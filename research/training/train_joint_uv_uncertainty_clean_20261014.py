#!/usr/bin/env python3
"""From-scratch RGB -> UV/PLV/PLH + UV sigma, four-rank resumable DDP.

Clean-data adaptation of the original loader8 trainer. Every full16 batch
uses the original four-rank DDP path. Only the final2 real images use the
explicit rank0/rank1 bare-model backward on loss/2 plus gradient SUM. This
places mean normalization before FP16 backward, avoiding redundant WORLD
amplification while preserving the exact two-real-image mean objective.
Only the frozen train/val splits are opened.  No test path, warm start, EMA,
teacher, or automatic post-training evaluation is accepted.  Run ``prepare``
before ``preflight`` or ``train``; the preparation writes immutable split and
epoch-order identities.  A production launch requires a successful native
1024 preflight on the target four GPUs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.matched_training_batching import order_sha256
from training.joint_clean_tail_20261014 import (
    iter_epoch_batches, resume_metadata, validate_resume_metadata,
    is_short_tail, tail_backward,
)
from training.train_uv_uncertainty_fullimage import (
    EpochMicroBatchSampler, atomic_torch_save, baseline_loader_orders,
    capture_rng, cpu_snapshot, digest, exclusive_lock, finite_all_ranks,
    make_loader, read_json, restore_rng, save_json, select_training_rows,
    state_equal, step_optimizer,
)

SCHEMA = "joint_uv_uncertainty_clean_20261014_v1"
COUNTS = {"train": 76642, "val": 10948}
WORLD = 4
EPOCHS = 20
BATCH = 16
MICRO = 1
VAL_KEYS = ("uv_nll", "l1_uv", "l1_plv", "l1_plh", "macro_l1", "sigma_mean")
CODE_FILES = (
    "training/train_joint_uv_uncertainty_clean_20261014.py",
    "training/joint_clean_tail_20261014.py",
    "training/train_uv_uncertainty_fullimage.py",
    "models/joint_uv_uncertainty_unet.py", "models/unet_baseline.py",
    "models/highres_patch_dataset.py", "models/matched_training_batching.py",
)


def validate_config(config):
    expected = {
        "schema": SCHEMA, "seed": 20260817,
        "model": {"base_channels": 32, "expected_parameters": 16909738, "ema": False},
        "expected_counts": COUNTS,
        "training": {"epochs": EPOCHS, "global_batch_size": BATCH,
                     "microbatch_size": MICRO, "world_size": WORLD,
                     "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
                     "amp": True, "initial_loss_scale": 65536,
                     "checkpoint_every_attempts": 1000,
                     "max_consecutive_amp_skips": 3},
        "objective": {"uv_nll": 1.0, "plv_l1": 1.0, "plh_l1": 1.0},
        "selection": "full_val_raw_equal_modality_macro_l1",
        "preprocess": "same_center_square_then_PIL_Lanczos_uint8_div255",
    }
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError("Frozen joint configuration differs: " + key)
    for key in ("output_dir", "dataset_root", "source_manifest", "source_manifest_sha256"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError("Missing configuration identity: " + key)
    sha = config["source_manifest_sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid source manifest SHA256")
    if config.get("dataloader") != {"num_workers": 8, "prefetch_factor": 1, "pin_memory": True}:
        raise ValueError("Dataloader protocol differs")
    if config.get("scheduler") != {"factor": .5, "patience": 2, "min_lr": 1e-6}:
        raise ValueError("Scheduler protocol differs")
    if config.get("early_stopping") != {"patience": 5, "min_delta": .0001}:
        raise ValueError("Early-stop protocol differs")
    if any(config.get(key) for key in ("checkpoint", "pretrained", "teacher", "ema", "test")):
        raise ValueError("Foreign initialization or test access prohibited")
    if config.get("execution_migration"):
        raise ValueError("Clean training must start from scratch, not migrate an old run")
    return config


def _train_val_rows(source):
    if digest(source) != SOURCE_SHA:
        raise RuntimeError("Canonical source manifest SHA256 mismatch")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("split") in COUNTS]
    splits = select_training_rows(rows, expected_counts=COUNTS)
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != sum(COUNTS.values()):
        raise RuntimeError("Duplicate train/val sample identity")
    for subset in splits.values():
        for row in subset:
            original = by_id[row["sample_id"]]
            for modality in ("plv", "plh"):
                for suffix in ("path", "sha256"):
                    key = modality + "_" + suffix
                    value = original.get(key)
                    if not isinstance(value, str) or not value:
                        raise ValueError("Missing joint modality identity: " + key)
                    row[key] = value
    return splits


SOURCE_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"


def prepare(config_path):
    config = validate_config(read_json(config_path))
    if config["source_manifest_sha256"] != SOURCE_SHA:
        raise RuntimeError("Source manifest is not the locked existing train/val split")
    out = Path(config["output_dir"])
    if out.exists():
        raise RuntimeError("Refusing to prepare into an existing directory")
    source = Path(config["source_manifest"])
    splits = _train_val_rows(source)
    code = {name: digest(ROOT / name) for name in CODE_FILES}
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        save_json(out / (name + "_manifest.json"), {"split": name, "rows": rows}, immutable=True)
    (out / "orders").mkdir()
    hashes = {}
    import torch
    for epoch, order in baseline_loader_orders(COUNTS["train"], config["seed"], EPOCHS):
        path = out / "orders" / ("epoch_%03d.json" % epoch)
        save_json(path, {"epoch": epoch, "indices": order,
                         "index_order_sha256": order_sha256(order)}, immutable=True)
        hashes[str(epoch)] = digest(path)
    protocol = {
        "schema": SCHEMA, "config": config, "config_sha256": digest(config_path),
        "code_sha256": code, "source_manifest_sha256": digest(source),
        "manifest_sha256": {s: digest(out / (s + "_manifest.json")) for s in COUNTS},
        "epoch_order_file_sha256": hashes, "order_generation_torch": torch.__version__,
        "order_rule": "actual_DataLoader_shuffle_seed_plus_one_based_epoch_persistent_workers",
        "batching": "global16_original_four_rank_DDP_micro1_tail2_rank0_rank1_loss_div2_grad_SUM_no_pad_no_drop",
        "objective": "1*UV_Gaussian_NLL+1*PLV_L1+1*PLH_L1",
        "objective_confirmed_by_user": True,
        "selection": "full_val_raw_equal_modality_macro_l1",
        "training_amp": "FP16_backbone_FP32_loss", "validation": "FP32_no_autocast_unpadded_bare_model",
        "initialization": "from_scratch", "ema_used": False,
        "test_split_used": False, "test_images_accessed": 0,
        "source_access_note": "Only train/val metadata retained; no test image paths opened",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(out / "protocol.json", protocol, immutable=True)
    save_json(out / "protocol_digest.json", {"sha256": digest(out / "protocol.json")}, immutable=True)
    print(json.dumps({"status": "prepared", "protocol_sha256": digest(out / "protocol.json"),
                      "train": COUNTS["train"], "val": COUNTS["val"], "test_split_used": False}), flush=True)


def load_frozen(config_path):
    config = validate_config(read_json(config_path))
    out = Path(config["output_dir"])
    protocol = read_json(out / "protocol.json")
    phash = digest(out / "protocol.json")
    if read_json(out / "protocol_digest.json")["sha256"] != phash:
        raise RuntimeError("Protocol digest mismatch")
    if protocol.get("schema") != SCHEMA or protocol.get("config") != config or protocol.get("config_sha256") != digest(config_path):
        raise RuntimeError("Frozen joint configuration mismatch")
    if digest(config["source_manifest"]) != SOURCE_SHA or protocol["source_manifest_sha256"] != SOURCE_SHA:
        raise RuntimeError("Canonical manifest changed")
    for name, sha in protocol["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("Frozen source code changed: " + name)
    splits = {}
    for split, count in COUNTS.items():
        path = out / (split + "_manifest.json")
        if digest(path) != protocol["manifest_sha256"][split]:
            raise RuntimeError("Frozen split manifest changed")
        rows = read_json(path)["rows"]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise RuntimeError("Frozen train/val split identity or coverage failure")
        splits[split] = rows
    if {r.get("split_group_id") for r in splits["train"] if r.get("split_group_id")} & {r.get("split_group_id") for r in splits["val"] if r.get("split_group_id")}:
        raise RuntimeError("Train/val subject-group overlap")
    return config, out, protocol, phash, splits


class JointDataset:
    """Read exactly four aligned full images using the existing preprocessing."""

    def __init__(self, rows, dataset_root, split):
        if split not in COUNTS or not rows or any(row.get("split") != split for row in rows):
            raise RuntimeError("Only frozen train/val rows are accepted")
        self.rows, self.dataset_root, self.split = rows, dataset_root, split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from models.highres_patch_dataset import center_square_geometry, resolve_raw_path, _load_rgb_crop
        row = self.rows[index]
        if row["split"] != self.split or self.split not in COUNTS:
            raise RuntimeError("Dataset split firewall failure")
        width, height = int(row["source_width"]), int(row["source_height"])
        left, top, side = center_square_geometry(width, height)
        if side < 1024:
            raise RuntimeError("Source below canonical 1024 resolution")
        box = (left, top, left + side, top + side)
        expected_size = (width, height)
        images = {}
        for name in ("rgb", "uv", "plv", "plh"):
            path = resolve_raw_path(self.dataset_root, row[name + "_path"])
            images[name] = _load_rgb_crop(path, expected_size, box, 1024, row["sample_id"], name)
        images["sample_id"] = row["sample_id"]
        images["index"] = index
        return images


def setup_distributed(config):
    import numpy as np
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != WORLD or rank not in range(WORLD) or local not in range(WORLD):
        raise RuntimeError("Use torchrun --nproc_per_node=4; clean tail2 compatibility is fixed to four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        raise RuntimeError("Four distinct CUDA devices are required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    return rank, torch.device("cuda", local)


def vectors_from_output(output, targets):
    import torch
    from models.joint_uv_uncertainty_unet import loss_components_per_image, validation_per_image, weighted_total_loss_per_image
    required = ("uv", "plv", "plh", "raw_log_variance", "log_variance", "variance", "sigma")
    if not isinstance(output, dict) or any(key not in output for key in required):
        raise RuntimeError("Missing joint mean/uncertainty head")
    if any(not bool(torch.isfinite(output[key]).all()) for key in required):
        raise FloatingPointError("Non-finite joint network output")
    if bool((output["variance"] <= 0).any()) or bool((output["sigma"] <= 0).any()):
        raise FloatingPointError("Non-positive joint variance/sigma")
    components = loss_components_per_image(output, targets)
    per_image_loss = weighted_total_loss_per_image(
        components, weights={"uv_nll": 1.0, "plv_l1": 1.0, "plh_l1": 1.0})
    metrics = validation_per_image(output, targets)
    metrics["uv_nll"] = components["uv_nll"]
    metrics["sigma_mean"] = output["sigma"].flatten(1).mean(1)
    if set(metrics) != set(VAL_KEYS) or any(not bool(torch.isfinite(v).all()) for v in metrics.values()):
        raise FloatingPointError("Incomplete or non-finite joint metrics")
    return per_image_loss, metrics


def new_state():
    return {"epoch": 1, "phase": "train", "next_batch": 0, "global_step": 0,
            "global_attempts": 0, "amp_overflow_skips": 0, "consecutive_amp_skips": 0,
            "history": [], "selection": {"best_val_macro_l1": None, "best_epoch": None,
                                        "early_stopping_anchor": None, "early_stopping_counter": 0},
            "blocked_reason": None}


def new_accumulator():
    return {"images": 0, "loss_sum": 0.0, "macro_l1_sum": 0.0,
            "l1_uv_sum": 0.0, "l1_plv_sum": 0.0, "l1_plh_sum": 0.0}


def update_selection(selection, value, epoch):
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite macro-L1 selection")
    updated = dict(selection)
    best = selection["best_val_macro_l1"]
    is_best = best is None or value < best
    if is_best:
        updated.update(best_val_macro_l1=value, best_epoch=epoch)
    anchor = selection["early_stopping_anchor"]
    significant = anchor is None or value < anchor - 1e-4
    if significant:
        updated.update(early_stopping_anchor=value, early_stopping_counter=0)
    else:
        updated["early_stopping_counter"] += 1
    return updated, is_best, updated["early_stopping_counter"] >= 5


def frozen_batching_metadata(order, next_batch):
    """The same four-rank micro-one identity written into every checkpoint."""
    return resume_metadata(order, next_batch, global_batch_size=BATCH,
                           world_size=WORLD, max_microbatch_size=MICRO)


def verify_frozen_batching_metadata(order, metadata):
    return validate_resume_metadata(order, metadata, global_batch_size=BATCH,
                                    world_size=WORLD, max_microbatch_size=MICRO)


def checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device, *, best=False):
    import torch.distributed as dist
    ranks = [None] * WORLD
    dist.all_gather_object(ranks, {"rank": rank, "rng": capture_rng(device), "epoch_accumulator": accum})
    if rank == 0:
        payload = {"schema": SCHEMA, "protocol_sha256": phash, "state": state,
                   "batching_resume": frozen_batching_metadata(order, state["next_batch"]),
                   "model_state_dict": core.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "rank_states": ranks, "ema_used": False, "test_split_used": False}
        if best:
            atomic_torch_save(out / "checkpoint_best.pt", payload)
        atomic_torch_save(out / "checkpoint_last.pt", payload)
        save_json(out / "history.json", state["history"])
    dist.barrier()


def verify_checkpoint(payload, phash, order):
    if payload.get("schema") != SCHEMA or payload.get("protocol_sha256") != phash:
        raise RuntimeError("Foreign checkpoint/protocol mismatch")
    if payload.get("ema_used") is not False or payload.get("test_split_used") is not False:
        raise RuntimeError("Foreign EMA or test-using checkpoint")
    ranks = payload.get("rank_states", [])
    if len(ranks) != WORLD or sorted(r["rank"] for r in ranks) != list(range(WORLD)):
        raise RuntimeError("Missing rank RNG/accumulator in checkpoint")
    if verify_frozen_batching_metadata(order, payload["batching_resume"]) != payload["state"]["next_batch"]:
        raise RuntimeError("Checkpoint batch offset mismatch")
    if payload["state"].get("blocked_reason"):
        raise RuntimeError("Unsafe blocked checkpoint requires user review")


def _load_val_records(path, rows, indices, phash, state_id):
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if not line.endswith("\n"):
                    raise RuntimeError("Truncated validation journal; preserve evidence")
                record = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NaN")))
                if (position >= len(indices) or record.get("index") != indices[position]
                        or record.get("sample_id") != rows[indices[position]]["sample_id"]
                        or record.get("protocol_sha256") != phash or record.get("state_id") != state_id
                        or set(record.get("metrics", {})) != set(VAL_KEYS)):
                    raise RuntimeError("Validation identity/protocol/order mismatch")
                if any(not math.isfinite(float(v)) for v in record["metrics"].values()):
                    raise FloatingPointError("Non-finite persisted validation metric")
                records.append(record)
    return records


def validate_full(core, dataset, rows, out, phash, state, config, rank, device):
    import torch
    import torch.distributed as dist
    indices = list(range(rank, len(rows), WORLD))
    state_id = "%s:e%d:a%d:s%d" % (phash, state["epoch"], state["global_attempts"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    records = _load_val_records(path, rows, indices, phash, state_id)
    totals = {k: sum(r["metrics"][k] for r in records) for k in VAL_KEYS}
    loader = make_loader(dataset, [[i] for i in indices[len(records):]], config,
                         config["seed"] + 500000 + state["epoch"] * WORLD + rank)
    core.eval()
    current_index = indices[len(records)] if len(records) < len(indices) else None
    try:
        with torch.inference_mode(), path.open("a", encoding="utf-8") as handle:
            for position, batch in enumerate(loader, len(records)):
                index = indices[position]
                current_index = index
                if batch["sample_id"] != [rows[index]["sample_id"]]:
                    raise RuntimeError("Val sample identity mismatch")
                rgb = batch["rgb"].to(device, non_blocking=True).float()
                targets = {name: batch[name].to(device, non_blocking=True).float()
                           for name in ("uv", "plv", "plh")}
                output = core(rgb)  # Bare model avoids unequal-rank DDP forward collectives.
                _, vectors = vectors_from_output(output, targets)
                metrics = {}
                for key in VAL_KEYS:
                    value = vectors[key]
                    if tuple(value.shape) != (1,):
                        raise RuntimeError("Val per-image metric shape mismatch")
                    metrics[key] = float(value.item())
                    totals[key] += metrics[key]
                handle.write(json.dumps({"index": index, "sample_id": rows[index]["sample_id"],
                                         "metrics": metrics, "protocol_sha256": phash,
                                         "state_id": state_id, "test_split_used": False},
                                        sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                if (position + 1) % 25 == 0:
                    os.fsync(handle.fileno())
                    save_json(out / ("validation_progress_rank%d.json" % rank),
                              {"status": "validating", "epoch": state["epoch"],
                               "completed_images": position + 1, "expected_images": len(indices),
                               "test_split_used": False})
                current_index = indices[position + 1] if position + 1 < len(indices) else None
            os.fsync(handle.fileno())
    except Exception:
        save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())),
                  {"status": "failed", "rank": rank, "index": current_index,
                   "sample_id": rows[current_index]["sample_id"] if current_index is not None else None,
                   "traceback": traceback.format_exc(), "protocol_sha256": phash}, immutable=True)
        raise
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    records = _load_val_records(path, rows, indices, phash, state_id)
    if len(records) != len(indices):
        raise RuntimeError("Validation readback incomplete")
    values = torch.tensor([len(records)] + [totals[k] for k in VAL_KEYS], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    merged = values.tolist()
    if int(merged[0]) != COUNTS["val"]:
        raise RuntimeError("Validation not exactly 11534 unique images")
    result = {key: merged[i + 1] / merged[0] for i, key in enumerate(VAL_KEYS)}
    result["images"] = int(merged[0])
    save_json(out / ("validation_progress_rank%d.json" % rank),
              {"status": "complete", "epoch": state["epoch"],
               "completed_images": len(records), "expected_images": len(indices),
               "record_sha256": digest(path), "test_split_used": False})
    return result


def _train_totals(accum, device):
    import torch
    import torch.distributed as dist
    keys = ("images", "loss_sum", "macro_l1_sum", "l1_uv_sum", "l1_plv_sum", "l1_plh_sum")
    tensor = torch.tensor([accum[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    values = tensor.tolist()
    count = int(values[0])
    return {"images": count, **{k[:-4]: values[i] / count if count else None
                                  for i, k in enumerate(keys[1:], 1)}}


def _log(out, event, rank):
    if rank == 0:
        event = dict(event, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     test_split_used=False)
        save_json(out / "training_progress.json", event)
        line = json.dumps(event, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def finalize_completed(out, state, phash):
    """Idempotently publish a finished checkpoint, including crash-after-LAST.

    A pre-existing summary/history is verified and never overwritten.  This
    function is called only on rank zero, after a complete LAST is loaded or
    immediately after the final complete checkpoint was atomically written.
    """
    if state.get("phase") != "complete" or not state.get("history") or not state.get("stop_reason"):
        raise RuntimeError("Cannot finalize an incomplete joint training state")
    history_path = out / "history.json"
    if history_path.exists():
        if read_json(history_path) != state["history"]:
            raise RuntimeError("Existing completed history differs from checkpoint; preserved")
    else:
        save_json(history_path, state["history"], immutable=True)
    best = out / "checkpoint_best.pt"
    last = out / "checkpoint_last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Complete state lacks BEST/LAST checkpoint")
    summary = {"status": "complete", "training_complete": True,
               "epochs_completed": len(state["history"]), "stop_reason": state["stop_reason"],
               "selection": state["selection"], "global_step": state["global_step"],
               "global_attempts": state["global_attempts"],
               "amp_overflow_skips": state["amp_overflow_skips"],
               "protocol_sha256": phash, "checkpoint_best_sha256": digest(best),
               "checkpoint_last_sha256": digest(last),
               "test_split_used": False, "test_images_accessed": 0, "ema_used": False}
    summary_path = out / "training_summary.json"
    if summary_path.exists():
        if read_json(summary_path) != summary:
            raise RuntimeError("Existing completed summary differs from checkpoint; preserved")
    else:
        save_json(summary_path, summary, immutable=True)
    return summary


def train(config_path, resume, preflight_output=None):
    config, out, protocol, phash, splits = load_frozen(config_path)
    preflight = Path(preflight_output) if preflight_output else out.parent / (out.name + "_preflight")
    for rank_id in range(WORLD):
        result_path = preflight / ("rank%d.json" % rank_id)
        if not result_path.exists() or read_json(result_path).get("status") != "complete":
            raise RuntimeError("All four native 1024 synthetic preflight rank reports required before training")
        report = read_json(result_path)
        if (report.get("rank") != rank_id or report.get("parameters") != config["model"]["expected_parameters"]
                or report.get("config_sha256") != digest(config_path)
                or report.get("code_sha256") != protocol["code_sha256"]
                or report.get("protocol_sha256") != phash
                or report.get("dataset_accessed") is not True
                or report.get("tail_global_batch") != 2
                or report.get("tail2_replay_pass") is not True
                or report.get("real_train_val_pass") is not True
                or report.get("post_tail_full_ddp_pass") is not True
                or report.get("checkpoint_weights_reused") is not False
                or report.get("formal_checkpoint_written") is not False
                or report.get("world_size") != WORLD
                or report.get("visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")
                or report.get("test_split_used") is not False
                or set(report.get("resume_replay_checks", {})) != {"model", "optimizer", "scaler", "state", "rng"}
                or not all(report["resume_replay_checks"].values())):
            raise RuntimeError("Preflight config/source identity mismatch")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models.joint_uv_uncertainty_unet import JointUVUncertaintyUNet
    rank, device = setup_distributed(config)
    own_preflight = read_json(preflight / ("rank%d.json" % rank))
    if own_preflight["gpu"] != torch.cuda.get_device_name(device):
        raise RuntimeError("Native preflight GPU model differs from training GPU")
    if own_preflight["train_peak_allocated_mib"] >= torch.cuda.get_device_properties(device).total_memory / 2 ** 20 * .9:
        raise RuntimeError("Native preflight peak left less than ten percent GPU memory headroom")
    lock = exclusive_lock(out / "training_coordinator.lock") if rank == 0 else contextlib.nullcontext()
    state, accum, train_loader, current_ids = new_state(), new_accumulator(), None, []
    try:
        with lock:
            dist.barrier()
            core = JointUVUncertaintyUNet(base_channels=32).float().to(device)
            if sum(p.numel() for p in core.parameters()) != config["model"]["expected_parameters"]:
                raise RuntimeError("Joint model parameter count changed")
            model = DDP(core, device_ids=[device.index], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=65536.0)
            last = out / "checkpoint_last.pt"
            saved_rng = None
            if last.exists():
                if resume == "none":
                    raise RuntimeError("Existing checkpoint: cannot overwrite from zero")
                payload = torch.load(last, map_location="cpu", weights_only=False)
                state = payload["state"]
                order = _load_order(out, protocol, state["epoch"])
                verify_checkpoint(payload, phash, order)
                core.load_state_dict(payload["model_state_dict"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                scheduler.load_state_dict(payload["scheduler_state_dict"])
                scaler.load_state_dict(payload["scaler_state_dict"])
                accum = payload["rank_states"][rank]["epoch_accumulator"]
                saved_rng = payload["rank_states"][rank]["rng"]
                del payload
            elif any((out / name).exists() for name in ("checkpoint_best.pt", "history.json", "training_summary.json")):
                raise RuntimeError("Training artifacts without LAST checkpoint; no silent restart")
            if state["phase"] == "complete":
                if rank == 0:
                    print(json.dumps({"status": "already_complete", "summary":
                                      finalize_completed(out, state, phash)}, sort_keys=True), flush=True)
                dist.barrier()
                return
            if state["phase"] == "epoch_complete":
                if state["epoch"] >= EPOCHS:
                    raise RuntimeError("Epoch20 checkpoint must be marked complete")
                state.update(epoch=state["epoch"] + 1, phase="next_epoch", next_batch=0)
            train_data = JointDataset(splits["train"], config["dataset_root"], "train")
            val_data = JointDataset(splits["val"], config["dataset_root"], "val")
            sampler = EpochMicroBatchSampler(rank)
            train_loader = make_loader(train_data, sampler, config, config["seed"] + rank * 1000)
            if saved_rng is not None:
                restore_rng(saved_rng, device)
            save_json(out / ("environment_rank%d_%d.json" % (rank, time.time_ns())),
                      {"python": sys.version.split()[0], "torch": torch.__version__,
                       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "gpu": torch.cuda.get_device_name(device), "rank": rank, "world_size": WORLD,
                       "parameters": sum(p.numel() for p in core.parameters()),
                       "training_amp": "FP16", "loss_precision": "FP32", "protocol_sha256": phash,
                       "from_scratch": saved_rng is None, "test_split_used": False}, immutable=True)
            batch_count = math.ceil(COUNTS["train"] / BATCH)
            session_start, initial_attempts = time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]
                order = _load_order(out, protocol, epoch)
                if state["phase"] == "next_epoch":
                    state.update(phase="train", next_batch=0)
                    accum = new_accumulator()
                if state["global_attempts"] == 0 and not last.exists():
                    checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device)
                if state["phase"] == "train":
                    plans = tuple(iter_epoch_batches(order, global_batch_size=BATCH, world_size=WORLD,
                                                     max_microbatch_size=MICRO, start_batch=state["next_batch"]))
                    sampler.set_plans(plans)
                    batches = iter(train_loader)
                    model.train()
                    for plan in plans:
                        optimizer.zero_grad(set_to_none=True)
                        current_ids = [splits["train"][i]["sample_id"] for i in plan.rank_batches[rank].indices]
                        if is_short_tail(plan):
                            _train_tail(core, plan, rank, scaler, device, batches, splits["train"], accum)
                        else:
                            for micro in plan.rank_batches[rank].microbatches:
                                batch = next(batches)
                                expected_ids = [splits["train"][i]["sample_id"] for i in micro.indices]
                                if batch["sample_id"] != expected_ids:
                                    raise RuntimeError("Train microbatch identity mismatch")
                                rgb = batch["rgb"].to(device, non_blocking=True).float()
                                targets = {name: batch[name].to(device, non_blocking=True).float()
                                           for name in ("uv", "plv", "plh")}
                                sync = contextlib.nullcontext() if micro.synchronize else model.no_sync()
                                with sync:
                                    failure = None
                                    try:
                                        with torch.cuda.amp.autocast(dtype=torch.float16):
                                            output = model(rgb)
                                        with torch.cuda.amp.autocast(enabled=False):
                                            loss, metrics = vectors_from_output(output, targets)
                                        valid = bool(torch.isfinite(loss).all())
                                    except Exception:
                                        failure, valid = traceback.format_exc(), False
                                    if not finite_all_ranks(valid, device):
                                        raise FloatingPointError("Invalid train output/loss: " + (failure or "another rank failed"))
                                    scaler.scale(loss.mean() * micro.loss_scale).backward()
                                n = len(micro.indices)
                                accum["images"] += n
                                accum["loss_sum"] += float(loss.detach().double().sum().item())
                                for name in ("macro_l1", "l1_uv", "l1_plv", "l1_plh"):
                                    accum[name + "_sum"] += float(metrics[name].detach().double().sum().item())
                                del batch, rgb, targets, output, loss, metrics
                        successful, gradient_norm = step_optimizer(model, optimizer, scaler, device, state)
                        state["next_batch"] = plan.batch_index + 1
                        if state["consecutive_amp_skips"] >= 3:
                            state["blocked_reason"] = "three_consecutive_amp_gradient_overflows"
                            checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device)
                            raise FloatingPointError(state["blocked_reason"])
                        if state["global_attempts"] % 1000 == 0:
                            checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device)
                        if state["global_attempts"] % 20 == 0 or state["next_batch"] == 1 or not successful:
                            totals = _train_totals(accum, device)
                            per_attempt = (time.monotonic() - session_start) / max(1, state["global_attempts"] - initial_attempts)
                            _log(out, {"status": "training" if successful else "AMP_OVERFLOW_SKIP",
                                       "epoch": epoch, "max_epochs": EPOCHS, "epoch_batch": state["next_batch"],
                                       "epoch_batches": batch_count, "train": totals,
                                       "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                                       "amp_overflow_skips": state["amp_overflow_skips"],
                                       "gradient_norm": gradient_norm, "loss_scale": scaler.get_scale(),
                                       "learning_rate": optimizer.param_groups[0]["lr"],
                                       "epoch_train_eta_seconds": per_attempt * (batch_count - state["next_batch"])}, rank)
                    totals = _train_totals(accum, device)
                    if totals["images"] != COUNTS["train"] or state["next_batch"] != batch_count:
                        raise RuntimeError("Epoch did not cover exactly 76642 train images")
                    state["phase"] = "val"
                    checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device)
                totals = _train_totals(accum, device)
                _log(out, {"status": "validating", "epoch": epoch, "max_epochs": EPOCHS,
                           "train": totals, "global_step": state["global_step"]}, rank)
                validation = validate_full(core, val_data, splits["val"], out, phash, state, config, rank, device)
                selected, best, early_stop = update_selection(state["selection"], validation["macro_l1"], epoch)
                state["selection"] = selected
                lr_before = optimizer.param_groups[0]["lr"]
                scheduler.step(validation["macro_l1"])
                row = {"epoch": epoch, "train": totals, "validation": validation,
                       "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                       "amp_overflow_skips": state["amp_overflow_skips"], "is_best": best,
                       "learning_rate": lr_before, "next_learning_rate": optimizer.param_groups[0]["lr"],
                       "test_split_used": False}
                state["history"].append(row)
                if early_stop or epoch == EPOCHS:
                    state.update(phase="complete", stop_reason="early_stopping" if early_stop else "max_epochs")
                else:
                    state["phase"] = "epoch_complete"
                checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accum, order, rank, device, best=best)
                _log(out, dict(row, status="epoch_complete", max_epochs=EPOCHS,
                               selection=selected), rank)
                if state["phase"] == "epoch_complete":
                    state.update(epoch=epoch + 1, phase="next_epoch", next_batch=0)
            if rank == 0:
                summary = finalize_completed(out, state, phash)
                _log(out, summary, rank)
            dist.barrier()
    except Exception:
        if out.exists():
            save_json(out / ("failure_rank%d_%d.json" % (rank, time.time_ns())),
                      {"status": "failed", "rank": rank, "state": state,
                       "current_train_sample_ids": current_ids,
                       "traceback": traceback.format_exc(), "test_split_used": False}, immutable=True)
        raise
    finally:
        if train_loader is not None:
            iterator = getattr(train_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
        if dist.is_initialized():
            dist.destroy_process_group()


def _load_order(out, protocol, epoch):
    # Identical original identity checks, bound to this clean train count.
    path = out / "orders" / ("epoch_%03d.json" % epoch)
    if digest(path) != protocol["epoch_order_file_sha256"][str(epoch)]:
        raise RuntimeError("Frozen epoch order changed")
    payload = read_json(path)
    order = payload["indices"]
    if (payload["epoch"] != epoch or len(order) != COUNTS["train"]
            or sorted(order) != list(range(COUNTS["train"])) or order_sha256(order) != payload["index_order_sha256"]):
        raise RuntimeError("Epoch index coverage/hash invalid")
    return order


def _train_tail(core, plan, rank, scaler, device, batches, rows, accum):
    """Only the two real tail samples; inactive ranks never consume a batch."""
    import torch

    def loss_closure():
        batch = next(batches)
        indices = plan.rank_batches[rank].indices
        expected_ids = [rows[i]["sample_id"] for i in indices]
        if batch["sample_id"] != expected_ids or len(indices) != 1:
            raise RuntimeError("Clean tail sample identity mismatch")
        rgb = batch["rgb"].to(device, non_blocking=True).float()
        targets = {name: batch[name].to(device, non_blocking=True).float()
                   for name in ("uv", "plv", "plh")}
        # The bare core deliberately avoids a DDP forward on only two ranks.
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = core(rgb)
        with torch.cuda.amp.autocast(enabled=False):
            return vectors_from_output(output, targets)

    result = tail_backward(core, plan, rank, scaler, device, loss_closure)
    if result is not None:
        loss, metrics = result
        accum["images"] += len(plan.rank_batches[rank].indices)
        accum["loss_sum"] += float(loss.detach().double().sum().item())
        for name in ("macro_l1", "l1_uv", "l1_plv", "l1_plh"):
            accum[name + "_sum"] += float(metrics[name].detach().double().sum().item())


def _synthetic_tail_update(core, model, optimizer, scaler, state, rank, device, *, size, seed):
    import torch
    plan = next(iter_epoch_batches(range(2), global_batch_size=BATCH,
                                   world_size=WORLD, max_microbatch_size=MICRO))
    optimizer.zero_grad(set_to_none=True)

    def loss_closure():
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        rgb = torch.rand(1, 3, size, size, generator=generator, device=device)
        targets = {name: torch.rand(1, 3, size, size, generator=generator, device=device)
                   for name in ("uv", "plv", "plh")}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = core(rgb)
        with torch.cuda.amp.autocast(enabled=False):
            return vectors_from_output(output, targets)

    tail_backward(core, plan, rank, scaler, device, loss_closure)
    succeeded, _ = step_optimizer(model, optimizer, scaler, device, state)
    torch.cuda.synchronize(device)
    return succeeded


def _real_data_smoke(core, splits, config, rank, device):
    """Disposable preflight reads only one frozen train/val capture per rank."""
    import torch
    from models.highres_patch_dataset import resolve_raw_path
    result = {}
    core.eval()
    for split in ("train", "val"):
        row = splits[split][rank]
        if row["split"] != split:
            raise RuntimeError("Real preflight split identity mismatch")
        file_hashes = {}
        for name in ("rgb", "uv", "plv", "plh"):
            path = resolve_raw_path(config["dataset_root"], row[name + "_path"])
            actual = digest(path)
            if actual != row[name + "_sha256"]:
                raise RuntimeError("Real preflight source hash mismatch: " + row["sample_id"] + ":" + name)
            file_hashes[name] = actual
        item = JointDataset([row], config["dataset_root"], split)[0]
        for name in ("rgb", "uv", "plv", "plh"):
            tensor = item[name]
            if tuple(tensor.shape) != (3, 1024, 1024) or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("Real preflight shape/finite failure: " + name)
            if float(tensor.min()) < 0 or float(tensor.max()) > 1:
                raise RuntimeError("Real preflight normalization mismatch")
        with torch.inference_mode():
            rgb = item["rgb"].unsqueeze(0).to(device).float()
            targets = {name: item[name].unsqueeze(0).to(device).float() for name in ("uv", "plv", "plh")}
            _, metrics = vectors_from_output(core(rgb), targets)
        result[split] = {"status": "complete", "sample_id": row["sample_id"],
                         "split": split, "source_sha256": file_hashes,
                         "shape": [3, 1024, 1024], "finite_metrics": sorted(metrics)}
    return {"status": "complete", "rows": result, "test_images_accessed": 0,
            "optimizer_updates": 0}


def _check_replicas(core, optimizer, scaler, state):
    """Preflight-only exact checksum of identical model/optimizer/scaler state."""
    import hashlib
    import torch
    import torch.distributed as dist
    hasher = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            hasher.update(str((array.dtype, array.shape)).encode("ascii"))
            hasher.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            hasher.update(type(value).__name__.encode("ascii"))
            for item in value:
                update(item)
        else:
            hasher.update(repr((type(value).__name__, value)).encode("utf-8"))

    update({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "state": state})
    checksums = [None] * WORLD
    dist.all_gather_object(checksums, hasher.hexdigest())
    if len(set(checksums)) != 1:
        raise RuntimeError("Preflight rank replicas diverged after tail2")
    return checksums[0]


def _synthetic_update(core, model, optimizer, scaler, state, rank, device, *, size, seed):
    import torch
    plan = next(iter_epoch_batches(range(BATCH), global_batch_size=BATCH,
                                   world_size=WORLD, max_microbatch_size=MICRO))
    optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    losses = []
    for micro in plan.rank_batches[rank].microbatches:
        rgb = torch.rand(1, 3, size, size, generator=generator, device=device)
        targets = {name: torch.rand(1, 3, size, size, generator=generator, device=device)
                   for name in ("uv", "plv", "plh")}
        sync = contextlib.nullcontext() if micro.synchronize else model.no_sync()
        with sync:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(rgb)
            with torch.cuda.amp.autocast(enabled=False):
                loss, _ = vectors_from_output(output, targets)
            if not finite_all_ranks(bool(torch.isfinite(loss).all()), device):
                raise FloatingPointError("Synthetic preflight loss non-finite")
            scaler.scale(loss.mean() * micro.loss_scale).backward()
        losses.append(float(loss.detach().mean().item()))
        del rgb, targets, output, loss
    succeeded, _ = step_optimizer(model, optimizer, scaler, device, state)
    torch.cuda.synchronize(device)
    return succeeded, losses


def preflight(config_path, output_path=None):
    config, out, protocol, phash, splits = load_frozen(config_path)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models.joint_uv_uncertainty_unet import JointUVUncertaintyUNet
    rank, device = setup_distributed(config)
    directory = Path(output_path) if output_path else out.parent / (out.name + "_preflight")
    try:
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        core = JointUVUncertaintyUNet(base_channels=32).float().to(device)
        model = DDP(core, device_ids=[device.index], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
        state = new_state()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        warmup = []
        for attempt in range(8):
            succeeded, losses = _synthetic_update(core, model, optimizer, scaler, state, rank, device,
                                                  size=1024, seed=19001 + attempt)
            warmup.append({"successful": succeeded, "losses": losses, "scale_after": scaler.get_scale()})
            if state["consecutive_amp_skips"] >= 3:
                raise FloatingPointError("Three consecutive native synthetic AMP gradient overflows")
            if state["global_step"] >= 2:
                break
        if state["global_step"] < 2:
            raise RuntimeError("Native preflight did not reach two finite optimizer updates in eight attempts")
        train_seconds = time.monotonic() - started
        warmup_skips = state["amp_overflow_skips"]
        # A replay check uses identical synthetic data and rank RNG at the same
        # update boundary. The in-memory snapshot is NEVER a training checkpoint.
        snapshot = io.BytesIO()
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "state": state,
                    "rng": capture_rng(device)}, snapshot)
        first_replay_success, _ = _synthetic_update(core, model, optimizer, scaler, state, rank, device,
                                                    size=1024, seed=19099)
        if not first_replay_success:
            raise FloatingPointError("Native synthetic replay update overflowed after warmup")
        expected = {"model": cpu_snapshot(core.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                    "scaler": cpu_snapshot(scaler.state_dict()), "state": cpu_snapshot(state),
                    "rng": cpu_snapshot(capture_rng(device))}
        snapshot.seek(0)
        saved = torch.load(snapshot, map_location="cpu", weights_only=False)
        core.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state.clear(); state.update(saved["state"])
        restore_rng(saved["rng"], device)
        second_replay_success, _ = _synthetic_update(core, model, optimizer, scaler, state, rank, device,
                                                     size=1024, seed=19099)
        if not second_replay_success:
            raise FloatingPointError("Restored native synthetic replay update overflowed")
        checks = {"model": state_equal(expected["model"], core.state_dict()),
                  "optimizer": state_equal(expected["optimizer"], optimizer.state_dict()),
                  "scaler": state_equal(expected["scaler"], scaler.state_dict()),
                  "state": state_equal(expected["state"], state),
                  "rng": state_equal(expected["rng"], capture_rng(device))}
        if not finite_all_ranks(all(checks.values()), device):
            raise RuntimeError("Native joint checkpoint replay mismatch: " + json.dumps(checks))
        # Only the newly authorized compatibility path is added: replay a
        # real-size synthetic two-image tail, then prove normal DDP still works.
        tail_snapshot = io.BytesIO()
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "state": state,
                    "rng": capture_rng(device)}, tail_snapshot)
        if not _synthetic_tail_update(core, model, optimizer, scaler, state, rank, device,
                                      size=1024, seed=19101):
            raise FloatingPointError("Native tail2 replay update overflowed")
        tail_replica_sha = _check_replicas(core, optimizer, scaler, state)
        full_after_tail, _ = _synthetic_update(core, model, optimizer, scaler, state, rank, device,
                                               size=1024, seed=19102)
        if not full_after_tail:
            raise FloatingPointError("Normal DDP update after tail2 overflowed")
        post_tail_replica_sha = _check_replicas(core, optimizer, scaler, state)
        tail_expected = {"model": cpu_snapshot(core.state_dict()),
                         "optimizer": cpu_snapshot(optimizer.state_dict()),
                         "scaler": cpu_snapshot(scaler.state_dict()), "state": cpu_snapshot(state),
                         "rng": cpu_snapshot(capture_rng(device))}
        tail_snapshot.seek(0)
        tail_saved = torch.load(tail_snapshot, map_location="cpu", weights_only=False)
        core.load_state_dict(tail_saved["model"], strict=True)
        optimizer.load_state_dict(tail_saved["optimizer"])
        scaler.load_state_dict(tail_saved["scaler"])
        state.clear(); state.update(tail_saved["state"])
        restore_rng(tail_saved["rng"], device)
        if not _synthetic_tail_update(core, model, optimizer, scaler, state, rank, device,
                                      size=1024, seed=19101):
            raise FloatingPointError("Restored native tail2 replay update overflowed")
        replay_full_after_tail, _ = _synthetic_update(core, model, optimizer, scaler, state, rank, device,
                                                      size=1024, seed=19102)
        if not replay_full_after_tail:
            raise FloatingPointError("Restored post-tail normal DDP update overflowed")
        tail_checks = {"model": state_equal(tail_expected["model"], core.state_dict()),
                       "optimizer": state_equal(tail_expected["optimizer"], optimizer.state_dict()),
                       "scaler": state_equal(tail_expected["scaler"], scaler.state_dict()),
                       "state": state_equal(tail_expected["state"], state),
                       "rng": state_equal(tail_expected["rng"], capture_rng(device))}
        if not finite_all_ranks(all(tail_checks.values()), device):
            raise RuntimeError("Native tail2 checkpoint replay mismatch: " + json.dumps(tail_checks))
        _check_replicas(core, optimizer, scaler, state)
        train_peak = torch.cuda.max_memory_allocated(device)
        train_reserved_peak = torch.cuda.max_memory_reserved(device)
        core.eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            rgb = torch.rand(1, 3, 1024, 1024, device=device)
            targets = {name: torch.rand_like(rgb) for name in ("uv", "plv", "plh")}
            output = core(rgb)
            _, metrics = vectors_from_output(output, targets)
            if set(metrics) != set(VAL_KEYS):
                raise RuntimeError("Native validation metrics incomplete")
        real_smoke = _real_data_smoke(core, splits, config, rank, device)
        result = {"status": "complete", "kind": "synthetic_native_1024_and_real_train_val_smoke_not_scientific_training",
                  "rank": rank, "gpu": torch.cuda.get_device_name(device),
                  "world_size": WORLD, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "parameters": sum(p.numel() for p in core.parameters()),
                  "train_peak_allocated_mib": train_peak / 2 ** 20,
                  "train_peak_reserved_mib": train_reserved_peak / 2 ** 20,
                  "validation_with_optimizer_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
                  "warmup_seconds": train_seconds, "warmup_attempts": warmup,
                  "warmup_successful_updates": 2, "warmup_amp_skips": warmup_skips,
                  "resume_replay_checks": checks, "config_sha256": digest(config_path),
                  "code_sha256": protocol["code_sha256"], "protocol_sha256": phash,
                  "tail_global_batch": 2, "tail2_replay_pass": True,
                  "tail2_resume_replay_checks": tail_checks, "post_tail_full_ddp_pass": True,
                  "tail_replica_sha256": tail_replica_sha, "post_tail_replica_sha256": post_tail_replica_sha,
                  "real_train_val_pass": True, "real_data_smoke": real_smoke,
                  "checkpoint_weights_reused": False, "formal_checkpoint_written": False,
                  "dataset_accessed": True, "test_split_used": False, "test_images_accessed": 0}
        save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        dist.barrier()
    except Exception:
        if directory.exists():
            save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())),
                      {"status": "failed", "rank": rank, "traceback": traceback.format_exc(),
                       "dataset_accessed": "at_most_train_val_smoke", "test_split_used": False}, immutable=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def smoke(device_name="cpu"):
    """Two synthetic images, one optimizer step; no real split or checkpoint."""
    import torch
    from models.joint_uv_uncertainty_unet import JointUVUncertaintyUNet
    device = torch.device(device_name)
    model = JointUVUncertaintyUNet(base_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260817)
        rgb = torch.rand(2, 3, 32, 32, device=device)
        targets = {name: torch.rand_like(rgb) for name in ("uv", "plv", "plh")}
        loss, metrics = vectors_from_output(model(rgb), targets)
        if tuple(loss.shape) != (2,) or tuple(metrics["macro_l1"].shape) != (2,):
            raise RuntimeError("Two-image synthetic shapes invalid")
        loss.mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Synthetic joint gradients non-finite")
        optimizer.step()
        print(json.dumps({"status": "two_image_synthetic_smoke_passed", "device": str(device),
                          "fixture_base_channels": 4, "formal_parameters": 16909738,
                          "training_started": False, "dataset_accessed": False,
                          "test_split_used": False}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "preflight", "train", "smoke"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--smoke-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(args.smoke_device)
        return
    if args.config is None:
        parser.error("--config is required except for synthetic smoke")
    if args.mode == "prepare":
        prepare(args.config)
    elif args.mode == "preflight":
        preflight(args.config, args.preflight_output)
    else:
        train(args.config, args.resume, args.preflight_output)


if __name__ == "__main__":
    main()
