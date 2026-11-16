#!/usr/bin/env python3
"""Frozen-protocol, from-zero RGB -> UV mean/variance training; never TEST.

prepare: freeze train/val metadata, audited-DataLoader epoch index orders and code.
preflight: disposable synthetic 1024 three-GPU updates; never real data/weights.
train: three-rank DDP, global image batch16, no padding, resumable at batch edges.

Only this run's checkpoint_last.pt can be resumed. No initialization checkpoint,
teacher, EMA, extra L1 objective, TEST loader, or automatic post-training test.
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.matched_training_batching import iter_epoch_batches, order_sha256, resume_metadata, validate_resume_metadata

SCHEMA = "uv_uncertainty_fullimage_training_v1"
COUNTS = {"train": 80740, "val": 11534}
VAL_KEYS = ("nll", "l1", "mse", "sigma_mean", "standardized_squared_error",
            "standardized_residual_mean", "coverage68", "coverage95", "coverage99",
            "near_variance_floor_fraction")
CODE_FILES = ("training/train_uv_uncertainty_fullimage.py", "models/uv_uncertainty_unet.py",
              "models/matched_training_batching.py", "models/unet_baseline.py", "models/highres_patch_dataset.py")


def reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, payload, immutable=False):
    path = Path(path)
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = path if immutable else path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if not immutable:
        os.replace(destination, path)


@contextlib.contextmanager
def exclusive_lock(path):
    import fcntl
    with Path(path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This training/preflight task already has an active coordinator")
        yield


def validate_config(config):
    required = ("output_dir", "dataset_root", "source_manifest", "source_manifest_sha256", "seed",
                "training", "dataloader", "scheduler", "early_stopping", "model", "expected_counts")
    if any(key not in config for key in required):
        raise ValueError("Missing required frozen configuration fields")
    expected_training = {"epochs": 20, "global_batch_size": 16, "microbatch_size": 2,
                         "world_size": 3, "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
                         "amp": True, "initial_loss_scale": 65536.0,
                         "checkpoint_every_attempts": 1000, "max_consecutive_amp_skips": 3}
    for key, value in expected_training.items():
        if config["training"].get(key) != value:
            raise ValueError("Unapproved training configuration: " + key)
    if config["expected_counts"] != COUNTS or config["model"].get("base_channels") != 32:
        raise ValueError("Frozen data count or backbone width changed")
    if config["model"].get("expected_parameters") != 7763140 or config["model"].get("ema") is not False:
        raise ValueError("Frozen uncertainty parameter count or no-EMA setting changed")
    for key, value in {"factor": .5, "patience": 2, "min_lr": 1e-6}.items():
        if config["scheduler"].get(key) != value:
            raise ValueError("Unapproved scheduler configuration: " + key)
    if config["early_stopping"].get("patience") != 5 or config["early_stopping"].get("min_delta") != .0001:
        raise ValueError("Unapproved early-stopping configuration")
    if config["seed"] != 20260817:
        raise ValueError("Frozen random seed changed")
    for key in ("output_dir", "dataset_root", "source_manifest"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError("Invalid configured path: " + key)
    sha = config["source_manifest_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid source manifest SHA256")
    workers = config["dataloader"].get("num_workers", 4)
    if not isinstance(workers, int) or workers < 0 or workers > 8:
        raise ValueError("dataloader num_workers must be within 0..8")
    if config["dataloader"].get("prefetch_factor", 1) != 1:
        raise ValueError("Only bounded prefetch_factor=1 is approved")
    if any(config.get(key) for key in ("checkpoint", "pretrained", "warmstart", "resume_checkpoint", "ema")):
        raise ValueError("Warm starts, foreign checkpoints and EMA are prohibited")
    return config


def select_training_rows(source_rows, expected_counts=None):
    """Inspect split metadata; retain only train/val rows, never TEST image paths."""
    expected_counts = COUNTS if expected_counts is None else expected_counts
    result = {"train": [], "val": []}
    keys = ("sample_id", "split", "source_width", "source_height", "rgb_path", "uv_path",
            "rgb_sha256", "uv_sha256", "subject_id", "split_group_id")
    for row in source_rows:
        split = row.get("split")
        if split not in result:
            continue
        if str(row.get("eligible_primary")) != "1":
            raise ValueError("Ineligible row in a frozen training/validation split")
        selected = {key: row.get(key) for key in keys}
        if any(not selected[key] for key in ("sample_id", "rgb_path", "uv_path", "rgb_sha256", "uv_sha256")):
            raise ValueError("Missing RGB/UV identity in train/val manifest")
        if min(int(selected["source_width"]), int(selected["source_height"])) < 1024:
            raise ValueError("Source image below canonical size")
        result[split].append(selected)
    if {key: len(value) for key, value in result.items()} != expected_counts:
        raise ValueError("Full train/val counts do not match frozen protocol")
    ids = [r["sample_id"] for rows in result.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated or overlapping train/val sample IDs")
    groups = [{r["split_group_id"] for r in result[s] if r.get("split_group_id")} for s in ("train", "val")]
    if groups[0] & groups[1]:
        raise ValueError("Train/val record-group overlap")
    return result


class IndexDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def baseline_loader_orders(size, seed, epochs, num_workers=1):
    """Reproduce baseline persistent-loader RNG consumption using index data.

    No images are read. The actual runtime's DataLoader owns the base-seed draw;
    its generator is reset to seed+one_based_epoch before each full iteration.
    Different Torch versions are not assumed to yield identical permutations.
    """
    import torch
    if num_workers < 1:
        raise ValueError("Persistent worker order reproduction requires >=1 worker")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(IndexDataset(size), batch_size=16, shuffle=True,
                                         generator=generator, num_workers=num_workers,
                                         persistent_workers=True, prefetch_factor=1, drop_last=False)
    try:
        for epoch in range(1, epochs + 1):
            generator.manual_seed(seed + epoch)
            order = [int(i) for batch in loader for i in batch.tolist()]
            if len(order) != size or sorted(order) != list(range(size)):
                raise RuntimeError("DataLoader index order did not cover the epoch exactly")
            yield epoch, order
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def prepare(config_path):
    config = validate_config(read_json(config_path))
    out = Path(config["output_dir"])
    if out.exists():
        raise RuntimeError("prepare requires a new output directory; use train --resume auto for this run")
    source = Path(config["source_manifest"])
    if digest(source) != config["source_manifest_sha256"]:
        raise RuntimeError("Canonical source manifest hash mismatch")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        splits = select_training_rows(csv.DictReader(handle))
    code = {name: digest(ROOT / name) for name in CODE_FILES}
    out.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        save_json(out / (split + "_manifest.json"), {"split": split, "rows": rows}, immutable=True)
    (out / "orders").mkdir()
    hashes = {}
    import torch
    for epoch, order in baseline_loader_orders(COUNTS["train"], config["seed"], 20):
        path = out / "orders" / ("epoch_%03d.json" % epoch)
        save_json(path, {"epoch": epoch, "indices": order, "index_order_sha256": order_sha256(order)}, immutable=True)
        hashes[str(epoch)] = digest(path)
        print(json.dumps({"stage": "prepare_index_order", "epoch": epoch, "samples": len(order)}), flush=True)
    protocol = {"schema": SCHEMA, "config": config, "config_sha256": digest(config_path),
                "code_sha256": code, "source_manifest_sha256": digest(source),
                "manifest_sha256": {s: digest(out / (s + "_manifest.json")) for s in COUNTS},
                "epoch_order_file_sha256": hashes, "order_generation_torch": torch.__version__,
                "order_rule": "actual_DataLoader_shuffle_generator_seed_plus_one_based_epoch_persistent_workers",
                "order_worker_count": 1, "original_baseline_order_worker_count": 8,
                "order_note": "Same source split order and DataLoader sampling algorithm; worker-count equivalence tested on CPU. Not bitwise old training replication.",
                "initialization": "from_zero_no_checkpoint_or_teacher", "ema_used": False,
                "objective": "Gaussian_NLL_FP32_only_no_L1_auxiliary", "selection": "full_val_mean_L1",
                "early_stopping": "Separate significant-improvement anchor; strict true-best L1 checkpoint saved independently",
                "canonical_size": 1024, "preprocessing": "same center-square RGB/UV; whole-image PIL Lanczos; uint8/255; no augmentation",
                "train_images": COUNTS["train"], "val_images": COUNTS["val"],
                "test_split_used": False, "test_images_accessed": 0, "random_crop_or_patch": False,
                "source_access_note": "Canonical CSV metadata filtered by split; TEST image paths are not retained or opened",
                "batching": "global16; rank6/5/5; micro2; loss scaling world_size*micro_n/global_n; short tail4=2/1/1",
                "validation": "FP32 batch1, unpadded rank-stride; no DDP forward collectives; per-image streamed resume",
                "validation_keys": list(VAL_KEYS), "amp": "FP16 convolution and FP32 Gaussian likelihood",
                "amp_overflow": "Synchronized gradient-only skip/backoff; 3 consecutive overflows block run; loss nonfinite is fatal",
                "numerical_difference_note": "DDP reduction and microbatching may change floating-point arithmetic relative to physical-batch16 baseline",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_json(out / "protocol.json", protocol, immutable=True)
    save_json(out / "protocol_digest.json", {"sha256": digest(out / "protocol.json")}, immutable=True)
    print(json.dumps({"status": "prepared", "protocol_sha256": digest(out / "protocol.json"), "test_split_used": False}), flush=True)


def load_frozen(config_path):
    config = validate_config(read_json(config_path))
    out = Path(config["output_dir"])
    protocol = read_json(out / "protocol.json")
    phash = digest(out / "protocol.json")
    if read_json(out / "protocol_digest.json")["sha256"] != phash:
        raise RuntimeError("Protocol hash mismatch")
    if protocol["schema"] != SCHEMA or protocol["config_sha256"] != digest(config_path) or protocol["config"] != config:
        raise RuntimeError("Frozen configuration changed")
    for name, sha in protocol["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("Frozen code changed: " + name)
    splits = {}
    for split, count in COUNTS.items():
        path = out / (split + "_manifest.json")
        if digest(path) != protocol["manifest_sha256"][split]:
            raise RuntimeError("Frozen split manifest changed")
        rows = read_json(path)["rows"]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise RuntimeError("Training-only split firewall/coverage failure")
        splits[split] = rows
    return config, out, protocol, phash, splits


def load_order(out, protocol, epoch):
    path = out / "orders" / ("epoch_%03d.json" % epoch)
    if digest(path) != protocol["epoch_order_file_sha256"][str(epoch)]:
        raise RuntimeError("Frozen epoch order changed")
    payload = read_json(path)
    order = payload["indices"]
    if (payload["epoch"] != epoch or len(order) != COUNTS["train"]
            or sorted(order) != list(range(COUNTS["train"])) or order_sha256(order) != payload["index_order_sha256"]):
        raise RuntimeError("Epoch index coverage/hash invalid")
    return order


def update_selection(selection, value, epoch, min_delta=.0001, patience=5):
    """Pure, testable strict-best versus significant-improvement ES logic."""
    if not math.isfinite(value):
        raise FloatingPointError("Validation selection metric is non-finite")
    result = dict(selection)
    best = selection.get("best_val_l1")
    is_best = best is None or value < best
    if is_best:
        result.update(best_val_l1=value, best_epoch=epoch)
    anchor = selection.get("early_stopping_anchor")
    significant = anchor is None or value < anchor - min_delta
    if significant:
        result.update(early_stopping_anchor=value, early_stopping_counter=0)
    else:
        result["early_stopping_counter"] = selection.get("early_stopping_counter", 0) + 1
    return result, is_best, result["early_stopping_counter"] >= patience


class EpochMicroBatchSampler:
    def __init__(self, rank):
        self.rank, self.plans = rank, ()

    def set_plans(self, plans):
        self.plans = tuple(plans)

    def __iter__(self):
        for plan in self.plans:
            for micro in plan.rank_batches[self.rank].microbatches:
                yield list(micro.indices)

    def __len__(self):
        return sum(len(p.rank_batches[self.rank].microbatches) for p in self.plans)


class TrainingOnlyDataset:
    def __init__(self, rows, root, split):
        if split not in COUNTS or any(row.get("split") != split for row in rows):
            raise RuntimeError("TEST/foreign rows prohibited in training dataset")
        from models.highres_patch_dataset import HighResolutionFullImageDataset
        self.rows = rows
        self.base = HighResolutionFullImageDataset(rows, "uv", root, split, canonical_size=1024, allow_test=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.rows[index].get("split") not in COUNTS:
            raise RuntimeError("Training-only split firewall failure")
        return self.base[index]


def seed_worker(worker_id):
    import torch
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_sampler, config, seed):
    import torch
    workers = config["dataloader"].get("num_workers", 4)
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_sampler=batch_sampler, num_workers=workers,
                  pin_memory=config["dataloader"].get("pin_memory", True),
                  generator=generator, worker_init_fn=seed_worker)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=1)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def setup_distributed(config):
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != 3 or rank not in range(3) or local_rank not in range(3):
        raise RuntimeError("Must run via torchrun --nproc_per_node=3 on the authorized node")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    import numpy as np
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    return rank, torch.device("cuda", local_rank)


def capture_rng(device):
    import torch
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    import torch
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def cpu_snapshot(value):
    """Independent CPU copies for disposable numerical-resume verification."""
    import copy
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def state_equal(left, right):
    import torch
    import numpy as np
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def atomic_torch_save(path, payload):
    import torch
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def synchronized_checkpoint(out, protocol_hash, core, optimizer, scheduler, scaler, state,
                            local_accumulator, order, rank, device, best=False):
    import torch.distributed as dist
    rank_payload = {"rank": rank, "rng": capture_rng(device), "epoch_accumulator": local_accumulator}
    rank_states = [None] * 3
    dist.all_gather_object(rank_states, rank_payload)
    if rank == 0:
        payload = {"schema": SCHEMA, "protocol_sha256": protocol_hash, "state": state,
                   "batching_resume": resume_metadata(order, state["next_batch"]),
                   "model_state_dict": core.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "rank_states": rank_states, "ema_used": False, "test_split_used": False,
                   "training_complete": state["phase"] == "complete"}
        if best:
            atomic_torch_save(out / "checkpoint_best.pt", payload)
        # A crash must not leave LAST claiming a new best that was never saved.
        atomic_torch_save(out / "checkpoint_last.pt", payload)
        save_json(out / "history.json", state["history"])
    dist.barrier()


def verify_checkpoint(payload, protocol_hash, order):
    if payload.get("schema") != SCHEMA or payload.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Checkpoint protocol identity mismatch; foreign warm starts prohibited")
    if payload.get("ema_used") is not False or payload.get("test_split_used") is not False:
        raise RuntimeError("Checkpoint violates training protocol")
    if len(payload.get("rank_states", [])) != 3 or sorted(r["rank"] for r in payload["rank_states"]) != [0, 1, 2]:
        raise RuntimeError("Checkpoint per-rank RNG/statistics missing")
    offset = validate_resume_metadata(order, payload["batching_resume"])
    if offset != payload["state"]["next_batch"]:
        raise RuntimeError("Checkpoint batch offset mismatch")
    if payload["state"].get("blocked_reason"):
        raise RuntimeError("Checkpoint explicitly blocked after an unsafe failure; user review required")


def finite_all_ranks(local_finite, device):
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(bool(local_finite)), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def validate_output(output, target):
    """Reject ALL non-finite heads even when the algebraic NLL remains finite."""
    import torch
    required = {"mean", "raw_log_variance", "log_variance", "variance", "sigma"}
    if not isinstance(output, dict) or not required.issubset(output):
        raise RuntimeError("Model output heads are incomplete")
    for key, value in output.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite model output: " + key)
    if output["mean"].shape != target.shape or tuple(output["variance"].shape) != (target.shape[0], 1, target.shape[2], target.shape[3]):
        raise ValueError("Model mean/variance shape mismatch")
    if not bool((output["variance"] > 0).all()) or not bool((output["sigma"] > 0).all()):
        raise FloatingPointError("Non-positive model variance/sigma")
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()) or bool((target > 1).any()):
        raise FloatingPointError("Invalid/nonfinite [0,1] training target")
    if bool((output["mean"] < 0).any()) or bool((output["mean"] > 1).any()):
        raise FloatingPointError("Mean prediction is outside the fixed [0,1] space")


def step_optimizer(model, optimizer, scaler, device, state):
    import torch
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    finite = finite_all_ranks(bool(torch.isfinite(norm).item()), device)
    state["global_attempts"] += 1
    if finite:
        scaler.step(optimizer)
        scaler.update()
        state["global_step"] += 1
        state["consecutive_amp_skips"] = 0
    else:
        state["amp_overflow_skips"] += 1
        state["consecutive_amp_skips"] += 1
        scale = float(scaler.get_scale()) * float(scaler.get_backoff_factor())
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError("AMP scale can no longer be safely reduced")
        scaler.update(new_scale=scale)
        scaler_state = scaler.state_dict()
        scaler_state["_growth_tracker"] = 0
        scaler.load_state_dict(scaler_state)
    optimizer.zero_grad(set_to_none=True)
    return finite, float(norm.item()) if finite else None


def reduced_accumulator(local, device):
    import torch
    import torch.distributed as dist
    value = torch.tensor([local["images"], local["nll_sum"], local["l1_sum"]], dtype=torch.float64, device=device)
    dist.all_reduce(value)
    count, nll, l1 = value.tolist()
    return {"images": int(count), "nll": nll / count if count else None, "l1": l1 / count if count else None}


def load_val_records(path, rows, indices, protocol_hash, state_id):
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if not line.endswith("\n"):
                    raise RuntimeError("Truncated validation record: preserve evidence, no silent truncation")
                record = json.loads(line, parse_constant=reject_constant)
                if position >= len(indices) or record["index"] != indices[position] or record["sample_id"] != rows[indices[position]]["sample_id"]:
                    raise RuntimeError("Validation identity/order mismatch")
                if record["protocol_sha256"] != protocol_hash or record["state_id"] != state_id or set(record["metrics"]) != set(VAL_KEYS):
                    raise RuntimeError("Validation model/protocol mismatch")
                if any(not math.isfinite(float(v)) for v in record["metrics"].values()):
                    raise FloatingPointError("Non-finite persisted validation metric")
                records.append(record)
    return records


def run_validation(core, dataset, rows, out, protocol_hash, state, config, rank, device):
    import torch
    import torch.distributed as dist
    from models.uv_uncertainty_unet import validation_per_image
    indices = list(range(rank, len(rows), 3))
    state_id = "%s:e%d:a%d:s%d" % (protocol_hash, state["epoch"], state["global_attempts"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    done = load_val_records(path, rows, indices, protocol_hash, state_id)
    totals = {key: sum(r["metrics"][key] for r in done) for key in VAL_KEYS}
    loader = make_loader(dataset, [[i] for i in indices[len(done):]], config,
                         config["seed"] + 500000 + state["epoch"] * 3 + rank)
    core.eval()
    start = time.monotonic()
    current_val_index = indices[len(done)] if len(done) < len(indices) else None
    try:
        with torch.inference_mode(), path.open("a", encoding="utf-8") as handle:
            for local_position, batch in enumerate(loader, len(done)):
                index = indices[local_position]
                current_val_index = index
                if batch["sample_id"] != [rows[index]["sample_id"]]:
                    raise RuntimeError("Validation dataset sample identity mismatch")
                rgb = batch["rgb"].to(device, non_blocking=True).float()
                target = batch["target"].to(device, non_blocking=True).float()
                output = core(rgb)  # Bare core: unequal val shard lengths have no DDP collectives.
                validate_output(output, target)
                vectors = validation_per_image(output, target)
                if set(vectors) != set(VAL_KEYS):
                    raise RuntimeError("Validation metric keys differ from the frozen complete set")
                metrics = {}
                for key in VAL_KEYS:
                    value = vectors[key]
                    if tuple(value.shape) != (1,) or not bool(torch.isfinite(value).all()):
                        raise FloatingPointError("Non-finite/invalid validation metric: " + key)
                    metrics[key] = float(value.item())
                    totals[key] += metrics[key]
                record = {"index": index, "sample_id": rows[index]["sample_id"], "metrics": metrics,
                          "protocol_sha256": protocol_hash, "state_id": state_id, "test_split_used": False}
                handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                handle.flush()
                if (local_position + 1) % 25 == 0:
                    os.fsync(handle.fileno())
                    save_json(out / ("validation_progress_rank%d.json" % rank), {
                        "status": "validating", "epoch": state["epoch"], "max_epochs": 20,
                        "completed_images": local_position + 1, "expected_images": len(indices),
                        "seconds_current_run": time.monotonic() - start, "test_split_used": False})
                del output, vectors, rgb, target, batch
                current_val_index = indices[local_position + 1] if local_position + 1 < len(indices) else None
            os.fsync(handle.fileno())
    except Exception:
        save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())), {
            "status": "failed", "rank": rank, "index": current_val_index,
            "sample_id": rows[current_val_index]["sample_id"] if current_val_index is not None else None,
            "traceback": traceback.format_exc(), "protocol_sha256": protocol_hash,
            "state_id": state_id, "test_split_used": False}, immutable=True)
        raise
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    records = load_val_records(path, rows, indices, protocol_hash, state_id)
    if len(records) != len(indices):
        raise RuntimeError("Validation incomplete after write/readback")
    values = torch.tensor([len(records)] + [totals[k] for k in VAL_KEYS], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    all_values = values.tolist()
    if int(all_values[0]) != COUNTS["val"]:
        raise RuntimeError("Full validation must contain exactly 11534 images")
    result = {key: all_values[i + 1] / all_values[0] for i, key in enumerate(VAL_KEYS)}
    result["images"] = int(all_values[0])
    save_json(out / ("validation_progress_rank%d.json" % rank), {
        "status": "complete", "epoch": state["epoch"], "completed_images": len(records),
        "expected_images": len(indices), "record_sha256": digest(path), "test_split_used": False})
    return result


def new_state():
    return {"epoch": 1, "phase": "train", "next_batch": 0, "global_step": 0, "global_attempts": 0,
            "amp_overflow_skips": 0, "consecutive_amp_skips": 0, "history": [],
            "selection": {"best_val_l1": None, "best_epoch": None, "early_stopping_anchor": None,
                          "early_stopping_counter": 0}, "blocked_reason": None}


def new_accumulator():
    return {"images": 0, "nll_sum": 0.0, "l1_sum": 0.0}


def advance_completed_epoch(state):
    """An epoch-complete checkpoint advances once, never repeats scheduler/ES."""
    if state["phase"] == "epoch_complete":
        if state["epoch"] >= 20:
            raise RuntimeError("An epoch20 checkpoint must be marked complete")
        state.update(epoch=state["epoch"] + 1, phase="next_epoch", next_batch=0)
    return state


def final_summary(out, state, protocol_hash):
    return {"status": "complete", "training_complete": True, "epochs_completed": len(state["history"]),
            "max_epochs": 20, "stop_reason": state["stop_reason"], "selection": state["selection"],
            "global_step": state["global_step"], "global_attempts": state["global_attempts"],
            "amp_overflow_skips": state["amp_overflow_skips"], "protocol_sha256": protocol_hash,
            "checkpoint_best": str(out / "checkpoint_best.pt"), "checkpoint_best_sha256": digest(out / "checkpoint_best.pt"),
            "checkpoint_last": str(out / "checkpoint_last.pt"), "checkpoint_last_sha256": digest(out / "checkpoint_last.pt"),
            "test_split_used": False, "test_images_accessed": 0, "ema_used": False}


def log_progress(out, event, rank):
    if rank == 0:
        event = dict(event, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test_split_used=False)
        save_json(out / "training_progress.json", event)
        line = json.dumps(event, allow_nan=False, sort_keys=True)
        print(line, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def train(config_path, resume):
    config, out, protocol, phash, splits = load_frozen(config_path)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models.uv_uncertainty_unet import UVUncertaintyUNet, gaussian_nll_per_image
    rank, device = setup_distributed(config)
    lock = exclusive_lock(out / "training_coordinator.lock") if rank == 0 else contextlib.nullcontext()
    state, accumulator, current_ids = new_state(), new_accumulator(), []
    train_loader = None
    try:
        with lock:
            dist.barrier()
            core = UVUncertaintyUNet(base_channels=32).float().to(device)
            if sum(p.numel() for p in core.parameters()) != config["model"].get("expected_parameters", 7763140):
                raise RuntimeError("Uncertainty model parameter count changed")
            model = DDP(core, device_ids=[device.index], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=65536.0)
            saved_rng = None
            last = out / "checkpoint_last.pt"
            if last.exists():
                if resume == "none":
                    raise RuntimeError("Existing checkpoint: refusing to overwrite with from-zero run")
                payload = torch.load(last, map_location="cpu", weights_only=False)
                state = payload["state"]
                order = load_order(out, protocol, state["epoch"])
                verify_checkpoint(payload, phash, order)
                core.load_state_dict(payload["model_state_dict"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                scheduler.load_state_dict(payload["scheduler_state_dict"])
                scaler.load_state_dict(payload["scaler_state_dict"])
                local = payload["rank_states"][rank]
                if local["rank"] != rank:
                    raise RuntimeError("Rank checkpoint ordering mismatch")
                accumulator, saved_rng = local["epoch_accumulator"], local["rng"]
                del payload
            elif any((out / name).exists() for name in ("checkpoint_best.pt", "history.json", "training_summary.json")):
                raise RuntimeError("Prior training artifacts without last checkpoint: no silent restart")
            if state["phase"] == "complete":
                if rank == 0:
                    summary = final_summary(out, state, phash)
                    save_json(out / "history.json", state["history"])
                    save_json(out / "training_summary.json", summary)
                    log_progress(out, dict(summary, status="already_complete"), rank)
                return
            advance_completed_epoch(state)
            train_dataset = TrainingOnlyDataset(splits["train"], config["dataset_root"], "train")
            val_dataset = TrainingOnlyDataset(splits["val"], config["dataset_root"], "val")
            sampler = EpochMicroBatchSampler(rank)
            train_loader = make_loader(train_dataset, sampler, config, config["seed"] + rank * 1000)
            if saved_rng is not None:
                restore_rng(saved_rng, device)
            save_json(out / ("environment_rank%d_%d.json" % (rank, time.time_ns())), {
                "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device), "rank": rank,
                "parameters": sum(p.numel() for p in core.parameters()), "world_size": 3,
                "tf32": False, "training_amp": "fp16", "nll_precision": "fp32", "pid": os.getpid(),
                "initialization": "this_run_checkpoint_resume" if saved_rng is not None else "from_zero",
                "protocol_sha256": phash}, immutable=True)
            session_start, session_attempts = time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]
                order = load_order(out, protocol, epoch)
                if state["phase"] == "next_epoch":
                    state.update(phase="train", next_batch=0)
                    accumulator = new_accumulator()
                if state["global_attempts"] == 0 and not last.exists():
                    synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                if state["phase"] == "train":
                    plans = tuple(iter_epoch_batches(order, start_batch=state["next_batch"]))
                    sampler.set_plans(plans)
                    batches = iter(train_loader)
                    model.train()
                    for plan in plans:
                        optimizer.zero_grad(set_to_none=True)
                        current_ids = [splits["train"][i]["sample_id"] for i in plan.rank_batches[rank].indices]
                        for micro in plan.rank_batches[rank].microbatches:
                            batch = next(batches)
                            expected_ids = [splits["train"][i]["sample_id"] for i in micro.indices]
                            if batch["sample_id"] != expected_ids:
                                raise RuntimeError("Train microbatch sample identity mismatch")
                            rgb = batch["rgb"].to(device, non_blocking=True).float()
                            target = batch["target"].to(device, non_blocking=True).float()
                            synchronization = contextlib.nullcontext() if micro.synchronize else model.no_sync()
                            with synchronization:
                                failure = None
                                try:
                                    with torch.cuda.amp.autocast(dtype=torch.float16):
                                        output = model(rgb)
                                    with torch.cuda.amp.autocast(enabled=False):
                                        validate_output(output, target)
                                        nll = gaussian_nll_per_image(output, target)
                                        l1 = (output["mean"].float() - target).abs().flatten(1).mean(1)
                                    valid = bool(torch.isfinite(nll).all() and torch.isfinite(l1).all())
                                except Exception:
                                    failure, valid = traceback.format_exc(), False
                                if not finite_all_ranks(valid, device):
                                    raise FloatingPointError("Nonfinite/invalid training forward or likelihood; " + (failure or "another rank failed"))
                                scaler.scale(nll.mean() * micro.loss_scale).backward()
                            accumulator["images"] += len(micro.indices)
                            accumulator["nll_sum"] += float(nll.double().sum().item())
                            accumulator["l1_sum"] += float(l1.double().sum().item())
                            del batch, rgb, target, output, nll, l1
                        successful, gradient_norm = step_optimizer(model, optimizer, scaler, device, state)
                        state["next_batch"] = plan.batch_index + 1
                        if state["consecutive_amp_skips"] >= 3:
                            state["blocked_reason"] = "three_consecutive_amp_gradient_overflows"
                            synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                            raise FloatingPointError(state["blocked_reason"])
                        if state["global_attempts"] % 1000 == 0:
                            synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                        if state["global_attempts"] % 20 == 0 or state["next_batch"] == 1 or not successful:
                            totals = reduced_accumulator(accumulator, device)
                            elapsed = time.monotonic() - session_start
                            per_attempt = elapsed / max(1, state["global_attempts"] - session_attempts)
                            log_progress(out, {"status": "training" if successful else "AMP_OVERFLOW_SKIP", "epoch": epoch,
                                "max_epochs": 20, "epoch_batch": state["next_batch"], "epoch_batches": 5047,
                                "train_images": totals["images"], "train_nll": totals["nll"], "train_l1": totals["l1"],
                                "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                                "amp_overflow_skips": state["amp_overflow_skips"], "gradient_norm": gradient_norm,
                                "loss_scale": scaler.get_scale(), "learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch_train_eta_seconds": per_attempt * (5047 - state["next_batch"]),
                                "eta_note": "observed-session estimate; validation and storage overhead can change it"}, rank)
                    totals = reduced_accumulator(accumulator, device)
                    if totals["images"] != COUNTS["train"] or state["next_batch"] != 5047:
                        raise RuntimeError("Epoch did not consume exactly 80740 train images")
                    state["phase"] = "val"
                    synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                totals = reduced_accumulator(accumulator, device)
                log_progress(out, {"status": "validating", "epoch": epoch, "max_epochs": 20,
                                  "train_images": totals["images"], "train_nll": totals["nll"], "train_l1": totals["l1"],
                                  "global_step": state["global_step"], "amp_overflow_skips": state["amp_overflow_skips"]}, rank)
                validation = run_validation(core, val_dataset, splits["val"], out, phash, state, config, rank, device)
                selection, is_best, early_stop = update_selection(state["selection"], validation["l1"], epoch)
                state["selection"] = selection
                lr_before = optimizer.param_groups[0]["lr"]
                scheduler.step(validation["l1"])
                history_row = {"epoch": epoch, "train_images": totals["images"], "val_images": validation["images"],
                               "train_nll": totals["nll"], "train_l1": totals["l1"], "validation": validation,
                               "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                               "amp_overflow_skips": state["amp_overflow_skips"], "is_best": is_best,
                               "learning_rate": lr_before, "next_learning_rate": optimizer.param_groups[0]["lr"],
                               "test_split_used": False, "random_crop_or_patch": False}
                state["history"].append(history_row)
                if early_stop or epoch == 20:
                    state.update(phase="complete", stop_reason="early_stopping" if early_stop else "max_epochs")
                else:
                    # Save the completed epoch/order first. On resume this phase advances once.
                    state["phase"] = "epoch_complete"
                synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, best=is_best)
                log_progress(out, dict(history_row, status="epoch_complete", max_epochs=20,
                                       best_epoch=selection["best_epoch"], best_val_l1=selection["best_val_l1"],
                                       early_stopping_counter=selection["early_stopping_counter"]), rank)
                if state["phase"] == "epoch_complete":
                    advance_completed_epoch(state)
            if rank == 0:
                summary = final_summary(out, state, phash)
                save_json(out / "training_summary.json", summary)
                log_progress(out, summary, rank)
    except Exception:
        failure = {"status": "failed", "rank": rank, "state": state, "current_train_sample_ids": current_ids,
                   "traceback": traceback.format_exc(), "test_split_used": False}
        save_json(out / ("failure_rank%d_%d.json" % (rank, time.time_ns())), failure, immutable=True)
        raise
    finally:
        if train_loader is not None and getattr(train_loader, "_iterator", None) is not None:
            train_loader._iterator._shutdown_workers()
        if dist.is_initialized():
            dist.destroy_process_group()


def preflight(config_path, output_path=None):
    """Three synthetic updates and FP32 validation; never reuse these weights."""
    config = validate_config(read_json(config_path))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models.uv_uncertainty_unet import UVUncertaintyUNet, gaussian_nll_per_image, validation_per_image
    rank, device = setup_distributed(config)
    directory = Path(output_path) if output_path else Path(config["output_dir"] + "_preflight")
    try:
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        core = UVUncertaintyUNet(base_channels=32).float().to(device)
        if sum(p.numel() for p in core.parameters()) != 7763140:
            raise RuntimeError("Preflight model parameter count mismatch")
        model = DDP(core, device_ids=[device.index], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
        state = new_state()
        plan = next(iter_epoch_batches(range(16)))
        timings, losses = [], []
        torch.cuda.reset_peak_memory_stats(device)
        def synthetic_update():
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for micro in plan.rank_batches[rank].microbatches:
                rgb = torch.rand(len(micro.indices), 3, 1024, 1024, device=device)
                target = torch.rand_like(rgb)
                ctx = contextlib.nullcontext() if micro.synchronize else model.no_sync()
                with ctx:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(rgb)
                    with torch.cuda.amp.autocast(enabled=False):
                        validate_output(output, target)
                        nll = gaussian_nll_per_image(output, target)
                    if not finite_all_ranks(bool(torch.isfinite(nll).all()), device):
                        raise FloatingPointError("Nonfinite synthetic likelihood")
                    scaler.scale(nll.mean() * micro.loss_scale).backward()
                losses.append(float(nll.mean().detach().item()))
                del rgb, target, output, nll
            success, _ = step_optimizer(model, optimizer, scaler, device, state)
            if not success:
                raise FloatingPointError("Synthetic preflight gradient overflow; formal training not approved by this result")
            torch.cuda.synchronize()
            return time.monotonic() - start

        for step in range(3):
            timings.append(synthetic_update())
        peak_train = torch.cuda.max_memory_allocated(device)
        # Disposable in-memory checkpoint: never a formal initialization asset.
        buffer = io.BytesIO()
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": capture_rng(device), "state": state}, buffer)
        synthetic_update()
        expected_weights = cpu_snapshot(core.state_dict())
        expected_optimizer = cpu_snapshot(optimizer.state_dict())
        expected_scaler = cpu_snapshot(scaler.state_dict())
        expected_rng = cpu_snapshot(capture_rng(device))
        expected_state = dict(state)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
        core.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        restore_rng(restored["rng"], device)
        state.clear()
        state.update(restored["state"])
        del restored, buffer
        synthetic_update()
        checks = {"weights": state_equal(expected_weights, core.state_dict()),
                  "optimizer": state_equal(expected_optimizer, optimizer.state_dict()),
                  "scaler": state_equal(expected_scaler, scaler.state_dict()),
                  "rng": state_equal(expected_rng, capture_rng(device)),
                  "counters": state_equal(expected_state, state)}
        if not finite_all_ranks(all(checks.values()), device):
            raise RuntimeError("Synthetic checkpoint/resume equality failed: " + json.dumps(checks))
        del expected_weights, expected_optimizer, expected_scaler, expected_rng
        # Deliberately retain optimizer and gradients: realistic train-to-val memory.
        core.eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            rgb = torch.rand(1, 3, 1024, 1024, device=device)
            target = torch.rand_like(rgb)
            started = time.monotonic()
            output = core(rgb)
            validate_output(output, target)
            vectors = validation_per_image(output, target)
            if set(vectors) != set(VAL_KEYS):
                raise RuntimeError("Preflight metric keys differ from the complete frozen set")
            if any(not bool(torch.isfinite(vectors[key]).all()) for key in VAL_KEYS):
                raise FloatingPointError("Nonfinite synthetic FP32 validation")
            torch.cuda.synchronize()
            inference_seconds = time.monotonic() - started
        result = {"status": "complete", "kind": "synthetic_preflight_not_scientific_training", "rank": rank,
                  "parameters": sum(p.numel() for p in core.parameters()), "gpu": torch.cuda.get_device_name(device),
                  "train_peak_allocated_mib": peak_train / 2 ** 20,
                  "validation_with_optimizer_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
                  "update_seconds": timings, "FP32_validation_seconds": inference_seconds, "synthetic_nll": losses,
                  "effective_batch": 16, "local_samples": len(plan.rank_batches[rank].indices),
                  "optimizer_updates": state["global_step"], "amp_overflow_skips": state["amp_overflow_skips"],
                  "warmup_optimizer_updates": 3, "executed_optimizer_steps_including_replay": 5,
                  "resume_equivalence_pass": True, "resume_equivalence_checks": checks,
                  "resume_probe": "BytesIO state after update3 -> update4 -> restore -> identical update4; weights/scaler/optimizer/RNG/counters",
                  "dataset_accessed": False, "checkpoint_written": False, "test_split_used": False,
                  "config_sha256": digest(config_path), "code_sha256": {p: digest(ROOT / p) for p in CODE_FILES}}
        save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        print(json.dumps(result, allow_nan=False), flush=True)
        dist.barrier()
    except Exception:
        if directory.exists():
            save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())), {
                "status": "failed", "traceback": traceback.format_exc(), "rank": rank,
                "test_split_used": False, "dataset_accessed": False}, immutable=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "preflight", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config)
    elif args.mode == "preflight":
        preflight(args.config, args.preflight_output)
    else:
        train(args.config, args.resume)


if __name__ == "__main__":
    main()
