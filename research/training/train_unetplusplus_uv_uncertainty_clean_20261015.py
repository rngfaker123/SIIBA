#!/usr/bin/env python3
"""Frozen from-scratch U-Net++ UV mean/sigma training, train/val only.

Four ranks execute equal 4/4/4/4 image counts using physical batch1. There
are NO DDP forward/backward collectives: gradients are explicitly averaged
once per global batch, after a common failure gate. No empty/padded samples.
This is not bitwise equivalent to original AMP U-Net or physical batch16.
Existing trainers are imported for pure helpers only, never monkeypatched.
"""
import argparse
import contextlib
import csv
import datetime
import random
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import train_uv_uncertainty_fullimage as ref
from models.matched_training_batching import order_sha256

SCHEMA = "unetplusplus_uv_fp32_clean_20261015_v1"
BATCHING = "global16_rank_stride_4_4_4_4_physical1_tail2_1_1_0_0_one_gradient_mean_v1"
COUNTS = {"train": 76642, "val": 10948}
WORLD = 4
EPOCH_BATCHES = (COUNTS["train"] + 15) // 16
SOURCE_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
REFERENCE_CLEAN_SHA = "621e35db9d4049e2dc9dbb2323f86086f53f1705f53f90592b2e6576a7b3f911"
SCIENTIFIC_SOURCE_SHA = {
    "training/configs/uv_uncertainty_unetplusplus_fullimage_1024_v1.0.json": "33d682d4580d2df33cbf2a1d927dbb5fbe0f2cd8f62b422d842bbec5fc58ee16",
    "training/train_uv_uncertainty_unetplusplus_v1.py": "b70052996cd398d38733a80e330a5a9182b02dfa30bd7cd9c86a55451e3d6231",
    "training/train_uv_uncertainty_fullimage.py": "e8f983cd5c6fa73193eab0b1a4ebe58ef5d6897bf824d2d92c35e312418600fc",
    "models/uv_uncertainty_unetplusplus.py": "3797b911113c258edd4e22b0175b7878b0a727e0bac6d600aa9be3b19886e972",
    "models/uv_uncertainty_unet.py": "48c2a9322b90cf8a92ef7458df696b4872c2473fa90f69940173d7129696401c",
    "models/unet_baseline.py": "3112bc3020b2005c2897716b3836099616f10afb34575e23224c5c5160310c05",
    "models/highres_patch_dataset.py": "7dbe88f46b4599926ce57ef384f2aeaf0acf55b2e473895e9b73b9cbdd085af7",
    "models/matched_training_batching.py": "02c664b2375a84bead4c9f5b95a26021c2adb19bcb7042037e5f0a86695f2ac4",
}
PARAMETERS = 9045924
CODE_FILES = ("training/train_unetplusplus_uv_uncertainty_clean_20261015.py",) + tuple(SCIENTIFIC_SOURCE_SHA)
read_json, digest, save_json = ref.read_json, ref.digest, ref.save_json


def runtime():
    import torch
    import numpy
    import PIL
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": numpy.__version__, "pillow": PIL.__version__}


def validate_config(config):
    if config.get("schema") != SCHEMA:
        raise ValueError("Foreign clean U-Net++ schema")
    if config.get("source_manifest_sha256") != SOURCE_SHA or config.get("reference_clean_protocol_sha256") != REFERENCE_CLEAN_SHA:
        raise ValueError("Frozen clean source/order protocol identity changed")
    original = read_json(ROOT / "training/configs/uv_uncertainty_unetplusplus_fullimage_1024_v1.0.json")
    if digest(ROOT / "training/configs/uv_uncertainty_unetplusplus_fullimage_1024_v1.0.json") != SCIENTIFIC_SOURCE_SHA["training/configs/uv_uncertainty_unetplusplus_fullimage_1024_v1.0.json"]:
        raise RuntimeError("Frozen original scientific configuration changed")
    for key in ("seed", "model", "dataloader", "scheduler", "early_stopping", "test_access_allowed", "after_training"):
        if config.get(key) != original[key]:
            raise ValueError("Original scientific configuration changed: " + key)
    if config.get("training") != dict(original["training"], world_size=WORLD):
        raise ValueError("Only world_size may differ from original training recipe")
    expected = {"epochs": 20, "global_batch_size": 16, "microbatch_size": 1,
                "world_size": WORLD, "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
                "amp": False, "precision": "FP32", "checkpoint_every_attempts": 1000,
                "checkpoint_every_seconds": 600}
    if config.get("training") != expected:
        raise ValueError("Unapproved training configuration")
    if config.get("model") != {"base_channels": 32, "expected_parameters": PARAMETERS,
                                "ema": False, "deep_supervision": False}:
        raise ValueError("Unapproved model configuration")
    if config.get("expected_counts") != COUNTS or config.get("seed") != 20260817:
        raise ValueError("Counts/seed differ from clean frozen reference")
    if config.get("scheduler") != {"factor": .5, "patience": 2, "min_lr": 1e-6}:
        raise ValueError("Unapproved scheduler")
    if config.get("early_stopping") != {"patience": 5, "min_delta": .0001}:
        raise ValueError("Unapproved early stopping")
    if config.get("dataloader") != {"num_workers": 4, "prefetch_factor": 1, "pin_memory": True}:
        raise ValueError("Unapproved dataloader")
    if config.get("test_access_allowed") is not False or config.get("after_training") != "stop_no_automatic_test_or_other_training":
        raise ValueError("Scope must stay train/val only")
    if any(config.get(k) for k in ("checkpoint", "pretrained", "warmstart", "resume_checkpoint", "ema", "execution_migration", "test")):
        raise ValueError("Foreign weights are prohibited")
    for key in ("output_dir", "reference_clean_order_dir", "dataset_root", "source_manifest"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError("Missing path " + key)
    if Path(config["output_dir"]).resolve() == Path(config["reference_clean_order_dir"]).resolve():
        raise ValueError("Output must not be the reference run")
    for key in ("source_manifest_sha256", "reference_clean_protocol_sha256"):
        if not isinstance(config.get(key), str) or len(config[key]) != 64 or any(c not in "0123456789abcdef" for c in config[key]):
            raise ValueError("Invalid hash " + key)
    return config


def verify_scientific_sources():
    for name, expected in SCIENTIFIC_SOURCE_SHA.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen original scientific implementation differs: " + name)


def reference_assets(config):
    directory = Path(config["reference_clean_order_dir"])
    if digest(directory / "protocol.json") != config["reference_clean_protocol_sha256"]:
        raise RuntimeError("Clean joint reference protocol hash mismatch")
    protocol = read_json(directory / "protocol.json")
    if (read_json(directory / "protocol_digest.json").get("sha256") != config["reference_clean_protocol_sha256"]
            or protocol.get("test_split_used") is not False
            or protocol.get("config", {}).get("seed") != config["seed"]
            or protocol.get("config", {}).get("expected_counts") != COUNTS):
        raise RuntimeError("Clean joint reference scope/configuration differs")
    if protocol["source_manifest_sha256"] != config["source_manifest_sha256"]:
        raise RuntimeError("Clean source manifest identity mismatch")
    if digest(config["source_manifest"]) != config["source_manifest_sha256"]:
        raise RuntimeError("Source CSV identity changed; no images accessed")
    with Path(config["source_manifest"]).open(encoding="utf-8-sig", newline="") as handle:
        expected_rows = ref.select_training_rows(csv.DictReader(handle), COUNTS)
    files = {}
    for split, count in COUNTS.items():
        name = split + "_manifest.json"
        if digest(directory / name) != protocol["manifest_sha256"][split]:
            raise RuntimeError("Reference manifest changed")
        rows = read_json(directory / name)["rows"]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise RuntimeError("Invalid reference train/val coverage")
        if any(any(row.get(key) != value for key, value in expected.items())
               for row, expected in zip(rows, expected_rows[split])):
            raise RuntimeError("Clean joint manifest order/identity differs from source CSV")
        files[name] = digest(directory / name)
    for epoch in range(1, 21):
        name = "orders/epoch_%03d.json" % epoch
        if digest(directory / name) != protocol["epoch_order_file_sha256"][str(epoch)]:
            raise RuntimeError("Reference epoch order changed")
        load_order(directory, protocol, epoch)
        files[name] = digest(directory / name)
    return directory, protocol, files


def prepare(config_path):
    config = validate_config(read_json(config_path))
    verify_scientific_sources()
    out = Path(config["output_dir"])
    if out.exists():
        raise RuntimeError("Prepare requires new independent directory")
    source, reference, files = reference_assets(config)
    out.mkdir(parents=True, exist_ok=False)
    (out / "orders").mkdir()
    for name, sha in files.items():
        with (source / name).open("rb") as src, (out / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(out / name) != sha:
            raise RuntimeError("Copied frozen reference artifact differs")
    protocol = {"schema": SCHEMA, "config": config, "config_sha256": digest(config_path),
                "code_sha256": {name: digest(ROOT / name) for name in CODE_FILES},
                "software": runtime(), "reference_clean_protocol_sha256": config["reference_clean_protocol_sha256"],
                "source_manifest_sha256": config["source_manifest_sha256"],
                "order_generation_torch": reference["order_generation_torch"],
                "manifest_sha256": reference["manifest_sha256"],
                "epoch_order_file_sha256": reference["epoch_order_file_sha256"],
                "order_rule": "byte_identical_copy_of_clean_joint_frozen_20_orders_no_regeneration",
                "preprocessing": "same_center_square_PIL_Lanczos_1024_uint8_div255_no_augmentation",
                "batching": BATCHING, "gradient_rule": "sum_local_4_times_image_NLL_div_actual_global_n_then_allreduce_mean4_idle_tail_ranks_zero_grad",
                "precision": "all_FP32_no_autocast_no_TF32", "ema_used": False,
                "architecture": "independent_nested_UNetplusplus_base32_no_deep_supervision",
                "initialization": "from_scratch_seed20260817_zero_logvariance_head",
                "objective": "original_Gaussian_NLL_FP32_only_no_L1_auxiliary",
                "selection": "lowest_full_val_mean_UV_L1", "validation": "FP32_batch1_unpadded_rank_stride",
                "training_difference": "original_UNetplusplus_FP32_physical1_same_science_clean_data_four_rank_manual_mean_not_bitwise_three_rank",
                "checkpointing": "initial_first_20th_each1000attempts_or600s_epoch_signal_full_state",
                "test_split_used": False, "test_images_accessed": 0,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_json(out / "protocol.json", protocol, immutable=True)
    save_json(out / "protocol_digest.json", {"sha256": digest(out / "protocol.json")}, immutable=True)
    print(json.dumps({"status": "prepared", "protocol_sha256": digest(out / "protocol.json")}), flush=True)


def load_frozen(config_path):
    config = validate_config(read_json(config_path))
    out = Path(config["output_dir"])
    protocol, phash = read_json(out / "protocol.json"), digest(out / "protocol.json")
    if read_json(out / "protocol_digest.json")["sha256"] != phash or protocol["schema"] != SCHEMA:
        raise RuntimeError("Frozen protocol identity mismatch")
    if protocol["config"] != config or protocol["config_sha256"] != digest(config_path):
        raise RuntimeError("Frozen configuration changed")
    if protocol["software"] != runtime():
        raise RuntimeError("Frozen software differs; do not silently migrate")
    if set(protocol["code_sha256"]) != set(CODE_FILES):
        raise RuntimeError("Frozen code inventory differs")
    for name, sha in protocol["code_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("Frozen code changed: " + name)
    if digest(Path(config["reference_clean_order_dir"]) / "protocol.json") != config["reference_clean_protocol_sha256"]:
        raise RuntimeError("Original reference protocol changed")
    splits = {}
    for split, count in COUNTS.items():
        path = out / (split + "_manifest.json")
        if digest(path) != protocol["manifest_sha256"][split]:
            raise RuntimeError("Frozen manifest changed")
        rows = read_json(path)["rows"]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise RuntimeError("Train/val firewall failed")
        splits[split] = rows
    if {r["sample_id"] for r in splits["train"]} & {r["sample_id"] for r in splits["val"]}:
        raise RuntimeError("Train/val sample overlap")
    return config, out, protocol, phash, splits


def setup_distributed(config):
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != WORLD or rank not in range(WORLD) or local_rank not in range(WORLD):
        raise RuntimeError("Must run via torchrun --nproc_per_node=4 on the authorized node")
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

def run_validation(core, dataset, rows, out, protocol_hash, state, config, rank, device):
    import torch
    import torch.distributed as dist
    from models.uv_uncertainty_unet import validation_per_image
    indices = list(range(rank, len(rows), WORLD))
    state_id = "%s:e%d:a%d:s%d" % (protocol_hash, state["epoch"], state["global_attempts"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    done = ref.load_val_records(path, rows, indices, protocol_hash, state_id)
    totals = {key: sum(r["metrics"][key] for r in done) for key in ref.VAL_KEYS}
    loader = ref.make_loader(dataset, [[i] for i in indices[len(done):]], config,
                         config["seed"] + 500000 + state["epoch"] * WORLD + rank)
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
                ref.validate_output(output, target)
                vectors = validation_per_image(output, target)
                if set(vectors) != set(ref.VAL_KEYS):
                    raise RuntimeError("Validation metric keys differ from the frozen complete set")
                metrics = {}
                for key in ref.VAL_KEYS:
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
    records = ref.load_val_records(path, rows, indices, protocol_hash, state_id)
    if len(records) != len(indices):
        raise RuntimeError("Validation incomplete after write/readback")
    values = torch.tensor([len(records)] + [totals[k] for k in ref.VAL_KEYS], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    all_values = values.tolist()
    if int(all_values[0]) != COUNTS["val"]:
        raise RuntimeError("Full validation must contain exactly 10948 images")
    result = {key: all_values[i + 1] / all_values[0] for i, key in enumerate(ref.VAL_KEYS)}
    result["images"] = int(all_values[0])
    save_json(out / ("validation_progress_rank%d.json" % rank), {
        "status": "complete", "epoch": state["epoch"], "completed_images": len(records),
        "expected_images": len(indices), "record_sha256": digest(path), "test_split_used": False})
    return result


def local_indices(global_indices, rank):
    values = tuple(global_indices)
    if len(values) not in (2, 16) or len(set(values)) != len(values) or rank not in range(WORLD):
        raise ValueError("Only exact full16 or tail2 four-rank batches are approved")
    return values[rank::WORLD]


def epoch_batches(order, start=0):
    if len(order) != COUNTS["train"] or len(set(order)) != len(order) or not 0 <= start <= EPOCH_BATCHES:
        raise ValueError("Invalid frozen epoch or resume offset")
    return [(position, tuple(order[position * 16:(position + 1) * 16])) for position in range(start, EPOCH_BATCHES)]


def batching_metadata(order, next_batch):
    epoch_batches(order, next_batch)
    return {"rule": BATCHING, "order_sha256": order_sha256(order), "global_batch_size": 16,
            "physical_batch_size": 1, "world_size": WORLD, "next_batch": next_batch, "num_samples": len(order)}


def tensor_fingerprint(value):
    """Deterministic nested state digest; no serialization/storage-ID dependence."""
    import torch
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous()
            h.update(str((str(a.dtype), tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda x: str(x)):
                h.update(repr(key).encode()); walk(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for item in v: walk(item)
        else:
            h.update(repr(v).encode())
    walk(value)
    return h.hexdigest()


def assert_replica_equal(core, optimizer):
    import torch.distributed as dist
    local = {"model": tensor_fingerprint(core.state_dict()), "optimizer": tensor_fingerprint(optimizer.state_dict())}
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, local)
    if any(item != gathered[0] for item in gathered):
        raise RuntimeError("Model/optimizer replicas differ")
    return local


def components(config, device):
    import torch
    import torch.distributed as dist
    from models.uv_uncertainty_unetplusplus import UVUncertaintyUNetPlusPlus
    core = UVUncertaintyUNetPlusPlus(base_channels=32).float().to(device)
    if sum(p.numel() for p in core.parameters()) != PARAMETERS:
        raise RuntimeError("Actual model parameter count differs")
    for value in core.state_dict().values():
        dist.broadcast(value, src=0)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    assert_replica_equal(core, optimizer)
    return core, optimizer, scheduler, scaler


def average_gradients(core):
    import torch.distributed as dist
    for parameter in core.parameters():
        if parameter.requires_grad:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(WORLD)


def global_update(core, optimizer, state, accumulator, indices, rank, device, get_pair):
    """Local work contains no collectives; every rank reaches one error gate.

    Each image contributes WORLD/N times its NLL. The final gradient mean across
    four ranks is exactly the image-equal global NLL gradient (up to FP32
    arithmetic), including the two-image tail. Clip only after reduction.
    """
    import torch
    from models.uv_uncertainty_unet import gaussian_nll_per_image
    core.train()
    optimizer.zero_grad(set_to_none=True)
    current, failure = None, None
    local = ref.new_accumulator()
    try:
        local_ids = local_indices(indices, rank)
        if not local_ids:
            # Exact clean tail2: idle ranks contribute zero, never fake images.
            for parameter in core.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)
        for current in local_ids:
            rgb, target = get_pair(current)
            if rgb.shape != (1, 3, 1024, 1024) or target.shape != rgb.shape:
                raise ValueError("Training shape is not physical1 full1024 RGB/UV")
            if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
                raise FloatingPointError("Invalid RGB input")
            output = core(rgb)
            ref.validate_output(output, target)
            nll = gaussian_nll_per_image(output, target)
            l1 = (output["mean"] - target).abs().mean()
            if not bool(torch.isfinite(nll).all()) or not bool(torch.isfinite(l1)):
                raise FloatingPointError("Nonfinite likelihood or L1")
            (nll.mean() * (float(WORLD) / len(indices))).backward()
            local["images"] += 1
            local["nll_sum"] += float(nll.item())
            local["l1_sum"] += float(l1.item())
            del rgb, target, output, nll, l1
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in core.parameters() if p.requires_grad):
            raise FloatingPointError("Missing/nonfinite local FP32 gradient")
    except Exception:
        failure = {"rank": rank, "sample_index": current, "traceback": traceback.format_exc()}
    if not ref.finite_all_ranks(failure is None, device):
        import torch.distributed as dist
        failures = [None] * WORLD
        dist.all_gather_object(failures, failure)
        raise FloatingPointError("Synchronized local update failure; no optimizer update: " + json.dumps(failures))
    average_gradients(core)
    norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    if not ref.finite_all_ranks(bool(torch.isfinite(norm)), device):
        raise FloatingPointError("Nonfinite reduced FP32 gradient; no optimizer update")
    step_failure = None
    try:
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in core.parameters()):
            raise FloatingPointError("Nonfinite model after optimizer update")
        for value in optimizer.state.values():
            if any(isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all()) for v in value.values()):
                raise FloatingPointError("Nonfinite optimizer state after update")
    except Exception:
        step_failure = {"rank": rank, "sample_indices": list(local_indices(indices, rank)), "traceback": traceback.format_exc()}
    if not ref.finite_all_ranks(step_failure is None, device):
        import torch.distributed as dist
        failures = [None] * WORLD
        dist.all_gather_object(failures, step_failure)
        raise FloatingPointError("Synchronized optimizer failure; partial update is NOT checkpointed: " + json.dumps(failures))
    optimizer.zero_grad(set_to_none=True)
    state["global_attempts"] += 1
    state["global_step"] += 1
    for key in local: accumulator[key] += local[key]
    return float(norm.item())


def save_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, loader_generator, best=False):
    import torch.distributed as dist
    replicas = assert_replica_equal(core, optimizer)
    local = {"rank": rank, "rng": ref.capture_rng(device), "epoch_accumulator": accumulator,
             "loader_generator_state": loader_generator.get_state()}
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, local)
    if rank == 0:
        payload = {"schema": SCHEMA, "protocol_sha256": phash, "state": state,
                   "model_state_dict": core.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "rank_states": gathered, "batching_resume": batching_metadata(order, state["next_batch"]),
                   "ema_used": False, "test_split_used": False, "training_complete": state["phase"] == "complete",
                   "replica_fingerprints": replicas}
        if best: ref.atomic_torch_save(out / "checkpoint_best.pt", payload)
        ref.atomic_torch_save(out / "checkpoint_last.pt", payload)
        save_json(out / "checkpoint_commit.json", {"sha256": digest(out / "checkpoint_last.pt"),
                  "epoch": state["epoch"], "phase": state["phase"], "next_batch": state["next_batch"],
                  "global_step": state["global_step"], "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_json(out / "history.json", state["history"])
    dist.barrier()


def verify_checkpoint(payload, phash, order):
    if payload.get("schema") != SCHEMA or payload.get("protocol_sha256") != phash:
        raise RuntimeError("Foreign checkpoint or protocol mismatch")
    if payload.get("ema_used") is not False or payload.get("test_split_used") is not False:
        raise RuntimeError("Checkpoint scope mismatch")
    state = payload["state"]
    if state.get("phase") not in ("train", "val", "epoch_complete", "complete") or not 1 <= state.get("epoch", 0) <= 20:
        raise RuntimeError("Invalid checkpoint phase/epoch")
    if payload["batching_resume"] != batching_metadata(order, state["next_batch"]):
        raise RuntimeError("Checkpoint order/layout/offset mismatch")
    if state.get("blocked_reason") or state["global_step"] != state["global_attempts"] or state["amp_overflow_skips"] != 0:
        raise RuntimeError("Invalid FP32 checkpoint counters or unsafe failure")
    if sorted(r["rank"] for r in payload["rank_states"]) != list(range(WORLD)):
        raise RuntimeError("Missing rank state")
    for row in payload["rank_states"]:
        if not all(k in row for k in ("rng", "epoch_accumulator", "loader_generator_state")):
            raise RuntimeError("Incomplete rank resume state")
        consumed = sum(len(local_indices(indices, row["rank"])) for _, indices in epoch_batches(order)[:state["next_batch"]])
        if row["epoch_accumulator"]["images"] != consumed:
            raise RuntimeError("Rank accumulator disagrees with consumed data offset")
        if any(not math.isfinite(float(row["epoch_accumulator"][k])) for k in ("nll_sum", "l1_sum")):
            raise RuntimeError("Nonfinite saved training accumulators")
    if payload["replica_fingerprints"] != {"model": tensor_fingerprint(payload["model_state_dict"]),
                                         "optimizer": tensor_fingerprint(payload["optimizer_state_dict"])}:
        raise RuntimeError("Checkpoint model/optimizer digest differs")


class StopAtBoundary:
    def __init__(self):
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self.handle)
    def handle(self, *_): self.requested = True


def any_rank_flag(flag, device):
    import torch
    import torch.distributed as dist
    x = torch.tensor(int(flag), dtype=torch.int32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return bool(x.item())


def check_preflight(directory, config_path, protocol_hash, assets=None):
    config, out, protocol, actual_protocol_hash, splits = assets or load_frozen(config_path)
    if actual_protocol_hash != protocol_hash:
        raise RuntimeError("Preflight gate protocol identity differs")
    train_indices = tuple(load_order(out, protocol, 1)[:18])
    replicas = []
    for rank in range(WORLD):
        item = read_json(Path(directory) / ("rank%d.json" % rank))
        if (item.get("status") != "passed" or item.get("rank") != rank or item.get("config_sha256") != digest(config_path)
                or item.get("protocol_sha256") != protocol_hash or item.get("software") != runtime()
                or item.get("code_sha256") != {p: digest(ROOT / p) for p in CODE_FILES}
                or item.get("synthetic_full_tail_replay_pass") is not True or item.get("real_train_val_pass") is not True):
            raise RuntimeError("Incomplete/mismatched synthetic and real preflight gate")
        if item.get("world_size") != WORLD or item.get("local_rank") != rank or item.get("visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("Preflight GPU binding differs from this launch")
        if (item.get("parameters") != PARAMETERS or item.get("global_batch") != 16
                or item.get("physical_batch") != 1 or item.get("tail_global_batch") != 2
                or item.get("rank_full_images") != 4 or item.get("rank_tail_images") != int(rank < 2)
                or item.get("precision") != "FP32" or item.get("amp") is not False
                or item.get("tail2_replay_pass") is not True
                or item.get("post_tail_full_manual_parallel_pass") is not True
                or item.get("formal_checkpoint_written") is not False
                or item.get("checkpoint_weights_reused") is not False
                or item.get("test_split_used") is not False or item.get("test_images_accessed") != 0):
            raise RuntimeError("Native preflight scientific/scope/tail contract differs")
        expected_checks = {"model", "optimizer", "scheduler", "scaler", "rng", "state", "accumulator", "loader_rng", "batching"}
        for key in ("resume_replay_checks", "tail2_resume_replay_checks"):
            if set(item.get(key, {})) != expected_checks or not all(v is True for v in item[key].values()):
                raise RuntimeError("Preflight full-state/tail replay evidence incomplete")
        assigned_train = local_indices(train_indices[:16], rank) + local_indices(train_indices[16:], rank)
        assigned_val = (rank, rank + WORLD)
        expected_real = [{"split": split, "index": i, "sample_id": splits[split][i]["sample_id"]}
                         for split, indices in (("train", assigned_train), ("val", assigned_val)) for i in indices]
        if item.get("real_samples") != expected_real:
            raise RuntimeError("Preflight exact real sample identity/order differs")
        validation = item.get("validation", [])
        if len(validation) != 2 or [r.get("sample_id") for r in validation] != [splits["val"][i]["sample_id"] for i in assigned_val]:
            raise RuntimeError("Preflight validation sample identity differs")
        for row in validation:
            metrics = row.get("metrics", {})
            if set(metrics) != set(ref.VAL_KEYS) or not all(math.isfinite(float(v)) for v in metrics.values()):
                raise RuntimeError("Preflight validation metrics incomplete/nonfinite")
        for key in ("train_peak_allocated_mib", "val_peak_allocated_mib", "synthetic_seconds", "real_train_two_updates_seconds"):
            if not isinstance(item.get(key), (int, float)) or not math.isfinite(item[key]) or item[key] <= 0:
                raise RuntimeError("Preflight resource/timing evidence invalid: " + key)
        fingerprint = item.get("replica_fingerprints", {})
        if set(fingerprint) != {"model", "optimizer"} or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in fingerprint.values()):
            raise RuntimeError("Preflight replica fingerprints missing/invalid")
        replicas.append(fingerprint)
    if any(row != replicas[0] for row in replicas):
        raise RuntimeError("Preflight model/optimizer replicas differ")


def train(config_path, resume="auto", preflight_directory=None):
    config, out, protocol, phash, splits = load_frozen(config_path)
    if preflight_directory is None: raise RuntimeError("Fresh verified --preflight-output directory required")
    check_preflight(preflight_directory, config_path, phash, (config, out, protocol, phash, splits))
    import torch
    import torch.distributed as dist
    rank, device = setup_distributed(config)
    stopper = StopAtBoundary()
    state, accumulator = ref.new_state(), ref.new_accumulator()
    loader = None
    try:
        with ref.exclusive_lock(out / "training_coordinator.lock") if rank == 0 else contextlib.nullcontext():
            dist.barrier()
            core, optimizer, scheduler, scaler = components(config, device)
            generator = torch.Generator().manual_seed(config["seed"] + rank * 1000)
            last = out / "checkpoint_last.pt"
            if last.exists():
                if resume == "none": raise RuntimeError("Refusing to overwrite existing run")
                if digest(last) != read_json(out / "checkpoint_commit.json")["sha256"]:
                    raise RuntimeError("LAST and checkpoint commit differ; preserve both for recovery")
                payload = torch.load(last, map_location="cpu", weights_only=False)
                state = payload["state"]
                order = load_order(out, protocol, state["epoch"])
                verify_checkpoint(payload, phash, order)
                core.load_state_dict(payload["model_state_dict"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                scheduler.load_state_dict(payload["scheduler_state_dict"])
                scaler.load_state_dict(payload["scaler_state_dict"])
                local = payload["rank_states"][rank]
                if local["rank"] != rank: raise RuntimeError("Rank state order mismatch")
                accumulator = local["epoch_accumulator"]
                generator.set_state(local["loader_generator_state"])
                ref.restore_rng(local["rng"], device)
                del payload, local
            elif any((out / p).exists() for p in ("checkpoint_best.pt", "history.json", "training_progress.json", "training_summary.json")):
                raise RuntimeError("Prior artifacts without LAST; no silent restart")
            assert_replica_equal(core, optimizer)
            if state["phase"] == "complete":
                if rank == 0: save_json(out / "training_summary.json", ref.final_summary(out, state, phash))
                return
            ref.advance_completed_epoch(state)
            train_dataset = ref.TrainingOnlyDataset(splits["train"], config["dataset_root"], "train")
            val_dataset = ref.TrainingOnlyDataset(splits["val"], config["dataset_root"], "val")
            save_json(out / ("environment_rank%d_%d.json" % (rank, time.time_ns())), {
                "software": runtime(), "gpu": torch.cuda.get_device_name(device), "rank": rank,
                "pid": os.getpid(), "parameters": PARAMETERS, "world_size": WORLD, "precision": "FP32",
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "protocol_sha256": phash}, immutable=True)
            last_save, session_start, session_attempts = time.monotonic(), time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]
                order = load_order(out, protocol, epoch)
                if state["phase"] == "next_epoch":
                    state.update(phase="train", next_batch=0); accumulator = ref.new_accumulator()
                if not last.exists():
                    save_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, generator)
                if state["phase"] == "train":
                    plans = epoch_batches(order, state["next_batch"])
                    sampler = [[i] for _, indices in plans for i in local_indices(indices, rank)]
                    loader = ref.make_loader(train_dataset, sampler, config, config["seed"] + rank * 1000)
                    loader.generator.set_state(generator.get_state())
                    iterator = iter(loader)
                    generator.set_state(loader.generator.get_state())
                    def get_pair(index):
                        batch = next(iterator)
                        if batch["sample_id"] != [splits["train"][index]["sample_id"]]:
                            raise RuntimeError("DataLoader sample identity mismatch")
                        return batch["rgb"].to(device, non_blocking=True).float(), batch["target"].to(device, non_blocking=True).float()
                    for position, indices in plans:
                        norm = global_update(core, optimizer, state, accumulator, indices, rank, device, get_pair)
                        state["next_batch"] = position + 1
                        stopping = any_rank_flag(stopper.requested, device)
                        due = any_rank_flag(time.monotonic() - last_save >= 600 if rank == 0 else False, device)
                        if state["global_attempts"] in (1, 20) or state["global_attempts"] % 1000 == 0 or due or stopping:
                            save_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, generator)
                            last_save = time.monotonic()
                        if state["global_attempts"] % 20 == 0 or state["global_attempts"] == 1:
                            totals = ref.reduced_accumulator(accumulator, device)
                            secs = (time.monotonic() - session_start) / max(1, state["global_attempts"] - session_attempts)
                            ref.log_progress(out, {"status": "training", "epoch": epoch, "max_epochs": 20,
                                "epoch_batch": state["next_batch"], "epoch_batches": EPOCH_BATCHES, "global_step": state["global_step"],
                                "global_attempts": state["global_attempts"], "amp_overflow_skips": 0, "precision": "FP32",
                                "train_images": totals["images"], "train_l1": totals["l1"], "train_nll": totals["nll"],
                                "gradient_norm": norm, "learning_rate": optimizer.param_groups[0]["lr"],
                                "seconds_per_global_batch": secs, "epoch_train_eta_seconds": secs * (EPOCH_BATCHES - state["next_batch"])}, rank)
                        if stopping:
                            ref.log_progress(out, {"status": "stopped_at_safe_boundary", "epoch": epoch,
                                                   "epoch_batch": state["next_batch"]}, rank)
                            return
                    if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
                    loader = None
                    totals = ref.reduced_accumulator(accumulator, device)
                    if totals["images"] != COUNTS["train"] or state["next_batch"] != EPOCH_BATCHES:
                        raise RuntimeError("Epoch train coverage differs")
                    state["phase"] = "val"
                    save_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, generator)
                totals = ref.reduced_accumulator(accumulator, device)
                ref.log_progress(out, {"status": "validating", "epoch": epoch, "max_epochs": 20,
                                      "global_step": state["global_step"], "amp_overflow_skips": 0}, rank)
                validation = run_validation(core, val_dataset, splits["val"], out, phash, state, config, rank, device)
                selection, is_best, early_stop = ref.update_selection(state["selection"], validation["l1"], epoch)
                state["selection"] = selection
                before = optimizer.param_groups[0]["lr"]
                scheduler.step(validation["l1"])
                row = {"epoch": epoch, "train_images": totals["images"], "val_images": validation["images"],
                       "train_nll": totals["nll"], "train_l1": totals["l1"], "validation": validation,
                       "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                       "amp_overflow_skips": 0, "is_best": is_best, "learning_rate": before,
                       "next_learning_rate": optimizer.param_groups[0]["lr"], "test_split_used": False}
                state["history"].append(row)
                if early_stop or epoch == 20:
                    state.update(phase="complete", stop_reason="early_stopping" if early_stop else "max_epochs")
                else: state["phase"] = "epoch_complete"
                save_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, generator, best=is_best)
                ref.log_progress(out, dict(row, status="epoch_complete", best_epoch=selection["best_epoch"],
                                           best_val_l1=selection["best_val_l1"], early_stopping_counter=selection["early_stopping_counter"]), rank)
                if state["phase"] == "epoch_complete": ref.advance_completed_epoch(state)
            if rank == 0:
                summary = ref.final_summary(out, state, phash)
                save_json(out / "training_summary.json", summary)
                ref.log_progress(out, summary, rank)
    except Exception:
        save_json(out / ("failure_rank%d_%d.json" % (rank, time.time_ns())), {
            "status": "failed", "rank": rank, "state": state, "traceback": traceback.format_exc(),
            "test_split_used": False}, immutable=True)
        raise
    finally:
        if loader is not None and getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
        if dist.is_initialized(): dist.destroy_process_group()


def preflight(config_path, output_path):
    config, out, protocol, phash, splits = load_frozen(config_path)
    import torch
    import torch.distributed as dist
    from models.uv_uncertainty_unet import validation_per_image
    rank, device = setup_distributed(config)
    directory = Path(output_path)
    if directory.resolve() == out.resolve() or out.resolve() in directory.resolve().parents:
        raise RuntimeError("Disposable preflight must be outside formal experiment")
    try:
        if rank == 0: directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        core, optimizer, scheduler, scaler = components(config, device)
        state, accumulator = ref.new_state(), ref.new_accumulator()
        generator = torch.Generator().manual_seed(config["seed"] + rank * 1000)
        torch.cuda.reset_peak_memory_stats(device)
        def synthetic(_):
            rgb = torch.rand(1, 3, 1024, 1024, device=device)
            return rgb, torch.rand_like(rgb)
        started = time.monotonic()
        global_update(core, optimizer, state, accumulator, tuple(range(16)), rank, device, synthetic)
        # Exercise optimizer/scheduler/scaler/RNG/offset/statistic replay, with tail2.
        state["next_batch"] = 1
        scheduler.step(.1)
        snap = {"model": core.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": ref.capture_rng(device), "state": state,
                "accumulator": accumulator, "loader_rng": generator.get_state()}
        buffer = io.BytesIO(); torch.save(snap, buffer)
        def tail_step():
            global_update(core, optimizer, state, accumulator, tuple(range(16, 18)), rank, device, synthetic)
            state["next_batch"] = 2
            scheduler.step(.2)
            _ = torch.rand(3, generator=generator)
        def snapshot():
            return ref.cpu_snapshot({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": ref.capture_rng(device),
                "state": state, "accumulator": accumulator, "loader_rng": generator.get_state(),
                "batching": {"world_size": WORLD, "global_batch": 16, "physical_batch": 1,
                             "next_batch": state["next_batch"], "local_images": accumulator["images"]}})
        def replay_sequence():
            tail_step()
            tail = snapshot()
            # Returning to a full batch proves zero-sample tail ranks recover
            # normal backward/update behavior without leftover gradient state.
            global_update(core, optimizer, state, accumulator, tuple(range(32, 48)), rank, device, synthetic)
            state["next_batch"] = 3
            scheduler.step(.3)
            _ = torch.rand(3, generator=generator)
            return tail, snapshot()
        expected_tail, expected = replay_sequence()
        buffer.seek(0); restored = torch.load(buffer, map_location="cpu", weights_only=False)
        core.load_state_dict(restored["model"], strict=True); optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"]); scaler.load_state_dict(restored["scaler"])
        ref.restore_rng(restored["rng"], device); generator.set_state(restored["loader_rng"])
        state, accumulator = restored["state"], restored["accumulator"]
        del restored, buffer, snap
        observed_tail, observed = replay_sequence()
        checks = {key: ref.state_equal(expected[key], observed[key]) for key in expected}
        tail_checks = {key: ref.state_equal(expected_tail[key], observed_tail[key]) for key in expected_tail}
        if not ref.finite_all_ranks(all(checks.values()) and all(tail_checks.values()), device):
            raise RuntimeError("Complete-state synthetic replay failed: " + json.dumps(checks))
        del expected_tail, expected, observed_tail, observed
        replica = assert_replica_equal(core, optimizer)
        synthetic_seconds = time.monotonic() - started
        # Fixed real data from frozen epoch1, plus two distinct val images/rank.
        order = load_order(out, protocol, 1)
        train_indices = tuple(order[:18])
        val_indices = (rank, rank + WORLD)
        image_hashes = []
        for split, indices in (("train", local_indices(train_indices[:16], rank) + local_indices(train_indices[16:], rank)), ("val", val_indices)):
            for i in indices:
                row = splits[split][i]
                for modality in ("rgb", "uv"):
                    path = Path(config["dataset_root"]) / row[modality + "_path"]
                    if digest(path) != row[modality + "_sha256"]: raise RuntimeError("Real preflight source hash mismatch: " + row["sample_id"])
                image_hashes.append({"split": split, "sample_id": row["sample_id"], "index": i})
        dataset = ref.TrainingOnlyDataset(splits["train"], config["dataset_root"], "train")
        def real_pair(index):
            row = dataset[index]
            if row["sample_id"] != splits["train"][index]["sample_id"]: raise RuntimeError("Real preflight identity mismatch")
            return row["rgb"].unsqueeze(0).to(device).float(), row["target"].unsqueeze(0).to(device).float()
        started = time.monotonic()
        for indices in (train_indices[:16], train_indices[16:]):
            global_update(core, optimizer, state, accumulator, indices, rank, device, real_pair)
        real_seconds = time.monotonic() - started
        peak_train = torch.cuda.max_memory_allocated(device) / 2 ** 20
        replica = assert_replica_equal(core, optimizer)
        dataset = ref.TrainingOnlyDataset(splits["val"], config["dataset_root"], "val")
        core.eval(); torch.cuda.reset_peak_memory_stats(device)
        vals = []
        with torch.inference_mode():
            for index in val_indices:
                row = dataset[index]
                rgb, target = row["rgb"].unsqueeze(0).to(device).float(), row["target"].unsqueeze(0).to(device).float()
                output = core(rgb); ref.validate_output(output, target)
                vectors = validation_per_image(output, target)
                if set(vectors) != set(ref.VAL_KEYS) or any(not bool(torch.isfinite(v).all()) for v in vectors.values()):
                    raise FloatingPointError("Invalid real FP32 validation metrics")
                vals.append({"sample_id": row["sample_id"], "metrics": {k: float(v.item()) for k, v in vectors.items()}})
                del rgb, target, output, vectors
        result = {"status": "passed", "rank": rank, "kind": "disposable_full1024_FP32_synthetic_and_real_train_val_preflight",
                  "synthetic_full_tail_replay_pass": True, "real_train_val_pass": True,
                  "resume_equivalence_checks": checks, "replica_fingerprints": replica,
                  "resume_replay_checks": checks, "tail2_resume_replay_checks": tail_checks,
                  "tail2_replay_pass": True, "post_tail_full_manual_parallel_pass": True,
                  "precision": "FP32", "amp": False, "test_images_accessed": 0,
                  "parameters": PARAMETERS, "gpu": torch.cuda.get_device_name(device),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "world_size": WORLD,
                  "local_rank": int(os.environ["LOCAL_RANK"]),
                  "global_batch": 16, "tail_global_batch": 2, "physical_batch": 1,
                  "rank_full_images": len(local_indices(range(16), rank)), "rank_tail_images": len(local_indices(range(2), rank)),
                  "synthetic_seconds": synthetic_seconds, "real_train_two_updates_seconds": real_seconds,
                  "train_peak_allocated_mib": peak_train,
                  "val_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
                  "real_samples": image_hashes, "validation": vals, "formal_checkpoint_written": False,
                  "checkpoint_weights_reused": False, "test_split_used": False,
                  "config_sha256": digest(config_path), "protocol_sha256": phash, "software": runtime(),
                  "code_sha256": {p: digest(ROOT / p) for p in CODE_FILES}}
        save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        if read_json(directory / ("rank%d.json" % rank)) != result: raise RuntimeError("Preflight JSON readback differs")
        dist.barrier()
        print(json.dumps({"rank": rank, "status": "passed", "train_peak_allocated_mib": peak_train}), flush=True)
    except Exception:
        if directory.exists():
            save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())), {
                "status": "failed", "rank": rank, "traceback": traceback.format_exc(), "test_split_used": False}, immutable=True)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "preflight", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.config)
    elif args.mode == "preflight":
        if args.preflight_output is None: parser.error("--preflight-output is required")
        preflight(args.config, args.preflight_output)
    else: train(args.config, args.resume, args.preflight_output)


if __name__ == "__main__": main()

