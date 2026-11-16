"""Independent production runner for the frozen proposed Swin, not the old model.

Synchronous data parallelism uses explicit SUM reductions, rather than padded
DistributedSampler batches or DDP's automatic reducer. Each real image has
weight 1 / actual_global_batch_size, including the final two-image batch.
FP16 overflow halves the common scale and replays the SAME batch and RNG state;
it never advances the sample cursor without an optimizer update.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

MODEL_SHA = "8dd8a7e20cfcc7228c896ef7880a81e6aaac61af858c3edbd1fe679f3c831314"
MANIFEST_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
REFERENCE_SHA = "621e35db9d4049e2dc9dbb2323f86086f53f1705f53f90592b2e6576a7b3f911"
MODEL_CONFIG = {"embed_dim": 96, "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24], "decoder_depths": [6, 2, 2],
                "window_size": 7, "patch_size": 4, "drop_path": 0.0}
ARCHITECTURE = "swin_t_encoder_unet_decoder_gaussian_proposed_v1"
SCHEMA = "swin_redesign_production_20261024_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_model_source(path):
    if sha(path) != MODEL_SHA:
        raise ValueError("Frozen proposed Swin model source SHA mismatch")
    spec = importlib.util.spec_from_file_location("frozen_proposed_swin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_image(path, size=1024):
    """Exact candidate transform; Pillow8 uses the same LANCZOS enum under Image."""
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((size, size), getattr(Image, "Resampling", Image).LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def validate_config(c):
    if c["architecture_id"] != ARCHITECTURE or c["model"] != MODEL_CONFIG:
        raise ValueError("Wrong architecture; reduced embed64/window8 is prohibited")
    if c["source_manifest_sha256"] != MANIFEST_SHA or c["seed"] != 20260817:
        raise ValueError("Frozen data/seed changed")
    if c["expected_counts"] != {"train": 76642, "val": 10948}:
        raise ValueError("Frozen cohort counts changed")
    t = c["training"]
    expected = {"global_batch_size": 16, "epochs": 20, "lr": 1e-4,
                "weight_decay": 1e-4, "grad_clip": 1, "precision": "FP16"}
    if any(t.get(k) != v for k, v in expected.items()) or t["microbatch_size"] != 1:
        raise ValueError("Frozen recipe or microbatch1 execution changed")
    if c["selection_metric"] != "l1":
        raise ValueError("Select checkpoint by full VAL L1")
    if not isinstance(c.get("activation_checkpointing", True), bool):
        raise ValueError("activation_checkpointing must be boolean")
    for k, default in (("num_workers", 2), ("checkpoint_every_steps", 100),
                       ("checkpoint_every_seconds", 300), ("max_amp_retries", 8)):
        value = c.get(k, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if k == "num_workers" else 1):
            raise ValueError("Invalid execution option " + k)


def synchronized_ok(ok, device, message):
    flag = torch.tensor(int(bool(ok)), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def all_finite(ok, device):
    flag = torch.tensor(int(bool(ok)), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def rank0_call(fn, rank, device):
    result, error = None, None
    if rank == 0:
        try:
            result = fn()
        except Exception as exc:
            error = str(exc)
    synchronized_ok(error is None, device, "Rank0 operation failed: " + str(error))
    return result


def allrank_call(fn, device, label):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = repr(exc)
    synchronized_ok(error is None, device, label + ": " + str(error))
    return result


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def gaussian_loss(output, truth):
    e2 = (output["mean"].float() - truth.float()).square().mean(1, keepdim=True)
    lv = output["log_variance"].float()
    return (.5 * (e2 * (-lv).exp() + lv)).flatten(1).mean(1)


def update_batch(model, optimizer, scaler, rgb, truth, actual_global_size,
                 device, microbatch=1, grad_clip=1, max_retries=8):
    """Image-weighted explicit-SUM update; zero-image ranks are supported."""
    model.train()
    initial_rng = capture_rng(device)
    local_size = len(rgb)
    synchronized_ok(rgb.shape == truth.shape and rgb.ndim == 4 and rgb.shape[1] == 3,
                    device, "Input/target batch shape mismatch")
    count = torch.tensor(local_size, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(count)
    if int(count) != actual_global_size:
        raise ValueError("Real global sample count differs from loss denominator")
    max_size = torch.tensor(local_size, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(max_size, op=dist.ReduceOp.MAX)
    if actual_global_size <= 0 or int(max_size) <= 0:
        raise ValueError("Empty global batch")
    for replay in range(max_retries + 1):
        restore_rng(initial_rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, int(max_size), microbatch):
            error = None
            try:
                if start < local_size:
                    x = rgb[start:start + microbatch].to(device, non_blocking=True)
                    y = truth[start:start + microbatch].to(device, non_blocking=True)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                        output = model(x)
                    if any(not bool(torch.isfinite(output[k]).all()) for k in
                           ("mean", "log_variance", "variance", "sigma")):
                        raise FloatingPointError("Nonfinite model output (not an AMP gradient overflow)")
                    losses = gaussian_loss(output, y)
                    if not bool(torch.isfinite(losses).all()):
                        raise FloatingPointError("Nonfinite FP32 likelihood")
                    loss_sum += losses.detach().double().sum()
                    scaler.scale(losses.sum() / actual_global_size).backward()
                    del x, y, output, losses
            except Exception as exc:
                error = repr(exc)
            synchronized_ok(error is None, device, "Forward/backward failed: " + str(error))
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                if dist.is_initialized():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        scaler.unscale_(optimizer)
        finite = all_finite(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                                if p.requires_grad), device)
        norm = None
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            finite = all_finite(bool(torch.isfinite(norm)), device)
        if finite:
            def commit_optimizer():
                scaler.step(optimizer)
                scaler.update()
            allrank_call(commit_optimizer, device, "Optimizer commit failed; retain prior durable checkpoint")
            if dist.is_initialized():
                dist.all_reduce(loss_sum)
            return {"nll": float(loss_sum / actual_global_size), "grad_norm": float(norm),
                    "amp_replays": replay, "loss_scale": scaler.get_scale()}
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite FP32 gradient; optimizer not updated")
        scale = scaler.get_scale() * .5
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError("Invalid AMP loss scale")
        scaler.update(new_scale=scale)
        if replay == max_retries:
            restore_rng(initial_rng, device)
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("AMP replay limit reached; cursor not advanced, no sample skipped")
    raise AssertionError("Unreachable")


class RankBatches(torch.utils.data.Dataset):
    def __init__(self, rows, order, start, root, source, rank, world, size=1024):
        self.rows, self.order, self.start = rows, order, start
        self.root, self.source, self.rank, self.world, self.size = root, source, rank, world, size

    def __len__(self):
        return math.ceil((len(self.order) - self.start) / 16)

    def __getitem__(self, index):
        offset = self.start + index * 16
        indices = self.order[offset:offset + 16]
        local = [self.rows[i] for i in indices[self.rank::self.world]]
        rgb, uv = [], []
        for row in local:
            rgb.append(read_image(Path(self.root) / row["rgb_path"], self.size))
            uv.append(read_image(Path(self.root) / row["uv_path"], self.size))
        empty = lambda: torch.empty((0, 3, self.size, self.size), dtype=torch.float32)
        return {"rgb": torch.stack(rgb) if rgb else empty(),
                "uv": torch.stack(uv) if uv else empty(), "offset": offset,
                "global_size": len(indices)}


def make_loader(dataset, c, rank, epoch, device):
    workers = c.get("num_workers", 2)
    kwargs = dict(batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
                  generator=torch.Generator().manual_seed(c["seed"] + epoch * 100 + rank))
    if workers:
        kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(dataset, **kwargs)


@torch.no_grad()
def validate_model(model, rows, root, source, rank, world, device, progress=None):
    model.eval()
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    local_rows = rows[rank::world]
    for i in range(math.ceil(len(rows) / world)):
        error = None
        try:
            if i < len(local_rows):
                row = local_rows[i]
                rgb = read_image(Path(root) / row["rgb_path"])[None].to(device)
                uv = read_image(Path(root) / row["uv_path"])[None].to(device)
                # Intentionally no autocast: model parameters and validation are FP32.
                output = model(rgb)
                values = torch.stack((gaussian_loss(output, uv).mean(), (output["mean"] - uv).abs().mean()))
                if not bool(torch.isfinite(values).all()) or any(not bool(torch.isfinite(output[k]).all())
                                                              for k in ("variance", "sigma")):
                    raise FloatingPointError("Nonfinite FP32 validation")
                sums[:2] += values.double()
                sums[2] += 1
                del rgb, uv, output, values
                if progress and i % 100 == 0:
                    progress(i + 1)
        except Exception as exc:
            error = repr(exc)
        synchronized_ok(error is None, device, "Validation failed: " + str(error))
    if dist.is_initialized():
        dist.all_reduce(sums)
    if int(sums[2]) != len(rows):
        raise ValueError("Validation coverage mismatch")
    return {"nll": float(sums[0] / sums[2]), "l1": float(sums[1] / sums[2]), "count": int(sums[2])}


def make_optimizer(model, c, device):
    t = c["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                          patience=2, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=65536)
    return optimizer, scheduler, scaler


def save_checkpoint(out, model, optimizer, scheduler, scaler, state, binding, rank, world, device,
                    best=False):
    rng = capture_rng(device)
    rngs = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(rngs, rng)
    else:
        rngs = [rng]
    def write():
        payload = {"schema": SCHEMA, "binding": binding, "world_size": world,
                   "model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "state": copy.deepcopy(state), "rank_rng": rngs}
        if best:
            atomic_checkpoint(out / "checkpoint_best.pt", payload)
        # LAST is the epoch/state commit record. Publish BEST first so a failed
        # BEST write cannot leave LAST claiming an unavailable new best model.
        atomic_checkpoint(out / "checkpoint_last.pt", payload)
    rank0_call(write, rank, device)


def load_checkpoint(path, model, optimizer, scheduler, scaler, binding, rank, world, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("schema") != SCHEMA or p.get("binding") != binding or p.get("world_size") != world:
        raise ValueError("Checkpoint code/config/order/cohort/world-size binding mismatch")
    if len(p["rank_rng"]) != world:
        raise ValueError("Missing rank RNG states")
    model.load_state_dict(p["model"], strict=True)
    optimizer.load_state_dict(p["optimizer"])
    scheduler.load_state_dict(p["scheduler"])
    scaler.load_state_dict(p["scaler"])
    restore_rng(p["rank_rng"][rank], device)
    return p["state"]


def orders_and_binding(c, source_path, world):
    reference = Path(c["reference_protocol"])
    if c.get("reference_protocol_sha256") != REFERENCE_SHA or sha(reference) != REFERENCE_SHA:
        raise ValueError("Original joint clean order protocol identity changed")
    protocol = read_json(reference)
    if (protocol.get("source_manifest_sha256") != MANIFEST_SHA or
            protocol.get("test_split_used") is not False):
        raise ValueError("Original order reference belongs to a different cohort")
    orders = {}
    for epoch in range(1, 21):
        path = Path(c["orders_dir"]) / ("epoch_%03d.json" % epoch)
        if (sha(path) != c["order_sha256"][str(epoch)] or
                c["order_sha256"][str(epoch)] != protocol["epoch_order_file_sha256"][str(epoch)]):
            raise ValueError("Frozen epoch order SHA mismatch")
        indices = read_json(path)["indices"]
        if sorted(indices) != list(range(76642)):
            raise ValueError("Epoch order is not an exact permutation of TRAIN")
        orders[epoch] = indices
    payload = {"config": c, "model_sha256": sha(source_path), "runner_sha256": sha(__file__),
               "world_size": world, "schema": SCHEMA}
    return orders, hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def verify_row_order(c, data):
    reference = Path(c["reference_protocol"])
    protocol = read_json(reference)
    for split in ("train", "val"):
        path = reference.parent / (split + "_manifest.json")
        if sha(path) != protocol["manifest_sha256"][split]:
            raise ValueError("Original reference split manifest changed")
        rows = read_json(path)["rows"]
        if [(r["sample_id"], r["subject_id"]) for r in rows] != [
                (r["sample_id"], r["subject_id"]) for r in data[split]]:
            raise ValueError("Order indices would address a different sample sequence")


def log_progress(out, event):
    event = {"unix_time": time.time(), **event}
    atomic_json(out / "progress.json", event)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def train(c, source, source_path, rank, world, device, resume):
    orders, binding, bound = orders_and_binding(c, source_path, world)
    data = allrank_call(lambda: source.training_data(c), device, "Frozen data verification failed")
    allrank_call(lambda: verify_row_order(c, data), device, "Frozen row order verification failed")
    out = Path(c["output_dir"])
    lock = None
    def prepare():
        nonlocal lock
        import fcntl
        if resume:
            if not (out / "checkpoint_last.pt").is_file():
                raise ValueError("Resume requires this run's checkpoint_last.pt")
            if read_json(out / "binding.json").get("binding") != binding:
                raise ValueError("Existing run binding differs; original files left unchanged")
        else:
            out.mkdir(parents=True, exist_ok=False)
        lock = (out / "training.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out / "binding.json", {"binding": binding, **bound})
        atomic_json(out / "protocol.json", {"schema": SCHEMA, "binding": binding,
            "architecture_id": ARCHITECTURE, "parameters": 41380360,
            "source_manifest_sha256": MANIFEST_SHA, "counts": c["expected_counts"],
            "test_accessed": False, "validation_precision": "FP32",
            "gradient_reduction": "explicit SUM of loss/global-real-count, idle ranks zero gradients",
            "overflow": "synchronized halve-scale and exact batch/RNG replay; no skipped samples",
            "checkpointing": "atomic global-batch-boundary last; only this code/config/world binding resumes",
            "execution_changes": ["microbatch1", "activation checkpointing", "parallel image reads",
                                  "batch-edge resume", "AMP original-batch replay instead of old skip",
                                  "Pillow8 LANCZOS enum compatibility; same crop/resize/uint8-div255"],
            "config": c})
    rank0_call(prepare, rank, device)
    model = allrank_call(lambda: source.SwinUVUncertainty(
        activation_checkpointing=c.get("activation_checkpointing", True)).float().to(device),
        device, "Model construction failed")
    optimizer, scheduler, scaler = make_optimizer(model, c, device)
    state = {"epoch": 1, "cursor": 0, "updates": 0, "amp_replays": 0,
             "best": None, "anchor": None, "bad_epochs": 0, "history": [], "complete": False}
    if resume:
        state = allrank_call(lambda: load_checkpoint(out / "checkpoint_last.pt", model, optimizer,
            scheduler, scaler, binding, rank, world, device), device, "Resume validation failed")
    else:
        save_checkpoint(out, model, optimizer, scheduler, scaler, state, binding, rank, world, device)
    start_time = time.monotonic()
    last_save = start_time
    try:
        while state["epoch"] <= 20 and not state["complete"]:
            epoch = state["epoch"]
            dataset = RankBatches(data["train"], orders[epoch], state["cursor"], c["dataset_root"], source, rank, world)
            loader = iter(make_loader(dataset, c, rank, epoch, device))
            while state["cursor"] < len(data["train"]):
                batch, error = None, None
                try:
                    batch = next(loader)
                    if int(batch["offset"]) != state["cursor"]:
                        raise ValueError("Loader cursor mismatch")
                except Exception as exc:
                    error = repr(exc)
                synchronized_ok(error is None, device, "Image loading failed: " + str(error))
                before = time.monotonic()
                result = update_batch(model, optimizer, scaler, batch["rgb"], batch["uv"],
                                      int(batch["global_size"]), device,
                                      max_retries=c.get("max_amp_retries", 8))
                state["cursor"] += int(batch["global_size"])
                state["updates"] += 1
                state["amp_replays"] += result["amp_replays"]
                if state["updates"] <= 10 or state["updates"] % 10 == 0:
                    rank0_call(lambda: log_progress(out, {"phase": "train", "epoch": epoch, "cursor": state["cursor"],
                        "train_count": len(data["train"]), "updates": state["updates"],
                        "step_seconds": time.monotonic() - before, "elapsed_seconds": time.monotonic() - start_time,
                        **result}), rank, device)
                due = torch.tensor(int(rank == 0 and (state["updates"] in (1, 10)
                    or state["updates"] % c.get("checkpoint_every_steps", 100) == 0
                    or time.monotonic() - last_save >= c.get("checkpoint_every_seconds", 300))), device=device)
                if dist.is_initialized():
                    dist.broadcast(due, 0)
                if bool(due):
                    save_checkpoint(out, model, optimizer, scheduler, scaler, state, binding, rank, world, device)
                    last_save = time.monotonic()
            del loader
            save_checkpoint(out, model, optimizer, scheduler, scaler, state, binding, rank, world, device)
            metrics = validate_model(model, data["val"], c["dataset_root"], source, rank, world, device,
                progress=(lambda n: log_progress(out, {"phase": "validation", "epoch": epoch,
                    "rank0_images_complete": n, "rank0_images_total": len(data["val"][0::world])})) if rank == 0 else None)
            value = metrics["l1"]
            is_best = state["best"] is None or value < state["best"]
            if is_best:
                state["best"] = value
            if state["anchor"] is None or value < state["anchor"] - 1e-4:
                state["anchor"], state["bad_epochs"] = value, 0
            else:
                state["bad_epochs"] += 1
            scheduler.step(value)
            state["history"].append({"epoch": epoch, **metrics})
            state["epoch"], state["cursor"] = epoch + 1, 0
            state["complete"] = epoch == 20 or state["bad_epochs"] >= 5
            save_checkpoint(out, model, optimizer, scheduler, scaler, state, binding, rank, world, device, best=is_best)
            def epoch_finished():
                log_progress(out, {"phase": "complete" if state["complete"] else "epoch_complete",
                    "epoch": epoch, "updates": state["updates"], "amp_replays": state["amp_replays"], **metrics})
                atomic_json(out / "history.json", state["history"])
            rank0_call(epoch_finished, rank, device)
        rank0_call(lambda: atomic_json(out / "training_summary.json", {"binding": binding,
            "state": state, "test_accessed": False}), rank, device)
    except Exception as exc:
        # An optimizer exception can leave partially mutated in-memory parameters.
        # NEVER replace the last durable checkpoint from this failure path.
        rank0_call(lambda: atomic_json(out / "failure.json", {"error": repr(exc),
            "in_memory_state_not_committed": state, "resume_from": "checkpoint_last.pt",
            "last_checkpoint_preserved": True, "time": time.time()}), rank, device)
        raise
    finally:
        if lock is not None:
            lock.close()


def preflight(c, source, source_path, out, rank, world, device):
    orders, binding, bound = orders_and_binding(c, source_path, world)
    data = allrank_call(lambda: source.training_data(c), device, "Preflight data verification failed")
    allrank_call(lambda: verify_row_order(c, data), device, "Preflight row order verification failed")
    rank0_call(lambda: out.mkdir(parents=True, exist_ok=False), rank, device)
    model = allrank_call(lambda: source.SwinUVUncertainty(
        activation_checkpointing=c.get("activation_checkpointing", True)).float().to(device),
        device, "Preflight model construction failed")
    optimizer, scheduler, scaler = make_optimizer(model, c, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ds = RankBatches(data["train"], orders[1][:16], 0, c["dataset_root"], source, rank, world)
    t = time.monotonic()
    full = allrank_call(lambda: ds[0], device, "Preflight TRAIN full16 reads failed")
    result = update_batch(model, optimizer, scaler, full["rgb"], full["uv"], 16, device,
                          max_retries=c.get("max_amp_retries", 8))
    full_seconds = time.monotonic() - t
    train_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    saved = {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
             "scaler": copy.deepcopy(scaler.state_dict()), "rng": capture_rng(device)}
    tail = allrank_call(lambda: RankBatches(data["train"], orders[1][-2:], 0,
        c["dataset_root"], source, rank, world)[0], device, "Preflight TRAIN tail2 reads failed")
    tail_result = update_batch(model, optimizer, scaler, tail["rgb"], tail["uv"], 2, device,
                               max_retries=c.get("max_amp_retries", 8))
    expected = {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
                "scaler": copy.deepcopy(scaler.state_dict()), "rng": capture_rng(device)}
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng"], device)
    replay_result = update_batch(model, optimizer, scaler, tail["rgb"], tail["uv"], 2, device,
                                 max_retries=c.get("max_amp_retries", 8))
    checks = {"model": state_equal(expected["model"], model.state_dict()),
              "optimizer": state_equal(expected["optimizer"], optimizer.state_dict()),
              "scaler": state_equal(expected["scaler"], scaler.state_dict()),
              "rng": state_equal(expected["rng"], capture_rng(device))}
    synchronized_ok(all(checks.values()), device, "Native tail checkpoint replay mismatch")
    del saved, expected
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t = time.monotonic()
    val = validate_model(model, data["val"][:max(world, 1)], c["dataset_root"], source, rank, world, device)
    report = {"rank": rank, "world_size": world, "binding": binding,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20 if device.type == "cuda" else None,
              "train_peak_allocated_mib": train_peak,
              "validation_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "full16_seconds_including_reads": full_seconds, "full16_update": result,
              "tail2_update": tail_result, "tail2_replay_update": replay_result, "resume_replay": checks,
              "fp32_validation": val, "validation_seconds_including_reads": time.monotonic() - t,
              "parameters": sum(p.numel() for p in model.parameters()), "native_shape": [1, 3, 1024, 1024],
              "formal_checkpoint_reused": False, "formal_checkpoint_written": False, "test_accessed": False}
    allrank_call(lambda: atomic_json(out / ("rank%d.json" % rank), report), device, "Preflight rank report write failed")
    if rank == 0:
        atomic_json(out / "preflight.json", {"passed": True, "binding": binding, **bound,
                                             "world_size": world, "all_ranks_reported": True})
        print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--mode", choices=("preflight", "train"), required=True)
    p.add_argument("--preflight-output")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    if not torch.cuda.is_available():
        raise RuntimeError("Production/preflight requires CUDA; use unit tests for CPU")
    rank, world, local = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0))
    if world not in (1, 2, 4, 8) or 16 % world:
        raise ValueError("Execution world size must divide global16")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = load_model_source(args.model_source)
    try:
        if args.mode == "preflight":
            if not args.preflight_output or args.resume:
                p.error("preflight requires new --preflight-output and prohibits --resume")
            preflight(c, source, args.model_source, Path(args.preflight_output), rank, world, device)
        else:
            if not args.preflight_output:
                p.error("train requires successful --preflight-output")
            orders, binding, _ = orders_and_binding(c, args.model_source, world)
            check = read_json(Path(args.preflight_output) / "preflight.json")
            if check.get("binding") != binding or check.get("passed") is not True:
                raise ValueError("Preflight does not bind this exact code/config/world")
            for r in range(world):
                report = read_json(Path(args.preflight_output) / ("rank%d.json" % r))
                if report["binding"] != binding or not all(report["resume_replay"].values()):
                    raise ValueError("Missing or invalid native preflight rank report")
            current = read_json(Path(args.preflight_output) / ("rank%d.json" % rank))
            if current["gpu"] != torch.cuda.get_device_name(device):
                raise ValueError("GPU differs from validated native preflight")
            total = torch.cuda.get_device_properties(device).total_memory / 2**20
            if (abs(total - current["gpu_total_mib"]) > 1 or
                    max(current["train_peak_allocated_mib"], current["validation_peak_allocated_mib"]) >= .95 * total):
                raise ValueError("GPU memory differs or native preflight lacks 5% allocated-memory headroom")
            train(c, source, args.model_source, rank, world, device, args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
