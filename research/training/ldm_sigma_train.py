"""Production phase C: frozen RGB->DDIM20->UV plus a learned Gaussian head.

Only the new 32->1 convolution is optimized. No UV enters model.forward; UV is
used by the FP32 Gaussian likelihood after generation. Outputs are independent
of old sampling-spread results. Explicit gradient SUM supports idle tail ranks.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import PIL
import torch
import torch.distributed as dist

import sigma_gate as gate

SCHEMA = "ldm_sigma_production_20261024_v1"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def sync_ok(ok, device, message):
    value = torch.tensor(int(bool(ok)), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if not value.item():
        raise RuntimeError(message)


def rank0_call(fn, rank, device):
    error = None
    if rank == 0:
        try:
            fn()
        except Exception as exc:
            error = repr(exc)
    sync_ok(error is None, device, "Rank0 IO failed: " + str(error))


def allrank_call(fn, device, label):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = repr(exc)
    sync_ok(error is None, device, label + ": " + str(error))
    return result


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(rng, device):
    random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], device)


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


def tensor_fingerprint(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().contiguous()
            h.update(str((a.dtype, tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def frozen_fingerprint(model):
    return {"ae": tensor_fingerprint(model.ae.state_dict()),
            "denoiser": tensor_fingerprint(model.denoiser.state_dict())}


def verify_trainables(model):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if [name for name, unused in named] != ["log_variance_head.weight", "log_variance_head.bias"]:
        raise ValueError("Only the new variance head may require gradients")
    if sum(p.numel() for unused, p in named) != 33:
        raise ValueError("Expected 32->1 pointwise variance head, 33 parameters")
    if model.ae.training or model.denoiser.training:
        raise ValueError("Frozen generator must stay in evaluation mode")
    if any(p.grad is not None for m in (model.ae, model.denoiser) for p in m.parameters()):
        raise ValueError("Frozen generator received gradients")
    return [p for unused, p in named]


def gaussian_loss(output, truth):
    e2 = (output["mean"].float() - truth.float()).square().mean(1, keepdim=True)
    lv = output["log_variance"].float()
    return (0.5 * (e2 * (-lv).exp() + lv)).flatten(1).mean(1)


def checked_prediction(model, rgb, sample_ids):
    # Deliberately no target/truth argument: actual RGB-conditioned generation.
    output = model(rgb, sample_ids)
    if output["mean"].shape != rgb.shape or output["sigma"].shape != (len(rgb), 1, *rgb.shape[-2:]):
        raise ValueError("Unexpected mean/sigma shape")
    if any(not bool(torch.isfinite(output[k]).all()) for k in ("mean", "log_variance", "variance", "sigma")):
        raise FloatingPointError("Nonfinite frozen generation/head output")
    if not bool((output["variance"] > 0).all()):
        raise FloatingPointError("Variance not positive")
    return output


def update_batch(model, optimizer, batch, device):
    model.train()
    trainables = allrank_call(lambda: verify_trainables(model), device, "Verify trainable head")
    optimizer.zero_grad(set_to_none=True)
    local = len(batch["sample_ids"])
    maximum = torch.tensor(local, dtype=torch.int64, device=device)
    count = torch.tensor(local, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(count)
    actual = int(batch["global_size"])
    if actual <= 0 or int(maximum) <= 0 or int(count) != actual:
        raise ValueError("Empty/mismatched actual global batch count")
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for i in range(int(maximum)):
        error = None
        try:
            if i < local:
                rgb = batch["rgb"][i:i+1].to(device)
                truth = batch["uv"][i:i+1].to(device)
                # No autocast: generator, head, loss and gradients are FP32.
                output = checked_prediction(model, rgb, [batch["sample_ids"][i]])
                loss = gaussian_loss(output, truth).sum()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite FP32 Gaussian likelihood")
                (loss / actual).backward()
                total += torch.stack((loss.detach().double(),
                                      (output["mean"] - truth).abs().mean().detach().double()))
                del rgb, truth, output, loss
        except Exception as exc:
            error = repr(exc)
        sync_ok(error is None, device, "Generation/backward failed: " + str(error))
    for p in trainables:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if dist.is_initialized():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    sync_ok(all(bool(torch.isfinite(p.grad).all()) for p in trainables), device, "Nonfinite head gradient")
    norm = torch.nn.utils.clip_grad_norm_(trainables, 1.)
    sync_ok(bool(torch.isfinite(norm)), device, "Nonfinite head gradient norm")
    allrank_call(lambda: verify_trainables(model), device, "Verify frozen gradients")
    error = None
    try:
        optimizer.step()
    except Exception as exc:
        error = repr(exc)
    sync_ok(error is None, device, "Optimizer step failed; do not checkpoint uncommitted step: " + str(error))
    if dist.is_initialized():
        dist.all_reduce(total)
    return {"nll": float(total[0] / actual), "l1": float(total[1] / actual), "grad_norm": float(norm)}


class RankBatches(torch.utils.data.Dataset):
    def __init__(self, rows, order, cursor, root, source, rank, world, size=1024):
        self.rows, self.order, self.cursor = rows, order, cursor
        self.root, self.source, self.rank, self.world, self.size = root, source, rank, world, size

    def __len__(self):
        return math.ceil((len(self.order) - self.cursor) / 16)

    def __getitem__(self, index):
        cursor = self.cursor + 16 * index
        indices = self.order[cursor:cursor+16]
        selected = [self.rows[j] for j in indices[self.rank::self.world]]
        rgb = [self.source.read_image(Path(self.root) / r["rgb_path"], self.size) for r in selected]
        uv = [self.source.read_image(Path(self.root) / r["uv_path"], self.size) for r in selected]
        empty = lambda: torch.empty((0, 3, self.size, self.size), dtype=torch.float32)
        return {"rgb": torch.stack(rgb) if rgb else empty(), "uv": torch.stack(uv) if uv else empty(),
                "sample_ids": [r["sample_id"] for r in selected], "offset": cursor, "global_size": len(indices)}


def make_loader(dataset, c, rank, epoch, device):
    workers = c.get("num_workers", 2)
    options = {"batch_size": None, "num_workers": workers, "pin_memory": device.type == "cuda",
               "generator": torch.Generator().manual_seed(c["seed"] + epoch * 100 + rank)}
    if workers:
        options["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(dataset, **options)


@torch.no_grad()
def validate(model, rows, c, source, rank, world, device, out=None, epoch=0, binding=""):
    model.eval()
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    local = rows[rank::world]
    error, done, handle = None, [], None
    try:
        state_id = hashlib.sha256((binding + str(epoch) + tensor_fingerprint(model.log_variance_head.state_dict())).encode()).hexdigest()
        if out is not None:
            directory = out / "validation" / ("epoch_%03d" % epoch)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("rank%d.jsonl" % rank)
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if not line.endswith("\n"):
                            raise ValueError("Truncated validation record; preserve and inspect")
                        row = json.loads(line)
                        if i >= len(local) or row["sample_id"] != local[i]["sample_id"] or row["state_id"] != state_id:
                            raise ValueError("Resumed VAL identity/model mismatch")
                        if not math.isfinite(row["nll"]) or not math.isfinite(row["l1"]):
                            raise ValueError("Nonfinite saved VAL")
                        done.append(row)
                for row in done:
                    sums += torch.tensor([row["nll"], row["l1"], 1], device=device, dtype=torch.float64)
            handle = path.open("a", encoding="utf-8")
    except Exception as exc:
        error = repr(exc)
    try:
        sync_ok(error is None, device, "Validation resume records failed: " + str(error))
        # All ranks visit the same chunk boundaries, including idle ranks and
        # shards with a longer saved prefix. A failure cannot wait through an
        # entire DDIM20 validation pass before its peers see it.
        for start in range(0, math.ceil(len(rows)/world), 25):
            error = None
            try:
                for index in range(max(start, len(done)), min(start+25, len(local))):
                    row = local[index]
                    rgb = source.read_image(Path(c["dataset_root"]) / row["rgb_path"])[None].to(device)
                    truth = source.read_image(Path(c["dataset_root"]) / row["uv_path"])[None].to(device)
                    prediction = checked_prediction(model, rgb, [row["sample_id"]])
                    nll = float(gaussian_loss(prediction, truth).mean())
                    l1 = float((prediction["mean"] - truth).abs().mean())
                    if not math.isfinite(nll) or not math.isfinite(l1):
                        raise FloatingPointError("Nonfinite VAL")
                    record = {"sample_id": row["sample_id"], "state_id": state_id, "nll": nll, "l1": l1}
                    if handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
                    sums += torch.tensor([nll, l1, 1], device=device, dtype=torch.float64)
                    done.append(record)
                    del rgb, truth, prediction
                if out is not None:
                    os.fsync(handle.fileno())
                    atomic_json(out / ("validation_progress_rank%d.json" % rank),
                                {"epoch": epoch, "completed": len(done), "total": len(local), "state_id": state_id})
            except Exception as exc:
                error = repr(exc)
            sync_ok(error is None, device, "Validation chunk failed: " + str(error))
    finally:
        if handle:
            handle.close()
    if dist.is_initialized():
        dist.all_reduce(sums)
    if int(sums[2]) != len(rows):
        raise ValueError("Incomplete VAL coverage")
    return {"nll": float(sums[0] / sums[2]), "l1": float(sums[1] / sums[2]), "images": int(sums[2])}


def save(out, model, optimizer, scheduler, state, binding, rank, world, device, best=False):
    rng, ranks = capture_rng(device), [None] * world
    fingerprint = tensor_fingerprint(model.log_variance_head.state_dict())
    if dist.is_initialized():
        dist.all_gather_object(ranks, {"rng": rng, "head_sha256": fingerprint})
    else:
        ranks = [{"rng": rng, "head_sha256": fingerprint}]
    sync_ok(all(r["head_sha256"] == fingerprint for r in ranks), device, "Head replicas differ")
    def write():
        payload = {"schema": SCHEMA, "binding": binding, "world_size": world,
                   "checkpoint_kind": "posttraining_sigma_head_only",
                   "head_state_dict": cpu_copy(model.log_variance_head.state_dict()),
                   "optimizer": cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                   "state": copy.deepcopy(state), "rank_rng": [r["rng"] for r in ranks]}
        if best:
            atomic_checkpoint(out / "checkpoint_best.pt", payload)
        # Write BEST first. If interrupted, LAST remains an earlier safe state
        # that deterministically replays validation; never mark completion while
        # the corresponding new BEST has not been committed.
        atomic_checkpoint(out / "checkpoint_last.pt", payload)
    rank0_call(write, rank, device)


def resume_checkpoint(path, model, optimizer, scheduler, binding, rank, world, device):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    if cp.get("schema") != SCHEMA or cp.get("binding") != binding or cp.get("world_size") != world:
        raise ValueError("Resume source/assets/config/world binding differs")
    if cp.get("checkpoint_kind") != "posttraining_sigma_head_only" or len(cp["rank_rng"]) != world:
        raise ValueError("Wrong checkpoint type or missing RNG")
    state = cp["state"]
    if state["cursor"] not in list(range(0, 76642, 16)) + [76642] or not 1 <= state["epoch"] <= 21:
        raise ValueError("Invalid batch-boundary cursor")
    if state["updates"] != (state["epoch"]-1)*4791 + math.ceil(state["cursor"]/16):
        raise ValueError("Update count and sample cursor differ")
    if len(state["history"]) != state["epoch"]-1 or (state["epoch"] == 21 and not state["complete"]):
        raise ValueError("Checkpoint history/complete flag differs")
    model.log_variance_head.load_state_dict(cp["head_state_dict"], strict=True)
    optimizer.load_state_dict(cp["optimizer"]); scheduler.load_state_dict(cp["scheduler"])
    restore_rng(cp["rank_rng"][rank], device)
    return state


def load_source(path):
    if gate.sha(path) != gate.MODEL_SHA:
        raise ValueError("Frozen LDM model/core source differs")
    spec = importlib.util.spec_from_file_location("ldm_proposed_frozen", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Pillow 8.2 names the same integer interpolation constant Image.LANCZOS.
    # An API alias preserves exact pixel operations without changing frozen code.
    if not hasattr(module.Image, "Resampling"):
        module.Image.Resampling = module.Image
    return module


def verified_data(source, config):
    data = source.training_data(config)
    keys = ("sample_id", "subject_id", "split", "rgb_path", "uv_path")
    for split in ("train", "val"):
        frozen = gate.read_json(gate.check_file(config["gate_assets"][split + "_manifest"]))["rows"]
        actual = data[split]
        if len(actual) != len(frozen) or any(tuple(a[k] for k in keys) != tuple(b[k] for k in keys)
                                             for a, b in zip(actual, frozen)):
            raise ValueError("CSV TRAIN/VAL order/paths differ from frozen index authority")
    return data


def config_and_binding(c, model_source, world):
    if c["architecture_id"] != gate.ARCHITECTURE or c["source_manifest_sha256"] != gate.MANIFEST_SHA:
        raise ValueError("Wrong scientific identity")
    if c["seed"] != 20260817 or c["expected_counts"] != {"train": 76642, "val": 10948}:
        raise ValueError("Wrong seed/counts")
    if c["selection_metric"] != "nll" or c["training"] != {
            "epochs": 20, "global_batch_size": 16, "microbatch_size": 1, "lr": 1e-4,
            "weight_decay": 1e-4, "grad_clip": 1, "precision": "FP32"}:
        raise ValueError("Frozen phase C recipe differs")
    for name, default in (("num_workers", 2), ("checkpoint_every_steps", 20), ("checkpoint_every_seconds", 300)):
        value = c.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "num_workers" else 1):
            raise ValueError("Invalid execution setting " + name)
    orders = {}
    for epoch in range(1, 21):
        path = Path(c["orders_dir"]) / ("epoch_%03d.json" % epoch)
        if gate.sha(path) != c["order_sha256"][str(epoch)]:
            raise ValueError("Frozen epoch order hash changed")
        indices = gate.read_json(path)["indices"]
        if sorted(indices) != list(range(76642)):
            raise ValueError("Order is not exact TRAIN permutation")
        orders[epoch] = indices
    bound = {"schema": SCHEMA, "config": c, "world_size": world,
             "source_sha256": {"model": gate.sha(model_source), "runner": gate.sha(__file__), "gate": gate.sha(gate.__file__)},
             "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "numpy": np.__version__, "pillow": PIL.__version__}}
    binding = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    return orders, binding, bound


def progress(out, record):
    record = {"unix_time": time.time(), "test_accessed": False, **record}
    atomic_json(out / "progress.json", record)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


def optimizer_for(model):
    optimizer = torch.optim.AdamW(verify_trainables(model), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
    return optimizer, scheduler


def train(c, source, model_source, rank, world, device, resume):
    orders, binding, bound = allrank_call(lambda: config_and_binding(c, model_source, world), device, "Bind configuration")
    gate_report = allrank_call(lambda: gate.verify_assets(c), device, "Frozen A/B gate")
    data = allrank_call(lambda: verified_data(source, c), device, "Frozen data gate")
    out, lock = Path(c["output_dir"]), None
    def prepare():
        nonlocal lock
        import fcntl
        if resume:
            if gate.read_json(out / "binding.json").get("binding") != binding:
                raise ValueError("Existing run differs; keep original intact")
        else:
            out.mkdir(parents=True, exist_ok=False)
        lock = (out / "training.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out / "binding.json", {"binding": binding, **bound})
        atomic_json(out / "protocol.json", {"binding": binding, **bound, "gate_report": gate_report,
            "trainable_parameters": 33, "trainable": "new_1x1_sigma_head_only", "test_accessed": False,
            "input": "TRAIN_RGB_actual_DDIM20_generation_no_truth_in_forward", "precision": "FP32",
            "loss": "image_equal_mean[0.5*(mean_RGB((mu-truth)^2)/variance+log(variance))]",
            "variance": "exp(s)+1e-6", "selection": "minimum_complete_VAL10948_NLL",
            "generator": "frozen_AE_and_BEST_denoiser;generation_index0;seed20260817",
            "sampling_spread_K8_is_not_this_sigma": True, "large_prediction_cache": False,
            "checkpoint": "atomic_head_only_plus_optimizer_scheduler_allrankRNG_bound_to_frozen_generator",
            "phase_C_additional_compute_not_equal_end_to_end_recipe": True})
    rank0_call(prepare, rank, device)
    model = allrank_call(lambda: source.build_training_model(c, device), device, "Load frozen generator")
    optimizer, scheduler = allrank_call(lambda: optimizer_for(model), device, "Initialize head optimizer")
    state = {"epoch": 1, "cursor": 0, "updates": 0, "best": None, "anchor": None,
             "bad_epochs": 0, "history": [], "complete": False}
    if resume:
        state = allrank_call(lambda: resume_checkpoint(out / "checkpoint_last.pt", model, optimizer, scheduler,
                                                     binding, rank, world, device), device, "Resume phase C")
    else:
        save(out, model, optimizer, scheduler, state, binding, rank, world, device)
    last_save = time.monotonic()
    try:
        while not state["complete"]:
            epoch = state["epoch"]
            ds = RankBatches(data["train"], orders[epoch], state["cursor"], c["dataset_root"], source, rank, world)
            loader = iter(make_loader(ds, c, rank, epoch, device))
            while state["cursor"] < 76642:
                batch, error = None, None
                try:
                    batch = next(loader)
                    if int(batch["offset"]) != state["cursor"]:
                        raise ValueError("DataLoader offset differs")
                except Exception as exc:
                    error = repr(exc)
                sync_ok(error is None, device, "Image load failed: " + str(error))
                before = time.monotonic()
                metrics = update_batch(model, optimizer, batch, device)
                state["cursor"] += int(batch["global_size"]); state["updates"] += 1
                rank0_call(lambda: progress(out, {"phase": "train", "epoch": epoch, "cursor": state["cursor"],
                    "updates": state["updates"], "step_seconds": time.monotonic()-before, **metrics}), rank, device)
                due = torch.tensor(int(rank == 0 and (state["updates"] in (1, 5)
                    or state["updates"] % c.get("checkpoint_every_steps", 20) == 0
                    or time.monotonic()-last_save >= c.get("checkpoint_every_seconds", 300))), device=device)
                if dist.is_initialized():
                    dist.broadcast(due, 0)
                if bool(due):
                    save(out, model, optimizer, scheduler, state, binding, rank, world, device)
                    last_save = time.monotonic()
            del loader
            save(out, model, optimizer, scheduler, state, binding, rank, world, device)
            rank0_call(lambda: progress(out, {"phase": "validation", "epoch": epoch}), rank, device)
            metrics = validate(model, data["val"], c, source, rank, world, device, out, epoch, binding)
            value = metrics["nll"]
            best = state["best"] is None or value < state["best"]
            if best:
                state["best"] = value
            if state["anchor"] is None or value < state["anchor"] - 1e-4:
                state["anchor"], state["bad_epochs"] = value, 0
            else:
                state["bad_epochs"] += 1
            scheduler.step(value)
            state["history"].append({"epoch": epoch, **metrics})
            state["epoch"], state["cursor"] = epoch+1, 0
            state["complete"] = epoch == 20 or state["bad_epochs"] >= 5
            save(out, model, optimizer, scheduler, state, binding, rank, world, device, best=best)
            rank0_call(lambda: progress(out, {"phase": "complete" if state["complete"] else "epoch_complete",
                "epoch": epoch, "best_nll": state["best"], **metrics}), rank, device)
        rank0_call(lambda: atomic_json(out / "training_summary.json", {"status": "complete",
            "binding": binding, "state": state, "test_accessed": False,
            "checkpoint_best_sha256": gate.sha(out / "checkpoint_best.pt"),
            "checkpoint_last_sha256": gate.sha(out / "checkpoint_last.pt")}), rank, device)
    except Exception as exc:
        # Preserve the previous atomic committed checkpoint. A failed optimizer
        # step may leave partial in-memory changes and must NEVER overwrite it.
        rank0_call(lambda: atomic_json(out / "failure.json", {"error": repr(exc), "state": state,
            "time": time.time(), "last_atomic_checkpoint_preserved": True}), rank, device)
        raise
    finally:
        if lock is not None:
            lock.close()


def preflight(c, source, model_source, out, rank, world, device):
    orders, binding, bound = allrank_call(lambda: config_and_binding(c, model_source, world), device, "Bind configuration")
    report_gate = allrank_call(lambda: gate.verify_assets(c), device, "Frozen A/B gate")
    rows = allrank_call(lambda: verified_data(source, c), device, "Frozen data gate")
    rank0_call(lambda: out.mkdir(parents=True, exist_ok=False), rank, device)
    model = allrank_call(lambda: source.build_training_model(c, device), device, "Load frozen generator")
    frozen_before = allrank_call(lambda: frozen_fingerprint(model), device, "Fingerprint generator")
    optimizer, scheduler = allrank_call(lambda: optimizer_for(model), device, "Initialize head optimizer")
    features = []
    handle = model.log_variance_head.register_forward_pre_hook(lambda unused, args: features.append(list(args[0].shape)))
    torch.cuda.reset_peak_memory_stats(device)
    full = allrank_call(lambda: RankBatches(rows["train"], orders[1][:16], 0, c["dataset_root"], source, rank, world)[0],
                        device, "Read native full16")
    t = time.monotonic()
    full_result = update_batch(model, optimizer, full, device)
    full_seconds = time.monotonic()-t
    train_peak = torch.cuda.max_memory_allocated(device)/2**20
    saved = {"head": cpu_copy(model.log_variance_head.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()), "rng": capture_rng(device)}
    replay_path = out / ("replay_rank%d.pt" % rank)
    error = None
    try:
        atomic_checkpoint(replay_path, saved)
    except Exception as exc:
        error = repr(exc)
    sync_ok(error is None, device, "Preflight replay checkpoint IO failed: " + str(error))
    tail = allrank_call(lambda: RankBatches(rows["train"], orders[1][-2:], 0, c["dataset_root"], source, rank, world)[0],
                        device, "Read native tail2")
    tail_result = update_batch(model, optimizer, tail, device)
    expected = {"head": cpu_copy(model.log_variance_head.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()), "rng": capture_rng(device)}
    saved = allrank_call(lambda: torch.load(replay_path, map_location="cpu", weights_only=False), device, "Load replay checkpoint")
    def restore():
        model.log_variance_head.load_state_dict(saved["head"]); optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], device)
    allrank_call(restore, device, "Restore replay checkpoint")
    update_batch(model, optimizer, tail, device)
    checks = {"head": state_equal(expected["head"], model.log_variance_head.state_dict()),
              "optimizer": state_equal(expected["optimizer"], optimizer.state_dict()),
              "rng": state_equal(expected["rng"], capture_rng(device))}
    handle.remove()
    checks["frozen_generator_unchanged"] = frozen_before == frozen_fingerprint(model)
    checks["feature_shapes"] = bool(features) and all(shape == [1, 32, 1024, 1024] for shape in features)
    checks["frozen_gradients_none"] = all(p.grad is None for m in (model.ae, model.denoiser) for p in m.parameters())
    with torch.no_grad():
        x = full["rgb"][:1].to(device)
        ids = full["sample_ids"][:1]
        expected_mean = source.ddim_sample(model.denoiser, x, ids, model.latent_mean, model.latent_std,
                                          model.ae, num_steps=20, generation_index=0, seed=20260817)
        actual_mean = checked_prediction(model, x, ids)["mean"]
        checks["mean_equals_original_generator"] = torch.equal(expected_mean, actual_mean)
    sync_ok(all(checks.values()), device, "Native C preflight failed")
    torch.cuda.reset_peak_memory_stats(device)
    t = time.monotonic()
    val = validate(model, rows["val"][:world], c, source, rank, world, device)
    result = {"rank": rank, "world_size": world, "binding": binding, "checks": checks,
              "gpu": torch.cuda.get_device_name(device), "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory/2**20,
              "train_peak_mib": train_peak, "validation_peak_mib": torch.cuda.max_memory_allocated(device)/2**20,
              "full16_seconds": full_seconds, "full16_update": full_result, "tail2_update": tail_result,
              "validation_seconds": time.monotonic()-t, "validation": val, "native_shape": [1, 3, 1024, 1024],
              "trainable_parameters": 33, "formal_weights_updated": False, "test_accessed": False}
    error = None
    try:
        atomic_json(out / ("rank%d.json" % rank), result)
    except Exception as exc:
        error = repr(exc)
    sync_ok(error is None, device, "Preflight report write failed: " + str(error))
    rank0_call(lambda: atomic_json(out / "preflight.json", {"passed": True, "binding": binding,
        **bound, "gate_report": report_gate}), rank, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--mode", choices=("preflight", "train"), required=True)
    p.add_argument("--preflight-output", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    c = gate.read_json(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Production/preflight requires CUDA")
    rank, world, local = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0))
    if world not in (1, 2, 4, 6, 8):
        raise ValueError("Unsupported fixed phase-C world")
    device = torch.device("cuda", local); torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = load_source(args.model_source)
    try:
        if args.mode == "preflight":
            if args.resume:
                p.error("preflight does not resume/update formal training")
            preflight(c, source, args.model_source, Path(args.preflight_output), rank, world, device)
        else:
            unused, binding, unused_bound = config_and_binding(c, args.model_source, world)
            pre = Path(args.preflight_output)
            manifest = gate.read_json(pre / "preflight.json")
            if manifest.get("passed") is not True or manifest.get("binding") != binding:
                raise ValueError("Native C preflight missing or wrong binding")
            for r in range(world):
                report = gate.read_json(pre / ("rank%d.json" % r))
                if report["binding"] != binding or not all(report["checks"].values()):
                    raise ValueError("Native C preflight rank failed")
            own = gate.read_json(pre / ("rank%d.json" % rank))
            total = torch.cuda.get_device_properties(device).total_memory/2**20
            if own["gpu"] != torch.cuda.get_device_name(device) or abs(own["gpu_total_mib"]-total)>1:
                raise ValueError("Hardware differs from preflight")
            if max(own["train_peak_mib"], own["validation_peak_mib"]) >= .95*total:
                raise ValueError("Preflight lacks 5% memory headroom")
            train(c, source, args.model_source, rank, world, device, args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
