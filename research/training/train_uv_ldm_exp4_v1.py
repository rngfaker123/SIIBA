#!/usr/bin/env python3
"""True, from-scratch two-stage UV LDM training. Train/val firewall; no TEST access.

Four replicas use manual gradient reduction and exact global 16 (tail 4),
physical batch one. BF16 forward, FP32 losses/coefficients/optimizer, no scaler.
Frozen orders and complete validation are inherited, not regenerated.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import train_uv_uncertainty_fullimage as ref
from models.matched_training_batching import order_sha256

SCHEMA = "uv_true_latent_diffusion_exp4_train_v1"
COUNTS = {"train": 80740, "val": 11534}
CODE_FILES = ("training/train_uv_ldm_exp4_v1.py", "models/uv_latent_diffusion_v1.py",
              "training/train_uv_uncertainty_fullimage.py", "models/highres_patch_dataset.py",
              "models/matched_training_batching.py")
read_json, digest, save_json = ref.read_json, ref.digest, ref.save_json


def runtime():
    import torch
    import numpy
    import PIL
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=numpy.__version__, pillow=PIL.__version__)


def validate_config(c):
    t = c["training"]
    expected = dict(epochs=20, global_batch_size=16, microbatch_size=1, world_size=4,
                    lr=1e-4, weight_decay=1e-4, grad_clip=1., precision="BF16",
                    checkpoint_every_attempts=1000, checkpoint_every_seconds=600)
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError("Training budget/precision differs from frozen LDM protocol")
    if c.get("expected_counts") != COUNTS or c.get("seed") != 20260817 or c.get("test_access_allowed") is not False:
        raise ValueError("Train/val counts, seed or TEST firewall differs")
    if c["scheduler"] != dict(factor=.5, patience=2, min_lr=1e-6):
        raise ValueError("Scheduler differs")
    if c["early_stopping"] != dict(patience=5, min_delta=.0001):
        raise ValueError("Early stopping differs")
    if c["model"] != dict(ae_base_channels=32, latent_channels=4, diffusion_base_channels=64, time_dimension=256):
        raise ValueError("Model configuration differs")
    if c["ae"].get("kl_weight") != 1e-6:
        raise ValueError("AE KL coefficient differs")
    d = c["diffusion"]
    if any(d.get(k) != v for k, v in dict(timesteps=1000, beta_start=1e-4, beta_end=.02,
                                         prediction_type="v_prediction", ddim_steps=20, eta=0).items()):
        raise ValueError("Diffusion training/sampling differs")
    for k in ("pretrained", "warmstart", "resume_checkpoint", "checkpoint", "ema"):
        if c.get(k): raise ValueError("Foreign weights prohibited")
    if Path(c["output_dir"]).resolve() == Path(c["reference_uncertainty_dir"]).resolve():
        raise ValueError("Output must be independent")
    return c


def reference_assets(c):
    source = Path(c["reference_uncertainty_dir"])
    if digest(source / "protocol.json") != c["reference_protocol_sha256"]:
        raise RuntimeError("Reference protocol changed")
    p = read_json(source / "protocol.json")
    if p["source_manifest_sha256"] != c["source_manifest_sha256"] or digest(c["source_manifest"]) != c["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    files = {}
    for split, n in COUNTS.items():
        name = split + "_manifest.json"
        if digest(source / name) != p["manifest_sha256"][split]: raise RuntimeError("Reference manifest changed")
        rows = read_json(source / name)["rows"]
        if len(rows) != n or len({r["sample_id"] for r in rows}) != n or any(r["split"] != split for r in rows):
            raise RuntimeError("Reference coverage/firewall invalid")
        dataset_root = Path(c["dataset_root"]).resolve()
        for row in rows:
            for modality in ("rgb", "uv"):
                raw = (dataset_root / row[modality + "_path"]).resolve()
                if dataset_root not in raw.parents or not raw.is_file() or raw.stat().st_size == 0:
                    raise RuntimeError("Missing/unsafe train-val input: " + row["sample_id"] + ":" + modality)
        files[name] = digest(source / name)
    for epoch in range(1, 21):
        name = "orders/epoch_%03d.json" % epoch
        ref.load_order(source, p, epoch)
        files[name] = digest(source / name)
    return source, p, files


def prepare(config_path):
    c = validate_config(read_json(config_path)); out = Path(c["output_dir"])
    source, original, files = reference_assets(c)
    if out.exists():
        load_frozen(config_path)
        print(json.dumps(dict(status="already_prepared", protocol_sha256=digest(out / "protocol.json"))))
        return
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("orders", "ae", "diffusion", "latent_stats"): (out / folder).mkdir()
    for name, sha in files.items():
        with (source / name).open("rb") as src, (out / name).open("xb") as dst:
            shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
        if digest(out / name) != sha: raise RuntimeError("Copied reference differs")
    p = dict(schema=SCHEMA, config=c, config_sha256=digest(config_path), software=runtime(),
             code_sha256={n: digest(ROOT / n) for n in CODE_FILES}, manifest_sha256=original["manifest_sha256"],
             epoch_order_file_sha256=original["epoch_order_file_sha256"],
             reference_protocol_sha256=c["reference_protocol_sha256"],
             preprocessing="center_square_PIL_Lanczos_1024_uint8_div255_no_augmentation",
             initialization="both_AE_and_latent_denoiser_from_scratch_no_EMA",
             precision="BF16_training_forward_FP32_losses_coefficients_and_parameters_no_GradScaler_TF32_false;FP32_validation",
             batching="manual_four_rank_gradient_mean_global16_tail4_physical1_no_padding",
             ae_objective="image_equal_L1_plus_1e-6_KL_sampled_posterior_train_mean_posterior_val",
             diffusion_objective="unweighted_v_MSE_uniform_t_frozen_BEST_AE_posterior_mean_latents",
             selection="strict_lowest_complete_val_UV_L1;diffusion_single_DDIM20_seed20260817_generation_index0",
             latent_normalization="all_train_BEST_AE_posterior_mean_per_channel_population_mean_std_float64_streaming",
             schedule_implementation="linear_beta_float64_cumprod_then_FP32_buffers_not_historical_pixel_schedule",
             parameter_counts=dict(ae=1754315, diffusion=21051204),
             test_split_used=False, test_images_accessed=0, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_json(out / "protocol.json", p, immutable=True)
    save_json(out / "protocol_digest.json", {"sha256": digest(out / "protocol.json")}, immutable=True)
    print(json.dumps(dict(status="prepared", protocol_sha256=digest(out / "protocol.json"))), flush=True)


def load_frozen(config_path):
    c = validate_config(read_json(config_path)); out = Path(c["output_dir"])
    p = read_json(out / "protocol.json"); ph = digest(out / "protocol.json")
    if p["schema"] != SCHEMA or read_json(out / "protocol_digest.json")["sha256"] != ph:
        raise RuntimeError("Frozen protocol identity changed")
    if p["config"] != c or p["config_sha256"] != digest(config_path): raise RuntimeError("Config changed")
    if p["software"] != runtime(): raise RuntimeError("Runtime differs; no silent migration")
    if p["code_sha256"] != {n: digest(ROOT / n) for n in CODE_FILES}: raise RuntimeError("Code changed")
    rows = {}
    for split, n in COUNTS.items():
        path = out / (split + "_manifest.json")
        if digest(path) != p["manifest_sha256"][split]: raise RuntimeError("Manifest changed")
        rows[split] = read_json(path)["rows"]
        if len(rows[split]) != n or len({r["sample_id"] for r in rows[split]}) != n or any(r["split"] != split for r in rows[split]):
            raise RuntimeError("Coverage/firewall invalid")
    if {r["sample_id"] for r in rows["train"]} & {r["sample_id"] for r in rows["val"]}:
        raise RuntimeError("Train/val overlap")
    return c, out, p, ph, rows


def local_indices(indices, rank):
    indices = tuple(indices)
    if len(indices) not in (4, 16) or len(set(indices)) != len(indices) or rank not in range(4):
        raise ValueError("Only unpadded global16/tail4 on four ranks")
    return indices[rank::4]


def epoch_batches(order, start=0):
    if len(order) != 80740 or len(set(order)) != 80740 or not 0 <= start <= 5047:
        raise ValueError("Invalid epoch/resume offset")
    return [(i, tuple(order[i * 16:(i + 1) * 16])) for i in range(start, 5047)]


def tensor_fingerprint(value):
    import torch
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous(); h.update(str((str(a.dtype), tuple(a.shape))).encode()); h.update(a.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(repr(k).encode()); walk(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for item in v: walk(item)
        else: h.update(repr(v).encode())
    walk(value); return h.hexdigest()


def setup(c):
    import torch
    import torch.distributed as dist
    import numpy as np
    rank, local, world = (int(os.environ.get(k, -1)) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 4 or rank not in range(4) or local not in range(4): raise RuntimeError("Requires torchrun four ranks")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported; no automatic precision change")
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.cuda.manual_seed(c["seed"])
    return rank, torch.device("cuda", local)


def replica_equal(core, optimizer):
    import torch.distributed as dist
    fp = dict(model=tensor_fingerprint(core.state_dict()), optimizer=tensor_fingerprint(optimizer.state_dict()))
    items = [None] * 4; dist.all_gather_object(items, fp)
    if any(v != items[0] for v in items): raise RuntimeError("Replicas differ")
    return fp


def frozen_ae(c, out, ph, device, require_stats=True):
    import torch
    from models.uv_latent_diffusion_v1 import KLUVAutoencoder
    folder = out / "ae"; summary = read_json(folder / "training_summary.json")
    path = folder / "checkpoint_best.pt"
    if summary["status"] != "complete" or summary["protocol_sha256"] != ph or digest(path) != summary["checkpoint_best_sha256"]:
        raise RuntimeError("AE BEST is not completed/frozen")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != ph or checkpoint["stage"] != "ae": raise RuntimeError("Wrong AE")
    ae = KLUVAutoencoder(base_channels=32, latent_channels=4).float().to(device)
    ae.load_state_dict(checkpoint["model_state_dict"], strict=True); ae.eval(); ae.requires_grad_(False)
    assets = dict(ae_checkpoint=str(path), ae_checkpoint_sha256=digest(path))
    stats = None
    if require_stats:
        sp = out / "latent_stats" / "statistics.json"; stats = read_json(sp)
        if stats["ae_checkpoint_sha256"] != assets["ae_checkpoint_sha256"] or stats["protocol_sha256"] != ph or stats["images"] != 80740:
            raise RuntimeError("Wrong/incomplete latent statistics")
        if len(stats["mean"]) != 4 or len(stats["std"]) != 4 or any(not math.isfinite(v) for v in stats["mean"] + stats["std"]) or any(v <= 0 for v in stats["std"]):
            raise RuntimeError("Invalid latent statistics; no flooring")
        assets.update(latent_stats=str(sp), latent_stats_sha256=digest(sp))
    return ae, stats, assets


def components(c, stage, out, ph, device):
    import torch
    import torch.distributed as dist
    from models.uv_latent_diffusion_v1 import KLUVAutoencoder, RGBLatentVDenoiser, VPLinearSchedule
    core = (KLUVAutoencoder(base_channels=32, latent_channels=4) if stage == "ae" else
            RGBLatentVDenoiser(base_channels=64, latent_channels=4, time_dimension=256)).float().to(device)
    if sum(parameter.numel() for parameter in core.parameters()) != (1754315 if stage == "ae" else 21051204):
        raise RuntimeError("Actual model parameter count differs from locked implementation")
    for v in core.state_dict().values(): dist.broadcast(v, src=0)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
    ae, stats, assets = (None, None, {}) if stage == "ae" else frozen_ae(c, out, ph, device)
    schedule = VPLinearSchedule().to(device)
    # Identical initialization, independent reproducible stochastic streams per rank.
    rank = dist.get_rank()
    import numpy as np
    random.seed(c["seed"] + rank * 1000)
    np.random.seed(c["seed"] + rank * 1000)
    torch.manual_seed(c["seed"] + rank * 1000)
    torch.cuda.manual_seed(c["seed"] + rank * 1000)
    return core, optimizer, scheduler, ae, stats, assets, schedule


def normalized_latent(ae, target, stats):
    import torch
    with torch.no_grad():
        mu, _ = ae.encode(target)
    mean = torch.tensor(stats["mean"], device=target.device, dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(stats["std"], device=target.device, dtype=torch.float32).view(1, 4, 1, 1)
    z = (mu.float() - mean) / std
    if not bool(torch.isfinite(z).all()): raise FloatingPointError("Nonfinite normalized latent")
    return z


def objective(core, stage, rgb, target, ae=None, stats=None, schedule=None):
    import torch
    from models.uv_latent_diffusion_v1 import kl_divergence
    if rgb.shape != target.shape or tuple(target.shape) != (1, 3, 1024, 1024): raise ValueError("Expected full1024 physical1")
    if any(not bool(torch.isfinite(v).all()) or bool((v < 0).any()) or bool((v > 1).any()) for v in (rgb, target)):
        raise FloatingPointError("Input outside finite [0,1]")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if stage == "ae":
            result = core(target, sample=True)
            for k in ("reconstruction", "mu", "logvar", "z"):
                if not bool(torch.isfinite(result[k]).all()): raise FloatingPointError("Nonfinite AE " + k)
            l1 = (result["reconstruction"].float() - target).abs().mean()
            loss = l1 + 1e-6 * kl_divergence(result["mu"].float(), result["logvar"].float())
        else:
            # Frozen AE statistics/coefficients remain FP32 even under BF16 denoiser forward.
            with torch.autocast(device_type="cuda", enabled=False):
                z = normalized_latent(ae, target, stats)
                noise = torch.randn_like(z)
                t = torch.randint(0, schedule.alphas_cumprod.numel(), (1,), device=z.device)
                noisy = schedule.add_noise(z, noise, t)
                wanted = schedule.v_target(z, noise, t)
            prediction = core(noisy, t, rgb)
            if not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Nonfinite velocity")
            loss = (prediction.float() - wanted.float()).square().mean()
            l1 = torch.zeros((), device=rgb.device)  # Not a generated-image training diagnostic.
    if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite objective")
    return loss, l1


def synchronized_failure(failure, device, label):
    import torch.distributed as dist
    if not ref.finite_all_ranks(failure is None, device):
        items = [None] * 4; dist.all_gather_object(items, failure)
        raise RuntimeError(label + ": " + json.dumps(items))


def global_update(core, optimizer, state, accumulator, indices, rank, device, get_pair,
                  stage="ae", ae=None, stats=None, schedule=None):
    import torch
    import torch.distributed as dist
    core.train(); optimizer.zero_grad(set_to_none=True)
    failure, current = None, None
    local = ref.new_accumulator()
    try:
        for current in local_indices(indices, rank):
            rgb, target = get_pair(current)
            loss, l1 = objective(core, stage, rgb, target, ae, stats, schedule)
            (loss * (4. / len(indices))).backward()
            local["images"] += 1; local["nll_sum"] += float(loss); local["l1_sum"] += float(l1)
            del rgb, target, loss, l1
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in core.parameters() if p.requires_grad):
            raise FloatingPointError("Missing/nonfinite local gradient")
    except Exception: failure = dict(rank=rank, sample_index=current, traceback=traceback.format_exc())
    synchronized_failure(failure, device, "Local forward/backward failed; no update")
    for p in core.parameters():
        if p.requires_grad: dist.all_reduce(p.grad); p.grad.div_(4)
    norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
    if not ref.finite_all_ranks(bool(torch.isfinite(norm)), device): raise FloatingPointError("Nonfinite reduced gradient")
    failure = None
    try:
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in core.parameters()): raise FloatingPointError("Nonfinite updated model")
        if any(not bool(torch.isfinite(v).all()) for row in optimizer.state.values() for v in row.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError("Nonfinite optimizer")
    except Exception: failure = dict(rank=rank, traceback=traceback.format_exc())
    synchronized_failure(failure, device, "Optimizer failed; partial state not checkpointed")
    optimizer.zero_grad(set_to_none=True)
    state["global_step"] += 1; state["global_attempts"] += 1
    for k in local: accumulator[k] += local[k]
    return float(norm)


def checkpoint_metadata(order, state):
    epoch_batches(order, state["next_batch"])
    return dict(rule="global16_four_rank_stride_no_padding", world_size=4, physical_batch=1,
                order_sha256=order_sha256(order), next_batch=state["next_batch"], samples=80740)


def save_checkpoint(out, stage, ph, core, optimizer, scheduler, state, accumulator, order, rank, device, generator, assets, best=False):
    import torch.distributed as dist
    fingerprints = replica_equal(core, optimizer)
    local = dict(rank=rank, rng=ref.capture_rng(device), epoch_accumulator=accumulator, loader_generator_state=generator.get_state())
    gathered = [None] * 4; dist.all_gather_object(gathered, local)
    if rank == 0:
        payload = dict(schema=SCHEMA, stage=stage, protocol_sha256=ph, state=state,
                       model_state_dict=core.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(), rank_states=gathered,
                       batching_resume=checkpoint_metadata(order, state), replica_fingerprints=fingerprints,
                       frozen_assets=assets, ema_used=False, test_split_used=False,
                       training_complete=state["phase"] == "complete", precision="BF16_forward_FP32_loss_no_scaler")
        if best: ref.atomic_torch_save(out / "checkpoint_best.pt", payload)
        ref.atomic_torch_save(out / "checkpoint_last.pt", payload)
        save_json(out / "checkpoint_commit.json", dict(sha256=digest(out / "checkpoint_last.pt"), epoch=state["epoch"],
                   global_step=state["global_step"], next_batch=state["next_batch"], phase=state["phase"]))
        save_json(out / "history.json", state["history"])
    dist.barrier()


def verify_checkpoint(cp, stage, ph, order, assets):
    if cp.get("schema") != SCHEMA or cp.get("stage") != stage or cp.get("protocol_sha256") != ph:
        raise RuntimeError("Foreign checkpoint")
    if cp.get("frozen_assets") != assets or cp.get("ema_used") is not False or cp.get("test_split_used") is not False:
        raise RuntimeError("Checkpoint asset/scope mismatch")
    s = cp["state"]
    if s["phase"] not in ("train", "val", "epoch_complete", "complete") or not 1 <= s["epoch"] <= 20:
        raise RuntimeError("Invalid checkpoint phase")
    if s["global_step"] != s["global_attempts"] or s["amp_overflow_skips"] != 0 or s.get("blocked_reason"):
        raise RuntimeError("Unsafe counters")
    if cp["batching_resume"] != checkpoint_metadata(order, s): raise RuntimeError("Resume order/layout changed")
    if sorted(v["rank"] for v in cp["rank_states"]) != list(range(4)): raise RuntimeError("Rank RNG missing")
    for v in cp["rank_states"]:
        consumed = sum(len(local_indices(indices, v["rank"])) for _, indices in epoch_batches(order)[:s["next_batch"]])
        if v["epoch_accumulator"]["images"] != consumed: raise RuntimeError("Accumulator offset differs")
        if any(not math.isfinite(v["epoch_accumulator"][k]) for k in ("nll_sum", "l1_sum")): raise RuntimeError("Saved nonfinite stats")
    fp = dict(model=tensor_fingerprint(cp["model_state_dict"]), optimizer=tensor_fingerprint(cp["optimizer_state_dict"]))
    if cp["replica_fingerprints"] != fp: raise RuntimeError("Saved state fingerprint differs")


def val_prediction(core, stage, rgb, target, sample_id, ae, stats, schedule):
    from models.uv_latent_diffusion_v1 import ddim_sample
    if stage == "ae": return core(target, sample=False)["reconstruction"]
    return ddim_sample(core, rgb, [sample_id], stats["mean"], stats["std"], ae,
                       num_steps=20, generation_index=0, seed=20260817, schedule=schedule)


def run_validation(core, stage, dataset, rows, out, ph, state, c, rank, device, ae, stats, schedule):
    import torch
    import torch.distributed as dist
    indices = list(range(rank, len(rows), 4)); core.eval()
    sid = "%s:%s:e%d:s%d" % (ph, stage, state["epoch"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"]); directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    done, failure = [], None
    try:
        if path.exists():
            for pos, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True)):
                if not line.endswith("\n"): raise RuntimeError("Truncated validation record; preserve")
                r = json.loads(line)
                if pos >= len(indices) or r["index"] != indices[pos] or r["sample_id"] != rows[indices[pos]]["sample_id"] or r["state_id"] != sid:
                    raise RuntimeError("Validation identity/state mismatch")
                if not math.isfinite(r["l1"]): raise FloatingPointError("Nonfinite persisted L1")
                done.append(r)
        loader = ref.make_loader(dataset, [[i] for i in indices[len(done):]], c, c["seed"] + rank + state["epoch"] * 4 + 500000)
        try:
            with torch.inference_mode(), path.open("a", encoding="utf-8") as handle:
                for pos, b in enumerate(loader, len(done)):
                    i = indices[pos]
                    if b["sample_id"] != [rows[i]["sample_id"]]: raise RuntimeError("Validation loader identity mismatch")
                    rgb, target = b["rgb"].to(device).float(), b["target"].to(device).float()
                    prediction = val_prediction(core, stage, rgb, target, rows[i]["sample_id"], ae, stats, schedule)
                    if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Invalid validation output")
                    l1 = float((prediction.float() - target).abs().mean())
                    r = dict(index=i, sample_id=rows[i]["sample_id"], state_id=sid, protocol_sha256=ph, l1=l1, test_split_used=False)
                    handle.write(json.dumps(r, allow_nan=False) + "\n"); handle.flush(); done.append(r)
                    if len(done) % 25 == 0:
                        os.fsync(handle.fileno()); save_json(out / ("validation_progress_rank%d.json" % rank), dict(status="validating", stage=stage, epoch=state["epoch"], completed_images=len(done), expected_images=len(indices)))
                    del rgb, target, prediction, b
                os.fsync(handle.fileno())
        finally:
            if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
        if len(done) != len(indices): raise RuntimeError("Incomplete validation")
    except Exception:
        failure = dict(rank=rank, traceback=traceback.format_exc()); save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())), failure, immutable=True)
    synchronized_failure(failure, device, "Validation failed")
    totals = torch.tensor([len(done), sum(r["l1"] for r in done)], dtype=torch.float64, device=device); dist.all_reduce(totals)
    if int(totals[0]) != 11534: raise RuntimeError("Full VAL coverage differs")
    save_json(out / ("validation_progress_rank%d.json" % rank), dict(status="complete", epoch=state["epoch"], completed_images=len(done), record_sha256=digest(path)))
    return dict(images=int(totals[0]), l1=float(totals[1] / totals[0]), single_generation=True, ddim20=stage == "diffusion", precision="FP32")


class StopAtBoundary:
    def __init__(self):
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, self.handle)
    def handle(self, *_): self.requested = True


def any_rank(flag, device):
    import torch
    import torch.distributed as dist
    value = torch.tensor(int(flag), device=device); dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def check_preflight(directory, config_path, ph, stage, assets):
    for rank in range(4):
        p = read_json(Path(directory) / ("rank%d.json" % rank))
        if (p.get("status") != "passed" or p.get("stage") != stage or p.get("rank") != rank or
            p.get("protocol_sha256") != ph or p.get("config_sha256") != digest(config_path) or
            p.get("code_sha256") != {n: digest(ROOT / n) for n in CODE_FILES} or p.get("software") != runtime() or
            p.get("frozen_assets") != assets or p.get("resume_equivalence_pass") is not True or
            p.get("visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise RuntimeError("Preflight gate incomplete/mismatched")


def train(config_path, stage, resume="auto", preflight_directory=None):
    c, root, p, ph, splits = load_frozen(config_path)
    if preflight_directory is None: raise RuntimeError("Fresh preflight required")
    import torch
    import torch.distributed as dist
    rank, device = setup(c); out = root / stage; loader = None
    state, acc = ref.new_state(), ref.new_accumulator(); stopper = StopAtBoundary()
    try:
        with ref.exclusive_lock(out / "training_coordinator.lock") if rank == 0 else contextlib.nullcontext():
            dist.barrier()
            core, opt, sch, ae, stats, assets, schedule = components(c, stage, root, ph, device)
            check_preflight(preflight_directory, config_path, ph, stage, assets)
            gen = torch.Generator().manual_seed(c["seed"] + rank * 1000)
            last = out / "checkpoint_last.pt"
            if last.exists():
                if resume == "none" or digest(last) != read_json(out / "checkpoint_commit.json")["sha256"]: raise RuntimeError("LAST/commit differs")
                cp = torch.load(last, map_location="cpu", weights_only=False); state = cp["state"]
                order = ref.load_order(root, p, state["epoch"]); verify_checkpoint(cp, stage, ph, order, assets)
                core.load_state_dict(cp["model_state_dict"], strict=True); opt.load_state_dict(cp["optimizer_state_dict"]); sch.load_state_dict(cp["scheduler_state_dict"])
                local = cp["rank_states"][rank]; acc = local["epoch_accumulator"]; gen.set_state(local["loader_generator_state"]); ref.restore_rng(local["rng"], device)
                del cp, local
            elif any((out / n).exists() for n in ("history.json", "checkpoint_best.pt", "training_progress.json")):
                raise RuntimeError("Prior artifacts without LAST; no restart")
            replica_equal(core, opt)
            if state["phase"] == "complete":
                if rank == 0:
                    summary = ref.final_summary(out, state, ph)
                    summary.update(stage=stage, true_diffusion=stage == "diffusion", frozen_assets=assets)
                    save_json(out / "training_summary.json", summary)
                return
            ref.advance_completed_epoch(state)
            train_data = ref.TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
            val_data = ref.TrainingOnlyDataset(splits["val"], c["dataset_root"], "val")
            save_json(out / ("environment_rank%d_%d.json" % (rank, time.time_ns())), dict(rank=rank, gpu=torch.cuda.get_device_name(device), software=runtime(), precision="BF16", parameters=sum(v.numel() for v in core.parameters()), test_split_used=False), immutable=True)
            last_save, started, first_attempt = time.monotonic(), time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]; order = ref.load_order(root, p, epoch)
                if state["phase"] == "next_epoch": state.update(phase="train", next_batch=0); acc = ref.new_accumulator()
                if not last.exists(): save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets)
                if state["phase"] == "train":
                    plans = epoch_batches(order, state["next_batch"])
                    loader = ref.make_loader(train_data, [[i] for _, indices in plans for i in local_indices(indices, rank)], c, c["seed"] + rank * 1000)
                    loader.generator.set_state(gen.get_state()); iterator = iter(loader); gen.set_state(loader.generator.get_state())
                    def get_pair(i):
                        b = next(iterator)
                        if b["sample_id"] != [splits["train"][i]["sample_id"]]: raise RuntimeError("Train identity mismatch")
                        return b["rgb"].to(device, non_blocking=True).float(), b["target"].to(device, non_blocking=True).float()
                    for pos, indices in plans:
                        norm = global_update(core, opt, state, acc, indices, rank, device, get_pair, stage, ae, stats, schedule)
                        state["next_batch"] = pos + 1
                        stop = any_rank(stopper.requested, device)
                        due = any_rank(time.monotonic() - last_save >= 600 if rank == 0 else False, device)
                        if state["global_attempts"] in (1, 20) or state["global_attempts"] % 1000 == 0 or due or stop:
                            save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets); last_save = time.monotonic()
                        if state["global_attempts"] % 20 == 0 or state["global_attempts"] == 1:
                            totals = ref.reduced_accumulator(acc, device); seconds = (time.monotonic() - started) / max(1, state["global_attempts"] - first_attempt)
                            ref.log_progress(out, dict(status="training", stage=stage, epoch=epoch, max_epochs=20, epoch_batch=state["next_batch"], epoch_batches=5047, global_step=state["global_step"], global_attempts=state["global_attempts"], amp_overflow_skips=0, train_images=totals["images"], train_loss=totals["nll"], train_l1=totals["l1"] if stage == "ae" else None, gradient_norm=norm, seconds_per_global_batch=seconds, epoch_train_eta_seconds=seconds * (5047 - state["next_batch"])), rank)
                        if stop: return
                    if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
                    loader = None; totals = ref.reduced_accumulator(acc, device)
                    if totals["images"] != 80740: raise RuntimeError("Training coverage differs")
                    state["phase"] = "val"; save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets)
                totals = ref.reduced_accumulator(acc, device)
                ref.log_progress(out, dict(status="validating", stage=stage, epoch=epoch, global_step=state["global_step"]), rank)
                val_started = time.monotonic()
                val = run_validation(core, stage, val_data, splits["val"], out, ph, state, c, rank, device, ae, stats, schedule)
                selection, best, early = ref.update_selection(state["selection"], val["l1"], epoch)
                state["selection"] = selection; sch.step(val["l1"])
                row = dict(epoch=epoch, stage=stage, train_images=totals["images"], train_loss=totals["nll"], validation=val,
                           validation_seconds=time.monotonic() - val_started, global_step=state["global_step"], is_best=best, amp_overflow_skips=0)
                state["history"].append(row)
                if early or epoch == 20: state.update(phase="complete", stop_reason="early_stopping" if early else "max_epochs")
                else: state["phase"] = "epoch_complete"
                save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets, best=best)
                ref.log_progress(out, dict(row, status="epoch_complete", best_epoch=selection["best_epoch"], best_val_l1=selection["best_val_l1"]), rank)
                if state["phase"] == "epoch_complete": ref.advance_completed_epoch(state)
            if rank == 0:
                summary = ref.final_summary(out, state, ph); summary.update(stage=stage, true_diffusion=stage == "diffusion", frozen_assets=assets)
                save_json(out / "training_summary.json", summary)
    except Exception:
        save_json(out / ("failure_rank%d_%d.json" % (rank, time.time_ns())), dict(status="failed", rank=rank, state=state, traceback=traceback.format_exc(), test_split_used=False), immutable=True)
        raise
    finally:
        if loader is not None and getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
        if dist.is_initialized(): dist.destroy_process_group()


def preflight(config_path, stage, directory):
    c, root, p, ph, splits = load_frozen(config_path)
    import torch
    import torch.distributed as dist
    rank, device = setup(c); directory = Path(directory)
    if directory.resolve() == root.resolve() or root.resolve() in directory.resolve().parents:
        raise RuntimeError("Preflight must be outside formal experiment")
    try:
        if rank == 0: directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        core, opt, sch, ae, stats, assets, schedule = components(c, stage, root, ph, device)
        state, acc = ref.new_state(), ref.new_accumulator(); gen = torch.Generator().manual_seed(c["seed"] + rank * 1000)
        torch.cuda.reset_peak_memory_stats(device)
        def synthetic(_):
            rgb = torch.rand(1, 3, 1024, 1024, device=device); return rgb, torch.rand_like(rgb)
        started = time.monotonic()
        global_update(core, opt, state, acc, tuple(range(16)), rank, device, synthetic, stage, ae, stats, schedule)
        snap = dict(model=core.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), rng=ref.capture_rng(device), state=state, accumulator=acc, loader_rng=gen.get_state())
        buffer = io.BytesIO(); torch.save(snap, buffer)
        def tail():
            global_update(core, opt, state, acc, tuple(range(16, 20)), rank, device, synthetic, stage, ae, stats, schedule)
            sch.step(.2); torch.rand(2, generator=gen)
        def snapshot():
            return ref.cpu_snapshot(dict(model=core.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), rng=ref.capture_rng(device), state=state, accumulator=acc, loader_rng=gen.get_state()))
        tail(); expected = snapshot(); buffer.seek(0); saved = torch.load(buffer, map_location="cpu", weights_only=False)
        core.load_state_dict(saved["model"], strict=True); opt.load_state_dict(saved["optimizer"]); sch.load_state_dict(saved["scheduler"])
        ref.restore_rng(saved["rng"], device); gen.set_state(saved["loader_rng"]); state, acc = saved["state"], saved["accumulator"]
        del saved, snap, buffer
        tail(); observed = snapshot(); checks = {k: ref.state_equal(expected[k], observed[k]) for k in expected}
        if not ref.finite_all_ranks(all(checks.values()), device): raise RuntimeError("Checkpoint/RNG replay differs: " + json.dumps(checks))
        del expected, observed
        train_data = ref.TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
        val_data = ref.TrainingOnlyDataset(splits["val"], c["dataset_root"], "val")
        order = ref.load_order(root, p, 1); real_indices = order[:2]
        samples = []
        for split, indices in (("train", real_indices), ("val", [0, 1])):
            for i in indices:
                row = splits[split][i]
                for mod in ("rgb", "uv"):
                    if digest(Path(c["dataset_root"]) / row[mod + "_path"]) != row[mod + "_sha256"]:
                        raise RuntimeError("Real source hash differs")
                samples.append(dict(split=split, sample_id=row["sample_id"]))
        def real_pair(i):
            item = train_data[real_indices[i % 2]]
            return item["rgb"].unsqueeze(0).to(device).float(), item["target"].unsqueeze(0).to(device).float()
        global_update(core, opt, state, acc, tuple(range(16)), rank, device, real_pair, stage, ae, stats, schedule)
        train_peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
        replica_equal(core, opt); core.eval(); values = []
        with torch.inference_mode():
            for i in (0, 1):
                item = val_data[i]; rgb, target = item["rgb"].unsqueeze(0).to(device).float(), item["target"].unsqueeze(0).to(device).float()
                prediction = val_prediction(core, stage, rgb, target, item["sample_id"], ae, stats, schedule)
                if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Real VAL generation invalid")
                values.append(dict(sample_id=item["sample_id"], l1=float((prediction - target).abs().mean())))
        result = dict(status="passed", stage=stage, rank=rank, world_size=4, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      protocol_sha256=ph, config_sha256=digest(config_path), software=runtime(), code_sha256={n: digest(ROOT / n) for n in CODE_FILES},
                      frozen_assets=assets, resume_equivalence_pass=True, resume_checks=checks, global_batch=16, tail_global_batch=4,
                      real_train_samples=2, real_val_samples=2, repeated_real_samples_only_for_disposable_global16_preflight=True,
                      real_samples=samples, validation=values, peak_train_mib=train_peak, total_seconds=time.monotonic() - started,
                      formal_checkpoint_written=False, weights_reused_for_training=False, test_split_used=False)
        save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        if read_json(directory / ("rank%d.json" % rank)) != result: raise RuntimeError("Preflight readback differs")
        dist.barrier(); print(json.dumps(dict(rank=rank, stage=stage, status="passed", peak_train_mib=train_peak)), flush=True)
    except Exception:
        if directory.exists(): save_json(directory / ("failure_rank%d_%d.json" % (rank, time.time_ns())), dict(status="failed", traceback=traceback.format_exc()), immutable=True)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def latent_stats(config_path):
    """Resumable, all-train posterior-mean statistics; no cached image/latent tensors."""
    c, root, p, ph, splits = load_frozen(config_path)
    import torch
    import torch.distributed as dist
    rank, device = setup(c); out = root / "latent_stats"
    try:
        ae, _, assets = frozen_ae(c, root, ph, device, require_stats=False)
        with ref.exclusive_lock(out / "coordinator.lock") if rank == 0 else contextlib.nullcontext():
            dist.barrier(); path = out / ("rank%d.json" % rank)
            indices = list(range(rank, 80740, 4)); done = 0
            sums = torch.zeros(4, dtype=torch.float64, device=device); squares = sums.clone(); count = 0
            if path.exists():
                old = read_json(path)
                if old["protocol_sha256"] != ph or old["assets"] != assets or old["rank"] != rank: raise RuntimeError("Stats identity differs")
                done, count = old["completed_images"], old["count_values_per_channel"]
                if not 0 <= done <= len(indices): raise RuntimeError("Stats offset invalid")
                if count != done * 256 * 256: raise RuntimeError("Stats count/offset differs")
                sums = torch.tensor(old["sums"], dtype=torch.float64, device=device); squares = torch.tensor(old["squares"], dtype=torch.float64, device=device)
            data = ref.TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
            loader = ref.make_loader(data, [[i] for i in indices[done:]], c, c["seed"] + rank + 900000)
            failure = None
            try:
                with torch.inference_mode():
                    for pos, item in enumerate(loader, done):
                        if item["sample_id"] != [splits["train"][indices[pos]]["sample_id"]]: raise RuntimeError("Stats identity mismatch")
                        mu, _ = ae.encode(item["target"].to(device).float()); z = mu.double()
                        if not bool(torch.isfinite(z).all()): raise FloatingPointError("Nonfinite AE posterior mean")
                        sums += z.sum((0, 2, 3)); squares += z.square().sum((0, 2, 3)); count += z.shape[0] * z.shape[2] * z.shape[3]; done = pos + 1
                        if done % 100 == 0 or done == len(indices):
                            save_json(path, dict(rank=rank, completed_images=done, count_values_per_channel=count, sums=sums.tolist(), squares=squares.tolist(), protocol_sha256=ph, assets=assets, test_split_used=False))
                        del item, mu, z
            except Exception: failure = dict(rank=rank, traceback=traceback.format_exc())
            finally:
                if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
            synchronized_failure(failure, device, "Latent statistics failed")
            totals = torch.cat([torch.tensor([done, count], dtype=torch.float64, device=device), sums, squares]); dist.all_reduce(totals)
            if int(totals[0]) != 80740: raise RuntimeError("Stats coverage differs")
            mean = totals[2:6] / totals[1]; variance = totals[6:10] / totals[1] - mean.square()
            if not bool(torch.isfinite(variance).all()) or not bool((variance > 0).all()): raise FloatingPointError("Invalid latent std; no flooring")
            result = dict(schema="uv_ldm_all_train_latent_statistics_v1", images=80740, count_values_per_channel=int(totals[1]), mean=mean.tolist(), std=variance.sqrt().tolist(),
                          protocol_sha256=ph, ae_checkpoint_sha256=assets["ae_checkpoint_sha256"], method="population_float64_all_train_posterior_mean", test_split_used=False)
            result.update(train_manifest_sha256=p["manifest_sha256"]["train"], source_manifest_sha256=c["source_manifest_sha256"],
                          latent_shape_per_image=[4, 256, 256], accumulator_dtype="float64", posterior_encoding_precision="FP32")
            if rank == 0:
                target = out / "statistics.json"
                if target.exists() and read_json(target) != result: raise RuntimeError("Existing stats differ; preserve")
                if not target.exists(): save_json(target, result, immutable=True)
            dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def audit(config_path, stage):
    c, root, p, ph, splits = load_frozen(config_path)
    folder = root / stage; path = folder / "training_summary.json"
    if not path.exists(): raise RuntimeError("Training summary not yet complete")
    summary = read_json(path)
    if summary["status"] != "complete" or summary["protocol_sha256"] != ph or summary["test_split_used"] is not False:
        raise RuntimeError("Invalid completed stage")
    for name in ("best", "last"):
        if digest(folder / ("checkpoint_" + name + ".pt")) != summary["checkpoint_" + name + "_sha256"]:
            raise RuntimeError("Completed checkpoint changed")
    print(json.dumps(dict(status="complete", stage=stage, summary=summary), allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("prepare", "preflight", "train", "latent-stats", "audit"))
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", choices=("ae", "diffusion"), default="ae")
    ap.add_argument("--resume", choices=("auto", "none"), default="auto")
    ap.add_argument("--preflight-output", type=Path)
    args = ap.parse_args()
    if args.mode == "prepare": prepare(args.config)
    elif args.mode == "latent-stats": latent_stats(args.config)
    elif args.mode == "audit": audit(args.config, args.stage)
    elif args.mode == "preflight":
        if args.preflight_output is None: ap.error("--preflight-output required")
        preflight(args.config, args.stage, args.preflight_output)
    else: train(args.config, args.stage, args.resume, args.preflight_output)


if __name__ == "__main__": main()
