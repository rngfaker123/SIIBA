# 待训练版本。标准 Swin-T 编码器尺寸，U 形 Swin 解码器，末端图像头和方差头
"""待训练版本。标准 Swin-T 编码器尺寸，U 形 Swin 解码器，末端图像头和方差头"""

from __future__ import annotations

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

VARIANCE_FLOOR = 1e-6

def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)

def window_partition(x, window_size):
    """"""
    b, h, w, c = x.shape
    if h % window_size or w % window_size:
        raise ValueError("Window partition requires divisible spatial dimensions")
    return x.reshape(b, h // window_size, window_size, w // window_size, window_size, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, c)

def window_reverse(windows, window_size, height, width, batch):
    """"""
    channels = windows.shape[-1]
    return windows.reshape(batch, height // window_size, width // window_size,
                           window_size, window_size, channels).permute(
        0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)

def effective_shift(height, width, window_size, shift_size):
    """"""
    return (0 if height <= window_size else shift_size,
            0 if width <= window_size else shift_size)

def window_attention_mask(height, width, window_size, shift_size, device=None):
    """"""
    hp = math.ceil(height / window_size) * window_size
    wp = math.ceil(width / window_size) * window_size
    sh, sw = effective_shift(height, width, window_size, shift_size)
    if sh == 0 and sw == 0 and hp == height and wp == width:
        return None
    yy = torch.arange(hp, device=device)
    xx = torch.arange(wp, device=device)

    source_y = (yy + sh) % hp
    source_x = (xx + sw) % wp
    valid = (source_y[:, None] < height) & (source_x[None, :] < width)
    groups = ((yy + sh >= hp).to(torch.int64)[:, None] * 2
              + (xx + sw >= wp).to(torch.int64)[None, :])
    valid_windows = window_partition(valid[None, :, :, None], window_size).squeeze(-1)
    group_windows = window_partition(groups[None, :, :, None], window_size).squeeze(-1)
    same_group = group_windows[:, :, None] == group_windows[:, None, :]
    allowed = same_group & valid_windows[:, :, None] & valid_windows[:, None, :]
    identity = torch.eye(window_size * window_size, dtype=torch.bool, device=device)[None]
    return allowed | ((~valid_windows)[:, :, None] & identity)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Attention dimensions must divide into whole heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.projection = nn.Linear(dim, dim, bias=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        flat = coords.flatten(1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous() + window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=True)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, windows, allowed=None):
        n, tokens, channels = windows.shape
        if tokens != self.window_size ** 2 or channels != self.dim:
            raise ValueError("Unexpected window token geometry")
        qkv = self.qkv(windows).reshape(n, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1).to(dtype=query.dtype)
        attention_bias = bias.unsqueeze(0)
        if allowed is not None:
            if n % allowed.shape[0] or allowed.shape[1:] != (tokens, tokens):
                raise ValueError("Attention-mask/window count mismatch")

            attention_bias = attention_bias.expand(allowed.shape[0], -1, -1, -1).masked_fill(
                ~allowed[:, None, :, :], float("-inf"))
            if n != allowed.shape[0]:
                attention_bias = attention_bias.repeat(n // allowed.shape[0], 1, 1, 1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(n, tokens, channels)
        return self.projection(attended)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        self._mask_key = None
        self._mask = None

    def forward(self, x):
        batch, height, width, channels = x.shape
        ws = self.window_size
        hp, wp = math.ceil(height / ws) * ws, math.ceil(width / ws) * ws
        sh, sw = effective_shift(height, width, ws, self.shift_size)
        mask_key = (height, width, x.device.type, x.device.index)
        if mask_key != self._mask_key:
            self._mask = window_attention_mask(height, width, ws, self.shift_size, x.device)
            self._mask_key = mask_key
        branch = self.norm1(x)
        if hp != height or wp != width:
            branch = F.pad(branch, (0, 0, 0, wp - width, 0, hp - height))
        if sh or sw:
            branch = torch.roll(branch, shifts=(-sh, -sw), dims=(1, 2))
        windows = self.attention(window_partition(branch, ws), self._mask)
        branch = window_reverse(windows, ws, hp, wp, batch)
        if sh or sw:
            branch = torch.roll(branch, shifts=(sh, sw), dims=(1, 2))
        branch = branch[:, :height, :width, :]
        x = x + branch
        return x + self.mlp(self.norm2(x))

class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, 0 if index % 2 == 0 else window_size // 2)
            for index in range(depth)])

    def forward(self, x, activation_checkpointing=False):
        for block in self.blocks:
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        if x.shape[1] % 2 or x.shape[2] % 2:
            x = F.pad(x, (0, 0, 0, x.shape[2] % 2, 0, x.shape[1] % 2))
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(x))

class PatchExpansion(nn.Module):
    """"""
    def __init__(self, dim, out_dim, scale):
        super().__init__()
        self.scale = scale
        self.out_dim = out_dim
        self.expansion = nn.Linear(dim, scale * scale * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        b, h, w, unused = x.shape
        s = self.scale
        expanded = self.expansion(x).reshape(b, h, w, s, s, self.out_dim)
        expanded = expanded.permute(0, 1, 3, 2, 4, 5).reshape(b, h * s, w * s, self.out_dim)
        return self.norm(expanded)

class SwinUVUncertainty(nn.Module):
    def __init__(self, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 decoder_depths=(6, 2, 2), window_size=7, patch_size=4,
                 activation_checkpointing=False):
        super().__init__()
        embed_dim = _positive_integer(embed_dim, "embed_dim")
        window_size = _positive_integer(window_size, "window_size")
        patch_size = _positive_integer(patch_size, "patch_size")
        if window_size < 2 or embed_dim < 2:
            raise ValueError("window_size and embed_dim must be at least two")
        depths = tuple(_positive_integer(value, "depth") for value in depths)
        num_heads = tuple(_positive_integer(value, "num_heads") for value in num_heads)
        decoder_depths = tuple(_positive_integer(value, "decoder_depth") for value in decoder_depths)
        if len(depths) != 4 or len(num_heads) != 4 or len(decoder_depths) != 3:
            raise ValueError("Expected four encoder stages and three decoder stages")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be bool")
        dimensions = [embed_dim * 2 ** index for index in range(4)]
        if any(dim % heads for dim, heads in zip(dimensions, num_heads)):
            raise ValueError("Each encoder dimension must be divisible by its head count")
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.decoder_depths = decoder_depths
        self.window_size = window_size
        self.patch_size = patch_size
        self.activation_checkpointing = activation_checkpointing
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.encoder_stages = nn.ModuleList([SwinStage(dim, depth, heads, window_size)
            for dim, depth, heads in zip(dimensions, depths, num_heads)])
        self.downsample = nn.ModuleList([PatchMerging(dim) for dim in dimensions[:-1]])
        self.bottleneck_norm = nn.LayerNorm(dimensions[-1])
        self.upsample = nn.ModuleList()
        self.skip_fusion = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for index, level in enumerate((2, 1, 0)):
            dim = dimensions[level]
            self.upsample.append(PatchExpansion(dimensions[level + 1], dim, 2))
            self.skip_fusion.append(nn.Linear(2 * dim, dim))
            self.decoder_stages.append(SwinStage(dim, decoder_depths[index], num_heads[level], window_size))
        self.decoder_norm = nn.LayerNorm(embed_dim)
        full_dim = embed_dim
        self.full_resolution_expansion = PatchExpansion(embed_dim, full_dim, patch_size)
        
        self.mean_head = nn.Conv2d(full_dim, 3, 1)
        self.log_variance_head = nn.Conv2d(full_dim, 1, 1)
        self.apply(self._initialize)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("Input must be a floating-point RGB tensor")
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[0], x.shape[2], x.shape[3]) < 1:
            raise ValueError("Input must have positive shape (batch,3,height,width)")
        height, width = x.shape[-2:]
        patch = self.patch_size
        pad_h, pad_w = (-height) % patch, (-width) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.patch_norm(self.patch_embedding(x).permute(0, 2, 3, 1))
        skips = []
        for index, stage in enumerate(self.encoder_stages):
            x = stage(x, self.activation_checkpointing)
            if index < 3:
                skips.append(x)
                x = self.downsample[index](x)
        x = self.bottleneck_norm(x)
        for upsample, fusion, stage, skip in zip(self.upsample, self.skip_fusion, self.decoder_stages, reversed(skips)):
            x = upsample(x)[:, :skip.shape[1], :skip.shape[2], :]
            x = fusion(torch.cat((x, skip), dim=-1))
            x = stage(x, self.activation_checkpointing)
        x = self.full_resolution_expansion(self.decoder_norm(x))
        x = x[:, :height, :width, :].permute(0, 3, 1, 2)
        mean = torch.sigmoid(self.mean_head(x)).float()
        raw_s = self.log_variance_head(x).float()
        log_variance = torch.logaddexp(raw_s, raw_s.new_tensor(math.log(VARIANCE_FLOOR)))
        return {"mean": mean, "raw_log_variance": raw_s, "log_variance": log_variance,
                "variance": torch.exp(log_variance), "sigma": torch.exp(.5 * log_variance)}

"""新版本训练入口的源模板。构建时复制进 Swin.py 和 LDM.py。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            value.update(part)
    return value.hexdigest()


def read_image(path, size=1024):
    # 与原训练一样，先居中裁成正方形，再用 Lanczos 缩放。
    with Image.open(path) as original:
        image = original.convert("RGB")
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def training_data(config):
    path = Path(config["source_manifest"])
    if file_sha256(path) != config["source_manifest_sha256"]:
        raise ValueError("冻结样本名单的哈希不一致")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {name: [r for r in rows if r["split"] == name and
                       r.get("eligible_primary", "1") == "1"] for name in ("train", "val")}
    for name, count in config["expected_counts"].items():
        if name in selected and len(selected[name]) != count:
            raise ValueError("清洗划分样本数不一致: " + name)
    subjects = [{r["subject_id"] for r in selected[name]} for name in ("train", "val")]
    if subjects[0] & subjects[1]:
        raise ValueError("训练集和验证集的 subject_id 有交集")
    for name in selected:
        ids = [r["sample_id"] for r in selected[name]]
        if len(ids) != len(set(ids)):
            raise ValueError("样本身份重复")
    return selected


def load_batch(rows, root, device):
    rgb = torch.stack([read_image(Path(root) / r["rgb_path"]) for r in rows]).to(device)
    uv = torch.stack([read_image(Path(root) / r["uv_path"]) for r in rows]).to(device)
    return rgb, uv


def gaussian_loss(output, truth):
    error2 = (output["mean"].float() - truth.float()).square().mean(1, keepdim=True)
    logvar = output["log_variance"].float()
    return (0.5 * (error2 * (-logvar).exp() + logvar)).flatten(1).mean(1)


def synchronized_check(ok, device, label):
    flag = torch.tensor(int(ok), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError(label)


def global_step(model, optimizer, scaler, rows, config, rank, world, device):
    """每张真实图像按 1/N 加权，汇总梯度后统一更新，不补假样本。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    size = config["training"]["microbatch_size"]
    local_rows = rows[rank::world]
    error = None
    try:
        for start in range(0, len(local_rows), size):
            batch = local_rows[start:start + size]
            rgb, truth = load_batch(batch, config["dataset_root"], device)
            # LDM 的生成路径固定使用 FP32，只训练新增方差头。
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                output = predict_training(model, rgb, batch)
            loss = gaussian_loss(output, truth).sum() / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限损失")
            scaler.scale(loss).backward()
    except Exception as exc:
        error = exc
    synchronized_check(error is None, device, "某个进程的前向或反向计算失败: " + str(error))
    # 空闲进程贡献零梯度；SUM 后已经是全局图像平均梯度。
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if dist.is_initialized():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    scaler.unscale_(optimizer)
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                 if p.requires_grad and p.grad is not None)
    synchronized_check(finite, device, "梯度溢出，停止运行且不覆盖上一轮断点")
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], config["training"]["grad_clip"])
    synchronized_check(torch.isfinite(norm), device, "梯度范数不是有限值")
    scaler.step(optimizer)
    scaler.update()


@torch.no_grad()
def validate_model(model, rows, config, rank, world, device):
    model.eval()
    total = torch.zeros(3, dtype=torch.float64, device=device)
    error = None
    try:
        for row in rows[rank::world]:
            rgb, uv = load_batch([row], config["dataset_root"], device)
            output = predict_training(model, rgb, [row])
            nll = gaussian_loss(output, uv).item()
            l1 = (output["mean"] - uv).abs().mean().item()
            if not np.isfinite([nll, l1]).all():
                raise FloatingPointError("验证值不是有限值")
            total += torch.tensor([nll, l1, 1], dtype=torch.float64, device=device)
    except Exception as exc:
        error = exc
    synchronized_check(error is None, device, "验证失败: " + str(error))
    if dist.is_initialized():
        dist.all_reduce(total)
    if total[2].item() != len(rows):
        raise ValueError("验证集没有被完整遍历")
    return {"nll": (total[0] / total[2]).item(), "l1": (total[1] / total[2]).item()}


def train_new_version(config_path, resume=None):
    """待训练版本的独立入口，不接续旧生产训练的中途断点。"""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_new_config(config)
    if (config["seed"] != 20260817 or config["expected_counts"] != {"train": 76642, "val": 10948}
            or config["source_manifest_sha256"] !=
            "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"):
        raise ValueError("本轮数据身份、数量或随机种子改变")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if world > 1:
        if device.type != "cuda":
            raise ValueError("多卡训练入口要求 CUDA")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = training_data(config)
    model = build_training_model(config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    amp = config["training"]["precision"] == "FP16"
    if amp and device.type != "cuda":
        raise ValueError("FP16 训练需要 CUDA")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=65536)
    out = Path(config["output_dir"])
    binding = hashlib.sha256((json.dumps(config, sort_keys=True) + file_sha256(__file__)).encode()).hexdigest()
    state = {"epoch": 0, "best": float("inf"), "anchor": float("inf"), "bad_epochs": 0}
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["schema"] != "readable_new_training_v1" or payload["binding"] != binding:
            raise ValueError("新版本断点的源码或配置不匹配")
        if payload["world_size"] != world:
            raise ValueError("恢复时不改变进程数")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        state = payload["state"]
        torch.set_rng_state(payload["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"][rank], device)
    if rank == 0:
        if not resume:
            out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    for epoch in range(state["epoch"] + 1, config["training"]["epochs"] + 1):
        order_path = Path(config["orders_dir"]) / ("epoch_%03d.json" % epoch)
        expected = config["order_sha256"][str(epoch)]
        if file_sha256(order_path) != expected:
            raise ValueError("训练顺序文件被改变")
        order = json.loads(order_path.read_text())["indices"]
        if sorted(order) != list(range(len(data["train"]))):
            raise ValueError("训练顺序没有完整且唯一地覆盖训练集")
        for start in range(0, len(order), 16):
            rows = [data["train"][i] for i in order[start:start + 16]]
            global_step(model, optimizer, scaler, rows, config, rank, world, device)
        metrics = validate_model(model, data["val"], config, rank, world, device)
        value = metrics[config["selection_metric"]]
        is_best = value < state["best"]
        state["best"] = min(state["best"], value)
        if value < state["anchor"] - 1e-4:
            state.update(anchor=value, bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        scheduler.step(value)
        state["epoch"] = epoch
        cuda_rng = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(cuda_rng, torch.cuda.get_rng_state(device))
        elif device.type == "cuda":
            cuda_rng = [torch.cuda.get_rng_state(device)]
        if rank == 0:
            payload = {"schema": "readable_new_training_v1", "binding": binding,
                       "world_size": world, "architecture_id": ARCHITECTURE_ID,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "state": state, "metrics": metrics, "rng": torch.get_rng_state(),
                       "cuda_rng": cuda_rng}
            temporary = out / "checkpoint_last.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, out / "checkpoint_last.pt")
            if is_best:
                torch.save(payload, out / "checkpoint_best.pt")
            with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        if state["bad_epochs"] >= 5:
            break
    if dist.is_initialized():
        dist.destroy_process_group()


def training_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="仅接受本版本保存的完整轮末断点")
    args = parser.parse_args()
    train_new_version(args.config, args.resume)

# 本段在构建时加入 Swin.py，模型定义和训练入口位于同一文件。
ARCHITECTURE_ID = "swin_t_encoder_unet_decoder_gaussian_proposed_v1"


def build_training_model(config, device):
    return SwinUVUncertainty().float().to(device)


def predict_training(model, rgb, rows):
    return model(rgb)


def validate_new_config(config):
    if config["architecture_id"] != ARCHITECTURE_ID:
        raise ValueError("不是这次讨论后的 Swin 结构")
    expected = {"embed_dim": 96, "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24], "decoder_depths": [6, 2, 2],
                "window_size": 7, "patch_size": 4, "drop_path": 0.0}
    if config["model"] != expected:
        raise ValueError("Swin 架构配置改变")
    if config["selection_metric"] != "l1":
        raise ValueError("图像重建按验证集 L1 选择断点")
    train = config["training"]
    if (train["global_batch_size"] != 16 or train["epochs"] != 20
            or train["lr"] != 1e-4 or train["weight_decay"] != 1e-4
            or train["grad_clip"] != 1 or train["precision"] != "FP16"):
        raise ValueError("训练设置与这份候选配置不一致")


if __name__ == "__main__":
    training_main()
