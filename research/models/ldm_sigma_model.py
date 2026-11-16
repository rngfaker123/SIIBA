# 原 LDM 生成核心。后接新增高斯方差头，该头尚未训练
"""原 LDM 生成核心。后接新增高斯方差头，该头尚未训练"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = "rgb_uv_f4_kl_latent_vdiffusion_v1"
ARCHITECTURE = {
    "autoencoder": {"base_channels": 32, "latent_channels": 4, "compression": 4,
                    "multipliers": [1, 2, 4], "blocks_per_level": 2,
                    "uv_skip_connections": False, "decoder_output": "sigmoid"},
    "denoiser": {"base_channels": 64, "multipliers": [1, 2, 4, 4],
                 "blocks_per_level": 2, "time_dimension": 256,
                 "rgb_condition": "learned_f4_convolutional_pyramid_concat",
                 "attention": "bottleneck_only_sdpa"},
    "diffusion": {"prediction_type": "v", "timesteps": 1000,
                  "beta_start": 1e-4, "beta_end": .02,
                  "schedule_construction": "float64_linear_and_cumprod_then_float32",
                  "eta": 0.0, "latent_clipping": False,
                  "ddim_timesteps": "round_linspace_999_to_0_inclusive",
                  "noise_namespace": "ldm_uv_v1"},
}

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Non-finite " + name)
    return value

def _image(value, name, channels=3, image_range=False):
    _require(isinstance(value, torch.Tensor) and value.is_floating_point()
             and value.ndim == 4 and value.shape[1] == channels
             and all(size > 0 for size in value.shape), "Invalid NCHW " + name)
    _finite(value, name)
    if image_range:
        _require(bool((value >= 0).all()) and bool((value <= 1).all()),
                 name + " must be in [0,1]")
    return value

def _norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels, eps=1e-6)

class ResBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_dimension=None):
        super().__init__()
        self.norm1 = _norm(incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = _norm(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
        self.time = nn.Linear(time_dimension, outgoing) if time_dimension else None

    def forward(self, value, time=None):
        hidden = self.conv1(F.silu(self.norm1(value)))
        if self.time is not None:
            _require(time is not None, "Missing time embedding")
            hidden = hidden + self.time(F.silu(time))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(value) + hidden

class KLUVAutoencoder(nn.Module):
    """"""

    def __init__(self, base_channels=32, latent_channels=4):
        super().__init__()
        _require(base_channels > 0 and latent_channels > 0, "Invalid autoencoder width")
        self.latent_channels = latent_channels
        self.compression = 4
        widths = [base_channels * factor for factor in (1, 2, 4)]
        self.encoder_in = nn.Conv2d(3, widths[0], 3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(ResBlock(width, width), ResBlock(width, width))
            for width in widths])
        self.encoder_down = nn.ModuleList([
            nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1) for i in range(2)])
        self.posterior = nn.Sequential(_norm(widths[-1]), nn.SiLU(),
                                       nn.Conv2d(widths[-1], 2 * latent_channels, 3, padding=1))
        self.decoder_in = nn.Conv2d(latent_channels, widths[-1], 3, padding=1)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(ResBlock(width, width), ResBlock(width, width))
            for width in reversed(widths)])
        self.decoder_up = nn.ModuleList([
            nn.Conv2d(widths[2 - i], widths[1 - i], 3, padding=1) for i in range(2)])
        self.decoder_out = nn.Sequential(_norm(widths[0]), nn.SiLU(),
                                         nn.Conv2d(widths[0], 3, 3, padding=1))

    def encode(self, image):
        _image(image, "UV autoencoder input", image_range=True)
        _require(image.shape[-2] % 4 == 0 and image.shape[-1] % 4 == 0,
                 "UV dimensions must be divisible by four")
        hidden = self.encoder_in(image)
        for index, block in enumerate(self.encoder_blocks):
            hidden = block(hidden)
            if index < len(self.encoder_down):
                hidden = self.encoder_down[index](hidden)
        mu, logvar = self.posterior(hidden).float().chunk(2, dim=1)
        return _finite(mu, "posterior mean"), _finite(logvar, "posterior log variance")

    def decode(self, latent):
        _image(latent, "autoencoder latent", channels=self.latent_channels)
        hidden = self.decoder_in(latent)
        for index, block in enumerate(self.decoder_blocks):
            hidden = block(hidden)
            if index < len(self.decoder_up):
                hidden = self.decoder_up[index](F.interpolate(hidden, scale_factor=2, mode="nearest"))
        logits = _finite(self.decoder_out(hidden).float(), "decoder logits")
        return _finite(torch.sigmoid(logits), "UV reconstruction")

    def forward(self, image, sample=True, generator=None):
        mu, logvar = self.encode(image)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            if sample:
                std = _finite(torch.exp(.5 * logvar), "posterior standard deviation")
                noise = torch.randn(mu.shape, dtype=torch.float32, device=mu.device, generator=generator)
                latent = _finite(mu + std * noise, "sampled latent")
            else:
                latent = mu
        return {"reconstruction": self.decode(latent), "mu": mu,
                "logvar": logvar, "z": latent}

def kl_divergence(mu, logvar):
    _require(mu.shape == logvar.shape and mu.device == logvar.device, "Posterior shape/device mismatch")
    _finite(mu, "KL posterior mean")
    _finite(logvar, "KL log variance")
    with torch.autocast(device_type=mu.device.type, enabled=False):
        variance = _finite(logvar.float().exp(), "KL posterior variance")
        result = .5 * (mu.float().square() + variance - 1 - logvar.float()).mean()
    return _finite(result, "KL loss")

def sinusoidal_time(timestep, dimension=256):
    _require(dimension >= 4 and dimension % 2 == 0 and timestep.ndim == 1,
             "Invalid sinusoidal time arguments")
    with torch.autocast(device_type=timestep.device.type, enabled=False):
        frequencies = torch.exp(-math.log(10000.) * torch.arange(
            dimension // 2, dtype=torch.float32, device=timestep.device) / (dimension // 2 - 1))
        angles = timestep.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)

class BottleneckAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.heads = 4 if channels % 4 == 0 else 1

    def forward(self, value):
        batch, channels, height, width = value.shape
        qkv = self.qkv(self.norm(value)).reshape(batch, 3, self.heads,
                                                 channels // self.heads, height * width)
        query, key, content = [qkv[:, index].transpose(-2, -1) for index in range(3)]
        hidden = F.scaled_dot_product_attention(query, key, content, dropout_p=0.)
        hidden = hidden.transpose(-2, -1).reshape(batch, channels, height, width)
        return value + self.out(hidden)

class RGBLatentVDenoiser(nn.Module):
    def __init__(self, base_channels=64, latent_channels=4, time_dimension=256):
        super().__init__()
        _require(base_channels > 0 and latent_channels > 0, "Invalid denoiser width")
        self.latent_channels, self.time_dimension = latent_channels, time_dimension
        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 4)]
        self.time = nn.Sequential(nn.Linear(time_dimension, time_dimension), nn.SiLU(),
                                  nn.Linear(time_dimension, time_dimension))
        self.condition_in = nn.Sequential(nn.Conv2d(3, base_channels // 2, 3, stride=2, padding=1),
                                          nn.SiLU(), nn.Conv2d(base_channels // 2, widths[0],
                                                              3, stride=2, padding=1), nn.SiLU())
        self.condition_down = nn.ModuleList([nn.Conv2d(widths[i], widths[i + 1], 3, stride=2,
                                                       padding=1) for i in range(3)])
        self.input = nn.Conv2d(latent_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            nn.ModuleList((ResBlock(2 * width, width, time_dimension),
                           ResBlock(width, width, time_dimension))) for width in widths])
        self.down = nn.ModuleList([nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1)
                                   for i in range(3)])
        self.attention = BottleneckAttention(widths[-1])
        self.middle = ResBlock(widths[-1], widths[-1], time_dimension)
        self.up_blocks = nn.ModuleList([
            nn.ModuleList((ResBlock(3 * width, width, time_dimension),
                           ResBlock(width, width, time_dimension))) for width in reversed(widths)])
        self.up = nn.ModuleList([nn.Conv2d(widths[3 - i], widths[2 - i], 3, padding=1)
                                 for i in range(3)])
        self.output = nn.Sequential(_norm(widths[0]), nn.SiLU(),
                                    nn.Conv2d(widths[0], latent_channels, 3, padding=1))

    def forward(self, noisy_latent, timestep, rgb):
        _image(noisy_latent, "noisy latent", self.latent_channels)
        _image(rgb, "RGB condition", image_range=True)
        _require(rgb.shape[0] == noisy_latent.shape[0] and rgb.device == noisy_latent.device
                 and rgb.shape[-2] == noisy_latent.shape[-2] * 4
                 and rgb.shape[-1] == noisy_latent.shape[-1] * 4,
                 "RGB/latent geometry mismatch")
        _require(noisy_latent.shape[-2] % 8 == 0 and noisy_latent.shape[-1] % 8 == 0,
                 "Latent dimensions must be divisible by eight")
        _require(timestep.ndim == 1 and timestep.shape[0] == rgb.shape[0]
                 and timestep.device == rgb.device, "Invalid denoiser timesteps")
        condition = self.condition_in(rgb)
        conditions = [condition]
        for layer in self.condition_down:
            condition = F.silu(layer(condition))
            conditions.append(condition)
        time = self.time(sinusoidal_time(timestep, self.time_dimension))
        hidden, skips = self.input(noisy_latent), []
        for index, blocks in enumerate(self.down_blocks):
            hidden = blocks[0](torch.cat((hidden, conditions[index]), dim=1), time)
            hidden = blocks[1](hidden, time)
            skips.append(hidden)
            if index < 3:
                hidden = self.down[index](hidden)
        hidden = self.middle(self.attention(hidden), time)
        for index, blocks in enumerate(self.up_blocks):
            level = 3 - index
            hidden = blocks[0](torch.cat((hidden, skips[level], conditions[level]), dim=1), time)
            hidden = blocks[1](hidden, time)
            if index < 3:
                hidden = self.up[index](F.interpolate(hidden, scale_factor=2, mode="nearest"))
        return _finite(self.output(hidden).float(), "latent velocity prediction")

def _alpha(alpha_bar, reference):
    alpha = torch.as_tensor(alpha_bar, dtype=torch.float32, device=reference.device)
    if alpha.ndim == 1 and alpha.shape[0] == reference.shape[0]:
        alpha = alpha[:, None, None, None]
    _require(alpha.ndim == 0 or tuple(alpha.shape) == (reference.shape[0], 1, 1, 1),
             "Invalid alpha broadcast")
    _finite(alpha, "alpha_bar")
    _require(bool((alpha > 0).all()) and bool((alpha <= 1).all()), "Alpha must be in (0,1]")
    return alpha

def velocity_target(clean, noise, alpha_bar):
    _require(clean.shape == noise.shape and clean.device == noise.device, "Latent/noise mismatch")
    _finite(clean, "clean latent")
    _finite(noise, "forward noise")
    with torch.autocast(device_type=clean.device.type, enabled=False):
        alpha = _alpha(alpha_bar, clean)
        result = alpha.sqrt() * noise.float() - (1 - alpha).sqrt() * clean.float()
    return _finite(result, "velocity target")

def v_to_x0_epsilon(sample, velocity, alpha_bar):
    _require(sample.shape == velocity.shape and sample.device == velocity.device,
             "Sample/velocity mismatch")
    _finite(sample, "noisy latent")
    _finite(velocity, "velocity")
    with torch.autocast(device_type=sample.device.type, enabled=False):
        alpha = _alpha(alpha_bar, sample)
        clean = alpha.sqrt() * sample.float() - (1 - alpha).sqrt() * velocity.float()
        noise = (1 - alpha).sqrt() * sample.float() + alpha.sqrt() * velocity.float()
    return _finite(clean, "predicted clean latent"), _finite(noise, "predicted noise")

class VPLinearSchedule(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=.02):
        super().__init__()
        _require(timesteps >= 2 and 0 < beta_start <= beta_end < 1, "Invalid VP schedule")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", (1 - betas).cumprod(0).float())

    def alpha_at(self, timestep):
        timestep = torch.as_tensor(timestep, device=self.alphas_cumprod.device)
        _require(timestep.dtype in (torch.int32, torch.int64) and bool((timestep >= 0).all())
                 and bool((timestep < self.timesteps).all()), "Invalid noise timestep")
        return self.alphas_cumprod[timestep.long()]

    def add_noise(self, clean, noise, timestep):
        _require(clean.shape == noise.shape and clean.device == noise.device, "Forward noise mismatch")
        _finite(clean, "forward clean latent")
        _finite(noise, "forward noise")
        with torch.autocast(device_type=clean.device.type, enabled=False):
            alpha = _alpha(self.alpha_at(timestep), clean)
            result = alpha.sqrt() * clean.float() + (1 - alpha).sqrt() * noise.float()
        return _finite(result, "forward noisy latent")

    def v_target(self, clean, noise, timestep):
        return velocity_target(clean, noise, self.alpha_at(timestep))

    def v_to_x0_epsilon(self, sample, velocity, timestep):
        return v_to_x0_epsilon(sample, velocity, self.alpha_at(timestep))

def _latent_statistics(mean, std, reference):
    result = []
    for value, name in ((mean, "latent mean"), (std, "latent std")):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=reference.device)
        _require(tensor.numel() == reference.shape[1], "Expected one " + name + " per channel")
        result.append(_finite(tensor.reshape(1, reference.shape[1], 1, 1), name))
    _require(bool((result[1] > 0).all()), "Latent standard deviations must be positive")
    return result

def latent_normalize(latent, mean, std):
    _finite(latent, "latent before normalization")
    with torch.autocast(device_type=latent.device.type, enabled=False):
        mean, std = _latent_statistics(mean, std, latent)
        return _finite((latent.float() - mean) / std, "normalized latent")

def latent_denormalize(latent, mean, std):
    _finite(latent, "normalized latent before decoding")
    with torch.autocast(device_type=latent.device.type, enabled=False):
        mean, std = _latent_statistics(mean, std, latent)
        return _finite(latent.float() * std + mean, "denormalized latent")

def stable_initial_noise(sample_ids, shape, seed=20260817, generation_index=0, device="cpu"):
    _require(len(shape) == 4 and len(sample_ids) == shape[0]
             and all(type(size) is int and size > 0 for size in shape), "Noise shape/IDs mismatch")
    _require(type(seed) is int and type(generation_index) is int and generation_index >= 0,
             "Invalid initial noise seed/index")
    values = []
    for sample_id in sample_ids:
        _require(isinstance(sample_id, str) and bool(sample_id), "Invalid sample ID")
        identity = ("ldm_uv_v1\0%d\0%d\0%s" % (seed, generation_index, sample_id)).encode("utf-8")
        number = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(number)
        values.append(torch.randn((1,) + tuple(shape[1:]), generator=generator, dtype=torch.float32))
    return torch.cat(values, dim=0).to(device)

def ddim_timesteps(timesteps=1000, num_steps=20):
    _require(type(num_steps) is int and 2 <= num_steps <= timesteps, "Invalid DDIM step count")
    sequence = torch.linspace(timesteps - 1, 0, num_steps, dtype=torch.float64).round().long()
    _require(bool((sequence[:-1] > sequence[1:]).all()), "DDIM timesteps must strictly decrease")
    return sequence

def ddim_sample(model, rgb, sample_ids, latent_mean, latent_std, ae, num_steps=20,
                generation_index=0, seed=20260817, schedule=None):
    """"""
    _image(rgb, "sampler RGB", image_range=True)
    _require(rgb.shape[-2] % 32 == 0 and rgb.shape[-1] % 32 == 0,
             "Sampler RGB dimensions must be divisible by 32")
    for module in (model, ae):
        _require(all(parameter.device == rgb.device and parameter.dtype == torch.float32
                     for parameter in module.parameters()), "FP32 sampler model device/dtype mismatch")
    schedule = VPLinearSchedule().to(rgb.device) if schedule is None else schedule
    _require(schedule.alphas_cumprod.device == rgb.device and schedule.alphas_cumprod.dtype == torch.float32,
             "FP32 schedule device/dtype mismatch")
    _require(model.latent_channels == ae.latent_channels, "Codec/denoiser latent channels differ")
    shape = (rgb.shape[0], model.latent_channels, rgb.shape[-2] // 4, rgb.shape[-1] // 4)
    latent = stable_initial_noise(sample_ids, shape, seed, generation_index, rgb.device)
    sequence = ddim_timesteps(schedule.timesteps, num_steps).tolist()
    model_training, ae_training = model.training, ae.training
    try:
        model.eval()
        ae.eval()
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            for index, timestep in enumerate(sequence):
                time = torch.full((rgb.shape[0],), timestep, device=rgb.device, dtype=torch.long)
                velocity = model(latent, time, rgb.float())
                clean, noise = schedule.v_to_x0_epsilon(latent, velocity, time)
                if index + 1 == len(sequence):
                    latent = clean
                else:
                    following = schedule.alphas_cumprod[sequence[index + 1]]
                    latent = _finite(following.sqrt() * clean + (1 - following).sqrt() * noise,
                                     "DDIM next latent")
            return _finite(ae.decode(latent_denormalize(latent, latent_mean, latent_std)),
                           "generated UV image")
    finally:
        model.train(model_training)
        ae.train(ae_training)

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

# 保留 AE、条件去噪器和 DDIM 生成路径，只新增一个图像域方差头。
ARCHITECTURE_ID = "ldm_fixed_generator_gaussian_head_proposed_v1"
VARIANCE_FLOOR = 1e-6


class LDMGaussian(nn.Module):
    """输入 RGB，经固定生成器得到 UV，并在解码特征上预测残差尺度。"""

    def __init__(self, ae=None, denoiser=None, latent_mean=None, latent_std=None):
        super().__init__()
        self.ae = KLUVAutoencoder() if ae is None else ae
        self.denoiser = RGBLatentVDenoiser() if denoiser is None else denoiser
        self.register_buffer("latent_mean", torch.tensor(
            [0.] * 4 if latent_mean is None else latent_mean, dtype=torch.float32))
        self.register_buffer("latent_std", torch.tensor(
            [1.] * 4 if latent_std is None else latent_std, dtype=torch.float32))
        channels = self.ae.decoder_out[2].in_channels
        self.log_variance_head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)
        self.ae.requires_grad_(False)
        self.denoiser.requires_grad_(False)
        self.ae.eval()
        self.denoiser.eval()

    def train(self, mode=True):
        super().train(mode)
        self.ae.eval()
        self.denoiser.eval()
        return self

    def forward(self, rgb, sample_ids):
        # 捕获原输出卷积之前的特征，不改变 AE 原有前向计算。
        # 这里的 F 在默认配置下为 [B,32,H,W]。
        captured = []
        handle = self.ae.decoder_out[2].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach()))
        try:
            mean = ddim_sample(self.denoiser, rgb, sample_ids,
                               self.latent_mean, self.latent_std, self.ae,
                               num_steps=20, generation_index=0, seed=20260817)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("没有唯一取得 AE 解码末端特征")
        # 生成部分不反传。只有这一层通过真实 UV 的高斯损失学习。
        raw_s = self.log_variance_head(captured[0]).float()
        logvar = torch.logaddexp(raw_s, raw_s.new_tensor(math.log(VARIANCE_FLOOR)))
        return {"mean": mean, "raw_log_variance": raw_s, "log_variance": logvar,
                "variance": logvar.exp(), "sigma": (0.5 * logvar).exp()}


def build_training_model(config, device):
    # 仅从指定且核验过的同轮 AE、去噪器与训练集统计初始化。
    loaded = {}
    for kind in ("ae", "denoiser", "latent_statistics"):
        descriptor = config[kind]
        if file_sha256(descriptor["path"]) != descriptor["sha256"]:
            raise ValueError(kind + " 来源哈希不匹配")
        if kind == "latent_statistics":
            loaded[kind] = json.loads(Path(descriptor["path"]).read_text())
        else:
            payload = torch.load(descriptor["path"], map_location="cpu", weights_only=False)
            loaded[kind] = payload[descriptor["state_key"]]
    stats = loaded["latent_statistics"]
    if (stats.get("test_split_used") is not False
            or stats.get("images") != config["expected_counts"]["train"]
            or stats.get("source_manifest_sha256") != config["source_manifest_sha256"]
            or stats.get("ae_checkpoint_sha256") != config["ae"]["sha256"]):
        raise ValueError("潜变量统计没有绑定本轮训练集和指定 AE")
    mean = stats[config["statistics_mean_key"]]
    std = stats[config["statistics_std_key"]]
    if len(mean) != 4 or len(std) != 4 or not np.isfinite(mean + std).all() or min(std) <= 0:
        raise ValueError("需要训练集得到的四通道有限均值和正标准差")
    model = LDMGaussian(latent_mean=mean, latent_std=std)
    model.ae.load_state_dict(loaded["ae"], strict=True)
    model.denoiser.load_state_dict(loaded["denoiser"], strict=True)
    return model.float().to(device)


def predict_training(model, rgb, rows):
    return model(rgb, [row["sample_id"] for row in rows])


def validate_new_config(config):
    if config["architecture_id"] != ARCHITECTURE_ID or config["selection_metric"] != "nll":
        raise ValueError("新增方差头按验证集 NLL 选择，固定生成图像的 L1 不能用来选择方差头")
    if config["training"]["precision"] != "FP32":
        raise ValueError("本版固定生成和方差头训练均使用 FP32")
    if (config["training"]["global_batch_size"] != 16
            or config["training"]["microbatch_size"] != 1
            or config["training"]["epochs"] != 20
            or config["training"]["lr"] != 1e-4
            or config["training"]["weight_decay"] != 1e-4
            or config["training"]["grad_clip"] != 1):
        raise ValueError("新增方差头使用全局 16、单次 1 张图像")




# 原 AE 和条件扩散训练源码，使用 --mode 调用；新增方差头使用 --sigma-config


"""原 AE 和条件扩散训练源码，使用 --mode 调用；新增方差头使用 --sigma-config"""

import os as _plain_os
from pathlib import Path as _PlainPath
from types import SimpleNamespace as _PlainNamespace

_PLAIN_WORKSPACE = _PlainPath(_plain_os.environ.get(
    "CLEAN_HANDOFF_WORKSPACE", str(_PlainPath(__file__).resolve().parents[2]))).resolve()

model__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'src/uv_latent_diffusion_v1.py')
"""True RGB-conditioned UV latent diffusion, trained from scratch in two stages.

The UV autoencoder has no input-to-output skip. The denoiser consumes RGB and
noisy latent only, never UV at inference. Latent normalization uses train-only
posterior-mean statistics, not a borrowed pretrained-model scale. All posterior
and diffusion algebra is FP32 with explicit non-finite failures, never clamps.
"""

import hashlib as model__hashlib
import math as model__math

import torch as model__torch
from torch import nn as model__nn
from torch.nn import functional as model__F

model__MODEL_VERSION = "rgb_uv_f4_kl_latent_vdiffusion_v1"
model__ARCHITECTURE = {
    "autoencoder": {"base_channels": 32, "latent_channels": 4, "compression": 4,
                    "multipliers": [1, 2, 4], "blocks_per_level": 2,
                    "uv_skip_connections": False, "decoder_output": "sigmoid"},
    "denoiser": {"base_channels": 64, "multipliers": [1, 2, 4, 4],
                 "blocks_per_level": 2, "time_dimension": 256,
                 "rgb_condition": "learned_f4_convolutional_pyramid_concat",
                 "attention": "bottleneck_only_sdpa"},
    "diffusion": {"prediction_type": "v", "timesteps": 1000,
                  "beta_start": 1e-4, "beta_end": .02,
                  "schedule_construction": "float64_linear_and_cumprod_then_float32",
                  "eta": 0.0, "latent_clipping": False,
                  "ddim_timesteps": "round_linspace_999_to_0_inclusive",
                  "noise_namespace": "ldm_uv_v1"},
}

def model___require(condition, message):
    if not condition:
        raise ValueError(message)

def model___finite(value, name):
    if not bool(model__torch.isfinite(value).all()):
        raise FloatingPointError("Non-finite " + name)
    return value

def model___image(value, name, channels=3, image_range=False):
    model___require(isinstance(value, model__torch.Tensor) and value.is_floating_point()
             and value.ndim == 4 and value.shape[1] == channels
             and all(size > 0 for size in value.shape), "Invalid NCHW " + name)
    model___finite(value, name)
    if image_range:
        model___require(bool((value >= 0).all()) and bool((value <= 1).all()),
                 name + " must be in [0,1]")
    return value

def model___norm(channels):
    return model__nn.GroupNorm(model__math.gcd(32, channels), channels, eps=1e-6)

class model__ResBlock(model__nn.Module):
    def __init__(self, incoming, outgoing, time_dimension=None):
        super().__init__()
        self.norm1 = model___norm(incoming)
        self.conv1 = model__nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = model___norm(outgoing)
        self.conv2 = model__nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = model__nn.Identity() if incoming == outgoing else model__nn.Conv2d(incoming, outgoing, 1)
        self.time = model__nn.Linear(time_dimension, outgoing) if time_dimension else None

    def forward(self, value, time=None):
        hidden = self.conv1(model__F.silu(self.norm1(value)))
        if self.time is not None:
            model___require(time is not None, "Missing time embedding")
            hidden = hidden + self.time(model__F.silu(time))[:, :, None, None]
        hidden = self.conv2(model__F.silu(self.norm2(hidden)))
        return self.skip(value) + hidden

class model__KLUVAutoencoder(model__nn.Module):
    """"""

    def __init__(self, base_channels=32, latent_channels=4):
        super().__init__()
        model___require(base_channels > 0 and latent_channels > 0, "Invalid autoencoder width")
        self.latent_channels = latent_channels
        self.compression = 4
        widths = [base_channels * factor for factor in (1, 2, 4)]
        self.encoder_in = model__nn.Conv2d(3, widths[0], 3, padding=1)
        self.encoder_blocks = model__nn.ModuleList([
            model__nn.Sequential(model__ResBlock(width, width), model__ResBlock(width, width))
            for width in widths])
        self.encoder_down = model__nn.ModuleList([
            model__nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1) for i in range(2)])
        self.posterior = model__nn.Sequential(model___norm(widths[-1]), model__nn.SiLU(),
                                       model__nn.Conv2d(widths[-1], 2 * latent_channels, 3, padding=1))
        self.decoder_in = model__nn.Conv2d(latent_channels, widths[-1], 3, padding=1)
        self.decoder_blocks = model__nn.ModuleList([
            model__nn.Sequential(model__ResBlock(width, width), model__ResBlock(width, width))
            for width in reversed(widths)])
        self.decoder_up = model__nn.ModuleList([
            model__nn.Conv2d(widths[2 - i], widths[1 - i], 3, padding=1) for i in range(2)])
        self.decoder_out = model__nn.Sequential(model___norm(widths[0]), model__nn.SiLU(),
                                         model__nn.Conv2d(widths[0], 3, 3, padding=1))

    def encode(self, image):
        model___image(image, "UV autoencoder input", image_range=True)
        model___require(image.shape[-2] % 4 == 0 and image.shape[-1] % 4 == 0,
                 "UV dimensions must be divisible by four")
        hidden = self.encoder_in(image)
        for index, block in enumerate(self.encoder_blocks):
            hidden = block(hidden)
            if index < len(self.encoder_down):
                hidden = self.encoder_down[index](hidden)
        mu, logvar = self.posterior(hidden).float().chunk(2, dim=1)
        return model___finite(mu, "posterior mean"), model___finite(logvar, "posterior log variance")

    def decode(self, latent):
        model___image(latent, "autoencoder latent", channels=self.latent_channels)
        hidden = self.decoder_in(latent)
        for index, block in enumerate(self.decoder_blocks):
            hidden = block(hidden)
            if index < len(self.decoder_up):
                hidden = self.decoder_up[index](model__F.interpolate(hidden, scale_factor=2, mode="nearest"))
        logits = model___finite(self.decoder_out(hidden).float(), "decoder logits")
        return model___finite(model__torch.sigmoid(logits), "UV reconstruction")

    def forward(self, image, sample=True, generator=None):
        mu, logvar = self.encode(image)
        with model__torch.autocast(device_type=mu.device.type, enabled=False):
            if sample:
                std = model___finite(model__torch.exp(.5 * logvar), "posterior standard deviation")
                noise = model__torch.randn(mu.shape, dtype=model__torch.float32, device=mu.device, generator=generator)
                latent = model___finite(mu + std * noise, "sampled latent")
            else:
                latent = mu
        return {"reconstruction": self.decode(latent), "mu": mu,
                "logvar": logvar, "z": latent}

def model__kl_divergence(mu, logvar):
    model___require(mu.shape == logvar.shape and mu.device == logvar.device, "Posterior shape/device mismatch")
    model___finite(mu, "KL posterior mean")
    model___finite(logvar, "KL log variance")
    with model__torch.autocast(device_type=mu.device.type, enabled=False):
        variance = model___finite(logvar.float().exp(), "KL posterior variance")
        result = .5 * (mu.float().square() + variance - 1 - logvar.float()).mean()
    return model___finite(result, "KL loss")

def model__sinusoidal_time(timestep, dimension=256):
    model___require(dimension >= 4 and dimension % 2 == 0 and timestep.ndim == 1,
             "Invalid sinusoidal time arguments")
    with model__torch.autocast(device_type=timestep.device.type, enabled=False):
        frequencies = model__torch.exp(-model__math.log(10000.) * model__torch.arange(
            dimension // 2, dtype=model__torch.float32, device=timestep.device) / (dimension // 2 - 1))
        angles = timestep.float()[:, None] * frequencies[None]
        return model__torch.cat((angles.sin(), angles.cos()), dim=1)

class model__BottleneckAttention(model__nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = model___norm(channels)
        self.qkv = model__nn.Conv2d(channels, channels * 3, 1)
        self.out = model__nn.Conv2d(channels, channels, 1)
        self.heads = 4 if channels % 4 == 0 else 1

    def forward(self, value):
        batch, channels, height, width = value.shape
        qkv = self.qkv(self.norm(value)).reshape(batch, 3, self.heads,
                                                 channels // self.heads, height * width)
        query, key, content = [qkv[:, index].transpose(-2, -1) for index in range(3)]
        hidden = model__F.scaled_dot_product_attention(query, key, content, dropout_p=0.)
        hidden = hidden.transpose(-2, -1).reshape(batch, channels, height, width)
        return value + self.out(hidden)

class model__RGBLatentVDenoiser(model__nn.Module):
    def __init__(self, base_channels=64, latent_channels=4, time_dimension=256):
        super().__init__()
        model___require(base_channels > 0 and latent_channels > 0, "Invalid denoiser width")
        self.latent_channels, self.time_dimension = latent_channels, time_dimension
        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 4)]
        self.time = model__nn.Sequential(model__nn.Linear(time_dimension, time_dimension), model__nn.SiLU(),
                                  model__nn.Linear(time_dimension, time_dimension))
        self.condition_in = model__nn.Sequential(model__nn.Conv2d(3, base_channels // 2, 3, stride=2, padding=1),
                                          model__nn.SiLU(), model__nn.Conv2d(base_channels // 2, widths[0],
                                                              3, stride=2, padding=1), model__nn.SiLU())
        self.condition_down = model__nn.ModuleList([model__nn.Conv2d(widths[i], widths[i + 1], 3, stride=2,
                                                       padding=1) for i in range(3)])
        self.input = model__nn.Conv2d(latent_channels, widths[0], 3, padding=1)
        self.down_blocks = model__nn.ModuleList([
            model__nn.ModuleList((model__ResBlock(2 * width, width, time_dimension),
                           model__ResBlock(width, width, time_dimension))) for width in widths])
        self.down = model__nn.ModuleList([model__nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1)
                                   for i in range(3)])
        self.attention = model__BottleneckAttention(widths[-1])
        self.middle = model__ResBlock(widths[-1], widths[-1], time_dimension)
        self.up_blocks = model__nn.ModuleList([
            model__nn.ModuleList((model__ResBlock(3 * width, width, time_dimension),
                           model__ResBlock(width, width, time_dimension))) for width in reversed(widths)])
        self.up = model__nn.ModuleList([model__nn.Conv2d(widths[3 - i], widths[2 - i], 3, padding=1)
                                 for i in range(3)])
        self.output = model__nn.Sequential(model___norm(widths[0]), model__nn.SiLU(),
                                    model__nn.Conv2d(widths[0], latent_channels, 3, padding=1))

    def forward(self, noisy_latent, timestep, rgb):
        model___image(noisy_latent, "noisy latent", self.latent_channels)
        model___image(rgb, "RGB condition", image_range=True)
        model___require(rgb.shape[0] == noisy_latent.shape[0] and rgb.device == noisy_latent.device
                 and rgb.shape[-2] == noisy_latent.shape[-2] * 4
                 and rgb.shape[-1] == noisy_latent.shape[-1] * 4,
                 "RGB/latent geometry mismatch")
        model___require(noisy_latent.shape[-2] % 8 == 0 and noisy_latent.shape[-1] % 8 == 0,
                 "Latent dimensions must be divisible by eight")
        model___require(timestep.ndim == 1 and timestep.shape[0] == rgb.shape[0]
                 and timestep.device == rgb.device, "Invalid denoiser timesteps")
        condition = self.condition_in(rgb)
        conditions = [condition]
        for layer in self.condition_down:
            condition = model__F.silu(layer(condition))
            conditions.append(condition)
        time = self.time(model__sinusoidal_time(timestep, self.time_dimension))
        hidden, skips = self.input(noisy_latent), []
        for index, blocks in enumerate(self.down_blocks):
            hidden = blocks[0](model__torch.cat((hidden, conditions[index]), dim=1), time)
            hidden = blocks[1](hidden, time)
            skips.append(hidden)
            if index < 3:
                hidden = self.down[index](hidden)
        hidden = self.middle(self.attention(hidden), time)
        for index, blocks in enumerate(self.up_blocks):
            level = 3 - index
            hidden = blocks[0](model__torch.cat((hidden, skips[level], conditions[level]), dim=1), time)
            hidden = blocks[1](hidden, time)
            if index < 3:
                hidden = self.up[index](model__F.interpolate(hidden, scale_factor=2, mode="nearest"))
        return model___finite(self.output(hidden).float(), "latent velocity prediction")

def model___alpha(alpha_bar, reference):
    alpha = model__torch.as_tensor(alpha_bar, dtype=model__torch.float32, device=reference.device)
    if alpha.ndim == 1 and alpha.shape[0] == reference.shape[0]:
        alpha = alpha[:, None, None, None]
    model___require(alpha.ndim == 0 or tuple(alpha.shape) == (reference.shape[0], 1, 1, 1),
             "Invalid alpha broadcast")
    model___finite(alpha, "alpha_bar")
    model___require(bool((alpha > 0).all()) and bool((alpha <= 1).all()), "Alpha must be in (0,1]")
    return alpha

def model__velocity_target(clean, noise, alpha_bar):
    model___require(clean.shape == noise.shape and clean.device == noise.device, "Latent/noise mismatch")
    model___finite(clean, "clean latent")
    model___finite(noise, "forward noise")
    with model__torch.autocast(device_type=clean.device.type, enabled=False):
        alpha = model___alpha(alpha_bar, clean)
        result = alpha.sqrt() * noise.float() - (1 - alpha).sqrt() * clean.float()
    return model___finite(result, "velocity target")

def model__v_to_x0_epsilon(sample, velocity, alpha_bar):
    model___require(sample.shape == velocity.shape and sample.device == velocity.device,
             "Sample/velocity mismatch")
    model___finite(sample, "noisy latent")
    model___finite(velocity, "velocity")
    with model__torch.autocast(device_type=sample.device.type, enabled=False):
        alpha = model___alpha(alpha_bar, sample)
        clean = alpha.sqrt() * sample.float() - (1 - alpha).sqrt() * velocity.float()
        noise = (1 - alpha).sqrt() * sample.float() + alpha.sqrt() * velocity.float()
    return model___finite(clean, "predicted clean latent"), model___finite(noise, "predicted noise")

class model__VPLinearSchedule(model__nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=.02):
        super().__init__()
        model___require(timesteps >= 2 and 0 < beta_start <= beta_end < 1, "Invalid VP schedule")
        self.timesteps = timesteps
        betas = model__torch.linspace(beta_start, beta_end, timesteps, dtype=model__torch.float64)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", (1 - betas).cumprod(0).float())

    def alpha_at(self, timestep):
        timestep = model__torch.as_tensor(timestep, device=self.alphas_cumprod.device)
        model___require(timestep.dtype in (model__torch.int32, model__torch.int64) and bool((timestep >= 0).all())
                 and bool((timestep < self.timesteps).all()), "Invalid noise timestep")
        return self.alphas_cumprod[timestep.long()]

    def add_noise(self, clean, noise, timestep):
        model___require(clean.shape == noise.shape and clean.device == noise.device, "Forward noise mismatch")
        model___finite(clean, "forward clean latent")
        model___finite(noise, "forward noise")
        with model__torch.autocast(device_type=clean.device.type, enabled=False):
            alpha = model___alpha(self.alpha_at(timestep), clean)
            result = alpha.sqrt() * clean.float() + (1 - alpha).sqrt() * noise.float()
        return model___finite(result, "forward noisy latent")

    def v_target(self, clean, noise, timestep):
        return model__velocity_target(clean, noise, self.alpha_at(timestep))

    def v_to_x0_epsilon(self, sample, velocity, timestep):
        return model__v_to_x0_epsilon(sample, velocity, self.alpha_at(timestep))

def model___latent_statistics(mean, std, reference):
    result = []
    for value, name in ((mean, "latent mean"), (std, "latent std")):
        tensor = model__torch.as_tensor(value, dtype=model__torch.float32, device=reference.device)
        model___require(tensor.numel() == reference.shape[1], "Expected one " + name + " per channel")
        result.append(model___finite(tensor.reshape(1, reference.shape[1], 1, 1), name))
    model___require(bool((result[1] > 0).all()), "Latent standard deviations must be positive")
    return result

def model__latent_normalize(latent, mean, std):
    model___finite(latent, "latent before normalization")
    with model__torch.autocast(device_type=latent.device.type, enabled=False):
        mean, std = model___latent_statistics(mean, std, latent)
        return model___finite((latent.float() - mean) / std, "normalized latent")

def model__latent_denormalize(latent, mean, std):
    model___finite(latent, "normalized latent before decoding")
    with model__torch.autocast(device_type=latent.device.type, enabled=False):
        mean, std = model___latent_statistics(mean, std, latent)
        return model___finite(latent.float() * std + mean, "denormalized latent")

def model__stable_initial_noise(sample_ids, shape, seed=20260817, generation_index=0, device="cpu"):
    model___require(len(shape) == 4 and len(sample_ids) == shape[0]
             and all(type(size) is int and size > 0 for size in shape), "Noise shape/IDs mismatch")
    model___require(type(seed) is int and type(generation_index) is int and generation_index >= 0,
             "Invalid initial noise seed/index")
    values = []
    for sample_id in sample_ids:
        model___require(isinstance(sample_id, str) and bool(sample_id), "Invalid sample ID")
        identity = ("ldm_uv_v1\0%d\0%d\0%s" % (seed, generation_index, sample_id)).encode("utf-8")
        number = int.from_bytes(model__hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)
        generator = model__torch.Generator(device="cpu").manual_seed(number)
        values.append(model__torch.randn((1,) + tuple(shape[1:]), generator=generator, dtype=model__torch.float32))
    return model__torch.cat(values, dim=0).to(device)

def model__ddim_timesteps(timesteps=1000, num_steps=20):
    model___require(type(num_steps) is int and 2 <= num_steps <= timesteps, "Invalid DDIM step count")
    sequence = model__torch.linspace(timesteps - 1, 0, num_steps, dtype=model__torch.float64).round().long()
    model___require(bool((sequence[:-1] > sequence[1:]).all()), "DDIM timesteps must strictly decrease")
    return sequence

def model__ddim_sample(model, rgb, sample_ids, latent_mean, latent_std, ae, num_steps=20,
                generation_index=0, seed=20260817, schedule=None):
    """"""
    model___image(rgb, "sampler RGB", image_range=True)
    model___require(rgb.shape[-2] % 32 == 0 and rgb.shape[-1] % 32 == 0,
             "Sampler RGB dimensions must be divisible by 32")
    for module in (model, ae):
        model___require(all(parameter.device == rgb.device and parameter.dtype == model__torch.float32
                     for parameter in module.parameters()), "FP32 sampler model device/dtype mismatch")
    schedule = model__VPLinearSchedule().to(rgb.device) if schedule is None else schedule
    model___require(schedule.alphas_cumprod.device == rgb.device and schedule.alphas_cumprod.dtype == model__torch.float32,
             "FP32 schedule device/dtype mismatch")
    model___require(model.latent_channels == ae.latent_channels, "Codec/denoiser latent channels differ")
    shape = (rgb.shape[0], model.latent_channels, rgb.shape[-2] // 4, rgb.shape[-1] // 4)
    latent = model__stable_initial_noise(sample_ids, shape, seed, generation_index, rgb.device)
    sequence = model__ddim_timesteps(schedule.timesteps, num_steps).tolist()
    model_training, ae_training = model.training, ae.training
    try:
        model.eval()
        ae.eval()
        with model__torch.no_grad(), model__torch.autocast(device_type=rgb.device.type, enabled=False):
            for index, timestep in enumerate(sequence):
                time = model__torch.full((rgb.shape[0],), timestep, device=rgb.device, dtype=model__torch.long)
                velocity = model(latent, time, rgb.float())
                clean, noise = schedule.v_to_x0_epsilon(latent, velocity, time)
                if index + 1 == len(sequence):
                    latent = clean
                else:
                    following = schedule.alphas_cumprod[sequence[index + 1]]
                    latent = model___finite(following.sqrt() * clean + (1 - following).sqrt() * noise,
                                     "DDIM next latent")
            return model___finite(ae.decode(model__latent_denormalize(latent, latent_mean, latent_std)),
                           "generated UV image")
    finally:
        model.train(model_training)
        ae.train(ae_training)

model__MODULE = _PlainNamespace(**{
    'ARCHITECTURE': model__ARCHITECTURE,
    'BottleneckAttention': model__BottleneckAttention,
    'F': model__F,
    'KLUVAutoencoder': model__KLUVAutoencoder,
    'MODEL_VERSION': model__MODEL_VERSION,
    'RGBLatentVDenoiser': model__RGBLatentVDenoiser,
    'ResBlock': model__ResBlock,
    'VPLinearSchedule': model__VPLinearSchedule,
    '_alpha': model___alpha,
    '_finite': model___finite,
    '_image': model___image,
    '_latent_statistics': model___latent_statistics,
    '_norm': model___norm,
    '_require': model___require,
    'ddim_sample': model__ddim_sample,
    'ddim_timesteps': model__ddim_timesteps,
    'hashlib': model__hashlib,
    'kl_divergence': model__kl_divergence,
    'latent_denormalize': model__latent_denormalize,
    'latent_normalize': model__latent_normalize,
    'math': model__math,
    'nn': model__nn,
    'sinusoidal_time': model__sinusoidal_time,
    'stable_initial_noise': model__stable_initial_noise,
    'torch': model__torch,
    'v_to_x0_epsilon': model__v_to_x0_epsilon,
    'velocity_target': model__velocity_target,
})

batches__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'src/matched_training_batching.py')
"""Exact, unpadded optimizer batches for unequal-size DDP rank batches.

This module performs no training, process initialization, or dataset access.
For global batch 16, three ranks, and a maximum physical microbatch of two,
rank sample counts are 6/5/5 and micro-round sizes are (2,2,2), (2,2,2),
(2,1,1). The final four-sample batch is (2,1,1), not a padded batch of 16.

Each microbatch loss MUST be a mean over its images (with identical per-image
reduction). Multiply it by ``loss_scale`` before backward. Default DDP averages
gradients across ranks, so ``world_size * micro_n / global_n`` produces exactly
the global image-mean gradient, including the short last batch. Do not divide
again by the number of accumulation rounds. Use DDP ``no_sync()`` around BOTH
forward and backward for every microbatch except ``synchronize=True``. Perform
clipping, optimizer/scaler step, and EMA update once after the complete plan.

All ranks perform the same number of nonempty forwards/backwards. Layouts that
cannot satisfy this without padding, dropping, or empty batches fail explicitly.
This assumes sample-independent layers (e.g. GroupNorm), no custom DDP reduction
hook, and synchronized optimizer/AMP overflow decisions across ranks. Unequal
microbatches are not equivalent for training-mode BatchNorm or batchwise losses.

The default ``epoch_order`` is a NEW explicitly named reproducible Python RNG
rule, NOT a claim to reproduce an existing DataLoader shuffle. For historical
matching, pass the exact frozen permutation produced by that audited loader to
``iter_epoch_batches``. In particular, DataLoader generator base-seed draws can
make a plain seed+epoch randperm differ from the historical loader's order.
"""

from dataclasses import dataclass as batches__dataclass
import hashlib as batches__hashlib
import random as batches__random
import struct as batches__struct

batches__ORDER_RULE = "python_random_seed_plus_epoch_shuffle_v1"
batches__BATCHING_RULE = "strided_rank_assignment_balanced_nonempty_microbatches_v1"

def batches___integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value

def batches___indices(values):
    result = tuple(values)
    for value in result:
        batches___integer(value, "sample index")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate sample indices are not allowed")
    return result

def batches__epoch_order(num_samples, seed, epoch):
    """"""
    batches___integer(num_samples, "num_samples", 1)
    batches___integer(seed, "seed")
    batches___integer(epoch, "epoch")
    order = list(range(num_samples))
    batches__random.Random(seed + epoch).shuffle(order)
    return tuple(order)

def batches__order_sha256(order):
    """"""
    order = batches___indices(order)
    digest = batches__hashlib.sha256(b"matched_training_order_v1\x00")
    for index in order:
        if index >= 2 ** 64:
            raise ValueError("Sample index exceeds uint64 hash encoding")
        digest.update(batches__struct.pack("<Q", index))
    return digest.hexdigest()

@batches__dataclass(frozen=True)
class batches__MicroBatch:
    indices: tuple
    loss_scale: float
    synchronize: bool

@batches__dataclass(frozen=True)
class batches__RankBatch:
    rank: int
    indices: tuple
    microbatches: tuple

@batches__dataclass(frozen=True)
class batches__GlobalBatch:
    batch_index: int
    indices: tuple
    rank_batches: tuple

    @property
    def num_samples(self):
        return len(self.indices)

    @property
    def micro_rounds(self):
        return len(self.rank_batches[0].microbatches)

def batches___layout(global_n, world_size, max_microbatch_size):
    batches___integer(global_n, "global batch sample count", 1)
    batches___integer(world_size, "world_size", 1)
    batches___integer(max_microbatch_size, "max_microbatch_size", 1)
    if global_n < world_size:
        raise ValueError(
            "Global batch has fewer samples than ranks; empty ranks, padding, "
            "and dropping are unsupported")
    counts = tuple(len(range(rank, global_n, world_size))
                   for rank in range(world_size))
    rounds = (max(counts) + max_microbatch_size - 1) // max_microbatch_size
    if min(counts) < rounds:
        raise ValueError(
            "Cannot give every rank the same number of nonempty microbatches "
            "within max_microbatch_size; increase the maximum or use a "
            "separately approved layout")
    return counts, rounds

def batches__plan_global_batch(indices, world_size=3, max_microbatch_size=2,
                      batch_index=0):
    """"""
    indices = batches___indices(indices)
    batches___integer(batch_index, "batch_index")
    counts, rounds = batches___layout(len(indices), world_size, max_microbatch_size)
    ranks = []
    for rank, local_n in enumerate(counts):
        local_indices = indices[rank::world_size]
        quotient, remainder = divmod(local_n, rounds)
        cursor = 0
        micros = []
        for micro_index in range(rounds):
            micro_n = quotient + int(micro_index < remainder)
            micros.append(batches__MicroBatch(
                indices=local_indices[cursor:cursor + micro_n],
                loss_scale=world_size * micro_n / len(indices),
                synchronize=(micro_index == rounds - 1),
            ))
            cursor += micro_n
        ranks.append(batches__RankBatch(rank, local_indices, tuple(micros)))
    return batches__GlobalBatch(batch_index, indices, tuple(ranks))

def batches__iter_epoch_batches(order, global_batch_size=16, world_size=3,
                       max_microbatch_size=2, start_batch=0):
    """"""
    order = batches___indices(order)
    if not order:
        raise ValueError("Epoch order must not be empty")
    batches___integer(global_batch_size, "global_batch_size", 1)
    batches___integer(start_batch, "start_batch")
    batches___layout(min(global_batch_size, len(order)), world_size, max_microbatch_size)
    tail = len(order) % global_batch_size
    if tail:
        batches___layout(tail, world_size, max_microbatch_size)
    count = (len(order) + global_batch_size - 1) // global_batch_size
    if start_batch > count:
        raise ValueError("start_batch exceeds the number of epoch batches")

    def plans():
        for batch_index in range(start_batch, count):
            start = batch_index * global_batch_size
            yield batches__plan_global_batch(
                order[start:start + global_batch_size], world_size,
                max_microbatch_size, batch_index)

    return plans()

def batches__resume_metadata(order, next_batch, global_batch_size=16, world_size=3,
                    max_microbatch_size=2):
    """"""
    order = batches___indices(order)
    batches__iter_epoch_batches(order, global_batch_size, world_size,
                       max_microbatch_size, next_batch)
    return {
        "batching_rule": batches__BATCHING_RULE,
        "order_sha256": batches__order_sha256(order),
        "num_samples": len(order),
        "global_batch_size": global_batch_size,
        "world_size": world_size,
        "max_microbatch_size": max_microbatch_size,
        "next_batch": next_batch,
    }

def batches__validate_resume_metadata(order, metadata, global_batch_size=16,
                             world_size=3, max_microbatch_size=2):
    """"""
    if not isinstance(metadata, dict) or "next_batch" not in metadata:
        raise ValueError("Missing batching resume metadata")
    expected = batches__resume_metadata(order, metadata["next_batch"], global_batch_size,
                               world_size, max_microbatch_size)
    if metadata != expected:
        raise ValueError("Resume order, sample count, or batching protocol mismatch")
    return expected["next_batch"]

batches__MODULE = _PlainNamespace(**{
    'BATCHING_RULE': batches__BATCHING_RULE,
    'GlobalBatch': batches__GlobalBatch,
    'MicroBatch': batches__MicroBatch,
    'ORDER_RULE': batches__ORDER_RULE,
    'RankBatch': batches__RankBatch,
    '_indices': batches___indices,
    '_integer': batches___integer,
    '_layout': batches___layout,
    'dataclass': batches__dataclass,
    'epoch_order': batches__epoch_order,
    'hashlib': batches__hashlib,
    'iter_epoch_batches': batches__iter_epoch_batches,
    'order_sha256': batches__order_sha256,
    'plan_global_batch': batches__plan_global_batch,
    'random': batches__random,
    'resume_metadata': batches__resume_metadata,
    'struct': batches__struct,
    'validate_resume_metadata': batches__validate_resume_metadata,
})

data__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'src/highres_patch_dataset.py')

"""Raw-image datasets for the frozen 4000x4000 protocol.

All crop coordinates are defined in a canonical 4000x4000 output space and
mapped back to the source center-square ROI. This preserves normalized field
of view across 8000x4000 and 9728x4864 captures without upsampling the dominant
cohort. RGB and target always receive the exact same spatial transform.
"""

import csv as data__csv
import hashlib as data__hashlib
import math as data__math
import os as data__os

import numpy as data__np
import torch as data__torch
from PIL import Image as data__Image
from torch.utils.data import Dataset as data__Dataset

data__TARGET_MODALITIES = ("uv", "plv", "plh")
data__TARGET_PATH_COLUMN = {"uv": "uv_path", "plv": "plv_path", "plh": "plh_path"}
data__REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "source_width",
    "source_height",
    "rgb_path",
    "uv_path",
    "plv_path",
    "plh_path",
    "eligible_primary",
)
data__EXPECTED_SPLIT_COUNTS = {"train": 80740, "val": 11534, "test": 23069}

class data__HighResolutionManifestError(RuntimeError):
    pass

class data__HighResolutionTestSplitAccessError(RuntimeError):
    pass

def data__load_frozen_raw_manifest(path):
    if not data__os.path.isfile(path):
        raise data__HighResolutionManifestError("Frozen raw manifest missing: %s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = data__csv.DictReader(handle)
        missing = [name for name in data__REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise data__HighResolutionManifestError(
                "Frozen raw manifest is missing columns: %s" % ", ".join(missing)
            )
        rows = [row for row in reader if str(row["eligible_primary"]).strip() == "1"]
    if not rows:
        raise data__HighResolutionManifestError("Frozen raw manifest has no eligible rows")
    return rows

def data__check_frozen_split_counts(rows, strict=True):
    observed = {}
    for row in rows:
        split = str(row["split"]).strip()
        observed[split] = observed.get(split, 0) + 1
    problems = []
    for split, expected in data__EXPECTED_SPLIT_COUNTS.items():
        if observed.get(split, 0) != expected:
            problems.append(
                "%s expected=%d observed=%d"
                % (split, expected, observed.get(split, 0))
            )
    if strict and problems:
        raise data__HighResolutionManifestError(
            "Frozen split counts changed; refusing to continue:\n  %s"
            % "\n  ".join(problems)
        )
    return observed

def data__filter_frozen_split(rows, split):
    return [row for row in rows if str(row["split"]).strip() == split]

def data__deterministic_subset(rows, count):
    if count is None or count <= 0 or count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        key = data__hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
        keyed.append((key, row["sample_id"], row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed[:count]]

def data__resolve_raw_path(dataset_root, manifest_path):
    if data__os.path.isabs(manifest_path):
        return manifest_path
    return data__os.path.join(dataset_root, manifest_path)

def data__center_square_geometry(width, height):
    side = min(int(width), int(height))
    left = (int(width) - side) // 2
    top = (int(height) - side) // 2
    return left, top, side

def data__canonical_patch_to_source_box(
    width, height, canonical_size, patch_size, output_x, output_y
):
    left, top, side = data__center_square_geometry(width, height)
    if side < canonical_size:
        raise data__HighResolutionManifestError(
            "Source side %d is below canonical size %d" % (side, canonical_size)
        )
    if patch_size <= 0 or patch_size > canonical_size:
        raise ValueError("patch_size must be in [1, canonical_size]")
    max_offset = canonical_size - patch_size
    if not (0 <= output_x <= max_offset and 0 <= output_y <= max_offset):
        raise ValueError(
            "Canonical patch origin out of bounds: (%d,%d), max=%d"
            % (output_x, output_y, max_offset)
        )

    scale = float(side) / float(canonical_size)
    source_left = left + int(data__math.floor(output_x * scale))
    source_top = top + int(data__math.floor(output_y * scale))
    source_right = left + int(data__math.ceil((output_x + patch_size) * scale))
    source_bottom = top + int(data__math.ceil((output_y + patch_size) * scale))
    square_right = left + side
    square_bottom = top + side
    source_left = max(left, min(source_left, square_right - 1))
    source_top = max(top, min(source_top, square_bottom - 1))
    source_right = max(source_left + 1, min(source_right, square_right))
    source_bottom = max(source_top + 1, min(source_bottom, square_bottom))
    return (source_left, source_top, source_right, source_bottom)

def data___lanczos():
    return getattr(getattr(data__Image, "Resampling", data__Image), "LANCZOS")

def data___load_rgb_crop(path, expected_size, crop_box, output_size, sample_id, role):
    if not data__os.path.isfile(path):
        raise data__HighResolutionManifestError(
            "Raw image missing: sample_id=%s role=%s path=%s"
            % (sample_id, role, path)
        )
    with data__Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != expected_size:
            raise data__HighResolutionManifestError(
                "Raw size mismatch for %s %s: manifest=%s actual=%s"
                % (sample_id, role, str(expected_size), str(image.size))
            )
        image = image.crop(crop_box)
        if image.size != (output_size, output_size):
            image = image.resize((output_size, output_size), data___lanczos())
        array = data__np.asarray(image, dtype=data__np.uint8)
    array = data__np.transpose(array, (2, 0, 1)).astype(data__np.float32) / 255.0
    return data__torch.from_numpy(data__np.ascontiguousarray(array))

class data__HighResolutionPatchDataset(data__Dataset):
    """"""

    def __init__(
        self,
        rows,
        target,
        dataset_root,
        split_name,
        canonical_size=4000,
        patch_size=512,
        seed=20260817,
        allow_test=False,
    ):
        if target not in data__TARGET_MODALITIES:
            raise ValueError("Unknown target: %s" % target)
        if split_name == "test" and not allow_test:
            raise data__HighResolutionTestSplitAccessError(
                "Refusing to load test images during high-resolution development"
            )
        if not rows:
            raise data__HighResolutionManifestError("No rows for split %s" % split_name)
        if patch_size <= 0 or patch_size > canonical_size:
            raise ValueError("Invalid patch_size=%d" % patch_size)
        self.rows = list(rows)
        self.target = target
        self.target_column = data__TARGET_PATH_COLUMN[target]
        self.dataset_root = dataset_root
        self.split_name = split_name
        self.canonical_size = int(canonical_size)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def patch_origin(self, row, index):
        max_offset = self.canonical_size - self.patch_size
        if max_offset == 0:
            return 0, 0
        token = "%d|%d|%s|%d" % (
            self.seed,
            self.epoch,
            row["sample_id"],
            int(index),
        )
        digest = data__hashlib.sha256(token.encode("utf-8")).digest()
        x_value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        y_value = int.from_bytes(digest[8:16], byteorder="big", signed=False)
        return x_value % (max_offset + 1), y_value % (max_offset + 1)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_id = row["sample_id"]
        width = int(row["source_width"])
        height = int(row["source_height"])
        output_x, output_y = self.patch_origin(row, index)
        source_box = data__canonical_patch_to_source_box(
            width,
            height,
            self.canonical_size,
            self.patch_size,
            output_x,
            output_y,
        )
        expected_size = (width, height)
        rgb_path = data__resolve_raw_path(self.dataset_root, row["rgb_path"])
        target_path = data__resolve_raw_path(self.dataset_root, row[self.target_column])
        rgb = data___load_rgb_crop(
            rgb_path,
            expected_size,
            source_box,
            self.patch_size,
            sample_id,
            "rgb",
        )
        target = data___load_rgb_crop(
            target_path,
            expected_size,
            source_box,
            self.patch_size,
            sample_id,
            self.target,
        )
        return {
            "rgb": rgb,
            "target": target,
            "sample_id": sample_id,
            "index": index,
            "canonical_x": output_x,
            "canonical_y": output_y,
            "source_box": data__torch.tensor(source_box, dtype=data__torch.int64),
        }

class data__HighResolutionFullImageDataset(data__Dataset):
    """"""

    def __init__(
        self,
        rows,
        target,
        dataset_root,
        split_name,
        canonical_size=4000,
        allow_test=False,
    ):
        if target not in data__TARGET_MODALITIES:
            raise ValueError("Unknown target: %s" % target)
        if split_name == "test" and not allow_test:
            raise data__HighResolutionTestSplitAccessError(
                "Refusing to load test images during high-resolution development"
            )
        if not rows:
            raise data__HighResolutionManifestError("No rows for split %s" % split_name)
        self.rows = list(rows)
        self.target = target
        self.target_column = data__TARGET_PATH_COLUMN[target]
        self.dataset_root = dataset_root
        self.split_name = split_name
        self.canonical_size = int(canonical_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_id = row["sample_id"]
        width = int(row["source_width"])
        height = int(row["source_height"])
        left, top, side = data__center_square_geometry(width, height)
        if side < self.canonical_size:
            raise data__HighResolutionManifestError(
                "Source side %d below canonical size %d for %s"
                % (side, self.canonical_size, sample_id)
            )
        box = (left, top, left + side, top + side)
        expected_size = (width, height)
        rgb = data___load_rgb_crop(
            data__resolve_raw_path(self.dataset_root, row["rgb_path"]),
            expected_size,
            box,
            self.canonical_size,
            sample_id,
            "rgb",
        )
        target = data___load_rgb_crop(
            data__resolve_raw_path(self.dataset_root, row[self.target_column]),
            expected_size,
            box,
            self.canonical_size,
            sample_id,
            self.target,
        )
        return {
            "rgb": rgb,
            "target": target,
            "sample_id": sample_id,
            "index": index,
        }

data__MODULE = _PlainNamespace(**{
    'Dataset': data__Dataset,
    'EXPECTED_SPLIT_COUNTS': data__EXPECTED_SPLIT_COUNTS,
    'HighResolutionFullImageDataset': data__HighResolutionFullImageDataset,
    'HighResolutionManifestError': data__HighResolutionManifestError,
    'HighResolutionPatchDataset': data__HighResolutionPatchDataset,
    'HighResolutionTestSplitAccessError': data__HighResolutionTestSplitAccessError,
    'Image': data__Image,
    'REQUIRED_COLUMNS': data__REQUIRED_COLUMNS,
    'TARGET_MODALITIES': data__TARGET_MODALITIES,
    'TARGET_PATH_COLUMN': data__TARGET_PATH_COLUMN,
    '_lanczos': data___lanczos,
    '_load_rgb_crop': data___load_rgb_crop,
    'canonical_patch_to_source_box': data__canonical_patch_to_source_box,
    'center_square_geometry': data__center_square_geometry,
    'check_frozen_split_counts': data__check_frozen_split_counts,
    'csv': data__csv,
    'deterministic_subset': data__deterministic_subset,
    'filter_frozen_split': data__filter_frozen_split,
    'hashlib': data__hashlib,
    'load_frozen_raw_manifest': data__load_frozen_raw_manifest,
    'math': data__math,
    'np': data__np,
    'os': data__os,
    'resolve_raw_path': data__resolve_raw_path,
    'torch': data__torch,
})

layers__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'src/unet_baseline.py')

"""U-Net baseline, architecture-locked to 7,763,107 parameters.

Structure (must match 15_overfit_unet.py):
  DoubleConv(i, o) = Conv3x3(i, o, bias=False) - GroupNorm - ReLU
                     Conv3x3(o, o, bias=False) - GroupNorm - ReLU
  Encoder : 3 -> 32 -> 64 -> 128 -> 256, bottleneck 512 (MaxPool2d(2))
  Decoder : ConvTranspose2d(2x, bias=True) + skip concat + DoubleConv
  Head    : Conv1x1(32, 3, bias=True) + Sigmoid
"""

import torch as layers__torch
import torch.nn as layers__nn
from torch.utils.checkpoint import checkpoint as layers__checkpoint

layers__EXPECTED_PARAMETER_COUNT = 7763107

class layers__ParameterCountMismatch(RuntimeError):
    pass

def layers___group_norm(num_channels, max_groups=8):
    groups = max_groups
    while groups > 1 and num_channels % groups != 0:
        groups = groups - 1
    return layers__nn.GroupNorm(num_groups=groups, num_channels=num_channels)

class layers__DoubleConv(layers__nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layers__DoubleConv, self).__init__()
        self.block = layers__nn.Sequential(
            layers__nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layers___group_norm(out_channels),
            layers__nn.ReLU(inplace=True),
            layers__nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layers___group_norm(out_channels),
            layers__nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class layers__UNetBaseline(layers__nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(layers__UNetBaseline, self).__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.encoder1 = layers__DoubleConv(in_channels, c1)
        self.encoder2 = layers__DoubleConv(c1, c2)
        self.encoder3 = layers__DoubleConv(c2, c3)
        self.encoder4 = layers__DoubleConv(c3, c4)
        self.bottleneck = layers__DoubleConv(c4, c5)

        self.pool = layers__nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = layers__nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.decoder4 = layers__DoubleConv(c5, c4)
        self.upconv3 = layers__nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = layers__DoubleConv(c4, c3)
        self.upconv2 = layers__nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = layers__DoubleConv(c3, c2)
        self.upconv1 = layers__nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = layers__DoubleConv(c2, c1)

        self.head = layers__nn.Conv2d(c1, out_channels, kernel_size=1)
        self.output_activation = layers__nn.Sigmoid()
        self.activation_checkpointing = False

    def set_activation_checkpointing(self, enabled=True):
        """"""
        self.activation_checkpointing = bool(enabled)

    def _checkpointed_forward(self, x):

        if not x.requires_grad:
            x = x.requires_grad_(True)
        e1 = layers__checkpoint(self.encoder1, x)
        e2 = layers__checkpoint(self.encoder2, self.pool(e1))
        e3 = layers__checkpoint(self.encoder3, self.pool(e2))
        e4 = layers__checkpoint(self.encoder4, self.pool(e3))
        b = layers__checkpoint(self.bottleneck, self.pool(e4))

        d4 = layers__checkpoint(self.decoder4, layers__torch.cat([self.upconv4(b), e4], dim=1))
        d3 = layers__checkpoint(self.decoder3, layers__torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = layers__checkpoint(self.decoder2, layers__torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = layers__checkpoint(self.decoder1, layers__torch.cat([self.upconv1(d2), e1], dim=1))
        return self.output_activation(self.head(d1))

    def forward(self, x):
        if self.training and self.activation_checkpointing:
            return self._checkpointed_forward(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(layers__torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(layers__torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(layers__torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(layers__torch.cat([self.upconv1(d2), e1], dim=1))

        return self.output_activation(self.head(d1))

def layers__count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def layers__build_unet(model_config, strict_parameter_check=True):
    """"""
    model = layers__UNetBaseline(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 3)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    expected = int(model_config.get("expected_parameter_count", layers__EXPECTED_PARAMETER_COUNT))
    actual = layers__count_parameters(model)
    if strict_parameter_check and actual != expected:
        raise layers__ParameterCountMismatch(
            "U-Net parameter count mismatch.\n"
            "  expected = %d\n"
            "  actual   = %d\n"
            "  delta    = %d\n"
            "The architecture must stay identical to the one validated in "
            "15_overfit_unet.py. Do not silently change it.\n"
            "Hint: the bias setting of the DoubleConv convolutions changes the "
            "count by exactly 2944." % (expected, actual, actual - expected)
        )
    return model

layers__MODULE = _PlainNamespace(**{
    'DoubleConv': layers__DoubleConv,
    'EXPECTED_PARAMETER_COUNT': layers__EXPECTED_PARAMETER_COUNT,
    'ParameterCountMismatch': layers__ParameterCountMismatch,
    'UNetBaseline': layers__UNetBaseline,
    '_group_norm': layers___group_norm,
    'build_unet': layers__build_unet,
    'checkpoint': layers__checkpoint,
    'count_parameters': layers__count_parameters,
    'nn': layers__nn,
    'torch': layers__torch,
})

uv_core__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'src/uv_uncertainty_unet.py')
"""Single-target RGB -> UV mean and per-pixel RGB residual sigma.

This is a heteroscedastic Gaussian U-Net, NOT a diffusion model and NOT
epistemic uncertainty.  Its mean branch preserves the independent U-Net.
The variance floor is a numerical regularizer in [0, 1] image units.
"""
import math as uv_core__math

import torch as uv_core__torch
from torch import nn as uv_core__nn

pass  

uv_core__VARIANCE_FLOOR = 1e-6
uv_core__EXPECTED_PARAMETER_COUNT = 7763140

class uv_core__UVUncertaintyUNet(layers__UNetBaseline):
    def __init__(self, base_channels=32):

        super().__init__(in_channels=3, out_channels=3,
                         base_channels=base_channels)
        self.log_variance_head = uv_core__nn.Conv2d(base_channels, 1, kernel_size=1)
        uv_core__nn.init.zeros_(self.log_variance_head.weight)
        uv_core__nn.init.zeros_(self.log_variance_head.bias)

    def forward(self, x):
        if self.training and self.activation_checkpointing:
            raise RuntimeError("Activation checkpointing is not in this frozen protocol")
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(uv_core__torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(uv_core__torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(uv_core__torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(uv_core__torch.cat([self.upconv1(d2), e1], dim=1))
        mean = self.output_activation(self.head(d1)).float()
        raw_s = self.log_variance_head(d1).float()
        
        log_variance = uv_core__torch.logaddexp(
            raw_s, raw_s.new_tensor(uv_core__math.log(uv_core__VARIANCE_FLOOR)))
        return {"mean": mean, "raw_log_variance": raw_s,
                "log_variance": log_variance,
                "variance": uv_core__torch.exp(log_variance),
                "sigma": uv_core__torch.exp(0.5 * log_variance)}

def uv_core__gaussian_nll_per_image(output, target):
    """先平均三个颜色通道的平方误差，再计算高斯负对数似然，最后对空间取平均。"""
    err2 = (output["mean"].float() - target.float()).square().mean(1, keepdim=True)
    logvar = output["log_variance"].float()
    return (0.5 * (err2 * uv_core__torch.exp(-logvar) + logvar)).flatten(1).mean(1)

def uv_core__validation_per_image(output, target, rgb=None):
    """逐图计算验证指标。单通道标准差在三个 UV 颜色通道间共享。"""
    error = output["mean"].float() - target.float()
    sigma = output["sigma"].float()
    z = error / sigma
    mean = lambda value: value.flatten(1).mean(1)
    return {
        "nll": uv_core__gaussian_nll_per_image(output, target),
        "l1": mean(error.abs()),
        "mse": mean(error.square()),
        "sigma_mean": mean(sigma),
        "standardized_squared_error": mean(z.square()),
        "standardized_residual_mean": mean(z),
        "coverage68": mean((z.abs() <= 1.0).float()),
        "coverage95": mean((z.abs() <= 1.959963984540054).float()),
        "coverage99": mean((z.abs() <= 2.5758293035489004).float()),
        "near_variance_floor_fraction": mean(
            (output["variance"] <= 2 * uv_core__VARIANCE_FLOOR).float()),
    }

uv_core__MODULE = _PlainNamespace(**{
    'EXPECTED_PARAMETER_COUNT': uv_core__EXPECTED_PARAMETER_COUNT,
    'UNetBaseline': layers__UNetBaseline,
    'UVUncertaintyUNet': uv_core__UVUncertaintyUNet,
    'VARIANCE_FLOOR': uv_core__VARIANCE_FLOOR,
    'gaussian_nll_per_image': uv_core__gaussian_nll_per_image,
    'math': uv_core__math,
    'nn': uv_core__nn,
    'torch': uv_core__torch,
    'validation_per_image': uv_core__validation_per_image,
})

training_common__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'tools/train_uv_uncertainty_fullimage.py')

"""Frozen-protocol, from-zero RGB -> UV mean/variance training; never TEST.

prepare: freeze train/val metadata, audited-DataLoader epoch index orders and code.
preflight: disposable synthetic 1024 three-GPU updates; never real data/weights.
train: three-rank DDP, global image batch16, no padding, resumable at batch edges.

Only this run's checkpoint_last.pt can be resumed. No initialization checkpoint,
teacher, EMA, extra L1 objective, TEST loader, or automatic post-training test.
"""

import argparse as training_common__argparse
import contextlib as training_common__contextlib
import csv as training_common__csv
import datetime as training_common__datetime
import hashlib as training_common__hashlib
import io as training_common__io
import json as training_common__json
import math as training_common__math
import os as training_common__os
from pathlib import Path as training_common__Path
import random as training_common__random
import sys as training_common__sys
import time as training_common__time
import traceback as training_common__traceback
import uuid as training_common__uuid

training_common__ROOT = training_common__Path(training_common__SOURCE_FILE).resolve().parents[1]
training_common__sys.path.insert(0, str(training_common__ROOT))
pass; pass; pass; pass  

training_common__SCHEMA = "uv_uncertainty_fullimage_training_v1"
training_common__COUNTS = {"train": 80740, "val": 11534}
training_common__VAL_KEYS = ("nll", "l1", "mse", "sigma_mean", "standardized_squared_error",
            "standardized_residual_mean", "coverage68", "coverage95", "coverage99",
            "near_variance_floor_fraction")
training_common__CODE_FILES = ("tools/train_uv_uncertainty_fullimage.py", "src/uv_uncertainty_unet.py",
              "src/matched_training_batching.py", "src/unet_baseline.py", "src/highres_patch_dataset.py")

def training_common__reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)

def training_common__read_json(path):
    return training_common__json.loads(training_common__Path(path).read_text(encoding="utf-8"), parse_constant=training_common__reject_constant)

def training_common__digest(path):
    h = training_common__hashlib.sha256()
    with training_common__Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def training_common__save_json(path, payload, immutable=False):
    path = training_common__Path(path)
    data = training_common__json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = path if immutable else path.with_name(path.name + "." + training_common__uuid.uuid4().hex + ".tmp")
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        training_common__os.fsync(handle.fileno())
    if not immutable:
        training_common__os.replace(destination, path)

@training_common__contextlib.contextmanager
def training_common__exclusive_lock(path):
    import fcntl
    with training_common__Path(path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This training/preflight task already has an active coordinator")
        yield

def training_common__validate_config(config):
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
    if config["expected_counts"] != training_common__COUNTS or config["model"].get("base_channels") != 32:
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

def training_common__select_training_rows(source_rows, expected_counts=None):
    """"""
    expected_counts = training_common__COUNTS if expected_counts is None else expected_counts
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

class training_common__IndexDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index

def training_common__baseline_loader_orders(size, seed, epochs, num_workers=1):
    """"""
    import torch
    if num_workers < 1:
        raise ValueError("Persistent worker order reproduction requires >=1 worker")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(training_common__IndexDataset(size), batch_size=16, shuffle=True,
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

def training_common__prepare(config_path):
    config = training_common__validate_config(training_common__read_json(config_path))
    out = training_common__Path(config["output_dir"])
    if out.exists():
        raise RuntimeError("prepare requires a new output directory; use train --resume auto for this run")
    source = training_common__Path(config["source_manifest"])
    if training_common__digest(source) != config["source_manifest_sha256"]:
        raise RuntimeError("Canonical source manifest hash mismatch")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        splits = training_common__select_training_rows(training_common__csv.DictReader(handle))
    code = {name: training_common__digest(training_common__ROOT / name) for name in training_common__CODE_FILES}
    out.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        training_common__save_json(out / (split + "_manifest.json"), {"split": split, "rows": rows}, immutable=True)
    (out / "orders").mkdir()
    hashes = {}
    import torch
    for epoch, order in training_common__baseline_loader_orders(training_common__COUNTS["train"], config["seed"], 20):
        path = out / "orders" / ("epoch_%03d.json" % epoch)
        training_common__save_json(path, {"epoch": epoch, "indices": order, "index_order_sha256": batches__order_sha256(order)}, immutable=True)
        hashes[str(epoch)] = training_common__digest(path)
        print(training_common__json.dumps({"stage": "prepare_index_order", "epoch": epoch, "samples": len(order)}), flush=True)
    protocol = {"schema": training_common__SCHEMA, "config": config, "config_sha256": training_common__digest(config_path),
                "code_sha256": code, "source_manifest_sha256": training_common__digest(source),
                "manifest_sha256": {s: training_common__digest(out / (s + "_manifest.json")) for s in training_common__COUNTS},
                "epoch_order_file_sha256": hashes, "order_generation_torch": torch.__version__,
                "order_rule": "actual_DataLoader_shuffle_generator_seed_plus_one_based_epoch_persistent_workers",
                "order_worker_count": 1, "original_baseline_order_worker_count": 8,
                "order_note": "Same source split order and DataLoader sampling algorithm; worker-count equivalence tested on CPU. Not bitwise old training replication.",
                "initialization": "from_zero_no_checkpoint_or_teacher", "ema_used": False,
                "objective": "Gaussian_NLL_FP32_only_no_L1_auxiliary", "selection": "full_val_mean_L1",
                "early_stopping": "Separate significant-improvement anchor; strict true-best L1 checkpoint saved independently",
                "canonical_size": 1024, "preprocessing": "same center-square RGB/UV; whole-image PIL Lanczos; uint8/255; no augmentation",
                "train_images": training_common__COUNTS["train"], "val_images": training_common__COUNTS["val"],
                "test_split_used": False, "test_images_accessed": 0, "random_crop_or_patch": False,
                "source_access_note": "Canonical CSV metadata filtered by split; TEST image paths are not retained or opened",
                "batching": "global16; rank6/5/5; micro2; loss scaling world_size*micro_n/global_n; short tail4=2/1/1",
                "validation": "FP32 batch1, unpadded rank-stride; no DDP forward collectives; per-image streamed resume",
                "validation_keys": list(training_common__VAL_KEYS), "amp": "FP16 convolution and FP32 Gaussian likelihood",
                "amp_overflow": "Synchronized gradient-only skip/backoff; 3 consecutive overflows block run; loss nonfinite is fatal",
                "numerical_difference_note": "DDP reduction and microbatching may change floating-point arithmetic relative to physical-batch16 baseline",
                "created_utc": training_common__time.strftime("%Y-%m-%dT%H:%M:%SZ", training_common__time.gmtime())}
    training_common__save_json(out / "protocol.json", protocol, immutable=True)
    training_common__save_json(out / "protocol_digest.json", {"sha256": training_common__digest(out / "protocol.json")}, immutable=True)
    print(training_common__json.dumps({"status": "prepared", "protocol_sha256": training_common__digest(out / "protocol.json"), "test_split_used": False}), flush=True)

def training_common__load_frozen(config_path):
    config = training_common__validate_config(training_common__read_json(config_path))
    out = training_common__Path(config["output_dir"])
    protocol = training_common__read_json(out / "protocol.json")
    phash = training_common__digest(out / "protocol.json")
    if training_common__read_json(out / "protocol_digest.json")["sha256"] != phash:
        raise RuntimeError("Protocol hash mismatch")
    if protocol["schema"] != training_common__SCHEMA or protocol["config_sha256"] != training_common__digest(config_path) or protocol["config"] != config:
        raise RuntimeError("Frozen configuration changed")
    for name, sha in protocol["code_sha256"].items():
        if training_common__digest(training_common__ROOT / name) != sha:
            raise RuntimeError("Frozen code changed: " + name)
    splits = {}
    for split, count in training_common__COUNTS.items():
        path = out / (split + "_manifest.json")
        if training_common__digest(path) != protocol["manifest_sha256"][split]:
            raise RuntimeError("Frozen split manifest changed")
        rows = training_common__read_json(path)["rows"]
        if len(rows) != count or len({r["sample_id"] for r in rows}) != count or any(r["split"] != split for r in rows):
            raise RuntimeError("Training-only split firewall/coverage failure")
        splits[split] = rows
    return config, out, protocol, phash, splits

def training_common__load_order(out, protocol, epoch):
    path = out / "orders" / ("epoch_%03d.json" % epoch)
    if training_common__digest(path) != protocol["epoch_order_file_sha256"][str(epoch)]:
        raise RuntimeError("Frozen epoch order changed")
    payload = training_common__read_json(path)
    order = payload["indices"]
    if (payload["epoch"] != epoch or len(order) != training_common__COUNTS["train"]
            or sorted(order) != list(range(training_common__COUNTS["train"])) or batches__order_sha256(order) != payload["index_order_sha256"]):
        raise RuntimeError("Epoch index coverage/hash invalid")
    return order

def training_common__update_selection(selection, value, epoch, min_delta=.0001, patience=5):
    """"""
    if not training_common__math.isfinite(value):
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

class training_common__EpochMicroBatchSampler:
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

class training_common__TrainingOnlyDataset:
    def __init__(self, rows, root, split):
        if split not in training_common__COUNTS or any(row.get("split") != split for row in rows):
            raise RuntimeError("TEST/foreign rows prohibited in training dataset")
        HighResolutionFullImageDataset = data__HighResolutionFullImageDataset
        self.rows = rows
        self.base = HighResolutionFullImageDataset(rows, "uv", root, split, canonical_size=1024, allow_test=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.rows[index].get("split") not in training_common__COUNTS:
            raise RuntimeError("Training-only split firewall failure")
        return self.base[index]

def training_common__seed_worker(worker_id):
    import torch
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    training_common__random.seed(seed)
    np.random.seed(seed)

def training_common__make_loader(dataset, batch_sampler, config, seed):
    import torch
    workers = config["dataloader"].get("num_workers", 4)
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_sampler=batch_sampler, num_workers=workers,
                  pin_memory=config["dataloader"].get("pin_memory", True),
                  generator=generator, worker_init_fn=training_common__seed_worker)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=1)
    return torch.utils.data.DataLoader(dataset, **kwargs)

def training_common__setup_distributed(config):
    import torch
    import torch.distributed as dist
    world = int(training_common__os.environ.get("WORLD_SIZE", "0"))
    rank = int(training_common__os.environ.get("RANK", "-1"))
    local_rank = int(training_common__os.environ.get("LOCAL_RANK", "-1"))
    if world != 3 or rank not in range(3) or local_rank not in range(3):
        raise RuntimeError("Must run via torchrun --nproc_per_node=3 on the authorized node")
    training_common__os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=training_common__datetime.timedelta(minutes=30))
    import numpy as np
    training_common__random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    return rank, torch.device("cuda", local_rank)

def training_common__capture_rng(device):
    import torch
    import numpy as np
    return {"python": training_common__random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device)}

def training_common__restore_rng(state, device):
    import torch
    import numpy as np
    training_common__random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)

def training_common__cpu_snapshot(value):
    """"""
    import copy
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: training_common__cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [training_common__cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(training_common__cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)

def training_common__state_equal(left, right):
    import torch
    import numpy as np
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(training_common__state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(training_common__state_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right

def training_common__atomic_torch_save(path, payload):
    import torch
    path = training_common__Path(path)
    temp = path.with_name(path.name + "." + training_common__uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        training_common__os.fsync(handle.fileno())
    training_common__os.replace(temp, path)

def training_common__synchronized_checkpoint(out, protocol_hash, core, optimizer, scheduler, scaler, state,
                            local_accumulator, order, rank, device, best=False):
    import torch.distributed as dist
    rank_payload = {"rank": rank, "rng": training_common__capture_rng(device), "epoch_accumulator": local_accumulator}
    rank_states = [None] * 3
    dist.all_gather_object(rank_states, rank_payload)
    if rank == 0:
        payload = {"schema": training_common__SCHEMA, "protocol_sha256": protocol_hash, "state": state,
                   "batching_resume": batches__resume_metadata(order, state["next_batch"]),
                   "model_state_dict": core.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "rank_states": rank_states, "ema_used": False, "test_split_used": False,
                   "training_complete": state["phase"] == "complete"}
        if best:
            training_common__atomic_torch_save(out / "checkpoint_best.pt", payload)
        
        training_common__atomic_torch_save(out / "checkpoint_last.pt", payload)
        training_common__save_json(out / "history.json", state["history"])
    dist.barrier()

def training_common__verify_checkpoint(payload, protocol_hash, order):
    if payload.get("schema") != training_common__SCHEMA or payload.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Checkpoint protocol identity mismatch; foreign warm starts prohibited")
    if payload.get("ema_used") is not False or payload.get("test_split_used") is not False:
        raise RuntimeError("Checkpoint violates training protocol")
    if len(payload.get("rank_states", [])) != 3 or sorted(r["rank"] for r in payload["rank_states"]) != [0, 1, 2]:
        raise RuntimeError("Checkpoint per-rank RNG/statistics missing")
    offset = batches__validate_resume_metadata(order, payload["batching_resume"])
    if offset != payload["state"]["next_batch"]:
        raise RuntimeError("Checkpoint batch offset mismatch")
    if payload["state"].get("blocked_reason"):
        raise RuntimeError("Checkpoint explicitly blocked after an unsafe failure; user review required")

def training_common__finite_all_ranks(local_finite, device):
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(bool(local_finite)), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())

def training_common__validate_output(output, target):
    """"""
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

def training_common__step_optimizer(model, optimizer, scaler, device, state):
    import torch
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    finite = training_common__finite_all_ranks(bool(torch.isfinite(norm).item()), device)
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
        if not training_common__math.isfinite(scale) or scale <= 0:
            raise FloatingPointError("AMP scale can no longer be safely reduced")
        scaler.update(new_scale=scale)
        scaler_state = scaler.state_dict()
        scaler_state["_growth_tracker"] = 0
        scaler.load_state_dict(scaler_state)
    optimizer.zero_grad(set_to_none=True)
    return finite, float(norm.item()) if finite else None

def training_common__reduced_accumulator(local, device):
    import torch
    import torch.distributed as dist
    value = torch.tensor([local["images"], local["nll_sum"], local["l1_sum"]], dtype=torch.float64, device=device)
    dist.all_reduce(value)
    count, nll, l1 = value.tolist()
    return {"images": int(count), "nll": nll / count if count else None, "l1": l1 / count if count else None}

def training_common__load_val_records(path, rows, indices, protocol_hash, state_id):
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if not line.endswith("\n"):
                    raise RuntimeError("Truncated validation record: preserve evidence, no silent truncation")
                record = training_common__json.loads(line, parse_constant=training_common__reject_constant)
                if position >= len(indices) or record["index"] != indices[position] or record["sample_id"] != rows[indices[position]]["sample_id"]:
                    raise RuntimeError("Validation identity/order mismatch")
                if record["protocol_sha256"] != protocol_hash or record["state_id"] != state_id or set(record["metrics"]) != set(training_common__VAL_KEYS):
                    raise RuntimeError("Validation model/protocol mismatch")
                if any(not training_common__math.isfinite(float(v)) for v in record["metrics"].values()):
                    raise FloatingPointError("Non-finite persisted validation metric")
                records.append(record)
    return records

def training_common__run_validation(core, dataset, rows, out, protocol_hash, state, config, rank, device):
    import torch
    import torch.distributed as dist
    validation_per_image = uv_core__validation_per_image
    indices = list(range(rank, len(rows), 3))
    state_id = "%s:e%d:a%d:s%d" % (protocol_hash, state["epoch"], state["global_attempts"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    done = training_common__load_val_records(path, rows, indices, protocol_hash, state_id)
    totals = {key: sum(r["metrics"][key] for r in done) for key in training_common__VAL_KEYS}
    loader = training_common__make_loader(dataset, [[i] for i in indices[len(done):]], config,
                         config["seed"] + 500000 + state["epoch"] * 3 + rank)
    core.eval()
    start = training_common__time.monotonic()
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
                output = core(rgb)  
                training_common__validate_output(output, target)
                vectors = validation_per_image(output, target)
                if set(vectors) != set(training_common__VAL_KEYS):
                    raise RuntimeError("Validation metric keys differ from the frozen complete set")
                metrics = {}
                for key in training_common__VAL_KEYS:
                    value = vectors[key]
                    if tuple(value.shape) != (1,) or not bool(torch.isfinite(value).all()):
                        raise FloatingPointError("Non-finite/invalid validation metric: " + key)
                    metrics[key] = float(value.item())
                    totals[key] += metrics[key]
                record = {"index": index, "sample_id": rows[index]["sample_id"], "metrics": metrics,
                          "protocol_sha256": protocol_hash, "state_id": state_id, "test_split_used": False}
                handle.write(training_common__json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                handle.flush()
                if (local_position + 1) % 25 == 0:
                    training_common__os.fsync(handle.fileno())
                    training_common__save_json(out / ("validation_progress_rank%d.json" % rank), {
                        "status": "validating", "epoch": state["epoch"], "max_epochs": 20,
                        "completed_images": local_position + 1, "expected_images": len(indices),
                        "seconds_current_run": training_common__time.monotonic() - start, "test_split_used": False})
                del output, vectors, rgb, target, batch
                current_val_index = indices[local_position + 1] if local_position + 1 < len(indices) else None
            training_common__os.fsync(handle.fileno())
    except Exception:
        training_common__save_json(directory / ("failure_rank%d_%d.json" % (rank, training_common__time.time_ns())), {
            "status": "failed", "rank": rank, "index": current_val_index,
            "sample_id": rows[current_val_index]["sample_id"] if current_val_index is not None else None,
            "traceback": training_common__traceback.format_exc(), "protocol_sha256": protocol_hash,
            "state_id": state_id, "test_split_used": False}, immutable=True)
        raise
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    records = training_common__load_val_records(path, rows, indices, protocol_hash, state_id)
    if len(records) != len(indices):
        raise RuntimeError("Validation incomplete after write/readback")
    values = torch.tensor([len(records)] + [totals[k] for k in training_common__VAL_KEYS], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    all_values = values.tolist()
    if int(all_values[0]) != training_common__COUNTS["val"]:
        raise RuntimeError("Full validation must contain exactly 11534 images")
    result = {key: all_values[i + 1] / all_values[0] for i, key in enumerate(training_common__VAL_KEYS)}
    result["images"] = int(all_values[0])
    training_common__save_json(out / ("validation_progress_rank%d.json" % rank), {
        "status": "complete", "epoch": state["epoch"], "completed_images": len(records),
        "expected_images": len(indices), "record_sha256": training_common__digest(path), "test_split_used": False})
    return result

def training_common__new_state():
    return {"epoch": 1, "phase": "train", "next_batch": 0, "global_step": 0, "global_attempts": 0,
            "amp_overflow_skips": 0, "consecutive_amp_skips": 0, "history": [],
            "selection": {"best_val_l1": None, "best_epoch": None, "early_stopping_anchor": None,
                          "early_stopping_counter": 0}, "blocked_reason": None}

def training_common__new_accumulator():
    return {"images": 0, "nll_sum": 0.0, "l1_sum": 0.0}

def training_common__advance_completed_epoch(state):
    """"""
    if state["phase"] == "epoch_complete":
        if state["epoch"] >= 20:
            raise RuntimeError("An epoch20 checkpoint must be marked complete")
        state.update(epoch=state["epoch"] + 1, phase="next_epoch", next_batch=0)
    return state

def training_common__final_summary(out, state, protocol_hash):
    return {"status": "complete", "training_complete": True, "epochs_completed": len(state["history"]),
            "max_epochs": 20, "stop_reason": state["stop_reason"], "selection": state["selection"],
            "global_step": state["global_step"], "global_attempts": state["global_attempts"],
            "amp_overflow_skips": state["amp_overflow_skips"], "protocol_sha256": protocol_hash,
            "checkpoint_best": str(out / "checkpoint_best.pt"), "checkpoint_best_sha256": training_common__digest(out / "checkpoint_best.pt"),
            "checkpoint_last": str(out / "checkpoint_last.pt"), "checkpoint_last_sha256": training_common__digest(out / "checkpoint_last.pt"),
            "test_split_used": False, "test_images_accessed": 0, "ema_used": False}

def training_common__log_progress(out, event, rank):
    if rank == 0:
        event = dict(event, updated_utc=training_common__time.strftime("%Y-%m-%dT%H:%M:%SZ", training_common__time.gmtime()), test_split_used=False)
        training_common__save_json(out / "training_progress.json", event)
        line = training_common__json.dumps(event, allow_nan=False, sort_keys=True)
        print(line, flush=True)
        with (out / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

def training_common__train(config_path, resume):
    config, out, protocol, phash, splits = training_common__load_frozen(config_path)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    UVUncertaintyUNet = uv_core__UVUncertaintyUNet; gaussian_nll_per_image = uv_core__gaussian_nll_per_image
    rank, device = training_common__setup_distributed(config)
    lock = training_common__exclusive_lock(out / "training_coordinator.lock") if rank == 0 else training_common__contextlib.nullcontext()
    state, accumulator, current_ids = training_common__new_state(), training_common__new_accumulator(), []
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
                order = training_common__load_order(out, protocol, state["epoch"])
                training_common__verify_checkpoint(payload, phash, order)
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
                    summary = training_common__final_summary(out, state, phash)
                    training_common__save_json(out / "history.json", state["history"])
                    training_common__save_json(out / "training_summary.json", summary)
                    training_common__log_progress(out, dict(summary, status="already_complete"), rank)
                return
            training_common__advance_completed_epoch(state)
            train_dataset = training_common__TrainingOnlyDataset(splits["train"], config["dataset_root"], "train")
            val_dataset = training_common__TrainingOnlyDataset(splits["val"], config["dataset_root"], "val")
            sampler = training_common__EpochMicroBatchSampler(rank)
            train_loader = training_common__make_loader(train_dataset, sampler, config, config["seed"] + rank * 1000)
            if saved_rng is not None:
                training_common__restore_rng(saved_rng, device)
            training_common__save_json(out / ("environment_rank%d_%d.json" % (rank, training_common__time.time_ns())), {
                "python": training_common__sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device), "rank": rank,
                "parameters": sum(p.numel() for p in core.parameters()), "world_size": 3,
                "tf32": False, "training_amp": "fp16", "nll_precision": "fp32", "pid": training_common__os.getpid(),
                "initialization": "this_run_checkpoint_resume" if saved_rng is not None else "from_zero",
                "protocol_sha256": phash}, immutable=True)
            session_start, session_attempts = training_common__time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]
                order = training_common__load_order(out, protocol, epoch)
                if state["phase"] == "next_epoch":
                    state.update(phase="train", next_batch=0)
                    accumulator = training_common__new_accumulator()
                if state["global_attempts"] == 0 and not last.exists():
                    training_common__synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                if state["phase"] == "train":
                    plans = tuple(batches__iter_epoch_batches(order, start_batch=state["next_batch"]))
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
                            synchronization = training_common__contextlib.nullcontext() if micro.synchronize else model.no_sync()
                            with synchronization:
                                failure = None
                                try:
                                    with torch.cuda.amp.autocast(dtype=torch.float16):
                                        output = model(rgb)
                                    with torch.cuda.amp.autocast(enabled=False):
                                        training_common__validate_output(output, target)
                                        nll = gaussian_nll_per_image(output, target)
                                        l1 = (output["mean"].float() - target).abs().flatten(1).mean(1)
                                    valid = bool(torch.isfinite(nll).all() and torch.isfinite(l1).all())
                                except Exception:
                                    failure, valid = training_common__traceback.format_exc(), False
                                if not training_common__finite_all_ranks(valid, device):
                                    raise FloatingPointError("Nonfinite/invalid training forward or likelihood; " + (failure or "another rank failed"))
                                scaler.scale(nll.mean() * micro.loss_scale).backward()
                            accumulator["images"] += len(micro.indices)
                            accumulator["nll_sum"] += float(nll.double().sum().item())
                            accumulator["l1_sum"] += float(l1.double().sum().item())
                            del batch, rgb, target, output, nll, l1
                        successful, gradient_norm = training_common__step_optimizer(model, optimizer, scaler, device, state)
                        state["next_batch"] = plan.batch_index + 1
                        if state["consecutive_amp_skips"] >= 3:
                            state["blocked_reason"] = "three_consecutive_amp_gradient_overflows"
                            training_common__synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                            raise FloatingPointError(state["blocked_reason"])
                        if state["global_attempts"] % 1000 == 0:
                            training_common__synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                        if state["global_attempts"] % 20 == 0 or state["next_batch"] == 1 or not successful:
                            totals = training_common__reduced_accumulator(accumulator, device)
                            elapsed = training_common__time.monotonic() - session_start
                            per_attempt = elapsed / max(1, state["global_attempts"] - session_attempts)
                            training_common__log_progress(out, {"status": "training" if successful else "AMP_OVERFLOW_SKIP", "epoch": epoch,
                                "max_epochs": 20, "epoch_batch": state["next_batch"], "epoch_batches": 5047,
                                "train_images": totals["images"], "train_nll": totals["nll"], "train_l1": totals["l1"],
                                "global_step": state["global_step"], "global_attempts": state["global_attempts"],
                                "amp_overflow_skips": state["amp_overflow_skips"], "gradient_norm": gradient_norm,
                                "loss_scale": scaler.get_scale(), "learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch_train_eta_seconds": per_attempt * (5047 - state["next_batch"]),
                                "eta_note": "observed-session estimate; validation and storage overhead can change it"}, rank)
                    totals = training_common__reduced_accumulator(accumulator, device)
                    if totals["images"] != training_common__COUNTS["train"] or state["next_batch"] != 5047:
                        raise RuntimeError("Epoch did not consume exactly 80740 train images")
                    state["phase"] = "val"
                    training_common__synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device)
                totals = training_common__reduced_accumulator(accumulator, device)
                training_common__log_progress(out, {"status": "validating", "epoch": epoch, "max_epochs": 20,
                                  "train_images": totals["images"], "train_nll": totals["nll"], "train_l1": totals["l1"],
                                  "global_step": state["global_step"], "amp_overflow_skips": state["amp_overflow_skips"]}, rank)
                validation = training_common__run_validation(core, val_dataset, splits["val"], out, phash, state, config, rank, device)
                selection, is_best, early_stop = training_common__update_selection(state["selection"], validation["l1"], epoch)
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
                    
                    state["phase"] = "epoch_complete"
                training_common__synchronized_checkpoint(out, phash, core, optimizer, scheduler, scaler, state, accumulator, order, rank, device, best=is_best)
                training_common__log_progress(out, dict(history_row, status="epoch_complete", max_epochs=20,
                                       best_epoch=selection["best_epoch"], best_val_l1=selection["best_val_l1"],
                                       early_stopping_counter=selection["early_stopping_counter"]), rank)
                if state["phase"] == "epoch_complete":
                    training_common__advance_completed_epoch(state)
            if rank == 0:
                summary = training_common__final_summary(out, state, phash)
                training_common__save_json(out / "training_summary.json", summary)
                training_common__log_progress(out, summary, rank)
    except Exception:
        failure = {"status": "failed", "rank": rank, "state": state, "current_train_sample_ids": current_ids,
                   "traceback": training_common__traceback.format_exc(), "test_split_used": False}
        training_common__save_json(out / ("failure_rank%d_%d.json" % (rank, training_common__time.time_ns())), failure, immutable=True)
        raise
    finally:
        if train_loader is not None and getattr(train_loader, "_iterator", None) is not None:
            train_loader._iterator._shutdown_workers()
        if dist.is_initialized():
            dist.destroy_process_group()

def training_common__preflight(config_path, output_path=None):
    """启动正式运行前检查数据、模型和执行配置。"""
    config = training_common__validate_config(training_common__read_json(config_path))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    UVUncertaintyUNet = uv_core__UVUncertaintyUNet; gaussian_nll_per_image = uv_core__gaussian_nll_per_image; validation_per_image = uv_core__validation_per_image
    rank, device = training_common__setup_distributed(config)
    directory = training_common__Path(output_path) if output_path else training_common__Path(config["output_dir"] + "_preflight")
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
        state = training_common__new_state()
        plan = next(batches__iter_epoch_batches(range(16)))
        timings, losses = [], []
        torch.cuda.reset_peak_memory_stats(device)
        def synthetic_update():
            start = training_common__time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for micro in plan.rank_batches[rank].microbatches:
                rgb = torch.rand(len(micro.indices), 3, 1024, 1024, device=device)
                target = torch.rand_like(rgb)
                ctx = training_common__contextlib.nullcontext() if micro.synchronize else model.no_sync()
                with ctx:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(rgb)
                    with torch.cuda.amp.autocast(enabled=False):
                        training_common__validate_output(output, target)
                        nll = gaussian_nll_per_image(output, target)
                    if not training_common__finite_all_ranks(bool(torch.isfinite(nll).all()), device):
                        raise FloatingPointError("Nonfinite synthetic likelihood")
                    scaler.scale(nll.mean() * micro.loss_scale).backward()
                losses.append(float(nll.mean().detach().item()))
                del rgb, target, output, nll
            success, _ = training_common__step_optimizer(model, optimizer, scaler, device, state)
            if not success:
                raise FloatingPointError("Synthetic preflight gradient overflow; formal training not approved by this result")
            torch.cuda.synchronize()
            return training_common__time.monotonic() - start

        for step in range(3):
            timings.append(synthetic_update())
        peak_train = torch.cuda.max_memory_allocated(device)
        
        buffer = training_common__io.BytesIO()
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": training_common__capture_rng(device), "state": state}, buffer)
        synthetic_update()
        expected_weights = training_common__cpu_snapshot(core.state_dict())
        expected_optimizer = training_common__cpu_snapshot(optimizer.state_dict())
        expected_scaler = training_common__cpu_snapshot(scaler.state_dict())
        expected_rng = training_common__cpu_snapshot(training_common__capture_rng(device))
        expected_state = dict(state)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
        core.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        training_common__restore_rng(restored["rng"], device)
        state.clear()
        state.update(restored["state"])
        del restored, buffer
        synthetic_update()
        checks = {"weights": training_common__state_equal(expected_weights, core.state_dict()),
                  "optimizer": training_common__state_equal(expected_optimizer, optimizer.state_dict()),
                  "scaler": training_common__state_equal(expected_scaler, scaler.state_dict()),
                  "rng": training_common__state_equal(expected_rng, training_common__capture_rng(device)),
                  "counters": training_common__state_equal(expected_state, state)}
        if not training_common__finite_all_ranks(all(checks.values()), device):
            raise RuntimeError("Synthetic checkpoint/resume equality failed: " + training_common__json.dumps(checks))
        del expected_weights, expected_optimizer, expected_scaler, expected_rng
        
        core.eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            rgb = torch.rand(1, 3, 1024, 1024, device=device)
            target = torch.rand_like(rgb)
            started = training_common__time.monotonic()
            output = core(rgb)
            training_common__validate_output(output, target)
            vectors = validation_per_image(output, target)
            if set(vectors) != set(training_common__VAL_KEYS):
                raise RuntimeError("Preflight metric keys differ from the complete frozen set")
            if any(not bool(torch.isfinite(vectors[key]).all()) for key in training_common__VAL_KEYS):
                raise FloatingPointError("Nonfinite synthetic FP32 validation")
            torch.cuda.synchronize()
            inference_seconds = training_common__time.monotonic() - started
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
                  "config_sha256": training_common__digest(config_path), "code_sha256": {p: training_common__digest(training_common__ROOT / p) for p in training_common__CODE_FILES}}
        training_common__save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        print(training_common__json.dumps(result, allow_nan=False), flush=True)
        dist.barrier()
    except Exception:
        if directory.exists():
            training_common__save_json(directory / ("failure_rank%d_%d.json" % (rank, training_common__time.time_ns())), {
                "status": "failed", "traceback": training_common__traceback.format_exc(), "rank": rank,
                "test_split_used": False, "dataset_accessed": False}, immutable=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def training_common__main():
    parser = training_common__argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "preflight", "train"))
    parser.add_argument("--config", type=training_common__Path, required=True)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--preflight-output", type=training_common__Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        training_common__prepare(args.config)
    elif args.mode == "preflight":
        training_common__preflight(args.config, args.preflight_output)
    else:
        training_common__train(args.config, args.resume)

training_common__MODULE = _PlainNamespace(**{
    'CODE_FILES': training_common__CODE_FILES,
    'COUNTS': training_common__COUNTS,
    'EpochMicroBatchSampler': training_common__EpochMicroBatchSampler,
    'IndexDataset': training_common__IndexDataset,
    'Path': training_common__Path,
    'ROOT': training_common__ROOT,
    'SCHEMA': training_common__SCHEMA,
    'TrainingOnlyDataset': training_common__TrainingOnlyDataset,
    'VAL_KEYS': training_common__VAL_KEYS,
    'advance_completed_epoch': training_common__advance_completed_epoch,
    'argparse': training_common__argparse,
    'atomic_torch_save': training_common__atomic_torch_save,
    'baseline_loader_orders': training_common__baseline_loader_orders,
    'capture_rng': training_common__capture_rng,
    'contextlib': training_common__contextlib,
    'cpu_snapshot': training_common__cpu_snapshot,
    'csv': training_common__csv,
    'datetime': training_common__datetime,
    'digest': training_common__digest,
    'exclusive_lock': training_common__exclusive_lock,
    'final_summary': training_common__final_summary,
    'finite_all_ranks': training_common__finite_all_ranks,
    'hashlib': training_common__hashlib,
    'io': training_common__io,
    'iter_epoch_batches': batches__iter_epoch_batches,
    'json': training_common__json,
    'load_frozen': training_common__load_frozen,
    'load_order': training_common__load_order,
    'load_val_records': training_common__load_val_records,
    'log_progress': training_common__log_progress,
    'main': training_common__main,
    'make_loader': training_common__make_loader,
    'math': training_common__math,
    'new_accumulator': training_common__new_accumulator,
    'new_state': training_common__new_state,
    'order_sha256': batches__order_sha256,
    'os': training_common__os,
    'preflight': training_common__preflight,
    'prepare': training_common__prepare,
    'random': training_common__random,
    'read_json': training_common__read_json,
    'reduced_accumulator': training_common__reduced_accumulator,
    'reject_constant': training_common__reject_constant,
    'restore_rng': training_common__restore_rng,
    'resume_metadata': batches__resume_metadata,
    'run_validation': training_common__run_validation,
    'save_json': training_common__save_json,
    'seed_worker': training_common__seed_worker,
    'select_training_rows': training_common__select_training_rows,
    'setup_distributed': training_common__setup_distributed,
    'state_equal': training_common__state_equal,
    'step_optimizer': training_common__step_optimizer,
    'synchronized_checkpoint': training_common__synchronized_checkpoint,
    'sys': training_common__sys,
    'time': training_common__time,
    'traceback': training_common__traceback,
    'train': training_common__train,
    'update_selection': training_common__update_selection,
    'uuid': training_common__uuid,
    'validate_config': training_common__validate_config,
    'validate_output': training_common__validate_output,
    'validate_resume_metadata': batches__validate_resume_metadata,
    'verify_checkpoint': training_common__verify_checkpoint,
})

train__SOURCE_FILE = str(_PLAIN_WORKSPACE / 'tools/train_uv_ldm_clean_multigpu_20261016.py')

"""True, from-scratch two-stage UV LDM training. Train/val firewall; no TEST access.

Two, four or six replicas use exact global 16 and clean tail 2,
physical batch one. BF16 forward, FP32 losses/coefficients/optimizer, no scaler.
Frozen orders and complete validation are inherited, not regenerated.
"""
import argparse as train__argparse
import contextlib as train__contextlib
import datetime as train__datetime
import hashlib as train__hashlib
import io as train__io
import json as train__json
import math as train__math
import os as train__os
from pathlib import Path as train__Path
import random as train__random
import shutil as train__shutil
import subprocess as train__subprocess
import signal as train__signal
import sys as train__sys
import time as train__time
import traceback as train__traceback

train__ROOT = train__Path(train__SOURCE_FILE).resolve().parents[1]
train__sys.path.insert(0, str(train__ROOT))
pass  
pass  

train__SCHEMA = "uv_true_latent_diffusion_clean_multigpu_train_20261016_v1"
train__COUNTS = {"train": 76642, "val": 10948}
train__WORLD_SIZE = 2  
train__ALLOWED_WORLD_SIZES = (2, 4, 6)
train__STAGE_ASSIGNMENT_SHA256 = None
train__STAGE_ASSIGNMENT = None
train__EPOCH_BATCHES = 4791
train__SOURCE_HASHES = {
    "tools/train_uv_ldm_exp4_v1.py": "1542d51233d84edc0e9c24296452b54c6d628f126c2b570541a9ccc4f82f4aa6",
    "configs/uv_ldm_exp4_20260927_v1.json": "8f9b5b4e0ec8e8583f7fb7f0ae47a3ff062d59e43ff8f61ab3301af3e05bcc2d",
    "src/uv_latent_diffusion_v1.py": "7886f0612812f70b02eb53feabe70d85e50059cc0971dd3942b64fc1c52ffe4c",
    "tools/train_uv_uncertainty_fullimage.py": "bcbb8cd9b1756061992e0e9c98180b3e1467b62056babadb2c284da7d8963691",
    "src/highres_patch_dataset.py": "7dbe88f46b4599926ce57ef384f2aeaf0acf55b2e473895e9b73b9cbdd085af7",
    "src/matched_training_batching.py": "02c664b2375a84bead4c9f5b95a26021c2adb19bcb7042037e5f0a86695f2ac4",
}
train__CODE_FILES = ("tools/train_uv_ldm_clean_multigpu_20261016.py", "src/uv_latent_diffusion_v1.py",
              "tools/train_uv_uncertainty_fullimage.py", "src/highres_patch_dataset.py",
              "src/matched_training_batching.py")
train__read_json, train__digest, train__save_json = training_common__read_json, training_common__digest, training_common__save_json

def train__runtime():
    import torch
    import numpy
    import PIL
    return dict(python=train__sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=numpy.__version__, pillow=PIL.__version__)

def train__validate_config(c):
    t = c["training"]
    expected = dict(epochs=20, global_batch_size=16, microbatch_size=1,
                    lr=1e-4, weight_decay=1e-4, grad_clip=1., precision="BF16",
                    checkpoint_every_attempts=1000, checkpoint_every_seconds=600)
    if t != expected:
        raise ValueError("Training budget/precision differs from frozen LDM protocol")
    if c.get("expected_counts") != train__COUNTS or c.get("seed") != 20260817 or c.get("test_access_allowed") is not False:
        raise ValueError("Train/val counts, seed or TEST firewall differs")
    if c["scheduler"] != dict(factor=.5, patience=2, min_lr=1e-6):
        raise ValueError("Scheduler differs")
    if c["early_stopping"] != dict(patience=5, min_delta=.0001):
        raise ValueError("Early stopping differs")
    if c["model"] != dict(ae_base_channels=32, latent_channels=4, diffusion_base_channels=64, time_dimension=256):
        raise ValueError("Model configuration differs")
    if c["ae"] != dict(kl_weight=1e-6, spatial_downsampling=4,
                       training_posterior="reparameterized_sample",
                       validation_posterior="mean", decoded_output="sigmoid", input_to_decoder_skip=False):
        raise ValueError("AE objective/architecture differs")
    if c.get("dataloader") != dict(num_workers=4, prefetch_factor=1, pin_memory=True):
        raise ValueError("Dataloader configuration differs")
    if c.get("actual_model_parameters") != dict(ae=1754315, denoiser=21051204):
        raise ValueError("Model parameter contract differs")
    d = c["diffusion"]
    if d != dict(timesteps=1000, beta_start=1e-4, beta_end=.02,
                 prediction_type="v_prediction", ddim_steps=20, eta=0):
        raise ValueError("Diffusion training/sampling differs")
    for k in ("pretrained", "warmstart", "resume_checkpoint", "checkpoint", "ema"):
        if c.get(k): raise ValueError("Foreign weights prohibited")
    if train__Path(c["output_dir"]).resolve() == train__Path(c["reference_uncertainty_dir"]).resolve():
        raise ValueError("Output must be independent")
    if c.get("schema") != "uv_ldm_clean_multigpu_training_protocol_20261016_v1":
        raise ValueError("Wrong clean training configuration")
    if c.get("execution") != dict(allowed_world_sizes=[2, 4, 6],
                                 topology_policy="freeze_assignment_per_stage_no_within_stage_world_migration",
                                 assignment_required=True):
        raise ValueError("Distributed execution contract differs")
    if (c.get("source_manifest_sha256") != "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
            or c.get("reference_protocol_sha256") != "621e35db9d4049e2dc9dbb2323f86086f53f1705f53f90592b2e6576a7b3f911"):
        raise ValueError("Wrong frozen clean dataset/reference protocol")
    return c

def train__original_sources():
    observed = {name: train__digest(train__ROOT / name) for name in train__SOURCE_HASHES}
    if observed != train__SOURCE_HASHES:
        raise RuntimeError("Original frozen scientific source changed")
    return observed

def train__load_order(out, protocol, epoch):
    if type(epoch) is not int or not 1 <= epoch <= 20:
        raise ValueError("Invalid frozen epoch")
    path = train__Path(out) / "orders" / ("epoch_%03d.json" % epoch)
    if train__digest(path) != protocol["epoch_order_file_sha256"][str(epoch)]:
        raise RuntimeError("Frozen epoch order changed")
    payload = train__read_json(path)
    order = payload["indices"]
    if (payload["epoch"] != epoch or len(order) != train__COUNTS["train"]
            or any(type(i) is not int for i in order)
            or sorted(order) != list(range(train__COUNTS["train"]))
            or batches__order_sha256(order) != payload["index_order_sha256"]):
        raise RuntimeError("Epoch index coverage/hash invalid")
    return order

def train__reference_assets(c):
    train__original_sources()
    source = train__Path(c["reference_uncertainty_dir"])
    if train__digest(source / "protocol.json") != c["reference_protocol_sha256"]:
        raise RuntimeError("Reference protocol changed")
    p = train__read_json(source / "protocol.json")
    if p["source_manifest_sha256"] != c["source_manifest_sha256"] or train__digest(c["source_manifest"]) != c["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    files, split_subjects, split_samples = {}, {}, {}
    for split, n in train__COUNTS.items():
        name = split + "_manifest.json"
        if train__digest(source / name) != p["manifest_sha256"][split]: raise RuntimeError("Reference manifest changed")
        rows = train__read_json(source / name)["rows"]
        if len(rows) != n or len({r["sample_id"] for r in rows}) != n or any(r["split"] != split for r in rows):
            raise RuntimeError("Reference coverage/firewall invalid")
        split_subjects[split] = {r["subject_id"] for r in rows}
        split_samples[split] = {r["sample_id"] for r in rows}
        dataset_root = train__Path(c["dataset_root"]).resolve()
        for row in rows:
            for modality in ("rgb", "uv"):
                raw = (dataset_root / row[modality + "_path"]).resolve()
                if dataset_root not in raw.parents or not raw.is_file() or raw.stat().st_size == 0:
                    raise RuntimeError("Missing/unsafe train-val input: " + row["sample_id"] + ":" + modality)
        files[name] = train__digest(source / name)
    if (split_subjects["train"] & split_subjects["val"]
            or split_samples["train"] & split_samples["val"]):
        raise RuntimeError("Reference train/val subject/sample overlap")
    for epoch in range(1, 21):
        name = "orders/epoch_%03d.json" % epoch
        train__load_order(source, p, epoch)
        files[name] = train__digest(source / name)
    return source, p, files

def train__prepare(config_path):
    c = train__validate_config(train__read_json(config_path)); out = train__Path(c["output_dir"])
    source, original, files = train__reference_assets(c)
    if out.exists():
        train__load_frozen(config_path)
        print(train__json.dumps(dict(status="already_prepared", protocol_sha256=train__digest(out / "protocol.json"))))
        return
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("orders", "ae", "diffusion", "latent_stats"): (out / folder).mkdir()
    for name, sha in files.items():
        with (source / name).open("rb") as src, (out / name).open("xb") as dst:
            train__shutil.copyfileobj(src, dst); dst.flush(); train__os.fsync(dst.fileno())
        if train__digest(out / name) != sha: raise RuntimeError("Copied reference differs")
    p = dict(schema=train__SCHEMA, config=c, config_sha256=train__digest(config_path), software=train__runtime(),
             code_sha256={n: train__digest(train__ROOT / n) for n in train__CODE_FILES}, manifest_sha256=original["manifest_sha256"],
             epoch_order_file_sha256=original["epoch_order_file_sha256"],
             reference_protocol_sha256=c["reference_protocol_sha256"],
             source_manifest_sha256=c["source_manifest_sha256"],
             original_scientific_sources_sha256=train__original_sources(),
             expected_counts=train__COUNTS, allowed_world_sizes=list(train__ALLOWED_WORLD_SIZES),
             topology_disclosure="each_stage_assignment_frozen_at_first_start_2_or4_or6_ranks_global16_unchanged;different_worlds_not_bitwise_identical_stochastic_trajectory",
             stochastic_streams="initialization_seed20260817_broadcast;training_rank_seed20260817_plus_rank_times1000;rank_rng_checkpointed",
             validation_loader_seed="20260817_plus_rank_plus_epoch_times_stage_world_plus500000",
             preprocessing="center_square_PIL_Lanczos_1024_uint8_div255_no_augmentation",
             initialization="both_AE_and_latent_denoiser_from_scratch_no_EMA",
             precision="BF16_training_forward_FP32_losses_coefficients_and_parameters_no_GradScaler_TF32_false;FP32_validation",
             batching="manual_stage_world_rank_gradient_mean_global16_tail2_physical1_no_padding_idle_tail_ranks_zero_gradient_no_forward",
             ae_objective="image_equal_L1_plus_1e-6_KL_sampled_posterior_train_mean_posterior_val",
             diffusion_objective="unweighted_v_MSE_uniform_t_frozen_BEST_AE_posterior_mean_latents",
             selection="strict_lowest_complete_val_UV_L1;diffusion_single_DDIM20_seed20260817_generation_index0",
             latent_normalization="all_train_BEST_AE_posterior_mean_per_channel_population_mean_std_float64_streaming",
             schedule_implementation="linear_beta_float64_cumprod_then_FP32_buffers_not_historical_pixel_schedule",
             parameter_counts=dict(ae=1754315, diffusion=21051204),
             test_split_used=False, test_images_accessed=0, created_utc=train__time.strftime("%Y-%m-%dT%H:%M:%SZ", train__time.gmtime()))
    train__save_json(out / "protocol.json", p, immutable=True)
    train__save_json(out / "protocol_digest.json", {"sha256": train__digest(out / "protocol.json")}, immutable=True)
    print(train__json.dumps(dict(status="prepared", protocol_sha256=train__digest(out / "protocol.json"))), flush=True)

def train__load_frozen(config_path):
    c = train__validate_config(train__read_json(config_path)); out = train__Path(c["output_dir"])
    p = train__read_json(out / "protocol.json"); ph = train__digest(out / "protocol.json")
    if p["schema"] != train__SCHEMA or train__read_json(out / "protocol_digest.json")["sha256"] != ph:
        raise RuntimeError("Frozen protocol identity changed")
    if p["config"] != c or p["config_sha256"] != train__digest(config_path): raise RuntimeError("Config changed")
    if p["software"] != train__runtime(): raise RuntimeError("Runtime differs; no silent migration")
    if p["code_sha256"] != {n: train__digest(train__ROOT / n) for n in train__CODE_FILES}: raise RuntimeError("Code changed")
    if p.get("original_scientific_sources_sha256") != train__original_sources():
        raise RuntimeError("Scientific source provenance differs")
    if (train__digest(c["source_manifest"]) != c["source_manifest_sha256"]
            or p.get("source_manifest_sha256") != c["source_manifest_sha256"]
            or train__digest(train__Path(c["reference_uncertainty_dir"]) / "protocol.json") != c["reference_protocol_sha256"]):
        raise RuntimeError("Frozen clean data/reference changed")
    rows = {}
    for split, n in train__COUNTS.items():
        path = out / (split + "_manifest.json")
        if train__digest(path) != p["manifest_sha256"][split]: raise RuntimeError("Manifest changed")
        rows[split] = train__read_json(path)["rows"]
        if len(rows[split]) != n or len({r["sample_id"] for r in rows[split]}) != n or any(r["split"] != split for r in rows[split]):
            raise RuntimeError("Coverage/firewall invalid")
    if {r["sample_id"] for r in rows["train"]} & {r["sample_id"] for r in rows["val"]}:
        raise RuntimeError("Train/val overlap")
    if {r["subject_id"] for r in rows["train"]} & {r["subject_id"] for r in rows["val"]}:
        raise RuntimeError("Train/val subject overlap")
    for epoch in range(1, 21):
        train__load_order(out, p, epoch)
    return c, out, p, ph, rows

def train__local_indices(indices, rank):
    indices = tuple(indices)
    if (len(indices) not in (2, 16) or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 for i in indices) or rank not in range(train__WORLD_SIZE)):
        raise ValueError("Only unpadded global16/tail2 on assigned two/four/six ranks")
    return indices[rank::train__WORLD_SIZE]

def train__epoch_batches(order, start=0):
    if (len(order) != train__COUNTS["train"] or any(type(i) is not int for i in order)
            or sorted(order) != list(range(train__COUNTS["train"]))
            or type(start) is not int or not 0 <= start <= train__EPOCH_BATCHES):
        raise ValueError("Invalid epoch/resume offset")
    return [(i, tuple(order[i * 16:(i + 1) * 16])) for i in range(start, train__EPOCH_BATCHES)]

def train__tensor_fingerprint(value):
    import torch
    h = train__hashlib.sha256()
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

def train__verified_assignment(path, ph, stage, world, visible_devices, observed_uuids):
    if path is None:
        raise RuntimeError("Explicit frozen stage assignment required")
    assignment = train__read_json(train__Path(path))
    if (assignment.get("schema") != "uv_clean_ldm_stage_assignment_v1"
            or assignment.get("protocol_sha256") != ph or assignment.get("stage") != stage
            or assignment.get("world_size") != world or world not in train__ALLOWED_WORLD_SIZES
            or assignment.get("visible_devices") != visible_devices
            or assignment.get("gpu_uuids") != list(observed_uuids)
            or len(observed_uuids) != world or len(set(observed_uuids)) != world):
        raise RuntimeError("Stage assignment/world/physical GPU binding differs")
    return assignment, train__digest(train__Path(path))

def train__physical_gpu_uuids(visible_devices):
    if not visible_devices:
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES required")
    selected = visible_devices.split(",")
    if any(not item.isdigit() for item in selected) or len(selected) != len(set(selected)):
        raise RuntimeError("Assignment must use unique numeric GPU indices")
    result = train__subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, stdout=train__subprocess.PIPE, stderr=train__subprocess.PIPE,
                            universal_newlines=True, timeout=20)
    inventory = {}
    for line in result.stdout.splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        if index in inventory or not uuid.startswith("GPU-"):
            raise RuntimeError("Invalid physical GPU inventory")
        inventory[index] = uuid
    if any(index not in inventory for index in selected):
        raise RuntimeError("Assigned physical GPU missing")
    return [inventory[index] for index in selected]

def train__topology_payload(ph, stage):
    if train__STAGE_ASSIGNMENT_SHA256 is None or train__STAGE_ASSIGNMENT is None:
        raise RuntimeError("Stage assignment was not verified")
    return dict(schema="uv_clean_ldm_stage_topology_v1", protocol_sha256=ph, stage=stage,
                world_size=train__WORLD_SIZE, assignment_sha256=train__STAGE_ASSIGNMENT_SHA256,
                visible_devices=train__STAGE_ASSIGNMENT["visible_devices"],
                gpu_uuids=train__STAGE_ASSIGNMENT["gpu_uuids"],
                policy="no_within_stage_world_or_assignment_migration")

def train__check_existing_topology(out, ph, stage):
    path = train__Path(out) / "execution_topology.json"
    expected = train__topology_payload(ph, stage)
    if path.exists() and train__read_json(path) != expected:
        raise RuntimeError("Stage topology already frozen; no world/assignment migration")
    return expected

def train__bind_stage_topology(out, ph, stage, rank, device):
    failure = None
    if rank == 0:
        try:
            expected = train__check_existing_topology(out, ph, stage)
            path = train__Path(out) / "execution_topology.json"
            if not path.exists():
                train__save_json(path, expected, immutable=True)
            if train__read_json(path) != expected:
                raise RuntimeError("Stage topology readback differs")
        except Exception:
            failure = dict(rank=rank, traceback=train__traceback.format_exc())
    train__synchronized_failure(failure, device, "Stage topology freeze failed")

def train__setup(c, ph, stage, assignment_path):
    if train__os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") != "PCI_BUS_ID":
        raise RuntimeError("Physical assignment requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    train__os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch
    import torch.distributed as dist
    import numpy as np
    global train__WORLD_SIZE, train__STAGE_ASSIGNMENT_SHA256, train__STAGE_ASSIGNMENT
    rank, local, world = (int(train__os.environ.get(k, -1)) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world not in train__ALLOWED_WORLD_SIZES or rank not in range(world) or local not in range(world):
        raise RuntimeError("Requires assigned torchrun two/four/six ranks")
    visible = train__os.environ.get("CUDA_VISIBLE_DEVICES")
    assignment, assignment_sha = train__verified_assignment(assignment_path, ph, stage, world,
                                                     visible, train__physical_gpu_uuids(visible))
    train__WORLD_SIZE, train__STAGE_ASSIGNMENT, train__STAGE_ASSIGNMENT_SHA256 = world, assignment, assignment_sha
    train__check_existing_topology(train__Path(c["output_dir"]) / stage, ph, stage)
    train__os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported; no automatic precision change")
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=train__datetime.timedelta(minutes=60))
    train__random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.cuda.manual_seed(c["seed"])
    return rank, torch.device("cuda", local)

def train__replica_equal(core, optimizer):
    import torch.distributed as dist
    fp = dict(model=train__tensor_fingerprint(core.state_dict()), optimizer=train__tensor_fingerprint(optimizer.state_dict()))
    items = [None] * train__WORLD_SIZE; dist.all_gather_object(items, fp)
    if any(v != items[0] for v in items): raise RuntimeError("Replicas differ")
    return fp

def train__frozen_ae(c, out, ph, device, require_stats=True):
    import torch
    KLUVAutoencoder = model__KLUVAutoencoder
    folder = out / "ae"; summary = train__read_json(folder / "training_summary.json")
    path = folder / "checkpoint_best.pt"
    if summary["status"] != "complete" or summary["protocol_sha256"] != ph or train__digest(path) != summary["checkpoint_best_sha256"]:
        raise RuntimeError("AE BEST is not completed/frozen")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != ph or checkpoint["stage"] != "ae": raise RuntimeError("Wrong AE")
    ae = KLUVAutoencoder(base_channels=32, latent_channels=4).float().to(device)
    ae.load_state_dict(checkpoint["model_state_dict"], strict=True); ae.eval(); ae.requires_grad_(False)
    assets = dict(ae_checkpoint=str(path), ae_checkpoint_sha256=train__digest(path))
    stats = None
    if require_stats:
        sp = out / "latent_stats" / "statistics.json"; stats = train__read_json(sp)
        if stats["ae_checkpoint_sha256"] != assets["ae_checkpoint_sha256"] or stats["protocol_sha256"] != ph or stats["images"] != train__COUNTS["train"]:
            raise RuntimeError("Wrong/incomplete latent statistics")
        if (stats.get("test_split_used") is not False
                or stats.get("count_values_per_channel") != train__COUNTS["train"] * 256 * 256
                or stats.get("source_manifest_sha256") != c["source_manifest_sha256"]
                or stats.get("train_manifest_sha256") != train__digest(out / "train_manifest.json")
                or stats.get("latent_shape_per_image") != [4, 256, 256]
                or stats.get("method") != "population_float64_all_train_posterior_mean"
                or stats.get("posterior_encoding_precision") != "FP32"):
            raise RuntimeError("Latent statistics coverage/source/precision differs")
        if len(stats["mean"]) != 4 or len(stats["std"]) != 4 or any(not train__math.isfinite(v) for v in stats["mean"] + stats["std"]) or any(v <= 0 for v in stats["std"]):
            raise RuntimeError("Invalid latent statistics; no flooring")
        assets.update(latent_stats=str(sp), latent_stats_sha256=train__digest(sp))
    return ae, stats, assets

def train__components(c, stage, out, ph, device):
    import torch
    import torch.distributed as dist
    KLUVAutoencoder = model__KLUVAutoencoder; RGBLatentVDenoiser = model__RGBLatentVDenoiser; VPLinearSchedule = model__VPLinearSchedule
    core = (KLUVAutoencoder(base_channels=32, latent_channels=4) if stage == "ae" else
            RGBLatentVDenoiser(base_channels=64, latent_channels=4, time_dimension=256)).float().to(device)
    if sum(parameter.numel() for parameter in core.parameters()) != (1754315 if stage == "ae" else 21051204):
        raise RuntimeError("Actual model parameter count differs from locked implementation")
    for v in core.state_dict().values(): dist.broadcast(v, src=0)
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2, min_lr=1e-6)
    ae, stats, assets = (None, None, {}) if stage == "ae" else train__frozen_ae(c, out, ph, device)
    schedule = VPLinearSchedule().to(device)
    
    rank = dist.get_rank()
    import numpy as np
    train__random.seed(c["seed"] + rank * 1000)
    np.random.seed(c["seed"] + rank * 1000)
    torch.manual_seed(c["seed"] + rank * 1000)
    torch.cuda.manual_seed(c["seed"] + rank * 1000)
    return core, optimizer, scheduler, ae, stats, assets, schedule

def train__normalized_latent(ae, target, stats):
    import torch
    with torch.no_grad():
        mu, _ = ae.encode(target)
    mean = torch.tensor(stats["mean"], device=target.device, dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(stats["std"], device=target.device, dtype=torch.float32).view(1, 4, 1, 1)
    z = (mu.float() - mean) / std
    if not bool(torch.isfinite(z).all()): raise FloatingPointError("Nonfinite normalized latent")
    return z

def train__objective(core, stage, rgb, target, ae=None, stats=None, schedule=None):
    import torch
    kl_divergence = model__kl_divergence
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
            
            with torch.autocast(device_type="cuda", enabled=False):
                z = train__normalized_latent(ae, target, stats)
                noise = torch.randn_like(z)
                t = torch.randint(0, schedule.alphas_cumprod.numel(), (1,), device=z.device)
                noisy = schedule.add_noise(z, noise, t)
                wanted = schedule.v_target(z, noise, t)
            prediction = core(noisy, t, rgb)
            if not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Nonfinite velocity")
            loss = (prediction.float() - wanted.float()).square().mean()
            l1 = torch.zeros((), device=rgb.device)  
    if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite objective")
    return loss, l1

def train__synchronized_failure(failure, device, label):
    import torch.distributed as dist
    if not training_common__finite_all_ranks(failure is None, device):
        items = [None] * train__WORLD_SIZE; dist.all_gather_object(items, failure)
        raise RuntimeError(label + ": " + train__json.dumps(items))

def train__global_update(core, optimizer, state, accumulator, indices, rank, device, get_pair,
                  stage="ae", ae=None, stats=None, schedule=None):
    import torch
    import torch.distributed as dist
    core.train(); optimizer.zero_grad(set_to_none=True)
    failure, current = None, None
    local = training_common__new_accumulator()
    try:
        selected = train__local_indices(indices, rank)
        for current in selected:
            rgb, target = get_pair(current)
            loss, l1 = train__objective(core, stage, rgb, target, ae, stats, schedule)
            (loss * (float(train__WORLD_SIZE) / len(indices))).backward()
            local["images"] += 1; local["nll_sum"] += float(loss); local["l1_sum"] += float(l1)
            del rgb, target, loss, l1
        if not selected:

            for parameter in core.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in core.parameters() if p.requires_grad):
            raise FloatingPointError("Missing/nonfinite local gradient")
    except Exception: failure = dict(rank=rank, sample_index=current, traceback=train__traceback.format_exc())
    train__synchronized_failure(failure, device, "Local forward/backward failed; no update")
    for p in core.parameters():
        if p.requires_grad: dist.all_reduce(p.grad); p.grad.div_(train__WORLD_SIZE)
    norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
    if not training_common__finite_all_ranks(bool(torch.isfinite(norm)), device): raise FloatingPointError("Nonfinite reduced gradient")
    failure = None
    try:
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in core.parameters()): raise FloatingPointError("Nonfinite updated model")
        if any(not bool(torch.isfinite(v).all()) for row in optimizer.state.values() for v in row.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError("Nonfinite optimizer")
    except Exception: failure = dict(rank=rank, traceback=train__traceback.format_exc())
    train__synchronized_failure(failure, device, "Optimizer failed; partial state not checkpointed")
    optimizer.zero_grad(set_to_none=True)
    state["global_step"] += 1; state["global_attempts"] += 1
    for k in local: accumulator[k] += local[k]
    return float(norm)

def train__checkpoint_metadata(order, state):
    train__epoch_batches(order, state["next_batch"])
    return dict(rule="global16_assigned_rank_stride_no_padding_idle_tail_zero_gradient", world_size=train__WORLD_SIZE, physical_batch=1,
                stage_assignment_sha256=train__STAGE_ASSIGNMENT_SHA256,
                order_sha256=batches__order_sha256(order), next_batch=state["next_batch"], samples=train__COUNTS["train"])

def train__save_checkpoint(out, stage, ph, core, optimizer, scheduler, state, accumulator, order, rank, device, generator, assets, best=False):
    import torch.distributed as dist
    fingerprints = train__replica_equal(core, optimizer)
    local = dict(rank=rank, rng=training_common__capture_rng(device), epoch_accumulator=accumulator, loader_generator_state=generator.get_state())
    gathered = [None] * train__WORLD_SIZE; dist.all_gather_object(gathered, local)
    if rank == 0:
        payload = dict(schema=train__SCHEMA, stage=stage, protocol_sha256=ph, state=state,
                       model_state_dict=core.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(), rank_states=gathered,
                       batching_resume=train__checkpoint_metadata(order, state), replica_fingerprints=fingerprints,
                       frozen_assets=assets, ema_used=False, test_split_used=False,
                       training_complete=state["phase"] == "complete", precision="BF16_forward_FP32_loss_no_scaler")
        if best: training_common__atomic_torch_save(out / "checkpoint_best.pt", payload)
        training_common__atomic_torch_save(out / "checkpoint_last.pt", payload)
        train__save_json(out / "checkpoint_commit.json", dict(sha256=train__digest(out / "checkpoint_last.pt"), epoch=state["epoch"],
                   global_step=state["global_step"], next_batch=state["next_batch"], phase=state["phase"]))
        train__save_json(out / "history.json", state["history"])
    dist.barrier()

def train__verify_checkpoint(cp, stage, ph, order, assets):
    if cp.get("schema") != train__SCHEMA or cp.get("stage") != stage or cp.get("protocol_sha256") != ph:
        raise RuntimeError("Foreign checkpoint")
    if cp.get("frozen_assets") != assets or cp.get("ema_used") is not False or cp.get("test_split_used") is not False:
        raise RuntimeError("Checkpoint asset/scope mismatch")
    s = cp["state"]
    if (s["phase"] not in ("train", "val", "epoch_complete", "complete")
            or type(s["epoch"]) is not int or not 1 <= s["epoch"] <= 20):
        raise RuntimeError("Invalid checkpoint phase")
    if (cp.get("precision") != "BF16_forward_FP32_loss_no_scaler"
            or cp.get("training_complete") is not (s["phase"] == "complete")):
        raise RuntimeError("Checkpoint precision/completion differs")
    if s["global_step"] != s["global_attempts"] or s["amp_overflow_skips"] != 0 or s.get("blocked_reason"):
        raise RuntimeError("Unsafe counters")
    if (s["global_step"] != (s["epoch"] - 1) * train__EPOCH_BATCHES + s["next_batch"]
            or (s["phase"] != "train" and s["next_batch"] != train__EPOCH_BATCHES)):
        raise RuntimeError("Checkpoint epoch/step/offset differs")
    expected_history = s["epoch"] if s["phase"] in ("epoch_complete", "complete") else s["epoch"] - 1
    if len(s["history"]) != expected_history:
        raise RuntimeError("Checkpoint history coverage differs")
    if cp["batching_resume"] != train__checkpoint_metadata(order, s): raise RuntimeError("Resume order/layout changed")
    if [v["rank"] for v in cp["rank_states"]] != list(range(train__WORLD_SIZE)): raise RuntimeError("Rank RNG missing/out of order")
    for v in cp["rank_states"]:
        consumed = sum(len(train__local_indices(indices, v["rank"])) for _, indices in train__epoch_batches(order)[:s["next_batch"]])
        if v["epoch_accumulator"]["images"] != consumed: raise RuntimeError("Accumulator offset differs")
        if any(not train__math.isfinite(v["epoch_accumulator"][k]) for k in ("nll_sum", "l1_sum")): raise RuntimeError("Saved nonfinite stats")
    fp = dict(model=train__tensor_fingerprint(cp["model_state_dict"]), optimizer=train__tensor_fingerprint(cp["optimizer_state_dict"]))
    if cp["replica_fingerprints"] != fp: raise RuntimeError("Saved state fingerprint differs")

def train__val_prediction(core, stage, rgb, target, sample_id, ae, stats, schedule):
    ddim_sample = model__ddim_sample
    if stage == "ae": return core(target, sample=False)["reconstruction"]
    return ddim_sample(core, rgb, [sample_id], stats["mean"], stats["std"], ae,
                       num_steps=20, generation_index=0, seed=20260817, schedule=schedule)

def train__run_validation(core, stage, dataset, rows, out, ph, state, c, rank, device, ae, stats, schedule):
    import torch
    import torch.distributed as dist
    indices = list(range(rank, len(rows), train__WORLD_SIZE)); core.eval()
    sid = "%s:%s:e%d:s%d" % (ph, stage, state["epoch"], state["global_step"])
    directory = out / "validation" / ("epoch_%03d" % state["epoch"]); directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("rank%d.jsonl" % rank)
    done, failure = [], None
    try:
        if path.exists():
            for pos, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True)):
                if not line.endswith("\n"): raise RuntimeError("Truncated validation record; preserve")
                r = train__json.loads(line)
                if pos >= len(indices) or r["index"] != indices[pos] or r["sample_id"] != rows[indices[pos]]["sample_id"] or r["state_id"] != sid:
                    raise RuntimeError("Validation identity/state mismatch")
                if not train__math.isfinite(r["l1"]): raise FloatingPointError("Nonfinite persisted L1")
                done.append(r)
        loader = training_common__make_loader(dataset, [[i] for i in indices[len(done):]], c, c["seed"] + rank + state["epoch"] * train__WORLD_SIZE + 500000)
        try:
            with torch.inference_mode(), path.open("a", encoding="utf-8") as handle:
                for pos, b in enumerate(loader, len(done)):
                    i = indices[pos]
                    if b["sample_id"] != [rows[i]["sample_id"]]: raise RuntimeError("Validation loader identity mismatch")
                    rgb, target = b["rgb"].to(device).float(), b["target"].to(device).float()
                    prediction = train__val_prediction(core, stage, rgb, target, rows[i]["sample_id"], ae, stats, schedule)
                    if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Invalid validation output")
                    l1 = float((prediction.float() - target).abs().mean())
                    r = dict(index=i, sample_id=rows[i]["sample_id"], state_id=sid, protocol_sha256=ph, l1=l1, test_split_used=False)
                    handle.write(train__json.dumps(r, allow_nan=False) + "\n"); handle.flush(); done.append(r)
                    if len(done) % 25 == 0:
                        train__os.fsync(handle.fileno()); train__save_json(out / ("validation_progress_rank%d.json" % rank), dict(status="validating", stage=stage, epoch=state["epoch"], completed_images=len(done), expected_images=len(indices)))
                    del rgb, target, prediction, b
                train__os.fsync(handle.fileno())
        finally:
            if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
        if len(done) != len(indices): raise RuntimeError("Incomplete validation")
    except Exception:
        failure = dict(rank=rank, traceback=train__traceback.format_exc()); train__save_json(directory / ("failure_rank%d_%d.json" % (rank, train__time.time_ns())), failure, immutable=True)
    train__synchronized_failure(failure, device, "Validation failed")
    totals = torch.tensor([len(done), sum(r["l1"] for r in done)], dtype=torch.float64, device=device); dist.all_reduce(totals)
    if int(totals[0]) != train__COUNTS["val"]: raise RuntimeError("Full VAL coverage differs")
    train__save_json(out / ("validation_progress_rank%d.json" % rank), dict(status="complete", epoch=state["epoch"], completed_images=len(done), record_sha256=train__digest(path)))
    return dict(images=int(totals[0]), l1=float(totals[1] / totals[0]), single_generation=True, ddim20=stage == "diffusion", precision="FP32")

class train__StopAtBoundary:
    def __init__(self):
        self.requested = False
        for sig in (train__signal.SIGTERM, train__signal.SIGINT, train__signal.SIGHUP): train__signal.signal(sig, self.handle)
    def handle(self, *_): self.requested = True

def train__any_rank(flag, device):
    import torch
    import torch.distributed as dist
    value = torch.tensor(int(flag), device=device); dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())

def train__check_preflight(directory, config_path, ph, stage, assets):
    if train__STAGE_ASSIGNMENT_SHA256 is None:
        raise RuntimeError("Verified stage assignment required before preflight gate")
    for rank in range(train__WORLD_SIZE):
        p = train__read_json(train__Path(directory) / ("rank%d.json" % rank))
        if (p.get("status") != "passed" or p.get("stage") != stage or p.get("rank") != rank or
            p.get("protocol_sha256") != ph or p.get("config_sha256") != train__digest(config_path) or
            p.get("code_sha256") != {n: train__digest(train__ROOT / n) for n in train__CODE_FILES} or p.get("software") != train__runtime() or
            p.get("frozen_assets") != assets or p.get("resume_equivalence_pass") is not True or
            p.get("world_size") != train__WORLD_SIZE or p.get("stage_assignment_sha256") != train__STAGE_ASSIGNMENT_SHA256 or
            p.get("global_batch") != 16 or p.get("tail_global_batch") != 2 or
            p.get("resume_sequence") != "full16_tail2_full16" or
            p.get("visible_devices") != train__os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise RuntimeError("Preflight gate incomplete/mismatched")
        expected_checks = {key: True for key in ("model", "optimizer", "scheduler", "rng", "state", "accumulator", "loader_rng")}
        if (p.get("resume_checks") != expected_checks
                or p.get("formal_checkpoint_written") is not False
                or p.get("weights_reused_for_training") is not False
                or p.get("test_split_used") is not False
                or p.get("real_train_samples") != 2 or p.get("real_val_samples") != 2):
            raise RuntimeError("Preflight replay/disposable real-data gate differs")
        validation = p.get("validation", [])
        if (len(validation) != 2
                or any(not train__math.isfinite(v.get("l1", float("nan"))) or v["l1"] < 0 for v in validation)
                or any(not train__math.isfinite(p.get(k, float("nan"))) or p[k] <= 0
                       for k in ("peak_train_mib", "total_seconds"))):
            raise RuntimeError("Preflight numerical/coverage diagnostics incomplete")

def train__train(config_path, stage, resume="auto", preflight_directory=None, assignment_path=None):
    c, root, p, ph, splits = train__load_frozen(config_path)
    if preflight_directory is None: raise RuntimeError("Fresh preflight required")
    import torch
    import torch.distributed as dist
    rank, device = train__setup(c, ph, stage, assignment_path); out = root / stage; loader = None
    state, acc = training_common__new_state(), training_common__new_accumulator(); stopper = train__StopAtBoundary()
    try:
        with training_common__exclusive_lock(out / "training_coordinator.lock") if rank == 0 else train__contextlib.nullcontext():
            dist.barrier()
            core, opt, sch, ae, stats, assets, schedule = train__components(c, stage, root, ph, device)
            train__check_preflight(preflight_directory, config_path, ph, stage, assets)
            train__bind_stage_topology(out, ph, stage, rank, device)
            gen = torch.Generator().manual_seed(c["seed"] + rank * 1000)
            last = out / "checkpoint_last.pt"
            if last.exists():
                if resume == "none" or train__digest(last) != train__read_json(out / "checkpoint_commit.json")["sha256"]: raise RuntimeError("LAST/commit differs")
                cp = torch.load(last, map_location="cpu", weights_only=False); state = cp["state"]
                order = train__load_order(root, p, state["epoch"]); train__verify_checkpoint(cp, stage, ph, order, assets)
                core.load_state_dict(cp["model_state_dict"], strict=True); opt.load_state_dict(cp["optimizer_state_dict"]); sch.load_state_dict(cp["scheduler_state_dict"])
                local = cp["rank_states"][rank]; acc = local["epoch_accumulator"]; gen.set_state(local["loader_generator_state"]); training_common__restore_rng(local["rng"], device)
                del cp, local
            elif any((out / n).exists() for n in ("history.json", "checkpoint_best.pt", "training_progress.json")):
                raise RuntimeError("Prior artifacts without LAST; no restart")
            train__replica_equal(core, opt)
            if state["phase"] == "complete":
                if rank == 0:
                    summary = training_common__final_summary(out, state, ph)
                    summary.update(stage=stage, true_diffusion=stage == "diffusion", frozen_assets=assets)
                    train__save_json(out / "training_summary.json", summary)
                return
            training_common__advance_completed_epoch(state)
            train_data = training_common__TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
            val_data = training_common__TrainingOnlyDataset(splits["val"], c["dataset_root"], "val")
            train__save_json(out / ("environment_rank%d_%d.json" % (rank, train__time.time_ns())), dict(rank=rank, gpu=torch.cuda.get_device_name(device), software=train__runtime(), precision="BF16", parameters=sum(v.numel() for v in core.parameters()), test_split_used=False), immutable=True)
            last_save, started, first_attempt = train__time.monotonic(), train__time.monotonic(), state["global_attempts"]
            while state["phase"] != "complete":
                epoch = state["epoch"]; order = train__load_order(root, p, epoch)
                if state["phase"] == "next_epoch": state.update(phase="train", next_batch=0); acc = training_common__new_accumulator()
                if not last.exists(): train__save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets)
                if state["phase"] == "train":
                    plans = train__epoch_batches(order, state["next_batch"])
                    loader = training_common__make_loader(train_data, [[i] for _, indices in plans for i in train__local_indices(indices, rank)], c, c["seed"] + rank * 1000)
                    loader.generator.set_state(gen.get_state()); iterator = iter(loader); gen.set_state(loader.generator.get_state())
                    def get_pair(i):
                        b = next(iterator)
                        if b["sample_id"] != [splits["train"][i]["sample_id"]]: raise RuntimeError("Train identity mismatch")
                        return b["rgb"].to(device, non_blocking=True).float(), b["target"].to(device, non_blocking=True).float()
                    for pos, indices in plans:
                        norm = train__global_update(core, opt, state, acc, indices, rank, device, get_pair, stage, ae, stats, schedule)
                        state["next_batch"] = pos + 1
                        stop = train__any_rank(stopper.requested, device)
                        due = train__any_rank(train__time.monotonic() - last_save >= 600 if rank == 0 else False, device)
                        if state["global_attempts"] in (1, 20) or state["global_attempts"] % 1000 == 0 or due or stop:
                            train__save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets); last_save = train__time.monotonic()
                        if state["global_attempts"] % 20 == 0 or state["global_attempts"] == 1:
                            totals = training_common__reduced_accumulator(acc, device); seconds = (train__time.monotonic() - started) / max(1, state["global_attempts"] - first_attempt)
                            training_common__log_progress(out, dict(status="training", stage=stage, epoch=epoch, max_epochs=20, epoch_batch=state["next_batch"], epoch_batches=train__EPOCH_BATCHES, global_step=state["global_step"], global_attempts=state["global_attempts"], amp_overflow_skips=0, train_images=totals["images"], train_loss=totals["nll"], train_l1=totals["l1"] if stage == "ae" else None, gradient_norm=norm, seconds_per_global_batch=seconds, epoch_train_eta_seconds=seconds * (train__EPOCH_BATCHES - state["next_batch"])), rank)
                        if stop: return
                    if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
                    loader = None; totals = training_common__reduced_accumulator(acc, device)
                    if totals["images"] != train__COUNTS["train"]: raise RuntimeError("Training coverage differs")
                    state["phase"] = "val"; train__save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets)
                totals = training_common__reduced_accumulator(acc, device)
                training_common__log_progress(out, dict(status="validating", stage=stage, epoch=epoch, global_step=state["global_step"]), rank)
                val_started = train__time.monotonic()
                val = train__run_validation(core, stage, val_data, splits["val"], out, ph, state, c, rank, device, ae, stats, schedule)
                selection, best, early = training_common__update_selection(state["selection"], val["l1"], epoch)
                state["selection"] = selection; sch.step(val["l1"])
                row = dict(epoch=epoch, stage=stage, train_images=totals["images"], train_loss=totals["nll"], validation=val,
                           validation_seconds=train__time.monotonic() - val_started, global_step=state["global_step"], is_best=best, amp_overflow_skips=0)
                state["history"].append(row)
                if early or epoch == 20: state.update(phase="complete", stop_reason="early_stopping" if early else "max_epochs")
                else: state["phase"] = "epoch_complete"
                train__save_checkpoint(out, stage, ph, core, opt, sch, state, acc, order, rank, device, gen, assets, best=best)
                training_common__log_progress(out, dict(row, status="epoch_complete", best_epoch=selection["best_epoch"], best_val_l1=selection["best_val_l1"]), rank)
                if state["phase"] == "epoch_complete": training_common__advance_completed_epoch(state)
            if rank == 0:
                summary = training_common__final_summary(out, state, ph); summary.update(stage=stage, true_diffusion=stage == "diffusion", frozen_assets=assets)
                train__save_json(out / "training_summary.json", summary)
    except Exception:
        train__save_json(out / ("failure_rank%d_%d.json" % (rank, train__time.time_ns())), dict(status="failed", rank=rank, state=state, traceback=train__traceback.format_exc(), test_split_used=False), immutable=True)
        raise
    finally:
        if loader is not None and getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
        if dist.is_initialized(): dist.destroy_process_group()

def train__preflight(config_path, stage, directory, assignment_path=None):
    c, root, p, ph, splits = train__load_frozen(config_path)
    import torch
    import torch.distributed as dist
    rank, device = train__setup(c, ph, stage, assignment_path); directory = train__Path(directory)
    if directory.resolve() == root.resolve() or root.resolve() in directory.resolve().parents:
        raise RuntimeError("Preflight must be outside formal experiment")
    try:
        if rank == 0: directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        core, opt, sch, ae, stats, assets, schedule = train__components(c, stage, root, ph, device)
        state, acc = training_common__new_state(), training_common__new_accumulator(); gen = torch.Generator().manual_seed(c["seed"] + rank * 1000)
        torch.cuda.reset_peak_memory_stats(device)
        def synthetic(_):
            rgb = torch.rand(1, 3, 1024, 1024, device=device); return rgb, torch.rand_like(rgb)
        started = train__time.monotonic()
        train__global_update(core, opt, state, acc, tuple(range(16)), rank, device, synthetic, stage, ae, stats, schedule)
        snap = dict(model=core.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), rng=training_common__capture_rng(device), state=state, accumulator=acc, loader_rng=gen.get_state())
        buffer = train__io.BytesIO(); torch.save(snap, buffer)
        def tail():
            train__global_update(core, opt, state, acc, tuple(range(16, 18)), rank, device, synthetic, stage, ae, stats, schedule)
            train__global_update(core, opt, state, acc, tuple(range(18, 34)), rank, device, synthetic, stage, ae, stats, schedule)
            sch.step(.2); torch.rand(2, generator=gen)
        def snapshot():
            return training_common__cpu_snapshot(dict(model=core.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), rng=training_common__capture_rng(device), state=state, accumulator=acc, loader_rng=gen.get_state()))
        tail(); expected = snapshot(); buffer.seek(0); saved = torch.load(buffer, map_location="cpu", weights_only=False)
        core.load_state_dict(saved["model"], strict=True); opt.load_state_dict(saved["optimizer"]); sch.load_state_dict(saved["scheduler"])
        training_common__restore_rng(saved["rng"], device); gen.set_state(saved["loader_rng"]); state, acc = saved["state"], saved["accumulator"]
        del saved, snap, buffer
        tail(); observed = snapshot(); checks = {k: training_common__state_equal(expected[k], observed[k]) for k in expected}
        if not training_common__finite_all_ranks(all(checks.values()), device): raise RuntimeError("Checkpoint/RNG replay differs: " + train__json.dumps(checks))
        del expected, observed
        train_data = training_common__TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
        val_data = training_common__TrainingOnlyDataset(splits["val"], c["dataset_root"], "val")
        order = train__load_order(root, p, 1); real_indices = order[:2]
        samples = []
        for split, indices in (("train", real_indices), ("val", [0, 1])):
            for i in indices:
                row = splits[split][i]
                for mod in ("rgb", "uv"):
                    if train__digest(train__Path(c["dataset_root"]) / row[mod + "_path"]) != row[mod + "_sha256"]:
                        raise RuntimeError("Real source hash differs")
                samples.append(dict(split=split, sample_id=row["sample_id"]))
        def real_pair(i):
            item = train_data[real_indices[i % 2]]
            return item["rgb"].unsqueeze(0).to(device).float(), item["target"].unsqueeze(0).to(device).float()
        train__global_update(core, opt, state, acc, tuple(range(16)), rank, device, real_pair, stage, ae, stats, schedule)
        train_peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
        train__replica_equal(core, opt); core.eval(); values = []
        with torch.inference_mode():
            for i in (0, 1):
                item = val_data[i]; rgb, target = item["rgb"].unsqueeze(0).to(device).float(), item["target"].unsqueeze(0).to(device).float()
                prediction = train__val_prediction(core, stage, rgb, target, item["sample_id"], ae, stats, schedule)
                if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()): raise FloatingPointError("Real VAL generation invalid")
                values.append(dict(sample_id=item["sample_id"], l1=float((prediction - target).abs().mean())))
        result = dict(status="passed", stage=stage, rank=rank, world_size=train__WORLD_SIZE, visible_devices=train__os.environ.get("CUDA_VISIBLE_DEVICES"),
                      protocol_sha256=ph, config_sha256=train__digest(config_path), software=train__runtime(), code_sha256={n: train__digest(train__ROOT / n) for n in train__CODE_FILES},
                      stage_assignment_sha256=train__STAGE_ASSIGNMENT_SHA256,
                      frozen_assets=assets, resume_equivalence_pass=True, resume_checks=checks, global_batch=16, tail_global_batch=2, resume_sequence="full16_tail2_full16",
                      real_train_samples=2, real_val_samples=2, repeated_real_samples_only_for_disposable_global16_preflight=True,
                      real_samples=samples, validation=values, peak_train_mib=train_peak, total_seconds=train__time.monotonic() - started,
                      formal_checkpoint_written=False, weights_reused_for_training=False, test_split_used=False)
        train__save_json(directory / ("rank%d.json" % rank), result, immutable=True)
        if train__read_json(directory / ("rank%d.json" % rank)) != result: raise RuntimeError("Preflight readback differs")
        dist.barrier(); print(train__json.dumps(dict(rank=rank, stage=stage, status="passed", peak_train_mib=train_peak)), flush=True)
    except Exception:
        if directory.exists(): train__save_json(directory / ("failure_rank%d_%d.json" % (rank, train__time.time_ns())), dict(status="failed", traceback=train__traceback.format_exc()), immutable=True)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

def train__latent_stats(config_path, assignment_path=None):
    """"""
    c, root, p, ph, splits = train__load_frozen(config_path)
    import torch
    import torch.distributed as dist
    rank, device = train__setup(c, ph, "latent_stats", assignment_path); out = root / "latent_stats"
    try:
        ae, _, assets = train__frozen_ae(c, root, ph, device, require_stats=False)
        with training_common__exclusive_lock(out / "coordinator.lock") if rank == 0 else train__contextlib.nullcontext():
            dist.barrier(); train__bind_stage_topology(out, ph, "latent_stats", rank, device)
            path = out / ("rank%d.json" % rank)
            indices = list(range(rank, train__COUNTS["train"], train__WORLD_SIZE)); done = 0
            sums = torch.zeros(4, dtype=torch.float64, device=device); squares = sums.clone(); count = 0
            if path.exists():
                old = train__read_json(path)
                if (old["protocol_sha256"] != ph or old["assets"] != assets or old["rank"] != rank
                        or old.get("test_split_used") is not False): raise RuntimeError("Stats identity differs")
                done, count = old["completed_images"], old["count_values_per_channel"]
                if type(done) is not int or not 0 <= done <= len(indices): raise RuntimeError("Stats offset invalid")
                if count != done * 256 * 256: raise RuntimeError("Stats count/offset differs")
                if (len(old["sums"]) != 4 or len(old["squares"]) != 4
                        or any(not train__math.isfinite(v) for v in old["sums"] + old["squares"])):
                    raise RuntimeError("Nonfinite/malformed latent moments")
                sums = torch.tensor(old["sums"], dtype=torch.float64, device=device); squares = torch.tensor(old["squares"], dtype=torch.float64, device=device)
            data = training_common__TrainingOnlyDataset(splits["train"], c["dataset_root"], "train")
            loader = training_common__make_loader(data, [[i] for i in indices[done:]], c, c["seed"] + rank + 900000)
            failure = None
            try:
                with torch.inference_mode():
                    for pos, item in enumerate(loader, done):
                        if item["sample_id"] != [splits["train"][indices[pos]]["sample_id"]]: raise RuntimeError("Stats identity mismatch")
                        mu, _ = ae.encode(item["target"].to(device).float()); z = mu.double()
                        if not bool(torch.isfinite(z).all()): raise FloatingPointError("Nonfinite AE posterior mean")
                        sums += z.sum((0, 2, 3)); squares += z.square().sum((0, 2, 3)); count += z.shape[0] * z.shape[2] * z.shape[3]; done = pos + 1
                        if done % 100 == 0 or done == len(indices):
                            train__save_json(path, dict(rank=rank, completed_images=done, count_values_per_channel=count, sums=sums.tolist(), squares=squares.tolist(), protocol_sha256=ph, assets=assets, test_split_used=False))
                        del item, mu, z
            except Exception: failure = dict(rank=rank, traceback=train__traceback.format_exc())
            finally:
                if getattr(loader, "_iterator", None) is not None: loader._iterator._shutdown_workers()
            train__synchronized_failure(failure, device, "Latent statistics failed")
            totals = torch.cat([torch.tensor([done, count], dtype=torch.float64, device=device), sums, squares]); dist.all_reduce(totals)
            if (int(totals[0]) != train__COUNTS["train"]
                    or int(totals[1]) != train__COUNTS["train"] * 256 * 256): raise RuntimeError("Stats coverage differs")
            mean = totals[2:6] / totals[1]; variance = totals[6:10] / totals[1] - mean.square()
            if not bool(torch.isfinite(variance).all()) or not bool((variance > 0).all()): raise FloatingPointError("Invalid latent std; no flooring")
            result = dict(schema="uv_ldm_all_train_latent_statistics_v1", images=train__COUNTS["train"], count_values_per_channel=int(totals[1]), mean=mean.tolist(), std=variance.sqrt().tolist(),
                          protocol_sha256=ph, ae_checkpoint_sha256=assets["ae_checkpoint_sha256"], method="population_float64_all_train_posterior_mean", test_split_used=False)
            result.update(train_manifest_sha256=p["manifest_sha256"]["train"], source_manifest_sha256=c["source_manifest_sha256"],
                          latent_shape_per_image=[4, 256, 256], accumulator_dtype="float64", posterior_encoding_precision="FP32")
            if rank == 0:
                target = out / "statistics.json"
                if target.exists() and train__read_json(target) != result: raise RuntimeError("Existing stats differ; preserve")
                if not target.exists(): train__save_json(target, result, immutable=True)
            dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

def train__audit(config_path, stage):
    c, root, p, ph, splits = train__load_frozen(config_path)
    folder = root / stage; path = folder / "training_summary.json"
    if not path.exists(): raise RuntimeError("Training summary not yet complete")
    summary = train__read_json(path)
    if summary["status"] != "complete" or summary["protocol_sha256"] != ph or summary["test_split_used"] is not False:
        raise RuntimeError("Invalid completed stage")
    for name in ("best", "last"):
        if train__digest(folder / ("checkpoint_" + name + ".pt")) != summary["checkpoint_" + name + "_sha256"]:
            raise RuntimeError("Completed checkpoint changed")
    print(train__json.dumps(dict(status="complete", stage=stage, summary=summary), allow_nan=False), flush=True)

def train__main():
    ap = train__argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("prepare", "preflight", "train", "latent-stats", "audit"))
    ap.add_argument("--config", required=True, type=train__Path)
    ap.add_argument("--stage", choices=("ae", "diffusion"), default="ae")
    ap.add_argument("--resume", choices=("auto", "none"), default="auto")
    ap.add_argument("--preflight-output", type=train__Path)
    ap.add_argument("--stage-assignment", type=train__Path)
    args = ap.parse_args()
    if args.mode in ("preflight", "train", "latent-stats") and args.stage_assignment is None:
        ap.error("--stage-assignment required for distributed execution")
    if args.mode == "prepare": train__prepare(args.config)
    elif args.mode == "latent-stats": train__latent_stats(args.config, args.stage_assignment)
    elif args.mode == "audit": train__audit(args.config, args.stage)
    elif args.mode == "preflight":
        if args.preflight_output is None: ap.error("--preflight-output required")
        train__preflight(args.config, args.stage, args.preflight_output, args.stage_assignment)
    else: train__train(args.config, args.stage, args.resume, args.preflight_output, args.stage_assignment)

train__MODULE = _PlainNamespace(**{
    'ALLOWED_WORLD_SIZES': train__ALLOWED_WORLD_SIZES,
    'CODE_FILES': train__CODE_FILES,
    'COUNTS': train__COUNTS,
    'EPOCH_BATCHES': train__EPOCH_BATCHES,
    'Path': train__Path,
    'ROOT': train__ROOT,
    'SCHEMA': train__SCHEMA,
    'SOURCE_HASHES': train__SOURCE_HASHES,
    'STAGE_ASSIGNMENT': train__STAGE_ASSIGNMENT,
    'STAGE_ASSIGNMENT_SHA256': train__STAGE_ASSIGNMENT_SHA256,
    'StopAtBoundary': train__StopAtBoundary,
    'WORLD_SIZE': train__WORLD_SIZE,
    'any_rank': train__any_rank,
    'argparse': train__argparse,
    'audit': train__audit,
    'bind_stage_topology': train__bind_stage_topology,
    'check_existing_topology': train__check_existing_topology,
    'check_preflight': train__check_preflight,
    'checkpoint_metadata': train__checkpoint_metadata,
    'components': train__components,
    'contextlib': train__contextlib,
    'datetime': train__datetime,
    'digest': train__digest,
    'epoch_batches': train__epoch_batches,
    'frozen_ae': train__frozen_ae,
    'global_update': train__global_update,
    'hashlib': train__hashlib,
    'io': train__io,
    'json': train__json,
    'latent_stats': train__latent_stats,
    'load_frozen': train__load_frozen,
    'load_order': train__load_order,
    'local_indices': train__local_indices,
    'main': train__main,
    'math': train__math,
    'normalized_latent': train__normalized_latent,
    'objective': train__objective,
    'order_sha256': batches__order_sha256,
    'original_sources': train__original_sources,
    'os': train__os,
    'physical_gpu_uuids': train__physical_gpu_uuids,
    'preflight': train__preflight,
    'prepare': train__prepare,
    'random': train__random,
    'read_json': train__read_json,
    'ref': training_common__MODULE,
    'reference_assets': train__reference_assets,
    'replica_equal': train__replica_equal,
    'run_validation': train__run_validation,
    'runtime': train__runtime,
    'save_checkpoint': train__save_checkpoint,
    'save_json': train__save_json,
    'setup': train__setup,
    'shutil': train__shutil,
    'signal': train__signal,
    'subprocess': train__subprocess,
    'synchronized_failure': train__synchronized_failure,
    'sys': train__sys,
    'tensor_fingerprint': train__tensor_fingerprint,
    'time': train__time,
    'topology_payload': train__topology_payload,
    'traceback': train__traceback,
    'train': train__train,
    'val_prediction': train__val_prediction,
    'validate_config': train__validate_config,
    'verified_assignment': train__verified_assignment,
    'verify_checkpoint': train__verify_checkpoint,
})

MODULE_EXPORTS = {
    'src.uv_latent_diffusion_v1': vars(model__MODULE),
    'src.matched_training_batching': vars(batches__MODULE),
    'src.highres_patch_dataset': vars(data__MODULE),
    'src.unet_baseline': vars(layers__MODULE),
    'src.uv_uncertainty_unet': vars(uv_core__MODULE),
    'tools.train_uv_uncertainty_fullimage': vars(training_common__MODULE),
    'tools.train_uv_ldm_clean_multigpu_20261016': vars(train__MODULE),
}
MODEL_EXPORTS = MODULE_EXPORTS['src.uv_latent_diffusion_v1']
TRAIN_EXPORTS = MODULE_EXPORTS['tools.train_uv_ldm_clean_multigpu_20261016']



if __name__ == "__main__":
    import sys
    if "--sigma-config" in sys.argv:
        sys.argv[sys.argv.index("--sigma-config")] = "--config"
        training_main()
    else:
        train__main()
