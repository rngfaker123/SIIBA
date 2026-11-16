"""True RGB-conditioned UV latent diffusion, trained from scratch in two stages.

The UV autoencoder has no input-to-output skip. The denoiser consumes RGB and
noisy latent only, never UV at inference. Latent normalization uses train-only
posterior-mean statistics, not a borrowed pretrained-model scale. All posterior
and diffusion algebra is FP32 with explicit non-finite failures, never clamps.
"""

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
    """f4 KL codec. RGB images here are UV's three image channels in [0,1]."""

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
    """Pure-noise DDIM eta=0 -> UV, no truth or UV encoder invocation.

    Four/eight sampling repeats are distinct generation_index values, not
    training repeats. Output averaging is intentionally absent here.
    """
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
