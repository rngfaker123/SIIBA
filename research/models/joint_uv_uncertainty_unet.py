"""Prepared joint three-output U-Net with a separate full UV sigma decoder.

No training entry point or frozen joint objective is supplied here. The
professor's plan specifies the UV Gaussian likelihood, but does not specify
PLV/PLH losses or joint weights. PLV/PLH L1 components below are explicit
candidate components, not evidence that the training protocol is approved.

Unlike the existing single-UV uncertainty U-Net, this architecture has one
shared encoder/bottleneck and FOUR independently parameterized full decoders.
It is a probabilistic deterministic feed-forward U-Net, NOT a diffusion model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import nn

from models.unet_baseline import DoubleConv


VARIANCE_FLOOR = 1e-6
ARCHITECTURE_CONTRACT = {
    "status": "prepared_not_trained_joint_loss_not_frozen",
    "model_kind": "joint_feedforward_heteroscedastic_unet_not_diffusion",
    "input_channels": 3,
    "default_encoder_channels": [32, 64, 128, 256],
    "default_bottleneck_channels": 512,
    "encoder": "shared original DoubleConv Conv3x3(no bias)-GroupNorm-ReLU twice",
    "decoder_names": ["decoder_uv", "decoder_plv", "decoder_plh", "decoder_uv_uncertainty"],
    "decoder_sharing": "none; four complete independent upsample-skip-DoubleConv paths",
    "mean_outputs": {"uv": 3, "plv": 3, "plh": 3},
    "mean_activation": "sigmoid",
    "uv_raw_log_variance_channels": 1,
    "variance": "exp(s)+1e-6; evaluated as logaddexp(s,log(1e-6)) then exp",
    "sigma": "sqrt(variance), one pixelwise value shared across UV color channels",
    "parameterization_note": "s denotes raw log variance, not professor pseudocode log standard deviation; without the floor log_sigma=s/2",
    "uncertainty_initialization": "last 1x1 convolution zero initialized; other sigma-decoder parameters independent",
    "uv_objective": "0.5*(channel_mean((uv_pred-uv_target)^2)/variance+log(variance)); spatial image mean",
    "candidate_other_objectives": {"plv_l1": "unweighted channel-pixel L1", "plh_l1": "unweighted channel-pixel L1"},
    "joint_weights": None,
    "requires_explicit_joint_weights": True,
    "selection_metric": "raw_equal_modality_macro_l1=(l1_uv+l1_plv+l1_plh)/3",
    "checkpoint_loading": "none; this module does not load or initialize from existing checkpoints",
}


class FullResolutionDecoder(nn.Module):
    """One independent original-style four-level U-Net decoder."""

    def __init__(self, base_channels: int, out_channels: int, *, sigmoid: bool):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_channels * multiplier for multiplier in (1, 2, 4, 8, 16))
        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(c5, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(c4, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(c3, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(c2, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        d4 = self.decoder4(torch.cat([self.upconv4(bottleneck), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.output_activation(self.head(d1))


class JointUVUncertaintyUNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        c1, c2, c3, c4, c5 = (base_channels * multiplier for multiplier in (1, 2, 4, 8, 16))
        self.encoder1 = DoubleConv(3, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder_uv = FullResolutionDecoder(base_channels, 3, sigmoid=True)
        self.decoder_plv = FullResolutionDecoder(base_channels, 3, sigmoid=True)
        self.decoder_plh = FullResolutionDecoder(base_channels, 3, sigmoid=True)
        self.decoder_uv_uncertainty = FullResolutionDecoder(base_channels, 1, sigmoid=False)
        nn.init.zeros_(self.decoder_uv_uncertainty.head.weight)
        nn.init.zeros_(self.decoder_uv_uncertainty.head.bias)

    def forward(self, rgb):
        if (not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or rgb.shape[0] < 1
                or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 16
                or any(size % 16 for size in rgb.shape[-2:]) or not rgb.is_floating_point()):
            raise ValueError("RGB must be nonempty floating [B,3,H,W] with H,W divisible by 16")
        e1 = self.encoder1(rgb)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        skips = (e1, e2, e3, e4)
        uv = self.decoder_uv(bottleneck, skips).float()
        plv = self.decoder_plv(bottleneck, skips).float()
        plh = self.decoder_plh(bottleneck, skips).float()
        raw_s = self.decoder_uv_uncertainty(bottleneck, skips).float()
        log_variance = torch.logaddexp(raw_s, raw_s.new_tensor(math.log(VARIANCE_FLOOR)))
        return {"uv": uv, "plv": plv, "plh": plh,
                "raw_log_variance": raw_s, "log_variance": log_variance,
                "variance": torch.exp(log_variance), "sigma": torch.exp(0.5 * log_variance)}


def _validate_mean_targets(output, targets):
    if not isinstance(output, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("output and targets must be mappings with uv/plv/plh tensors")
    shape = None
    for modality in ("uv", "plv", "plh"):
        prediction, target = output[modality], targets[modality]
        for name, value in (("prediction", prediction), ("target", target)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] < 1
                    or value.shape[1] != 3 or min(value.shape[-2:]) < 1 or not value.is_floating_point()):
                raise ValueError(modality + " " + name + " must be floating nonempty [B,3,H,W]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(modality + " " + name + " is non-finite")
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(modality + " " + name + " is outside [0,1]; no implicit clipping")
        if prediction.shape != target.shape or prediction.device != target.device:
            raise ValueError(modality + " prediction/target shape or device mismatch")
        if shape is None:
            shape = prediction.shape
            device = prediction.device
        elif prediction.shape != shape or prediction.device != device:
            raise ValueError("all modalities must share B,H,W and device")
    return shape


def loss_components_per_image(output, targets):
    """Return [B] UV-NLL and candidate PLV/PLH L1; do not combine implicitly.

    Sigma is shared across RGB channels, so UV squared error is channel-mean
    before multiplication by inverse variance. The constant .5*log(2*pi) is
    omitted just as in the existing single-UV model. These components do not
    assert that PLV/PLH L1 or any particular joint weights are frozen.
    """
    shape = _validate_mean_targets(output, targets)
    log_variance = output["log_variance"]
    if (not isinstance(log_variance, torch.Tensor) or log_variance.shape != (shape[0], 1, shape[2], shape[3])
            or log_variance.device != output["uv"].device or not log_variance.is_floating_point()
            or not bool(torch.isfinite(log_variance).all())):
        raise ValueError("UV log_variance must be finite floating [B,1,H,W] on the image device")
    uv_squared = (output["uv"].float() - targets["uv"].float()).square().mean(1, keepdim=True)
    log_variance = log_variance.float()
    components = {
        "uv_nll": (0.5 * (uv_squared * torch.exp(-log_variance) + log_variance)).flatten(1).mean(1),
        "plv_l1": (output["plv"].float() - targets["plv"].float()).abs().flatten(1).mean(1),
        "plh_l1": (output["plh"].float() - targets["plh"].float()).abs().flatten(1).mean(1),
    }
    if any(not bool(torch.isfinite(value).all()) for value in components.values()):
        raise ValueError("non-finite loss arithmetic; no nan_to_num, clipping, or silent omission")
    return components


def validate_joint_weights(weights):
    """Require explicit finite nonnegative weights; there is no default recipe."""
    required = {"uv_nll", "plv_l1", "plh_l1"}
    if not isinstance(weights, Mapping) or set(weights) != required:
        raise ValueError("explicit weights must contain exactly uv_nll, plv_l1, plh_l1")
    validated = {}
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(name + " weight must be nonnegative finite real")
        validated[name] = float(value)
    if not any(value > 0 for value in validated.values()):
        raise ValueError("at least one objective weight must be positive")
    return validated


def weighted_total_loss_per_image(components, *, weights):
    """Return explicitly weighted [B] loss; trainer must still select reduction.

    Merely passing weights does not establish scientific approval of them.
    They must later be frozen in the reviewed training configuration.
    """
    validated = validate_joint_weights(weights)
    if not isinstance(components, Mapping) or set(components) != set(validated):
        raise ValueError("components must contain exactly uv_nll, plv_l1, plh_l1")
    shape, device = None, None
    for value in components.values():
        if (not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < 1
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError("loss components must be finite floating [B] tensors")
        if shape is None:
            shape, device = value.shape, value.device
        elif value.shape != shape or value.device != device:
            raise ValueError("loss component shape/device mismatch")
    total = sum(components[name] * validated[name] for name in validated)
    if not bool(torch.isfinite(total).all()):
        raise ValueError("weighted loss is non-finite")
    return total


def validation_per_image(output, targets):
    """Original raw three-modality equal-weight L1 for joint checkpoint choice."""
    _validate_mean_targets(output, targets)
    metrics = {"l1_" + name: (output[name].float() - targets[name].float()).abs().flatten(1).mean(1)
               for name in ("uv", "plv", "plh")}
    metrics["macro_l1"] = (metrics["l1_uv"] + metrics["l1_plv"] + metrics["l1_plh"]) / 3.0
    return metrics
