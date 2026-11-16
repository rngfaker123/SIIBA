"""Single-target RGB -> UV mean and per-pixel RGB residual sigma.

This is a heteroscedastic Gaussian U-Net, NOT a diffusion model and NOT
epistemic uncertainty.  Its mean branch preserves the independent U-Net.
The variance floor is a numerical regularizer in [0, 1] image units.
"""
import math

import torch
from torch import nn

from models.unet_baseline import UNetBaseline

VARIANCE_FLOOR = 1e-6
EXPECTED_PARAMETER_COUNT = 7763140


class UVUncertaintyUNet(UNetBaseline):
    def __init__(self, base_channels=32):
        # Construct the entire original mean branch first: at the same RNG
        # state its initial weights are exactly those of UNetBaseline.
        super().__init__(in_channels=3, out_channels=3,
                         base_channels=base_channels)
        self.log_variance_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def forward(self, x):
        if self.training and self.activation_checkpointing:
            raise RuntimeError("Activation checkpointing is not in this frozen protocol")
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        mean = self.output_activation(self.head(d1)).float()
        raw_s = self.log_variance_head(d1).float()
        # FP32 log(exp(s) + floor), without directly exponentiating raw_s.
        log_variance = torch.logaddexp(
            raw_s, raw_s.new_tensor(math.log(VARIANCE_FLOOR)))
        return {"mean": mean, "raw_log_variance": raw_s,
                "log_variance": log_variance,
                "variance": torch.exp(log_variance),
                "sigma": torch.exp(0.5 * log_variance)}


def gaussian_nll_per_image(output, target):
    """Mean over RGB and pixels, omitting the fixed 0.5*log(2*pi).

One sigma is shared across the three UV-image color channels at each pixel.
This sigma is not the standard deviation of UV luminance.
"""
    err2 = (output["mean"].float() - target.float()).square().mean(1, keepdim=True)
    logvar = output["log_variance"].float()
    return (0.5 * (err2 * torch.exp(-logvar) + logvar)).flatten(1).mean(1)


def validation_per_image(output, target, rgb=None):
    """Image-equal full-image summaries; sigma calibration is descriptive.

rgb is accepted for trainer compatibility but does not change the full-image
population. Region-specific calibration is a separate, frozen analysis.
"""
    error = output["mean"].float() - target.float()
    sigma = output["sigma"].float()
    z = error / sigma
    mean = lambda value: value.flatten(1).mean(1)
    return {
        "nll": gaussian_nll_per_image(output, target),
        "l1": mean(error.abs()),
        "mse": mean(error.square()),
        "sigma_mean": mean(sigma),
        "standardized_squared_error": mean(z.square()),
        "standardized_residual_mean": mean(z),
        "coverage68": mean((z.abs() <= 1.0).float()),
        "coverage95": mean((z.abs() <= 1.959963984540054).float()),
        "coverage99": mean((z.abs() <= 2.5758293035489004).float()),
        "near_variance_floor_fraction": mean(
            (output["variance"] <= 2 * VARIANCE_FLOOR).float()),
    }
