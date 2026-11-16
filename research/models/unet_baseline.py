# -*- coding: utf-8 -*-
"""U-Net baseline, architecture-locked to 7,763,107 parameters.

Structure (must match 15_overfit_unet.py):
  DoubleConv(i, o) = Conv3x3(i, o, bias=False) - GroupNorm - ReLU
                     Conv3x3(o, o, bias=False) - GroupNorm - ReLU
  Encoder : 3 -> 32 -> 64 -> 128 -> 256, bottleneck 512 (MaxPool2d(2))
  Decoder : ConvTranspose2d(2x, bias=True) + skip concat + DoubleConv
  Head    : Conv1x1(32, 3, bias=True) + Sigmoid
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


EXPECTED_PARAMETER_COUNT = 7763107


class ParameterCountMismatch(RuntimeError):
    pass


def _group_norm(num_channels, max_groups=8):
    groups = max_groups
    while groups > 1 and num_channels % groups != 0:
        groups = groups - 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(UNetBaseline, self).__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.encoder1 = DoubleConv(in_channels, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(c5, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(c4, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(c3, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(c2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()
        self.activation_checkpointing = False

    def set_activation_checkpointing(self, enabled=True):
        """Trade recomputation for memory without changing learned parameters."""
        self.activation_checkpointing = bool(enabled)

    def _checkpointed_forward(self, x):
        # The server's PyTorch checkpoint API is reentrant and requires an input
        # with gradients enabled. The input gradient itself is not used.
        if not x.requires_grad:
            x = x.requires_grad_(True)
        e1 = checkpoint(self.encoder1, x)
        e2 = checkpoint(self.encoder2, self.pool(e1))
        e3 = checkpoint(self.encoder3, self.pool(e2))
        e4 = checkpoint(self.encoder4, self.pool(e3))
        b = checkpoint(self.bottleneck, self.pool(e4))

        d4 = checkpoint(self.decoder4, torch.cat([self.upconv4(b), e4], dim=1))
        d3 = checkpoint(self.decoder3, torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = checkpoint(self.decoder2, torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = checkpoint(self.decoder1, torch.cat([self.upconv1(d2), e1], dim=1))
        return self.output_activation(self.head(d1))

    def forward(self, x):
        if self.training and self.activation_checkpointing:
            return self._checkpointed_forward(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.output_activation(self.head(d1))


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_unet(model_config, strict_parameter_check=True):
    """Build the U-Net and hard-fail if the parameter count drifted."""
    model = UNetBaseline(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 3)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    expected = int(model_config.get("expected_parameter_count", EXPECTED_PARAMETER_COUNT))
    actual = count_parameters(model)
    if strict_parameter_check and actual != expected:
        raise ParameterCountMismatch(
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
