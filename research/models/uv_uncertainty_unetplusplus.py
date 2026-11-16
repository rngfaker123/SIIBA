"""Untrained U-Net++-style UV mean/sigma candidate; NOT a frozen experiment.

Independent nested dense-skip implementation inspired by Zhou et al.,
https://arxiv.org/abs/1807.10165, not the authors' official implementation.
It reuses this project's original DoubleConv (GroupNorm/ReLU), four pooling
levels and widths 32/64/128/256/512. Each nested edge has its own learned
2x2 stride-2 transposed convolution, reducing C[i+1] to C[i] before concat.
No deep supervision, pretrained weights, dropout, teacher, or diffusion.

The final X(0,4) has UV sigmoid-mean and zero-initialized log-variance heads.
Sigma has precisely the existing single-UV Gaussian residual-scale meaning;
this is neither a latent diffusion model nor the professor's three-output
probabilistic architecture. Config freezing, trainer/resource preflight,
checkpoint replay, and launch authorization are still separate pending gates.
"""

import math

import torch
from torch import nn

from models.unet_baseline import DoubleConv
from models.uv_uncertainty_unet import VARIANCE_FLOOR, gaussian_nll_per_image


ARCHITECTURE_ID = "nested_unetplusplus_uv_aleatoric_candidate_v1"
PREPARATION_STATE = "candidate_not_frozen"
CHANNEL_MULTIPLIERS = (1, 2, 4, 8, 16)
DEEP_SUPERVISION = False


def nested_topology(base_channels=32):
    """Pure-Python topological specification, also used to build the network.

    Encoders X(i,0) are ordered first, then columns j=1..4. For j>0,
    X(i,j)=DoubleConv(cat(X(i,0),...,X(i,j-1),Up(X(i+1,j-1)))).
    Up changes the parent channel width to the receiving row's width.
    """
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
        raise ValueError("base_channels must be a positive integer")
    channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8, 16)]
    nodes = []
    for i in range(5):
        nodes.append({"name": "x%d_0" % i, "row": i, "column": 0,
                      "out_channels": channels[i],
                      "in_channels": 3 if i == 0 else channels[i - 1],
                      "same_row_parents": [], "up_parent": None,
                      "pool_parent": None if i == 0 else "x%d_0" % (i - 1)})
    for j in range(1, 5):
        for i in range(5 - j):
            nodes.append({"name": "x%d_%d" % (i, j), "row": i, "column": j,
                          "out_channels": channels[i], "in_channels": (j + 1) * channels[i],
                          "same_row_parents": ["x%d_%d" % (i, k) for k in range(j)],
                          "up_parent": "x%d_%d" % (i + 1, j - 1),
                          "pool_parent": None})
    return nodes


class UVUncertaintyUNetPlusPlus(nn.Module):
    """RGB BCHW -> UV mean B3HW and shared Gaussian sigma B1HW.

    base_channels=32 is the proposed comparison candidate. Other widths exist
    solely to permit small numerical tests or explicitly separate candidates;
    they must not silently replace the proposed production configuration.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        self.topology = nested_topology(base_channels)
        self.base_channels = base_channels
        self.blocks = nn.ModuleDict()
        self.upsampling = nn.ModuleDict()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        by_name = {node["name"]: node for node in self.topology}
        for node in self.topology:
            self.blocks[node["name"]] = DoubleConv(node["in_channels"], node["out_channels"])
            if node["up_parent"] is not None:
                parent_channels = by_name[node["up_parent"]]["out_channels"]
                self.upsampling[node["name"]] = nn.ConvTranspose2d(
                    parent_channels, node["out_channels"], kernel_size=2, stride=2, bias=True)
        self.mean_head = nn.Conv2d(base_channels, 3, kernel_size=1, bias=True)
        self.log_variance_head = nn.Conv2d(base_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def architecture_metadata(self):
        return {
            "architecture_id": ARCHITECTURE_ID,
            "state": PREPARATION_STATE,
            "implementation": "independent_U_Net_plus_plus_style_not_official",
            "reference": "https://arxiv.org/abs/1807.10165",
            "task": "RGB_to_UV_mean_and_Gaussian_aleatoric_sigma",
            "base_channels": self.base_channels,
            "channels": [self.base_channels * multiplier for multiplier in CHANNEL_MULTIPLIERS],
            "downsampling_levels": 4,
            "double_conv": "original_bias_false_3x3_GroupNorm_up_to8_ReLU_twice",
            "upsampling": "independent_ConvTranspose2d_2x2_stride2_bias_true_Ci_plus1_to_Ci_per_nested_edge",
            "deep_supervision": DEEP_SUPERVISION,
            "head_source": "x0_4",
            "mean": "sigmoid_3_channels",
            "sigma": "sqrt(exp(raw_log_variance)+1e-6)_one_channel_shared_by_UV_RGB",
            "variance_floor": VARIANCE_FLOOR,
            "loss": "reuse_src.uv_uncertainty_unet.gaussian_nll_per_image",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()
                                        if parameter.requires_grad),
            "topology": [dict(node) for node in self.topology],
            "initialization": "random_from_scratch_zero_log_variance_head",
            "budget_note": "same_widths_do_not_mean_equal_parameters_FLOPs_or_only_a_depth_change",
            "pending": ["config_and_protocol_freeze", "trainer_integration",
                        "real_train_val_resource_preflight", "checkpoint_resume_replay", "explicit_launch"],
        }

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Expected RGB input with shape [B,3,H,W]")
        if rgb.shape[0] < 1 or min(rgb.shape[-2:]) < 16 or any(side % 16 for side in rgb.shape[-2:]):
            raise ValueError("Input batch must be nonempty and H,W >=16 divisible by16; no implicit resize or pad")
        if not rgb.is_floating_point():
            raise ValueError("RGB input must be floating point")
        features = {}
        for node in self.topology:
            if node["column"] == 0:
                source = rgb if node["pool_parent"] is None else self.pool(features[node["pool_parent"]])
            else:
                source = torch.cat(
                    [features[parent] for parent in node["same_row_parents"]]
                    + [self.upsampling[node["name"]](features[node["up_parent"]])], dim=1)
            features[node["name"]] = self.blocks[node["name"]](source)
        final_features = features["x0_4"]
        mean = torch.sigmoid(self.mean_head(final_features)).float()
        raw_s = self.log_variance_head(final_features).float()
        log_variance = torch.logaddexp(raw_s, raw_s.new_tensor(math.log(VARIANCE_FLOOR)))
        return {"mean": mean, "raw_log_variance": raw_s, "log_variance": log_variance,
                "variance": torch.exp(log_variance), "sigma": torch.exp(0.5 * log_variance)}
