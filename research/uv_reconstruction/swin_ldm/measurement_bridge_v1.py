"""Draft one-forward bridge to unchanged frozen TEST measurement functions.

Preparation only: no CLI, model loader, source import, file access, ownership or
report finalizer. The controller must independently bind every injected helper,
input array, row, mask and runtime before calling this module. Importing this
module does not import torch or execute a scientific measurement.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


SHAPE = (1024, 1024)
MASK_DOMAINS = frozenset(("P", "D", "O", "F"))
ANATOMICAL_ROIS = frozenset(("forehead", "image_left_cheek", "image_right_cheek",
                            "nose", "chin", "F_reference"))
TECHNICAL_STATUSES = frozenset(("no_unique_face", "invalid_geometry"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _row_identity(row):
    require(isinstance(row, dict) and row.get("split") == "test", "Original TEST row required")
    require(all(isinstance(row.get(key), str) and bool(row[key]) for key in ("sample_id", "subject_id")),
            "Original nonempty sample/subject ID strings required")
    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")
    return dict(sample_id=row["sample_id"], subject_id=row["subject_id"],
                source_row_sha256=hashlib.sha256(encoded).hexdigest())


def _image(array, label):
    require(isinstance(array, np.ndarray) and array.dtype == np.float32
            and array.shape == (3, *SHAPE), label + " must be float32 CHW RGB1024")
    require(bool(np.isfinite(array).all()) and bool(((array >= 0) & (array <= 1)).all()),
            label + " must preserve finite encoded image intensities in [0,1]")


def _pair(prediction, sigma):
    _image(prediction, "Mean")
    require(isinstance(sigma, np.ndarray) and sigma.dtype == np.float32 and sigma.shape == SHAPE,
            "Sigma must be float32 HW1024, not variance/log variance or RGB channels")
    require(bool(np.isfinite(sigma).all()) and bool((sigma > 0).all()), "Sigma must be finite and positive")


@dataclass(frozen=True)
class FrozenMetricHelpers:
    """Already source-hash-verified modules; this class does not import or attest them."""
    old: object
    combined: object
    regional_adapter: object
    regional_runner: object
    numeric: object
    rank: object


class TorchTensorIO:
    """Future execution converter; tests inject a fake instead of importing torch.

    The selected device must already be UUID-bound by the new controller. Actual
    loader_adapter_v1.PredictionAdapter independently checks its input device.
    """
    def __init__(self, device):
        self.device = device
        self._input_device = None

    def as_input(self, rgb):
        import torch
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        self._input_device = tensor.device
        return tensor

    def prediction_pair(self, output):
        import torch
        require(isinstance(output, dict), "Source output dictionary required")
        mean, sigma = output.get("mean"), output.get("sigma")
        for value, shape, label in ((mean, (1, 3, *SHAPE), "mean"),
                                    (sigma, (1, 1, *SHAPE), "sigma")):
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                    and tuple(value.shape) == shape, "Invalid original source tensor " + label)
            require(not value.requires_grad, "Frozen inference tensor must not require gradients")
        require(self._input_device is not None and mean.device == sigma.device == self._input_device,
                "Source output device differs from the actual input device")
        # 直接提取均值与σ
        return mean[0].float().cpu().numpy(), sigma[0, 0].float().cpu().numpy()


def measure_one(adapter, tensor_io, helpers, frozen_row, rgb, truth, weights, rois, roi_audit):
    """One loader-adapter predict call, same extracted arrays passed to all helpers.

    rgb/truth are already preprocessed, descriptor-bound arrays. Geometry must be
    prepared by the unchanged regional.prepare_geometry/runner.roi_weights path.
    This function is for a valid published mask; use unavailable_measurements for
    an original technical status. Exceptions propagate; no partial success record
    is returned. The return value is an unsealed payload, not a production record.
    """
    identity = _row_identity(frozen_row)
    _image(rgb, "RGB input")
    _image(truth, "UV truth")
    require(isinstance(weights, dict) and set(weights) == MASK_DOMAINS, "Exactly original P/D/O/F weights required")
    require(isinstance(rois, dict) and set(rois) == ANATOMICAL_ROIS, "Exactly original six anatomical ROIs required")
    require(isinstance(roi_audit, dict) and isinstance(roi_audit.get("landmark_source"), str)
            and bool(roi_audit["landmark_source"]), "Original landmark-source audit required")
    # 模型只读RGB和索引行
    output = adapter.predict(tensor_io.as_input(rgb), frozen_row)
    require(_row_identity(frozen_row) == identity, "Adapter mutated the original TEST row")
    prediction, sigma = tensor_io.prediction_pair(output)
    _pair(prediction, sigma)
    regions = helpers.old.measure(truth, prediction, sigma, weights, helpers.numeric)
    table4 = helpers.combined.table4_measure(truth, prediction, sigma, weights,
                                           helpers.old, helpers.numeric, helpers.rank)
    regional = helpers.regional_adapter.measure_same_forward(helpers.regional_runner,
                                                            truth, prediction, rois, roi_audit)
    require(_row_identity(frozen_row) == identity, "Measurement helper mutated the original TEST row")
    return dict(status="complete", **identity, forward_count=1,
                combined=dict(regions=regions, table4=table4), regional=regional)


def unavailable_measurements(helpers, frozen_row, mask_status):
    """Preserve an original technical case with no model/tensor input or forward."""
    identity = _row_identity(frozen_row)
    require(mask_status in TECHNICAL_STATUSES, "Only original unavailable geometry statuses are supported")
    reason = "mask_" + mask_status
    regions = helpers.old.unavailable_regions(reason)
    table4 = helpers.combined.unavailable_table4(reason)
    regional = helpers.regional_adapter.unavailable(reason)
    require(_row_identity(frozen_row) == identity, "Technical helper mutated the original TEST row")
    return dict(status="unavailable_geometry", **identity, forward_count=0,
                combined=dict(regions=regions, table4=table4), regional=regional)
