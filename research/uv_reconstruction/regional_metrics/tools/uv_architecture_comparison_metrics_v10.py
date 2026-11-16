"""Array-only architecture comparison using externally frozen native V10 weights.

No image loading, segmentation, Otsu threshold, inference or training occurs here.
The caller must supply weights from the approved native-image V10 projection and
persist its detector, mask and projection provenance separately. SSIM and Gaussian
calibration retain v1 definitions; only the regional spatial measure changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from tools.uv_architecture_comparison_metrics_v1 import (
    COVERAGE_Z, GAUSSIAN_SIGMA_KIND, _psnr, _validate_sigma,
)
from tools.uv_brightspot_metrics_v1 import (
    SSIM_RADIUS, protocol_definition as image_metric_protocol,
    ssim_map_rgb, validate_image,
)

SCHEMA = "uv_architecture_comparison_metrics_native_v10_area_weighted_v1"
REGIONS = ("whole", "foreground", "bright", "nonbright")


def protocol_definition() -> dict[str, Any]:
    return {
        "version": SCHEMA,
        "state": "prepared_metrics_not_an_experiment_result",
        "input": "target/prediction floating 3HW in [0,1]; no resizing or clipping",
        "required_evaluation_spatial_shape": [1024, 1024],
        "preprocessing_external_contract": "frozen original center-square crop then Lanczos",
        "region_source": "externally approved native V10 mask, BOX fractional-coverage projection",
        "mask_generation": "none; no Otsu or threshold selection inside this module",
        "caller_provenance_required": "detector version/hash, approval hash, native ROI/mask provenance, crop geometry, projection audit",
        "whole_image": "all H*W pixels including background, unit weights",
        "foreground": "frozen eroded RGB nonblack model-grid domain, binary 0/1 weights",
        "partition": "V10 bright + nonbright = frozen foreground; original projection weights are not binarized",
        "region_interpretation": "approved UV high-brightness/local-contrast proxy; no biological identity established here",
        "errors": "RGB-channel mean error then sum(pixel error * area weight)/sum(area weight)",
        "psnr": "per-image per-region 10*log10(1/MSE), data_range=1; never convert aggregate MSE",
        "ssim": image_metric_protocol()["ssim"],
        "regional_ssim": {
            "definition": "one unmasked full-image RGB-mean SSIM map; area-weighted mean by each valid window center's regional weight",
            "windows_may_cross_ROI_boundary": True,
            "valid_centers": "exclude outer 5 pixels, no further ROI erosion",
            "zero_fill_or_flatten_ROI": False,
        },
        "sigma": {
            "accepted_kind": GAUSSIAN_SIGMA_KIND,
            "layout": "positive finite HW standard deviation shared across UV RGB channels",
            "variance": "sigma**2; no floor or fitted rescaling",
            "standardized_residual": "z=(prediction-target)/sigma; channel mean then area-weighted regional mean",
            "coverage": "area-weighted channel-pixel indicator abs(z)<=threshold, not all-three-channels joint coverage",
            "z_thresholds": COVERAGE_Z,
            "nll": "area-weighted channel mean of 0.5*(z**2+log(sigma**2)); Gaussian constant omitted",
            "not_accepted": "diffusion sampling diversity, empirical residual std, epistemic score",
        },
        "undefined_statistics": "JSON null with explicit reason; zero-MSE PSNR means mathematical positive infinity",
        "aggregate_contract": "image-equal means of per-image values with valid/undefined counts; not pooled pixel-area means",
    }


def _weights(weights: Mapping[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping of precomputed V10 arrays")
    result = {"whole": np.ones(shape, dtype=np.float64)}
    for name in REGIONS[1:]:
        if name not in weights:
            raise ValueError("missing V10 weight: " + name)
        value = weights[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or not np.issubdtype(value.dtype, np.floating)):
            raise ValueError(name + " must be a floating HW weight array")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(name + " weights must be finite and within [0,1]")
        result[name] = value.astype(np.float64, copy=False)
    f, b, n = (result[key] for key in REGIONS[1:])
    if np.any((f != 0) & (f != 1)):
        raise ValueError("foreground must retain the frozen binary 0/1 evaluation domain")
    if np.any(b > f) or np.any(n > f):
        raise ValueError("V10 regional weights escape frozen foreground")
    if not np.allclose(b + n, f, rtol=0, atol=1e-8):
        raise ValueError("V10 bright and nonbright do not partition frozen foreground")
    return result


def _weighted_mean(value: np.ndarray, weight: np.ndarray, label: str) -> float | None:
    active = weight > 0
    if not active.any():
        return None
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            result = float(np.sum(value[active] * weight[active], dtype=np.float64)
                           / np.sum(weight[active], dtype=np.float64))
    except FloatingPointError as exc:
        raise ValueError(label + ": invalid weighted metric arithmetic") from exc
    if not math.isfinite(result):
        raise ValueError(label + ": non-finite weighted metric arithmetic")
    return result


def _gaussian_region(delta: np.ndarray, sigma: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    active = weight > 0
    result = {
        "status": "ok" if active.any() else "empty_region",
        "reason": None if active.any() else "empty_region",
        "positive_weight_cells": int(active.sum()),
        "effective_pixel_area": float(weight.sum(dtype=np.float64)),
    }
    keys = ("sigma_mean", "nll", "standardized_residual_mean",
            "standardized_squared_error", *COVERAGE_Z)
    if not active.any():
        return {**result, **{key: None for key in keys}}
    error, region_sigma, w = delta[:, active], sigma[active], weight[active]
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            variance = np.square(region_sigma)
            if np.any(variance <= 0) or not np.isfinite(variance).all():
                raise ValueError("sigma squared is outside finite positive float64 range")
            z = error / region_sigma[None, :]
            z2 = np.square(z)
            nll = .5 * (z2 + np.log(variance)[None, :])
            arrays = {
                "sigma_mean": region_sigma,
                "nll": nll.mean(axis=0),
                "standardized_residual_mean": z.mean(axis=0),
                "standardized_squared_error": z2.mean(axis=0),
                **{key: (np.abs(z) <= threshold).mean(axis=0)
                   for key, threshold in COVERAGE_Z.items()},
            }
    except FloatingPointError as exc:
        raise ValueError("non-finite Gaussian calibration arithmetic") from exc
    return {**result, **{key: _weighted_mean(value, w, key) for key, value in arrays.items()}}


def evaluate_uv_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: Mapping[str, np.ndarray],
    *,
    sigma: np.ndarray | None = None,
    sigma_kind: str | None = None,
    expected_spatial_shape: tuple[int, int] = (1024, 1024),
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate frozen UV output using pre-generated V10 fractional region weights.

    Shape override is for synthetic numerical fixtures only and is recorded.
    Absence of Gaussian sigma is unavailable, never zero uncertainty; diffusion
    diversity must be measured using its own distinct protocol.
    """
    if (not isinstance(expected_spatial_shape, tuple) or len(expected_spatial_shape) != 2
            or any(not isinstance(n, int) or isinstance(n, bool) or n < 11
                   for n in expected_spatial_shape)):
        raise ValueError("expected_spatial_shape must contain two integers >= 11")
    if sample_id is not None and (not isinstance(sample_id, str) or not sample_id.strip()):
        raise ValueError("sample_id must be a nonempty string when present")
    target, prediction = validate_image(target, "UV truth"), validate_image(prediction, "UV prediction")
    if target.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    if target.shape[1:] != expected_spatial_shape:
        raise ValueError("image spatial shape differs from declared evaluation shape")
    checked_weights = _weights(weights, expected_spatial_shape)
    if sigma is None and sigma_kind is not None:
        raise ValueError("sigma_kind provided without sigma")
    checked_sigma = None if sigma is None else _validate_sigma(sigma, expected_spatial_shape, sigma_kind)
    delta = prediction - target
    absolute, squared = np.abs(delta).mean(axis=0), np.square(delta).mean(axis=0)
    ssim = ssim_map_rgb(target, prediction)
    regions = {}
    for name, weight in checked_weights.items():
        area = float(weight.sum(dtype=np.float64))
        center_weight = weight[SSIM_RADIUS:-SSIM_RADIUS, SSIM_RADIUS:-SSIM_RADIUS]
        center_area = float(center_weight.sum(dtype=np.float64))
        mse = _weighted_mean(squared, weight, name + " MSE")
        regions[name] = {
            "status": "ok" if area else "empty_region",
            "positive_weight_cells": int(np.count_nonzero(weight)),
            "effective_pixel_area": area,
            "l1": _weighted_mean(absolute, weight, name + " L1"),
            "mse": mse,
            "psnr": _psnr(mse),
            "psnr_reason": "empty_region" if mse is None else "zero_mse_psnr_positive_infinity" if mse == 0 else None,
            "ssim": _weighted_mean(ssim, center_weight, name + " SSIM"),
            "ssim_reason": None if center_area else "no_valid_window_centers_in_region",
            "ssim_valid_center_positive_weight_cells": int(np.count_nonzero(center_weight)),
            "ssim_valid_center_effective_area": center_area,
        }
    calibration = {"status": "unavailable", "reason": "model_does_not_supply_declared_gaussian_sigma",
                   "kind": None, "regions": None}
    if checked_sigma is not None:
        calibration = {"status": "measured", "reason": None, "kind": sigma_kind,
                       "regions": {name: _gaussian_region(delta, checked_sigma, weight)
                                   for name, weight in checked_weights.items()}}
    return {"protocol_version": SCHEMA, "sample_id": sample_id,
            "spatial_shape": list(expected_spatial_shape),
            "locked_1024_shape": expected_spatial_shape == (1024, 1024),
            "region_measure": "V10_fractional_area_weighted",
            "regions": regions, "gaussian_calibration": calibration}
