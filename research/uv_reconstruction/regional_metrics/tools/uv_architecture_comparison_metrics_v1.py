"""Model-independent, array-only metrics prepared for architecture comparisons.

This module does not read images/checkpoints, perform inference, train, choose
thresholds on evaluation results, or launch jobs. Historical masks and the SSIM
kernel are reused without changing their implementation. These functions being
available does NOT mean architecture comparison results have been measured.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tools.uv_brightspot_metrics_v1 import (
    PRIMARY_BRIGHT,
    PRIMARY_NONBRIGHT,
    SSIM_RADIUS,
    build_uv_region_masks,
    protocol_definition as historical_protocol_definition,
    ssim_map_rgb,
    validate_image,
)


GAUSSIAN_SIGMA_KIND = "gaussian_aleatoric_per_pixel_channel_shared_standard_deviation"
COVERAGE_Z = {"coverage68": 1.0, "coverage95": 1.959963984540054,
              "coverage99": 2.5758293035489004}


def protocol_definition() -> dict[str, Any]:
    historical = historical_protocol_definition()
    return {
        "version": "uv_architecture_comparison_metrics_v1",
        "state": "prepared_metrics_not_an_experiment_result",
        "input": "already preprocessed floating CHW RGB arrays in [0,1]; no internal resizing or clipping",
        "required_evaluation_spatial_shape": [1024, 1024],
        "preprocessing_external_contract": "same original center-square crop then Lanczos as locked evaluation",
        "historical_region_protocol": historical,
        "primary_region": "unchanged historical GT-UV luminance Otsu inside RGB nonblack foreground eroded 5 pixels",
        "V10_mask": "not used; may not substitute for the historical primary mask",
        "region_interpretation": "UV brightness proxy; no biological identity is established by this module",
        "whole_image": "all H*W pixels, including background; not just foreground",
        "errors": "per-image RGB-channel mean of absolute/squared error followed by spatial region mean",
        "psnr": "per-image per-region 10*log10(1/MSE), data_range=1; aggregate per-image PSNR, never convert aggregate MSE",
        "ssim": historical["ssim"],
        "regional_ssim": {
            "definition": "compute one unmasked full-image RGB-channel-mean SSIM map; average valid map entries whose centers are in ROI",
            "windows_may_cross_ROI_boundary": True,
            "zero_fill_or_flatten_ROI": False,
            "valid_centers": "exclude outer 5 pixels of image, no additional ROI erosion",
            "interpretation": "local structural similarity centered in ROI, not similarity using exclusively ROI pixels",
            "reporting": "newly available comparison statistic; not asserted to exist in previous architecture summaries",
        },
        "sigma": {
            "accepted_kind": GAUSSIAN_SIGMA_KIND,
            "layout": "HW, strictly positive finite standard deviation shared by three UV channels",
            "variance": "sigma**2 from model output; no extra variance floor or fitted rescaling",
            "standardized_residual": "z=(prediction-target)/sigma; mean and mean square over channels and region pixels",
            "coverage": "mean over channel-pixel indicators abs(z)<=threshold, not all-three-channels joint coverage",
            "z_thresholds": COVERAGE_Z,
            "nll": "mean(0.5*((prediction-target)**2/sigma**2+log(sigma**2))); Gaussian constant omitted",
            "not_accepted": "diffusion sampling diversity, residual sample std, epistemic uncertainty, or arbitrary uncertainty score",
            "limitation": "coverage near nominal alone does not establish calibration; complete-image includes background",
        },
        "undefined_statistics": "JSON null with explicit reason; zero-MSE PSNR is mathematically +infinity, not invalid model output",
        "aggregate_contract": "image-equal means of per-image values, each metric with valid/undefined counts; no pixel-area weighted substitution",
    }


def _finite_mean(values: np.ndarray, label: str) -> float:
    result = float(values.mean())
    if not math.isfinite(result):
        raise ValueError(label + ": non-finite metric arithmetic")
    return result


def _psnr(mse: float | None) -> float | None:
    if mse is None or mse == 0:
        return None
    # Preserve historical arithmetic for ordinary inputs. The algebraically
    # identical log form avoids reciprocal overflow for a positive subnormal
    # MSE, without clipping errors or changing the zero-error convention.
    if mse < np.finfo(np.float64).tiny:
        return -10.0 * math.log10(mse)
    return 10.0 * math.log10(1.0 / mse)


def _validate_sigma(sigma: np.ndarray, shape: tuple[int, int], kind: str | None) -> np.ndarray:
    if kind != GAUSSIAN_SIGMA_KIND:
        raise ValueError("sigma requires its explicit Gaussian aleatoric standard-deviation kind; diffusion diversity is not interchangeable")
    if (not isinstance(sigma, np.ndarray) or sigma.shape != shape
            or not np.issubdtype(sigma.dtype, np.floating)):
        raise ValueError("sigma must be a floating HW array matching the UV image")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and strictly positive")
    return sigma.astype(np.float64, copy=False)


def _gaussian_region(delta: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"status": "empty_region", "pixel_count": 0, "reason": "empty_region",
                **{key: None for key in ("sigma_mean", "nll", "standardized_residual_mean",
                                        "standardized_squared_error", *COVERAGE_Z)}}
    error = delta[:, mask]
    region_sigma = sigma[mask]
    # Do not sanitize an underflowed/overflowed variance or invalid prediction.
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            variance = np.square(region_sigma)
            if np.any(variance <= 0) or not np.isfinite(variance).all():
                raise ValueError("sigma squared is outside finite positive float64 range")
            standardized = error / region_sigma[None, :]
            squared = np.square(standardized)
            nll = 0.5 * (squared + np.log(variance)[None, :])
    except FloatingPointError as error:
        raise ValueError("non-finite Gaussian calibration arithmetic") from error
    return {
        "status": "ok", "pixel_count": int(mask.sum()), "reason": None,
        "sigma_mean": _finite_mean(region_sigma, "sigma_mean"),
        "nll": _finite_mean(nll, "nll"),
        "standardized_residual_mean": _finite_mean(standardized, "standardized_residual_mean"),
        "standardized_squared_error": _finite_mean(squared, "standardized_squared_error"),
        **{key: _finite_mean(np.abs(standardized) <= threshold, key)
           for key, threshold in COVERAGE_Z.items()},
    }


def evaluate_uv_prediction(
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    sigma: np.ndarray | None = None,
    sigma_kind: str | None = None,
    expected_spatial_shape: tuple[int, int] = (1024, 1024),
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen prediction, without doing inference or file I/O.

    The size override exists for numerical unit fixtures and must be recorded;
    production comparison callers must retain the locked (1024,1024) default.
    Missing sigma is an explicit unavailable measurement, not a zero estimate.
    Any future LDM sampling variability needs a separate protocol and cannot be
    passed as ``sigma`` here.
    """
    if (not isinstance(expected_spatial_shape, tuple) or len(expected_spatial_shape) != 2
            or any(not isinstance(n, int) or isinstance(n, bool) or n < 11 for n in expected_spatial_shape)):
        raise ValueError("expected_spatial_shape must contain two integers >= 11")
    if sample_id is not None and (not isinstance(sample_id, str) or not sample_id.strip()):
        raise ValueError("sample_id, if present, must be a nonempty string")
    rgb = validate_image(rgb, "RGB input")
    target = validate_image(target, "UV truth")
    prediction = validate_image(prediction, "UV prediction")
    if rgb.shape != target.shape or prediction.shape != target.shape:
        raise ValueError("RGB, truth and prediction shapes differ")
    if target.shape[1:] != expected_spatial_shape:
        raise ValueError("image spatial shape differs from the declared evaluation shape")
    if sigma is None and sigma_kind is not None:
        raise ValueError("sigma_kind provided without sigma")
    checked_sigma = None if sigma is None else _validate_sigma(sigma, target.shape[1:], sigma_kind)

    historical_masks, mask_metadata = build_uv_region_masks(rgb, target)
    masks = {"whole": np.ones(target.shape[1:], dtype=bool),
             **{key: historical_masks[key] for key in ("foreground", PRIMARY_BRIGHT, PRIMARY_NONBRIGHT)}}
    delta = prediction - target
    absolute, squared = np.abs(delta).mean(axis=0), np.square(delta).mean(axis=0)
    ssim = ssim_map_rgb(target, prediction)
    regions: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        center_mask = mask[SSIM_RADIUS:-SSIM_RADIUS, SSIM_RADIUS:-SSIM_RADIUS]
        center_count = int(center_mask.sum())
        l1 = _finite_mean(absolute[mask], name + " L1") if count else None
        mse = _finite_mean(squared[mask], name + " MSE") if count else None
        reason = "empty_region" if not count else ("zero_mse_psnr_positive_infinity" if mse == 0 else None)
        regions[name] = {
            "status": "ok" if count else "empty_region", "pixel_count": count,
            "l1": l1, "mse": mse,
            "psnr": _psnr(mse),
            "psnr_reason": reason,
            "ssim": _finite_mean(ssim[center_mask], name + " SSIM") if center_count else None,
            "ssim_reason": None if center_count else "no_valid_window_centers_in_region",
            "ssim_valid_center_count": center_count,
        }
    calibration = {"status": "unavailable", "reason": "model_does_not_supply_declared_gaussian_sigma",
                   "kind": None, "regions": None}
    if checked_sigma is not None:
        calibration = {"status": "measured", "reason": None, "kind": sigma_kind,
                       "regions": {name: _gaussian_region(delta, checked_sigma, mask)
                                   for name, mask in masks.items()}}
    return {"protocol_version": "uv_architecture_comparison_metrics_v1", "sample_id": sample_id,
            "spatial_shape": list(expected_spatial_shape),
            "locked_1024_shape": expected_spatial_shape == (1024, 1024),
            "regions": regions, "mask_metadata": mask_metadata, "gaussian_calibration": calibration}
