"""CPU-only, explicitly defined UV bright-region diagnostics.

All image arguments are floating numpy arrays with shape [3, H, W] in [0, 1].
Masks are derived from RGB and ground truth only, never from predictions.  The
foreground is a nonblack proxy, not a skin segmentation; UV brightness is not a
validated porphyrin measurement.  Array-returning helpers are marked as such;
all metric/metadata results contain JSON-safe native values and no NaN/Infinity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import ndimage


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_RADIUS = SSIM_WINDOW // 2
PRIMARY_BRIGHT = "primary_bright"
PRIMARY_NONBRIGHT = "primary_nonbright"
LOCAL_BRIGHT = "local_bright"
LOCAL_NONBRIGHT = "local_nonbright"


def protocol_definition() -> dict[str, Any]:
    """Return the exact algorithm settings, suitable for a frozen JSON record."""
    return {
        "version": "uv_brightspot_metrics_v1",
        "image_layout": "CHW_RGB",
        "image_dtype": "floating numpy array; float64 metric arithmetic",
        "image_range": [0.0, 1.0],
        "foreground": {
            "name": "nonblack_foreground_proxy",
            "is_skin_mask": False,
            "threshold": 8.0 / 255.0,
            "comparison": "max(R,G,B) > threshold",
            "erosion": "5 iterations of 3x3 all-ones structure; border_value=0",
            "erosion_radius_px": 5,
        },
        "uv_luminance": {"coefficients": list(LUMA_WEIGHTS), "linearize_srgb": False},
        "primary_region": {
            "name": "uv_bright_intensity_proxy",
            "source": "ground-truth UV luminance inside foreground",
            "method": "Otsu between-class variance, fixed histogram",
            "histogram_bins": 256,
            "histogram_range": [0.0, 1.0],
            "threshold": "center of earliest maximizing histogram bin",
            "comparison": "luminance > threshold",
            "degenerate_behavior": "threshold null; both primary masks empty; explicit status",
        },
        "local_contrast_sensitivity": {
            "definition": "Y - gaussian_filter(Y, sigma=5, mode=reflect, truncate=4)",
            "sigma": 5.0,
            "gaussian_support_radius_px": 20,
            "domain": "raw RGB nonblack foreground eroded by 20 pixels (41x41 square support)",
            "domain_erosion": "20 iterations of 3x3 all-ones structure; border_value=0",
            "percentile": 95.0,
            "percentile_method": "linear",
            "comparison": "local_contrast > local_domain percentile, inside local_domain only",
            "is_primary": False,
        },
        "local_contrast_luminance_strata": {
            "edges": [i / 10.0 for i in range(11)],
            "intervals": "[left,right), except last includes 1",
            "source": "ground-truth UV luminance, independent of prediction",
        },
        "error": "per-pixel RGB channel-mean absolute/squared error, then masked spatial mean",
        "psnr": {"data_range": 1.0, "definition": "10*log10(1/regional_mse)",
                 "zero_mse": "null with reason zero_mse_psnr_positive_infinity"},
        "ssim": {
            "data_range": 1.0,
            "window_size": SSIM_WINDOW,
            "gaussian_sigma": SSIM_SIGMA,
            "gaussian_truncation": "explicit normalized 11-tap kernel",
            "K1": 0.01,
            "K2": 0.03,
            "covariance": "population",
            "channel_aggregation": "RGB channel mean of SSIM map",
            "padding": "none in retained valid map",
            "valid_map_shape": "(H-10,W-10)",
            "regional_center_ssim": "mean valid SSIM whose window center belongs to region",
            "regional_full_window_ssim": "mean valid SSIM whose whole 11x11 window belongs to region",
        },
        "bootstrap": {
            "iterations": 2000, "seed": 20260817,
            "resampling_unit": "subject if independently marked reliable; otherwise image",
            "estimand": "image-equal-weight mean paired candidate-reference difference",
            "interval": "two-sided percentile, numpy quantile(method=linear), [0.025,0.975]",
            "one_cluster": "CI unavailable; no image or normal-approximation fallback",
        },
    }


def validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate strict CHW/[0,1]/finite input and return float64 array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name}: expected a numpy array")
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError(f"{name}: expected shape [3,H,W]; received {image.shape}")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(f"{name}: expected floating dtype, received {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"{name}: non-finite image values")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError(f"{name}: values outside [0,1]")
    return image.astype(np.float64, copy=False)


def luminance(image_chw: np.ndarray) -> np.ndarray:
    """Return HxW luminance array, without nonlinear colorspace conversion."""
    image = validate_image(image_chw)
    return np.einsum("c,chw->hw", np.asarray(LUMA_WEIGHTS), image)


def _mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(f"{name}: expected boolean mask of shape {shape}")
    return mask


def _otsu(values: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    if values.size == 0:
        return None, {"status": "empty_foreground", "occupied_bins": 0}
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    occupied = int(np.count_nonzero(histogram))
    if occupied < 2:
        return None, {"status": "no_histogram_separation", "occupied_bins": occupied}
    probability = histogram.astype(np.float64) / float(values.size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative_weight = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * centers)
    total_mean = float(cumulative_mean[-1])
    valid = (cumulative_weight > 0.0) & (cumulative_weight < 1.0)
    variance = np.zeros(256, dtype=np.float64)
    variance[valid] = (
        (total_mean * cumulative_weight[valid] - cumulative_mean[valid]) ** 2
        / (cumulative_weight[valid] * (1.0 - cumulative_weight[valid]))
    )
    if not np.isfinite(variance).all():
        raise ValueError("Otsu between-class variance is non-finite")
    best = int(np.argmax(variance))
    if not valid[best] or float(variance[best]) <= 0.0:
        return None, {"status": "no_between_class_variance", "occupied_bins": occupied}
    return float(centers[best]), {
        "status": "ok", "occupied_bins": occupied, "threshold_bin": best,
        "between_class_variance": float(variance[best]),
    }


def build_uv_region_masks(
    rgb_chw: np.ndarray, uv_chw: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return boolean HxW masks and separately JSON-safe metadata.

    Every mask is based on RGB or UV ground truth.  Degenerate Otsu images have
    both primary masks empty and an explicit unclassified foreground count.
    Sensitivity strata cannot substitute for the primary masks.
    """
    rgb = validate_image(rgb_chw, "RGB")
    uv = validate_image(uv_chw, "UV ground truth")
    if rgb.shape != uv.shape:
        raise ValueError("RGB and UV shapes differ")
    initial_foreground = np.max(rgb, axis=0) > (8.0 / 255.0)
    foreground = ndimage.binary_erosion(
        initial_foreground, structure=np.ones((3, 3), dtype=bool),
        iterations=5, border_value=0,
    )
    y = luminance(uv)
    threshold, primary_metadata = _otsu(y[foreground])
    bright = np.zeros(foreground.shape, dtype=bool)
    nonbright = np.zeros(foreground.shape, dtype=bool)
    if threshold is not None:
        bright = foreground & (y > threshold)
        nonbright = foreground & ~bright
        if not bright.any():
            primary_metadata["status"] = "empty_bright_region"
        elif not nonbright.any():
            primary_metadata["status"] = "empty_nonbright_region"
    local_contrast = y - ndimage.gaussian_filter(y, sigma=5.0, mode="reflect", truncate=4.0)
    if not np.isfinite(local_contrast).all():
        raise ValueError("local contrast contains non-finite values")
    # Every retained Gaussian footprint must stay inside the raw nonblack RGB
    # proxy, so black-background boundaries cannot create a local bright rim.
    # The Otsu/primary foreground still uses its independently frozen 5px erosion.
    local_domain = ndimage.binary_erosion(
        initial_foreground, structure=np.ones((3, 3), dtype=bool),
        iterations=20, border_value=0,
    )
    local_bright = np.zeros(foreground.shape, dtype=bool)
    local_nonbright = np.zeros(foreground.shape, dtype=bool)
    local_threshold: float | None = None
    local_status = "empty_local_domain"
    if local_domain.any():
        # Linear interpolation is the NumPy default both before and after 1.22;
        # omitting the renamed keyword preserves identical numeric semantics.
        local_threshold = float(np.percentile(local_contrast[local_domain], 95.0))
        local_bright = local_domain & (local_contrast > local_threshold)
        local_nonbright = local_domain & ~local_bright
        local_status = "ok" if local_bright.any() and local_nonbright.any() else "empty_local_region"
    masks = {
        "foreground": foreground,
        PRIMARY_BRIGHT: bright,
        PRIMARY_NONBRIGHT: nonbright,
        "local_domain": local_domain,
        LOCAL_BRIGHT: local_bright,
        LOCAL_NONBRIGHT: local_nonbright,
    }
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        in_stratum = (y >= lower) & ((y < upper) if index < 9 else (y <= upper))
        masks[f"luminance_{index:02d}_local_bright"] = local_bright & in_stratum
        masks[f"luminance_{index:02d}_local_nonbright"] = local_nonbright & in_stratum
    metadata = {
        "foreground_name": "nonblack_foreground_proxy", "is_skin_mask": False,
        "image_shape_chw": [int(v) for v in rgb.shape],
        "initial_foreground_pixels": int(initial_foreground.sum()),
        "foreground_pixels": int(foreground.sum()),
        "primary": dict(primary_metadata, threshold=threshold, proxy_name="uv_bright_intensity_proxy"),
        "primary_unclassified_foreground_pixels": int((foreground & ~(bright | nonbright)).sum()),
        "local_contrast_sensitivity": {
            "status": local_status, "threshold": local_threshold,
            "domain_name": "nonblack_foreground_proxy_eroded20",
            "domain_pixels": int(local_domain.sum()),
            "domain_erosion_radius_px": 20,
            "gaussian_support_radius_px": 20,
            "excluded_primary_foreground_pixels": int((foreground & ~local_domain).sum()),
        },
        "pixel_counts": {key: int(mask.sum()) for key, mask in masks.items()},
    }
    return masks, metadata


def ssim_map_rgb(target_chw: np.ndarray, prediction_chw: np.ndarray) -> np.ndarray:
    """Return valid (H-10)x(W-10) RGB-mean SSIM map from full images.

    No background masking, zero filling or regional flattening is performed.
    This reproduces an explicit normalized 11x11 Gaussian/population-covariance
    definition, rather than delegating to library-dependent default settings.
    """
    target = validate_image(target_chw, "target")
    prediction = validate_image(prediction_chw, "prediction")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    if min(target.shape[1:]) < SSIM_WINDOW:
        raise ValueError("SSIM requires H and W >= 11")
    coordinates = np.arange(-SSIM_RADIUS, SSIM_RADIUS + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates ** 2) / (2.0 * SSIM_SIGMA ** 2))
    kernel /= kernel.sum()

    def smooth(value: np.ndarray) -> np.ndarray:
        result = ndimage.convolve1d(value, kernel, axis=-1, mode="constant", cval=0.0)
        result = ndimage.convolve1d(result, kernel, axis=-2, mode="constant", cval=0.0)
        return result[:, SSIM_RADIUS:-SSIM_RADIUS, SSIM_RADIUS:-SSIM_RADIUS]

    mean_t, mean_p = smooth(target), smooth(prediction)
    # Population variances may be slightly negative from floating cancellation.
    # This is an arithmetic correction of variance, never clipping model outputs.
    variance_t = np.maximum(smooth(target * target) - mean_t * mean_t, 0.0)
    variance_p = np.maximum(smooth(prediction * prediction) - mean_p * mean_p, 0.0)
    covariance = smooth(target * prediction) - mean_t * mean_p
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    result = (
        ((2.0 * mean_t * mean_p + c1) * (2.0 * covariance + c2))
        / ((mean_t ** 2 + mean_p ** 2 + c1) * (variance_t + variance_p + c2))
    ).mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError("SSIM map contains non-finite values")
    return result


def _mean_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    result = float(values[mask].mean())
    if not math.isfinite(result):
        raise ValueError("regional mean is non-finite")
    return result


def _ssim_region_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = mask[SSIM_RADIUS:-SSIM_RADIUS, SSIM_RADIUS:-SSIM_RADIUS]
    full = ndimage.minimum_filter(mask, size=SSIM_WINDOW, mode="constant", cval=0)
    full = full[SSIM_RADIUS:-SSIM_RADIUS, SSIM_RADIUS:-SSIM_RADIUS]
    return center, full


def evaluate_regions(
    target_chw: np.ndarray,
    prediction_chw: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate all fixed masks using one SSIM map; return only native values.

    ``masks['foreground']`` is required.  Other masks must be subsets of it.
    Each region includes signed error, brightness, area, and error contribution.
    ``brightness_contrasts`` records paired region contrast and retention; this
    describes intensity compression and is not a biological recovery metric.
    """
    target = validate_image(target_chw, "target")
    prediction = validate_image(prediction_chw, "prediction")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    shape = target.shape[1:]
    if "foreground" not in masks:
        raise ValueError("masks must include foreground")
    foreground = _mask(masks["foreground"], shape, "foreground")
    for name, mask in masks.items():
        _mask(mask, shape, name)
        if np.any(mask & ~foreground):
            raise ValueError(f"{name}: regional pixels outside foreground")
    delta = prediction - target
    absolute = np.abs(delta).mean(axis=0)
    squared = np.square(delta).mean(axis=0)
    signed = delta.mean(axis=0)
    gt_y, pred_y = luminance(target), luminance(prediction)
    ssim = ssim_map_rgb(target, prediction)
    foreground_count = int(foreground.sum())
    foreground_absolute_sum = float(absolute[foreground].sum())
    regions: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        center_mask, full_mask = _ssim_region_masks(mask)
        center_count, full_count = int(center_mask.sum()), int(full_mask.sum())
        mse = _mean_or_none(squared, mask)
        psnr = None if mse is None or mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
        psnr_reason = "empty_region" if mse is None else (
            "zero_mse_psnr_positive_infinity" if mse == 0.0 else None
        )
        absolute_sum = float(absolute[mask].sum())
        regions[name] = {
            "status": "ok" if count else "empty_region",
            "pixel_count": count,
            "image_fraction": count / float(shape[0] * shape[1]),
            "foreground_fraction": count / float(foreground_count) if foreground_count else None,
            "l1": _mean_or_none(absolute, mask), "mse": mse,
            "psnr": psnr, "psnr_reason": psnr_reason,
            "signed_mean_error": _mean_or_none(signed, mask),
            "signed_channel_mean_error_rgb": [_mean_or_none(channel, mask) for channel in delta],
            "target_luminance_mean": _mean_or_none(gt_y, mask),
            "prediction_luminance_mean": _mean_or_none(pred_y, mask),
            "signed_luminance_mean_error": _mean_or_none(pred_y - gt_y, mask),
            "absolute_error_sum_rgb_mean": absolute_sum,
            "foreground_absolute_error_contribution": (
                absolute_sum / foreground_absolute_sum if foreground_absolute_sum > 0.0 else None
            ),
            "foreground_absolute_error_contribution_reason": (
                None if foreground_absolute_sum > 0.0 else "zero_foreground_absolute_error"
            ),
            "regional_center_ssim": _mean_or_none(ssim, center_mask),
            "regional_center_ssim_window_count": center_count,
            "regional_center_ssim_reason": None if center_count else "no_valid_window_centers_in_region",
            "regional_full_window_ssim": _mean_or_none(ssim, full_mask),
            "regional_full_window_ssim_window_count": full_count,
            "regional_full_window_ssim_reason": None if full_count else "no_complete_11x11_windows_in_region",
        }
    contrasts: dict[str, Any] = {}
    pairs = [("primary", PRIMARY_BRIGHT, PRIMARY_NONBRIGHT), ("local", LOCAL_BRIGHT, LOCAL_NONBRIGHT)]
    pairs.extend((f"luminance_{i:02d}_local", f"luminance_{i:02d}_local_bright",
                  f"luminance_{i:02d}_local_nonbright") for i in range(10))
    for name, bright_key, rest_key in pairs:
        if bright_key not in regions or rest_key not in regions:
            continue
        bright_region, rest_region = regions[bright_key], regions[rest_key]
        if not bright_region["pixel_count"] or not rest_region["pixel_count"]:
            contrasts[name] = {"status": "empty_comparison_region", "target_difference": None,
                               "prediction_difference": None, "retention_ratio": None}
            continue
        target_difference = bright_region["target_luminance_mean"] - rest_region["target_luminance_mean"]
        prediction_difference = bright_region["prediction_luminance_mean"] - rest_region["prediction_luminance_mean"]
        valid = abs(target_difference) > 1e-12
        contrasts[name] = {
            "status": "ok" if valid else "target_contrast_too_small",
            "target_difference": target_difference, "prediction_difference": prediction_difference,
            "retention_ratio": prediction_difference / target_difference if valid else None,
        }
    return {"regions": regions, "brightness_contrasts": contrasts}


def paired_bootstrap(
    candidate: Sequence[float],
    reference: Sequence[float],
    subject_ids: Sequence[str] | None = None,
    subject_ids_reliable: bool = False,
    higher_is_better: bool = False,
    iterations: int = 2000,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Paired percentile bootstrap with image-weighted cluster resampling.

    Resampling subjects retains every image in each drawn subject, including
    repeated groups. Thus each replicate is sum(image differences)/image count,
    not an equal-weight mean of subject means. Unreliable/missing subject IDs
    explicitly use image bootstrap. A single reliable subject yields no CI.
    Missing/nonfinite measurements must be explicitly handled by the caller;
    this function raises instead of silently removing them.
    """
    if not isinstance(subject_ids_reliable, bool) or not isinstance(higher_is_better, bool):
        raise TypeError("subject_ids_reliable and higher_is_better must be explicit bools")
    left, right = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("paired vectors must be nonempty, 1D, and equal length")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired vectors contain non-finite values; silent omission is prohibited")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if subject_ids is not None and len(subject_ids) != left.size:
        raise ValueError("subject ID count does not match paired images")
    differences = left - right
    if not np.isfinite(differences).all():
        raise ValueError("paired differences are non-finite")
    if subject_ids_reliable:
        if subject_ids is None or any(not isinstance(s, str) or not s.strip() for s in subject_ids):
            raise ValueError("reliable IDs require a nonempty string for every image")
        unique_subjects = sorted(set(subject_ids))
        subject_index = {s: index for index, s in enumerate(unique_subjects)}
        inverse = np.asarray([subject_index[s] for s in subject_ids], dtype=np.int64)
        group_count = len(unique_subjects)
        group_sizes = np.bincount(inverse, minlength=group_count)
        group_sums = np.bincount(inverse, weights=differences, minlength=group_count)
        method = "paired_subject_cluster_percentile_bootstrap"
        limitation = None
        unique_subject_count: int | None = group_count
    else:
        group_count = int(left.size)
        group_sizes = np.ones(group_count, dtype=np.int64)
        group_sums = differences
        method = "paired_image_percentile_bootstrap"
        limitation = "Reliable subject IDs unavailable; within-subject dependence is not accounted for."
        unique_subject_count = None
    improved = differences > 0.0 if higher_is_better else differences < 0.0
    ties = differences == 0.0
    interval: list[float] | None = None
    reason: str | None = None
    if group_count < 2:
        reason = "fewer_than_two_reliable_subjects" if subject_ids_reliable else "fewer_than_two_images"
    else:
        rng = np.random.default_rng(seed)
        bootstrap_means = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            draw = rng.integers(0, group_count, size=group_count)
            bootstrap_means[index] = group_sums[draw].sum() / group_sizes[draw].sum()
        if not np.isfinite(bootstrap_means).all():
            raise ValueError("bootstrap produced non-finite replicate means")
        interval = [float(v) for v in np.quantile(bootstrap_means, [0.025, 0.975])]
    return {
        "status": "ok" if interval is not None else "insufficient_resampling_units",
        "candidate_mean": float(left.mean()), "reference_mean": float(right.mean()),
        "paired_difference_mean": float(differences.mean()),
        "difference_definition": "candidate minus reference", "higher_is_better": higher_is_better,
        "image_count": int(left.size), "unique_subject_count": unique_subject_count,
        "resampling_unit_count": group_count, "subject_ids_reliable": subject_ids_reliable,
        "improved_count": int(improved.sum()), "improved_fraction": float(improved.mean()),
        "tied_count": int(ties.sum()), "tied_fraction": float(ties.mean()),
        "worsened_count": int((~improved & ~ties).sum()),
        "ci95": interval, "ci95_reason": reason,
        "method": method, "estimand": "image-equal-weight mean paired difference",
        "iterations": iterations, "seed": seed,
        "interval_algorithm": "percentile [2.5,97.5], numpy quantile(method=linear)",
        "statistical_limitation": limitation,
    }
