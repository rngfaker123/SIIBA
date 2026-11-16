"""UV local-darkness candidates v2; engineering pilot, not a UV-spot truth.

Only UV pixels and a supplied geometric analysis mask are used. This module
does not read RGB, porphyrin masks, labels, or model predictions. Parameters
are fixed engineering starting values, not optimized or clinically validated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class DarkSpotConfig:
    max_work_height: int = 1600
    sigma_small: float = 4.0
    sigma_large: float = 12.0
    small_abs: float = 4.0
    large_abs: float = 7.0
    small_relative: float = 0.06
    large_relative: float = 0.12
    min_value: float = 20.0
    min_background: float = 30.0
    min_area: int = 8
    max_area: int = 1200
    max_aspect_ratio: float = 4.0
    min_circularity: float = 0.12
    min_solidity: float = 0.25
    min_minor_axis: float = 3.0
    max_extreme_dark_core_fraction: float = 0.12
    median_kernel: int = 25
    median_dark_abs: float = 4.0
    reject_boundary_contact: bool = True


def _validate_config(config):
    cfg = DarkSpotConfig() if config is None else DarkSpotConfig(**config) if isinstance(config, dict) else config
    if not isinstance(cfg, DarkSpotConfig):
        raise TypeError("config must be DarkSpotConfig, dict, or None")
    values = asdict(cfg)
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError("All configuration values must be finite")
    if not isinstance(cfg.max_work_height, int) or cfg.max_work_height < 16:
        raise ValueError("max_work_height must be an integer >= 16")
    if not 0 < cfg.sigma_small < cfg.sigma_large:
        raise ValueError("Require 0 < sigma_small < sigma_large")
    if cfg.min_area < 1 or cfg.max_area < cfg.min_area:
        raise ValueError("Invalid component area bounds")
    if min(cfg.small_abs, cfg.large_abs, cfg.small_relative, cfg.large_relative) <= 0:
        raise ValueError("Darkness thresholds must be positive")
    if not 0 <= cfg.min_value < cfg.min_background <= 255:
        raise ValueError("Require 0 <= min_value < min_background <= 255")
    if cfg.max_aspect_ratio < 1 or not 0 <= cfg.min_circularity <= 1 or not 0 <= cfg.min_solidity <= 1 or cfg.min_minor_axis < 0:
        raise ValueError("Invalid shape threshold")
    if not 0 <= cfg.max_extreme_dark_core_fraction <= 1:
        raise ValueError("Invalid extreme-dark core fraction")
    if not isinstance(cfg.median_kernel, int) or cfg.median_kernel < 3 or cfg.median_kernel % 2 != 1 or cfg.median_dark_abs <= 0:
        raise ValueError("median_kernel must be odd >=3 and median_dark_abs positive")
    return cfg


def _background(value, valid, sigma):
    """Normalized Gaussian average: invalid/outside pixels do not darken it."""
    weights = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(value * weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    return numerator / np.maximum(denominator, 1e-6)


def detect_uv_darkspots(image_bgr, valid_mask, config=None):
    """Return native-size masks, retained components, diagnostics, work maps.

    image_bgr: uint8 [H,W,3], native UV. valid_mask: bool or uint8 [H,W].
    mask_native / candidate_mask_native: uint8 [H,W] with only 0 and 255.
    mask_native is a subset of candidate_mask_native and valid_mask.
    Work maps are not diagnostic probabilities. Native masks are nearest-neighbor
    projections of new detections at at most 1600 pixels high, not native-detail
    detections. Inputs are never mutated.
    """
    cfg = _validate_config(config)
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must be a uint8 numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError("image_bgr must have nonempty shape H,W,3")
    if not isinstance(valid_mask, np.ndarray) or valid_mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise TypeError("valid_mask must be bool or uint8 numpy array")
    if valid_mask.shape != image_bgr.shape[:2]:
        raise ValueError("valid_mask must have the same native H,W as UV")
    valid_native = valid_mask != 0
    native_h, native_w = valid_native.shape
    work_h = min(native_h, cfg.max_work_height)
    work_w = max(1, round(native_w * work_h / native_h))
    scale_x, scale_y = work_w / native_w, work_h / native_h
    if (work_h, work_w) == (native_h, native_w):
        work = image_bgr.copy()
        valid = valid_native.copy()
    else:
        # Mask-aware area resampling prevents out-of-domain colors leaking into
        # work pixels near a geometric boundary. Only valid source pixels vote.
        native_weights = valid_native.astype(np.float32)
        work_weights = cv2.resize(native_weights, (work_w, work_h), interpolation=cv2.INTER_AREA)
        work = np.empty((work_h, work_w, 3), np.uint8)
        for channel in range(3):
            numerator = cv2.resize(image_bgr[..., channel].astype(np.float32) * native_weights,
                                   (work_w, work_h), interpolation=cv2.INTER_AREA)
            work[..., channel] = np.clip(np.rint(numerator / np.maximum(work_weights, 1e-6)), 0, 255).astype(np.uint8)
        valid = cv2.resize(valid_native.astype(np.uint8), (work_w, work_h), interpolation=cv2.INTER_NEAREST) != 0
    value = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    back_small = _background(value, valid, cfg.sigma_small)
    back_large = _background(value, valid, cfg.sigma_large)
    residual_small = np.maximum(back_small - value, 0)
    residual_large = np.maximum(back_large - value, 0)
    relative_small = residual_small / np.maximum(back_small, 1)
    relative_large = residual_large / np.maximum(back_large, 1)
    # Gaussian averages can be raised by a bright dot and create a false dark
    # ring on otherwise uniform skin. A local-median confirmation rejects that
    # synthetic artifact. Keep only pixels with complete valid median support;
    # this is detection support, not a replacement anatomical region.
    median_background = cv2.medianBlur(value.astype(np.uint8), cfg.median_kernel).astype(np.float32)
    median_residual = median_background - value
    median_support = cv2.erode(valid.astype(np.uint8), np.ones((cfg.median_kernel, cfg.median_kernel), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    candidate = (valid & (value > cfg.min_value) & (back_large >= cfg.min_background)
                 & (residual_small >= cfg.small_abs) & (residual_large >= cfg.large_abs)
                 & (relative_small >= cfg.small_relative) & (relative_large >= cfg.large_relative)
                 & median_support & (median_residual >= cfg.median_dark_abs))
    # The one-pixel ring is only used to flag clipped components; no expansion
    # of either the supplied region or the final detections is performed.
    boundary_ring = valid & (cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    selected = np.zeros_like(candidate, dtype=np.uint8)
    rejected = Counter()
    spots = []
    rejected_spots = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        reasons = []
        component = {"component_label_work": label, "bbox_work_xyxy": [x, y, x + width, y + height],
                     "area_work_pixels": area, "accepted": False}
        if area < cfg.min_area:
            rejected["too_small"] += 1
            rejected_spots.append({**component, "reject_reason": ["too_small"]})
            continue
        if area > cfg.max_area:
            rejected["too_large"] += 1
            rejected_spots.append({**component, "reject_reason": ["too_large"]})
            continue
        local = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        if cfg.reject_boundary_contact and np.any(local.astype(bool) & boundary_ring[y:y + height, x:x + width]):
            reasons.append("region_boundary_contact")
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = min(1.0, 4 * np.pi * contour_area / max(perimeter * perimeter, 1e-6))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / max(hull_area, 1e-6)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        yy, xx = np.nonzero(local)
        centered = np.stack((xx - xx.mean(), yy - yy.mean()), axis=1)
        covariance = centered.T @ centered / len(xx)
        minor_axis = 4 * np.sqrt(max(0.0, float(np.linalg.eigvalsh(covariance)[0])))
        # A pixelwise V floor alone could leave a ring around a black cavity.
        # Inspect its outer-contour interior, including excluded central holes.
        filled = np.zeros_like(local)
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        core_domain = (filled != 0) & valid[y:y + height, x:x + width]
        dark_core = core_domain & (value[y:y + height, x:x + width] <= cfg.min_value)
        dark_core_fraction = float(dark_core.sum() / max(1, core_domain.sum()))
        if dark_core_fraction > cfg.max_extreme_dark_core_fraction:
            reasons.append("extreme_dark_core")
        if aspect_ratio > cfg.max_aspect_ratio:
            reasons.append("elongated_bbox")
        if circularity < cfg.min_circularity:
            reasons.append("low_circularity")
        if solidity < cfg.min_solidity:
            reasons.append("low_solidity")
        if minor_axis < cfg.min_minor_axis:
            reasons.append("thin_minor_axis")
        if reasons:
            rejected.update(reasons)
            rejected_spots.append({**component, "reject_reason": reasons,
                                   "circularity": float(circularity), "solidity": float(solidity),
                                   "bbox_aspect_ratio": float(aspect_ratio), "minor_axis_work_pixels": float(minor_axis),
                                   "extreme_dark_core_fraction": dark_core_fraction})
            continue
        selected[y:y + height, x:x + width][local != 0] = 255
        ys, xs = yy + y, xx + x
        cx, cy = map(float, centers[label])
        spots.append({
            "id": len(spots) + 1, "component_label_work": label,
            "accepted": True, "reject_reason": [],
            "center_work_xy": [cx, cy],
            "center_native_xy": [(cx + 0.5) / scale_x - 0.5, (cy + 0.5) / scale_y - 0.5],
            "bbox_work_xyxy": [x, y, x + width, y + height],
            "bbox_native_xyxy": [int(np.floor(x / scale_x)), int(np.floor(y / scale_y)),
                                  min(native_w, int(np.ceil((x + width) / scale_x))), min(native_h, int(np.ceil((y + height) / scale_y)))],
            "area_work_pixels": area, "contour_area_work": contour_area,
            "area_native_approx": float(area / (scale_x * scale_y)),
            "circularity": float(circularity), "solidity": float(solidity),
            "bbox_aspect_ratio": float(aspect_ratio), "minor_axis_work_pixels": float(minor_axis),
            "extreme_dark_core_fraction": dark_core_fraction,
            "mean_V": float(value[ys, xs].mean()),
            "mean_dark_residual_small": float(residual_small[ys, xs].mean()),
            "mean_dark_residual_large": float(residual_large[ys, xs].mean()),
            "mean_relative_contrast_large": float(relative_large[ys, xs].mean()),
        })
    if (work_h, work_w) != (native_h, native_w):
        mask_native = cv2.resize(selected, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        candidate_native = cv2.resize(candidate.astype(np.uint8) * 255, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_native = selected.copy()
        candidate_native = candidate.astype(np.uint8) * 255
    mask_native[~valid_native] = 0
    candidate_native[~valid_native] = 0
    if np.any((mask_native != 0) & ((candidate_native == 0) | ~valid_native)):
        raise AssertionError("Native subset invariant failed")
    valid_count = int(valid_native.sum())
    diagnostics = {
        "algorithm": "UV V-channel dual-scale normalized-Gaussian darkness with local-median confirmation v2",
        "status": "engineering_pilot_not_validated_ground_truth",
        "parameter_status": "fixed engineering starting values, not optimized",
        "config": asdict(cfg), "native_shape_hw": [native_h, native_w],
        "work_shape_hw": [work_h, work_w], "scale_native_to_work_xy": [scale_x, scale_y],
        "work_resize": "mask-weighted INTER_AREA for each UV color channel, divided by mask coverage; INTER_NEAREST for supplied mask; no upsampling",
        "native_projection": "INTER_NEAREST then intersection with exact supplied native mask",
        "sigma_units": "pixels on work image; kept fixed for this pilot",
        "V_definition": "OpenCV HSV V = max(B,G,R) on uint8 UV",
        "candidate_rule": "inside supplied region AND V>min_value AND large background>=min_background AND absolute/relative darkness tests at both Gaussian scales AND local-median darkness confirmation; all values in config",
        "median_support": "Require complete median_kernel square inside valid work mask. Boundary pixels without support are unassessed for dark candidates; supplied native analysis mask itself is unchanged.",
        "median_support_work_pixels": int(median_support.sum()),
        "median_unsupported_valid_work_pixels": int((valid & ~median_support).sum()),
        "v2_rationale": "Synthetic bright disk on uniform background caused Gaussian-only false dark rings in v1; fixed local-median confirmation added before reviewing v2 pilot cases. Mask-aware downsampling fixes out-of-domain color leakage.",
        "valid_native_pixels": valid_count,
        "candidate_work_pixels": int(candidate.sum()),
        "candidate_native_pixels": int(np.count_nonzero(candidate_native)),
        "selected_work_pixels": int(np.count_nonzero(selected)),
        "selected_native_pixels": int(np.count_nonzero(mask_native)),
        "selected_fraction_of_valid": float(np.count_nonzero(mask_native) / valid_count) if valid_count else None,
        "candidate_components": count - 1, "kept_components": len(spots),
        "rejected_components": count - 1 - len(spots),
        "rejection_reason_counts": dict(rejected),
        "rejection_counts_note": "area failures short-circuit; shape failures can have multiple reasons",
        "limitations": ["Only local dark candidates, not biological UV-spot labels", "Moles, hair fragments, texture and shadows may survive", "Weak/large dark lesions may be missed", "Geometric input region can omit peripheral skin", "Native mask is a projection of lower-resolution detection", "No RGB comparison or porphyrin mask was used"],
    }
    return {"mask_native": mask_native, "candidate_mask_native": candidate_native,
            "spots": spots, "rejected_spots": rejected_spots, "diagnostics": diagnostics,
            "maps_work": {"value": value, "background_small": back_small, "background_large": back_large,
                          "dark_residual_small": residual_small, "dark_residual_large": residual_large,
                          "median_background": median_background, "median_dark_residual": median_residual,
                          "median_support": median_support.astype(np.uint8) * 255,
                          "relative_contrast_large": relative_large, "candidate_mask": candidate.astype(np.uint8) * 255,
                          "selected_mask": selected}}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_write(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _image_write(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise RuntimeError("Image encoding failed")
    with Path(path).open("xb") as f:
        f.write(encoded.tobytes())


def run_pilot(source_results, regions_results, output, cases):
    """Independent additive pilot; no historical files are overwritten."""
    cv2.setNumThreads(2)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Pilot output must be a new directory")
    source = json.loads(Path(source_results).read_text(encoding="utf-8"))
    regions = json.loads(Path(regions_results).read_text(encoding="utf-8"))
    by_folder = {c["folder"]: c for c in source["cases"]}
    geometry = {c["folder"]: c for c in regions["cases"]}
    if any(c not in by_folder or c not in geometry for c in cases):
        raise ValueError("Unknown requested case")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    shutil.copyfile(__file__, output / "source" / Path(__file__).name)
    _json_write(output / "protocol.json", {
        "schema": "uv_darkspot_pilot_v2", "case_order": cases,
        "source_results": str(Path(source_results).resolve()), "source_results_sha256": _sha(source_results),
        "regions_results": str(Path(regions_results).resolve()), "regions_results_sha256": _sha(regions_results),
        "script_sha256": _sha(__file__), "config": asdict(DarkSpotConfig()),
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "purpose": "Engineering visual pilot only; no performance claim, label, tuning, RGB input, or porphyrin-based dark candidate selection",
        "pixel_mapping": "same-frame geometry supplied by independent prior region build",
    })
    records = []
    for name in cases:
        started = time.monotonic()
        c, g = by_folder[name], geometry[name]
        for key in ("sample_id", "ordinal", "width", "height", "uv_sha256"):
            if c[key] != g[key]:
                raise ValueError(f"Geometry binding differs: {name}/{key}")
        if g["status"] != "ok" or _sha(g["mask_path"]) != g["region_mask_sha256"] or _sha(c["uv_source"]) != c["uv_sha256"]:
            raise ValueError("Source hash or geometry status check failed")
        image = cv2.imdecode(np.fromfile(c["uv_source"], dtype=np.uint8), cv2.IMREAD_COLOR)
        valid = cv2.imdecode(np.fromfile(g["mask_path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        result = detect_uv_darkspots(image, valid)
        folder = output / name
        folder.mkdir()
        _image_write(folder / "darkspot_mask_native.png", result["mask_native"])
        _image_write(folder / "candidate_mask_native.png", result["candidate_mask_native"])
        preview = cv2.resize(image, (1600, round(image.shape[0] * 1600 / image.shape[1])), interpolation=cv2.INTER_AREA)
        preview_mask = cv2.resize(result["mask_native"], (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
        alpha = (preview_mask.astype(np.float32) / 255 * 0.8)[..., None]
        color = np.zeros_like(preview); color[:] = (0, 215, 255)
        overlay = np.rint(preview * (1 - alpha) + color * alpha).astype(np.uint8)
        _image_write(folder / "uv_preview.jpg", preview)
        _image_write(folder / "darkspot_overlay_preview.jpg", overlay)
        _image_write(folder / "darkspot_mask_preview.png", preview_mask)
        _image_write(folder / "dark_residual_large_work.png", np.clip(result["maps_work"]["dark_residual_large"] * 5, 0, 255).astype(np.uint8))
        _json_write(folder / "spots.json", result["spots"])
        _json_write(folder / "rejected_spots.json", result["rejected_spots"])
        _json_write(folder / "diagnostics.json", result["diagnostics"])
        record = {"folder": name, "sample_id": c["sample_id"], "ordinal": c["ordinal"],
                  "native_shape_hw": list(image.shape[:2]), "uv_source": c["uv_source"], "uv_sha256": c["uv_sha256"],
                  "region_source": g["mask_path"], "region_mask_sha256": g["region_mask_sha256"],
                  "darkspot_mask_path": str((folder / "darkspot_mask_native.png").resolve()),
                  "darkspot_mask_sha256": _sha(folder / "darkspot_mask_native.png"),
                  "candidate_mask_sha256": _sha(folder / "candidate_mask_native.png"),
                  "spot_count": len(result["spots"]), "selected_native_pixels": result["diagnostics"]["selected_native_pixels"],
                  "selected_fraction_of_valid": result["diagnostics"]["selected_fraction_of_valid"],
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        _json_write(folder / "summary.json", record)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    _json_write(output / "results.json", {"status": "complete", "schema": "uv_darkspot_pilot_v2", "count": len(records),
                                           "protocol_sha256": _sha(output / "protocol.json"), "cases": records})


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=root / "output/porphyrin_company_review_20261021_v1/results.json")
    parser.add_argument("--regions-results", type=Path, default=root / "output/porphyrin_mediapipe_review_20261021_v1/regions_center_square/results.json")
    parser.add_argument("--output", type=Path, default=root / "output/uv_darkspot_pilot_20261021_v2")
    parser.add_argument("--cases", nargs="+", default=["case02", "case04", "case05", "case19"])
    args = parser.parse_args()
    run_pilot(args.source_results, args.regions_results, args.output, args.cases)


