"""Development v4 UV-dark candidate module, not validated biological labels.

Known porphyrin and anatomical exclusions enter BEFORE background estimation.
Multiscale robust background + strong/weak connected reconstruction expands a
dark seed only into measured weaker darkness, never by arbitrary mask dilation.
v1/v2, company porphyrin rules, training and frozen results are not modified.
"""
from dataclasses import asdict, dataclass
from collections import Counter
import cv2
import numpy as np


@dataclass(frozen=True)
class DarkSpotConfig:
    max_work_height: int = 1600
    median_kernels: tuple = (25, 65, 121)
    smooth_sigma: float = 2.5
    texture_kernel: int = 25
    texture_seed_factor: float = 3.0
    missing_fill_sigma: float = 18.0
    seed_abs: float = 4.0
    seed_relative: float = 0.035
    weak_abs: float = 2.0
    weak_relative: float = 0.015
    min_context_fraction: float = 0.55
    porphyrin_guard_radius: int = 2
    red_hue_low_max: int = 22
    red_hue_high_min: int = 170
    red_min_saturation: int = 70
    red_min_value: int = 25
    red_channel_ratio: float = 1.10
    min_value: float = 20.0
    min_background: float = 30.0
    min_area: int = 18
    max_area: int = 60000
    max_aspect_ratio: float = 4.0
    min_circularity: float = 0.16
    min_solidity: float = 0.35
    min_fill_fraction: float = 0.45
    min_minor_axis: float = 4.0
    max_extreme_dark_core_fraction: float = 0.12
    min_ring_pixels: int = 24
    ring_contrast_abs: float = 3.0
    ring_contrast_relative: float = 0.025


def _config(config):
    c = DarkSpotConfig() if config is None else DarkSpotConfig(**config) if isinstance(config, dict) else config
    if not isinstance(c, DarkSpotConfig):
        raise TypeError("config must be DarkSpotConfig, dict, or None")
    for key, value in asdict(c).items():
        if key == "median_kernels":
            if not value or any(not isinstance(k, int) or k < 3 or k % 2 != 1 for k in value):
                raise ValueError("median_kernels must contain positive odd integers >=3")
        elif not np.isfinite(value):
            raise ValueError("All config values must be finite")
    if not isinstance(c.max_work_height, int) or c.max_work_height < 16:
        raise ValueError("Invalid max_work_height")
    if not isinstance(c.porphyrin_guard_radius, int) or not 0 <= c.porphyrin_guard_radius <= 20:
        raise ValueError("Invalid porphyrin guard radius")
    if not 0 < c.weak_abs <= c.seed_abs or not 0 < c.weak_relative <= c.seed_relative:
        raise ValueError("Require positive weak thresholds <= seed thresholds")
    if c.smooth_sigma <= 0 or c.missing_fill_sigma <= 0 or not 0 < c.min_context_fraction <= 1:
        raise ValueError("Invalid smoothing or support threshold")
    if not isinstance(c.texture_kernel, int) or c.texture_kernel < 3 or c.texture_kernel % 2 != 1 or c.texture_seed_factor < 0:
        raise ValueError("Invalid local texture support or seed factor")
    if not 0 <= c.min_value < c.min_background <= 255 or c.min_area < 1 or c.max_area < c.min_area:
        raise ValueError("Invalid intensity or area bounds")
    if c.max_aspect_ratio < 1 or c.min_minor_axis < 0 or c.min_ring_pixels < 1 or c.ring_contrast_abs < 0 or c.ring_contrast_relative < 0:
        raise ValueError("Invalid shape or contrast threshold")
    if any(not 0 <= v <= 1 for v in (c.min_circularity, c.min_solidity, c.min_fill_fraction, c.max_extreme_dark_core_fraction)):
        raise ValueError("Shape fractions must be in [0,1]")
    return c


def _mask(value, shape, name, optional=False):
    if value is None and optional:
        return np.zeros(shape, bool)
    if not isinstance(value, np.ndarray) or value.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise TypeError(name + " must be bool or uint8")
    if value.shape != shape:
        raise ValueError(name + " must match native UV shape")
    return value != 0


def _smooth(value, known, sigma):
    weights = known.astype(np.float32)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    numerator = cv2.GaussianBlur(value * weights, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    return numerator / np.maximum(denominator, 1e-6), denominator


def detect_uv_darkspots(image_bgr, valid_mask, porphyrin_mask=None, anatomical_exclusion=None, config=None):
    """Detect UV-only dark candidates, explicitly excluding provided bright/anatomy.

    All input masks: native bool/uint8 [H,W]. Image: uint8 [H,W,3] BGR.
    Main outputs are native-size 0/255 masks projected from a work image <=1600h.
    eligible != assessed: unavailable background support is separately recorded.
    No RGB input, learned biological classifier, or clinical truth is involved.
    """
    c = _config(config)
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must be uint8 numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError("image_bgr must have nonempty H,W,3 shape")
    h, w = image_bgr.shape[:2]
    native_valid = _mask(valid_mask, (h, w), "valid_mask")
    native_p = _mask(porphyrin_mask, (h, w), "porphyrin_mask", True)
    native_a = _mask(anatomical_exclusion, (h, w), "anatomical_exclusion", True)
    wh = min(h, c.max_work_height)
    ww = max(1, round(w * wh / h))
    sx, sy = ww / w, wh / h
    size = (ww, wh)
    native_known = native_valid & ~native_p & ~native_a
    if (wh, ww) == (h, w):
        work = image_bgr.copy()
        work[~native_known] = 0
        geometry = native_valid.copy()
        p, anatomy = native_p.copy(), native_a.copy()
    else:
        weight = native_known.astype(np.float32)
        area_weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        work = np.empty((wh, ww, 3), np.uint8)
        for channel in range(3):
            numerator = cv2.resize(image_bgr[..., channel].astype(np.float32) * weight, size, interpolation=cv2.INTER_AREA)
            work[..., channel] = np.clip(np.rint(numerator / np.maximum(area_weight, 1e-6)), 0, 255).astype(np.uint8)
        geometry = cv2.resize(native_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        # Any native excluded-pixel coverage excludes the corresponding work cell.
        p = cv2.resize(native_p.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        anatomy = cv2.resize(native_a.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    radius = c.porphyrin_guard_radius
    guard = cv2.dilate(p.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))) != 0
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    hue, saturation, value = [hsv[..., j].astype(np.float32) for j in range(3)]
    blue, green, red = [work[..., j].astype(np.float32) for j in range(3)]
    red_veto = (((hue <= c.red_hue_low_max) | (hue >= c.red_hue_high_min))
                & (saturation >= c.red_min_saturation) & (value > c.red_min_value)
                & (red > green * c.red_channel_ratio) & (red > blue * c.red_channel_ratio)
                & geometry & ~guard & ~anatomy)
    eligible = geometry & ~guard & ~anatomy & ~red_veto
    smooth, smooth_weight = _smooth(value, eligible, c.smooth_sigma)
    fill, fill_weight = _smooth(value, eligible, c.missing_fill_sigma)
    fallback = float(np.median(value[eligible])) if np.any(eligible) else 0.0
    fill[fill_weight < 1e-5] = fallback
    smooth[smooth_weight < 1e-5] = fill[smooth_weight < 1e-5]
    filled = np.where(eligible, smooth, fill)
    median_input = np.clip(np.rint(filled), 0, 255).astype(np.uint8)
    # Robust local high-frequency magnitude: isolated broad dark spots have
    # nearly constant interiors, while fine alternating texture retains a
    # nonzero local median residual. Excluded colors never enter this estimate.
    residual = np.abs(value - smooth)
    residual_fill, residual_weight = _smooth(residual, eligible, c.missing_fill_sigma)
    residual_fallback = float(np.median(residual[eligible])) if np.any(eligible) else 0.0
    residual_fill[residual_weight < 1e-5] = residual_fallback
    residual_filled = np.where(eligible, residual, residual_fill)
    texture = cv2.medianBlur(np.clip(np.rint(residual_filled), 0, 255).astype(np.uint8), c.texture_kernel).astype(np.float32)
    seed_threshold = np.maximum(c.seed_abs, c.texture_seed_factor * texture)
    seeds = np.zeros((wh, ww), bool)
    weak = np.zeros_like(seeds)
    assessment = np.zeros_like(seeds)
    maps = {"value": value, "smooth_value": smooth, "local_texture_magnitude": texture,
            "seed_threshold": seed_threshold}
    scales = []
    for kernel in c.median_kernels:
        background = cv2.medianBlur(median_input, kernel).astype(np.float32)
        # Full geometric support avoids asymmetric off-face background. Excluded
        # P/anatomy are imputed from eligible neighbors, never their own colors.
        geometric_support = cv2.erode(geometry.astype(np.uint8), np.ones((kernel, kernel), np.uint8),
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
        coverage = cv2.boxFilter(eligible.astype(np.float32), -1, (kernel, kernel), normalize=True,
                                 borderType=cv2.BORDER_CONSTANT)
        supported = eligible & geometric_support & (coverage >= c.min_context_fraction)
        raw_dark = background - value
        smooth_dark = background - smooth
        relative = raw_dark / np.maximum(background, 1)
        finite_lit = (value > c.min_value) & (background >= c.min_background)
        strong_k = (supported & finite_lit & (raw_dark >= seed_threshold) & (smooth_dark >= seed_threshold)
                    & (relative >= c.seed_relative))
        weak_k = (supported & finite_lit & (raw_dark >= c.weak_abs) & (smooth_dark >= c.weak_abs * 0.5)
                  & (relative >= c.weak_relative))
        seeds |= strong_k
        weak |= weak_k
        assessment |= supported
        maps[f"background_{kernel}"] = background
        scales.append({"kernel": kernel, "supported_pixels": int(supported.sum()),
                       "seed_pixels": int(strong_k.sum()), "weak_pixels": int(weak_k.sum())})
    count, labels, stats, centers = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    seed_labels = set(map(int, np.unique(labels[seeds]))) - {0}
    candidate = np.isin(labels, list(seed_labels)) if seed_labels else np.zeros_like(weak)
    selected = np.zeros((wh, ww), np.uint8)
    spots, rejected_spots = [], []
    rejected = Counter()
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        record = {"component_label_work": label, "bbox_work_xyxy": [x, y, x + bw, y + bh],
                  "area_work_pixels": area, "accepted": False}
        reasons = []
        if label not in seed_labels:
            reasons = ["no_strong_seed"]
        elif area < c.min_area:
            reasons = ["too_small"]
        elif area > c.max_area:
            reasons = ["too_large"]
        if reasons:
            rejected.update(reasons)
            rejected_spots.append({**record, "reject_reason": reasons})
            continue
        local = (labels[y:y + bh, x:x + bw] == label).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        filled_contour = np.zeros_like(local)
        cv2.drawContours(filled_contour, [contour], -1, 1, cv2.FILLED)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = min(1.0, 4 * np.pi * contour_area / max(1e-6, perimeter * perimeter))
        solidity = contour_area / max(1e-6, float(cv2.contourArea(cv2.convexHull(contour))))
        # Excluded P/anatomy/red pixels are missing observations, NOT dark-zero
        # measurements or evidence that the dark component is hollow.
        observed_interior = (filled_contour != 0) & eligible[y:y + bh, x:x + bw]
        observed_interior_count = int(observed_interior.sum())
        fill_fraction = area / max(1, observed_interior_count)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        yy, xx = np.nonzero(local)
        centered = np.stack((xx - xx.mean(), yy - yy.mean()), axis=1)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(xx))
        minor_axis = 4 * np.sqrt(max(0.0, float(eigenvalues[0])))
        black_core = float(np.count_nonzero(observed_interior & (value[y:y + bh, x:x + bw] <= c.min_value)) / max(1, observed_interior_count))
        if aspect > c.max_aspect_ratio: reasons.append("elongated_bbox")
        if circularity < c.min_circularity: reasons.append("low_circularity")
        if solidity < c.min_solidity: reasons.append("low_solidity")
        if fill_fraction < c.min_fill_fraction: reasons.append("hollow_or_fragmented")
        if minor_axis < c.min_minor_axis: reasons.append("thin_component")
        if black_core > c.max_extreme_dark_core_fraction: reasons.append("extreme_dark_core")
        # Check an external local ring, excluding other weak-dark components and
        # every pre-excluded pixel. This rejects large shading/edge fragments.
        ring_radius = max(3, min(15, round(np.sqrt(area / np.pi) * 0.25)))
        x0, y0 = max(0, x - ring_radius), max(0, y - ring_radius)
        x1, y1 = min(ww, x + bw + ring_radius), min(wh, y + bh + ring_radius)
        patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
        patch[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = filled_contour
        expanded = cv2.dilate(patch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)))
        ring = (expanded != 0) & (patch == 0) & eligible[y0:y1, x0:x1] & ~weak[y0:y1, x0:x1]
        ring_count = int(ring.sum())
        component_value = float(np.median(value[yy + y, xx + x]))
        ring_value = float(np.median(value[y0:y1, x0:x1][ring])) if ring_count else None
        ring_delta = ring_value - component_value if ring_count else None
        if ring_count < c.min_ring_pixels:
            reasons.append("insufficient_background_ring")
        elif ring_delta < c.ring_contrast_abs or ring_delta / max(1.0, ring_value) < c.ring_contrast_relative:
            reasons.append("insufficient_local_ring_contrast")
        record.update(circularity=circularity, solidity=solidity, fill_fraction=float(fill_fraction),
                      bbox_aspect_ratio=float(aspect), minor_axis_work_pixels=float(minor_axis),
                      extreme_dark_core_fraction=black_core, observed_interior_pixels=observed_interior_count, ring_radius_work=ring_radius,
                      ring_pixels=ring_count, median_component_V=component_value,
                      median_ring_V=ring_value, ring_contrast=ring_delta)
        if reasons:
            rejected.update(reasons)
            rejected_spots.append({**record, "reject_reason": reasons})
            continue
        selected[y:y + bh, x:x + bw][local != 0] = 255
        cx, cy = map(float, centers[label])
        record.update(id=len(spots) + 1, accepted=True, reject_reason=[],
                      center_work_xy=[cx, cy], center_native_xy=[(cx + .5) / sx - .5, (cy + .5) / sy - .5],
                      bbox_native_xyxy=[int(np.floor(x / sx)), int(np.floor(y / sy)),
                                        min(w, int(np.ceil((x + bw) / sx))), min(h, int(np.ceil((y + bh) / sy)))])
        spots.append(record)
    def project(mask):
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) != 0
    guard_native = (project(guard) | native_p) & native_valid
    anatomy_native = (project(anatomy) | native_a) & native_valid
    red_native = project(red_veto) & native_valid
    eligible_native = native_valid & ~guard_native & ~anatomy_native & ~red_native
    assessed_native = project(assessment) & eligible_native
    mask_native = project(selected) & assessed_native
    candidate_native = project(candidate) & assessed_native
    unassessed_native = eligible_native & ~assessed_native
    assert not np.any(mask_native & (~candidate_native | native_p | native_a))
    maps.update(seed_mask=seeds.astype(np.uint8) * 255, strong_mask=seeds.astype(np.uint8) * 255,
                weak_mask=weak.astype(np.uint8) * 255, candidate_mask=candidate.astype(np.uint8) * 255,
                selected_mask=selected, eligible_mask=eligible.astype(np.uint8) * 255,
                assessment_mask=assessment.astype(np.uint8) * 255,
                porphyrin_guard=guard.astype(np.uint8) * 255,
                anatomical_exclusion=anatomy.astype(np.uint8) * 255,
                red_veto=red_veto.astype(np.uint8) * 255,
                unassessed_mask=(eligible & ~assessment).astype(np.uint8) * 255)
    diagnostics = {
        "algorithm": "v4 spatially coherent, texture-adaptive dark seeds with multiscale robust background and hysteresis",
        "status": "development_candidate_not_biological_ground_truth", "parameter_status": "fixed engineering values; intentional development-set optimization",
        "config": asdict(c), "native_shape_hw": [h, w], "work_shape_hw": [wh, ww],
        "scale_native_to_work_xy": [sx, sy], "work_scales": scales,
        "native_projection": "nearest-neighbor projection of <=1600h detections, NOT native-detail detection",
        "background_protocol": "P/anatomy excluded before mask-weighted area resize; P guard and independent red veto excluded before Gaussian/median background. Missing neighbors imputed from eligible pixels, never excluded pixel values.",
        "red_veto_meaning": "dark-only reddish-color exclusion; does not modify or enlarge company porphyrin labels",
        "support_protocol": "each median scale requires full supplied geometric support and >=55% eligible context; missing support is explicitly unassessed",
        "hysteresis": "8-connected weak measured-dark regions retained only when containing >=1 strong seed; no final dilation or contour filling",
        "texture_protocol": "At each eligible work pixel compute |V - normalized Gaussian(V,sigma=2.5)|; median magnitude in 25x25 neighborhood, imputing missing support only from eligible pixels. Strong absolute threshold=max(4,3*local median residual). Weak measured-dark support is unchanged. Engineering texture rejection, not a validated noise or clinical model.",
        "ring_protocol": "outer ring median V must exceed component median V; excludes every known exclusion and weak-dark pixels",
        "black_core_and_fill_protocol": "Only eligible pixels inside the external contour enter dark-core/fill denominators; known excluded pixels are missing, not black evidence",
        "valid_native_pixels": int(native_valid.sum()), "eligible_native_pixels": int(eligible_native.sum()),
        "assessed_native_pixels": int(assessed_native.sum()), "unassessed_native_pixels": int(unassessed_native.sum()),
        "porphyrin_guard_native_pixels": int(guard_native.sum()), "anatomical_exclusion_native_pixels": int(anatomy_native.sum()),
        "red_veto_native_pixels": int(red_native.sum()), "candidate_native_pixels": int(candidate_native.sum()),
        "selected_native_pixels": int(mask_native.sum()),
        "selected_fraction_of_valid": float(mask_native.sum() / native_valid.sum()) if native_valid.any() else None,
        "weak_components": count - 1, "seeded_components": len(seed_labels), "kept_components": len(spots),
        "rejected_components": len(rejected_spots), "rejection_reason_counts": dict(rejected),
        "exclusion_mask_pixels_can_overlap": True,
        "limitations": ["Fixed engineering candidate rules, not validated UV spot segmentation", "Dark lesions may be missed; retained texture/shadow/moles remain possible", "Anatomical exclusion can remove genuine signals", "Known bright guard and unassessed support reduce eligible coverage", "Remaining area is not proven normal skin", "No RGB pixels, training or original company detector changes"],
    }
    return {"mask_native": mask_native.astype(np.uint8) * 255,
            "candidate_mask_native": candidate_native.astype(np.uint8) * 255,
            "eligible_mask_native": eligible_native.astype(np.uint8) * 255,
            "assessment_mask_native": assessed_native.astype(np.uint8) * 255,
            "porphyrin_guard_native": guard_native.astype(np.uint8) * 255,
            "anatomical_exclusion_native": anatomy_native.astype(np.uint8) * 255,
            "red_veto_native": red_native.astype(np.uint8) * 255,
            "unassessed_mask_native": unassessed_native.astype(np.uint8) * 255,
            "spots": spots, "rejected_spots": rejected_spots,
            "diagnostics": diagnostics, "maps_work": maps}


