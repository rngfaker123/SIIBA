"""Company-framework UV-dark candidate adaptation; not validated UV-spot labels.

Unchanged: OpenCV contour order, area>=1, moments centroid, the exact supplied
deduplicate_spots function, min_dist=10 native pixels, and center-domain gate.
New: one broad HSV intensity gate plus median-neighborhood dark contrast;
whole-object porphyrin/anatomy conflicts are isolated, never pixel-subtracted.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import ast
import hashlib

import cv2
import numpy as np


COMPANY_SOURCE_SHA256 = "de1c4d52ab781d3440b694afb0d9480eb0af1359a365aa5290ae0d750a31a0e9"
COMPANY_SOURCE = Path(__file__).resolve().parents[1] / "output/porphyrin_company_review_20261021_v1/source/porphyrin_module.py"


@dataclass(frozen=True)
class DarkSpotConfig:
    minimum_value: int = 20
    median_short_side_fraction: float = 0.025
    minimum_dark_difference: float = 6.0
    minimum_contour_area: float = 1.0
    min_dist: float = 10.0


@lru_cache(maxsize=4)
def load_company_primitives(source_path=None):
    """Execute only the reviewed, byte-bound original dedup function via AST.

    Also read, rather than re-invent, the exact 11 HSV bands from the business
    function detect_orange_spots_m. No proprietary module import is executed.
    """
    if source_path is None:
        adjacent = Path(__file__).resolve().with_name("porphyrin_module.py")
        source_path = adjacent if adjacent.exists() else COMPANY_SOURCE
    path = Path(source_path).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != COMPANY_SOURCE_SHA256:
        raise ValueError("Company source differs from reviewed immutable copy")
    source = raw.decode("utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    node = functions["deduplicate_spots"]
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    arrays = {}
    for n in functions["detect_orange_spots_m"].body:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
            continue
        name = n.targets[0].id
        if not name.startswith(("lower_orange", "upper_orange")):
            continue
        if not isinstance(n.value, ast.Call) or len(n.value.args) != 1:
            raise ValueError("Unexpected company HSV declaration")
        arrays[name] = ast.literal_eval(n.value.args[0])
    order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "_bright", "_dark"]
    bands = [(arrays["lower_orange" + s], arrays["upper_orange" + s]) for s in order]
    function_source = ast.get_source_segment(source, node)
    return scope["deduplicate_spots"], bands, {
        "source_path": str(path), "source_sha256": digest,
        "deduplicate_function_sha256": hashlib.sha256(function_source.encode("utf-8")).hexdigest(),
        "executed_functions": ["deduplicate_spots"], "company_bands_entry": "detect_orange_spots_m",
    }


def _mask(mask, shape, name, optional=False):
    if mask is None and optional:
        return np.zeros(shape, bool)
    if not isinstance(mask, np.ndarray) or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)) or mask.shape != shape:
        raise ValueError(name + " must be native-size bool or uint8 [H,W]")
    return mask != 0


def detect_uv_darkspots(image_bgr, valid_mask, porphyrin_mask=None, anatomical_exclusion=None, config=None):
    """Native BGR UV + fixed geometry/P masks -> dark objects and conflicts.

    No RGB input, photometric enhancement, resampling, shape heuristics, learned
    classifier, dilation of P, or subtraction of P pixels is used for detection.
    """
    c = DarkSpotConfig() if config is None else DarkSpotConfig(**config) if isinstance(config, dict) else config
    if not isinstance(c, DarkSpotConfig):
        raise TypeError("config must be DarkSpotConfig, dict, or None")
    if not all(np.isfinite(v) for v in asdict(c).values()):
        raise ValueError("Config must be finite")
    if not isinstance(c.minimum_value, int) or not 0 <= c.minimum_value < 255:
        raise ValueError("Invalid minimum_value")
    if not 0 < c.median_short_side_fraction <= 0.5 or c.minimum_dark_difference <= 0 or c.minimum_contour_area < 1 or c.min_dist < 0:
        raise ValueError("Invalid contrast, area, distance, or neighborhood parameter")
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError("image_bgr must be nonempty native uint8 H,W,3 BGR")
    h, w = image_bgr.shape[:2]
    valid = _mask(valid_mask, (h, w), "valid_mask")
    p = _mask(porphyrin_mask, (h, w), "porphyrin_mask", True)
    anatomy = _mask(anatomical_exclusion, (h, w), "anatomical_exclusion", True)
    deduplicate, company_bands, company_meta = load_company_primitives()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    kernel = max(3, int(round(min(h, w) * c.median_short_side_fraction)) | 1)
    background = cv2.medianBlur(value, kernel)
    difference = background.astype(np.int16) - value.astype(np.int16)
    support = cv2.erode(valid.astype(np.uint8), np.ones((kernel, kernel), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    broad_hsv = cv2.inRange(hsv, np.array([0, 0, c.minimum_value + 1]), np.array([179, 255, 255]))
    raw = (broad_hsv != 0) & support & (difference >= c.minimum_dark_difference)
    company_union = np.zeros((h, w), np.uint8)
    for lower, upper in company_bands:
        cv2.bitwise_or(company_union, cv2.inRange(hsv, np.array(lower), np.array(upper)), dst=company_union)
    # Preserve the exact company order and centroid implementation; do not sort.
    contours, _ = cv2.findContours(raw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_spots = []
    for contour_index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= c.minimum_contour_area:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = 0, 0
            raw_spots.append({"contour": contour, "center": (cx, cy), "area": area, "contour_index": contour_index})
    retained = deduplicate(raw_spots, min_dist=c.min_dist)
    retained_ids = {s["contour_index"] for s in retained}
    accepted = np.zeros((h, w), np.uint8)
    p_conflict = np.zeros_like(accepted)
    a_conflict = np.zeros_like(accepted)
    region_rejected = np.zeros_like(accepted)
    records = []
    count = {"accepted": 0, "porphyrin_conflict": 0, "anatomy_conflict": 0,
             "center_outside_region": 0, "dedup_suppressed": 0}
    for spot in raw_spots:
        contour, (cx, cy), area = spot["contour"], spot["center"], spot["area"]
        record = {"contour_index": spot["contour_index"], "center": [cx, cy], "contour_area": float(area),
                  "contour": contour.reshape(-1, 2).tolist(), "retained_after_dedup": spot["contour_index"] in retained_ids}
        if spot["contour_index"] not in retained_ids:
            record.update(status="dedup_suppressed", accepted=False, reasons=["company_original_distance_suppression"])
            count["dedup_suppressed"] += 1
            records.append(record)
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        footprint = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(footprint, [contour - np.array([[[x, y]]])], -1, 1, cv2.FILLED)
        fp = footprint != 0
        # Mask exports retain the original measured candidate pixels. Filled
        # contour is used only for whole-object conflict checks, including holes.
        pixels = fp & raw[y:y+bh, x:x+bw]
        p_count = int(np.count_nonzero(fp & p[y:y+bh, x:x+bw]))
        hsv_count = int(np.count_nonzero(fp & (company_union[y:y+bh, x:x+bw] != 0)))
        a_count = int(np.count_nonzero(fp & anatomy[y:y+bh, x:x+bw]))
        outside_count = int(np.count_nonzero(fp & ~valid[y:y+bh, x:x+bw]))
        center_gate = 0 <= cy < h and 0 <= cx < w and valid[cy, cx] and not anatomy[cy, cx]
        reasons = []
        if not center_gate:
            reasons.append("company_style_center_gate_failed")
        if p_count or hsv_count:
            reasons.append("porphyrin_object_conflict")
            p_conflict[y:y+bh, x:x+bw][pixels] = 255
            count["porphyrin_conflict"] += 1
        if a_count or outside_count:
            reasons.append("anatomy_or_region_whole_object_conflict")
            a_conflict[y:y+bh, x:x+bw][pixels] = 255
            count["anatomy_conflict"] += 1
        if not center_gate:
            region_rejected[y:y+bh, x:x+bw][pixels] = 255
            count["center_outside_region"] += 1
        if not reasons:
            accepted[y:y+bh, x:x+bw][pixels] = 255
            count["accepted"] += 1
        record.update(status="accepted" if not reasons else "conflict_or_region_rejected", accepted=not reasons,
                      reasons=reasons, center_gate_passed=bool(center_gate), bbox_xyxy=[x,y,x+bw,y+bh],
                      candidate_pixels=int(pixels.sum()), filled_footprint_pixels=int(fp.sum()),
                      retained_p_overlap_pixels=p_count, company_hsv_union_overlap_pixels=hsv_count,
                      anatomy_overlap_pixels=a_count, outside_region_pixels=outside_count,
                      median_dark_difference=float(np.median(difference[y:y+bh, x:x+bw][pixels])))
        records.append(record)
    conflict = cv2.bitwise_or(p_conflict, a_conflict)
    # An exact object rejection should already prevent any P overlap; there is
    # deliberately NO post-hoc P subtraction to manufacture this property.
    assert not np.any((accepted != 0) & ((company_union != 0) | p | anatomy | ~valid))
    unresolved = raw & (accepted == 0) & (conflict == 0)
    return {
        "raw_mask_native": raw.astype(np.uint8)*255,
        "accepted_mask_native": accepted,
        "porphyrin_conflict_mask_native": p_conflict,
        "anatomy_conflict_mask_native": a_conflict,
        "conflict_mask_native": conflict,
        "center_rejected_mask_native": region_rejected,
        "unresolved_mask_native": unresolved.astype(np.uint8)*255,
        "support_mask_native": support.astype(np.uint8)*255,
        "company_hsv_union_native": company_union,
        "records": records,
        "diagnostics": {
            "algorithm": "company-framework dark-candidate adaptation v1",
            "status": "development_candidate_not_validated_UV_spot_or_biological_ground_truth",
            "config": asdict(c), "native_shape_hw": [h,w], "work_shape_hw": [h,w],
            "resampling": "none; detector runs on native UV", "median_kernel_native": kernel,
            "company": company_meta, "company_hsv_bands": company_bands,
            "unchanged": ["findContours RETR_EXTERNAL/CHAIN_APPROX_SIMPLE and order", "area>=1 and moments centroid", "AST-executed original deduplicate_spots", "min_dist=10 native pixels", "center-gate logic"],
            "new": ["single HSV gate V>20, all H/S; not company dark-lesion thresholds", "single median neighborhood difference>=6; no RGB subtraction", "whole-neighborhood geometric support", "whole-object P/11-band-HSV and anatomical conflict isolation; no P pixel subtraction"],
            "parameter_status": "fixed initial development choices; not validation-selected or clinically validated",
            "conflict_semantics": "Conflicting objects remain exported in full original candidate support. Filled external contour, including holes, is used only to test conflict; company P unchanged. P/anatomy conflict layers may overlap each other.",
            "counts": {"all_contours": len(contours), "area_qualified": len(raw_spots), "after_company_dedup": len(retained), **count},
            "pixels": {"valid": int(valid.sum()), "support": int(support.sum()), "raw": int(raw.sum()),
                       "accepted": int(np.count_nonzero(accepted)), "porphyrin_conflict": int(np.count_nonzero(p_conflict)),
                       "anatomy_conflict": int(np.count_nonzero(a_conflict)), "conflict_union": int(np.count_nonzero(conflict)),
                       "unresolved": int(unresolved.sum())},
            "limitations": ["May select texture, moles, acne or shading; cannot identify their biology", "One median scale may miss broad/shallow lesions", "Company HSV color-candidate conflicts may reject genuine dark objects", "Unresolved or excluded pixels are not normal-skin labels", "No model training, RGB photometric selection, vendor changes or old-results changes"],
        },
        "maps_native": {"V": value.copy(), "median_background": background, "dark_difference": difference},
    }
