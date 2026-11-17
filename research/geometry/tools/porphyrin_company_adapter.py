"""Independent, read-only adapter for the supplied company point detector.

No V10/OTSU is used. Original thresholds, contour order and suppression remain
unchanged. Contour/HSV masks are diagnostic exports, NOT validated segmentation.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types

import cv2
import numpy as np


def load_vendor(path):
    """Load reviewed source, omitting only its unused proprietary import."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8-sig"), filename=str(path))
    removed = []
    for node in list(tree.body):
        if (isinstance(node, ast.ImportFrom)
                and node.module == "core.service.region_service.compute_region"):
            names = [a.asname or a.name for a in node.names]
            if names != ["Compute_region"] or any(
                isinstance(n, ast.Name) and n.id in names for n in ast.walk(tree)
            ):
                raise ValueError("Proprietary dependency is used; full company dependency required")
            tree.body.remove(node)
            removed.append(node.module + ".Compute_region")
    module = types.ModuleType("company_porphyrin_review")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    module._adapter_metadata = {
        "source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "removed_unused_import": removed,
        "entry": "detect_orange_spots_m (Porphyrin_detect business detection path)",
    }
    return module


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape != shape:
        raise ValueError(f"{name} must have the same H,W as the native image")
    if value.dtype not in (np.dtype(bool), np.dtype("uint8")):
        raise ValueError(f"{name} must be bool or uint8")
    return (value > 0).astype(np.uint8) * 255


def detect_uv(module, image_bgr, *, roi=None, t_mask=None, u_mask=None, min_dist=10.0):
    """Run the unchanged 11-band detector on native uint8 BGR.

    roi is an analysis-domain gate, not a company anatomical T/U partition.
    If it is used, T receives this gate and U is empty; no T/U or skin-type
    interpretation is made. No ROI means explicit full-frame diagnostic mode.
    Use a separate loaded module per concurrent worker (capture is local).
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3
            or min(image_bgr.shape[:2]) == 0):
        raise ValueError("image_bgr must be nonempty uint8 [H,W,3] BGR, not RGB/float")
    if not np.isfinite(min_dist) or min_dist < 0:
        raise ValueError("min_dist must be finite and nonnegative")
    shape = image_bgr.shape[:2]
    if (t_mask is None) != (u_mask is None):
        raise ValueError("Both t_mask and u_mask must be supplied together")
    if roi is not None and t_mask is not None:
        raise ValueError("Supply roi OR anatomical T/U masks, not both")
    if t_mask is not None:
        tm, um = _mask(t_mask, shape, "T"), _mask(u_mask, shape, "U")
        roi_mode = "provided_anatomical_T_U"
    else:
        tm = np.full(shape, 255, np.uint8) if roi is None else _mask(roi, shape, "ROI")
        um = np.zeros(shape, np.uint8)
        roi_mode = "full_frame_diagnostic" if roi is None else "analysis_ROI_no_anatomical_T_U"
    gate = cv2.bitwise_or(tm, um)
    captured = {}
    union = np.zeros(shape, np.uint8)
    bands = []
    original_dedup, original_cv2 = module.deduplicate_spots, module.cv2

    def capture_dedup(spots, min_dist=10.0):
        kept = original_dedup(spots, min_dist=min_dist)
        captured.update(raw_count=len(spots), kept=kept)
        return kept

    class CaptureCV2:
        def __getattr__(self, name):
            return getattr(original_cv2, name)

        def inRange(self, image, lower, upper):
            mask = original_cv2.inRange(image, lower, upper)
            original_cv2.bitwise_or(union, mask, dst=union)
            bands.append({"lower": np.asarray(lower).tolist(), "upper": np.asarray(upper).tolist()})
            return mask

    module.deduplicate_spots, module.cv2 = capture_dedup, CaptureCV2()
    try:
        marks, count, t_count, u_count = module.detect_orange_spots_m(
            image_bgr, tm, um, min_dist=float(min_dist)
        )
    finally:
        module.deduplicate_spots, module.cv2 = original_dedup, original_cv2
    if len(bands) != 11:
        raise RuntimeError("Unexpected company source: expected 11 HSV bands")
    valid = [s for s in captured["kept"] if gate[s["center"][1], s["center"][0]] > 0]
    if len(valid) != count or count != t_count + u_count:
        raise RuntimeError("Adapter trace differs from company point count")
    contour_mask = np.zeros(shape, np.uint8)
    if valid:
        cv2.drawContours(contour_mask, [s["contour"] for s in valid], -1, 255, cv2.FILLED)
    cv2.bitwise_and(contour_mask, gate, dst=contour_mask)
    cv2.bitwise_and(union, gate, dst=union)
    # Identical alpha repair to Porphyrin_detect; RGB circle pixels unchanged.
    marks[:, :, 3] = np.any(marks[:, :, :3] > 0, axis=2).astype(np.uint8) * 255
    spots = [{"center": [int(v) for v in s["center"]], "contour_area": float(s["area"]),
              "contour": s["contour"].reshape(-1, 2).tolist()} for s in valid]
    return {
        "point_count": int(count), "raw_count": captured["raw_count"],
        "kept_count": len(captured["kept"]), "roi_mode": roi_mode,
        "marks_bgra": marks, "hsv_mask": union, "contour_mask": contour_mask,
        "spots": spots, "hsv_bands": bands,
        "roi_pixels": int(np.count_nonzero(gate)),
        "vendor": module._adapter_metadata,
    }
