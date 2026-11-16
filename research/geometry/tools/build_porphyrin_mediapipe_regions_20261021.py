"""Build independent MediaPipe geometric face regions for the 11 review cases.

No porphyrin detector, old ROI, threshold optimization, training, or test access.
Face oval minus both eye contours and the outer lip contour. No dilation,
eyebrow removal, manually drawn nostrils, semantic skin claims, or registration.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
CONNECTIONS = ROOT / "assets/skin_mi_subject1000_v1/face_mesh_connections.py"
DEFAULT_SOURCE = ROOT / "output/porphyrin_company_review_20261021_v1/results.json"
DEFAULT_RGB = ROOT / "output/roi_intersection_review_20261020"
DEFAULT_OUT = ROOT / "output/porphyrin_mediapipe_review_20261021_v1/regions"
DEFAULT_MODEL = ROOT / "output/porphyrin_mediapipe_runtime_20261021/face_landmarker.task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def connection_sets():
    """Read only literal edge sets from the existing official MediaPipe file."""
    names = {"FACEMESH_FACE_OVAL", "FACEMESH_LEFT_EYE", "FACEMESH_RIGHT_EYE", "FACEMESH_LIPS"}
    result = {}
    for node in ast.parse(CONNECTIONS.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                value = node.value
                if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "frozenset":
                    raise ValueError("Unexpected edge-set syntax")
                result[name] = set(map(tuple, ast.literal_eval(value.args[0])))
    if set(result) != names:
        raise ValueError("Missing required official edge sets")
    return result


def ordered_cycle(edges, start):
    adjacent = {}
    for a, b in edges:
        adjacent.setdefault(a, set()).add(b)
        adjacent.setdefault(b, set()).add(a)
    if start not in adjacent:
        raise ValueError("Starting landmark absent")
    result = [start]
    previous = None
    current = start
    while True:
        if len(adjacent[current]) != 2:
            raise ValueError("Contour is not a simple cycle")
        nxt = sorted(adjacent[current] - ({previous} if previous is not None else set()))[0]
        if nxt == start:
            break
        if nxt in result:
            raise ValueError("Unexpected repeated contour vertex")
        result.append(nxt)
        previous, current = current, nxt
    return result


def polygon_indices():
    edges = connection_sets()
    polygons = {
        "face_oval": ordered_cycle(edges["FACEMESH_FACE_OVAL"], 10),
        "left_eye": ordered_cycle(edges["FACEMESH_LEFT_EYE"], 263),
        "right_eye": ordered_cycle(edges["FACEMESH_RIGHT_EYE"], 33),
        "outer_lips": ordered_cycle(edges["FACEMESH_LIPS"], 61),
    }
    if [len(x) for x in polygons.values()] != [36, 16, 16, 20]:
        raise ValueError("Unexpected official contour lengths")
    if 78 in polygons["outer_lips"]:
        raise ValueError("Inner lips mistakenly selected")
    return polygons


def ensure_model(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        if temporary.exists():
            raise FileExistsError(temporary)
        with urllib.request.urlopen(MODEL_URL, timeout=90) as response, temporary.open("xb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if sha256(temporary) != MODEL_SHA:
            raise ValueError("Downloaded model hash differs from frozen official model")
        temporary.rename(path)
    if sha256(path) != MODEL_SHA:
        raise ValueError("Model hash differs")


def make_region(normalized, width, height, polygons):
    xy = normalized[:, :2] * np.array([width, height]) - 0.5
    integer_xy = np.floor(xy + 0.5).astype(np.int32)
    selected = integer_xy[sorted({p for indices in polygons.values() for p in indices})]
    if not np.isfinite(normalized).all() or (selected < 0).any() or (selected[:, 0] >= width).any() or (selected[:, 1] >= height).any():
        raise ValueError("Nonfinite or out-of-image polygon vertex: no automatic clipping")
    oval = np.zeros((height, width), np.uint8)
    excluded = np.zeros_like(oval)
    cv2.fillPoly(oval, [integer_xy[polygons["face_oval"]]], 255)
    for name in ("left_eye", "right_eye", "outer_lips"):
        cv2.fillPoly(excluded, [integer_xy[polygons[name]]], 255)
    mask = cv2.bitwise_and(oval, cv2.bitwise_not(excluded))
    if np.any((mask > 0) & ((oval == 0) | (excluded > 0))):
        raise AssertionError("Region set subtraction failed")
    return mask, xy, integer_xy, int(np.count_nonzero(oval)), int(np.count_nonzero((oval > 0) & (excluded > 0)))


def draw_boundaries(image, normalized, polygons):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    xy = normalized[:, :2] * np.array(image.size) - 0.5
    for name, indices in polygons.items():
        points = [tuple(map(float, xy[i])) for i in indices]
        color = (45, 235, 120) if name == "face_oval" else (255, 85, 210)
        draw.line(points + points[:1], fill=color, width=3)
    return image


def detect(detector, image):
    data = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=data))
    return [np.asarray([[p.x, p.y, p.z] for p in face], dtype=np.float64) for face in result.face_landmarks]


def detect_mapped(detector, image, view):
    width, height = image.size
    side = min(width, height)
    box = ((width - side) // 2, (height - side) // 2,
           (width - side) // 2 + side, (height - side) // 2 + side)
    if view == "full":
        box = (0, 0, width, height)
    raw_faces = detect(detector, image.crop(box))
    mapped = []
    for raw in raw_faces:
        face = raw.copy()
        face[:, 0] = (raw[:, 0] * (box[2] - box[0]) + box[0]) / width
        face[:, 1] = (raw[:, 1] * (box[3] - box[1]) + box[1]) / height
        face[:, 2] = raw[:, 2] * (box[2] - box[0]) / width
        mapped.append(face)
    return mapped, raw_faces, list(box)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rgb-review", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--landmark-view", choices=("full", "center-square"), default="full")
    args = parser.parse_args()
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Keep prior output immutable; choose another --output")
    source = read_json(args.source_results)
    rgb_summary = read_json(args.rgb_review / "review_summary.json")
    rgb_by_id = {x["sample_id"]: x for x in rgb_summary["cases"]}
    polygons = polygon_indices()
    ensure_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "company_porphyrin_mediapipe_geometry_v1",
        "scope": "11 development review cases; geometry only, no change to historical ROI or training",
        "source_results": str(args.source_results.resolve()),
        "source_results_sha256": sha256(args.source_results),
        "script_sha256": sha256(Path(__file__)),
        "model_path": str(args.model.resolve()), "model_url": MODEL_URL,
        "model_sha256": MODEL_SHA,
        "runtime": {"python": platform.python_version(), "mediapipe": mp.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "connections_path": str(CONNECTIONS.resolve()), "connections_sha256": sha256(CONNECTIONS),
        "polygons": polygons,
        "region_definition": "official face oval minus left eye, right eye, and outer lip polygons",
        "excluded_structures": ["left_eye", "right_eye", "outer_lips"],
        "eyebrows_excluded": False, "nostrils_excluded": False,
        "dilation": False, "erosion": False, "old_roi_used": False,
        "color_threshold_used": False, "detector_results_used_to_adjust_geometry": False,
        "landmark_source": "same-subject 1600px RGB preview from prior read-only review; UV preview only if RGB missing or 0 faces",
        "landmark_view": args.landmark_view,
        "landmark_preprocess": "all cases use the same integer center square if selected, only for landmark inference; then map landmark coordinates back to complete preview and native UV frame; company detector input is unchanged",
        "mapping": "native_xy = normalized_xy * [native_width,native_height] - 0.5; raster vertex floor(native_xy + 0.5)",
        "mapping_limit": "same-size modality coordinates are reused without registration; exact per-pixel RGB-UV alignment is not asserted",
        "model_options": {"delegate": "CPU", "num_faces": 2, "mode": "IMAGE", "min_face_detection_confidence": 0.5, "min_face_presence_confidence": 0.5, "min_tracking_confidence": 0.5},
        "failure_rule": "0 or >1 faces after allowed fallback => explicit failed case, no fabricated mask",
        "region_limit": "Geometric face region, not a semantic skin mask or anatomical truth; eyebrows, hair, nose and within-oval artifacts may remain",
    }
    write_json(args.output / "protocol.json", protocol)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve()), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    records = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for case in source["cases"]:
            started = time.monotonic()
            folder = args.output / case["folder"]
            folder.mkdir(exist_ok=False)
            record = {"ordinal": case["ordinal"], "folder": case["folder"], "sample_id": case["sample_id"], "description": case["description"], "width": case["width"], "height": case["height"], "native_shape_hw": [case["height"], case["width"]], "uv_sha256": case["uv_sha256"], "rgb_preview_sha256": None, "stress_only": case["stress_only"]}
            try:
                uv_path = Path(case["uv_source"])
                if sha256(uv_path) != case["uv_sha256"]:
                    raise ValueError("UV original hash changed")
                rgb_info = rgb_by_id[case["sample_id"]]
                if rgb_info["native_shape_hw"] != [case["height"], case["width"]] or rgb_info["source_hashes"]["uv"] != case["uv_sha256"]:
                    raise ValueError("RGB preview identity or native geometry binding mismatch")
                with Image.open(uv_path) as image:
                    if image.size != (case["width"], case["height"]):
                        raise ValueError("UV geometry mismatch")
                    uv_preview = image.convert("RGB").resize((1600, round(case["height"] * 1600 / case["width"])), Image.Resampling.LANCZOS)
                rgb_path = args.rgb_review / "assets" / (rgb_info["stem"] + "_rgb_original.jpg")
                rgb_image = None
                if rgb_path.exists():
                    with Image.open(rgb_path) as image:
                        rgb_image = image.convert("RGB")
                    if list(rgb_image.size) != rgb_info["preview_size_wh"] or rgb_image.width * case["height"] != rgb_image.height * case["width"]:
                        raise ValueError("RGB preview geometry mismatch")
                    faces, raw_faces, detection_box = detect_mapped(detector, rgb_image, args.landmark_view)
                    record.update(rgb_preview_path=str(rgb_path.resolve()), rgb_preview_sha256=sha256(rgb_path), rgb_original_sha256=rgb_info["source_hashes"]["rgb"], rgb_face_count=len(faces), landmark_source="RGB_preview")
                else:
                    faces = []
                    record.update(rgb_face_count=None, landmark_source="RGB_preview_missing")
                if len(faces) == 0:
                    faces, raw_faces, detection_box = detect_mapped(detector, uv_preview, args.landmark_view)
                    record.update(landmark_source="UV_preview_fallback", uv_fallback_face_count=len(faces))
                record.update(landmark_view=args.landmark_view, detector_input_bbox_preview_xyxy=detection_box)
                write_json(folder / "detections.json", {"sample_id": case["sample_id"], "landmark_source": record["landmark_source"], "detector_input_bbox_preview_xyxy": detection_box, "detector_normalized_xyz_all_faces": [x.tolist() for x in raw_faces], "full_preview_normalized_xyz_all_faces": [x.tolist() for x in faces]})
                if len(faces) != 1:
                    raise ValueError("Expected exactly one face, found %d" % len(faces))
                lm = faces[0]
                mask, xy, integer_xy, oval_pixels, removed_pixels = make_region(lm, case["width"], case["height"], polygons)
                Image.fromarray(mask).save(folder / "region_mask_native.png")
                Image.fromarray(mask).resize(uv_preview.size, Image.Resampling.NEAREST).save(folder / "region_mask_preview.png")
                draw_boundaries(uv_preview, lm, polygons).save(folder / "boundary_uv_preview.jpg", quality=95, subsampling=0)
                if rgb_image is not None:
                    draw_boundaries(rgb_image, lm, polygons).save(folder / "boundary_rgb_preview.jpg", quality=95, subsampling=0)
                record.update(status="ok", landmark_count=len(lm), region_pixels=int(np.count_nonzero(mask)), face_oval_pixels=oval_pixels, removed_eye_lip_pixels=removed_pixels, region_fraction_of_native_image=float(np.count_nonzero(mask) / mask.size), mask_path=str((folder / "region_mask_native.png").resolve()), mask_sha256=sha256(folder / "region_mask_native.png"), uv_source=str(uv_path.resolve()), uv_sha256=case["uv_sha256"])
                write_json(folder / "landmarks.json", {"schema": protocol["schema"], "source": record, "normalized_xyz": lm.tolist(), "normalized_xyz_reference": "complete preview/native frame", "detector_normalized_xyz": raw_faces[0].tolist(), "detector_input_bbox_preview_xyxy": detection_box, "native_xy": xy.tolist(), "rasterized_native_xy": integer_xy.tolist(), "polygons": {name: {"indices": ids, "native_xy": xy[ids].tolist(), "rasterized_native_xy": integer_xy[ids].tolist()} for name, ids in polygons.items()}, "mapping": protocol["mapping"], "model_sha256": MODEL_SHA})
                record["region_mask_sha256"] = record["mask_sha256"]
                record["landmarks_sha256"] = sha256(folder / "landmarks.json")
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(folder / "summary.json", record)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    result = {"schema": protocol["schema"], "status": "complete" if all(x["status"] == "ok" for x in records) else "complete_with_failures", "count": len(records), "success_count": sum(x["status"] == "ok" for x in records), "fallback_count": sum(x.get("landmark_source") == "UV_preview_fallback" for x in records), "protocol_sha256": sha256(args.output / "protocol.json"), "cases": records}
    write_json(args.output / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
