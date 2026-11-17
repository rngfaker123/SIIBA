"""Pinned MediaPipe geometry stage for the approved native MI mask adapter.

Input RGB preview must use the original native->PIL Lanczos->JPEG92/4:4:4
pipeline, with its hash bound to the native RGB. This stage uses a different
OpenCV/NumPy environment than dark-mask inference, deliberately.
"""
from pathlib import Path
import hashlib
import json

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from build_porphyrin_mediapipe_regions import (
    detect_mapped, make_region, polygon_indices, MODEL_SHA,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def create_detector(model_path):
    if (mp.__version__, cv2.__version__, np.__version__) != ('0.10.21', '4.11.0', '1.26.4'):
        raise RuntimeError('Geometry requires MediaPipe 0.10.21 / OpenCV 4.11.0 / NumPy 1.26.4')
    if sha256(model_path) != MODEL_SHA:
        raise ValueError('Frozen MediaPipe task file changed')
    root = Path(__file__).resolve().parents[1]
    if sha256(Path(__file__).with_name('build_porphyrin_mediapipe_regions.py')) != 'a63c338527290341e700ef47ca6bf2a2735944f80adff7e0e7b8e625498f088b':
        raise ValueError('Original geometry functions changed')
    if sha256(root / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py') != 'cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87':
        raise ValueError('Original landmark polygon definitions changed')
    return mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model_path).resolve()), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5))


def extract_geometry(detector, rgb_preview_path, native_uv_path):
    """No new registration, mirrored inference, clipping or failure fallback."""
    with Image.open(native_uv_path) as image:
        width, height = image.size
        uv_preview = image.convert('RGB').resize((1600, round(height * 1600 / width)), Image.Resampling.LANCZOS)
    with Image.open(rgb_preview_path) as image:
        rgb_preview = image.convert('RGB')
    if rgb_preview.size != (min(1600, width), max(1, round(height * min(1600, width) / width))):
        raise ValueError('Unexpected native RGB preview shape')
    faces, raw_faces, box = detect_mapped(detector, rgb_preview, 'center-square')
    source = 'RGB_preview'
    if len(faces) == 0:
        faces, raw_faces, box = detect_mapped(detector, uv_preview, 'center-square')
        source = 'UV_preview_fallback'
    if len(faces) != 1:
        raise ValueError('Expected exactly one face; no mask fabricated')
    normalized = faces[0]
    polygons = polygon_indices()
    valid, xy, raster, _, _ = make_region(normalized, width, height, polygons)
    landmarks = {'normalized_xyz': normalized.tolist(), 'native_xy': xy.tolist(),
                 'rasterized_native_xy': raster.tolist(),
                 'normalized_xyz_reference': 'complete preview/native frame',
                 'detector_normalized_xyz': raw_faces[0].tolist(),
                 'detector_input_bbox_preview_xyxy': box,
                 'source': {'native_shape_hw': [height, width], 'landmark_source': source},
                 'rgb_preview_sha256': sha256(rgb_preview_path), 'uv_sha256': sha256(native_uv_path),
                 'model_sha256': MODEL_SHA,
                 'runtime': {'mediapipe': mp.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__}}
    return valid, landmarks


def save_geometry(output_dir, valid, landmarks):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(valid).save(folder / 'old_valid_native.png')
    landmarks['old_valid_file_sha256'] = sha256(folder / 'old_valid_native.png')
    with (folder / 'landmarks.json').open('x', encoding='utf-8') as stream:
        json.dump(landmarks, stream, indent=2, allow_nan=False)
    summary = {'status': 'complete', 'native_shape': list(valid.shape),
               'input_sha256': {'rgb_preview': landmarks['rgb_preview_sha256'], 'uv': landmarks['uv_sha256'],
                                'model': landmarks['model_sha256']},
               'artifact_sha256': {name: sha256(folder / name) for name in ['old_valid_native.png', 'landmarks.json']},
               'source_sha256': sha256(__file__), 'runtime': landmarks['runtime']}
    with (folder / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb-preview', type=Path, required=True)
    parser.add_argument('--uv', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with create_detector(args.model) as detector:
        print(json.dumps(save_geometry(args.output, *extract_geometry(detector, args.rgb_preview, args.uv))), flush=True)
