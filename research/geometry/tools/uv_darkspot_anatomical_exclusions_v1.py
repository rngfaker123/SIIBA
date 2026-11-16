"""Dark-spot-only geometric exclusions from already cached MediaPipe points.

These are conservative engineering guard regions, not anatomical segmentation
truth. The company porphyrin detector and its masks are never read or modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CONNECTIONS = ROOT / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py'
REGIONS = ROOT / 'output/porphyrin_mediapipe_review_20261021_v1/regions_center_square'
DEFAULT_OUTPUT = ROOT / 'output/uv_darkspot_anatomy_20261021_v1'
NOSTRIL_ANCHORS = {
    # Locally inspected lower-nasal mesh anchors; these are not official
    # MediaPipe nostril segmentation contours. The nose tip/bridge are omitted.
    'right_nostril_guard': [218, 237, 44, 20, 242, 99, 97, 240, 219],
    'left_nostril_guard': [438, 457, 274, 250, 462, 328, 326, 460, 439],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def official_eye_brow_ids():
    wanted = {'FACEMESH_LEFT_EYE', 'FACEMESH_RIGHT_EYE', 'FACEMESH_LEFT_EYEBROW', 'FACEMESH_RIGHT_EYEBROW'}
    found = {}
    for node in ast.parse(CONNECTIONS.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted:
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != 'frozenset':
                raise ValueError('Unexpected official connection syntax')
            edges = ast.literal_eval(node.value.args[0])
            found[node.targets[0].id] = sorted({i for edge in edges for i in edge})
    if set(found) != wanted:
        raise ValueError('Missing official eye/brow connection set')
    return found


def _filled_guard(xy, ids, shape_hw, margin):
    vertices = np.floor(xy[ids] + .5).astype(np.int32)
    hull = cv2.convexHull(vertices).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) <= 0:
        raise ValueError('Degenerate landmark guard region')
    mask = np.zeros(shape_hw, np.uint8)
    # Exact local rasterization avoids applying a large structuring element to
    # an otherwise empty 32-million-pixel native frame.
    h, w = shape_hw
    x0 = max(0, int(hull[:, 0].min()) - margin - 1)
    y0 = max(0, int(hull[:, 1].min()) - margin - 1)
    x1 = min(w, int(hull[:, 0].max()) + margin + 2)
    y1 = min(h, int(hull[:, 1].max()) + margin + 2)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Guard outside image')
    local = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillPoly(local, [hull - np.array([x0, y0])], 255)
    if margin:
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    mask[y0:y1, x0:x1] = local
    return mask, hull


def build_dark_exclusions(landmarks_dict, shape_hw):
    """Return mask_native, four component_masks, and transparent diagnostics.

    Upper eyelids are covered by the hull of the corresponding eye and eyebrow
    points, not merely the very narrow closed-eye contour. The nostril guards
    are small separate lower-nasal hulls, never the entire nose.
    """
    if len(shape_hw) != 2 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in shape_hw):
        raise ValueError('shape_hw must be positive integer H,W')
    shape_hw = tuple(map(int, shape_hw))
    if 'source' in landmarks_dict and 'native_shape_hw' in landmarks_dict['source']:
        if tuple(landmarks_dict['source']['native_shape_hw']) != shape_hw:
            raise ValueError('Cached landmark source shape differs')
    xy = np.asarray(landmarks_dict['native_xy'], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] < 468 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite native [N>=468,2] landmark array')
    groups = official_eye_brow_ids()
    component_masks = {}
    components = {}
    for side, corners in [('left', (263, 362)), ('right', (33, 133))]:
        eye = groups[f'FACEMESH_{side.upper()}_EYE']
        brow = groups[f'FACEMESH_{side.upper()}_EYEBROW']
        ids = sorted(set(eye + brow))
        eye_width = float(np.linalg.norm(xy[corners[0]] - xy[corners[1]]))
        if eye_width <= 1:
            raise ValueError('Degenerate eye-width anchor')
        margin = max(1, round(.04 * eye_width))
        name = f'{side}_eye_brow_guard'
        component_masks[name], hull = _filled_guard(xy, ids, shape_hw, margin)
        components[name] = {'landmark_ids': ids, 'hull_native_xy': hull.tolist(), 'margin_native_pixels': margin,
                            'margin_rule': 'round(0.04 * distance between eye corners), minimum 1 pixel',
                            'includes': 'eye contour, upper eyelid between eye and brow, eyebrow, small guard margin'}
    nose_width = float(np.linalg.norm(xy[98] - xy[327]))
    if nose_width <= 1:
        raise ValueError('Degenerate nasal width anchor')
    nasal_margin = max(1, round(.035 * nose_width))
    for name, ids in NOSTRIL_ANCHORS.items():
        component_masks[name], hull = _filled_guard(xy, ids, shape_hw, nasal_margin)
        components[name] = {'landmark_ids': ids, 'hull_native_xy': hull.tolist(), 'margin_native_pixels': nasal_margin,
                            'margin_rule': 'round(0.035 * distance between lower nasal anchors 98 and 327), minimum 1 pixel',
                            'includes': 'engineering lower-nasal guard for nostril cavity and nearby nasal hair; not all nose'}
    used = sorted({i for c in components.values() for i in c['landmark_ids']})
    h, w = shape_hw
    if (xy[used] < -.5).any() or (xy[used, 0] >= w - .5).any() or (xy[used, 1] >= h - .5).any():
        raise ValueError('Guard anchors outside source image; no fabricated geometry')
    mask = np.zeros(shape_hw, np.uint8)
    for m in component_masks.values():
        mask |= m
    return {'mask_native': mask, 'component_masks': component_masks,
            'diagnostics': {'schema': 'dark_spot_anatomy_guards_v1', 'native_shape_hw': list(shape_hw),
                            'source': 'previously cached RGB MediaPipe landmarks, same native coordinate mapping; no new landmark inference',
                            'application': 'dark-spot branch only; does not change company porphyrin or existing analysis mask',
                            'components': components, 'total_guard_pixels': int(np.count_nonzero(mask)),
                            'component_guard_pixels': {k: int(np.count_nonzero(m)) for k, m in component_masks.items()},
                            'parameters': 'fixed engineering margins for all 11 development cases; not validated anatomical ground truth',
                            'limitations': ['RGB-to-UV registration and landmark errors can shift guards', 'Exclusion removes some genuine skin near eyes and nostrils from dark-spot assessment', 'Hair outside the guard and other shadows can remain', 'Nostril anchors are engineering selections, not official labelled nostril contours', 'Excluded pixels are unassessed for dark spots, not proven normal skin']}}


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_image(path, array):
    ok, encoded = cv2.imencode(Path(path).suffix, array)
    if not ok:
        raise ValueError('Image encode failed')
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous output; choose a new output directory')
    cv2.setNumThreads(2)
    source_path = REGIONS / 'results.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True)
    (args.output / 'source').mkdir()
    shutil.copyfile(__file__, args.output / 'source' / Path(__file__).name)
    protocol = {'schema': 'uv_darkspot_anatomy_review_v1', 'source_results': str(source_path), 'source_results_sha256': sha(source_path),
                'source_code_sha256': sha(__file__), 'official_connections_path': str(CONNECTIONS), 'official_connections_sha256': sha(CONNECTIONS),
                'new_mediapipe_inference': False, 'company_detector_or_existing_masks_modified': False,
                'opencv': cv2.__version__, 'numpy': np.__version__, 'case_order': [c['folder'] for c in source['cases']],
                'rules': 'convex hull of official eye+brow anchors, margin 4% eye width; paired lower-nasal anchor hulls, margin 3.5% nasal width',
                'warning': 'Engineering exclusion region for dark branch, not true anatomy segmentation or normal skin mask'}
    _write_json(args.output / 'protocol.json', protocol)
    records = []
    for case in source['cases']:
        name = case['folder']
        folder = args.output / name
        folder.mkdir()
        lm_path = REGIONS / name / 'landmarks.json'
        if case['status'] != 'ok' or sha(lm_path) != case['landmarks_sha256']:
            raise ValueError('Landmark source binding mismatch')
        lm = json.loads(lm_path.read_text(encoding='utf-8'))
        shape = (case['height'], case['width'])
        result = build_dark_exclusions(lm, shape)
        mask_path = folder / 'anatomical_exclusion_native.png'
        _write_image(mask_path, result['mask_native'])
        for key, mask in result['component_masks'].items():
            _write_image(folder / f'{key}_native.png', mask)
        _write_json(folder / 'diagnostics.json', result['diagnostics'])
        uv_path = ROOT / 'output/roi_intersection_review_20261020/assets' / f'{name}_uv_original.jpg'
        uv = cv2.imdecode(np.fromfile(uv_path, np.uint8), cv2.IMREAD_COLOR)
        if uv is None:
            raise ValueError('Missing bound UV preview')
        small = cv2.resize(result['mask_native'], (1600, 800), interpolation=cv2.INTER_NEAREST)
        layer = uv.copy()
        for key, mask in result['component_masks'].items():
            color = (160, 90, 250) if 'eye' in key else (70, 210, 255)
            resized = cv2.resize(mask, (1600, 800), interpolation=cv2.INTER_NEAREST)
            active = resized != 0
            layer[active] = np.rint(layer[active] * .55 + np.array(color) * .45).astype(np.uint8)
            contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(layer, contours, -1, color, 2)
        _write_image(folder / 'anatomical_exclusion_preview.png', small)
        _write_image(folder / 'uv_anatomy_overlay.jpg', layer)
        panel = np.concatenate([uv[:, 300:1300], layer[:, 300:1300]], axis=1)
        _write_image(folder / 'full_face_before_after.jpg', panel)
        record = {'folder': name, 'sample_id': case['sample_id'], 'ordinal': case['ordinal'], 'native_shape_hw': list(shape),
                  'uv_sha256': case['uv_sha256'], 'landmarks_sha256': sha(lm_path), 'existing_valid_sha256': case['region_mask_sha256'],
                  'mask_path': str(mask_path.resolve()), 'mask_sha256': sha(mask_path),
                  'guard_pixels': result['diagnostics']['total_guard_pixels'], 'diagnostics_path': str((folder / 'diagnostics.json').resolve())}
        _write_json(folder / 'summary.json', record)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    _write_json(args.output / 'results.json', {'schema': protocol['schema'], 'status': 'complete', 'count': len(records),
                                             'protocol_sha256': sha(args.output / 'protocol.json'), 'cases': records})


if __name__ == '__main__':
    main()
