"""Independent UV three-candidate-mask review; preserves all historical files."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from uv_darkspot_module_v2 import DarkSpotConfig, detect_uv_darkspots
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/uv_three_masks_review_20261021_v1'
MP = ROOT / 'output/porphyrin_mediapipe_review_20261021_v1'
PAIRS = ROOT / 'output/rgb_uv_darkspot_pairs_20261021_v1'
VENDOR = Path('/study/vendor/porphyrin_module.py')
VENDOR_HASH = 'de1c4d52ab781d3440b694afb0d9480eb0af1359a365aa5290ae0d750a31a0e9'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_image(path, gray=False):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError('Image encoding failed')
    with Path(path).open('xb') as f:
        f.write(encoded.tobytes())


def partition_masks(porphyrin_mask, dark_mask, valid_mask):
    """Declare bright priority; retain raw overlap. Remainder is not normal skin."""
    inputs = (porphyrin_mask, dark_mask, valid_mask)
    if any(not isinstance(m, np.ndarray) or m.ndim != 2 or m.dtype not in (np.dtype('uint8'), np.dtype('bool')) for m in inputs):
        raise ValueError('Masks must be two-dimensional uint8 or bool arrays')
    if len({m.shape for m in inputs}) != 1:
        raise ValueError('Mask dimensions differ')
    valid = valid_mask != 0
    p = (porphyrin_mask != 0) & valid
    d = (dark_mask != 0) & valid
    return {name: value.astype(np.uint8) * 255 for name, value in {
        'porphyrin': p, 'uv_spot': d & ~p, 'other': valid & ~(p | d),
        'outside': ~valid, 'overlap_raw': p & d}.items()}


def porphyrin_pixel_support(uv, valid, spots, bands):
    """Derive pixels from retained vendor contours AND unchanged HSV candidates.

    Retain centers in the same MediaPipe gate; do not rerun detection/dedup or
    treat the display circles as lesion areas. This is an adapter definition,
    not a segmentation mask supplied or validated by the company.
    """
    hsv = cv2.cvtColor(uv, cv2.COLOR_BGR2HSV)
    color_union = np.zeros(valid.shape, np.uint8)
    for band in bands:
        color_union |= cv2.inRange(hsv, np.array(band['lower'], np.uint8), np.array(band['upper'], np.uint8))
    contour_support = np.zeros(valid.shape, np.uint8)
    retained = []
    h, w = valid.shape
    for spot in spots:
        x, y = spot['center']
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError('Company center out of image')
        if valid[y, x]:
            retained.append(spot)
            cv2.drawContours(contour_support, [np.array(spot['contour'], np.int32).reshape(-1, 1, 2)], -1, 255, cv2.FILLED)
    mask = contour_support & color_union
    mask[valid == 0] = 0
    return mask, retained


def color_layer(mask, bgr, size):
    alpha = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    layer = np.empty((*alpha.shape, 4), np.uint8)
    layer[..., :3] = bgr
    layer[..., 3] = alpha
    return layer


def build_case(c, g, pair):
    name = c['folder']
    folder = OUT / name
    folder.mkdir()
    for key in ('sample_id', 'ordinal', 'width', 'height', 'uv_sha256'):
        if c[key] != g[key]:
            raise ValueError(f'Geometry binding mismatch {name}/{key}')
    if c['sample_id'] != pair['sample_id'] or c['uv_sha256'] != pair['source_original_sha256']['uv']:
        raise ValueError('RGB/UV pair binding mismatch')
    if sha(c['uv_source']) != c['uv_sha256'] or sha(g['mask_path']) != g['region_mask_sha256'] or g['status'] != 'ok':
        raise ValueError('Source checksum mismatch')
    if sha(PAIRS / pair['rgb']) != pair['source_preview_sha256']['rgb']:
        raise ValueError('RGB preview checksum mismatch')
    uv, valid = read_image(c['uv_source']), read_image(g['mask_path'], True)
    raw_spots_path = MP / 'detection' / name / 'spots.json'
    raw_spots = read_json(raw_spots_path)
    p, retained = porphyrin_pixel_support(uv, valid, raw_spots, c['hsv_bands'])
    dark = detect_uv_darkspots(uv, valid, DarkSpotConfig())
    masks = partition_masks(p, dark['mask_native'], valid)
    membership = sum((masks[k] != 0).astype(np.uint8) for k in ('porphyrin', 'uv_spot', 'other'))
    if not np.array_equal(membership, (valid != 0).astype(np.uint8)):
        raise AssertionError('Three masks not disjoint and exhaustive within valid geometry')
    support = cv2.resize(dark['maps_work']['median_support'], (uv.shape[1], uv.shape[0]), interpolation=cv2.INTER_NEAREST)
    unsupported = ((valid != 0) & (support == 0)).astype(np.uint8) * 255
    masks.update({'dark_raw': dark['mask_native'], 'dark_candidates': dark['candidate_mask_native'],
                  'valid': valid, 'dark_unassessed': unsupported})
    labels = (masks['porphyrin'] != 0).astype(np.uint8) + 2 * (masks['uv_spot'] != 0).astype(np.uint8) + 3 * (masks['other'] != 0).astype(np.uint8)
    write_image(folder / 'labels_native.png', labels)
    mask_hashes = {}
    for key, mask in masks.items():
        path = folder / f'{key}_native.png'
        write_image(path, mask)
        mask_hashes[key] = sha(path)
    write_json(folder / 'dark_diagnostics.json', dark['diagnostics'])
    write_json(folder / 'dark_components.json', dark['spots'])
    write_json(folder / 'dark_rejected_components.json', dark['rejected_spots'])
    write_json(folder / 'porphyrin_retained_spots.json', retained)
    size = (3200, 1600)
    uv_small = cv2.resize(uv, size, interpolation=cv2.INTER_AREA)
    write_image(folder / 'uv.jpg', uv_small)
    shutil.copyfile(PAIRS / pair['rgb'], folder / 'rgb.jpg')
    colors = {'porphyrin': (40, 120, 255), 'uv_spot': (255, 210, 0), 'dark_raw': (255, 210, 0),
              'dark_candidates': (160, 220, 250), 'other': (170, 170, 170), 'overlap_raw': (240, 0, 240),
              'dark_unassessed': (100, 70, 180)}
    for key, color in colors.items():
        write_image(folder / f'{key}_layer.png', color_layer(masks[key], color, size))
    label_color = np.zeros((*valid.shape, 3), np.uint8)
    for key in ('porphyrin', 'uv_spot', 'other'):
        label_color[masks[key] != 0] = colors[key]
    write_image(folder / 'three_masks.png', cv2.resize(label_color, size, interpolation=cv2.INTER_NEAREST))
    boundary = np.zeros((*uv_small.shape[:2], 4), np.uint8)
    small_valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small_valid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, (255, 255, 255, 220), 2)
    write_image(folder / 'boundary.png', boundary)
    counts = {k: int(np.count_nonzero(v)) for k, v in masks.items()}
    fractions = {k: counts[k] / counts['valid'] for k in masks if k != 'outside'}
    record = {'folder': name, 'ordinal': c['ordinal'], 'sample_id': c['sample_id'], 'stress_only': c['stress_only'],
              'native_shape_hw': list(valid.shape), 'uv_source': c['uv_source'], 'uv_sha256': c['uv_sha256'],
              'region_source': g['mask_path'], 'region_sha256': g['region_mask_sha256'],
              'company_spots_sha256': sha(raw_spots_path), 'company_point_count': len(retained),
              'dark_component_count_before_priority': len(dark['spots']),
              'counts': counts, 'fractions': fractions, 'mask_sha256': mask_hashes,
              'rgb_preview_sha256': pair['source_preview_sha256']['rgb'],
              'rgb_native_source_sha256': pair['source_original_sha256']['rgb'],
              'focus': pair['focus'], 'description': pair['description']}
    write_json(folder / 'summary.json', record)
    # Honest focus preview: original RGB, original UV, dark raw overlay, and
    # the exported exclusive three-class mask; no new image enhancement.
    x0, y0, x1, y1 = pair['focus']
    rgb_small = read_image(folder / 'rgb.jpg')
    crop_uv = uv_small[y0*2:y1*2, x0*2:x1*2]
    layer = read_image(folder / 'uv.jpg')  # source unchanged; overlay separate
    alpha = cv2.resize(masks['dark_raw'], size, interpolation=cv2.INTER_AREA)[y0*2:y1*2, x0*2:x1*2].astype(np.float32)[..., None] / 255 * .8
    overlay = np.rint(crop_uv * (1-alpha) + np.array(colors['dark_raw'])*(alpha)).astype(np.uint8)
    label_preview = cv2.resize(label_color, size, interpolation=cv2.INTER_NEAREST)[y0*2:y1*2, x0*2:x1*2]
    crops = [rgb_small[y0:y1, x0:x1], crop_uv, overlay, label_preview]
    tile_w = 440
    tile_h = round(tile_w * (y1-y0)/(x1-x0))
    board = Image.new('RGB', (4 * tile_w, tile_h + 45), 'white')
    drawer = ImageDraw.Draw(board)
    for index, (img, title) in enumerate(zip(crops, ['RGB', 'UV original', 'UV dark candidates', 'P / dark / remainder'])):
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        board.paste(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)), (index*tile_w, 45))
        drawer.text((index*tile_w+10, 12), f'{name} | {title}', fill='black')
    board.save(folder / 'focus_comparison.jpg', quality=94)
    return record


def main():
    if OUT.exists():
        raise FileExistsError('Use a fresh output directory; no overwriting historical results')
    if sha(VENDOR) != VENDOR_HASH:
        raise ValueError('Company source has changed; stop')
    cv2.setNumThreads(2)
    src = MP / 'detection/results.json'
    geo = MP / 'regions_center_square/results.json'
    pairs_path = PAIRS / 'pairs.json'
    cases = read_json(src)['cases']
    geometries = {c['folder']: c for c in read_json(geo)['cases']}
    pairs = {p['ordinal']: p for p in read_json(pairs_path)}
    OUT.mkdir(parents=True)
    (OUT / 'source').mkdir()
    for path in (Path(__file__), ROOT / 'tools/uv_darkspot_module_v2.py', VENDOR, ROOT / 'tools/uv_three_masks_viewer_template.html'):
        shutil.copyfile(path, OUT / 'source' / path.name)
    write_json(OUT / 'protocol.json', {
        'schema': 'uv_three_masks_visual_prototype_v1', 'status': 'candidate_masks_not_validated_ground_truth',
        'vendor_sha256': VENDOR_HASH, 'company_rules_modified': False, 'training_or_frozen_results_modified': False,
        'company_cached_detection': str(src), 'company_results_sha256': sha(src),
        'geometry_results_sha256': sha(geo), 'pair_manifest_sha256': sha(pairs_path),
        'dark_config': asdict(DarkSpotConfig()), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'dark_module_sha256': sha(ROOT / 'tools/uv_darkspot_module_v2.py'), 'runner_sha256': sha(__file__),
        'dark_input': 'UV only plus previously frozen MediaPipe face-oval-minus-eyes-and-lips geometry; no RGB pixel values, no porphyrin mask',
        'bright_pixel_mask': 'Adapter-derived: retained company external contours AND unchanged 11-band HSV union AND supplied geometric mask; original circles are display markers only',
        'partition': 'P=bright; D=raw_dark minus P; other=valid minus(P union raw_dark); raw overlap exported; bright priority is engineering convention',
        'other_meaning': 'not assigned to the two candidate types, NOT established normal skin; includes dark-unassessed boundary',
        'dark_unassessed': 'median 25x25 support incomplete; exported separately, not a fourth biological class',
        'known_limitations': ['Dark cores rather than full lesion extents', 'Hair, moles, shadows, texture may be selected', 'Weak or large spots may be missed', 'Geometry omits peripheral skin and does not explicitly exclude eyebrows or nostrils'],
        'stress_only': [18, 20], 'no_accuracy_claim': True,
        'sources': 'native UV; RGB visual reference is an existing 1600x800 preview with original-source identity hash; no extra registration or enhancement',
    })
    records = []
    start = time.monotonic()
    for c in cases:
        record = build_case(c, geometries[c['folder']], pairs[c['ordinal']])
        records.append(record)
        print(json.dumps({'case': record['folder'], 'dark_components': record['dark_component_count_before_priority'], 'fractions': record['fractions']}, ensure_ascii=False), flush=True)
        gc.collect()
    if sha(VENDOR) != VENDOR_HASH:
        raise AssertionError('Company source changed during run')
    result = {'status': 'complete', 'cases': records, 'count': len(records), 'elapsed_seconds': round(time.monotonic()-start, 3),
              'protocol_sha256': sha(OUT / 'protocol.json'), 'company_source_unchanged': True}
    write_json(OUT / 'results.json', result)
    html = (ROOT / 'tools/uv_three_masks_viewer_template.html').read_text(encoding='utf-8').replace('__CASES_JSON__', json.dumps(records, ensure_ascii=False).replace('</', '<\\/'))
    with (OUT / 'viewer.html').open('x', encoding='utf-8') as f:
        f.write(html)
    print(json.dumps({'output': str(OUT), 'count': len(records), 'seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
