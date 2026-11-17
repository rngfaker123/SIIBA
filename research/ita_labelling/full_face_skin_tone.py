"""Versioned ITA on every native pixel of the already-approved final F.

Never edits a mask, detection rule, model, prior ITA label or source image.
Final F is NOT merely the old face oval: its frozen forehead cap uses UV
nonblack canvas support. Only the color values used for ITA are RGB-only.
Native-F reconstruction belongs to a separate replay helper; this module only
consumes it after verifying archived file/array SHA256 and native pixel counts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

try:
    from . import rgb_skin_tone as base
except ImportError:
    import rgb_skin_tone as base

BASE_SHA = 'ad37e8a0f63918f023a9a86c4a7469a25f1489308cc40fe3d4f5aa240caca05d'
METHOD = copy.deepcopy(base.METHOD)
for _key in ('roi_centers_landmark_mean', 'axis_landmarks', 'ellipse_radius_interoutereye',
             'min_native_pixels_per_patch', 'bilateral_ITA_disagreement_warning_degrees'):
    METHOD.pop(_key)
METHOD.update(
    version='rgb_ita_full_f_v1',
    base_color_implementation_sha256=BASE_SHA,
    min_native_pixels=256,
    aggregation='componentwise median physical Lab across ALL native final-F pixels; ITA of that single median Lab vector',
    geometry='exact archived final F: original RGB-preview face oval minus eye/lip holes, plus frozen UV-nonblack-clipped forehead cap; no re-inference',
    roi_is_rgb_only=False,
    color_values_are_rgb_only=True,
    pixel_selection='all native final-F pixels, including original eyebrow/nostril/hair support; no semantic skin filtering, lightness trimming, P/D/O selection, dilation or erosion',
    native_F_binding='must match original F PNG SHA256 AND uint8 array SHA256 AND native pixel count; never binarize 1024 coverage',
    qc_failure_policy='retain sample ID; null ITA with reason; does not exclude from original evaluation',
)

# 沿用确定性读写
require, encoded, digest, compact_digest = base.require, base.encoded, base.digest, base.compact_digest
sha, read, save, descriptor, bound = base.sha, base.read, base.save, base.descriptor, base.bound
checked_mask, srgb_to_lab, ita_from_lab, stratum = base.checked_mask, base.srgb_to_lab, base.ita_from_lab, base.stratum


def verify_color_source():
    require(sha(base.__file__) == BASE_SHA, 'Frozen ITA color implementation changed')


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native_evidence(rec):
    """Read only the original archived geometry/mask ledgers, not projected F."""
    require(rec['status'] == 'complete', 'Native F unavailable for technical failure')
    paths = {name: bound(rec['evidence'][name]) for name in
             ('geometry/landmarks.json', 'geometry/summary.json', 'mask/audit.json', 'mask/summary.json')}
    lm, gs, audit, ms = (read(paths[name]) for name in paths)
    require(gs['status'] == ms['status'] == 'complete', 'Incomplete native evidence')
    require(gs['native_shape'] == ms['native_shape'] == rec['native_shape'], 'Native evidence shape mismatch')
    require(gs['artifact_sha256']['landmarks.json'] == rec['evidence']['geometry/landmarks.json']['sha256'], 'Geometry ledger binding mismatch')
    require(ms['artifact_sha256']['audit.json'] == rec['evidence']['mask/audit.json']['sha256'], 'Mask audit ledger binding mismatch')
    require(ms['input_sha256']['landmarks'] == rec['evidence']['geometry/landmarks.json']['sha256'], 'Mask/geometry binding mismatch')
    require(ms['input_sha256']['uv'] == lm['uv_sha256'] == rec['sources']['uv']['sha256'], 'Native UV binding mismatch')
    require(ms['input_sha256']['old_valid'] == lm['old_valid_file_sha256'] == gs['artifact_sha256']['old_valid_native.png'], 'Old-F file ledger mismatch')
    require(ms['counts']['F'] == audit['stages']['F']['pixels'] == rec['counts_native']['F'], 'Final-F count ledgers differ')
    require(ms['array_sha256']['F'] == audit['stages']['F']['array_sha256'], 'Final-F array ledgers differ')
    require(ms['artifact_sha256']['F.png'] == rec['native_mask_file_sha256']['F'], 'Final-F file ledgers differ')
    return lm, audit, ms


def verify_f_array(raw, rec, audit, ms):
    require(raw.dtype == np.uint8 and raw.shape == tuple(rec['native_shape']) and
            np.all((raw == 0) | (raw == 255)), 'Require binary native uint8 F; projected weights are forbidden')
    require(int(np.count_nonzero(raw)) == rec['counts_native']['F'], 'Final-F pixel count mismatch')
    require(array_sha(raw) == audit['stages']['F']['array_sha256'] == ms['array_sha256']['F'], 'Final-F array SHA mismatch')
    return raw


def load_native_f(path, rec, audit, ms):
    path = Path(path)
    require(sha(path) == rec['native_mask_file_sha256']['F'], 'Final-F PNG SHA mismatch')
    with Image.open(path) as image:
        require(image.mode == 'L', 'Final-F PNG must have original grayscale mode')
        raw = np.asarray(image).copy()
    return verify_f_array(raw, rec, audit, ms)


def evaluate_full_f(image, mask):
    """No UV/P/D/O color values are inputs; every selected native pixel counts."""
    require(image.mode == 'RGB' and mask.dtype == np.uint8 and mask.shape == (image.height, image.width)
            and np.all((mask == 0) | (mask == 255)), 'Native RGB/F type or geometry differs')
    pixels = np.asarray(image)[mask != 0]
    n = len(pixels)
    result = {'status': 'invalid', 'reason': None, 'ita_degrees': None, 'stratum': None,
              'n_native_pixels': n, 'qc_warnings': []}
    if n < METHOD['min_native_pixels']:
        result['reason'] = 'ita_full_f_too_small'
        return result
    clipped = float(np.any((pixels == 0) | (pixels == 255), axis=1).mean())
    lab = srgb_to_lab(pixels)
    median = np.median(lab, axis=0)
    result.update(exact_clipped_fraction=clipped, aggregate_lab=median.tolist(),
                  lab_q10=np.percentile(lab, 10, axis=0).tolist(),
                  lab_q90=np.percentile(lab, 90, axis=0).tolist())
    if clipped > METHOD['max_exact_0_or_255_channel_clipped_fraction']:
        result['reason'] = 'ita_full_f_excessive_channel_clipping'
    elif median[2] <= METHOD['minimum_patch_median_b_star_exclusive']:
        result['reason'] = 'ita_full_f_unstable_or_nonpositive_b_star'
    else:
        ita = ita_from_lab(median)
        result.update(status='ok', ita_degrees=ita, stratum=stratum(ita))
    return result


def preview(image, mask, destination, sample_id):
    """Show exact F support; preview scaling is never used for color analysis."""
    w, h = image.size
    scale = min(1., 1600 / w)
    size = (round(w * scale), round(h * scale))
    resampling = getattr(Image, 'Resampling', Image)
    canvas = image.resize(size, resampling.LANCZOS)
    mask_small = np.asarray(Image.fromarray(mask).resize(size, resampling.NEAREST)) != 0
    rgb = np.asarray(canvas).copy()
    rgb[mask_small] = np.rint(.8 * rgb[mask_small] + .2 * np.array([255, 220, 0])).astype(np.uint8)
    canvas = Image.fromarray(rgb)
    ImageDraw.Draw(canvas).text((10, 10), sample_id + ' | exact native final F (preview only)', fill='white')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        canvas.save(stream, format='PNG')


def process_case(row, index, mask_path, protocol_sha, preview_path=None, source_root=None, *, native_f_path=None):
    verify_color_source()
    rec = checked_mask(mask_path)
    require(rec.get('schema') == 'uv_three_region_test_mask_case_v1' and
            rec['sample_id'] == row['sample_id'] and rec['subject_id'] == row['subject_id'] and
            rec['index'] == index and rec['native_shape'] == row['native_shape'] and
            rec['source_row_sha256'] == digest(row), 'Mask/sample identity mismatch')
    expected = Path(row['rgb']['path'])
    if not expected.is_absolute():
        require(source_root is not None, 'Relative source requires declared data_root')
        expected = Path(source_root) / expected
    require(rec['sources']['rgb'] == {'path': str(expected.resolve()), 'sha256': row['rgb']['sha256']}, 'RGB source descriptor mismatch')
    result = {'sample_id': row['sample_id'], 'subject_id': row['subject_id'], 'index': index,
              'protocol_sha256': protocol_sha, 'mask_receipt': descriptor(mask_path), 'rgb': rec['sources']['rgb'],
              'status': 'invalid', 'reason': None, 'ita_degrees': None, 'stratum': None,
              'method': METHOD['version'], 'roi_is_rgb_only': False, 'color_values_are_rgb_only': True}
    if rec['status'] != 'complete':
        require(rec['status'] in ('invalid_geometry', 'no_unique_face'), 'Unexpected mask failure')
        result['reason'] = 'upstream_' + rec['status']
        return result
    lm, audit, ms = native_evidence(rec)
    result['geometry'] = rec['evidence']['geometry/landmarks.json']
    result['landmark_source'] = lm['source']['landmark_source']
    if result['landmark_source'] != 'RGB_preview':
        require(result['landmark_source'] == 'UV_preview_fallback', 'Unknown landmark source')
        result['reason'] = 'rgb_geometry_unavailable_uv_fallback_not_used'
        return result
    require(native_f_path is not None, 'Require archived-hash-verified native final F, not old-F or 1024 coverage')
    f = load_native_f(native_f_path, rec, audit, ms)
    result['native_f'] = {'png_sha256': rec['native_mask_file_sha256']['F'],
                          'array_sha256': array_sha(f), 'pixels': int(np.count_nonzero(f)),
                          'shape': list(f.shape), 'original_evidence': rec['evidence']['mask/audit.json']}
    rgb_path = bound(rec['sources']['rgb'])
    with Image.open(rgb_path) as original:
        require(original.mode == 'RGB' and list(reversed(original.size)) == row['native_shape'], 'Native RGB type/shape differs')
        require(original.getexif().get(274, 1) == 1, 'Unsupported EXIF orientation; no automatic transformation')
        icc = original.info.get('icc_profile')
        result['icc_sha256'] = hashlib.sha256(icc).hexdigest() if icc else None
        image = original.copy()
        if icc:
            image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile('sRGB'), renderingIntent=1, outputMode='RGB')
            result['color_interpretation'] = 'embedded_ICC_to_sRGB_relative_colorimetric'
        else:
            result['color_interpretation'] = 'assumed_sRGB_no_embedded_ICC'
        result.update(evaluate_full_f(image, f))
        if preview_path is not None:
            preview(image, f, preview_path, row['sample_id'])
            result['preview'] = descriptor(preview_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask-receipt', type=Path, required=True)
    parser.add_argument('--row', type=Path, required=True, help='Exact one-row JSON from frozen TEST catalog')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--native-f', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = process_case(read(args.row), args.index, args.mask_receipt, args.protocol_sha256,
                          preview_path=args.preview, source_root=args.source_root, native_f_path=args.native_f)
    record['record_sha256'] = digest(record)
    save(args.output, record)
    print(json.dumps({'status': record['status'], 'record': descriptor(args.output)}), flush=True)
