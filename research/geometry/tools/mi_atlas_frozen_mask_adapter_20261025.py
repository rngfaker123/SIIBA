"""Frozen native three-mask pipeline for MI; no detector/threshold redesign.

The approved endpoint is uv_porphyrin_priority_review_20261024_v1. This adapter
calls the byte-bound original functions in their historical order; it does not
substitute its final candidate detector for the earlier baseline branch.
Geometry and mask processing use separately pinned runtimes. See source hash
checks and the independent 11-case pixel replay before using new samples.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import gc
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
VENDOR_SHA = 'de1c4d52ab781d3440b694afb0d9480eb0af1359a365aa5290ae0d750a31a0e9'
CONNECTIONS_SHA = 'cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87'
FROZEN_SHA = {
    'porphyrin_company_adapter_20261021.py': 'b3a90168162d5224b79cc7682dc2749c0abad886db5d178aa7537941eca45783',
    # Public copy points to the supplied vendor module under /study/vendor.
    'build_uv_three_masks_review_20261021.py': '44f21a26a36c41db67547a5789fc8ea2c89100b809dd7a319e280da061016a14',
    'build_porphyrin_mediapipe_regions_20261021.py': 'a63c338527290341e700ef47ca6bf2a2735944f80adff7e0e7b8e625498f088b',
    'uv_darkspot_anatomical_exclusions_v1.py': 'f79295f0689e5a0c282382496b40c25c4acd3c7bbbf8e41a79a53eb421027291',
    'uv_darkspot_module_v4.py': '91a6cfbf236fabaaff8ebc30c88d1333ca7f897c6377c20fc630c46d9856da4b',
    'uv_darkspot_refinement_v5.py': '51f968650f6214515613fe97dd15824c83ea26ee262a2de5ba5827b39fcaa4f8',
    'uv_darkspot_native_red_guard_v1.py': 'c81a7a82dfb3f40892ac5026cacb36993b905d27a1c716297207b382e10c8cb0',
    'uv_spot_local_contrast_filter_20261024.py': 'a9a7d8f128eefa732c6da0b2934107eb42f2c98398ff3050859ee4a01386980c',
    'uv_forehead_domain_20261024.py': '20e5b5687c7fd16b28843a98d756475f7ab685e1264066d4f914717964b36eaf',
    'uv_spots_company_v2.py': 'c05ff90e7363b739c48ec2e03d9ffa18ff458de693c57b55f6d6f94c4ffd9c20',
    'uv_darkspot_company_based_v1.py': 'c937a4f31cabc1518ca5ea2e8aadfba07bceda0df79280775283bcbb2dc2c11d',
    'uv_spot_recovery_20261024.py': 'dee7a9ad91f9b49138247857fc1b6ed43536f106812d8fc0ac605649e9388bce',
    'uv_porphyrin_priority_20261024.py': '2d0322b8d341db69829704af06c70b101a62099711bed61f6e03e65f86cf899f',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(vendor_path=None, *, require_runtime=True):
    """Fail closed if a production source or numerical runtime changed."""
    for name, expected in FROZEN_SHA.items():
        if sha256(TOOLS / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    connections = ROOT / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py'
    if sha256(connections) != CONNECTIONS_SHA:
        raise ValueError('Frozen MediaPipe connections changed')
    vendor = Path(vendor_path) if vendor_path else TOOLS / 'porphyrin_module.py'
    if not vendor.exists() and vendor_path is None:
        vendor = ROOT / 'output/porphyrin_company_review_20261021_v1/source/porphyrin_module.py'
    if sha256(vendor) != VENDOR_SHA:
        raise ValueError('Company module differs from approved source')
    if require_runtime and (cv2.__version__ != '4.10.0' or np.__version__ != '2.3.4'):
        raise RuntimeError('Mask pipeline requires approved OpenCV 4.10.0 / NumPy 2.3.4; geometry is a separate runtime')
    return vendor


def _original_functions(filename, names, scope=None):
    """Execute unchanged named functions, without importing viewer/MP runtimes."""
    path = TOOLS / filename
    if sha256(path) != FROZEN_SHA[filename]:
        raise ValueError('Frozen function source changed: ' + filename)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Required original functions missing')
    namespace = {'np': np, 'cv2': cv2, 'ast': ast,
                 'CONNECTIONS': ROOT / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py'}
    namespace.update(scope or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def geometry_from_landmarks(landmarks, shape_hw):
    """Rasterize original normalized landmarks; never resize an existing mask."""
    if cv2.__version__ != '4.11.0' or np.__version__ != '1.26.4':
        raise RuntimeError('Old-F rasterization requires geometry OpenCV 4.11.0 / NumPy 1.26.4')
    funcs = _original_functions('build_porphyrin_mediapipe_regions_20261021.py',
                                ['connection_sets', 'ordered_cycle', 'polygon_indices', 'make_region'])
    h, w = map(int, shape_hw)
    normalized = np.asarray(landmarks['normalized_xyz'], dtype=np.float64)
    valid, xy, _, _, _ = funcs['make_region'](normalized, w, h, funcs['polygon_indices']())
    if 'native_xy' in landmarks and not np.array_equal(xy, np.asarray(landmarks['native_xy'], np.float64)):
        raise ValueError('Landmark normalized/native mapping mismatch')
    return valid


def moderate_recovery(proposal, recovery, *, minimum_area=70, minimum_depth=11.0):
    """Exact professor moderation on added proposals only, not baseline D."""
    h, w = proposal.shape
    hh = min(h, 1600)
    ww = round(w * hh / h)
    small = (cv2.resize((proposal != 0).astype(np.float32), (ww, hh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(small, 8)
    records = recovery['records']
    if len(records) != n - 1:
        raise ValueError('Recovery label binding mismatch')
    accepted = np.zeros(n, np.uint8)
    for rec in records:
        if rec['area_work'] != int(stats[rec['label'], cv2.CC_STAT_AREA]):
            raise ValueError('Recovery area binding mismatch')
        if rec['accepted'] and rec['area_work'] >= minimum_area and rec['core_depth'] >= minimum_depth:
            accepted[rec['label']] = 1
    projected = cv2.resize(accepted[labels], (w, h), interpolation=cv2.INTER_NEAREST) != 0
    return ((recovery['mask_native'] != 0) & projected).astype(np.uint8) * 255


def build_masks(uv_bgr, landmarks, old_valid_native, *, vendor_path=None, raw_company=None, stage_hook=None):
    """Single CPU sample; original native UV and frozen-geometry landmarks.

    old_valid_native must be generated by geometry_from_landmarks in its pinned
    OpenCV 4.11 runtime, not re-rasterized under the mask runtime (4.10 differs).
    raw_company is optional only for replay/caching, bound externally to native
    UV hash. New samples leave it None to execute the original company entry.
    stage_hook(name, array) may compare/write small-stage audits; no algorithm
    decisions depend on its return value. Inputs and historical outputs unchanged.
    """
    vendor = verify_sources(vendor_path)
    if not isinstance(uv_bgr, np.ndarray) or uv_bgr.dtype != np.uint8 or uv_bgr.ndim != 3 or uv_bgr.shape[2] != 3:
        raise ValueError('Expected native uint8 BGR UV')
    cv2.setNumThreads(2)
    from porphyrin_company_adapter_20261021 import load_vendor, detect_uv
    from uv_darkspot_module_v4 import DarkSpotConfig, detect_uv_darkspots
    from uv_darkspot_refinement_v5 import RefinementConfig, refine_darkspots
    from uv_darkspot_native_red_guard_v1 import NativeRedGuardConfig, build_native_red_guard
    from uv_spot_local_contrast_filter_20261024 import LocalContrastConfig, filter_local_spots
    from uv_forehead_domain_20261024 import ForeheadDomainConfig, extend_forehead_domain
    from uv_spots_company_v2 import UVSpotsConfig, detect_uv_spots
    from uv_spot_recovery_20261024 import RecoveryConfig, recover_dark_proposals
    from uv_porphyrin_priority_20261024 import company_p_support, enforce_porphyrin_priority
    from uv_darkspot_company_based_v1 import load_company_primitives
    from uv_darkspot_anatomical_exclusions_v1 import build_dark_exclusions
    # Both company callers must bind to the same immutable copy. The original
    # primitive loader supports an adjacent file in a deployment package.
    _, _, primitive_source = load_company_primitives()
    if primitive_source['source_sha256'] != sha256(vendor):
        raise ValueError('Company primitive and point detector versions differ')
    support = _original_functions('build_uv_three_masks_review_20261021.py', ['porphyrin_pixel_support'])['porphyrin_pixel_support']
    started = time.monotonic()
    audit = {'stages': {}, 'company_source_sha256': VENDOR_SHA}
    def emit(name, array):
        audit['stages'][name] = {'shape': list(array.shape), 'pixels': int(np.count_nonzero(array)),
                                  'array_sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        if stage_hook is not None:
            stage_hook(name, array)
    if not isinstance(old_valid_native, np.ndarray) or old_valid_native.dtype != np.uint8 or old_valid_native.shape != uv_bgr.shape[:2]:
        raise ValueError('Expected geometry-runtime native uint8 F with matching UV shape')
    if np.any((old_valid_native != 0) & (old_valid_native != 255)):
        raise ValueError('F must be binary 0/255')
    old_f = old_valid_native
    anatomy = build_dark_exclusions(landmarks, uv_bgr.shape[:2])['mask_native']
    emit('old_F', old_f); emit('anatomy', anatomy)
    if raw_company is None:
        detected = detect_uv(load_vendor(vendor), uv_bgr, roi=None, min_dist=10.0)
        spots, bands = detected['spots'], detected['hsv_bands']
        audit['raw_company_count'] = len(spots)
        del detected
    else:
        spots, bands = raw_company['spots'], raw_company['hsv_bands']
        audit['raw_company_count'] = len(spots)
    old_p, _ = support(uv_bgr, old_f, spots, bands)
    emit('old_P', old_p)
    base_config = replace(DarkSpotConfig(), min_circularity=0.0)
    base = detect_uv_darkspots(uv_bgr, old_f, old_p, anatomy, base_config)
    emit('no_circularity_D', base['mask_native'])
    red = build_native_red_guard(uv_bgr, old_f, old_p, NativeRedGuardConfig())['guard_native']
    emit('old_red_guard', red)
    refined = refine_darkspots(base, base['mask_native'], red, RefinementConfig(), old_f)['mask_native']
    emit('refined_D', refined)
    del base
    gc.collect()
    baseline = filter_local_spots(uv_bgr, refined, old_f, old_p | anatomy | red, LocalContrastConfig())['mask_native']
    emit('shadow_D', baseline)
    del red
    expanded = extend_forehead_domain(uv_bgr, old_f, ForeheadDomainConfig())
    valid = expanded['valid_native']; emit('F', valid)
    del expanded
    p, retained = support(uv_bgr, valid, spots, bands)
    emit('P', p)
    # Candidate generator historically receives un-clipped original filled
    # contour footprint; final priority guard is separately clipped to F.
    footprint = np.zeros_like(p)
    for spot in retained:
        cv2.drawContours(footprint, [np.asarray(spot['contour'], np.int32).reshape(-1, 1, 2)], -1, 255, cv2.FILLED)
    company = detect_uv_spots(uv_bgr, valid, footprint, p, anatomy, UVSpotsConfig())
    proposal = company['accepted_support_native']; new_red = company['native_red_guard']
    emit('company_proposals', proposal); emit('new_red_guard', new_red)
    del company, footprint
    forbidden = p | anatomy | new_red
    all_recovered = []
    moderated = []
    for tag, candidate in [('old', refined), ('company', proposal)]:
        recovery = recover_dark_proposals(uv_bgr, candidate, valid, forbidden, RecoveryConfig())
        emit('recovery_' + tag, recovery['mask_native'])
        all_recovered.append(recovery['mask_native'])
        moderated.append(moderate_recovery(candidate, recovery))
        emit('moderated_' + tag, moderated[-1])
        del recovery
    v1 = baseline | all_recovered[0] | all_recovered[1]
    v1[(valid == 0) | (forbidden != 0)] = 0
    # 70/11 implies earlier 60/10, so intersecting with the original v1 is
    # identical to the historical professor wrapper's intersection with v2.
    pre_priority = (baseline | moderated[0] | moderated[1]) & v1
    emit('professor_D', pre_priority)
    del all_recovered, moderated, v1, refined, proposal, old_p, old_f, baseline
    guard = company_p_support(valid.shape, retained, valid)
    priority = enforce_porphyrin_priority(pre_priority, p, valid, guard)
    d = priority['dark_native']; emit('D', d)
    other = ((valid != 0) & (p == 0) & (d == 0)).astype(np.uint8) * 255
    emit('O', other)
    if np.any((p != 0) & (d != 0)) or not np.array_equal((p != 0) | (d != 0) | (other != 0), valid != 0):
        raise AssertionError('Three-mask partition failure')
    audit.update(seconds=round(time.monotonic() - started, 3), retained_company_count=len(retained),
                 priority_stats=priority['stats'], runtime={'opencv': cv2.__version__, 'numpy': np.__version__},
                 configs={'v4': asdict(base_config), 'refinement': asdict(RefinementConfig()),
                          'red_guard': asdict(NativeRedGuardConfig()), 'local_contrast': asdict(LocalContrastConfig()),
                          'forehead': asdict(ForeheadDomainConfig()), 'company_dark': asdict(UVSpotsConfig()),
                          'recovery': asdict(RecoveryConfig()), 'moderation': {'area_work': 70, 'depth': 11.0}})
    return {'P': p, 'D': d, 'O': other, 'F': valid, 'audit': audit,
            'raw_company': {'spots': spots, 'hsv_bands': bands}, 'retained_company': retained}


def run_files(uv_path, landmarks_path, old_valid_path, output_dir, vendor_path=None):
    """Auditable single-sample CLI boundary; never writes historical outputs."""
    import json
    vendor = verify_sources(vendor_path)
    input_paths = {'uv': Path(uv_path), 'landmarks': Path(landmarks_path),
                   'old_valid': Path(old_valid_path), 'vendor': vendor}
    inputs = {key: sha256(path) for key, path in input_paths.items()}
    landmarks = json.loads(input_paths['landmarks'].read_text(encoding='utf-8'))
    if landmarks.get('uv_sha256') != inputs['uv']:
        raise ValueError('Geometry is not bound to this native UV')
    if landmarks.get('old_valid_file_sha256') != inputs['old_valid']:
        raise ValueError('Geometry native F file hash mismatch')
    uv = cv2.imdecode(np.fromfile(input_paths['uv'], np.uint8), cv2.IMREAD_COLOR)
    valid = cv2.imdecode(np.fromfile(input_paths['old_valid'], np.uint8), cv2.IMREAD_GRAYSCALE)
    if uv is None or valid is None:
        raise ValueError('Cannot read native inputs')
    result = build_masks(uv, landmarks, valid, vendor_path=vendor)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    counts = {}
    for key in ['P', 'D', 'O', 'F']:
        array = result[key]
        ok, encoded = cv2.imencode('.png', array)
        if not ok:
            raise ValueError('PNG encode failed')
        with (folder / (key + '.png')).open('xb') as stream:
            stream.write(encoded.tobytes())
        counts[key] = int(np.count_nonzero(array))
    for name, value in [('company_raw.json', result['raw_company']), ('audit.json', result['audit'])]:
        with (folder / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    if inputs != {key: sha256(path) for key, path in input_paths.items()}:
        raise ValueError('Native inputs changed during processing')
    artifacts = ['P.png', 'D.png', 'O.png', 'F.png', 'company_raw.json', 'audit.json']
    summary = {'status': 'complete', 'counts': counts, 'native_shape': list(valid.shape),
               'partition_verified': True, 'input_sha256': inputs,
               'artifact_sha256': {name: sha256(folder / name) for name in artifacts},
               'array_sha256': {key: result['audit']['stages'][key]['array_sha256'] for key in ['P', 'D', 'O', 'F']},
               'source_sha256': sha256(__file__), 'frozen_sources': FROZEN_SHA,
               'runtime': result['audit']['runtime'], 'seconds': result['audit']['seconds']}
    with (folder / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--uv', type=Path, required=True)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--old-valid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vendor', type=Path)
    args = parser.parse_args()
    print(json.dumps(run_files(args.uv, args.landmarks, args.old_valid, args.output, args.vendor)), flush=True)
