"""Replay archived native F only, in the original two separate runtimes.

geometry: archived landmarks -> old F, using the frozen geometry rasterizer.
final: old F + original UV canvas -> final F, using the frozen forehead helper.
build-f: original completed geometry -> the same F rule, without P/D detection.
No detector, MediaPipe inference, P/D segmentation, resize or threshold change.
Every generated PNG must match its historical bytes, array hash and pixel count.
Outputs are new immutable artifacts; this helper never removes or edits inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image


SCHEMA = 'uv_ita_native_f_replay_v1'
FROZEN = {
    'mi_atlas_frozen_mask_adapter_20261025.py': 'db52cafb66d40f3b40592dff11724888ce6c6fbf37731df2cc56e250b4320e0e',
    'mi_atlas_frozen_geometry_20261025.py': '35018515b38b8b8a61afad13448e6dce800a4298d9d927a689ecf37309cd0ff1',
    'build_porphyrin_mediapipe_regions_20261021.py': 'a63c338527290341e700ef47ca6bf2a2735944f80adff7e0e7b8e625498f088b',
    'uv_forehead_domain_20261024.py': '20e5b5687c7fd16b28843a98d756475f7ab685e1264066d4f914717964b36eaf',
}
CONNECTIONS_SHA = 'cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87'
RUNTIMES = {
    'geometry': {'cv2': '4.11.0', 'numpy': '1.26.4', 'PIL': '12.0.0'},
    'final': {'cv2': '4.10.0', 'numpy': '2.3.4', 'PIL': '12.0.0'},
}
FOREHEAD_CONFIG = {'upper_arc_fraction': .25, 'upward_extension_fraction': .18,
                   'taper_power': 1.5, 'nonblack_value': 8}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def bound(desc):
    path = Path(desc['path'])
    require(path.is_absolute() and sha(path) == desc['sha256'], 'Archived input SHA mismatch: ' + str(path))
    return path


def array_sha(raw):
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()


def verify_array(raw, shape, stage, label):
    require(raw.dtype == np.uint8 and list(raw.shape) == list(shape) and
            np.all((raw == 0) | (raw == 255)), label + ': native binary uint8 required')
    require(stage['shape'] == list(shape), label + ': archived shape mismatch')
    require(int(np.count_nonzero(raw)) == stage['pixels'], label + ': pixel count mismatch')
    require(array_sha(raw) == stage['array_sha256'], label + ': array SHA mismatch')


def evidence(mask_receipt):
    """Resolve only sealed original TEST evidence; technical failures stay upstream."""
    rec = read(mask_receipt)
    require(rec.get('record_sha256') == digest({k: v for k, v in rec.items() if k != 'record_sha256'}),
            'Mask receipt seal mismatch')
    require(rec.get('schema') == 'uv_three_region_test_mask_case_v1' and rec['status'] == 'complete',
            'Native F replay requires a complete original TEST case')
    names = ('geometry/landmarks.json', 'geometry/summary.json', 'mask/audit.json', 'mask/summary.json')
    lm, gs, audit, ms = [read(bound(rec['evidence'][name])) for name in names]
    require(gs['status'] == ms['status'] == 'complete', 'Incomplete archived geometry/mask')
    require(gs['native_shape'] == ms['native_shape'] == rec['native_shape'] == lm['source']['native_shape_hw'],
            'Archived native shapes differ')
    require(gs['source_sha256'] == FROZEN['mi_atlas_frozen_geometry_20261025.py'] and
            ms['source_sha256'] == FROZEN['mi_atlas_frozen_mask_adapter_20261025.py'],
            'Original geometry/mask implementation differs')
    require(gs['artifact_sha256']['landmarks.json'] == rec['evidence']['geometry/landmarks.json']['sha256'] ==
            ms['input_sha256']['landmarks'], 'Archived landmarks binding differs')
    require(ms['artifact_sha256']['audit.json'] == rec['evidence']['mask/audit.json']['sha256'],
            'Archived mask audit binding differs')
    require(gs['artifact_sha256']['old_valid_native.png'] == lm['old_valid_file_sha256'] ==
            ms['input_sha256']['old_valid'], 'Archived old F file bindings differ')
    require(lm['uv_sha256'] == ms['input_sha256']['uv'] == rec['sources']['uv']['sha256'],
            'Archived UV bindings differ')
    require(ms['artifact_sha256']['F.png'] == rec['native_mask_file_sha256']['F'],
            'Archived final F PNG bindings differ')
    require(ms['counts']['F'] == audit['stages']['F']['pixels'] == rec['counts_native']['F'] and
            ms['array_sha256']['F'] == audit['stages']['F']['array_sha256'],
            'Archived final F count/array bindings differ')
    require(audit['runtime'] == {'opencv': '4.10.0', 'numpy': '2.3.4'} and
            lm['runtime']['opencv'] == '4.11.0' and lm['runtime']['numpy'] == '1.26.4',
            'Archived numerical runtimes differ')
    return rec, lm, gs, audit, ms


def source_bindings(frozen_tools):
    folder = Path(frozen_tools).resolve()
    result = {}
    for name, expected in FROZEN.items():
        path = folder / name
        require(sha(path) == expected, 'Frozen source changed: ' + name)
        result[name] = descriptor(path)
    connections = folder.parent / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py'
    require(sha(connections) == CONNECTIONS_SHA, 'Frozen landmark connections changed')
    result['face_mesh_connections.py'] = descriptor(connections)
    return result


def check_runtime(stage):
    import cv2
    actual = {'cv2': cv2.__version__, 'numpy': np.__version__, 'PIL': PIL.__version__}
    require(actual == RUNTIMES[stage], 'Wrong ' + stage + ' runtime: ' + repr(actual))
    cv2.setNumThreads(1)
    return actual


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry_array(frozen_tools, landmarks, shape):
    adapter = load_module(Path(frozen_tools) / 'mi_atlas_frozen_mask_adapter_20261025.py',
                          '_ita_native_f_frozen_adapter')
    # 从地标重建原掩膜
    return adapter.geometry_from_landmarks(landmarks, shape)


def final_array_and_png(frozen_tools, uv_path, old, shape, config):
    import cv2
    module = load_module(Path(frozen_tools) / 'uv_forehead_domain_20261024.py',
                         '_ita_native_f_frozen_forehead')
    defaults = module.ForeheadDomainConfig()
    require(asdict(defaults) == config, 'Archived forehead configuration differs')
    uv = cv2.imdecode(np.fromfile(uv_path, np.uint8), cv2.IMREAD_COLOR)
    require(uv is not None and uv.dtype == np.uint8 and list(uv.shape) == list(shape) + [3],
            'Native UV decode/geometry differs')
    raw = module.extend_forehead_domain(uv, old, defaults)['valid_native']
    ok, png = cv2.imencode('.png', raw)
    require(ok, 'Original final F PNG encoder failed')
    return raw, png.tobytes()


def png_array(path, expected_sha):
    require(sha(path) == expected_sha, 'Old F PNG SHA mismatch')
    with Image.open(path) as image:
        require(image.mode == 'L', 'Old F PNG must be original grayscale')
        return np.asarray(image).copy()


def publish(output, name, raw, png, expected, mask_receipt, sources, stage, runtime, extra=None):
    """Only publish after every historical equality passed; never overwrite."""
    require(hashlib.sha256(png).hexdigest() == expected, stage + ': PNG SHA mismatch')
    destination = Path(output).resolve()
    require(not destination.exists(), 'Replay output already exists; inspect, never overwrite')
    # 禁止覆盖来源目录
    require(destination != Path(mask_receipt).resolve() and destination not in Path(mask_receipt).resolve().parents,
            'Replay output overlaps historical evidence')
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / name
    with target.open('xb') as stream:
        stream.write(png)
    record = {'schema': SCHEMA, 'status': 'complete', 'stage': stage,
              'mask_receipt': descriptor(mask_receipt), 'sources': sources,
              'runtime': runtime, 'png': descriptor(target), 'shape': list(raw.shape),
              'array_sha256': array_sha(raw), 'pixels': int(np.count_nonzero(raw)),
              'native_pixels_replayed_exactly': True, 'detection_or_masks_modified': False}
    record.update(extra or {})
    record['record_sha256'] = digest(record)
    with (destination / 'replay.json').open('xb') as stream:
        stream.write(encoded(record))
    return record


def replay_geometry(mask_receipt, frozen_tools, output):
    before = descriptor(mask_receipt)
    rec, lm, _, audit, _ = evidence(mask_receipt)
    sources = source_bindings(frozen_tools)
    runtime = check_runtime('geometry')
    raw = geometry_array(frozen_tools, lm, rec['native_shape'])
    verify_array(raw, rec['native_shape'], audit['stages']['old_F'], 'old F')
    stream = io.BytesIO()
    Image.fromarray(raw).save(stream, format='PNG')
    evidence(mask_receipt)
    require(descriptor(mask_receipt) == before, 'Receipt changed during replay')
    return publish(output, 'old_valid_native.png', raw, stream.getvalue(), lm['old_valid_file_sha256'],
                   mask_receipt, sources, 'geometry', runtime,
                   {'sample_id': rec['sample_id'], 'index': rec['index']})


def replay_final(mask_receipt, frozen_tools, old_f, output):
    before = descriptor(mask_receipt)
    rec, lm, _, audit, ms = evidence(mask_receipt)
    sources = source_bindings(frozen_tools)
    runtime = check_runtime('final')
    old = png_array(old_f, lm['old_valid_file_sha256'])
    verify_array(old, rec['native_shape'], audit['stages']['old_F'], 'old F')
    uv = bound(rec['sources']['uv'])
    raw, png = final_array_and_png(frozen_tools, uv, old, rec['native_shape'], audit['configs']['forehead'])
    verify_array(raw, rec['native_shape'], audit['stages']['F'], 'final F')
    require(not np.any((old != 0) & (raw == 0)), 'Final F unexpectedly removes original support')
    bound(rec['sources']['uv'])
    require(sha(old_f) == lm['old_valid_file_sha256'], 'Old F changed during replay')
    evidence(mask_receipt)
    require(descriptor(mask_receipt) == before, 'Receipt changed during replay')
    return publish(output, 'F.png', raw, png, ms['artifact_sha256']['F.png'], mask_receipt,
                   sources, 'final', runtime, {'sample_id': rec['sample_id'], 'index': rec['index'],
                                             'old_f': descriptor(old_f), 'uv': rec['sources']['uv']})


def build_f(geometry_dir, uv_path, frozen_tools, output, *, expected_geometry_summary_sha256,
            expected_uv_sha256):
    """F-only construction after the ORIGINAL geometry stage; no P/D claim.

    This is not the already-archived-mask replay path. Its provenance explicitly
    distinguishes same-rule F construction from historical final-F parity.
    The caller binds these inputs to the fixed catalog and preserves all IDs.
    """
    folder = Path(geometry_dir).resolve()
    gs_path, lm_path, old_path = (folder / name for name in
                                  ('summary.json', 'landmarks.json', 'old_valid_native.png'))
    require(sha(gs_path) == expected_geometry_summary_sha256, 'Geometry summary SHA mismatch')
    gs = read(gs_path)
    require(gs['status'] == 'complete' and gs['source_sha256'] == FROZEN['mi_atlas_frozen_geometry_20261025.py'],
            'Require completed original frozen geometry')
    require(sha(lm_path) == gs['artifact_sha256']['landmarks.json'], 'Geometry landmarks SHA mismatch')
    lm = read(lm_path)
    require(gs['native_shape'] == lm['source']['native_shape_hw'], 'Original geometry shapes differ')
    require(lm['source']['landmark_source'] in ('RGB_preview', 'UV_preview_fallback'), 'Unknown geometry source')
    require(lm['runtime']['opencv'] == '4.11.0' and lm['runtime']['numpy'] == '1.26.4' and
            lm['runtime']['mediapipe'] == '0.10.21' and gs['runtime'] == lm['runtime'],
            'Original geometry runtime differs')
    require(gs['input_sha256']['uv'] == lm['uv_sha256'] == expected_uv_sha256 == sha(uv_path),
            'Original geometry/UV binding differs')
    require(gs['input_sha256']['rgb_preview'] == lm['rgb_preview_sha256'] and
            gs['input_sha256']['model'] == lm['model_sha256'], 'Original geometry source bindings differ')
    require(gs['artifact_sha256']['old_valid_native.png'] == lm['old_valid_file_sha256'],
            'Original old-F bindings differ')
    old = png_array(old_path, lm['old_valid_file_sha256'])
    shape = gs['native_shape']
    require(old.dtype == np.uint8 and list(old.shape) == shape and np.all((old == 0) | (old == 255)),
            'Original old F must be native binary uint8')
    sources = source_bindings(frozen_tools)
    runtime = check_runtime('final')
    raw, png = final_array_and_png(frozen_tools, uv_path, old, shape, FOREHEAD_CONFIG)
    require(raw.dtype == np.uint8 and list(raw.shape) == shape and np.all((raw == 0) | (raw == 255)),
            'Constructed F must be native binary uint8')
    require(not np.any((old != 0) & (raw == 0)), 'Constructed F removes original support')
    require(sha(uv_path) == expected_uv_sha256 and sha(gs_path) == expected_geometry_summary_sha256 and
            sha(lm_path) == gs['artifact_sha256']['landmarks.json'] and sha(old_path) == lm['old_valid_file_sha256'],
            'Original inputs changed during F-only construction')
    destination = Path(output).resolve()
    require(not destination.exists(), 'F-only output already exists; inspect, never overwrite')
    require(destination != folder and destination not in folder.parents and destination != Path(uv_path).resolve(),
            'F-only output overlaps original inputs')
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'F.png'
    with target.open('xb') as stream:
        stream.write(png)
    record = {'schema': 'uv_ita_native_f_from_geometry_v1', 'status': 'complete', 'stage': 'build-f',
              'geometry_summary': descriptor(gs_path), 'landmarks': descriptor(lm_path),
              'old_f': descriptor(old_path), 'uv': descriptor(uv_path), 'sources': sources, 'runtime': runtime,
              'png': descriptor(target), 'shape': shape, 'array_sha256': array_sha(raw),
              'pixels': int(np.count_nonzero(raw)), 'old_f_array_sha256': array_sha(old),
              'old_f_pixels': int(np.count_nonzero(old)), 'forehead_config': FOREHEAD_CONFIG,
              'landmark_source': lm['source']['landmark_source'],
              'same_approved_f_rule': True, 'historical_final_f_parity_available': False,
              'p_d_masks_computed': False, 'mi_eligibility_determined': False,
              'detection_or_masks_modified': False}
    record['record_sha256'] = digest(record)
    with (destination / 'replay.json').open('xb') as stream:
        stream.write(encoded(record))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('geometry', 'final', 'build-f'))
    parser.add_argument('--mask-receipt', type=Path)
    parser.add_argument('--frozen-tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New immutable stage directory')
    parser.add_argument('--old-f', type=Path, help='Verified geometry-stage old_valid_native.png; final stage only')
    parser.add_argument('--geometry-dir', type=Path)
    parser.add_argument('--uv', type=Path)
    parser.add_argument('--uv-sha256')
    parser.add_argument('--geometry-summary-sha256')
    args = parser.parse_args()
    if args.stage == 'geometry':
        require(args.mask_receipt is not None, 'Geometry replay requires a mask receipt')
        require(args.old_f is None, 'Geometry stage does not accept an old-F override')
        result = replay_geometry(args.mask_receipt, args.frozen_tools, args.output)
    elif args.stage == 'final':
        require(args.mask_receipt is not None, 'Final replay requires a mask receipt')
        require(args.old_f is not None, 'Final stage requires the original-hash-matched old F')
        result = replay_final(args.mask_receipt, args.frozen_tools, args.old_f, args.output)
    else:
        require(args.mask_receipt is None and args.old_f is None, 'F-only construction cannot override old F or use a mask receipt')
        require(all(value is not None for value in (args.geometry_dir, args.uv, args.uv_sha256,
                                                   args.geometry_summary_sha256)), 'Missing F-only input bindings')
        result = build_f(args.geometry_dir, args.uv, args.frozen_tools, args.output,
                         expected_geometry_summary_sha256=args.geometry_summary_sha256,
                         expected_uv_sha256=args.uv_sha256)
    print(json.dumps(result, ensure_ascii=False), flush=True)
