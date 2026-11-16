"""Read-only-dependency, F-only TEST ITA fast track. Controller uses stdlib only.

Existing original masks are replayed and hash-verified. Missing masks do NOT
block ITA: the original preview/geometry stages and forehead helper construct F
without running P/D detection. MI eligibility stays pending until the ORIGINAL
mask receipt exists. No MI, training, source-image or old-task writes occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

CATALOG_SHA = '81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89'
TEST_COUNT = 21899
STRATA = ('very_light', 'light', 'intermediate', 'tan', 'brown', 'dark', 'unclassified')
TECHNICAL = {'ValueError: Expected exactly one face; no mask fabricated': 'no_unique_face',
             'ValueError: Nonfinite or out-of-image polygon vertex: no automatic clipping': 'invalid_geometry'}
SOURCES = ('workflow.py', 'uv_rgb_ita_full_f_20261025.py', 'uv_rgb_ita_20261025.py',
           'uv_ita_native_f_replay_20261025.py')
CPU = [60, 61, 62, 63]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'),
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def desc(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def bound(value):
    path = Path(value['path'])
    require(path.is_absolute() and sha(path) == value['sha256'], 'Frozen input changed: ' + str(path))
    return path


def seal(value):
    return dict(value, record_sha256=digest(value))


def checked(value):
    require(value.get('record_sha256') == digest({k: v for k, v in value.items() if k != 'record_sha256'}),
            'Record seal mismatch')
    return value


def save(path, value, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    if not mutable and path.exists():
        require(path.read_bytes() == payload, 'Refuse frozen overwrite: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if mutable:
        os.replace(temporary, path)
    else:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == payload, 'Concurrent frozen overwrite')
        finally:
            temporary.unlink()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def env(runtime=None):
    result = os.environ.copy()
    result.pop('PYTHONPATH', None)
    result.pop('PYTHONHOME', None)
    if runtime and runtime.get('pythonpath'):
        result['PYTHONPATH'] = os.pathsep.join(runtime['pythonpath'])
    result.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONNOUSERSITE='1',
                  PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return result


def identity(pid):
    text = Path('/proc/%d/stat' % pid).read_text()
    tail = text[text.rfind(')') + 2:].split()
    return {'pid': pid, 'starttick': int(tail[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def color_probe(python, source):
    command = [python, str(source / 'workflow.py'), 'color-probe', '--out', str(source.parent)]
    return json.loads(subprocess.check_output(command, env=env(), text=True))


def configure(args):
    out, root = args.out.resolve(), args.eval_root.resolve()
    require(out.parent == root and out.name == 'ita_full_f_fasttrack_v1', 'Require distinct approved fast-track child directory')
    require(1 <= args.workers <= 4, 'Workers must be 1..4')
    catalog_path = root / 'catalog.json'
    require(sha(catalog_path) == CATALOG_SHA, 'Frozen TEST catalog changed')
    catalog = read(catalog_path)
    require(catalog['expected_count'] == TEST_COUNT and len(catalog['rows']) == TEST_COUNT,
            'Must preserve all 21899 clean TEST rows')
    require(len({r['sample_id'] for r in catalog['rows']}) == TEST_COUNT, 'Duplicate TEST IDs')
    batch = load(args.batch_module, '_ita_frozen_batch')
    old = batch.load_settings(args.frozen_config, project_root=args.batch_module.resolve().parents[1])
    require(read(old['output'] / 'batch_contract.json') == old['contract'], 'Original contract changed')
    source = out / 'source'
    color = color_probe(args.color_python, source)
    require(color['runtime'] == {'numpy': '1.21.2', 'Pillow': '8.2.0'}, 'ITA must use original color runtime')
    protocol = {'schema': 'ita_full_f_fasttrack_protocol_v1',
        'authorization': 'Human selected entire confirmed F, requested accelerated Table 2; no masks/company/training changes',
        'catalog': desc(catalog_path), 'sample_count': TEST_COUNT,
        'sample_order_sha256': hashlib.sha256(json.dumps([r['sample_id'] for r in catalog['rows']],
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'original_mask_contract': desc(root / 'masks/mask_contract.json'),
        'original_eval_protocol': desc(root / 'evaluation/protocol.json'),
        'original_launch_request': desc(root / 'evidence/launch_request.json'),
        'original_batch_contract': desc(old['output'] / 'batch_contract.json'),
        'frozen_config': desc(args.frozen_config), 'batch_module': desc(args.batch_module),
        'sources': {name: desc(source / name) for name in SOURCES},
        'frozen_tools': str(args.batch_module.resolve().parent),
        'stages': {name: desc(old['scripts'][name]) for name in ('preview', 'geometry')},
        'model': desc(old['files']['model']), 'runtimes': old['runtimes'],
        'color_runtime': {'python': os.path.abspath(args.color_python), 'python_sha256': sha(args.color_python),
                          'pythonpath': [], 'versions': color['runtime']},
        'method': color['method'], 'workers': args.workers, 'cpu_affinity': CPU,
        'nice': 19, 'ionice': 'idle3', 'cuda': False,
        'geometry_policy': 'Archived same-case geometry replay where available; otherwise original frozen preview and geometry inference; UV fallback never yields an ITA label',
        'fresh_F_policy': 'Original forehead extension only; no P/D calls, no semantic skin filtering; every native F pixel used',
        'mi_eligibility': 'Only original complete P/D/O/F native-count receipts can confirm >=1000; missing evidence is pending, never excluded',
        'outputs': 'ITA labels and Table 2A counts only; no MI estimation or model evaluation',
        'data_unit': 'sample_group; independent patient identity not established'}
    for folder in ('labels', 'cases', 'evidence'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', protocol)
    print(json.dumps({'status': 'configured', 'protocol': desc(out / 'protocol.json')}), flush=True)


def verify(out):
    p = read(out / 'protocol.json')
    for key in ('catalog', 'original_mask_contract', 'original_eval_protocol', 'original_launch_request',
                'original_batch_contract', 'frozen_config', 'batch_module', 'model'):
        bound(p[key])
    for value in list(p['sources'].values()) + list(p['stages'].values()):
        bound(value)
    require(p['catalog']['sha256'] == CATALOG_SHA and p['sample_count'] == TEST_COUNT and
            p['cpu_affinity'] == CPU and 1 <= p['workers'] <= 4, 'Protocol invariants changed')
    require(sha(p['color_runtime']['python']) == p['color_runtime']['python_sha256'], 'Color Python changed')
    probe = color_probe(p['color_runtime']['python'], out / 'source')
    require(probe['runtime'] == p['color_runtime']['versions'] and probe['method'] == p['method'], 'Color implementation/runtime changed')
    batch = load(bound(p['batch_module']), '_ita_frozen_batch_verify')
    old = batch.load_settings(bound(p['frozen_config']), project_root=Path(p['frozen_tools']).parent)
    require(old['runtimes'] == p['runtimes'], 'Frozen stage runtimes changed')
    return p, read(p['catalog']['path'])


def mask_record(p, catalog, index):
    root = Path(p['catalog']['path']).parent
    path = root / 'masks/cases' / ('case_%06d' % index) / 'receipt.json'
    if not path.exists():
        return path, None
    rec = checked(read(path))
    row = catalog['rows'][index]
    require(rec['schema'] == 'uv_three_region_test_mask_case_v1' and rec['index'] == index and
            rec['sample_id'] == row['sample_id'] and rec['subject_id'] == row['subject_id'] and
            rec['source_row_sha256'] == digest(row) and rec['catalog_sha256'] == CATALOG_SHA and
            rec['contract_sha256'] == digest(read(p['original_mask_contract']['path'])) and
            rec['native_shape'] == row['native_shape'], 'Original mask identity/contract differs')
    require(rec['status'] in ('complete', 'no_unique_face', 'invalid_geometry'), 'Unexpected original mask failure')
    for name in ('rgb', 'uv', 'plv', 'plh'):
        native = Path(row[name]['path'])
        if not native.is_absolute():
            native = Path(catalog['data_root']) / native
        require(rec['sources'][name] == {'path': str(native.resolve()), 'sha256': row[name]['sha256']},
                'Original mask source descriptors differ')
    if rec['status'] != 'complete':
        log = bound(rec['failure_stderr'])
        require(TECHNICAL.get(last_line(log)) == rec['status'], 'Technical failure evidence differs')
    return path, rec


def eligibility(rec, path=None):
    if rec is None:
        return {'status': 'pending', 'reason': 'original_P_D_O_F_mask_receipt_not_yet_available'}
    result = {'original_mask_receipt': desc(path)} if path else {}
    if rec['status'] != 'complete':
        return dict(result, status='not_eligible', reason=rec['status'])
    counts = rec['counts_native']
    require(set(counts) == {'F', 'P', 'D', 'O'} and all(type(v) is int and v >= 0 for v in counts.values()),
            'Invalid original native counts')
    small = [name for name in ('P', 'D', 'O', 'F') if counts[name] < 1000]
    return dict(result, status='not_eligible' if small else 'eligible',
                reason='native_region_below_1000' if small else None,
                counts_native=counts, insufficient_domains=small)


def reconcile_label(label, original):
    """When original P/D processing catches up, independently compare the F.

    Never update a frozen label or replace a discrepancy with a new exclusion.
    """
    if original is None:
        return
    require(label['geometry_status'] == original['status'], 'Fresh/original geometry status differs')
    if original['status'] != 'complete':
        return
    final = checked(read(bound(label['artifacts']['final_F_receipt'])))
    require(final['png']['sha256'] == original['native_mask_file_sha256']['F'] and
            final['pixels'] == original['counts_native']['F'], 'Fresh/original native F PNG/count differs')
    audit = read(bound(original['evidence']['mask/audit.json']))
    require(final['array_sha256'] == audit['stages']['F']['array_sha256'] and
            final['pixels'] == audit['stages']['F']['pixels'], 'Fresh/original native F array differs')


def last_line(path):
    lines = [line.strip() for line in Path(path).read_text(encoding='utf8', errors='replace').splitlines() if line.strip()]
    return lines[-1] if lines else ''


def execute(command, runtime, case, name, technical=False):
    log = case / (name + '.log')
    require(not log.exists(), 'Stage already attempted; inspect preserved evidence: ' + str(log))
    with log.open('xb') as stream:
        done = subprocess.run(command, env=env(runtime), stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT, check=False)
    if done.returncode:
        status = TECHNICAL.get(last_line(log)) if technical else None
        if status:
            return status, desc(log)
        raise RuntimeError('Unexpected stage failure; no sample excluded: %s exit=%s log=%s' % (name, done.returncode, log))
    return 'complete', desc(log)


def normalized(row, data_root, name):
    path = Path(row[name]['path'])
    if not path.is_absolute():
        path = Path(data_root) / path
    require(sha(path) == row[name]['sha256'], 'Native image hash changed: ' + name)
    return str(path.resolve())


def produce(out, p, catalog, index, *, force_fresh=False, destination=None, preview=False):
    row = catalog['rows'][index]
    target = destination or out / 'labels' / ('%06d.json' % index)
    protocol_sha = sha(out / 'protocol.json')
    if target.exists():
        result = checked(read(target))
        require(result['index'] == index and result['sample_id'] == row['sample_id'] and
                result['protocol_sha256'] == protocol_sha and result['source_row_sha256'] == digest(row),
                'Existing label identity/protocol mismatch')
        for value in result.get('artifacts', {}).values():
            bound(value)
        _, original = mask_record(p, catalog, index)
        reconcile_label(result, original)
        return result
    case = out / 'cases' / (('pilot_fresh_%06d' if force_fresh else 'case_%06d') % index)
    require(not case.exists(), 'Incomplete case preserved; inspect before any retry: ' + str(case))
    case.mkdir(parents=True)
    mask_path, original = mask_record(p, catalog, index)
    source = out / 'source'
    script = source / 'uv_ita_native_f_replay_20261025.py'
    base = {'schema': 'ita_full_f_fasttrack_label_v1', 'index': index,
            'sample_id': row['sample_id'], 'subject_id': row['subject_id'],
            'source_row_sha256': digest(row), 'protocol_sha256': protocol_sha,
            'method': p['method']['version'], 'status': 'invalid', 'reason': None,
            'ita_degrees': None, 'stratum': None, 'mi_eligibility_at_label': eligibility(original, mask_path),
            'artifacts': {}, 'fresh_geometry': force_fresh or original is None}
    if original and original['status'] != 'complete' and not force_fresh:
        base.update(reason='upstream_' + original['status'], geometry_status=original['status'])
        base['artifacts']['original_mask_receipt'] = desc(mask_path)
    else:
        rgb = normalized(row, catalog['data_root'], 'rgb')
        uv = normalized(row, catalog['data_root'], 'uv')
        if original and not force_fresh:
            execute([p['runtimes']['geometry']['python'], str(script), 'geometry', '--mask-receipt', str(mask_path),
                     '--frozen-tools', p['frozen_tools'], '--output', str(case / 'geometry')],
                    p['runtimes']['geometry'], case, 'replay_geometry')
            execute([p['runtimes']['mask']['python'], str(script), 'final', '--mask-receipt', str(mask_path),
                     '--frozen-tools', p['frozen_tools'], '--old-f', str(case / 'geometry/old_valid_native.png'),
                     '--output', str(case / 'final')], p['runtimes']['mask'], case, 'replay_final')
            geometry_status = 'complete'
            base['artifacts']['original_mask_receipt'] = desc(mask_path)
            base['artifacts']['geometry_replay'] = desc(case / 'geometry/replay.json')
        else:
            execute([p['runtimes']['preview']['python'], p['stages']['preview']['path'], '--rgb', rgb,
                     '--expected-sha256', row['rgb']['sha256'], '--output', str(case / 'preview')],
                    p['runtimes']['preview'], case, 'preview')
            geometry_status, geometry_log = execute([p['runtimes']['geometry']['python'], p['stages']['geometry']['path'],
                     '--rgb-preview', str(case / 'preview/rgb_preview.jpg'), '--uv', uv,
                     '--model', p['model']['path'], '--output', str(case / 'geometry')],
                     p['runtimes']['geometry'], case, 'geometry', technical=True)
            base['artifacts']['geometry_log'] = geometry_log
            if geometry_status == 'complete':
                gs = case / 'geometry/summary.json'
                execute([p['runtimes']['mask']['python'], str(script), 'build-f', '--geometry-dir', str(case / 'geometry'),
                         '--uv', uv, '--uv-sha256', row['uv']['sha256'], '--geometry-summary-sha256', sha(gs),
                         '--frozen-tools', p['frozen_tools'], '--output', str(case / 'final')],
                        p['runtimes']['mask'], case, 'build_f')
                base['artifacts']['geometry_summary'] = desc(gs)
        base['geometry_status'] = geometry_status
        if geometry_status != 'complete':
            base.update(reason='upstream_' + geometry_status)
            # 单列本次技术失败
        else:
            replay = checked(read(case / 'final/replay.json'))
            require(replay['status'] == 'complete' and replay['shape'] == row['native_shape'], 'F construction incomplete')
            base['artifacts']['final_F_receipt'] = desc(case / 'final/replay.json')
            base['artifacts']['native_F'] = replay['png']
            request = {'base': base, 'row': row, 'source_root': catalog['data_root'], 'rgb': {'path': rgb, 'sha256': row['rgb']['sha256']},
                       'original_mask_receipt': desc(mask_path) if original and not force_fresh else None,
                       'landmarks': desc(case / 'geometry/landmarks.json') if base['fresh_geometry'] else original['evidence']['geometry/landmarks.json'],
                       'final_F': desc(case / 'final/replay.json'), 'preview': str(case / 'preview_F.png') if preview else None}
            save(case / 'color_request.json', request)
            execute([p['color_runtime']['python'], str(source / 'workflow.py'), 'color', '--out', str(out),
                     '--request', str(case / 'color_request.json'), '--request-sha256', sha(case / 'color_request.json'),
                     '--output', str(target)], p['color_runtime'], case, 'color')
            return checked(read(target))
    result = seal(base)
    save(target, result)
    return result


def color(args):
    """Numerical child only: unchanged native Lab estimator in original runtime."""
    sys.path.insert(0, str(args.out / 'source'))
    import uv_rgb_ita_full_f_20261025 as m
    import numpy as np
    import PIL
    from PIL import Image, ImageCms
    import io
    m.verify_color_source()
    if args.mode == 'color-probe':
        print(json.dumps({'runtime': {'numpy': np.__version__, 'Pillow': PIL.__version__}, 'method': m.METHOD}), flush=True)
        return
    require(sha(args.request) == args.request_sha256, 'Color request changed')
    request = read(args.request)
    result = request['base']
    f_record = checked(read(bound(request['final_F'])))
    f_path = bound(f_record['png'])
    lm = read(bound(request['landmarks']))
    require(lm['source']['native_shape_hw'] == request['row']['native_shape'], 'Color geometry shape differs')
    result['landmark_source'] = lm['source']['landmark_source']
    if request['original_mask_receipt'] is not None:
        details = m.process_case(request['row'], result['index'], bound(request['original_mask_receipt']),
            result['protocol_sha256'], preview_path=request['preview'], source_root=request['source_root'], native_f_path=f_path)
        result.update(details)
    elif result['landmark_source'] == 'UV_preview_fallback':
        result['reason'] = 'rgb_geometry_unavailable_uv_fallback_not_used'
    else:
        require(result['landmark_source'] == 'RGB_preview', 'Unknown RGB geometry source')
        with Image.open(f_path) as mask_image:
            require(mask_image.mode == 'L', 'F must be binary grayscale')
            f = np.asarray(mask_image).copy()
        require(f.dtype == np.uint8 and list(f.shape) == request['row']['native_shape'] and
                np.all((f == 0) | (f == 255)) and m.array_sha(f) == f_record['array_sha256'] and
                int(np.count_nonzero(f)) == f_record['pixels'], 'F array/count binding differs')
        with Image.open(bound(request['rgb'])) as original:
            require(original.mode == 'RGB' and list(reversed(original.size)) == request['row']['native_shape'], 'Native RGB geometry/type differs')
            require(original.getexif().get(274, 1) == 1, 'Unsupported EXIF orientation')
            icc = original.info.get('icc_profile')
            image = original.copy()
            result['icc_sha256'] = hashlib.sha256(icc).hexdigest() if icc else None
            if icc:
                image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                    ImageCms.createProfile('sRGB'), renderingIntent=1, outputMode='RGB')
                result['color_interpretation'] = 'embedded_ICC_to_sRGB_relative_colorimetric'
            else:
                result['color_interpretation'] = 'assumed_sRGB_no_embedded_ICC'
            result.update(m.evaluate_full_f(image, f))
            result['native_f'] = {'png_sha256': f_record['png']['sha256'], 'array_sha256': f_record['array_sha256'],
                                  'pixels': f_record['pixels'], 'shape': f_record['shape']}
            if request['preview']:
                m.preview(image, f, request['preview'], result['sample_id'])
                result['preview'] = desc(request['preview'])
    result.update(roi_is_rgb_only=False, color_values_are_rgb_only=True)
    save(args.output, seal(result))
    print(json.dumps({'status': result['status'], 'index': result['index'], 'stratum': result['stratum']}), flush=True)


def pilot(out):
    p, catalog = verify(out)
    require(not (out / 'evidence/launch_receipt.json').exists(), 'Pilot cannot rerun after launch')
    records = [produce(out, p, catalog, i, preview=True) for i in range(12)]
    index = next(i for i, rec in enumerate(records) if rec['status'] == 'ok')
    fresh_path = out / 'evidence' / ('pilot_fresh_%06d.json' % index)
    fresh = produce(out, p, catalog, index, force_fresh=True, destination=fresh_path, preview=True)
    archived = records[index]
    require(fresh['status'] == archived['status'] and fresh['native_f']['array_sha256'] == archived['native_f']['array_sha256'] and
            fresh['native_f']['png_sha256'] == archived['native_f']['png_sha256'] and
            fresh['native_f']['pixels'] == archived['native_f']['pixels'] and
            fresh['aggregate_lab'] == archived['aggregate_lab'] and fresh['ita_degrees'] == archived['ita_degrees'] and
            fresh['stratum'] == archived['stratum'], 'Fresh/archive F and color parity failed; do not launch')
    result = {'status': 'complete', 'protocol': desc(out / 'protocol.json'), 'sample_slots': 12,
              'labels': {str(i): desc(out / 'labels' / ('%06d.json' % i)) for i in range(12)},
              'status_counts': dict(Counter(r['status'] for r in records)),
              'reasons': dict(Counter(r['reason'] for r in records if r['status'] != 'ok')),
              'fresh_archive_parity': {'index': index, 'fresh': desc(fresh_path), 'native_F_png_array_count_exact': True,
                                       'aggregate_Lab_ITA_stratum_exact': True},
              'does_not_claim_all_landmark_floats_bitwise_equal': True}
    save(out / 'evidence/pilot.json', seal(result))
    print(json.dumps(result), flush=True)


def summarize_records(records, current_eligibility):
    require(len(records) == len(current_eligibility), 'Label/eligibility row counts differ')
    rows = {name: {'stratum': name, 'test_sample_groups': 0, 'eligible_for_MI': 0,
                   'not_included_in_MI': 0, 'eligibility_pending': 0} for name in STRATA}
    reasons = Counter()
    for rec, eligible in zip(records, current_eligibility):
        key = rec['stratum'] if rec['status'] == 'ok' else 'unclassified'
        require(key in rows, 'Unknown ITA stratum')
        row = rows[key]
        row['test_sample_groups'] += 1
        # 跳过无ITA值记录
        if rec['status'] != 'ok':
            row['not_included_in_MI'] += 1
            reasons['ITA_unavailable:' + str(rec['reason'])] += 1
        elif eligible['status'] == 'pending':
            row['eligibility_pending'] += 1
        elif eligible['status'] == 'eligible':
            row['eligible_for_MI'] += 1
        elif eligible['status'] == 'not_eligible':
            row['not_included_in_MI'] += 1
            reasons[eligible['reason']] += 1
        else:
            raise ValueError('Unknown MI eligibility status')
    total = {'stratum': 'total'}
    for key in ('test_sample_groups', 'eligible_for_MI', 'not_included_in_MI', 'eligibility_pending'):
        total[key] = sum(row[key] for row in rows.values())
    require(total['test_sample_groups'] == sum(total[k] for k in ('eligible_for_MI', 'not_included_in_MI', 'eligibility_pending')),
            'Cohort table does not partition')
    return {'rows': list(rows.values()) + [total], 'not_included_reasons': dict(reasons)}


def summarize(out, p=None, catalog=None):
    if p is None:
        p, catalog = verify(out)
    records, current = [], []
    for i, row in enumerate(catalog['rows']):
        path = out / 'labels' / ('%06d.json' % i)
        if not path.exists():
            continue
        rec = checked(read(path))
        require(rec['index'] == i and rec['sample_id'] == row['sample_id'] and
                rec['protocol_sha256'] == sha(out / 'protocol.json'), 'Summary label identity differs')
        mask_path, original = mask_record(p, catalog, i)
        reconcile_label(rec, original)
        records.append(rec)
        current.append(eligibility(original, mask_path))
    summary = dict(summarize_records(records, current), schema='ita_full_f_test_composition_v1',
        status='complete' if len(records) == TEST_COUNT else 'partial', labelled=len(records), expected=TEST_COUNT,
        missing_labels=TEST_COUNT-len(records), protocol=desc(out / 'protocol.json'),
        mi_calculated=False, eligibility_is_snapshot=True, time=time.time())
    save(out / 'evidence/table2a_current.json', summary, mutable=True)
    return summary


def run(out):
    import fcntl
    p, catalog = verify(out)
    require(sorted(os.sched_getaffinity(0)) == CPU and os.getpriority(os.PRIO_PROCESS, 0) == 19 and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU/GPU isolation differs')
    pilot_record = checked(read(out / 'evidence/pilot.json'))
    require(pilot_record['status'] == 'complete' and pilot_record['protocol'] == desc(out / 'protocol.json') and
            pilot_record['fresh_archive_parity']['native_F_png_array_count_exact'] is True and
            pilot_record['fresh_archive_parity']['aggregate_Lab_ITA_stratum_exact'] is True,
            'Bound successful pilot required before run')
    lock = (out / 'evidence/run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'labelling', 'identity': identity(os.getpid()), 'protocol': desc(out / 'protocol.json'),
             'completed': 0, 'expected': TEST_COUNT, 'workers': p['workers'], 'strata_counts': {},
             'invalid_reasons': {}, 'active_indices': [], 'mi_calculated': False}
    counts, reasons, records = Counter(), Counter(), [None] * TEST_COUNT
    def publish():
        state['time'] = time.time()
        save(out / 'evidence/status.json', state, mutable=True)
    try:
        publish()
        with ThreadPoolExecutor(max_workers=p['workers']) as pool:
            active, next_index = {}, 0
            while next_index < TEST_COUNT or active:
                while next_index < TEST_COUNT and len(active) < p['workers']:
                    active[pool.submit(produce, out, p, catalog, next_index)] = next_index
                    next_index += 1
                state['active_indices'] = sorted(active.values())
                publish()
                ready, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = active.pop(future)
                    rec = future.result()
                    records[index] = rec
                    counts[rec['stratum'] if rec['status'] == 'ok' else 'unclassified'] += 1
                    if rec['status'] != 'ok':
                        reasons[rec['reason']] += 1
                    state.update(completed=state['completed'] + 1, strata_counts=dict(counts), invalid_reasons=dict(reasons))
                    print(json.dumps({'event': 'full_f_ita_complete', 'index': index, 'status': rec['status'],
                                      'stratum': rec['stratum'], 'completed': state['completed']}), flush=True)
                publish()
        summary = summarize(out, p, catalog)
        require(summary['status'] == 'complete' and len([r for r in records if r]) == TEST_COUNT, 'Incomplete labels')
        save(out / 'labels.json', {'schema': 'ita_full_f_test_labels_v1', 'protocol': desc(out / 'protocol.json'),
             'sample_count': TEST_COUNT, 'records': records})
        save(out / 'completion.json', seal({'status': 'ITA_complete', 'protocol': desc(out / 'protocol.json'),
            'labels': desc(out / 'labels.json'), 'sample_count': TEST_COUNT, 'mi_calculated': False,
            'eligibility_snapshot': summary, 'time': time.time()}))
        state.update(status='ITA_complete', active_indices=[])
        publish()
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        publish()
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def launch(out):
    p, _ = verify(out)
    pilot_record = checked(read(out / 'evidence/pilot.json'))
    require(pilot_record['status'] == 'complete' and pilot_record['protocol'] == desc(out / 'protocol.json'), 'Bound pilot required')
    for name in ('launch_receipt.json', 'launch_reservation.json'):
        require(not (out / 'evidence' / name).exists(), 'Already launched/reserved; inspect exact process')
    save(out / 'evidence/launch_reservation.json', {'protocol': desc(out / 'protocol.json'), 'time': time.time()})
    command = ['taskset', '-c', ','.join(map(str, CPU)), 'nice', '-n', '19', 'ionice', '-c', '3',
               sys.executable, str(out / 'source/workflow.py'), 'run', '--out', str(out)]
    with (out / 'evidence/workflow.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=str(out), env=env(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = {'status': 'launched_not_yet_verified_computing', 'identity': identity(child.pid),
               'command': command, 'protocol': desc(out / 'protocol.json'), 'time': time.time()}
    save(out / 'evidence/launch_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('configure', 'pilot', 'launch', 'run', 'summarize', 'color', 'color-probe'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path)
    parser.add_argument('--frozen-config', type=Path)
    parser.add_argument('--batch-module', type=Path)
    parser.add_argument('--color-python')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--request-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode in ('color', 'color-probe'):
        color(args)
    elif args.mode == 'configure':
        require(all((args.eval_root, args.frozen_config, args.batch_module, args.color_python)), 'Missing configuration inputs')
        configure(args)
    elif args.mode == 'summarize':
        print(json.dumps(summarize(args.out)), flush=True)
    else:
        globals()[args.mode](args.out)


if __name__ == '__main__':
    main()
