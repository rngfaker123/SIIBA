"""New Table 3 + 4 records from one frozen CPU forward, old outputs read-only.

Uses the original TEST evaluation functions verbatim. No training, new masks,
thresholds, inferred uncertainty, calibration fitting or inference seed repeats.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

SCHEMA = 'three_region_combined_eval_v1'
OLD_SHA = '874bffe20046d7b7e368651906f9d22a4403333d77dbdfc033f2457214d51ff4'
RANK_SHA = 'ef439884b3ef8b73cca3abf9302cc01b2b659293bcd667e9e20263000be63a0d'
CATALOG_SHA = '81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89'
COUNT = 21899
MODELS = ('unet_single', 'unet_multi', 'unetplusplus')
CPUS = {'unet_single': [72, 73], 'unet_multi': [74, 75], 'unetplusplus': [76, 77]}
REGIONS = ('P', 'D', 'O')
TABLE4_METRICS = ('gaussian_nll', 'rank_corr', 'mean_z2', 'coverage95')
RTOL, ATOL = 1e-7, 1e-9


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


def bound(item):
    path = Path(item['path'])
    require(path.is_absolute() and sha(path) == item['sha256'], 'Bound file changed: ' + str(path))
    return path


def seal(item):
    require('record_sha256' not in item, 'Cannot seal an already sealed record')
    return dict(item, record_sha256=digest(item))


def checked(item):
    require(item.get('record_sha256') == digest({k: v for k, v in item.items() if k != 'record_sha256'}), 'Record seal differs')
    return item


def save(path, value, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    if not mutable and path.exists():
        require(path.read_bytes() == payload, 'Refuse immutable overwrite: ' + str(path))
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
            require(path.read_bytes() == payload, 'Concurrent differing immutable record')
        finally:
            temporary.unlink()


def load(path, name, expected):
    require(sha(path) == expected, 'Frozen Python source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(pid):
    text = Path('/proc/%d/stat' % pid).read_text()
    tail = text[text.rfind(')') + 2:].split()
    return {'pid': pid, 'starttick': int(tail[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def isolate(model_id):
    import subprocess
    os.sched_setaffinity(0, CPUS[model_id])
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    require(sorted(os.sched_getaffinity(0)) == CPUS[model_id] and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU/GPU isolation differs')


@contextlib.contextmanager
def lock(out, name):
    import fcntl
    path = out / 'locks' / (name + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def configure(args):
    out, root, masks = args.out.resolve(), args.eval_root.resolve(), args.mask_continuation.resolve()
    require(out.name == 'evaluation' and out.parent.name == 'three_region_test_parallel_20261025_v1', 'Require isolated new evaluation root')
    require(masks == out.parent / 'masks' and root != out.parent and root not in out.parents,
            'New masks/evaluation must not overlap old task')
    old = load(args.old_runner, '_combined_old_eval', OLD_SHA)
    catalog_path, config_path = root / 'catalog.json', root / 'eval_config.json'
    require(sha(catalog_path) == CATALOG_SHA, 'Frozen TEST catalog changed')
    catalog, config = read(catalog_path), read(config_path)
    rows = old.validate_catalog(catalog, catalog_path.parent)
    require(len(rows) == COUNT and set(config['models']) == set(MODELS), 'Require exact original TEST/models')
    frozen = old.FrozenSources(config['sources'], config_path.parent)
    frozen.load('tools/uv_architecture_comparison_metrics_v10.py')
    require(sha(args.rank_helper) == RANK_SHA, 'Rank helper differs')
    for model in MODELS:
        old.validate_model_assets(model, config['models'][model], config_path.parent)
    mask_contract = masks / 'mask_contract.json'
    require(sha(mask_contract) == sha(root / 'masks/mask_contract.json'), 'Scientific mask contract changed')
    old.validate_mask_contract(read(mask_contract), CATALOG_SHA, rows)
    old_protocol = read(root / 'evaluation/protocol.json')
    require(old_protocol['catalog']['sha256'] == CATALOG_SHA and old_protocol['config']['sha256'] == sha(config_path) and
            old_protocol['mask_contract']['sha256'] == sha(mask_contract), 'Original protocol/config/catalog binding differs')
    runtime = old.runtime_info('cpu', 2)
    require(all(runtime.get(k) == v for k, v in config['expected_runtime'].items()), 'Original model runtime differs')
    protocol = {'schema': SCHEMA + '_protocol', 'original_root': str(root), 'mask_root': str(masks),
        'source': desc(__file__), 'old_runner': desc(args.old_runner), 'rank_helper': desc(args.rank_helper),
        'catalog': desc(catalog_path), 'eval_config': desc(config_path), 'mask_contract': desc(mask_contract),
        'mask_execution_amendment': desc(masks / 'execution_amendment.json'),
        'old_evaluation_protocol': desc(root / 'evaluation/protocol.json'),
        'sample_count': COUNT, 'model_order': list(MODELS), 'runtime': runtime,
        'source_pins': old.SOURCE_PINS, 'model_pins': old.MODEL_PINS,
        'resources': {'cpus_per_model': CPUS, 'cpu_threads': 2, 'nice': 19, 'ionice': 3, 'cuda': False},
        'one_forward': 'One original frozen FP32 CPU forward supplies both Table3 and Table4 for every valid image/model; no seed repeats',
        'table3': 'Original prepare_image/infer/measure exactly reused; P/D/O L1 PSNR SSIM mean_sigma; F primary L1 PSNR only; extra raw F diagnostics not main outcomes',
        'table4': {'regions': list(REGIONS), 'metrics': list(TABLE4_METRICS),
            'gaussian_nll': 'Area-weighted channel mean 0.5*((prediction-target)^2/sigma^2+log(sigma^2)); Gaussian constant omitted',
            'mean_z2': 'Area-weighted channel mean of ((prediction-target)/sigma)^2; target1',
            'coverage95': 'Area-weighted channel-pixel marginal abs(z)<=1.959963984540054; untruncated intervals, target.95',
            'rank_corr': 'Weighted Pearson of ordinary average ranks of sigma and channel-mean absolute error within positive-weight support; rho_w, not unweighted Spearman',
            'sigma': 'Unchanged positive Gaussian residual scale shared across three channels; no variance floor or fitted rescaling',
            'summary': 'Equal weight per sample-group; mean and sample SD ddof1; own validN/undefined reasons per cell; no new t-test family'},
        'old_table3_parity': {'rtol': RTOL, 'atol': ATOL, 'exact_reported': True,
            'policy': 'Re-forward every old-complete case; verify original record/source/checkpoint/mask bindings; stop beyond tolerance; new result preserved without overwriting old scalar'},
        'mask_refs': 'sealed refs/NNNNNN.json; follow receipt descriptor only, never guess generated receipt paths',
        'no_old_writes': True, 'no_training_changes': True}
    for name in ('records', 'progress', 'locks', 'pilot', 'exports'):
        (out / name).mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', protocol)
    print(json.dumps({'status': 'configured', 'protocol': desc(out / 'protocol.json')}), flush=True)


def context(out):
    p = read(out / 'protocol.json')
    require(p['schema'] == SCHEMA + '_protocol' and p['source'] == desc(__file__) and
            p['catalog']['sha256'] == CATALOG_SHA and p['model_order'] == list(MODELS) and
            p['resources']['cpus_per_model'] == CPUS, 'Combined protocol changed')
    for key in ('old_runner', 'rank_helper', 'catalog', 'eval_config', 'mask_contract', 'mask_execution_amendment', 'old_evaluation_protocol'):
        bound(p[key])
    old = load(bound(p['old_runner']), '_combined_old_eval', OLD_SHA)
    rank = load(bound(p['rank_helper']), '_combined_rank_helper', RANK_SHA)
    catalog, config = read(p['catalog']['path']), read(p['eval_config']['path'])
    rows = old.validate_catalog(catalog, Path(p['catalog']['path']).parent)
    old.validate_mask_contract(read(p['mask_contract']['path']), CATALOG_SHA, rows)
    frozen = old.FrozenSources(config['sources'], Path(p['eval_config']['path']).parent)
    numeric = frozen.load('tools/uv_architecture_comparison_metrics_v10.py')
    runtime = old.runtime_info('cpu', 2)
    require(runtime == p['runtime'], 'CPU FP32 runtime changed')
    for model in MODELS:
        old.validate_model_assets(model, config['models'][model], Path(p['eval_config']['path']).parent)
    return {'protocol': p, 'protocol_sha': sha(out / 'protocol.json'), 'catalog': catalog, 'rows': rows,
            'config': config, 'old': old, 'rank': rank, 'numeric': numeric, 'frozen': frozen}


def validate_ref(value, row, index, p):
    checked(value)
    require(value['schema'] == 'three_region_mask_receipt_ref_v1' and value['index'] == index and
            value['sample_id'] == row['sample_id'] and value['mask_contract'] == p['mask_contract'] and
            value['origin'] in ('original_readonly', 'continuation') and
            value['execution_amendment'] == p['mask_execution_amendment'], 'Mask reference identity/contract differs')
    bound(value['mask_contract'])
    bound(value['execution_amendment'])
    return bound(value['receipt'])


def wait_ref(out, ctx, model_id, index, timeout, pilot_refs=False):
    p, row = ctx['protocol'], ctx['rows'][index]
    root = Path(p['mask_root'])
    path = root / ('pilot/refs' if pilot_refs else 'refs') / ('%06d.json' % index)
    started = time.monotonic()
    while not path.exists():
        state_path = root / 'continuation_status.json'
        if state_path.exists():
            state = read(state_path)
            require(state.get('status') != 'failed', 'Mask continuation failed; no fabricated evaluation')
            require(state.get('status') != 'complete', 'Mask continuation complete but reference missing')
        require(time.monotonic() - started < timeout, 'Timed out waiting for sealed mask reference')
        publish(out, model_id, status='waiting_for_mask_reference', next_index=index)
        time.sleep(5)
    ref = checked(read(path))
    receipt_path = validate_ref(ref, row, index, p)
    old = ctx['old']
    receipt = old.validate_mask_receipt(read(receipt_path), row, index, CATALOG_SHA,
                                      p['mask_contract']['sha256'], ctx['catalog']['data_root'])
    require(ref['status'] == receipt['status'], 'Mask reference status differs')
    if receipt['status'] in old.TECHNICAL:
        old.validate_technical_evidence(receipt, receipt_path)
    return path, receipt_path, receipt


def table4_measure(truth, prediction, sigma, weights, old, numeric, rank, shape=(1024, 1024)):
    target = numeric.validate_image(truth, 'UV truth')
    pred = numeric.validate_image(prediction, 'UV prediction')
    require(target.shape == pred.shape == (3, *shape), 'Table4 output shape differs')
    scale = numeric._validate_sigma(sigma, shape, numeric.GAUSSIAN_SIGMA_KIND)
    checked_weights = old.validate_weights(weights, shape)
    delta, result = pred - target, {}
    for region in REGIONS:
        weight = checked_weights[region]
        gaussian = numeric._gaussian_region(delta, scale, weight)
        rho = rank.weighted_sigma_error_spearman(scale, target, pred, weight)
        empty = gaussian['status'] == 'empty_region'
        result[region] = {
            'gaussian_nll': gaussian['nll'], 'gaussian_nll_reason': gaussian['reason'],
            'rank_corr': rho['value'], 'rank_corr_reason': rho['reason'],
            'mean_z2': gaussian['standardized_squared_error'], 'mean_z2_reason': gaussian['reason'],
            'coverage95': gaussian['coverage95'], 'coverage95_reason': gaussian['reason'],
            'effective_pixel_area': gaussian['effective_pixel_area'],
            'positive_weight_cells': gaussian['positive_weight_cells'],
            'status': 'empty_region' if empty else 'complete'}
    validate_table4(result)
    return result


def unavailable_table4(reason):
    return {region: {**{key: None for key in TABLE4_METRICS},
            **{key + '_reason': reason for key in TABLE4_METRICS},
            'effective_pixel_area': None, 'positive_weight_cells': None, 'status': 'unavailable_geometry'} for region in REGIONS}


def validate_table4(regions):
    require(set(regions) == set(REGIONS), 'Table4 requires exactly P/D/O')
    for region, row in regions.items():
        require(row['status'] in ('complete', 'empty_region', 'unavailable_geometry'), 'Invalid Table4 regional status')
        for key in TABLE4_METRICS:
            value, reason = row[key], row[key + '_reason']
            if value is None:
                require(isinstance(reason, str) and bool(reason), 'Undefined Table4 metric needs reason')
            else:
                require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and reason is None,
                        'Nonfinite or wrongly labelled Table4 metric')
                if key == 'coverage95':
                    require(0 <= value <= 1, 'Coverage outside[0,1]')
                elif key == 'rank_corr':
                    require(-1 <= value <= 1, 'Rank correlation outside[-1,1]')
                elif key == 'mean_z2':
                    require(value >= 0, 'Negative standardized squared residual')
        if row['status'] != 'complete':
            require(all(row[key] is None for key in TABLE4_METRICS), 'Unavailable/empty region cannot contain measurements')


def compare_values(previous, current, rtol=RTOL, atol=ATOL):
    differences = []
    exact = True
    def visit(a, b, location):
        nonlocal exact
        if isinstance(a, dict):
            require(isinstance(b, dict) and set(a) == set(b), 'Parity key mismatch: ' + location)
            for key in a:
                visit(a[key], b[key], location + '/' + key)
        elif isinstance(a, bool) or a is None or isinstance(a, str):
            require(a == b and type(a) is type(b), 'Parity status/reason mismatch: ' + location)
        elif isinstance(a, (int, float)):
            require(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(a) and math.isfinite(b), 'Parity numeric type differs')
            difference = abs(a - b)
            differences.append(difference)
            exact = exact and a == b
            if isinstance(a, int) and isinstance(b, int):
                require(a == b, 'Parity count mismatch: ' + location)
            else:
                require(math.isclose(a, b, rel_tol=rtol, abs_tol=atol), 'Table3 parity exceeded frozen tolerance: ' + location)
        else:
            raise ValueError('Unsupported parity field type: ' + location)
    visit(previous, current, 'table3')
    return {'status': 'passed', 'exact': exact, 'maximum_absolute_difference': max(differences, default=0.),
            'rtol': rtol, 'atol': atol, 'numeric_fields_compared': len(differences)}


def old_parity(ctx, row, index, model_id, status, regions, receipt):
    root = Path(ctx['protocol']['original_root'])
    path = root / 'evaluation/records' / model_id / ('%06d.json' % index)
    if not path.exists():
        return {'status': 'old_record_not_available', 'old_record': None, 'exact': None}
    original_mask_path = root / 'masks/cases' / ('case_%06d' % index) / 'receipt.json'
    old = ctx['old']
    original_mask = old.validate_mask_receipt(read(original_mask_path), row, index, CATALOG_SHA,
        ctx['protocol']['mask_contract']['sha256'], ctx['catalog']['data_root'])
    require(original_mask['status'] == receipt['status'], 'Old/new geometry differs')
    if receipt['status'] == 'complete':
        require(original_mask['native_mask_file_sha256'] == receipt['native_mask_file_sha256'] and
                original_mask['counts_native'] == receipt['counts_native'], 'Old/new native masks differ')
    prior = old.verify_record(read(path), row, index, model_id,
                             ctx['protocol']['old_evaluation_protocol']['sha256'], sha(original_mask_path))
    require(prior['checkpoint_sha256'] == old.MODEL_PINS[model_id]['sha'], 'Old checkpoint binding differs')
    result = compare_values({'status': prior['status'], 'regions': prior['regions']}, {'status': status, 'regions': regions})
    result['old_record'] = desc(path)
    result['old_mask_receipt'] = desc(original_mask_path)
    return result


def verify_record(record, ctx, row, index, model_id, receipt_sha):
    checked(record)
    require(record['schema'] == SCHEMA + '_sample' and record['index'] == index and record['sample_id'] == row['sample_id'] and
            record['subject_id'] == row['subject_id'] and record['model_id'] == model_id and
            record['checkpoint_sha256'] == ctx['old'].MODEL_PINS[model_id]['sha'] and
            record['contract_sha256'] == ctx['protocol_sha'] and record['source_row_sha256'] == digest(row) and
            record['mask_receipt_sha256'] == receipt_sha, 'Combined record identity/provenance differs')
    ref_path = bound(record['mask_ref'])
    receipt_path = bound(record['mask_receipt'])
    require(validate_ref(read(ref_path), row, index, ctx['protocol']) == receipt_path and
            record['mask_receipt']['sha256'] == receipt_sha, 'Recorded receipt/ref binding differs')
    require(record['status'] in ('complete', 'unavailable_geometry') and
            record['same_forward_table3_table4'] is (record['status'] == 'complete') and
            record['forward_count_for_this_record'] == int(record['status'] == 'complete'), 'Same-forward declaration differs')
    parity = record['table3_parity']
    require(parity['status'] in ('passed', 'old_record_not_available'), 'Invalid old Table3 parity status')
    if parity['status'] == 'passed':
        require(parity['rtol'] == RTOL and parity['atol'] == ATOL and isinstance(parity['exact'], bool), 'Parity tolerance differs')
    if record['table3_parity'].get('old_record'):
        bound(record['table3_parity']['old_record'])
        bound(record['table3_parity']['old_mask_receipt'])
    ctx['old'].verify_record(record, row, index, model_id, ctx['protocol_sha'], receipt_sha)
    validate_table4(record['table4'])
    for region in REGIONS:
        metrics = record['table4'][region]
        if record['status'] == 'complete':
            require(metrics['status'] in ('complete', 'empty_region'), 'Valid case cannot have unavailable Table4 geometry')
        else:
            reason = record['regions'][region]['l1_reason']
            require(metrics['status'] == 'unavailable_geometry' and
                    all(metrics[key] is None and metrics[key + '_reason'] == reason for key in TABLE4_METRICS),
                    'Technical Table4 case must retain original unavailability reason')
    return record


def evaluate_one(out, ctx, model_id, index, models, timeout, pilot_refs=False):
    import numpy as np
    row, old = ctx['rows'][index], ctx['old']
    ref_path, receipt_path, receipt = wait_ref(out, ctx, model_id, index, timeout, pilot_refs)
    path = out / 'records' / model_id / ('%06d.json' % index)
    if path.exists():
        for key in ('rgb', 'uv'):
            old.resolve_descriptor(row[key], ctx['catalog']['data_root'])
        if receipt['status'] == 'complete':
            old.resolve_descriptor(receipt['npz'], receipt_path.parent)
        return verify_record(read(path), ctx, row, index, model_id, sha(receipt_path))
    if receipt['status'] in old.TECHNICAL:
        reason = 'mask_' + receipt['status']
        regions, table4, status, forwarded = old.unavailable_regions(reason), unavailable_table4(reason), 'unavailable_geometry', False
    else:
        projection = old.resolve_descriptor(receipt['npz'], receipt_path.parent)
        with np.load(projection, allow_pickle=False) as bundle:
            weights = {name: bundle[name] for name in bundle.files}
        old.validate_weights(weights)
        old.validate_projection_audit(receipt, receipt_path, weights)
        rgb = old.prepare_image(row['rgb'], row['native_shape'], ctx['catalog']['data_root'])
        truth = old.prepare_image(row['uv'], row['native_shape'], ctx['catalog']['data_root'])
        if model_id not in models:
            models[model_id] = old.load_model(model_id, ctx['config']['models'][model_id],
                Path(ctx['protocol']['eval_config']['path']).parent, ctx['frozen'], 'cpu')
        prediction, sigma = old.infer(models[model_id], rgb, model_id, 'cpu')
        regions = old.measure(truth, prediction, sigma, weights, ctx['numeric'])
        table4 = table4_measure(truth, prediction, sigma, weights, old, ctx['numeric'], ctx['rank'])
        status, forwarded = 'complete', True
    parity = old_parity(ctx, row, index, model_id, status, regions, receipt)
    result = seal({'schema': SCHEMA + '_sample', 'status': status, 'index': index,
        'sample_id': row['sample_id'], 'subject_id': row['subject_id'], 'model_id': model_id,
        'checkpoint_sha256': old.MODEL_PINS[model_id]['sha'], 'contract_sha256': ctx['protocol_sha'],
        'source_row_sha256': digest(row), 'mask_receipt_sha256': sha(receipt_path),
        'mask_ref': desc(ref_path), 'mask_receipt': desc(receipt_path), 'regions': regions, 'table4': table4,
        'same_forward_table3_table4': forwarded, 'forward_count_for_this_record': int(forwarded),
        'table3_parity': parity})
    verify_record(result, ctx, row, index, model_id, sha(receipt_path))
    save(path, result)
    return result


def publish(out, model_id, **fields):
    save(out / 'progress' / (model_id + '.json'), {'identity': identity(os.getpid()), 'model_id': model_id,
        'protocol': desc(out / 'protocol.json'), 'time': time.time(), **fields}, mutable=True)


def pilot(out, timeout):
    ctx = context(out)
    mask_pilot_path = Path(ctx['protocol']['mask_root']) / 'pilot/receipt.json'
    mask_pilot = checked(read(mask_pilot_path))
    require(mask_pilot['schema'] == 'three_region_mask_parallel_pilot_v1' and mask_pilot['status'] == 'passed' and
            mask_pilot['case_count'] == 12 and mask_pilot['valid_cases'] == 11 and mask_pilot['technical_cases'] == 1 and
            mask_pilot['all_complete_native_png_array_projection_exact'] is True and
            mask_pilot['execution_amendment'] == ctx['protocol']['mask_execution_amendment'], 'Passed, bound mask pilot required')
    for comparison in mask_pilot['comparisons']:
        bound(comparison['old'])
        bound(comparison['new'])
    require(not any((out / 'completion' / (model + '.json')).exists() for model in MODELS), 'Cannot pilot after production completion')
    outcomes = {}
    for model in MODELS:
        isolate(model)
        with lock(out, model):
            models = {}
            records = [evaluate_one(out, ctx, model, i, models, timeout, pilot_refs=True) for i in range(12)]
            require(all(r['table3_parity']['status'] == 'passed' for r in records), 'Pilot needs prior Table3 records for all12')
            outcomes[model] = {'count': 12, 'forwarded': sum(r['same_forward_table3_table4'] for r in records),
                'all_table3_exact': all(r['table3_parity']['exact'] for r in records),
                'max_absolute_difference': max(r['table3_parity']['maximum_absolute_difference'] for r in records),
                'records': {str(i): desc(out / 'records' / model / ('%06d.json' % i)) for i in range(12)}}
            models.clear()
            gc.collect()
    result = seal({'status': 'complete', 'protocol': desc(out / 'protocol.json'), 'models': outcomes,
                   'mask_pilot': desc(mask_pilot_path),
                   'note': 'Same frozen forward; Table4 metrics measured, not inferred from old scalars'})
    save(out / 'pilot/receipt.json', result)
    print(json.dumps(result), flush=True)


def run(out, model_id, timeout):
    isolate(model_id)
    ctx = context(out)
    gate = checked(read(out / 'pilot/receipt.json'))
    require(gate['status'] == 'complete' and gate['protocol'] == desc(out / 'protocol.json'), 'Bound passed pilot required')
    bound(gate['mask_pilot'])
    for group in gate['models'].values():
        for item in group['records'].values():
            bound(item)
    with lock(out, model_id):
        models, completed, exact, tolerant, missing = {}, 0, 0, 0, 0
        try:
            for index in range(COUNT):
                publish(out, model_id, status='computing', completed=completed, expected=COUNT, active_index=index)
                result = evaluate_one(out, ctx, model_id, index, models, timeout)
                completed += 1
                parity = result['table3_parity']
                if parity['status'] == 'passed':
                    exact += parity['exact']
                    tolerant += not parity['exact']
                else:
                    missing += 1
                publish(out, model_id, status='computing', completed=completed, expected=COUNT,
                        old_parity_exact=exact, old_parity_within_tolerance=tolerant, old_not_available=missing)
                print(json.dumps({'event': 'combined_case_complete', 'model': model_id, 'index': index,
                                   'completed': completed, 'status': result['status']}), flush=True)
            save(out / 'completion' / (model_id + '.json'), seal({'status': 'complete', 'model_id': model_id,
                'sample_count': COUNT, 'protocol': desc(out / 'protocol.json'),
                'old_parity_exact': exact, 'old_parity_within_tolerance': tolerant, 'old_not_available': missing}))
            publish(out, model_id, status='complete', completed=COUNT, expected=COUNT)
        except BaseException as exc:
            publish(out, model_id, status='failed', completed=completed, expected=COUNT,
                    error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            models.clear()
            gc.collect()


def summary_cell(records, region, metric):
    values, reasons = [], Counter()
    for record in records:
        row = record['table4'][region]
        value = row[metric]
        if value is None:
            reasons[row[metric + '_reason']] += 1
        else:
            values.append(value)
    return {'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) >= 2 else None,
            'sample_sd_reason': None if len(values) >= 2 else 'fewer_than_two_valid_samples',
            'valid_n': len(values), 'missing_n': len(records) - len(values), 'missing_reasons': dict(reasons), 'ddof': 1}


def compact_order(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finalize(out):
    ctx = context(out)
    models, table4_models, summary = [], [], []
    with lock(out, 'finalize'):
        for model in MODELS:
            complete = checked(read(out / 'completion' / (model + '.json')))
            require(complete['status'] == 'complete' and complete['sample_count'] == COUNT and complete['model_id'] == model and
                    complete['protocol'] == desc(out / 'protocol.json'), 'Missing/incompatible model completion')
            records, table3_rows, table4_rows = [], [], []
            for index, row in enumerate(ctx['rows']):
                path = out / 'records' / model / ('%06d.json' % index)
                rec = read(path)
                checked(rec)
                verify_record(rec, ctx, row, index, model, rec['mask_receipt']['sha256'])
                records.append(rec)
                table3_rows.append(seal({'schema': ctx['old'].SCHEMA + '_sample',
                    **{key: rec[key] for key in ('status', 'index', 'sample_id', 'subject_id', 'model_id',
                       'checkpoint_sha256', 'contract_sha256', 'source_row_sha256', 'mask_receipt_sha256', 'regions')},
                    'combined_source': desc(path)}))
                table4_rows.append({'sample_id': rec['sample_id'], 'subject_id': rec['subject_id'], 'index': index,
                     'model_id': model, 'status': rec['status'], 'table4': rec['table4'], 'combined_source': desc(path)})
            table3_desc = ctx['old'].export_jsonl(out / 'exports' / (model + '_table3.jsonl'), table3_rows)
            table4_desc = ctx['old'].export_jsonl(out / 'exports' / (model + '_table4.jsonl'), table4_rows)
            common = {'model_id': model, 'checkpoint_sha256': ctx['old'].MODEL_PINS[model]['sha']}
            models.append(dict(common, records=table3_desc))
            table4_models.append(dict(common, records=table4_desc))
            for region in REGIONS:
                for metric in TABLE4_METRICS:
                    summary.append(dict(model_id=model, region=region, metric=metric,
                                        **summary_cell(records, region, metric)))
        require(len(summary) == 36, 'Incomplete Table4 summary')
        unit = {'name': 'sample_group', 'independent_subject_identity_verified': False,
                'note': 'Folder-derived groups, not independently established biological patients'}
        manifest = {'schema': 'uv_three_region_statistics_input_v1', 'models': models, 'model_order': list(MODELS),
            'expected_sample_count': COUNT, 'expected_sample_order_sha256': compact_order([r['sample_id'] for r in ctx['rows']]),
            'statistical_unit': unit, 'evaluation_protocol': desc(out / 'protocol.json')}
        save(out / 'table3_stats_manifest.json', manifest)
        save(out / 'table4_summary.json', seal({'schema': SCHEMA + '_table4_summary', 'status': 'complete',
            'sample_count': COUNT, 'cells': summary, 'expected_cells': 36, 'model_order': list(MODELS),
            'models': table4_models, 'protocol': desc(out / 'protocol.json'), 'statistical_unit': unit,
            'new_t_tests': False, 'sample_sd_not_training_or_inference_seed_variation': True}))
        save(out / 'finalization.json', seal({'status': 'complete', 'protocol': desc(out / 'protocol.json'),
            'sample_count_per_model': COUNT, 'table3_stats_manifest': desc(out / 'table3_stats_manifest.json'),
            'table4_summary': desc(out / 'table4_summary.json')}))
    print(json.dumps({'status': 'finalized', 'receipt': desc(out / 'finalization.json')}), flush=True)


def status(out):
    print(json.dumps({'root': str(out), 'models': {model: read(out / 'progress' / (model + '.json'))
        if (out / 'progress' / (model + '.json')).exists() else None for model in MODELS},
        'finalized': (out / 'finalization.json').exists()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('configure', 'pilot', 'run', 'finalize', 'status'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--eval-root', type=Path)
    parser.add_argument('--mask-continuation', type=Path)
    parser.add_argument('--old-runner', type=Path)
    parser.add_argument('--rank-helper', type=Path)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--wait-timeout', type=float, default=86400)
    args = parser.parse_args()
    require(math.isfinite(args.wait_timeout) and args.wait_timeout > 0, 'Wait timeout must be positive and finite')
    if args.mode == 'configure':
        require(all((args.eval_root, args.mask_continuation, args.old_runner, args.rank_helper)), 'Missing configure inputs')
        configure(args)
    elif args.mode == 'run':
        require(args.model in MODELS, 'Production requires one explicitly selected model per process')
        run(args.out.resolve(), args.model, args.wait_timeout)
    elif args.mode == 'pilot':
        pilot(args.out.resolve(), args.wait_timeout)
    else:
        globals()[args.mode](args.out.resolve())


if __name__ == '__main__':
    main()
