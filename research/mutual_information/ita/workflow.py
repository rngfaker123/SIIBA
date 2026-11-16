"""Versioned full-clean-TEST ITA-stratified MI; original scientific code stays read-only.

configure --request FILE; scan|pilot|run|verify --root DIR. scan is non-generating:
it only joins published mask evidence. run additionally resolves native sources,
recovers missing OLD native masks with exact parity, and runs the frozen MI runner.
All six strata retain every qualified group, without caps, replacement or merging.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

SCHEMA = 'ita_stratified_test_mi_20261026_v1'
COUNT = 21899
CPUS = [60, 61, 62, 63]
SEEDS, SIZES = [20261025, 20261026, 20261027], [100, 500, 1000]
MODS, DOMAINS = ('rgb', 'uv', 'plv', 'plh'), ('P', 'D', 'O', 'F')
STRATA = ('very_light', 'light', 'intermediate', 'tan', 'brown', 'dark')
BOUNDS = {'very_light': '>55', 'light': '(41,55]', 'intermediate': '(28,41]',
          'tan': '(10,28]', 'brown': '(-30,10]', 'dark': '<=-30'}
METRICS = {'D_rgb_uv': ('D', 'uv'), 'P_rgb_uv': ('P', 'uv'), 'O_rgb_uv': ('O', 'uv'),
           'full_rgb_uv': ('F', 'uv'), 'full_rgb_plh': ('F', 'plh'), 'full_rgb_plv': ('F', 'plv')}
CATALOG_SHA = '81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89'
MASK_CONTRACT_SHA = '0ab265ad6470a6072e947fb2a69404b125c5abd30305dad386a241712daff067'
MI_SHA = '7ac6c09e9eb81cc2f91bd94e830222ccc2b7b0b5ba8a2da33c2f262c217167f9'
ESTIMATOR_SHA = 'bb6f7fbf704a3baff582a712fefbaa6bb6d13c9d962ff4fa108347db05a19b0d'
BATCH_SHA = '0c47a4f306749ac7913d5e8e9132e8b77df481651da5ffc15e728eaebd289c1a'
DESCRIPTORS = ('catalog', 'ita_protocol', 'labels', 'ita_completion', 'mask_contract',
               'execution_amendment', 'frozen_config', 'batch_module', 'mi_runner',
               'estimator', 'reference_mi_protocol', 'authorization')
PROTOCOL_FIELDS = ('seeds', 'sizes', 'noise_seed', 'k', 'modality_order', 'domain_order',
                   'metrics', 'sampling', 'sampling_rng', 'dequantization', 'geometry',
                   'regional_weighting', 'shortage_policy', 'summary',
                   'negative_estimates_clipped', 'spatial_points_are_independent_people',
                   'identities', 'runner_sha256', 'estimator_sha256', 'software')
SCIENCE_FIELDS = ('company_module_sha256', 'mask_pipeline_sha256', 'mask_config_sha256', 'parity_report_sha256')
TECHNICAL = {'no_unique_face': 'ValueError: Expected exactly one face; no mask fabricated',
             'invalid_geometry': 'ValueError: Nonfinite or out-of-image polygon vertex: no automatic clipping'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def seal(value):
    require('record_sha256' not in value, 'Already sealed')
    return dict(value, record_sha256=digest(value))


def checked(value):
    require(value.get('record_sha256') == digest({k: v for k, v in value.items() if k != 'record_sha256'}), 'Record seal mismatch')
    return value


def desc(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def bound(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str), 'Descriptor required')
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and sha(path) == value['sha256'], 'Bound source missing/changed: ' + str(path))
    return path


def save_bytes(path, payload, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable:
        require(path.read_bytes() == payload, 'Refuse to overwrite frozen file: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if mutable:
            os.replace(str(temporary), str(path))
        else:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError:
                require(path.read_bytes() == payload, 'Concurrent differing publication')
    finally:
        if temporary.exists():
            temporary.unlink()


def save(path, value, mutable=False):
    save_bytes(path, encoded(value), mutable)


def stratum_of(value):
    require(type(value) in (int, float) and math.isfinite(value) and -90 <= value <= 90, 'Invalid ITA angle')
    for cutoff, code in ((55, 'very_light'), (41, 'light'), (28, 'intermediate'), (10, 'tan'), (-30, 'brown')):
        if value > cutoff:
            return code
    return 'dark'


def normalized(row, data_root):
    result = dict(row)
    for mod in MODS:
        p = Path(row[mod]['path'])
        p = (p if p.is_absolute() else Path(data_root) / p).resolve()
        p.relative_to(Path(data_root).resolve())
        result[mod] = {'path': str(p), 'sha256': row[mod]['sha256']}
    return result


def check_labels(bundle, catalog, protocol_descriptor):
    require(bundle['schema'] == 'ita_full_f_test_labels_v1' and bundle['sample_count'] == COUNT and
            bundle['protocol'] == protocol_descriptor and len(bundle['records']) == COUNT, 'Full-F label coverage/protocol differs')
    counts = Counter()
    for index, (row, label) in enumerate(zip(catalog['rows'], bundle['records'])):
        checked(label)
        require(label['index'] == index and label['sample_id'] == row['sample_id'] and label['subject_id'] == row['subject_id'] and
                label['source_row_sha256'] == digest(row) and label['protocol_sha256'] == protocol_descriptor['sha256'] and
                label['method'] == 'rgb_ita_full_f_v1', 'Label identity/method differs')
        if label['status'] == 'ok':
            require(label['reason'] is None and label['stratum'] == stratum_of(label['ita_degrees']) and
                    label['geometry_status'] == 'complete', 'Valid ITA classification differs')
            counts[label['stratum']] += 1
        else:
            require(label['status'] == 'invalid' and label['ita_degrees'] is None and label['stratum'] is None and
                    isinstance(label['reason'], str) and bool(label['reason']), 'Invalid ITA must remain explicitly null')
            counts['unclassified'] += 1
    return counts


def load_request(request):
    require(request['schema'] == SCHEMA + '_request' and request['allow_old_native_recovery'] is True, 'Unsupported request/recovery scope')
    paths = {k: bound(request[k]) for k in DESCRIPTORS}
    if request.get('table1_manifest'):
        bound(request['table1_manifest'])
    expected = {'catalog': CATALOG_SHA, 'mask_contract': MASK_CONTRACT_SHA, 'mi_runner': MI_SHA,
                'estimator': ESTIMATOR_SHA, 'batch_module': BATCH_SHA}
    require(all(request[k]['sha256'] == v for k, v in expected.items()), 'Frozen scientific source changed')
    require(paths['estimator'] == paths['mi_runner'].with_name('mi_knn_estimator_v1.py'), 'Estimator is not runner dependency')
    root = Path(request['root']).resolve()
    oldroot, newroot = Path(request['old_mask_root']).resolve(), Path(request['parallel_mask_root']).resolve()
    require(root not in (oldroot, newroot) and root not in oldroot.parents and oldroot not in root.parents and
            root not in newroot.parents and newroot not in root.parents, 'Output overlaps original mask tasks')
    for path in paths.values():
        require(path != root and root not in path.parents, 'Scientific input inside new output')
    catalog, labels, ip = read(paths['catalog']), read(paths['labels']), read(paths['ita_protocol'])
    require(catalog['expected_count'] == COUNT and len(catalog['rows']) == COUNT and
            len({r['sample_id'] for r in catalog['rows']}) == COUNT and
            len({r['subject_id'] for r in catalog['rows']}) == COUNT and
            all(r['split'] == 'test' for r in catalog['rows']), 'Require entire frozen clean TEST')
    require(ip['catalog'] == request['catalog'] and ip['method']['version'] == 'rgb_ita_full_f_v1' and
            ip['method']['strata'] == BOUNDS and ip['sample_count'] == COUNT and
            ip['original_mask_contract'] == request['mask_contract'] and
            ip['frozen_config'] == request['frozen_config'] and ip['batch_module'] == request['batch_module'], 'ITA method/source differs')
    completion = checked(read(paths['ita_completion']))
    require(completion['status'] == 'ITA_complete' and completion['sample_count'] == COUNT and
            completion['labels'] == request['labels'] and completion['protocol'] == request['ita_protocol'], 'ITA completion binding differs')
    counts = check_labels(labels, catalog, request['ita_protocol'])
    snapshot_counts = {r['stratum']: r['test_sample_groups'] for r in completion['eligibility_snapshot']['rows'] if r['stratum'] != 'total'}
    require({k: counts[k] for k in STRATA + ('unclassified',)} == snapshot_counts, 'ITA complete counts differ')
    amendment = checked(read(paths['execution_amendment']))
    require(amendment['schema'] == 'three_region_mask_execution_amendment_v1' and
            amendment['catalog'] == request['catalog'] and amendment['original_mask_contract'] == request['mask_contract'] and
            amendment['copied_mask_contract'] == desc(newroot / 'mask_contract.json') and
            amendment['original_mask_root'] == str(oldroot) and amendment['new_mask_root'] == str(newroot / 'generated') and
            amendment['scientific_contract_unchanged'] is True and amendment['original_outputs_readonly'] is True,
            'Parallel mask execution binding differs')
    bound(amendment['wrapper'])
    bound(amendment['request'])
    reference = read(paths['reference_mi_protocol'])
    require(reference['seeds'] == SEEDS and reference['sizes'] == SIZES and reference['noise_seed'] == 20260817 and
            reference['k'] == 5 and reference['runner_sha256'] == MI_SHA and reference['estimator_sha256'] == ESTIMATOR_SHA and
            reference['frozen_provenance']['test_catalog_sha256'] == CATALOG_SHA and
            reference['negative_estimates_clipped'] is False and set(reference['metrics']) == set(METRICS), 'Reference MI science differs')
    rows = [normalized(row, catalog['data_root']) for row in catalog['rows']]
    for row in rows:
        require(all(root not in Path(row[m]['path']).parents for m in MODS), 'Output contains a native input')
    runtime = ip['runtimes']['mi']
    require(sha(runtime['python']) == runtime['python_sha256'], 'MI runtime changed')
    return dict(root=root, request=request, catalog=catalog, rows=rows, labels=labels['records'], ita_protocol=ip,
                counts=counts, amendment=amendment, mask_contract=read(paths['mask_contract']), reference=reference,
                runtime=runtime, oldroot=oldroot, newroot=newroot)


def configure(request_path):
    request = read(request_path)
    ctx = load_request(request)
    root = ctx['root']
    save(root / 'request.json', request)
    contract = seal(dict(schema=SCHEMA + '_contract', request=desc(root / 'request.json'),
        source={'workflow': desc(__file__), 'native_sources': desc(Path(__file__).with_name('native_sources.py'))},
        sample_count=COUNT, sample_order_sha256=digest([r['sample_id'] for r in ctx['rows']]),
        strata=list(STRATA), ita_bounds=BOUNDS, groups_per_stratum_cap=None, stratum_merging=False,
        stratum_execution_order=sorted(STRATA, key=lambda s: (ctx['counts'][s], STRATA.index(s))),
        selection='All fixed TEST groups with valid frozen full-F ITA in this stratum and native P/D/O/F each >=1000; catalog order',
        common_subset_all_six_metrics_sizes_seeds=True, replacement_count=0, minimum_native_points=1000,
        seeds=SEEDS, sizes=SIZES, noise_seed=20260817, k=5, estimator='KSG_1', unit='bits', ddof=1,
        reference_mi_protocol=request['reference_mi_protocol'], cpu_affinity=CPUS, workers=4, nice=19, no_gpu=True,
        old_native_recovery='Only missing old native files in a new owned directory; original process_case; exact PNG/array/count parity',
        original_results_readonly=True, source_mutations=False, training_actions=False,
        empty_stratum='Explicit not_estimable; do not merge, lower quota, replace or invent results'))
    save(root / 'contract.json', contract)
    return {'status': 'configured', 'contract': desc(root / 'contract.json')}


def context(root):
    root = Path(root).resolve()
    contract = checked(read(root / 'contract.json'))
    require(bound(contract['source']['workflow']).resolve() == Path(__file__).resolve() and
            bound(contract['source']['native_sources']).resolve() == Path(__file__).with_name('native_sources.py').resolve(), 'Frozen local execution source changed')
    require(bound(contract['request']) == root / 'request.json', 'Wrong request root')
    ctx = load_request(read(root / 'request.json'))
    require(ctx['root'] == root and contract['sample_order_sha256'] == digest([r['sample_id'] for r in ctx['rows']]) and
            contract['cpu_affinity'] == CPUS and contract['workers'] == 4 and contract['groups_per_stratum_cap'] is None,
            'New task contract differs')
    ctx['contract'] = contract
    return ctx


def verify_compact(ctx, index, path):
    item = checked(read(path))
    row, original = ctx['rows'][index], ctx['catalog']['rows'][index]
    require(item['schema'] == 'uv_three_region_test_mask_case_v1' and item['index'] == index and
            item['sample_id'] == row['sample_id'] and item['subject_id'] == row['subject_id'] and
            item['catalog_sha256'] == ctx['request']['catalog']['sha256'] and
            item['contract_sha256'] == digest(ctx['mask_contract']) and item['source_row_sha256'] == digest(original) and
            item['native_shape'] == row['native_shape'] and item['sources'] == {m: row[m] for m in MODS}, 'Mask identity/source/contract mismatch')
    for descriptor in item['evidence'].values():
        bound(descriptor)
    if item['status'] == 'complete':
        bound(item['npz'])
        projection = checked(read(bound(item['projection_audit'])))
        require(projection['npz'] == item['npz'] and projection['counts_native'] == item['counts_native'] and
                projection['contract_sha256'] == item['contract_sha256'] and projection['source_row_sha256'] == item['source_row_sha256'],
                'Projection/source binding differs')
        require(set(item['counts_native']) == set(DOMAINS) and all(type(n) is int and n >= 0 for n in item['counts_native'].values()), 'Invalid native counts')
        require(item['mask_config_sha256'] == ctx['reference']['frozen_provenance']['mask_config_sha256'], 'Mask parameters changed')
    else:
        require(item['status'] in TECHNICAL and item['counts_native'] is None and item['npz'] is None, 'Unknown mask failure is fatal')
        lines = [line.strip() for line in bound(item['failure_stderr']).read_text(encoding='utf8', errors='replace').splitlines() if line.strip()]
        require(lines and lines[-1] == TECHNICAL[item['status']], 'Technical reason evidence differs')
    return item


def read_mask(ctx, index):
    ref_path = ctx['newroot'] / 'refs' / ('%06d.json' % index)
    if ref_path.exists():
        ref = checked(read(ref_path))
        require(ref['schema'] == 'three_region_mask_receipt_ref_v1' and ref['index'] == index and
                ref['sample_id'] == ctx['rows'][index]['sample_id'] and
                ref['execution_amendment'] == ctx['request']['execution_amendment'] and
                ref['mask_contract'] == ctx['amendment']['copied_mask_contract'] and
                ref['origin'] in ('original_readonly', 'continuation'), 'New mask ref provenance differs')
        base = ctx['oldroot'] if ref['origin'] == 'original_readonly' else ctx['newroot'] / 'generated'
        path = base / 'cases' / ('case_%06d' % index) / 'receipt.json'
        require(bound(ref['receipt']).resolve() == path.resolve(), 'Ref points to another case')
        item = verify_compact(ctx, index, path)
        require(ref['status'] == item['status'], 'Ref/receipt status differs')
        if ref['origin'] == 'original_readonly':
            require(ref.get('native_retention') is None, 'Old ref cannot invent native retention')
        else:
            retention = checked(read(bound(ref['native_retention'])))
            require(retention['compact_receipt'] == ref['receipt'] and retention['counts_native'] == item['counts_native'] and
                    retention['status'] == item['status'], 'New native retention differs')
        return item, ref, desc(path), desc(ref_path)
    path = ctx['oldroot'] / 'cases' / ('case_%06d' % index) / 'receipt.json'
    if path.exists():
        item = verify_compact(ctx, index, path)
        return item, {'origin': 'original_readonly', 'native_retention': None}, desc(path), None
    return None, None, None, None


def reconcile(label, item):
    require(label['geometry_status'] == item['status'], 'ITA/mask geometry status mismatch; never exclude a discrepancy')
    if item['status'] != 'complete':
        return
    final = checked(read(bound(label['artifacts']['final_F_receipt'])))
    audit = read(bound(item['evidence']['mask/audit.json']))
    require(final['png']['sha256'] == item['native_mask_file_sha256']['F'] and
            final['array_sha256'] == audit['stages']['F']['array_sha256'] and
            final['pixels'] == item['counts_native']['F'] == audit['stages']['F']['pixels'], 'ITA and scientific mask F differ')


def classify(label, item):
    if label['status'] != 'ok':
        return 'not_included', 'ITA_unavailable:' + label['reason'], []
    if item is None:
        return 'pending', 'mask_receipt_not_yet_published', []
    if item['status'] != 'complete':
        return 'not_included', item['status'], []
    short = [d for d in DOMAINS if item['counts_native'][d] < 1000]
    return ('not_included', 'native_region_below_1000', short) if short else ('eligible', None, [])


def build_outcome(ctx, index, item, receipt, ref_descriptor):
    label, row = ctx['labels'][index], ctx['rows'][index]
    status, reason, short = classify(label, item)
    return seal(dict(schema=SCHEMA + '_case', index=index, sample_id=row['sample_id'], subject_id=row['subject_id'],
        source_row_sha256=digest(ctx['catalog']['rows'][index]), contract=desc(ctx['root'] / 'contract.json'),
        label_record_sha256=label['record_sha256'], stratum=label['stratum'], ita_degrees=label['ita_degrees'],
        status=status, reason=reason, insufficient_domains=short, counts_native=item['counts_native'] if item else None,
        compact_receipt=receipt, mask_ref=ref_descriptor))


def scan(ctx, resolver=None, ready_cache=None):
    ready_cache = ready_cache if ready_cache is not None else {}
    outcomes, ready = {}, {}
    for index, label in enumerate(ctx['labels']):
        out = ctx['root'] / 'outcomes' / ('%06d.json' % index)
        if out.exists():
            value = checked(read(out))
            require(value['index'] == index and value['sample_id'] == label['sample_id'] and
                    value['label_record_sha256'] == label['record_sha256'] and value['contract'] == desc(ctx['root'] / 'contract.json'),
                    'Cached outcome binding differs')
            if value['compact_receipt']:
                bound(value['compact_receipt'])
            if value['mask_ref']:
                bound(value['mask_ref'])
            outcomes[index] = value
        else:
            item, ref, receipt, ref_descriptor = read_mask(ctx, index) if label['status'] == 'ok' else (None, None, None, None)
            if item is not None:
                reconcile(label, item)
            value = build_outcome(ctx, index, item, receipt, ref_descriptor)
            outcomes[index] = value
            if value['status'] != 'pending':
                save(out, value)
        if value['status'] == 'eligible' and index in ready_cache:
            # 估计前校验哈希
            # 复用已绑定掩膜

            bound(ready_cache[index]['proof'])
            ready[index] = ready_cache[index]
        elif resolver is not None and value['status'] == 'eligible':
            item = verify_compact(ctx, index, bound(value['compact_receipt']))
            if value['mask_ref']:
                ref = checked(read(bound(value['mask_ref'])))
            else:
                ref = {'origin': 'original_readonly', 'native_retention': None}
            result = resolver.resolve(index, item, ref, receipt_descriptor=value['compact_receipt'])
            require(result['status'] in ('ready', 'pending'), 'Unexpected native resolution status')
            if result['status'] == 'ready':
                require(set(result['masks']) == set(DOMAINS), 'Incomplete native mask descriptors')
                ready[index] = result
                ready_cache[index] = result
        if (index + 1) % 100 == 0:
            publish_state(ctx, 'joining', outcomes, ready, inspected=index + 1)
    publish_state(ctx, 'joined', outcomes, ready, inspected=COUNT)
    return outcomes, ready


def composition(outcomes, ready):
    result = {s: dict(stratum=s, test_sample_groups=0, eligible_for_MI=0, not_included_in_MI=0,
                     eligibility_pending=0, native_ready=0, native_pending=0) for s in STRATA + ('unclassified',)}
    reasons = Counter()
    for index, value in outcomes.items():
        row = result[value['stratum'] or 'unclassified']
        row['test_sample_groups'] += 1
        if value['status'] == 'eligible':
            row['eligible_for_MI'] += 1
            row['native_ready' if index in ready else 'native_pending'] += 1
        elif value['status'] == 'not_included':
            row['not_included_in_MI'] += 1
            reasons[value['reason']] += 1
        else:
            require(value['status'] == 'pending', 'Unknown eligibility')
            row['eligibility_pending'] += 1
    rows = list(result.values())
    total = {'stratum': 'total'}
    for key in rows[0]:
        if key != 'stratum':
            total[key] = sum(r[key] for r in rows)
    require(all(r['test_sample_groups'] == r['eligible_for_MI'] + r['not_included_in_MI'] + r['eligibility_pending'] for r in rows + [total]),
            'Composition counting mismatch')
    return rows + [total], dict(reasons)


def publish_state(ctx, status, outcomes=None, ready=None, **fields):
    value = dict(schema=SCHEMA + '_progress', status=status, updated_unix=time.time(), **fields)
    if outcomes is not None:
        displayed = dict(outcomes)
        for i, label in enumerate(ctx['labels']):
            if i not in displayed:
                displayed[i] = dict(stratum=label['stratum'], status='pending' if label['status'] == 'ok' else 'not_included',
                    reason=None if label['status'] == 'ok' else 'ITA_unavailable:' + label['reason'])
        rows, reasons = composition(displayed, ready or {})
        value.update(table2a=rows, not_included_reasons=reasons)
    save(ctx['root'] / 'progress.json', value, mutable=True)


def freeze_stratum(ctx, stratum, outcomes, ready):
    indices = [i for i, r in enumerate(ctx['labels']) if r['stratum'] == stratum]
    require(all(i in outcomes for i in indices), 'Incomplete stratum join')
    if any(outcomes[i]['status'] == 'pending' for i in indices):
        return None
    admitted = [i for i in indices if outcomes[i]['status'] == 'eligible']
    if any(i not in ready for i in admitted):
        return None
    root = ctx['root'] / 'strata' / stratum
    selection = seal(dict(schema=SCHEMA + '_selection', status='frozen' if admitted else 'not_estimable',
        stratum=stratum, ita_range=BOUNDS[stratum], contract=desc(ctx['root'] / 'contract.json'),
        test_sample_count=len(indices), admitted_count=len(admitted), excluded_count=len(indices) - len(admitted),
        exclusion_reasons=dict(Counter(outcomes[i]['reason'] for i in indices if outcomes[i]['status'] != 'eligible')),
        catalog_indices=indices, admitted_catalog_indices=admitted,
        sample_order=[ctx['rows'][i]['sample_id'] for i in admitted],
        outcomes=[desc(ctx['root'] / 'outcomes' / ('%06d.json' % i)) for i in indices],
        native_proofs=[ready[i]['proof'] for i in admitted],
        groups_per_stratum_cap=None, replacement_count=0, stratum_merging=False,
        same_subset_all_metrics_sizes_seeds=True, empty_reason=None if admitted else 'no_qualified_groups'))
    save(root / 'selection.json', selection)
    if not admitted:
        return selection
    provenance = {k: ctx['reference']['frozen_provenance'][k] for k in SCIENCE_FIELDS}
    provenance.update(stratification_contract_sha256=digest(ctx['contract']), ita_protocol_sha256=ctx['request']['ita_protocol']['sha256'],
        labels_sha256=ctx['request']['labels']['sha256'], selection_record_sha256=selection['record_sha256'],
        test_catalog_sha256=ctx['request']['catalog']['sha256'])
    rows = [dict(sample_id=ctx['rows'][i]['sample_id'], subject_id=ctx['rows'][i]['subject_id'], native_shape=ctx['rows'][i]['native_shape'],
                 **{m: ctx['rows'][i][m] for m in MODS}, **ready[i]['masks']) for i in admitted]
    manifest = dict(schema='mi_atlas_six_metrics_input_v1', status='frozen',
        cohort_id='full_clean_TEST_ITA_' + stratum + '_' + selection['record_sha256'][:12],
        expected_sample_count=len(rows), frozen_provenance=provenance, rows=rows)
    save(root / 'mi_input.json', manifest)
    return selection


def verify_stratum(ctx, stratum):
    root = ctx['root'] / 'strata' / stratum
    selection = checked(read(root / 'selection.json'))
    expected_indices = [i for i, r in enumerate(ctx['labels']) if r['stratum'] == stratum]
    require(selection['contract'] == desc(ctx['root'] / 'contract.json') and selection['stratum'] == stratum and
            selection['catalog_indices'] == expected_indices and selection['test_sample_count'] == len(expected_indices) and
            selection['same_subset_all_metrics_sizes_seeds'] is True and selection['groups_per_stratum_cap'] is None and
            selection['replacement_count'] == 0 and selection['stratum_merging'] is False, 'Selection scope differs')
    records = [checked(read(bound(d))) for d in selection['outcomes']]
    require([r['index'] for r in records] == expected_indices and all(r['status'] in ('eligible', 'not_included') for r in records), 'Selection not final')
    for item in records:
        i = item['index']
        require(item['label_record_sha256'] == ctx['labels'][i]['record_sha256'] and item['sample_id'] == ctx['rows'][i]['sample_id'] and
                item['contract'] == selection['contract'] and item['stratum'] == stratum, 'Selection outcome differs')
        compact = verify_compact(ctx, i, bound(item['compact_receipt'])) if item['compact_receipt'] else None
        require(compact is not None, 'A valid-ITA final outcome needs an actual mask receipt')
        reconcile(ctx['labels'][i], compact)
        status, reason, short = classify(ctx['labels'][i], compact)
        require((item['status'], item['reason'], item['insufficient_domains'], item['counts_native']) ==
                (status, reason, short, compact['counts_native']), 'Outcome eligibility differs from original evidence')
        if item['mask_ref']:
            bound(item['mask_ref'])
    admitted = [r['index'] for r in records if r['status'] == 'eligible']
    n = len(admitted)
    require(selection['admitted_count'] == n and selection['admitted_catalog_indices'] == admitted and
            selection['sample_order'] == [ctx['rows'][i]['sample_id'] for i in admitted] and
            selection['excluded_count'] == len(expected_indices) - n and
            selection['exclusion_reasons'] == dict(Counter(r['reason'] for r in records if r['status'] != 'eligible')),
            'Selection count/order differs')
    if n == 0:
        require(selection['status'] == 'not_estimable' and selection['native_proofs'] == [], 'Empty stratum must be explicit')
        return dict(stratum=stratum, status='not_estimable', sample_count=0, reason='no_qualified_groups', selection=desc(root / 'selection.json'), cells=[])
    require(selection['status'] == 'frozen' and len(selection['native_proofs']) == n, 'Missing native provenance')
    native_proofs = [checked(read(bound(proof))) for proof in selection['native_proofs']]
    manifest = read(root / 'mi_input.json')
    require(manifest['schema'] == 'mi_atlas_six_metrics_input_v1' and manifest['status'] == 'frozen' and manifest['expected_sample_count'] == n and
            [r['sample_id'] for r in manifest['rows']] == selection['sample_order'] and
            [r['subject_id'] for r in manifest['rows']] == [ctx['rows'][i]['subject_id'] for i in admitted], 'Manifest cohort differs')
    require(all(manifest['frozen_provenance'][k] == ctx['reference']['frozen_provenance'][k] for k in SCIENCE_FIELDS) and
            manifest['frozen_provenance']['selection_record_sha256'] == selection['record_sha256'], 'Manifest science/selection differs')
    compact_records = {}
    eligible_records = [r for r in records if r['status'] == 'eligible']
    for index, row, proof, outcome in zip(admitted, manifest['rows'], native_proofs, eligible_records):
        original = ctx['rows'][index]
        compact = checked(read(bound(outcome['compact_receipt'])))
        compact_records[index] = compact
        require(row['native_shape'] == original['native_shape'] and all(row[m] == original[m] for m in MODS), 'Manifest native source differs')
        require(proof['schema'] == 'ita_stratified_native_sources_v1' and proof['status'] == 'ready' and
                proof['index'] == index and proof['sample_id'] == row['sample_id'] and proof['subject_id'] == row['subject_id'] and
                proof['request_sha256'] == digest(ctx['request']) and proof['helper_sha256'] == ctx['contract']['source']['native_sources']['sha256'] and
                proof['compact_receipt'] == outcome['compact_receipt'] and proof['counts_native'] == compact['counts_native'] and
                proof['mask_contract'] == ctx['request']['mask_contract'] and proof['original_outputs_modified'] is False and
                proof['continuation_regenerated'] is False and all(proof['masks'][d] == row[d] for d in DOMAINS) and
                all(row[d]['sha256'] == compact['native_mask_file_sha256'][d] for d in DOMAINS), 'Native proof/manifest binding differs')
        native_summary = read(bound(proof['original_summary']))
        require(proof['native_array_sha256'] == native_summary['array_sha256'] and native_summary['counts'] == compact['counts_native'], 'Native array proof differs')
        for evidence in proof['supporting_evidence'].values():
            bound(evidence)
    folder = root / 'mi_run'
    protocol = read(folder / 'protocol.json')
    require(protocol['input_manifest_sha256'] == sha(root / 'mi_input.json') and protocol['input_contract_sha256'] == digest(manifest) and
            protocol['expected_sample_count'] == n and protocol['sample_order'] == selection['sample_order'] and
            protocol['subject_order'] == [r['subject_id'] for r in manifest['rows']] and protocol['frozen_provenance'] == manifest['frozen_provenance'] and
            all(protocol[k] == ctx['reference'][k] for k in PROTOCOL_FIELDS), 'MI protocol differs')
    prep = checked(read(folder / 'preparation.json'))
    require(prep['status'] == 'complete' and prep['protocol_sha256'] == digest(protocol) and prep['expected_sample_count'] == n and
            prep['sample_order'] == selection['sample_order'] and len(prep['features']) == n and
            all(prep[k] is True for k in ('all_samples_all_domains_at_least_1000', 'no_population_replacement', 'no_duplicate_padding')), 'MI preparation incomplete')
    for index, (row, feature) in enumerate(zip(manifest['rows'], prep['features'])):
        feature_path = folder / 'features' / ('sample_%06d.npz' % index)
        metadata_path = feature_path.with_suffix('.json')
        require(feature['index'] == index and feature['sample_id'] == row['sample_id'] and feature['subject_id'] == row['subject_id'] and
                feature['feature_path'] == feature_path.relative_to(folder).as_posix() and sha(feature_path) == feature['feature_sha256'] and
                sha(metadata_path) == feature['metadata_sha256'], 'Feature inventory differs')
        meta = checked(read(metadata_path))
        require(meta['index'] == index and meta['sample_id'] == row['sample_id'] and meta['subject_id'] == row['subject_id'] and
                meta['protocol_sha256'] == digest(protocol) and meta['source_row_sha256'] == digest(row) and
                meta['feature_sha256'] == feature['feature_sha256'] and meta['partition_verified'] is True and
                meta['source_hashes_verified'] is True and meta['sampled_points_per_domain_per_seed'] == 1000 and
                meta['source_sha256'] == {key: row[key]['sha256'] for key in MODS + DOMAINS} and
                meta['mask_pixels'] == compact_records[admitted[index]]['counts_native'] and
                set(meta['mask_pixels']) == set(DOMAINS) and all(v >= 1000 for v in meta['mask_pixels'].values()), 'Feature seal/provenance differs')
    require(len(list((folder / 'features').glob('sample_*.npz'))) == n and len(list((folder / 'features').glob('sample_*.json'))) == n, 'Extra/missing features')
    summary = checked(read(folder / 'summary.json'))
    require(summary['status'] == 'complete' and summary['actual_result_count'] == summary['expected_result_count'] == 54 and
            summary['protocol_sha256'] == digest(protocol) and summary['preparation_sha256'] == prep['record_sha256'] and
            summary['unit'] == 'bits' and len(summary['summaries']) == 18, 'Summary completion binding differs')
    cells, inventory = [], {}
    lookup = {(r['metric'], r['points_per_sample']): r for r in summary['summaries']}
    require(len(lookup) == 18, 'Duplicate summary cell')
    for metric, (domain, target) in METRICS.items():
        for size in SIZES:
            values = []
            for seed in SEEDS:
                name = '%s_n%d_seed%d.json' % (metric, size, seed)
                record = checked(read(folder / 'results' / name))
                expected = dict(protocol_sha256=digest(protocol), preparation_sha256=prep['record_sha256'],
                    metric=metric, domain=domain, target=target, points_per_sample=size, sampling_seed=seed,
                    sample_count=n, observations=n * size, k=5, dx=3, dy=3, status='complete')
                require(all(record.get(k) == v for k, v in expected.items()), 'Raw MI result binding differs')
                result = record['result']
                require(result['n'] == n * size and result['k'] == 5 and result['dx'] == result['dy'] == 3 and
                        result['estimator'] == 'KSG_1' and result['unit'] == 'bits' and result['negative_estimates_clipped'] is False and
                        1 <= result['workers'] <= 4 and
                        type(result['mi_bits']) in (int, float) and math.isfinite(result['mi_bits']), 'Raw MI numeric metadata differs')
                values.append(result['mi_bits'])
                inventory[name] = record['record_sha256']
            mean, std = statistics.mean(values), statistics.stdev(values)
            reported = lookup[(metric, size)]
            require(reported['mi_bits_by_seed'] == values and reported['sampling_seeds'] == SEEDS and reported['sample_count'] == n and
                    reported['observations_per_repeat'] == n * size and reported['ddof'] == 1 and reported['repetitions'] == 3 and
                    math.isclose(reported['mean_bits'], mean, rel_tol=1e-12, abs_tol=1e-12) and
                    math.isclose(reported['sample_sd_bits'], std, rel_tol=1e-12, abs_tol=1e-12), 'Independent mean/sample SD differs')
            cells.append(dict(metric=metric, points_per_sample=size, sample_count=n, observations_per_repeat=n * size,
                mean_bits=mean, sample_sd_bits=std, mi_bits_by_seed=values, ddof=1, repetitions=3))
    require(inventory == summary['result_record_sha256'] and set(inventory) == {p.name for p in (folder / 'results').glob('*.json')}, 'Raw result inventory differs')
    return dict(stratum=stratum, status='complete', sample_count=n, cells=cells, result_count=54,
                selection=desc(root / 'selection.json'), mi_input=desc(root / 'mi_input.json'),
                protocol=desc(folder / 'protocol.json'), preparation=desc(folder / 'preparation.json'), summary=desc(folder / 'summary.json'))


def report(ctx):
    strata = [verify_stratum(ctx, s) for s in STRATA]
    outcomes = {i: checked(read(ctx['root'] / 'outcomes' / ('%06d.json' % i))) for i in range(COUNT)}
    for index, outcome in outcomes.items():
        label = ctx['labels'][index]
        require(outcome['index'] == index and outcome['sample_id'] == label['sample_id'] and
                outcome['subject_id'] == label['subject_id'] and outcome['label_record_sha256'] == label['record_sha256'] and
                outcome['contract'] == desc(ctx['root'] / 'contract.json'), 'Global report outcome differs')
        if label['status'] != 'ok':
            require(outcome['status'] == 'not_included' and outcome['reason'] == 'ITA_unavailable:' + label['reason'], 'Invalid ITA counted differently')
    ready = {i: True for i, r in outcomes.items() if r['status'] == 'eligible'}
    rows, reasons = composition(outcomes, ready)
    require(rows[-1]['test_sample_groups'] == COUNT and rows[-1]['eligibility_pending'] == 0, 'Final TEST coverage incomplete')
    notes = ['All eligible groups in each fixed TEST ITA stratum; no cap, replacement or merging.',
        'Six metrics share one stratum-specific subset; native P/D/O/F each have at least 1000 pixels.',
        'Mean +/- sample SD across three sampling seeds (ddof=1), bits; SD is not a population confidence interval.',
        'Different stratum sizes affect estimation; tiny strata are descriptive and do not establish skin-tone-wide consistency.',
        'Folder-derived sample groups are not independently verified patients. Pixel MI does not prove physical irrecoverability.']
    summary = seal(dict(schema=SCHEMA + '_report', status='complete', contract=desc(ctx['root'] / 'contract.json'),
        labels=ctx['request']['labels'], table2a=rows, not_included_reasons=reasons, strata=strata,
        verified_results=sum(r.get('result_count', 0) for r in strata), independently_recomputed_cells=sum(len(r['cells']) for r in strata),
        unit='bits', ddof=1, notes=notes))
    lines = ['Table 2: ITA-stratified Cross-modal Mutual Information', '', 'Full clean TEST: 21899 sample groups', '',
             'Stratum | TEST groups | MI eligible | Not included']
    lines += ['%s | %d | %d | %d' % (r['stratum'], r['test_sample_groups'], r['eligible_for_MI'], r['not_included_in_MI']) for r in rows]
    for result in strata:
        lines += ['', result['stratum'] + ': N=' + str(result['sample_count']), 'Metric | 100 points | 500 points | 1000 points']
        if result['status'] == 'not_estimable':
            lines.append('Not estimable: no qualified groups')
            continue
        lookup = {(r['metric'], r['points_per_sample']): r for r in result['cells']}
        for metric in METRICS:
            values = [lookup[(metric, size)] for size in SIZES]
            lines.append(metric + ' | ' + ' | '.join('%.6f +/- %.6f' % (v['mean_bits'], v['sample_sd_bits']) for v in values))
    lines += ['', *notes]
    text = '\n'.join(lines) + '\n'
    output = ctx['root'] / 'report'
    save(output / 'summary.json', summary)
    save_bytes(output / 'report.txt', text.encode('utf8'))
    save_bytes(output / 'report.html', ('<!doctype html><meta charset="utf-8"><title>ITA-stratified MI</title><pre>' + html.escape(text) + '</pre>').encode('utf8'))
    receipt = seal(dict(status='complete', contract=summary['contract'], verified_results=summary['verified_results'],
        independently_recomputed_cells=summary['independently_recomputed_cells'],
        artifacts={n: desc(output / n) for n in ('summary.json', 'report.txt', 'report.html')}))
    save(output / 'receipt.json', receipt)
    save(ctx['root'] / 'completion.json', seal(dict(status='complete', contract=summary['contract'], report=desc(output / 'receipt.json'),
        sample_count=COUNT, strata_count=6, mi_calculated=True, verified_results=summary['verified_results'])))
    return receipt


def identity(pid):
    value = Path('/proc/%d/stat' % pid).read_text()
    fields = value[value.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def active_mi_child(folder):
    active = []
    for path in sorted(folder.glob('child_*.json')):
        receipt = checked(read(path))
        require(receipt['selection'] == desc(folder / 'selection.json') and receipt['mi_input'] == desc(folder / 'mi_input.json'),
                'Existing MI child belongs to another selection')
        expected = receipt['identity']
        try:
            actual = identity(expected['pid'])
        except FileNotFoundError:
            continue
        if actual != expected:
            continue
        stat = Path('/proc/%d/stat' % expected['pid']).read_text()
        if stat[stat.rfind(')') + 2:].split()[0] == 'Z':
            continue
        command = [part.decode() for part in Path('/proc/%d/cmdline' % expected['pid']).read_bytes().split(b'\0') if part]
        require(command == receipt['command'], 'Live child identity matches but command differs; stop investigation before launch')
        active.append(receipt)
    require(len(active) <= 1, 'Multiple live MI children for one stratum')
    return active[0] if active else None


def enforce_resources():
    require(hasattr(os, 'sched_setaffinity'), 'Production execution requires Linux CPU affinity')
    os.sched_setaffinity(0, CPUS)
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 19:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)


@contextmanager
def task_lock(root):
    import fcntl
    path = root / 'workflow.lock'
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(identity(os.getpid())))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_env(runtime):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    env['PYTHONPATH'] = os.pathsep.join(runtime.get('pythonpath', []))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def verify_mi_runtime(ctx):
    code = 'import sys,json,numpy,scipy;print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,scipy=scipy.__version__)))'
    result = subprocess.run([ctx['runtime']['python'], '-c', code], env=runtime_env(ctx['runtime']),
                            capture_output=True, text=True, check=True, timeout=120)
    actual = json.loads(result.stdout)
    require(actual == ctx['reference']['software'], 'Frozen MI runtime software differs before launch')
    return actual


def pilot(root):
    """Exercise one genuinely missing OLD native case before general dispatch."""
    enforce_resources()
    ctx = context(root)
    software = verify_mi_runtime(ctx)
    from native_sources import NativeResolver
    resolver = NativeResolver(ctx['request'], ctx['rows'], ctx['root'] / 'native_sources')
    with task_lock(ctx['root']):
        proof_descriptor = None
        reused = []
        for index, label in enumerate(ctx['labels']):
            if label['status'] != 'ok':
                continue
            item, ref, receipt, _ = read_mask(ctx, index)
            if item is None or ref['origin'] != 'original_readonly':
                continue
            reconcile(label, item)
            if classify(label, item)[0] != 'eligible':
                continue
            result = resolver.resolve(index, item, ref, receipt_descriptor=receipt)
            require(result['status'] == 'ready', 'Authorized native pilot failed to resolve')
            proof = checked(read(bound(result['proof'])))
            if proof['mode'] == 'old_native_exact_recovery':
                proof_descriptor = result['proof']
                break
            reused.append(result['proof'])
        require(proof_descriptor is not None or reused, 'No qualifying OLD native source available for pilot')
        result = seal(dict(schema=SCHEMA + '_pilot', status='passed', contract=desc(ctx['root'] / 'contract.json'),
            software=software, recovery_path_exercised=proof_descriptor is not None,
            recovered_native_proof=proof_descriptor, preceding_exact_existing_proofs=reused,
            png_array_count_exact=True, original_outputs_modified=False, continuation_regenerated=False))
        save(ctx['root'] / 'pilot/receipt.json', result)
        return result


def run(root, poll_seconds=120):
    enforce_resources()
    ctx = context(root)
    verify_mi_runtime(ctx)
    gate = checked(read(ctx['root'] / 'pilot/receipt.json'))
    require(gate['schema'] == SCHEMA + '_pilot' and gate['status'] == 'passed' and
            gate['contract'] == desc(ctx['root'] / 'contract.json') and gate['png_array_count_exact'] is True and
            gate['original_outputs_modified'] is False and gate['continuation_regenerated'] is False and
            gate['software'] == ctx['reference']['software'], 'Native pilot not accepted')
    for proof in ([gate['recovered_native_proof']] if gate['recovered_native_proof'] else gate['preceding_exact_existing_proofs']):
        checked(read(bound(proof)))
    from native_sources import NativeResolver
    resolver = NativeResolver(ctx['request'], ctx['rows'], ctx['root'] / 'native_sources')
    ready_cache = {}
    with task_lock(ctx['root']):
        save(ctx['root'] / 'processes' / ('%d.json' % os.getpid()), seal(dict(identity=identity(os.getpid()),
            contract=desc(ctx['root'] / 'contract.json'), cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0))))
        try:
            while True:
                outcomes, ready = scan(ctx, resolver, ready_cache)
                table2a, reasons = composition(outcomes, ready)
                snapshot = seal(dict(schema=SCHEMA + '_table2a', contract=desc(ctx['root'] / 'contract.json'), rows=table2a, reasons=reasons))
                path = ctx['root'] / 'snapshots' / (snapshot['record_sha256'] + '.json')
                save(path, snapshot)
                save(ctx['root'] / 'table2a_current.json', dict(snapshot=desc(path)), mutable=True)
                active = [child for s in STRATA for child in [active_mi_child(ctx['root'] / 'strata' / s)] if child]
                require(len(active) <= 1, 'Multiple live stratum runners; refuse additional CPU work')
                if active:
                    publish_state(ctx, 'existing_mi_child_running', outcomes, ready, child=active[0])
                    time.sleep(poll_seconds)
                    continue
                completed = []
                for stratum in ctx['contract']['stratum_execution_order']:
                    selection = freeze_stratum(ctx, stratum, outcomes, ready)
                    if selection is None:
                        continue
                    folder = ctx['root'] / 'strata' / stratum
                    if selection['admitted_count'] and not (folder / 'mi_run/summary.json').exists():
                        active = active_mi_child(folder)
                        if active:
                            publish_state(ctx, 'existing_mi_child_running', outcomes, ready, stratum=stratum, child=active)
                            break
                        require(not (folder / 'mi_run/mi_atlas.lock').exists(),
                                'Original MI lock remains without a matching live child; inspect exact process before explicit recovery')
                        log_path = folder / ('mi_' + uuid.uuid4().hex + '.log')
                        command = [ctx['runtime']['python'], str(bound(ctx['request']['mi_runner'])),
                                   '--manifest', str(folder / 'mi_input.json'), '--output-dir', str(folder / 'mi_run'), '--workers', '4', '--phase', 'all']
                        with log_path.open('xb') as stream:
                            child = subprocess.Popen(command, env=runtime_env(ctx['runtime']), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                            child_receipt = seal(dict(identity=identity(child.pid), command=command, log=str(log_path), stratum=stratum,
                                selection=desc(folder / 'selection.json'), mi_input=desc(folder / 'mi_input.json')))
                            save(folder / ('child_%d.json' % child.pid), child_receipt)
                            publish_state(ctx, 'mi_running', outcomes, ready, stratum=stratum, child=child_receipt)
                            require(child.wait() == 0, 'Frozen MI runner failed: ' + str(log_path))
                    verify_stratum(ctx, stratum)
                    completed.append(stratum)
                if len(completed) == len(STRATA):
                    result = report(ctx)
                    publish_state(ctx, 'complete', outcomes, ready, report=result)
                    return result
                publish_state(ctx, 'waiting_for_published_masks', outcomes, ready, completed_strata=completed)
                time.sleep(poll_seconds)
        except Exception as error:
            publish_state(ctx, 'failed', error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('configure', 'scan', 'pilot', 'run', 'verify'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--poll-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.action == 'configure':
        parser.error('--request required') if args.request is None else None
        result = configure(args.request)
    else:
        parser.error('--root required') if args.root is None else None
        require(30 <= args.poll_seconds <= 3600, 'Polling interval must be 30..3600 seconds')
        if args.action == 'run':
            result = run(args.root, args.poll_seconds)
        elif args.action == 'pilot':
            result = pilot(args.root)
        elif args.action == 'verify':
            result = report(context(args.root))
        else:
            ctx = context(args.root)
            with task_lock(ctx['root']):
                outcomes, ready = scan(ctx)
            rows, reasons = composition(outcomes, ready)
            result = dict(status='joined_without_native_generation', table2a=rows, reasons=reasons)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
