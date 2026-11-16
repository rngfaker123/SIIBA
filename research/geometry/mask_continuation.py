"""Execution-only continuation of frozen TEST masks; original artifacts read-only.

The historical mask contract retains workers=1. A separate sealed amendment
records actual workers=4 and new output paths. Original process_one/projector
implementations are loaded by exact SHA; no scientific function is replaced.
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

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
MASK_SHA = '65030669c34f9b1b90ae0e75313734ea60d25fe6f7236979ac51cbb487555fe9'
BATCH_SHA = '0c47a4f306749ac7913d5e8e9132e8b77df481651da5ffc15e728eaebd289c1a'
BUILDER_SHA = 'aa05bbca42235676637fea1cb5fa53832947e110bc172ef7441fc680506d07ff'
CATALOG_SHA = '81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89'
CONTRACT_SHA = '0ab265ad6470a6072e947fb2a69404b125c5abd30305dad386a241712daff067'
COUNT, WORKERS, CPUS = 21899, 4, [68, 69, 70, 71]
STATUSES = ('complete', 'no_unique_face', 'invalid_geometry')
DOMAINS = ('F', 'P', 'D', 'O')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def desc(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def bound(value):
    path = Path(value['path'])
    require(path.is_absolute() and sha(path) == value['sha256'], 'Bound artifact changed: '+str(path))
    return path


def seal(value):
    return dict(value, record_sha256=digest(value))


def checked(value):
    require(value.get('record_sha256') == digest({k:v for k,v in value.items() if k != 'record_sha256'}), 'Invalid record seal')
    return value


def save_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Refuse overwrite: '+str(path))
        return
    tmp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(path.read_bytes() == data, 'Concurrent differing artifact')
    finally:
        if tmp.exists():
            tmp.unlink()


def save(path, value, mutable=False):
    if not mutable:
        return save_bytes(path, encoded(value))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        f.write(encoded(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def import_file(path, name, expected):
    require(sha(path) == expected, 'Frozen code differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a != b and a not in b.parents and b not in a.parents


def validate_request(request):
    require(request.get('schema') == 'three_region_mask_continuation_request_v1', 'Unsupported request')
    require(request.get('workers') == WORKERS and request.get('cpu_affinity') == CPUS and request.get('nice') == 19 and request.get('ionice_class') == 3 and request.get('no_gpu') is True, 'Execution allocation differs')
    for key, expected in (('mask_module',MASK_SHA), ('batch_module',BATCH_SHA), ('catalog',CATALOG_SHA), ('original_mask_contract',CONTRACT_SHA)):
        require(request[key]['sha256'] == expected, 'Unapproved '+key)
        bound(request[key])
    bound(request['frozen_config'])
    auth = read(bound(request['authorization']))
    require(isinstance(auth.get('human_request'), str) and auth['human_request'].strip(), 'Missing human authorization')
    require(all(auth.get(k) is True for k in ('masks_unchanged','company_unchanged','no_training_mutations','original_outputs_readonly')), 'Authorization safeguards absent')
    root, oldroot = Path(request['root']).resolve(), Path(request['original_mask_root']).resolve()
    require(Path(request['root']).is_absolute() and disjoint(root, oldroot), 'New output must be disjoint from original mask tree')
    require(bound(request['original_mask_contract']).resolve() == oldroot/'mask_contract.json', 'Wrong original mask contract location')
    for key in ('mask_module','batch_module','catalog','original_mask_contract','frozen_config','authorization'):
        p = bound(request[key]).resolve()
        require(p != root and root not in p.parents, 'Frozen input inside mutable new mask root')
    return root, oldroot


def make_context(request):
    root, oldroot = validate_request(request)
    masks = import_file(bound(request['mask_module']), 'original_mask_implementation', MASK_SHA)
    batchpath = bound(request['batch_module'])
    batch = import_file(batchpath, 'original_mask_batch', BATCH_SHA)
    builder = import_file(batchpath.with_name('mi_atlas_cohort_20261025.py'), 'original_catalog_builder', BUILDER_SHA)
    old = batch.load_settings(bound(request['frozen_config']))
    require(read(old['output']/'batch_contract.json') == old['contract'], 'Original batch contract differs')
    catalog, rows = masks.validate_catalog(batch, builder, bound(request['catalog']), old)
    contract = read(bound(request['original_mask_contract']))
    # 重建原TEST协议
    expected = dict(old['contract'])
    expected.update(schema='uv_three_region_test_mask_contract_v1',catalog=batch.descriptor(bound(request['catalog'])),
        original_config=batch.descriptor(bound(request['frozen_config'])),original_contract=batch.descriptor(old['output']/'batch_contract.json'),
        mask_projector_sha256=MASK_SHA,source_row_order_sha256=batch.digest(catalog['rows']),normalized_source_rows_sha256=batch.digest(rows),
        sample_order=[r['sample_id'] for r in rows],test_sample_count=len(rows),mask_workers=1,
        projection=dict(size=1024,dtype='float32',resampling='Pillow_BOX_mode_F',crop='center_square_floor_offsets',partition_tolerance=2e-6),
        eligibility='Every clean TEST row retained; no minimum region count; no replacement; metrics use own effective support and valid N',
        invalid_geometry_exception=masks.INVALID_GEOMETRY,empty_domain_is_exclusion=False,minimum_pixel_count_filter_applied=False,
        source_row_digest_encoding='json.dumps ensure_ascii=False sort_keys=True indent=2 allow_nan=False plus newline UTF8')
    require(contract == expected and len(rows) == COUNT, 'Original scientific contract/catalog changed')
    require(disjoint(root, old['output']), 'New output overlaps original VAL batch')
    for row in rows:
        for mod in batch.MODS:
            p = Path(row[mod]['path']).resolve()
            require(p != root and root not in p.parents, 'New output contains native source')
    settings = dict(old, rows=rows, output=root/'generated/private_native', contract=contract, workers=WORKERS, batch_module=batchpath)
    # 保留原分辨率掩膜


    masks.delete_owned_native = retention_hook((root/'generated',root/'pilot'))
    return dict(root=root,oldroot=oldroot,batch=batch,masks=masks,settings=settings,catalog=catalog,request=request)


def configure(config):
    request = read(config)
    ctx = make_context(request)
    root = ctx['root']
    save(root/'request.json', request)
    save_bytes(root/'mask_contract.json', bound(request['original_mask_contract']).read_bytes())
    amendment = seal(dict(schema='three_region_mask_execution_amendment_v1',request=desc(root/'request.json'),
        wrapper=desc(__file__),original_mask_contract=request['original_mask_contract'],copied_mask_contract=desc(root/'mask_contract.json'),
        actual_workers=WORKERS,historical_contract_workers=1,cpu_affinity=CPUS,nice=19,ionice_class=3,no_gpu=True,
        original_mask_root=str(ctx['oldroot']),new_mask_root=str(root/'generated'),catalog=request['catalog'],
        sample_count=COUNT,sample_order_sha256=digest([r['sample_id'] for r in ctx['settings']['rows']]),
        scientific_contract_unchanged=True,company_unchanged=True,original_outputs_readonly=True,
        storage_policy='New owned native directories retained after sealed compact publication; loaded-module delete_owned_native replaced by named verified retention hook only. Original caches are read-only; missing old native masks are not reconstructed here.',
        legacy_compact_receipt_cleanup_note_overridden_by_this_storage_amendment=True,
        scientific_functions_replaced=False,cleanup_hook_replaced=True,
        execution_note='Historical contract workers=1 is preserved for compatible scientific receipts; actual scheduler uses four workers in new output only.'))
    save(root/'execution_amendment.json', amendment)
    return dict(status='configured',execution_amendment=desc(root/'execution_amendment.json'))


def context(root):
    root = Path(root).resolve()
    amendment = checked(read(root/'execution_amendment.json'))
    require(bound(amendment['wrapper']).resolve() == Path(__file__).resolve(), 'Wrong continuation wrapper')
    require(bound(amendment['request']) == root/'request.json', 'Wrong request binding')
    ctx = make_context(read(root/'request.json'))
    require(ctx['root'] == root and sha(root/'mask_contract.json') == CONTRACT_SHA and amendment['actual_workers'] == WORKERS, 'Execution freeze differs')
    bound(amendment['copied_mask_contract'])
    ctx['amendment'] = amendment
    return ctx


def verify_receipt(ctx,index,path):
    item = ctx['masks'].verify_compact(ctx['batch'],ctx['settings'],ctx['catalog'],index,Path(path))
    require(item['status'] in STATUSES, 'Unknown mask status')
    return item


def receipt_path(root,index):
    return Path(root)/'cases'/('case_%06d'%index)/'receipt.json'


def retention_hook(allowed_roots):
    allowed = {Path(p).resolve() for p in allowed_roots}
    def retain_owned_native(batch,root,index,owner,receipt_file):
        root = Path(root).resolve()
        require(root in allowed,'Retention restricted to this wrapper owned roots')
        parent = root/'private_native/cases'
        case = parent/('case_%06d'%index)
        require(case.exists() and not any(p.is_symlink() for p in (root,root/'private_native',parent,case)), 'Missing or symlinked retention target')
        require(case.resolve().parent == parent.resolve() and len(case.resolve().relative_to(root).parts) == 3, 'Unsafe retention target')
        require(read(case/'owner.json') == owner,'Native ownership differs')
        receipt_file = Path(receipt_file).resolve()
        require(receipt_file == receipt_path(root,index),'Wrong compact receipt retention target')
        item = checked(read(receipt_file))
        require(item['index'] == index and item['sample_id'] == owner['sample_id'] and item['contract_sha256'] == owner['contract_sha256'] and item['status'] in STATUSES,'Unbound retention receipt')
        for descriptor in item['evidence'].values():
            bound(descriptor)
        masks = None
        if item['status'] == 'complete':
            bound(item['npz']); bound(item['projection_audit'])
            native = checked(read(case/'case_receipt.json'))
            require(native['index'] == index and native['sample_id'] == item['sample_id'] and native['contract_sha256'] == item['contract_sha256'] and native['sources'] == item['sources'] and native['counts'] == item['counts_native'], 'Native receipt retention binding differs')
            masks = native['masks']
            require(set(masks) == set(DOMAINS),'Incomplete retained native masks')
            for d,descriptor in masks.items():
                require(bound(descriptor) == case/'mask'/(d+'.png') and descriptor['sha256'] == item['native_mask_file_sha256'][d], 'Retained PNG differs')
            require(sha(case/'mask/summary.json') == item['evidence']['mask/summary.json']['sha256'], 'Retained array evidence differs')
            arrays = native_arrays(item)
        else:
            arrays = None
            bound(item['failure_stderr'])
        result = seal(dict(schema='three_region_owned_native_retention_v1',index=index,sample_id=item['sample_id'],status=item['status'],
            compact_receipt=desc(receipt_file),owner=desc(case/'owner.json'),native_directory=str(case),native_masks=masks,
            native_array_sha256=arrays,counts_native=item['counts_native'],cleanup_replaced_by_retention=True,original_outputs_touched=False))
        save(receipt_file.parent/'retention.json',result)
        return result
    return retain_owned_native


def native_arrays(item):
    key = 'mask/summary.json'
    require(key in item['evidence'], 'Native array evidence absent')
    value = read(bound(item['evidence'][key]))
    require(set(value['array_sha256']) == set(DOMAINS) and value['counts'] == item['counts_native'], 'Native array/count evidence differs')
    return value['array_sha256']


def compare_pair(old,new):
    require(old['status'] == new['status'], 'Pilot status mismatch')
    for key in ('index','sample_id','subject_id','native_shape','sources','source_row_sha256','catalog_sha256','contract_sha256','counts_native','native_mask_file_sha256'):
        require(old[key] == new[key], 'Pilot '+key+' mismatch')
    result = dict(index=old['index'],status=old['status'],native_png_exact=None,native_array_exact=None,projection_arrays_exact=None)
    if old['status'] == 'complete':
        import numpy as np
        require(native_arrays(old) == native_arrays(new), 'Pilot native array mismatch')
        with np.load(bound(old['npz']),allow_pickle=False) as a, np.load(bound(new['npz']),allow_pickle=False) as b:
            require(set(a.files) == set(b.files) == set(DOMAINS), 'Pilot projection domains differ')
            for d in DOMAINS:
                require(a[d].dtype == b[d].dtype == np.float32 and a[d].shape == b[d].shape == (1024,1024) and np.array_equal(a[d],b[d]), 'Pilot projection array mismatch: '+d)
        result.update(native_png_exact=True,native_array_exact=True,projection_arrays_exact=True)
    return result


def bounded_parallel(indices,worker,consume,workers=WORKERS):
    """At most four original case generators. Unexpected exceptions are fatal."""
    iterator = iter(indices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        def fill():
            while len(active)<workers:
                try:
                    i = next(iterator)
                except StopIteration:
                    break
                active[pool.submit(worker,i)] = i
        fill()
        while active:
            ready,_ = wait(active,return_when=FIRST_COMPLETED)
            # 先核对已完成任务
            resolved = [(f, f.result()) for f in ready]
            for f,value in resolved:
                consume(active.pop(f),value)
            fill()


def enforce_resources():
    require(sys.platform.startswith('linux'), 'Execution requires original Linux runtime')
    require(sorted(os.sched_getaffinity(0)) == CPUS and os.getpriority(os.PRIO_PROCESS,0) == 19, 'Launch with taskset 68-71 nice19')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be disabled')
    done = subprocess.run(['ionice','-p',str(os.getpid())],capture_output=True,text=True,check=True)
    require(done.stdout.strip().startswith('idle'), 'Launch with ionice -c3')


def identity(pid):
    raw = Path('/proc/%d/stat'%pid).read_text()
    rest = raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,starttick=int(rest[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=rest[0])


def verify_handover(path,oldroot):
    value = read(path)
    require(value.get('schema') == 'three_region_mask_handover_v1' and Path(value['original_mask_root']).resolve() == oldroot, 'Wrong handover target')
    processes = value.get('old_processes',[])
    require({'supervisor','masks'}.issubset({p.get('role') for p in processes}), 'Require original supervisor and mask identities')
    for p in processes:
        require(type(p.get('pid')) is int and p['pid']>1 and type(p.get('starttick')) is int and isinstance(p.get('boot_id'),str), 'Incomplete process identity')
        try:
            current = identity(p['pid'])
        except (FileNotFoundError,ProcessLookupError):
            continue
        same = all(current[k] == p[k] for k in ('pid','starttick','boot_id'))
        require(not same or current['state'] in ('Z','X'), 'Original writer still active; no continuation launched')
    return value


def pilot(root):
    enforce_resources()
    ctx = context(root)
    batch, masks, root = ctx['batch'],ctx['masks'],ctx['root']
    destination = root/'pilot'
    settings = dict(ctx['settings'],output=destination/'private_native')
    with batch.batch_lock(destination):
        journal = batch.Journal(dict(settings,output=destination))
        comparisons = {}
        olditems = {i:verify_receipt(ctx,i,receipt_path(ctx['oldroot'],i)) for i in range(12)}
        def worker(i):
            masks.process_one(batch,settings,ctx['catalog'],i,journal,destination)
            return verify_receipt(ctx,i,receipt_path(destination,i))
        def consume(i,item):
            comparisons[i] = dict(compare_pair(olditems[i],item),old=desc(receipt_path(ctx['oldroot'],i)),new=desc(receipt_path(destination,i)))
        bounded_parallel(range(12),worker,consume)
        require(Counter(v['status'] for v in comparisons.values())['complete'] == 11 and len(comparisons) == 12, 'Expected original first12: eleven valid and one technical case')
        report = seal(dict(schema='three_region_mask_parallel_pilot_v1',status='passed',actual_workers=WORKERS,case_count=12,
            valid_cases=11,technical_cases=1,all_complete_native_png_array_projection_exact=True,
            execution_amendment=desc(root/'execution_amendment.json'),comparisons=[comparisons[i] for i in range(12)]))
        save(destination/'receipt.json',report)
        for i in range(12):
            publish_ref(ctx,i,receipt_path(ctx['oldroot'],i),'original_readonly',olditems[i],refroot=destination/'refs')
        return report


def verify_ref(ctx,index,path):
    ref = checked(read(path))
    require(ref.get('schema') == 'three_region_mask_receipt_ref_v1' and ref['index'] == index and ref['sample_id'] == ctx['settings']['rows'][index]['sample_id'], 'Reference index/source differs')
    require(ref['mask_contract'] == desc(ctx['root']/'mask_contract.json') and ref['execution_amendment'] == desc(ctx['root']/'execution_amendment.json'), 'Reference contract/amendment differs')
    require(ref['origin'] in ('original_readonly','continuation'), 'Unknown reference origin')
    expected = receipt_path(ctx['oldroot'] if ref['origin'] == 'original_readonly' else ctx['root']/'generated',index)
    require(bound(ref['receipt']).resolve() == expected.resolve(), 'Unexpected reference target')
    item = verify_receipt(ctx,index,expected)
    require(ref['status'] == item['status'], 'Reference status differs')
    if ref['origin'] == 'continuation':
        retention_file = expected.parent/'retention.json'
        require(bound(ref['native_retention']) == retention_file, 'Reference retention binding differs')
        retained = checked(read(retention_file))
        require(retained['compact_receipt'] == ref['receipt'], 'Retained native/compact binding differs')
        expected_native = ctx['root']/'generated/private_native/cases'/('case_%06d'%index)
        require(Path(retained['native_directory']) == expected_native and bound(retained['owner']) == expected_native/'owner.json', 'Retained native directory differs')
        require(set(retained['native_masks'] or {}) == (set(DOMAINS) if item['status'] == 'complete' else set()), 'Retained native inventory differs')
        for d,descriptor in (retained['native_masks'] or {}).items():
            require(bound(descriptor) == expected_native/'mask'/(d+'.png') and descriptor['sha256'] == item['native_mask_file_sha256'][d], 'Retained native reference differs')
        if item['status'] == 'complete':
            require(retained['native_array_sha256'] == native_arrays(item), 'Retained native array binding differs')
        require(retained['counts_native'] == item['counts_native'] and retained['status'] == item['status'], 'Retained native status/counts differ')
    else:
        require(ref.get('native_retention') is None, 'Old native storage must not be fabricated')
    return ref,item


def publish_ref(ctx,index,path,origin,item,refroot=None):
    retention = Path(path).parent/'retention.json'
    value = seal(dict(schema='three_region_mask_receipt_ref_v1',index=index,sample_id=item['sample_id'],status=item['status'],origin=origin,
        receipt=desc(path),mask_contract=desc(ctx['root']/'mask_contract.json'),execution_amendment=desc(ctx['root']/'execution_amendment.json'),
        native_retention=desc(retention) if origin == 'continuation' else None,
        native_availability='retained_new_owned' if origin == 'continuation' else 'not_guaranteed_in_original_compact_cache'))
    save((refroot or ctx['root']/'refs')/('%06d.json'%index),value)
    return value


def run(root,handover):
    enforce_resources()
    ctx = context(root)
    root,batch,masks = ctx['root'],ctx['batch'],ctx['masks']
    verify_handover(handover,ctx['oldroot'])
    pilot_report = checked(read(root/'pilot/receipt.json'))
    require(pilot_report.get('status') == 'passed' and pilot_report.get('case_count') == 12 and pilot_report.get('all_complete_native_png_array_projection_exact') is True and pilot_report['execution_amendment'] == desc(root/'execution_amendment.json'), 'Parallel pilot has not passed')
    for pair in pilot_report['comparisons']:
        bound(pair['old']); bound(pair['new'])
    with batch.batch_lock(root):
        save(root/'handover.json',dict(handover=desc(handover),execution_amendment=desc(root/'execution_amendment.json')))
        journal = batch.Journal(dict(ctx['settings'],output=root))
        refs, counts, origins = {},Counter(),Counter()
        state = dict(status='running',process=identity(os.getpid()),execution_amendment=desc(root/'execution_amendment.json'),sample_count=COUNT)
        def publish():
            state.update(published=len(refs),status_counts=dict(counts),origin_counts=dict(origins),updated_unix=time.time())
            save(root/'continuation_status.json',state,mutable=True)
        publish()
        def worker(i):
            refpath = root/'refs'/('%06d.json'%i)
            if refpath.exists():
                return verify_ref(ctx,i,refpath)
            oldpath = receipt_path(ctx['oldroot'],i)
            if oldpath.exists():
                item = verify_receipt(ctx,i,oldpath)
                masks.mark_published(journal,i,item['status'])
                journal.emit('mask_original_readonly_reused',index=i,sample_id=item['sample_id'],mask_status=item['status'])
                return publish_ref(ctx,i,oldpath,'original_readonly',item),item
            masks.process_one(batch,ctx['settings'],ctx['catalog'],i,journal,root/'generated')
            newpath = receipt_path(root/'generated',i)
            item = verify_receipt(ctx,i,newpath)
            return publish_ref(ctx,i,newpath,'continuation',item),item
        def consume(i,pair):
            ref,item = pair
            refs[i] = ref
            counts[item['status']] += 1
            origins[ref['origin']] += 1
            publish()
        try:
            journal.emit('masks_running',test_samples=COUNT,workers=WORKERS,execution_amendment=desc(root/'execution_amendment.json'))
            bounded_parallel(range(COUNT),worker,consume)
            require(set(refs) == set(range(COUNT)), 'Incomplete TEST coverage')
            manifest = seal(dict(schema='three_region_mask_receipt_refs_v1',status='complete',catalog=ctx['request']['catalog'],sample_count=COUNT,
                execution_amendment=desc(root/'execution_amendment.json'),mask_contract=desc(root/'mask_contract.json'),
                status_counts=dict(counts),origin_counts=dict(origins),minimum_pixel_count_filter_applied=False,replacement_count=0,
                refs=[desc(root/'refs'/('%06d.json'%i)) for i in range(COUNT)]))
            save(root/'receipt_refs.json',manifest)
            save(root/'receipt.json',seal(dict(status='complete',sample_count=COUNT,manifest=desc(root/'receipt_refs.json'),execution_amendment=desc(root/'execution_amendment.json'))))
            state['status'] = 'complete'; publish()
            journal.emit('complete',all_test_rows_accounted=True,status_counts=dict(counts))
            return dict(status='complete',sample_count=COUNT,receipt=desc(root/'receipt.json'))
        except BaseException as exc:
            state.update(status='failed',error_type=type(exc).__name__,error=str(exc)); publish()
            journal.emit('failed',error_type=type(exc).__name__,error=str(exc))
            raise


def status(root):
    root = Path(root).resolve()
    value = read(root/'continuation_status.json') if (root/'continuation_status.json').exists() else {'status':'not_started'}
    value['pilot_passed'] = (root/'pilot/receipt.json').exists()
    value['completion_exists'] = (root/'receipt.json').exists()
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('configure','pilot','run','status'))
    p.add_argument('--config',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--handover',type=Path)
    a = p.parse_args()
    if a.mode == 'configure':
        require(a.config is not None,'configure requires --config'); result = configure(a.config)
    else:
        require(a.root is not None,'require --root')
        if a.mode == 'run':
            require(a.handover is not None,'run requires --handover'); result = run(a.root,a.handover)
        else:
            result = globals()[a.mode](a.root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
