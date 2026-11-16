"""Authorized program-consistent FP32 TEST execution.

Original science and CUDA values are unchanged. Cross-backend numerical
differences are diagnostic only under the new explicit human authorization;
categorical, geometry, truth, binding, finite and domain failures remain fatal.
Historical failures remain failures under their original comparison rules.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
import backend_compare as backend
import regional_adapter as regional

BASE = Path('/study/siiba/experiments')
P = BASE/'three_region_test_parallel_20261025_v1'
R = BASE/'regional_signal_agreement_20261026_v1'
OLD = BASE/'three_region_test_eval_20261025_v1'
PARENT = BASE/'test_gpu_boundary_review_20261026_v1'
ROOT = BASE/'test_gpu_program_consistency_20261027_v1'
MODELS = ('unet_single','unet_multi','unetplusplus')
COUNT = 21899
COMBINED_SHA = '55de1279c7c3ec692467017a9fc27bf96db85ee39debfc6dffea7af9db1010c7'
REGIONAL_PROTOCOL_SHA = 'ff37b3c5cdc86af7e2f781107cb46c7724c04c50da66b2b26e44ed2ff92eeffa'
TOLERANCE_SHA = '5d2a593b078f6e14629b9a18b88573d345fb8efad5240580ceea508a1ec925f2'
PARENT_WORKER_SHA = '5e10a2e4a075d22a7cee909b2339ea6113a5d3c28c727e3e53ea0bc6d3895045'
PARENT_FAILURE_SHA = '317c7b68c21a0daaea2881e35391fed1ec50e561db68729a1bb341a9d86e224e'
SOURCE_NAMES = ('worker.py','backend_compare.py','regional_adapter.py','aggregate.py',
                'validate_independent.py','build_validation_gate.py')
POLICY = 'program_semantics_consistency_v1'
_parent = None

def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def desc(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def bound(value):
    path=Path(value['path'])
    need(path.is_absolute() and path.is_file() and sha(path)==value['sha256'],'Bound file changed: '+str(path))
    return path


def seal(value):
    return dict(value,record_sha256=digest(value))


def checked(value):
    need(value.get('record_sha256')==digest({k:v for k,v in value.items() if k!='record_sha256'}),'Record seal differs')
    return value


def save(path,value,mutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=encoded(value)
    if path.exists() and not mutable:
        need(path.read_bytes()==data,'Refuse overwrite: '+str(path));return
    temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        if mutable:
            os.replace(str(temporary),str(path))
        else:
            try:os.link(str(temporary),str(path))
            except FileExistsError:need(path.read_bytes()==data,'Concurrent different record')
    finally:
        if temporary.exists():temporary.unlink()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_context(gpu=None):
    """Validate the original CPU contract, then explicitly select CUDA if used."""
    visible='' if gpu is None else str(gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=visible,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    need(sha(P/'source/combined_eval.py')==COMBINED_SHA,'Original combined source differs')
    need(sha(R/'protocol.json')==REGIONAL_PROTOCOL_SHA,'Original regional protocol differs')
    rp=read(R/'protocol.json')
    for value in rp['sources'].values():bound(value)
    c=load(P/'source/combined_eval.py','_gpu_original_combined')
    # 导入后恢复GPU可见性

    sys.path.insert(0,str(R/'source'))
    try:r=load(R/'source/runner.py','_gpu_original_regional')
    finally:sys.path.pop(0)
    os.environ['CUDA_VISIBLE_DEVICES']=visible
    need(rp['sources']==r.source_files(),'Regional source inventory differs')
    need(rp['original_evaluation']==desc(P/'evaluation/protocol.json'),'Original regional/combined binding differs')
    ctx=c.context(P/'evaluation')
    need(len(ctx['rows'])==COUNT and tuple(c.MODELS)==MODELS,'Original TEST/models changed')
    return dict(combined=c,regional_runner=r,ctx=ctx,regional_protocol=rp)


def bundle_technical_statuses():
    # 沿用掩膜状态
    return ('no_unique_face','invalid_geometry')


@contextmanager
def shard_lock(root,shard):
    import fcntl
    path=Path(root)/'locks'/('shard_%d.lock'%shard);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def load_mask(bundle,index,wait=False):
    c,ctx=bundle['combined'],bundle['ctx'];row=ctx['rows'][index]
    path=P/'masks/refs'/('%06d.json'%index)
    while not path.exists():
        need(wait,'Missing sealed mask reference during verification')
        state=P/'masks/continuation_status.json'
        if state.exists():need(read(state).get('status') not in ('failed','complete'),'Mask producer failed or incomplete reference after completion')
        time.sleep(5)
    receipt_path=c.validate_ref(c.checked(read(path)),row,index,ctx['protocol'])
    receipt=ctx['old'].validate_mask_receipt(read(receipt_path),row,index,c.CATALOG_SHA,
        ctx['protocol']['mask_contract']['sha256'],ctx['catalog']['data_root'])
    if receipt['status'] in ctx['old'].TECHNICAL:ctx['old'].validate_technical_evidence(receipt,receipt_path)
    return path,receipt_path,receipt


def gpu_runtime(bundle,contract,gpu):
    need(os.environ['CUDA_VISIBLE_DEVICES']==contract['gpu_inventory'][gpu]['uuid'],'Explicit GPU visibility changed')
    actual=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    need(actual==contract['gpu_inventory'][gpu]['uuid'],'Physical GPU identity changed')
    runtime=bundle['ctx']['old'].runtime_info('cuda:0',2)
    need(runtime['precision']=='FP32_no_autocast_no_TF32' and runtime['deterministic_algorithms'] is True,'CUDA precision/runtime differs')
    return dict(runtime=runtime,physical_gpu_id=gpu,uuid=actual,cuda_visible_devices=actual,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])



def sources():
    here = Path(__file__).resolve().parent
    return {Path(name).stem:desc(here/name) for name in SOURCE_NAMES}


def parent():
    """Load and verify the original G2 verifier without altering its source."""
    global _parent
    if _parent is None:
        source = PARENT/'source/worker.py'
        need(sha(source) == PARENT_WORKER_SHA, 'Frozen G2 worker differs')
        sys.path.insert(0, str(source.parent))
        saved = {name:sys.modules.get(name) for name in ('backend_compare','regional_adapter','boundary_review')}
        try:
            for name in saved:
                sys.modules.pop(name, None)
            _parent = load(source, '_program_consistency_original_g2_worker')
            _parent.verify_contract(PARENT)
        finally:
            sys.path.pop(0)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
    return _parent


def _numeric_metric_path(path, kind):
    parts = path.split('/')
    if kind == 'table3':
        return (len(parts) == 3 and parts[0] == kind and parts[1] in ('P','D','O','F')
                and parts[2] in backend.THRESHOLDS[kind])
    if kind == 'table4':
        return (len(parts) == 3 and parts[0] == kind and parts[1] in ('P','D','O')
                and parts[2] in backend.THRESHOLDS[kind])
    return (kind == 'regional' and len(parts) == 5 and parts[:2] == ['regional','regions']
            and parts[2] in regional.REGIONS and parts[3] == 'prediction' and parts[4] in regional.SIGNALS)


def _metric_domain(path, value, kind):
    need(isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value),
         'Nonfinite or invalid metric: '+path)
    name = path.rsplit('/',1)[-1]
    if kind == 'regional':
        need(-1e-12 <= value <= 1+1e-12, 'Regional encoded intensity outside domain: '+path)
    elif name in ('l1','mse'):
        need(0 <= value <= 1, 'Encoded-image error outside domain: '+path)
    elif name in ('mean_sigma','mean_z2'):
        need(value >= 0, 'Nonnegative metric outside domain: '+path)
    elif name in ('rank_corr','ssim'):
        need(-1 <= value <= 1, 'Correlation/SSIM outside domain: '+path)
    elif name == 'coverage95':
        need(0 <= value <= 1, 'Coverage outside domain: '+path)
    elif name == 'psnr':
        need(value >= 0, 'PSNR outside encoded-image domain: '+path)


def compare_program_payloads(reference, candidate, kind):
    """Independently enforce semantics; retain the original numeric report."""
    need(kind in ('table3','table4','regional'), 'Unknown scientific payload kind')
    def walk(a,b,path):
        if isinstance(a,dict):
            need(isinstance(b,dict) and set(a)==set(b), 'Mapping structure differs: '+path)
            for key in sorted(a):walk(a[key],b[key],path+'/'+key)
        elif isinstance(a,(list,tuple)):
            need(type(a) is type(b) and len(a)==len(b), 'Sequence structure differs: '+path)
            for index,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(index))
        elif a is None or isinstance(a,(str,bool)):
            need(type(a) is type(b) and a==b, 'Categorical/null/reason differs: '+path)
        elif _numeric_metric_path(path,kind):
            _metric_domain(path,a,kind);_metric_domain(path,b,kind)
            need(type(a) is type(b), 'Metric numeric representation differs: '+path)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            need(type(a) is type(b) and math.isfinite(a) and math.isfinite(b) and a==b,
                 'Exact truth/area/count/metadata differs: '+path)
        else:
            raise ValueError('Unsupported scientific field: '+path)
    walk(reference,candidate,kind)
    result = backend.compare_payload(reference,candidate,kind)
    check_program_comparison({kind:result})
    return result


def check_program_comparison(comparison):
    """Reject semantic mismatches, never reject finite allowed metric deltas."""
    checked_kinds=[]; numeric_violations=0
    for kind in ('table3','table4','regional'):
        result=comparison.get(kind)
        if result is None:continue
        need(result.get('schema')=='cuda_fp32_comparison_v1' and result.get('kind')==kind,
             'Invalid original diagnostic comparison schema')
        need(result.get('status') in ('passed','failed') and isinstance(result.get('violations'),list),
             'Invalid diagnostic comparison status')
        need(result['violation_count']==len(result['violations']) and
             result['status']==('failed' if result['violations'] else 'passed'), 'Diagnostic count/status differs')
        for item in result['violations']:
            path=item.get('path','')
            need('reason' not in item and _numeric_metric_path(path,kind),
                 'Structural/exact-field backend discrepancy: '+path)
            _metric_domain(path,item.get('reference'),kind);_metric_domain(path,item.get('candidate'),kind)
            need(math.isfinite(item.get('absolute_difference',float('nan'))) and
                 item['absolute_difference']==abs(item['candidate']-item['reference']),
                 'Invalid numerical diagnostic difference: '+path)
            numeric_violations+=1
        checked_kinds.append(kind)
    return dict(status='passed',policy=POLICY,compared_kinds=checked_kinds,
                original_numeric_violation_count=numeric_violations,numeric_differences_are_diagnostic_only=True)


def check_program_tensors(reference,candidate):
    """Shape/dtype/finite/domain are mandatory; tensor deltas are reported."""
    import numpy as np
    need(set(reference)==set(candidate)=={'prediction','sigma'}, 'Tensor field structure differs')
    for tensors in (reference,candidate):
        p,s=np.asarray(tensors['prediction']),np.asarray(tensors['sigma'])
        need(p.dtype==s.dtype==np.float32 and p.ndim==3 and p.shape[0]==3 and s.shape==p.shape[1:],
             'FP32 prediction/sigma shape or dtype differs')
        need(np.isfinite(p).all() and np.isfinite(s).all() and np.all((p>=0)&(p<=1)) and np.all(s>0),
             'Prediction/sigma finite or domain failure')
    return backend.compare_tensors(reference,candidate)


def comparison_status(comparison):
    statuses=[comparison[k]['status'] for k in ('table3','table4','regional') if comparison.get(k) is not None]
    return 'unavailable' if not statuses else ('failed' if 'failed' in statuses else 'passed')


def configure(root,authorization,tolerances,gpu_inventory):
    root=Path(root).resolve()
    need(root==ROOT and not (root/'contract.json').exists(),'Dedicated new program-consistency root required')
    need(sha(tolerances)==TOLERANCE_SHA,'Historical diagnostic thresholds changed')
    backend.normalized_thresholds(read(tolerances))
    inventory=read(gpu_inventory)
    if isinstance(inventory,dict):inventory=inventory['gpu_inventory']
    inventory=[dict(index=v['index'],uuid=v['uuid']) for v in inventory]
    need([v['index'] for v in inventory]==[0,1,2,3] and len({v['uuid'] for v in inventory})==4,
         'Require four explicitly bound unique GPUs')
    bundle=load_context();ctx=bundle['ctx'];original=parent().verify_contract(PARENT,bundle)
    plan=checked(read(root/'validation_plan.json'))
    need(plan.get('expected_comparisons')==60 and len(plan['selected_groups'])==17,
         'Prospective program validation plan scope differs')
    failure=PARENT/'failures/shard_0_index_003804_unetplusplus.json'
    need(sha(failure)==PARENT_FAILURE_SHA,'Original rank failure evidence differs')
    value=seal(dict(schema='test_gpu_program_contract_v1',sample_count=COUNT,models=list(MODELS),
        source=sources(),authorization=desc(authorization),thresholds=desc(tolerances),
        validation_plan=desc(root/'validation_plan.json'),parent_contract=desc(PARENT/'contract.json'),
        parent_failure=desc(failure),original_combined_protocol=desc(P/'evaluation/protocol.json'),
        original_regional_protocol=desc(R/'protocol.json'),catalog=ctx['protocol']['catalog'],
        original_stats=desc(OLD/'source/tools/uv_three_region_stats_20261025.py'),
        sample_order_sha256=bundle['combined'].compact_order([r['sample_id'] for r in ctx['rows']]),model_pins=ctx['old'].MODEL_PINS,
        gpu_inventory=inventory,gpu_inventory_source=desc(gpu_inventory),
        execution=dict(precision='FP32_no_autocast_no_TF32',batch_size=1,shard_count=4,shard_rule='catalog_index % 4',
            cpu_threads=2,cpu_affinity={str(i):[80+2*i,81+2*i] for i in range(4)},nice=19,ionice_class=3),
        acceptance_policy=POLICY,
        human_policy_change='User explicitly authorized identical programs/semantics with CPU/CUDA numerical deltas diagnostic only. This supersedes numeric acceptance gates; it does not relabel any historical failure as passed.',
        numerical_comparison='Original scalar/tensor comparisons and actual CPU/CUDA numbers retained, including failed status. No numerical threshold governs acceptance.',
        mandatory='Same frozen programs, models, sample/mask/truth/geometry, scientific definitions, FP32 runtime and deterministic bindings. Structures, counts, exact truth/areas, null/reasons, finite values and domains remain mandatory.',
        parent_reuse='Only original G2 accepted records verified by its frozen original verifier. Unchanged CUDA payloads and original hardware provenance; new_forward_count=0. Failed records are recomputed, never imported.',
        primary_results='CUDA outputs only; no CPU replacement, rounding, snapping, selective deletion or scientific parameter changes.',
        same_forward='One unchanged CUDA forward supplies reconstruction, calibration and anatomical intensity.',
        failure_policy='Stop on semantic, binding, structural, truth, finite/domain or runtime failures. Finite cross-backend numeric differences are diagnostic only.'))
    save(root/'contract.json',value)
    return value


def verify_contract(root,bundle=None):
    root=Path(root);contract=checked(read(root/'contract.json'))
    need(contract['schema']=='test_gpu_program_contract_v1' and contract['source']==sources() and
         contract['sample_count']==COUNT and contract['models']==list(MODELS) and
         contract['acceptance_policy']==POLICY,'Program contract/source differs')
    need(contract['thresholds']['sha256']==TOLERANCE_SHA,'Diagnostic thresholds SHA differs')
    for key in ('authorization','thresholds','validation_plan','parent_contract','parent_failure',
                'original_combined_protocol','original_regional_protocol','catalog','original_stats','gpu_inventory_source'):
        bound(contract[key])
    need(contract['validation_plan']==desc(root/'validation_plan.json') and
         contract['parent_contract']==desc(PARENT/'contract.json') and
         contract['parent_failure']['sha256']==PARENT_FAILURE_SHA,'Prospective or historical binding changed')
    backend.normalized_thresholds(read(bound(contract['thresholds'])))
    need(contract['original_combined_protocol']==desc(P/'evaluation/protocol.json') and
         contract['original_regional_protocol']==desc(R/'protocol.json'),'Scientific contracts changed')
    parent().verify_contract(PARENT,bundle)
    if bundle:
        ctx=bundle['ctx']
        need(contract['catalog']==ctx['protocol']['catalog'] and contract['model_pins']==ctx['old'].MODEL_PINS and
             contract['sample_order_sha256']==bundle['combined'].compact_order([r['sample_id'] for r in ctx['rows']]),
             'TEST/model assets differ')
    return contract


def verify_gate(root,contract):
    root=Path(root);gate=checked(read(root/'pilot_acceptance.json'))
    need(gate['schema']=='test_gpu_program_validation_gate_v1' and gate['status']=='passed' and
         gate['contract']==desc(root/'contract.json') and gate['validation_plan']==contract['validation_plan'],
         'Program validation gate missing or unbound')
    plan=checked(read(bound(gate['validation_plan'])))
    checked(read(bound(gate['validation_receipt'])))
    required={(g,i,m) for g in range(4) for i in plan['by_gpu'][str(g)] for m in MODELS}
    need(len(required)==60 and plan['expected_comparisons']==60 and len(plan['selected_groups'])==17,
         'Independent program validation scope differs')
    seen=set();shared={}
    for descriptor in gate['records']:
        raw=checked(read(bound(descriptor)));key=(raw['gpu'],raw['index'],raw['model_id'])
        need(key in required and key not in seen and raw['schema']=='test_gpu_program_validation_case_v1' and
             raw['contract']==desc(root/'contract.json') and raw['plan']==contract['validation_plan'],
             'Program validation record scope/binding differs')
        seen.add(key)
        for name in ('source','cpu_reference','mask_ref','mask_receipt'):bound(raw[name])
        need(raw['source']==contract['source']['validate_independent'] and raw['cpu_replay_exact'] is True and
             raw['gpu_repeat_exact'] is True and raw['structural_program_checks']['status']=='passed',
             'Program/replay/repeat verification did not pass')
        prior=checked(read(bound(raw['cpu_reference'])))
        need(prior['index']==raw['index'] and prior['model_id']==raw['model_id'] and
             raw['actual_cpu']['combined']==dict(regions=prior['regions'],table4=prior['table4']),
             'Validation CPU replay does not exactly equal archived values')
        comparisons={}
        for name,field in (('table3','regions'),('table4','table4')):
            comparisons[name]=compare_program_payloads(raw['actual_cpu']['combined'][field],raw['actual_gpu']['combined'][field],name)
            need(raw[name]==comparisons[name],'Validation numeric diagnostics changed')
        # 记录历史前向比对
        keys=('status','reason','regions','roi_audit')
        comparisons['regional']=compare_program_payloads({k:raw['actual_cpu']['regional'][k] for k in keys},
                                                         {k:raw['actual_gpu']['regional'][k] for k in keys},'regional')
        need(raw['regional']==comparisons['regional'],'Regional diagnostic changed')
        check_program_comparison(comparisons)
        need(raw['tensor']['schema']=='cuda_fp32_tensor_comparison_v1' and raw['tensor']['status'] in ('passed','failed'),
             'Tensor diagnostic missing')
        if all(raw['index'] in plan['by_gpu'][str(g)] for g in range(4)):
            value=(raw['actual_gpu'],raw['arrays'])
            old=shared.setdefault(raw['model_id'],value)
            # 跨卡数值仅供诊断
            # 严格核对输入与几何
            for name,field in (('table3','regions'),('table4','table4')):
                compare_program_payloads(old[0]['combined'][field],value[0]['combined'][field],name)
            compare_program_payloads({k:old[0]['regional'][k] for k in keys},
                                     {k:value[0]['regional'][k] for k in keys},'regional')
            need(old[1]['truth']==value[1]['truth'],'Shared-case truth input differs across devices')
            for name in ('gpu_prediction','gpu_sigma'):
                need(all(old[1][name][field]==value[1][name][field] for field in ('shape','dtype')),
                     'Shared-case output tensor semantics differ')
    need(seen==required and set(shared)==set(MODELS),'Incomplete four-GPU program validation')
    return gate


def compare_cpu(bundle,contract,index,model,ref_path,receipt_path,receipt,combined,regional_fields):
    c,r,ctx=bundle['combined'],bundle['regional_runner'],bundle['ctx'];row=ctx['rows'][index]
    result={'combined_record':None,'regional_record':None,'table3':None,'table4':None,'regional':None}
    previous=P/'evaluation/records'/model/('%06d.json'%index)
    if previous.exists():
        old=c.verify_record(read(previous),ctx,row,index,model,sha(receipt_path))
        result['combined_record']=desc(previous)
        result['table3']=compare_program_payloads(old['regions'],combined['regions'],'table3')
        result['table4']=compare_program_payloads(old['table4'],combined['table4'],'table4')
    prior=R/'records'/model/('%06d.json'%index)
    if prior.exists():
        need(previous.exists(),'Regional CPU record lacks combined evidence')
        fields=regional.verify_old_record(r,c,ctx,index,model,prior,previous,ref_path,receipt_path,contract['original_regional_protocol']['sha256'])
        result['regional_record']=desc(prior)
        keys=('status','reason','regions','roi_audit')
        result['regional']=compare_program_payloads({k:fields[k] for k in keys},{k:regional_fields[k] for k in keys},'regional')
    check_program_comparison(result)
    return result


def prior_failed_record(index,model):
    if (index,model)!=(3804,'unetplusplus'):return None
    path=PARENT/'failures/shard_0_index_003804_unetplusplus.json'
    need(sha(path)==PARENT_FAILURE_SHA,'Original failed record differs')
    return desc(path)


def _verify_continuation(value,original,forwarded):
    continuation=value['continuation']
    need(continuation['cpu_numeric_gate']=='diagnostic_only' and
         continuation['original_cpu_comparison_status']==comparison_status(value['execution']['cpu_comparison']),
         'Numeric diagnostics relabelled or made an acceptance gate')
    if original is not None:
        need(continuation['status']=='reused_accepted_g2' and continuation['new_forward_count']==0 and
             continuation['historical_parent_continuation']==original['continuation'], 'G2 reuse declaration differs')
        for key in original:
            if key not in ('record_sha256','contract_sha256','continuation'):
                need(value.get(key)==original[key],'Reused G2 payload/provenance changed: '+key)
        need(continuation['original_failed_record'] is None,'Failed original imported as accepted')
    else:
        need(continuation['status']=='computed_program_v1' and continuation['parent_record'] is None and
             continuation['historical_parent_continuation'] is None and continuation['new_forward_count']==int(forwarded) and
             continuation['original_failed_record']==prior_failed_record(value['index'],value['model_id']),
             'Fresh program execution declaration differs')


def verify_record(root,bundle,contract,model,index,record=None):
    c,ctx=bundle['combined'],bundle['ctx'];row=ctx['rows'][index]
    value=checked(read(Path(root)/'records'/model/('%06d.json'%index)) if record is None else record)
    need(value['schema']=='test_gpu_sample_v1' and value['index']==index and value['model_id']==model and
         value['sample_id']==row['sample_id'] and value['subject_id']==row['subject_id'] and
         value['source_row_sha256']==digest(row) and value['contract_sha256']==sha(Path(root)/'contract.json') and
         value['checkpoint_sha256']==ctx['old'].MODEL_PINS[model]['sha'],'Unified record identity differs')
    ref_path,receipt_path,receipt=load_mask(bundle,index)
    need(value['mask_ref']==desc(ref_path) and value['mask_receipt']==desc(receipt_path),'Unified mask source differs')
    forwarded=receipt['status']=='complete'
    need(value['status']==('complete' if forwarded else 'unavailable_geometry') and
         value['execution']['forward_count']==int(forwarded) and value['execution']['physical_gpu_id']==index%4 and
         value['execution']['device']=='cuda:0','Backend/forward identity differs')
    continuation=value['continuation'];original=None
    if continuation['status']=='reused_accepted_g2':
        path=bound(continuation['parent_record'])
        need(path==PARENT/'records'/model/('%06d.json'%index),'Only original accepted G2 records can be imported')
        original=parent().verify_record(PARENT,bundle,parent().verify_contract(PARENT),model,index)
    else:
        runtime=read(bound(value['execution']['runtime']))
        need(runtime['runtime']==dict(ctx['protocol']['runtime'],device='cuda:0') and
             runtime['physical_gpu_id']==index%4 and runtime['uuid']==contract['gpu_inventory'][index%4]['uuid'] and
             runtime['cuda_visible_devices']==runtime['uuid'],'Current CUDA runtime/physical device differs')
    _verify_continuation(value,original,forwarded)
    need(set(value['combined'])=={'regions','table4'},'Combined fields differ')
    projection=seal(dict(index=index,sample_id=row['sample_id'],subject_id=row['subject_id'],model_id=model,
        checkpoint_sha256=value['checkpoint_sha256'],contract_sha256=value['contract_sha256'],source_row_sha256=value['source_row_sha256'],
        mask_receipt_sha256=sha(receipt_path),status=value['status'],regions=value['combined']['regions']))
    ctx['old'].verify_record(projection,row,index,model,value['contract_sha256'],sha(receipt_path))
    c.validate_table4(value['combined']['table4']);regional.validate_fields(value['regional'])
    for kind,field in (('table3','regions'),('table4','table4')):
        compare_program_payloads(value['combined'][field],value['combined'][field],kind)
    need(value['regional']['status']==value['status'],'Regional status differs')
    origin=dict(kind='gpu_new' if forwarded else 'technical_derived',record=None)
    need(value['provenance']==dict(combined=origin,regional=origin),'Mixed CPU/GPU primary values prohibited')
    if not forwarded:
        reason='mask_'+receipt['status']
        need(value['combined']==dict(regions=ctx['old'].unavailable_regions(reason),table4=c.unavailable_table4(reason)) and
             value['regional']==regional.unavailable(reason),'Technical fields changed')
    comparison=compare_cpu(bundle,contract,index,model,ref_path,receipt_path,receipt,value['combined'],value['regional'])
    need(comparison==value['execution']['cpu_comparison'],'CPU comparison/provenance changed after publication')
    check_program_comparison(comparison)
    return value


def read_unified(root,bundle,contract,model,index):
    return verify_record(root,bundle,contract,model,index)


def import_parent(root,bundle,contract,model,index):
    path=PARENT/'records'/model/('%06d.json'%index)
    if not path.exists():return None
    original=parent().verify_record(PARENT,bundle,parent().verify_contract(PARENT),model,index)
    value={k:v for k,v in original.items() if k!='record_sha256'}
    value['contract_sha256']=sha(Path(root)/'contract.json')
    value['continuation']=dict(status='reused_accepted_g2',parent_record=desc(path),new_forward_count=0,
        cpu_numeric_gate='diagnostic_only',original_cpu_comparison_status=comparison_status(original['execution']['cpu_comparison']),
        historical_parent_continuation=original['continuation'],original_failed_record=None)
    return seal(value)


def run(root,gpu,shard,max_groups=None):
    root=Path(root).resolve();need(root==ROOT and gpu==shard and shard in range(4),'Fixed four-GPU shard required')
    os.sched_setaffinity(0,{80+2*shard,81+2*shard});os.setpriority(os.PRIO_PROCESS,0,19)
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True,capture_output=True)
    contract=verify_contract(root)
    bundle=load_context(contract['gpu_inventory'][gpu]['uuid']);c,r,ctx=bundle['combined'],bundle['regional_runner'],bundle['ctx']
    contract=verify_contract(root,bundle);verify_gate(root,contract)
    smoke=max_groups is not None
    if smoke:
        need(max_groups==3 and not (root/'launch_reservation.json').exists(),'Smoke requires exactly3 groups before production reservation')
    else:
        launch=checked(read(root/'launch_receipt.json'));configuration=checked(read(bound(launch['configuration'])))
        request=read(bound(configuration['request']))
        need(launch['configuration']==desc(root/'configuration.json') and launch['pilot_gate']==desc(root/'pilot_acceptance.json') and
             request['execution_contract']==desc(root/'contract.json') and request['worker_source']==desc(__file__),
             'Bound program-consistent supervisor launch and worker source required')
    runtime=gpu_runtime(bundle,contract,gpu)
    need(runtime['runtime']==dict(ctx['protocol']['runtime'],device='cuda:0'),'Original software/runtime changed')
    save(root/'runtimes'/('gpu_%d.json'%gpu),runtime);runtime_desc=desc(root/'runtimes'/('gpu_%d.json'%gpu))
    completed=0;models={};indices=list(range(shard,COUNT,4))
    if smoke:indices=indices[:max_groups]
    forward_counts=dict(new_cuda=0,reused_records=0,technical_records=0)
    def publish(status,**fields):
        save(root/('smoke_progress' if smoke else 'progress')/('shard_%d.json'%shard),dict(status=status,shard=shard,gpu=gpu,pid=os.getpid(),
            completed_groups=completed,expected_groups=len(indices),time=time.time(),forward_counts=forward_counts,**fields),mutable=True)
    candidate_evidence=None
    with shard_lock(root,shard):
        try:
            for index in indices:
                candidate_evidence=None
                publish('waiting_or_verifying',active_index=index)
                row=ctx['rows'][index];ref_path,receipt_path,receipt=load_mask(bundle,index,wait=True)
                existing={m:(root/'records'/m/('%06d.json'%index)).exists() for m in MODELS}
                for model in MODELS:
                    if not existing[model]:
                        inherited=import_parent(root,bundle,contract,model,index)
                        if inherited is not None:
                            verify_record(root,bundle,contract,model,index,inherited)
                            save(root/'records'/model/('%06d.json'%index),inherited)
                            existing[model]=True;forward_counts['reused_records']+=1
                if all(existing.values()):
                    for model in MODELS:verify_record(root,bundle,contract,model,index)
                    completed+=1;publish('cache_verified',active_index=index);continue
                technical=receipt['status'] in ctx['old'].TECHNICAL
                if not technical:
                    weights,rois,audit=regional.prepare_geometry(r,c,ctx,row,receipt_path,receipt)
                    rgb=ctx['old'].prepare_image(row['rgb'],row['native_shape'],ctx['catalog']['data_root'])
                    truth=ctx['old'].prepare_image(row['uv'],row['native_shape'],ctx['catalog']['data_root'])
                for model in MODELS:
                    if existing[model]:verify_record(root,bundle,contract,model,index);continue
                    publish('evaluating',active_index=index,model=model)
                    if technical:
                        reason='mask_'+receipt['status'];status='unavailable_geometry'
                        combined=dict(regions=ctx['old'].unavailable_regions(reason),table4=c.unavailable_table4(reason))
                        regional_fields=regional.unavailable(reason);forward_counts['technical_records']+=1
                    else:
                        if model not in models:
                            models[model]=ctx['old'].load_model(model,ctx['config']['models'][model],Path(ctx['protocol']['eval_config']['path']).parent,ctx['frozen'],'cuda:0')
                        prediction,sigma=ctx['old'].infer(models[model],rgb,model,'cuda:0')
                        check_program_tensors(dict(prediction=prediction,sigma=sigma),dict(prediction=prediction,sigma=sigma))
                        combined=dict(regions=ctx['old'].measure(truth,prediction,sigma,weights,ctx['numeric']),
                            table4=c.table4_measure(truth,prediction,sigma,weights,ctx['old'],ctx['numeric'],ctx['rank']))
                        regional_fields=regional.measure_same_forward(r,truth,prediction,rois,audit)
                        status='complete';forward_counts['new_cuda']+=1
                    candidate_evidence=dict(index=index,model_id=model,sample_id=row['sample_id'],
                        contract=desc(root/'contract.json'),mask_ref=desc(ref_path),mask_receipt=desc(receipt_path),
                        current_runtime=runtime_desc,status=status,combined=combined,regional=regional_fields)
                    comparison=compare_cpu(bundle,contract,index,model,ref_path,receipt_path,receipt,combined,regional_fields)
                    origin=dict(kind='technical_derived' if technical else 'gpu_new',record=None)
                    value=seal(dict(schema='test_gpu_sample_v1',index=index,sample_id=row['sample_id'],subject_id=row['subject_id'],
                        model_id=model,checkpoint_sha256=ctx['old'].MODEL_PINS[model]['sha'],contract_sha256=sha(root/'contract.json'),
                        source_row_sha256=digest(row),mask_ref=desc(ref_path),mask_receipt=desc(receipt_path),status=status,
                        combined=combined,regional=regional_fields,provenance=dict(combined=origin,regional=origin),
                        execution=dict(forward_count=0 if technical else 1,physical_gpu_id=gpu,device='cuda:0',runtime=runtime_desc,cpu_comparison=comparison),
                        continuation=dict(status='computed_program_v1',parent_record=None,new_forward_count=0 if technical else 1,
                            cpu_numeric_gate='diagnostic_only',original_cpu_comparison_status=comparison_status(comparison),
                            historical_parent_continuation=None,original_failed_record=prior_failed_record(index,model))))
                    verify_record(root,bundle,contract,model,index,value)
                    save(root/'records'/model/('%06d.json'%index),value)
                    candidate_evidence=None
                completed+=1;publish('running',active_index=index)
                print(json.dumps(dict(event='gpu_group_complete',index=index,shard=shard,completed_groups=completed)),flush=True)
            if smoke:
                save(root/'smoke'/('shard_%d.json'%shard),seal(dict(schema='test_gpu_worker_smoke_v1',status='smoke_passed',shard=shard,gpu=gpu,
                    completed_groups=completed,indices=indices,models=list(MODELS),contract=desc(root/'contract.json'),
                    current_runtime=runtime_desc,forward_counts=forward_counts,gate=desc(root/'pilot_acceptance.json'),
                    records=[desc(root/'records'/model/('%06d.json'%index)) for index in indices for model in MODELS])))
                publish('smoke_complete')
            else:
                save(root/'completion'/('shard_%d.json'%shard),seal(dict(status='complete',shard=shard,gpu=gpu,
                    completed_groups=completed,indices=indices,models=list(MODELS),contract=desc(root/'contract.json'),
                    current_runtime=runtime_desc,forward_counts=forward_counts)))
                publish('complete')
        except BaseException as error:
            if candidate_evidence is not None:
                save(root/'failures'/('shard_%d_index_%06d_%s.json'%(shard,candidate_evidence['index'],candidate_evidence['model_id'])),
                     seal(dict(schema='test_gpu_program_semantic_failure_v1',candidate=candidate_evidence,
                               error_type=type(error).__name__,error=str(error),accepted=False)))
            publish('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('configure','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--authorization',type=Path);parser.add_argument('--tolerances',type=Path);parser.add_argument('--gpu-inventory',type=Path)
    parser.add_argument('--gpu',type=int);parser.add_argument('--shard',type=int);parser.add_argument('--max-groups',type=int)
    args=parser.parse_args()
    if args.action=='configure':
        need(all((args.authorization,args.tolerances,args.gpu_inventory)),'Configure requires authorization, diagnostic thresholds and inventory')
        print(json.dumps(configure(args.root,args.authorization,args.tolerances,args.gpu_inventory)),flush=True)
    else:run(args.root,args.gpu,args.shard,args.max_groups)


if __name__=='__main__':main()
