"""Independent audit of structural consistency; numeric parity is diagnostic.

This audit does not invoke worker comparison/acceptance helper implementations.
Full tensor comparison summaries remain hash-bound remote harness evidence.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path('/study/siiba/experiments/test_gpu_program_consistency_20261027_v1')
REGIONS=('forehead','image_left_cheek','image_right_cheek','nose','chin','F_reference')
SIGNALS=('R','G','B','RGB_mean_secondary')

def need(v,s):
    if not v:raise ValueError(s)

def variable(path):
    p=path.split('/')
    return (len(p)==3 and ((p[0]=='table3' and p[1] in ('P','D','O','F') and p[2] in ('l1','mse','psnr','ssim','mean_sigma')) or
            (p[0]=='table4' and p[1] in ('P','D','O') and p[2] in ('gaussian_nll','rank_corr','mean_z2','coverage95')))) or (
            len(p)==5 and p[0:2]==['regional','regions'] and p[2] in REGIONS and p[3]=='prediction' and p[4] in SIGNALS)

def independent_walk(a,b,path):
    """Different finite metric values allowed; all other quantities exactly match."""
    if isinstance(a,dict):
        need(type(b)==dict and set(a)==set(b),'Mapping differs '+path)
        for k in sorted(a):independent_walk(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        need(type(b)==list and len(a)==len(b),'Sequence differs '+path)
        for i,(x,y) in enumerate(zip(a,b)):independent_walk(x,y,path+'/'+str(i))
    elif isinstance(a,float):
        need(type(b)==float and math.isfinite(a) and math.isfinite(b),'Nonfinite/type differs '+path)
        if not variable(path):need(a==b,'Exact source quantity differs '+path)
        else:
            key=path.split('/')[-1]
            for value in (a,b):
                if key in ('l1','mse','mean_sigma','mean_z2'):need(value>=0,'Negative '+path)
                if key in ('l1','mse'):need(value<=1,'Encoded image error domain '+path)
                if key=='psnr':need(value>=0,'PSNR domain '+path)
                if key=='coverage95':need(0<=value<=1,'Coverage domain '+path)
                if key in ('rank_corr','ssim'):need(-1<=value<=1,'Correlation domain '+path)
                if path.startswith('regional/'):need(-1e-12<=value<=1+1e-12,'Intensity domain '+path)
    else:need(type(a)==type(b) and a==b,'Exact value/type differs '+path)

def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ah(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def expected(plan):
    need(plan['schema']=='test_gpu_program_validation_plan_v1' and plan['expected_comparisons']==60 and plan['expected_unique_groups']==17,'Plan scope')
    indices=set();shared=None
    for g in range(4):
        v=plan['by_gpu'][str(g)]
        need(len(v)==len(set(v))==5 and plan['shared_index'] in v,'Plan duplicates/shared')
        need(all(i==plan['shared_index'] or i%4==g for i in v),'Plan shard assignment')
        indices.update(v);shared=set(v) if shared is None else shared.intersection(v)
    need(len(indices)==17 and shared=={plan['shared_index']} and set(plan['selected_groups'])==set(map(str,indices)) and not indices.intersection(plan['excluded_indices']),'Heldout group membership')
    return {(g,i,m) for g in range(4) for i in plan['by_gpu'][str(g)] for m in plan['models']}

def main():
    raw=json.loads((ROOT/'contract.json').read_text())
    source=ROOT/'source/worker.py'
    need(raw['source']['worker']['sha256']==file_sha(source),'Worker source differs before import')
    sys.path.insert(0,str(source.parent));import worker as w
    contract=w.verify_contract(ROOT);plan=w.checked(w.read(w.bound(contract['validation_plan'])))
    required=expected(plan)
    need(plan['sample_order_sha256']==contract['sample_order_sha256'],'Sample order differs')
    bundle=w.load_context();w.verify_contract(ROOT,bundle)
    c,r,ctx=bundle['combined'],bundle['regional_runner'],bundle['ctx']
    refs=[];shards=[];seen=set();shared={};shared_diagnostics=[];geometry={};num_failed={k:0 for k in ('table3','table4','regional','tensor')};maxima={k:0.0 for k in num_failed}
    for g in range(4):
        sp=ROOT/'validation'/('gpu%d_receipt.json'%g);sr=w.checked(w.read(sp));shards.append(w.desc(sp))
        need(sr['schema']=='test_gpu_program_validation_shard_v1' and sr['status']=='passed' and sr['gpu']==g and sr['indices']==plan['by_gpu'][str(g)] and sr['models']==list(w.MODELS) and sr['count']==15 and len(sr['records'])==15,'Shard receipt scope')
        need(sr['contract']==w.desc(ROOT/'contract.json') and sr['plan']==contract['validation_plan'] and sr['source']==w.desc(ROOT/'source/validate_independent.py'),'Shard binding')
        for rd in sr['records']:
            v=w.checked(w.read(w.bound(rd)));key=(g,v['index'],v['model_id']);i=key[1];m=key[2]
            need(key in required and key not in seen and v['gpu']==g,'Case membership/duplicate');seen.add(key)
            need(v['schema']=='test_gpu_program_validation_case_v1' and v['status']=='passed' and v['contract']==w.desc(ROOT/'contract.json') and v['plan']==contract['validation_plan'] and v['source']==sr['source'],'Case binding')
            need(v['cpu_replay_exact'] and v['gpu_repeat_exact'] and v['same_scientific_functions'] and v['inputs_unchanged'] and v['numeric_differences_are_diagnostic'] and v['structural_program_checks']['status']=='passed','Program consistency evidence')
            need(v['model_state_before']==v['model_state_after'] and len(v['model_state_before'])==64,'Model state changed')
            runtime=v['runtime'];need(runtime['physical_gpu_id']==g and runtime['uuid']==contract['gpu_inventory'][g]['uuid'] and runtime['cuda_visible_devices']==runtime['uuid'] and runtime['cublas_workspace_config']==':4096:8','GPU UUID binding')
            need(runtime['runtime']==dict(ctx['protocol']['runtime'],device='cuda:0'),'Runtime differs')
            selected=plan['selected_groups'][str(i)]
            need(v['mask_ref']==selected['mask_ref'] and v['mask_receipt']==selected['mask_receipt'] and v['cpu_reference'] in selected['cpu_references'],'Input selection differs')
            for k in ('mask_ref','mask_receipt','cpu_reference'):w.bound(v[k])
            row=ctx['rows'][i];prior=c.verify_record(w.read(w.bound(v['cpu_reference'])),ctx,row,i,m,v['mask_receipt']['sha256'])
            need(v['actual_cpu']['combined']==dict(regions=prior['regions'],table4=prior['table4']),'CPU full scalar replay differs')
            for kind, a,b in [('table3',v['actual_cpu']['combined']['regions'],v['actual_gpu']['combined']['regions']),('table4',v['actual_cpu']['combined']['table4'],v['actual_gpu']['combined']['table4']),('regional',v['actual_cpu']['regional'],v['actual_gpu']['regional'])]:
                independent_walk(a,b,kind)
                # Recompute diagnostic comparisons, never require their status passed.
                diag=w.backend.compare_payload(a,b,kind);need(diag==v[kind],'Diagnostic comparison altered')
                num_failed[kind]+=int(diag['status']=='failed');maxima[kind]=max(maxima[kind],diag['maximum_absolute_difference'])
            arrays=v['arrays'];tensor=v['tensor']
            need(tensor['schema']=='cuda_fp32_tensor_comparison_v1' and set(tensor['fields'])=={'prediction','sigma'},'Tensor evidence schema')
            for name in ('prediction','sigma'):
                a,b=arrays['cpu_'+name],arrays['gpu_'+name];td=tensor['fields'][name]
                need(a['shape']==b['shape']==td['shape'] and a['dtype']==b['dtype']==td['dtype']=='float32','Tensor geometry/type')
                need(math.isfinite(td['maximum_absolute_difference']) and td['maximum_absolute_difference']>=0 and td['backend_violation_count']>=0,'Invalid tensor summary')
                need(b==arrays['repeat_'+name],'CUDA repeated array hashes differ')
                maxima['tensor']=max(maxima['tensor'],td['maximum_absolute_difference'])
            num_failed['tensor']+=int(tensor['status']=='failed')
            if i not in geometry:
                receipt=w.read(w.bound(v['mask_receipt']))
                weights,rois,audit=w.regional.prepare_geometry(r,c,ctx,row,w.bound(v['mask_receipt']),receipt)
                rgb=ctx['old'].prepare_image(row['rgb'],row['native_shape'],ctx['catalog']['data_root'])
                truth=ctx['old'].prepare_image(row['uv'],row['native_shape'],ctx['catalog']['data_root'])
                geometry[i]=dict(hashes=dict(original_regions={k:ah(x) for k,x in weights.items()},anatomical_regions={k:ah(x) for k,x in rois.items()}),rgb=ah(rgb),truth=ah(truth),truth_means=r.intensities(truth,rois),audit=audit)
            geo=geometry[i]
            need(v['geometry']==geo['hashes'] and arrays['rgb']==geo['rgb'] and arrays['truth']==geo['truth'],'Actual frozen inputs/geometry differ')
            for region in REGIONS:
                q=v['actual_gpu']['regional']['regions'][region];t=geo['truth_means'][region]
                need(q['truth']==t['values'] and q['effective_area']==t['area'] and q['reason']==t['reason'],'Independent truth/ROI differs')
            need(v['actual_gpu']['regional']['roi_audit']==geo['audit'],'Landmark audit differs')
            if i==plan['shared_index']:
                value=(v['actual_gpu'],arrays['gpu_prediction'],arrays['gpu_sigma'],v['geometry'])
                old=shared.setdefault(m,value)
                independent_walk(old[0]['combined']['regions'],value[0]['combined']['regions'],'table3')
                independent_walk(old[0]['combined']['table4'],value[0]['combined']['table4'],'table4')
                independent_walk(old[0]['regional'],value[0]['regional'],'regional')
                need(old[3]==value[3],'Shared raw geometry differs')
                shared_diagnostics.append(dict(gpu=g,model=m,actual_values_and_tensor_hashes_exact=old==value,
                    numeric_comparisons={kind:w.backend.compare_payload(a,b,kind) for kind,a,b in [
                        ('table3',old[0]['combined']['regions'],value[0]['combined']['regions']),
                        ('table4',old[0]['combined']['table4'],value[0]['combined']['table4']),
                        ('regional',old[0]['regional'],value[0]['regional'])]}))
            refs.append(rd)
    need(seen==required and len(refs)==60 and set(shared)==set(w.MODELS),'Incomplete prospective validation')
    report=w.seal(dict(schema='test_gpu_program_validation_audit_v1',status='passed',contract=w.desc(ROOT/'contract.json'),plan=contract['validation_plan'],source=w.desc(__file__),
        unique_groups=17,comparison_count=60,all_current_four_gpus=True,cpu_replay_exact=True,cuda_repeat_exact=True,
        shared_case_semantics_consistent=True,shared_case_exact=all(x['actual_values_and_tensor_hashes_exact'] for x in shared_diagnostics),shared_case_diagnostics=shared_diagnostics,independent_raw_input_and_geometry_recheck=True,
        numerical_diagnostic_failed_counts=num_failed,numerical_max_absolute_differences=maxima,old_numerical_thresholds_enforced=False,
        tensor_scope='Full tensor hashes and comparisons are pinned remote harness evidence; this audit independently rechecks scalar payloads, inputs/truth/geometry and all diagnostic scalar comparisons, not full model tensors.',
        shards=shards,records=refs,time=time.time()))
    receipt_path=ROOT/'validation/independent_audit.json';w.save(receipt_path,report)
    gate=w.seal(dict(schema='test_gpu_program_validation_gate_v1',status='passed',contract=w.desc(ROOT/'contract.json'),validation_plan=contract['validation_plan'],validation_receipt=w.desc(receipt_path),records=refs,
        parent_gate=contract.get('parent_gate'),cpu_cuda_numeric_equality_required=False,numerical_differences_reported=True,
        shared_case_semantics_consistent=True,shared_case_exact=all(x['actual_values_and_tensor_hashes_exact'] for x in shared_diagnostics),time=time.time()))
    w.save(ROOT/'pilot_acceptance.json',gate)
    w.verify_gate(ROOT,contract)
    print(json.dumps(dict(gate=w.desc(ROOT/'pilot_acceptance.json'),audit=w.desc(receipt_path),count=60,numeric_failed=num_failed,max_abs=maxima)))

if __name__=='__main__':main()
