"""Use exactly the same frozen scientific functions on CPU and current GPUs.

Numerical backend differences are recorded, never disguised as equality.
The user-authorized gate concerns program/data/structural consistency.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
ROOT=Path('/study/siiba/experiments/test_gpu_program_consistency_20261027_v1')

def ah(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())

def geometry_hashes(weights,rois):
    return dict(original_regions={k:ah(v) for k,v in weights.items()},anatomical_regions={k:ah(v) for k,v in rois.items()})

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    assert args.gpu in range(4)
    sys.path.insert(0,str(ROOT/'source'))
    import worker as w
    contract=w.verify_contract(ROOT)
    plan=w.checked(w.read(ROOT/'validation_plan.json'))
    assert plan['expected_comparisons']==60 and len(plan['selected_groups'])==17
    assert sorted(os.sched_getaffinity(0))==[80+2*args.gpu,81+2*args.gpu]
    bundle=w.load_context(contract['gpu_inventory'][args.gpu]['uuid']);w.verify_contract(ROOT,bundle)
    import numpy as np
    import torch
    c,r,ctx=bundle['combined'],bundle['regional_runner'],bundle['ctx']
    runtime=w.gpu_runtime(bundle,contract,args.gpu)
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    receipt_path=out/('gpu%d_receipt.json'%args.gpu)
    assert not receipt_path.exists()
    records=[]
    for model_id in w.MODELS:
        model=ctx['old'].load_model(model_id,ctx['config']['models'][model_id],Path(ctx['protocol']['eval_config']['path']).parent,ctx['frozen'],'cuda:0')
        fixed_model_state=state_hash(model)
        for index in plan['by_gpu'][str(args.gpu)]:
            dest=out/('gpu%d_%s_%06d.json'%(args.gpu,model_id,index));assert not dest.exists()
            row=ctx['rows'][index];ref,rp,rec=w.load_mask(bundle,index)
            selected=plan['selected_groups'][str(index)]
            assert w.desc(ref)==selected['mask_ref'] and w.desc(rp)==selected['mask_receipt'] and rec['status']=='complete'
            pp=w.P/'evaluation/records'/model_id/('%06d.json'%index)
            assert w.desc(pp) in selected['cpu_references']
            prior=c.verify_record(w.read(pp),ctx,row,index,model_id,w.sha(rp))
            weights,rois,audit=w.regional.prepare_geometry(r,c,ctx,row,rp,rec)
            rgb=ctx['old'].prepare_image(row['rgb'],row['native_shape'],ctx['catalog']['data_root'])
            truth=ctx['old'].prepare_image(row['uv'],row['native_shape'],ctx['catalog']['data_root'])
            original_geo=geometry_hashes(weights,rois);inputs=dict(rgb=ah(rgb),truth=ah(truth))
            start=time.monotonic()
            gp,gs=ctx['old'].infer(model,rgb,model_id,'cuda:0')
            repeat_p,repeat_s=ctx['old'].infer(model,rgb,model_id,'cuda:0')
            gpu_seconds=time.monotonic()-start
            assert ah(gp)==ah(repeat_p) and ah(gs)==ah(repeat_s), 'CUDA repeat differs'
            model.cpu();start=time.monotonic()
            cp,cs=ctx['old'].infer(model,rgb,model_id,'cpu')
            cpu_seconds=time.monotonic()-start;model.to('cuda:0')
            cpu3=ctx['old'].measure(truth,cp,cs,weights,ctx['numeric'])
            cpu4=c.table4_measure(truth,cp,cs,weights,ctx['old'],ctx['numeric'],ctx['rank'])
            assert cpu3==prior['regions'] and cpu4==prior['table4'], 'CPU program replay differs'
            gpu3=ctx['old'].measure(truth,gp,gs,weights,ctx['numeric'])
            gpu4=c.table4_measure(truth,gp,gs,weights,ctx['old'],ctx['numeric'],ctx['rank'])
            cpu_r=w.regional.measure_same_forward(r,truth,cp,rois,audit)
            gpu_r=w.regional.measure_same_forward(r,truth,gp,rois,audit)
            comparison={k:w.compare_program_payloads(a,b,k) for k,a,b in [('table3',cpu3,gpu3),('table4',cpu4,gpu4),('regional',cpu_r,gpu_r)]}
            checks=w.check_program_comparison(comparison)
            tensor=w.check_program_tensors(dict(prediction=cp,sigma=cs),dict(prediction=gp,sigma=gs))
            assert original_geo==geometry_hashes(weights,rois) and inputs==dict(rgb=ah(rgb),truth=ah(truth)), 'Inputs mutated'
            after_state=state_hash(model)
            assert fixed_model_state==after_state,'Model state changed during CPU/CUDA evaluation'
            item=w.seal(dict(schema='test_gpu_program_validation_case_v1',gpu=args.gpu,index=index,model_id=model_id,status='passed',
                source=w.desc(__file__),contract=w.desc(ROOT/'contract.json'),plan=w.desc(ROOT/'validation_plan.json'),runtime=runtime,
                cpu_reference=w.desc(pp),mask_ref=w.desc(ref),mask_receipt=w.desc(rp),structural_program_checks=checks,
                table3=comparison['table3'],table4=comparison['table4'],regional=comparison['regional'],tensor=tensor,
                cpu_replay_exact=True,cpu_replay_details=dict(table3=True,table4=True),gpu_repeat_exact=True,
                same_scientific_functions=True,inputs_unchanged=True,geometry=original_geo,
                model_state_before=fixed_model_state,model_state_after=after_state,
                actual_cpu=dict(combined=dict(regions=cpu3,table4=cpu4),regional=cpu_r),
                actual_gpu=dict(combined=dict(regions=gpu3,table4=gpu4),regional=gpu_r),
                arrays=dict(rgb=inputs['rgb'],truth=inputs['truth'],cpu_prediction=ah(cp),cpu_sigma=ah(cs),gpu_prediction=ah(gp),gpu_sigma=ah(gs),repeat_prediction=ah(repeat_p),repeat_sigma=ah(repeat_s)),
                forwards=dict(cpu=1,cuda=2),timing=dict(cpu=cpu_seconds,cuda_two=gpu_seconds),numeric_differences_are_diagnostic=True))
            w.save(dest,item);records.append(w.desc(dest))
            print(json.dumps(dict(gpu=args.gpu,index=index,model=model_id,status='program_consistent',numeric_gate={k:v['status'] for k,v in comparison.items()})),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(records)==15
    receipt=w.seal(dict(schema='test_gpu_program_validation_shard_v1',status='passed',gpu=args.gpu,indices=plan['by_gpu'][str(args.gpu)],models=list(w.MODELS),
        source=w.desc(__file__),contract=w.desc(ROOT/'contract.json'),plan=w.desc(ROOT/'validation_plan.json'),count=len(records),records=records,runtime=runtime))
    w.save(receipt_path,receipt)
    print(json.dumps(dict(status='validation_complete',receipt=w.desc(receipt_path))),flush=True)

if __name__=='__main__':main()
