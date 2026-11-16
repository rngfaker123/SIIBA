"""Three nonoverlapping group lanes; one forward per missing model/group."""
from a6000_common import *
import numpy as np,traceback
class OriginalCPUScoreEngine:
 def __init__(self,nu):self.nu=nu
 def compute(self,truth,pred,sigma,weights,cal=None):
  return self.nu.cpu_scores(truth,pred,sigma,weights),self.nu.cpu_evaluate(truth,pred,sigma,weights,cal) if cal else {}
def direct_intervals(truth,pred,sg,weights,cal,evaluation,nu):
 error=np.abs(pred.astype(np.float64)-truth.astype(np.float64));sigma=sg.astype(np.float64);rows=[]
 for r,w in weights.items():
  w=w.astype(np.float64);area=float(w.sum())
  for item in evaluation[r]:
   if not area:assert item['coverage'] is None;continue
   scaled=item['method']!='group_conformal_absolute'
   q=1.959963984540054 if item['method']=='Gaussian_1.96_sigma' else nu.threshold(cal['pixel_sigma' if scaled else 'pixel_absolute'])
   # 直接比较标准化残差

   included=error/sigma[None]<=q if scaled else error<=q
   cov=sum(float((included[k].astype(np.float64)*w).sum()) for k in range(3))/(3*area)
   assert abs(cov-item['coverage'])<2e-15
   alternative=error<=q*sigma[None] if scaled else error<=q
   alt=sum(float((alternative[k].astype(np.float64)*w).sum()) for k in range(3))/(3*area)
   rows.append(dict(region=r,method=item['method'],independent_direct_coverage=cov,original_coverage=item['coverage'],absolute_difference=abs(cov-item['coverage']),algebraic_multiply_alternative_coverage=alt,algebraic_multiply_alternative_difference=alt-cov,actual_event_definition='abs(pred-truth)/sigma <= q for scaled intervals, matching frozen program'))
 return rows
def run(lane):
 q,p=context();os.sched_setaffinity(0,q['worker_cpus'][lane]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
 uuid=q['gpus'][lane%3]['uuid']
 published = Path(__file__).resolve().parents[1]
 cpu_source = Path(__file__).resolve().parent / 'cpu'
 sys.path.insert(0,str(published));sys.path.insert(0,str(cpu_source))
 infer=module(published/'inference.py','continuation_original_GPU_inference');nu=module(published/'numerics.py','continuation_original_numerics');cache=module(published/'cache_io.py','continuation_original_cache');helpers=module(cpu_source/'worker.py','preserved_CPU_pure_helpers')
 start=time.monotonic();model=infer.Original(p,uuid)
 for mid in MODELS:model.model(mid)
 import torch
 assert torch.cuda.is_initialized() and torch.cuda.get_device_name(0)=='NVIDIA RTX A6000' and torch.get_num_threads()==2
 assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
 engine=OriginalCPUScoreEngine(nu);load=time.monotonic()-start;selection=read(verify(p['selection']));validation=[]
 for j,index in enumerate(q['pilot_indices']):
  t=time.monotonic();c,rgb,row,ready=helpers.case(p,selection,cache,model,index);prep=time.monotonic()-t;assert c['missing_reason'] is None
  for mid in MODELS:
   t=time.monotonic();pred,sg=model.predict(mid,rgb,row);torch.cuda.synchronize();fw=time.monotonic()-t;helpers.check_arrays(pred,sg)
   repeated=None
   if j==0:
    p2,s2=model.predict(mid,rgb,row);torch.cuda.synchronize();assert np.array_equal(pred,p2) and np.array_equal(sg,s2),'Same card deterministic repeat changed';repeated=True
   t=time.monotonic();scores,_=engine.compute(c['truth'],pred,sg,c['weights']);torch.cuda.synchronize();score_seconds=time.monotonic()-t
   fake=dict(pixel_sigma=nu.finite_threshold([scores['normalized']]*40),pixel_absolute=nu.finite_threshold([scores['absolute']]*40))
   t=time.monotonic();_,ev=engine.compute(c['truth'],pred,sg,c['weights'],fake);torch.cuda.synchronize();eval_seconds=time.monotonic()-t
   diagnostics=direct_intervals(c['truth'],pred,sg,c['weights'],fake,ev,nu)
   assert all(0<=z['coverage']<=1 and z['width']>=0 for rr in ev.values() for z in rr if z['coverage'] is not None)
   prior=checked(read(R/('validation/'+mid+'.json')))['cases'][j]
   cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);delta=np.abs(pred.astype(np.float64)-cached.astype(np.float64))
   validation.append(dict(index=index,model=mid,input_bindings=c['inputs'],checkpoint_sha256=p['checkpoint_sha256'][mid],same_card_repeat_exact=repeated,forward_seconds=fw,shared_case_load_seconds=prep,score_seconds=score_seconds,evaluation_pass_including_score_seconds=eval_seconds,CPU_saved_pilot_scores=prior['scores'],actual_CUDA_scores=scores,CPU_CUDA_saved_pilot_score_differences=compare(prior['scores'],scores),independent_CPU_direct_interval_event_checks=diagnostics,score_device='cpu_original_FP64_program',CUDA_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),CUDA_sigma_sha256=hashlib.sha256(sg.tobytes()).hexdigest(),mean_vs_historical_CUDA=dict(max_absolute_difference=float(delta.max()),mean_absolute_difference=float(delta.mean()))))
   progress('lane'+str(lane),'prospective_GPU_validation',cases=len(validation),gpu_uuid=uuid,last_model=mid)
 save(A/('validation/lane'+str(lane)+'.json'),dict(time=now(),status='passed',lane=lane,identity=identity(),gpu_uuid=uuid,runtime=model.runtime,model_load_seconds=load,cases=validation,request=desc(A/'request.json'),precision=q['precision']))
 while not (A/'ownership_transfer_receipt.json').exists():no_failures();time.sleep(3)
 transfer=checked(read(A/'ownership_transfer_receipt.json'));assert transfer['status']=='exclusive_pending_ownership_transferred'
 preservation=checked(read(verify(transfer['preservation_manifest'])));preserved={v['path']:v for v in preservation['files']}
 work=checked(read(verify(transfer['pending_plan'])));manifest={v['path']:v for v in read(verify(p['bindings']['record_manifest']))}
 for stage in ('cal','eval'):
  if stage=='eval':
   while not (C/'calibration_frozen.json').exists():no_failures();progress('lane'+str(lane),'waiting_original_calibration_freeze');time.sleep(5)
   frozen=checked(read(C/'calibration_frozen.json'));assert frozen['EVAL_records_at_freeze']==0;cal=frozen['models']
  else:cal=None;assert not list((C/'eval/records').glob('*/*.json.gz'))
  tasks=work[stage];own=[z for z in tasks if z['index']%9==lane];byindex={}
  for z in own:byindex.setdefault(z['index'],[]).append(z['model'])
  new=valid=missing=0;timings=[];started=time.monotonic();written=0
  for index,mids in sorted(byindex.items()):
   no_failures();assert index%9==lane
   save(A/('claims/'+stage+'_%06d.json'%index),dict(time=now(),lane=lane,index=index,models=mids,identity=identity(),request=desc(A/'request.json'),transfer=desc(A/'ownership_transfer_receipt.json')))
   t=time.monotonic();c,rgb,row,ready=helpers.case(p,selection,cache,model,index)
   for mid in MODELS:
    out=record_path(stage,mid,index)
    if mid not in mids:
     assert str(out) in preserved and sha(out)==preserved[str(out)]['sha256'];continue
    assert not out.exists(),'Nonoverlapping pending plan violated'
    old_claim=checked(read(C/(stage+'/claims/'+mid+'_%06d.json'%index)));assert old_claim['CPU_execution_request']==q['CPU_request'] and old_claim['model']==mid and old_claim['index']==index
    tt=time.monotonic()
    if c['missing_reason'] is not None:
     fields=dict(status='unavailable_geometry',forward_count=0,scores=None,tasks=None,pixel_evaluation=None,task_evaluation=None);missing+=1
    else:
     pred,sg=model.predict(mid,rgb,row);helpers.check_arrays(pred,sg);scores,pe=engine.compute(c['truth'],pred,sg,c['weights'],cal[mid] if cal else None)
     tasks_v,source=helpers.frozen_task(p,index,mid,manifest,nu,scores['regions']['F']['mean_sigma']);te=nu.task_evaluate(tasks_v,scores['regions']['F']['mean_sigma'],cal[mid]) if cal else None
     cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);delta=np.abs(pred.astype(np.float64)-cached.astype(np.float64))
     fields=dict(status='ok',forward_count=1,scores=scores,tasks=tasks_v,task_record=source,pixel_evaluation=pe if cal else None,task_evaluation=te,mean_numerical_diagnostics=dict(max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean()),cached_prediction=ready['outputs'][mid]['prediction'],new_prediction_tensor_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),new_sigma_tensor_sha256=hashlib.sha256(sg.tobytes()).hexdigest(),historical_sigma_sha256=ready['outputs'][mid]['sigma_diagnostic']['sha256']));valid+=1
     if index in q['QC_indices'][stage]:
      qp=C/('QC/'+stage+'_'+mid+'_%06d.npz'%index);qp.parent.mkdir(parents=True,exist_ok=True)
      with qp.open('xb') as f:np.savez_compressed(f,truth=c['truth'],prediction=pred,sigma=sg,**c['weights'])
      written+=qp.stat().st_size
    save(out,dict(time=now(),phase=stage,index=index,group_id=c['group_id'],model=mid,device='cuda',score_device='cpu_original_FP64_program',gpu_uuid=uuid,identity=identity(),inputs=c['inputs'],missing_reason=c['missing_reason'],checkpoint_sha256=p['checkpoint_sha256'][mid],calibration=desc(C/'calibration_frozen.json') if cal else None,request_sha256=p['bindings']['scientific_request']['sha256'],CPU_execution_request=q['CPU_request'],GPU_execution_request=desc(A/'request.json'),ownership_transfer=desc(A/'ownership_transfer_receipt.json'),job=desc(N/'jobs/conformal/manifest.json'),elapsed_seconds=time.monotonic()-tt,**fields))
    new+=1;written+=out.stat().st_size;assert written<q['storage_cap_bytes']//3
   timings.append(time.monotonic()-t)
   progress('lane'+str(lane),'five_models_GPU_'+stage,gpu_uuid=uuid,groups_done=len(timings),groups_total=len(byindex),new_records=new,valid_forwards=valid,missing_records=missing,last_index=index,recent_group_seconds=timings[-30:],elapsed_seconds=time.monotonic()-started)
  save(A/('completion/'+stage+'_lane'+str(lane)+'.json'),dict(time=now(),status='complete',identity=identity(),lane=lane,gpu_uuid=uuid,new_records=new,valid_forwards=valid,missing_records=missing,groups=len(timings),elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':
 try:run(int(sys.argv[1]))
 except BaseException:
  save(A/('failures/worker_'+str(os.getpid())+'.json'),dict(time=now(),identity=identity(),traceback=traceback.format_exc()));raise
