"""Five-model CPU FP32 inference and original split-conformal numerical program."""
from cpu_common import *
import traceback
import numpy as np

def programs(q,mid):
 os.sched_setaffinity(0,q['resources']['model_cpus'][mid]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
 sys.path.insert(0,str(N/'source'))
 cpu=module(R/'cpu_inference.py','cpu_inference_adapter')
 nu=module(R/'frozen_numerics.py','cpu_conformal_numeric')
 cache=module(N/'source/cache_io.py','cpu_conformal_original_cache')
 model=cpu.Original(q,len(q['resources']['model_cpus'][mid]));t=time.monotonic();model.model(mid);load=time.monotonic()-t
 import torch
 assert not torch.cuda.is_initialized() and torch.get_num_threads()==len(q['resources']['model_cpus'][mid])
 return model,nu,cache,load

def case(q,s,cache,model,index):
 c=cache.load_case(index,q,s)
 if c['missing_reason'] is not None:return c,None,None,None
 rgb,row,ready=model.rgb(c);return c,rgb,row,ready

def frozen_task(q,index,mid,manifest,nu,scale):
 p=core_record(mid,index);assert sha(p)==manifest[str(p)]['sha256'];v=checked(read(p))
 assert v['index']==index and v['model']==mid and v['status']=='ok'
 return nu.task_values(v,scale),desc(p)

def check_arrays(pred,sigma):
 assert pred.shape==(3,1024,1024) and sigma.shape==(1024,1024) and pred.dtype==sigma.dtype==np.float32
 assert np.isfinite(pred).all() and np.isfinite(sigma).all() and pred.min()>=0 and pred.max()<=1 and sigma.min()>0

def timing(mid):
 q=context();start=time.monotonic();model,nu,cache,load=programs(q,mid);s=read(verify(q['selection']));rows=[]
 for index in q['pilot_indices']:
  t=time.monotonic();c,rgb,row,ready=case(q,s,cache,model,index);prep=time.monotonic()-t;assert c['missing_reason'] is None
  t=time.monotonic();pred,sg=model.predict(mid,rgb,row);first=time.monotonic()-t;check_arrays(pred,sg)
  t=time.monotonic();p2,s2=model.predict(mid,rgb,row);repeat=time.monotonic()-t
  assert np.array_equal(pred,p2) and np.array_equal(sg,s2),'Same CPU execution repeat differs'
  t=time.monotonic();scores=nu.cpu_scores(c['truth'],pred,sg,c['weights']);scoretime=time.monotonic()-t
  fake=dict(pixel_sigma=nu.finite_threshold([scores['normalized']]*40),pixel_absolute=nu.finite_threshold([scores['absolute']]*40))
  t=time.monotonic();evaluation=nu.cpu_evaluate(c['truth'],pred,sg,c['weights'],fake);evaltime=time.monotonic()-t
  # Independent direct interval event and width checks on real saved tensors.
  error=np.abs(pred.astype(np.float64)-c['truth'].astype(np.float64));sigma=sg.astype(np.float64)
  for region,w in c['weights'].items():
   area=float(w.astype(np.float64).sum())
   for item in evaluation[region]:
    if area==0:assert item['coverage'] is None;continue
    scaled=item['method']!='group_conformal_absolute';z=1.959963984540054 if item['method']=='Gaussian_1.96_sigma' else (scores['normalized'] if scaled else scores['absolute'])
    included=(error<=z*sigma[None]) if scaled else (error<=z)
    direct=sum(float((included[k].astype(np.float64)*w).sum()) for k in range(3))/(3*area)
    assert abs(item['coverage']-direct)<2e-15
  t=time.monotonic();cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);delta=np.abs(pred.astype(np.float64)-cached.astype(np.float64));diagnostic_time=time.monotonic()-t
  rows.append(dict(index=index,input_bindings=c['inputs'],preparation_seconds=prep,forward_seconds=first,repeat_forward_seconds=repeat,score_seconds=scoretime,evaluation_seconds=evaltime,
                   diagnostic_seconds=diagnostic_time,estimated_CAL_group_seconds=prep+repeat+scoretime+diagnostic_time,estimated_EVAL_group_seconds=prep+repeat+scoretime+evaltime+diagnostic_time,
                   same_CPU_repeat_exact=True,CPU_scores_and_direct_interval_events_verified=True,mean_diagnostic=dict(max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean())),scores=scores,
                   cpu_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),cpu_sigma_sha256=hashlib.sha256(sg.tobytes()).hexdigest(),historical_sigma_sha256=ready['outputs'][mid]['sigma_diagnostic']['sha256']))
  progress('pilot_'+mid,'CPU_real_end_to_end_validation',model=mid,cases=len(rows),elapsed_seconds=time.monotonic()-start,last=rows[-1])
 import torch
 save(R/('validation/'+mid+'.json'),dict(time=now(),status='passed',identity=identity(),model=mid,device='cpu',cuda_initialized=torch.cuda.is_initialized(),runtime=model.runtime,model_load_seconds=load,cases=rows,
                                     checkpoint_sha256=q['checkpoint_sha256'][mid],request=desc(R/'request.json')))

def work(mid,stage):
 q=context();assert checked(read(R/'validation/gate.json'))['status']=='passed';s=read(verify(q['selection']));split=checked(read(verify(q['conformal_policy']['split'])))
 indices=split['calibration_indices' if stage=='cal' else 'evaluation_indices'];calfile=C/'calibration_frozen.json'
 calibrations=checked(read(calfile))['models'] if stage=='eval' else None
 model,nu,cache,load=programs(q,mid)
 manifest={x['path']:x for x in read(verify(q['bindings']['record_manifest']))};done=0;valid=0;times=[];start=time.monotonic();written_bytes=0
 for index in indices:
  no_failures();out=record_path(stage,mid,index);assert not out.exists(),'No blind resume or duplicate execution'
  claim=C/(stage+'/claims/'+mid+'_%06d.json'%index)
  reservation=checked(read(claim));assert reservation['CPU_execution_request']==desc(R/'request.json') and reservation['model']==mid and reservation['index']==index
  t=time.monotonic();c,rgb,row,ready=case(q,s,cache,model,index)
  if c['missing_reason'] is not None:
   fields=dict(status='unavailable_geometry',forward_count=0,scores=None,tasks=None,pixel_evaluation=None,task_evaluation=None)
  else:
   pred,sg=model.predict(mid,rgb,row);check_arrays(pred,sg);scores=nu.cpu_scores(c['truth'],pred,sg,c['weights'])
   ev=nu.cpu_evaluate(c['truth'],pred,sg,c['weights'],calibrations[mid]) if calibrations else None
   tasks,taskrecord=frozen_task(q,index,mid,manifest,nu,scores['regions']['F']['mean_sigma'])
   tev=nu.task_evaluate(tasks,scores['regions']['F']['mean_sigma'],calibrations[mid]) if calibrations else None
   cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);delta=np.abs(pred.astype(np.float64)-cached.astype(np.float64))
   fields=dict(status='ok',forward_count=1,scores=scores,tasks=tasks,task_record=taskrecord,pixel_evaluation=ev,task_evaluation=tev,
               mean_numerical_diagnostics=dict(max_abs_difference=float(delta.max()),mean_abs_difference=float(delta.mean()),cached_prediction=ready['outputs'][mid]['prediction'],
               new_prediction_tensor_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),new_sigma_tensor_sha256=hashlib.sha256(sg.tobytes()).hexdigest(),historical_sigma_sha256=ready['outputs'][mid]['sigma_diagnostic']['sha256']))
   if index in q['QC_indices'][stage]:
    p=C/('QC/'+stage+'_'+mid+'_%06d.npz'%index);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,truth=c['truth'],prediction=pred,sigma=sg,**c['weights'])
    written_bytes+=p.stat().st_size
   valid+=1
  elapsed=time.monotonic()-t;times.append(elapsed);done+=1
  save(out,dict(time=now(),phase=stage,index=index,group_id=c['group_id'],model=mid,device='cpu',identity=identity(),inputs=c['inputs'],missing_reason=c['missing_reason'],
                checkpoint_sha256=q['checkpoint_sha256'][mid],calibration=desc(calfile) if calibrations else None,request_sha256=q['bindings']['scientific_request']['sha256'],CPU_execution_request=desc(R/'request.json'),job=desc(N/'jobs/conformal/manifest.json'),elapsed_seconds=elapsed,**fields))
  written_bytes+=out.stat().st_size;assert written_bytes<q['storage_cap_bytes']//10,'Owned per-model/stage storage cap exceeded'
  progress(stage+'_'+mid,'producing_'+stage,model=mid,done=done,valid=valid,total=len(indices),elapsed_seconds=time.monotonic()-start,last_index=index,recent_group_seconds=times[-20:],model_load_seconds=load,owned_output_bytes=written_bytes)
 import torch
 assert not torch.cuda.is_initialized()
 save(R/('conformal/'+stage+'/model_completion/'+mid+'.json'),dict(time=now(),status='complete',model=mid,records=done,valid=valid,elapsed_seconds=time.monotonic()-start,identity=identity(),cuda_initialized=False,request=desc(R/'request.json')))

if __name__=='__main__':
 try:
  if sys.argv[1]=='pilot':timing(sys.argv[2])
  else:work(sys.argv[2],sys.argv[3])
 except BaseException:
  save(R/('failures/worker_'+str(os.getpid())+'.json'),dict(time=now(),identity=identity(),argv=sys.argv,traceback=traceback.format_exc()));raise
