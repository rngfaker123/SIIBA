"""Detached supervisor. Validation first, then explicit audited exclusive handoff, no retries."""
from a6000_common import *
import traceback
def final_audit(q,p):
 acceptance=checked(read(C/'final_acceptance.json'));assert acceptance['status']=='passed' and acceptance['records']==109495
 for k in ('summary','calibration','manifest','raw'):verify(acceptance[k])
 for d in acceptance['CPU_QC']:verify(d);assert checked(read(d['path']))['status']=='passed'
 transfer=checked(read(A/'ownership_transfer_receipt.json'));old=checked(read(verify(transfer['preservation_manifest'])))
 preserved={d['path']:d for d in old['files']}
 for d in old['files']:verify(d)
 original_claims=checked(read(verify(transfer['original_claim_manifest'])))
 assert original_claims['count']==109495
 for d in original_claims['files']:verify(d)
 plan=checked(read(verify(transfer['pending_plan'])));split=checked(read(verify(q['split'])));s=read(verify(q['selection']));valid=set(s['geometry_valid_indices'])
 counts=dict(records=0,forwards=0,missing=0,preserved_CPU=0,new_CUDA=0);uuids={g['uuid'] for g in q['gpus']};pending={(stage,z['model'],z['index']) for stage in ('cal','eval') for z in plan[stage]}
 for stage in ('cal','eval'):
  for m in MODELS:
   for i in split['calibration_indices' if stage=='cal' else 'evaluation_indices']:
    path=record_path(stage,m,i);v=checked(read(path));assert v['CPU_execution_request']==q['CPU_request']
    assert v['request_sha256']==p['bindings']['scientific_request']['sha256'] and v['checkpoint_sha256']==p['checkpoint_sha256'][m]
    assert (v['index'],v['group_id'],v['phase'],v['model'])==(i,s['all_group_ids'][i],stage,m)
    if str(path) in preserved:assert v['device']=='cpu' and sha(path)==preserved[str(path)]['sha256'];counts['preserved_CPU']+=1
    else:
     assert (stage,m,i) in pending and v['device']=='cuda' and v['gpu_uuid'] in uuids and v['GPU_execution_request']==desc(A/'request.json') and v['ownership_transfer']==desc(A/'ownership_transfer_receipt.json');counts['new_CUDA']+=1
    if i in valid:assert v['forward_count']==1 and v['status']=='ok';counts['forwards']+=1
    else:assert v['forward_count']==0 and v['status']=='unavailable_geometry' and v['missing_reason'];counts['missing']+=1
    counts['records']+=1
 assert (counts['records'],counts['forwards'],counts['missing'])==(109495,107110,2385)
 assert counts['new_CUDA']==len(pending)
 context()
 save(A/'independent_execution_acceptance.json',dict(time=now(),status='passed',counts=counts,all_preserved_byte_hashes_unchanged=True,original_numerical_acceptance=desc(C/'final_acceptance.json'),request=desc(A/'request.json'),ownership_transfer=desc(A/'ownership_transfer_receipt.json'),device_policy='Disclosed preserved CPU records plus new original-program CUDA records; no old results relabelled. Finite numerical differences remain diagnostic.',limits='Original collector performs all-record scientific binding checks and fixed actual-tensor CPU QC/statistics. This independent execution audit does not replay every forward or original detector.'))
 save(A/'scientific_completion.json',dict(time=now(),status='complete_full_five_model_Conformal',acceptance=desc(A/'independent_execution_acceptance.json'),summary=desc(C/'summary.json'),full_pixel_sigma_and_risk_coverage=True,device_provenance='Preserved CPU and prospectively validated CUDA continuation',request=desc(A/'request.json')))
def run():
 q,p=context();os.sched_setaffinity(0,q['supervisor_cpus']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
 save(A/'active.json',dict(time=now(),identity=identity(),request=desc(A/'request.json')));children=[]
 try:
  for lane in range(9):
   with (A/('logs/lane'+str(lane)+'.log')).open('xb') as log:
    child=subprocess.Popen([str(PY),'-B','-u',str(A/'worker.py'),str(lane)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env(q['gpus'][lane%3]['uuid']))
   children.append(child);save(A/('worker_launches/lane'+str(lane)+'.json'),dict(time=now(),lane=lane,identity=identity(child.pid),gpu_uuid=q['gpus'][lane%3]['uuid'],request=desc(A/'request.json')))
  files=[A/('validation/lane'+str(lane)+'.json') for lane in range(9)]
  while not all(x.exists() for x in files):
   no_failures();assert all(x.poll() is None for x in children),'Worker exited before validation';progress('owner','prospective_nine_lane_three_card_validation',ready=sum(x.exists() for x in files));time.sleep(5)
  rows=[checked(read(x)) for x in files]
  assert all(v['status']=='passed' and len(v['cases'])==10 for v in rows)
  assert all(z['same_card_repeat_exact'] for v in rows for z in v['cases'][:5])
  cross=[]
  for j in range(10):
   a=[v['cases'][j] for v in rows];assert len({(x['index'],x['model']) for x in a})==1
   cross.append(dict(index=a[0]['index'],model=a[0]['model'],GPU_UUIDs=[v['gpu_uuid'] for v in rows],prediction_hashes=[z['CUDA_prediction_sha256'] for z in a],sigma_hashes=[z['CUDA_sigma_sha256'] for z in a],score_differences=[compare(a[0]['actual_CUDA_scores'],z['actual_CUDA_scores']) for z in a[1:]],cross_card_exact_observed=len({z['CUDA_prediction_sha256'] for z in a})==len({z['CUDA_sigma_sha256'] for z in a})==1))
  save(A/'validation/gate.json',dict(time=now(),status='passed',actual_GPU_model_cases=90,same_card_repeat_comparisons=45,all_five_models_on_all_three_cards=True,independent_CPU_formula_comparisons=90,validations=[desc(x) for x in files],cross_card_diagnostics=cross,source_science_unchanged=True,request=desc(A/'request.json')))
  while not (A/'ownership_transfer_receipt.json').exists():no_failures();assert all(x.poll() is None for x in children);progress('owner','validated_waiting_exclusive_CPU_handoff');time.sleep(5)
  assert checked(read(A/'ownership_transfer_receipt.json'))['status']=='exclusive_pending_ownership_transferred'
  while any(x.poll() is None for x in children):
   no_failures();assert all(x.poll() in (None,0) for x in children),'Owned GPU worker failed';progress('owner','producing_remaining_Conformal_GPU',worker_states=[dict(pid=x.pid,exit_code=x.poll()) for x in children]);time.sleep(10)
  assert all(x.returncode==0 for x in children)
  while not (C/'final_acceptance.json').exists():no_failures();progress('owner','waiting_unchanged_original_statistics_and_tensor_QC');time.sleep(10)
  final_audit(q,p);save(A/'supervisor_completion.json',dict(time=now(),status='complete',scientific_completion=desc(A/'scientific_completion.json'),identity=identity()))
  progress('owner','complete_full_five_model_Conformal')
 except BaseException:
  save(A/('failures/owner_'+str(os.getpid())+'.json'),dict(time=now(),identity=identity(),traceback=traceback.format_exc()))
  for x in children:
   if x.poll() is None:x.terminate()
  for x in children:
   try:x.wait(timeout=30)
   except subprocess.TimeoutExpired:x.kill();x.wait()
  raise
def launch():
 q,p=context();assert checked(read(A/'tests.json'))['status']=='passed'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 save(A/'launch_reservation.json',dict(time=now(),identity=identity(),request=desc(A/'request.json')))
 (A/'logs').mkdir(exist_ok=True)
 with (A/'logs/owner.log').open('xb') as log:
  child=subprocess.Popen([str(PY),'-B','-u',__file__,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env())
 save(A/'launch_receipt.json',dict(time=now(),identity=identity(child.pid),request=desc(A/'request.json'),detached=True,stdin='DEVNULL',training_jobs_touched=False))
 print(json.dumps(dict(status='GPU_validation_owner_started_once',identity=identity(child.pid),request=desc(A/'request.json'))))
if __name__=='__main__':
 if sys.argv[1]=='launch':launch()
 else:run()
