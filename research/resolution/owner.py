"""Unique detached owner: CPU selection/preparation, explicit grant, exact CUDA MI."""
import argparse,csv,math,traceback,gc
from common import *

def fail(action):
 save(ROOT/'failures'/('%s_%d.json'%(action,os.getpid())),dict(time=now(),identity=identity(),action=action,traceback=traceback.format_exc(),other_experiments_modified=False))
def configure():
 assert not (ROOT/'request.json').exists(),'Request already frozen'
 old=checked(read(OLD/'request.json'));v=identity();retire=E/'pl_controls_original_uv_pipeline_20261031_v1'
 paths=dict(protocol=ROOT/'protocol_v1.json',authorization=ROOT/'authorization.json',catalog=E/'three_region_test_eval_20261025_v1/catalog.json',old_request=OLD/'request.json',old_table1_input=E/'mi_atlas_test1000_20261025_v1/mi_input.json',native_helper=E/'ita_stratified_mi_test_20261026_v1_inputs/native_sources.py',ita_request=I/'request.json',ita_contract=I/'contract.json',detector_core=CORE,detector_protocol=DETECT/'protocol.json',accepted_detector_validation=E/'multimodal_resolution_native_mi_20261030_v1/validation/detector_gate.json',training_correction_authority=retire/'human_request.json',retirement_completion=retire/'retirement/side/completion.json',main_resource_grant=retire/'resource_grants/side_MI.json',main_resource_request=retire/'request.json')
 bindings={n:desc(p) for n,p in paths.items()}
 bindings['return_marker_writer']=desc(E/'siiba_experiment_chain_20261030_v1/source/chain_common.py')
 assert bindings['catalog']['sha256']=='81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89'
 assert bindings['native_helper']['sha256']=='10ab9c799d115f509ab6be135d4911a6d0f240acc7397021bb28decc2dd0e25b'
 assert sha(ROOT/'source/native_detector.py')==old['sources']['native_detector.py']['sha256']
 assert sha(ROOT/'source/ksg_stability_cuda_v2.py')==old['sources']['ksg_stability_cuda_v2.py']['sha256']
 assert sha(ROOT/'source/ksg_stability_kernels_v1.cu')==old['sources']['ksg_stability_kernels_v1.cu']['sha256']
 assert read(paths['retirement_completion'])['status']=='cancelled_by_human'
 assert bindings['main_resource_grant']['sha256']=='3390ba2cb296f7271aa28fd275bc249db3803733eb867d5a864f7f69fe14e001'
 assert bindings['main_resource_request']['sha256']=='3a8655b0ddb272976f73f51a2cec56bf3e70282c4c3b4369af88c76a8541c8c8'
 tasks=[]
 for n in BUDGETS:
  for f in FACTORS:
   for seed in SEEDS:
    for domain in DOMAINS:
     for x,y in PAIRS:tasks.append(dict(budget=n,factor=f,seed=seed,domain=domain,x=x,y=y,lane=len(tasks)%3))
 assert len(tasks)==864
 save(ROOT/'request.json',dict(schema='common1000_sixpair_native_resolution_MI_v1',time=now(),bindings=bindings,sources={p.name:desc(p) for p in (ROOT/'source').iterdir() if p.is_file()},original_runner=old['original_runner'],original_estimator=old['original_estimator'],host=dict(boot_id=v['boot_id'],pid1_starttick=v['pid1_starttick'],gpus=inventory()),tasks=tasks,raw_slots=864,summary_cells=288,CPU_only_until_explicit_GPU_grant=True,old_results_readonly=True))
 print('CONFIGURED',sha(ROOT/'request.json'),flush=True)
def tests():
 from prepare import ordered_selection,partition
 count=0
 records=[dict(index=i,eligible=i%3!=0) for i in range(30)]
 assert ordered_selection(records,5)==[1,2,4,5,7];count+=1
 assert ordered_selection(records,100)==[r['index'] for r in records if r['eligible']];count+=1
 assert ordered_selection([],100)==[];count+=1
 v=dict(a=1,b=[2,3]);sealed=dict(v,record_sha256=digest(v));checked(sealed);count+=1
 try:checked(dict(sealed,a=2));raise RuntimeError('Corruption accepted')
 except AssertionError:count+=1
 l=dict(v,record_sha256=hashlib.sha256((json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest());legacy(l);count+=1
 try:checked(l);raise RuntimeError('Wrong seal origin accepted')
 except AssertionError:count+=1
 mask=np.ones((40,50),bool)
 for f in (2,4,8):
  ids=sampler(mask,'UNIT',f,'P');assert ids.shape==(3,1000) and np.array_equal(ids,sampler(mask,'UNIT',f,'P'));count+=1
  for a in ids:assert len(np.unique(a))==1000 and mask.ravel()[a].all();count+=1
  for n in BUDGETS:assert np.array_equal(ids[:,:n],ids[:,:1000][:,:n]);count+=1
 rng=np.random.default_rng(2);a=Image.fromarray(rng.integers(0,256,(32,64,3),dtype=np.uint8))
 for f in FACTORS:
  b=lower(a,f);assert b.size==(64//f,32//f);count+=1
  if f==1:assert np.array_equal(np.asarray(a),np.asarray(b));count+=1
 q,*_=context();assert len({task_key(t) for t in q['tasks']})==864;count+=1
 assert {sum(t['lane']==i for t in q['tasks']) for i in range(3)}=={288};count+=1
 save(ROOT/'validation/cpu_tests.json',dict(time=now(),passed=True,checks=count,source=desc(__file__),request=desc(ROOT/'request.json')))
 print('CPU_TESTS_PASSED',count,flush=True)
def cpu_gate():
 q,runner,est,cat,cached=context()
 gate=checked(read(verify(q['bindings']['accepted_detector_validation'])));assert gate['passed'] and gate['no_TEST_used']
 old=checked(read(verify(q['bindings']['old_request'])))
 assert sha(ROOT/'source/native_detector.py')==old['sources']['native_detector.py']['sha256']
 for d in gate['reports']:verify(d)
 save(ROOT/'validation/detector_gate.json',dict(time=now(),passed=True,accepted_original_validation=q['bindings']['accepted_detector_validation'],native_detector_bytes_unchanged=True,TEST_threshold_tuning=False,reports=gate['reports'],request=desc(ROOT/'request.json')))
 # 固定首组验证
 import prepare
 prepare.initializer();r=prepare.screen_one(0)
 feature=None
 if r['eligible']:
  prepare.feature_one(0);feature=desc(group_path(0))
 save(ROOT/'validation/real_CPU_smoke.json',dict(time=now(),passed=True,catalogue_index=0,candidate=desc(candidate_path(0)),feature=feature,CUDA_used=False,request=desc(ROOT/'request.json')))
def grant_valid(q):
 p=verify(q['bindings']['main_resource_grant']);g=checked(read(p))
 assert g['purpose']=='common1000_allpairs_MI' and g['exclusive_three_side_UUIDs'] is True
 assert g['request']==q['bindings']['main_resource_request'] and g['authorization']==q['bindings']['training_correction_authority']
 assert g['host']['boot_id']==q['host']['boot_id'] and g['host']['pid1_starttick']==q['host']['pid1_starttick']
 assert g['host']['gpus']==[{k:v for k,v in a.items() if k!='pci'} for a in q['host']['gpus']]
 verify(g['request']);verify(g['authorization']);retired=read(verify(q['bindings']['retirement_completion']))
 assert retired['status']=='cancelled_by_human' and retired['actual_GPU_compute_empty']
 return True
def wait_resources(q):
 while not grant_valid(q):
  no_failures();check_host(q);state(phase='CPU_ready_waiting_explicit_GPU_resource_grant',N=1000,raw_done=0,total_raw=864);time.sleep(30)
 observations=[]
 for _ in range(2):
  no_failures();check_host(q);assert not gpu_processes(),'GPU still owned; do not compete'
  observations.append(dict(time=now(),identity=identity(),inventory=inventory(),compute=[]));time.sleep(5)
 save(ROOT/'GPU_resource_acceptance.json',dict(time=now(),grant=q['bindings']['main_resource_grant'],observations=observations,passed=True,request=desc(ROOT/'request.json')))
def gpu_validate(lane,backend,est,q):
 reports=[]
 for j in range(4):
  rng=np.random.default_rng(20261030+j);x=rng.normal(size=(257,3));y=rng.normal(size=(257,3))
  if j==1:y=.7*x+.3*y
  if j==2:x=x*np.array([.01,1,100]);y=y*np.array([.1,2,30])
  v,d=backend.estimate_ksg_gpu(x,y,est,k=5,workers=2,validation_query_indices=np.arange(257),return_arrays=True)
  assert v['mi_bits']==est.ksg_mi(x,y,k=5,workers=2)['mi_bits']
  arrays=d.pop('arrays');again,dr=backend.estimate_ksg_gpu(x,y,est,k=5,workers=2,validation_query_indices=np.arange(257),return_arrays=True)
  ar=dr.pop('arrays');assert v['mi_bits']==again['mi_bits'] and all(np.array_equal(arrays[k],ar[k]) for k in arrays)
  p=ROOT/'validation'/('lane%d_fixture%d.npz'%(lane,j))
  with p.open('xb') as f:np.savez(f,x=x,y=y,**arrays)
  reports.append(dict(fixture=j,mi_bits=v['mi_bits'],diagnostic=d,repeat_diagnostic=dr,same_card_repeat_exact=True,arrays=desc(p)))
 save(ROOT/'validation'/('lane%d.json'%lane),dict(time=now(),passed=True,identity=identity(),reports=reports,request=desc(ROOT/'request.json')))
def lane_run(lane):
 q,runner,est,cat,cached=context();check_host(q);no_failures();assert grant_valid(q)
 os.sched_setaffinity(0,[80+lane*2,81+lane*2]);gpu=q['host']['gpus'][lane]
 os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid'];sys.path.insert(0,str(E/'ita_gpu_exact_20261029_v1/runtime_deps'))
 import ksg_stability_cuda_v2 as backend
 gpu_validate(lane,backend,est,q)
 while not all((ROOT/'validation'/('lane%d.json'%j)).exists() for j in range(3)):no_failures();time.sleep(2)
 for j in range(3):assert checked(read(ROOT/'validation'/('lane%d.json'%j)))['passed']
 prep=checked(read(ROOT/'preparation_complete.json'));assert prep['N']==1000
 tasks=[t for t in q['tasks'] if t['lane']==lane]
 for number,t in enumerate(tasks):
  no_failures();check_host(q);key=task_key(t);start=time.monotonic()
  save(ROOT/'claims'/(key+'.json'),dict(time=now(),task=t,identity=identity(),request=desc(ROOT/'request.json')))
  save(ROOT/'progress'/('lane%d.json'%lane),dict(time=now(),phase='estimating',task=key,done=number,total=len(tasks)),True)
  pp=[ROOT/'pools'/pool_name(t['factor'],t['seed'],t['domain'],m) for m in (t['x'],t['y'])]
  values=[]
  for p in pp:
   verify(prep['pools'][p.name]);a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(1000,1000,3)
   values.append(np.ascontiguousarray(a[:,:t['budget'],:].reshape(-1,3)));del a
  x,y=values;obs=1000*t['budget'];rng=np.random.default_rng(20261031+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
  ids=np.sort(rng.choice(obs,min(512,obs),replace=False))
  result,diagnostic=backend.estimate_ksg_gpu(x,y,est,k=5,workers=2,validation_query_indices=ids)
  assert result['n']==obs and diagnostic['cpu_validation']['passed'] and diagnostic['cuda_executed']
  pci=lambda s:tuple(int(z,16) for z in s.lower().replace('.',':').split(':'))
  assert pci(diagnostic['runtime']['device_pci_bus_id'])==pci(gpu['pci'])
  elapsed=time.monotonic()-start
  save(ROOT/'results'/(key+'.json'),dict(time=now(),task=t,status='complete',groups=1000,observations=obs,result=result,diagnostic=diagnostic,x_pool=desc(pp[0]),y_pool=desc(pp[1]),GPU=gpu,seconds=elapsed,identity=identity(),request=desc(ROOT/'request.json'),cohort=prep['cohort'],reused=False))
  print('MI_RESULT',key,result['mi_bits'],'seconds',elapsed,flush=True);del x,y,values;gc.collect()
 save(ROOT/'completion'/('lane%d.json'%lane),dict(time=now(),status='complete',slots=len(tasks),identity=identity(),request=desc(ROOT/'request.json')))
def summarize(q):
 records={}
 for p in (ROOT/'results').glob('*.json'):
  v=checked(read(p));assert v['request']==desc(ROOT/'request.json') and v['groups']==1000 and v['reused'] is False
  records[task_key(v['task'])]=v
 assert len(records)==864 and set(records)=={task_key(t) for t in q['tasks']}
 rows=[]
 for n in BUDGETS:
  for x,y in PAIRS:
   for d in DOMAINS:
    for f in FACTORS:
     rr=[records[task_key(dict(x=x,y=y,domain=d,factor=f,budget=n,seed=s))] for s in SEEDS];v=[r['result']['mi_bits'] for r in rr]
     mean=math.fsum(v)/3;sd=math.sqrt(math.fsum((a-mean)**2 for a in v)/2)
     rows.append(dict(budget=n,x=x,y=y,domain=d,factor=f,N_groups=1000,pooled_observations=1000*n,mean_bits=mean,sampling_sd_bits_ddof1=sd,draws=v))
 assert len(rows)==288
 save(ROOT/'summary.json',dict(time=now(),status='complete',cells=rows,raw_records=864,new_CUDA=864,cohort=desc(ROOT/'cohort.json'),request=desc(ROOT/'request.json'),SD_is_three_sampling_draws_not_patient_CI=True))
 with (ROOT/'summary.csv').open('x',newline='',encoding='utf8') as stream:
  fields=[k for k in rows[0] if k!='draws'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
 sys.path.insert(0,str(E/'ksg_stability_20261030_v1/runtime_plot'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 figures={}
 for n in BUDGETS:
  fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
  for ax,(x,y) in zip(axes.ravel(),PAIRS):
   for d in ('P','D','O','F'):
    rr=[r for r in rows if r['budget']==n and r['x']==x and r['y']==y and r['domain']==d]
    ax.errorbar(range(4),[r['mean_bits'] for r in rr],yerr=[r['sampling_sd_bits_ddof1'] for r in rr],marker='o',label=d)
   ax.set_title('%s–%s'%(x.upper(),y.upper()));ax.set_xticks(range(4));ax.set_xticklabels(['1x','2x','4x','8x']);ax.set_xlabel('Direct native downsampling');ax.set_ylabel('MI (bits; mean ± sampling SD)');ax.legend(frameon=False);ax.grid(alpha=.15)
  fig.suptitle('Same 1,000 groups; %d points per group; true-UV-defined regions'%n)
  for ext in ('png','pdf'):
   p=ROOT/('resolution_MI_n%d.%s'%(n,ext));fig.savefig(p,dpi=220);figures[p.name]=desc(p)
  plt.close(fig)
 for d in q['bindings'].values():verify(d)
 for d in q['sources'].values():verify(d)
 save(ROOT/'scientific_completion.json',dict(time=now(),status='complete',raw=864,cells=288,summary=desc(ROOT/'summary.json'),CSV=desc(ROOT/'summary.csv'),figures=figures,request=desc(ROOT/'request.json'),independent_final_audit_still_required=True))
 state(phase='scientific_complete_pending_independent_acceptance',raw_done=864,summary_cells=288)
def acceptance_and_return(q):
 p=subprocess.run([str(PY),'-B','-u',str(ROOT/'source/audit.py')],check=False)
 assert p.returncode==0,'Independent saved-artifact audit failed; no resource return'
 audit=checked(read(ROOT/'independent_scientific_acceptance.json'));assert audit['passed'] and audit['raw_records']==864 and audit['summary_cells']==288
 observations=[]
 for _ in range(2):
  check_host(q);no_failures();assert not gpu_processes(),'GPU resource not actually released'
  observations.append(dict(time=now(),inventory=inventory(),compute=[]));time.sleep(5)
 save(ROOT/'GPU_resource_return_acceptance.json',dict(time=now(),passed=True,observations=observations,scientific_acceptance=desc(ROOT/'independent_scientific_acceptance.json'),request=desc(ROOT/'request.json')))
 writer=module(verify(q['bindings']['return_marker_writer']),'common1000_original_return_writer')
 target=Path(q['bindings']['main_resource_grant']['path']).with_name('side_MI_returned.json')
 assert target==E/'pl_controls_original_uv_pipeline_20261031_v1/resource_grants/side_MI_returned.json'
 writer.save(target,dict(time=now(),grant=q['bindings']['main_resource_grant'],scientific_completion=desc(ROOT/'independent_scientific_acceptance.json'),machine_completion=desc(ROOT/'scientific_completion.json'),actual_resource_release=desc(ROOT/'GPU_resource_return_acceptance.json'),original_sources_unchanged=True))
 save(ROOT/'resource_return_receipt.json',dict(time=now(),marker=desc(target),request=desc(ROOT/'request.json')))
def run():
 q,*_=context();check_host(q);no_failures();os.sched_setaffinity(0,[62,63]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
 save(ROOT/'active.json',dict(time=now(),identity=identity(),request=desc(ROOT/'request.json')))
 state(phase='CPU_validation');cpu_gate();import prepare
 selected=prepare.screen()
 if len(selected)!=1000:
  state(phase='catalogue_exhausted_shortfall',actual_N=len(selected),required_N=1000,scientific_complete=False)
  save(ROOT/'supervisor_completion.json',dict(time=now(),status='incomplete_catalogue_shortfall',cohort=desc(ROOT/'cohort.json'),scientific_complete=False));return
 prepare.features(selected);prepare.pools(selected);os.sched_setaffinity(0,[62,63]);wait_resources(q);children=[]
 for lane in range(3):
  save(ROOT/'owners'/('lane%d_reservation.json'%lane),dict(time=now(),lane=lane,request=desc(ROOT/'request.json')))
  with (ROOT/'owners'/('lane%d.log'%lane)).open('xb') as log:
   p=subprocess.Popen([str(PY),'-B','-u',__file__,'lane','--lane',str(lane)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  save(ROOT/'owners'/('lane%d_launch.json'%lane),dict(time=now(),lane=lane,identity=identity(p.pid),request=desc(ROOT/'request.json')));children.append(p)
 while any(p.poll() is None for p in children):
  no_failures();assert all(p.poll() in (None,0) for p in children),'GPU child failed'
  state(phase='exact_GPU_KSG',raw_done=len(list((ROOT/'results').glob('*.json'))),total_raw=864);time.sleep(10)
 assert all(p.returncode==0 for p in children);summarize(q);acceptance_and_return(q)
 save(ROOT/'supervisor_completion.json',dict(time=now(),status='complete',scientific_completion=desc(ROOT/'scientific_completion.json'),request=desc(ROOT/'request.json'),identity=identity()))
def launch():
 q,*_=context();check_host(q);no_failures();assert not (ROOT/'launch_reservation.json').exists(),'Unique launch already reserved'
 assert checked(read(ROOT/'validation/cpu_tests.json'))['passed']
 for d in ('validation','screening','masks','features','pools','owners','claims','progress','results','completion','failures'):(ROOT/d).mkdir(exist_ok=True)
 save(ROOT/'launch_reservation.json',dict(time=now(),identity=identity(),request=desc(ROOT/'request.json')))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 with (ROOT/'owner.log').open('xb') as log:
  p=subprocess.Popen([str(PY),'-B','-u',__file__,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=env)
 time.sleep(.4);assert p.poll() is None
 save(ROOT/'launch_receipt.json',dict(time=now(),identity=identity(p.pid),request=desc(ROOT/'request.json'),source=desc(__file__),detached=True,stdin_devnull=True))
 print('LAUNCHED',p.pid,flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['configure','tests','launch','run','lane']);parser.add_argument('--lane',type=int);a=parser.parse_args()
 try:
  if a.action=='lane':lane_run(a.lane)
  else:globals()[a.action]()
 except BaseException:
  if a.action not in ('configure','tests','launch'):fail(a.action)
  raise
if __name__=='__main__':main()
