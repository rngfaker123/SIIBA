"""TEST group-count sensitivity. Frozen1000pixels/k5, CPU-only, old outputs read-only."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util,statistics,traceback,subprocess,csv,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
B=Path('/study/siiba/experiments');R=B/'ksg_group_fraction_20261102_v1'
EST=B/'mi_atlas_six_20261025_v1/source/tools/mi_knn_estimator_v1.py'
RUN=B/'mi_atlas_six_20261025_v1/source/tools/mi_atlas_six_metrics_20261025.py'
ESTSHA='bb6f7fbf704a3baff582a712fefbaa6bb6d13c9d962ff4fa108347db05a19b0d';RUNSHA='7ac6c09e9eb81cc2f91bd94e830222ccc2b7b0b5ba8a2da33c2f262c217167f9'
METRICS={'P_rgb_uv':('P','uv'),'D_rgb_uv':('D','uv'),'O_rgb_uv':('O','uv'),'full_rgb_uv':('F','uv'),'full_rgb_plh':('F','plh'),'full_rgb_plv':('F','plv')}
SEEDS=[20261102,20261103,20261104];PCTS=[10,30,50,70,100];POOLS={};SELECTIONS={};REQ=None;AUD=None;runner=None;est=None
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def seal(v):return dict(v,record_sha256=hashlib.sha256(encoded(v)).hexdigest())
def check(v):
 z=dict(v);h=z.pop('record_sha256');assert hashlib.sha256(encoded(z)).hexdigest()==h;return v
def put(p,v,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=encoded(seal(v));t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 try:
  if mutable:os.replace(t,p)
  else:os.link(t,p)
 finally:
  if t.exists():t.unlink()
def desc(p):return dict(path=str(p),sha256=sha(p))
def state(phase,**x):put(R/'status.json',dict(time=now(),phase=phase,**x),mutable=True)
def module(p,name):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def selections(n):
 out={}
 for seed in SEEDS:
  perm=np.random.Generator(np.random.PCG64(seed)).permutation(n)
  for pct in PCTS:
   size=n if pct==100 else n*pct//100;ids=np.sort(perm[:size]);assert len(ids)==len(np.unique(ids))==size
   out[seed,pct]=ids
 return out
def identity(pid=None):
 pid=pid or os.getpid();s=(Path('/proc')/str(pid)/'stat').read_text();a=s[s.rfind(')')+2:].split()
 return dict(pid=pid,starttick=int(a[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]),cpus=list(os.sched_getaffinity(pid)))
def unchanged_training():
 expected={350:285433677,355:285433692,360:285433697,366:285433702,396:285433938,397:285433940,401:285433952,402:285433954,788:285438327}
 out=[]
 for pid,birth in expected.items():
  try:v=identity(pid);v['same_original_birth']=v['starttick']==birth
  except FileNotFoundError:v=dict(pid=pid,absent=True)
  out.append(v)
 return out
def cache_row(v):
 p=v['prepared'];row=p['row'];root=Path(p['out']);fr=p['feature'];a=root/fr['feature_path'];m=a.with_suffix('.json')
 assert sha(a)==fr['feature_sha256'] and sha(m)==fr['metadata_sha256'];meta=runner.check_seal(read(m))
 assert meta['source_row_sha256']==runner.digest(row) and meta['feature_sha256']==fr['feature_sha256']
 assert meta['sample_id']==v['sample_id'] and all(meta['mask_pixels'][d]==v['counts'][d] for d in ['P','D','O','F'])
 return row,runner.validate_features(runner.load_npz(a),row),dict(feature=desc(a),metadata=desc(m),original_row_sha256=runner.digest(row))
def prepare_extra():
 extra=[v for v in AUD['eligible'] if 'prepared' not in v]
 if not extra:return {}
 old=read(Path(next(iter(AUD['source_manifests'].values()))['manifest_path']));rows=[];proofs=[]
 for v in extra:
  row=dict(v['native_without_ITA']['catalog_row']);refp=B/'three_region_test_parallel_20261025_v1/masks/refs'/('%06d.json'%v['index']);ref=check(read(refp));retp=Path(ref['native_retention']['path']);assert sha(retp)==ref['native_retention']['sha256'];ret=check(read(retp));assert ret['counts_native']==v['counts']
  for mod in ['rgb','uv','plh','plv']:
   row[mod]=dict(row[mod]);p=Path(row[mod]['path']);row[mod]['path']=str(p if p.is_absolute() else Path('/study/images')/p)
  for domain in ['P','D','O','F']:
   row[domain]=ret['native_masks'][domain];assert sha(row[domain]['path'])==row[domain]['sha256']
  rows.append(row);proofs.append(dict(index=v['index'],reference=desc(refp),retention=desc(retp)))
 manifest=dict(old,cohort_id='TEST_native_MI_eligible_without_ITA_filter_6_extra',expected_sample_count=len(rows),rows=rows)
 mpth=R/'extra_native/mi_input.json';mpth.parent.mkdir();mpth.write_bytes(runner.encoded(manifest));runner.prepare(mpth,R/'extra_native/mi_run')
 prep=runner.check_seal(read(R/'extra_native/mi_run/preparation.json'));out={}
 for i,(v,row,fr) in enumerate(zip(extra,rows,prep['features'])):out[v['sample_id']]=dict(row=row,feature=fr,out=str(R/'extra_native/mi_run'),stratum='not_used')
 put(R/'extra_native/provenance.json',dict(new_detector=False,new_image_samples_only=len(rows),native_old_masks_reused=True,proofs=proofs,manifest=desc(mpth),preparation=desc(R/'extra_native/mi_run/preparation.json')))
 return out
def validate_sources():
 assert sha(EST)==ESTSHA and sha(RUN)==RUNSHA
 assert sha(R/'protocol.json')==REQ['protocol']['sha256']
 for name,d in REQ['new_sources'].items():assert sha(d['path'])==d['sha256'],name
def prepared_group(arg):
 i,v=arg;row,a,proof=cache_row(v);values={}
 for r in ['P','D','O','F']:
  ri=runner.DOMAINS.index(r);coords=a['linear_indices'][0,ri]
  for mod in (['rgb','uv','plh','plv'] if r=='F' else ['rgb','uv']):
   raw=a['full_values'][0,runner.MODS.index(mod)] if r=='F' else a['region_values'][0,ri-1,runner.MODS.index(mod)]
   values[r,mod]=runner.dequantize_native(raw,coords,v['sample_id'],row['native_shape'][1],mod)
 return i,values,dict(index=v['index'],sample_id=v['sample_id'],**proof)
def prep_worker_init():
 k=(mp.current_process()._identity[0]-1)%6;os.sched_setaffinity(0,set(range(96+k*4,100+k*4)))
def create_pools():
 n=AUD['eligible_N'];extra=prepare_extra();domains=['P','D','O','F'];proofs=[];args=[]
 for r in domains:
  for mod in (['rgb','uv','plh','plv'] if r=='F' else ['rgb','uv']):POOLS[r,mod]=np.empty((n,1000,3),np.float64)
 for i,v0 in enumerate(AUD['eligible']):
  v=dict(v0)
  if 'prepared' not in v:v['prepared']=extra[v['sample_id']]
  args.append((i,v))
 with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('fork'),initializer=prep_worker_init) as pool:
  for completed,(i,values,proof) in enumerate(pool.map(prepared_group,args,chunksize=8),1):
   proofs.append(proof)
   for key,value in values.items():POOLS[key][i]=value
   if completed%250==0:state('preparing_fixed1000pixel_pools',prepared_groups=completed,total_groups=n)
 for a in POOLS.values():assert a.shape==(n,1000,3) and np.isfinite(a).all();a.flags.writeable=False
 put(R/'prepared_input_receipt.json',dict(time=now(),cohort=desc(R/'cohort_audit.json'),N=n,pixel_seed=20261025,noise_seed=20260817,new_native_groups=len(extra),group_features=proofs,pools=[dict(region=r,modality=m,shape=list(a.shape),dtype=str(a.dtype),sha256=ah(a)) for (r,m),a in POOLS.items()],all_pools_memory_only=True))
def job(task):
 metric,seed,pct=task;domain,target=METRICS[metric];ids=SELECTIONS[seed,pct];key=metric+'_groups'+str(pct)+'_seed'+str(seed);r=R/'raw'/(key+'.json')
 assert not r.exists();put(R/'claims'/(key+'.json'),dict(time=now(),task=task,identity=identity(),request=desc(R/'request.json')))
 x=np.ascontiguousarray(POOLS[domain,'rgb'][ids].reshape(-1,3));y=np.ascontiguousarray(POOLS[domain,target][ids].reshape(-1,3));t=time.monotonic()
 put(R/'progress'/(key+'.json'),dict(time=now(),phase='original_CPU_KSG',task=task,N_groups=len(ids),observations=len(x),identity=identity()),mutable=True)
 result=est.ksg_mi(x,y,k=5,workers=8);assert result['n']==len(ids)*1000 and result['k']==5
 value=dict(time=now(),metric=metric,group_seed=seed,percent=pct,N_groups=len(ids),points_per_group=1000,observations=len(x),pixel_seed=20261025,selection=desc(R/'selections'/('%d_%d.json'%(seed,pct))),input_x_sha256=ah(x),input_y_sha256=ah(y),result=result,elapsed_seconds=time.monotonic()-t,identity=identity(),workers=8,device='CPU',new_training=False,request=desc(R/'request.json'),reused=False)
 put(r,value);put(R/'progress'/(key+'.json'),dict(time=now(),phase='complete',result=desc(r),elapsed_seconds=value['elapsed_seconds']),mutable=True)
 return value
def worker_init():
 # 每路独占八核
 k=(mp.current_process()._identity[0]-1)%3;os.sched_setaffinity(0,set(range(96+k*8,104+k*8)));os.nice(19) if os.getpriority(os.PRIO_PROCESS,0)<19 else None
def pilot():
 from scipy.spatial import cKDTree
 from scipy.special import digamma
 rng=np.random.default_rng(20261102);x=rng.normal(size=(257,3));y=.7*x+rng.normal(size=x.shape)*.3
 a=est.ksg_mi(x,y,k=5,workers=1);dist=cKDTree(np.c_[x,y]).query(np.c_[x,y],k=7,p=np.inf,workers=1)[0];rad=np.nextafter(dist[:,5],0);nx=cKDTree(x).query_ball_point(x,rad,p=np.inf,return_length=True);ny=cKDTree(y).query_ball_point(y,rad,p=np.inf,return_length=True);val=(digamma(5)+digamma(len(x))-np.mean(digamma(nx)+digamma(ny)))/np.log(2);assert a['mi_bits']==val
 for n in [17,1000,19651]:
  z=selections(n)
  for seed in SEEDS:
   assert all(set(z[seed,a]).issubset(set(z[seed,b])) for a,b in zip(PCTS,PCTS[1:]));assert np.array_equal(z[seed,100],np.arange(n))
 assert all(np.array_equal(SELECTIONS[SEEDS[0],100],SELECTIONS[s,100]) for s in SEEDS)
 real=[]
 for metric,(domain,target) in METRICS.items():
  ids=SELECTIONS[SEEDS[0],10][:100];x=POOLS[domain,'rgb'][ids].reshape(-1,3);y=POOLS[domain,target][ids].reshape(-1,3);t=time.monotonic();a=est.ksg_mi(x,y,k=5,workers=2);real.append(dict(metric=metric,N_groups=len(ids),observations=len(x),mi_bits=a['mi_bits'],elapsed_seconds=time.monotonic()-t))
 put(R/'prospective_validation.json',dict(time=now(),passed=True,independent_formula_exact=True,source_hashes_passed=True,group_sampling_nested_unique_complete=True,fixed_pixel_draw=20261025,full_fraction_same_groups_verified=True,real_CPU_pilots=real,identity=identity()))
def summarize():
 for metric in METRICS:
  first=R/'raw'/(metric+'_groups100_seed'+str(SEEDS[0])+'.json');a=check(read(first))
  for seed in SEEDS[1:]:
   p=R/'raw'/(metric+'_groups100_seed'+str(seed)+'.json');v={k:z for k,z in a.items() if k!='record_sha256'};v.update(group_seed=seed,reused=True,reused_numeric_estimate=desc(first),selection=desc(R/'selections'/('%d_100.json'%seed)),elapsed_seconds=0.0);put(p,v)
 cells=[];raw=[]
 for metric in METRICS:
  for pct in PCTS:
   vv=[check(read(R/'raw'/(metric+'_groups'+str(pct)+'_seed'+str(seed)+'.json'))) for seed in SEEDS];values=[v['result']['mi_bits'] for v in vv]
   assert all(v['N_groups']==len(SELECTIONS[v['group_seed'],pct]) and v['points_per_group']==1000 for v in vv)
   sd=statistics.stdev(values);assert np.isclose(sd,np.std(values,ddof=1),rtol=1e-12,atol=1e-15)
   if pct==100:assert sd==0 and len({(v['input_x_sha256'],v['input_y_sha256']) for v in vv})==1
   cells.append(dict(metric=metric,percent=pct,N_groups=vv[0]['N_groups'],pooled_observations=vv[0]['observations'],k=5,points_per_group=1000,mean_bits=statistics.mean(values),group_sampling_SD_ddof1=sd,draw1=values[0],draw2=values[1],draw3=values[2]));raw.extend(vv)
 assert len(cells)==30 and len(raw)==90
 put(R/'summary.json',dict(time=now(),status='complete',cells=cells,group_sampling_SD_only=True,full_fraction_SD_zero_by_design=True,request=desc(R/'request.json'),source_audit=desc(R/'cohort_audit.json')))
 with (R/'summary.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
 validate_sources();put(R/'independent_acceptance.json',dict(time=now(),status='passed',raw_bindings=90,unique_estimates=78,mean_sampleSD_cells=30,original_sources_unchanged=True,no_extra_ITA_filter=True,all_group_selections_saved_before_estimates=True,summary=desc(R/'summary.json'),prepared_inputs=desc(R/'prepared_input_receipt.json'),training_observation=unchanged_training()))
 put(R/'completion.json',dict(time=now(),status='complete',summary=desc(R/'summary.json'),acceptance=desc(R/'independent_acceptance.json'),request=desc(R/'request.json')));state('complete',cells=30,raw_bindings=90)
def main():
 global REQ,AUD,runner,est,SELECTIONS
 REQ=check(read(R/'request.json'));AUD=read(R/'cohort_audit.json');validate_sources();runner=module(RUN,'original_group_sensitivity_runner');est=module(EST,'original_group_sensitivity_estimator')
 assert sha(R/'cohort_audit.json')==REQ['cohort']['sha256'];assert AUD['eligible_N']==19651 and len(AUD['eligible'])==19651
 assert len({v['sample_id'] for v in AUD['eligible']})==len({v['subject_id'] for v in AUD['eligible']})==19651
 state('freeze_group_selections',N_groups=AUD['eligible_N']);SELECTIONS=selections(AUD['eligible_N'])
 for (seed,pct),ids in SELECTIONS.items():put(R/'selections'/('%d_%d.json'%(seed,pct)),dict(group_seed=seed,percent=pct,positions=ids.tolist(),catalogue_indices=[AUD['eligible'][i]['index'] for i in ids],sample_ids=[AUD['eligible'][i]['sample_id'] for i in ids],points_per_group=1000,pixel_seed=20261025,cohort=REQ['cohort']))
 state('preparing_fixed1000pixel_pools',total_groups=AUD['eligible_N']);create_pools();state('prospective_validation');pilot();state('production',unique_tasks=78,lanes=3,CPU_threads_per_lane=8)
 tasks=[(metric,seed,pct) for pct in PCTS for metric in METRICS for seed in (SEEDS[:1] if pct==100 else SEEDS)]
 t=time.monotonic();done=0
 with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('fork'),initializer=worker_init) as pool:
  futures=[pool.submit(job,task) for task in tasks]
  try:
   for future in as_completed(futures):
    v=future.result();done+=1;state('production',done=done,total=78,last_metric=v['metric'],last_percent=v['percent'],last_seconds=v['elapsed_seconds'],elapsed_seconds=time.monotonic()-t);print('MI_RESULT',v['metric'],v['percent'],v['group_seed'],v['result']['mi_bits'],v['elapsed_seconds'],flush=True)
  except BaseException:
   put(R/'failure.json',dict(time=now(),identity=identity(),traceback=traceback.format_exc()))
   for f in futures:f.cancel()
   for p in pool._processes.values():p.terminate()
   raise
 state('independent_final_checks');summarize()
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (R/'failure.json').exists():put(R/'failure.json',dict(time=now(),identity=identity(),traceback=traceback.format_exc()))
  state('failed_preserved');raise
