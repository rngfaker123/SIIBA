"""Full eligible TEST MI; same pixel-draw SD in Table1A and both sensitivities.

All output is confined to this new directory. Original results are read-only.
CLI: configure, pilot, launch, run TOKEN, audit. Never retry a reserved launch.
"""
from pathlib import Path
import os, sys, json, hashlib, datetime, time, statistics, traceback, secrets
import importlib.util, subprocess, multiprocessing as mp, gc, math, io, csv
from concurrent.futures import ProcessPoolExecutor
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

R=Path(__file__).resolve().parent
B=Path('/study/siiba/experiments')
G=B/'ksg_group_fraction_20261102_v1'
PL=B/'mi_rgb_pl_regions_20261104_v1'
SEEDS=(20261025,20261026,20261027)
DOMAINS=('P','D','O','F')
TARGETS=('uv','plh','plv')
FRACTIONS=(10,30,50,70,100)
KS=(1,3,5,10,20)
POINTS=(100,500,1000)
N=19651
POOL={}; ROWS=[]; PROOFS=[]; SELECT={}; CTX={}

def require(condition, message):
    if not condition: raise ValueError(message)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def enc(x): return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def digest(x): return hashlib.sha256(enc(x)).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def ah(x): return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()
def desc(p): return dict(path=str(p),sha256=sha(p))
def verify(d):
    p=Path(d['path']); require(sha(p)==d['sha256'],'Source hash differs: '+str(p)); return p
def seal(v): return dict(v,record_sha256=digest(v))
def checked(v):
    w=dict(v); old=w.pop('record_sha256'); require(digest(w)==old,'Record seal differs'); return v
def publish(p,data,mutable=False):
    p=Path(p).resolve(); require(R.resolve() in p.parents,'Output outside owned directory')
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp.'+secrets.token_hex(8))
    with temp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    try:
        if mutable: os.replace(temp,p)
        else: os.link(temp,p)
    finally:
        if temp.exists(): temp.unlink()
def put(p,v,mutable=False): publish(p,enc(seal(v)),mutable)
def status(phase,**kw): put(R/'status.json',dict(time=now(),phase=phase,**kw),True)
def fail(label): put(R/'failures'/(label+'_'+str(os.getpid())+'_'+str(time.time_ns())+'.json'),dict(time=now(),traceback=traceback.format_exc()))
def no_failure(): require(not list((R/'failures').glob('*.json')),'Failure evidence preserved; no automatic retry')
def identity(pid=None):
    pid=pid or os.getpid(); q=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttick=int(q[19]),ppid=int(q[1]),cpus=sorted(os.sched_getaffinity(pid)),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]))
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,str(p)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def metric(d,m): return ('full' if d=='F' else d)+'_rgb_'+m
def key(t): return '%s_p%d_n%d_k%d_s%d'%(metric(t['domain'],t['target']),t['percent'],t['points'],t['k'],t['seed'])
def task(d,m,p,n,k,s): return dict(domain=d,target=m,percent=p,points=n,k=k,seed=s,N_groups=N*p//100)
def tasks():
    # 主表、组比例、近邻数
    out=[]
    for n in (1000,100,500):
        for s in SEEDS:
            for m in TARGETS:
                for d in DOMAINS: out.append(task(d,m,100,n,5,s))
    for p in (10,30,50,70):
        for s in SEEDS:
            for m in TARGETS:
                for d in DOMAINS: out.append(task(d,m,p,1000,5,s))
    for k in (1,3,10,20):
        for s in SEEDS:
            for m in TARGETS:
                for d in DOMAINS: out.append(task(d,m,100,1000,k,s))
    require(len(out)==396 and len({key(t) for t in out})==396,'Task plan duplicates')
    return out
def selections(n=N):
    perm=np.random.Generator(np.random.PCG64(20261102)).permutation(n)
    return {p:np.sort(perm[:n*p//100]) for p in FRACTIONS}
def cpu_list(v):
    out=set()
    for chunk in v.strip().split(','):
        x=chunk.split('-'); out.update(range(int(x[0]),int(x[-1])+1))
    return out
def resource_check():
    r=CTX['resources']; ident=identity()
    require(all(ident[k]==r['host'][k] for k in ('boot','pid1_starttick')),'Host/container identity differs')
    allocated=[c for lane in r['lanes'] for c in lane]+r['owner_cpus']
    require(len(allocated)==len(set(allocated)),'CPU allocations overlap')
    # 读取在线核数
    online=cpu_list(Path('/sys/devices/system/cpu/online').read_text())
    require(set(allocated)<=online,'Allocated CPUs unavailable')
    for c in allocated:
        siblings=cpu_list(Path('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list'%c).read_text())
        require(not siblings.intersection(r['protected_cpus']),'Physical CPU overlaps protected workload')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available>r['reserve_free_gib']*2**30,'Insufficient free RAM reserve')
    limit=Path('/sys/fs/cgroup/memory.max')
    if limit.exists() and limit.read_text().strip()!='max':
        require(int(limit.read_text())>(r['memory_budget_gib']+r['reserve_free_gib'])*2**30,'Container RAM budget too small')
    return ident
def load():
    p=read(R/'protocol.json'); b=read(R/'baseline_pins.json'); r=read(R/'resources.json')
    for item in b['files']: verify(item)
    run=module(verify(p['runner']),'new_fulltest_original_runner')
    est=module(verify(p['estimator']),'new_fulltest_original_estimator')
    require(sha(Path(sys.modules[run.ksg_mi.__module__].__file__))==p['estimator']['sha256'],'Runner estimator import differs')
    require(run.SEEDS==SEEDS and run.MODS==('rgb','uv','plv','plh') and run.DOMAINS==('F','P','D','O'),'Original constants differ')
    require(run.NOISE_SEED==20260817 and run.K==5 and run.MAX_POINTS==1000,'Original sampling constants differ')
    helper=module(verify(p['PL_helper']),'new_fulltest_readonly_PL_helper'); helper.CTX['runner']=run
    def no_old_write(*args,**kwargs): raise RuntimeError('Old experiment mutation disabled')
    helper.publish_bytes=no_old_write; helper.put=no_old_write
    CTX.update(protocol=p,baseline=b,resources=r,runner=run,est=est,helper=helper)
    require(np.__version__=='2.3.4' and est.scipy.__version__=='1.16.2','Original numerical runtime differs')
    if (R/'request.json').exists():
        req=checked(read(R/'request.json'))
        for d in req['frozen_files']: verify(d)
        CTX['request']=req
    return p
def load_rows():
    global ROWS, PROOFS, SELECT
    aud=read(G/'cohort_audit.json'); prep=checked(read(G/'prepared_input_receipt.json'))
    require(aud['total']==21899 and aud['eligible_N']==N and prep['N']==N,'Full eligible cohort differs')
    proof={x['sample_id']:x for x in prep['group_features']}; extra=read(G/'extra_native/mi_input.json')
    extra_rows={x['sample_id']:x for x in extra['rows']}
    ROWS=[]; PROOFS=[]
    for e in aud['eligible']:
        row=e['prepared']['row'] if 'prepared' in e else extra_rows[e['sample_id']]
        v=proof[e['sample_id']]
        require(CTX['runner'].digest(row)==v['original_row_sha256'],'Original row binding differs')
        ROWS.append(row); PROOFS.append(v)
    require(len(ROWS)==len(set(x['sample_id'] for x in ROWS))==N,'Cohort duplicate or missing group')
    SELECT=selections()
    for p,ids in SELECT.items():
        old=checked(read(G/'selections'/('20261102_%d.json'%p)))
        # 独立核对原编号
        CTX.setdefault('selection_records',{})[p]=old
    return aud
def original(i):
    row=ROWS[i]; proof=PROOFS[i]; run=CTX['runner']; p=verify(proof['feature']); meta=run.check_seal(read(verify(proof['metadata'])))
    require(meta['feature_sha256']==proof['feature']['sha256'] and meta['source_row_sha256']==run.digest(row),'Feature provenance mismatch')
    require(meta['sample_id']==row['sample_id'] and meta['partition_verified'] and meta['source_hashes_verified'],'Original feature verification absent')
    require(all(meta['mask_pixels'][d]>=1000 for d in run.DOMAINS),'Original eligibility changed')
    a=run.validate_features(run.load_npz(p),row)
    require(ah(a['linear_indices'])==meta['linear_indices_sha256'],'Original coordinate hash differs')
    return row,a,meta
def pl_values(row,a):
    helper=CTX['helper']; token=helper.cache_key(row,a); p=PL/'sampled_PL_cache'/(token+'.npy'); q=p.with_suffix('.json')
    if q.exists():
        v=checked(read(q)); require(v['cache_key']==token and v['request']==desc(PL/'request.json'),'PL cache binding differs')
        require(v['source_PL']=={m:row[m] for m in ('plh','plv')} and v['linear_indices_sha256']==ah(a['linear_indices']),'PL cache source/coordinate differs')
        require(v['all_saved_F_values_bit_exact'] is True,'Missing F-value equality evidence')
        verify(v['array']); values=np.load(p,allow_pickle=False); proof=dict(array=desc(p),receipt=desc(q),reused=True)
    else:
        require(not p.exists(),'Incomplete historical PL cache')
        p=R/'extra_PL_cache'/(token+'.npy'); q=p.with_suffix('.json')
        if q.exists():
            v=checked(read(q)); verify(v['array']); require(v['cache_key']==token,'Own PL cache differs'); values=np.load(p,allow_pickle=False)
        else:
            require(not p.exists(),'Unsealed own PL cache')
            values=helper.extract_pl_arrays(row,a,B,CTX['runner'].read_bound_array)
            stream=io.BytesIO(); np.save(stream,values,allow_pickle=False); publish(p,stream.getvalue())
            put(q,dict(cache_key=token,sample_id=row['sample_id'],array=desc(p),source_PL={m:row[m] for m in ('plh','plv')},linear_indices_sha256=ah(a['linear_indices']),all_saved_F_values_bit_exact=True))
        proof=dict(array=desc(p),receipt=desc(q),reused=False)
    require(values.dtype==np.uint8 and values.shape==(3,3,2,1000,3),'PL cache layout differs')
    return values,proof
def prepare_one(i):
    row,a,meta=original(i); pl,plproof=pl_values(row,a); run=CTX['runner']; data=np.empty((3,4,4,1000,3),np.float64)
    for si,s in enumerate(SEEDS):
        for di,d in enumerate(('F','P','D','O')):
            coords=a['linear_indices'][si,di]
            for mi,m in enumerate(('rgb','uv','plh','plv')):
                if d=='F': raw=a['full_values'][si,run.MODS.index(m)]
                elif m in ('rgb','uv'): raw=a['region_values'][si,di-1,run.MODS.index(m)]
                else: raw=pl[si,di-1,('plh','plv').index(m)]
                data[si,di,mi]=run.dequantize_native(raw,coords,row['sample_id'],row['native_shape'][1],m)
    require(np.isfinite(data).all(),'Nonfinite pool data')
    return i,data,dict(sample_id=row['sample_id'],original=PROOFS[i],coordinates_sha256=meta['linear_indices_sha256'],PL=plproof)
def prep_init(counter):
    with counter.get_lock(): lane=counter.value; counter.value+=1
    os.sched_setaffinity(0,set(CTX['resources']['lanes'][lane]))
def build_pools(positions=None,save=True):
    global POOL
    positions=list(range(N)) if positions is None else list(positions); n=len(positions)
    resource_check()
    if save:
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        require(available>(CTX['resources']['memory_budget_gib']+CTX['resources']['reserve_free_gib'])*2**30,'Production RAM allocation unavailable')
    POOL={(s,d,m):np.empty((n,1000,3),np.float64) for s in SEEDS for d in ('F','P','D','O') for m in ('rgb','uv','plh','plv')}
    ctx=mp.get_context('fork'); counter=ctx.Value('i',0); proofs=[]; started=time.monotonic()
    with ProcessPoolExecutor(max_workers=len(CTX['resources']['lanes']),mp_context=ctx,initializer=prep_init,initargs=(counter,)) as ex:
        for start in range(0,n,12):
            futures=[ex.submit(prepare_one,i) for i in positions[start:start+12]]
            for offset,f in enumerate(futures):
                i,values,proof=f.result(); require(i==positions[start+offset],'Preparation ordering differs')
                for si,s in enumerate(SEEDS):
                    for di,d in enumerate(('F','P','D','O')):
                        for mi,m in enumerate(('rgb','uv','plh','plv')): POOL[s,d,m][start+offset]=values[si,di,mi]
                proofs.append(proof); del values
            if save and (start%120==0 or start+12>=n): status('preparing_pools',groups=min(start+12,n),total=n,elapsed_seconds=time.monotonic()-started)
    pools=[]
    for (s,d,m),a in POOL.items():
        a.flags.writeable=False; pools.append(dict(seed=s,domain=d,modality=m,shape=list(a.shape),sha256=ah(a)))
    result=dict(time=now(),positions=positions,groups=n,pools=pools,features=proofs,elapsed_seconds=time.monotonic()-started,request=desc(R/'request.json'))
    if save: put(R/'prepared.json',result)
    usage=sum(p.stat().st_size for p in (R/'extra_PL_cache').glob('*') if p.is_file())
    require(usage<CTX['resources']['cache_disk_budget_gib']*2**30,'Compact extraction cache budget exceeded')
    return result
def validate_result(v,t,n):
    require(v['n']==n and v['k']==t['k'] and math.isfinite(v['mi_bits']),'Invalid numeric estimate')
    require(v['unit']=='bits' and v['dtype']=='float64' and v['estimator']=='KSG_1' and v['dx']==v['dy']==3,'KSG definition differs')
    require(v['negative_estimates_clipped'] is False and v['jitter_added_inside_estimator'] is False,'Changed estimator clipping/jitter')
    require(v['marginal_boundary']=='strict_less_than_joint_kth_radius' and v['self_in_marginal_digamma_count'] is True,'Changed KSG boundaries')
def fixture():
    from scipy.special import digamma
    rng=np.random.Generator(np.random.PCG64(20261104)); x=rng.normal(size=(127,3)); y=.7*x+.3*rng.normal(size=x.shape)
    z=np.concatenate([x,y],axis=1); d=np.max(np.abs(z[:,None]-z[None,:]),axis=2); xx=np.max(np.abs(x[:,None]-x[None,:]),axis=2); yy=np.max(np.abs(y[:,None]-y[None,:]),axis=2); out=[]
    for k in KS:
        radius=np.nextafter(np.sort(d,axis=1)[:,k],0); nx=(xx<=radius[:,None]).sum(1); ny=(yy<=radius[:,None]).sum(1)
        expected=float((digamma(k)+digamma(len(x))-np.mean(digamma(nx)+digamma(ny)))/np.log(2)); actual=CTX['est'].ksg_mi(x,y,k=k,workers=1)
        require(actual['mi_bits']==expected,'Independent brute-force fixture differs'); out.append(dict(k=k,mi_bits=expected))
    return out
def configure():
    require(not (R/'request.json').exists() and not (R/'launch_reservation.json').exists(),'Already configured/reserved')
    load(); resource_check(); aud=load_rows()
    group_ids=[x['sample_id'] for x in ROWS]
    for p,ids in SELECT.items():
        old=CTX['selection_records'][p]
        # 沿用固定键名
        require(old['sample_ids']==[group_ids[i] for i in ids],'Original fixed group subset differs')
        put(R/'selections'/('%d.json'%p),dict(percent=p,indices=ids.tolist(),sample_ids=[group_ids[i] for i in ids],group_seed=20261102,pixel_repeats_share_groups=True,original=desc(G/'selections'/('20261102_%d.json'%p))))
    put(R/'task_plan.json',dict(tasks=tasks(),unique_raw=396,unique_cells=132,table_cells=36,fraction_cells=60,k_cells=60))
    frozen=[desc(R/x) for x in ['protocol.json','baseline_pins.json','resources.json','owner.py','test_owner.py','task_plan.json']]
    frozen.extend(desc(p) for p in sorted((R/'selections').glob('*.json')))
    put(R/'request.json',dict(time=now(),schema='fulltest_pixel_sampling_SD_v1',frozen_files=frozen,host=identity(),old_results_readonly=True))
    print('CONFIGURED_396_UNIQUE_RAW',flush=True)
def pilot():
    load(); no_failure(); resource_check(); load_rows(); os.sched_setaffinity(0,set(CTX['resources']['owner_cpus'])); os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    put(R/'pilot_reservation.json',dict(time=now(),identity=identity(),request=desc(R/'request.json')))
    independent=fixture(); mask_evidence=[]; positions=[0,1]
    run=CTX['runner']
    for i in positions:
        row,a,_=original(i); masks,counts=run.validate_masks({d:run.read_bound_array(B,row[d],True) for d in run.DOMAINS},row['native_shape'])
        require(np.array_equal(run.sample_indices(masks,row['sample_id']),a['linear_indices']),'Original sampling replay differs')
        require(all(np.all(masks[d].reshape(-1)[a['linear_indices'][:,di]]) for di,d in enumerate(run.DOMAINS)),'Coordinate-mask membership differs')
        mask_evidence.append(dict(sample_id=row['sample_id'],counts=counts,all12000_memberships=True,all_three_seed_replay_exact=True))
    prep=build_pools(positions,False); out=[]
    os.sched_setaffinity(0,set(CTX['resources']['lanes'][0]))
    for s in SEEDS:
        for m in TARGETS:
            for d in DOMAINS:
                x=POOL[s,d,'rgb'].reshape(-1,3); y=POOL[s,d,m].reshape(-1,3); t=time.monotonic(); v=CTX['est'].ksg_mi(x,y,k=5,workers=8)
                validate_result(v,dict(k=5),2000); out.append(dict(seed=s,domain=d,target=m,result=v,seconds=time.monotonic()-t,input_x_sha256=ah(x),input_y_sha256=ah(y)))
    os.sched_setaffinity(0,set(CTX['resources']['owner_cpus'])); POOL.clear(); gc.collect()
    put(R/'pilot_pass.json',dict(time=now(),status='passed',real_estimates=out,independent_all_k_fixture=independent,mask_evidence=mask_evidence,preparation=prep,request=desc(R/'request.json'),not_production=True))
    print('PILOT_PASSED_36_REAL_ESTIMATES_AND_5_K_FIXTURES',flush=True)
def compatible_old(t):
    if t['seed']!=SEEDS[0] or t['points']!=1000 or t['k']!=5: return None
    if t['target']!='uv' and t['domain']!='F': return None
    return G/'raw'/('%s_groups%d_seed20261102.json'%(metric(t['domain'],t['target']),t['percent']))
def one(t,lane):
    no_failure(); name=key(t); ident=identity(); put(R/'claims'/(name+'.json'),dict(time=now(),task=t,identity=ident,request=desc(R/'request.json')))
    put(R/'lanes'/('%d_status.json'%lane),dict(time=now(),phase='KSG',task=t,identity=ident),True)
    ids=SELECT[t['percent']]; x=np.ascontiguousarray(POOL[t['seed'],t['domain'],'rgb'][ids,:t['points']].reshape(-1,3)); y=np.ascontiguousarray(POOL[t['seed'],t['domain'],t['target']][ids,:t['points']].reshape(-1,3))
    xh,yh=ah(x),ah(y); started=time.monotonic(); oldpath=compatible_old(t)
    if oldpath is not None:
        old=checked(read(oldpath)); require(old['input_x_sha256']==xh and old['input_y_sha256']==yh,'Reuse input bytes differ')
        require(old['pixel_seed']==t['seed'] and old['percent']==t['percent'] and old['N_groups']==len(ids) and old['points_per_group']==1000,'Reuse scientific scope differs')
        require(old['selection']==desc(G/'selections'/('20261102_%d.json'%t['percent'])),'Reuse selection differs')
        require(old['request']==desc(G/'request.json'),'Reuse request differs'); result=old['result']; reused=desc(oldpath)
    else: result=CTX['est'].ksg_mi(x,y,k=t['k'],workers=8); reused=None
    validate_result(result,t,len(ids)*t['points'])
    put(R/'raw'/(name+'.json'),dict(time=now(),task=t,result=result,input_x_sha256=xh,input_y_sha256=yh,prepared=desc(R/'prepared.json'),selection=desc(R/'selections'/('%d.json'%t['percent'])),request=desc(R/'request.json'),identity=ident,reused_from=reused,seconds=time.monotonic()-started))
def lane_run(lane,queue):
    try:
        os.sched_setaffinity(0,set(CTX['resources']['lanes'][lane])); put(R/'lanes'/('%d_started.json'%lane),dict(time=now(),identity=identity()))
        count=0
        while True:
            t=queue.get()
            if t is None: break
            one(t,lane); count+=1
        put(R/'lanes'/('%d_complete.json'%lane),dict(time=now(),identity=identity(),count=count))
    except BaseException: fail('lane%d'%lane); raise
def summarize(audit=False):
    raw={key(t):checked(read(R/'raw'/(key(t)+'.json'))) for t in tasks()}; cells={}
    for t in tasks():
        k=(t['domain'],t['target'],t['percent'],t['points'],t['k'])
        if k in cells: continue
        ts=[dict(t,seed=s) for s in SEEDS]; vv=[raw[key(x)] for x in ts]; vals=[v['result']['mi_bits'] for v in vv]
        for expected,v in zip(ts,vv):
            require(v['task']==expected and v['request']==desc(R/'request.json'),'Raw contract mismatch')
            validate_result(v['result'],expected,expected['N_groups']*expected['points'])
            claim=checked(read(R/'claims'/(key(expected)+'.json'))); require(claim['task']==expected and claim['identity']==v['identity'],'Claim/raw identity differs')
            require(v['selection']==desc(R/'selections'/('%d.json'%expected['percent'])) and v['prepared']==desc(R/'prepared.json'),'Raw input provenance differs')
        mean=statistics.mean(vals); sd=statistics.stdev(vals)
        require(math.isclose(mean,float(np.mean(vals)),abs_tol=1e-14,rel_tol=0) and math.isclose(sd,float(np.std(vals,ddof=1)),abs_tol=1e-14,rel_tol=0),'Independent mean/SD disagreement')
        cells[k]=dict(domain=t['domain'],target=t['target'],percent=t['percent'],points=t['points'],k=t['k'],N_groups=t['N_groups'],seeds=list(SEEDS),values=vals,mean=mean,sample_SD=sd,SD_definition='pixel sampling; ddof=1; fixed sample groups',raw=[desc(R/'raw'/(key(x)+'.json')) for x in ts])
    table=[cells[d,m,100,n,5] for m in TARGETS for d in DOMAINS for n in POINTS]
    fraction=[cells[d,m,p,1000,5] for m in TARGETS for d in DOMAINS for p in FRACTIONS]
    kval=[cells[d,m,100,1000,k] for m in TARGETS for d in DOMAINS for k in KS]
    for m in TARGETS:
        for d in DOMAINS:
            anchor=cells[d,m,100,1000,5]; require(anchor in table and anchor in fraction and anchor in kval,'Shared endpoint absent')
    out=dict(time=now(),status='complete',unique_cells=list(cells.values()),table1A=table,group_fraction=fraction,k_sensitivity=kval,N=N,omitted_N_column=True,request=desc(R/'request.json'))
    if not audit:
        put(R/'summary.json',out)
        for name,rows in [('Table1A_full_precision',table),('group_fraction_pixel_SD',fraction),('k_sensitivity_pixel_SD',kval)]:
            stream=io.StringIO(newline=''); fields=['domain','target','percent','points','k','N_groups','mean','sample_SD','SD_definition']
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
            writer.writerows({f:row[f] for f in fields} for row in rows)
            publish(R/'tables'/(name+'.csv'),stream.getvalue().encode('utf-8'))
        stream=io.StringIO(newline=''); writer=csv.writer(stream,lineterminator='\n'); writer.writerow(['Modality pair','Region','100 points','500 points','1000 points'])
        native=[]
        for m in TARGETS:
            for d in DOMAINS:
                row=['RGB–'+m.upper(),d]+['%.3f ± %.3f'%(cells[d,m,100,n,5]['mean'],cells[d,m,100,n,5]['sample_SD']) for n in POINTS]
                writer.writerow(row); native.append(row)
        publish(R/'tables/Table1A_three_decimal.csv',stream.getvalue().encode('utf-8'))
        put(R/'tables/Table1A_native_cells.json',dict(columns=['Modality pair','Region','100 points','500 points','1000 points'],rows=native,caption='All 19,651 MI-eligible TEST groups from the original 21,899; unchanged 2,248 exclusions. KSG1 k=5; mean ± sample SD across pixel seeds20261025/26/27; fixed groups, original paired native coordinates. P/D/O are UV-defined supports for all three targets.'))
    else:
        prior=checked(read(R/'summary.json'))
        for field in ('unique_cells','table1A','group_fraction','k_sensitivity'): require(prior[field]==out[field],'Saved summary differs')
    return out
def audit():
    load(); no_failure(); load_rows()
    expected={key(t)+'.json' for t in tasks()}; require({p.name for p in (R/'raw').glob('*.json')}==expected,'Raw coverage differs')
    prep=checked(read(R/'prepared.json')); require(prep['groups']==N and prep['positions']==list(range(N)),'Prepared cohort differs')
    for i,proof in enumerate(prep['features']):
        require(proof['sample_id']==ROWS[i]['sample_id'],'Prepared order differs')
        for d in (proof['original']['feature'],proof['original']['metadata'],proof['PL']['array'],proof['PL']['receipt']): verify(d)
        if i%500==0: status('final_audit',groups_verified=i,total=N)
    for lane in range(len(CTX['resources']['lanes'])):
        a=checked(read(R/'lanes'/('%d_started.json'%lane))); b=checked(read(R/'lanes'/('%d_complete.json'%lane)))
        require(a['identity']==b['identity'] and a['identity']['cpus']==CTX['resources']['lanes'][lane],'Worker lifecycle differs')
    pools={(p['seed'],p['domain'],p['modality']):p for p in prep['pools']}
    for t in tasks():
        raw=checked(read(R/'raw'/(key(t)+'.json')))
        if t['percent']==100 and t['points']==1000:
            require(raw['input_x_sha256']==pools[t['seed'],t['domain'],'rgb']['sha256'] and raw['input_y_sha256']==pools[t['seed'],t['domain'],t['target']]['sha256'],'Full-pool input hash mismatch')
        if raw['reused_from'] is not None:
            old=checked(read(verify(raw['reused_from'])))
            require(raw['result']==old['result'] and raw['input_x_sha256']==old['input_x_sha256'] and raw['input_y_sha256']==old['input_y_sha256'],'Reused estimate differs')
    out=summarize(True)
    put(R/'acceptance.json',dict(time=now(),status='passed',unique_raw=396,unique_cells=132,three_views_share_endpoint_exactly=True,pixel_SD_only=True,source_files_preserved=True,limits=['Independent coverage, source hashes, bindings and arithmetic; not a second full KSG rerun.','Conditional on fixed group subsets; pixel SD is not group-selection uncertainty or a patient-level confidence interval.'],summary=desc(R/'summary.json'),request=desc(R/'request.json')))
def launch():
    load(); no_failure(); resource_check(); p=checked(read(R/'pilot_pass.json')); require(p['status']=='passed' and p['request']==desc(R/'request.json'),'Pilot not accepted')
    require(not (R/'launch_reservation.json').exists(),'Launch already reserved')
    token=secrets.token_hex(32); put(R/'launch_reservation.json',dict(time=now(),token_sha256=hashlib.sha256(token.encode()).hexdigest(),request=desc(R/'request.json')))
    log=(R/'owner.log').open('xb'); child=subprocess.Popen(['ionice','-c','3',sys.executable,'-B','-u',str(Path(__file__)),'run',token],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
    put(R/'launch_receipt.json',dict(time=now(),identity=identity(child.pid),detached=True,request=desc(R/'request.json'))); print(json.dumps(dict(launched=True,pid=child.pid)),flush=True)
def owner(token):
    import fcntl
    lock=(R/'owner.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    load(); no_failure(); resource_check(); r=checked(read(R/'launch_reservation.json')); require(r['token_sha256']==hashlib.sha256(token.encode()).hexdigest(),'Owner launch token differs')
    os.sched_setaffinity(0,set(CTX['resources']['owner_cpus'])); os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    put(R/'owner_started.json',dict(time=now(),identity=identity(),request=desc(R/'request.json')))
    load_rows(); build_pools(); ctx=mp.get_context('fork'); queue=ctx.Queue()
    for t in tasks(): queue.put(t)
    for _ in CTX['resources']['lanes']: queue.put(None)
    children=[ctx.Process(target=lane_run,args=(i,queue)) for i in range(len(CTX['resources']['lanes']))]
    for c in children: c.start()
    while any(c.is_alive() for c in children):
        status('production',completed=len(list((R/'raw').glob('*.json'))),total=396,failures=len(list((R/'failures').glob('*.json')))); time.sleep(15)
    for c in children: c.join(); require(c.exitcode==0,'Worker failed; evidence preserved')
    queue.close(); POOL.clear(); gc.collect(); summarize(); audit()
    put(R/'completion.json',dict(time=now(),status='complete',acceptance=desc(R/'acceptance.json'),summary=desc(R/'summary.json'),request=desc(R/'request.json'))); status('complete',raw=396)
if __name__=='__main__':
    try:
        cmd=sys.argv[1]
        if cmd=='run': owner(sys.argv[2])
        else: {'configure':configure,'pilot':pilot,'launch':launch,'audit':audit}[cmd]()
    except BaseException:
        fail(sys.argv[1] if len(sys.argv)>1 else 'no_command'); raise
