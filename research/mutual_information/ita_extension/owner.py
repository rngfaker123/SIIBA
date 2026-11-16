"""Missing RGB--PL regional MI. Frozen coordinates, original CPU KSG, no GPU.

CLI: configure; pilot; launch; owner TOKEN; audit. A failed or reserved owner
is never automatically relaunched. New files live exclusively beneath ROOT.
"""
from pathlib import Path
import os, sys, json, hashlib, datetime, time, importlib.util, statistics
import subprocess, traceback, math, multiprocessing as mp, gc, secrets, csv
from concurrent.futures import ProcessPoolExecutor

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261025, 20261026, 20261027)
DOMAINS = ('P', 'D', 'O')
TARGETS = ('plh', 'plv')
COHORTS = (('dark',10),('very_light',309),('table1',1000),('light',2498),
           ('brown',2793),('intermediate',5979),('tan',8056))
CTX = {}; POOLS = {}; CURRENT = None

def require(ok, why):
    if not ok: raise ValueError(why)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def enc(x): return (json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
def digest(x): return hashlib.sha256(enc(x)).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def ah(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def desc(p): return {'path':str(Path(p)), 'sha256':sha(p)}
def verify(d):
    p=Path(d['path']); require(sha(p)==d['sha256'], 'Source changed: '+str(p)); return p
def unseal(v):
    z=dict(v); old=z.pop('record_sha256', None)
    require(old==digest(z), 'Record seal mismatch'); return v
def seal(v): return dict(v, record_sha256=digest(v))
def safe_output(p):
    p=Path(p).resolve(); require(ROOT.resolve() in p.parents, 'Output outside new root'); return p
def publish_bytes(p, data, mutable=False):
    p=safe_output(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+secrets.token_hex(4))
    with t.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    try:
        if mutable: os.replace(t,p)
        else: os.link(t,p)
    finally:
        if t.exists(): t.unlink()
def put(p,v,mutable=False): publish_bytes(p,enc(seal(v)),mutable)
def status(phase,**kw): put(ROOT/'status.json',dict(time=now(),phase=phase,**kw),True)
def module(p,name):
    s=importlib.util.spec_from_file_location(name,str(p)); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m
def identity(pid=None):
    pid=pid or os.getpid(); a=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttick=int(a[19]),ppid=int(a[1]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]),cpus=sorted(os.sched_getaffinity(pid)))
def fail(label):
    p=ROOT/'failures'/(label+'_'+str(os.getpid())+'_'+str(time.time_ns())+'.json')
    put(p,dict(time=now(),traceback=traceback.format_exc()))
def no_failure(): require(not list((ROOT/'failures').glob('*.json')), 'Failure preserved: manual diagnosis required')
def task_key(t): return '%s__%s_rgb_%s__n1000_seed%d'%(t['cohort'],t['domain'],t['target'],t['seed'])
def tasks():
    return [dict(cohort=c,N=n,domain=d,target=m,seed=s,k=5,points=1000)
            for c,n in COHORTS for d in DOMAINS for m in TARGETS for s in SEEDS]
def parse_cpu_list(text):
    out=set()
    for part in text.strip().split(','):
        a=part.split('-'); out.update(range(int(a[0]),int(a[-1])+1))
    return out
def resources_check(r, check_protected=True):
    ident=identity(); require(all(ident[k]==r['host'][k] for k in ('boot','pid1_starttick')), 'Host/container changed')
    lanes=r['lane_cpus']; require(len(lanes)==8 and all(len(x)==8 for x in lanes),'Expected 8 lanes x 8 CPUs')
    allc=[c for lane in lanes for c in lane]; require(len(set(allc))==len(allc),'Overlapping CPU lanes')
    require(not set(allc).intersection(r['owner_cpus']),'Owner/lanes overlap')
    require(set(allc+r['owner_cpus']).issubset(os.sched_getaffinity(0)),'CPU allocation unavailable')
    protected=set(r['protected_cpus'])
    for c in allc+r['owner_cpus']:
        sib=parse_cpu_list(Path('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list'%c).read_text())
        require(not sib.intersection(protected),'Physical CPU overlaps training')
    if check_protected:
        for p in r['protected_processes']:
            q=identity(p['pid']); require(q['starttick']==p['starttick'] and q['cpus']==p['cpus'],'Protected process identity/affinity changed')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available>(r['memory_budget_gib']+r['reserve_free_gib'])*2**30,'Insufficient memory headroom')
    require(0<r['cache_budget_gib']<=4 and r['memory_budget_gib']>=32,'Resource budget insufficient')
    return ident

def load_science():
    p=read(ROOT/'protocol.json'); b=read(ROOT/'baseline_pins.json')
    run=module(verify(p['runner']),'rgb_pl_original_runner')
    est=module(verify(p['estimator']),'rgb_pl_original_estimator')
    require(sha(Path(sys.modules[run.ksg_mi.__module__].__file__))==p['estimator']['sha256'],'Runner imported wrong estimator')
    require(run.SEEDS==SEEDS and run.MODS==('rgb','uv','plv','plh') and run.DOMAINS==('F','P','D','O'), 'Original scientific constants changed')
    require(run.NOISE_SEED==20260817 and run.K==5 and run.MAX_POINTS==1000,'Original numerical contract changed')
    CTX.update(protocol=p,baseline=b,runner=run,est=est)
    return p,b

def cohort_meta(c):
    run=CTX['runner']; spec=CTX['baseline']['cohorts'][c]
    m=read(verify(spec['input'])); prep=run.check_seal(read(verify(spec['preparation']))); pro=read(verify(spec['protocol']))
    n=dict(COHORTS)[c]; rows=m['rows']; require(len(rows)==len(prep['features'])==n,'Cohort count changed')
    ids=[x['sample_id'] for x in rows]; require(len(set(ids))==n,'Duplicate original group')
    require(ids==prep['sample_order']==pro['sample_order'],'Original sample order changed')
    require(pro['input_manifest_sha256']==spec['input']['sha256'] and pro['input_contract_sha256']==run.digest(m),'Input binding mismatch')
    require(prep['protocol_sha256']==run.digest(pro) and prep['status']=='complete','Preparation binding mismatch')
    require(pro['k']==5 and pro['seeds']==list(SEEDS) and pro['noise_seed']==20260817,'Original scientific settings changed')
    require(pro['software']['python']==sys.version.split()[0] and pro['software']['numpy']==np.__version__ and pro['software']['scipy']==CTX['est'].scipy.__version__,'Original runtime differs')
    return dict(name=c,N=n,rows=rows,prep=prep,pro=pro,spec=spec,base=str(Path(spec['input']['path']).parent),out=str(Path(spec['preparation']['path']).parent))

def verify_baseline():
    for d in CTX['baseline']['preserved']: verify(d)
    for c,_ in COHORTS: cohort_meta(c)

def original_feature(meta,i):
    run=CTX['runner']; row=meta['rows'][i]; fr=meta['prep']['features'][i]
    p=Path(meta['out'])/fr['feature_path']; q=p.with_suffix('.json')
    require(fr['index']==i and fr['sample_id']==row['sample_id'] and fr['subject_id']==row['subject_id'],'Feature identity mismatch')
    require(sha(p)==fr['feature_sha256'] and sha(q)==fr['metadata_sha256'],'Feature bytes changed')
    m=run.check_seal(read(q)); require(m['source_row_sha256']==run.digest(row) and m['sample_id']==row['sample_id'],'Row provenance mismatch')
    require(m['protocol_sha256']==run.digest(meta['pro']) and m['feature_sha256']==fr['feature_sha256'],'Feature protocol mismatch')
    require(m['index']==i and m['subject_id']==row['subject_id'],'Feature metadata identity changed')
    require(m['source_sha256']=={k:row[k]['sha256'] for k in run.MODS+run.DOMAINS},'Source bindings changed')
    require(m['partition_verified'] is True and m['source_hashes_verified'] is True and all(m['mask_pixels'][d]>=1000 for d in run.DOMAINS),'Original eligibility invalid')
    a=run.validate_features(run.load_npz(p),row); require(ah(a['linear_indices'])==m['linear_indices_sha256'],'Coordinates changed')
    return row,a,dict(feature=desc(p),metadata=desc(q),sample_id=row['sample_id'],source_row_sha256=m['source_row_sha256'],linear_indices_sha256=m['linear_indices_sha256'])

def extract_pl_arrays(row,a,base,reader):
    """All PL pixel values use the original saved native coordinates, without RNG."""
    out=np.empty((3,3,2,1000,3),np.uint8)
    for j,mod in enumerate(TARGETS):
        image=reader(base,row[mod]); require(image.shape==tuple(row['native_shape'])+(3,) and image.dtype==np.uint8,'PL image geometry/dtype differs')
        flat=image.reshape(-1,3); k=CTX['runner'].MODS.index(mod)
        require(np.array_equal(flat[a['linear_indices'][:,0]],a['full_values'][:,k]),'Decoded PL values differ from original F cache')
        out[:,:,j]=flat[a['linear_indices'][:,1:]]
        del image,flat
    return out

def cache_key(row,a):
    return digest(dict(sample_id=row['sample_id'],native_shape=row['native_shape'],coordinates=ah(a['linear_indices']),sources={m:row[m] for m in TARGETS}))

def extraction_worker(i):
    meta=CURRENT; row,a,proof=original_feature(meta,i); token=cache_key(row,a)
    p=ROOT/'sampled_PL_cache'/(token+'.npy'); receipt=p.with_suffix('.json')
    if receipt.exists():
        v=unseal(read(receipt)); require(v['cache_key']==token and v['request']==desc(ROOT/'request.json'),'Cached request mismatch')
        require(sha(p)==v['array']['sha256'],'PL cache bytes changed'); raw=np.load(p,allow_pickle=False)
        require(raw.dtype==np.uint8 and raw.shape==(3,3,2,1000,3),'PL cache layout changed')
    else:
        require(not p.exists(),'Unsealed PL cache; no automatic overwrite')
        raw=extract_pl_arrays(row,a,Path(meta['base']),CTX['runner'].read_bound_array)
        import io
        stream=io.BytesIO(); np.save(stream,raw,allow_pickle=False); publish_bytes(p,stream.getvalue())
        put(receipt,dict(time=now(),cache_key=token,sample_id=row['sample_id'],source_PL={m:row[m] for m in TARGETS},
            linear_indices_sha256=proof['linear_indices_sha256'],all_saved_F_values_bit_exact=True,array=desc(p),
            request=desc(ROOT/'request.json'),original_geometry=True,no_resampling=True))
    proof.update(new_PL_cache=desc(p),new_PL_cache_receipt=desc(receipt))
    # 逐组去量化
    out=np.empty((3,3,3,1000,3),np.float64)
    for si in range(3):
        for di in range(3):
            coords=a['linear_indices'][si,di+1]
            values=[a['region_values'][si,di,0],raw[si,di,0],raw[si,di,1]]
            for mi,mod in enumerate(('rgb','plh','plv')):
                out[si,di,mi]=CTX['runner'].dequantize_native(values[mi],coords,row['sample_id'],row['native_shape'][1],mod)
    require(np.isfinite(out).all(),'Non-finite dequantization')
    return i,out,proof

def worker_init(counter):
    with counter.get_lock(): lane=counter.value; counter.value+=1
    os.sched_setaffinity(0,set(CTX['resources']['lane_cpus'][lane]))

def make_pools(meta,positions=None,write_receipt=True):
    global CURRENT,POOLS
    CURRENT=meta; positions=list(range(meta['N'])) if positions is None else list(positions); n=len(positions)
    estimate=2*2**30+meta['N']*1000*(27*24+8*384)
    require(estimate<CTX['resources']['memory_budget_gib']*2**30,'Memory estimate exceeds quota')
    POOLS={(s,d,m):np.empty((n,1000,3),np.float64) for s in SEEDS for d in DOMAINS for m in ('rgb','plh','plv')}
    proofs=[]; started=time.monotonic(); context=mp.get_context('fork'); counter=context.Value('i',0)
    with ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=worker_init,initargs=(counter,)) as pool:
        # 限制在途批数
        for start in range(0,n,32):
            fs=[pool.submit(extraction_worker,i) for i in positions[start:start+32]]
            for offset,f in enumerate(fs):
                index,v,proof=f.result(); slot=start+offset; require(index==positions[slot],'Extraction order changed')
                for si,s in enumerate(SEEDS):
                    for di,d in enumerate(DOMAINS):
                        for mi,m in enumerate(('rgb','plh','plv')): POOLS[s,d,m][slot]=v[si,di,mi]
                proofs.append(proof); del v
            status('extract_PL_at_original_coordinates',cohort=meta['name'],done=min(start+32,n),total=n,elapsed_seconds=time.monotonic()-started)
    descriptors=[]
    for (s,d,m),a in POOLS.items():
        a.flags.writeable=False; descriptors.append(dict(seed=s,domain=d,modality=m,shape=list(a.shape),sha256=ah(a)))
    anchors=[]
    if positions==list(range(meta['N'])):
        for s in SEEDS:
            for d in DOMAINS:
                p=Path(meta['out'])/'results'/('%s_rgb_uv_n1000_seed%d.json'%(d,s)); old=CTX['runner'].check_seal(read(p))
                require(old['sample_count']==meta['N'] and old['points_per_sample']==1000 and old['sampling_seed']==s and old['domain']==d,'Anchor scope mismatch')
                require(old['preparation_sha256']==meta['prep']['record_sha256'] and old['protocol_sha256']==CTX['runner'].digest(meta['pro']),'Anchor provenance mismatch')
                require(ah(POOLS[s,d,'rgb'])==old['input_x_sha256'],'Original RGB input hash differs')
                anchors.append(desc(p))
    v=dict(time=now(),cohort=meta['name'],N=n,positions=positions,pools=descriptors,features=proofs,
           RGB_anchors=anchors,elapsed_seconds=time.monotonic()-started,request=desc(ROOT/'request.json'),
           limitations=['Original RGB/masks inherited via hash-bound features; no image-level registration revalidation.'])
    if write_receipt: put(ROOT/'prepared'/(meta['name']+'.json'),v)
    usage=sum(p.stat().st_size for p in (ROOT/'sampled_PL_cache').iterdir())
    require(usage<CTX['resources']['cache_budget_gib']*2**30,'Compact cache quota exceeded')
    return v

def independent_fixture():
    from scipy.special import digamma
    rng=np.random.Generator(np.random.PCG64(20261104)); x=rng.normal(size=(127,3)); y=.7*x+.3*rng.normal(size=x.shape)
    z=np.concatenate([x,y],axis=1); distance=np.max(np.abs(z[:,None]-z[None,:]),axis=2)
    strict=np.nextafter(np.sort(distance,axis=1)[:,5],0)
    nx=(np.max(np.abs(x[:,None]-x[None,:]),axis=2)<=strict[:,None]).sum(1)
    ny=(np.max(np.abs(y[:,None]-y[None,:]),axis=2)<=strict[:,None]).sum(1)
    expected=float((digamma(5)+digamma(len(x))-np.mean(digamma(nx)+digamma(ny)))/np.log(2))
    got=CTX['est'].ksg_mi(x,y,k=5,workers=1)
    require(got['mi_bits']==expected,'Independent brute-force KSG mismatch')
    return dict(n=127,k=5,bit_exact=True,mi_bits=expected)

def pilot_masks(meta,positions):
    run=CTX['runner']; records=[]
    for i in positions:
        row,a,_=original_feature(meta,i)
        masks,counts=run.validate_masks({d:run.read_bound_array(Path(meta['base']),row[d],True) for d in run.DOMAINS},row['native_shape'])
        for di,d in enumerate(run.DOMAINS):
            require(bool(np.all(masks[d].reshape(-1)[a['linear_indices'][:,di]])), 'Pilot original coordinate/mask membership failed')
        require(np.array_equal(run.sample_indices(masks,row['sample_id']),a['linear_indices']), 'Pilot frozen sampling replay differs')
        records.append(dict(sample_id=row['sample_id'],original_position=i,mask_counts=counts,
            source_mask_sha256={d:row[d]['sha256'] for d in run.DOMAINS},partition_and_12000_memberships_verified=True,sampling_replay_exact_diagnostic_only=True))
    return records

def configure():
    require(not (ROOT/'request.json').exists() and not (ROOT/'launch_reservation.json').exists(),'Already configured/reserved')
    p,b=load_science(); r=read(ROOT/'resources.json'); CTX['resources']=r; ident=resources_check(r)
    verify_baseline()
    # 冻结原始结果
    original_results=[]
    for c,_ in COHORTS:
        meta=cohort_meta(c); rr=sorted((Path(meta['out'])/'results').glob('*.json'))
        require(len(rr)==54,'Original result coverage differs')
        for path in rr:
            old=CTX['runner'].check_seal(read(path))
            require(old['protocol_sha256']==CTX['runner'].digest(meta['pro']) and old['preparation_sha256']==meta['prep']['record_sha256'],'Original result binding changed')
            if c=='table1':
                summary=CTX['runner'].check_seal(read(verify(meta['spec']['summary'])))
                require(summary['result_record_sha256'][path.name]==old['record_sha256'],'Table1 original result changed')
            original_results.append(desc(path))
    plan=tasks(); require(len(plan)==126 and len({task_key(t) for t in plan})==126,'Task plan invalid')
    put(ROOT/'task_plan.json',dict(tasks=plan,raw=126,summary_cells=42,cohort_order=[c for c,_ in COHORTS]))
    put(ROOT/'request.json',dict(schema='RGB_PL_UV_defined_regions_MI_v1',time=now(),protocol=desc(ROOT/'protocol.json'),
        baseline=desc(ROOT/'baseline_pins.json'),resources=desc(ROOT/'resources.json'),task_plan=desc(ROOT/'task_plan.json'),
        sources={p.name:desc(p) for p in sorted((ROOT/'source').glob('*.py'))},original_results=original_results,host=ident))
    print('CONFIGURED_126_RAW_42_CELLS',flush=True)

def context():
    req=unseal(read(ROOT/'request.json'))
    for k in ('protocol','baseline','resources','task_plan'): verify(req[k])
    for d in req['sources'].values(): verify(d)
    load_science(); CTX.update(request=req,resources=read(ROOT/'resources.json'))
    require(unseal(read(ROOT/'task_plan.json'))['tasks']==tasks(),'Frozen task plan changed')
    return req

def pilot():
    context(); no_failure(); resources_check(CTX['resources']); os.sched_setaffinity(0,set(CTX['resources']['owner_cpus']))
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    put(ROOT/'pilot_reservation.json',dict(time=now(),identity=identity(),request=desc(ROOT/'request.json')))
    fixture=independent_fixture(); outputs=[]; masks=[]
    for c in ('table1','dark'):
        meta=cohort_meta(c); masks.extend(pilot_masks(meta,[0,1])); prep=make_pools(meta,[0,1],False)
        for d in DOMAINS:
            for m in TARGETS:
                x=POOLS[SEEDS[0],d,'rgb'].reshape(-1,3); y=POOLS[SEEDS[0],d,m].reshape(-1,3)
                os.sched_setaffinity(0,set(CTX['resources']['lane_cpus'][0]))
                start=time.monotonic(); v=CTX['est'].ksg_mi(x,y,k=5,workers=8)
                os.sched_setaffinity(0,set(CTX['resources']['owner_cpus']))
                require(math.isfinite(v['mi_bits']),'Pilot nonfinite')
                outputs.append(dict(cohort=c,domain=d,target=m,N=2,points=1000,seed=SEEDS[0],result=v,
                    input_x_sha256=ah(x),input_y_sha256=ah(y),elapsed_seconds=time.monotonic()-start))
        put(ROOT/'pilot'/(c+'_extraction.json'),prep); POOLS.clear(); gc.collect()
    verify_baseline()
    put(ROOT/'pilot_pass.json',dict(time=now(),status='passed',fixture=fixture,real_estimates=outputs,mask_checks=masks,
        request=desc(ROOT/'request.json'),not_production_estimates=True,small_pilot_does_not_estimate_full_runtime=True))
    print('PILOT_PASSED_12_REAL_ESTIMATES',flush=True)

def lane(c,i):
    try:
        os.sched_setaffinity(0,set(CTX['resources']['lane_cpus'][i]))
        selected=[t for j,t in enumerate([x for x in tasks() if x['cohort']==c]) if j%8==i]
        put(ROOT/'lanes'/('%s_%d_launch.json'%(c,i)),dict(time=now(),identity=identity(),tasks=[task_key(t) for t in selected],request=desc(ROOT/'request.json')))
        prep=desc(ROOT/'prepared'/(c+'.json'))
        for t in selected:
            no_failure(); key=task_key(t); put(ROOT/'claims'/(key+'.json'),dict(task=t,identity=identity(),request=desc(ROOT/'request.json')))
            x=POOLS[t['seed'],t['domain'],'rgb'].reshape(-1,3); y=POOLS[t['seed'],t['domain'],t['target']].reshape(-1,3)
            require(x.shape==y.shape==(t['N']*1000,3),'Pool geometry differs')
            put(ROOT/'lanes'/('%s_%d_status.json'%(c,i)),dict(time=now(),phase='original_CPU_KSG',task=t,identity=identity()),True)
            start=time.monotonic(); v=CTX['est'].ksg_mi(x,y,k=5,workers=8)
            require(v['n']==t['N']*1000 and v['k']==5 and math.isfinite(v['mi_bits']) and v['negative_estimates_clipped'] is False,'Invalid estimate')
            put(ROOT/'raw'/(key+'.json'),dict(time=now(),task=t,result=v,input_x_sha256=ah(x),input_y_sha256=ah(y),
                request=desc(ROOT/'request.json'),prepared=prep,identity=identity(),device='CPU',elapsed_seconds=time.monotonic()-start))
        put(ROOT/'lanes'/('%s_%d_complete.json'%(c,i)),dict(time=now(),status='complete',identity=identity(),tasks=len(selected),request=desc(ROOT/'request.json')))
    except BaseException: fail('lane_'+c+'_'+str(i)); raise

def summary_cohort(c):
    cells=[]
    for d in DOMAINS:
        for m in TARGETS:
            ts=[t for t in tasks() if t['cohort']==c and t['domain']==d and t['target']==m]
            records=[unseal(read(ROOT/'raw'/(task_key(t)+'.json'))) for t in ts]
            values=[x['result']['mi_bits'] for x in records]
            cells.append(dict(cohort=c,domain=d,target=m,N=dict(COHORTS)[c],points=1000,k=5,seeds=list(SEEDS),
                raw_bits=values,mean_bits=statistics.mean(values),sample_sd_bits=statistics.stdev(values),
                SD_definition='three pixel-sampling draws; sample SD ddof=1'))
    put(ROOT/'summaries'/(c+'.json'),dict(time=now(),status='complete',cells=cells,request=desc(ROOT/'request.json')))

def audit():
    context(); no_failure(); verify_baseline()
    for d in CTX['request']['original_results']: verify(d)
    plan=tasks(); expected={task_key(t)+'.json' for t in plan}
    require({p.name for p in (ROOT/'raw').glob('*.json')}==expected,'Raw coverage mismatch')
    results=[]
    for c,n in COHORTS:
        identities={}
        for lane_id in range(8):
            start=unseal(read(ROOT/'lanes'/('%s_%d_launch.json'%(c,lane_id))))
            end=unseal(read(ROOT/'lanes'/('%s_%d_complete.json'%(c,lane_id))))
            expected_lane=[task_key(t) for j,t in enumerate([x for x in plan if x['cohort']==c]) if j%8==lane_id]
            require(start['tasks']==expected_lane and end['tasks']==len(expected_lane) and end['status']=='complete','Lane coverage/completion differs')
            require(start['identity']==end['identity'] and start['identity']['cpus']==CTX['resources']['lane_cpus'][lane_id],'Lane identity changed')
            require(start['request']==end['request']==desc(ROOT/'request.json'),'Lane request changed')
            for name in expected_lane: identities[name]=start['identity']
        meta=cohort_meta(c); prep=unseal(read(ROOT/'prepared'/(c+'.json')))
        require(prep['N']==n and prep['positions']==list(range(n)) and len(prep['features'])==n,'Prepared coverage mismatch')
        require(len(prep['RGB_anchors'])==9,'Missing original RGB anchors')
        for d in prep['RGB_anchors']: verify(d)
        for i,v in enumerate(prep['features']):
            for f in ('feature','metadata','new_PL_cache','new_PL_cache_receipt'): verify(v[f])
            require(v['sample_id']==meta['rows'][i]['sample_id'],'Prepared ordering changed')
        pools={(p['seed'],p['domain'],p['modality']):p for p in prep['pools']}
        sm=unseal(read(ROOT/'summaries'/(c+'.json'))); require(len(sm['cells'])==6,'Summary cell count mismatch')
        for t in [t for t in plan if t['cohort']==c]:
            path=ROOT/'raw'/(task_key(t)+'.json'); v=unseal(read(path)); claim=unseal(read(ROOT/'claims'/path.name))
            require(v['task']==claim['task']==t and v['request']==claim['request']==desc(ROOT/'request.json'),'Raw binding changed')
            require(v['identity']==claim['identity']==identities[task_key(t)],'Claim/raw/owned lane identity mismatch')
            require(v['prepared']==desc(ROOT/'prepared'/(c+'.json')),'Prepared raw binding changed')
            require(v['input_x_sha256']==pools[t['seed'],t['domain'],'rgb']['sha256'] and v['input_y_sha256']==pools[t['seed'],t['domain'],t['target']]['sha256'],'Input pool hashes differ')
            r=v['result']; require(r['n']==n*1000 and r['k']==5 and r['unit']=='bits' and r['dtype']=='float64' and r['distance']=='chebyshev_max_norm' and r['negative_estimates_clipped'] is False,'Estimator contract differs')
            require(r['estimator']=='KSG_1' and r['dx']==r['dy']==3 and r['jitter_added_inside_estimator'] is False and r['self_in_marginal_digamma_count'] is True and r['marginal_boundary']=='strict_less_than_joint_kth_radius','Scientific estimator settings differ')
            require(math.isfinite(r['mi_bits']) and math.isfinite(r['mi_nats']) and math.isclose(r['mi_bits']*math.log(2),r['mi_nats'],abs_tol=1e-14,rel_tol=0),'Nonfinite/inconsistent MI units')
            results.append(desc(path))
        require({(v['domain'],v['target']) for v in sm['cells']}=={(d,m) for d in DOMAINS for m in TARGETS},'Summary identities duplicated/missing')
        for cell in sm['cells']:
            require(cell['cohort']==c and cell['N']==n and cell['points']==1000 and cell['k']==5 and cell['seeds']==list(SEEDS),'Summary scope differs')
            vals=[unseal(read(ROOT/'raw'/(task_key(t)+'.json')))['result']['mi_bits'] for t in plan if t['cohort']==c and t['domain']==cell['domain'] and t['target']==cell['target']]
            require(len(vals)==3 and cell['raw_bits']==vals,'Summary draw mismatch')
            require(math.isclose(cell['mean_bits'],float(np.mean(vals)),abs_tol=1e-14,rel_tol=0) and math.isclose(cell['sample_sd_bits'],float(np.std(vals,ddof=1)),abs_tol=1e-14,rel_tol=0),'Independent mean/SD mismatch')
    put(ROOT/'independent_acceptance.json',dict(time=now(),status='passed',raw_count=126,cells=42,results=results,
        original_bytes_preserved=True,request=desc(ROOT/'request.json'),
        limits=['Independent arithmetic/provenance audit; original KSG not independently rerun.',
                'UV-defined regional supports; not PL-native signal detection or entire-image information.']))

def owner(token):
    import fcntl
    lock=(ROOT/'owner.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    context(); no_failure(); reservation=unseal(read(ROOT/'launch_reservation.json'))
    require(reservation['token_sha256']==hashlib.sha256(token.encode()).hexdigest(),'Invalid owner launch token')
    resources_check(CTX['resources']); os.sched_setaffinity(0,set(CTX['resources']['owner_cpus']))
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    put(ROOT/'owner_started.json',dict(time=now(),identity=identity(),request=desc(ROOT/'request.json')))
    require(unseal(read(ROOT/'pilot_pass.json'))['status']=='passed','Pilot has not passed')
    verify_baseline()
    for c,n in COHORTS:
        no_failure(); status('preparing_cohort',cohort=c,N=n)
        make_pools(cohort_meta(c)); children=[mp.get_context('fork').Process(target=lane,args=(c,i)) for i in range(8)]
        for p in children: p.start()
        while any(p.is_alive() for p in children):
            status('production',cohort=c,raw=len(list((ROOT/'raw').glob('*.json'))),total=126,failed_lanes=len(list((ROOT/'failures').glob('*.json'))))
            time.sleep(10)
        for p in children: p.join(); require(p.exitcode==0,'Failed lane preserved; no automatic retry')
        summary_cohort(c); POOLS.clear(); gc.collect()
    status('final_provenance_and_arithmetic_audit')
    subprocess.run([sys.executable,str(Path(__file__)),'audit'],check=True)
    put(ROOT/'completion.json',dict(time=now(),status='complete',raw=126,cells=42,acceptance=desc(ROOT/'independent_acceptance.json'),
        summaries=[desc(ROOT/'summaries'/(c+'.json')) for c,_ in COHORTS],request=desc(ROOT/'request.json')))
    status('complete',raw=126,cells=42)

def launch():
    context(); no_failure(); resources_check(CTX['resources'])
    v=unseal(read(ROOT/'pilot_pass.json')); require(v['status']=='passed' and v['request']==desc(ROOT/'request.json'),'Pilot/request mismatch')
    require(not (ROOT/'launch_reservation.json').exists(),'Already reserved; no relaunch')
    token=secrets.token_hex(32); put(ROOT/'launch_reservation.json',dict(time=now(),identity=identity(),request=desc(ROOT/'request.json'),token_sha256=hashlib.sha256(token.encode()).hexdigest()))
    log=(ROOT/'owner.log').open('xb')
    proc=subprocess.Popen(['ionice','-c','3',sys.executable,'-u',str(Path(__file__)),'owner',token],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
    put(ROOT/'launch_receipt.json',dict(time=now(),identity=identity(proc.pid),request=desc(ROOT/'request.json'),detached=True))
    print(json.dumps(dict(detached=True,identity=identity(proc.pid))),flush=True)

if __name__=='__main__':
    try:
        command=sys.argv[1]
        if command=='owner': owner(sys.argv[2])
        else: {'configure':configure,'pilot':pilot,'launch':launch,'audit':audit}[command]()
    except BaseException:
        fail(sys.argv[1] if len(sys.argv)>1 else 'missing_command'); raise
