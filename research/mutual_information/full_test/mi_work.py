"""Frozen scientific inputs; three CUDA query lanes plus three original CPU lanes."""
import os,sys,time,json,math,statistics,csv,io,traceback,multiprocessing as mp
from pathlib import Path
import control as c
base=None
backend=None
def setup():
    global base,backend
    base=c.module(c.M/'owner.py','priority_frozen_original_MI')
    base.load();base.load_rows()
    def forbid(*a,**kw):raise RuntimeError('Old MI publication forbidden')
    base.publish=forbid;base.put=forbid
    sys.path.insert(0,str(c.B/'ita_gpu_exact_20261029_v1/runtime_deps'))
    backend=c.module(c.B/'ksg_stability_20261030_v2/source/ksg_stability_cuda_v2.py','priority_frozen_exact_CUDA')
def prepare(q):
    setup();c.status('preparing_identical_full_cohort_pools',old_completed=c.read(c.R/'mi_handoff.json')['N_completed'])
    p=base.build_pools(save=False)
    old=base.checked(base.read(c.M/'prepared.json'))
    expected={(v['seed'],v['domain'],v['modality']):v for v in old['pools']}
    for v in p['pools']:
        z=expected[v['seed'],v['domain'],v['modality']]
        c.need(v['shape']==z['shape'] and v['sha256']==z['sha256'],'Rebuilt pool differs from original running computation')
    c.need(p['features']==old['features'] and p['positions']==old['positions'],'Feature/order provenance differs')
    c.put(c.R/'prepared.json',dict(time=c.now(),groups=base.N,pools=p['pools'],features_exact_original=True,original_prepared=c.desc(c.M/'prepared.json'),elapsed_seconds=p['elapsed_seconds']))
def gpu_estimate(x,y,k,all_queries=False,arrays=False):
    import numpy as np
    ids=np.arange(len(x)) if all_queries else np.unique(np.linspace(0,len(x)-1,min(512,len(x)),dtype=np.int64))
    return backend.estimate_ksg_gpu(x,y,base.CTX['est'],k=k,device=0,workers=8,chunk_size=32768,
        validation_query_indices=ids,return_arrays=arrays)
def validate_lane(lane,q):
    try:
        os.sched_setaffinity(0,set(range(80+lane*8,88+lane*8)))
        os.environ['CUDA_VISIBLE_DEVICES']=q['rank_uuids'][lane]
        c.put(c.R/'workers'/('validation%d_birth.json'%lane),dict(identity=c.identity(),lane=lane,uuid=q['rank_uuids'][lane]))
        import numpy as np
        from scipy.special import digamma
        rng=np.random.Generator(np.random.PCG64(20261105));checks=[]
        for fixture in range(4):
            x=rng.normal(size=(127,3));y=(.7*x+.3*rng.normal(size=x.shape)) if fixture%2 else rng.normal(size=x.shape)
            joint=np.concatenate([x,y],axis=1);d=np.max(np.abs(joint[:,None]-joint[None,:]),axis=2)
            xx=np.max(np.abs(x[:,None]-x[None,:]),axis=2);yy=np.max(np.abs(y[:,None]-y[None,:]),axis=2)
            for k in base.KS:
                result,diag=gpu_estimate(x,y,k,True,True);arr=diag.pop('arrays')
                radius=np.sort(d,axis=1)[:,k];strict=np.nextafter(radius,0)
                nx=(xx<=strict[:,None]).sum(1);ny=(yy<=strict[:,None]).sum(1)
                oracle=float((digamma(k)+digamma(len(x))-np.mean(digamma(nx)+digamma(ny)))/np.log(2))
                cpu=base.CTX['est'].ksg_mi(x,y,k=k,workers=8)
                c.need(np.array_equal(arr['radii'],radius) and np.array_equal(arr['nx_plus_one'],nx) and np.array_equal(arr['ny_plus_one'],ny),'Independent radius/count oracle differs')
                c.need(result['mi_bits']==cpu['mi_bits']==oracle,'Independent KSG formula differs')
                checks.append(dict(fixture=fixture,k=k,mi_bits=oracle,all_radii_counts_exact=True,diagnostics=diag))
        real=[]
        for d in base.DOMAINS:
            target=base.TARGETS[lane]
            x=base.POOL[base.SEEDS[0],d,'rgb'][:100].reshape(-1,3)
            y=base.POOL[base.SEEDS[0],d,target][:100].reshape(-1,3)
            t=time.monotonic();cpu=base.CTX['est'].ksg_mi(x,y,k=5,workers=8);cpu_s=time.monotonic()-t
            result,diag=gpu_estimate(x,y,5)
            c.need(cpu['mi_bits']==result['mi_bits'],'Real 100-group CPU/GPU result differs')
            real.append(dict(domain=d,target=target,N=100,points=1000,cpu_seconds=cpu_s,diagnostics=diag,CPU_GPU_scalar_exact=True))
        c.put(c.R/'validation'/('gpu%d.json'%lane),dict(time=c.now(),passed=True,identity=c.identity(),uuid=q['rank_uuids'][lane],independent_fixtures=checks,real=real))
    except BaseException:
        c.put(c.R/'failures'/('validation%d.json'%lane),dict(time=c.now(),traceback=traceback.format_exc()));raise
def wait_children(children,phase):
    while any(p.is_alive() for p in children):
        bad=[p for p in children if p.exitcode is not None and p.exitcode!=0]
        if bad:
            for p in children:
                if p.is_alive():p.terminate()
            raise RuntimeError('Worker failed in '+phase)
        c.status(phase,new_raw=len(list((c.R/'raw').glob('*.json'))),old_raw=c.read(c.R/'mi_handoff.json')['N_completed'])
        time.sleep(10)
    for p in children:p.join();c.need(p.exitcode==0,'Worker failed in '+phase)
def validate(q):
    c.status('validating_exact_CUDA_all_three_cards')
    ctx=mp.get_context('fork');children=[ctx.Process(target=validate_lane,args=(i,q)) for i in range(3)]
    for p in children:p.start()
    wait_children(children,'validating_exact_CUDA_all_three_cards')
    for i in range(3):c.need(c.checked(c.read(c.R/'validation'/('gpu%d.json'%i)))['passed'],'GPU validation missing')
    c.need(not c.apps(),'Validation did not release GPUs')
    c.put(c.R/'gpu_gate.json',dict(time=c.now(),passed=True,independent_fixtures=60,real_100group_comparisons=12,all_three_cards=True,records=[c.desc(c.R/'validation'/('gpu%d.json'%i)) for i in range(3)]))
def one(t,lane,q):
    import numpy as np
    key=base.key(t);ident=c.identity();c.put(c.R/'claims'/(key+'.json'),dict(time=c.now(),task=t,identity=ident,request=c.desc(c.R/'request.json')))
    c.put(c.R/'lanes'/('%d.json'%lane),dict(time=c.now(),phase='query',task=t,identity=ident,backend='GPU' if lane<3 else 'CPU'),True)
    ids=base.SELECT[t['percent']]
    x=np.ascontiguousarray(base.POOL[t['seed'],t['domain'],'rgb'][ids,:t['points']].reshape(-1,3))
    y=np.ascontiguousarray(base.POOL[t['seed'],t['domain'],t['target']][ids,:t['points']].reshape(-1,3))
    xh,yh=base.ah(x),base.ah(y);start=time.monotonic();reused=None;diag=None
    oldpath=base.compatible_old(t)
    if oldpath is not None:
        old=base.checked(base.read(oldpath));c.need(old['input_x_sha256']==xh and old['input_y_sha256']==yh,'Old reusable inputs differ')
        c.need(old['pixel_seed']==t['seed'] and old['percent']==t['percent'] and old['N_groups']==len(ids),'Old reuse scope differs')
        c.need(old['request']==base.desc(base.G/'request.json') and old['selection']==base.desc(base.G/'selections'/('20261102_%d.json'%t['percent'])),'Old reuse binding differs')
        result=old['result'];reused=c.desc(oldpath)
    elif lane<3:
        result,diag=gpu_estimate(x,y,t['k'])
        c.need(diag['cuda_executed'] and diag['cpu_validation']['passed'] and diag['cpu_validation']['full_reference_used'],'Missing full-reference query validation')
        c.need(diag['arrays_sha256']['x']==xh and diag['arrays_sha256']['y']==yh,'CUDA inputs differ')
    else:result=base.CTX['est'].ksg_mi(x,y,k=t['k'],workers=8)
    base.validate_result(result,t,len(x))
    c.put(c.R/'raw'/(key+'.json'),dict(time=c.now(),task=t,result=result,identity=ident,lane=lane,input_x_sha256=xh,input_y_sha256=yh,
        original_request=c.desc(c.M/'request.json'),request=c.desc(c.R/'request.json'),prepared=c.desc(c.M/'prepared.json'),
        selection=c.desc(c.M/'selections'/('%d.json'%t['percent'])),reused_from=reused,diagnostics=diag,seconds=time.monotonic()-start))
def lane_run(lane,queue,q):
    try:
        os.sched_setaffinity(0,set(range(80+lane*8,88+lane*8)));os.environ['CUDA_VISIBLE_DEVICES']=q['rank_uuids'][lane] if lane<3 else ''
        c.put(c.R/'workers'/('production%d_birth.json'%lane),dict(identity=c.identity(),lane=lane,backend='GPU' if lane<3 else 'CPU'))
        while True:
            t=queue.get()
            if t is None:break
            one(t,lane,q)
        c.put(c.R/'workers'/('production%d_complete.json'%lane),dict(time=c.now(),identity=c.identity()))
    except BaseException:
        c.put(c.R/'failures'/('production%d.json'%lane),dict(time=c.now(),traceback=traceback.format_exc()));raise
def merged_raw():
    result={}
    for d in c.checked(c.read(c.R/'mi_handoff.json'))['completed']:
        c.need(c.sha(d['path'])==d['sha256'],'Preserved original result changed')
        p=Path(d['path']);v=base.checked(base.read(p));key=base.key(v['task'])
        c.need(key not in result,'Duplicate historical result');result[key]=(v,d)
    for p in (c.R/'raw').glob('*.json'):
        v=c.checked(c.read(p));key=base.key(v['task']);c.need(key not in result,'Duplicate GPU/CPU handoff result');result[key]=(v,c.desc(p))
    return result
def aggregate():
    import numpy as np
    raw=merged_raw();plan=base.tasks();c.need(set(raw)=={base.key(t) for t in plan},'Missing/extra raw tasks')
    pools={(p['seed'],p['domain'],p['modality']):p['sha256'] for p in c.read(c.R/'prepared.json')['pools']}
    cells={};gpu=cpu=reuse=0
    for t in plan:
        v,d=raw[base.key(t)];base.validate_result(v['result'],t,t['N_groups']*t['points'])
        c.need(v['task']==t and v['prepared']==c.desc(c.M/'prepared.json') and v['selection']==c.desc(c.M/'selections'/('%d.json'%t['percent'])),'Raw scientific binding differs')
        old=Path(d['path']).parent.parent==c.M
        if old:
            c.need(v['request']==c.desc(c.M/'request.json'),'Old request differs');claim=base.checked(base.read(c.M/'claims'/Path(d['path']).name))
        else:
            c.need(v['request']==c.desc(c.R/'request.json') and v['original_request']==c.desc(c.M/'request.json'),'New request differs');claim=c.checked(c.read(c.R/'claims'/Path(d['path']).name))
        c.need(claim['task']==t and claim['identity']==v['identity'],'Claim identity differs')
        if t['percent']==100 and t['points']==1000:
            c.need(v['input_x_sha256']==pools[t['seed'],t['domain'],'rgb'] and v['input_y_sha256']==pools[t['seed'],t['domain'],t['target']],'Full endpoint pool differs')
        if v.get('diagnostics'):
            diag=v['diagnostics'];c.need(diag['cpu_validation']['passed'] and len(diag['cpu_validation']['query_indices'])==512,'GPU query coverage differs');gpu+=1
        elif v['reused_from'] is not None:reuse+=1
        else:cpu+=1
        key=(t['domain'],t['target'],t['percent'],t['points'],t['k'])
        if key in cells:continue
        records=[raw[base.key(dict(t,seed=s))] for s in base.SEEDS];values=[r[0]['result']['mi_bits'] for r in records]
        mean=statistics.mean(values);sd=statistics.stdev(values)
        c.need(abs(mean-np.mean(values))<1e-13 and abs(sd-np.std(values,ddof=1))<1e-13,'Independent mean/SD mismatch')
        cells[key]=dict(domain=t['domain'],target=t['target'],percent=t['percent'],points=t['points'],k=t['k'],N_groups=t['N_groups'],values=values,mean=mean,sample_SD=sd,seeds=list(base.SEEDS),SD_definition='pixel sampling; ddof=1; fixed sample groups',raw=[r[1] for r in records])
    table=[cells[d,m,100,n,5] for m in base.TARGETS for d in base.DOMAINS for n in base.POINTS]
    fraction=[cells[d,m,p,1000,5] for m in base.TARGETS for d in base.DOMAINS for p in base.FRACTIONS]
    kval=[cells[d,m,100,1000,k] for m in base.TARGETS for d in base.DOMAINS for k in base.KS]
    for d in base.DOMAINS:
        for m in base.TARGETS:
            anchor=cells[d,m,100,1000,5];c.need(anchor in table and anchor in fraction and anchor in kval,'Shared mean/SD endpoint differs')
    c.req()
    summary=dict(time=c.now(),status='complete',N=base.N,unique_cells=list(cells.values()),table1A=table,group_fraction=fraction,k_sensitivity=kval,backend_counts=dict(GPU=gpu,CPU=cpu,reused=reuse),request=c.desc(c.R/'request.json'))
    c.put(c.R/'summary.json',summary)
    for name,rows in [('Table1A_full_precision',table),('group_fraction_pixel_SD',fraction),('k_sensitivity_pixel_SD',kval)]:
        s=io.StringIO(newline='');fields=['domain','target','percent','points','k','N_groups','mean','sample_SD','SD_definition'];writer=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
        p=c.R/'tables'/(name+'.csv');p.parent.mkdir(exist_ok=True);p.write_text(s.getvalue(),encoding='utf-8')
    rows=[['RGB–'+m.upper(),d]+['%.3f ± %.3f'%(cells[d,m,100,n,5]['mean'],cells[d,m,100,n,5]['sample_SD']) for n in base.POINTS] for m in base.TARGETS for d in base.DOMAINS]
    c.put(c.R/'tables/Table1A_native_cells.json',dict(columns=['Modality pair','Region','100 points','500 points','1000 points'],rows=rows,caption='All19,651 eligible TEST groups;2,248 original exclusions unchanged. Three pixel seeds; sampleSDddof1. UV-defined P/D/O/F supports for all three targets.'))
    c.put(c.R/'acceptance.json',dict(time=c.now(),status='passed',raw=396,cells=132,source_preserved=True,old_records_unchanged=True,shared_mean_and_SD_endpoint=True,GPU_exact_full_reference_spot_checks=True,summary=c.desc(c.R/'summary.json')))
    c.put(c.R/'completion.json',dict(time=c.now(),status='complete',acceptance=c.desc(c.R/'acceptance.json')))
def production(q):
    previous=merged_raw();todo=[t for t in base.tasks() if base.key(t) not in previous]
    c.put(c.R/'remaining_plan.json',dict(time=c.now(),tasks=todo,old_completed=len(previous),total=396,nonoverlapping=True))
    ctx=mp.get_context('fork');queue=ctx.Queue()
    for t in todo:queue.put(t)
    for _ in range(6):queue.put(None)
    children=[ctx.Process(target=lane_run,args=(i,queue,q)) for i in range(6)]
    for p in children:p.start()
    wait_children(children,'GPU_CPU_production');queue.close();base.POOL.clear()
    c.need(not c.apps(),'MI production resources not released');c.status('final_source_and_arithmetic_audit');aggregate()
