"""Versioned full-cohort spatial extension: frozen saved masks and equations.

No model, detector, training, old-result rewrite, or desktop automation.
"""
import argparse,copy,csv,hashlib,io as binaryio,json,math,os,subprocess,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy

B=Path('/study/siiba');BASE=B/'experiments/signal_objects_3cd_20261029_v1'
ROOT=B/'experiments/signal_objects_3d_full_20261030_v1'
PREVIOUS=B/'experiments/signal_objects_3cd_statistics_parallel_20261030_v2'
PY=B/'experiments/mi_atlas_six_20261025_v1/runtime_mask/bin/python'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'uv_objects'))
import io_utils as io
import metrics as m
import collector as c
import validation_metrics as va
import final_audit as original_audit
MODELS=io.MODELS;SIGNALS=('P','D');METRICS=('dice','precision','recall','f1','mean_center_error')
STATE={}

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(io.seal(value),f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
def status(phase,**value):io.save(ROOT/'status.json',dict(time=io.now(),phase=phase,**value),True)
def checked(path):return io.checked(io.read(path))
def desc(path):return io.desc(path)
def verify(d):return io.verify(d)
def context():
    q=checked(ROOT/'request.json')
    for d in q['frozen_sources'].values():verify(d)
    verify(q['source']);verify(q['base_protocol']);verify(q['selection']);verify(q['prior_numeric_acceptance'])
    assert np.__version__=='2.3.4' and scipy.__version__=='1.16.2'
    assert m.TOLERANCES==(2,5,10) and m.RANDOM_REPEATS==10 and m.RANDOM_SEED==20261030
    selection=io.read(verify(q['selection']))
    return q,selection
def init():
    q,s=context();STATE.update(q=q,selection=s,valid=set(s['geometry_valid_indices']),old=set(s['spatial_indices']),qc=set(q['independent_qc_indices']),base_protocol=io.read(BASE/'protocol.json'))
    os.sched_setaffinity(0,q['cpu_affinity']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
def npz(d,keys):
    p=Path(d['path']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==d['sha256'],str(p)
    with np.load(binaryio.BytesIO(data),allow_pickle=False) as z:return {k:z[k] for k in keys}
def compare_nested(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict)
        for k in a:
            if k=='pairs':continue
            assert k in b;compare_nested(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare_nested(x,y,path+'/'+str(i))
    elif isinstance(a,float):assert isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),path
    else:assert a==b,path

def group_work(index,validation=False):
    if not STATE:init()
    q,s=STATE['q'],STATE['selection'];gid=s['all_group_ids'][index];start=time.monotonic()
    out=ROOT/'records'/('%06d.json'%index)
    assert not out.exists(),'Existing record must be independently inspected, never duplicated'
    save(ROOT/'claims'/('%06d.json'%index),dict(time=io.now(),index=index,identity=io.identity(),request=desc(ROOT/'request.json')))
    rows={mid:checked(BASE/('records/%s/%06d.json'%(mid,index))) for mid in MODELS}
    old_descriptors={mid:desc(BASE/('records/%s/%06d.json'%(mid,index))) for mid in MODELS}
    for mid,v in rows.items():
        assert (v['index'],v['group_id'],v['model'])==(index,gid,mid)
        assert v['protocol_sha256']==q['base_protocol']['sha256']
    if index not in STATE['valid']:
        reason=next(x['reason'] for x in s['missing'] if x['index']==index)
        assert all(v['status']=='unavailable_geometry' and v['missing_reason']==reason and v['forward_count']==0 for v in rows.values())
        value=dict(time=io.now(),index=index,group_id=gid,status='unavailable_geometry',missing_reason=reason,models={},
                   original_records=old_descriptors,request=desc(ROOT/'request.json'),elapsed_seconds=time.monotonic()-start)
        save(out,value);return dict(index=index,valid=False,reused=False,seconds=value['elapsed_seconds'])
    folder=BASE/('staged/%06d'%index);ready=checked(folder/'ready.json');ready_desc=desc(folder/'ready.json')
    assert ready['index']==index and ready['group_id']==gid and ready['protocol_sha256']==q['base_protocol']['sha256']
    geo=npz(ready['geometry'],m.ROI_ORDER);truth_detector=checked(verify(rows[MODELS[0]]['truth_detector']))
    truth=npz(truth_detector['arrays'],('P','D','F'));f=truth['F'];assert f.dtype==np.bool_ and f.shape==(1024,1024) and f.any()
    domains=m.make_roi_domains(f,geo)
    recompute=index not in STATE['old'] or validation
    truth_objects={sig:m.extract_objects(truth[sig],f_mask=f,domains=domains) for sig in SIGNALS} if recompute else {}
    facts={};input_arrays={'truth':truth_detector['arrays'],'geometry':ready['geometry']};audits={}
    for mid,record in rows.items():
        assert record['status']=='ok' and record['input_binding']==ready_desc
        assert record['truth_detector']==rows[MODELS[0]]['truth_detector'] and record['sources']==STATE['base_protocol']['sources']
        assert record['prediction']==ready['outputs'][mid] and record['prediction']['forward_count']==1
        pred=npz(record['predicted_signals'],SIGNALS);input_arrays[mid]=record['predicted_signals'];facts[mid]={}
        for sig in SIGNALS:
            source=record['signals'][sig];t,p=truth[sig],pred[sig]
            assert t.dtype==p.dtype==np.bool_ and t.shape==p.shape==f.shape and not np.any((t|p)&~f)
            if recompute:
                po=m.extract_objects(p,f_mask=f,domains=domains)
                assert po['count']==source['prediction']['count'] and truth_objects[sig]['count']==source['truth']['count']
                sp=m.spatial_metrics(t,p,f,geo,index,sig,truth_objects[sig],po,include_pairs=False)
                if index in STATE['old']:compare_nested(sp,source['spatial'])
            else:sp=source['spatial']
            c.validate_spatial(sp)
            original_audit.spatial_coherence(sp,t,p,source['truth']['count'],source['prediction']['count'])
            if validation or index in STATE['qc']:
                independent=dict(source,spatial=sp)
                audits[mid+'_'+sig]=va.audit_signal_arrays(t,p,f,geo,independent,index,sig,check_matching=True)
            facts[mid][sig]=dict(spatial=sp,truth_count=source['truth']['count'],prediction_count=source['prediction']['count'],
                count_abs_error=abs(source['prediction']['count']-source['truth']['count']),
                area_abs_error=abs(source['prediction']['area_fraction']-source['truth']['area_fraction']))
    value=dict(time=io.now(),index=index,group_id=gid,status='ok',missing_reason=None,models=facts,original_records=old_descriptors,
               original_ready=ready_desc,saved_arrays=input_arrays,reused_prespecified_spatial=index in STATE['old'],
               recomputed_for_validation=bool(validation),independent_qc=audits,request=desc(ROOT/'request.json'),
               elapsed_seconds=time.monotonic()-start,identity=io.identity())
    save(out,value);return dict(index=index,valid=True,reused=index in STATE['old'],seconds=value['elapsed_seconds'],independent_audits=len(audits))

def configure():
    assert not (ROOT/'request.json').exists()
    protocol=io.read(BASE/'protocol.json');s=io.read(verify(protocol['selection']))
    assert (len(s['all_group_ids']),len(s['geometry_valid_indices']),len(s['spatial_indices']),len(s['missing']))==(21899,21422,1000,477)
    accept=io.read(PREVIOUS/'report/independent_audit_v1/acceptance.json');assert accept['status']=='passed' and accept['model_records']==109495
    rng=np.random.default_rng(20261030);old=sorted(s['spatial_indices']);new=sorted(set(s['geometry_valid_indices'])-set(old))
    pilot_old=sorted(map(int,rng.choice(old,4,replace=False)));qc_new=sorted(map(int,rng.choice(new,32,replace=False)))
    frozen={k:protocol['sources'][k] for k in ('metrics.py','collector.py','io_utils.py','validation_metrics.py','final_audit.py')}
    for d in frozen.values():verify(d)
    save(ROOT/'request.json',dict(time=io.now(),authorization='Human2026-10-30 explicitly requests full3D and final fullcohort table with STD',
        source=desc(__file__),frozen_sources=frozen,base_protocol=desc(BASE/'protocol.json'),selection=protocol['selection'],
        prior_numeric_acceptance=desc(PREVIOUS/'report/independent_audit_v1/acceptance.json'),prior_summary=desc(PREVIOUS/'report/summary.json'),
        host_identity=io.host_identity(),workers=40,cpu_affinity=list(range(48)),pilot_indices=pilot_old+qc_new[:4],
        independent_qc_indices=pilot_old+qc_new,full_catalog=21899,geometry_valid=21422,geometry_missing=477,
        original1000_reused=True,original1000_preserved=True,new_spatial_groups=20422,no_training=True,no_model_or_detector_execution=True,
        matching=m.runtime_definition(),bootstrap_repeats=2000,bootstrap_seed=20261029,
        statistics='original mean_summary, paired_test and original Holm family sizes80/20/10; old scalar errors reused for family accounting',
        presentation='Final main Table3D uses full-cohort extension as authorized; original prespecified1000 result retained and scope expansion transparent',
        independent_audit='allgroup bytehash/recordseal/count/Dice/null/coherence;36fixedQC full independent CC/matching/RNG; independent streaming summaries',
        disk='only new JSON/CSV/summary/plots; no copies of prediction tensors or masks'))
    print('CONFIGURED_FULL3D_21422_VALID_20422_NEW_1000_REUSED')

def validate():
    q,s=context();assert not (ROOT/'validation.json').exists();init();results=[];start=time.monotonic()
    for index in q['pilot_indices']:results.append(group_work(index,validation=True))
    elapsed=time.monotonic()-start
    save(ROOT/'validation.json',dict(time=io.now(),passed=True,request=desc(ROOT/'request.json'),groups=results,
         real_groups=8,model_signal_audits=80,elapsed_seconds=elapsed,mean_group_seconds=elapsed/8,
         projected_compute_seconds_40workers=(sum(x['seconds'] for x in results)/8)*20422/40,
         original1000_spatial_reproduction=True,no_inference=True))
    print('REAL_END_TO_END_VALIDATION_PASSED',elapsed,'seconds',flush=True)

def independent_mean(vals):
    a=[float(x) for x in vals if x is not None];n=len(a);mu=math.fsum(a)/n if n else None
    return dict(n=n,mean=mu,sample_sd=math.sqrt(math.fsum((x-mu)**2 for x in a)/(n-1)) if n>1 else None)
def summarize_model(mid):
    q,s=context();values={sig:{'spatial':[],'groups':[]} for sig in SIGNALS};missing={};manifest=[]
    comparative={sig:{k:{} for k in ('count_abs_error','area_abs_error','dice','f1','random_f1')} for sig in SIGNALS}
    for index,gid in enumerate(s['all_group_ids']):
        p=ROOT/'records'/('%06d.json'%index);v=checked(p);assert v['request']==desc(ROOT/'request.json') and (v['index'],v['group_id'])==(index,gid)
        if v['status']!='ok':missing[v['missing_reason']]=missing.get(v['missing_reason'],0)+1;continue
        for sig in SIGNALS:
            a=v['models'][mid][sig];sp=a['spatial'];values[sig]['spatial'].append(sp);values[sig]['groups'].append(gid)
            d=comparative[sig]
            for key in ('count_abs_error','area_abs_error'):d[key][gid]=a[key]
            d['dice'][gid]=sp['dice'];d['f1'][gid]=sp['by_tolerance']['5']['f1']
            rand=[x['by_tolerance']['5']['f1'] for x in sp['random_repeats'] if x['by_tolerance']['5']['f1'] is not None]
            d['random_f1'][gid]=float(np.mean(rand)) if rand else None
    assert sum(missing.values())==477
    summary={};audit_checks=0
    for sig in SIGNALS:
        entries=values[sig]['spatial'];assert len(entries)==21422;summary[sig]={}
        for tol in ('2','5','10'):
            z={}
            for key in METRICS:
                vals=[sp['dice'] if key=='dice' else sp['by_tolerance'][tol][key] for sp in entries]
                z[key]=c.mean_summary([c.finite_or_nan(x) for x in vals],2000 if tol=='5' else 0,20261029)
                z[key]['n_undefined']=21422-z[key]['n'];a=independent_mean(vals)
                for k in a:
                    if a[k] is None:assert z[key][k] is None
                    else:assert math.isclose(a[k],z[key][k],rel_tol=1e-11,abs_tol=1e-11)
                    audit_checks+=1
            z['correct_negative_groups']=sum(sp['correct_negative'] for sp in entries);z['selected_groups']=21422
            z['tp_fp_fn_pooled_descriptive']={k:sum(sp['by_tolerance'][tol][k] for sp in entries) for k in ('tp','fp','fn')}
            z['conditional_random_centers']={}
            for key in METRICS[1:]:
                vals=[]
                for sp in entries:
                    a=[rr['by_tolerance'][tol][key] for rr in sp['random_repeats'] if rr['by_tolerance'][tol][key] is not None]
                    vals.append(float(np.mean(a)) if a else None)
                z['conditional_random_centers'][key]=c.mean_summary([c.finite_or_nan(x) for x in vals],2000 if tol=='5' else 0,20261029)
                a=independent_mean(vals)
                for k in a:
                    b=z['conditional_random_centers'][key][k]
                    assert (a[k] is None and b is None) or math.isclose(a[k],b,rel_tol=1e-11,abs_tol=1e-11);audit_checks+=1
            summary[sig][tol]=z
    save(ROOT/'models'/(mid+'.json'),dict(model=mid,spatial=summary,coverage=dict(all=21899,valid=21422,missing_reasons=missing),
        comparisons_input=comparative,independent_summary_checks=audit_checks,request=desc(ROOT/'request.json')))
    return mid

def collect():
    q,s=context();status('full_cohort_statistics_and_independent_summary_checks',completed_groups=21899)
    with ProcessPoolExecutor(max_workers=5) as pool:list(pool.map(summarize_model,MODELS))
    models={mid:checked(ROOT/'models'/(mid+'.json')) for mid in MODELS}
    comparisons=c.comparisons({mid:v['comparisons_input'] for mid,v in models.items()})
    exported={mid:{k:v[k] for k in ('spatial','coverage','independent_summary_checks')} for mid,v in models.items()}
    save(ROOT/'summary.json',dict(time=io.now(),models=exported,comparisons=comparisons,request=desc(ROOT/'request.json'),
        bootstrap_repeats=2000,bootstrap_seed=20261029,STD='Across sample groups, ddof1; not training-seed or verifiedpatient CI',
        original1000_preserved=True,full_cohort_scope_extension=True))
    sys.path.insert(0,str(B/'experiments/ksg_stability_20261030_v1/runtime_plot'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,6.5));fig.subplots_adjust(left=.02,right=.99,top=.95,bottom=.02);ax.set_axis_off()
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.01,.97,'Table 3D. Full-cohort spatial recovery of UV signal objects',fontsize=15,weight='bold',va='top')
    ax.text(.01,.91,'21,422 geometry-valid groups; five frozen models; 5-pixel matching tolerance on the 1024 grid.',fontsize=10,color='.3')
    xs=[.01,.07,.42,.56,.70,.84,.98];heads=['Signal','Metric','Single U-Net','Multi U-Net','U-Net++','Swin','LDM']
    ax.hlines(.86,.01,.99,color='black',linewidth=1.5)
    for i,h in enumerate(heads):ax.text(xs[i],.825,h,fontsize=10,weight='bold',ha='left' if i<2 else 'right')
    ax.hlines(.805,.01,.99,color='black',linewidth=.8)
    y=.755
    for sig in SIGNALS:
        for key,label in [('dice','Dice'),('precision','Object precision'),('recall','Object recall'),('f1','Object F1'),('mean_center_error','Center error, px (N)')]:
            ax.text(xs[0],y,sig,fontsize=10);ax.text(xs[1],y,label,fontsize=10)
            for j,mid in enumerate(MODELS):
                v=exported[mid]['spatial'][sig]['5'][key]
                label='NA' if v['mean'] is None else '%.3f ± %.3f'%(v['mean'],v['sample_sd'])
                if key=='mean_center_error':label+=' (%d)'%v['n']
                ax.text(xs[j+2],y,label,fontsize=10,ha='right')
            y-=.054
    ax.hlines(y+.018,.01,.99,color='black',linewidth=1.5)
    ax.text(.01,y-.025,'Mean ± across-group sample SD (ddof=1). Own endpoint N/95% CIs and 2/10-pixel sensitivity are in the supplement.',fontsize=9,color='.3')
    ax.text(.01,y-.065,'Center error is conditional on successful matches and must be read with recall/F1. Algorithm-defined objects; no clinical tolerance.',fontsize=9,color='.3')
    fig.savefig(ROOT/'table3d_full_with_SD.png',dpi=240);fig.savefig(ROOT/'table3d_full_with_SD.pdf');plt.close(fig)
    with (ROOT/'spatial_values.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['index','group_id','model','signal','tolerance','dice','tp','fp','fn','precision','recall','f1','mean_center_error','reused1000'])
        for index,gid in enumerate(s['all_group_ids']):
            v=checked(ROOT/'records'/('%06d.json'%index))
            if v['status']!='ok':continue
            for mid in MODELS:
                for sig in SIGNALS:
                    sp=v['models'][mid][sig]['spatial']
                    for tol in ('2','5','10'):
                        z=sp['by_tolerance'][tol];w.writerow([index,gid,mid,sig,tol,sp['dice']]+[z[k] for k in ('tp','fp','fn','precision','recall','f1','mean_center_error')]+[v['reused_prespecified_spatial']])
    manifest=[];qc_count=0;reused=0
    for index in range(21899):
        p=ROOT/'records'/('%06d.json'%index);v=checked(p);manifest.append(desc(p));reused+=int(v.get('reused_prespecified_spatial',False));qc_count+=len(v.get('independent_qc',{}))
    assert reused==1000 and qc_count==360
    save(ROOT/'record_manifest.json',dict(files=manifest))
    save(ROOT/'numerical_acceptance.json',dict(time=io.now(),status='passed',all_catalog_groups=21899,valid_groups=21422,
        unavailable_groups=477,reused1000=reused,new_spatial_groups=20422,independent_QC_model_signals=qc_count,
        independent_summary_checks=sum(v['independent_summary_checks'] for v in models.values()),summary=desc(ROOT/'summary.json'),
        record_manifest=desc(ROOT/'record_manifest.json'),paired_values=desc(ROOT/'spatial_values.csv'),source=desc(__file__),
        primary_1000_preserved=True,no_model_or_detector_execution=True))
    status('waiting_prior_final_provenance_acceptance',completed_groups=21899)
    while not (PREVIOUS/'scientific_completion.json').exists():
        assert not list(PREVIOUS.glob('array_audit_failure*')),'Prior provenance failed; preserve and stop'
        time.sleep(30)
    prior=io.read(PREVIOUS/'scientific_completion.json');assert prior['status']=='complete'
    save(ROOT/'completion.json',dict(time=io.now(),status='complete',numerical=desc(ROOT/'numerical_acceptance.json'),
        prior_scientific_completion=desc(PREVIOUS/'scientific_completion.json'),summary=desc(ROOT/'summary.json'),
        table_png=desc(ROOT/'table3d_full_with_SD.png'),table_pdf=desc(ROOT/'table3d_full_with_SD.pdf'),all_old_results_preserved=True))
    status('complete',completed_groups=21899)

def run():
    q,s=context();assert checked(ROOT/'validation.json')['passed'];status('production_saved_mask_spatial_extension')
    init();todo=iter(i for i in range(21899) if i not in q['pilot_indices']);done=len(q['pilot_indices']);times=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=q['workers'],initializer=init) as pool:
        pending={}
        for _ in range(q['workers']*2):
            i=next(todo,None)
            if i is not None:pending[pool.submit(group_work,i)]=i
        while pending:
            finished,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                index=pending.pop(future);v=future.result();assert v['index']==index;done+=1;times.append(v['seconds'])
                if done%50==0:status('production_saved_mask_spatial_extension',completed_groups=done,total_groups=21899,elapsed_seconds=time.monotonic()-start,
                    measured_groups_per_second=(done-len(q['pilot_indices']))/(time.monotonic()-start))
                i=next(todo,None)
                if i is not None:pending[pool.submit(group_work,i)]=i
    assert done==21899;collect()

def launch():
    q,s=context();assert checked(ROOT/'validation.json')['passed'];assert io.host_identity()==q['host_identity']
    save(ROOT/'launch_reservation.json',dict(time=io.now(),request=desc(ROOT/'request.json'),source=desc(__file__)))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (ROOT/'owner.log').open('xb') as log:
        p=subprocess.Popen([str(PY),'-u','-B',str(Path(__file__).resolve()),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
    time.sleep(.4);assert p.poll() is None
    save(ROOT/'launch_receipt.json',dict(time=io.now(),identity=io.identity(p.pid),request=desc(ROOT/'request.json'),source=desc(__file__)))
    print('DETACHED_FULL3D',p.pid,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['configure','validate','launch','run']);args=a.parse_args()
    try:{'configure':configure,'validate':validate,'launch':launch,'run':run}[args.action]()
    except BaseException:
        save(ROOT/'failures'/('%s_%d.json'%(args.action,os.getpid())),dict(time=io.now(),identity=io.identity(),traceback=traceback.format_exc()));raise
