import os,sys,time,math,csv,statistics,traceback,subprocess
from pathlib import Path
R=Path('/study/siiba/experiments/pl_controls_original_uv_pipeline_20261031_v1');sys.path.insert(0,str(R/'source'))
from chain_common import *
from numerics import *


def worker(host,local,phase):
    from cache_io import load_case
    from inference import Original
    q=request();check_host(q,host);g=q['hosts'][host]['gpus'][local];os.sched_setaffinity(0,[80+2*local,81+2*local]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES=g['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    job=checked(read(R/'jobs/conformal/manifest.json'));plan=checked(read(verify(job['plan'])));split=checked(read(verify(q['conformal_policy']['split'])));s=read(verify(q['selection']));stage='cal' if phase=='conformal_cal' else 'eval';indices=split['calibration_indices' if stage=='cal' else 'evaluation_indices'];assert len(indices)==(10949 if stage=='cal' else 10950)
    calfile=R/'conformal/calibration_frozen.json';calibrations=checked(read(calfile))['models'] if stage=='eval' else None
    if stage=='eval':assert checked(read(calfile))['split']==q['conformal_policy']['split']
    original=Original(q,g['uuid']);engine=GPU();done=0;start=time.monotonic()
    # 逐卡复测与CPU核分

    for mid in MODELS:
        no_failures();pilot=plan['validation_pilots'][g['uuid']];c=load_case(pilot,q,s);assert c['missing_reason'] is None;rgb,row,ready=original.rgb(c);p,sg=original.predict(mid,rgb,row);p2,sg2=original.predict(mid,rgb,row);assert np.array_equal(p,p2) and np.array_equal(sg,sg2)
        a,_=engine.compute(c['truth'],p,sg,c['weights']);b=cpu_scores(c['truth'],p,sg,c['weights']);diagnostics=[]
        for r in REGIONS:
            for key in ('normalized_score','absolute_score','area','mean_absolute_error','mean_sigma'):
                x,y=b['regions'][r][key],a['regions'][r][key];assert (x is None)==(y is None)
                if x is not None:assert math.isfinite(x) and math.isfinite(y)
                diagnostics.append(dict(region=r,metric=key,cpu=x,cuda=y,absolute_difference=abs(x-y) if x is not None else None))
        fake=dict(pixel_sigma=finite_threshold([a['normalized']]*40),pixel_absolute=finite_threshold([a['absolute']]*40));_,av=engine.compute(c['truth'],p,sg,c['weights'],fake);bv=cpu_evaluate(c['truth'],p,sg,c['weights'],fake)
        for r in REGIONS:
            for aa,bb in zip(av[r],bv[r]):
                for key in ('coverage','width'):
                    x,y=bb[key],aa[key];assert (x is None)==(y is None)
                    diagnostics.append(dict(region=r,metric=aa['method']+'_'+key,cpu=x,cuda=y,absolute_difference=abs(x-y) if x is not None else None))
        cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);delta=np.abs(p.astype(np.float64)-cached.astype(np.float64))
        save(R/('conformal/validation/'+stage+'_'+host+'_'+str(local)+'_'+mid+'.json'),dict(status='passed',time=now(),identity=identity(),uuid=g['uuid'],model=mid,pilot=pilot,same_card_repeat_exact=True,score_metric_diagnostics=diagnostics,CPU_model_forward_performed=False,CPU_score_program_on_actual_CUDA_tensors=True,cached_mean_max_abs_difference=float(delta.max()),cached_mean_mean_abs_difference=float(delta.mean()),original_program_runtime=original.runtime,checkpoint_sha256=q['checkpoint_sha256'][mid],job=desc(R/'jobs/conformal/manifest.json')))
    for i in indices:
        no_failures();needed=[mid for mid in MODELS if not (R/('conformal/'+stage+'/records/'+mid+'/%06d.json.gz'%i)).exists()]
        if not needed:continue
        c=None;rgb=row=ready=None
        for mid in needed:
            out=R/('conformal/'+stage+'/records/'+mid+'/%06d.json.gz'%i)
            if not claim(R/('conformal/'+stage+'/claims/'+mid+'_%06d.json'%i),dict(time=now(),index=i,model=mid,uuid=g['uuid'],identity=identity(),job=desc(R/'jobs/conformal/manifest.json'))):continue
            if c is None:
                c=load_case(i,q,s)
                if c['missing_reason'] is None:rgb,row,ready=original.rgb(c)
            if c['missing_reason'] is None:
                t=time.monotonic();pred,sigma=original.predict(mid,rgb,row);scores,ev=engine.compute(c['truth'],pred,sigma,c['weights'],calibrations[mid] if calibrations else None);record=BASE/('records/'+mid+'/%06d.json'%i);old=checked(read(record));assert old['index']==i and old['group_id']==c['group_id'] and old['status']=='ok';tasks=task_values(old,scores['regions']['F']['mean_sigma']);tev=task_evaluate(tasks,scores['regions']['F']['mean_sigma'],calibrations[mid]) if calibrations else None
                cached=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False);d=np.abs(pred.astype(np.float64)-cached.astype(np.float64));mu_diag=dict(max_abs_difference=float(d.max()),mean_abs_difference=float(d.mean()),cached_prediction=ready['outputs'][mid]['prediction'],new_prediction_tensor_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),new_sigma_tensor_sha256=hashlib.sha256(sigma.tobytes()).hexdigest(),sigma_hash_equal_to_historical=hashlib.sha256(sigma.tobytes()).hexdigest()==ready['outputs'][mid]['sigma_diagnostic']['sha256'])
                fields=dict(status='ok',forward_count=1,scores=scores,tasks=tasks,task_record=desc(record),pixel_evaluation=ev if calibrations else None,task_evaluation=tev,mean_numerical_diagnostics=mu_diag,elapsed_seconds=time.monotonic()-t)
                if i in plan[stage+'_QC_indices']:
                    p=R/('conformal/QC/'+stage+'_'+mid+'_%06d.npz'%i);p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('xb') as f:np.savez_compressed(f,truth=c['truth'],prediction=pred,sigma=sigma,**c['weights'])
            else:fields=dict(status='unavailable_geometry',forward_count=0,scores=None,tasks=None,pixel_evaluation=None,task_evaluation=None)
            save(out,dict(index=i,group_id=c['group_id'],model=mid,phase=stage,missing_reason=c['missing_reason'],inputs=c['inputs'],uuid=g['uuid'],identity=identity(),checkpoint_sha256=q['checkpoint_sha256'][mid],calibration=desc(calfile) if calibrations else None,request_sha256=sha(R/'request.json'),job=desc(R/'jobs/conformal/manifest.json'),time=now(),**fields));done+=1
            if done%5==0:touch_progress('conformal_'+stage+'_'+host+'_'+str(local),'conformal_'+stage,records=done,elapsed_seconds=time.monotonic()-start,last_index=i,last_model=mid)
    # 仅记本卡完成
    save(R/('conformal/'+stage+'/card_completion/'+host+'_'+str(local)+'.json'),dict(status='complete',time=now(),records=done,identity=identity(),uuid=g['uuid']))


def load_records(stage,split,q,s):
    data={};manifest=[];missing={};allids=set(split['calibration_indices' if stage=='cal' else 'evaluation_indices']);valid=set(s['geometry_valid_indices'])
    for mid in MODELS:
        records={}
        for i in sorted(allids):
            p=R/('conformal/'+stage+'/records/'+mid+'/%06d.json.gz'%i);v=checked(read(p));assert v['index']==i and v['group_id']==s['all_group_ids'][i] and v['phase']==stage and v['model']==mid and v['request_sha256']==sha(R/'request.json') and v['checkpoint_sha256']==q['checkpoint_sha256'][mid];assert v['job']==desc(R/'jobs/conformal/manifest.json');manifest.append(desc(p))
            if i in valid:
                assert v['status']=='ok' and v['forward_count']==1;assert set(v['tasks'])==set(TASKS) and set(v['scores']['regions'])==set(REGIONS);assert math.isfinite(v['scores']['normalized']) and math.isfinite(v['scores']['absolute']);records[i]=v
                if stage=='eval':assert v['calibration']==desc(R/'conformal/calibration_frozen.json')
            else:assert v['status']=='unavailable_geometry' and v['forward_count']==0;key=mid+':'+v['missing_reason'];missing[key]=missing.get(key,0)+1
        assert len(records)==len(allids&valid);data[mid]=records
    return data,manifest,missing


def freeze_calibration(data,q,manifest,missing):
    models={}
    for mid,values in data.items():
        records=list(values.values());tasks={}
        for name in TASKS:
            ts=[v['tasks'][name] for v in records];tasks[name]=dict(constant_width=finite_threshold([v['error'] for v in ts if v['error'] is not None]),sigma_F_proxy=finite_threshold([v['normalized_error'] for v in ts if v['normalized_error'] is not None]),n_missing=sum(v['error'] is None for v in ts))
        models[mid]=dict(pixel_sigma=finite_threshold([v['scores']['normalized'] for v in records]),pixel_absolute=finite_threshold([v['scores']['absolute'] for v in records]),tasks=tasks)
    save(R/'conformal/calibration_record_manifest.json',dict(files=manifest));save(R/'conformal/calibration_frozen.json',dict(status='frozen_before_EVAL',time=now(),models=models,split=q['conformal_policy']['split'],request=desc(R/'request.json'),job=desc(R/'jobs/conformal/manifest.json'),calibration_records=desc(R/'conformal/calibration_record_manifest.json'),missing_reasons=missing,EVAL_records_at_freeze=sum(len(list((R/('conformal/eval/records/'+m)).glob('*.json.gz'))) for m in MODELS),retrospective=True))


def qc(stage):
    from numerics import cpu_scores,cpu_evaluate
    q=request();job=checked(read(R/'jobs/conformal/manifest.json'));plan=checked(read(verify(job['plan'])));cal=checked(read(R/'conformal/calibration_frozen.json'))['models'] if stage=='eval' else None;results=[]
    for mid in MODELS:
        for i in plan[stage+'_QC_indices']:
            p=R/('conformal/QC/'+stage+'_'+mid+'_%06d.npz'%i)
            with np.load(p,allow_pickle=False) as z:
                a=cpu_scores(z['truth'],z['prediction'],z['sigma'],{r:z[r] for r in REGIONS});ev=cpu_evaluate(z['truth'],z['prediction'],z['sigma'],{r:z[r] for r in REGIONS},cal[mid]) if cal else None
            v=checked(read(R/('conformal/'+stage+'/records/'+mid+'/%06d.json.gz'%i)));ds=[]
            for r in REGIONS:
                for key in ('normalized_score','absolute_score','area','mean_absolute_error','mean_sigma'):
                    x,y=a['regions'][r][key],v['scores']['regions'][r][key];assert (x is None)==(y is None)
                    if x is not None:assert math.isfinite(x) and math.isfinite(y)
                    ds.append(dict(region=r,metric=key,cpu=x,cuda=y,absolute_difference=abs(x-y) if x is not None else None))
                if ev:
                    for aa,bb in zip(ev[r],v['pixel_evaluation'][r]):
                        for k in ('coverage','width'):
                            x,y=aa[k],bb[k];assert (x is None)==(y is None);ds.append(dict(region=r,metric=aa['method']+'_'+k,cpu=x,cuda=y,absolute_difference=abs(x-y) if x is not None else None))
            results.append(dict(index=i,model=mid,arrays=desc(p),diagnostics=ds))
    save(R/('conformal/evidence/'+stage+'_independent_CPU_QC.json'),dict(status='passed',time=now(),cases=len(results),results=results,source=desc(__file__),CPU_model_forward_performed=False,CPU_score_recomputed_from_actual_saved_tensors=True))


def statistics_eval(data,cal,q,s):
    ids=sorted(next(iter(data.values())));assert all(set(v)==set(ids) for v in data.values());n=len(ids);cols=[];labels=[];curves=[]
    def add(values,**label):cols.append(values);labels.append(label)
    for mid in MODELS:
        records=[data[mid][i] for i in ids];c=cal[mid]
        for method in METHODS:
            for r in REGIONS:
                for metric in ('coverage','width'):
                    add([next(z for z in v['pixel_evaluation'][r] if z['method']==method)[metric] for v in records],model=mid,analysis='pixel_region',region_or_task=r,method=method,metric=metric)
            scaled=method!='group_conformal_absolute';qv=1.959963984540054 if method==METHODS[0] else threshold(c['pixel_sigma'] if scaled else c['pixel_absolute']);key='normalized' if scaled else 'absolute'
            add([float(v['scores'][key]<=qv) for v in records],model=mid,analysis='joint_group_event',region_or_task='all_nonempty_P_D_O_F',method=method,metric='coverage')
            add([float(all(z['coverage'] is None or z['coverage']>=.95 for r in REGIONS for z in v['pixel_evaluation'][r] if z['method']==method)) for v in records],model=mid,analysis='direct_weighted_coverage_event',region_or_task='all_nonempty_P_D_O_F',method=method,metric='coverage')
        for task in TASKS:
            for method in ('constant_width','sigma_F_proxy'):
                for metric in ('coverage','width'):
                    add([next(z for z in v['task_evaluation'][task] if z['method']==method)[metric] for v in records],model=mid,analysis='task',region_or_task=task,method=method,metric=metric)
        # 正缩放保留排序

        outcomes={r:([v['scores']['regions'][r]['mean_absolute_error'] for v in records],[v['scores']['regions'][r]['mean_sigma'] for v in records]) for r in REGIONS}
        outcomes.update({task:([v['tasks'][task]['error'] for v in records],[v['scores']['regions']['F']['mean_sigma'] for v in records]) for task in TASKS})
        for name,(errors,scores) in outcomes.items():
            ee=np.array(errors,float);ss=np.array(scores,float);curves.extend([dict(model=mid,outcome=name,ranking='Original mean-sigma; global scaling leaves ranking unchanged',**z) for z in risk_curve(ee,ss)])
            # 固定阈值按组重采样

            ok=np.isfinite(ee)&np.isfinite(ss);order=np.where(ok)[0][np.argsort(ss[ok],kind='mergesort')]
            for frac in (1.,.9,.8,.5):
                k=max(1,math.ceil(len(order)*frac)) if len(order) else 0;a=np.full(n,np.nan);a[order[:k]]=ee[order[:k]];add(a.tolist(),model=mid,analysis='selective_risk_fixed_EVAL_order',region_or_task=name,method='mean_sigma_keep_'+str(frac),metric='mean_absolute_error')
    x=np.array(cols,dtype=float).T;assert x.shape==(n,len(labels));finite=np.isfinite(x);filled=np.where(finite,x,0.);den=finite.astype(float);bs=[];rng=np.random.default_rng(20261030)
    for start in range(0,2000,16):
        ix=rng.integers(n,size=(min(16,2000-start),n));ct=np.array([np.bincount(a,minlength=n) for a in ix],float);dn=ct@den;bs.extend(np.divide(ct@filled,dn,out=np.full_like(dn,np.nan),where=dn>0))
    bs=np.array(bs);cells=[]
    for j,label in enumerate(labels):
        z=x[:,j];z=z[np.isfinite(z)];b=bs[:,j];b=b[np.isfinite(b)];cells.append(dict(label,n=len(z),n_unavailable=n-len(z),mean=statistics.fmean(z) if len(z) else None,sample_SD_ddof1=statistics.stdev(z) if len(z)>1 else None,CI95_low=float(np.quantile(b,.025)) if len(b) else None,CI95_high=float(np.quantile(b,.975)) if len(b) else None))
    with (R/'conformal/all_mean_SD_CI_N.csv').open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    with (R/'conformal/risk_coverage_curves.csv').open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    with (R/'conformal/paired_values.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['index','group_id']+['|'.join(str(d[k]) for k in ('model','analysis','region_or_task','method','metric')) for d in labels])
        for i,values in zip(ids,x):w.writerow([i,s['all_group_ids'][i]]+[float(v) if np.isfinite(v) else None for v in values])
    save(R/'conformal/summary.json',dict(status='complete',time=now(),cells=cells,EVAL_valid_groups=n,retrospective=True,group_unit='Frozen sample group, not verified patient',coverage_definition='95%groups with >=95%weighted RGB-component coverage simultaneously across nonempty P/D/O/F; encoded unclipped intervals',ranking='Global positive q leaves sigma order unchanged; selection gains are separate empirical results',limitations='Retrospective split of already inspected TEST; exchangeability unverified; no external patient or clinical decision guarantee; F sigma is a task proxy, not task standard deviation',infinite_width_policy='NA numeric width with explicit infinite=true in raw records; never average infinity as zero',bootstrap=dict(draws=2000,seed=20261030,unit='paired sample group',selective_risk='fixed original EVAL retention order within bootstrap, descriptive'),calibration=desc(R/'conformal/calibration_frozen.json'),request=desc(R/'request.json')));render(cells,curves)


def render(cells,curves):
    sys.path.insert(0,str(E/'ksg_stability_20261030_v1/runtime_plot'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
    for j,method in enumerate(METHODS):
        v=[next(z for z in cells if z['analysis']=='joint_group_event' and z['model']==m and z['method']==method) for m in MODELS];axs[0].plot(range(5),[z['mean'] for z in v],'o-',label=method)
    axs[0].axhline(.95,color='gray',ls='--');axs[0].set(xticks=range(5),xticklabels=('U-Net','Multi','U-Net++','Swin','LDM'),ylabel='Group event coverage',ylim=(0,1.02));axs[0].legend(fontsize=6,frameon=False)
    for mid in MODELS:
        c=[z for z in curves if z['model']==mid and z['outcome']=='F'];axs[1].plot([z['retained_fraction'] for z in c],[z['risk'] for z in c],'o-',label=mid)
    axs[1].set(xlabel='Retained groups',ylabel='F mean absolute error');axs[1].legend(fontsize=6,frameon=False)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(R/'conformal/Figure_pixel_coverage_and_risk.png',dpi=300);fig.savefig(R/'conformal/Figure_pixel_coverage_and_risk.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
    for ax,signal in zip(axs,('P','D')):
        for method in ('constant_width','sigma_F_proxy'):
            z=[next(v for v in cells if v['analysis']=='task' and v['model']==m and v['region_or_task']==signal+'_count' and v['method']==method and v['metric']=='coverage') for m in MODELS];ax.errorbar(range(5),[v['mean'] for v in z],yerr=[[v['mean']-v['CI95_low'] for v in z],[v['CI95_high']-v['mean'] for v in z]],fmt='o-',label=method)
        ax.axhline(.95,color='gray',ls='--');ax.set(title=signal+' algorithm-defined object count',xticks=range(5),xticklabels=('U-Net','Multi','U-Net++','Swin','LDM'),ylabel='Task interval coverage',ylim=(0,1.02));ax.legend(fontsize=6,frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(R/'conformal/Figure_task_coverage.png',dpi=300);fig.savefig(R/'conformal/Figure_task_coverage.pdf');plt.close(fig)


def collect():
    os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');os.sched_setaffinity(0,list(range(96,112)));os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    q=request();job=checked(read(R/'jobs/conformal/manifest.json'));assert job['policy_sha256']==hashlib.sha256(encoded(q['conformal_policy'])).hexdigest();split=checked(read(verify(q['conformal_policy']['split'])));s=read(verify(q['selection']));allmanifest=[]
    for stage,total in (('cal',10949),('eval',10950)):
        while True:
            no_failures();counts={m:len(list((R/('conformal/'+stage+'/records/'+m)).glob('*.json.gz'))) for m in MODELS};touch_progress('conformal_collector','waiting_'+stage,counts=counts,expected_each=total)
            if all(v==total for v in counts.values()):break
            time.sleep(10)
        data,manifest,missing=load_records(stage,split,q,s);qc(stage);allmanifest.extend(manifest)
        if stage=='cal':freeze_calibration(data,q,manifest,missing);assert checked(read(R/'conformal/calibration_frozen.json'))['EVAL_records_at_freeze']==0
        else:statistics_eval(data,checked(read(R/'conformal/calibration_frozen.json'))['models'],q,s)
    assert len(allmanifest)==21899*5;save(R/'conformal/all_record_manifest.json',dict(files=allmanifest));summary=checked(read(R/'conformal/summary.json'));save(R/'conformal/final_acceptance.json',dict(status='passed',time=now(),records=109495,valid_predictions=21422*5,missing_predictions=477*5,summary=desc(R/'conformal/summary.json'),calibration=desc(R/'conformal/calibration_frozen.json'),manifest=desc(R/'conformal/all_record_manifest.json'),CPU_QC=[desc(R/('conformal/evidence/'+stage+'_independent_CPU_QC.json')) for stage in ('cal','eval')],raw=desc(R/'conformal/paired_values.csv'),request=desc(R/'request.json'),job=desc(R/'jobs/conformal/manifest.json')))


if __name__=='__main__':
    try:
        if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]),sys.argv[4])
        else:collect()
    except BaseException:
        save(R/('failures/conformal_%d.json'%os.getpid()),dict(time=now(),traceback=traceback.format_exc()));raise
