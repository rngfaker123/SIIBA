import csv, gzip, math, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from common import *
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
import numpy as np

def run():
    os.sched_setaffinity(0,list(range(24)));os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    q=request();start=time.monotonic()
    while not all((ROOT/('validation/lane%d.json'%k)).exists() for k in range(5)):
        no_failures();time.sleep(2)
    validations=[checked(read(ROOT/('validation/lane%d.json'%k))) for k in range(5)]
    assert all(v['status']=='passed' and v['same_card_repeat_exact'] and v['synthetic_checks']==25 for v in validations)
    # 逐卡核验含缺失项
    for lane,v in enumerate(validations):
        assert v['uuid']==q['lanes'][lane]['uuid'] and len(v['diagnostics'])==80
        for d in v['diagnostics']:
            assert (d['cpu'] is None)==(d['cuda'] is None)
            if d['cuda'] is not None:assert math.isfinite(d['cuda']) and math.isfinite(d['cpu'])
    save(ROOT/'validation_gate.json',dict(time=now(),status='passed',request=desc(ROOT/'request.json'),validation_files=[desc(ROOT/('validation/lane%d.json'%k)) for k in range(5)],
          program='Fixed transforms, kernels, masks, coordinates, metrics and null semantics; CPU/CUDA actual values/differences retained',small_real_CPU_CUDA_cells=400))
    while not all((ROOT/('completion/lane%d.json'%k)).exists() for k in range(5)):
        no_failures();progress=[checked(read(ROOT/('progress/lane%d.json'%k))) if (ROOT/('progress/lane%d.json'%k)).exists() else dict(done=0) for k in range(5)]
        save(ROOT/'status.json',dict(time=now(),phase='GPU_production',done=sum(p['done'] for p in progress),total=21899,lanes=progress),mutable=True);time.sleep(15)
    no_failures();save(ROOT/'status.json',dict(time=now(),phase='independent_coverage_and_statistics',total=21899),mutable=True)
    selection=read(verify(q['selection']));valid=set(selection['geometry_valid_indices']);values=[];ids=[];missing={};manifest=[];checks=0;qsha=sha(ROOT/'request.json')
    def load_one(i):
        p=ROOT/('records/%06d.json.gz'%i);v=checked(read(p));return i,v,desc(p)
    rawpath=ROOT/'all_conditions_per_group.csv.gz'
    with gzip.open(rawpath,'xt',encoding='utf8',newline='') as stream,ThreadPoolExecutor(max_workers=8) as pool:
        writer=csv.writer(stream);writer.writerow(['index','group_id','sigma_px','shift_px','dx','dy','region','L1','MSE','PSNR_dB','SSIM','effective_area'])
        for i,v,d in pool.map(load_one,range(21899)):
            assert v['index']==i and v['group_id']==selection['all_group_ids'][i] and v['lane']==i%5 and v['request_sha256']==qsha
            manifest.append(d)
            if i not in valid:
                assert v['status']=='unavailable_geometry' and v['metrics'] is None;missing[v['missing_reason']]=missing.get(v['missing_reason'],0)+1;continue
            assert v['status']=='ok';a=np.array(v['metrics'],dtype=float);assert a.shape==(85,4,4)
            areas=np.array(v['areas']);assert np.all(areas>=0) and np.isclose(areas[0]+areas[1]+areas[2],areas[3],rtol=0,atol=.02)
            for c,cell in zip(CONDITIONS,a):
                for r,metric in enumerate(cell):
                    if not areas[r]:assert np.isnan(metric).all();continue
                    l1,mse,psnr,ssim=metric
                    assert all(np.isfinite([l1,mse,ssim])) and -1e-9<=mse<=l1+2e-7 and l1*l1<=mse+2e-7
                    if mse:assert math.isclose(psnr,10*math.log10(1/mse),rel_tol=1e-12,abs_tol=1e-12)
                    else:assert np.isnan(psnr)
                    checks+=1;writer.writerow([i,v['group_id'],c['sigma'],c['radius'],c['dx'],c['dy'],REGIONS[r]]+[None if not np.isfinite(x) else float(x) for x in metric]+[areas[r]])
            assert np.nanmax(a[0,:,0])==0 and np.nanmax(a[0,:,1])==0
            collapsed=[]
            for sigma in SIGMAS:
                for radius in MAGNITUDES:
                    ix=[j for j,c in enumerate(CONDITIONS) if c['sigma']==sigma and c['radius']==radius]
                    block=a[ix];finite=np.isfinite(block);n=finite.sum(0)
                    avg=np.divide(np.nansum(block,axis=0),n,out=np.full((4,4),np.nan),where=n>0);collapsed.append(avg)
            values.append(collapsed);ids.append(i)
    assert len(ids)==21422 and sum(missing.values())==477
    data=np.asarray(values,dtype=np.float64).reshape(21422,-1);finite=np.isfinite(data);n=finite.sum(0);sums=np.nansum(data,axis=0)
    means=np.divide(sums,n,out=np.full_like(sums,np.nan),where=n>0)
    sd=np.sqrt(np.divide(np.nansum((data-means)**2,axis=0),n-1,out=np.full_like(means,np.nan),where=n>1))
    # 各指标共用组抽样
    rng=np.random.default_rng(20261030);draws=[];filled=np.nan_to_num(data,nan=0);den=finite.astype(float)
    for k in range(0,2000,32):
        sample=rng.integers(21422,size=(min(32,2000-k),21422));count=np.array([np.bincount(x,minlength=21422) for x in sample],dtype=float)
        numer=count@filled;denom=count@den;draws.extend(np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0))
    draws=np.array(draws);summary=[];independent_checks=0
    for col in range(data.shape[1]):
        z=data[:,col];a=z[np.isfinite(z)];mu=math.fsum(a)/len(a) if len(a) else None
        ss=math.sqrt(math.fsum((x-mu)**2 for x in a)/(len(a)-1)) if len(a)>1 else None
        assert (mu is None and np.isnan(means[col])) or math.isclose(mu,means[col],rel_tol=1e-10,abs_tol=1e-10)
        assert (ss is None and np.isnan(sd[col])) or math.isclose(ss,sd[col],rel_tol=1e-10,abs_tol=1e-10);independent_checks+=2
        ci=draws[:,col];ci=ci[np.isfinite(ci)];cell=col//16;ri=(col//4)%4;mi=col%4;sigma=SIGMAS[cell//5];radius=MAGNITUDES[cell%5]
        summary.append(dict(sigma_px=sigma,shift_px=radius,region=REGIONS[ri],metric=('L1','MSE','PSNR_dB','SSIM')[mi],
            n=int(n[col]),mean=mu,sample_SD_ddof1=ss,n_undefined=int(21422-n[col]),CI95_low=float(np.quantile(ci,.025)) if len(ci) else None,
            CI95_high=float(np.quantile(ci,.975)) if len(ci) else None,bootstrap_valid_draws=len(ci),
            undefined_reason='zero_MSE_positive_infinity' if sigma==0 and radius==0 and mi==2 else 'empty_fixed_reference_region_or_zero_MSE'))
    save(ROOT/'summary.json',dict(time=now(),status='complete',cells=summary,coverage=dict(catalog=21899,valid=21422,missing_reasons=missing),request=desc(ROOT/'request.json'),
        scientific_scope='UV vs synthetically perturbed UV; not a model or an estimate of observed RGB-UV misregistration',directions_averaged_within_group=True,bootstrap_draws=2000,seed=20261030))
    with (ROOT/'summary_mean_SD_CI_N.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save(ROOT/'record_manifest.json',dict(files=manifest));save(ROOT/'numerical_acceptance.json',dict(time=now(),status='passed',records=21899,valid=21422,missing=477,
        record_and_metric_coherence_checks=checks,independent_mean_SD_checks=independent_checks,real_CPU_CUDA_validation_cells=400,
        source_bindings_verified_in_workers=True,old_files_written=False,new_image_copies=False,summary=desc(ROOT/'summary.json')))
    render(summary)
    request();save(ROOT/'completion.json',dict(time=now(),status='complete',summary=desc(ROOT/'summary.json'),numerical=desc(ROOT/'numerical_acceptance.json'),
        raw=desc(rawpath),manifest=desc(ROOT/'record_manifest.json'),request=desc(ROOT/'request.json'),elapsed_seconds=time.monotonic()-start))
    save(ROOT/'status.json',dict(time=now(),phase='complete',groups=21899,valid=21422),mutable=True)

def render(cells):
    sys.path.insert(0,str(E/'ksg_stability_20261030_v1/runtime_plot'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'P':'#8e4585','D':'#2878b5','O':'#39874c','F':'#555555'}
    for sigma in SIGMAS:
        fig,axes=plt.subplots(1,3,figsize=(7.2,2.6),constrained_layout=True)
        for ax,metric,title in zip(axes,['L1','PSNR_dB','SSIM'],['L1 error','PSNR (dB)','SSIM']):
            for region in REGIONS:
                a=[x for x in cells if x['sigma_px']==sigma and x['region']==region and x['metric']==metric and x['mean'] is not None]
                xx=[x['shift_px'] for x in a];yy=[x['mean'] for x in a]
                ax.plot(xx,yy,'o-',label=region,color=colors[region],lw=1.2,ms=3)
                ax.fill_between(xx,[x['CI95_low'] for x in a],[x['CI95_high'] for x in a],color=colors[region],alpha=.16)
            ax.set(title=title,xlabel='Translation (1024-grid pixels)',xticks=MAGNITUDES);ax.grid(axis='y',color='.9',lw=.6)
        if sigma==0:axes[1].text(.05,.04,'0 px: PSNR = +infinity',transform=axes[1].transAxes,fontsize=7)
        axes[0].legend(frameon=False,fontsize=8)
        stem='Figure_translation_only' if sigma==0 else 'Supplement_shift_blur_sigma_%s'%str(sigma).replace('.','p')
        fig.savefig(ROOT/(stem+'.png'),dpi=300);fig.savefig(ROOT/(stem+'.pdf'));plt.close(fig)
    text='UV self-perturbation control: original UV vs translated Gaussian-blurred copies. All21,422 geometry-valid TEST groups; fixed original P/D/O/F fractional masks. Shifts0/1/2/3/5pixels in four cardinal directions; directions averaged within group before between-group means. Blur sigma0/0.5/1/2/4pixels, radiusceil(4sigma), applied before translation. All conditions share26-pixel image-border exclusion; masks are never moved or redetected. Gaussian SSIM11x11/sigma1.5, encoded-intensity L1 and per-condition PSNR(data range1). Shading:95% paired sample-group bootstrap CI(2000draws,seed20261030), not training-seed variability or a verified-patient CI. Full groupSD,N/nulls and direction-level values supplied. Zero-MSE PSNR is+infinity, not floored. Synthetic sensitivity does not estimate actual RGB-UV displacement or prove adequate alignment.\n'
    (ROOT/'figure_caption.txt').write_text(text,encoding='utf8')

if __name__=='__main__':
    try:run()
    except BaseException:
        save(ROOT/('failures/collector_%d.json'%os.getpid()),dict(time=now(),identity=identity(),traceback=traceback.format_exc()));raise
