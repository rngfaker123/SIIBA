import os,sys,time,math,statistics,traceback,csv
os.environ.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from io_control import *
def collapsed(v,family):
 from engine import pick
 a=np.array(v['curves'],float);assert a.shape==(5,17,5,4,4);rows=[];keys=[]
 mean=np.stack([np.nanmean(a[:,[k for k,s in enumerate(SHIFTS) if max(abs(s[0]),abs(s[1]))==r]],axis=1) for r in RADII],axis=1)
 for kind,z in [('direction_mean',mean),('direction_mean_delta_from_same_sigma_zero',mean-mean[:,0:1])]:
  rows.extend(z.ravel())
  keys.extend([dict(kind=kind,sigma_px=sigma,shift_px=r,model=m,region=roi,metric=metric) for sigma in SIGMAS for r in RADII for m in MODELS for roi in REGIONS for metric in METRICS])
 if family=='common':
  b=np.array(v['corrected'],float);scores=np.array(v['grid_scores'],float);assert b.shape==(5,5,4,4) and scores.shape==(5,121,5,4)
  for kind,z in [('common_F_121_corrected',b),('common_F_121_delta_from_same_sigma_zero',b-a[:,0])]:
   rows.extend(z.ravel());keys.extend([dict(kind=kind,sigma_px=sigma,shift_px=5,model=m,region=roi,metric=metric) for sigma in SIGMAS for m in MODELS for roi in REGIONS for metric in METRICS])
  for kind in ['F_oracle_L1_by_max_radius','ROI_optimistic_min_L1_by_max_radius']:
   out=[]
   for si in range(5):
    for r in RADII:
     ks=[k for k,s in enumerate(GRID) if max(abs(s[0]),abs(s[1]))<=r];block=scores[si,ks]
     z=block[np.argmin(block[:,:,3],axis=0),np.arange(5)] if kind.startswith('F_') else np.nanmin(block,axis=0)
     out.extend(z.ravel())
   rows.extend(out);keys.extend([dict(kind=kind,sigma_px=sigma,shift_px=r,model=m,region=roi,metric='L1_selection_FP64') for sigma in SIGMAS for r in RADII for m in MODELS for roi in REGIONS])
 return np.array(rows),keys
def qc(family,q):
 from worker import load
 from engine import cpu_metric,cpu_score_grid,pick
 old,pe=old_modules();s=old.read(old.verify(old.request()['selection']));checks=0;maxima={k:0. for k in METRICS};allfiles=[]
 for i in q['QC_indices']:
  c,reuse=load(i,q,s);r=checked(read(ROOT/(family+'/records/%06d.json.gz'%i)));z=np.array(r['values']['curves'],float);ds=[]
  for sigma,dx,dy in [(0.,0,0),(.5,1,0),(1.,-2,0),(2.,0,3),(4.,0,-5)]:
   a=cpu_metric(c,sigma,dx,dy,family);b=z[SIGMAS.index(sigma),SHIFTS.index((dx,dy))];assert np.array_equal(np.isfinite(a),np.isfinite(b));d=np.abs(a-b)
   for mi,m in enumerate(METRICS):
    if np.isfinite(d[...,mi]).any():maxima[m]=max(maxima[m],float(np.nanmax(d[...,mi])))
   ds.append(dict(sigma=sigma,dx=dx,dy=dy,cpu=a.tolist(),actual=b.tolist(),absolute_difference=d.tolist()));checks+=80
  grids=[]
  if family=='common':
   sc=np.array(r['values']['grid_scores'],float)
   for sigma in [0.,4.]:
    cpu=cpu_score_grid(c,sigma);actual=sc[SIGMAS.index(sigma)];ix,_=pick(actual)
    assert np.all(actual[ix,np.arange(5),3]==actual[:,:,3].min(0));grids.append(dict(sigma=sigma,cpu=cpu.tolist(),actual=actual.tolist(),cpu_choice=pick(cpu)[0].tolist(),actual_choice=ix.tolist()))
  p=ROOT/(family+'/evidence/CPU_QC_%06d.json'%i);save(p,dict(index=i,inputs=c['inputs'],metric_diagnostics=ds,grid_diagnostics=grids));allfiles.append(desc(p))
 save(ROOT/(family+'/evidence/CPU_QC_acceptance.json'),dict(status='passed',time=now(),metric_cells=checks,actual_maximum_differences=maxima,files=allfiles,new_forward_count=0,CUDA_executed=False,selection='Same pre-fixed objective, all121 candidate minima independently checked; finite score and metric differences retained'))
def render(family,cells):
 sys.path.insert(0,str(ROOT.parent/'ksg_stability_20261030_v1/runtime_plot'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
 for sigma in SIGMAS:
  for metric in ['L1','SSIM']:
   fig,axes=plt.subplots(1,5,figsize=(13,2.6),constrained_layout=True,sharey=True)
   for j,ax in enumerate(axes):
    for ri,roi in enumerate(REGIONS):
     z=[v for v in cells if v['kind']=='direction_mean' and v['model']==MODELS[j] and v['region']==roi and v['metric']==metric and v['sigma_px']==sigma];z.sort(key=lambda v:v['shift_px']);xx=[v['shift_px'] for v in z];yy=np.array([v['mean'] for v in z]);lo=np.array([v['CI95_low'] for v in z]);hi=np.array([v['CI95_high'] for v in z]);ax.plot(xx,yy,'o-',label=roi,color=plt.cm.tab10(ri),ms=3);ax.fill_between(xx,lo,hi,alpha=.12,color=plt.cm.tab10(ri))
    ax.set(title=MODELS[j],xlabel='Added shift (1024-grid px)',xticks=RADII);ax.spines[['top','right']].set_visible(False)
   axes[0].set_ylabel(metric);axes[-1].legend(frameon=False);fig.suptitle('%s: Gaussian sigma=%g px; four-direction group average'%(family,sigma))
   p=ROOT/(family+'/Figure_sigma%g_%s'%(sigma,metric));fig.savefig(str(p)+'.png',dpi=220);fig.savefig(str(p)+'.pdf');plt.close(fig)
 if family=='common':
  for sigma in [0.,2.]:
   fig,axes=plt.subplots(1,5,figsize=(13,2.6),constrained_layout=True,sharey=True)
   for j,ax in enumerate(axes):
    for ri,roi in enumerate(REGIONS):
     z=[v for v in cells if v['kind']=='F_oracle_L1_by_max_radius' and v['model']==MODELS[j] and v['region']==roi and v['sigma_px']==sigma];z.sort(key=lambda v:v['shift_px']);ax.plot([v['shift_px'] for v in z],[v['mean'] for v in z],'o-',label=roi,color=plt.cm.tab10(ri),ms=3)
    ax.set(title=MODELS[j],xlabel='Allowed correction radius (px)',xticks=RADII);ax.spines[['top','right']].set_visible(False)
   axes[0].set_ylabel('L1 (FP64 selection score)');axes[-1].legend(frameon=False);fig.suptitle('Truth-assisted common-F translation bound; sigma=%g px'%sigma);p=ROOT/('common/Figure_oracle_sigma%g'%sigma);fig.savefig(str(p)+'.png',dpi=220);fig.savefig(str(p)+'.pdf');plt.close(fig)
def run(family):
 q=request();os.sched_setaffinity(0,list(range(32,48)));os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));start=time.monotonic()
 while not all((ROOT/(family+'/card%d_completion.json'%i)).exists() for i in range(3)):
  no_failures();save(ROOT/('progress/collector_'+family+'.json'),dict(time=now(),phase='waiting_GPU_'+family,records=len(list((ROOT/(family+'/records')).glob('*.json.gz')))),True);time.sleep(15)
 old,pe=old_modules();s=old.read(old.verify(old.request()['selection']));valid=set(s['geometry_valid_indices']);rows=[];ids=[];missing={};manifest=[];keys=None
 def load(i):
  p=ROOT/(family+'/records/%06d.json.gz'%i);return i,checked(read(p)),desc(p)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,v,d in pool.map(load,range(21899)):
   no_failures();assert v['index']==i and v['group_id']==s['all_group_ids'][i] and v['request_sha256']==sha(ROOT/'request.json') and v['new_forward_count']==0;manifest.append(d)
   if i not in valid:assert v['status']=='unavailable_geometry';missing[v['missing_reason']]=missing.get(v['missing_reason'],0)+1;continue
   assert v['status']=='ok';z,k=collapsed(v['values'],family);keys=k if keys is None else keys;assert keys==k;rows.append(z);ids.append(i)
 assert len(ids)==21422 and sum(missing.values())==477;data=np.array(rows);del rows;finite=np.isfinite(data);filled=np.nan_to_num(data,nan=0);den=finite.astype(float);boots=[];rng=np.random.default_rng(20261031)
 for k in range(0,2000,32):
  ix=rng.integers(len(ids),size=(min(32,2000-k),len(ids)));ct=np.array([np.bincount(v,minlength=len(ids)) for v in ix],float);dn=ct@den;boots.extend(np.divide(ct@filled,dn,out=np.full_like(dn,np.nan),where=dn>0))
 boots=np.array(boots);cells=[]
 for col,key in enumerate(keys):
  z=data[:,col];z=z[np.isfinite(z)];b=boots[:,col];b=b[np.isfinite(b)];mu=statistics.fmean(z) if len(z) else None;sd=statistics.stdev(z) if len(z)>1 else None
  assert not len(z) or math.isclose(mu,float(z.mean()),abs_tol=1e-10,rel_tol=1e-10)
  assert len(z)<2 or math.isclose(sd,float(z.std(ddof=1)),abs_tol=1e-10,rel_tol=1e-10)
  cells.append(dict(key,n=len(z),n_undefined=21422-len(z),mean=mu,sample_SD_ddof1=sd,CI95_low=float(np.quantile(b,.025)) if len(b) else None,CI95_high=float(np.quantile(b,.975)) if len(b) else None))
 p=ROOT/(family+'/paired_values.npz')
 with p.open('xb') as f:np.savez_compressed(f,indices=np.array(ids),values=data)
 save(ROOT/(family+'/paired_keys.json'),dict(keys=keys));save(ROOT/(family+'/record_manifest.json'),dict(files=manifest));save(ROOT/(family+'/summary.json'),dict(status='complete',time=now(),family=family,cells=cells,catalog=21899,valid=21422,missing_reasons=missing,request=desc(ROOT/'request.json'),interpretation=q['interpretation']))
 with (ROOT/(family+'/summary_mean_SD_CI_N.csv')).open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
 qc(family,q);render(family,cells)
 save(ROOT/(family+'/final_acceptance.json'),dict(status='passed',time=now(),catalog=21899,valid=21422,records=21899,new_model_forwards=0,summary=desc(ROOT/(family+'/summary.json')),paired=desc(p),paired_keys=desc(ROOT/(family+'/paired_keys.json')),manifest=desc(ROOT/(family+'/record_manifest.json')),QC=desc(ROOT/(family+'/evidence/CPU_QC_acceptance.json')),independent_mean_SD_checks=2*len(cells),bootstrap_draws=2000,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException:save(ROOT/('failures/collector_%d.json'%os.getpid()),dict(time=now(),traceback=traceback.format_exc()));raise
