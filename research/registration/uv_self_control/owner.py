import argparse, os, subprocess, sys, time, traceback
from common import *

def probe(host):
    v=identity();v['gpus']=inventory();v['host']=host
    save(ROOT/('inventory/'+host+'.json'),v)

def configure():
    hosts={}
    for h,n in [('a6000',1),('four4090',4)]:
        v=checked(read(ROOT/('inventory/'+h+'.json')));assert len(v['gpus'])==n
        hosts[h]={k:v[k] for k in ('boot_id','pid1_starttick','gpus')}
    selection=BASE/'selection.json';s=read(selection);valid=set(s['geometry_valid_indices']);assert len(valid)==21422
    lanes=[]
    for h in hosts:
        for local,gpu in enumerate(hosts[h]['gpus']):
            lane=len(lanes);pilot=min(i for i in s['spatial_indices'] if i%5==lane)
            lanes.append(dict(host=h,local_index=local,uuid=gpu['uuid'],cpus=[80+2*local,81+2*local],pilot_index=pilot))
    q=dict(schema='uv_misalignment_sensitivity_v1',time=now(),sources={p.name:desc(p) for p in sorted((ROOT/'source').glob('*.py'))},
        authorization=desc(ROOT/'authorization.json'),document_review=desc(ROOT/'document_review.json'),hosts=hosts,lanes=lanes,
        selection=desc(selection),prior_scientific_completion=desc(E/'signal_objects_3cd_statistics_parallel_20261030_v2/scientific_completion.json'),
        original_protocol=desc(BASE/'protocol.json'),runtime_versions=['2.4.1+cu124','1.21.2','1.6.3'],conditions=CONDITIONS,regions=REGIONS,
        full_catalog=21899,geometry_valid=21422,geometry_missing=477,coordinate_grid=[1024,1024],max_shift=5,blur_sigmas=SIGMAS,
        definitions=dict(reference='Original unchanged saved UV truth',candidate='Gaussian blur then exact integer translation of the same UV',
          directions='Four cardinal directions per nonzero Euclidean shift magnitude; no diagonal distance mixing',
          gaussian='Separable normalized discrete Gaussian, radius ceil(4*sigma), whole-image PyTorch reflect (SciPy mirror)',
          masks='Original Table3A frozen P/D/O/F fractional weights; fixed reference coordinates; never redetect or move masks',
          common_support='Fixed26-pixel image border removed for all conditions/metrics;5maxshift+16maxblur+5SSIMradius; no padding/wrap contamination',
          SSIM='Original11x11 separable Gaussian sigma1.5 population covariance; RGB channel mean; same fixed-center fractional weights',
          precision='CUDA FP32; noAMP/noTF32; FP64 weighted accumulation; actual CPU/CUDA numerical differences retained',
          PSNR='Per-condition10log10(1/MSE); zero MSE is +infinity, exported null with explicit count; no epsilon floor',
          statistics='Average four directions within group per metric before equal-group mean, sampleSDddof1, paired group bootstrap2000 seed20261030',
          interpretation='Sensitivity control only; no measured real RGB-UV displacement, no proof of registration adequacy, no physical impossibility claim'),
        execution=dict(ownership='index mod5',models_loaded=False,training=False,inference=False,detector=False,new_image_copies=False,
                       background='Detached server supervisors, source-pinned one-shot reservations, executable validation gate, independent host failures preserved'))
    assert all(len([i for i in range(21899) if i%5==k])>0 for k in range(5))
    save(ROOT/'request.json',q);print('CONFIGURED',sha(ROOT/'request.json'),flush=True)

def launch(host):
    q=request();check_host(q,host);assert not (ROOT/('owners/'+host+'_reservation.json')).exists(),'Host launch already reserved'
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not active,'GPUs busy; preserve active task'
    save(ROOT/('owners/'+host+'_reservation.json'),dict(time=now(),request=desc(ROOT/'request.json'),identity=identity(),host=host))
    log=ROOT/('owners/'+host+'.log');f=log.open('ab');env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([str(PY),str(ROOT/'source/owner.py'),'run','--host',host],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    save(ROOT/('owners/'+host+'_launch.json'),dict(time=now(),identity=identity(p.pid),request=desc(ROOT/'request.json'),host=host));print('LAUNCHED',host,p.pid,flush=True)

def run(host):
    q=request();check_host(q,host);os.sched_setaffinity(0,[64,65]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    children=[]
    for i,lane in enumerate(q['lanes']):
        if lane['host']!=host:continue
        f=(ROOT/('owners/lane%d.log'%i)).open('ab');p=subprocess.Popen([str(PY),str(ROOT/'source/worker.py'),'--lane',str(i)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(ROOT/('owners/lane%d_launch.json'%i),dict(time=now(),identity=identity(p.pid),lane=i,request=desc(ROOT/'request.json')));children.append((i,p))
    if host=='four4090':
        f=(ROOT/'owners/collector.log').open('ab');p=subprocess.Popen([str(MASKPY),str(ROOT/'source/collect.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        save(ROOT/'owners/collector_launch.json',dict(time=now(),identity=identity(p.pid),request=desc(ROOT/'request.json')));children.append(('collector',p))
    save(ROOT/('owners/'+host+'_active.json'),dict(time=now(),identity=identity(),children=[dict(name=k,identity=identity(p.pid)) for k,p in children],request=desc(ROOT/'request.json')))
    for name,p in children:
        rc=p.wait();assert rc==0,'Child failure '+str(name)+' rc'+str(rc)
    save(ROOT/('completion/'+host+'_supervisor.json'),dict(time=now(),status='complete',identity=identity(),request=desc(ROOT/'request.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['probe','configure','launch','run']);ap.add_argument('--host');a=ap.parse_args()
    try:
        if a.mode=='probe':probe(a.host)
        elif a.mode=='configure':configure()
        elif a.mode=='launch':launch(a.host)
        else:run(a.host)
    except BaseException:
        save(ROOT/('failures/owner_%s_%d.json'%(a.mode,os.getpid())),dict(time=now(),identity=identity(),traceback=traceback.format_exc()));raise
