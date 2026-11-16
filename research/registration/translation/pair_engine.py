import argparse, importlib.util, os, time, traceback
from concurrent.futures import ThreadPoolExecutor
from common import *
from cache_io import load_case


def inputs(index,q,selection):
    import numpy as np
    case=load_case(index,q,selection)
    if case['missing_reason'] is not None:return case
    ready=checked(read(verify(case['inputs']['ready'])));pred=[];bindings={}
    for name in MODELS:
        d=ready['outputs'][name];assert d['checkpoint_sha256']==q['checkpoint_sha256'][name]
        p=verify(d['prediction']);a=np.load(p,allow_pickle=False)
        assert a.shape==(3,1024,1024) and a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        assert hashlib.sha256(a.tobytes()).hexdigest()==d['tensor']['sha256']
        pred.append(a);bindings[name]=dict(prediction=d['prediction'],tensor=d['tensor'],checkpoint_sha256=d['checkpoint_sha256'])
    case['predictions']=np.stack(pred);case['inputs']['existing_predictions']=bindings
    return case


class PairEngine:
    def __init__(self):
        from gpu_math import Engine
        self.e=Engine()
    def evaluate(self,truth,pred,weights):
        import numpy as np,torch
        x=torch.as_tensor(truth,device='cuda:0').unsqueeze(0);y=torch.as_tensor(pred,device='cuda:0')
        assert x.dtype==y.dtype==torch.float32
        m=MARGIN;w=torch.as_tensor(np.stack([weights[k] for k in REGIONS]),device='cuda:0',dtype=torch.float64)[:,m:-m,m:-m]
        area=w.flatten(1).sum(1);safe=area.clamp_min(1e-300);flat=w.flatten(1).T.contiguous()
        mt=self.e.smooth(x);mp=self.e.smooth(y)
        vt=(self.e.smooth(x*x)-mt*mt).clamp_min(0);vp=(self.e.smooth(y*y)-mp*mp).clamp_min(0);cov=self.e.smooth(x*y)-mt*mp
        sm=(((2*mt*mp+.0001)*(2*cov+.0009))/((mt*mt+mp*mp+.0001)*(vt+vp+.0009))).mean(1)
        d=y-x;maps=[d.abs().mean(1)[:,m:-m,m:-m],d.square().mean(1)[:,m:-m,m:-m],sm[:,m-5:-(m-5),m-5:-(m-5)]]
        vals=[((a.flatten(1).double()@flat)/safe).cpu().numpy() for a in maps];out=[]
        for j in range(len(pred)):
            rows=[]
            for r in range(4):
                if not area[r].item():rows.append([None]*4);continue
                l1,mse,ssim=[float(v[j,r]) for v in vals]
                assert all(math.isfinite(v) for v in (l1,mse,ssim)) and l1>=0 and mse>=0
                rows.append([l1,mse,10*math.log10(1/mse) if mse else None,ssim])
            out.append(rows)
        return out,area.cpu().tolist()


def oracle(case):
    import numpy as np
    protocol=read(E/'three_region_test_parallel_20261025_v1/evaluation/protocol.json')
    spec=importlib.util.spec_from_file_location('_support_old',verify(protocol['old_runner']));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=read(protocol['eval_config']['path']);numeric=old.FrozenSources(cfg['sources'],Path(protocol['eval_config']['path']).parent).load('tools/uv_architecture_comparison_metrics_v10.py')
    x=case['truth'].astype(float);out=[];m=MARGIN
    for y in case['predictions'].astype(float):
        d=y-x;sm=numeric.ssim_map_rgb(x,y);rows=[]
        for k in REGIONS:
            w=case['weights'][k].astype(float)[m:-m,m:-m];a=w.sum()
            if not a:rows.append([None]*4);continue
            l1=float((abs(d).mean(0)[m:-m,m:-m]*w).sum()/a);mse=float((np.square(d).mean(0)[m:-m,m:-m]*w).sum()/a)
            rows.append([l1,mse,10*math.log10(1/mse) if mse else None,float((sm[m-5:-(m-5),m-5:-(m-5)]*w).sum()/a)])
        out.append(rows)
    return out


def validation(engine,case):
    import numpy as np
    a,_=engine.evaluate(case['truth'],case['predictions'],case['weights']);b,_=engine.evaluate(case['truth'],case['predictions'],case['weights']);assert a==b
    cpu=oracle(case);diagnostics=[]
    for j,model in enumerate(MODELS):
        for ri,r in enumerate(REGIONS):
            for mi,metric in enumerate(('L1','MSE','PSNR_dB','SSIM')):
                x,y=cpu[j][ri][mi],a[j][ri][mi];assert (x is None)==(y is None)
                diagnostics.append(dict(model=model,region=r,metric=metric,cpu=x,cuda=y,absolute_difference=abs(x-y) if x is not None else None))
    x=np.full((3,80,80),.2,np.float32);p=np.full((5,3,80,80),.4,np.float32);w={k:np.ones((80,80),np.float32) for k in REGIONS}
    z,_=engine.evaluate(x,p,w)
    for row in z:
        for v in row:
            assert abs(v[0]-.2)<1e-6 and abs(v[1]-.04)<1e-6 and abs(v[3]-(.1601/.2001))<1e-4
    w['P'][:]=0;z,_=engine.evaluate(x,np.stack([x]*5),w)
    assert all(row[0]==[None]*4 and all(v[0]==0 and v[1]==0 and v[2] is None and abs(v[3]-1)<1e-5 for v in row[1:]) for row in z)
    return diagnostics


def run(lane):
    q=request();item=q['lanes'][lane];check_host(q,item['host']);os.sched_setaffinity(0,item['cpus']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES=item['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    import torch,numpy,scipy
    assert [torch.__version__,numpy.__version__,scipy.__version__]==q['runtime_versions']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    selection=read(verify(q['selection']));engine=PairEngine();pilot=inputs(item['pilot_index'],q,selection);t=time.monotonic();ds=validation(engine,pilot)
    save(ROOT/('validation/lane%d.json'%lane),dict(status='passed',time=now(),lane=lane,uuid=item['uuid'],identity=identity(),same_card_repeat_exact=True,pilot=pilot['index'],diagnostics=ds,seconds=time.monotonic()-t))
    while not (ROOT/'validation_gate.json').exists():no_failures();time.sleep(2)
    no_failures();assert checked(read(ROOT/'validation_gate.json'))['status']=='passed'
    todo=list(range(lane,21899,5));start=time.monotonic();times=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(inputs,todo[0],q,selection)
        for pos,index in enumerate(todo):
            no_failures();case=future.result()
            if pos+1<len(todo):future=pool.submit(inputs,todo[pos+1],q,selection)
            t=time.monotonic();save(ROOT/('claims/%06d.json'%index),dict(index=index,lane=lane,time=now(),request=desc(ROOT/'request.json')))
            if case['missing_reason'] is None:
                metrics,areas=engine.evaluate(case['truth'],case['predictions'],case['weights']);status='ok'
            else:metrics=areas=None;status='unavailable_geometry'
            save(ROOT/('records/%06d.json.gz'%index),dict(index=index,group_id=case['group_id'],status=status,missing_reason=case['missing_reason'],metrics=metrics,areas=areas,inputs=case['inputs'],lane=lane,request_sha256=sha(ROOT/'request.json'),uuid=item['uuid'],time=now()))
            times.append(time.monotonic()-t)
            if (pos+1)%10==0:save(ROOT/('progress/lane%d.json'%lane),dict(time=now(),lane=lane,done=pos+1,total=len(todo),elapsed_seconds=time.monotonic()-start,recent_compute_seconds=times[-20:]),mutable=True)
    save(ROOT/('completion/lane%d.json'%lane),dict(status='complete',time=now(),lane=lane,groups=len(todo),identity=identity(),elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lane',type=int,required=True);a=ap.parse_args()
    try:run(a.lane)
    except BaseException:
        save(ROOT/('failures/worker_%d_%d.json'%(a.lane,os.getpid())),dict(time=now(),traceback=traceback.format_exc()));raise
