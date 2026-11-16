import argparse, importlib.util, io, math, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import *

def load_case(index,q,selection):
    import numpy as np
    p=BASE/('records/unet_single/%06d.json'%index);rec=checked(read(p));gid=selection['all_group_ids'][index]
    assert (rec['index'],rec['group_id'])==(index,gid)
    inputs=dict(original_record=desc(p))
    if index not in set(selection['geometry_valid_indices']):
        assert rec['status']=='unavailable_geometry';return dict(index=index,group_id=gid,missing_reason=rec['missing_reason'],inputs=inputs)
    ready=checked(read(verify(rec['input_binding'])));assert ready['index']==index and ready['group_id']==gid
    # 按原回执字节核验

    receipt=read(verify(ready['mask_receipt']));assert receipt['status']=='complete' and receipt['index']==index and receipt['sample_id']==gid
    truth_d=ready['arrays']['truth'];mask_d=receipt['npz']
    truth=np.load(verify(truth_d),allow_pickle=False)
    with np.load(verify(mask_d),allow_pickle=False) as z:w={k:z[k] for k in REGIONS}
    assert truth.shape==(3,1024,1024) and truth.dtype==np.float32 and np.isfinite(truth).all() and truth.min()>=0 and truth.max()<=1
    assert all(v.shape==(1024,1024) and v.dtype==np.float32 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in w.values())
    assert np.allclose(w['P'].astype(float)+w['D']+w['O'],w['F'],atol=2e-6,rtol=0)
    inputs.update(ready=rec['input_binding'],truth=truth_d,original_mask_receipt=ready['mask_receipt'],original_fractional_masks=mask_d)
    return dict(index=index,group_id=gid,truth=truth,weights=w,inputs=inputs,missing_reason=None)

def cpu_oracle(case,conditions,q):
    import numpy as np
    from scipy import ndimage
    # 沿用冻结指标实现
    protocol=read(E/'three_region_test_parallel_20261025_v1/evaluation/protocol.json')
    spec=importlib.util.spec_from_file_location('_misalign_original_runner',verify(protocol['old_runner']));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=read(protocol['eval_config']['path']);frozen=old.FrozenSources(cfg['sources'],Path(protocol['eval_config']['path']).parent)
    numeric=frozen.load('tools/uv_architecture_comparison_metrics_v10.py')
    from gpu_math import kernel
    x=case['truth'].astype(np.float64);out=[]
    for c in conditions:
        y=x.copy();s=c['sigma']
        if s:
            k=kernel(s,int(math.ceil(4*s)));y=ndimage.convolve1d(y,k,axis=-1,mode='mirror');y=ndimage.convolve1d(y,k,axis=-2,mode='mirror')
        dx,dy=c['dx'],c['dy'];z=np.zeros_like(y);xs,xe=max(0,-dx),min(1024,1024-dx);ys,ye=max(0,-dy),min(1024,1024-dy)
        z[:,ys+dy:ye+dy,xs+dx:xe+dx]=y[:,ys:ye,xs:xe]
        delta=z-x;sm=numeric.ssim_map_rgb(x,z);rows=[];m=MARGIN
        for name in REGIONS:
            w=case['weights'][name].astype(np.float64)[m:-m,m:-m];area=w.sum()
            if not area:rows.append([None]*4);continue
            l1=float((np.abs(delta).mean(0)[m:-m,m:-m]*w).sum()/area);mse=float((np.square(delta).mean(0)[m:-m,m:-m]*w).sum()/area)
            ssim=float((sm[m-5:-(m-5),m-5:-(m-5)]*w).sum()/area)
            rows.append([l1,mse,10*math.log10(1/mse) if mse else None,ssim])
        out.append(rows)
    return out

def synthetic(engine):
    import torch,numpy as np
    from gpu_math import shift
    a=torch.zeros((1,3,80,80),device=engine.device);a[:,:,40,40]=1
    n=0
    for r,dx,dy in SHIFTS:
        z=shift(a,dx,dy);assert z[0,0,40+dy,40+dx].item()==1 and z.sum().item()==3;n+=1
    const=torch.full_like(a,.4)
    for s in SIGMAS:
        z=engine.blur(const,s);assert torch.max(torch.abs(z-const)).item()<2e-6;n+=1
    impulse=engine.blur(a,2.);assert abs(impulse.sum().item()-3)<1e-5 and torch.allclose(impulse[0,0,40,35:46],impulse[0,0,40,35:46].flip(0),atol=1e-7);n+=1
    truth=np.full((3,80,80),.4,dtype=np.float32);w={k:np.full((80,80),v,dtype=np.float32) for k,v in dict(P=.25,D=.25,O=.5,F=1).items()}
    v,areas=engine.evaluate(truth,w,CONDITIONS[:1]);assert all(x[0]==0 and x[1]==0 and x[2] is None and abs(x[3]-1)<1e-5 for x in v[0]);n+=1
    w['P'][:]=0;v,_=engine.evaluate(truth,w,CONDITIONS[:1]);assert v[0][0]==[None]*4;n+=1
    return n

def save_case(case,engine,lane,q):
    t=time.monotonic();index=case['index'];p=ROOT/('records/%06d.json.gz'%index)
    assert not p.exists(),'Duplicate case claim'
    save(ROOT/('claims/%06d.json'%index),dict(time=now(),index=index,lane=lane,identity=identity(),request=desc(ROOT/'request.json')))
    if case['missing_reason'] is not None:
        v=dict(index=index,group_id=case['group_id'],status='unavailable_geometry',missing_reason=case['missing_reason'],metrics=None,areas=None)
    else:
        metrics,areas=engine.evaluate(case['truth'],case['weights']);v=dict(index=index,group_id=case['group_id'],status='ok',missing_reason=None,metrics=metrics,areas=areas)
    v.update(time=now(),lane=lane,inputs=case['inputs'],request_sha256=sha(ROOT/'request.json'),identity=identity(),GPU_uuid=q['lanes'][lane]['uuid'],seconds=time.monotonic()-t)
    save(p,v);return v

def run(lane):
    q=request();item=q['lanes'][lane];check_host(q,item['host']);os.sched_setaffinity(0,item['cpus']);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES=item['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    import torch,numpy,scipy
    assert [torch.__version__,numpy.__version__,scipy.__version__]==q['runtime_versions']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    from gpu_math import Engine
    engine=Engine();selection=read(verify(q['selection']));pilot=item['pilot_index'];t=time.monotonic()
    tests=synthetic(engine);case=load_case(pilot,q,selection)
    cs=[CONDITIONS[i] for i in [0,1,8,34,84]];cuda,_=engine.evaluate(case['truth'],case['weights'],cs);repeat,_=engine.evaluate(case['truth'],case['weights'],cs)
    assert cuda==repeat,'Same-card deterministic repeat changed';cpu=cpu_oracle(case,cs,q);diagnostics=[]
    for i,(a,b) in enumerate(zip(cpu,cuda)):
        for r in range(4):
            for k,name in enumerate(('l1','mse','psnr','ssim')):
                assert (a[r][k] is None)==(b[r][k] is None),'Null semantics changed'
                diagnostics.append(dict(condition=cs[i],region=REGIONS[r],metric=name,cpu=a[r][k],cuda=b[r][k],absolute_difference=abs(a[r][k]-b[r][k]) if a[r][k] is not None else None))
    save(ROOT/('validation/lane%d.json'%lane),dict(status='passed',time=now(),lane=lane,identity=identity(),uuid=item['uuid'],
        synthetic_checks=tests,same_card_repeat_exact=True,real_group=pilot,diagnostics=diagnostics,seconds=time.monotonic()-t,
        precision='CUDA FP32 no AMP/no TF32; original FP64 weighted accumulation; finite CPU/CUDA deltas diagnostic'))
    from PIL import Image,ImageDraw
    import numpy as np
    from gpu_math import shift
    x=torch.as_tensor(case['truth'],device='cuda:0').unsqueeze(0)
    board=Image.new('RGB',(2048,552),'white');draw=ImageDraw.Draw(board)
    for j,(sigma,dx,label) in enumerate([(0,0,'Original UV'),(0,5,'UV shifted +5px'),(2,0,'UV Gaussian sigma2'),(2,5,'UV sigma2 then +5px')]):
        yy=shift(engine.blur(x,sigma),dx,0)[0].cpu().numpy();im=Image.fromarray(np.rint(yy.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).resize((512,512))
        board.paste(im,(j*512,40));draw.text((j*512+8,12),label,fill='black')
    (ROOT/'QC').mkdir(exist_ok=True);board.save(ROOT/('QC/lane%d_pilot%d.png'%(lane,pilot)))
    while not (ROOT/'validation_gate.json').exists():no_failures();time.sleep(2)
    no_failures();assert checked(read(ROOT/'validation_gate.json'))['status']=='passed'
    todo=list(range(lane,21899,5));done=0;start=time.monotonic();timings=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,todo[0],q,selection)
        for pos,index in enumerate(todo):
            no_failures();case=future.result()
            if pos+1<len(todo):future=pool.submit(load_case,todo[pos+1],q,selection)
            row=save_case(case,engine,lane,q);done+=1
            if row['status']=='ok':timings.append(row['seconds'])
            if done%5==0:save(ROOT/('progress/lane%d.json'%lane),dict(time=now(),lane=lane,done=done,total=len(todo),last_index=index,elapsed_seconds=time.monotonic()-start,recent_compute_seconds=timings[-20:]),mutable=True)
    check_host(q,item['host']);save(ROOT/('completion/lane%d.json'%lane),dict(status='complete',time=now(),lane=lane,groups=len(todo),identity=identity(),elapsed_seconds=time.monotonic()-start,source=desc(__file__)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lane',type=int,required=True);a=ap.parse_args()
    try:run(a.lane)
    except BaseException:
        save(ROOT/('failures/worker_lane%d_%d.json'%(a.lane,os.getpid())),dict(time=now(),identity=identity(),traceback=traceback.format_exc()));raise
