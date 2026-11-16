import math,importlib.util
import numpy as np
from chain_common import *


def numeric_helpers():
    p=read(E/'three_region_test_parallel_20261025_v1/evaluation/protocol.json');spec=importlib.util.spec_from_file_location('_chain_original_numeric_runner',verify(p['old_runner']));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg=read(p['eval_config']['path']);numeric=old.FrozenSources(cfg['sources'],Path(p['eval_config']['path']).parent).load('tools/uv_architecture_comparison_metrics_v10.py')
    return old,numeric


def cpu(truth,pred,sigma,weights):
    old,numeric=numeric_helpers();r=old.measure(truth,pred,sigma,weights,numeric)
    return [[r[k]['l1'],r[k]['mse'],r[k]['psnr'],r[k]['ssim']] for k in REGIONS]


class GPU:
    def __init__(self):
        from gpu_math import Engine
        self.e=Engine()
    def measure(self,truth,pred,weights):
        import torch
        x=torch.as_tensor(truth,device='cuda:0')[None];y=torch.as_tensor(pred,device='cuda:0')[None];d=y-x
        w=torch.as_tensor(np.stack([weights[k] for k in REGIONS]),device='cuda:0',dtype=torch.float64);a=w.flatten(1).sum(1);safe=a.clamp_min(1e-300);flat=w.flatten(1).T.contiguous()
        l1=(d.abs().mean(1).flatten(1).double()@flat/safe)[0];mse=(d.square().mean(1).flatten(1).double()@flat/safe)[0]
        mt=self.e.smooth(x);mp=self.e.smooth(y);vt=(self.e.smooth(x*x)-mt*mt).clamp_min(0);vp=(self.e.smooth(y*y)-mp*mp).clamp_min(0);cov=self.e.smooth(x*y)-mt*mp
        sm=(((2*mt*mp+.0001)*(2*cov+.0009))/((mt*mt+mp*mp+.0001)*(vt+vp+.0009))).mean(1);wc=w[:,5:-5,5:-5];ac=wc.flatten(1).sum(1);ssim=(sm.flatten(1).double()@wc.flatten(1).T/ac.clamp_min(1e-300))[0]
        aa=torch.stack([l1,mse,ssim]).cpu().numpy();areas=a.cpu().tolist();centers=ac.cpu().tolist();out=[]
        for ri in range(4):
            if not areas[ri]:out.append([None]*4);continue
            u,v,z=[float(x) for x in aa[:,ri]];assert math.isfinite(u) and math.isfinite(v) and u>=0 and v>=0
            if centers[ri]:assert math.isfinite(z)
            out.append([u,v,10*math.log10(1/v) if v else None,z if centers[ri] else None])
        return out,areas,centers


def GPU_setup(q,host,local):
    check_host(q,host);os.sched_setaffinity(0,[80+2*local,81+2*local]);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));g=q['hosts'][host]['gpus'][local]
    os.environ.update(CUDA_VISIBLE_DEVICES=g['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    return g,torch
