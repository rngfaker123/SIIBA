import argparse, importlib.util, io, math, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from chain_common import *

def load_case(index,q,selection):
    import numpy as np
    p=BASE/('records/unet_single/%06d.json'%index);rec=checked(read(p));gid=selection['all_group_ids'][index]
    assert (rec['index'],rec['group_id'])==(index,gid)
    inputs=dict(original_record=desc(p))
    if index not in set(selection['geometry_valid_indices']):
        assert rec['status']=='unavailable_geometry';return dict(index=index,group_id=gid,missing_reason=rec['missing_reason'],inputs=inputs)
    ready=checked(read(verify(rec['input_binding'])));assert ready['index']==index and ready['group_id']==gid
    # Historical mask receipts use a different canonical seal encoding; their
    # original ready.json byte-SHA binding is the authority, never reinterpret it.
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
    # Load the unchanged original scalar/SSIM implementation from its frozen sources.
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
