"""Exercise study architectures and released configuration bindings on synthetic CPU data."""
from pathlib import Path
import json
import sys
import hashlib

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent),str(HERE)]

import torch
from models.joint_uv_uncertainty_unet import JointUVUncertaintyUNet
from models.uv_uncertainty_unet import UVUncertaintyUNet
from models.uv_uncertainty_unetplusplus import UVUncertaintyUNetPlusPlus
from models.uv_latent_diffusion_v1 import KLUVAutoencoder, RGBLatentVDenoiser
from training import run


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    rgb=torch.rand(1,3,64,64)
    with torch.no_grad():
        for model,expected,name in ((UVUncertaintyUNet(),7763140,'mean'),(JointUVUncertaintyUNet(),16909738,'uv'),
                                    (UVUncertaintyUNetPlusPlus(),9045924,'mean')):
            assert sum(p.numel() for p in model.parameters())==expected
            out=model.eval()(rgb)
            assert out[name].shape==rgb.shape and torch.isfinite(out[name]).all()
        swin=run.select_module('swin','uv',{})
        source=swin.load_model_source(HERE.parent/'models/swin_model.py')
        model=source.SwinUVUncertainty().eval()
        assert sum(p.numel() for p in model.parameters())==41380360
        assert model(rgb)['mean'].shape==rgb.shape
        ae,denoiser=KLUVAutoencoder().eval(), RGBLatentVDenoiser().eval()
        prediction=ae(rgb,sample=False)
        assert prediction['reconstruction'].shape==rgb.shape
        z=prediction['mu']
        assert denoiser(z,torch.zeros(1,dtype=torch.long),rgb).shape==z.shape
    for arch in ('single_unet','multi_unet','unetplusplus','ldm','swin'):
        for target in (('joint',) if arch=='multi_unet' else ('uv','plh','plv')):
            name=arch if target=='joint' else arch+'_'+target
            config=json.loads((HERE/'configs'/(name+'.json')).read_bytes())
            runner=run.select_module(arch,target,config)
            runner.validate_config(config)
            if arch in ('single_unet','unetplusplus'):
                runner.verify_scientific_sources()
            if arch=='ldm':
                runner.original_sources()
    sigma=run.select_module('sigma','uv',{})
    assert sigma.load_source(HERE.parent/'models/ldm_sigma_model.py')
    print('Passed five model forwards, thirteen target configurations, source checks and sigma model import (CPU).')


if __name__=='__main__':
    main()
