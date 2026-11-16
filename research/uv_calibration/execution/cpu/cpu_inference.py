"""Thin, source-bound access to the already accepted five-model inference program."""
import sys,importlib.util,os,hashlib
from pathlib import Path
import numpy as np
from chain_common import *


class Original:
    def __init__(self,q,uuid):
        sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/'source'))
        spec=importlib.util.spec_from_file_location('_conformal_original_infer',BASE/'source/infer_pipeline.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.w,self.manifest,self.science=mod.original_science();self.runtime=self.science['ctx']['old'].runtime_info('cpu',int(uuid));self.models={};self.q=q
        assert self.runtime['precision']=='FP32_no_autocast_no_TF32'
    def model(self,mid):
        if mid not in self.models:
            if mid in ('swin','ldm'):
                bundle=self.w.training_bundle(self.manifest,mid);assert bundle.best_checkpoint.sha256==self.q['checkpoint_sha256'][mid]
                self.models[mid]=self.w.loader.load_adapter(mid,bundle,'cpu')
            else:
                c=self.science['ctx'];assert c['old'].MODEL_PINS[mid]['sha']==self.q['checkpoint_sha256'][mid]
                self.models[mid]=c['old'].load_model(mid,c['config']['models'][mid],Path(c['protocol']['eval_config']['path']).parent,c['frozen'],'cpu')
        return self.models[mid]
    def predict(self,mid,rgb,row):
        import torch
        model=self.model(mid)
        if mid in ('swin','ldm'):
            converter=self.w.bridge.TorchTensorIO('cpu');x=converter.as_input(rgb)
            with torch.inference_mode():out=model.predict(x,row)
            pred,sigma=converter.prediction_pair(out)
        else:pred,sigma=self.science['ctx']['old'].infer(model,rgb,mid,'cpu')
        assert pred.shape==(3,1024,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
        assert sigma.shape==(1024,1024) and sigma.dtype==np.float32 and np.isfinite(sigma).all() and (sigma>0).all()
        return pred,sigma
    def rgb(self,c):
        ready=checked(read(verify(c['inputs']['ready'])));row=ready['row'];rgb=self.science['ctx']['old'].prepare_image(row['rgb'],row['native_shape'],self.q['data_root']);assert hashlib.sha256(rgb.tobytes()).hexdigest()==ready['inputs']['rgb']['sha256'];return rgb,row,ready
