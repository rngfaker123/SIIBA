"""Original archived-landmark old-F rasterization in its original runtime."""
import argparse
import json
from pathlib import Path
import numpy as np
from detector import prepare_old_f_native, project_native_mask
from io_utils import FROZENROOT, read, save_array, save, seal, desc, now, verify

def run(meta_path, output):
    meta=read(meta_path); lm=read(verify(meta['landmarks']))
    original=prepare_old_f_native(lm,meta['row']['native_shape'],tools_dir=FROZENROOT/'tools')
    old=project_native_mask(original,meta['row']['native_shape'])
    receipt=save_array(output,old)
    save(str(output)+'.json',seal({'time':now(),'input':desc(meta_path),'landmarks':meta['landmarks'],
        'old_f_coverage':receipt,'native_area':int(np.count_nonzero(original))}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--meta',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.meta,a.output)
