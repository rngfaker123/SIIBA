"""Issue a stage assignment from the local CUDA visibility and prepared LDM protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from train_uv_ldm_clean_multigpu_20261016 import physical_gpu_uuids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--stage',choices=('ae','diffusion','latent_stats'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    visible=os.environ['CUDA_VISIBLE_DEVICES']
    ids=physical_gpu_uuids(visible)
    doc=dict(schema='uv_clean_ldm_stage_assignment_v1',
             protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
             stage=a.stage,world_size=len(ids),visible_devices=visible,gpu_uuids=ids)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2)


if __name__=='__main__':
    main()
