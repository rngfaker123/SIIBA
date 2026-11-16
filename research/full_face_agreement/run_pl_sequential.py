"""Replay frozen PL predictions; validate the original whole-F channel measurements."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import time
import fcntl

from pl_cases import ARCHIVE, iter_cases
from measure3b import ROOT, completed, flush, save_record

TASKS = (("unet_single","plh"), ("unet_single","plv"), ("unet_multi","both"),
         ("unetplusplus","plh"), ("unetplusplus","plv"), ("swin","plh"))

@lru_cache(maxsize=4)
def original_chunk(model,key,start):
    path = ARCHIVE / "chunks" / (model + "_" + key) / ("%05d.jsonl.gz" % start)
    with gzip.open(path,"rt",encoding="utf8") as stream:
        return {r["index"]: r for r in map(json.loads,stream)}

def persist(status, lane):
    dest = ROOT / "logs" / ("pl%d_progress.json" % lane)
    partial = dest.with_suffix(".partial")
    partial.write_text(json.dumps(status), encoding="utf8")
    os.replace(partial,dest)

def run(lane, stop):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ["TABLE3B_LANE"] = "pl%d" % lane
    with (ROOT / "logs" / ("pl%d.lock" % lane)).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for model, key in TASKS:
            heads = ("plh","plv") if key == "both" else (key,)
            existing = {head:completed(model,head) for head in heads}
            skip = set.intersection(*existing.values())
            stamp = time.monotonic()
            count = 0
            max_delta = 0
            state = dict(pid=os.getpid(), lane=lane, gpu=torch.cuda.get_device_name(0),
                         model=model, target=key, prior=len(skip), start=datetime.now(timezone.utc).isoformat())
            persist(dict(state,phase="loading"),lane)
            for case in iter_cases(lane,stop,3,model,key,"cuda:0",skip_indices=skip):
                idx,head = case["index"],case["target"]
                if idx in existing[head]:
                    continue
                orig = original_chunk(model,key,(idx//32)*32)[idx]
                if orig["sample_id"] != case["group_id"] or orig["architecture"] != model:
                    raise ValueError("Original test groups do not match current inputs")
                if case.get("missing_reason") is None:
                    full = orig["measurements"][head]["anatomy"]["F_reference"]
                    case["archived_full_face"] = {name:[full[old][c] for c in "RGB"]
                                                   for name,old in (("truth","truth"),("pred","prediction"))}
                record = save_record(case)
                max_delta = max(max_delta, record.get("max_full_face_prediction_delta",0))
                count += 1
                if count <= 3 or count % 32 == 0:
                    flush()
                    persist(dict(state,phase="running",index=idx,new_rows=count,
                                 max_full_face_delta=max_delta,elapsed_s=time.monotonic()-stamp,
                                 time=datetime.now(timezone.utc).isoformat()),lane)
            flush()
            print(model,key,count,"rows",time.monotonic()-stamp,"sec",flush=True)
        persist(dict(pid=os.getpid(), phase="complete",time=datetime.now(timezone.utc).isoformat()),lane)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("lane",type=int,choices=range(3))
    parser.add_argument("--stop",type=int,default=21899)
    args=parser.parse_args()
    run(args.lane,args.stop)
