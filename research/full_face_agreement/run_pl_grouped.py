"""Evaluate all accepted PL checkpoints after one frozen image/geometry load per group."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import os
from pathlib import Path
import sys
import time

from measure3b import ROOT, completed, flush, save_record
from pl_cases import REGIONS, TARGETS, _context, _predictor
from run_pl_sequential import TASKS, original_chunk, persist


# Six configurations share each archive chunk across 32 adjacent TEST indices.
chunk = lru_cache(maxsize=len(TASKS) * 2)(original_chunk.__wrapped__)


def heads(key):
    return TARGETS if key == "both" else (key,)


def load_predictors():
    return {(model, key): _predictor(model, key, heads(key), "cuda:0")
            for model, key in TASKS}


def measure_indices(lane, indices, ctx, predictors):
    import torch
    prior = {(model, head): completed(model, head)
             for model, key in TASKS for head in heads(key)}
    summary = dict(pid=os.getpid(), lane=lane, gpu=torch.cuda.get_device_name(0),
                   mode="grouped", new_rows=0, max_full_face_delta=0.0,
                   start=datetime.now(timezone.utc).isoformat())
    start = time.monotonic()
    for idx in indices:
        required = {(model, key): [head for head in heads(key) if idx not in prior[model, head]]
                    for model, key in TASKS}
        if not any(required.values()):
            continue
        case = ctx.case(idx, TARGETS)
        gid = case["group_id"]
        reason = case["missing_reason"]
        seeds = {} if reason else {name: case["rois"][name] for name in REGIONS}
        for (model, key), pending in required.items():
            if not pending:
                continue
            old = chunk(model, key, idx // 32 * 32)[idx]
            if old["sample_id"] != gid or old["architecture"] != model:
                raise ValueError(f"Frozen group or architecture differs for {idx} {model}")
            output = {}
            if not reason:
                if old["input_binding"]["rgb_tensor_sha256"] != case["inputs"]["rgb_tensor_sha256"]:
                    raise ValueError(f"Frozen PL RGB differs at {idx} {model}")
                output = predictors[model, key].predict(case)
                if set(output) != set(heads(key)):
                    raise ValueError(f"Unexpected predictions for {model} {key}")
            for head in pending:
                archived = old["measurements"][head]
                if reason != archived["missing_reason"]:
                    raise ValueError(f"Frozen technical status differs for {idx} {model} {head}")
                record = dict(index=idx, group_id=gid, model=model, target=head, missing_reason=reason)
                if not reason:
                    whole = archived["anatomy"]["F_reference"]
                    reference = {name: [whole[col][c] for c in "RGB"]
                                 for name, col in (("truth", "truth"), ("pred", "prediction"))}
                    record.update(f_weights=case["weights"]["F"], old_weights=seeds,
                                  truth=case["truths"][head], pred=output[head].detach().cpu().numpy(),
                                  archived_full_face=reference)
                result = save_record(record)
                summary["max_full_face_delta"] = max(summary["max_full_face_delta"],
                                                     result.get("max_full_face_prediction_delta", 0))
                summary["new_rows"] += 1
            del output
        flush()
        summary.update(index=idx, elapsed_s=time.monotonic()-start, time=datetime.now(timezone.utc).isoformat())
        if summary["new_rows"] <= 7 or (idx // 3) % 16 == 0:
            persist(dict(summary, phase="running"), lane)
    return summary


def run(lane, stop):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ["TABLE3B_LANE"] = f"pl{lane}"
    with (ROOT / "logs" / f"pl{lane}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        persist(dict(pid=os.getpid(), lane=lane, phase="loading", mode="grouped"), lane)
        predictors = load_predictors()
        summary = measure_indices(lane, range(lane, stop, 3), _context(), predictors)
        persist(dict(summary, phase="complete"), lane)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", type=int, choices=range(3))
    parser.add_argument("--stop", type=int, default=21899)
    args = parser.parse_args()
    try:
        run(args.lane, args.stop)
    except RuntimeError as exc:
        import torch
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and "cuda out of memory" not in str(exc).lower():
            raise
        flush()
        print("Grouped GPU allocation exceeded capacity; resuming with one checkpoint at a time.", flush=True)
        worker = Path(__file__).with_name("run_pl_sequential.py")
        os.execv(sys.executable, [sys.executable, "-u", str(worker), *sys.argv[1:]])


if __name__ == "__main__":
    main()
