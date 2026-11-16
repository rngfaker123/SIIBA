"""Read the five archived UV outputs together on each TEST group, using CPU lanes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path("/study/siiba/experiments/table3b_fullf_20261112_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--stop", type=int, default=21899)
    args = parser.parse_args()
    if not 0 <= args.lane < args.lanes or not 0 <= args.stop <= 21899:
        parser.error("invalid lane or TEST range")
    tag = "uv%02d" % args.lane
    os.environ.update(TABLE3B_LANE=tag, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    import measure3b
    import uv_cases
    from uv_cases import CHECKPOINTS, iter_cases
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    progress = logs / (tag + ".progress.json")
    lock = (logs / (tag + ".lock")).open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    _, _, missing = uv_cases.cohort()
    processed = 0
    saved = 0
    skipped = 0
    missing_inputs = []

    def update(status, index):
        state = dict(status=status, lane=tag, index=index, pid=os.getpid(), processed=processed,
                     saved=saved, previously_done=skipped, elapsed=time.monotonic()-start,
                     missing_cache_count=len(missing_inputs), missing_cache=missing_inputs)
        tmp = progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(progress)

    update("running", None)
    index = None
    try:
        for index in range(args.lane, args.stop, args.lanes):
            todo = [model for model in CHECKPOINTS if not measure3b.done(model, "uv", index)]
            skipped += len(CHECKPOINTS) - len(todo)
            ready = None
            if todo and index not in missing:
                ready = uv_cases.stage(index)
            for model in todo:
                if index not in missing and (ready is None or not Path(ready["outputs"][model]["prediction"]["path"]).is_file()):
                    missing_inputs.append([index, model])
                    continue
                case = next(iter_cases(index, index+1, 1, model, "uv", "cpu"))
                measure3b.save_record(case)
                saved += 1
            processed += 1
            if processed == 1 or processed % 20 == 0:
                measure3b.flush()
                update("running", index)
        measure3b.flush()
        update("incomplete_cache" if missing_inputs else "complete", index)
    except Exception:
        measure3b.flush()
        update("failed", index)
        raise
    if missing_inputs:
        raise RuntimeError("Missing frozen prediction caches; see lane status for replay assignments")


if __name__ == "__main__":
    main()
