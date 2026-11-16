"""Measure paired channel means on the five exhaustive regions and the archived F."""
import atexit
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
from frozen_f_partition import partition_frozen_f
from fullface_partition import REGION_NAMES

ROOT = Path(os.getenv("TABLE3B_ROOT", str(Path(__file__).resolve().parents[1])))
_last = None
_db = None
_pending = 0

def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def layout(index, gid, f, old):
    global _last
    if _last is not None and _last[:2] == (index, gid) and _last[2] is f:
        return _last[3]
    key = "%06d" % index
    receipt = ROOT / "label_receipts" / (key + ".json")
    if receipt.is_file():
        meta = json.loads(receipt.read_bytes())
        if (meta["status"], meta["index"], meta["group_id"]) != ("complete", index, gid):
            raise ValueError("Geometry receipt identity differs: " + key)
        data = (ROOT / "labels" / (key + ".png")).read_bytes()
        if (hashlib.sha256(data).hexdigest() != meta["labels_sha256"] or
                sha(f) != meta["frozen_F_sha256"] or
                any(sha(old[n]) != meta["seed_sha256"][n] for n in REGION_NAMES)):
            raise ValueError("Archived fractional F, seeds or labels differ: " + key)
        labels = np.asarray(Image.open(io.BytesIO(data)))
        if sha(labels) != meta["label_array_sha256"]:
            raise ValueError("Decoded labels differ: " + key)
    else:
        labels, _ = partition_frozen_f(f, old)
    if labels.shape != f.shape or labels.max() != 5 or not np.array_equal(labels > 0, f > 0):
        raise ValueError("Incomplete five-region partition: " + key)
    bins = labels.flatten().astype(np.intp)
    weights = f.flatten().astype(np.float64)
    areas = np.bincount(bins, weights=weights, minlength=6)[1:]
    if np.any(areas <= 0):
        raise ValueError("Empty region: " + key)
    geometry = (bins, weights, np.append(areas, weights.sum()), sha(labels))
    _last = (index, gid, f, geometry)
    return geometry

def channel_means(image, bins, weights, areas):
    arr = np.asarray(image)
    if arr.shape != (3,1024,1024) or arr.dtype != np.float32 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError("Expected finite encoded RGB at the frozen evaluation grid")
    result = np.empty((6,3), dtype=np.float64)
    for ch, plane in enumerate(arr):
        weighted = plane.ravel() * weights
        sums = np.bincount(bins, weights=weighted, minlength=6)[1:]
        full_sum = weighted.sum()
        if not np.isclose(sums.sum(), full_sum, rtol=2e-11, atol=1e-9):
            raise ValueError("Regional sums do not reconstruct original F")
        result[:,ch] = np.append(sums,full_sum) / areas
    return result.tolist()

def measure_case(f_weights, old_weights, truth, pred, index, group_id):
    bins, weights, areas, identity = layout(index,group_id,f_weights,old_weights)
    return dict(truth=channel_means(truth,bins,weights,areas),
                pred=channel_means(pred,bins,weights,areas), area=areas.tolist(), labels=identity)

def db():
    global _db
    if _db is None:
        lane = os.environ["TABLE3B_LANE"]
        folder = ROOT / "records"
        folder.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(folder / (lane + ".sqlite")))
        _db.execute("CREATE TABLE IF NOT EXISTS measurements (model TEXT, target TEXT, idx INTEGER, "
                    "gid TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(model,target,idx), UNIQUE(model,target,gid))")
        _db.commit()
    return _db

def done(model,target,index):
    return db().execute("SELECT 1 FROM measurements WHERE model=? AND target=? AND idx=?",
                        (model,target,index)).fetchone() is not None

def completed(model,target):
    return {r[0] for r in db().execute("SELECT idx FROM measurements WHERE model=? AND target=?", (model,target))}

def flush():
    global _pending
    if _db is not None:
        _db.commit()
    _pending = 0

atexit.register(flush)

def save_record(case):
    global _pending
    model, target, idx, gid = (case[k] for k in ("model","target","index","group_id"))
    if case.get("missing_reason"):
        record = {"missing_reason": case["missing_reason"]}
    else:
        record = measure_case(case["f_weights"],case["old_weights"],case["truth"],case["pred"],idx,gid)
        if "archived_full_face" in case:
            ref = case["archived_full_face"]
            if not np.allclose(record["truth"][-1],ref["truth"],rtol=0,atol=2e-11):
                raise ValueError("Measured whole-face reference mismatch at index %d" % idx)
            record["max_full_face_prediction_delta"] = float(np.max(np.abs(
                np.asarray(record["pred"][-1])-ref["pred"])))
    data = json.dumps(record, allow_nan=False, separators=(",",":"))
    db().execute("INSERT INTO measurements VALUES (?,?,?,?,?)",(model,target,idx,gid,data))
    _pending += 1
    if _pending >= 16 or idx < 8:
        flush()
    return record
