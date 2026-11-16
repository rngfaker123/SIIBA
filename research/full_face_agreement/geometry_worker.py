"""Compute the original-F five-zone TEST label cache; publish each verified PNG before its receipt."""
import argparse
import errno
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
from PIL import Image

from frozen_f_partition import REGION_NAMES, VERSION, partition_frozen_f


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "signal_objects_3cd_20261029_v1"
COUNT = 21899
SELECTION = "c3d63a9610ada7bca6161de225e848132a0679e9514d00b92b0f53f14cbc720f"
PROTOCOL = "d7fd5ad88b042b3bd551d92cd3efd2a67cf52b376e2c7b8e4c9b3716401f93da"
MANIFEST = "01f9e286a2cbc868383e34dc4753d9479549442aee855a7544d094d6be9212c5"
CATALOG = "81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89"
GRID = (1024, 1024)
IDS = []
MISSING = {}
STAGE = {}
ALGORITHM = ""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def raw(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound(path, digest):
    data = Path(path).read_bytes()
    require(sha(data) == digest, "Source digest differs: " + str(path))
    return data


def payload(pin):
    return bound(pin["path"], pin["sha256"])


def publish(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".geometry-", delete=False) as out:
        temp = Path(out.name)
        try:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temp.unlink()
            raise
    try:
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def setup():
    global IDS, MISSING, STAGE, ALGORITHM
    selection = json.loads(bound(SRC / "selection.json", SELECTION))
    protocol = json.loads(bound(SRC / "protocol.json", PROTOCOL))
    require(protocol["selection"]["sha256"] == SELECTION and
            selection["source_catalog"]["sha256"] == CATALOG, "Selection/catalog mismatch")
    catalog = json.loads(payload(selection["source_catalog"]))
    IDS = selection["all_group_ids"]
    require(IDS == [r["sample_id"] for r in catalog["rows"]] and catalog["split"] == "test" and
            all(r["split"] == "test" for r in catalog["rows"]) and
            len(IDS) == len(set(IDS)) == COUNT, "Unexpected TEST cohort")
    MISSING = {r["index"]: r["reason"] for r in selection["missing"]}
    require(len(MISSING) == len(selection["missing"]) == 477 and
            all(r["group_id"] == IDS[r["index"]] and r["reason"] in ("no_unique_face", "invalid_geometry")
                for r in selection["missing"]) and
            set(range(COUNT)) - set(MISSING) == set(selection["geometry_valid_indices"]), "Eligibility mismatch")
    path = SRC / "report/independent_audit_v1/consumed_files_sha256.jsonl"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if b'/ready.json"' in line:
                rec = json.loads(line)
                ready = Path(rec["path"])
                if ready.parent.parent == SRC / "staged" and ready.name == "ready.json":
                    i = int(ready.parent.name)
                    require(ready.parent.name == "%06d" % i and i not in STAGE, "Duplicate ready.json")
                    STAGE[i] = rec["sha256"]
    require(digest.hexdigest() == MANIFEST and set(STAGE) == set(range(COUNT)), "Stage audit mismatch")
    sources = {name: sha((ROOT / "code" / name).read_bytes())
               for name in ("frozen_f_partition.py", "fullface_partition.py")}
    ALGORITHM = sha(raw(sources))
    return dict(sources=sources, algorithm_sha256=ALGORITHM, driver_sha256=sha(Path(__file__).read_bytes()),
                selection_sha256=SELECTION, stage_audit_sha256=MANIFEST, protocol_sha256=PROTOCOL)


def decoded(blob):
    with Image.open(io.BytesIO(blob)) as im:
        require(im.format == "PNG" and im.mode == "L" and im.size == GRID, "Invalid label PNG")
        return np.array(im)


def work(i):
    stamp = ROOT / "label_receipts" / ("%06d.json" % i)
    path = ROOT / "labels" / ("%06d.png" % i)
    started = time.monotonic()
    try:
        rd = SRC / ("staged/%06d/ready.json" % i)
        ready = json.loads(bound(rd, STAGE[i]))
        seal = ready.pop("record_sha256")
        require(sha(raw(ready)) == seal and (ready["index"], ready["group_id"], ready["row"]["sample_id"],
                ready["row"]["split"], ready["protocol_sha256"]) == (i, IDS[i], IDS[i], "test", PROTOCOL),
                "Sealed stage or row identity mismatch")
        receipt = json.loads(payload(ready["mask_receipt"]))
        require((receipt["index"], receipt["sample_id"], receipt["catalog_sha256"]) ==
                (i, IDS[i], CATALOG), "Original mask receipt identity mismatch")
        base = dict(index=i, group_id=IDS[i], ready_sha256=STAGE[i], algorithm_sha256=ALGORITHM,
                    selection_sha256=SELECTION, mask_receipt_sha256=ready["mask_receipt"]["sha256"])
        if i in MISSING:
            require(receipt["status"] == MISSING[i] and "geometry" not in ready and not path.exists(),
                    "Wrong technical case")
            row = dict(base, status="technical_missing", missing_reason=MISSING[i])
            if not stamp.exists() or json.loads(stamp.read_bytes()) != row:
                publish(stamp, raw(row))
            return i, "technical_missing", time.monotonic() - started
        require(receipt["status"] == "complete" and Path(ready["geometry"]["path"]) == rd.with_name("geometry.npz"),
                "Missing valid geometry")
        geometry = payload(ready["geometry"])
        with np.load(io.BytesIO(geometry), allow_pickle=False) as z:
            require(set(z.files) == set(REGION_NAMES) | {"F_reference", "F_coverage"}, "Bad geometry fields")
            f = z["F_coverage"]
            require(f.shape == GRID and f.dtype == np.float32 and np.array_equal(f, z["F_reference"]),
                    "Frozen F reference mismatch")
            seeds = {name: z[name] for name in REGION_NAMES}
        with np.load(io.BytesIO(payload(receipt["npz"])), allow_pickle=False) as z:
            require(np.array_equal(f, z["F"]), "F differs from original fractional projection")
        seed_shas = {name: sha(v.tobytes("C")) for name, v in seeds.items()}
        require(all(v.shape == GRID and v.dtype == np.float32 and
                    seed_shas[n] == ready["roi_audit"]["regions"][n]["weight_array_sha256"]
                    for n, v in seeds.items()), "Frozen anatomy seed mismatch")
        base.update(geometry_sha256=ready["geometry"]["sha256"], frozen_F_sha256=sha(f.tobytes("C")),
                    seed_sha256=seed_shas, original_weights_sha256=receipt["npz"]["sha256"])
        if stamp.exists() and path.exists():
            cached = json.loads(stamp.read_bytes())
            if (all(cached.get(k) == v for k, v in base.items()) and cached.get("status") == "complete"
                    and cached.get("record_sha256") == sha(raw({k:v for k,v in cached.items()
                                                                 if k != "record_sha256"}))):
                blob = path.read_bytes()
                if sha(blob) == cached["labels_sha256"] and sha(decoded(blob).tobytes("C")) == cached["label_array_sha256"]:
                    return i, "resumed", time.monotonic() - started
        labels, qc = partition_frozen_f(f, seeds)
        buf = io.BytesIO()
        Image.fromarray(labels).save(buf, format="PNG", compress_level=3)
        blob = buf.getvalue()
        restored = decoded(blob)
        require(np.array_equal(restored, labels) and np.array_equal(restored > 0, f > 0) and
                all(np.all(restored[seeds[n] > 0] == j) for j, n in enumerate(REGION_NAMES, 1)) and
                qc["old_seeds_preserved"] and qc["exact_frozen_F"] and qc["frozen_F_sha256"] == base["frozen_F_sha256"],
                "Output label round trip or geometry invariants failed")
        row = dict(base, status="complete", labels_sha256=sha(blob), label_array_sha256=sha(restored.tobytes("C")),
                   labels_path=str(path.relative_to(ROOT)), version=VERSION,
                   qc={k:v for k,v in qc.items() if k != "islands"})
        row["record_sha256"] = sha(raw(row))
        publish(path, blob)
        publish(stamp, raw(row))
        return i, "complete", time.monotonic() - started
    except Exception:
        err = traceback.format_exc()
        publish(ROOT / "geometry_failures" / ("%06d.json" % i), raw(dict(index=i, group_id=IDS[i], error=err)))
        return i, "failed", err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--run-name", required=True, choices=("pilot", "full"))
    args = parser.parse_args()
    for subdir in ("labels", "label_receipts", "geometry_failures", "geometry_progress"):
        (ROOT / subdir).mkdir(exist_ok=True)
    out = ROOT / "geometry_progress"
    with (out / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = setup()
        indices = args.indices if args.indices is not None else range(COUNT)
        require(len(set(indices)) == len(indices) and all(0 <= i < COUNT for i in indices), "Bad indices")
        counts = dict(complete=0, technical_missing=0, resumed=0, failed=0)
        started = time.time()
        state = dict(meta, pid=os.getpid(), workers=args.workers, requested=len(indices), started=started,
                     status="running", counts=counts)
        def save():
            state["updated"] = time.time()
            state["elapsed_seconds"] = state["updated"] - started
            publish(out / (args.run_name + ".json"), raw(state))
        save()
        with mp.get_context("fork").Pool(args.workers) as pool:
            for index, status, detail in pool.imap_unordered(work, indices, chunksize=1):
                counts[status] += 1
                state["last_index"] = index
                if status == "failed":
                    print(index, detail, flush=True)
                if time.time() - state["updated"] >= 5 or sum(counts.values()) <= 2:
                    save()
        state["status"] = "failed" if counts["failed"] else "complete"
        save()
        print(json.dumps(state, sort_keys=True), flush=True)
        return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
