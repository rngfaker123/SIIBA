"""Publish Full-F TEST ICC(A,1) only after a complete model/target cohort passes."""
import argparse
from collections import defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import agreement_stats


CODE = Path(__file__).resolve().parent
MODELS = {"unet_single": "Single-target U-Net", "unet_multi": "Multi-target U-Net",
          "unetplusplus": "U-Net++", "swin": "Swin", "ldm": "LDM"}
TARGETS = ("uv", "plh", "plv")
REGIONS = ("Forehead", "Image-left cheek", "Image-right cheek", "Nose", "Chin", "Full valid face (F)")
CHANNELS = "RGB"
SELECTION_SHA = "c3d63a9610ada7bca6161de225e848132a0679e9514d00b92b0f53f14cbc720f"
CATALOG_SHA = "81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89"
STATS_SHA = "cac2f1219f819575aa4973236f6d2843736a8631512a9f386dac61b2e0eeff70"
REFERENCE_SHA = "6ed87757eb3ca71ba193d6e66c11b4bdf0a841793dd20fb5e0ace22e477887d0"
KEYS = ("truth", "pred", "area", "labels")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_pinned(path, digest):
    contents = path.read_bytes()
    require(hashlib.sha256(contents).hexdigest() == digest, "Changed pinned file: " + str(path))
    return contents


def original_cohort(root):
    select = root.parent / "signal_objects_3cd_20261029_v1/selection.json"
    selection = json.loads(read_pinned(select, SELECTION_SHA))
    desc = selection["source_catalog"]
    require(desc["sha256"] == CATALOG_SHA, "Unexpected catalog binding")
    catalog = json.loads(read_pinned(Path(desc["path"]), CATALOG_SHA))
    ids = selection["all_group_ids"]
    require(ids == [r["sample_id"] for r in catalog["rows"]] and len(set(ids)) == len(ids) == 21899
            and catalog["split"] == "test" and all(r["split"] == "test" for r in catalog["rows"]),
            "Official TEST identities differ")
    missing = {r["index"]: r["reason"] for r in selection["missing"]}
    require(len(missing) == len(selection["missing"]) == 477 and
            all(ids[r["index"]] == r["group_id"] and r["reason"] in ("no_unique_face", "invalid_geometry")
                for r in selection["missing"]) and
            set(selection["geometry_valid_indices"]) == set(range(21899)) - set(missing)
            and len(selection["geometry_valid_indices"]) == 21422, "Official TEST eligibility differs")
    return ids, missing


def references():
    source = read_pinned(CODE / "original_table3b_for_F_only.tsv", REFERENCE_SHA)
    names = {label: key for key, label in MODELS.items()}
    result = {}
    for row in list(csv.reader(io.StringIO(source.decode("utf-8-sig")), delimiter="\t"))[2:]:
        if row[1] != REGIONS[-1]:
            continue
        for i, target in enumerate(TARGETS):
            values = row[2 + 3*i:5 + 3*i]
            if all(values):
                require(all(re.fullmatch(r"-?\d\.\d{3}", x) for x in values), "Malformed F reference")
                result[names[row[0]], target] = values
    require(len(result) == 12, "Unexpected original TEST coverage")
    return result


def database(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)


def inventory(paths, refs):
    parts = defaultdict(dict)
    for path in paths:
        with closing(database(path)) as db:
            query = "SELECT model,target,COUNT(*) FROM measurements GROUP BY model,target"
            for model, target, count in db.execute(query):
                require((model, target) in refs, "Unexpected model/target in production records: " + str((model, target)))
                parts[model, target][path] = count
    return parts


def records(paths, pair):
    for path in paths:
        with closing(database(path)) as db:
            rows = db.execute("SELECT idx,gid,payload FROM measurements WHERE model=? AND target=?", pair).fetchall()
        yield from rows


def verified_estimates(rows, ids, missing, reference):
    # Fill by official index irrespective of lane order. Reject incomplete and overlapping shards.
    size = len(ids)
    seen = np.zeros(size, dtype=bool)
    values = np.zeros((size, 2, 6, 3), dtype=np.float64)
    areas = np.zeros((size, 6), dtype=np.float64)
    label_hashes = np.zeros((size, 32), dtype=np.uint8)
    row_hashes = np.zeros((size, 32), dtype=np.uint8)
    for idx, gid, raw in rows:
        require(type(idx) is int and 0 <= idx < size and ids[idx] == gid, "Wrong TEST group at index " + str(idx))
        require(not seen[idx], "Duplicate measurement for TEST index " + str(idx))
        seen[idx] = True
        row_hashes[idx] = np.frombuffer(hashlib.sha256(raw.encode("utf8")).digest(), dtype=np.uint8)
        p = json.loads(raw)
        if idx in missing:
            require(p == {"missing_reason": missing[idx]}, "Wrong technical exclusion at index " + str(idx))
            continue
        require(set(p) == set(KEYS) and re.fullmatch("[0-9a-f]{64}", p["labels"]),
                "Invalid geometry or payload at index " + str(idx))
        a, x, y = (np.asarray(p[k]) for k in ("area", "truth", "pred"))
        require(a.shape == (6,) and x.shape == y.shape == (6, 3) and
                all(z.dtype.kind in "fiu" for z in (a, x, y)), "Invalid numeric shape at index " + str(idx))
        areas[idx], values[idx, 0], values[idx, 1] = a, x, y
        label_hashes[idx] = np.frombuffer(bytes.fromhex(p["labels"]), dtype=np.uint8)
    require(seen.all(), "Incomplete official TEST cohort")
    valid = np.ones(size, dtype=bool)
    valid[list(missing)] = False
    a, v = areas[valid], values[valid]
    require(np.isfinite(a).all() and (a > 0).all() and np.isfinite(v).all() and
            v.min() >= 0 and v.max() <= 1 + 1e-12, "Nonfinite or out-of-range measurements")
    require(np.allclose(a[:, :5].sum(axis=1), a[:, 5], rtol=1e-9, atol=1e-8),
            "Region areas do not reconstruct F")
    total = np.einsum("nr,nmrc->nmc", a[:, :5], v[:, :, :5]) / a[:, 5, None, None]
    require(np.allclose(total, v[:, :, 5], rtol=2e-8, atol=2e-9), "Region means do not reconstruct F")
    iccs = []
    for region in range(6):
        estimates = []
        for ch in range(3):
            stat = agreement_stats.summarize_pairs(v[:, 0, region, ch], v[:, 1, region, ch], bootstrap_repeats=0)
            require(stat["n"] == len(v) and stat["icc_a1"] is not None, "Undefined paired ICC(A,1)")
            estimates.append(stat["icc_a1"])
        iccs.append(estimates)
    rounded = [format(e, ".3f") for e in iccs[-1]]
    require(rounded == reference, "Recalculated whole-face F differs: " + str(rounded) + " vs " + str(reference))
    return dict(records_read=size, valid_pairs=len(v), technical_missing=len(missing), icc_a1=iccs,
                original_F_3dp=reference, reproduced_F_3dp=rounded,
                labels_by_official_index_sha256=hashlib.sha256(label_hashes.tobytes()).hexdigest(),
                payloads_by_official_index_sha256=hashlib.sha256(row_hashes.tobytes()).hexdigest())


def publish(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def publish_json(path, obj):
    publish(path, (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def table(accepted):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Model", "Anatomical region"] + [t.upper() + "_" + c for t in TARGETS for c in CHANNELS])
    for model, label in MODELS.items():
        for r, name in enumerate(REGIONS):
            row = [label, name]
            for target in TARGETS:
                data = accepted.get((model, target))
                row.extend(["TBD"] * 3 if data is None else [format(v, ".3f") for v in data["icc_a1"][r]])
            writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def collect(root, output, ids, missing, refs, accepted):
    parts = inventory(sorted((root / "records").glob("*.sqlite")), refs)
    errors = {}
    progress = {}
    for pair, ref in refs.items():
        key = "/".join(pair)
        files = parts[pair]
        count = sum(files.values())
        progress[key] = dict(records_committed=count, required=len(ids), lanes={p.name: n for p, n in files.items()})
        if count != len(ids):
            accepted.pop(pair, None)
            if count > len(ids):
                errors[key] = "Overlapping shards: more rows than the official cohort"
            continue
        if pair not in accepted:
            try:
                result = verified_estimates(records(list(files), pair), ids, missing, ref)
                label_sets = {v["labels_by_official_index_sha256"] for v in accepted.values()}
                require(not label_sets or result["labels_by_official_index_sha256"] in label_sets,
                        "Regional partition identity differs across accepted model/target pairs")
                result.update(model=pair[0], target=pair[1], accepted_at=now(), selection_sha256=SELECTION_SHA,
                              catalog_sha256=CATALOG_SHA, agreement_stats_sha256=STATS_SHA, reference_table_sha256=REFERENCE_SHA)
                accepted[pair] = result
            except (ValueError, KeyError, TypeError) as exc:
                errors[key] = str(exc)
    csv_bytes = table(accepted)
    publish(output / "Table3B_fullF.csv", csv_bytes)
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    audit = {"selection_sha256": SELECTION_SHA, "original_table_for_F_regression_only_sha256": REFERENCE_SHA,
             "measurement": "paired sample-group ICC(A,1), single measurement, absolute agreement",
             "region_order": REGIONS, "channels": list(CHANNELS),
             "table_sha256": csv_hash, "pairs": {"/".join(k): v for k, v in accepted.items()}}
    publish_json(output / "accepted.json", audit)
    status = dict(updated_utc=now(), pid=os.getpid(), accepted_pairs=len(accepted), expected_pairs=len(refs),
                  records_committed=sum(x["records_committed"] for x in progress.values()),
                  records_read=sum(i["records_read"] for i in accepted.values()), pairs=progress,
                  errors=errors, table_sha256=csv_hash, status="complete" if len(accepted) == len(refs) else
                  ("validation_error" if errors else "collecting"))
    publish_json(output / "progress.json", status)
    return status


def main():
    import fcntl
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=CODE.parent)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    require(args.interval >= 10, "Invalid poll interval")
    output = args.root / "output/collection"
    output.mkdir(exist_ok=True, parents=True)
    lock = (output / "collector.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    read_pinned(Path(agreement_stats.__file__), STATS_SHA)
    refs, (ids, missing) = references(), original_cohort(args.root)
    accepted = {}
    while True:
        try:
            state = collect(args.root, output, ids, missing, refs, accepted)
        except (sqlite3.Error, OSError, ValueError) as exc:
            state = dict(status="collection_error", updated_utc=now(), pid=os.getpid(), error=str(exc))
            publish_json(output / "progress.json", state)
        if args.once or state["status"] == "complete":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

