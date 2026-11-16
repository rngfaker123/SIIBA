"""Independently reconcile every archived TEST geometry with its committed label file."""
import argparse
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import geometry_worker as g

FOLLOW = False


def validate(i):
    stem = "%06d" % i
    receipt_path = g.ROOT / "label_receipts" / (stem + ".json")
    while FOLLOW and not receipt_path.exists():
        status = json.loads((g.ROOT / "geometry_progress/full.json").read_bytes())
        g.require(status["status"] == "running" and time.time() - status["updated"] < 180,
                  "No committed receipt: " + stem)
        time.sleep(1)
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload)
    png = g.ROOT / "labels" / (stem + ".png")
    g.require((receipt["index"], receipt["group_id"], receipt["ready_sha256"],
               receipt["selection_sha256"], receipt["algorithm_sha256"]) ==
              (i, g.IDS[i], g.STAGE[i], g.SELECTION, g.ALGORITHM), "Bad identity: " + stem)
    if i in g.MISSING:
        g.require(receipt["status"] == "technical_missing" and receipt["missing_reason"] == g.MISSING[i]
                  and not png.exists(), "Misclassified technical case: " + stem)
        return g.sha(payload), None
    seal = receipt.pop("record_sha256")
    g.require(receipt["status"] == "complete" and g.sha(g.raw(receipt)) == seal, "Invalid label receipt: " + stem)
    lab = g.decoded(g.bound(png, receipt["labels_sha256"]))
    g.require(g.sha(lab.tobytes("C")) == receipt["label_array_sha256"] and lab.max() == 5, "Bad labels: " + stem)
    ready = json.loads(g.bound(g.SRC / "staged" / stem / "ready.json", g.STAGE[i]))
    g.require(ready["geometry"]["sha256"] == receipt["geometry_sha256"], "Source geometry mismatch: " + stem)
    with np.load(io.BytesIO(g.payload(ready["geometry"])), allow_pickle=False) as z:
        f = z["F_coverage"]
        g.require(f.shape == g.GRID and g.sha(f.tobytes("C")) == receipt["frozen_F_sha256"] and
                  np.array_equal(f, z["F_reference"]) and np.array_equal(lab != 0, f > 0),
                  "Frozen support mismatch: " + stem)
        for label, name in enumerate(g.REGION_NAMES, 1):
            seed = z[name]
            g.require(g.sha(seed.tobytes("C")) == receipt["seed_sha256"][name] ==
                      ready["roi_audit"]["regions"][name]["weight_array_sha256"] and
                      np.all(lab[seed > 0] == label), "Reassigned old seed: " + stem + "/" + name)
        areas = np.bincount(lab.ravel(), weights=f.ravel(), minlength=6)
        orig = float(f.sum(dtype=np.float64))
        info = receipt["qc"]
        expected = [info["regions"][name]["effective_area"] for name in g.REGION_NAMES]
        g.require(areas[0] == 0 and np.all(areas[1:] > 0) and
                  np.allclose(areas[1:], expected, rtol=2e-12, atol=1e-9) and
                  np.isclose(areas[1:].sum(), orig, rtol=2e-12, atol=1e-9),
                  "Fractional area mismatch: " + stem)
    return g.sha(payload), (info["satellite_count"], info["satellite_effective_area"], orig)


def main():
    global FOLLOW
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    FOLLOW = args.follow
    started = time.time()
    setup = g.setup()
    out = g.ROOT / "geometry_progress"
    full = json.loads((out / "full.json").read_bytes())
    g.require((full["status"] == "complete" or (FOLLOW and full["status"] == "running"))
              and full["requested"] == g.COUNT and not full["counts"]["failed"]
              and full["algorithm_sha256"] == setup["algorithm_sha256"], "Production state inconsistent")
    inventory = hashlib.sha256()
    qa = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (digest, metric) in enumerate(pool.imap(validate, range(g.COUNT), chunksize=4)):
            inventory.update(("%06d %s\n" % (i, digest)).encode())
            if metric is not None:
                qa.append(metric)
            if i % 500 == 0:
                g.publish(out / "audit_progress.json", g.raw(dict(last_index=i, updated=time.time())))
    expected = {"%06d.json" % i for i in range(g.COUNT)}
    g.require({p.name for p in (g.ROOT / "label_receipts").glob("*.json")} == expected,
              "Extra or missing TEST receipts")
    labels = {"%06d.png" % i for i in range(g.COUNT) if i not in g.MISSING}
    g.require({p.name for p in (g.ROOT / "labels").glob("*.png")} == labels, "Extra or missing label PNGs")
    numbers = np.array(qa)
    audit = dict(status="passed", cohort=g.COUNT, complete=len(qa), original_technical_missing=len(g.MISSING),
                 cases_with_satellites=int(np.sum(numbers[:, 0] != 0)), satellite_count=int(numbers[:, 0].sum()),
                 satellite_area_fraction_of_F_overall=float(numbers[:, 1].sum()/numbers[:, 2].sum()),
                 max_single_case_satellite_F_fraction=float(np.max(numbers[:, 1]/numbers[:, 2])),
                 original_positive_F_support_mismatches=0, reassigned_seed_cells=0,
                 fractional_area_mismatches=0, receipt_inventory_sha256=inventory.hexdigest(),
                 algorithm_sha256=g.ALGORITHM, audit_source_sha256=g.sha(Path(__file__).read_bytes()),
                 elapsed_seconds=time.time()-started)
    g.publish(out / "audit.json", g.raw(audit))
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
