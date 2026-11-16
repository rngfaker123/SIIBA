"""Optional archived-UV-minus-PL group-paired contrasts after PL pair acceptance."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np

import pl_collect as c


EXPERIMENTS = Path("/study/siiba/experiments")
COMPLETIONS = (
    ("test_gpu_program_consistency_20261027_v1/completion.json",
     "2cebb37a621d33c4819998ee294decc2a3e690e39bf03838592780014abf1fdf"),
    ("test_resume_eight_gpu_20261028_v1/scientific_completion.json",
     "7628f303ef30778806ac8f41cacb6bd29bfd017e84f347ccb131d07fda866a3a"),
)


def original_inventory(arch):
    newer = arch in ("swin", "ldm")
    rel, pin = COMPLETIONS[int(newer)]
    complete = EXPERIMENTS / rel
    c.require(c.file_sha(complete) == pin, "Historical UV completion differs")
    manifest_path = complete.parent / "record_manifest.json"
    desc, = [v for v in c.read(complete)["outputs"] if v["path"] == str(manifest_path)]
    manifest = c.read(c.checked(desc, EXPERIMENTS))
    entries = [v for v in manifest["records"] if v["model"] == arch]
    mapped = {v["index"]: v for v in entries}
    c.require(len(entries) == c.N_TEST and set(mapped) == set(range(c.N_TEST)), "UV inventory incomplete")
    return complete, desc, mapped


def verified_reference(data, arch):
    path = c.HERE / "reference" / "uv_reference.json"
    c.require(c.file_sha(path) == "411e9c64af1d82baed13e532ffa9c5bd620c4637cfd6ea3adab98c4dedabe0d2",
              "Frozen UV summary differs")
    refs = [v for v in c.read(path)["reconstruction"] if v["model"] == arch]
    for v in refs:
        ri, mi = c.DOMAINS.index(v["region"]), c.METRICS.index(v["metric"])
        a = data[:, ri, mi]
        a = a[np.isfinite(a)]
        c.require(len(a) == v["n_valid"] and math.isclose(float(np.mean(a)), v["mean"], abs_tol=2e-10)
                  and math.isclose(float(np.std(a, ddof=1)), v["sample_std_ddof1"], abs_tol=2e-10),
                  "UV raw-to-historical-summary mismatch for " + str((arch, v["region"], v["metric"])))
    return c.descriptor(path), len(refs)


def contrasts(root, arch, target):
    root = Path(root).resolve()
    c.require(arch in c.MODEL and target in c.TARGETS, "Unrecognized PL pair")
    folder = root / "reports" / (arch + "_" + target)
    receipt = c.read(folder / "acceptance.json")
    c.require(receipt["status"] == "passed" and receipt["architecture"] == arch and receipt["target"] == target,
              "PL pair has not passed acceptance")
    manifest = c.read(c.checked(receipt["input_manifest"], root))
    summary = c.read(c.checked(receipt["outputs"]["summary"], root))
    c.require(summary["inputs"] == receipt["input_manifest"], "PL report binding differs")
    ids, valid, original_missing, _, _ = c.cohort(root, c.read(c.checked(manifest["protocol"], root)))
    complete, uv_manifest, inventory = original_inventory(arch)
    uv = np.full((len(ids), 4, 4), np.nan)
    pl = np.full_like(uv, np.nan)
    present = np.zeros((4, 4), dtype=int)
    missing = {(r, m): Counter() for r in c.DOMAINS for m in c.METRICS}
    uv_checkpoint = set()
    seen = 0
    for shard in manifest["chunks"]:
        blob = (root / shard["path"]).read_bytes()
        c.require(c.digest(blob) == shard["sha256"], "PL chunk bytes changed")
        for line in gzip.decompress(blob).splitlines():
            p = json.loads(line)
            i = seen
            seen += 1
            c.require(p["index"] == i and ids[i] == p["sample_id"], "PL index/order differs")
            ref = inventory[i]
            path = Path(ref["source"]["path"])
            c.require(path == complete.parent / "records" / arch / ("%06d.json" % i), "UV source path differs")
            raw = path.read_bytes()
            c.require(c.digest(raw) == ref["source"]["sha256"], "Historical UV record bytes changed")
            u = json.loads(raw)
            c.require(u["index"] == i and u["sample_id"] == ids[i] and u["model_id"] == arch
                      and u["record_sha256"] == ref["record_sha256"], "UV paired index, model or seal differs")
            uv_checkpoint.add(u["checkpoint_sha256"])
            for k in ("mask_ref", "mask_receipt"):
                c.require(p["input_binding"][k] == u[k], "PL/UV frozen mask identity differs")
            if i not in valid:
                c.require(u["status"] == "unavailable_geometry", "UV technical status differs")
                continue
            c.require(u["status"] == "complete", "UV group is not available")
            for ri, region in enumerate(c.DOMAINS):
                a = u["combined"]["regions"][region]
                b = p["measurements"][target]["regions"][region]
                for old, new in (("effective_pixel_area", "effective_area"),
                                 ("ssim_effective_center_area", "ssim_center_area")):
                    c.require(math.isclose(a[old], b[new], rel_tol=1e-10, abs_tol=1e-6),
                              f"UV/PL support mismatch at {i}, {region}, {old}")
                for mi, m in enumerate(c.METRICS):
                    if m not in a:
                        continue
                    present[ri, mi] += 1
                    x, y = a[m], b[m]
                    if x is not None:
                        c.require(c.finite(x), "Nonfinite historical UV value")
                        uv[i, ri, mi] = x
                    if y is not None:
                        c.require(c.finite(y), "Nonfinite PL value")
                        pl[i, ri, mi] = y
                    if x is None or y is None:
                        # MSE 沿用 L1 缺失原因
                        xr = a.get(m + "_reason") or (a.get("l1_reason") if m == "mse" else None)
                        yr = b[m + "_reason"]
                        c.require(x is not None or bool(xr), "UV null without original reason")
                        missing[region, m]["UV=" + (xr or "valid") + ",PL=" + (yr or "valid")] += 1
    c.require(seen == len(ids) and len(uv_checkpoint) == 1 and
              np.all((present == 0) | (present == len(valid))), "Incomplete UV field/row or mixed checkpoint")
    ref, n_checked = verified_reference(uv, arch)
    delta = uv - pl
    draws = c.bootstrap_means(delta, valid)
    cells = []
    for ri, region in enumerate(c.DOMAINS):
        for mi, m in enumerate(c.METRICS):
            if not present[ri, mi]:
                cells.append(dict(region=region, metric=m, status="historical_raw_field_absent", n=None))
                continue
            mask = np.isfinite(delta[:, ri, mi])
            v = delta[mask, ri, mi].tolist()
            finite = draws[:, ri, mi]
            finite = finite[np.isfinite(finite)]
            ci = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
            reasons = Counter(original_missing.values()) + missing[region, m]
            c.require(len(v) + sum(reasons.values()) == len(ids), "Paired missingness differs")
            raw_uv = uv[:, ri, mi]
            raw_uv = raw_uv[np.isfinite(raw_uv)]
            cells.append(dict(region=region, metric=m, status="verified", n=len(v),
                              mean=statistics.fmean(v) if v else None,
                              sample_SD_ddof1=statistics.stdev(v) if len(v) > 1 else None,
                              ci95_low=ci[0], ci95_high=ci[1], bootstrap_valid=len(finite),
                              valid_index_mask_sha256=c.digest(mask.tobytes()), missing_reasons=dict(reasons),
                              UV_raw_n=len(raw_uv), UV_raw_mean=float(raw_uv.mean()) if len(raw_uv) else None,
                              UV_raw_SD_ddof1=float(raw_uv.std(ddof=1)) if len(raw_uv) > 1 else None))
    result = dict(status="passed", architecture=arch, target=target, difference="UV_minus_" + target.upper(),
                  n_total=len(ids), bootstrap_repeats=2000, seed=20261030, unit="paired_sample_group",
                  UV_checkpoint_sha256=uv_checkpoint.pop(), UV_completion=c.descriptor(complete),
                  UV_record_manifest=uv_manifest, accepted_PL=c.descriptor(folder / "acceptance.json", root),
                  UV_summary=ref, UV_historical_summary_cells_reconciled=n_checked, cells=cells)
    c.emit_json(folder / "UV_minus_PL.json", result)
    c.render_tables(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--architecture", choices=c.MODEL, required=True)
    parser.add_argument("--target", choices=c.TARGETS, required=True)
    args = parser.parse_args()
    data = contrasts(args.root, args.architecture, args.target)
    print(data["status"], data["architecture"], data["difference"])
