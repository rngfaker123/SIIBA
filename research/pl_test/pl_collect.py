"""Verify complete, fixed TEST shards before publishing any PL endpoint."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import gzip
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
MODEL = {"unet_single": "单目标 U-Net", "unet_multi": "多目标 U-Net",
         "unetplusplus": "U-Net++", "swin": "Swin", "ldm": "LDM"}
TARGETS = ("plh", "plv")
DOMAINS = ("P", "D", "O", "F")
METRICS = ("l1", "mse", "psnr", "ssim")
ROI = {"forehead": "额头", "image_left_cheek": "图像左颊",
       "image_right_cheek": "图像右颊", "nose": "鼻部", "chin": "下巴",
       "F_reference": "F参考"}
SIGNALS = ("R", "G", "B", "RGB_mean_secondary")
N_TEST, N_GEOMETRY, WIDTH = 21899, 21422, 32
REF_PINS = {
    "agreement_stats.py": "cac2f1219f819575aa4973236f6d2843736a8631512a9f386dac61b2e0eeff70",
    "report.py": "e312aa4314d28be732c4b63ae90710257afcb369027c9cdf25c5e8a6b6246c3b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def descriptor(path, relative_to=None):
    path = Path(path)
    return {"path": str(path.relative_to(relative_to) if relative_to else path), "sha256": file_sha(path)}


def checked(desc, root):
    path = root / desc["path"]
    require(file_sha(path) == desc["sha256"], "SHA mismatch: " + str(path))
    return path


def publish(path, data, mutable=False):
    """Publish as one file. A retry may accept byte-identical immutable output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(data, bytes):
        data = data.encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=".staging-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mutable:
            os.replace(name, str(path))
        else:
            try:
                os.link(name, str(path))
            except FileExistsError:
                require(path.read_bytes() == data, "Refuse conflicting result: " + str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def emit_json(path, value):
    publish(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def references():
    import importlib.util
    folder = HERE / "reference"
    for name, pin in REF_PINS.items():
        require(file_sha(folder / name) == pin, "Native 3B source changed: " + name)
    spec = importlib.util.spec_from_file_location("agreement_stats", folder / "agreement_stats.py")
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    # 审计沿用原模块名
    prior = sys.modules.get("agreement_stats")
    sys.modules["agreement_stats"] = stats
    try:
        spec = importlib.util.spec_from_file_location("pl_native_3b_report", folder / "report.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
    finally:
        if prior is None:
            del sys.modules["agreement_stats"]
        else:
            sys.modules["agreement_stats"] = prior
    return stats.summarize_pairs, report.independent_point_check


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def check_reconstruction(cell):
    area, centers = cell["effective_area"], cell["ssim_center_area"]
    require(finite(area) and finite(centers) and 0 <= centers <= area + 1e-6,
            "Invalid reconstruction effective area")
    for m in METRICS:
        x, reason = cell[m], cell[m + "_reason"]
        missing = area == 0 or (m == "psnr" and cell["mse"] == 0) or (m == "ssim" and centers == 0)
        require((x is None and isinstance(reason, str) and bool(reason)) if missing
                else (finite(x) and reason is None), "Inconsistent " + m + " or undefined reason")
    if area:
        require(0 <= cell["l1"] <= 1.000001 and 0 <= cell["mse"] <= 1.000001,
                "L1/MSE outside encoded image range")
        if cell["mse"] > 0:
            require(math.isclose(cell["psnr"], -10 * math.log10(cell["mse"]), abs_tol=1e-6),
                    "PSNR/MSE mismatch")


def check_anatomy(cell):
    area = cell["effective_area"]
    require(finite(area) and area >= 0, "Invalid anatomical ROI area")
    for key in ("truth", "prediction"):
        means = cell[key]
        require(set(means) == set(SIGNALS), "Anatomical signal set differs")
        if area == 0:
            require(all(v is None for v in means.values()) and cell["reason"] == "empty_anatomical_roi",
                    "Empty anatomical ROI has fabricated data or missing reason")
        else:
            require(all(finite(v) and 0 <= v <= 1.000001 for v in means.values()) and cell["reason"] is None,
                    "Invalid anatomical mean")
            require(math.isclose(statistics.fmean(means[c] for c in SIGNALS[:3]), means[SIGNALS[-1]],
                                 abs_tol=1e-6), "Arithmetic RGB secondary signal differs")


def cohort(root, protocol):
    require(protocol["selection"]["sha256"] == "c3d63a9610ada7bca6161de225e848132a0679e9514d00b92b0f53f14cbc720f",
            "Unrecognized frozen TEST selection")
    selection = read(checked(protocol["selection"], root))
    ids = selection["all_group_ids"]
    valid = set(selection["geometry_valid_indices"])
    require(len(ids) == len(set(ids)) == N_TEST and all(isinstance(i, str) and i for i in ids),
            "Wrong ordered TEST group inventory")
    require(len(selection["geometry_valid_indices"]) == len(valid) == N_GEOMETRY and
            all(type(i) is int and 0 <= i < len(ids) for i in valid), "Wrong frozen geometry index set")
    missing = {v["index"]: v["reason"] for v in selection["missing"]}
    require(set(missing) == set(range(len(ids))) - valid, "Frozen missing inventory differs")
    catalog = read(checked(selection["source_catalog"], root))
    rows = catalog["rows"]
    require(catalog["split"] == "test" and [r["sample_id"] for r in rows] == ids
            and all(r["split"] == "test" for r in rows), "Frozen TEST catalog or order differs")
    return ids, valid, missing, rows, selection["source_catalog"]["sha256"]


def load_pair(root, arch, target, ids, valid, original_missing, rows, catalog_sha,
              protocol_sha, model_sha, targets, lanes):
    key = "both" if arch == "unet_multi" else target
    folder = root / "chunks" / (arch + "_" + key)
    expected = {f"{i:05d}.jsonl.gz" for i in range(0, len(ids), WIDTH)}
    actual = {p.name for p in folder.glob("*.jsonl.gz")}
    require(actual == expected, f"{arch}/{target}: shard inventory incomplete or unexpected "
            f"(missing {len(expected - actual)}, extra {len(actual - expected)})")
    recon = np.full((len(ids), 4, 4), np.nan)
    pairs = np.full((len(ids), len(ROI), len(SIGNALS), 2), np.nan)
    reasons = {(r, m): Counter() for r in DOMAINS for m in METRICS}
    anatomical_missing = {r: Counter() for r in ROI}
    technical = Counter()
    shards, devices = [], Counter()
    for start in range(0, len(ids), WIDTH):
        path = folder / f"{start:05d}.jsonl.gz"
        blob = path.read_bytes()
        lines = gzip.decompress(blob).splitlines()
        require(len(lines) == min(WIDTH, len(ids) - start), f"Short or duplicate chunk {start}")
        shards.append({"path": str(path.relative_to(root)), "sha256": digest(blob), "rows": len(lines)})
        for i, line in enumerate(lines, start):
            row = json.loads(line)
            require(type(row["index"]) is int and row["index"] == i and row["sample_id"] == ids[i] and
                    row["architecture"] == arch and row["model_identity_sha256"] == model_sha and
                    row["protocol_sha256"] == protocol_sha and set(row["measurements"]) == set(targets),
                    f"Order, identity or measurement target mismatch at {i}")
            device = row["runtime"]["device_uuid"]
            require(device == lanes[(start // WIDTH) % 3], "Shard/device pilot lane mismatch")
            require(type(row["forward_count"]) is int and row["forward_count"] == int(i in valid),
                    "Wrong production forward count")
            binding = row["input_binding"]
            require(binding["catalog_sha256"] == catalog_sha and
                    binding["targets"] == {t: rows[i][t] for t in targets} and
                    binding["rgb"] == rows[i]["rgb"] and binding["uv"] == rows[i]["uv"],
                    f"Frozen input/PL target evidence differs at {i}")
            devices[device] += 1
            v = row["measurements"][target]
            if i not in valid:
                reason = v["missing_reason"]
                require(v["status"] == "unavailable_geometry" and isinstance(reason, str) and reason and
                        v["regions"] is None and v["anatomy"] is None, f"Technical record differs at {i}")
                require(reason in (original_missing[i], "mask_" + original_missing[i]),
                        f"Frozen technical reason differs at {i}")
                technical[reason] += 1
                continue
            require(v["status"] == "ok" and v["missing_reason"] is None and
                    set(v["regions"]) == set(DOMAINS) and set(v["anatomy"]) == set(ROI),
                    f"Geometry-valid endpoint inventory differs at {i}")
            for ri, r in enumerate(DOMAINS):
                region = v["regions"][r]
                check_reconstruction(region)
                for mi, metric in enumerate(METRICS):
                    if region[metric] is None:
                        reasons[r, metric][region[metric + "_reason"]] += 1
                    else:
                        recon[i, ri, mi] = region[metric]
            for ri, region in enumerate(ROI):
                item = v["anatomy"][region]
                check_anatomy(item)
                if item["effective_area"] == 0:
                    anatomical_missing[region][item["reason"]] += 1
                    continue
                for si, signal in enumerate(SIGNALS):
                    pairs[i, ri, si] = item["truth"][signal], item["prediction"][signal]
    return recon, pairs, reasons, anatomical_missing, technical, shards, devices


def bootstrap_means(recon, valid):
    """Original 3F paired group resampling over the frozen geometry-valid cohort."""
    z = recon[sorted(valid)].reshape(len(valid), -1)
    mask = np.isfinite(z)
    both = np.concatenate([np.where(mask, z, 0), mask], axis=1)
    rng = np.random.default_rng(20261030)
    draws = np.full((2000, z.shape[1]), np.nan)
    for first in range(0, len(draws), 32):
        n = min(32, len(draws) - first)
        ix = rng.integers(len(z), size=(n, len(z)))
        counts = np.stack([np.bincount(row, minlength=len(z)) for row in ix])
        sums, den = np.split(counts.astype(float) @ both, 2, axis=1)
        np.divide(sums, den, out=draws[first:first + n], where=den > 0)
    return draws.reshape(-1, 4, 4)


def summarize(recon, pairs, valid, reasons, anatomical_missing, technical):
    summarize_pairs, independent_check = references()
    reconstruction, agreement = [], []
    draws = bootstrap_means(recon, valid)
    for ri, region in enumerate(DOMAINS):
        for mi, metric in enumerate(METRICS):
            col = recon[:, ri, mi]
            mask = np.isfinite(col)
            z = col[mask].tolist()
            missing = technical + reasons[region, metric]
            require(sum(missing.values()) + len(z) == len(col), "3F loss accounting differs")
            mean = statistics.fmean(z) if z else None
            sd = statistics.stdev(z) if len(z) > 1 else None
            if mean is not None:
                require(math.isclose(mean, np.mean(z), abs_tol=1e-11), "3F mean audit failed")
            if sd is not None:
                require(math.isclose(sd, np.std(z, ddof=1), abs_tol=1e-11), "3F SD audit failed")
            bs = draws[:, ri, mi]
            bs = bs[np.isfinite(bs)]
            ci = np.quantile(bs, [.025, .975]).tolist() if len(bs) else [None, None]
            reconstruction.append(dict(region=region, metric=metric, n=len(z), mean=mean,
                                       sample_SD_ddof1=sd, ci95_low=ci[0], ci95_high=ci[1],
                                       bootstrap_valid=len(bs), undefined_reasons=dict(missing),
                                       valid_index_mask_sha256=digest(mask.tobytes())))
    for ri, region in enumerate(ROI):
        for si, signal in enumerate(SIGNALS):
            z = pairs[:, ri, si]
            mask = np.isfinite(z).all(axis=1)
            x, y = z[mask].T
            missing = technical + anatomical_missing[region]
            require(len(x) + sum(missing.values()) == len(z), "3B loss accounting differs")
            stats = summarize_pairs(x, y, bootstrap_repeats=2000, seed=20261026)
            audit = independent_check(x, y, stats)
            agreement.append(dict(region=region, signal=signal, n=len(x), statistics=stats,
                                  undefined_reasons=dict(missing), independent_point_audit=audit,
                                  valid_index_mask_sha256=digest(mask.tobytes())))
    return reconstruction, agreement


def as_tsv(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, dict)
                      else v for k, v in row.items()} for row in rows)
    return out.getvalue()


def collect_pair(output_root, architecture, target):
    """Accept one complete PL task. Joint tasks share shards and have separate reports."""
    root = Path(output_root).resolve()
    require(architecture in MODEL and target in TARGETS, "Unknown architecture/target")
    key = architecture + "_" + ("both" if architecture == "unet_multi" else target)
    targets = list(TARGETS) if architecture == "unet_multi" else [target]
    protocol_path, binding_path = root / "protocol.json", root / "bindings" / (key + ".json")
    protocol_sha = file_sha(protocol_path)
    protocol, binding = read(protocol_path), read(binding_path)
    ids, valid, missing, rows, catalog_sha = cohort(root, protocol)
    ident = binding["model_identity_sha256"]
    require(binding["architecture"] == architecture and binding["targets"] == targets and
            binding["protocol_sha256"] == protocol_sha and isinstance(ident, str) and
            len(ident) == 64 and all(c in "0123456789abcdef" for c in ident), "Model binding differs")
    devices = []
    for desc in binding["pilot_acceptances"]:
        pilot = read(checked(desc, root))
        require(pilot["status"] == "passed" and isinstance(pilot["device_uuid"], str) and
                bool(pilot["device_uuid"]) and pilot["model_identity_sha256"] == ident and
                pilot["protocol_sha256"] == protocol_sha and pilot["architecture"] == architecture
                and pilot["targets"] == targets, "Unbound or unsuccessful device pilot")
        devices.append(pilot["device_uuid"])
    require(len(set(devices)) == len(devices) == 3, "Three distinct device pilot gates are required")
    recon, pairs, reasons, anatomical_missing, technical, shards, used = load_pair(
        root, architecture, target, ids, valid, missing, rows, catalog_sha, protocol_sha, ident, targets, devices)
    require(set(used) <= set(devices), "Unvalidated device in TEST results")
    folder = root / "reports" / (architecture + "_" + target)
    provenance = {"protocol": descriptor(protocol_path, root), "binding": descriptor(binding_path, root),
                  "selection": protocol["selection"], "pilots": binding["pilot_acceptances"],
                  "native_regional_sources": REF_PINS, "collector_sha256": file_sha(__file__),
                  "model_identity_sha256": ident, "chunks": shards}
    manifest = folder / "input_manifest.json"
    emit_json(manifest, provenance)
    acceptance_path = folder / "acceptance.json"
    if acceptance_path.exists():
        accepted = read(acceptance_path)
        require(accepted["status"] == "passed" and accepted["input_manifest"] == descriptor(manifest, root),
                "Existing acceptance no longer matches provenance")
        for desc in accepted["outputs"].values():
            checked(desc, root)
        render_tables(root)
        return read(checked(accepted["outputs"]["summary"], root))
    re, ag = summarize(recon, pairs, valid, reasons, anatomical_missing, technical)
    summary = {"schema": "PL_TEST_3F_3B_v1", "status": "complete", "architecture": architecture,
               "target": target, "split": "TEST", "cohort": len(ids), "geometry_valid": len(valid),
               "geometry_missing": dict(technical), "rows_per_device": dict(used), "3F": re, "3B": ag,
               "3F_bootstrap": {"repeats": 2000, "seed": 20261030, "unit": "geometry_valid_sample_group"},
               "inputs": descriptor(manifest, root)}
    summ_path = folder / "summary.json"
    emit_json(summ_path, summary)
    publish(folder / "Table3F.tsv", as_tsv(re))
    publish(folder / "Table3B.tsv", as_tsv([
        {"region": x["region"], "signal": x["signal"], **x["statistics"], "undefined_reasons": x["undefined_reasons"]}
        for x in ag]))
    emit_json(acceptance_path, {"schema": "PL_TEST_pair_acceptance_v1", "status": "passed",
              "architecture": architecture, "target": target, "records": len(ids), "geometry_valid": len(valid),
              "reconstruction_cells": len(re), "agreement_cells": len(ag),
              "input_manifest": descriptor(manifest, root),
              "outputs": {"summary": descriptor(summ_path, root), "3F": descriptor(folder / "Table3F.tsv", root),
                          "3B": descriptor(folder / "Table3B.tsv", root)}})
    render_tables(root)
    return summary


def number(x):
    return "NA" if x is None else format(x, ".3f")


def interval(s, key):
    return "[" + number(s[key + "_ci95_low"]) + ", " + number(s[key + "_ci95_high"]) + "]"


def table(caption, header, rows):
    def cells(tag, row):
        return "<tr>" + "".join("<" + tag + ">" + html.escape(str(v)) + "</" + tag + ">" for v in row) + "</tr>"
    return ("<table><caption>" + html.escape(caption) + "</caption><thead>" + cells("th", header) +
            "</thead><tbody>" + "".join(cells("td", row) for row in rows) + "</tbody></table>")


@contextmanager
def render_lock(folder):
    """Serialize read-and-publish so simultaneous pair completions cannot hide a newer result."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".tables.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def render_tables(output_root):
    root = Path(output_root).resolve()
    with render_lock(root / "reports"):
        return _render_tables(root)


def _render_tables(root):
    re, ag, states = {}, {}, {}
    for arch in MODEL:
        for target in TARGETS:
            receipt = root / "reports" / (arch + "_" + target) / "acceptance.json"
            if not receipt.exists():
                states[arch, target] = "待验收"
                continue
            accepted = read(receipt)
            require(accepted["architecture"] == arch and accepted["target"] == target and
                    accepted["status"] == "passed", "Invalid table source")
            s = read(checked(accepted["outputs"]["summary"], root))
            require(s["target"] == target and s["architecture"] == arch and s["status"] == "complete" and
                    s["cohort"] == N_TEST and s["geometry_valid"] == N_GEOMETRY, "Summary identity differs")
            require(len(s["3F"]) == 16 and len(s["3B"]) == 24, "Summary dimensions differ")
            re.update({(arch, target, x["region"], x["metric"]): x for x in s["3F"]})
            ag.update({(arch, target, x["region"], x["signal"]): x for x in s["3B"]})
            states[arch, target] = "已验收"
    def cell(a, t, r, k, fmt):
        if states[a, t] == "待验收":
            return "待填"
        return fmt(re[a, t, r, k])

    def ms(x):
        return number(x["mean"]) + " ± " + number(x["sample_SD_ddof1"])

    def icc(a, t, r, sig):
        return "待填" if states[a, t] == "待验收" else number(ag[a, t, r, sig]["statistics"]["icc_a1"])

    body = ["<h1>PL 目标的 TEST 重建与区域强度一致性</h1>",
            "<p>已通过整组验收的模型－目标组合 " + str(sum(s == "已验收" for s in states.values())) +
            " / 10。待填表示等待完整 TEST 和统计验收。NA 表示已验收单元中相应统计量无定义，原因保存在汇总文件。</p>",
            "<p>每个样本组等权。重建沿用同组真实 UV 定义的 P（卟啉）、D（UV spots）、O（其他）与 F（全有效域）。"
            "解剖强度为目标图像的编码强度。多目标 U-Net 的两个输出头共用原联合模型。</p>"]
    combos = [(a, t) for a in MODEL for t in TARGETS]
    body.append(table("表 3F · PL 区域 L1（均值 ± 样本 SD）", ["模型", "目标", *DOMAINS],
                      [[MODEL[a], t.upper()] + [cell(a, t, r, "l1", ms) for r in DOMAINS] for a, t in combos]))
    body.append(table("表 3B · PL 区域平均编码强度 ICC(A,1)", ["模型", "目标", "通道", *ROI.values()],
                      [[MODEL[a], t.upper(), sig] + [icc(a, t, r, sig) for r in ROI]
                       for a, t in combos for sig in SIGNALS[:3]]))
    body.append("<details><summary>S3F · 各区域的完整重建指标</summary>")
    for r in DOMAINS:
        rows = []
        for a, t in combos:
            ns = "/".join(cell(a, t, r, k, lambda x: str(x["n"])) for k in METRICS)
            rows.append([MODEL[a], t.upper(), ns] + [cell(a, t, r, k, ms) for k in METRICS])
        body.append(table("S3F · " + r, ["模型", "目标", "有效 N（按指标顺序）", "L1 ↓", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"], rows))
    body.append("</details><details><summary>S3B · 各解剖区域、原色通道与辅助算术均值</summary>")
    body.append("<p>偏差为预测减真实，差值 SD 用 ddof=1。BA 一致性界限为偏差 ± 1.96 倍差值 SD。"
                "ICC 采用双向混合绝对一致单次测量模型。组级配对 bootstrap 2,000 次，种子为 20261026。"
                "每项完整区间（包括 BA 上下界自身的 CI）见已验收组合的 Table3B.tsv。</p>")
    for r, label in ROI.items():
        rows = []
        for a, t in combos:
            for sig in SIGNALS:
                stat = ag.get((a, t, r, sig))
                values = ["待填"] * 4
                if stat is not None:
                    z = stat["statistics"]
                    values = [str(stat["n"]), number(z["bias"]) + " ± " + number(z["difference_sample_sd_ddof1"]),
                              number(z["icc_a1"]) + " " + interval(z, "icc_a1"),
                              "[" + number(z["loa_low"]) + ", " + number(z["loa_high"]) + "]"]
                rows.append([MODEL[a], t.upper(), "RGB算术均值（辅助）" if sig == SIGNALS[-1] else sig, *values])
        body.append(table("S3B · " + label, ["模型", "目标", "通道", "有效 N", "偏差 ± 差值 SD",
                                               "ICC(A,1) [95%CI]", "BA 95%一致性界限"], rows))
    body.append("</details>")
    uv_path = HERE / "reference" / "uv_reference.json"
    if uv_path.exists():
        body.extend(uv_tables(uv_path))
    body.append("<details><summary>UV − PL · 同组配对的区域重建指标差异</summary>"
                "<p>仅对冻结 UV 逐组记录和 PL 结果均已核验的组合计算。正差值表示 UV 指标数值较大。"
                "各指标按共同有效组报告，配对组 bootstrap 2,000 次，种子为 20261030。</p>")
    archived = 0
    for a, t in combos:
        source = root / "reports" / (a + "_" + t) / "UV_minus_PL.json"
        if not source.is_file():
            continue
        z = read(source)
        require(states[a, t] == "已验收" and z["status"] == "passed" and z["architecture"] == a and
                z["target"] == t and z["accepted_PL"] == descriptor(source.parent / "acceptance.json", root),
                "UV contrast is stale or belongs to a different pair")
        archived += 1
        rows = []
        for x in z["cells"]:
            values = ["原始记录无该项"] * 4
            if x["status"] == "verified":
                values = [str(x["n"]), ms(x), "[" + number(x["ci95_low"]) + ", " + number(x["ci95_high"]) + "]",
                          f"{number(x['UV_raw_mean'])} ± {number(x['UV_raw_SD_ddof1'])} ({x['UV_raw_n']})"]
            rows.append([x["region"], x["metric"].upper(), *values])
        body.append(table(MODEL[a] + " · UV − " + t.upper(), ["区域", "指标", "配对 N", "差值均值 ± SD",
                          "均值差 95%CI", "原始 UV 均值 ± SD（自身 N）"], rows))
    if archived == 0:
        body.append("<p>当前尚无已核验的 UV−PL 配对结果。</p>")
    body.append("</details>")
    style = "body{font:15px/1.55 Arial,'Microsoft YaHei',sans-serif;color:#182329;max-width:1100px;margin:32px auto;padding:0 16px}table{border-collapse:collapse;width:100%;margin:15px 0 30px;font-variant-numeric:tabular-nums}caption{text-align:left;font-size:1.1em;font-weight:bold;padding:8px 0}th,td{padding:5px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}summary{font-weight:bold;cursor:pointer;padding:6px 0}"
    export = root / "reports" / "PL_TEST_tables.html"
    publish(export, "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>PL TEST 统计表</title><style>"
            + style + "</style><body>" + "\n".join(body) + "</body></html>", mutable=True)
    return export


def uv_tables(path):
    require(file_sha(path) == "411e9c64af1d82baed13e532ffa9c5bd620c4637cfd6ea3adab98c4dedabe0d2",
            "Portable UV reference changed")
    uv = read(path)
    require(uv["status"] == "verified_native_UV_reference", "Unverified UV source")
    data = {(x["model"], x["region"], x["metric"]): x for x in uv["reconstruction"]}
    pairs = {(x["model"], x["region"], x["signal"]): x for x in uv["agreement"]}
    body = ["<details><summary>已验收 UV 对照 · 原模型的冻结 TEST 汇总</summary>",
            "<p>以下原始 UV 结果与各 PL 目标分列。原 UV 汇总含区域 L1、PSNR 和 P/D/O 的 SSIM；"
            "各区 MSE 及 F-SSIM 需另行核验原始逐组记录。所有数字从原始未舍入单元直接显示。</p>"]
    for metric, label in [("l1", "L1"), ("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
        rows = []
        for a in MODEL:
            vals = []
            for r in DOMAINS:
                x = data.get((a, r, metric))
                vals.append("待核原始记录" if x is None else
                            f"{number(x['mean'])} ± {number(x['sample_std_ddof1'])} ({x['n_valid']})")
            rows.append([MODEL[a], *vals])
        body.append(table("UV 对照 · " + label + "，均值 ± 样本 SD（有效 N）", ["模型", *DOMAINS], rows))
    body.append(table("UV 对照 · 区域平均编码强度 ICC(A,1)", ["模型", "通道", *ROI.values()],
                      [[MODEL[a], sig] + [number(pairs[a, r, sig]["statistics"]["icc_a1"]) for r in ROI]
                       for a in MODEL for sig in SIGNALS[:3]]))
    body.append("</details>")
    return body


def pack_uv(source):
    """Prepare an immutable portable copy after cross-checking the accepted full-support tables."""
    require(file_sha(source) == "291ef871a45719a5053dd05690b7616b034aaadcba5eb6f2c4a7c8b37e692be0",
            "UV source is not the frozen accepted v3 summary")
    j = read(source)
    require(j["source_status"] == "independent_record_and_statistics_audit_passed" and
            j["sample_count_per_model"] == N_TEST and list(MODEL) == j["models"], "UV scope differs")
    pins = [j["sources"][i] for i in (0, 1, 4, 5)]
    native = [read(checked(p, Path.cwd())) for p in pins]
    acceptance = j["independent_final_acceptance"]
    require(read(checked(acceptance, Path.cwd()))["status"] == "passed", "UV audit failed")
    tr = [dict(model=a, region=r, metric=m, **v) for a, regions in native[0]["models"].items()
          for r, metrics in regions.items() for m, v in metrics.items()] + native[1]["cells"]
    reg = native[2]["cells"] + native[3]["cells"]
    def index(rows, axis):
        return {(v["model"], v["region"], v[axis]): v for v in rows}
    require(index(tr, "metric") == index(j["table3_cells"], "metric") and len(tr) == 70 and
            index(reg, "signal") == index(j["regional_cells"], "signal") and len(reg) == 120,
            "Merged UV table disagrees with original archived accepted reports")
    recon = [v for v in tr if v["metric"] in METRICS]
    require(len(recon) == 55 and all(v["n_valid"] + sum(v["undefined_reasons"].values()) == N_TEST for v in recon),
            "UV 3F source cells incomplete")
    require(all(v["n_valid"] == v["statistics"]["n"] and v["n_valid"] + v["n_undefined"] == N_TEST
                and v["statistics"]["bootstrap_repeats"] == 2000 and v["statistics"]["bootstrap_seed"] == 20261026
                for v in reg), "UV 3B source cells incomplete")
    obj = dict(status="verified_native_UV_reference", source=descriptor(source), original_reports=pins,
               acceptance=acceptance, reconstruction=recon, agreement=reg,
               missing_from_accepted_reconstruction=["P/D/O/F MSE", "F SSIM"])
    emit_json(HERE / "reference" / "uv_reference.json", obj)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    pair = cmd.add_parser("pair")
    pair.add_argument("--root", type=Path, required=True)
    pair.add_argument("--architecture", required=True, choices=list(MODEL))
    pair.add_argument("--target", required=True, choices=TARGETS)
    cmd.add_parser("tables").add_argument("--root", type=Path, required=True)
    cmd.add_parser("pack-uv").add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.command == "pair":
        r = collect_pair(args.root, args.architecture, args.target)
        print(r["status"], r["architecture"], r["target"], r["cohort"])
    elif args.command == "tables":
        print(render_tables(args.root))
    else:
        pack_uv(args.source.resolve())


if __name__ == "__main__":
    main()
