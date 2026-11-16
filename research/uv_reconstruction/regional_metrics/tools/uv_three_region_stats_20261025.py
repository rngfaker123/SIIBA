#!/usr/bin/env python3
"""Read-only aggregation of three frozen models on the same TEST sample groups.

No inference, mask generation, resampling, or training occurs here. The declared
family contains 78 paired tests, including unavailable tests (Holm placeholder
p=1); it is never reduced after looking at results. Requires scipy, not pandas.
CLI: --manifest stats_manifest.json --output-dir NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import html
from itertools import combinations
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from scipy.stats import t as student_t

SCHEMA = "uv_three_region_statistics_v1"
INPUT_SCHEMA = "uv_three_region_statistics_input_v1"
MODEL_ORDER = ("unet_single", "unet_multi", "unetplusplus")
REGIONS = ("P", "D", "O")
METRICS = ("l1", "psnr", "ssim", "mean_sigma")
F_METRICS = ("l1", "psnr")
FAMILY_SIZE = 78


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_value(value, location):
    require(isinstance(value, Real) and not isinstance(value, bool), "Non-numeric metric: " + location)
    value = float(value)
    require(math.isfinite(value), "NaN/Infinity is forbidden; use null plus reason: " + location)
    return value


def validate_unit(unit):
    require(isinstance(unit, dict), "statistical_unit is required")
    require(unit.get("name") in ("sample_group", "subject"), "Unsupported statistical unit")
    verified = unit.get("independent_subject_identity_verified")
    require(type(verified) is bool, "Explicit independent_subject_identity_verified required")
    require(isinstance(unit.get("note"), str) and bool(unit["note"].strip()), "Statistical-unit caveat required")
    if unit["name"] == "subject":
        require(verified, "Cannot label unverified groups as independent subjects")
    return dict(unit)


def validate_models(records_by_model, unit):
    require(set(records_by_model) == set(MODEL_ORDER), "Exactly three frozen model IDs required")
    by_model, sample_order, subject_reference = {}, None, None
    for model in MODEL_ORDER:
        rows = records_by_model[model]
        require(isinstance(rows, list) and bool(rows), "Nonempty record list required: " + model)
        mapped, subjects = {}, {}
        for row in rows:
            require(isinstance(row, dict), "Record must be an object")
            sid = row.get("sample_id")
            require(isinstance(sid, str) and bool(sid.strip()) and sid not in mapped, "Missing/duplicate sample ID")
            require(row.get("model_id", model) == model, "Record model_id differs")
            regions = row.get("regions")
            require(isinstance(regions, dict), "Missing regions: " + sid)
            copied = {}
            for region in REGIONS + ("F",):
                entry = regions.get(region)
                require(isinstance(entry, dict), "Missing region " + region + " in " + sid)
                target = {}
                for metric in METRICS if region in REGIONS else F_METRICS:
                    where = "/".join((model, sid, region, metric))
                    require(metric in entry, "Missing metric: " + where)
                    value, reason = entry[metric], entry.get(metric + "_reason")
                    if value is None:
                        require(isinstance(reason, str) and bool(reason.strip()), "Null metric requires explicit reason: " + where)
                    else:
                        value = finite_value(value, where)
                        require(reason in (None, ""), "Finite metric cannot have an undefined reason: " + where)
                        require(metric != "mean_sigma" or value > 0, "Sigma must be positive")
                        require(metric not in ("l1", "psnr") or value >= 0, "Negative L1/PSNR")
                        require(metric != "ssim" or -1.0000001 <= value <= 1.0000001, "SSIM outside [-1,1]")
                    target[metric] = value
                    target[metric + "_reason"] = reason if value is None else None
                copied[region] = target
            subject = row.get("subject_id")
            require(subject is None or isinstance(subject, str), "Invalid subject ID")
            subjects[sid] = subject
            mapped[sid] = {"regions": copied, "subject_id": subject}
        if sample_order is None:
            sample_order = list(mapped)
            subject_reference = subjects
        else:
            require(set(mapped) == set(sample_order), "Models must cover exactly the same sample IDs")
            require(subjects == subject_reference, "Subject mapping differs across models")
        if unit["independent_subject_identity_verified"]:
            ids = list(subjects.values())
            require(all(isinstance(v, str) and bool(v.strip()) for v in ids), "Verified subjects require nonblank IDs")
            require(len(set(ids)) == len(ids), "Repeated subject IDs: per-capture t would pseudoreplicate; aggregate subjects explicitly first")
        by_model[model] = mapped
    return by_model, sample_order


def sample_summary(values, reasons):
    require(len(values) == len(reasons), "Summary values/reasons length differs")
    good = [v for v in values if v is not None]
    n = len(good)
    return {"mean": statistics.fmean(good) if n else None,
            "mean_reason": None if n else "no_finite_valid_samples",
            "sample_std_ddof1": statistics.stdev(good) if n >= 2 else None,
            "sample_std_reason": None if n >= 2 else "fewer_than_two_valid_samples",
            "n_valid": n, "n_undefined": len(values) - n,
            "undefined_reasons": dict(sorted(Counter(r for v, r in zip(values, reasons) if v is None).items()))}


def paired_test(left, right, *, sample_ids, left_reasons, right_reasons):
    require(len(left) == len(right) == len(sample_ids) == len(left_reasons) == len(right_reasons), "Unpaired vectors")
    valid, missing = [], Counter()
    for i, (a, b) in enumerate(zip(left, right)):
        if a is None or b is None:
            missing["left=" + (left_reasons[i] if a is None else "valid") +
                    ";right=" + (right_reasons[i] if b is None else "valid")] += 1
        else:
            valid.append((sample_ids[i], a, b, a - b))
    ds = [r[3] for r in valid]
    require(all(math.isfinite(v) for v in ds), "Nonfinite paired difference")
    n = len(ds)
    mean = statistics.fmean(ds) if n else None
    sd = statistics.stdev(ds) if n >= 2 else None
    result = {"n_pairs": n, "n_excluded": len(left) - n, "excluded_reasons": dict(sorted(missing.items())),
              "paired_sample_ids_sha256": digest([r[0] for r in valid]),
              "left_paired_mean": statistics.fmean(r[1] for r in valid) if n else None,
              "right_paired_mean": statistics.fmean(r[2] for r in valid) if n else None,
              "mean_difference_left_minus_right": mean, "difference_sample_std_ddof1": sd,
              "df": n - 1 if n >= 2 else None, "t_statistic": None,
              "p_raw_two_sided": None, "p_underflow_to_zero": False,
              "difference_ci95_unadjusted": None,
              "ci_note": "Individual Student-t interval; not a simultaneous/multiplicity-adjusted interval",
              "status": "unavailable", "reason": "fewer_than_two_complete_pairs"}
    if n >= 2 and sd == 0:
        result["reason"] = "constant_zero_difference" if mean == 0 else "constant_nonzero_difference"
        return result
    if n >= 2:
        se = sd / math.sqrt(n)
        statistic = mean / se
        require(math.isfinite(statistic), "Nonfinite t statistic outside explicit degenerate case")
        p = float(2.0 * student_t.sf(abs(statistic), n - 1))
        critical = float(student_t.ppf(.975, n - 1))
        low, high = mean - critical * se, mean + critical * se
        require(math.isfinite(p) and 0 <= p <= 1 and math.isfinite(low) and math.isfinite(high), "Invalid t distribution result")
        result.update(status="ok", reason=None, t_statistic=statistic, p_raw_two_sided=p,
                      p_underflow_to_zero=p == 0, difference_ci95_unadjusted=[low, high])
    return result


def holm_adjust(p_values, family_size=FAMILY_SIZE):
    """Step-down Holm; unavailable hypotheses retain conservative p=1 slots."""
    require(len(p_values) == family_size, "Declared Holm family must not shrink")
    for p in p_values:
        require(p is None or (math.isfinite(p) and 0 <= p <= 1), "Invalid p value")
    order = sorted(range(family_size), key=lambda i: (1.0 if p_values[i] is None else p_values[i], i))
    adjusted, running = [None] * family_size, 0.0
    for rank, index in enumerate(order):
        p = 1.0 if p_values[index] is None else p_values[index]
        running = max(running, min(1.0, (family_size - rank) * p))
        if p_values[index] is not None:
            adjusted[index] = running
    return adjusted


def test_plan():
    plan = []
    for model in MODEL_ORDER:
        for a, b in combinations(REGIONS, 2):
            for metric in METRICS:
                plan.append(("within_model_regions", model, a, model, b, metric))
    for a, b in combinations(MODEL_ORDER, 2):
        for region in REGIONS:
            for metric in METRICS:
                plan.append(("across_models_regions", a, region, b, region, metric))
        for metric in F_METRICS:
            plan.append(("across_models_F", a, "F", b, "F", metric))
    require(len(plan) == FAMILY_SIZE and len(set(plan)) == FAMILY_SIZE, "Invalid fixed test inventory")
    return plan


def summarize_models(records_by_model, *, statistical_unit):
    unit = validate_unit(statistical_unit)
    data, ids = validate_models(records_by_model, unit)

    def column(model, region, metric):
        entries = [data[model][sid]["regions"][region] for sid in ids]
        return [v[metric] for v in entries], [v[metric + "_reason"] for v in entries]

    summaries = {}
    for model in MODEL_ORDER:
        summaries[model] = {}
        for region in REGIONS + ("F",):
            summaries[model][region] = {metric: sample_summary(*column(model, region, metric))
                                        for metric in (METRICS if region in REGIONS else F_METRICS)}
    tests = []
    for kind, lm, lr, rm, rr, metric in test_plan():
        left, lreason = column(lm, lr, metric)
        right, rreason = column(rm, rr, metric)
        test = paired_test(left, right, sample_ids=ids, left_reasons=lreason, right_reasons=rreason)
        test.update(test_id="/".join((kind, lm, lr, rm, rr, metric)), kind=kind, metric=metric,
                    left={"model": lm, "region": lr}, right={"model": rm, "region": rr})
        tests.append(test)
    adjusted = holm_adjust([r["p_raw_two_sided"] for r in tests])
    for test, p in zip(tests, adjusted):
        test["p_holm_global78"] = p
        test["reject_holm_alpha_0_05"] = p <= .05 if p is not None else None
    return {"schema": SCHEMA, "status": "complete", "model_order": list(MODEL_ORDER),
            "sample_count": len(ids), "sample_order_sha256": digest(ids), "statistical_unit": unit,
            "aggregation": "Equal weight per sample-group metric; SD across samples, ddof=1; never pooled pixels",
            "pairing": "Same sample_id, complete pair for each endpoint; no pixel or seed pseudoreplicates",
            "inference_caveat": "Paired-t p values assume independent sample groups. Unverified identity does not establish biological independence; correlated repeated captures invalidate that assumption.",
            "no_equivalence_claim": "Nonsignificance is not equivalence; no expected ordering is imposed",
            "seed_repeats": "None; frozen deterministic inference is not repeated for a seed SD",
            "multiplicity": {"method": "Holm step-down", "family_size": FAMILY_SIZE, "alpha": .05,
                             "within_model_regions": 36, "across_models_regions": 36, "across_models_F": 6,
                             "unavailable_tests": "p=1 placeholder for correction; actual/adjusted p remain null",
                             "ci_adjusted": False},
            "undefined_policy": "Null with explicit reason; no zero filling or clipping; infinite PSNR excluded from finite summary with reason/count",
            "models": summaries, "tests": tests}


def display(value):
    return "NA" if value is None else format(value, ".7g")


def report_text(summary):
    lines = ["三模型 × 三类区域：固定 TEST 统计报告", "",
             "P=卟啉候选；D=UV spot候选；O=其余未分类区域；F=有效分析范围。",
             "样本组数：" + str(summary["sample_count"]),
             "统计单位：" + summary["statistical_unit"]["name"] + "；" + summary["statistical_unit"]["note"],
             "均值与SD：先每图区域统计，再样本等权；SD是跨样本SD（ddof=1），不是5seed SD。",
             "模型/区域均在同一sample_id上配对；像素不是独立样本。",
             "配对t依赖样本组独立性；身份未验证时，不宣称已证明个体独立。",
             "全部78项检验预先统一Holm校正；CI为单项未校正95% t区间。",
             "空区域/无SSIM中心/数学无限PSNR等以NA及原因计数；不填0、不任意封顶。",
             "差值始终左减右；L1低较好，PSNR/SSIM高较好，平均σ没有单独的优劣方向。",
             "不显著不等于等效；不预设P/D/O的排序。", "", "描述统计："]
    for model in MODEL_ORDER:
        for region, metrics in summary["models"][model].items():
            for metric, item in metrics.items():
                lines.append("%s / %s / %s: %s ± %s; N=%d; NA=%d; %s" %
                             (model, region, metric, display(item["mean"]), display(item["sample_std_ddof1"]),
                              item["n_valid"], item["n_undefined"], json.dumps(item["undefined_reasons"], ensure_ascii=False)))
    lines.extend(["", "配对检验（完整预注册78项）："])
    for item in summary["tests"]:
        ci = item["difference_ci95_unadjusted"]
        lines.append("%s: N=%d; Δ=%s; t=%s; df=%s; p=%s; Holm78=%s; CI=%s; reason=%s%s" %
                     (item["test_id"], item["n_pairs"], display(item["mean_difference_left_minus_right"]),
                      display(item["t_statistic"]), display(item["df"]), display(item["p_raw_two_sided"]),
                      display(item["p_holm_global78"]), "NA" if ci is None else "[%s, %s]" % tuple(map(display, ci)),
                      item["reason"], "; p值浮点下溢，存储0不等于数学概率0" if item["p_underflow_to_zero"] else ""))
    return "\n".join(lines) + "\n"


def report_html(summary):
    text = report_text(summary)
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>三类区域统计报告</title><style>body{margin:30px;font:16px/1.6 system-ui,sans-serif;color:#17202a}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}</style>'
            '<h1>三模型 × 三类区域</h1><pre>' + html.escape(text) + '</pre></html>\n')


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf8"))
    require(manifest.get("schema") == INPUT_SCHEMA, "Unsupported statistics manifest")
    require(manifest.get("model_order") == list(MODEL_ORDER), "Frozen model order differs")
    expected_count = manifest.get("expected_sample_count")
    expected_order = manifest.get("expected_sample_order_sha256")
    require(type(expected_count) is int and expected_count > 0, "Frozen expected_sample_count required")
    require(isinstance(expected_order, str) and len(expected_order) == 64 and
            all(c in "0123456789abcdef" for c in expected_order), "Frozen expected_sample_order_sha256 required")
    entries = manifest.get("models")
    require(isinstance(entries, list) and len(entries) == 3, "Exactly three model descriptors required")
    rows, sources = {}, []
    for item in entries:
        model = item.get("model_id")
        require(model in MODEL_ORDER and model not in rows, "Duplicate/unknown model descriptor")
        descriptor = item.get("records", {})
        raw_path = descriptor.get("path")
        require(isinstance(raw_path, str) and raw_path, "Missing records path")
        record_path = Path(raw_path)
        record_path = record_path.resolve() if record_path.is_absolute() else (path.parent / record_path).resolve()
        require(file_sha(record_path) == descriptor.get("sha256"), "Record file hash differs: " + model)
        checkpoint = item.get("checkpoint_sha256")
        require(isinstance(checkpoint, str) and len(checkpoint) == 64 and all(c in "0123456789abcdef" for c in checkpoint), "Checkpoint SHA256 required")
        if record_path.suffix.lower() == ".jsonl":
            lines = record_path.read_text(encoding="utf8").splitlines()
            require(bool(lines) and all(v.strip() for v in lines), "Blank/empty JSONL record")
            rows[model] = [json.loads(line) for line in lines]
        else:
            rows[model] = json.loads(record_path.read_text(encoding="utf8"))
        require(isinstance(rows[model], list) and len(rows[model]) == expected_count,
                "Incomplete/extra TEST records: " + model)
        require(all(isinstance(r, dict) and isinstance(r.get("sample_id"), str) for r in rows[model]),
                "Invalid TEST sample record: " + model)
        require(digest([r["sample_id"] for r in rows[model]]) == expected_order,
                "Frozen TEST sample inventory/order differs: " + model)
        sources.append(dict(item, records={"path": str(record_path), "sha256": file_sha(record_path)}))
    return rows, validate_unit(manifest.get("statistical_unit")), {"manifest": str(path),
        "manifest_sha256": file_sha(path), "models": sources, "statistics_source_sha256": file_sha(__file__),
        "expected_sample_count": expected_count, "expected_sample_order_sha256": expected_order}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), "Output exists; never overwrite statistics")
    rows, unit, provenance = load_manifest(args.manifest)
    summary = summarize_models(rows, statistical_unit=unit)
    summary["provenance"] = provenance
    outputs = {"summary.json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
               "report.txt": report_text(summary), "report.html": report_html(summary)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        with (args.output_dir / name).open("x", encoding="utf8", newline="\n") as stream:
            stream.write(content)
    print(json.dumps({"status": "complete", "samples": summary["sample_count"], "tests": len(summary["tests"]),
                      "output": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
