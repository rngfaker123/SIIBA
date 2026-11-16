#!/usr/bin/env python3
"""Read-only verification of completed native MI batch; new TXT/HTML report.

Pure standard library. Independently recomputes all 18 mean/sample-SD cells
from exactly 54 sealed results, then checks them against the runner summary.
The source chain is verified through sealed case/feature records and hashes;
large original image files are NOT re-read or reprocessed for this report.

CLI: python tools/mi_atlas_report_20261025.py --batch-dir COMPLETE_BATCH
         --output-dir NEW_REPORT_FOLDER
An existing, byte-identical report is verified as a no-op; nothing is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import tempfile


VERSION = "mi_atlas_report_20261025_v1"
INITIAL_COUNT = 1000
SEEDS = (20261025, 20261026, 20261027)
SIZES = (100, 500, 1000)
METRICS = {"full_rgb_plv": ("F", "plv"), "full_rgb_plh": ("F", "plh"),
           "full_rgb_uv": ("F", "uv"), "P_rgb_uv": ("P", "uv"),
           "D_rgb_uv": ("D", "uv"), "O_rgb_uv": ("O", "uv")}
LABELS = {"full_rgb_plv": "有效分析域 F：RGB ↔ PLV", "full_rgb_plh": "有效分析域 F：RGB ↔ PLH",
          "full_rgb_uv": "有效分析域 F：RGB ↔ UV", "P_rgb_uv": "卟啉区域 P：RGB ↔ UV",
          "D_rgb_uv": "UV spot 区域 D：RGB ↔ UV", "O_rgb_uv": "其他区域 O：RGB ↔ UV"}
SOURCES = ("rgb", "uv", "plv", "plh")
DOMAINS = ("F", "P", "D", "O")
NOTES = (
    "每格为 3 个抽样随机种子的均值 ± 样本标准差（ddof=1），单位 bits。标准差只描述抽样重复波动，不是人群置信区间。",
    "初始队列按提供的 subject_id 去重；未在此独立核验跨会话真实身份，不能将其称作 1,000 名独立患者。",
    "每个合格样本在每个区域分别抽取 1,000 个不重复原生像素坐标，100/500 点为其前缀；三个 F 域指标共享同一组坐标。",
    "全部 6 指标、3 点数、3 随机种子采用同一合格子集；技术排除与点数不足分别记录，不补人、不重复填点、不降低配额。",
    "以所有合格样本的配对原生三通道像素合并计算 KSG-1（k=5），不是逐图 MI 再平均；保留负估计值，不截断为零。",
    "P/D/O 是冻结规则得到的互斥区域，合并为有效分析域 F。Other（O）只表示其余区域，不是临床确认的正常皮肤。",
    "区域条件下的像素互信息不等于整张 RGB 的全部信息，也不能单独证明物理不可恢复性；像素坐标对应继承原数据，本报告没有重新配准。",
    "报告核验完整 54 结果和冻结记录的来源绑定，并独立重算 18 格统计量；报告阶段不重读原始大图、不重算掩膜或 MI。",
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(value):
    require(value.get("record_sha256") == digest({k: v for k, v in value.items() if k != "record_sha256"}), "Sealed record digest mismatch")
    return value


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def valid_descriptor(value):
    return isinstance(value, dict) and isinstance(value.get("path"), str) and bool(value["path"]) and valid_sha(value.get("sha256"))


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite_number(a) and finite_number(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def verify(batch_dir):
    """No writes and no detector/model imports. Every expected result is checked."""
    root = Path(batch_dir).resolve()
    fixed = {name: root / name for name in ("receipt.json", "batch_contract.json", "eligibility.json", "mi_input.json")}
    fixed.update({"mi_run/" + name: root / "mi_run" / name for name in ("summary.json", "protocol.json", "preparation.json")})
    source_hashes = {name: sha(path) for name, path in fixed.items()}
    receipt = checked(read_json(fixed["receipt.json"]))
    contract = read_json(fixed["batch_contract.json"])
    eligibility = checked(read_json(fixed["eligibility.json"]))
    manifest = read_json(fixed["mi_input.json"])
    summary = checked(read_json(fixed["mi_run/summary.json"]))
    protocol = read_json(fixed["mi_run/protocol.json"])
    preparation = checked(read_json(fixed["mi_run/preparation.json"]))
    contract_hash, protocol_hash = digest(contract), digest(protocol)
    require(receipt.get("status") == "complete" and receipt.get("result_count") == 54, "Batch is not complete with 54 results")
    require(receipt.get("contract_sha256") == contract_hash == eligibility.get("contract_sha256"), "Batch contract binding differs")
    require(receipt.get("initial_count") == INITIAL_COUNT == contract.get("original_sample_count") == eligibility.get("initial_count"), "Initial frozen cohort count differs")
    require(receipt.get("mi_input_sha256") == source_hashes["mi_input.json"] and receipt.get("eligibility_sha256") == source_hashes["eligibility.json"], "Batch manifest/eligibility hashes differ")
    desc = receipt.get("summary", {})
    require(valid_descriptor(desc) and Path(desc["path"]).resolve() == fixed["mi_run/summary.json"] and desc["sha256"] == source_hashes["mi_run/summary.json"], "Batch summary descriptor differs")
    require(receipt.get("replacement_count") == 0 == eligibility.get("replacement_count") and eligibility.get("same_subset_all_sizes_seeds_metrics") is True, "Unexpected replacement or differing subsets")
    require(all(receipt.get(key) is True for key in ("company_unchanged", "masks_frozen", "cpu_only")), "Incomplete frozen-source completion receipt")
    rows = manifest.get("rows", [])
    n = len(rows)
    require(n > 0 and n == manifest.get("expected_sample_count") == receipt.get("admitted_count") == eligibility.get("admitted_count") == protocol.get("expected_sample_count"), "Common eligible cohort count differs")
    require(manifest.get("schema") == "mi_atlas_six_metrics_input_v1" and manifest.get("status") == "frozen", "MI input is not frozen")
    ids, subjects = [r.get("sample_id") for r in rows], [r.get("subject_id") for r in rows]
    require(len(set(ids)) == n and len(set(subjects)) == n and all(isinstance(x, str) and x for x in ids + subjects), "Duplicate/missing supplied identities")
    require(ids == eligibility.get("admitted_sample_ids") == receipt.get("eligible_sample_order") == protocol.get("sample_order") == preparation.get("sample_order") and subjects == protocol.get("subject_order"), "Common eligible sample order differs")
    require(protocol.get("input_manifest_sha256") == source_hashes["mi_input.json"] and protocol.get("input_contract_sha256") == digest(manifest), "Protocol input binding differs")
    require(protocol.get("runner_sha256") == contract.get("stage_code_sha256", {}).get("mi"), "Runner source binding differs")
    require(protocol.get("seeds") == list(SEEDS) and protocol.get("sizes") == list(SIZES) and protocol.get("metrics") == list(METRICS) and protocol.get("k") == 5 and protocol.get("noise_seed") == 20260817, "Frozen MI protocol differs")
    require(protocol.get("negative_estimates_clipped") is False and protocol.get("spatial_points_are_independent_people") is False, "Numerical/identity interpretation differs")
    provenance = manifest.get("frozen_provenance", {})
    require(provenance == protocol.get("frozen_provenance"), "Protocol provenance differs")
    expected_pipeline = digest({"stage_code": contract.get("stage_code_sha256"), "mask_sources": contract.get("frozen_source_sha256")})
    require(provenance.get("company_module_sha256") == contract.get("vendor_sha256") and provenance.get("mask_pipeline_sha256") == expected_pipeline and provenance.get("parity_report_sha256") == contract.get("parity_report_sha256") and provenance.get("initial_cohort_sha256") == contract.get("cohort_sha256") and provenance.get("eligibility_record_sha256") == eligibility["record_sha256"], "Mask/cohort/parity source binding differs")
    require(preparation.get("status") == "complete" and preparation.get("protocol_sha256") == protocol_hash and preparation.get("expected_sample_count") == n, "Preparation is incomplete/unbound")
    require(all(preparation.get(k) is True for k in ("all_samples_all_domains_at_least_1000", "no_population_replacement", "no_duplicate_padding")), "Preparation quota assurance missing")
    require(summary.get("status") == "complete" and summary.get("actual_result_count") == 54 == summary.get("expected_result_count") and summary.get("protocol_sha256") == protocol_hash and summary.get("preparation_sha256") == preparation["record_sha256"] and summary.get("unit") == "bits", "Summary completion/protocol/preparation differs")

    initial_ids = contract.get("sample_order", [])
    require(len(initial_ids) == INITIAL_COUNT and len(set(initial_ids)) == INITIAL_COUNT and set(eligibility.get("case_record_sha256", {})) == set(initial_ids), "Initial cohort coverage incomplete")
    admitted, exclusions = [], []
    case_hashes = {}
    for index, sample_id in enumerate(initial_ids):
        case_dir = root / "cases" / ("case_%06d" % index)
        case = checked(read_json(case_dir / "case_receipt.json"))
        require(case.get("record_sha256") == eligibility["case_record_sha256"][sample_id] and case.get("index") == index and case.get("sample_id") == sample_id and case.get("contract_sha256") == contract_hash, "Initial case binding differs")
        raw_row = dict(sample_id=sample_id, subject_id=case.get("subject_id"), native_shape=case.get("native_shape"),
                       initial_cohort_index=index, **case.get("sources", {}))
        require(set(case.get("sources", {})) == set(SOURCES) and all(valid_descriptor(case["sources"][key]) for key in SOURCES) and case.get("source_row_sha256") == digest(raw_row), "Case native source binding differs")
        expected_stages = {"preview", "geometry"} if case.get("status") == "technical_excluded_no_unique_face" else {"preview", "geometry", "mask"}
        require(set(case.get("stage_receipt_sha256", {})) == expected_stages, "Case stage receipt coverage differs")
        for stage, expected_sha in case["stage_receipt_sha256"].items():
            require(stage in ("preview", "geometry", "mask") and sha(case_dir / (stage + "_receipt.json")) == expected_sha, "Case stage receipt changed")
            stage_receipt = checked(read_json(case_dir / (stage + "_receipt.json")))
            binding = stage_receipt.get("binding", {})
            require(binding.get("contract_sha256") == contract_hash and binding.get("index") == index and binding.get("sample_id") == sample_id and binding.get("row_sha256") == digest(raw_row) and binding.get("stage") == stage and binding.get("code_sha256") == contract.get("stage_code_sha256", {}).get(stage) and binding.get("runtime") == contract.get("runtimes", {}).get(stage), "Stage code/runtime/source binding differs")
            if stage_receipt.get("status") == "technical_excluded_no_unique_face":
                require(stage == "geometry" and case.get("status") == "technical_excluded_no_unique_face", "Technical stage status differs")
                for prefix in ("log", "stderr"):
                    require(sha(stage_receipt[prefix + "_path"]) == stage_receipt.get(prefix + "_sha256"), "Technical geometry log changed")
                stderr_lines = Path(stage_receipt["stderr_path"]).read_text(encoding="utf8", errors="replace").splitlines()
                require(next((x for x in reversed(stderr_lines) if x.strip()), "") == "ValueError: Expected exactly one face; no mask fabricated", "Technical exclusion reason differs")
            else:
                require(stage_receipt.get("status") == "complete" and stage_receipt.get("summary_sha256") == sha(case_dir / stage / "summary.json"), "Stage summary binding differs")
        if case.get("status") == "technical_excluded_no_unique_face":
            require(case.get("masks") is None and case.get("counts") is None and case.get("eligible") is False and case.get("exclusion_kind") == "technical" and case.get("exclusion_reason") == "no_unique_face_geometry", "Technical exclusion fabricated masks/counts")
        else:
            require(case.get("status") == "complete" and case.get("mask_config_sha256") == provenance.get("mask_config_sha256"), "Case mask config/status differs")
            counts = case.get("counts", {})
            require(set(counts) == set(DOMAINS) and all(type(v) is int and v >= 0 for v in counts.values()) and sum(counts[k] for k in ("P", "D", "O")) == counts["F"], "Case partition counts differ")
            shortage = {key: counts[key] for key in DOMAINS if counts[key] < 1000}
            require(case.get("eligible") is (not shortage) and case.get("insufficient_domains") == shortage, "Case quota decision differs")
            if shortage:
                require(case.get("exclusion_kind") == "insufficient_pixels" and case.get("exclusion_reason") == "insufficient_1000_native_pixels", "Quota exclusion label differs")
        if case["eligible"]:
            row = rows[len(admitted)]
            require(row == dict(raw_row, **case["masks"]), "Admitted raw/mask source descriptors differ")
            admitted.append(sample_id)
        else:
            exclusions.append(dict(index=index, sample_id=sample_id, subject_id=case["subject_id"], counts=case["counts"],
                reason=case["exclusion_reason"], kind=case["exclusion_kind"], insufficient_domains=case["insufficient_domains"]))
        case_hashes[sample_id] = sha(case_dir / "case_receipt.json")
    technical = sum(x["kind"] == "technical" for x in exclusions)
    shortage_count = sum(x["kind"] == "insufficient_pixels" for x in exclusions)
    require(admitted == ids and exclusions == eligibility.get("exclusions") and len(exclusions) == receipt.get("excluded_count") == eligibility.get("excluded_count") and n + len(exclusions) == INITIAL_COUNT, "Exclusion coverage differs")
    require(technical == eligibility.get("technical_excluded_count") and shortage_count == eligibility.get("insufficient_pixels_excluded_count"), "Technical and pixel exclusions conflated")

    feature_records = preparation.get("features", [])
    require(len(feature_records) == n, "Sampled feature coverage incomplete")
    feature_files, metadata_files = set(), set()
    for index, (row, item) in enumerate(zip(rows, feature_records)):
        feature_name = "features/sample_%06d.npz" % index
        metadata_name = "features/sample_%06d.json" % index
        feature, metadata_path = root / "mi_run" / feature_name, root / "mi_run" / metadata_name
        require(item.get("index") == index and item.get("sample_id") == row["sample_id"] and item.get("subject_id") == row["subject_id"] and item.get("feature_path") == feature_name, "Feature identity/order differs")
        require(sha(feature) == item.get("feature_sha256") and sha(metadata_path) == item.get("metadata_sha256"), "Sampled feature artifact changed")
        metadata = checked(read_json(metadata_path))
        require(metadata.get("protocol_sha256") == protocol_hash and metadata.get("source_row_sha256") == digest(row) and metadata.get("source_sha256") == {key: row[key]["sha256"] for key in SOURCES + DOMAINS} and metadata.get("feature_sha256") == item["feature_sha256"], "Sampled feature source binding differs")
        require(metadata.get("source_hashes_verified") is True and metadata.get("partition_verified") is True and metadata.get("sampled_points_per_domain_per_seed") == 1000 and all(metadata.get("mask_pixels", {}).get(key, 0) >= 1000 for key in DOMAINS), "Feature quota/partition verification missing")
        feature_files.add(feature.name)
        metadata_files.add(metadata_path.name)
    require({p.name for p in (root / "mi_run/features").glob("sample_*.npz")} == feature_files and {p.name for p in (root / "mi_run/features").glob("sample_*.json")} == metadata_files, "Unexpected sampled feature artifacts")

    expected_names, recomputed, result_hashes = set(), [], {}
    summary_rows = summary.get("summaries", [])
    require(len(summary_rows) == 18, "Summary does not contain exactly 18 cells")
    indexed = {(r.get("metric"), r.get("points_per_sample")): r for r in summary_rows}
    require(len(indexed) == 18 and set(indexed) == {(m, s) for m in METRICS for s in SIZES}, "Summary cell inventory differs")
    for size in SIZES:
        for metric, (domain, target) in METRICS.items():
            values = []
            for seed in SEEDS:
                name = "%s_n%d_seed%d.json" % (metric, size, seed)
                expected_names.add(name)
                path = root / "mi_run/results" / name
                result = checked(read_json(path))
                expected = dict(status="complete", protocol_sha256=protocol_hash, preparation_sha256=preparation["record_sha256"],
                    metric=metric, domain=domain, target=target, points_per_sample=size, sampling_seed=seed,
                    sample_count=n, observations=n * size, k=5, dx=3, dy=3)
                require(all(result.get(k) == v for k, v in expected.items()), "Individual result binding differs: " + name)
                numeric = result.get("result", {})
                require(numeric.get("n") == n * size and numeric.get("k") == 5 and numeric.get("dx") == 3 and numeric.get("dy") == 3 and numeric.get("estimator") == "KSG_1" and numeric.get("negative_estimates_clipped") is False and finite_number(numeric.get("mi_bits")), "Invalid MI numerical result: " + name)
                require(valid_sha(result.get("input_x_sha256")) and valid_sha(result.get("input_y_sha256")), "Missing paired feature hashes")
                require(summary.get("result_record_sha256", {}).get(name) == result["record_sha256"], "Summary result inventory hash differs")
                values.append(numeric["mi_bits"])
                result_hashes[name] = sha(path)
            mean, sd = statistics.mean(values), statistics.stdev(values)
            reported = indexed[metric, size]
            require(reported.get("mi_bits_by_seed") == values and reported.get("sampling_seeds") == list(SEEDS) and reported.get("ddof") == 1 and reported.get("repetitions") == 3 and reported.get("sample_count") == n and reported.get("observations_per_repeat") == n * size, "Summary repeat metadata differs")
            require(close(mean, reported.get("mean_bits")) and close(sd, reported.get("sample_sd_bits")), "Independent mean/sample-SD recomputation differs")
            recomputed.append(dict(metric=metric, points_per_sample=size, mean_bits=mean, sample_sd_bits=sd, mi_bits_by_seed=values))
    require(set(summary.get("result_record_sha256", {})) == expected_names and {p.name for p in (root / "mi_run/results").glob("*.json")} == expected_names, "Result inventory not exactly the expected 54")
    return dict(batch_dir=str(root), initial_count=INITIAL_COUNT, eligible_count=n, excluded_count=len(exclusions),
        technical_excluded_count=technical, insufficient_pixels_excluded_count=shortage_count,
        source_hashes=source_hashes, case_receipt_file_sha256=case_hashes, result_file_sha256=result_hashes,
        contract_sha256=contract_hash, protocol_sha256=protocol_hash,
        frozen_provenance=provenance, estimator_sha256=protocol.get("estimator_sha256"),
        runner_sha256=protocol.get("runner_sha256"), model_sha256=contract.get("model_sha256"),
        controller_sha256=contract.get("controller_sha256"), software=protocol.get("software"),
        summaries=recomputed, exclusions=exclusions)


def render(data):
    cells = {(r["metric"], r["points_per_sample"]): r for r in data["summaries"]}
    def cell(metric, size):
        value = cells[metric, size]
        return "%.6f ± %.6f" % (value["mean_bits"], value["sample_sd_bits"])
    title = "皮肤影像信息边界：六项互信息实验汇总"
    population = "初始固定队列：%d 组；共同合格子集：%d 组；不补充替代样本。" % (data["initial_count"], data["eligible_count"])
    exclusions = "技术排除（无唯一可用人脸）：%d 组；像素不足排除（任一 P/D/O/F <1,000）：%d 组。" % (data["technical_excluded_count"], data["insufficient_pixels_excluded_count"])
    heading = "指标 | 每样本/区域 100 点 | 每样本/区域 500 点 | 每样本/区域 1,000 点"
    lines = [title, "", "验收：54/54 结果完整，18/18 格已用 statistics.mean/stdev 独立复算通过。", population,
             exclusions, "", "均值 ± 样本标准差（3 个抽样种子，bits）", heading]
    lines.extend(" | ".join([LABELS[m]] + [cell(m, s) for s in SIZES]) for m in METRICS)
    lines.extend(["", "统计口径与限制"] + ["%d. %s" % (i + 1, note) for i, note in enumerate(NOTES)])
    hashes = {"初始队列文件": data["frozen_provenance"]["initial_cohort_sha256"],
        "11 例完整链路校验报告": data["frozen_provenance"]["parity_report_sha256"],
        "公司卟啉模块": data["frozen_provenance"]["company_module_sha256"],
        "冻结 mask 流程": data["frozen_provenance"]["mask_pipeline_sha256"],
        "冻结 mask 配置": data["frozen_provenance"]["mask_config_sha256"],
        "KSG 估计器源码": data["estimator_sha256"], "MI runner 源码": data["runner_sha256"],
        "MI 输入清单": data["source_hashes"]["mi_input.json"], "批次完成凭证": data["source_hashes"]["receipt.json"]}
    lines.extend(["", "冻结来源与 SHA256", "原批次：" + data["batch_dir"], "抽样种子：" + ", ".join(map(str, SEEDS))])
    lines.extend(name + ": " + str(value) for name, value in hashes.items())
    text = ("\n".join(lines) + "\n").encode("utf8")
    esc = html.escape
    table = "".join("<tr><th scope='row'>" + esc(LABELS[m]) + "</th>" + "".join("<td>" + esc(cell(m, s)) + "</td>" for s in SIZES) + "</tr>" for m in METRICS)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + esc(title) + """</title><style>
body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#172334;background:#f3f5f7;margin:0}main{max-width:1180px;margin:30px auto;padding:30px;background:white;border-radius:12px}h1{font-size:27px;margin:0 0 12px}h2{font-size:20px;margin-top:28px}.status{color:#185d3b}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #dce1e6;padding:13px 12px;text-align:left;white-space:nowrap}thead{background:#e9eef4}td{font-family:ui-monospace,Consolas,monospace}caption{text-align:left;margin:12px 0}li{margin:7px 0}dl{font-size:13px}dt{font-weight:650;margin-top:10px}dd{margin:0;overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace}.source{overflow-wrap:anywhere;font-size:13px}@media print{body{background:white}main{margin:0;padding:0}table{font-size:12px}}
</style><main><h1>""" + esc(title) + "</h1><p class='status'>54/54 结果完整；18/18 格独立复算通过。</p><p>" + esc(population) + "<br>" + esc(exclusions) + "</p><div class='tablewrap'><table><caption>均值 ± 样本标准差（3 个抽样种子，bits）</caption><thead><tr><th>指标</th><th>每样本/区域 100 点</th><th>每样本/区域 500 点</th><th>每样本/区域 1,000 点</th></tr></thead><tbody>" + table + "</tbody></table></div><h2>统计口径与限制</h2><ol>" + "".join("<li>" + esc(note) + "</li>" for note in NOTES) + "</ol><h2>冻结来源与 SHA256</h2><p class='source'>原批次：" + esc(data["batch_dir"]) + "<br>抽样种子：" + ", ".join(map(str, SEEDS)) + "</p><dl>" + "".join("<dt>" + esc(k) + "</dt><dd>" + esc(str(v)) + "</dd>" for k, v in hashes.items()) + "</dl></main></html>\n"
    return {"report.txt": text, "report.html": page.encode("utf8")}


def run(batch_dir, output_dir):
    data = verify(batch_dir)
    root, output = Path(batch_dir).resolve(), Path(output_dir).resolve()
    require(output != root and not any(p == output or p in output.parents for p in (root / "cases", root / "mi_run")), "Report cannot replace batch or scientific artifact directories")
    payloads = render(data)
    receipt = dict(schema=VERSION, status="complete", batch_dir=str(root), report_source_sha256=sha(__file__),
        verified_result_count=54, independently_recomputed_cell_count=18, ddof=1, repetitions=3,
        verification=data, artifact_sha256={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()})
    receipt["record_sha256"] = digest(receipt)
    payloads["receipt.json"] = encoded(receipt)
    if output.exists():
        require(output.is_dir() and {p.name for p in output.iterdir()} == set(payloads), "Existing report folder has unexpected files; nothing overwritten")
        require(all((output / name).read_bytes() == payload for name, payload in payloads.items()), "Existing report differs from frozen inputs/code; nothing overwritten")
        return dict(status="verified_noop", report_dir=str(output), report_txt=str(output / "report.txt"), report_html=str(output / "report.html"))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=str(output.parent)))
    for name, payload in payloads.items():
        (staging / name).write_bytes(payload)
    # No overwrite; a concurrent publication is verified byte-for-byte.
    try:
        staging.rename(output)
    except FileExistsError:
        require(output.is_dir() and {p.name for p in output.iterdir()} == set(payloads) and all((output / name).read_bytes() == payload for name, payload in payloads.items()), "Concurrent report differs; partial folder retained")
        return dict(status="verified_noop", report_dir=str(output), retained_identical_partial=str(staging))
    return dict(status="created", report_dir=str(output), report_txt=str(output / "report.txt"), report_html=str(output / "report.html"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.batch_dir, args.output_dir), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
