"""Finalize the two new TEST models without changing the original 78-test family.

No inference, geometry generation, exclusions from missing files, model fitting,
or calibration is performed. Every statistic uses its own explicit valid N.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import io
import math
from pathlib import Path
import sys

import numpy as np

import common as c

MODELS = ("swin", "ldm")
COUNT = 21899
REGIONS = ("forehead", "image_left_cheek", "image_right_cheek", "nose", "chin", "F_reference")
SIGNALS = ("R", "G", "B", "RGB_mean_secondary")
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261026


def owner(model, index):
    c.need(model in MODELS and type(index) is int and index >= 0, "Invalid record owner identity")
    return (0 if model == "swin" else 4) + index % 4


def independent_summary(values):
    n = len(values)
    mean = math.fsum(values) / n if n else None
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else None
    return mean, sd


def close(expected, actual, label):
    if expected is None:
        c.need(actual is None, "Independent undefined mismatch: " + label)
        return 0.0
    c.need(actual is not None and math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-11),
           "Independent statistic mismatch: " + label)
    return abs(expected - actual)


def _finite(value, reason, label):
    if value is None:
        c.need(isinstance(reason, str) and bool(reason.strip()), "Null without explicit reason: " + label)
    else:
        c.need(type(value) in (int, float) and math.isfinite(value) and reason in (None, ""),
               "Invalid finite statistic: " + label)


def validate_table3(record, stats):
    fields = record["combined"]["regions"]
    c.need(set(fields) == set(stats.REGIONS + ("F",)), "Table3 region inventory differs")
    for region in stats.REGIONS + ("F",):
        for metric in stats.METRICS if region in stats.REGIONS else stats.F_METRICS:
            value, reason = fields[region][metric], fields[region].get(metric + "_reason")
            _finite(value, reason, region + "/" + metric)
            if value is not None:
                c.need(metric != "mean_sigma" or value > 0, "Nonpositive sigma")
                c.need(metric not in ("l1", "psnr") or value >= 0, "Negative L1/PSNR")
                c.need(metric != "ssim" or -1.0000001 <= value <= 1.0000001, "Invalid SSIM")
            if record["status"] == "unavailable_geometry":
                c.need(value is None, "Technical record contains a numerical Table3 value")


def verify_inventory(root, manifest, science, worker, expected_count=COUNT):
    """Require complete fixed ownership, exact filenames, seals and all source bindings."""
    root = Path(root).resolve()
    rows = science["ctx"]["rows"]
    c.need(manifest["sample_count"] == len(rows) == expected_count and tuple(manifest["models"]) == MODELS,
           "Full TEST cohort/model inventory differs")
    ids = [r["sample_id"] for r in rows]
    c.need(len(set(ids)) == expected_count, "Duplicate catalog sample IDs")
    gpu_inventory = c.validate_inventory(manifest)
    c.validate_ownership(manifest["ownership"])
    manifest_desc = c.desc(root / "manifest.json")
    completion_dir = root / "completion"
    c.need({p.name for p in completion_dir.glob("*.json")} == {"shard_%d.json" % g for g in range(8)},
           "Exactly eight shard completions are required before collecting")
    shard_descriptors, declared = [], {}
    for gpu in range(8):
        model = MODELS[gpu // 4]
        indices = list(range(gpu % 4, expected_count, 4))
        path = completion_dir / ("shard_%d.json" % gpu)
        shard = c.checked(c.read(path))
        c.need(shard["schema"] == "secondary_test_shard_completion_v1" and shard["status"] == "complete"
               and shard["gpu"] == gpu and shard["model"] == model and shard["indices"] == indices
               and shard["manifest"] == manifest_desc and len(shard["records"]) == len(indices),
               "Shard identity/ownership/completeness differs")
        for index, descriptor in zip(indices, shard["records"]):
            path = root / "records" / model / ("%06d.json" % index)
            c.need(Path(descriptor["path"]).resolve() == path and (model, index) not in declared,
                   "Duplicate or wrongly owned shard record")
            c.need(c.bound(descriptor) == path, "Shard record descriptor differs")
            declared[model, index] = descriptor
        shard_descriptors.append(c.desc(completion_dir / ("shard_%d.json" % gpu)))
    records, inventory, runtimes = {}, [], {}
    expected_names = {"%06d.json" % i for i in range(expected_count)}
    for model in MODELS:
        folder = root / "records" / model
        c.need({p.name for p in folder.glob("*.json")} == expected_names, "Missing or extra model records")
        records[model] = []
        for index, row in enumerate(rows):
            value = c.checked(worker.read_record(root, manifest, science, model, index))
            path = folder / ("%06d.json" % index)
            c.need(c.desc(path) == declared[model, index], "Record changed after shard verification")
            c.need(value["schema"] == "secondary_test_sample_v1" and value["index"] == index
                   and value["model_id"] == model and value["sample_id"] == row["sample_id"]
                   and value["subject_id"] == row["subject_id"] and value["source_row_sha256"] == c.digest(row)
                   and value["manifest_sha256"] == manifest_desc["sha256"], "Record identity/manifest differs")
            c.need(value["status"] in ("complete", "unavailable_geometry"), "Unexpected record status")
            execution = value["execution"]
            gpu = owner(model, index)
            gpu_item = gpu_inventory[gpu]
            c.need(execution["global_gpu_id"] == gpu
                   and execution["physical_gpu_id"] == gpu_item["local_index"]
                   and execution["host"] == gpu_item["host"] and execution["uuid"] == gpu_item["uuid"]
                   and execution["forward_count"] == int(value["status"] == "complete"), "Record ownership/forward count differs")
            if value["status"] == "complete":
                c.need(execution["device"] == "cuda:0", "Valid result did not use the pinned CUDA worker")
            kind = "gpu_new" if value["status"] == "complete" else "technical_derived"
            for domain in ("combined", "regional"):
                c.need(value["provenance"][domain] == {"kind": kind, "record": None}, "Historical/reused result in new cohort")
            for key in ("mask_ref", "mask_receipt"):
                c.bound(value[key])
            c.bound(execution["runtime"])
            runtime = c.read(execution["runtime"]["path"])
            c.need(all(runtime[key] == execution[key] for key in ("global_gpu_id", "physical_gpu_id", "host", "uuid")),
                   "Record runtime ownership differs")
            runtimes[execution["runtime"]["path"]] = execution["runtime"]
            records[model].append(value)
            inventory.append(dict(model=model, index=index, sample_id=row["sample_id"], source=declared[model, index],
                                  record_sha256=value["record_sha256"], checkpoint_sha256=value["checkpoint_sha256"],
                                  source_row_sha256=value["source_row_sha256"], mask_ref=value["mask_ref"],
                                  mask_receipt=value["mask_receipt"], runtime=execution["runtime"]))
    c.need(len(inventory) == expected_count * 2, "Incomplete final record inventory")
    for index in range(expected_count):
        a, b = records["swin"][index], records["ldm"][index]
        c.need(all(a[key] == b[key] for key in ("status", "mask_ref", "mask_receipt")),
               "Model cohorts use different mask cases")
        left, right = a["regional"], b["regional"]
        c.need(left["roi_audit"] == right["roi_audit"], "Model cohorts use different anatomical geometry")
        if a["status"] == "complete":
            for region in REGIONS:
                c.need(all(left["regions"][region][key] == right["regions"][region][key]
                           for key in ("truth", "effective_area", "reason")), "Model cohorts use different regional truth")
    return records, inventory, shard_descriptors, list(runtimes.values())


def summarize_table3(records, stats):
    cells, audits = [], []
    c.need(tuple(stats.REGIONS) == ("P", "D", "O") and tuple(stats.METRICS) == ("l1", "psnr", "ssim", "mean_sigma")
           and tuple(stats.F_METRICS) == ("l1", "psnr") and stats.FAMILY_SIZE == 78, "Frozen Table3 definitions differ")
    for model in MODELS:
        for record in records[model]:
            validate_table3(record, stats)
        for region in stats.REGIONS + ("F",):
            for metric in stats.METRICS if region in stats.REGIONS else stats.F_METRICS:
                fields = [r["combined"]["regions"][region] for r in records[model]]
                values = [r[metric] for r in fields]
                reasons = [r.get(metric + "_reason") for r in fields]
                result = stats.sample_summary(values, reasons)
                good = [v for v in values if v is not None]
                missing = dict(sorted(Counter(r for v, r in zip(values, reasons) if v is None).items()))
                mean, sd = independent_summary(good)
                differences = [close(mean, result["mean"], metric), close(sd, result["sample_std_ddof1"], metric + " SD")]
                c.need(result["n_valid"] == len(good) and result["n_undefined"] == len(values) - len(good)
                       and result["undefined_reasons"] == missing
                       and result["mean_reason"] == (None if good else "no_finite_valid_samples")
                       and result["sample_std_reason"] == (None if len(good) >= 2 else "fewer_than_two_valid_samples"),
                       "Table3 N/null/reason accounting differs")
                cells.append(dict(model=model, region=region, metric=metric, **result))
                audits.append(dict(model=model, region=region, metric=metric, status="passed",
                                   maximum_absolute_difference=max(differences), independently_checked_N_and_nulls=True))
    c.need(len(cells) == 28, "Exactly 14 Table3 descriptive cells per model required")
    return cells, audits


def verify_production_children(root, manifest, runtimes):
    """Join each actual record runtime to the controller's owned production birth."""
    root = Path(root).resolve()
    request = c.checked(c.read(c.bound(manifest["request"])))
    c.need(manifest["request"] == c.desc(root / "request.json"), "Production launch request differs")
    gpu_inventory = c.validate_inventory(manifest)
    c.need(request["hosts"] == manifest["hosts"] and request["gpu_inventory"] == gpu_inventory
           and request["ownership"] == manifest["ownership"], "Production request topology differs")
    c.need(len(runtimes) == 8, "Exactly eight production runtimes required")
    by_gpu = {}
    for descriptor in runtimes:
        runtime = c.read(c.bound(descriptor))
        gpu = runtime["global_gpu_id"]
        c.need(type(gpu) is int and gpu in range(8) and gpu not in by_gpu, "Duplicate production runtime owner")
        gpu_item = c.validate_probe(manifest, gpu, runtime["host_identity"], runtime["local_gpu_inventory"])
        c.need(runtime["physical_gpu_id"] == gpu_item["local_index"] and runtime["host"] == gpu_item["host"]
               and runtime["uuid"] == gpu_item["uuid"]
               and runtime["identity"]["boot_id"] == runtime["host_identity"]["boot_id"],
               "Production runtime host/physical GPU ownership differs")
        by_gpu[gpu] = runtime
    children = []
    for gpu in range(8):
        path = root / "children" / ("production_gpu%d.json" % gpu)
        child = c.checked(c.read(path))
        c.need(child["schema"] == "secondary_owned_child_v1" and child["phase"] == "production"
               and child["name"] == "production_gpu%d" % gpu and child["request"] == manifest["request"]
               and child["host"] == gpu_inventory[gpu]["host"],
               "Owned production child scope/request differs")
        birth, actual = child["identity"], by_gpu[gpu]["identity"]
        c.need(all(birth[key] == actual[key] for key in ("pid", "starttick", "boot_id", "command")),
               "Production runtime does not match the owned child birth")
        c.need(type(birth["pid"]) is int and birth["pid"] > 0
               and type(birth["starttick"]) is int and birth["starttick"] > 0,
               "Invalid production child birth")
        command = child["command"]
        c.need(command == birth["command"] and len(command) == 7 and command[0] == request["python"]
               and Path(command[1]).resolve() == root / "source/worker.py" and command[2:4] == ["run", "--root"]
               and Path(command[4]).resolve() == root and command[5:] == ["--gpu", str(gpu)],
               "Owned production command/global GPU differs")
        children.append(c.desc(path))
    return children


def summarize_table4(records, combined):
    cells, audits = [], []
    c.need(tuple(combined.REGIONS) == ("P", "D", "O")
           and tuple(combined.TABLE4_METRICS) == ("gaussian_nll", "rank_corr", "mean_z2", "coverage95"),
           "Frozen Table4 definitions differ")
    for model in MODELS:
        payloads = [{"table4": r["combined"]["table4"]} for r in records[model]]
        for record, payload in zip(records[model], payloads):
            combined.validate_table4(payload["table4"])
            if record["status"] == "unavailable_geometry":
                c.need(all(row["status"] == "unavailable_geometry" for row in payload["table4"].values()),
                       "Technical Table4 status differs")
        for region in combined.REGIONS:
            for metric in combined.TABLE4_METRICS:
                result = combined.summary_cell(payloads, region, metric)
                values = [r["table4"][region][metric] for r in payloads if r["table4"][region][metric] is not None]
                reasons = Counter(r["table4"][region][metric + "_reason"] for r in payloads if r["table4"][region][metric] is None)
                mean, sd = independent_summary(values)
                diffs = [close(mean, result["mean"], metric), close(sd, result["sample_sd"], metric + " SD")]
                c.need(result["valid_n"] == len(values) and result["missing_n"] == len(payloads) - len(values)
                       and result["missing_reasons"] == dict(reasons) and result["ddof"] == 1
                       and result["sample_sd_reason"] == (None if len(values) >= 2 else "fewer_than_two_valid_samples"),
                       "Table4 N/null/reason accounting differs")
                cells.append(dict(model=model, region=region, metric=metric, **result))
                audits.append(dict(model=model, region=region, metric=metric, status="passed", maximum_absolute_difference=max(diffs)))
    c.need(len(cells) == 24, "Exactly 12 Table4 cells per model required")
    return cells, audits


def summarize_regional(records, rows, adapter, stats, report):
    data = np.full((2, len(REGIONS), len(SIGNALS), len(rows), 2), np.nan, dtype=np.float64)
    reasons, cells, audits = {}, [], []
    for mi, model in enumerate(MODELS):
        for index, record in enumerate(records[model]):
            fields = adapter.validate_fields(record["regional"])
            c.need(fields["status"] == record["status"], "Regional/primary status differs")
            if fields["status"] == "complete":
                c.bound(fields["roi_audit"]["landmarks"])
            for ri, region in enumerate(REGIONS):
                field = fields["regions"][region] if fields["status"] == "complete" else None
                reason = field["reason"] if field is not None else fields["reason"]
                reasons[mi, ri, index] = reason
                if reason is None:
                    for si, signal in enumerate(SIGNALS):
                        data[mi, ri, si, index] = field["truth"][signal], field["prediction"][signal]
        for ri, region in enumerate(REGIONS):
            missing = dict(sorted(Counter(reasons[mi, ri, i] for i in range(len(rows)) if reasons[mi, ri, i]).items()))
            for si, signal in enumerate(SIGNALS):
                pairs = data[mi, ri, si]
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
                result = stats.summarize_pairs(pairs[:, 0], pairs[:, 1], bootstrap_repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED)
                audit = report.independent_point_check(pairs[:, 0], pairs[:, 1], result)
                c.need(result["n"] == len(pairs) and result["n_input_pairs"] == len(pairs)
                       and result["n_omitted_pairs"] == 0 and result["bootstrap_repeats"] == 2000
                       and result["bootstrap_seed"] == 20261026 and len(pairs) + sum(missing.values()) == len(rows),
                       "Regional paired-N/bootstrap contract differs")
                cells.append(dict(model=model, region=region, signal=signal, n_total=len(rows), n_valid=len(pairs),
                                  n_undefined=len(rows)-len(pairs), undefined_reasons=missing, statistics=result,
                                  analysis_role="reference_domain" if region == "F_reference" else "anatomical_subroi",
                                  signal_role="secondary_diagnostic" if signal == "RGB_mean_secondary" else "primary_channel"))
                audits.append(dict(model=model, region=region, signal=signal, **audit))
                print(c.encoded(dict(event="secondary_regional_summary_cell", model=model, region=region,
                                     signal=signal, n_valid=len(pairs))).decode().strip(), flush=True)
    c.need(len(cells) == 48, "Exactly 24 regional agreement cells per model required")
    return cells, audits, data, reasons


def load_statistics(manifest, science):
    table3 = c.load_verified(manifest["original_stats"], "_secondary_frozen_table3_statistics")
    protocol = science["regional_protocol"]
    c.need(protocol == c.read(c.bound(manifest["original_regional_protocol"])), "Regional protocol binding differs")
    agreement = science["regional_runner"].stats
    c.need(Path(agreement.__file__).resolve() == c.bound(protocol["sources"]["agreement_stats.py"]), "Agreement estimator source differs")
    report = c.load_verified(protocol["sources"]["report.py"], "_secondary_frozen_regional_report")
    c.need(report.summarize_pairs is agreement.summarize_pairs, "Regional report resolved a different estimator")
    return table3, agreement, report


def write_report(folder, title, summary, audit, report, additional=None):
    """Write inspectable JSON, CSV, text, HTML and sealed output receipt."""
    folder = Path(folder)
    c.save(folder / "summary.json", c.seal(summary))
    c.save(folder / "independent_audit.json", c.seal(dict(status="passed", checks=audit, check_count=len(audit))))
    cells = summary["cells"]
    columns = list(cells[0])
    def shown(value):
        if isinstance(value, (dict, list)):
            return c.encoded(value).decode().strip().replace("\n", " ")
        return "NA" if value is None else str(value)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows([[shown(row.get(k)) for k in columns] for row in cells])
    report.save_bytes(folder / "summary.csv", output.getvalue().encode("utf8"))
    note = ("Frozen descriptive estimators; own valid N and explicit undefined reasons per endpoint. "
            "Sample groups are not verified independent patients. Original three-UNet 78-test Holm family remains unchanged; "
            "no new paired-test family or significance claim is introduced. 68% coverage is not implemented.")
    report.save_bytes(folder / "report.txt", (title + "\n\n" + note + "\n\n" + output.getvalue()).encode("utf8"))
    body = "<!doctype html><html><meta charset='utf-8'><title>" + html.escape(title) + "</title>"
    body += "<style>body{font:14px Arial;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%}</style>"
    body += "<h1>" + html.escape(title) + "</h1><p>" + html.escape(note) + "</p><table><tr>"
    body += "".join("<th>" + html.escape(k) + "</th>" for k in columns) + "</tr>"
    body += "".join("<tr>" + "".join("<td>" + html.escape(shown(row.get(k))) + "</td>" for k in columns) + "</tr>" for row in cells) + "</table>"
    for path in additional or []:
        relative = path.relative_to(folder).as_posix()
        body += "<p><a href='" + html.escape(relative, quote=True) + "'>" + html.escape(relative) + "</a></p>"
        if path.suffix == ".png":
            body += "<img src='" + html.escape(relative, quote=True) + "' alt='Regional paired agreement'>"
    report.save_bytes(folder / "report.html", (body + "</html>").encode("utf8"))
    paths = [folder / name for name in ("summary.json", "independent_audit.json", "summary.csv", "report.txt", "report.html")] + list(additional or [])
    receipt = c.seal(dict(schema="secondary_test_report_receipt_v1", status="verified", manifest=summary["manifest"],
                          cell_count=len(cells), independent_check_count=len(audit), outputs=[c.desc(p) for p in paths]))
    c.save(folder / "receipt.json", receipt)
    return c.desc(folder / "receipt.json")


def paired_csv(folder, rows, data, reasons, report):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("sample_id", "sample_index", "model", "region", "signal", "truth", "prediction", "prediction_minus_truth", "valid_pair", "undefined_reason"))
    for mi, model in enumerate(MODELS):
        for index, row in enumerate(rows):
            for ri, region in enumerate(REGIONS):
                for si, signal in enumerate(SIGNALS):
                    x, y = data[mi, ri, si, index]
                    valid = bool(np.isfinite([x, y]).all())
                    writer.writerow((row["sample_id"], index, model, region, signal, float(x) if valid else "",
                                     float(y) if valid else "", float(y-x) if valid else "", int(valid), reasons[mi, ri, index] or ""))
    path = folder / "paired_values.csv"
    report.save_bytes(path, output.getvalue().encode("utf8"))
    return path


def verify_completion(root, manifest_desc):
    complete = c.checked(c.read(Path(root) / "scientific_completion.json"))
    c.need(complete["schema"] == "secondary_test_scientific_completion_v1" and complete["status"] == "complete_two_models"
           and complete["manifest"] == manifest_desc and complete["models"] == list(MODELS)
           and complete["sample_count_per_model"] == COUNT and complete["record_count"] == 2 * COUNT
           and complete["table3_descriptive_cells"] == 28 and complete["table4_cells"] == 24
           and complete["regional_cells"] == 48 and complete["new_paired_tests"] == 0
           and complete["original_78_test_family_modified"] is False, "Scientific completion contract differs")
    for descriptor in complete["outputs"]:
        c.bound(descriptor)
    record_manifest = c.checked(c.read(Path(root) / "record_manifest.json"))
    c.need(record_manifest["schema"] == "secondary_test_final_record_manifest_v1"
           and record_manifest["manifest"] == manifest_desc and record_manifest["record_count"] == 2 * COUNT
           and len(record_manifest["records"]) == 2 * COUNT, "Final provenance inventory differs")
    roster = {(item["model"], item["index"]) for item in record_manifest["records"]}
    c.need(roster == {(model, index) for model in MODELS for index in range(COUNT)}, "Final provenance ownership incomplete")
    verified = set()
    def check(descriptor):
        key = descriptor["path"], descriptor["sha256"]
        if key not in verified:
            c.bound(descriptor)
            verified.add(key)
    for item in record_manifest["records"]:
        c.need(Path(item["source"]["path"]).resolve() == Path(root).resolve() / "records" / item["model"] / ("%06d.json" % item["index"]),
               "Final provenance record path differs")
        for key in ("source", "mask_ref", "mask_receipt", "runtime"):
            check(item[key])
    for descriptor in record_manifest["shard_completions"] + record_manifest["runtimes"]:
        check(descriptor)
    c.need(len(record_manifest["shard_completions"]) == 8, "Final provenance needs eight shards")
    for descriptor in record_manifest["sources"].values():
        check(descriptor)
    for descriptor in record_manifest["frozen_estimator_sources"].values():
        check(descriptor)
    for key in ("catalog", "original_stats", "original_combined_protocol", "original_regional_protocol", "training_gate"):
        check(record_manifest[key])
    manifest = c.checked(c.read(c.bound(manifest_desc)))
    c.need(verify_production_children(root, manifest, record_manifest["runtimes"]) == record_manifest["production_children"],
           "Production child provenance changed")
    collector = c.checked(c.read(Path(root) / "collector_receipt.json"))
    c.need(collector["status"] == "verified" and collector["manifest"] == manifest_desc
           and collector["record_count"] == 2 * COUNT and collector["report_receipts"] == complete["report_receipts"],
           "Collector receipt binding differs")
    for descriptor in complete["report_receipts"]:
        receipt = c.checked(c.read(c.bound(descriptor)))
        c.need(receipt["status"] == "verified" and receipt["manifest"] == manifest_desc, "Report receipt differs")
        for output in receipt["outputs"]:
            c.bound(output)
    return complete


def finalize(root, worker):
    root = Path(root).resolve()
    manifest = worker.load_manifest(root)
    manifest_desc = c.desc(root / "manifest.json")
    with c.lock(root, "collect"):
        if (root / "scientific_completion.json").exists():
            return verify_completion(root, manifest_desc)
        science = worker.load_science(manifest)
        records, inventory, shards, runtimes = verify_inventory(root, manifest, science, worker, expected_count=COUNT)
        production_children = verify_production_children(root, manifest, runtimes)
        stats3, agreement, report = load_statistics(manifest, science)
        frozen_sources = {"table3_statistics": c.desc(stats3.__file__), "combined": c.desc(science["combined"].__file__),
                          "agreement_statistics": c.desc(agreement.__file__), "regional_report": c.desc(report.__file__),
                          "regional_adapter": c.desc(science["helpers"].regional_adapter.__file__)}
        provenance = dict(manifest=manifest_desc, catalog=manifest["catalog"], sources=manifest["sources"],
                          original_stats=manifest["original_stats"], original_combined_protocol=manifest["original_combined_protocol"],
                          original_regional_protocol=manifest["original_regional_protocol"], training_bundles=manifest["training_bundles"],
                          training_gate=manifest["training_gate"], shard_completions=shards, runtimes=runtimes,
                          frozen_estimator_sources=frozen_sources, production_children=production_children)
        c.save(root / "record_manifest.json", c.seal(dict(schema="secondary_test_final_record_manifest_v1", status="complete",
                                                        record_count=len(inventory), records=inventory, **provenance)))
        shared = dict(schema="secondary_test_descriptive_report_v1", status="complete_two_models", manifest=manifest_desc,
                      models=list(MODELS), sample_count=COUNT, record_manifest=c.desc(root / "record_manifest.json"),
                      status_counts={m: dict(Counter(r["status"] for r in records[m])) for m in MODELS},
                      statistical_unit="sample_group", independent_patient_identity_verified=False,
                      new_paired_tests=0, original_78_test_family_modified=False)
        cells3, audit3 = summarize_table3(records, stats3)
        receipts = [write_report(root / "table3_report", "Additional frozen model TEST descriptive metrics", dict(shared, cells=cells3), audit3, report)]
        cells4, audit4 = summarize_table4(records, science["combined"])
        receipts.append(write_report(root / "table4_report", "Additional frozen model uncertainty and calibration", dict(shared, cells=cells4, coverage68_implemented=False), audit4, report))
        rows = science["ctx"]["rows"]
        regional, audit, data, reasons = summarize_regional(records, rows, science["helpers"].regional_adapter, agreement, report)
        folder = root / "regional_report"
        csv_path = paired_csv(folder, rows, data, reasons, report)
        plots = report.make_plots(folder, MODELS, data, regional)
        c.need(len(plots) == 32 and sum(p.suffix == ".png" for p in plots) == 16
               and sum(p.suffix == ".pdf" for p in plots) == 16, "Complete PNG/PDF regional figure inventory required")
        receipts.append(write_report(folder, "Additional frozen model regional UV agreement", dict(shared, cells=regional,
                                     bootstrap_repeats=2000, bootstrap_seed=20261026), audit, report, [csv_path] + plots))
        # 汇总后复核来源
        c.need(c.desc(root / "manifest.json") == manifest_desc, "Manifest changed while collecting")
        for item in inventory:
            c.bound(item["source"])
        for descriptor in shards + runtimes + production_children:
            c.bound(descriptor)
        for descriptor in list(manifest["sources"].values()) + list(frozen_sources.values()):
            c.bound(descriptor)
        c.save(root / "collector_receipt.json", c.seal(dict(schema="secondary_test_collector_receipt_v1", status="verified",
               manifest=manifest_desc, record_count=len(inventory), shard_completions=shards,
               independent_checks=dict(table3=28, table4=24, regional=48), report_receipts=receipts,
               outputs=[c.desc(root / "record_manifest.json")] + receipts)))
        completion = c.seal(dict(schema="secondary_test_scientific_completion_v1", status="complete_two_models",
                                manifest=manifest_desc, models=list(MODELS), sample_count_per_model=COUNT,
                                record_count=2 * COUNT, table3_descriptive_cells=28, table4_cells=24, regional_cells=48,
                                new_paired_tests=0, original_78_test_family_modified=False, coverage68_implemented=False,
                                report_receipts=receipts, outputs=[c.desc(root / "record_manifest.json"), c.desc(root / "collector_receipt.json")] + receipts))
        c.save(root / "scientific_completion.json", completion)
        return verify_completion(root, manifest_desc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = c.load_manifest(root)
    worker = c.load_verified(manifest["sources"]["worker"], "_secondary_collector_verified_worker")
    print(c.encoded(finalize(root, worker)).decode(), flush=True)


if __name__ == "__main__":
    main()
