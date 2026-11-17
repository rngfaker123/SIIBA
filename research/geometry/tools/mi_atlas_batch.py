#!/usr/bin/env python3
"""Checkpointed three-runtime frozen-mask batch -> common subset -> 54 MI jobs.

This controller uses ONLY the standard library. Native RGB preview, MediaPipe
geometry, masks and MI always execute in independent configured subprocesses.
No source images, company rules or historical masks are changed. A bound report
with fresh-preview AND all-11 geometry/mask exact parity is mandatory.

Config schema=mi_atlas_batch_config_v1; cohort, parity_report, model, vendor are
{path,sha256} descriptors (relative to config or absolute). runtimes has preview,
geometry,mask,mi, each {python,pythonpath:[optional isolated package paths]}.
output_root is a new batch folder; mask_workers=1..4(default4), pilot_count=2.
Parity report binds model_sha256 and runtime_fingerprints for preview/geometry/
mask: each has python (lexical absolute path), python_sha256, pythonpath and
versions (python plus exact keys in RUNTIME_MODULES). Versions are freshly
probed before each controller start, including resume; no package installs run.
Initial cohort comes from mi_atlas_cohort.py (flat modality fields,
paths relative to cohort.data_root). It is never refilled or reordered.

Each complete case has frozen stage receipts, P/D/O/F PNGs and native four-source
hashes. Count<1000 exclusions and the exact frozen geometry no-unique-face error
are separately reported; other subprocess/hash/runtime failures stop the batch.
No geometry-failure masks are fabricated. Incomplete stage
folders are preserved in that case's interrupted/ directory before retry.
All admitted cases use the identical subset for every N/seed/metric.

CLI: python tools/mi_atlas_batch.py --config FILE --phase all|masks|mi
Outputs JSON/JSONL only. No CSV/XLSX or full-pixel feature arrays are exported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = "mi_atlas_batch_20261025_v1"
COHORT_SCHEMA = "mi_atlas_initial_clean_val_cohort_20261025_v1"
STAGE_FILES = {"preview": "mi_atlas_frozen_preview.py",
               "geometry": "mi_atlas_frozen_geometry.py",
               "mask": "mi_atlas_frozen_mask_adapter.py",
               "mi": "mi_atlas_six_metrics.py"}
MODS = ("rgb", "uv", "plv", "plh")
DOMAINS = ("F", "P", "D", "O")
RUNTIME_MODULES = {"preview": ("PIL", "numpy"),
                   "geometry": ("PIL", "numpy", "cv2", "mediapipe"),
                   "mask": ("PIL", "numpy", "cv2")}
MINIMUM = 1000
INITIAL_COUNT = 1000
COHORT_SEED = 20261025
CLEAN_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
DEVELOPMENT_SHA = "2e2f5d1aca155ad06f655bf68c4d3f1cebc107e6d9e28cba846bba74ce9b3ae5"
SOURCE_FIELDS = ("sample_id", "subject_id", "source_height", "source_width") + tuple(
    name + suffix for name in MODS for suffix in ("_path", "_sha256"))
NO_FACE_ERROR = "ValueError: Expected exactly one face; no mask fabricated"
NO_FACE_STATUS = "technical_excluded_no_unique_face"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def seal(value):
    return dict(value, record_sha256=digest(value))


def checked(value):
    if value.get("record_sha256") != digest({k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("Receipt record digest mismatch")
    return value


def save_json(path, value, mutable=False):
    path, payload = Path(path), encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable:
        if path.read_bytes() != payload:
            raise ValueError("Refusing to overwrite frozen artifact: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if mutable:
            os.replace(str(temporary), str(path))
        else:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError("Concurrent differing frozen artifact")
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve(base, path):
    value = Path(path)
    return (value if value.is_absolute() else Path(base) / value).resolve()


def lexical_absolute(base, path):
    """Keep a venv's executable symlink path: resolving it loses venv context."""
    value = Path(path)
    return Path(os.path.abspath(str(value if value.is_absolute() else Path(base) / value)))


def probe_runtime(runtime, modules):
    """Version check under the exact child environment, before cache reuse."""
    code = ("import importlib,json,platform; modules=" + repr(list(modules)) + "; "
            "v={name:str(importlib.import_module(name).__version__) for name in modules}; "
            "v['python']=platform.python_version(); print(json.dumps(v,sort_keys=True))")
    result = subprocess.run([runtime["python"], "-c", code], env=child_env(runtime),
        capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)


def bound_file(base, descriptor):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str) or not valid_sha(descriptor.get("sha256")):
        raise ValueError("Require path/SHA256 descriptor")
    path = resolve(base, descriptor["path"])
    if not path.is_file() or sha(path) != descriptor["sha256"]:
        raise ValueError("Frozen file missing/changed: " + str(path))
    return path


def safe_child(parent, name):
    path = (Path(parent) / name).resolve()
    path.relative_to(Path(parent).resolve())
    return path


def descriptor(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def stage_source(project_root, filename):
    public_path = project_root / "tools" / filename
    if public_path.is_file():
        return public_path
    archived_path = public_path.with_name(public_path.stem + "_20261025.py")
    if archived_path.is_file():
        return archived_path
    if filename == STAGE_FILES["mi"] and project_root == ROOT:
        return ROOT.parent / "mutual_information" / "estimator" / filename
    return public_path


def load_settings(config_path, project_root=ROOT):
    """No masks or batch folders are created until this exact-parity gate passes."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if config.get("schema") != "mi_atlas_batch_config_v1":
        raise ValueError("Unsupported batch config schema")
    files = {}
    for key in ("cohort", "parity_report", "model", "vendor"):
        item = config.get(key)
        # 接受同级哈希字段
        if isinstance(item, str):
            item = dict(path=item, sha256=config.get(key + "_sha256"))
        files[key] = bound_file(config_path.parent, item)
    cohort, gate = read_json(files["cohort"]), read_json(files["parity_report"])
    if cohort.get("schema") != COHORT_SCHEMA or cohort.get("status") != "frozen_initial_cohort_before_images_and_masks" or not isinstance(cohort.get("rows"), list) or len(cohort["rows"]) != INITIAL_COUNT:
        raise ValueError("Unsupported/empty frozen initial cohort")
    selection = cohort.get("selection", {})
    if selection.get("seed") != COHORT_SEED or selection.get("selected_count") != INITIAL_COUNT or selection.get("requested_count") != INITIAL_COUNT or selection.get("split") != "val" or selection.get("preimage_membership") is not True or selection.get("replacement_allowed") is not False:
        raise ValueError("Initial cohort is not the fixed preimage VAL selection")
    source_rows = [{key: row[key] for key in SOURCE_FIELDS} for row in cohort["rows"]]
    if cohort.get("membership_sha256") != digest(source_rows):
        raise ValueError("Initial cohort membership digest changed")
    if cohort.get("source_manifest", {}).get("sha256") != CLEAN_SHA or cohort.get("development_manifest", {}).get("sha256") != DEVELOPMENT_SHA or cohort.get("development_manifest", {}).get("records") != 20:
        raise ValueError("Initial cohort source/development freeze differs")
    if gate.get("status") != "complete" or gate.get("all11_exact") is not True or gate.get("case_count") != 11 or gate.get("preview_all11_exact") is not True or not {"geometry", "mask"}.issubset(gate.get("stage_types", [])):
        raise ValueError("Formal batch blocked: fresh preview + all11 geometry/mask exact parity required")
    project_root = Path(project_root).resolve()
    if cohort.get("provenance", {}).get("builder_sha256") != sha(stage_source(project_root, "mi_atlas_cohort.py")):
        raise ValueError("Initial cohort builder source changed")
    scripts = {key: stage_source(project_root, filename) for key, filename in STAGE_FILES.items()}
    for key, report_key in (("preview", "preview_sha256"), ("geometry", "geometry_sha256"), ("mask", "adapter_sha256")):
        if not valid_sha(gate.get(report_key)) or sha(scripts[key]) != gate[report_key]:
            raise ValueError("Parity report does not bind current " + key + " source")
    sources = gate.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Parity report lacks frozen source dependencies")
    for name, expected in sources.items():
        if not valid_sha(expected):
            raise ValueError("Invalid parity dependency hash")
        path = project_root / (name if "/" in name or "\\" in name else "tools/" + name)
        path = path.resolve()
        path.relative_to(project_root)
        if sha(path) != expected:
            raise ValueError("Parity dependency changed: " + name)
    if gate.get("vendor_sha256") is not None and gate["vendor_sha256"] != sha(files["vendor"]):
        raise ValueError("Parity company module hash differs")
    if gate.get("model_sha256") != sha(files["model"]):
        raise ValueError("Parity MediaPipe model hash differs")
    rows, samples, subjects = [], set(), set()
    data_root = resolve(files["cohort"].parent, cohort["data_root"])
    for index, source in enumerate(cohort["rows"]):
        if source.get("cohort_index") != index or source.get("split") != "val" or source.get("source_sha") != CLEAN_SHA:
            raise ValueError("Initial cohort index/order changed")
        for key, used in (("sample_id", samples), ("subject_id", subjects)):
            value = source.get(key)
            if not isinstance(value, str) or not value or value in used:
                raise ValueError("Missing/duplicate initial " + key)
            used.add(value)
        shape = source.get("native_shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape):
            raise ValueError("Invalid native_shape")
        if shape != [source["source_height"], source["source_width"]]:
            raise ValueError("Native shape disagrees with frozen source metadata")
        row = dict(sample_id=source["sample_id"], subject_id=source["subject_id"],
                   native_shape=shape, initial_cohort_index=index)
        for mod in MODS:
            path = resolve(data_root, source[mod + "_path"])
            expected = source[mod + "_sha256"]
            if not path.is_file() or not valid_sha(expected):
                raise ValueError("Missing/unbound native source")
            row[mod] = dict(path=str(path), sha256=expected)
        rows.append(row)
    runtime_config = config.get("runtimes", {})
    runtimes = {}
    for key in STAGE_FILES:
        raw = runtime_config.get(key, {})
        python = lexical_absolute(config_path.parent, raw.get("python", ""))
        if not python.is_file():
            raise ValueError("Missing isolated Python runtime: " + key)
        pythonpath = raw.get("pythonpath", [])
        if isinstance(pythonpath, str):
            pythonpath = [pythonpath]
        if not isinstance(pythonpath, list) or any(not isinstance(v, str) for v in pythonpath):
            raise ValueError("pythonpath must be a string or list")
        pythonpath = [str(resolve(config_path.parent, path)) for path in pythonpath]
        if any(not Path(p).is_dir() for p in pythonpath):
            raise ValueError("Missing isolated package directory")
        runtimes[key] = dict(python=str(python), python_sha256=sha(python), pythonpath=pythonpath)
    fingerprints = gate.get("runtime_fingerprints", {})
    if not isinstance(fingerprints, dict) or not set(RUNTIME_MODULES).issubset(fingerprints):
        raise ValueError("Parity report lacks isolated runtime fingerprints")
    for key, modules in RUNTIME_MODULES.items():
        expected = fingerprints[key]
        if not isinstance(expected, dict) or any(expected.get(field) != runtimes[key][field]
            for field in ("python", "python_sha256", "pythonpath")):
            raise ValueError("Parity runtime executable/package path differs: " + key)
        versions = expected.get("versions")
        if not isinstance(versions, dict) or set(versions) != set(modules) | {"python"}:
            raise ValueError("Parity runtime version inventory differs: " + key)
        actual = probe_runtime(runtimes[key], modules)
        if actual != versions:
            raise ValueError("Parity runtime library versions changed: " + key)
        runtimes[key]["versions"] = actual
    workers, pilot = config.get("mask_workers", 4), config.get("pilot_count", 2)
    if type(workers) is not int or not 1 <= workers <= 4 or type(pilot) is not int or not 1 <= pilot <= len(rows):
        raise ValueError("mask_workers must be1..4; pilot_count must fit nonempty cohort")
    output = resolve(config_path.parent, config["output_root"])
    for row in rows:
        for mod in MODS:
            path = Path(row[mod]["path"])
            if path == output or output in path.parents:
                raise ValueError("Batch output cannot contain original source images")
    for path in list(files.values()) + [config_path]:
        if any(path == output / folder or output / folder in path.parents for folder in ("cases", "mi_run")):
            raise ValueError("Frozen inputs cannot lie in mutable stage output trees")
    code_hashes = {key: sha(path) for key, path in scripts.items()}
    contract = dict(schema=VERSION, config_sha256=sha(config_path), cohort_sha256=sha(files["cohort"]),
        parity_report_sha256=sha(files["parity_report"]), original_sample_count=len(rows),
        sample_order=[r["sample_id"] for r in rows], rows_sha256=digest(rows),
        runtimes=runtimes, stage_code_sha256=code_hashes, frozen_source_sha256=sources,
        model_sha256=sha(files["model"]), vendor_sha256=sha(files["vendor"]),
        controller_sha256=sha(__file__), mask_workers=workers, sequential_pilot_count=pilot,
        mask_cv_threads=2, blas_threads=1, mi_workers=4,
        eligibility="Unique frozen geometry and all P/D/O/F counts >=1000; same ordered subset for every metric/size/seed; no replacements",
        count_failures_are_exclusions=True, no_unique_face_is_technical_exclusion=True,
        runtime_or_hash_failures_are_exclusions=False)
    return dict(config_path=config_path, config=config, cohort=cohort, gate=gate,
                files=files, scripts=scripts, rows=rows, output=output,
                runtimes=runtimes, contract=contract, workers=workers, pilot=pilot)


def child_env(runtime):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if runtime["pythonpath"]:
        env["PYTHONPATH"] = os.pathsep.join(runtime["pythonpath"])
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


class Journal:
    def __init__(self, settings):
        self.root = settings["output"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.state = dict(schema=VERSION, status="starting", pid=os.getpid(),
            contract_sha256=digest(settings["contract"]), initial_samples=len(settings["rows"]),
            completed=0, eligible=0, excluded=0, technical_excluded=0, active={})

    def emit(self, event, **fields):
        with self.lock:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            record = dict(time_utc=timestamp, event=event, **fields)
            index = str(fields.get("index", ""))
            if event == "stage_started":
                self.state["active"][index] = {k: fields[k] for k in ("sample_id", "stage")}
            if event == "case_complete":
                self.state["active"].pop(index, None)
                self.state["completed"] += 1
                self.state["eligible" if fields["eligible"] else "excluded"] += 1
                if fields.get("exclusion_kind") == "technical":
                    self.state["technical_excluded"] += 1
            if event in ("masks_running", "masks_complete", "mi_running", "complete", "failed"):
                self.state["status"] = event
            self.state["updated_at_utc"] = timestamp
            self.state["last_event"] = record
            with (self.root / "events.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            save_json(self.root / "progress.json", self.state, mutable=True)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def verify_native(row):
    for name in MODS:
        if sha(row[name]["path"]) != row[name]["sha256"]:
            raise ValueError("Native source changed: " + row["sample_id"] + "/" + name)


def validate_stage_folder(folder, stage, expected_inputs, code_sha, shape):
    summary = read_json(folder / "summary.json")
    if summary.get("status") != "complete" or summary.get("input_sha256") != expected_inputs or summary.get("source_sha256") != code_sha or summary.get("native_shape") != shape:
        raise ValueError("Stage output not bound to exact inputs/code/native geometry: " + stage)
    expected = {"preview": {"rgb_preview.jpg"},
                "geometry": {"old_valid_native.png", "landmarks.json"},
                "mask": {"P.png", "D.png", "O.png", "F.png", "company_raw.json", "audit.json"}}[stage]
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("Unexpected stage artifact inventory")
    for name, value in hashes.items():
        if not valid_sha(value) or sha(safe_child(folder, name)) != value:
            raise ValueError("Stage artifact SHA mismatch: " + name)
    if stage == "mask":
        counts = summary.get("counts", {})
        if set(counts) != set(DOMAINS) or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts[d] for d in ("P", "D", "O")) != counts["F"] or counts["F"] > shape[0] * shape[1] or summary.get("partition_verified") is not True:
            raise ValueError("Invalid native mask partition/counts")
        configs = read_json(folder / "audit.json").get("configs")
        if not isinstance(configs, dict) or not configs:
            raise ValueError("Missing explicit frozen mask configurations")
    return summary


def last_nonempty_line(path):
    lines = Path(path).read_text(encoding="utf8", errors="replace").splitlines()
    return next((line for line in reversed(lines) if line.strip()), "")


def verify_technical_exclusion(receipt):
    if receipt.get("status") != NO_FACE_STATUS or receipt.get("binding", {}).get("stage") != "geometry" or receipt.get("counts") is not None:
        raise ValueError("Invalid geometry technical exclusion receipt")
    for prefix in ("log", "stderr"):
        if sha(receipt[prefix + "_path"]) != receipt[prefix + "_sha256"]:
            raise ValueError("Geometry exclusion log changed")
    if last_nonempty_line(receipt["stderr_path"]) != NO_FACE_ERROR:
        raise ValueError("Geometry exclusion reason is not the exact frozen exception")
    return receipt


def stage_run(settings, row, index, stage, arguments, inputs, journal):
    case = settings["output"] / "cases" / ("case_%06d" % index)
    case.mkdir(parents=True, exist_ok=True)
    folder, receipt_path = case / stage, case / (stage + "_receipt.json")
    runtime = settings["runtimes"][stage]
    script = settings["scripts"][stage]
    if sha(script) != settings["contract"]["stage_code_sha256"][stage] or sha(runtime["python"]) != runtime["python_sha256"]:
        raise ValueError("Runtime or stage source changed after batch freeze")
    input_sha = {key: sha(path) for key, path in inputs.items()}
    binding = dict(contract_sha256=digest(settings["contract"]), index=index,
        sample_id=row["sample_id"], row_sha256=digest(row), stage=stage,
        input_sha256=input_sha, code_sha256=sha(script), runtime=runtime)
    journal.emit("stage_started", index=index, sample_id=row["sample_id"], stage=stage)
    if receipt_path.exists():
        receipt = checked(read_json(receipt_path))
        if receipt.get("binding") != binding:
            raise ValueError("Cached stage belongs to different freeze/inputs")
        if receipt.get("status") == NO_FACE_STATUS:
            verify_technical_exclusion(receipt)
            journal.emit("stage_technical_exclusion_cached", index=index, sample_id=row["sample_id"], stage=stage)
            return receipt
        if receipt.get("status") != "complete":
            raise ValueError("Cached stage has unsupported status")
        summary = validate_stage_folder(folder, stage, input_sha, binding["code_sha256"], row["native_shape"])
        if receipt.get("summary_sha256") != sha(folder / "summary.json"):
            raise ValueError("Cached stage summary changed")
        journal.emit("stage_cached", index=index, sample_id=row["sample_id"], stage=stage)
        return summary
    if folder.exists():
        interrupted = case / "interrupted"
        interrupted.mkdir(exist_ok=True)
        destination = interrupted / (stage + "_" + uuid.uuid4().hex)
        folder.resolve().relative_to(case.resolve())
        destination.resolve().relative_to(case.resolve())
        folder.rename(destination)
        journal.emit("partial_stage_preserved", index=index, sample_id=row["sample_id"], stage=stage, path=str(destination))
    log = case / (stage + "_" + uuid.uuid4().hex + ".log")
    stderr_log = log.with_suffix(".stderr.log")
    command = [runtime["python"], str(script)] + [str(v) for v in arguments] + ["--output", str(folder)]
    started = time.monotonic()
    with log.open("xb") as stream, stderr_log.open("xb") as error_stream:
        completed = subprocess.run(command, env=child_env(runtime), cwd=str(ROOT),
                                   stdout=stream, stderr=error_stream, check=False)
    if input_sha != {key: sha(path) for key, path in inputs.items()}:
        raise ValueError("Stage inputs changed while running")
    if completed.returncode != 0:
        if stage == "geometry" and last_nonempty_line(stderr_log) == NO_FACE_ERROR:
            receipt = seal(dict(schema=VERSION, status=NO_FACE_STATUS, binding=binding,
                counts=None, log_path=str(log), log_sha256=sha(log), stderr_path=str(stderr_log),
                stderr_sha256=sha(stderr_log), elapsed_seconds=time.monotonic() - started))
            save_json(receipt_path, receipt)
            journal.emit("stage_technical_exclusion", index=index, sample_id=row["sample_id"],
                         stage=stage, reason="no_unique_face_geometry", stderr=str(stderr_log))
            return receipt
        raise RuntimeError("Stage failed, no sample excluded: %s/%s exit=%d log=%s stderr=%s" % (row["sample_id"], stage, completed.returncode, log, stderr_log))
    summary = validate_stage_folder(folder, stage, input_sha, binding["code_sha256"], row["native_shape"])
    receipt = seal(dict(schema=VERSION, status="complete", binding=binding,
        summary_sha256=sha(folder / "summary.json"), artifact_sha256=summary["artifact_sha256"],
        log_path=str(log), stderr_path=str(stderr_log), elapsed_seconds=time.monotonic() - started))
    save_json(receipt_path, receipt)
    journal.emit("stage_complete", index=index, sample_id=row["sample_id"], stage=stage,
                 seconds=receipt["elapsed_seconds"])
    return summary


def process_case(settings, index, journal):
    row = settings["rows"][index]
    verify_native(row)
    case = settings["output"] / "cases" / ("case_%06d" % index)
    rgb, uv = Path(row["rgb"]["path"]), Path(row["uv"]["path"])
    stage_run(settings, row, index, "preview", ["--rgb", rgb, "--expected-sha256", row["rgb"]["sha256"]], {"rgb": rgb}, journal)
    preview = case / "preview/rgb_preview.jpg"
    model, vendor = settings["files"]["model"], settings["files"]["vendor"]
    geometry = stage_run(settings, row, index, "geometry", ["--rgb-preview", preview, "--uv", uv, "--model", model],
              {"rgb_preview": preview, "uv": uv, "model": model}, journal)
    if geometry.get("status") == NO_FACE_STATUS:
        item = seal(dict(schema=VERSION, status=NO_FACE_STATUS, index=index,
            sample_id=row["sample_id"], subject_id=row["subject_id"], native_shape=row["native_shape"],
            contract_sha256=digest(settings["contract"]), source_row_sha256=digest(row),
            sources={name: row[name] for name in MODS}, masks=None, counts=None,
            eligible=False, exclusion_reason="no_unique_face_geometry", exclusion_kind="technical",
            insufficient_domains={}, mask_config_sha256=None,
            stage_receipt_sha256={s: sha(case / (s + "_receipt.json")) for s in ("preview", "geometry")}))
        save_json(case / "case_receipt.json", item)
        journal.emit("case_complete", index=index, sample_id=row["sample_id"], eligible=False,
                     counts=None, exclusion_kind="technical", reason="no_unique_face_geometry")
        return item
    landmarks, valid = case / "geometry/landmarks.json", case / "geometry/old_valid_native.png"
    summary = stage_run(settings, row, index, "mask", ["--uv", uv, "--landmarks", landmarks,
                        "--old-valid", valid, "--vendor", vendor],
                        {"uv": uv, "landmarks": landmarks, "old_valid": valid, "vendor": vendor}, journal)
    if summary.get("frozen_sources") != settings["gate"]["source_sha256"]:
        # 按文件名核验掩膜哈希

        gate_names = {Path(k).name: v for k, v in settings["gate"]["source_sha256"].items()}
        if any(gate_names.get(k) != v for k, v in summary.get("frozen_sources", {}).items()) or not summary.get("frozen_sources"):
            raise ValueError("Mask stage source inventory differs from parity gate")
    counts = summary["counts"]
    shortage = {name: counts[name] for name in DOMAINS if counts[name] < MINIMUM}
    configs = read_json(case / "mask/audit.json")["configs"]
    item = seal(dict(schema=VERSION, status="complete", index=index,
        sample_id=row["sample_id"], subject_id=row["subject_id"], native_shape=row["native_shape"],
        contract_sha256=digest(settings["contract"]), source_row_sha256=digest(row),
        sources={name: row[name] for name in MODS},
        masks={name: descriptor(case / "mask" / (name + ".png")) for name in DOMAINS},
        counts=counts, eligible=not shortage, exclusion_reason="insufficient_1000_native_pixels" if shortage else None,
        exclusion_kind="insufficient_pixels" if shortage else None,
        insufficient_domains=shortage, mask_config_sha256=digest(configs),
        stage_receipt_sha256={s: sha(case / (s + "_receipt.json")) for s in ("preview", "geometry", "mask")}))
    save_json(case / "case_receipt.json", item)
    journal.emit("case_complete", index=index, sample_id=row["sample_id"], eligible=not shortage, counts=counts)
    return item


def freeze_subset(settings, receipts):
    if len(receipts) != len(settings["rows"]) or any(r["index"] != i or r["sample_id"] != settings["rows"][i]["sample_id"] for i, r in enumerate(receipts)):
        raise ValueError("Initial cohort coverage/order incomplete")
    configs = {r["mask_config_sha256"] for r in receipts if r["status"] == "complete"}
    if len(configs) > 1:
        raise ValueError("Mask configurations differed across samples")
    admitted = [r for r in receipts if r["eligible"]]
    excluded = [dict(index=r["index"], sample_id=r["sample_id"], subject_id=r["subject_id"],
                     counts=r["counts"], reason=r["exclusion_reason"], kind=r["exclusion_kind"],
                     insufficient_domains=r["insufficient_domains"])
                for r in receipts if not r["eligible"]]
    eligibility = seal(dict(schema=VERSION, status="complete", contract_sha256=digest(settings["contract"]),
        initial_count=len(receipts), admitted_count=len(admitted), excluded_count=len(excluded),
        technical_excluded_count=sum(r["status"] == NO_FACE_STATUS for r in receipts),
        insufficient_pixels_excluded_count=sum(r["exclusion_kind"] == "insufficient_pixels" for r in receipts),
        admitted_sample_ids=[r["sample_id"] for r in admitted], exclusions=excluded,
        case_record_sha256={r["sample_id"]: r["record_sha256"] for r in receipts},
        replacement_count=0, min_required_each_domain=MINIMUM,
        same_subset_all_sizes_seeds_metrics=True))
    save_json(settings["output"] / "eligibility.json", eligibility)
    if not admitted:
        raise ValueError("No common eligible samples; no MI launched")
    rows = []
    for item in admitted:
        row = dict(sample_id=item["sample_id"], subject_id=item["subject_id"],
                   native_shape=item["native_shape"], initial_cohort_index=item["index"],
                   **item["sources"], **item["masks"])
        rows.append(row)
    manifest = dict(schema="mi_atlas_six_metrics_input_v1", status="frozen",
        cohort_id="initial_clean_val_common_mask1000_" + settings["contract"]["cohort_sha256"][:12],
        expected_sample_count=len(rows),
        frozen_provenance=dict(company_module_sha256=settings["contract"]["vendor_sha256"],
            mask_pipeline_sha256=digest({"stage_code": settings["contract"]["stage_code_sha256"],
                                         "mask_sources": settings["gate"]["source_sha256"]}),
            mask_config_sha256=next(iter(configs)),
            parity_report_sha256=settings["contract"]["parity_report_sha256"],
            initial_cohort_sha256=settings["contract"]["cohort_sha256"],
            eligibility_record_sha256=eligibility["record_sha256"]),
        selection="Initial frozen cohort filtered by unique frozen geometry and >=1000 native pixels in every P/D/O/F; technical failures reported separately; no replacements",
        rows=rows)
    path = settings["output"] / "mi_input.json"
    save_json(path, manifest)
    return path, eligibility


def run_masks(settings, journal):
    journal.emit("masks_running", workers=settings["workers"], pilot_count=settings["pilot"])
    results = [None] * len(settings["rows"])
    for index in range(settings["pilot"]):
        results[index] = process_case(settings, index, journal)
    journal.emit("pilot_complete", cases=settings["pilot"], workers=settings["workers"])
    # 失败停止派发
    next_index = settings["pilot"]
    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        active = {}
        for _ in range(settings["workers"]):
            if next_index < len(results):
                active[executor.submit(process_case, settings, next_index, journal)] = next_index
                next_index += 1
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            completed_now = []
            for future in finished:
                index = active.pop(future)
                results[index] = future.result()
                completed_now.append(index)
            for _ in completed_now:
                if next_index < len(results):
                    active[executor.submit(process_case, settings, next_index, journal)] = next_index
                    next_index += 1
    path, eligibility = freeze_subset(settings, results)
    journal.emit("masks_complete", initial=len(results), admitted=eligibility["admitted_count"], excluded=eligibility["excluded_count"], manifest=str(path))
    return path, eligibility


def validated_existing_subset(settings):
    receipts = []
    for index, row in enumerate(settings["rows"]):
        case = settings["output"] / "cases" / ("case_%06d" % index)
        item = checked(read_json(case / "case_receipt.json"))
        if item.get("index") != index or item.get("source_row_sha256") != digest(row) or item.get("contract_sha256") != digest(settings["contract"]):
            raise ValueError("Cached case binding differs")
        verify_native(row)
        if item["status"] == NO_FACE_STATUS:
            if item["masks"] is not None or item["counts"] is not None or item["eligible"] is not False:
                raise ValueError("Technical failure cannot fabricate masks or counts")
            verify_technical_exclusion(checked(read_json(case / "geometry_receipt.json")))
            stages = ("preview", "geometry")
        elif item["status"] == "complete":
            for domain in DOMAINS:
                bound_file(case, item["masks"][domain])
            stages = ("preview", "geometry", "mask")
        else:
            raise ValueError("Unsupported cached case status")
        for stage in stages:
            if sha(case / (stage + "_receipt.json")) != item["stage_receipt_sha256"][stage]:
                raise ValueError("Cached stage receipt changed")
        receipts.append(item)
    return freeze_subset(settings, receipts)


def run_mi(settings, manifest_path, eligibility, journal):
    runtime, script = settings["runtimes"]["mi"], settings["scripts"]["mi"]
    if sha(script) != settings["contract"]["stage_code_sha256"]["mi"] or sha(runtime["python"]) != runtime["python_sha256"]:
        raise ValueError("Frozen MI runtime/code changed")
    folder = settings["output"] / "mi_run"
    log = settings["output"] / ("mi_" + uuid.uuid4().hex + ".log")
    command = [runtime["python"], str(script), "--manifest", str(manifest_path),
               "--output-dir", str(folder), "--phase", "all", "--workers", "4"]
    journal.emit("mi_running", admitted=eligibility["admitted_count"], expected_results=54, log=str(log))
    with log.open("xb") as stream:
        completed = subprocess.run(command, env=child_env(runtime), cwd=str(ROOT),
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("MI subprocess failed; resume preserves completed results. log=" + str(log))
    summary = checked(read_json(folder / "summary.json"))
    if summary.get("status") != "complete" or summary.get("actual_result_count") != 54 or summary.get("expected_result_count") != 54:
        raise ValueError("MI returned without all 54 frozen results")
    mi_protocol = read_json(folder / "protocol.json")
    preparation = checked(read_json(folder / "preparation.json"))
    if mi_protocol.get("input_manifest_sha256") != sha(manifest_path) or mi_protocol.get("expected_sample_count") != eligibility["admitted_count"] or summary.get("protocol_sha256") != digest(mi_protocol) or preparation.get("protocol_sha256") != digest(mi_protocol) or summary.get("preparation_sha256") != preparation["record_sha256"]:
        raise ValueError("MI outputs are not bound to this exact common subset")
    result_records = summary.get("result_record_sha256", {})
    if len(result_records) != 54:
        raise ValueError("MI result inventory is incomplete")
    for name, expected in result_records.items():
        record = checked(read_json(safe_child(folder / "results", name)))
        if record["record_sha256"] != expected or record.get("protocol_sha256") != digest(mi_protocol):
            raise ValueError("MI result record binding differs")
    receipt = seal(dict(schema=VERSION, status="complete", contract_sha256=digest(settings["contract"]),
        initial_count=len(settings["rows"]), admitted_count=eligibility["admitted_count"], excluded_count=eligibility["excluded_count"],
        eligible_sample_order=eligibility["admitted_sample_ids"], replacement_count=0,
        mi_input_sha256=sha(manifest_path), eligibility_sha256=sha(settings["output"] / "eligibility.json"),
        summary=descriptor(folder / "summary.json"), result_count=54,
        company_unchanged=True, masks_frozen=True, cpu_only=True))
    save_json(settings["output"] / "receipt.json", receipt)
    journal.emit("complete", results=54, admitted=eligibility["admitted_count"], receipt=str(settings["output"] / "receipt.json"))
    return receipt


@contextmanager
def batch_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "batch.lock"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Batch lock exists; check process before clearing: " + str(path))
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), created_unix=time.time())))
        yield
    finally:
        path.unlink()


def run(config_path, phase="all"):
    settings = load_settings(config_path)
    with batch_lock(settings["output"]):
        save_json(settings["output"] / "batch_contract.json", settings["contract"])
        journal = Journal(settings)
        try:
            if phase == "mi":
                manifest, eligibility = validated_existing_subset(settings)
            else:
                manifest, eligibility = run_masks(settings, journal)
            if phase != "masks":
                return run_mi(settings, manifest, eligibility, journal)
            return eligibility
        except BaseException as exc:
            journal.emit("failed", error_type=type(exc).__name__, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("all", "masks", "mi"), default="all")
    args = parser.parse_args()
    run(args.config, args.phase)


if __name__ == "__main__":
    main()
