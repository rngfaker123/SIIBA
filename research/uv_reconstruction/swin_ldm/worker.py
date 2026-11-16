"""Versioned Swin/LDM TEST worker; original scientific functions remain unchanged."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from common import (need, encoded, digest, sha, read, desc, bound, seal, checked,
                    save, load_verified, lock, identity)
from common import load_manifest as _load_manifest
from common import GPU_COUNT, expected_cpus, runtime_probe, validate_inventory, validate_probe
import loader_adapter_v1 as loader
import measurement_bridge_v1 as bridge

ROOT = Path("/study/siiba/experiments/test_resume_eight_gpu_20261028_v1")
MODELS = ("swin", "ldm")
COUNT = 21899
G3_PINS = {
    "g3_worker": "0f66d305f861c9c90d5f2a77d12dfa0a8656dd7e76df20c1a4849bceacedf0e1",
    "g3_backend_compare": "d2b85eb2885af31e10d572cf11c99b81d6e0b07e64c48e4d480cca44c33bc8b5",
    "g3_regional_adapter": "eedb37baacecaf3f082aa8e7198c0262b321c0404848560671b14e22005b6194",
}


def owner(model, index):
    need(model in MODELS and isinstance(index, int) and not isinstance(index, bool)
         and 0 <= index < COUNT, "Invalid fixed model/index")
    return (0 if model == "swin" else 4) + index % 4


def model_for_gpu(gpu):
    need(type(gpu) is int and gpu in range(GPU_COUNT), "Eight fixed global GPU IDs only")
    return MODELS[gpu // 4]


def indices_for_gpu(gpu):
    model_for_gpu(gpu)
    return list(range(gpu % 4, COUNT, 4))


def training_bundle(manifest, model):
    need(model in MODELS, "Unknown model")
    fields = manifest["training_bundles"][model]
    need(set(fields) == {"source", "config", "binding", "protocol", "summary", "best_checkpoint"},
         "Exactly six final training descriptors required")
    return loader.TrainingBundle(**{k: loader.FrozenFile(**v) for k, v in fields.items()})


def load_manifest(root):
    root = Path(root)
    need(root.resolve() == ROOT, "New secondary TEST root required")
    manifest = _load_manifest(root)
    need(manifest["schema"] == "secondary_eight_gpu_manifest_v1"
         and manifest["models"] == list(MODELS) and manifest["sample_count"] == COUNT,
         "Frozen secondary model/cohort inventory differs")
    validate_inventory(manifest)
    for name, expected in G3_PINS.items():
        need(manifest["sources"][name]["sha256"] == expected, "Frozen G3 helper differs: " + name)
    need(manifest["sources"]["swin_reader"]["sha256"] == loader.PINS["swin"]["runner"],
         "Original Swin preprocessing reader required")
    for model in MODELS:
        loader.inspect_metadata(model, training_bundle(manifest, model))
    return manifest


def load_science(manifest, gpu=None):
    """Load immutable G3 context without altering its three-model inventory."""
    if gpu is not None:
        # 核对分片所属主机
        runtime_probe(gpu, manifest, select_cuda=True)
    for name, module_name in (("g3_backend_compare", "backend_compare"),
                              ("g3_regional_adapter", "regional_adapter")):
        sys.modules[module_name] = load_verified(manifest["sources"][name], module_name)
    g3 = load_verified(manifest["sources"]["g3_worker"], "_secondary_original_g3")
    visible = None if gpu is None else manifest["gpu_inventory"][gpu]["uuid"]
    science = g3.load_context(visible)
    ctx = science["ctx"]
    need(desc(g3.P / "evaluation/protocol.json") == manifest["original_combined_protocol"],
         "Original combined protocol differs")
    need(desc(g3.R / "protocol.json") == manifest["original_regional_protocol"],
         "Original regional protocol differs")
    need(desc(g3.bound(ctx["protocol"]["catalog"])) == manifest["catalog"],
         "Original complete TEST catalog differs")
    need(len(ctx["rows"]) == COUNT, "Original 21899 TEST rows required")
    science["g3"] = g3
    science["helpers"] = bridge.FrozenMetricHelpers(ctx["old"], science["combined"], g3.regional,
        science["regional_runner"], ctx["numeric"], ctx["rank"])
    return science


def load_context(root, gpu=None):
    manifest = load_manifest(root)
    return load_science(manifest, gpu)


def runtime_for(manifest, science, gpu):
    need(sorted(os.sched_getaffinity(0)) == manifest["gpu_inventory"][gpu]["cpus"],
         "Worker CPU affinity differs")
    probe = runtime_probe(gpu, manifest, select_cuda=False)
    need(probe["cuda_visible_devices"] == manifest["gpu_inventory"][gpu]["uuid"],
         "Explicit physical GPU visibility changed")
    runtime = science["ctx"]["old"].runtime_info("cuda:0", 2)
    need(runtime == dict(science["ctx"]["protocol"]["runtime"], device="cuda:0"),
         "Original metric software/precision/runtime differs")
    need(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Original deterministic cuBLAS setting differs")
    return dict(probe, runtime=runtime, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                identity=identity(os.getpid()), manifest_sha256=digest(manifest))


def verify_runtime(value, manifest, science, gpu):
    item = manifest["gpu_inventory"][gpu]
    validate_probe(manifest, gpu, value["host_identity"], value["local_gpu_inventory"])
    need(value["global_gpu_id"] == gpu and value["host"] == item["host"]
         and value["physical_gpu_id"] == item["local_index"] and value["uuid"] == item["uuid"]
         and value["cuda_visible_devices"] == item["uuid"]
         and value["cublas_workspace_config"] == ":4096:8", "Record physical runtime differs")
    need(value["runtime"] == dict(science["ctx"]["protocol"]["runtime"], device="cuda:0"),
         "Record precision/software runtime differs")
    need(value["manifest_sha256"] == digest(manifest), "Runtime manifest differs")
    need(value["identity"]["affinity"] == item["cpus"] and value["identity"]["nice"] == 19
         and value["identity"]["pid"] > 0 and value["identity"]["starttick"] > 0
         and value["identity"]["boot_id"] == manifest["hosts"][item["host"]]["boot_id"],
         "Recorded worker identity/CPU priority differs")
    return value


def prepare_case(science, index, wait=False):
    g3, ctx = science["g3"], science["ctx"]
    row = ctx["rows"][index]
    ref_path, receipt_path, receipt = g3.load_mask(science, index, wait=wait)
    case = dict(row=row, ref_path=ref_path, receipt_path=receipt_path, receipt=receipt,
                technical=receipt["status"] in ctx["old"].TECHNICAL)
    if not case["technical"]:
        weights, rois, audit = g3.regional.prepare_geometry(science["regional_runner"],
            science["combined"], ctx, row, receipt_path, receipt)
        case.update(weights=weights, rois=rois, roi_audit=audit,
            rgb=ctx["old"].prepare_image(row["rgb"], row["native_shape"], ctx["catalog"]["data_root"]),
            truth=ctx["old"].prepare_image(row["uv"], row["native_shape"], ctx["catalog"]["data_root"]))
    return case


def array_descriptor(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


class CapturingIO(bridge.TorchTensorIO):
    """Retain the exact pair used by all helpers for diagnostic evidence only."""
    def prediction_pair(self, output):
        self.pair = super().prediction_pair(output)
        return self.pair


def evaluate_case(adapter, science, case, device="cuda:0"):
    if case["technical"]:
        return bridge.unavailable_measurements(science["helpers"], case["row"], case["receipt"]["status"]), None
    converter = CapturingIO(device)
    payload = bridge.measure_one(adapter, converter, science["helpers"], case["row"],
        case["rgb"], case["truth"], case["weights"], case["rois"], case["roi_audit"])
    science["g3"].check_program_tensors(dict(prediction=converter.pair[0], sigma=converter.pair[1]),
                                       dict(prediction=converter.pair[0], sigma=converter.pair[1]))
    return payload, converter.pair


def validate_payload(science, payload):
    need(set(payload["combined"]) == {"regions", "table4"}, "Unexpected combined fields")
    g3 = science["g3"]
    for kind, fields in (("table3", payload["combined"]["regions"]),
                         ("table4", payload["combined"]["table4"]), ("regional", payload["regional"])):
        g3.compare_program_payloads(fields, fields, kind)
    science["combined"].validate_table4(payload["combined"]["table4"])
    g3.regional.validate_fields(payload["regional"])
    need(payload["status"] == payload["regional"]["status"], "Regional/main status differs")


def make_record(root, manifest, science, model, index, gpu, runtime_desc, case, payload, pair):
    row = case["row"]
    item = manifest["gpu_inventory"][gpu]
    origin = dict(kind="technical_derived" if case["technical"] else "gpu_new", record=None)
    return seal(dict(schema="secondary_test_sample_v1", index=index, sample_id=row["sample_id"],
        subject_id=row["subject_id"], model_id=model, source_row_sha256=digest(row),
        checkpoint_sha256=manifest["training_bundles"][model]["best_checkpoint"]["sha256"],
        manifest_sha256=sha(Path(root) / "manifest.json"), status=payload["status"],
        mask_ref=desc(case["ref_path"]), mask_receipt=desc(case["receipt_path"]),
        inputs={k: row[k] for k in ("rgb", "uv", "native_shape")},
        input_arrays=None if case["technical"] else {k: array_descriptor(case[k]) for k in ("rgb", "truth")},
        output_arrays=None if pair is None else dict(prediction=array_descriptor(pair[0]), sigma=array_descriptor(pair[1])),
        sources=manifest["sources"], training_bundle=manifest["training_bundles"][model],
        combined=payload["combined"], regional=payload["regional"],
        provenance=dict(combined=origin, regional=origin),
        execution=dict(forward_count=payload["forward_count"], global_gpu_id=gpu,
                       physical_gpu_id=item["local_index"], host=item["host"], uuid=item["uuid"], device="cuda:0",
                       runtime=runtime_desc, numeric_cpu_cuda_policy="diagnostic_only"),
        continuation=dict(status="computed_secondary_v1", parent_record=None,
                          new_forward_count=payload["forward_count"])))


def verify_record(root, manifest, science, model, index, value=None):
    root = Path(root)
    record = checked(read(root / "records" / model / ("%06d.json" % index))) if value is None else checked(value)
    ctx = science["ctx"]; row = ctx["rows"][index]; gpu = owner(model, index)
    need(record["schema"] == "secondary_test_sample_v1" and record["index"] == index
         and record["model_id"] == model and record["sample_id"] == row["sample_id"]
         and record["subject_id"] == row["subject_id"] and record["source_row_sha256"] == digest(row),
         "Record TEST identity differs")
    need(record["manifest_sha256"] == sha(root / "manifest.json")
         and record["checkpoint_sha256"] == manifest["training_bundles"][model]["best_checkpoint"]["sha256"]
         and record["training_bundle"] == manifest["training_bundles"][model]
         and record["sources"] == manifest["sources"], "Record sources/model/manifest differ")
    need(record["inputs"] == {k: row[k] for k in ("rgb", "uv", "native_shape")}, "Original input descriptor binding differs")
    ref, receipt_path, receipt = science["g3"].load_mask(science, index, wait=False)
    need(record["mask_ref"] == desc(ref) and record["mask_receipt"] == desc(receipt_path),
         "Original mask reference/receipt differs")
    technical = receipt["status"] in ctx["old"].TECHNICAL
    expected_status = "unavailable_geometry" if technical else "complete"
    execution = record["execution"]
    item = manifest["gpu_inventory"][gpu]
    need(record["status"] == expected_status and execution["forward_count"] == int(not technical)
         and execution["global_gpu_id"] == gpu and execution["physical_gpu_id"] == item["local_index"]
         and execution["host"] == item["host"] and execution["uuid"] == item["uuid"] and execution["device"] == "cuda:0"
         and execution["numeric_cpu_cuda_policy"] == "diagnostic_only", "Record ownership/status/forward count differs")
    verify_runtime(read(bound(execution["runtime"])), manifest, science, gpu)
    expected_origin = dict(kind="technical_derived" if technical else "gpu_new", record=None)
    need(record["provenance"] == dict(combined=expected_origin, regional=expected_origin)
         and record["continuation"] == dict(status="computed_secondary_v1", parent_record=None,
                                           new_forward_count=int(not technical)), "No imported/duplicate inference provenance permitted")
    if technical:
        expected = bridge.unavailable_measurements(science["helpers"], row, receipt["status"])
        need(record["combined"] == expected["combined"] and record["regional"] == expected["regional"]
             and record["input_arrays"] is None and record["output_arrays"] is None, "Technical reason/output differs")
    else:
        need(set(record["input_arrays"]) == {"rgb", "truth"} and set(record["output_arrays"]) == {"prediction", "sigma"},
             "Actual input/output array hashes missing")
        for name, item in dict(record["input_arrays"], **record["output_arrays"]).items():
            shape = [1024, 1024] if name == "sigma" else [3, 1024, 1024]
            need(item["shape"] == shape and item["dtype"] == "float32" and loader.valid_sha(item["sha256"]),
                 "Output/input tensor descriptor differs")
    validate_payload(science, record)
    return record


validate_record = verify_record


def read_record(root, manifest, science, model, index):
    return verify_record(root, manifest, science, model, index)


def verify_gate(root, manifest):
    root = Path(root)
    gate = checked(read(root / "validation/acceptance.json"))
    need(gate["status"] == "passed" and gate["manifest"] == desc(root / "manifest.json")
         and gate["models"] == list(MODELS) and gate["comparisons"] == 80
         and gate["same_card_repeat_required"] is True and gate["numeric_differences"] == "diagnostic_only",
         "Prospective secondary validation gate absent or differs")
    need(len(gate["records"]) == len({d["path"] for d in gate["records"]}) == 80
         and len(gate["receipts"]) == len({d["path"] for d in gate["receipts"]}) == GPU_COUNT
         and gate["cpu_forwards"] == 80 and gate["cuda_forwards"] == 160
         and gate["independent_raw_tensor_and_metric_recalculation"] is True
         and gate["source"] == manifest["sources"]["validation"],
         "Validation gate lacks independent complete raw evidence")
    bound(gate["plan"])
    for descriptor in gate["records"] + gate["receipts"]:
        bound(descriptor)
    return gate


def run(root, gpu):
    root = Path(root); model = model_for_gpu(gpu)
    manifest = load_manifest(root)
    os.sched_setaffinity(0, set(manifest["gpu_inventory"][gpu]["cpus"]))
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True, capture_output=True)
    science = load_science(manifest, gpu)
    verify_gate(root, manifest)
    runtime = runtime_for(manifest, science, gpu)
    runtime_path = root / "runtimes" / ("gpu_%d.json" % gpu)
    indices = indices_for_gpu(gpu)
    progress_path = root / "progress" / ("shard_%d.json" % gpu)
    completed = []; adapter = None; active_index = None
    with lock(root, "gpu_%d" % gpu):
        try:
            save(runtime_path, runtime)
            for index in indices:
                active_index = index
                save(progress_path, dict(status="waiting_or_verifying", gpu=gpu, model=model,
                    active_index=index, completed=len(completed), expected=len(indices), time=time.time()), mutable=True)
                path = root / "records" / model / ("%06d.json" % index)
                if path.exists():
                    read_record(root, manifest, science, model, index)
                else:
                    case = prepare_case(science, index, wait=True)
                    if not case["technical"] and adapter is None:
                        adapter = loader.load_adapter(model, training_bundle(manifest, model), "cuda:0")
                    payload, pair = evaluate_case(adapter, science, case)
                    record = make_record(root, manifest, science, model, index, gpu, desc(runtime_path), case, payload, pair)
                    verify_record(root, manifest, science, model, index, record)
                    save(path, record)
                completed.append(desc(path))
                save(progress_path, dict(status="running", gpu=gpu, model=model, active_index=index,
                    completed=len(completed), expected=len(indices), time=time.time()), mutable=True)
            save(root / "completion" / ("shard_%d.json" % gpu), seal(dict(
                schema="secondary_test_shard_completion_v1", status="complete", gpu=gpu, model=model,
                indices=indices, manifest=desc(root / "manifest.json"), records=completed, runtime=desc(runtime_path))))
            save(progress_path, dict(status="complete", gpu=gpu, model=model, completed=len(completed),
                 expected=len(indices), time=time.time()), mutable=True)
        except BaseException as error:
            save(root / "failure" / ("shard_%d.json" % gpu), seal(dict(
                schema="secondary_test_worker_failure_v1", status="failed", gpu=gpu, model=model,
                index=active_index, completed_records=len(completed), manifest=desc(root / "manifest.json"),
                error_type=type(error).__name__, error=str(error), time=time.time())))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.gpu)


if __name__ == "__main__":
    main()
