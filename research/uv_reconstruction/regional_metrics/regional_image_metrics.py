"""Frozen single/joint/U-Net++ evaluation on approved P/D/O/F coverage.

This entry point never segments, trains, chooses checkpoints, or changes masks.
It consumes the separately sealed TEST mask receipts, preserves unavailable
cases, and evaluates all three BEST checkpoints in FP32, one at a time.
"""
from __future__ import annotations

import argparse
import builtins
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import types

SOURCE_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
SCHEMA = "uv_three_region_eval_20261025_v1"
MODEL_IDS = ("unet_single", "unet_multi", "unetplusplus")
REGIONS = ("P", "D", "O", "F")
METRICS = ("l1", "psnr", "ssim", "mean_sigma")
TECHNICAL = ("no_unique_face", "invalid_geometry")
TECHNICAL_ERRORS = {"no_unique_face": "ValueError: Expected exactly one face; no mask fabricated", "invalid_geometry": "ValueError: Nonfinite or out-of-image polygon vertex: no automatic clipping"}
MASK_ADAPTER_SHA = "db52cafb66d40f3b40592dff11724888ce6c6fbf37731df2cc56e250b4320e0e"
VENDOR_SHA = "de1c4d52ab781d3440b694afb0d9480eb0af1359a365aa5290ae0d750a31a0e9"
SOURCE_PINS = {
    "src/unet_baseline.py": "3112bc3020b2005c2897716b3836099616f10afb34575e23224c5c5160310c05",
    "src/uv_uncertainty_unet.py": "669401ef8beeafffba233844d2c2728651e1afffaf203b8847517f6d8fa5dc8f",
    "src/joint_uv_uncertainty_unet.py": "a3c179708a4c48337e297f15977294a283a5894cd6bee77e9e1f558032778be3",
    "src/uv_uncertainty_unetplusplus.py": "8205ff7c5f9fb4c70eab38a86145fedac56d1917a5c769b984679592f5739589",
    "tools/uv_brightspot_metrics_v1.py": "3042ed387139d4577166eb223eab7db07f9470e678a05e01421b32cc53ee6a1d",
    "tools/uv_architecture_comparison_metrics_v1.py": "343d12ca534f2013081ec4117cd5d9361ffa9a75caf638917549168d02806798",
    "tools/uv_architecture_comparison_metrics_v10.py": "128889d990dca37176b682a48c3d01e9c7052b97efd31fbaca6dd75659333a04",
}
MODEL_PINS = {
    "unet_single": dict(sha="6f9e5c2a8745a3f67794f993961bfc6612421f04be178f41288f704c53d3cbfb", epoch=18, parameters=7763140, source="src/uv_uncertainty_unet.py", cls="UVUncertaintyUNet", output="mean", selection="full_VAL_UV_L1"),
    "unet_multi": dict(sha="204dbffedc15fa2cc6053ce6b6d1aa1464f9978c103c5814f84692d69c80844e", epoch=11, parameters=16909738, source="src/joint_uv_uncertainty_unet.py", cls="JointUVUncertaintyUNet", output="uv", selection="full_VAL_equal_modality_macro_L1"),
    "unetplusplus": dict(sha="81d7dc5d40f2b36d2ac64a10ca6a5a443af3854d70fc88c75027ca5ecd2dec5f", epoch=17, parameters=9045924, source="src/uv_uncertainty_unetplusplus.py", cls="UVUncertaintyUNetPlusPlus", output="mean", selection="full_VAL_UV_L1"),
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def seal(value):
    value = dict(value)
    if "record_sha256" in value:
        raise ValueError("Cannot reseal an already sealed record")
    return {**value, "record_sha256": digest(value)}


def verify_seal(value):
    body = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("record_sha256") != digest(body):
        raise ValueError("Record seal differs")
    return value


def write_new(path, value):
    """Immutable atomic publication; an identical existing object is resumable."""
    path = Path(path)
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to overwrite different evidence: " + str(path))
        return
    temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink()
        raise ValueError("Concurrent publication: " + str(path))
    os.replace(temporary, path)


def resolve_descriptor(desc, base):
    if not isinstance(desc, dict) or not isinstance(desc.get("path"), str) or not isinstance(desc.get("sha256"), str):
        raise ValueError("Expected path/sha256 descriptor")
    path = Path(desc["path"])
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve()
    if sha(path) != desc["sha256"]:
        raise ValueError("Artifact SHA mismatch: " + str(path))
    return path


class FrozenSources:
    """Import unchanged frozen files under a private namespace, never src.*.

    The import hook redirects only these files' absolute project imports; source
    bytes and Python code are not rewritten. Ordinary package imports are intact.
    """
    def __init__(self, descriptors, base):
        if set(descriptors) != set(SOURCE_PINS):
            raise ValueError("Frozen source closure is incomplete or has extra files")
        self.paths = {}
        for logical, expected in SOURCE_PINS.items():
            if descriptors[logical]["sha256"] != expected:
                raise ValueError("Unapproved frozen source: " + logical)
            self.paths[logical] = resolve_descriptor(descriptors[logical], base)
        self.namespace = "_uv_three_region_frozen_" + digest(descriptors)[:16]
        for name in (self.namespace, self.namespace + ".src", self.namespace + ".tools"):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module

    def import_hook(self, name, globals=None, locals=None, fromlist=(), level=0):
        logical = name.replace(".", "/") + ".py"
        if level == 0 and logical in self.paths:
            module = self.load(logical)
            if fromlist:
                return module
            return sys.modules[self.namespace + "." + name.split(".")[0]]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, logical):
        name = self.namespace + "." + logical[:-3].replace("/", ".")
        if name in sys.modules:
            return sys.modules[name]
        path = self.paths[logical]
        if sha(path) != SOURCE_PINS[logical]:
            raise ValueError("Frozen source changed before import")
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = name.rsplit(".", 1)[0]
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self.import_hook}
        sys.modules[name] = module
        try:
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        except BaseException:
            del sys.modules[name]
            raise
        setattr(sys.modules[module.__package__], name.rsplit(".", 1)[1], module)
        return module


def validate_catalog(catalog, base):
    if catalog.get("schema") != "uv_three_region_test_catalog_v1" or catalog.get("status") != "frozen":
        raise ValueError("Unfrozen or foreign TEST catalog")
    if catalog.get("source_manifest", {}).get("sha256") != SOURCE_SHA:
        raise ValueError("Not the approved clean split")
    resolve_descriptor(catalog["source_manifest"], base)
    rows = catalog.get("rows", [])
    if len(rows) != 21899:
        raise ValueError("Formal evaluation requires all 21,899 TEST groups")
    for key in ("sample_id", "subject_id"):
        ids = [row.get(key) for row in rows]
        if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("Missing or repeated " + key)
    for row in rows:
        if row.get("split") != "test" or len(row.get("native_shape", [])) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v < 1024 for v in row["native_shape"]):
            raise ValueError("Invalid TEST row/shape")
        for key in ("rgb", "uv", "plv", "plh"):
            if not isinstance(row.get(key), dict) or not row[key].get("path") or len(row[key].get("sha256", "")) != 64:
                raise ValueError("Missing raw descriptor")
    return rows


def validate_mask_receipt(receipt, row, index, catalog_sha, contract_sha, raw_base=None):
    verify_seal(receipt)
    for key, expected in (("index", index), ("sample_id", row["sample_id"]), ("subject_id", row["subject_id"]), ("native_shape", row["native_shape"]), ("source_row_sha256", digest(row)), ("catalog_sha256", catalog_sha), ("contract_sha256", contract_sha)):
        if receipt.get(key) != expected:
            raise ValueError("Mask receipt binding differs: " + key)
    if receipt.get("schema") != "uv_three_region_test_mask_case_v1":
        raise ValueError("Foreign mask receipt")
    for key in ("rgb", "uv", "plv", "plh"):
        expected = Path(row[key]["path"])
        if not expected.is_absolute() and raw_base is not None:
            expected = Path(raw_base) / expected
        actual = receipt.get("sources", {}).get(key, {})
        if actual.get("sha256") != row[key]["sha256"] or Path(actual.get("path", "")).resolve() != expected.resolve():
            raise ValueError("Mask raw-source descriptors differ: " + key)
    h, w = row["native_shape"]
    side, left, top = min(h, w), (w - min(h, w)) // 2, (h - min(h, w)) // 2
    if receipt["status"] == "complete":
        if receipt.get("artifact_kind") != "projected_region_coverage" or receipt.get("projected_shape") != [1024, 1024] or receipt.get("crop_box_xyxy") != [left, top, left + side, top + side]:
            raise ValueError("Mask projection geometry differs from model crop")
        if not receipt.get("npz"):
            raise ValueError("Complete mask without coverage")
    elif receipt["status"] not in TECHNICAL or receipt.get("npz") is not None:
        raise ValueError("Unapproved mask status")
    return receipt


def validate_weights(weights, shape=(1024, 1024)):
    import numpy as np
    if set(weights) != set(REGIONS):
        raise ValueError("Require exactly P/D/O/F coverage arrays")
    checked = {}
    for name in REGIONS:
        a = weights[name]
        if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
            raise ValueError("Invalid fractional coverage: " + name)
        checked[name] = a.astype(np.float64)
    if not np.allclose(checked["P"] + checked["D"] + checked["O"], checked["F"], rtol=0, atol=2e-6):
        raise ValueError("Projected masks do not partition F")
    return checked


def validate_mask_contract(contract, catalog_sha, rows):
    if (contract.get("schema") != "uv_three_region_test_mask_contract_v1"
            or contract.get("catalog", {}).get("sha256") != catalog_sha
            or contract.get("stage_code_sha256", {}).get("mask") != MASK_ADAPTER_SHA
            or contract.get("vendor_sha256") != VENDOR_SHA
            or contract.get("test_sample_count") != len(rows)
            or contract.get("sample_order") != [r["sample_id"] for r in rows]
            or contract.get("source_row_order_sha256") != digest(rows)
            or contract.get("minimum_pixel_count_filter_applied") is not False
            or contract.get("empty_domain_is_exclusion") is not False):
        raise ValueError("Mask contract differs from frozen approved TEST protocol")
    projection = contract.get("projection", {})
    if projection != dict(size=1024, dtype="float32", resampling="Pillow_BOX_mode_F", crop="center_square_floor_offsets", partition_tolerance=2e-6):
        raise ValueError("Mask projection contract differs")


def validate_projection_audit(receipt, receipt_path, weights):
    audit_path = resolve_descriptor(receipt["projection_audit"], Path(receipt_path).parent)
    audit = verify_seal(read_json(audit_path))
    for key in ("npz", "source_row_sha256", "contract_sha256", "native_shape", "crop_box_xyxy", "projected_shape", "counts_native", "counts_native_in_crop", "counts_native_outside_crop"):
        if audit.get(key) != receipt.get(key):
            raise ValueError("Projection audit binding differs: " + key)
    for region in REGIONS:
        actual = float(weights[region].astype("float64").sum())
        expected = receipt.get("effective_projected_area", {}).get(region)
        if not isinstance(expected, (int, float)) or not math.isfinite(expected) or abs(actual - expected) > 1e-7:
            raise ValueError("Projected support area differs: " + region)
    return audit


def validate_technical_evidence(receipt, receipt_path):
    path = resolve_descriptor(receipt["failure_stderr"], Path(receipt_path).parent)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines or lines[-1] != TECHNICAL_ERRORS[receipt["status"]]:
        raise ValueError("Technical unavailability lacks the exact original exception")


def measure(target, prediction, sigma, weights, numeric, shape=(1024, 1024)):
    """Generalize the frozen numeric helpers, not the old V10 domain builder."""
    import numpy as np
    target = numeric.validate_image(target, "UV truth")
    prediction = numeric.validate_image(prediction, "UV prediction")
    if target.shape != (3, *shape) or prediction.shape != target.shape:
        raise ValueError("Unexpected model output geometry")
    sigma = numeric._validate_sigma(sigma, shape, numeric.GAUSSIAN_SIGMA_KIND)
    checked = validate_weights(weights, shape)
    delta = prediction - target
    absolute, squared = np.abs(delta).mean(axis=0), np.square(delta).mean(axis=0)
    ssim = numeric.ssim_map_rgb(target, prediction)
    result = {}
    radius = numeric.SSIM_RADIUS
    for name in REGIONS:
        weight = checked[name]
        center = weight[radius:-radius, radius:-radius]
        area, center_area = float(weight.sum()), float(center.sum())
        mse = numeric._weighted_mean(squared, weight, name + " MSE")
        result[name] = {
            "l1": numeric._weighted_mean(absolute, weight, name + " L1"),
            "mse": mse, "psnr": numeric._psnr(mse),
            "ssim": numeric._weighted_mean(ssim, center, name + " SSIM"),
            "mean_sigma": numeric._weighted_mean(sigma, weight, name + " sigma"),
            "l1_reason": None if area else "empty_region_after_model_crop",
            "psnr_reason": "empty_region_after_model_crop" if mse is None else "zero_mse_psnr_positive_infinity" if mse == 0 else None,
            "ssim_reason": None if center_area else "no_valid_window_centers_in_region",
            "mean_sigma_reason": None if area else "empty_region_after_model_crop",
            "effective_pixel_area": area, "ssim_effective_center_area": center_area,
            "positive_weight_cells": int(np.count_nonzero(weight)),
        }
    return result


def unavailable_regions(reason):
    return {region: {**{metric: None for metric in METRICS}, **{metric + "_reason": reason for metric in METRICS}, "mse": None, "effective_pixel_area": None, "ssim_effective_center_area": None, "positive_weight_cells": None} for region in REGIONS}


def prepare_image(desc, native_shape, base):
    import numpy as np
    from PIL import Image
    path = resolve_descriptor(desc, base)
    h, w = native_shape
    side, left, top = min(h, w), (w - min(h, w)) // 2, (h - min(h, w)) // 2
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (w, h):
            raise ValueError("Raw image dimensions differ from paired catalog")
        image = image.crop((left, top, left + side, top + side))
        if image.size != (1024, 1024):
            image = image.resize((1024, 1024), getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
        array = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(array.transpose(2, 0, 1).astype(np.float32) / 255.0)


def validate_model_assets(model_id, config, base):
    """Always run, including on a resume that requires no new inference."""
    pin = MODEL_PINS[model_id]
    path = resolve_descriptor(config["checkpoint"], base)
    if config["checkpoint"]["sha256"] != pin["sha"]:
        raise ValueError("Not the training-frozen BEST: " + model_id)
    pp = resolve_descriptor(config["training_protocol"], base)
    sp = resolve_descriptor(config["training_summary"], base)
    protocol, summary = read_json(pp), read_json(sp)
    if (protocol.get("source_manifest_sha256") != SOURCE_SHA or protocol.get("test_split_used") is not False or protocol.get("ema_used") is not False or protocol.get("config", {}).get("seed") != 20260817):
        raise ValueError("Training protocol not clean/train-only/frozen seed")
    if summary.get("status") != "complete" or summary.get("training_complete") is not True or summary.get("test_split_used") is not False:
        raise ValueError("Training not confirmed complete")
    if summary.get("protocol_sha256") != sha(pp) or summary.get("checkpoint_best_sha256", summary.get("best_sha256")) != pin["sha"]:
        raise ValueError("Training summary is not bound to the BEST and protocol")
    needed = ["src/unet_baseline.py", pin["source"]]
    if model_id == "unetplusplus":
        needed.append("src/uv_uncertainty_unet.py")
    for source in needed:
        if protocol.get("code_sha256", {}).get(source) != SOURCE_PINS[source]:
            raise ValueError("Training-frozen inference source differs: " + source)
    return path, sha(pp)


def load_model(model_id, config, base, frozen, device):
    """Same strict state_dict/FP32 eval path as the frozen production loaders."""
    import torch
    pin = MODEL_PINS[model_id]
    path, protocol_sha = validate_model_assets(model_id, config, base)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state", {})
    if (checkpoint.get("protocol_sha256") != protocol_sha or checkpoint.get("ema_used") is not False or checkpoint.get("test_split_used") is not False or state.get("epoch") != pin["epoch"] or state.get("selection", {}).get("best_epoch") != pin["epoch"]):
        raise ValueError("Checkpoint provenance/selection mismatch")
    if state.get("phase") not in ("epoch_complete", "complete") or state.get("blocked_reason"):
        raise ValueError("Blocked or partial BEST checkpoint")
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict) or any(not bool(torch.isfinite(v).all()) for v in weights.values()):
        raise ValueError("Invalid frozen weights")
    model = getattr(frozen.load(pin["source"]), pin["cls"])(base_channels=32)
    model.load_state_dict(weights, strict=True)
    if sum(v.numel() for v in model.parameters()) != pin["parameters"]:
        raise ValueError("Architecture parameter count mismatch")
    return model.float().eval().requires_grad_(False).to(device)


def infer(model, rgb, model_id, device):
    import numpy as np
    import torch
    with torch.inference_mode():
        output = model(torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32))
        prediction = output[MODEL_PINS[model_id]["output"]][0].float().cpu().numpy()
        sigma = output["sigma"][0, 0].float().cpu().numpy()
    if prediction.shape != (3, 1024, 1024) or sigma.shape != (1024, 1024) or not np.isfinite(prediction).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Invalid model output")
    return prediction, sigma


def wait_json(path, timeout):
    start = time.monotonic()
    while not Path(path).exists():
        if time.monotonic() - start >= timeout:
            raise TimeoutError("Waiting for separately sealed mask evidence: " + str(path))
        time.sleep(max(0, min(5, timeout - (time.monotonic() - start))))
    return read_json(path)


def runtime_info(device, threads):
    import numpy
    import PIL
    import scipy
    import torch
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device != "cpu":
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("GPU must be explicitly selected and available")
        free, _ = torch.cuda.mem_get_info(device)
        if free < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB free on explicitly selected GPU; no fallback")
    return dict(python=platform.python_version(), numpy=numpy.__version__, Pillow=PIL.__version__, scipy=scipy.__version__, torch=torch.__version__, device=device, cpu_threads=threads, interop_threads=1, precision="FP32_no_autocast_no_TF32", deterministic_algorithms=True, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def verify_record(record, row, index, model_id, contract_sha, mask_receipt_sha):
    verify_seal(record)
    for key, value in (("index", index), ("sample_id", row["sample_id"]), ("subject_id", row["subject_id"]), ("model_id", model_id), ("contract_sha256", contract_sha), ("source_row_sha256", digest(row)), ("mask_receipt_sha256", mask_receipt_sha)):
        if record.get(key) != value:
            raise ValueError("Stored evaluation record binding differs: " + key)
    if set(record.get("regions", {})) != set(REGIONS):
        raise ValueError("Incomplete stored metrics")
    for metrics in record["regions"].values():
        for key in METRICS:
            value = metrics.get(key)
            if value is None:
                if not metrics.get(key + "_reason"):
                    raise ValueError("Undefined result without reason")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Nonfinite stored result")
    return record


def export_jsonl(path, records):
    path = Path(path)
    data = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for r in records)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Existing export differs")
    else:
        temp = path.with_name(path.name + ".partial." + str(os.getpid()))
        with temp.open("xb") as handle:
            handle.write(data)
        os.replace(temp, path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def pid_alive(pid):
    if os.name == "nt":
        # Unlike POSIX, os.kill(pid, 0) can terminate a Windows process.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # No such PID.
                return False
            raise RuntimeError("Cannot safely inspect lock owner process")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError("Cannot query lock owner process")
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path, resume):
    """Only an explicitly resumed, dead-owner lock may be retired."""
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError("Output lock exists")
        try:
            pid = int(path.read_text().strip())
        except (ValueError, OSError):
            raise ValueError("Unverifiable output lock; inspect rather than remove")
        if pid < 1:
            raise ValueError("Invalid owner PID")
        if pid_alive(pid):
            raise ValueError("Output owner still alive; refusing concurrent evaluation")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, (str(os.getpid()) + "\n").encode())
    os.close(descriptor)


def run(args):
    import numpy as np
    config_path, catalog_path = Path(args.config).resolve(), Path(args.catalog).resolve()
    config, catalog = read_json(config_path), read_json(catalog_path)
    if config.get("schema") != "uv_three_region_eval_config_v1" or set(config.get("models", {})) != set(MODEL_IDS):
        raise ValueError("Expected frozen configuration with exactly three models")
    rows = validate_catalog(catalog, catalog_path.parent)
    raw_base = Path(catalog.get("data_root", catalog_path.parent))
    if not raw_base.is_absolute():
        raw_base = catalog_path.parent / raw_base
    frozen = FrozenSources(config["sources"], config_path.parent)
    numeric = frozen.load("tools/uv_architecture_comparison_metrics_v10.py")
    for model_id in MODEL_IDS:
        validate_model_assets(model_id, config["models"][model_id], config_path.parent)
    mask_root, output = Path(args.mask_root).resolve(), Path(args.output_dir).resolve()
    mask_contract_path = mask_root / "mask_contract.json"
    mask_contract = wait_json(mask_contract_path, args.wait_timeout)
    validate_mask_contract(mask_contract, sha(catalog_path), rows)
    mask_contract_sha = sha(mask_contract_path)
    runtime = runtime_info(args.device, args.cpu_threads)
    expected_runtime = config.get("expected_runtime", {})
    if any(runtime.get(k) != v for k, v in expected_runtime.items()):
        raise ValueError("Runtime differs from explicit configuration")
    contract = {"schema": SCHEMA, "config": {"path": str(config_path), "sha256": sha(config_path)}, "catalog": {"path": str(catalog_path), "sha256": sha(catalog_path)}, "mask_contract": {"path": str(mask_contract_path), "sha256": mask_contract_sha}, "runner_sha256": sha(__file__), "runtime": runtime, "source_pins": SOURCE_PINS, "model_pins": MODEL_PINS, "sample_count": len(rows), "preprocess": "same center-square crop then PIL LANCZOS1024 uint8/255; no EXIF transform", "coverage": "native approved P/D/O/F same crop then float32 BOX1024; P+D+O=F atol2e-6", "statistics": "image-equal per-sample metrics; across-sample sample SD(ddof1), not five inference seeds; paired tests separately sealed", "region_ssim": "unmasked full-image RGB-mean 11x11 Gaussian SSIM map, regional coverage at valid centers; windows may cross categories", "sigma": "Gaussian residual scale shared across three UV channels; no calibration fit or posthoc rescaling", "technical_cases": "retain every TEST sample ID; all region metrics null with explicit reason on unavailable geometry; no replacement", "background": "F is frozen valid domain, not whole rectangular image and not validated semantic skin", "metric_helpers": ["validate_image", "_validate_sigma", "_weighted_mean", "_psnr", "ssim_map_rgb"]}
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "protocol.json"
    if contract_path.exists() and not args.resume:
        raise ValueError("Existing output requires --resume")
    write_new(contract_path, contract)
    contract_sha = sha(contract_path)
    lock = output / ".run.lock"
    acquire_lock(lock, args.resume)
    try:
        selected = MODEL_IDS if args.model == "all" else (args.model,)
        models = {}
        catalog_sha = sha(catalog_path)
        # Sample-outer execution confirms all three real outputs on the first
        # available sample; only one model is forwarded at a time. CPU weights
        # may remain resident. GPU mode keeps only one model resident.
        for index, row in enumerate(rows):
            receipt_path = mask_root / "cases" / ("case_%06d" % index) / "receipt.json"
            receipt = validate_mask_receipt(wait_json(receipt_path, args.wait_timeout), row, index, catalog_sha, mask_contract_sha, raw_base)
            if receipt["status"] in TECHNICAL:
                validate_technical_evidence(receipt, receipt_path)
            receipt_sha = sha(receipt_path)
            weights = rgb = truth = None
            for model_id in selected:
                target = output / "records" / model_id / ("%06d.json" % index)
                if target.exists():
                    if not args.resume:
                        raise ValueError("Existing record requires --resume")
                    # Resume verifies inputs, rather than treating an old result as a new measurement.
                    for key in ("rgb", "uv"):
                        resolve_descriptor(row[key], raw_base)
                    if receipt["status"] == "complete":
                        resolve_descriptor(receipt["npz"], receipt_path.parent)
                    record = verify_record(read_json(target), row, index, model_id, contract_sha, receipt_sha)
                else:
                    if receipt["status"] in TECHNICAL:
                        regions, status = unavailable_regions("mask_" + receipt["status"]), "unavailable_geometry"
                    else:
                        if weights is None:
                            coverage_path = resolve_descriptor(receipt["npz"], receipt_path.parent)
                            with np.load(coverage_path, allow_pickle=False) as bundle:
                                weights = {key: bundle[key] for key in bundle.files}
                            validate_weights(weights)
                            validate_projection_audit(receipt, receipt_path, weights)
                            rgb = prepare_image(row["rgb"], row["native_shape"], raw_base)
                            truth = prepare_image(row["uv"], row["native_shape"], raw_base)
                        if model_id not in models:
                            if args.device.startswith("cuda:"):
                                models.clear()
                                gc.collect()
                                import torch
                                torch.cuda.empty_cache()
                            models[model_id] = load_model(model_id, config["models"][model_id], config_path.parent, frozen, args.device)
                        prediction, sigma = infer(models[model_id], rgb, model_id, args.device)
                        regions, status = measure(truth, prediction, sigma, weights, numeric), "complete"
                    record = seal({"schema": SCHEMA + "_sample", "status": status, "index": index, "sample_id": row["sample_id"], "subject_id": row["subject_id"], "model_id": model_id, "checkpoint_sha256": MODEL_PINS[model_id]["sha"], "contract_sha256": contract_sha, "source_row_sha256": digest(row), "mask_receipt_sha256": receipt_sha, "regions": regions})
                    verify_record(record, row, index, model_id, contract_sha, receipt_sha)
                    write_new(target, record)
                if index % 10 == 0 or index + 1 == len(rows):
                    event = seal({"model_id": model_id, "completed_count": index + 1, "sample_count": len(rows), "contract_sha256": contract_sha, "last_record_sha256": record["record_sha256"]})
                    write_new(output / "progress" / model_id / ("%06d.json" % index), event)
                    print(json.dumps({"model": model_id, "completed": index + 1, "total": len(rows)}), flush=True)
            del weights, rgb, truth
        models.clear()
        gc.collect()
        for model_id in selected:
            records = []
            for index, row in enumerate(rows):
                receipt_path = mask_root / "cases" / ("case_%06d" % index) / "receipt.json"
                records.append(verify_record(read_json(output / "records" / model_id / ("%06d.json" % index)), row, index, model_id, contract_sha, sha(receipt_path)))
            exported = export_jsonl(output / (model_id + ".jsonl"), records)
            write_new(output / (model_id + "_completion.json"), seal({"model_id": model_id, "count": len(rows), "records": exported, "checkpoint_sha256": MODEL_PINS[model_id]["sha"], "contract_sha256": contract_sha, "status": "complete"}))
            del records
            gc.collect()
            if args.device.startswith("cuda:"):
                import torch
                torch.cuda.empty_cache()
        if all((output / (mid + "_completion.json")).exists() for mid in MODEL_IDS):
            models = []
            for mid in MODEL_IDS:
                finished = verify_seal(read_json(output / (mid + "_completion.json")))
                if finished["contract_sha256"] != contract_sha or finished["count"] != len(rows):
                    raise ValueError("Incompatible model completions")
                resolve_descriptor(finished["records"], output)
                models.append({"model_id": mid, "records": finished["records"], "checkpoint_sha256": finished["checkpoint_sha256"]})
            order_sha = hashlib.sha256(json.dumps([r["sample_id"] for r in rows], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            manifest = {"schema": "uv_three_region_statistics_input_v1", "models": models, "model_order": list(MODEL_IDS), "expected_sample_count": len(rows), "expected_sample_order_sha256": order_sha, "statistical_unit": {"name": "sample_group", "independent_subject_identity_verified": False, "note": "One folder-derived unique subject_id per TEST group; not independently verified biological identity"}, "evaluation_protocol": {"path": str(contract_path), "sha256": contract_sha}}
            write_new(output / "stats_manifest.json", manifest)
            write_new(output / "completion.json", seal({"schema": SCHEMA + "_completion", "status": "complete", "sample_count": len(rows), "model_count": 3, "statistics_manifest": {"path": str(output / "stats_manifest.json"), "sha256": sha(output / "stats_manifest.json")}, "contract_sha256": contract_sha}))
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", help="Explicit cpu or cuda:N; no auto GPU selection")
    parser.add_argument("--model", choices=("all", *MODEL_IDS), default="all")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.wait_timeout <= 0:
        parser.error("cpu-threads/wait-timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
