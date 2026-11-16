"""Original TEST UV data, fractional F, anatomy seeds and five frozen UV predictors.

The normal path consumes the audited 2026-10-29 staged output. It does not
load Torch. Missing stored arrays can be recovered from the original sources
and BEST checkpoints. Verification errors propagate, never become exclusions.
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

EXPERIMENTS = Path("/study/siiba/experiments")
STAGE = EXPERIMENTS / "signal_objects_3cd_20261029_v1"
ORIGINAL = EXPERIMENTS / "three_region_test_eval_20261025_v1"
PROTOCOL_SHA = "d7fd5ad88b042b3bd551d92cd3efd2a67cf52b376e2c7b8e4c9b3716401f93da"
AUDIT_SHA = "01f9e286a2cbc868383e34dc4753d9479549442aee855a7544d094d6be9212c5"
OLD_SHA = "874bffe20046d7b7e368651906f9d22a4403333d77dbdfc033f2457214d51ff4"
CONFIG_SHA = "d4d7ad6cd3cf6aaf96006723dd70174ead25164bd8ee5f937a2f750663bf4bf9"
MASK_CONTRACT_SHA = "0ab265ad6470a6072e947fb2a69404b125c5abd30305dad386a241712daff067"
REGIONAL_SHA = "ff37b3c5cdc86af7e2f781107cb46c7724c04c50da66b2b26e44ed2ff92eeffa"
CHECKPOINTS = {
    "unet_single": "6f9e5c2a8745a3f67794f993961bfc6612421f04be178f41288f704c53d3cbfb",
    "unet_multi": "204dbffedc15fa2cc6053ce6b6d1aa1464f9978c103c5814f84692d69c80844e",
    "unetplusplus": "81d7dc5d40f2b36d2ac64a10ca6a5a443af3854d70fc88c75027ca5ecd2dec5f",
    "swin": "5c9da944dd83f9478fbdcb96e75a4108f1fc2ae35243786b0f05d139fc6aadad",
    "ldm": "e294864b955abf20461dd51c66cd08d8b12508aa8746b2e718e6e11556b76dbf",
}
REGIONS = ("forehead", "image_left_cheek", "image_right_cheek", "nose", "chin")
COUNT = 21899


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verified(path, expected):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected, "File SHA differs: " + str(path))
    return payload


def bound(desc):
    return verified(desc["path"], desc["sha256"])


@lru_cache(maxsize=1)
def cohort():
    protocol = json.loads(verified(STAGE / "protocol.json", PROTOCOL_SHA))
    catalog = json.loads(bound(protocol["catalog"]))
    selection = json.loads(bound(protocol["selection"]))
    ids = [row["sample_id"] for row in catalog["rows"]]
    require(ids == selection["all_group_ids"] and len(set(ids)) == len(ids) == COUNT
            and catalog["split"] == "test" and all(row["split"] == "test" for row in catalog["rows"]),
            "Frozen TEST catalog differs")
    missing = {item["index"]: item["reason"] for item in selection["missing"]}
    valid = set(selection["geometry_valid_indices"])
    require(len(valid) == 21422 and len(missing) == 477 and not valid.intersection(missing)
            and valid.union(missing) == set(range(COUNT))
            and all(item["group_id"] == ids[item["index"]] and
                    item["reason"] in {"no_unique_face", "invalid_geometry"} for item in selection["missing"]),
            "Frozen geometry inventory differs")
    return protocol, catalog, missing


@lru_cache(maxsize=1)
def stage_hashes():
    path = STAGE / "report/independent_audit_v1/consumed_files_sha256.jsonl"
    hashes = {}
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            if b'/ready.json"' in line:
                item = json.loads(line)
                file = Path(item["path"])
                if file.parent.parent == STAGE / "staged" and file.name == "ready.json":
                    index = int(file.parent.name)
                    require(file.parent.name == "%06d" % index and index not in hashes,
                            "Duplicate or noncanonical audited stage")
                    hashes[index] = item["sha256"]
    require(hasher.hexdigest() == AUDIT_SHA and set(hashes) == set(range(COUNT)),
            "Historical independent stage audit differs")
    return hashes


def checked_image(array, what):
    require(array.shape == (3, 1024, 1024) and array.dtype == np.float32 and
            np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1, "Invalid " + what)
    return array


def load_module(name, desc):
    bound(desc)
    spec = importlib.util.spec_from_file_location(name, desc["path"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def legacy():
    return load_module("_table3b_original_uv_runner", {
        "path": str(ORIGINAL / "source/tools/uv_three_region_eval_20261025.py"), "sha256": OLD_SHA})


@lru_cache(maxsize=1)
def original_geometry():
    protocol = json.loads(verified(EXPERIMENTS / "regional_signal_agreement_20261026_v1/protocol.json",
                                  REGIONAL_SHA))
    anatomy = load_module("_table3b_original_uv_anatomy", protocol["sources"]["anatomy_regions.py"])
    vertices, faces = anatomy.load_canonical(protocol["sources"]["canonical_face_model.obj"]["path"])
    inventory = json.loads(bound(cohort()[0]["mask_inventory"]))
    return anatomy, vertices, faces, inventory


def recover_geometry(index, row):
    """Same F and frozen anatomy.build_weights call used by runner.roi_weights."""
    protocol, catalog, _ = cohort()
    anatomy, vertices, faces, inventory = original_geometry()
    entry = inventory[index]
    require(entry["index"] == index, "Mask inventory index differs")
    descriptor = entry["receipt"]
    receipt = legacy().validate_mask_receipt(json.loads(bound(descriptor)), row, index,
                protocol["catalog"]["sha256"], MASK_CONTRACT_SHA, catalog["data_root"])
    require(receipt["status"] == "complete", "Original mask unexpectedly incomplete")
    path = Path(descriptor["path"])
    with np.load(legacy().resolve_descriptor(receipt["npz"], path.parent), allow_pickle=False) as z:
        weights = {key: z[key] for key in z.files}
    legacy().validate_weights(weights)
    legacy().validate_projection_audit(receipt, path, weights)
    lm = json.loads(bound(receipt["evidence"]["geometry/landmarks.json"]))
    require(lm["source"]["native_shape_hw"] == row["native_shape"] and
            lm["uv_sha256"] == row["uv"]["sha256"], "Archived landmark binding differs")
    rois, _ = anatomy.build_weights(lm["native_xy"], row["native_shape"], weights["F"], vertices, faces)
    return weights["F"], rois


@lru_cache(maxsize=1)
def stage(index):
    row = cohort()[1]["rows"][index]
    path = STAGE / ("staged/%06d/ready.json" % index)
    if not path.is_file():
        return None
    ready = json.loads(verified(path, stage_hashes()[index]))
    require((ready["index"], ready["group_id"], ready["row"], ready["protocol_sha256"]) ==
            (index, row["sample_id"], row, PROTOCOL_SHA), "Staged row identity differs")
    return ready


@lru_cache(maxsize=1)
def geometry(index):
    ready = stage(index)
    if ready is None or not Path(ready["geometry"]["path"]).is_file():
        return recover_geometry(index, cohort()[1]["rows"][index])
    with np.load(io.BytesIO(bound(ready["geometry"])), allow_pickle=False) as archive:
        require(set(archive.files) == set(REGIONS) | {"F_reference", "F_coverage"}, "Geometry fields differ")
        f = archive["F_coverage"]
        require(np.array_equal(f, archive["F_reference"]), "F coverage and reference differ")
        return f, {name: archive[name] for name in REGIONS}


def actual_image(index, kind):
    catalog = cohort()[1]
    row = catalog["rows"][index]
    return legacy().prepare_image(row[kind], row["native_shape"], catalog["data_root"])


@lru_cache(maxsize=1)
def truth_array(index):
    ready = stage(index)
    if ready is not None and Path(ready["arrays"]["truth"]["path"]).is_file():
        truth = np.load(io.BytesIO(bound(ready["arrays"]["truth"])), allow_pickle=False)
    else:
        truth = actual_image(index, "uv")
        if ready is not None:
            require(hashlib.sha256(truth.tobytes(order="C")).hexdigest() == ready["inputs"]["truth"]["sha256"],
                    "Source UV preprocessing differs from archived truth")
    return checked_image(truth, "UV truth")


@lru_cache(maxsize=1)
def frozen_model(model, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old = legacy()
    old.runtime_info(device, 2)
    if model in ("swin", "ldm"):
        manifest = json.loads(bound(cohort()[0]["original_manifest"]))
        loader = load_module("_table3b_original_uv_loader", manifest["sources"]["loader_adapter_v1"])
        bridge = load_module("_table3b_original_uv_bridge", manifest["sources"]["measurement_bridge_v1"])
        files = manifest["training_bundles"][model]
        require(files["best_checkpoint"]["sha256"] == CHECKPOINTS[model], "Frozen checkpoint differs")
        bundle = loader.TrainingBundle(**{key: loader.FrozenFile(**value) for key, value in files.items()})
        return loader.load_adapter(model, bundle, device), bridge.TorchTensorIO(device)
    cfg = json.loads(verified(ORIGINAL / "eval_config.json", CONFIG_SHA))
    frozen = old.FrozenSources(cfg["sources"], ORIGINAL)
    return old.load_model(model, cfg["models"][model], ORIGINAL, frozen, device), None


def prediction(index, model, device):
    ready = stage(index)
    if ready is not None:
        output = ready["outputs"][model]
        require(output["checkpoint_sha256"] == CHECKPOINTS[model] and output["forward_count"] == 1,
                "Staged prediction provenance differs")
        if Path(output["prediction"]["path"]).is_file():
            array = np.load(io.BytesIO(bound(output["prediction"])), allow_pickle=False)
            return checked_image(array, "staged UV prediction")
    rgb = actual_image(index, "rgb")
    if ready is not None:
        require(hashlib.sha256(rgb.tobytes(order="C")).hexdigest() == ready["inputs"]["rgb"]["sha256"],
                "Source RGB preprocessing differs from archived input")
    net, converter = frozen_model(model, device)
    if converter is None:
        array, _ = legacy().infer(net, rgb, model, device)
    else:
        import torch
        with torch.inference_mode():
            result = net.predict(converter.as_input(rgb), cohort()[1]["rows"][index])
            array, _ = converter.prediction_pair(result)
    return checked_image(array, "frozen UV prediction")


def iter_cases(start, stop, step, model, target, device):
    """Yield indexed original TEST groups. Stop is exclusive."""
    require(model in CHECKPOINTS and target.lower() == "uv", "Unknown UV model or target")
    require(step > 0 and 0 <= start <= stop <= COUNT, "Invalid TEST range")
    _, catalog, missing = cohort()
    for index in range(start, stop, step):
        base = dict(index=index, group_id=catalog["rows"][index]["sample_id"], model=model, target=target)
        if index in missing:
            yield dict(base, missing_reason=missing[index])
        else:
            f, rois = geometry(index)
            yield dict(base, f_weights=f, old_weights=rois, truth=truth_array(index),
                       pred=prediction(index, model, str(device)))
