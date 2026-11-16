#!/usr/bin/env python3
"""Frozen native-pixel six-metric MI; new artifacts only, no detector/training.

Input JSON schema: mi_atlas_six_metrics_input_v1, status=frozen, cohort_id,
expected_sample_count, frozen_provenance with company_module_sha256,
mask_pipeline_sha256 and mask_config_sha256; rows in frozen order. Each row:
sample_id, subject_id, native_shape=[H,W], and rgb/uv/plv/plh/P/D/O/F descriptors
{path,sha256}. Paths are relative to the manifest, or absolute. Arrays may be
image files (RGB color / single-channel masks) or .npy; all must be uint8.
Binary masks use 0/1 or 0/255; P,D,O partition F. EVERY domain in EVERY row
must have >=1000 native pixels. Failure stops preparation; no replacement,
padding, point duplication, quota reduction, or mask recomputation is allowed.

Three seeds each sample 1000 unique pixels independently per person/domain.
100/500 are prefixes. The three full-face pairs share F coordinates. Features
contain ONLY those sampled values/indices. The 54 pooled estimates use native
3D/3D KSG-1 k=5 and fixed identity-keyed half-LSB dequantization. Summaries are
mean and sample SD (ddof=1) across three sampling seeds, NOT confidence bounds.

CLI: --manifest FILE --output-dir NEW_DIR --phase prepare|run|summarize|all
CPU only, estimator workers <=4. Run/summarize validate exact feature coverage
and bound hashes; run also rechecks all original image/mask hashes. Source data
and company code are never modified. Do not remove a stale lock without first
checking that its recorded process is no longer running.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mi_knn_estimator_v1 import ksg_mi

INPUT_SCHEMA = "mi_atlas_six_metrics_input_v1"
VERSION = "mi_atlas_six_metrics_20261025_v1"
SEEDS = (20261025, 20261026, 20261027)
SIZES = (100, 500, 1000)
NOISE_SEED, K, MAX_POINTS = 20260817, 5, 1000
MODS = ("rgb", "uv", "plv", "plh")
DOMAINS = ("F", "P", "D", "O")
METRICS = {"full_rgb_plv": ("F", "plv"), "full_rgb_plh": ("F", "plh"),
           "full_rgb_uv": ("F", "uv"), "P_rgb_uv": ("P", "uv"),
           "D_rgb_uv": ("D", "uv"), "O_rgb_uv": ("O", "uv")}
ESTIMATOR_PATH = ROOT / "tools/mi_knn_estimator_v1.py"
ESTIMATOR_SHA = "bb6f7fbf704a3baff582a712fefbaa6bb6d13c9d962ff4fa108347db05a19b0d"


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
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    with Path(path).open(encoding="utf8") as stream:
        return json.load(stream, parse_constant=invalid)


def seal(value):
    return dict(value, record_sha256=digest(value))


def check_seal(value):
    if value.get("record_sha256") != digest({k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("Artifact record digest mismatch")
    return value


def publish(path, payload):
    """Atomically publish a new file, refusing to overwrite different bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Refusing to replace frozen artifact: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("Concurrent differing artifact: " + str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(path, value):
    publish(path, encoded(value))


def save_npz(path, arrays):
    path = Path(path)
    if path.exists():
        existing = load_npz(path)
        if set(existing) != set(arrays) or any(existing[k].dtype != arrays[k].dtype or
                not np.array_equal(existing[k], arrays[k]) for k in arrays):
            raise ValueError("Existing sampled features differ: " + str(path))
        return
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    publish(path, stream.getvalue())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def progress(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs), ensure_ascii=False, allow_nan=False), flush=True)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def descriptor_path(base, descriptor):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str) or not descriptor["path"] or not valid_sha(descriptor.get("sha256")):
        raise ValueError("Each source requires path and lowercase SHA256")
    path = Path(descriptor["path"])
    return (path if path.is_absolute() else base / path).resolve()


def load_contract(manifest_path, output=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get("status") != "frozen":
        raise ValueError("Require the frozen input schema")
    if not isinstance(manifest.get("cohort_id"), str) or not manifest["cohort_id"]:
        raise ValueError("Missing cohort_id")
    count = positive_int(manifest.get("expected_sample_count"), "expected_sample_count")
    provenance = manifest.get("frozen_provenance", {})
    for key in ("company_module_sha256", "mask_pipeline_sha256", "mask_config_sha256"):
        if not valid_sha(provenance.get(key)):
            raise ValueError("Missing frozen provenance: " + key)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Frozen cohort row count mismatch")
    samples, subjects = set(), set()
    source_paths = {manifest_path}
    for row in rows:
        for key, used in (("sample_id", samples), ("subject_id", subjects)):
            value = row.get(key)
            if not isinstance(value, str) or not value or value in used:
                raise ValueError("Duplicate/missing " + key + "; no silent deduplication")
            used.add(value)
        shape = row.get("native_shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("native_shape must be integer [height,width]")
        for name in MODS + DOMAINS:
            path = descriptor_path(manifest_path.parent, row.get(name))
            if not path.is_file():
                raise ValueError("Missing source: " + str(path))
            source_paths.add(path)
    if output is not None:
        out = Path(output).resolve()
        if any(p == out or out in p.parents for p in source_paths):
            raise ValueError("Output must not contain any scientific input")
    if sha(ESTIMATOR_PATH) != ESTIMATOR_SHA:
        raise ValueError("Frozen KSG/dequantization dependency changed")
    protocol = dict(schema=VERSION, input_manifest_sha256=sha(manifest_path),
        input_contract_sha256=digest(manifest), cohort_id=manifest["cohort_id"],
        expected_sample_count=count, sample_order=[r["sample_id"] for r in rows],
        subject_order=[r["subject_id"] for r in rows], frozen_provenance=provenance,
        seeds=list(SEEDS), sizes=list(SIZES), noise_seed=NOISE_SEED, k=K,
        modality_order=list(MODS), domain_order=list(DOMAINS), metrics=list(METRICS),
        sampling="1000 unique pixels per sample/domain/seed; 100 and 500 are ordered prefixes",
        sampling_rng="NumPy PCG64; seed=first128bits SHA256(JSON[version,seed,sample_id,domain])",
        dequantization="(uint8 + U[-0.5,0.5])/255, SHA keyed by noise seed/sample_id/NATIVE:width:linear_id/modality/channel",
        geometry="Unmodified full-native uint8 RGB values at identical paired coordinates; no resize or normalization",
        regional_weighting="Equal sampled pixel count per sample; pool samples, not mean per-image MI",
        shortage_policy="Fail entire preparation; no replacement, padding or quota reduction",
        summary="Three sampling repeats: mean and sample SD ddof=1; not CI or independent subjects",
        negative_estimates_clipped=False, spatial_points_are_independent_people=False,
        identities="Uniqueness checked on provided subject_id; biological identity is adapter responsibility",
        runner_sha256=sha(__file__), estimator_sha256=ESTIMATOR_SHA,
        software=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__))
    return manifest_path, manifest, protocol


def read_bound_array(base, descriptor, is_mask=False):
    path = descriptor_path(base, descriptor)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError("Frozen source SHA256 changed: " + str(path))
    if path.suffix.lower() == ".npy":
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    else:
        with Image.open(io.BytesIO(payload)) as im:
            if is_mask:
                if im.mode != "L":
                    raise ValueError("Mask image must be single-channel uint8 L: " + str(path))
                array = np.asarray(im, dtype=np.uint8).copy()
            else:
                if im.mode != "RGB":
                    raise ValueError("Color image must be RGB, no implicit channel conversion: " + str(path))
                array = np.asarray(im, dtype=np.uint8).copy()
    if array.dtype != np.uint8:
        raise ValueError("Source must be uint8: " + str(path))
    return array


def validate_masks(mask_arrays, shape):
    masks = {}
    counts = {}
    for domain in DOMAINS:
        raw = mask_arrays[domain]
        if raw.dtype != np.uint8 or raw.shape != tuple(shape):
            raise ValueError("Mask dtype/geometry mismatch: " + domain)
        values = np.unique(raw)
        if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
            raise ValueError("Mask is not binary uint8: " + domain)
        masks[domain] = raw != 0
        counts[domain] = int(np.count_nonzero(masks[domain]))
    occupancy = masks["P"].astype(np.uint8) + masks["D"] + masks["O"]
    if np.any(occupancy > 1) or not np.array_equal(occupancy != 0, masks["F"]):
        raise ValueError("P/D/O must be disjoint and their union exactly F")
    if any(counts[d] < MAX_POINTS for d in DOMAINS):
        raise ValueError("Insufficient native pixels; no padding/replacement: " + json.dumps(counts))
    return masks, counts


def sample_indices(masks, sample_id):
    """Return [seed, F/P/D/O, rank] native row-major linear indices."""
    result = np.empty((len(SEEDS), len(DOMAINS), MAX_POINTS), np.uint64)
    for domain_index, domain in enumerate(DOMAINS):
        candidates = np.flatnonzero(masks[domain].ravel())
        if len(candidates) < MAX_POINTS:
            raise ValueError("Insufficient candidates: " + domain)
        for seed_index, seed in enumerate(SEEDS):
            payload = json.dumps([VERSION, seed, sample_id, domain], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf8")
            integer = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
            generator = np.random.Generator(np.random.PCG64(integer))
            ranks = generator.choice(len(candidates), MAX_POINTS, replace=False)
            result[seed_index, domain_index] = candidates[ranks]
    return result


def feature_paths(out, index):
    stem = "sample_%06d" % index
    return out / "features" / (stem + ".npz"), out / "features" / (stem + ".json")


def validate_features(arrays, row):
    expected = {"linear_indices", "full_values", "region_values", "native_shape", "seeds"}
    if set(arrays) != expected:
        raise ValueError("Unexpected sampled feature fields")
    indices = arrays["linear_indices"]
    if indices.dtype != np.uint64 or indices.shape != (3, 4, MAX_POINTS):
        raise ValueError("Sampled coordinate layout differs")
    if not np.array_equal(arrays["native_shape"], row["native_shape"]) or not np.array_equal(arrays["seeds"], SEEDS):
        raise ValueError("Sampled geometry/seeds changed")
    if np.any(indices >= int(row["native_shape"][0]) * int(row["native_shape"][1])):
        raise ValueError("Sampled point outside image")
    for group in indices.reshape(-1, MAX_POINTS):
        if len(np.unique(group)) != MAX_POINTS:
            raise ValueError("Duplicate sampled point")
    for seed_index in range(3):
        regions = indices[seed_index, 1:].ravel()
        if len(np.unique(regions)) != 3 * MAX_POINTS:
            raise ValueError("P/D/O coordinate collision")
    for name, shape in (("full_values", (3, 4, MAX_POINTS, 3)),
                        ("region_values", (3, 3, 2, MAX_POINTS, 3))):
        if arrays[name].dtype != np.uint8 or arrays[name].shape != shape:
            raise ValueError("Sampled intensity layout differs: " + name)
    return arrays


def verify_sources(base, row):
    for name in MODS + DOMAINS:
        if sha(descriptor_path(base, row[name])) != row[name]["sha256"]:
            raise ValueError("Frozen source SHA256 changed: " + row["sample_id"] + "/" + name)


def prepare(manifest_path, out):
    manifest_path, manifest, protocol = load_contract(manifest_path, out)
    out = Path(out).resolve()
    save_json(out / "protocol.json", protocol)
    binding = digest(protocol)
    inventory = []
    for index, row in enumerate(manifest["rows"]):
        feature_path, metadata_path = feature_paths(out, index)
        if metadata_path.exists():
            metadata = check_seal(read_json(metadata_path))
            if metadata.get("protocol_sha256") != binding or metadata.get("source_row_sha256") != digest(row):
                raise ValueError("Resumed feature binding differs")
            verify_sources(manifest_path.parent, row)
            if sha(feature_path) != metadata["feature_sha256"]:
                raise ValueError("Resumed feature SHA256 differs")
            validate_features(load_npz(feature_path), row)
        else:
            raw_masks = {d: read_bound_array(manifest_path.parent, row[d], True) for d in DOMAINS}
            masks, counts = validate_masks(raw_masks, row["native_shape"])
            indices = sample_indices(masks, row["sample_id"])
            del raw_masks, masks
            full = np.empty((3, 4, MAX_POINTS, 3), np.uint8)
            regional = np.empty((3, 3, 2, MAX_POINTS, 3), np.uint8)
            for mod_index, modality in enumerate(MODS):
                raw = read_bound_array(manifest_path.parent, row[modality])
                if raw.shape != tuple(row["native_shape"]) + (3,):
                    raise ValueError("Paired source shape differs: " + modality)
                flat = raw.reshape(-1, 3)
                full[:, mod_index] = flat[indices[:, 0]]
                if mod_index < 2:
                    regional[:, :, mod_index] = flat[indices[:, 1:]]
                del raw, flat
            arrays = dict(linear_indices=indices, full_values=full, region_values=regional,
                          native_shape=np.asarray(row["native_shape"], np.int64),
                          seeds=np.asarray(SEEDS, np.int64))
            validate_features(arrays, row)
            save_npz(feature_path, arrays)
            metadata = seal(dict(schema=VERSION, status="complete", index=index,
                sample_id=row["sample_id"], subject_id=row["subject_id"],
                protocol_sha256=binding, source_row_sha256=digest(row),
                source_sha256={name: row[name]["sha256"] for name in MODS + DOMAINS},
                mask_pixels=counts, feature_sha256=sha(feature_path),
                linear_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                partition_verified=True, source_hashes_verified=True,
                sampled_points_per_domain_per_seed=MAX_POINTS, full_pixel_export=False))
            save_json(metadata_path, metadata)
            del arrays, full, regional, indices
        inventory.append(dict(index=index, sample_id=row["sample_id"],
            subject_id=row["subject_id"], feature_path=feature_path.relative_to(out).as_posix(),
            feature_sha256=metadata["feature_sha256"], metadata_sha256=sha(metadata_path)))
        progress("prepared", completed=index + 1, total=len(manifest["rows"]), sample_id=row["sample_id"])
    summary = seal(dict(schema=VERSION, status="complete", protocol_sha256=binding,
        expected_sample_count=len(manifest["rows"]), sample_order=protocol["sample_order"],
        features=inventory, all_samples_all_domains_at_least_1000=True,
        no_population_replacement=True, no_duplicate_padding=True))
    save_json(out / "preparation.json", summary)
    return summary


def verify_prepared(manifest_path, out, recheck_sources=False):
    manifest_path, manifest, protocol = load_contract(manifest_path, out)
    out = Path(out).resolve()
    if read_json(out / "protocol.json") != protocol:
        raise ValueError("Frozen runtime/protocol changed; do not mix outputs")
    preparation = check_seal(read_json(out / "preparation.json"))
    binding = digest(protocol)
    records = preparation.get("features", [])
    if preparation.get("protocol_sha256") != binding or preparation.get("status") != "complete" or preparation.get("sample_order") != protocol["sample_order"] or len(records) != len(manifest["rows"]):
        raise ValueError("Incomplete or mismatched frozen sample coverage")
    for index, (row, record) in enumerate(zip(manifest["rows"], records)):
        feature_path, metadata_path = feature_paths(out, index)
        if record.get("index") != index or record.get("sample_id") != row["sample_id"] or record.get("subject_id") != row["subject_id"] or record.get("feature_path") != feature_path.relative_to(out).as_posix():
            raise ValueError("Feature cohort order/identity mismatch")
        if sha(feature_path) != record["feature_sha256"] or sha(metadata_path) != record["metadata_sha256"]:
            raise ValueError("Frozen feature/metadata SHA256 changed")
        metadata = check_seal(read_json(metadata_path))
        if metadata.get("source_row_sha256") != digest(row) or metadata.get("protocol_sha256") != binding or metadata.get("feature_sha256") != record["feature_sha256"] or metadata.get("partition_verified") is not True or any(metadata.get("mask_pixels", {}).get(d, 0) < MAX_POINTS for d in DOMAINS):
            raise ValueError("Frozen feature provenance/coverage mismatch")
        validate_features(load_npz(feature_path), row)
        if recheck_sources:
            verify_sources(manifest_path.parent, row)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            progress("verified", completed=index + 1, total=len(records), sources_rechecked=recheck_sources)
    if len(list((out / "features").glob("sample_*.npz"))) != len(records) or len(list((out / "features").glob("sample_*.json"))) != len(records):
        raise ValueError("Unexpected extra sampled feature artifacts")
    return manifest, protocol, preparation


def dequantize_native(raw, indices, sample_id, width, modality):
    """Fast exact realization of old dequantize_u8 with NATIVE point IDs.

    Noise identity excludes region, seed of sampling, and requested sample size.
    Thus a reused physical pixel gets exactly the same noise in every estimate.
    """
    if raw.dtype != np.uint8 or raw.shape != (len(indices), 3) or indices.dtype != np.uint64 or len(np.unique(indices)) != len(indices):
        raise ValueError("Invalid/duplicated native dequantization observations")
    positive_int(width, "width")
    if modality not in MODS:
        raise ValueError("Unknown modality")
    prefix = json.dumps(["mi_u8_dequantization_v1", NOISE_SEED, sample_id],
                        ensure_ascii=False, separators=(",", ":")).encode("utf8")[:-1] + b","
    native_prefix = ('"NATIVE:%d:' % width).encode("ascii")
    suffixes = [b"," + json.dumps([modality, c], separators=(",", ":")).encode("ascii")[1:] for c in range(3)]
    output = raw.astype(np.float64)
    denominator = float(1 << 53)
    for row, linear in enumerate(indices):
        base = prefix + native_prefix + str(int(linear)).encode("ascii") + b'"'
        for channel, suffix in enumerate(suffixes):
            integer = int.from_bytes(hashlib.sha256(base + suffix).digest()[:8], "big") >> 11
            output[row, channel] += integer / denominator - .5
    return output / 255.


def pooled_domain(out, manifest, seed_index, domain):
    domain_index = DOMAINS.index(domain)
    modalities = MODS if domain == "F" else MODS[:2]
    count = len(manifest["rows"])
    result = {modality: np.empty((count, MAX_POINTS, 3), np.float64) for modality in modalities}
    for index, row in enumerate(manifest["rows"]):
        feature_path, _ = feature_paths(Path(out), index)
        arrays = validate_features(load_npz(feature_path), row)
        indices = arrays["linear_indices"][seed_index, domain_index]
        values = arrays["full_values"][seed_index] if domain == "F" else arrays["region_values"][seed_index, domain_index - 1]
        for mod_index, modality in enumerate(modalities):
            result[modality][index] = dequantize_native(values[mod_index], indices,
                row["sample_id"], row["native_shape"][1], modality)
        if (index + 1) % 100 == 0 or index + 1 == count:
            progress("dequantized", domain=domain, seed=SEEDS[seed_index], completed=index + 1, total=count)
    return result


def result_path(out, metric, size, seed):
    return Path(out) / "results" / ("%s_n%d_seed%d.json" % (metric, size, seed))


def result_binding(protocol, preparation, metric, size, seed):
    return dict(schema=VERSION, protocol_sha256=digest(protocol),
        preparation_sha256=preparation["record_sha256"], metric=metric,
        domain=METRICS[metric][0], target=METRICS[metric][1], points_per_sample=size,
        sampling_seed=seed, sample_count=protocol["expected_sample_count"],
        observations=protocol["expected_sample_count"] * size, k=K, dx=3, dy=3)


def existing_result(path, expected):
    if not Path(path).exists():
        return None
    record = check_seal(read_json(path))
    if any(record.get(key) != value for key, value in expected.items()) or record.get("status") != "complete":
        raise ValueError("Cached MI result binding differs: " + str(path))
    result = record.get("result", {})
    if result.get("n") != expected["observations"] or result.get("k") != K or result.get("dx") != 3 or result.get("dy") != 3 or result.get("estimator") != "KSG_1" or result.get("negative_estimates_clipped") is not False or not isinstance(result.get("mi_bits"), (int, float)) or not np.isfinite(result["mi_bits"]):
        raise ValueError("Invalid cached numerical result")
    return record


def run_mi(manifest_path, out, workers=4):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Estimator CPU workers must be 1..4")
    manifest, protocol, preparation = verify_prepared(manifest_path, out, recheck_sources=True)
    completed = 0
    for seed_index, seed in enumerate(SEEDS):
        for domain in DOMAINS:
            metrics = [m for m, pair in METRICS.items() if pair[0] == domain]
            tasks = []
            for size in SIZES:
                for metric in metrics:
                    binding = result_binding(protocol, preparation, metric, size, seed)
                    path = result_path(out, metric, size, seed)
                    if existing_result(path, binding) is None:
                        tasks.append((metric, size, path, binding))
                    else:
                        completed += 1
            if not tasks:
                continue
            values = pooled_domain(out, manifest, seed_index, domain)
            for metric, size, path, binding in tasks:
                # 逐组取样再合并
                x = np.ascontiguousarray(values["rgb"][:, :size, :].reshape(-1, 3))
                y = np.ascontiguousarray(values[METRICS[metric][1]][:, :size, :].reshape(-1, 3))
                if len(x) != binding["observations"]:
                    raise RuntimeError("Lost fixed-cohort observations")
                progress("mi_started", metric=metric, points_per_sample=size, seed=seed, observations=len(x))
                started = time.monotonic()
                result = ksg_mi(x, y, k=K, workers=workers)
                record = seal(dict(binding, status="complete", result=result,
                    input_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    input_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                    elapsed_seconds=time.monotonic() - started))
                save_json(path, record)
                completed += 1
                progress("mi_complete", completed=completed, total=54, metric=metric,
                         points_per_sample=size, seed=seed, mi_bits=result["mi_bits"])
                del x, y
            del values
            gc.collect()
    return summarize(manifest_path, out)


def describe_repeats(values):
    values = np.asarray(values, np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Exactly three finite sampling repeats required")
    return dict(mean_bits=float(np.mean(values)), sample_sd_bits=float(np.std(values, ddof=1)),
                ddof=1, repetitions=3, sampling_seeds=list(SEEDS), mi_bits_by_seed=values.tolist())


def summarize(manifest_path, out):
    _, protocol, preparation = verify_prepared(manifest_path, out)
    summaries, result_hashes = [], {}
    for size in SIZES:
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                path = result_path(out, metric, size, seed)
                record = existing_result(path, result_binding(protocol, preparation, metric, size, seed))
                if record is None:
                    raise ValueError("Incomplete 54-result coverage: " + str(path))
                values.append(record["result"]["mi_bits"])
                result_hashes[path.name] = record["record_sha256"]
            summaries.append(dict(metric=metric, points_per_sample=size,
                sample_count=protocol["expected_sample_count"],
                observations_per_repeat=protocol["expected_sample_count"] * size,
                **describe_repeats(values)))
    actual_names = {p.name for p in (Path(out) / "results").glob("*.json")}
    if actual_names != set(result_hashes):
        raise ValueError("Unexpected or missing result files")
    result = seal(dict(schema=VERSION, status="complete", protocol_sha256=digest(protocol),
        preparation_sha256=preparation["record_sha256"], expected_result_count=54,
        actual_result_count=len(result_hashes), result_record_sha256=result_hashes,
        summaries=summaries, unit="bits", summary_definition="Mean +/- sample SD (ddof=1) across sampling seeds",
        limitations=["Sampling-repeat SD is not a confidence interval or population uncertainty.",
                     "P/D/O are frozen rule-derived regions; O is other, not clinically certified normal skin.",
                     "Region-conditioned pixel MI is not whole-image information or proof of physical impossibility.",
                     "Native coordinate correspondence is inherited, not re-registered or medically validated here."]))
    save_json(Path(out) / "summary.json", result)
    progress("complete", results=54, summary=str(Path(out) / "summary.json"))
    return result


@contextmanager
def output_lock(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "mi_atlas.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Output lock exists; check process before clearing: " + str(lock))
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), created_unix=time.time())))
        yield
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "summarize", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1..4")
    load_contract(args.manifest, args.output_dir)
    with output_lock(args.output_dir):
        if args.phase in ("prepare", "all"):
            prepare(args.manifest, args.output_dir)
        if args.phase in ("run", "all"):
            run_mi(args.manifest, args.output_dir, args.workers)
        elif args.phase == "summarize":
            summarize(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
