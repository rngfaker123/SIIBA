#!/usr/bin/env python3
"""Freeze an initial clean-VAL cohort before opening images or computing masks.

Production inputs, count and seed are pinned. No old MI job is resumed. Empty
masks, failed geometry and insufficient pixels must be reported downstream;
they never cause replacement by another capture in this initial cohort.
Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re


SCHEMA = "mi_atlas_initial_clean_val_cohort_20261025_v1"
CLEAN_MANIFEST_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
DEVELOPMENT_MANIFEST_SHA = "2e2f5d1aca155ad06f655bf68c4d3f1cebc107e6d9e28cba846bba74ce9b3ae5"
EXPECTED_SPLIT_COUNTS = {"train": 76642, "val": 10948, "test": 21899}
EXPECTED_DEV_COUNT = 20
SEED = 20261025
COUNT = 1000
MODALITIES = ("rgb", "uv", "plv", "plh")
SOURCE_FIELDS = ("sample_id", "subject_id", "source_height", "source_width") + tuple(
    modality + suffix for modality in MODALITIES for suffix in ("_path", "_sha256")
)
IDENTITY_CAVEAT = (
    "subject_id is folder-derived, not independently verified biological person "
    "identity; one row per subject_id is not proof of one independent person."
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pinned_bytes(path, expected):
    data = Path(path).read_bytes()
    if sha256_bytes(data) != expected:
        raise ValueError("Frozen input SHA256 mismatch: " + str(path))
    return data


def safe_relative_path(value):
    """Validate native source paths without touching the image filesystem."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected a nonempty POSIX relative image path")
    if any(ord(char) < 32 for char in value):
        raise ValueError("Control character in image path")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Unsafe relative image path: " + value)
    return value


def validate_source_row(row):
    """Preserve source fields while enforcing complete native-image bindings."""
    result = dict(row)
    for key in ("sample_id", "subject_id"):
        if (not isinstance(row.get(key), str) or row[key] in (".", "..")
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", row[key])):
            raise ValueError("Missing or invalid " + key)
    for key in ("source_height", "source_width"):
        text = str(row.get(key, ""))
        if not re.fullmatch(r"[1-9][0-9]*", text):
            raise ValueError("Invalid native dimension: " + key)
        result[key] = int(text)
    for modality in MODALITIES:
        safe_relative_path(row.get(modality + "_path"))
        value = row.get(modality + "_sha256")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Missing or invalid source SHA256: " + modality)
    return result


def read_clean_manifest(path):
    """Read the pinned clean CSV including its BOM; never inspect raw images."""
    payload = pinned_bytes(path, CLEAN_MANIFEST_SHA)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    required = set(SOURCE_FIELDS) | {"split", "eligible_primary", "full_decode_qc"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Clean CSV missing required columns")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Duplicate CSV column")
    rows = []
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV record at line " + str(line))
        row["source_manifest_row_1based"] = line
        rows.append(row)
    counts = dict(Counter(row["split"] for row in rows))
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Frozen clean split counts differ: " + str(counts))
    seen = set()
    subject_splits = {}
    for row in rows:
        sid, subject = row["sample_id"], row["subject_id"]
        if not sid or not subject or sid in seen:
            raise ValueError("Duplicate or missing clean source identity")
        seen.add(sid)
        if subject in subject_splits and subject_splits[subject] != row["split"]:
            raise ValueError("Cross-split subject_id overlap")
        subject_splits[subject] = row["split"]
        if row["eligible_primary"] != "1" or row["full_decode_qc"] != "pass":
            raise ValueError("Frozen clean row is not eligible/full-decode pass")
        if row.get("exclusion_reason", ""):
            raise ValueError("Excluded row unexpectedly present in clean manifest")
    return rows, counts


def read_development_manifest(path):
    data = json.loads(pinned_bytes(path, DEVELOPMENT_MANIFEST_SHA).decode("utf-8-sig"))
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_DEV_COUNT:
        raise ValueError("Expected the entire fixed development20 manifest")
    validated = [validate_source_row(row) for row in rows]
    if len({row["sample_id"] for row in validated}) != EXPECTED_DEV_COUNT:
        raise ValueError("Duplicate development sample ID")
    # Historic split/person statements in this old manifest are not inherited.
    return validated


def order_key(domain, identity, seed=SEED):
    """SHA ordering is deterministic across platforms and independent of CSV order."""
    value = json.dumps([int(seed), domain, identity], ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
    return sha256_bytes(value)


def select_rows(rows, development, *, count=COUNT, seed=SEED):
    """Choose one record per metadata subject, then the first fixed-count subjects.

    Development exclusion matches sample/subject IDs, per-modality source paths,
    or exact image hashes. No UV brightness, mask, image or outcome is consulted.
    The optional count/seed arguments are for synthetic unit tests; CLI is pinned.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    dev = [validate_source_row(row) for row in development]
    sample_ids = {row["sample_id"] for row in dev}
    subject_ids = {row["subject_id"] for row in dev}
    paths = {modality: {row[modality + "_path"] for row in dev} for modality in MODALITIES}
    hashes = {modality: {row[modality + "_sha256"] for row in dev} for modality in MODALITIES}
    groups, excluded, seen = {}, [], set()
    val_count = 0
    for source in rows:
        sid = source.get("sample_id")
        if not sid or sid in seen:
            raise ValueError("Duplicate or missing sample_id in selection input")
        seen.add(sid)
        if source.get("split") != "val":
            continue
        val_count += 1
        row = validate_source_row(source)
        reasons = []
        if row["sample_id"] in sample_ids:
            reasons.append("development_sample_id")
        if row["subject_id"] in subject_ids:
            reasons.append("development_subject_id")
        for modality in MODALITIES:
            if row[modality + "_path"] in paths[modality]:
                reasons.append("development_" + modality + "_path")
            if row[modality + "_sha256"] in hashes[modality]:
                reasons.append("development_" + modality + "_sha256")
        if reasons:
            excluded.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                             "reasons": reasons})
        else:
            groups.setdefault(row["subject_id"], []).append(row)
    subjects = sorted(groups, key=lambda subject: (order_key("subject", subject, seed), subject))
    if len(subjects) < count:
        raise ValueError("Insufficient eligible non-development VAL subjects: %d < %d" % (len(subjects), count))
    queue, duplicate_records = [], []
    for subject in subjects:
        records = sorted(groups[subject], key=lambda row: (order_key("record", row["sample_id"], seed), row["sample_id"]))
        queue.append(records[0])
        duplicate_records.extend({"sample_id": row["sample_id"], "subject_id": subject,
                                  "retained_sample_id": records[0]["sample_id"]} for row in records[1:])
    selected = []
    for index, source in enumerate(queue[:count]):
        row = dict(source)
        row.update(cohort_index=index, native_shape=[row["source_height"], row["source_width"]],
                   source_sha=CLEAN_MANIFEST_SHA)
        selected.append(row)
    audit = {
        "clean_val_rows": val_count,
        "development_excluded_val_rows": len(excluded),
        "available_non_development_val_rows": sum(map(len, groups.values())),
        "available_unique_subject_ids": len(subjects),
        "subject_deduplicated_records": len(duplicate_records),
        "development_excluded_rows": sorted(excluded, key=lambda row: row["sample_id"]),
        "deduplicated_rows": sorted(duplicate_records, key=lambda row: row["sample_id"]),
        "development_sample_ids_absent_from_clean": sorted(sample_ids - seen),
        "development_sample_ids": sorted(sample_ids),
        "development_subject_ids": sorted(subject_ids),
    }
    return selected, audit


def build_cohort(manifest, development, data_root):
    rows, counts = read_clean_manifest(manifest)
    development_rows = read_development_manifest(development)
    selected, audit = select_rows(rows, development_rows)
    source_rows = [{key: row[key] for key in SOURCE_FIELDS} for row in selected]
    return {
        "schema": SCHEMA,
        "status": "frozen_initial_cohort_before_images_and_masks",
        "data_root": str(data_root),
        "source_manifest": {"path": str(Path(manifest).resolve()), "sha256": CLEAN_MANIFEST_SHA,
                            "source_split_counts": counts},
        "development_manifest": {"path": str(Path(development).resolve()),
                                 "sha256": DEVELOPMENT_MANIFEST_SHA, "records": EXPECTED_DEV_COUNT},
        "provenance": {"builder_sha256": sha256_bytes(Path(__file__).read_bytes()),
                       "source_sha_meaning": "SHA256 of the full clean frozen CSV; per-image hashes are preserved separately",
                       "source_manifest_row_1based_meaning": "CSV row number including header as row 1",
                       "path_base": "All four modality paths are relative to data_root",
                       "raw_images_read": False, "masks_read": False,
                       "old_cancelled_job_resumed": False},
        "selection": {
            "split": "val", "seed": SEED, "requested_count": COUNT, "selected_count": len(selected),
            "algorithm": "Sort subjects by SHA256(JSON([seed,'subject',subject_id])); choose one record per subject by SHA256(JSON([seed,'record',sample_id])); take first count. Hash ties broken lexically by ID.",
            "hash_json_encoding": "UTF-8, ensure_ascii=False, compact comma/colon separators",
            "subject_identity_caveat": IDENTITY_CAVEAT,
            "development_exclusion": "Exclude all fixed development20 by sample_id OR subject_id OR corresponding source path OR corresponding source image SHA256",
            "preimage_membership": True,
            "replacement_allowed": False,
            "outcome_based_filtering": False,
            "downstream_failures": "Report geometry/read/mask failures and insufficient region pixels explicitly; do not replace captures. A later common eligible subset must bind to this initial cohort and preserve its relative order.",
            "manual_review_caveat": "Clean means the frozen exclusion-only protocol; it does not imply every retained image was manually reviewed.",
        },
        "exclusions": audit,
        "membership_sha256": sha256_bytes(encoded(source_rows)),
        "rows": selected,
    }


def write_new_json(path, value):
    """Exclusive create; an identical existing freeze is a no-op, never overwritten."""
    path = Path(path)
    payload = encoded(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Refusing to overwrite a different frozen cohort: " + str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Pinned clean manifest_frozen.csv")
    parser.add_argument("--development", type=Path, required=True, help="Pinned complete development20 manifest.json")
    parser.add_argument("--data-root", required=True, help="Native source root; no images are read")
    parser.add_argument("--output", type=Path, required=True, help="New frozen cohort.json")
    args = parser.parse_args()
    if args.output.resolve() in (args.manifest.resolve(), args.development.resolve(), Path(__file__).resolve()):
        raise ValueError("Output cannot replace an input or this builder")
    result = build_cohort(args.manifest, args.development, args.data_root)
    created = write_new_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "created": created,
                      "schema": result["schema"], "count": len(result["rows"]),
                      "membership_sha256": result["membership_sha256"],
                      "output_sha256": sha256_bytes(encoded(result))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
