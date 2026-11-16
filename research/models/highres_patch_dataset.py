# -*- coding: utf-8 -*-
"""Raw-image datasets for the frozen 4000x4000 protocol.

All crop coordinates are defined in a canonical 4000x4000 output space and
mapped back to the source center-square ROI. This preserves normalized field
of view across 8000x4000 and 9728x4864 captures without upsampling the dominant
cohort. RGB and target always receive the exact same spatial transform.
"""

from __future__ import print_function

import csv
import hashlib
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGET_MODALITIES = ("uv", "plv", "plh")
TARGET_PATH_COLUMN = {"uv": "uv_path", "plv": "plv_path", "plh": "plh_path"}
REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "source_width",
    "source_height",
    "rgb_path",
    "uv_path",
    "plv_path",
    "plh_path",
    "eligible_primary",
)
EXPECTED_SPLIT_COUNTS = {"train": 80740, "val": 11534, "test": 23069}


class HighResolutionManifestError(RuntimeError):
    pass


class HighResolutionTestSplitAccessError(RuntimeError):
    pass


def load_frozen_raw_manifest(path):
    if not os.path.isfile(path):
        raise HighResolutionManifestError("Frozen raw manifest missing: %s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise HighResolutionManifestError(
                "Frozen raw manifest is missing columns: %s" % ", ".join(missing)
            )
        rows = [row for row in reader if str(row["eligible_primary"]).strip() == "1"]
    if not rows:
        raise HighResolutionManifestError("Frozen raw manifest has no eligible rows")
    return rows


def check_frozen_split_counts(rows, strict=True):
    observed = {}
    for row in rows:
        split = str(row["split"]).strip()
        observed[split] = observed.get(split, 0) + 1
    problems = []
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if observed.get(split, 0) != expected:
            problems.append(
                "%s expected=%d observed=%d"
                % (split, expected, observed.get(split, 0))
            )
    if strict and problems:
        raise HighResolutionManifestError(
            "Frozen split counts changed; refusing to continue:\n  %s"
            % "\n  ".join(problems)
        )
    return observed


def filter_frozen_split(rows, split):
    return [row for row in rows if str(row["split"]).strip() == split]


def deterministic_subset(rows, count):
    if count is None or count <= 0 or count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        key = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
        keyed.append((key, row["sample_id"], row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed[:count]]


def resolve_raw_path(dataset_root, manifest_path):
    if os.path.isabs(manifest_path):
        return manifest_path
    return os.path.join(dataset_root, manifest_path)


def center_square_geometry(width, height):
    side = min(int(width), int(height))
    left = (int(width) - side) // 2
    top = (int(height) - side) // 2
    return left, top, side


def canonical_patch_to_source_box(
    width, height, canonical_size, patch_size, output_x, output_y
):
    left, top, side = center_square_geometry(width, height)
    if side < canonical_size:
        raise HighResolutionManifestError(
            "Source side %d is below canonical size %d" % (side, canonical_size)
        )
    if patch_size <= 0 or patch_size > canonical_size:
        raise ValueError("patch_size must be in [1, canonical_size]")
    max_offset = canonical_size - patch_size
    if not (0 <= output_x <= max_offset and 0 <= output_y <= max_offset):
        raise ValueError(
            "Canonical patch origin out of bounds: (%d,%d), max=%d"
            % (output_x, output_y, max_offset)
        )

    scale = float(side) / float(canonical_size)
    source_left = left + int(math.floor(output_x * scale))
    source_top = top + int(math.floor(output_y * scale))
    source_right = left + int(math.ceil((output_x + patch_size) * scale))
    source_bottom = top + int(math.ceil((output_y + patch_size) * scale))
    square_right = left + side
    square_bottom = top + side
    source_left = max(left, min(source_left, square_right - 1))
    source_top = max(top, min(source_top, square_bottom - 1))
    source_right = max(source_left + 1, min(source_right, square_right))
    source_bottom = max(source_top + 1, min(source_bottom, square_bottom))
    return (source_left, source_top, source_right, source_bottom)


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _load_rgb_crop(path, expected_size, crop_box, output_size, sample_id, role):
    if not os.path.isfile(path):
        raise HighResolutionManifestError(
            "Raw image missing: sample_id=%s role=%s path=%s"
            % (sample_id, role, path)
        )
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != expected_size:
            raise HighResolutionManifestError(
                "Raw size mismatch for %s %s: manifest=%s actual=%s"
                % (sample_id, role, str(expected_size), str(image.size))
            )
        image = image.crop(crop_box)
        if image.size != (output_size, output_size):
            image = image.resize((output_size, output_size), _lanczos())
        array = np.asarray(image, dtype=np.uint8)
    array = np.transpose(array, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array))


class HighResolutionPatchDataset(Dataset):
    """Deterministic epoch-varying aligned patches from frozen raw images.

    Call set_epoch(epoch) before constructing each epoch's DataLoader iterator.
    Use persistent_workers=False so workers receive the updated epoch value.
    """

    def __init__(
        self,
        rows,
        target,
        dataset_root,
        split_name,
        canonical_size=4000,
        patch_size=512,
        seed=20260817,
        allow_test=False,
    ):
        if target not in TARGET_MODALITIES:
            raise ValueError("Unknown target: %s" % target)
        if split_name == "test" and not allow_test:
            raise HighResolutionTestSplitAccessError(
                "Refusing to load test images during high-resolution development"
            )
        if not rows:
            raise HighResolutionManifestError("No rows for split %s" % split_name)
        if patch_size <= 0 or patch_size > canonical_size:
            raise ValueError("Invalid patch_size=%d" % patch_size)
        self.rows = list(rows)
        self.target = target
        self.target_column = TARGET_PATH_COLUMN[target]
        self.dataset_root = dataset_root
        self.split_name = split_name
        self.canonical_size = int(canonical_size)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def patch_origin(self, row, index):
        max_offset = self.canonical_size - self.patch_size
        if max_offset == 0:
            return 0, 0
        token = "%d|%d|%s|%d" % (
            self.seed,
            self.epoch,
            row["sample_id"],
            int(index),
        )
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        x_value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        y_value = int.from_bytes(digest[8:16], byteorder="big", signed=False)
        return x_value % (max_offset + 1), y_value % (max_offset + 1)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_id = row["sample_id"]
        width = int(row["source_width"])
        height = int(row["source_height"])
        output_x, output_y = self.patch_origin(row, index)
        source_box = canonical_patch_to_source_box(
            width,
            height,
            self.canonical_size,
            self.patch_size,
            output_x,
            output_y,
        )
        expected_size = (width, height)
        rgb_path = resolve_raw_path(self.dataset_root, row["rgb_path"])
        target_path = resolve_raw_path(self.dataset_root, row[self.target_column])
        rgb = _load_rgb_crop(
            rgb_path,
            expected_size,
            source_box,
            self.patch_size,
            sample_id,
            "rgb",
        )
        target = _load_rgb_crop(
            target_path,
            expected_size,
            source_box,
            self.patch_size,
            sample_id,
            self.target,
        )
        return {
            "rgb": rgb,
            "target": target,
            "sample_id": sample_id,
            "index": index,
            "canonical_x": output_x,
            "canonical_y": output_y,
            "source_box": torch.tensor(source_box, dtype=torch.int64),
        }


class HighResolutionFullImageDataset(Dataset):
    """Full canonical 4000x4000 pairs for final tiled validation."""

    def __init__(
        self,
        rows,
        target,
        dataset_root,
        split_name,
        canonical_size=4000,
        allow_test=False,
    ):
        if target not in TARGET_MODALITIES:
            raise ValueError("Unknown target: %s" % target)
        if split_name == "test" and not allow_test:
            raise HighResolutionTestSplitAccessError(
                "Refusing to load test images during high-resolution development"
            )
        if not rows:
            raise HighResolutionManifestError("No rows for split %s" % split_name)
        self.rows = list(rows)
        self.target = target
        self.target_column = TARGET_PATH_COLUMN[target]
        self.dataset_root = dataset_root
        self.split_name = split_name
        self.canonical_size = int(canonical_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_id = row["sample_id"]
        width = int(row["source_width"])
        height = int(row["source_height"])
        left, top, side = center_square_geometry(width, height)
        if side < self.canonical_size:
            raise HighResolutionManifestError(
                "Source side %d below canonical size %d for %s"
                % (side, self.canonical_size, sample_id)
            )
        box = (left, top, left + side, top + side)
        expected_size = (width, height)
        rgb = _load_rgb_crop(
            resolve_raw_path(self.dataset_root, row["rgb_path"]),
            expected_size,
            box,
            self.canonical_size,
            sample_id,
            "rgb",
        )
        target = _load_rgb_crop(
            resolve_raw_path(self.dataset_root, row[self.target_column]),
            expected_size,
            box,
            self.canonical_size,
            sample_id,
            self.target,
        )
        return {
            "rgb": rgb,
            "target": target,
            "sample_id": sample_id,
            "index": index,
        }
