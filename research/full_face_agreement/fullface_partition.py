"""Landmark-seeded five-zone partition of an existing fractional facial mask.

The five archived anatomical ROIs are immutable seeds on the evaluation grid.
Every cell of positive F weight is assigned to a zone by four-neighbour
shortest-path distance within F. Distances are in image-grid steps. Ties use
the order in REGION_NAMES. Weights are copied verbatim from the frozen F.
"""
from collections import deque
from itertools import permutations

import numpy as np
from scipy import ndimage


REGION_NAMES = ("forehead", "image_left_cheek", "image_right_cheek", "nose", "chin")
CROSS = ndimage.generate_binary_structure(2, 1)
VERSION = "F5_seeded_geodesic_4_v1"


class PartitionQCError(ValueError):
    """Case did not satisfy the partition requirements; retain it in the QC log."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__("%s: %s" % (code, detail))


def _inputs(f_weights, old_weights):
    f = np.asarray(f_weights)
    if (f.ndim != 2 or not f.size or f.dtype.kind not in "buif"
            or not np.isfinite(f).all() or np.any(f < 0) or np.any(f > 1)):
        raise PartitionQCError("invalid_F", "Expected a finite 2-D coverage array in [0, 1].")
    if "F_reference" in old_weights and not np.array_equal(old_weights["F_reference"], f):
        raise PartitionQCError("F_reference_mismatch", "Supplied F differs from the archived reference.")
    inside = f > 0
    seeds = np.zeros(f.shape, np.uint8)
    for idx, name in enumerate(REGION_NAMES, 1):
        if name not in old_weights:
            raise PartitionQCError("missing_seed", name)
        old = np.asarray(old_weights[name])
        if (old.shape != f.shape or old.dtype.kind not in "buif"
                or not np.isfinite(old).all() or np.any(old < 0) or np.any(old > 1)):
            raise PartitionQCError("invalid_seed", name)
        mask = old > 0
        if not mask.any():
            raise PartitionQCError("empty_seed", name)
        if np.any(mask & ~inside):
            raise PartitionQCError("seed_outside_F", name)
        if old.dtype.kind == "f" and np.any(old > f):
            raise PartitionQCError("seed_weight_exceeds_F", name)
        if np.any(mask & (seeds != 0)):
            raise PartitionQCError("overlapping_seeds", name)
        seeds[mask] = idx
    return f, inside, seeds


def _components(mask):
    return ndimage.label(mask, structure=CROSS)


def _check_support(inside, seeds):
    groups, count = _components(inside)
    if count > 1:
        seeded = set()
        for idx, name in enumerate(REGION_NAMES, 1):
            components = np.unique(groups[seeds == idx])
            if components.size != 1:
                raise PartitionQCError("seed_crosses_disconnected_F", name)
            seeded.add(int(components[0]))
        if len(seeded) != count:
            raise PartitionQCError("unseeded_F_component",
                                   "A disconnected component of F has no old ROI seed.")
    return int(count)


def _flood(inside, seeds):
    """Level-order multi-source flood with label-priority ties."""
    blanks = inside & (seeds == 0)
    border = ndimage.binary_dilation(blanks, structure=CROSS) & (seeds != 0)
    start = np.pad(np.where(border, seeds, 0), 1).ravel()
    padded = np.pad(np.where(inside, seeds, 255).astype(np.uint8), 1, constant_values=255)
    width = padded.shape[1]
    pixels = bytearray(padded)
    queue = deque()
    for idx in range(1, 6):
        queue.extend(np.flatnonzero(start == idx).tolist())
    steps = (-width, -1, 1, width)
    while queue:
        p = queue.popleft()
        label = pixels[p]
        for delta in steps:
            q = p + delta
            if pixels[q] == 0:
                pixels[q] = label
                queue.append(q)
    labels = np.frombuffer(pixels, dtype=np.uint8).reshape(padded.shape)[1:-1, 1:-1].copy()
    labels[~inside] = 0
    return labels


def _attach_island(inside, claimed, idx):
    """Bridge one disconnected seed island by the shortest open path."""
    seeds = claimed == idx
    parts, count = _components(seeds)
    if count <= 1:
        return True
    sizes = np.bincount(parts.ravel())[1:]
    main = parts == (1 + int(np.argmax(sizes)))
    goals = seeds & ~main
    # 0 = open, 1..4 = direction to parent, 5 = start, 6 = goal, 7 = barrier.
    terrain = np.full(inside.shape, 7, dtype=np.uint8)
    terrain[inside & (claimed == 0)] = 0
    terrain[goals] = 6
    terrain[main] = 5
    padded = np.pad(terrain, 1, constant_values=7)
    width = padded.shape[1]
    steps = (-width, -1, 1, width)
    pixels = bytearray(padded)
    start = main & ndimage.binary_dilation(terrain == 0, structure=CROSS)
    queue = deque(np.flatnonzero(np.pad(start, 1)).tolist())
    while queue:
        p = queue.popleft()
        for code, delta in enumerate(steps, 1):
            q = p + delta
            if pixels[q] == 0:
                pixels[q] = code
                queue.append(q)
            elif pixels[q] == 6:
                while pixels[p] != 5:
                    y, x = divmod(p, width)
                    claimed[y-1, x-1] = idx
                    p -= steps[pixels[p]-1]
                return True
    return False


def validate_partition(f_weights, old_weights, labels):
    """Validate the actual delivered label image against the original arrays."""
    f, inside, seeds = _inputs(f_weights, old_weights)
    label = np.asarray(labels)
    if label.shape != f.shape or label.dtype.kind not in "ui" or np.any(label < 0) or np.any(label > 5):
        raise PartitionQCError("invalid_labels", "Expected integer labels 0..5.")
    if not np.array_equal(label != 0, inside):
        raise PartitionQCError("support_mismatch", "Union must equal all of positive F.")
    if not np.array_equal(label[seeds != 0], seeds[seeds != 0]):
        raise PartitionQCError("seed_not_contained", "One or more old cells changed anatomical class.")
    restored = np.zeros_like(f)
    regions = {}
    for idx, name in enumerate(REGION_NAMES, 1):
        m = label == idx
        components = int(_components(m)[1])
        if components != 1:
            raise PartitionQCError("noncontiguous_zone", "%s has %s components" % (name, components))
        restored[m] = f[m]
        regions[name] = dict(components=components, old_seed_cells=int(np.sum(seeds == idx)),
                             new_cells=int(m.sum()), effective_area=float(f[m].sum(dtype=np.float64)))
    if not np.array_equal(restored, f):
        raise PartitionQCError("weight_mismatch", "Summed regional weights must equal F exactly.")
    return regions


def partition_face(f_weights, old_weights):
    """Return uint8 labels and auditable per-zone coverage; raise on failed QC.

    Zero outside F, labels 1..5 in REGION_NAMES order. The weight map for
    label i is np.where(labels == i, f_weights, 0). For split old seeds, try
    the ordinary flood first. If it cannot connect all zones, reserve minimal
    four-connected paths joining islands without crossing any other seed,
    then reflood. Try fragmented-class priority orders deterministically.
    Failure of this finite repair is a QC issue, not a proof of impossibility.
    """
    f, inside, seeds = _inputs(f_weights, old_weights)
    frame_components = _check_support(inside, seeds)
    labels = _flood(inside, seeds)
    additions = np.zeros(5, dtype=int)
    bridge_order = []
    if any(_components(labels == i)[1] != 1 for i in range(1, 6)):
        fragmented = [i for i in range(1, 6) if _components(seeds == i)[1] > 1]
        for order in permutations(fragmented):
            claimed = seeds.copy()
            blocked = False
            for idx in order:
                while _components(claimed == idx)[1] > 1:
                    if not _attach_island(inside, claimed, idx):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                candidate = _flood(inside, claimed)
                if all(_components(candidate == i)[1] == 1 for i in range(1, 6)):
                    labels = candidate
                    additions = np.bincount(claimed[(claimed != 0) & (seeds == 0)], minlength=6)[1:]
                    bridge_order = list(order)
                    break
        else:
            raise PartitionQCError("seed_islands_not_connected",
                                   "No valid solution found by the documented bridge search.")
    regions = validate_partition(f, old_weights, labels)
    for idx, name in enumerate(REGION_NAMES):
        regions[name]["bridge_cells"] = int(additions[idx])
    audit = dict(protocol=VERSION, connectivity=4, distance="shortest path on F > 0",
                 tie_priority=list(REGION_NAMES), F_components=frame_components, bridge_order=bridge_order,
                 positive_F_cells=int(inside.sum()), effective_F_area=float(f.sum(dtype=np.float64)),
                 union_equals_F=True, old_seeds_preserved=True, regions=regions)
    return labels, audit
