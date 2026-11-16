"""Partition the original fractional F, including every detached component."""
import hashlib

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from fullface_partition import CROSS, REGION_NAMES, PartitionQCError, _inputs, partition_face


VERSION = "F5_frozen_F_core_geodesic_nearest_islands_v1"


def _assign_islands(core_labels, components, sat_ids):
    """Measure every island against fixed seeded regions, never against other islands."""
    lut = np.zeros(components.max() + 1, np.int32)
    lut[sat_ids] = np.arange(1, len(sat_ids) + 1)
    index = lut[components]
    points = np.argwhere(index != 0)
    group = index[tuple(points.T)] - 1
    squared = np.full((len(sat_ids), len(REGION_NAMES)), np.iinfo(np.int64).max, np.int64)
    for i in range(len(REGION_NAMES)):
        mask = core_labels == i+1
        boundary = np.argwhere(mask & ~ndimage.binary_erosion(mask, structure=CROSS))
        _, near = cKDTree(boundary).query(points)
        delta = points.astype(np.int64) - boundary[near]
        np.minimum.at(squared[:, i], group, np.sum(delta*delta, axis=1))
    # argmin chooses first canonical region on exact squared-distance ties.
    chosen = np.argmin(squared, axis=1) + 1
    labels = core_labels.copy()
    labels[tuple(points.T)] = chosen[group]
    return labels, chosen, squared


def partition_frozen_f(f_weights, old_weights):
    """Return (uint8 labels, JSON-ready QA). Positive F only; labels 1–5 in REGION_NAMES order.

    Raises PartitionQCError with a machine-readable code on inconsistent old
    geometry, empty seeds, split seeds across F components, or failed validation.
    """
    f, inside, seeds = _inputs(f_weights, old_weights)
    for i, name in enumerate(REGION_NAMES, 1):
        s = np.asarray(old_weights[name])
        if np.any(s[seeds == i] > f[seeds == i]):
            raise PartitionQCError("seed_weight_exceeds_F", name)
    components, n = ndimage.label(inside, structure=CROSS)
    seeded = []
    for i, name in enumerate(REGION_NAMES, 1):
        ids = np.unique(components[seeds == i])
        if len(ids) != 1:
            raise PartitionQCError("seed_crosses_disconnected_F", name)
        seeded.append(int(ids[0]))
    seeded = sorted(set(seeded))
    sat_ids = np.setdiff1d(np.arange(1,n+1), seeded)
    core = np.isin(components, seeded) if len(sat_ids) else inside
    f_core = np.where(core, f, 0) if len(sat_ids) else f
    core_labels, geometry = partition_face(f_core, {s:old_weights[s] for s in REGION_NAMES})
    entries=[]
    if len(sat_ids):
        labels, chosen, squared = _assign_islands(core_labels, components, sat_ids)
        counts = np.bincount(components.ravel())
        areas = np.bincount(components.ravel(),weights=f.ravel())
        for j,c in enumerate(sat_ids):
            i=int(chosen[j])
            entries.append(dict(component=int(c),positive_cells=int(counts[c]),effective_area=float(areas[c]),
                                assigned_region=REGION_NAMES[i-1],label=i,distance_squared=int(squared[j,i-1]),
                                distance_tied_labels=(np.flatnonzero(squared[j]==squared[j,i-1])+1).tolist()))
    else:
        labels = core_labels
    if not np.array_equal(labels>0,inside) or not np.array_equal(labels[seeds>0],seeds[seeds>0]):
        raise PartitionQCError("support_or_seed_mismatch","Full F/old labels differ")
    reconstructed = np.zeros_like(f)
    per_region={}
    for i,name in enumerate(REGION_NAMES,1):
        mask = labels==i
        n_core = ndimage.label(mask & core,structure=CROSS)[1]
        n_all = ndimage.label(mask,structure=CROSS)[1]
        attached = [a for a in entries if a["label"]==i]
        if n_core!=1 or n_all!=1+len(attached):
            raise PartitionQCError("unexpected_components",name)
        reconstructed[mask]=f[mask]
        per_region[name]=dict(positive_cells=int(mask.sum()),effective_area=float(f[mask].sum(dtype=np.float64)),
                              core_components=int(n_core),all_components=int(n_all),satellites=len(attached),
                              extra_effective_area=float(sum(a["effective_area"] for a in attached)))
    if not np.array_equal(reconstructed,f):
        raise PartitionQCError("weight_mismatch","Reconstructed weights differ from frozen F.")
    return labels, dict(protocol=VERSION, core_protocol=geometry["protocol"], old_seeds_preserved=True,
                        exact_frozen_F=True, frozen_F_sha256=hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(),
                        original_F_components=int(n),seeded_F_components=seeded,satellite_count=len(entries),
                        effective_F_area=float(f.sum(dtype=np.float64)),positive_F_cells=int(inside.sum()),
                        satellite_effective_area=float(sum(a["effective_area"] for a in entries)),
                        core_bridge_order=geometry["bridge_order"],regions=per_region,islands=entries)
