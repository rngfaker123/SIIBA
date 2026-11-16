"""Frozen CPU definitions for algorithm-defined signal objects (Table 3C/3D).

No model, detector, filesystem, training or threshold fitting is performed here.
All masks are on the common image grid; array coordinates are (y, x).  The
caller binds this source, protocol, SciPy/NumPy versions, and actual inputs.
"""
import math

import numpy as np
import scipy
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import (connected_components,
                                  maximum_bipartite_matching,
                                  min_weight_full_bipartite_matching)

ROI_ORDER = ("forehead", "image_left_cheek", "image_right_cheek", "nose", "chin")
REMAINDER = "F_remainder"
TOLERANCES = (2, 5, 10)
RANDOM_SEED = 20261030
RANDOM_REPEATS = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _mask(value, shape=None, label="mask"):
    value = np.asarray(value)
    require(value.ndim == 2 and value.dtype == np.bool_, label + " must be a 2-D boolean mask")
    if shape is not None:
        require(value.shape == shape, label + " shape mismatch")
    return value


def _centres(value):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0:
        value = value.reshape(0, 2)
    require(value.ndim == 2 and value.shape[1] == 2, "centres must have shape (N,2), in y,x order")
    require(np.isfinite(value).all(), "nonfinite object centre")
    return value


def make_roi_domains(f_mask, roi_masks):
    """Disjoint fixed anatomical partition for centroid assignment/randomisation.

    Archived fractional ROI masks are rasterised at >=0.5.  A shared boundary
    pixel belongs to the first ROI in ROI_ORDER, independent of signal/model.
    F_reference, when supplied, is not a sixth anatomical region.  All remaining
    F pixels form the explicitly separate F_remainder domain.
    """
    f = _mask(f_mask, label="F")
    require(set(roi_masks) in (set(ROI_ORDER), set(ROI_ORDER) | {"F_reference"}),
            "Exactly five named anatomical ROIs required (optional F_reference)")
    domains = {}
    unassigned = f.copy()
    for name in ROI_ORDER:
        raw = np.asarray(roi_masks[name])
        require(raw.shape == f.shape and np.isfinite(raw).all(), "Invalid archived ROI " + name)
        require(np.all((raw >= 0) & (raw <= 1)), "ROI coverage outside [0,1]")
        domains[name] = (raw >= 0.5) & unassigned
        unassigned &= ~domains[name]
    domains[REMAINDER] = unassigned
    return domains


def extract_objects(mask, roi_masks=None, f_mask=None, domains=None):
    """8-connected full-F objects, scan-order IDs, unweighted mean pixel centres.

    An object's anatomical assignment uses the nearest raster pixel to its
    geometric centroid: floor(coordinate+0.5), clamped to the image bounds.
    Concave objects can have centres outside F; these stay full-F objects, are
    assigned F_remainder, and are explicitly counted in QC.
    """
    mask = _mask(mask)
    f = np.ones_like(mask) if f_mask is None else _mask(f_mask, mask.shape, "F")
    require(not np.any(mask & ~f), "Signal mask extends outside common F")
    if domains is None:
        domains = make_roi_domains(f, roi_masks) if roi_masks is not None else {REMAINDER: f}
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    if count:
        centres = np.asarray(ndimage.center_of_mass(mask, labels, np.arange(1, count + 1)), dtype=np.float64)
        points = np.floor(centres + 0.5).astype(np.int64)
        points[:, 0] = np.clip(points[:, 0], 0, mask.shape[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, mask.shape[1] - 1)
        membership = np.full(count, REMAINDER, dtype=object)
        for name in reversed(ROI_ORDER):
            if name in domains:
                membership[domains[name][points[:, 0], points[:, 1]]] = name
        outside = int(np.count_nonzero(~f[points[:, 0], points[:, 1]]))
    else:
        centres = np.empty((0, 2), dtype=np.float64)
        membership = np.empty(0, dtype=object)
        outside = 0
    return {"centres_yx": centres, "areas": areas, "roi_labels": membership,
            "count": int(count), "pixel_area": int(mask.sum()),
            "qc": {"centroids_outside_F": outside,
                   "single_pixel_objects": int(np.count_nonzero(areas == 1)),
                   "largest_object_area": int(areas.max()) if count else 0,
                   "objects_ge_1pct_F": int(np.count_nonzero(areas >= max(1, math.ceil(f.sum() * 0.01))))}}


def match_centres(truth_yx, prediction_yx, tolerance, include_pairs=True):
    """Maximum cardinality, then minimum sum of Euclidean centroid distances.

    Build only gated edges with a cKDTree and solve the sparse bipartite graph.
    Every truth row has a private unmatched dummy.
    Real cost=d+1; dummy C=(min(n,m)+1)*(tolerance+1)+1.  This makes losing a
    match cost more than every possible real-edge cost change in the graph;
    for fixed cardinality the objective is exactly total centre distance.
    Maximum cardinality is checked independently using an unweighted solver.

    Tie rule: stable scan-order object IDs, sorted CSR column indices, fixed
    SciPy sparse LAP implementation/version, no perturbation or rounding of
    distances.  The fixed solver's deterministic traversal resolves exact
    equal-cost solutions; no claim of unique biological correspondence.
    """
    truth, prediction = _centres(truth_yx), _centres(prediction_yx)
    tolerance = float(tolerance)
    require(np.isfinite(tolerance) and tolerance > 0, "Invalid distance tolerance")
    nt, npred = len(truth), len(prediction)
    pairs = []
    candidate_count = 0
    ambiguous_truth = ambiguous_prediction = 0
    largest_component_vertices = 0
    if nt and npred:
        neighbourhood = cKDTree(prediction).query_ball_point(truth, tolerance, return_sorted=True)
        degree = np.fromiter((len(js) for js in neighbourhood), dtype=np.int64, count=nt)
        if int(degree.sum()):
            rr = np.repeat(np.arange(nt, dtype=np.int64), degree)
            cc = np.concatenate(neighbourhood).astype(np.int64, copy=False)
            dd = np.linalg.norm(truth[rr] - prediction[cc], axis=1)
            # 包含容差边界
            keep = dd <= tolerance
            rr, cc, dd = rr[keep], cc[keep], dd[keep]
            candidate_count = len(rr)
            adjacency = sparse.csr_matrix((np.ones(len(rr), dtype=np.int8), (rr, cc)), shape=(nt, npred))
            adjacency.sort_indices()
            max_cardinality = int(np.count_nonzero(maximum_bipartite_matching(adjacency, perm_type="column") >= 0))
            ambiguous_truth = int(np.count_nonzero(np.bincount(rr, minlength=nt) > 1))
            ambiguous_prediction = int(np.count_nonzero(np.bincount(cc, minlength=npred) > 1))
            full = sparse.csr_matrix((np.ones(2 * len(rr), dtype=np.int8),
                                      (np.r_[rr, cc + nt], np.r_[cc + nt, rr])), shape=(nt + npred, nt + npred))
            _, components = connected_components(full, directed=False)
            if len(rr):
                sizes = np.bincount(components)
                largest_component_vertices = int(sizes[np.unique(components[rr])].max())
            else:
                largest_component_vertices = 0
            # 全局稀疏匹配
            # 各对象独用虚拟节点

            penalty = (min(nt, npred) + 1) * (tolerance + 1) + 1
            costs = sparse.csr_matrix((np.r_[dd + 1.0, np.full(nt, penalty)],
                                       (np.r_[rr, np.arange(nt)], np.r_[cc, npred + np.arange(nt)])),
                                      shape=(nt, npred + nt))
            costs.sort_indices()
            assigned_r, assigned_c = min_weight_full_bipartite_matching(costs)
            matched = assigned_c < npred
            matched_r, matched_c = assigned_r[matched], assigned_c[matched]
            distances = np.linalg.norm(truth[matched_r] - prediction[matched_c], axis=1)
            require(np.all(distances <= tolerance), "LAP returned an out-of-gate edge")
            pairs = [(int(i), int(j), float(d)) for i, j, d in zip(matched_r, matched_c, distances)]
            require(len(pairs) == max_cardinality, "Weighted matching lost cardinality")
    pairs.sort(key=lambda value: (value[0], value[1]))
    tp = len(pairs)
    require(len({p[0] for p in pairs}) == tp and len({p[1] for p in pairs}) == tp, "Matching is not one-to-one")
    result = {"tp": tp, "fp": npred - tp, "fn": nt - tp,
              "truth_objects": nt, "prediction_objects": npred,
              "precision": tp / npred if npred else None,
              "recall": tp / nt if nt else None,
              "f1": 2.0 * tp / (nt + npred) if nt + npred else None,
              "mean_center_error": float(np.mean([p[2] for p in pairs])) if tp else None,
              "center_error_sum": float(sum(p[2] for p in pairs)) if tp else None,
              "matched_count": tp,
              "correct_negative": nt == 0 and npred == 0,
              "qc": {"candidate_edges": candidate_count,
                     "truth_with_multiple_candidates": ambiguous_truth,
                     "prediction_with_multiple_candidates": ambiguous_prediction,
                     "largest_component_vertices": largest_component_vertices}}
    if include_pairs:
        result["pairs"] = [[int(i), int(j), d] for i, j, d in pairs]
    return result


def mask_dice(truth_mask, prediction_mask):
    truth = _mask(truth_mask)
    prediction = _mask(prediction_mask, truth.shape)
    denominator = int(truth.sum()) + int(prediction.sum())
    return (2.0 * int(np.count_nonzero(truth & prediction)) / denominator) if denominator else None


def window_observations(truth_mask, prediction_mask, f_mask, window):
    f = _mask(f_mask, label="F")
    truth = _mask(truth_mask, f.shape, "truth")
    prediction = _mask(prediction_mask, f.shape, "prediction")
    require(not np.any((truth | prediction) & ~f), "Window masks extend outside F")
    require(window in (64, 128), "Only frozen 64/128 pixel windows supported")
    require(f.shape[0] % window == 0 and f.shape[1] % window == 0, "Grid not divisible by window")
    # 覆盖率含50%
    def blocks(x):
        return x.reshape(x.shape[0] // window, window, x.shape[1] // window, window).sum(axis=(1, 3), dtype=np.int64)
    valid = blocks(f)
    t, p = blocks(truth), blocks(prediction)
    keep = valid * 2 >= window * window
    y, x = np.nonzero(keep)
    labels = (t[keep] > 0).astype(np.uint8)
    return {"labels": labels.tolist(), "scores": (p[keep] / valid[keep]).tolist(),
            "window_origin_yx": np.column_stack([y * window, x * window]).tolist(),
            "F_pixel_count": valid[keep].tolist(),
            "positive_windows": int(labels.sum()), "negative_windows": int(len(labels) - labels.sum()),
            "window_count": int(len(labels)),
            "prevalence": float(labels.mean()) if len(labels) else None}


def randomise_centres(roi_labels, domains, group_index, signal, repeat):
    """Uniform valid raster centres with replacement, preserving each ROI count.

    This conditional location control does not preserve within-ROI density,
    signal shapes, or create any artificial mask Dice.
    """
    require(signal in ("P", "D"), "Unknown signal")
    require(isinstance(group_index, (int, np.integer)) and group_index >= 0, "Invalid stable group index")
    require(isinstance(repeat, (int, np.integer)) and 0 <= repeat < RANDOM_REPEATS, "Invalid frozen repeat")
    labels = np.asarray(roi_labels, dtype=object)
    require(labels.ndim == 1 and all(x in domains for x in labels), "Unknown object ROI")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([RANDOM_SEED, int(group_index), 0 if signal == "P" else 1, int(repeat)])))
    centres = np.empty((len(labels), 2), dtype=np.float64)
    for name in ROI_ORDER + (REMAINDER,):
        chosen = np.flatnonzero(labels == name)
        if not len(chosen):
            continue
        domain = _mask(domains[name])
        pixels = np.flatnonzero(domain)
        require(len(pixels) > 0, "Prediction object assigned to empty randomisation domain " + name)
        selected = rng.choice(pixels, size=len(chosen), replace=True)
        centres[chosen, 0], centres[chosen, 1] = np.unravel_index(selected, domain.shape)
    return centres


def spatial_metrics(truth_mask, prediction_mask, f_mask, roi_masks, group_index, signal,
                    truth_objects=None, prediction_objects=None, include_pairs=False):
    f = _mask(f_mask)
    domains = make_roi_domains(f, roi_masks)
    truth_objects = truth_objects if truth_objects is not None else extract_objects(truth_mask, f_mask=f, domains=domains)
    prediction_objects = prediction_objects if prediction_objects is not None else extract_objects(prediction_mask, f_mask=f, domains=domains)
    t, p = truth_objects["centres_yx"], prediction_objects["centres_yx"]
    result = {"dice": mask_dice(truth_mask, prediction_mask),
              "correct_negative": not len(t) and not len(p),
              "by_tolerance": {str(g): match_centres(t, p, g, include_pairs=include_pairs) for g in TOLERANCES},
              "random_repeats": []}
    for repeat in range(RANDOM_REPEATS):
        rp = randomise_centres(prediction_objects["roi_labels"], domains, group_index, signal, repeat)
        result["random_repeats"].append({"repeat": repeat,
                                        "by_tolerance": {str(g): match_centres(t, rp, g, include_pairs=False) for g in TOLERANCES}})
    return result


def scalar_signal_measurements(mask, contrast, f_mask, objects, domains):
    """Whole-F and disjoint ROI count, area fraction, encoded contrast load."""
    f = _mask(f_mask)
    mask = _mask(mask, f.shape)
    require(not np.any(mask & ~f), "Signal outside F")
    contrast = np.asarray(contrast, dtype=np.float64)
    require(contrast.shape == f.shape and np.isfinite(contrast).all() and np.all(contrast >= 0), "Invalid contrast field")
    def one(domain, object_count):
        area = int(domain.sum())
        return {"count": int(object_count), "signal_pixels": int(np.count_nonzero(mask & domain)),
                "valid_pixels": area,
                "area_fraction": float(np.count_nonzero(mask & domain) / area) if area else None,
                "contrast_burden": float(np.sum(contrast[mask & domain], dtype=np.float64) / area) if area else None,
                "reason": None if area else "empty_anatomical_domain"}
    return one(f, objects["count"]), {name: one(domain, np.count_nonzero(objects["roi_labels"] == name)) for name, domain in domains.items()}


def measure_signal(truth_mask, prediction_mask, truth_contrast, prediction_contrast,
                   f_mask, roi_masks, group_index, signal, spatial=False, include_pairs=False):
    """Unsealed record for one signal/model/group; no inference inside this call."""
    domains = make_roi_domains(f_mask, roi_masks)
    truth_objects = extract_objects(truth_mask, f_mask=f_mask, domains=domains)
    prediction_objects = extract_objects(prediction_mask, f_mask=f_mask, domains=domains)
    truth, tr = scalar_signal_measurements(truth_mask, truth_contrast, f_mask, truth_objects, domains)
    prediction, pr = scalar_signal_measurements(prediction_mask, prediction_contrast, f_mask, prediction_objects, domains)
    result = {"truth": truth, "prediction": prediction,
              "rois": {name: {"truth": tr[name], "prediction": pr[name]} for name in domains},
              "windows": {str(w): window_observations(truth_mask, prediction_mask, f_mask, w) for w in (64, 128)},
              "qc": {"truth": truth_objects["qc"], "prediction": prediction_objects["qc"]}}
    if spatial:
        result["spatial"] = spatial_metrics(truth_mask, prediction_mask, f_mask, roi_masks, group_index, signal,
                                             truth_objects, prediction_objects, include_pairs=include_pairs)
    return result


def runtime_definition():
    return {"numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "coordinate_order": "y,x", "connectivity": 8,
            "centroid": "unweighted arithmetic mean of all object pixel coordinates",
            "roi_order": list(ROI_ORDER), "roi_raster_threshold": 0.5,
            "centroid_roi_pixel": "floor(coordinate+0.5), bounded to grid",
            "matching": "max_cardinality_then_min_total_Euclidean_distance",
            "ties": "stable scan-order IDs, sorted CSR, pinned deterministic SciPy sparse LAP traversal; no jitter",
            "tolerances_pixels": list(TOLERANCES), "random_seed": RANDOM_SEED,
            "random_repeats": RANDOM_REPEATS, "random_rng": "PCG64/SeedSequence[seed,group_index,P=0_D=1,repeat]",
            "random_sampling": "uniform valid raster pixel centres with replacement within fixed nonoverlapping anatomical domain",
            "random_baseline_dice": "not_defined_no_random_masks"}
