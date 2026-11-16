"""Independent CPU audit of saved pilot signal arrays and metric records.

Uses row-run union/find for 8-connectivity (not scipy.ndimage), chunked all-pair
distance enumeration (not cKDTree), dynamic programming for small matching
components and a separate dense LAP for larger components. No production
records, detector thresholds, source images or model weights are modified.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

import metrics


def check(condition, message):
    if not condition:
        raise ValueError("Independent metric audit: " + message)


def close(actual, expected, label, tolerance=1e-12):
    if expected is None:
        check(actual is None, label + " must be null")
    else:
        check(actual is not None and np.isfinite(actual) and abs(actual - expected) <= tolerance,
              label + " value mismatch")


class UnionFind:
    def __init__(self):
        self.parent = []
    def add(self):
        i = len(self.parent); self.parent.append(i); return i
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def independent_objects(mask):
    """8-neighbour union of horizontal runs and exact pixel-coordinate sums."""
    mask = np.asarray(mask)
    check(mask.dtype == np.bool_ and mask.ndim == 2, "boolean 2-D mask required")
    uf, runs, previous = UnionFind(), [], []
    for y, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        current = []
        previous_start = 0
        for left, right in zip(starts, ends):
            ident = uf.add(); current.append((int(left), int(right), ident))
            runs.append((y, int(left), int(right), ident))
            while previous_start < len(previous) and previous[previous_start][1] < left - 1:
                previous_start += 1
            j = previous_start
            while j < len(previous) and previous[j][0] <= right + 1:
                uf.union(ident, previous[j][2]); j += 1
        previous = current
    sums = {}
    for y, left, right, ident in runs:
        root = uf.root(ident); length = right - left + 1
        v = sums.setdefault(root, [0, 0, 0, y * mask.shape[1] + left])
        v[0] += length; v[1] += y * length; v[2] += (left + right) * length
    ordered = sorted(sums.values(), key=lambda v: v[3])
    centres = np.asarray([[v[1] / v[0], v[2] / (2 * v[0])] for v in ordered], dtype=np.float64).reshape(-1, 2)
    return {"areas": np.asarray([v[0] for v in ordered], dtype=np.int64), "centres_yx": centres}


def independent_matching_objective(truth, prediction, gate):
    """All gated edges, independent component DP/dense-LAP objective checks."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    nt, npred = len(truth), len(prediction)
    if not nt or not npred:
        return 0, 0.0, {"dp_components": 0, "dense_components": 0}
    uf = UnionFind()
    for _ in range(nt + npred):
        uf.add()
    edges = []
    for start in range(0, nt, 256):
        distances = np.sqrt(np.sum((truth[start:start + 256, None, :] - prediction[None, :, :]) ** 2, axis=2))
        ii, jj = np.where(distances <= gate)
        for i, j in zip(ii, jj):
            a, b, d = int(start + i), int(j), float(distances[i, j])
            edges.append((a, b, d)); uf.union(a, nt + b)
    grouped = {}
    for a, b, d in edges:
        grouped.setdefault(uf.root(a), []).append((a, b, d))
    total_n, total_cost = 0, 0.
    counts = {"dp_components": 0, "dense_components": 0}
    for group in grouped.values():
        ti, pj = sorted({v[0] for v in group}), sorted({v[1] for v in group})
        imap, jmap = {v: i for i, v in enumerate(ti)}, {v: j for j, v in enumerate(pj)}
        n, m = len(ti), len(pj)
        by_row = [[] for _ in range(n)]
        for a, b, d in group:
            by_row[imap[a]].append((jmap[b], d))
        if n <= 10 and m <= 10:
            states = {0: 0.}
            for row in by_row:
                new = dict(states)
                for used, value in states.items():
                    for j, distance in row:
                        if not (used >> j) & 1:
                            key = used | (1 << j); candidate = value + distance
                            if key not in new or candidate < new[key]:
                                new[key] = candidate
                states = new
            k = max(bin(used).count("1") for used in states)
            cost = min(value for used, value in states.items() if bin(used).count("1") == k)
            counts["dp_components"] += 1
        else:
            check(n * (m + n) <= 12000000, "Pilot independent dense component exceeds fixed 96MB cap")
            # A different cost representation and dense solver from production.
            penalty = (min(n, m) + 1) * (gate + 1) + 2
            matrix = np.full((n, m + n), np.inf)
            matrix[np.arange(n), m + np.arange(n)] = penalty
            for i, row in enumerate(by_row):
                for j, distance in row:
                    matrix[i, j] = distance
            ii, jj = linear_sum_assignment(matrix)
            keep = jj < m
            k, cost = int(np.count_nonzero(keep)), float(matrix[ii[keep], jj[keep]].sum())
            counts["dense_components"] += 1
        total_n += k; total_cost += cost
    return total_n, total_cost, counts


def _independent_domains(f, roi_masks):
    domains, assigned = {}, np.zeros_like(f)
    for name in metrics.ROI_ORDER:
        region = (np.asarray(roi_masks[name]) >= .5) & f & ~assigned
        domains[name] = region; assigned |= region
    domains[metrics.REMAINDER] = f & ~assigned
    check(np.array_equal(sum(v.astype(np.uint8) for v in domains.values()), f.astype(np.uint8)), "ROI partition")
    return domains


def audit_signal_arrays(truth_mask, prediction_mask, f_mask, roi_masks, record, group_index, signal,
                        truth_contrast=None, prediction_contrast=None, check_matching=True):
    f = np.asarray(f_mask); check(f.dtype == np.bool_ and f.any(), "Nonempty boolean F required")
    domains = _independent_domains(f, roi_masks)
    objects = {}
    checks = {"object_count_and_centroid": 0, "scalar": 0, "windows": 0, "matching": 0,
              "random_centres": 0, "dp_components": 0, "dense_components": 0}
    for side, mask, contrast in (("truth", truth_mask, truth_contrast), ("prediction", prediction_mask, prediction_contrast)):
        check(np.asarray(mask).shape == f.shape and not np.any(mask & ~f), side + " mask/F binding")
        obj = independent_objects(mask); objects[side] = obj
        produced = metrics.extract_objects(mask, f_mask=f, domains=domains)
        check(np.array_equal(obj["areas"], produced["areas"]), side + " independent 8CC areas")
        check(np.array_equal(obj["centres_yx"], produced["centres_yx"]), side + " independent geometric centroid sums")
        check(record[side]["count"] == len(obj["areas"]), side + " object count")
        check(record[side]["signal_pixels"] == int(mask.sum()), side + " mask pixel area")
        check(record[side]["valid_pixels"] == int(f.sum()), side + " F area")
        close(record[side]["area_fraction"], float(mask.sum() / f.sum()), side + " area fraction")
        if contrast is not None:
            close(record[side]["contrast_burden"], float(np.sum(contrast[mask], dtype=np.float64) / f.sum()), side + " contrast burden")
        checks["object_count_and_centroid"] += len(obj["areas"])
        checks["scalar"] += 5 if contrast is not None else 4
        labels = []
        for cy, cx in obj["centres_yx"]:
            y, x = min(f.shape[0] - 1, math.floor(cy + .5)), min(f.shape[1] - 1, math.floor(cx + .5))
            labels.append(next((name for name in metrics.ROI_ORDER if domains[name][y, x]), metrics.REMAINDER))
        obj["roi_labels"] = np.asarray(labels, dtype=object)
        for name, domain in domains.items():
            r = record["rois"][name][side]; n = int(domain.sum())
            check(r["count"] == labels.count(name), side + "/" + name + " centroid ROI count")
            close(r["area_fraction"], float(np.count_nonzero(mask & domain) / n) if n else None, side + "/" + name + " area")
            if contrast is not None:
                close(r["contrast_burden"], float(np.sum(contrast[mask & domain], dtype=np.float64) / n) if n else None, side + "/" + name + " burden")
    for width in (64, 128):
        labels, scores, starts, valid_counts = [], [], [], []
        for y in range(0, f.shape[0], width):
            for x in range(0, f.shape[1], width):
                sl = np.s_[y:y + width, x:x + width]; valid = int(f[sl].sum())
                if valid * 2 >= width ** 2:
                    labels.append(int(np.any(truth_mask[sl])))
                    scores.append(int(prediction_mask[sl].sum()) / valid)
                    starts.append([y, x]); valid_counts.append(valid)
        window = record["windows"][str(width)]
        check(window["labels"] == labels and window["scores"] == scores, "Independent window labels/scores")
        check(window["window_origin_yx"] == starts and window["F_pixel_count"] == valid_counts, "Window domain counts")
        checks["windows"] += len(labels)
    if "spatial" in record:
        spatial = record["spatial"]
        denominator = int(truth_mask.sum()) + int(prediction_mask.sum())
        dice = 2. * int(np.logical_and(truth_mask, prediction_mask).sum()) / denominator if denominator else None
        close(spatial["dice"], dice, "Dice")
        t = objects["truth"]["centres_yx"]; p = objects["prediction"]["centres_yx"]
        def check_match(a, b, observed, gate):
            nt, npred = len(a), len(b)
            k, cost, diagnostics = independent_matching_objective(a, b, gate)
            check((observed["tp"], observed["fp"], observed["fn"]) == (k, npred - k, nt - k), "Matching counts")
            close(observed["mean_center_error"], cost / k if k else None, "Matching per-group error", 1e-10)
            close(observed["f1"], 2. * k / (nt + npred) if nt + npred else None, "Matching F1")
            close(observed["precision"], k / npred if npred else None, "Matching precision")
            close(observed["recall"], k / nt if nt else None, "Matching recall")
            checks["matching"] += 1
            for name, count in diagnostics.items():
                checks[name] += count
        for gate in metrics.TOLERANCES:
            if check_matching:
                check_match(t, p, spatial["by_tolerance"][str(gate)], gate)
        check(len(spatial["random_repeats"]) == metrics.RANDOM_REPEATS, "Frozen random repeat count")
        for repeat, observation in enumerate(spatial["random_repeats"]):
            check("dice" not in observation, "Random centres must not have synthetic Dice")
            check(observation["repeat"] == repeat, "Random repeat ordering")
            random_centres = metrics.randomise_centres(objects["prediction"]["roi_labels"], domains, group_index, signal, repeat)
            check(len(random_centres) == len(p), "Random total object count")
            for centre, name in zip(random_centres.astype(int), objects["prediction"]["roi_labels"]):
                check(domains[name][tuple(centre)], "Random centre leaves conditional anatomical domain")
            checks["random_centres"] += len(random_centres)
            # Independently generated RNG arrays with the same frozen specification.
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261030, int(group_index), 0 if signal == "P" else 1, repeat])))
            expected = np.empty_like(random_centres)
            for name in metrics.ROI_ORDER + (metrics.REMAINDER,):
                indices = [i for i, lab in enumerate(objects["prediction"]["roi_labels"]) if lab == name]
                if indices:
                    pixels = np.where(domains[name].reshape(-1))[0]
                    picked = rng.choice(pixels, len(indices), replace=True)
                    expected[indices, 0] = picked // f.shape[1]
                    expected[indices, 1] = picked % f.shape[1]
            check(np.array_equal(expected, random_centres), "Independent random draw reproducibility")
            if check_matching:
                for gate in metrics.TOLERANCES:
                    check_match(t, random_centres, observation["by_tolerance"][str(gate)], gate)
    return {"status": "passed", "checks": checks,
            "independent_methods": ["row-run union/find 8CC and integer coordinate sums",
                                    "explicit window loops", "chunked all-pair distances",
                                    "small-component exact matching DP / separate dense LAP",
                                    "independent conditional RNG redraw and pixel membership"],
            "scope": "saved pilot detector output arrays; no claim of independent clinical annotation",
            "matching_checked": bool(check_matching)}
