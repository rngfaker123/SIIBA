"""Frozen, group-based Table 3C/3D statistics; no model or detector execution.

Window scores are sorted exactly once. Bootstrap weights multiply sparse
score-by-group count matrices: no discretisation, independent-window resampling,
or repeated window sort is used. Only the primary 64px window receives intervals.
"""
import argparse
from contextlib import ExitStack
import csv
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse, stats

MODELS = ("unet_single", "unet_multi", "unetplusplus", "swin", "ldm")
SIGNALS = ("P", "D")
SEED = 20261029
BOOTSTRAP_REPEATS = 2000
WINDOW_BOOTSTRAP_REPEATS = 500
SCALARS = ("count", "area_fraction", "contrast_burden")
SPATIAL_METRICS = ("precision", "recall", "f1", "mean_center_error")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    with open(path, "x", encoding="utf8") as f:
        json.dump(clean(value), f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def finite_or_nan(value):
    if value is None:
        return np.nan
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite numerical value is not a valid missing value")
    return value


def interval(draws, repeats):
    a = np.asarray(draws, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {"low": float(np.quantile(a, .025)) if len(a) else None,
            "high": float(np.quantile(a, .975)) if len(a) else None,
            "level": .95, "method": "paired_sample_group_percentile_bootstrap",
            "draws_requested": int(repeats), "draws_valid": len(a),
            "draws_undefined": int(repeats - len(a))}


def icc_a1(x, y):
    """Two-way mixed/single measurement/absolute agreement ANOVA ICC(A,1).

    No clipping. Constants with undefined denominator return NaN. Rows are
    paired sample groups, columns are truth and prediction.
    """
    a = np.stack([x, y], axis=-1).astype(np.float64)
    n = a.shape[-2]
    if n < 2:
        return np.full(a.shape[:-2], np.nan) if a.ndim > 2 else float("nan")
    grand = a.mean(axis=(-2, -1), keepdims=True)
    row = a.mean(axis=-1, keepdims=True)
    col = a.mean(axis=-2, keepdims=True)
    msr = 2 * np.sum((row - grand) ** 2, axis=(-2, -1)) / (n - 1)
    msc = n * np.sum((col - grand) ** 2, axis=(-2, -1))
    mse = np.sum((a - row - col + grand) ** 2, axis=(-2, -1)) / (n - 1)
    den = msr + mse + 2 * (msc - mse) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(den > 0, (msr - mse) / den, np.nan)
    return float(result) if result.ndim == 0 else result


def paired_summary(truth, prediction, repeats=BOOTSTRAP_REPEATS, seed=SEED):
    x = np.asarray(truth, dtype=np.float64)
    y = np.asarray(prediction, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Pair arrays must have equal one-dimensional shape")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    result = {"n": n, "n_undefined": int(len(valid) - n),
              "mae": float(np.abs(y-x).mean()) if n else None,
              "bias": float((y-x).mean()) if n else None,
              "icc_a1": clean(icc_a1(x, y)),
              "truth_mean": float(x.mean()) if n else None,
              "prediction_mean": float(y.mean()) if n else None,
              "resampling_unit": "sample_group", "bootstrap_seed": seed}
    if repeats and n:
        rng = np.random.default_rng(seed)
        mae, icc = [], []
        # 限制索引批量
        batch = max(1, min(64, 2500000 // n))
        for start in range(0, repeats, batch):
            ids = rng.integers(n, size=(min(batch, repeats-start), n))
            xb, yb = x[ids], y[ids]
            mae.extend(np.abs(yb-xb).mean(axis=1).tolist())
            icc.extend(np.atleast_1d(icc_a1(xb, yb)).tolist())
        result.update(mae_ci95=interval(mae, repeats), icc_a1_ci95=interval(icc, repeats))
    elif repeats:
        result.update(mae_ci95=interval([], repeats), icc_a1_ci95=interval([], repeats))
    return result


def mean_summary(values, repeats=BOOTSTRAP_REPEATS, seed=SEED):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    result = {"n": n, "mean": float(a.mean()) if n else None,
              "sample_sd": float(a.std(ddof=1)) if n > 1 else None}
    if repeats and n:
        rng = np.random.default_rng(seed)
        means = []
        batch = max(1, min(64, 2500000 // n))
        for start in range(0, repeats, batch):
            ids = rng.integers(n, size=(min(batch, repeats-start), n))
            means.extend(a[ids].mean(axis=1).tolist())
        result["mean_ci95"] = interval(means, repeats)
    elif repeats:
        result["mean_ci95"] = interval([], repeats)
    return result


def curves_from_counts(positive, negative):
    """Counts at EXACT descending scores; tied scores enter together.

    PR starts at recall=0, precision=1. Trapezoid PR-AUC and noninterpolated AP
    are intentionally separate. With only one label class all AUCs are NA.
    """
    p, q = np.asarray(positive, float), np.asarray(negative, float)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Invalid score counts")
    use = (p + q) > 0
    p, q = p[use], q[use]
    np_, nn = p.sum(), q.sum()
    result = {"positive_windows": int(np_), "negative_windows": int(nn),
              "window_count": int(np_+nn),
              "prevalence": float(np_/(np_+nn)) if np_+nn else None,
              "auroc": None, "pr_auc_trapezoid": None, "ap_noninterpolated": None}
    if not np_ or not nn:
        result["undefined_reason"] = "no_positive_windows" if not np_ else "no_negative_windows"
        return result
    tp, fp = np.cumsum(p), np.cumsum(q)
    recall = tp / np_
    precision = tp / (tp+fp)
    fpr = np.r_[0., fp/nn]
    tpr = np.r_[0., recall]
    rr, pp = np.r_[0., recall], np.r_[1., precision]
    result.update(auroc=float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])*.5)),
                  pr_auc_trapezoid=float(np.sum(np.diff(rr)*(pp[:-1]+pp[1:])*.5)),
                  ap_noninterpolated=float(np.sum(np.diff(rr)*precision)))
    return result


class WindowAccumulator:
    """Per group input. Object stores only arrays, not full record dictionaries."""
    def __init__(self):
        self.labels, self.scores, self.groups = [], [], []

    def add(self, group, labels, scores):
        label = np.asarray(labels)
        score = np.asarray(scores, np.float64)
        if label.ndim != 1 or label.shape != score.shape:
            raise ValueError("Window label/score shape mismatch")
        if not np.all((label == 0) | (label == 1)) or not np.isfinite(score).all():
            raise ValueError("Invalid window label or score")
        if np.any(score < 0) or np.any(score > 1):
            raise ValueError("Area fraction score outside [0,1]")
        self.groups.append(str(group))
        self.labels.append(label.astype(np.uint8))
        self.scores.append(score)

    def summarize(self, repeats=WINDOW_BOOTSTRAP_REPEATS, seed=SEED):
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("Duplicate window sample group")
        n = len(self.groups)
        lengths = np.asarray([len(x) for x in self.labels])
        labels = np.concatenate(self.labels) if n else np.empty(0, np.uint8)
        scores = np.concatenate(self.scores) if n else np.empty(0)
        # 原值去重并降序
        unique, score_ids = np.unique(scores, return_inverse=True)
        score_ids = len(unique)-1-score_ids
        group_ids = np.repeat(np.arange(n), lengths)
        shape = (len(unique), n)
        positive = sparse.csr_matrix((labels.astype(float), (score_ids, group_ids)), shape=shape)
        negative = sparse.csr_matrix(((1-labels).astype(float), (score_ids, group_ids)), shape=shape)
        positive.eliminate_zeros()
        negative.eliminate_zeros()
        result = curves_from_counts(np.asarray(positive.sum(axis=1)).ravel(),
                                    np.asarray(negative.sum(axis=1)).ravel())
        result.update(group_count=n, groups_with_windows=int(np.count_nonzero(lengths)),
                      groups_without_windows=int(np.count_nonzero(lengths == 0)),
                      unique_exact_score_count=len(unique), score_quantized=False,
                      resampling_unit="sample_group", bootstrap_seed=seed,
                      score_definition="prediction signal area / common F pixels in window; not probability")
        if repeats:
            rng = np.random.default_rng(seed)
            draws = {k: [] for k in ("auroc", "pr_auc_trapezoid", "ap_noninterpolated")}
            for _ in range(repeats if n else 0):
                weights = np.bincount(rng.integers(n, size=n), minlength=n)
                one = curves_from_counts(positive @ weights, negative @ weights)
                for k in draws:
                    draws[k].append(np.nan if one[k] is None else one[k])
            result["ci95"] = {k: interval(v, repeats) for k, v in draws.items()}
        return result


def paired_test(a, b):
    keys = sorted(set(a) & set(b))
    dif = np.asarray([a[k]-b[k] for k in keys if a[k] is not None and b[k] is not None], float)
    dif = dif[np.isfinite(dif)]
    n = len(dif)
    nz = int(np.count_nonzero(dif))
    out = {"n_paired": n, "n_nonzero_differences": nz, "n_zero_differences": n-nz,
           "mean_paired_difference": float(dif.mean()) if n else None,
           "median_paired_difference": float(np.median(dif)) if n else None,
           "statistic": None, "p_raw": None,
           "test": "two-sided Wilcoxon signed-rank; wilcox zeros; normal approximation; no continuity correction"}
    if not n:
        out["undefined_reason"] = "no_defined_paired_groups"
    elif nz == 0:
        out.update(statistic=0., p_raw=1.)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            test = stats.wilcoxon(dif, zero_method="wilcox", correction=False,
                                  alternative="two-sided", method="approx")
        out.update(statistic=float(test.statistic), p_raw=float(test.pvalue))
    return out


def holm(family):
    """Retain undefined prespecified tests in family size (conservative)."""
    valid = sorted([(r["p_raw"], i) for i, r in enumerate(family) if r["p_raw"] is not None])
    running = 0.
    for rank, (p, idx) in enumerate(valid):
        running = min(1., max(running, p * (len(family)-rank)))
        family[idx]["p_holm"] = running
    for row in family:
        row.setdefault("p_holm", None)
        row["prespecified_family_size"] = len(family)
    return family


def comparisons(values):
    endpoints = ("count_abs_error", "area_abs_error", "dice", "f1")
    model_tests, signal_tests, random_tests = [], [], []
    for signal, endpoint in itertools.product(SIGNALS, endpoints):
        for a, b in itertools.combinations(MODELS, 2):
            model_tests.append(dict(signal=signal, endpoint=endpoint, model_a=a, model_b=b,
                                    **paired_test(values[a][signal][endpoint], values[b][signal][endpoint])))
    for model, endpoint in itertools.product(MODELS, endpoints):
        signal_tests.append(dict(model=model, endpoint=endpoint, signal_a="P", signal_b="D",
                                 **paired_test(values[model]["P"][endpoint], values[model]["D"][endpoint])))
    for model, signal in itertools.product(MODELS, SIGNALS):
        random_tests.append(dict(model=model, signal=signal, endpoint="f1_real_minus_conditional_random_mean",
                                 **paired_test(values[model][signal]["f1"], values[model][signal]["random_f1"])))
    return {"model_pairs_primary_80": holm(model_tests), "within_model_P_vs_D_20": holm(signal_tests),
            "model_vs_conditional_random_10": holm(random_tests)}


def validate_spatial(spatial):
    if spatial.get("dice") is not None and not 0 <= spatial["dice"] <= 1:
        raise ValueError("Dice outside [0,1]")
    for tol in ("2", "5", "10"):
        x = spatial["by_tolerance"][tol]
        tp, fp, fn = [x[k] for k in ("tp", "fp", "fn")]
        if any(int(v) != v or v < 0 for v in (tp, fp, fn)):
            raise ValueError("Object totals must be nonnegative integers")
        den = 2*tp+fp+fn
        expected = 2*tp/den if den else None
        if (expected is None) != (x["f1"] is None) or (expected is not None and abs(expected-x["f1"]) > 1e-12):
            raise ValueError("Object F1 disagrees with TP/FP/FN")
        for key, expected in (("precision", tp/(tp+fp) if tp+fp else None),
                              ("recall", tp/(tp+fn) if tp+fn else None)):
            if (expected is None) != (x[key] is None) or (expected is not None and abs(expected-x[key]) > 1e-12):
                raise ValueError(key+" disagrees with TP/FP/FN")
        if (tp == 0) != (x["mean_center_error"] is None):
            raise ValueError("Center error null status disagrees with TP")
        if tp and not 0 <= x["mean_center_error"] <= float(tol)+1e-12:
            raise ValueError("Mean center error exceeds matching gate")
        if not den and spatial["dice"] is not None:
            raise ValueError("Correct-negative Dice must be undefined")


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def write_tables(out, summaries):
    # 五模型按列对照
    rows_c, rows_d = [], []
    for signal in SIGNALS:
        for metric in SCALARS:
            for stat in ("mae", "icc_a1"):
                rows_c.append([signal, metric+" / "+stat] +
                              [_fmt(summaries[m][signal]["F"][metric][stat]) for m in MODELS])
        for stat in ("auroc", "pr_auc_trapezoid"):
            rows_c.append([signal, "64px / "+stat] +
                          [_fmt(summaries[m][signal]["windows"]["64"][stat]) for m in MODELS])
        for metric in ("dice", "precision", "recall", "f1", "mean_center_error"):
            rows_d.append([signal, metric] +
                          [_fmt(summaries[m][signal]["spatial"]["5"][metric]["mean"]) for m in MODELS])
    for name, rows in (("table3c", rows_c), ("table3d", rows_d)):
        with open(out/(name+".tsv"), "x", encoding="utf8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Signal", "Metric"] + list(MODELS))
            writer.writerows(rows)
        # 另列区间与缺失
        cells = lambda row: "".join("<td>"+str(v)+"</td>" for v in row)
        html = '<meta charset="utf-8"><style>table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;font:13px Arial}thead{border-bottom:1px solid}td,th{border:0;padding:8px 14px;white-space:nowrap;text-align:right}td:first-child,td:nth-child(2){text-align:left}</style><table><thead><tr>'
        html += "".join("<th>"+x+"</th>" for x in ["Signal", "Metric"]+list(MODELS))
        html += "</tr></thead><tbody>"+"".join("<tr>"+cells(row)+"</tr>" for row in rows)+"</tbody></table>"
        with open(out/(name+".html"), "x", encoding="utf8") as f:
            f.write(html)
    # 逐指标列有效数
    # 记未定义抽样数
    detail, window_counts, spatial_counts, roi_rows = [], [], [], []
    for model, signal in itertools.product(MODELS, SIGNALS):
        result = summaries[model][signal]
        for metric in SCALARS:
            cell = result["F"][metric]
            for stat in ("mae", "icc_a1"):
                ci = cell.get(stat+"_ci95", {})
                detail.append([model, signal, "3C", "F/"+metric+"/"+stat, cell[stat],
                               cell["n"], cell["n_undefined"], ci.get("low"), ci.get("high"),
                               ci.get("draws_valid"), ci.get("draws_undefined")])
        for window, cell in result["windows"].items():
            window_counts.append([model, signal, window]+[cell[k] for k in (
                "group_count", "groups_with_windows", "groups_without_windows", "window_count",
                "positive_windows", "negative_windows", "prevalence")]+[cell.get("undefined_reason")])
            for metric in ("auroc", "pr_auc_trapezoid", "ap_noninterpolated"):
                ci = cell.get("ci95", {}).get(metric, {})
                detail.append([model, signal, "3C", window+"px/"+metric, cell[metric],
                               cell["groups_with_windows"], cell["groups_without_windows"],
                               ci.get("low"), ci.get("high"), ci.get("draws_valid"), ci.get("draws_undefined")])
        for tolerance, spatial in result["spatial"].items():
            pooled = spatial["tp_fp_fn_pooled_descriptive"]
            spatial_counts.append([model, signal, tolerance, spatial["selected_groups"],
                                   spatial["correct_negative_groups"], pooled["tp"], pooled["fp"], pooled["fn"]])
            for metric in ("dice",)+SPATIAL_METRICS:
                cell = spatial[metric]; ci = cell.get("mean_ci95", {})
                detail.append([model, signal, "3D", tolerance+"px/"+metric, cell["mean"], cell["n"],
                               cell["n_undefined"], ci.get("low"), ci.get("high"),
                               ci.get("draws_valid"), ci.get("draws_undefined")])
            for metric, cell in spatial["conditional_random_centers"].items():
                ci = cell.get("mean_ci95", {})
                detail.append([model, signal, "3D_conditional_random", tolerance+"px/"+metric,
                               cell["mean"], cell["n"], spatial["selected_groups"]-cell["n"],
                               ci.get("low"), ci.get("high"), ci.get("draws_valid"), ci.get("draws_undefined")])
        for roi, results in result["rois_descriptive"].items():
            for metric, cell in results.items():
                roi_rows.append([model, signal, roi, metric]+[cell[k] for k in (
                    "n", "n_undefined", "mae", "bias", "icc_a1", "truth_mean", "prediction_mean")])
    supplements = {
        "supplement_endpoint_N_CI": (["model", "signal", "table", "endpoint", "estimate", "N_groups", "N_undefined_groups",
                                       "CI95_low", "CI95_high", "bootstrap_valid_draws", "bootstrap_undefined_draws"], detail),
        "supplement_window_counts": (["model", "signal", "window_px", "N_groups", "groups_with_windows", "groups_without_windows",
                                      "N_windows", "positive_windows", "negative_windows", "prevalence", "undefined_reason"], window_counts),
        "supplement_spatial_counts": (["model", "signal", "tolerance_px", "N_selected_groups", "correct_negative_groups", "TP", "FP", "FN"], spatial_counts),
        "supplement_anatomical_ROI": (["model", "signal", "ROI", "metric", "N", "N_undefined", "MAE", "bias", "ICC_A1", "truth_mean", "prediction_mean"], roi_rows),
    }
    for name, (header, rows) in supplements.items():
        with open(out/(name+".tsv"), "x", encoding="utf8", newline="") as f:
            writer = csv.writer(f, delimiter="\t"); writer.writerow(header); writer.writerows(rows)


def _collect(record_paths_by_model, expected_group_ids, spatial_group_ids, output_dir,
            bootstrap_repeats=BOOTSTRAP_REPEATS, window_bootstrap_repeats=WINDOW_BOOTSTRAP_REPEATS,
            seed=SEED, require_seals=True, _stack=None):
    """Read unique terminal records; no partial/failure records can become results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    expected = set(map(str, expected_group_ids))
    selected = set(map(str, spatial_group_ids))
    if len(expected) != len(expected_group_ids) or not selected <= expected:
        raise ValueError("Catalog IDs duplicated or spatial sample outside catalog")
    provenance, summaries, comparative = [], {}, {}
    truth_binding = {}
    valid_ids_by_model = {}
    scalar_file = _stack.enter_context(open(out/"paired_values.csv", "x", encoding="utf8", newline=""))
    scalar_writer = csv.writer(scalar_file)
    scalar_writer.writerow(["group_id", "model", "signal", "roi", "metric", "truth", "prediction"])
    spatial_file = _stack.enter_context(open(out/"spatial_values.csv", "x", encoding="utf8", newline=""))
    spatial_writer = csv.writer(spatial_file)
    spatial_writer.writerow(["group_id", "model", "signal", "tolerance_pixels", "dice", "tp", "fp", "fn", "precision", "recall", "f1", "mean_center_error"])
    for model in MODELS:
        ids, valid_ids, missing = set(), set(), {}
        data = {s: {"F": {k: [[], []] for k in SCALARS}, "rois": {},
                    "windows": {str(w): WindowAccumulator() for w in (64, 128)},
                    "spatial": [], "face": {"positive": 0, "negative": 0}} for s in SIGNALS}
        comparative[model] = {s: {e: {} for e in ("count_abs_error", "area_abs_error", "dice", "f1", "random_f1")} for s in SIGNALS}
        spatial_ids = set()
        for path in sorted(map(Path, record_paths_by_model[model])):
            record = json.loads(path.read_text(encoding="utf8"))
            seal = record.get("record_sha256")
            if require_seals and (seal is None or seal != canonical_sha({k: v for k, v in record.items() if k != "record_sha256"})):
                raise ValueError("Record seal mismatch: "+str(path))
            group = str(record["group_id"])
            if record["model"] != model or group not in expected or group in ids:
                raise ValueError("Record model/group ownership mismatch or duplicate")
            ids.add(group)
            provenance.append({"path": str(path), "sha256": sha256_file(path), "model": model, "group_id": group})
            if record["status"] == "unavailable_geometry":
                reason = record.get("missing_reason")
                if not reason:
                    raise ValueError("Geometry null record lacks reason")
                missing[reason] = missing.get(reason, 0)+1
                continue
            if record["status"] != "ok":
                raise ValueError("Nonterminal or failed record cannot be summarized")
            valid_ids.add(group)
            truth_payload = {s: {"truth": record["signals"][s]["truth"],
                                  "rois": {k: x["truth"] for k, x in record["signals"][s].get("rois", {}).items()},
                                  "windows": {w: {k: v for k, v in x.items() if k != "scores"}
                                              for w, x in record["signals"][s]["windows"].items()}}
                             for s in SIGNALS}
            truth_hash = canonical_sha(truth_payload)
            if model == MODELS[0]:
                truth_binding[group] = truth_hash
            elif truth_binding[group] != truth_hash:
                raise ValueError("Truth/ROI/window identity differs between models")
            for signal in SIGNALS:
                v = record["signals"][signal]
                for key in SCALARS:
                    x, y = finite_or_nan(v["truth"][key]), finite_or_nan(v["prediction"][key])
                    data[signal]["F"][key][0].append(x)
                    data[signal]["F"][key][1].append(y)
                    scalar_writer.writerow([group, model, signal, "F", key, clean(x), clean(y)])
                comparative[model][signal]["count_abs_error"][group] = abs(v["prediction"]["count"]-v["truth"]["count"])
                comparative[model][signal]["area_abs_error"][group] = abs(v["prediction"]["area_fraction"]-v["truth"]["area_fraction"])
                data[signal]["face"]["positive" if v["truth"]["count"] else "negative"] += 1
                for roi, rv in v.get("rois", {}).items():
                    dest = data[signal]["rois"].setdefault(roi, {k: [[], []] for k in SCALARS})
                    for key in SCALARS:
                        x, y = finite_or_nan(rv["truth"][key]), finite_or_nan(rv["prediction"][key])
                        dest[key][0].append(x); dest[key][1].append(y)
                        scalar_writer.writerow([group, model, signal, roi, key, clean(x), clean(y)])
                for w in ("64", "128"):
                    data[signal]["windows"][w].add(group, v["windows"][w]["labels"], v["windows"][w]["scores"])
                sp = v.get("spatial")
                if (group in selected) != (sp is not None):
                    raise ValueError("Spatial record ownership disagrees with frozen sample")
                if sp is not None:
                    validate_spatial(sp)
                    if [r.get("repeat") for r in sp.get("random_repeats", [])] != list(range(10)):
                        raise ValueError("Conditional random baseline does not have ten fixed repeats")
                    for repeat in sp["random_repeats"]:
                        validate_spatial({"dice": None, "by_tolerance": repeat["by_tolerance"]})
                    for match_set in [sp] + sp["random_repeats"]:
                        for match in match_set["by_tolerance"].values():
                            if (match["tp"]+match["fn"] != v["truth"]["count"] or
                                    match["tp"]+match["fp"] != v["prediction"]["count"]):
                                raise ValueError("Spatial or random baseline object counts differ from full-F counts")
                    spatial_ids.add(group)
                    data[signal]["spatial"].append(sp)
                    comparative[model][signal]["dice"][group] = sp["dice"]
                    comparative[model][signal]["f1"][group] = sp["by_tolerance"]["5"]["f1"]
                    rvals = [r["by_tolerance"]["5"]["f1"] for r in sp.get("random_repeats", [])]
                    rvals = [r for r in rvals if r is not None]
                    comparative[model][signal]["random_f1"][group] = float(np.mean(rvals)) if rvals else None
                    for tol, match in sp["by_tolerance"].items():
                        spatial_writer.writerow([group, model, signal, tol, sp["dice"]] + [match[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "mean_center_error")])
        if ids != expected or spatial_ids != selected:
            raise ValueError("Coverage incomplete for "+model)
        if len(expected) == 21899 and (len(valid_ids) != 21422 or sum(missing.values()) != 477 or len(selected) != 1000):
            raise ValueError("Frozen full TEST or spatial cohort counts differ")
        valid_ids_by_model[model] = valid_ids
        if valid_ids != valid_ids_by_model[MODELS[0]]:
            raise ValueError("Geometry-valid cohort differs across models")
        summaries[model] = {"coverage": {"all": len(ids), "geometry_valid": len(valid_ids), "missing_reasons": missing,
                                          "spatial_selected": len(spatial_ids)}}
        for signal in SIGNALS:
            d = data[signal]
            result = {"F": {k: paired_summary(*d["F"][k], repeats=bootstrap_repeats, seed=seed) for k in SCALARS},
                      "rois_descriptive": {roi: {k: paired_summary(*vals[k], repeats=0) for k in SCALARS} for roi, vals in d["rois"].items()},
                      "whole_face_presence_descriptive": d["face"], "windows": {}, "spatial": {}}
            for w in ("64", "128"):
                result["windows"][w] = d["windows"][w].summarize(window_bootstrap_repeats if w == "64" else 0, seed)
            spatial = d["spatial"]
            for tol in ("2", "5", "10"):
                summary = {k: mean_summary([finite_or_nan(x["by_tolerance"][tol][k]) for x in spatial], bootstrap_repeats if tol == "5" else 0, seed) for k in SPATIAL_METRICS}
                summary["dice"] = mean_summary([finite_or_nan(x["dice"]) for x in spatial], bootstrap_repeats if tol == "5" else 0, seed)
                summary["tp_fp_fn_pooled_descriptive"] = {k: sum(x["by_tolerance"][tol][k] for x in spatial) for k in ("tp", "fp", "fn")}
                summary["correct_negative_groups"] = sum(x["by_tolerance"][tol]["tp"]+x["by_tolerance"][tol]["fp"]+x["by_tolerance"][tol]["fn"] == 0 for x in spatial)
                summary["selected_groups"] = len(spatial)
                for k in ("dice",)+SPATIAL_METRICS:
                    summary[k]["n_undefined"] = len(spatial)-summary[k]["n"]
                random_means = {k: [] for k in SPATIAL_METRICS}
                repeat_counts = []
                for sp in spatial:
                    repeats = sp.get("random_repeats", [])
                    repeat_counts.append(len(repeats))
                    for k in SPATIAL_METRICS:
                        vals = [r["by_tolerance"][tol][k] for r in repeats if r["by_tolerance"][tol][k] is not None]
                        random_means[k].append(float(np.mean(vals)) if vals else np.nan)
                summary["conditional_random_centers"] = {k: mean_summary(v, bootstrap_repeats if tol == "5" else 0, seed) for k, v in random_means.items()}
                summary["conditional_random_repeat_counts"] = sorted(set(repeat_counts))
                result["spatial"][tol] = summary
            summaries[model][signal] = result
        # 保存阶段汇总
        write_json(out/(model+"_summary.json"), summaries[model])
        del data
        scalar_file.flush(); spatial_file.flush()
    scalar_file.close(); spatial_file.close()
    result = {"schema": "signal_objects_3cd_statistics_v1", "models": summaries,
              "comparisons": comparisons(comparative), "seed": seed,
              "bootstrap_repeats": bootstrap_repeats, "window_bootstrap_repeats": window_bootstrap_repeats,
              "interpretation_limits": ["algorithm-defined objects, not clinical lesions", "sample groups are not verified patients",
                                        "no detected paired difference does not prove equivalence", "center error is conditional on successful matches",
                                        "encoded contrast burden is not concentration", "random centers do not define a Dice mask"]}
    write_json(out/"summary.json", result)
    write_json(out/"record_manifest.json", provenance)
    write_tables(out, summaries)
    # 完成前核验输入
    if any(sha256_file(p["path"]) != p["sha256"] for p in provenance):
        raise ValueError("Record changed during collection")
    write_json(out/"completion.json", {"status": "complete", "record_count": len(provenance),
                                      "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      "collector_source_sha256": sha256_file(__file__),
                                      "expected_group_count": len(expected), "spatial_group_count": len(selected),
                                      "summary_sha256": sha256_file(out/"summary.json"),
                                      "record_manifest_sha256": sha256_file(out/"record_manifest.json"),
                                      "independent_audit_required": True})
    return result


def collect(record_paths_by_model, expected_group_ids, spatial_group_ids, output_dir,
            bootstrap_repeats=BOOTSTRAP_REPEATS, window_bootstrap_repeats=WINDOW_BOOTSTRAP_REPEATS,
            seed=SEED, require_seals=True):
    with ExitStack() as stack:
        return _collect(record_paths_by_model, expected_group_ids, spatial_group_ids, output_dir,
                        bootstrap_repeats, window_bootstrap_repeats, seed, require_seals, _stack=stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--catalog-json", required=True)
    parser.add_argument("--spatial-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = json.loads(Path(args.catalog_json).read_text(encoding="utf8"))
    spatial_ids = json.loads(Path(args.spatial_json).read_text(encoding="utf8"))
    ids = [str(x["group_id"]) if isinstance(x, dict) else str(x) for x in catalog]
    selected = [str(x["group_id"]) if isinstance(x, dict) else str(x) for x in spatial_ids]
    collect({m: list((Path(args.records_root)/m).glob("*.json")) for m in MODELS}, ids, selected, args.output)
