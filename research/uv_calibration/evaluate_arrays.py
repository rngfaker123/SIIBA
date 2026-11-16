"""Fit CAL group thresholds and evaluate uncertainty on independent EVAL group arrays."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from numerics import METHODS, cpu_evaluate, cpu_scores, finite_threshold, risk_curve, threshold


REGIONS = ("P", "D", "O", "F")


def read_group(path):
    with np.load(path, allow_pickle=False) as group:
        x, y, sigma = (np.asarray(group[k], dtype=np.float64) for k in ("truth", "prediction", "sigma"))
        weights = {r: np.asarray(group[r], dtype=np.float64) for r in REGIONS}
    if x.ndim != 3 or x.shape[0] != 3 or y.shape != x.shape or sigma.shape != x.shape[1:]:
        raise ValueError("Expected CHW RGB truth and prediction, with one spatial sigma map")
    for a in (x, y, *weights.values()):
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError("Images and masks must be finite and within [0,1]")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Expected positive finite predictive standard deviations")
    if any(w.shape != sigma.shape for w in weights.values()):
        raise ValueError("Region shape differs from the sigma grid")
    if not weights["F"].any() or not np.allclose(weights["P"] + weights["D"] + weights["O"], weights["F"], rtol=0, atol=2e-6):
        raise ValueError("P, D and O must sum to a nonempty F")
    return x, y, sigma, weights


def evaluate(cal_files, eval_files):
    if not cal_files or not eval_files:
        raise ValueError("Both splits must contain geometry-valid groups")
    keys = [p.stem for p in [*cal_files, *eval_files]]
    if len(keys) != len(set(keys)):
        raise ValueError("Group keys must be distinct within and across splits")
    cal_scores = [cpu_scores(*read_group(p)) for p in cal_files]
    thresholds = {"pixel_sigma": finite_threshold([v["normalized"] for v in cal_scores]),
                  "pixel_absolute": finite_threshold([v["absolute"] for v in cal_scores])}
    groups = {r: [] for r in REGIONS}
    estimates = {r: {m: [] for m in METHODS} for r in REGIONS}
    simultaneous = {m: [] for m in METHODS}
    direct = {m: [] for m in METHODS}
    count_errors = {s: [] for s in ("P", "D")}
    count_sigmas = {s: [] for s in count_errors}
    for path in eval_files:
        args = read_group(path)
        scores = cpu_scores(*args)
        rows = cpu_evaluate(*args, thresholds)
        minimum = {m: 1.0 for m in METHODS}
        for region in REGIONS:
            if scores["regions"][region]["area"] == 0:
                continue
            groups[region].append(scores["regions"][region])
            for row in rows[region]:
                method = row["method"]
                estimates[region][method].append(row)
                minimum[method] = min(minimum[method], row["coverage"])
        for method in METHODS:
            direct[method].append(float(minimum[method] >= 0.95))
        for method, score, cutoff in zip(METHODS, ("normalized", "normalized", "absolute"),
                (1.959963984540054, threshold(thresholds["pixel_sigma"]), threshold(thresholds["pixel_absolute"]))):
            simultaneous[method].append(float(scores[score] <= cutoff))
        with np.load(path, allow_pickle=False) as group:
            for signal in count_errors:
                keys = [signal + "_count_" + field for field in ("truth", "prediction")]
                if all(k not in group for k in keys):
                    continue
                if any(k not in group for k in keys):
                    raise ValueError("Each object count needs both a truth and prediction")
                values = [np.asarray(group[k]).item() for k in keys]
                if any(type(v) is not int or v < 0 for v in values):
                    raise ValueError("Expected nonnegative integer counts")
                count_errors[signal].append(abs(values[0] - values[1]))
                count_sigmas[signal].append(scores["regions"]["F"]["mean_sigma"])
    results = {}
    for region in REGIONS:
        errors = [g["mean_absolute_error"] for g in groups[region]]
        sigmas = [g["mean_sigma"] for g in groups[region]]
        intervals = {}
        for method, rows in estimates[region].items():
            widths = [v["width"] for v in rows]
            intervals[method] = {
                "coverage": statistics.mean(v["coverage"] for v in rows) if rows else None,
                "width": statistics.mean(widths) if widths and None not in widths else None,
                "infinite_width_groups": sum(v["width_infinite"] for v in rows),
            }
        results[region] = dict(valid_groups=len(errors), mean_sigma=statistics.mean(sigmas) if sigmas else None,
                               intervals=intervals, intensity_risk=risk_curve(errors, sigmas))
    return dict(calibration_groups=len(cal_files), evaluation_groups=len(eval_files),
                thresholds=thresholds, simultaneous_group_coverage={m: statistics.mean(v) for m, v in simultaneous.items()},
                direct_weighted_group_coverage={m: statistics.mean(v) for m, v in direct.items()},
                count_risk={s: risk_curve(count_errors[s], count_sigmas[s]) for s in count_errors if count_errors[s]},
                regions=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", type=Path, required=True, help="Directory of CAL group NPZ files")
    parser.add_argument("--eval", type=Path, required=True, help="Directory of EVAL group NPZ files")
    args = parser.parse_args()
    print(json.dumps(evaluate(sorted(args.cal.glob("*.npz")), sorted(args.eval.glob("*.npz"))),
                     indent=2, allow_nan=False))
