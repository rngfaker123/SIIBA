"""Agreement of paired per-sample regional UV signal means.

Inputs are one regional scalar per sample group per method, never pixels. The
two fixed methods are observed UV and a particular frozen model's output.

ICC(A,1): two-way mixed effects, absolute agreement, single measurement. With
k=2, (MS_rows-MS_error)/(MS_rows+(k-1)*MS_error+k*(MS_columns-MS_error)/n).
McGraw & Wong (1996), doi:10.1037/1082-989X.1.1.30 (and correction 1.4.390).
Bland & Altman (1986), doi:10.1016/S0140-6736(86)90837-8:
https://www-users.york.ac.uk/~mb55/meas/ba.htm

Bias and limits use prediction minus truth. Normal-theory limits are descriptive
bias +/- 1.96 sample SD, not confidence limits and not a clinical acceptability
test. Percentile limits are reported as a distributional diagnostic. Bootstrap
confidence intervals resample complete paired sample-group rows. Independent
patient identity is not established by this interface.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np

METRICS = (
    "truth_mean", "prediction_mean", "bias", "difference_sample_sd_ddof1",
    "loa_low", "loa_high", "icc_a1", "difference_p025", "difference_p975",
)


def _vector(values, name):
    # Reject booleans, nulls and numeric strings rather than coerce or drop rows.
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(name + " must be a one-dimensional paired vector")
    if raw.dtype.kind not in "iuf":
        raise ValueError(name + " must contain real numeric scalars only")
    a = raw.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(name + " contains nonfinite data; silent omission is forbidden")
    return a


def _estimate(truth, prediction):
    """One batch of rows, returning a [batch, metric] float64 array.

    NaN represents only a mathematically undefined estimate internally; public
    outputs replace it with explicit null and a reason. No ICC clipping occurs.
    """
    n = truth.shape[1]
    values = np.full((truth.shape[0], len(METRICS)), np.nan, dtype=np.float64)
    if not n:
        return values
    tm, pm = truth.mean(axis=1), prediction.mean(axis=1)
    d = prediction - truth
    bias = d.mean(axis=1)
    q = np.quantile(d, [0.025, 0.975], axis=1, interpolation="linear")
    values[:, 0], values[:, 1], values[:, 2] = tm, pm, bias
    values[:, 7], values[:, 8] = q[0], q[1]
    if n < 2:
        return values
    sd = d.std(axis=1, ddof=1)
    values[:, 3], values[:, 4], values[:, 5] = sd, bias - 1.96 * sd, bias + 1.96 * sd

    # Two-way ANOVA, specialized to two methods for lower bootstrap memory use.
    row_mean = 0.5 * (truth + prediction)
    grand = 0.5 * (tm + pm)
    ms_rows = 2.0 * np.square(row_mean - grand[:, None]).sum(axis=1) / (n - 1)
    ms_columns = n * (np.square(tm - grand) + np.square(pm - grand))
    # Each method's interaction residual is +/- (d_i - d_bar)/2.
    ms_error = 0.5 * np.square(d - bias[:, None]).sum(axis=1) / (n - 1)
    denom = ms_rows + ms_error + 2.0 * (ms_columns - ms_error) / n
    # Repeated decimal constants can acquire tiny fake variance through mean
    # rounding. Exactly identical constant pairs have an undefined ICC.
    all_one_value = ((np.ptp(truth, axis=1) == 0.0)
                     & (np.ptp(prediction, axis=1) == 0.0)
                     & (truth[:, 0] == prediction[:, 0]))
    denom[all_one_value] = 0.0
    np.divide(ms_rows - ms_error, denom, out=values[:, 6], where=denom > 0.0)
    return values


def _reason(metric, n):
    if n == 0:
        return "no_valid_paired_sample_groups"
    if n < 2:
        return "fewer_than_two_valid_paired_sample_groups"
    if metric == "icc_a1":
        return "icc_zero_or_nonpositive_anova_denominator"
    return "nonfinite_statistic"


def summarize_pairs(truth, pred, bootstrap_repeats=2000, seed=20261026):
    """Return scalar JSON fields for paired sample-group agreement.

    Missing/failed regions must be explicitly accounted for by the caller before
    passing the matched valid pair vectors. This function never selects samples.
    Empty vectors return an undefined summary; unequal/nonfinite inputs fail.
    ``bootstrap_repeats=0`` explicitly disables intervals. One or negative
    bootstrap repeats is invalid. Samples are resampled with replacement as
    matched pairs; this does not retrain, resample pixels or calibrate predictions.
    """
    x, y = _vector(truth, "truth"), _vector(pred, "prediction")
    if x.shape != y.shape:
        raise ValueError("truth and prediction must have identical paired lengths")
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, Integral) or bootstrap_repeats < 0 or bootstrap_repeats == 1:
        raise ValueError("bootstrap_repeats must be zero or an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bootstrap_repeats, seed = int(bootstrap_repeats), int(seed)
    n = int(x.size)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        point = _estimate(x[None, :], y[None, :])[0]
    result = {
        "schema": "regional_signal_agreement_statistics_v1",
        "n": n,
        "n_input_pairs": n,
        "n_omitted_pairs": 0,
        "statistical_unit": "sample_group",
        "independent_patient_identity_verified": False,
        "bootstrap_unit": "paired_sample_group",
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
        "bootstrap_ci_method": "paired_percentile_numpy_quantile_linear_2.5_97.5",
        "difference_definition": "prediction_minus_truth",
        "icc_definition": "ICC(A,1): two-way mixed, absolute agreement, single measurement; fixed two methods",
        "loa_definition": "bias_plus_or_minus_1.96_times_difference_sample_sd_ddof1",
        "loa_normality_assumption": "normal-theory limits assume approximately Gaussian differences; inspect paired BA and difference distribution",
        "statistical_limitation": "Sample-group resampling does not account for unverified repeated patient identity; intervals are not established patient-population confidence intervals.",
    }
    draws = None
    if n >= 2 and bootstrap_repeats:
        draws = np.empty((bootstrap_repeats, len(METRICS)), dtype=np.float64)
        rng = np.random.default_rng(seed)
        # Bound working memory; sequence is invariant to batch size.
        batch = max(1, min(32, 524288 // n))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for first in range(0, bootstrap_repeats, batch):
                count = min(batch, bootstrap_repeats - first)
                indices = rng.integers(0, n, size=(count, n))
                draws[first:first + count] = _estimate(x[indices], y[indices])
    for j, metric in enumerate(METRICS):
        value = float(point[j]) if math.isfinite(point[j]) else None
        reason = None if value is not None else _reason(metric, n)
        result[metric] = value
        result[metric + "_reason"] = reason
        result[metric + "_ci95_low"] = None
        result[metric + "_ci95_high"] = None
        valid = int(np.count_nonzero(np.isfinite(draws[:, j]))) if draws is not None else 0
        attempted = bootstrap_repeats if draws is not None else 0
        result[metric + "_bootstrap_valid"] = valid
        result[metric + "_bootstrap_undefined"] = attempted - valid
        if value is None:
            ci_reason = "point_estimate_undefined:" + reason
        elif bootstrap_repeats == 0:
            ci_reason = "bootstrap_disabled"
        elif n < 2:
            ci_reason = "fewer_than_two_valid_paired_sample_groups"
        elif valid != bootstrap_repeats:
            # Do not silently condition ICC intervals on nondegenerate draws.
            ci_reason = "undefined_bootstrap_replicates"
        else:
            ci = np.quantile(draws[:, j], [0.025, 0.975], interpolation="linear")
            result[metric + "_ci95_low"], result[metric + "_ci95_high"] = map(float, ci)
            ci_reason = None
        result[metric + "_ci95_reason"] = ci_reason
    return result
