"""Full-face RGB paired with UV and PLH, with one difference per acquisition.

Example: python paired_capture_mi.py features.json --workers 8

Run against native frozen feature NPZ files. The input is a JSON array with
one row per eligible acquisition, in its original order. Required row fields
are sample_id, native_width, and feature (a path/sha256 object). Paths can be
relative to the JSON file. The before-pooling values are distinct from the
pooled-pixel estimates displayed by the grouped bars in Figure 2a.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

DOMAIN_ORDER = ("F", "P", "D", "O")
MODALITY_ORDER = ("rgb", "uv", "plv", "plh")
SEEDS = (20261025, 20261026, 20261027)


def acquisition_mi(sample_id, width, arrays, dequantize, ksg):
    """Return (UV, PLH) mean MI over the three frozen draws of 1,000 pixels."""
    coords = arrays["linear_indices"]
    values = arrays["full_values"]
    if coords.shape != (3, 4, 1000) or values.shape != (3, 4, 1000, 3):
        raise ValueError("Unexpected frozen feature dimensions")
    result = np.empty((3, 2))
    for draw in range(3):
        indices = coords[draw, DOMAIN_ORDER.index("F")]
        def pixels(modality):
            return dequantize(values[draw, MODALITY_ORDER.index(modality)],
                              indices, sample_id, width, modality)
        x = pixels("rgb")
        for i, target in enumerate(("uv", "plh")):
            result[draw, i] = ksg(x, pixels(target), k=5, workers=1)["mi_bits"]
    return result.mean(axis=0)


def summarize(pairs):
    """Two-sided conditional exact binomial sign test, omitting exact ties."""
    pairs = np.asarray(pairs, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.isfinite(pairs).all():
        raise ValueError("Expected a finite UV/PLH pair for every acquisition")
    delta = pairs[:, 1] - pairs[:, 0]
    plus = int(np.count_nonzero(delta > 0))
    minus = int(np.count_nonzero(delta < 0))
    n = plus + minus
    j = np.arange(min(plus, minus) + 1, dtype=float)
    logcdf = logsumexp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)) - n * math.log(2)
    logp = min(0.0, math.log(2) + float(logcdf))
    log10p = logp / math.log(10)
    exponent = math.floor(log10p)
    return dict(
        domain="F", contrast="MI(RGB,PLH) minus MI(RGB,UV)", unit="paired acquisition",
        k=5, pixels_per_acquisition_per_draw=1000, pixel_seeds=list(SEEDS),
        n_acquisitions=len(delta), n_positive=plus, n_negative=minus, n_ties=len(delta)-n,
        median_uv_bits=float(np.median(pairs[:, 0])), median_plh_bits=float(np.median(pairs[:, 1])),
        median_paired_difference_bits=float(np.median(delta)),
        method="two-sided exact sign test conditional on non-tied acquisitions",
        null="P(paired difference > 0 | nonzero difference) = 0.5",
        log10_p=log10p, p_scientific=f"{10**(log10p-exponent):.8f}e{exponent:+d}")


def evaluate_row(item):
    from estimator.mi_atlas_six_metrics import dequantize_native, ksg_mi
    row, root = item
    path = root / row["feature"]["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["feature"]["sha256"]:
        raise ValueError(f"Feature checksum differs at {path}")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        if not np.array_equal(arrays["seeds"], SEEDS):
            raise ValueError("Unexpected pixel draw seeds")
        if int(arrays["native_shape"][1]) != row["native_width"]:
            raise ValueError("Native image width differs from manifest")
        return acquisition_mi(row["sample_id"], row["native_width"], arrays, dequantize_native, ksg_mi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf8"))
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Use exactly one entry per eligible acquisition")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pairs = list(pool.map(evaluate_row, ((row, args.manifest.parent) for row in rows), chunksize=8))
    print(json.dumps(summarize(pairs), indent=2))


if __name__ == "__main__":
    main()
