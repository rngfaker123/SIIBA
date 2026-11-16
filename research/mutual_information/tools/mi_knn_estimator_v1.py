"""Frozen, training-free k-NN mutual-information estimators.

KSG estimator 1 (Kraskov, Stogbauer & Grassberger, 2004):
    https://arxiv.org/abs/cond-mat/0305641
Ross continuous/discrete estimator (2014), equations 1--2:
    https://doi.org/10.1371/journal.pone.0087357

Inputs are observations, not independent-subject claims. These functions do not
provide confidence intervals or convert repeated points into independent people.
They never clip negative finite-sample MI estimates or secretly add jitter. A
caller must explicitly perform and record any quantization preprocessing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.special import digamma


ESTIMATOR_VERSION = "mi_knn_estimator_v1"
LUMINANCE_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
_LOG2 = float(np.log(2.0))


def _matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("%s must be a non-empty N x d array" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s contains non-finite values; observations were not dropped" % name)
    return np.ascontiguousarray(array)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _duplicate_count(array: np.ndarray) -> int:
    return int(len(array) - len(np.unique(array, axis=0)))


def _assert_no_duplicate_rows(array: np.ndarray, name: str) -> None:
    duplicates = _duplicate_count(array)
    if duplicates:
        raise ValueError(
            "%s has %d duplicate continuous observations; use an explicit frozen "
            "dequantization protocol, not hidden estimator jitter" % (name, duplicates)
        )


def _neighbor_radii(array: np.ndarray, k: int, workers: int) -> np.ndarray:
    """Distances to the kth OTHER point; reject ties at that boundary."""
    count = len(array)
    if count <= k:
        raise ValueError("Need N > k, got N=%d and k=%d" % (count, k))
    # Self is rank 0. Inspect one rank above/below the chosen boundary so that a
    # tied kth radius cannot silently change strict-count conventions.
    ranks = min(count, k + 2)
    distances, _ = cKDTree(array).query(array, k=ranks, p=np.inf, workers=workers)
    radius = distances[:, k]
    tied = radius <= 0
    if k > 1:
        tied |= radius == distances[:, k - 1]
    if ranks > k + 1:
        tied |= radius == distances[:, k + 1]
    if np.any(tied):
        raise ValueError(
            "Tied or zero kth-neighbor radius at %d observations; no silent jitter" % int(tied.sum())
        )
    if not np.isfinite(radius).all():
        raise ValueError("Non-finite neighbor radius")
    return radius


def _ball_counts(array: np.ndarray, radii: np.ndarray, workers: int) -> np.ndarray:
    """Inclusive radii, including self. SciPy >= 1.6 API is required."""
    return np.asarray(
        cKDTree(array).query_ball_point(array, radii, p=np.inf, workers=workers, return_length=True),
        dtype=np.int64,
    )


def _base_result(estimate: float, n: int, k: int, workers: int, estimator: str) -> Dict[str, Any]:
    if not np.isfinite(estimate):
        raise FloatingPointError("Non-finite mutual-information estimate")
    return {
        "estimator": estimator,
        "implementation_version": ESTIMATOR_VERSION,
        "mi_nats": float(estimate),
        "mi_bits": float(estimate / _LOG2),
        "unit": "bits",
        "n": int(n),
        "k": int(k),
        "distance": "chebyshev_max_norm",
        "negative_estimates_clipped": False,
        "jitter_added_inside_estimator": False,
        "dtype": "float64",
        "workers": int(workers),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
    }


def ksg_mi(x: Any, y: Any, k: int = 5, workers: int = 4) -> Dict[str, Any]:
    """KSG-1 I(X;Y), multivariate X/Y, common physical scales, raw bits.

    Uses psi(k)+psi(N)-mean(psi(nx+1)+psi(ny+1)). nx/ny exclude
    self and count distances STRICTLY LESS than the joint kth-neighbor radius.
    nextafter(radius, 0) makes that convention explicit in cKDTree.
    No standardization, normalization, dimensional MI summation, or filtering.
    """
    x = _matrix(x, "x")
    y = _matrix(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same N")
    k = _positive_integer(k, "k")
    workers = _positive_integer(workers, "workers")
    _assert_no_duplicate_rows(x, "x")
    _assert_no_duplicate_rows(y, "y")
    joint = np.concatenate([x, y], axis=1)
    radii = _neighbor_radii(joint, k, workers)
    strict_radii = np.nextafter(radii, 0.0)
    # Counts include self and hence equal nx+1 / ny+1 directly.
    nx_plus_one = _ball_counts(x, strict_radii, workers)
    ny_plus_one = _ball_counts(y, strict_radii, workers)
    estimate = digamma(k) + digamma(len(x)) - np.mean(digamma(nx_plus_one) + digamma(ny_plus_one))
    result = _base_result(float(estimate), len(x), k, workers, "KSG_1")
    result.update({
        "dx": int(x.shape[1]),
        "dy": int(y.shape[1]),
        "marginal_boundary": "strict_less_than_joint_kth_radius",
        "self_in_marginal_digamma_count": True,
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "reference": "https://arxiv.org/abs/cond-mat/0305641",
    })
    return result


def ross_mi(continuous: Any, labels: Any, k: int = 5, workers: int = 4) -> Dict[str, Any]:
    """Ross (2014) I(continuous; discrete), without adaptive-k or clamping.

    Fixed k requires every class to contain at least k+1 observations. The kth
    within-class neighbor is INCLUDED in the full-sample neighbor count m;
    self is EXCLUDED. Thus psi(N)+psi(k)-mean(psi(N_class)+psi(m)).
    Integer/bool/string labels are categorical; floating labels are rejected.
    """
    x = _matrix(continuous, "continuous")
    categorical = np.asarray(labels)
    if categorical.ndim != 1 or len(categorical) != len(x):
        raise ValueError("labels must be a one-dimensional vector of length N")
    if categorical.dtype.kind not in "biuUS":
        raise ValueError("labels must be integer, boolean, or string categories")
    k = _positive_integer(k, "k")
    workers = _positive_integer(workers, "workers")
    _assert_no_duplicate_rows(x, "continuous")
    classes, inverse, class_counts = np.unique(categorical, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least two observed classes")
    if np.any(class_counts <= k):
        raise ValueError("Every class must have > k observations; no class dropped and no adaptive k")
    radii = np.empty(len(x), dtype=np.float64)
    class_n = class_counts[inverse]
    for class_index in range(len(classes)):
        mask = inverse == class_index
        radii[mask] = _neighbor_radii(x[mask], k, workers)
    # Detect additional exact radius ties from OTHER classes too. This is an
    # explicit verification of the continuous-distribution/no-tie convention.
    strict_counts = _ball_counts(x, np.nextafter(radii, 0.0), workers)
    inclusive_counts = _ball_counts(x, radii, workers)
    if not np.all(inclusive_counts - strict_counts == 1):
        raise ValueError("Continuous-distance boundary ties across classes; no silent jitter")
    m = inclusive_counts - 1  # Include kth other point, exclude self.
    if np.any(m < k):
        raise RuntimeError("Inconsistent nearest-neighbor counts")
    estimate = digamma(len(x)) + digamma(k) - np.mean(digamma(class_n) + digamma(m))
    result = _base_result(float(estimate), len(x), k, workers, "Ross_2014")
    probabilities = class_counts / float(len(x))
    result.update({
        "dx": int(x.shape[1]),
        "dy": "categorical",
        "class_counts": {str(label): int(count) for label, count in zip(classes, class_counts)},
        "class_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "full_boundary": "inclusive_kth_within_class_radius_excluding_self",
        "reference": "https://doi.org/10.1371/journal.pone.0087357",
    })
    return result


def dequantize_u8(
    values: Any,
    sample_ids: Sequence[Any],
    point_ids: Sequence[Any],
    modality: str,
    seed: int = 20260817,
) -> np.ndarray:
    """Explicit, row-order-independent (u8 + U[-.5,.5])/255.

    Each noise value is fixed by SHA256(domain, seed, sample_id, point_id,
    modality, channel). No clipping: black/white may lie just outside [0,1].
    Identical source observations use the SAME identifiers when reused; row
    duplication does not manufacture independent jittered observations.
    Point IDs must be unique inside each sample in the supplied array.
    """
    raw = _matrix(values, "values")
    if np.any(raw < 0) or np.any(raw > 255) or np.any(raw != np.floor(raw)):
        raise ValueError("values must contain integer-valued 8-bit intensities, not normalized floats")
    if len(sample_ids) != len(raw) or len(point_ids) != len(raw):
        raise ValueError("sample_ids and point_ids must both have length N")
    if not isinstance(modality, str) or not modality:
        raise ValueError("A non-empty, frozen modality name is required")
    pairs = [(str(sample), str(point)) for sample, point in zip(sample_ids, point_ids)]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate sample/point IDs; do not independently jitter repeated observations")
    output = raw.copy()
    denominator = float(1 << 53)
    for row, (sample, point) in enumerate(pairs):
        for channel in range(raw.shape[1]):
            payload = json.dumps(
                ["mi_u8_dequantization_v1", int(seed), sample, point, modality, channel],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
            uniform = integer / denominator
            output[row, channel] += uniform - 0.5
    return output / 255.0


def luminance(dequantized_rgb: Any) -> np.ndarray:
    """Luma-like scalar from the SAME RGB realization; no additional jitter."""
    values = _matrix(dequantized_rgb, "dequantized_rgb")
    if values.shape[1] != 3:
        raise ValueError("Expected N x 3 RGB values")
    return values @ LUMINANCE_WEIGHTS


def subject_point_permutation(
    subject_ids: Sequence[Any], point_ids: Sequence[Any], seed: int = 20260817,
) -> np.ndarray:
    """Return target-row indices for a whole-subject cyclic derangement.

    A single deterministic subject mapping is applied at EVERY point location,
    preserving within-target-person relationships and the location distribution.
    Requires the same complete set of point IDs for each subject, one row per
    (subject,point). Ragged regional samples must be handled explicitly by the
    caller, e.g. separate per-location analyses, and are never silently filled.
    """
    if len(subject_ids) != len(point_ids) or len(subject_ids) == 0:
        raise ValueError("subject_ids and point_ids need the same non-zero length")
    pairs = [(str(subject), str(point)) for subject, point in zip(subject_ids, point_ids)]
    lookup = {pair: row for row, pair in enumerate(pairs)}
    if len(lookup) != len(pairs):
        raise ValueError("Repeated subject/point row")
    subjects = set(subject for subject, _ in pairs)
    points = set(point for _, point in pairs)
    if len(subjects) < 2:
        raise ValueError("Need at least two different subjects")
    if len(pairs) != len(subjects) * len(points):
        raise ValueError("Ragged subject/point design; no complete global subject permutation exists")
    def digest(subject: str) -> str:
        payload = json.dumps(["mi_subject_permutation_v1", int(seed), subject], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    ordered = sorted(subjects, key=lambda subject: (digest(subject), subject))
    shift_hash = hashlib.sha256(("mi_subject_shift_v1:%d" % int(seed)).encode("ascii")).digest()
    shift = 1 + int.from_bytes(shift_hash[:8], "big") % (len(ordered) - 1)
    mapping = {subject: ordered[(index + shift) % len(ordered)] for index, subject in enumerate(ordered)}
    result = np.asarray([lookup[(mapping[subject], point)] for subject, point in pairs], dtype=np.int64)
    if len(np.unique(result)) != len(result) or any(pairs[row][0] == pairs[target][0] for row, target in enumerate(result)):
        raise RuntimeError("Subject derangement invariants failed")
    return result


def sweep_ksg(x: Any, y: Any, ks: Iterable[int] = (3, 5, 10), workers: int = 4) -> Dict[str, Any]:
    """Convenience wrapper. Caller supplies already-frozen/dequantized arrays."""
    return {str(k): ksg_mi(x, y, k=k, workers=workers) for k in ks}

