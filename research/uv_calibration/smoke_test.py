"""Run the public numerical endpoints on synthetic RGB, masks, objects and uncertainty."""
from pathlib import Path
import sys
import tempfile

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
for folder in ("full_face_agreement", "uv_objects", "registration", "pl_test", "uv_reconstruction/regional_metrics"):
    sys.path.insert(0, str(RESEARCH / folder))

from agreement_stats import summarize_pairs
from engine import GRID, cpu_score_grid, pick
from evaluate_arrays import evaluate
from frozen_f_partition import REGION_NAMES, partition_frozen_f
from metrics import ROI_ORDER, extract_objects, make_roi_domains, match_centres, spatial_metrics
from pl_measure import cpu_regions
from tools import uv_architecture_comparison_metrics_v10 as numeric
from regional_image_metrics import measure


def main():
    shape = (32, 34)
    x = np.full((3, *shape), 0.25, np.float32)
    y = x + 0.125
    sigma = np.full(shape, 0.125, np.float32)
    f = np.ones(shape, np.float32)
    p = np.zeros_like(f)
    p[:, :10] = 1
    d = np.zeros_like(f)
    d[:, 10:20] = 1
    weights = dict(P=p, D=d, O=f-p-d, F=f)
    uv = measure(x, y, sigma, weights, numeric, shape)
    pl = cpu_regions(numeric, x, y, weights)
    for region in weights:
        assert uv[region]["l1"] == pl[region]["l1"] == 0.125
        assert uv[region]["mse"] == pl[region]["mse"] == 0.015625
        assert np.isclose(uv[region]["ssim"], pl[region]["ssim"])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cal, ev = root / "cal", root / "eval"
        cal.mkdir()
        ev.mkdir()
        for i in range(19):
            np.savez(cal / f"cal_{i:02d}.npz", truth=x, prediction=y, sigma=sigma, **weights)
        for i, delta in enumerate((0.0625, 0.25)):
            np.savez(ev / f"eval_{i}.npz", truth=x, prediction=x+delta, sigma=sigma*(1+i),
                     P_count_truth=4, P_count_prediction=4+i, **weights)
        result = evaluate(sorted(cal.glob("*.npz")), sorted(ev.glob("*.npz")))
        assert result["thresholds"]["pixel_sigma"]["rank"] == 19
        assert result["thresholds"]["pixel_sigma"]["threshold"] == 1.0
        assert result["simultaneous_group_coverage"]["group_conformal_absolute"] == 0.5
        assert result["regions"]["F"]["intervals"]["group_conformal_sigma"]["width"] == 0.375
        assert result["count_risk"]["P"][-1]["risk"] == 0

    seeds = {}
    for j, name in enumerate(REGION_NAMES):
        seed = np.zeros_like(f)
        seed[8 + j, 4 + 5*j] = 1.0
        seeds[name] = seed
    labels, _ = partition_frozen_f(f, seeds)
    assert set(np.unique(labels)) == set(range(1,6))
    reassembled = sum(np.where(labels == i, f, 0.) for i in range(1, 6))
    np.testing.assert_array_equal(f, reassembled)
    a = np.zeros(shape, bool)
    a[8, 10] = True
    b = np.roll(a, 1, axis=1)
    roi = {name: (labels == i) for i, name in enumerate(ROI_ORDER, 1)}
    domains = make_roi_domains(f > 0, roi)
    ta, pb = [extract_objects(mask, f_mask=f > 0, domains=domains)["centres_yx"] for mask in (a,b)]
    assert match_centres(ta, pb, 2)["mean_center_error"] == 1.0
    spatial = spatial_metrics(a, b, f > 0, roi, group_index=0, signal="P")
    assert spatial["by_tolerance"]["2"]["f1"] == 1.0
    assert spatial["dice"] == 0
    rng = np.random.default_rng(14)
    texture = rng.random((3, 72, 72), dtype=np.float32)
    shifted = np.roll(texture, shift=(1,-2), axis=(1,2))
    one = np.ones((72,72), np.float32)
    weights_72 = dict(P=0.25*one, D=0.25*one, O=0.5*one, F=one)
    pair = dict(truth=texture, predictions=shifted[None], weights=weights_72)
    for blur in (0.0, 1.0):
        chosen, minimum = pick(cpu_score_grid(pair, blur))
        assert GRID[chosen[0]] == (2, -1)
        np.testing.assert_allclose(minimum, 0, atol=1e-12)
    agreement = summarize_pairs([1,2,3,4], [1,2,3,4], bootstrap_repeats=0)
    assert agreement["icc_a1"] == 1.0
    print("PASS: regional UV/PL, weighted intervals and risk, object matching, registration selection, five-region full-F and agreement")


if __name__ == "__main__":
    main()
