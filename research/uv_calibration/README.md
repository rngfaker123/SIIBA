# Figure 4 uncertainty, split conformal calibration and selective risk

The study compares a model's predicted UV RGB mean and its per-pixel Gaussian standard deviation against the paired UV image. Run each model separately using the same prospectively assigned sample-group split. The archived study retrospectively split 21,899 TEST groups into 10,949 CAL and 10,950 EVAL groups before excluding invalid geometry. A group is the unit of calibration and resampling. Patient-level independence was unverified.

From the repository root, Python 3.10+, NumPy and SciPy suffice for both public commands.

```bash
python research/uv_calibration/smoke_test.py
python research/uv_calibration/evaluate_arrays.py --cal inputs/CAL --eval inputs/EVAL > calibration_summary.json
```

Each NPZ holds one geometry-valid group. The filename stem serves as its unique group key. Provide disjoint CAL and EVAL directories. The program reads one model at a time and writes aggregate JSON to stdout.

| NPZ field | Required content |
|---|---|
| `truth`, `prediction` | Encoded RGB in [0, 1], shape (3, H, W). The study used 1024 × 1024. |
| `sigma` | Positive Gaussian standard deviation, shape (H, W), in the same encoded-intensity units. |
| `P`, `D`, `O`, `F` | Frozen fractional supports, shape (H, W). Porphyrin (P), dark spot (D) and the remaining facial region (O) sum to the full valid face (F). |
| `P_count_truth`, `P_count_prediction`, `D_count_truth`, `D_count_prediction` | Optional EVAL integer scalars from the object detector. Supply each observed truth/prediction pair together. |

For each calibration group, the estimator finds the weighted 95th percentile of absolute error divided by sigma within each nonempty region, pooling the three channel components with equal weight. The maximum across regions is the group's score. With *n* calibration groups, the frozen multiplier is the order statistic at rank `ceil(0.95 × (n + 1))`. A rank larger than *n* yields an infinite interval. The absolute-error comparator uses the same procedure without dividing by sigma.

The output includes Gaussian, scale-normalized conformal and constant-width conformal results. Coverage is the fraction of weighted RGB components covered, averaged equally across eligible EVAL groups. Width uses the original, unclipped interval. The joint score event and its direct weighted-coverage check are reported separately. Intensity-risk curves rank groups by the region's mean sigma. Count-risk curves rank groups by full-face mean sigma and measure absolute count error. Empty regions and infinite widths remain explicit.

[`numerics.py`](numerics.py) contains the original CPU and CUDA scores, finite-sample thresholds, scalar-task equations and risk curves. [`conformal_phase.py`](conformal_phase.py) contains the archive's paired 2,000-draw group bootstrap, frozen retention orders and six object-task summaries. The array command returns point estimates. The archived collector additionally requires the ordered split, technical-missingness records, model-specific outputs and signed cohort manifests. The published chart uses the released table of estimates and confidence intervals in [figures](../../figures/README.md).

The smoke test constructs all arrays in memory or in a temporary directory. It checks known results for UV/PL reconstruction, interval width, intensity and count risk, spatial object matching, candidate-shift selection, full-face partition conservation and ICC(A,1).
