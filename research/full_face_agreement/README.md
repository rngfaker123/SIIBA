# Five-region full-face agreement

[`frozen_f_partition.py`](frozen_f_partition.py) takes the original fractional full-face support and its five anatomical seed masks. `partition_frozen_f(f_weights, old_weights)` returns integer labels and quality-control counts. Labels 1–5 follow the order forehead, image-left cheek, image-right cheek, nose and chin. Seeded components use four-neighbour geodesic distances. Disconnected unseeded islands use distance to the fixed seeded regions. Partition weights sum cell-by-cell to the original F.

[`measure3b.py`](measure3b.py) measures fractional-weighted RGB channel means for each region and the original F. [`agreement_stats.py`](agreement_stats.py) computes ICC(A,1), paired bias, the sample SD of differences, Bland–Altman limits and paired-group bootstrap intervals. [`collect_fullf_3b.py`](collect_fullf_3b.py) requires the official ordered cohort, technical-missingness evidence and nonoverlapping per-group measurement shards. [`original_table3b_for_F_only.tsv`](original_table3b_for_F_only.tsv) is the pre-partition aggregate reference used to check the full-F result.

Run the array-only geometry tests with NumPy and SciPy.

```bash
python research/full_face_agreement/test_frozen_f_partition.py
```

For the archived TEST experiment, supply the original 1024-grid F and seed masks, native truth and predictions for UV/PLH/PLV, and the study's signed selection and catalog. [`run_uv.py`](run_uv.py) and [`run_pl.py`](run_pl.py) show how measurements were bound to those sources.
