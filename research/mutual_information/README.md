# Regional mutual information

The CPU estimator operates on paired RGB and target vectors in native pixel coordinates. For discrete 8-bit images it first dequantizes each sampled observation with deterministic, pixel- and modality-specific noise. KSG-1 estimates are reported in bits. The study computes the sample standard deviation across three pixel draws.

Run the synthetic tests from the repository root:

```sh
python research/mutual_information/test_mi.py -v
```

The tests construct four synthetic modalities and an F/P/D/O partition, run all 54 estimates in the original six-comparison CPU protocol, and exercise sampling, resume, and invalid-input handling. To use that protocol with authorized data:

```sh
python research/mutual_information/estimator/mi_atlas_six_metrics_20261025.py --phase all --manifest /path/to/manifest.json --output-dir /path/to/new-output
```

The module docstring describes the manifest fields and mask checks. The test fixture gives a minimal working example. For continuous observations, the underlying estimator is `tools/mi_knn_estimator_v1.py`.

The full TEST experiment extends the comparisons to RGB–UV, RGB–PLH and RGB–PLV in each eligible region. `full_test/mi_work.py` holds the GPU execution and aggregation, including the group-fraction and neighbour-count analyses. `ita/` and `ita_extension/` provide the stratified analysis. The original full-cohort workflows read frozen cohort orders, sampling metadata and cache receipts from the controlled study tree. The final plotted cohort aggregates are in [the Figure 2 inputs](../../figures/inputs/figure2/).

The [ITA implementation](../ita_labelling/uv_rgb_ita_full_f_20261025.py) converts native RGB inside the original F mask to CIELAB, takes the componentwise median, and derives the angle and stratum. `evaluate_full_f(image, mask)` accepts a PIL RGB image and a same-size NumPy uint8 mask with values 0 and 255. The [resolution experiment](../resolution/) uses the same eligible groups across scale factors and recomputes region support at each scale.
