# UV signal object measurements

[`metrics.py`](metrics.py) computes 8-connected components, unweighted pixel-centre centroids, counts, signal-mask area fractions and encoded contrast burden on the frozen facial support. `measure_signal` takes truth and prediction signal masks, their contrast maps, a valid-face mask, five anatomical ROI maps, a group index and a P/D signal label. It measures each prediction from its own signal mask. `match_centres` performs maximum-cardinality matching followed by minimum total Euclidean distance among the admissible pairs. Tolerances use grid pixels.

[`detector.py`](detector.py) implements study-side scaling and binds external detector components by digest. Running the original masks from raw data requires the licensed third-party porphyrin component and the source geometry pipeline. Supply their outputs to run object metrics directly. [`validation_metrics.py`](validation_metrics.py) contains independent checks. [`collector.py`](collector.py) accepts the per-model records root, catalog, spatial-subset group list and output directory.

The [smoke test](../uv_calibration/smoke_test.py) measures synthetic displaced components, including an exact one-pixel centroid error. The [full-cohort spatial extension](../uv_spatial/README.md) reuses the same object metrics.
