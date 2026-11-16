# Registration sensitivity

[`engine.py`](engine.py) evaluates displacement and blur on paired truth and model outputs using the original fractional support. For the oracle translation, the 121 integer shifts in the ±5-pixel grid are ordered by squared radius followed by vertical then horizontal shift. The full-face L1 minimum breaks ties in this frozen order. `cpu_score_grid` independently recomputes the L1 grid using mirrored Gaussian filtering. The computed image metrics use the fixed 26-pixel interior margin.

[`translation`](translation) supplies the archived pairwise engine and data helpers. [`uv_self_control`](uv_self_control) applies the displacement pipeline to the UV truth paired with itself. [`worker.py`](worker.py) and [`collect.py`](collect.py) bind the prediction-only and common-blur controls to frozen per-group inputs and report cohort summaries. Study replay needs registered image pairs, weights, the ordered cohort, prediction receipts and the original zero-shift and blur control records. Interpreting the grid minimum uses it as an oracle sensitivity statistic.

The [numerical smoke test](../uv_calibration/smoke_test.py) creates a translated RGB image and recovers the known displacement by evaluating all 121 shifts, with and without Gaussian blur.
