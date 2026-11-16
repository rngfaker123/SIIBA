# PLH and PLV paired TEST

Run the per-group CPU measurement for either polarized target. The NPZ holds registered 3 × 1024 × 1024 floating RGB arrays `truth` and `prediction` and the four float32 true-UV-derived region weights P/D/O/F.

```bash
python research/uv_reconstruction/evaluate_group.py group.npz --target plh
python research/uv_reconstruction/evaluate_group.py group.npz --target plv
```

[`pl_measure.py`](pl_measure.py) computes region-weighted L1, MSE, PSNR and SSIM and measures paired channel intensities from precomputed anatomical masks. [`pl_models.py`](pl_models.py) describes architecture-specific checkpoint selection and the one-forward evaluation. [`pl_execution.py`](pl_execution.py) contains the original CPU reference, GPU comparison and inference loop. [`pl_collect.py`](pl_collect.py) verifies complete cohorts and summarizes agreement. [`pl_uv_contrasts.py`](pl_uv_contrasts.py) pairs the original per-group UV and PL measurements for contrasts. Aggregate UV reference cells are in [reference](reference).

To replay the study cohort, supply the 21,899 ordered group identities, both modality-specific truth images, model checkpoints, frozen projection and ROI receipts, and archived model protocol. This public source describes the original study pipeline. The current run's hardware assignments and device-specific resume owner belong to the execution site. The [full-face analysis](../full_face_agreement/README.md) gives the exhaustive five-region recalculation.
