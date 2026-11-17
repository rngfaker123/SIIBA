# UV reconstruction measurements

Run the paired image-level metrics from the repository root.

```bash
python research/uv_reconstruction/evaluate_group.py group.npz --target uv
```

Provide float RGB `truth` and `prediction` in [0,1] with shape (3,1024,1024), a positive `sigma` image of shape (1024,1024), and float32 fractional P, D, O and F masks on the same grid. The three disjoint signal-domain weights sum to F. The archived analysis projected masks from the true UV image and passed only RGB to the generator. The command reads precomputed predictions, computes L1, MSE, PSNR, SSIM and mean sigma per region, and returns JSON. It retains the SSIM valid-center weighting and the zero-MSE and empty-region reasons.

[`regional_metrics/regional_image_metrics.py`](regional_metrics/regional_image_metrics.py) binds the archived P/D/O/F measures to the numerical helpers in [tools](regional_metrics/tools). [three_unets](three_unets) holds the single-target U-Net, multi-target U-Net and U-Net++ TEST worker and aggregate path. [swin_ldm](swin_ldm) contains the Swin/LDM model loader, one-forward measurement bridge and collector. Published cohort means weight valid groups equally. A full model replay needs the frozen manifest, trained weights, image transformations, signed stage receipts and independently generated region masks.

Public study path examples use `/study/siiba` for the archive and `/study/images` for image inputs. Archive workers verify historical source and input digests and expect the experiment directories recorded in their manifests. Use the array entry point for a new collection of predictions.
