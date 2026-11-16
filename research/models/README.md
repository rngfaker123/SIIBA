# Model sources

| Model | File | Class |
| --- | --- | --- |
| Single-target U-Net | `uv_uncertainty_unet.py` | `UVUncertaintyUNet` |
| Multi-target U-Net | `joint_uv_uncertainty_unet.py` | `JointUVUncertaintyUNet` |
| U-Net++ | `uv_uncertainty_unetplusplus.py` | `UVUncertaintyUNetPlusPlus` |
| Swin | `swin_model.py` | `SwinUVUncertainty` |
| Conditional LDM | `uv_latent_diffusion_v1.py` and `ldm_sigma_model.py` | `KLUVAutoencoder`, `RGBLatentVDenoiser`, `LDMGaussian` |

`unet_baseline.py` defines the shared convolutional blocks. `matched_training_batching.py` and `highres_patch_dataset.py` provide batch partitioning and image transforms used by the native trainers. Swin and the LDM sigma trainer verify the complete model-source file against its training-time hash.

Training instructions and private input schemas are in [the training guide](../training/README.md).
