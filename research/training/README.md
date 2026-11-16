# Image-to-image model training

The study used paired 1024 × 1024 RGB inputs and UV, PLH or PLV targets. The public trainers retain the original optimizer, losses, full validation selection and partial-batch handling. The configuration files use Linux mount points for the private images and protocols. Training results are written under `/study/siiba/runs`.

| Entry | Target argument | Output and training objective | Training recipe |
| --- | --- | --- | --- |
| `single_unet` | `uv`, `plh`, `plv` | One RGB target mean and one shared-channel variance, fitted by Gaussian NLL | Separate U-Net per target, 2 ranks |
| `multi_unet` | `joint` | Shared encoder and four decoders. UV NLL + PLH L1 + PLV L1 with unit weights | One model predicts all three modalities, 4 ranks |
| `unetplusplus` | `uv`, `plh`, `plv` | One target mean and variance, Gaussian NLL | FP32, UV 4 ranks, PLH 3, PLV 2 |
| `swin` | `uv`, `plh`, `plv` | Swin encoder–decoder, target mean and variance, Gaussian NLL | FP16 backbone, microbatch 1, FP32 validation. The latest PLV continuation used 4 ranks |
| `ldm`, then `sigma` | `uv`, `plh`, `plv` | KL autoencoder, RGB-conditioned latent v-prediction, deterministic DDIM20. A Gaussian head fits conditional variance with the generator frozen | AE and denoiser BF16 forward with FP32 loss, sigma FP32. The latest PLV sigma continuation used 2 ranks |

All five use a global batch of 16 and retain the final two real training samples. The multi-output U-Net selects the checkpoint with the lowest mean validation L1 across the three modalities. The single-target networks select by target L1. The LDM generator stages select by validation L1 and its variance head by validation NLL.

## Environment and private inputs

The CPU smoke test ran on Linux with Python 3.8.12, PyTorch 2.4.1+cu124, NumPy 1.21.2 and Pillow 8.2.0. GPU training requires CUDA and NCCL. The historical model-source checks hash file bytes. Preserve LF line endings in checkout.

- Mount the image root at `/study/images`. Each `*_path` in the private manifest locates the corresponding RGB image in that tree.
- Supply `/study/siiba/data_protocol/dataset_clean_v1.0_20261013/manifest_frozen.csv`. The study trainers verify its SHA-256, 76,642 TRAIN groups and 10,948 VAL groups. Required fields include `sample_id`, `subject_id`, `split_group_id`, `split`, `eligible_primary`, `source_width`, `source_height`, and the path and SHA-256 for each of `rgb`, `uv`, `plh` and `plv`.
- For U-Net++, Swin and LDM, mount the original private reference directory specified in the config. It supplies the original `protocol.json`, `protocol_digest.json`, `train_manifest.json`, `val_manifest.json` and 20 files named `orders/epoch_NNN.json`. Preserve file bytes. The hash checks bind the actual source cohort, order and split identities.
- For LDM phase C, supply the selected AE and denoiser checkpoints, their LAST checkpoints, the two training summaries, the original protocols and split manifests, and the TRAIN posterior latent statistics. Each supplied sigma config identifies the frozen asset paths and hashes. A fresh generator run needs a fresh sigma configuration from `sigma_gate.py`.

Each command below runs from the repository root. For a CPU-only installation the following commands exercise all five model families, verify the published recipe bindings and run the native joint-U-Net synthetic update.

```bash
python research/training/smoke.py
python research/training/run.py multi_unet joint --mode smoke --smoke-device cpu
```

## U-Net training

Use `run.py` for training. It applies the native modality-specific selectors and matches the U-Net++ reduction to the configured number of ranks. For example, to train the single-target PLH U-Net:

```bash
R=research/training/run.py
P=/study/siiba/preflight/single_unet_plh
mkdir -p /study/siiba/preflight /study/siiba/runs
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python "$R" single_unet plh --mode prepare
torchrun --standalone --nproc_per_node=2 "$R" single_unet plh --mode preflight --preflight-output "$P"
torchrun --standalone --nproc_per_node=2 "$R" single_unet plh --mode train --preflight-output "$P" --resume auto
```

For a multi-target run, use `multi_unet joint` and four visible GPUs. For U-Net++, use `unetplusplus` with its target-specific rank count from the table and a distinct preflight directory per run. The `prepare` stage is a one-time CPU operation.

## Four-GPU Swin and two-GPU LDM sigma

Each native preflight runs full-size images and checks checkpoint replay before training. For Swin PLV:

```bash
R=research/training/run.py
P=/study/siiba/preflight/swin_plv_w4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
torchrun --standalone --nproc_per_node=4 "$R" swin plv --mode preflight --preflight-output "$P"
torchrun --standalone --nproc_per_node=4 "$R" swin plv --mode train --preflight-output "$P"
```

For LDM PLV phase C with the supplied frozen-generator configuration:

```bash
R=research/training/run.py
P=/study/siiba/preflight/sigma_plv_w2
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
torchrun --standalone --nproc_per_node=2 "$R" sigma plv --mode preflight --preflight-output "$P"
torchrun --standalone --nproc_per_node=2 "$R" sigma plv --mode train --preflight-output "$P"
```

Pass `--resume` to the train command to restore a checkpoint created by that same published recipe, runtime binding and world size. Production checkpoints from the private historical installation used different paths and source hashes. The historical Swin migration preserved the single-rank checkpoint and validated an uncommitted 16-image update against four ranks before changing the execution binding.

## LDM generator and new sigma configuration

For each target, run AE training, review the preselected 24 validation groups, compute TRAIN posterior-mean latent statistics, then train the v-prediction denoiser. Both generator stages use the assignment file generated for their stage and visible devices. This example starts a fresh two-rank PLV run.

```bash
R=research/training/run.py
G=/study/siiba/runs/ldm_plv
P=/study/siiba/preflight
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$P"
python "$R" ldm plv --mode prepare
for stage in ae latent_stats diffusion; do
  python research/training/assign_ldm_stage.py --stage "$stage" --protocol "$G/protocol.json" --output "$P/ldm_plv_$stage.json"
done
torchrun --standalone --nproc_per_node=2 "$R" ldm plv --stage ae --mode preflight --stage-assignment "$P/ldm_plv_ae.json" --preflight-output "$P/ldm_plv_ae"
torchrun --standalone --nproc_per_node=2 "$R" ldm plv --stage ae --mode train --stage-assignment "$P/ldm_plv_ae.json" --preflight-output "$P/ldm_plv_ae"
python research/training/review_pl_ae.py --generator "$G" --selection /study/siiba/private/fixed_val24.json --output "$G/ae/fixed_val24_summary.json"
torchrun --standalone --nproc_per_node=2 "$R" ldm plv --mode latent-stats --stage-assignment "$P/ldm_plv_latent_stats.json"
torchrun --standalone --nproc_per_node=2 "$R" ldm plv --stage diffusion --mode preflight --stage-assignment "$P/ldm_plv_diffusion.json" --preflight-output "$P/ldm_plv_diffusion"
torchrun --standalone --nproc_per_node=2 "$R" ldm plv --stage diffusion --mode train --stage-assignment "$P/ldm_plv_diffusion.json" --preflight-output "$P/ldm_plv_diffusion"
python research/training/sigma_gate.py --prepare --target plv --formal-root "$G" --b-root "$G" --orders-dir "$G/orders" --source-manifest /study/siiba/data_protocol/dataset_clean_v1.0_20261013/manifest_frozen.csv --data-root /study/images --output-dir /study/siiba/runs/sigma_plv --config "$G/sigma_plv.json"
```

Add `--config "$G/sigma_plv.json"` to both sigma commands above to use this newly verified generator. The PL reconstruction review uses L1 values recorded at the AE's selected validation epoch and requires the private list of 24 predetermined IDs. The original UV study also used its separately documented UV fluorescence review.
