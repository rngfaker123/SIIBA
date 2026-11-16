# SIIBA

SIIBA studies the information shared between RGB skin images and paired ultraviolet (UV) and polarized-light (PLH, PLV) images. The code covers regional mutual information, reconstruction with five model families, anatomical agreement, and the relationship between uncertainty and UV reconstruction error.

| Component | Implementation |
| --- | --- |
| Cohort, masks, skin-tone strata | [Cohort split](research/cohort/), [geometry and masks](research/geometry/), [native full-face ITA](research/ita_labelling/) |
| Shared information and sensitivity | [Regional mutual information](research/mutual_information/), [resolution](research/resolution/), [context probes](research/context/) |
| Model training | [Training commands and configuration files](research/training/README.md), [architectures](research/models/README.md) |
| Reconstruction and agreement | [UV reconstruction](research/uv_reconstruction/README.md), [PL evaluation](research/pl_test/README.md), [full-face anatomical agreement](research/full_face_agreement/README.md) |
| UV objects and spatial controls | [Counts and burden](research/uv_objects/README.md), [spatial matching](research/uv_spatial/README.md), [registration](research/registration/README.md) |
| Uncertainty | [Raw and conformal intervals, coverage and selective risk](research/uv_calibration/README.md) |
| Figures 1–4 | [Plotting instructions, source data and input manifest](figures/README.md) |

## Run the analysis checks

From the repository root, install the CPU analysis dependencies in your Python environment. The commands below use generated arrays or public figure inputs.

```sh
python -m pip install numpy scipy pillow
python research/mutual_information/test_mi.py -v
python research/uv_calibration/smoke_test.py
python figures/render_figures.py --check-public
```

The training guide includes a CPU forward pass for each architecture and a synthetic training update. See the figure guide for the plotting environment and the individual figure commands.

## Study inputs

The original study uses a frozen manifest of paired acquisitions, the registered modality images and their analysis masks. The split contains 76,642 training, 10,948 validation and 21,899 test groups. Metric-specific sample counts depend on geometry and regional support. F denotes the full valid face, P the porphyrin region, D the UV-spot region and O the remaining valid skin. The five anatomical subregions used for intensity agreement form a separate partition of F.

The public repository supplies study code, configurations, plotting inputs and synthetic checks. Clinical images, participant-level manifests and model checkpoints are held separately. The model guide lists the files required to use the original training protocols. Rendering panels that show a clinical example requires the group's image assets. Archived cohort runners additionally refer to frozen protocols, licensed detector inputs and site-specific resource assignments. Their provenance checks depend on those original artifacts.
