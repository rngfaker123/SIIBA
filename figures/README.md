# SIIBA figures

The four main figures use a shared model order and region palette. The plotting code is in `siiba_figures/`. Aggregated inputs, including the individual points shown in the quantitative panels, are in `inputs/`. The manifest identifies each input and its SHA-256 checksum.

## Build

The repository homepage displays the manuscript version of [Figure 1](../assets/figure1.png). To regenerate it, use the figure command below with `--figures 1` and the group inputs. The separate [workflow schematic](../assets/study-workflow.png) can be regenerated with `python figures/study_workflow.py` using the inputs in this repository.

Use Python 3.11–3.13 and the package versions in `requirements.txt`. Figure 1 also uses Chrome, Edge or Chromium and Poppler’s `pdftoppm`. Set `SIIBA_CHROME` and `SIIBA_PDFTOPPM` if the executables are outside the search path. The original figures use Arial.

From the repository root, run:

```sh
python -m pip install -r figures/requirements.txt
python figures/render_figures.py --private-inputs /path/to/private_inputs --output /path/to/review_outputs
```

The group input folder has subfolders `figure1/` through `figure4/`, with paths listed as `group` in the manifest. It contains clinical image examples and internal audit metadata. The Figure 1 preview on the homepage includes eye masking in all photographs. For a single figure add `--figures 3`. To verify all published inputs on a standalone checkout, run `python figures/render_figures.py --check-public`.

Figure 3 also produces `Figure3_two_pages.pdf`, with panels a–c on page 1 and d–g on page 2 at the same native scale. For the manuscript, save this PDF as `figures/fig3.pdf` and use [fig3.tex](fig3.tex). The template includes both pages at the same width and assigns the corresponding caption to each page.

## Data represented

| Figure | Panels and source |
| --- | --- |
| 1 | Cohort selection, paired capture, six skin-tone strata, valid regions and experimental design. The figure uses the annotated layout derived from the approved SVG. |
| 2 | Regional mutual information, sensitivity to sampling and resolution, and six ITA strata. Five CSVs contain the pooled estimates and sampling variability. The full-face paired sign test uses one RGB–UV and one RGB–PLH estimate per acquisition. See the [aggregate result](inputs/figure2/full_face_paired_mi.json) and [analysis code](../research/mutual_information/paired_capture_mi.py). |
| 3 | Matched UV, PLH and PLV crops from one held-out capture. Each row shows the measured target and five RGB-based reconstructions. The remaining panels show regional L1, paired full-face L1, regional ICC(A,1), UV object performance and spatial-shift sensitivity. |
| 4 | Matched UV, uncertainty and error images, regional mean uncertainty, uncertainty–error correlations, retention risk, coverage and interval width. |

The November 15 acceptance export adds the regional LDM–PLH L1 means and between-capture standard deviations (P 0.022 ± 0.005, D/O/F 0.019 ± 0.004). Panel 3b uses the full-precision means and SDs from the acceptance-linked native summary. The reference table gives them to three decimals. For panel 3d it adds six regional ICC estimates for each of the three channels. Panel 3b now contains 52 estimates, panel 3d contains 234, and panel 3c contains eight verified paired contrasts with 21,422 matched captures each. The LDM UV–PLH estimate comes from the November 17 matched-record calculation using the archived contrast script and 2,000 bootstrap replicates. The cohort-level quantitative tables in this snapshot have no accepted entries for Swin–PLV and LDM–PLV. Panel 3a includes their single-case PLV reconstructions from the validation-selected Swin checkpoint and frozen LDM image generator. Figure 3 uses dashes for unavailable values in panels b and c and plots the available ICC estimates in panel d.
