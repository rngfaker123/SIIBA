# SIIBA figures

The four main figures use a shared model order and region palette. The plotting code is in `siiba_figures/`. Aggregated inputs, including the individual points shown in the quantitative panels, are in `inputs/`. The manifest identifies each input and its SHA-256 checksum.

## Build

Use Python 3.11–3.13 and the package versions in `requirements.txt`. Figure 1 also uses Chrome, Edge or Chromium and Poppler’s `pdftoppm`. Set `SIIBA_CHROME` and `SIIBA_PDFTOPPM` if the executables are outside the search path. The original figures use Arial.

From the repository root, run:

```sh
python -m pip install -r figures/requirements.txt
python figures/render_figures.py --private-inputs /path/to/private_inputs --output /path/to/review_outputs
```

The group input folder has subfolders `figure1/` through `figure4/`, with paths listed as `group` in the manifest. It contains clinical image examples and internal audit metadata. Generated figures that include these photographs belong in the group's review directory. For a single figure add `--figures 3`. To verify all published inputs on a standalone checkout, run `python figures/render_figures.py --check-public`.

## Data represented

| Figure | Panels and source |
| --- | --- |
| 1 | Cohort selection, paired capture, six skin-tone strata, valid regions and experimental design. The figure uses the annotated layout derived from the approved SVG. |
| 2 | Regional mutual information, sensitivity to sampling and resolution, and six ITA strata. The five published CSVs contain the plotted estimates and sampling variability. |
| 3 | Five models’ measured and predicted UV crops, regional L1, paired full-face L1, regional ICC(A,1), UV object performance and spatial-shift sensitivity. The case bundle contains one shared crop and six source-verified pixel arrays. |
| 4 | Matched UV, uncertainty and error images, regional mean uncertainty, uncertainty–error correlations, retention risk, coverage and interval width. |

The November 15 acceptance export adds the regional LDM–PLH L1 means and between-capture standard deviations (P 0.022 ± 0.005, D/O/F 0.019 ± 0.004). Panel 3b uses the full-precision means and SDs from the acceptance-linked native summary. The reference table gives them to three decimals. For panel 3d it adds six regional ICC estimates for each of the three channels. Panel 3b now contains 52 estimates, panel 3d contains 234, and panel 3c contains eight verified paired contrasts with 21,422 matched captures each. The LDM UV–PLH estimate comes from the November 17 matched-record calculation using the archived contrast script and 2,000 bootstrap replicates. Swin–PLV and LDM–PLV have no accepted entries in this snapshot. Figure 3 uses dashes for unavailable values in panels b and c and plots the available ICC estimates in panel d.
