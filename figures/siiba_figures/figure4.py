"""Revise Figure 4 using the accepted uncertainty and calibration records."""
import csv
import json
import hashlib
from pathlib import Path
from .input_paths import INPUT_ROOT

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from .style import (FONT, REGION_ORDER, REGION_COLORS as COLORS, REGION_NAMES as NAMES,
                    REGION_MARKERS, MODEL_ORDER as MODEL_NAMES, MODEL_COLORS,
                    MODEL_MARKERS as MARKERS)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = INPUT_ROOT / "figure4"
MODELS = ("unet_single", "unet_multi", "unetplusplus", "swin", "ldm")
RETAINED = (50, 80, 90, 100)


def verify_inputs():
    """Check source hashes."""
    manifest = json.loads((ASSETS / "provenance.json").read_text(encoding="utf8"))
    for item in manifest["inputs"]:
        path = ASSETS / item["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['file']}")


def load_risk_values():
    """Read selection estimates."""
    with (ASSETS / "Table4B_plotted.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {}
    for row in rows:
        key = (row["analysis"], row["region_or_task"], row["metric"],
               row["method"], row["model"])
        mean = float(row["mean"])
        low, high = float(row["CI95_low"]), float(row["CI95_high"])
        if key in values or not low <= mean <= high:
            raise ValueError(f"Invalid estimate {key}")
        values[key] = np.array([mean, low, high])
    return values


def build(output_dir):
    """Render the accepted Figure 4."""
    verify_inputs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context():
        return _render(output_dir)


def _render(OUT):
    VALUES = load_risk_values()
    with (ASSETS / "Figure4_verified_data.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    with (ASSETS / "Table4A_group_distributions.json").open(encoding="utf8") as stream:
        distributions = {(r["metric"], r["model"], r["region"]): r for r in json.load(stream)}

    INK = "#202126"
    GRID = "#E4E3E3"
    REGIONS = REGION_ORDER[:3]
    RAW, CAL = "Gaussian_1.96_sigma", "group_conformal_sigma"
    mpl.rcParams.update({
        "font.family": FONT, "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "axes.labelcolor": INK, "axes.edgecolor": "#666970", "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False,
    })
    WIDTH, HEIGHT = 8.5, 8.72
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")


    def axat(x, y, width, height):
        return fig.add_axes((x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT))


    def label(x, y, text, **kwargs):
        return fig.text(x / WIDTH, 1 - y / HEIGHT, text, va="center", **kwargs)


    def heading(x, y, letter, title):
        label(x, y, letter, fontsize=14, weight="bold")
        label(x + .22, y, title, fontsize=11, weight="bold")


    def lattice(ax, axis="both"):
        ax.set_axisbelow(True)
        ax.grid(axis=axis, color=GRID, lw=.5)
        ax.tick_params(pad=2.5, labelsize=8)


    def read(section, model, region, method):
        row, = [r for r in records if
                (r["table"], r["model"], r["region_or_task"], r["method"])
                == (section, model, region, method)]
        return np.array([float(row[k]) for k in ("estimate", "ci_low", "ci_high")])


    heading(.10, .17, "a", "Measured UV, uncertainty and error")
    images = [
        ("measured_uv.png", "Measured UV"),
        ("uncertainty.png", "Predicted σ"),
        ("absolute_error.png", "Absolute UV error"),
    ]
    for i, (filename, title) in enumerate(images):
        x = .12 + 1.42 * i
        label(x + .685, .46, title, ha="center", fontsize=8)
        ax = axat(x, .62, 1.37, 1.37)
        ax.axis("off")
        with Image.open(ASSETS / filename) as image:
            ax.imshow(np.asarray(image), interpolation="none")
    label(1.66, 2.16, "σ and error", fontsize=8, ha="right")
    ax = axat(1.86, 2.12, 1.15, .075)
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap="gray", vmin=0, vmax=1,
              aspect="auto", extent=(0, .14, 0, 1))
    ax.set_yticks([])
    ax.set_xticks((0, .14), ["0", "≥0.14"])
    ax.tick_params(axis="x", labelsize=8, length=1.5, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.4)

    heading(4.50, .17, "b", "Mean uncertainty by region")
    table_regions = ("D", "P", "O")
    table_names = {"UNet single": "Single U-Net", "UNet multi": "Multi U-Net",
                   "UNet++": "U-Net++", "Swin": "Swin", "LDM": "LDM"}
    sigma_table = {}
    with (ASSETS / "Table4A.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        assert next(reader) == ["Mean σ", *table_regions]
        for name, *cells in reader:
            if not name:
                break
            assert len(cells) == 3
            for region, cell in zip(table_regions, cells):
                sigma_table[table_names[name], region] = tuple(map(float, cell.split("±")))
    assert len(sigma_table) == 15

    ax_b = axat(4.95, .74, 3.35, 1.23)
    lattice(ax_b, "y")
    ax_b.set(ylim=(0, .125), yticks=[0, .04, .08, .12], xlim=(-.50, 4.50))
    ax_b.set_yticklabels(["0", "0.04", "0.08", "0.12"])
    ax_b.set_xticks(range(5), ["Single\nU-Net", "Multi\nU-Net", "U-Net++", "Swin", "LDM"])
    ax_b.set_ylabel("Mean predicted σ", fontsize=8, labelpad=3)
    ax_b.tick_params(axis="x", length=0, pad=4, labelsize=8)
    for region, offset in zip(REGIONS, [-.24, 0, .24]):
        means, sds = np.asarray([sigma_table[model, region] for model in MODEL_NAMES]).T
        ax_b.bar(np.arange(5) + offset, means, width=.22, color=COLORS[region],
                 linewidth=0, label=f"{region}  {NAMES[region]}", yerr=sds, zorder=3,
                 error_kw={"ecolor": "#30343A", "elinewidth": .8, "capsize": 2, "capthick": .8})
    ax_b.legend(loc="lower center", bbox_to_anchor=(.5, 1.05), ncol=3, fontsize=8,
                handlelength=.9, handletextpad=.3, columnspacing=.65, borderaxespad=0)

    heading(.10, 2.57, "c", "Within-face uncertainty–error correlation")
    peak = max((np.asarray(distributions["rank_corr", model, region]["counts"])
                / distributions["rank_corr", model, region]["n"] / .02).max()
               for model in MODEL_NAMES for region in REGIONS)
    for j, region in enumerate(REGIONS):
        ax = axat(1.0 + 2.47 * j, 3.00, 2.29, 1.10)
        ax.set(xlim=(-1, 1), ylim=(-.23, 4.96), xticks=(-1, 0, 1), yticks=np.arange(4, -1, -1))
        ax.set_yticklabels(MODEL_NAMES if j == 0 else [""] * 5)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=8)
        ax.tick_params(axis="x", pad=2, labelsize=8)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color="#aaa8a5", lw=.65, ls=(0, (2, 2)), zorder=0)
        ax.set_title(f"{region}  {NAMES[region]}", fontsize=8.5, color=COLORS[region], weight="bold", pad=5)
        for i, model in enumerate(MODEL_NAMES):
            dist = distributions["rank_corr", model, region]
            edges, counts = np.asarray(dist["edges"]), np.asarray(dist["counts"])
            density = counts / np.diff(edges) / dist["n"]
            assert counts.sum() == dist["n"] and abs(np.diff(edges).mean() - .02) < 1e-9
            y = 4 - i
            heights = np.r_[density, density[-1]] * .76 / peak
            ax.hlines(y, -1, 1, color="#cfcecc", lw=.4, zorder=0)
            ax.fill_between(edges, y, y + heights, step="post", color=COLORS[region], alpha=.55, lw=0)
            ax.step(edges, y + heights, where="post", color=COLORS[region], lw=.7)
            ax.vlines(dist["q50"], y, y + .12, color="white", lw=.8)
            ax.plot(dist["mean"], y, marker="o", ms=3.1, color=INK, mec="white", mew=.3, zorder=3)
    label(4.60, 4.36, "Within-image Spearman ρ", ha="center", fontsize=8)
    label(8.28, 2.57, "●  Mean", ha="right", fontsize=8)

    heading(.10, 4.72, "d", "Coverage and interval width before and after calibration")
    method_legend = [
        Line2D([], [], ls="", marker="o", mec="#41444b", mfc=fill, ms=4.5, label=title)
        for title, fill in [("Original", "white"), ("Conformal", "#41444b")]]
    fig.legend(handles=method_legend, loc="center right", bbox_to_anchor=(8.29 / WIDTH, 1 - 4.99 / HEIGHT),
               ncol=2, fontsize=8, handlelength=.75, columnspacing=1.2, handletextpad=.35, borderpad=0)
    facets = [("P", "4B.1", 1.0), ("D", "4B.1", 2.84),
              ("P", "4B.2", 4.72), ("D", "4B.2", 6.56)]
    for column, (region, section, left) in enumerate(facets):
        ax = axat(left, 5.40, 1.70, 1.0)
        lattice(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.set(ylim=(4.46, -.49), yticks=range(5))
        ax.set_yticklabels(MODEL_NAMES if column == 0 else [""] * 5)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=8)
        ax.set_title(f"{region}  {NAMES[region]}", color=COLORS[region], weight="bold", fontsize=8.5, pad=5)
        coverage = section == "4B.1"
        if coverage:
            ax.set(xlim=(74, 102), xticks=(75, 85, 95, 100))
            ax.axvline(95, color="#777a80", lw=.75, ls=(0, (3, 2)))
        else:
            ax.set(xlim=(0, 1.35), xticks=(0, .4, .8, 1.2), xticklabels=["0", "0.4", "0.8", "1.2"])
        for i, model in enumerate(MODEL_NAMES):
            means = [read(section, model, region, method)[0] * (100 if coverage else 1) for method in (RAW, CAL)]
            ax.plot(means, [i, i], color=COLORS[region], alpha=.8, lw=1.0)
            for value, fill in zip(means, ("white", COLORS[region])):
                ax.plot(value, i, marker=REGION_MARKERS[region], ls="", mfc=fill,
                        mec=COLORS[region], mew=.9, ms=4.4)
    label(2.77, 6.66, "Pixel-component coverage (%)", ha="center", fontsize=8)
    label(6.49, 6.66, "Unclipped width (encoded intensity)", ha="center", fontsize=8)

    heading(.10, 7.01, "e", "Risk across four retention levels")
    model_legend = [Line2D([], [], color=color, marker=marker, lw=1.1, ms=4.0, label=model)
                    for color, marker, model in zip(MODEL_COLORS, MARKERS, MODEL_NAMES)]
    fig.legend(handles=model_legend, loc="center", bbox_to_anchor=(4.42 / WIDTH, 1 - 7.30 / HEIGHT),
               ncol=5, fontsize=8, columnspacing=1.6, handlelength=1.4, handletextpad=.4, borderpad=0)
    risk_titles = ("P · intensity", "D · intensity", "P · count", "D · count")
    risk_ticks = ((.08, .10, .12), (.04, .05, .06), (300, 400, 500, 600), (60, 90, 120, 150))
    risk_limits = ((.077, .124), (.032, .063), (260, 615), (46, 154))
    for i, (task, title) in enumerate(zip(("P", "D", "P_count", "D_count"), risk_titles)):
        ax = axat((.58, 2.57, 4.74, 6.66)[i], 7.71, (1.68, 1.61, 1.60, 1.60)[i], .63)
        lattice(ax, "y")
        ax.set(xlim=(47, 103), xticks=RETAINED, ylim=risk_limits[i], yticks=risk_ticks[i])
        for model, color, marker in zip(MODELS, MODEL_COLORS, MARKERS):
            estimates = [VALUES["selective_risk_fixed_EVAL_order", task, "mean_absolute_error",
                                f"mean_sigma_keep_{retention / 100:.1f}", model] for retention in RETAINED]
            mean, low, high = np.asarray(estimates).T
            ax.errorbar(RETAINED, mean, yerr=[mean - low, high - mean], color=color, marker=marker,
                        markeredgecolor="white", markeredgewidth=.35, ms=4.0,
                        linewidth=1.15, elinewidth=.7, capsize=1.5, zorder=2)
        ax.set_title(title, loc="left", fontsize=8.5, weight="bold", pad=5)
        if i in (0, 2):
            ax.set_ylabel("Regional L1" if i == 0 else "Count MAE", fontsize=8, labelpad=3)
    label(4.42, 8.59, "Evaluation groups retained (%)", ha="center", fontsize=8)

    fig.canvas.draw()
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 8, (text.get_text(), text.get_fontsize())
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"Figure4.{extension}", dpi=400, facecolor="white")
    plt.close(fig)
    with Image.open(OUT / "Figure4.png") as image:
        image.thumbnail((1700, 1800), Image.Resampling.LANCZOS)
        image.save(OUT / "Figure4_preview.png")

    caption_en = """Figure 4. Predictive uncertainty and calibration of UV reconstruction from RGB.
    (a) Measured UV, predicted uncertainty (σ) and absolute reconstruction error from the same calibration record. Black rectangles conceal the eyes.
    (b) Mean predicted σ in porphyrin regions (P), UV spots (D) and other valid skin (O) for Single-target U-Net, Multi-target U-Net, U-Net++, Swin Transformer and the latent diffusion model (LDM). Bars show regional means and error bars show one standard deviation across records.
    (c) Distributions of within-image Spearman correlations between predicted σ and absolute pixel error. Each ridge represents one model and region. Dark points mark means.
    (d) Pixel-component coverage and unclipped interval width in P and D before calibration (open symbols) and after σ-scaled conformal calibration (filled symbols). The dashed lines mark 95% nominal coverage.
    (e) Regional intensity L1 and count mean absolute error at four uncertainty-based sample-retention levels. Points and error bars show mean error and bootstrap 95% confidence intervals. Lower retention keeps records with lower mean predicted uncertainty."""
    caption_cn = """图4 从 RGB 重建 UV 的预测不确定性与校准。
    （a）同一条校准记录的实测 UV、预测不确定性（σ）及重建绝对误差，眼部用黑框遮挡。
    （b）单目标 U-Net、多目标 U-Net、U-Net++、Swin Transformer 与潜在扩散模型（LDM）在卟啉区（P）、UV 斑区（D）和其余有效皮肤（O）的平均预测 σ。柱高为区域平均值，误差线为记录间的一个标准差。
    （c）各模型在每个区域内，逐图计算的预测 σ 与像素绝对误差之间的 Spearman 相关系数分布。深色点标示均值。
    （d）P 和 D 区内校准前（空心符号）与按 σ 缩放的共形校准后（实心符号）的像素通道分量覆盖率及未截断区间宽度。虚线标示 95% 名义覆盖率。
    （e）按预测不确定性保留不同样本比例时的区域强度 L1 与计数平均绝对误差。点及误差线分别表示平均误差和自助法 95% 置信区间。较低保留率对应保留平均预测不确定性较低的记录。"""
    (OUT / "Figure4_caption_EN.txt").write_text(caption_en.replace("\n    ", "\n") + "\n", encoding="utf8")
    (OUT / "Figure4_caption_CN.txt").write_text(caption_cn.replace("\n    ", "\n") + "\n", encoding="utf8")
    return [OUT / name for name in ("Figure4.pdf", "Figure4.svg", "Figure4.png",
            "Figure4_preview.png", "Figure4_caption_EN.txt", "Figure4_caption_CN.txt")]
