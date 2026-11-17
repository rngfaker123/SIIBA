"""Build the README workflow using only the public study illustrations."""
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from siiba_figures.style import FONT, REGION_COLORS, REGION_NAMES, REGION_ORDER


FIGURES = Path(__file__).resolve().parent
ILLUSTRATIONS = FIGURES / "inputs" / "figure1"
OUTPUT = FIGURES.parent / "assets" / "study-workflow.png"
INK, ACCENT, RULE = "#192B3D", "#285783", "#BECBD5"


def build():
    width, height = 1000, 555
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"]})
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")

    def label(x, y, value, size=11, **options):
        ax.text(x, y, value, fontsize=size, color=options.pop("color", INK),
                va="center", **options)

    def segment(xs, ys, color=RULE, **options):
        ax.plot(xs, ys, color=color, linewidth=options.pop("lw", 1.2), **options)

    def arrow(start, finish, color=RULE):
        ax.annotate("", xy=finish, xytext=start, arrowprops={
            "arrowstyle": "-|>", "color": color, "lw": 1.1,
            "mutation_scale": 9, "shrinkA": 0, "shrinkB": 0})

    def heading(x, y, number, title):
        label(x, y, str(number), 16, color=ACCENT, weight="bold")
        label(x + 28, y, title, 16, weight="bold")

    def illustration(image, x, y, w, h):
        ax.imshow(image, extent=(x, x + w, y + h, y), interpolation="lanczos")

    with Image.open(ILLUSTRATIONS / "assets" / "acquisition_device.png") as device:
        illustration(device, 16, 8, 123, 123)
    label(160, 30, "Four modalities in one acquisition", 18, weight="bold")
    modalities = [
        (160, "RGB", "White light"), (362, "UV", "UV-excited fluorescence"),
        (565, "PLH", "Parallel-polarized light"), (775, "PLV", "Cross-polarized light"),
    ]
    for x, key, name in modalities:
        label(x, 76, key, 15, color=ACCENT, weight="bold")
        label(x, 103, name, 11)

    segment([500, 500], [123, 143])
    segment([250, 750], [143, 143])
    for x in [250, 750]:
        arrow((x, 143), (x, 161))
    segment([500, 500], [176, 535], color="#DFE5EB", lw=.8)

    heading(27, 190, 1, "Regional mutual information")
    label(27, 220, "Compare RGB with each target in four facial regions.")
    source_centres = {"P": 491, "D": 790, "O": 1091, "F": 195}
    with Image.open(ILLUSTRATIONS / "assets" / "signal_domains.png") as regions:
        for centre, key in zip([78, 190, 303, 415], REGION_ORDER):
            cx = source_centres[key]
            tile = regions.crop((cx - 150, 0, cx + 150, 375))
            illustration(tile, centre - 36, 240, 72, 90)
            label(centre, 349, REGION_NAMES[key], 11, ha="center",
                  color=REGION_COLORS[key], weight="bold")

    heading(528, 190, 3, "Reconstruction from RGB")
    label(528, 220, "Compare each prediction with its measured target.")
    label(536, 285, "RGB", 14, color=ACCENT, weight="bold")
    arrow((582, 285), (615, 285), ACCENT)
    ax.add_patch(FancyBboxPatch((619, 248), 190, 71, boxstyle="round,pad=0,rounding_size=8",
                                edgecolor=ACCENT, linewidth=1, facecolor="#EFF5FA"))
    label(714, 270, "5 models", 13, ha="center", weight="bold")
    label(714, 297, "U-Nets · Swin · LDM", 10.5, ha="center")
    segment([810, 846], [285, 285], color=ACCENT)
    segment([846, 846], [257, 313], color=ACCENT)
    for y, target in zip([257, 285, 313], ["UV", "PLH", "PLV"]):
        arrow((846, y), (876, y), ACCENT)
        label(887, y, target, 13, weight="bold")
    label(528, 349, "UV objects are assessed for count, area and location.", 10.7)

    for x in [250, 750]:
        arrow((x, 365), (x, 388))
    heading(27, 409, 2, "Skin-tone strata")
    label(27, 440, "Repeat the regional comparison within each ITA stratum.")
    svg = ElementTree.parse(ILLUSTRATIONS / "layout_template.svg")
    panel = svg.find(".//*[@id='panel-c']")
    colors = [e.attrib["fill"] for e in panel.findall("{http://www.w3.org/2000/svg}rect")]
    names = ["Very light", "Light", "Intermediate", "Tan", "Brown", "Dark"]
    for index, (name, color) in enumerate(zip(names, colors, strict=True)):
        centre = 61 + index * 75
        ax.add_patch(Rectangle((centre - 32, 468), 64, 19, edgecolor="none", facecolor=color))
        label(centre, 510, name, 9.4, ha="center")

    heading(528, 409, 4, "UV uncertainty")
    label(528, 440, "Test error ranking and calibrate UV-object count intervals.")
    for index, bar_height in enumerate([8, 13, 18, 23, 28]):
        ax.add_patch(Rectangle((575 + 18 * index, 492 - bar_height), 12, bar_height,
                               edgecolor="none", facecolor="#766D98"))
    segment([571, 664], [493, 493], color="#766D98", lw=.8)
    label(617, 515, "Rank by predicted σ", 10.5, ha="center")
    segment([793, 921], [479, 479], color="#766D98", lw=2)
    for x in [793, 921]:
        segment([x, x], [471, 487], color="#766D98", lw=2)
    ax.plot(857, 479, "o", color="#52496F", ms=5)
    label(857, 515, "95% target coverage", 10.5, ha="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
