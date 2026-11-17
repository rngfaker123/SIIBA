"""Shared figure conventions."""

from matplotlib.lines import Line2D

FONT = "Arial"
REGION_ORDER = ("P", "D", "O", "F")
REGION_COLORS = {
    "P": "#C34359", "D": "#416FA3", "O": "#C89B3C", "F": "#747A83",
}
REGION_NAMES = {
    "P": "Porphyrin", "D": "UV spots", "O": "Other", "F": "Full face",
}
REGION_MARKERS = {"P": "o", "D": "s", "O": "^", "F": "o"}
REGION_LINESTYLES = {"P": "-", "D": "--", "O": "-.", "F": ":"}
MODEL_ORDER = ("Single U-Net", "Multi U-Net", "U-Net++", "Swin", "LDM")
MODEL_MARKERS = ("o", "s", "^", "D", "v")
MODEL_COLORS = ("#285783", "#24847C", "#B45C3C", "#944E76", "#7564A4")
TARGET_ORDER = ("UV", "PLH", "PLV")


REGION_MARKER_SIZES = {"P": 3.6, "D": 3.7, "O": 3.9, "F": 6.4}


def region_line_style(region, scale=1.0):
    color = REGION_COLORS[region]
    return dict(color=color, ls=REGION_LINESTYLES[region], lw=scale,
                marker=REGION_MARKERS[region], ms=REGION_MARKER_SIZES[region] * scale,
                mfc="none" if region == "F" else color, mec=color,
                mew=(.9 if region == "F" else .4) * scale)


def add_region_legend(fig, regions, anchor, *, fontsize, loc, borderaxespad=.5):
    """Keep symbol, dash and spacing proportions at each figure's text size."""
    handles = [Line2D([], [], label=f"{r}  {REGION_NAMES[r]}",
                      **region_line_style(r, scale=fontsize / 8)) for r in regions]
    return fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=len(handles),
                      prop={"family": FONT, "size": fontsize}, labelcolor="#171717",
                      frameon=False, handlelength=1.4, handletextpad=.3,
                      columnspacing=.75, borderaxespad=borderaxespad)
