"""Shared figure conventions."""

FONT = "Arial"
REGION_ORDER = ("P", "D", "O", "F")
REGION_COLORS = {
    "P": "#C34359", "D": "#416FA3", "O": "#C89B3C", "F": "#747A83",
}
REGION_NAMES = {
    "P": "Porphyrin", "D": "UV spots", "O": "Other skin", "F": "Full face",
}
REGION_MARKERS = {"P": "o", "D": "s", "O": "^", "F": "o"}
REGION_LINESTYLES = {"P": "-", "D": "--", "O": "-.", "F": ":"}
MODEL_ORDER = ("Single U-Net", "Multi U-Net", "U-Net++", "Swin", "LDM")
MODEL_MARKERS = ("o", "s", "^", "D", "v")
MODEL_COLORS = ("#285783", "#24847C", "#B45C3C", "#944E76", "#7564A4")
TARGET_ORDER = ("UV", "PLH", "PLV")
