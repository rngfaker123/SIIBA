"""Locate the verified plotting inputs assembled by the command-line runner."""
import os
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("SIIBA_FIGURE_INPUTS", Path(__file__).resolve().parents[1] / "inputs"))
