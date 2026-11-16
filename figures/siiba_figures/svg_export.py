"""Render vector layouts with Chromium and Poppler."""

from pathlib import Path
from xml.etree import ElementTree as ET
import os
import shutil
import subprocess
import tempfile

from PIL import Image


def find_browser():
    """Locate Chrome or Edge."""
    override = os.environ.get("SIIBA_CHROME")
    if override:
        found = shutil.which(override) or (override if Path(override).is_file() else None)
        if not found:
            raise FileNotFoundError("SIIBA_CHROME does not point to a browser executable.")
        return str(found)
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if found := shutil.which(name):
            return found
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))/"Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))/"Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))/"Google/Chrome/Application/chrome.exe",
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError("Chrome/Chromium/Edge not found. Set SIIBA_CHROME to its executable.")


def export_svg(svg_path, width_mm=215.9):
    """Export PDF, PNG and preview."""
    svg_path = Path(svg_path).resolve()
    browser = find_browser()
    pdftoppm = os.environ.get("SIIBA_PDFTOPPM") or shutil.which("pdftoppm")
    if not pdftoppm:
        raise FileNotFoundError("pdftoppm not found. Install Poppler or set SIIBA_PDFTOPPM.")
    root = ET.parse(svg_path).getroot()
    width, height = map(float, root.get("viewBox").split()[2:])
    source = ET.tostring(root, encoding="unicode")
    pdf_path = svg_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="siiba-render-") as temporary:
        folder = Path(temporary)
        page = folder/"render.html"
        page.write_text(
            '<!doctype html><meta charset="utf-8"><style>'
            f'@page{{size:{width_mm}mm {width_mm*height/width:.8f}mm;margin:0}}'
            'html,body{margin:0;padding:0}'
            f'body>svg{{width:{width_mm}mm;height:{width_mm*height/width:.8f}mm;display:block}}'
            '</style>'+source, encoding="utf8")
        subprocess.run([
            browser, "--headless", "--disable-gpu", "--disable-extensions",
            f"--user-data-dir={folder/'profile'}", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}", page.as_uri(),
        ], capture_output=True, check=True, timeout=90)
    subprocess.run([
        str(pdftoppm), "-r", "300", "-singlefile", "-png", str(pdf_path),
        str(svg_path.with_suffix("")),
    ], capture_output=True, check=True, timeout=120)
    with Image.open(svg_path.with_suffix(".png")) as image:
        image.thumbnail((1550, 1400), Image.Resampling.LANCZOS)
        image.save(svg_path.with_name(svg_path.stem+"_preview.png"))
