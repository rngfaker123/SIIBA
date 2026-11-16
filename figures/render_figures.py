"""Verify a figure snapshot and render the selected figures."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parent


def assemble_inputs(figures, private, destination, public_only=False):
    entries = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf8"))["files"]
    selected = {f"figure{number}" for number in figures}
    missing = []
    valid = 0
    for relative, entry in entries.items():
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid input path {relative}")
        if relative.parts[0] not in selected:
            continue
        is_public = entry["access"] == "public"
        if public_only and not is_public:
            continue
        folder = ROOT / "inputs" if is_public else private
        if folder is None or not (folder / relative).is_file():
            missing.append(str(relative))
            continue
        source = folder / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Unexpected checksum for {relative}")
        if destination:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        valid += 1
    if missing:
        raise FileNotFoundError("Supply the research-group inputs via --private-inputs. Missing: " + ", ".join(missing))
    print(f"Verified {valid} inputs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=range(1, 5), default=[1, 2, 3, 4])
    parser.add_argument("--private-inputs", type=Path, help="Root of the research-group inputs.")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--check-inputs", action="store_true", help="Verify inputs without rendering.")
    parser.add_argument("--check-public", action="store_true", help="Verify only the published inputs.")
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.figures))
    private = args.private_inputs.expanduser().resolve() if args.private_inputs else None
    if args.check_inputs or args.check_public:
        assemble_inputs(selected, private, None, public_only=args.check_public)
        return
    output = args.output.expanduser().resolve()
    protected = [ROOT / "inputs", ROOT / "siiba_figures"]
    if private:
        protected.append(private)
    if output == ROOT or any(output.is_relative_to(path) for path in protected):
        parser.error("Choose a separate output folder.")
    if not __debug__:
        parser.error("Run without -O so all source-data validations are active.")
    with tempfile.TemporaryDirectory(prefix="siiba-figures-") as stage:
        inputs = Path(stage)
        assemble_inputs(selected, private, inputs)
        os.environ["SIIBA_FIGURE_INPUTS"] = str(inputs)
        output.mkdir(parents=True, exist_ok=True)
        for number in selected:
            print(f"Rendering Figure {number} ...", flush=True)
            module = importlib.import_module(f"siiba_figures.figure{number}")
            module.build(output)
            print(f"Saved Figure {number}.", flush=True)


if __name__ == "__main__":
    main()
