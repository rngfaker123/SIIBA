"""Measure one UV, PLH or PLV reconstruction from decoded arrays at the study grid."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "regional_metrics"))
sys.path.insert(0, str(HERE.parent / "pl_test"))
from tools import uv_architecture_comparison_metrics_v10 as numeric
from uv_three_region_eval_20261025 import measure, validate_weights
from pl_measure import cpu_regions


def evaluate_group(input_path, target):
    with np.load(input_path, allow_pickle=False) as group:
        truth = numeric.validate_image(group["truth"], "truth")
        prediction = numeric.validate_image(group["prediction"], "prediction")
        shape = (1024, 1024)
        if truth.shape != (3, *shape) or prediction.shape != truth.shape:
            raise ValueError("Expected 3 x 1024 x 1024 RGB arrays")
        weights = {r: group[r] for r in ("P","D","O","F")}
        if target == "uv":
            regions = measure(truth, prediction, group["sigma"], weights, numeric, shape)
        elif target in ("plh", "plv"):
            regions = cpu_regions(numeric, truth, prediction, validate_weights(weights, shape))
        else:
            raise ValueError("Expected uv, plh or plv")
    return dict(target=target, regions=regions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=Path)
    parser.add_argument("--target", choices=("uv", "plh", "plv"), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_group(args.group, args.target), indent=2, allow_nan=False))
