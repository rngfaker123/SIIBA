#!/usr/bin/env python3
"""Synthetic tests only: no user images, detectors, training or production data.

Run: python research/mutual_information/test_mi.py -v
Fixtures/artifacts live in TemporaryDirectory and are removed by unittest.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from estimator import mi_atlas_six_metrics_20261025 as runner
import numpy as np
from tools.mi_knn_estimator_v1 import dequantize_u8, ksg_mi


def masks64():
    f = np.ones((64, 96), np.uint8)
    p, d, o = (np.zeros_like(f) for _ in range(3))
    p[:, :32], d[:, 32:64], o[:, 64:] = 1, 1, 1
    return dict(F=f, P=p, D=d, O=o)


def fixture(root, count=2):
    source = root / "inputs"
    source.mkdir()
    rows = []
    for index in range(count):
        rng = np.random.default_rng(900 + index)
        rgb = rng.integers(25, 230, (64, 96, 3), dtype=np.uint8)
        uv = np.clip(rgb.astype(np.int16) + rng.integers(-30, 31, rgb.shape), 0, 255).astype(np.uint8)
        plv = np.clip(rgb.astype(np.int16) + rng.integers(-12, 13, rgb.shape), 0, 255).astype(np.uint8)
        plh = rng.integers(0, 256, rgb.shape, dtype=np.uint8)
        arrays = dict(rgb=rgb, uv=uv, plv=plv, plh=plh, **masks64())
        row = dict(sample_id="synthetic_%d" % index, subject_id="subject_%d" % index,
                   native_shape=[64, 96])
        for name, array in arrays.items():
            path = source / ("%d_%s.npy" % (index, name))
            np.save(path, array, allow_pickle=False)
            row[name] = dict(path=path.name, sha256=runner.sha(path))
        rows.append(row)
    manifest = dict(schema=runner.INPUT_SCHEMA, status="frozen", cohort_id="synthetic_not_real_people",
        expected_sample_count=count,
        frozen_provenance={key: "a" * 64 for key in
            ("company_module_sha256", "mask_pipeline_sha256", "mask_config_sha256")}, rows=rows)
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    return path, manifest


class SamplingTests(unittest.TestCase):
    def test_reproducible_unique_nested_and_mask_membership(self):
        masks, counts = runner.validate_masks(masks64(), (64, 96))
        a = runner.sample_indices(masks, "person_A")
        b = runner.sample_indices(masks, "person_A")
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a[0], a[1]))
        self.assertFalse(np.array_equal(a, runner.sample_indices(masks, "person_B")))
        self.assertEqual(counts, dict(F=6144, P=2048, D=2048, O=2048))
        for si in range(3):
            for di, domain in enumerate(runner.DOMAINS):
                indices = a[si, di]
                self.assertEqual(len(np.unique(indices)), 1000)
                self.assertTrue(masks[domain].ravel()[indices].all())
                self.assertTrue(np.array_equal(indices[:100], indices[:500][:100]))
                self.assertTrue(np.array_equal(indices[:500], indices[:1000][:500]))

    def test_uint255_and_invalid_masks(self):
        a, _ = runner.validate_masks(masks64(), (64, 96))
        b, _ = runner.validate_masks({k: v * 255 for k, v in masks64().items()}, (64, 96))
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        overlap = masks64()
        overlap["P"][0, 40] = 1
        with self.assertRaisesRegex(ValueError, "disjoint"):
            runner.validate_masks(overlap, (64, 96))
        hole = masks64()
        hole["O"][0, 80] = 0
        with self.assertRaisesRegex(ValueError, "union"):
            runner.validate_masks(hole, (64, 96))
        shortage = masks64()
        retained = np.flatnonzero(shortage["P"].ravel())[:999]
        shortage["P"][:] = 0
        shortage["P"].ravel()[retained] = 1
        shortage["D"] = ((shortage["F"] != 0) & (shortage["P"] == 0) & (shortage["O"] == 0)).astype(np.uint8)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            runner.validate_masks(shortage, (64, 96))
        bad = masks64()
        bad["P"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            runner.validate_masks(bad, (64, 96))
        with self.assertRaisesRegex(ValueError, "dtype"):
            runner.validate_masks({k: v.astype(bool) for k, v in masks64().items()}, (64, 96))

    def test_native_dequantization_matches_frozen_reference_bitwise(self):
        raw = np.array([[0, 255, 128], [12, 90, 177], [18, 22, 31], [0, 0, 0]], np.uint8)
        indices = np.array([900, 0, 532, 31], np.uint64)
        for sample in ("synthetic_A", "人\\\"sample"):
            for modality in runner.MODS:
                actual = runner.dequantize_native(raw, indices, sample, 96, modality)
                expected = dequantize_u8(raw, [sample] * 4,
                    ["NATIVE:96:%d" % i for i in indices], modality, seed=runner.NOISE_SEED)
                self.assertTrue(np.array_equal(actual, expected))
                subset = [2, 0]
                reordered = runner.dequantize_native(raw[subset], indices[subset], sample, 96, modality)
                self.assertTrue(np.array_equal(reordered, actual[subset]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            runner.dequantize_native(raw[:2], np.array([1, 1], np.uint64), "a", 96, "rgb")

    def test_sd_is_sample_sd_and_negative_is_preserved(self):
        value = runner.describe_repeats([-1., 0., 1.])
        self.assertEqual(value["mean_bits"], 0.)
        self.assertEqual(value["sample_sd_bits"], 1.)
        self.assertEqual(value["mi_bits_by_seed"][0], -1.)
        self.assertEqual(value["ddof"], 1)
        with self.assertRaises(ValueError):
            runner.describe_repeats([1., 2.])


class ProtocolTests(unittest.TestCase):
    def test_image_input_and_worker_limit(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="mi_atlas_test_") as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root, 1)
            row = manifest["rows"][0]
            for name in runner.MODS + runner.DOMAINS:
                original = np.load(manifest_path.parent / row[name]["path"])
                if name in runner.DOMAINS:
                    original = original * 255
                path = manifest_path.parent / (name + ".png")
                Image.fromarray(original).save(path)
                row[name] = dict(path=path.name, sha256=runner.sha(path))
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")
            with contextlib.redirect_stdout(io.StringIO()):
                runner.prepare(manifest_path, root / "out")
                runner.verify_prepared(manifest_path, root / "out", recheck_sources=True)
            with self.assertRaisesRegex(ValueError, "1..4"):
                runner.run_mi(manifest_path, root / "out", workers=5)
            with self.assertRaisesRegex(ValueError, "contain"):
                runner.load_contract(manifest_path, root)

    def test_prepare_resume_and_exact_raw_pairing(self):
        with tempfile.TemporaryDirectory(prefix="mi_atlas_test_") as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root)
            out = root / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                first = runner.prepare(manifest_path, out)
                second = runner.prepare(manifest_path, out)
                runner.verify_prepared(manifest_path, out, recheck_sources=True)
            self.assertEqual(first, second)
            self.assertEqual(len(first["features"]), 2)
            for i, row in enumerate(manifest["rows"]):
                z = runner.load_npz(runner.feature_paths(out, i)[0])
                for mi, modality in enumerate(runner.MODS):
                    raw = np.load(manifest_path.parent / row[modality]["path"]).reshape(-1, 3)
                    self.assertTrue(np.array_equal(z["full_values"][:, mi], raw[z["linear_indices"][:, 0]]))
                    if mi < 2:
                        self.assertTrue(np.array_equal(z["region_values"][:, :, mi], raw[z["linear_indices"][:, 1:]]))
                self.assertEqual(sum(a.nbytes for a in z.values()), 186040)
            source = manifest_path.parent / manifest["rows"][0]["P"]["path"]
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                runner.verify_prepared(manifest_path, out, recheck_sources=True)

    def test_incomplete_identity_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory(prefix="mi_atlas_test_") as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root)
            manifest["rows"][1]["subject_id"] = manifest["rows"][0]["subject_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "subject_id"):
                runner.load_contract(manifest_path)
            manifest["rows"][1]["subject_id"] = "unique_again"
            manifest["expected_sample_count"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "row count"):
                runner.load_contract(manifest_path)

    def test_prepare_shortage_never_publishes_complete_marker(self):
        with tempfile.TemporaryDirectory(prefix="mi_atlas_test_") as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root, 1)
            row = manifest["rows"][0]
            data = masks64()
            data["P"][:] = 0
            data["D"] = (data["F"] - data["O"]).astype(np.uint8)
            for domain in ("P", "D"):
                path = manifest_path.parent / row[domain]["path"]
                np.save(path, data[domain], allow_pickle=False)
                row[domain]["sha256"] = runner.sha(path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")
            out = root / "out"
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                runner.prepare(manifest_path, out)
            self.assertFalse((out / "preparation.json").exists())
            self.assertFalse((out / "summary.json").exists())

    def test_all_54_real_ksg_and_resume_without_recompute(self):
        with tempfile.TemporaryDirectory(prefix="mi_atlas_test_") as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root)
            out = root / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                runner.prepare(manifest_path, out)
                with self.assertRaisesRegex(ValueError, "Incomplete 54"):
                    runner.summarize(manifest_path, out)
                summary = runner.run_mi(manifest_path, out, workers=1)
                with patch.object(runner, "ksg_mi", side_effect=AssertionError("must use cached results")):
                    resumed = runner.run_mi(manifest_path, out, workers=1)
            self.assertEqual(summary, resumed)
            self.assertEqual(summary["actual_result_count"], 54)
            self.assertEqual(len(summary["summaries"]), 18)
            for record in summary["summaries"]:
                expected = np.asarray(record["mi_bits_by_seed"])
                self.assertEqual(record["sample_sd_bits"], float(expected.std(ddof=1)))
                self.assertEqual(record["observations_per_repeat"], 2 * record["points_per_sample"])
            # Full-F RGB inputs are bit-identical across its three target metrics.
            for seed in runner.SEEDS:
                for size in runner.SIZES:
                    hashes = {runner.read_json(runner.result_path(out, m, size, seed))["input_x_sha256"]
                              for m in ("full_rgb_plv", "full_rgb_plh", "full_rgb_uv")}
                    self.assertEqual(len(hashes), 1)
            # A corrupted feature is rejected even when every MI result is cached.
            feature_path = runner.feature_paths(out, 0)[0]
            feature_path.write_bytes(feature_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                runner.verify_prepared(manifest_path, out)


class EstimatorSmokeTests(unittest.TestCase):
    def test_independent_and_correlated_gaussian(self):
        rng = np.random.default_rng(20261025)
        x = rng.normal(size=(4000, 3))
        independent = rng.normal(size=x.shape)
        rho = .5
        correlated = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=x.shape)
        null = ksg_mi(x, independent, k=5, workers=1)["mi_bits"]
        measured = ksg_mi(x, correlated, k=5, workers=1)["mi_bits"]
        theoretical = -1.5 * np.log2(1 - rho ** 2)
        self.assertLess(abs(null), .12)
        self.assertLess(abs(measured - theoretical), .2)
        self.assertGreater(measured, null + .3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
