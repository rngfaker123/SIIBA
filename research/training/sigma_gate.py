"""Fail-closed asset gate and config preparation for post-training LDM sigma.

This only reads completed TRAIN/VAL artifacts. No TEST pixels, new generator
training, checkpoint relabelling, or changes to the historical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MANIFEST_SHA = "737d8654fd99c1e3790ac8ec6aa9866f4002bad47ada24835bd97a43b03e4a2c"
MODEL_SHA = "2b90dcba5617d2e564097eb5771b2bfbcaaa381fe53de35e9204fe1479b83dd2"
ARCHITECTURE = "ldm_fixed_generator_gaussian_head_proposed_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def descriptor(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def check_file(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("Frozen asset hash differs: " + str(path))
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_stage(summary, best, last, protocol_sha, stage, expected_best_sha,
                 expected_last_sha, expected_assets=None):
    """Pure metadata check; file hashes are independently checked by the caller."""
    require(summary.get("status") == "complete" and summary.get("training_complete") is True,
            stage + " is not complete")
    require(summary.get("protocol_sha256") == protocol_sha and summary.get("stage") == stage,
            stage + " summary protocol/stage differs")
    require(summary.get("test_split_used") is False and summary.get("test_images_accessed") == 0,
            stage + " summary used TEST or has unknown TEST access")
    require(summary.get("checkpoint_best_sha256") == expected_best_sha
            and summary.get("checkpoint_last_sha256") == expected_last_sha,
            stage + " summary checkpoint hashes differ")
    for cp in (best, last):
        require(cp.get("schema") == "uv_true_latent_diffusion_clean_multigpu_train_20261016_v1",
                stage + " unknown checkpoint schema")
        require(cp.get("stage") == stage and cp.get("protocol_sha256") == protocol_sha,
                stage + " checkpoint stage/protocol differs")
        require(cp.get("test_split_used") is False and cp.get("ema_used") is False,
                stage + " checkpoint has TEST/EMA mismatch")
        require(cp.get("precision") == "BF16_forward_FP32_loss_no_scaler",
                stage + " checkpoint precision changed")
        if expected_assets is not None:
            assets = cp.get("frozen_assets", {})
            for k, value in expected_assets.items():
                require(assets.get(k) == value, stage + " frozen asset differs: " + k)
    state = last["state"]
    require(last.get("training_complete") is True and state.get("phase") == "complete",
            stage + " last checkpoint not complete")
    epochs = state.get("epoch")
    require(isinstance(epochs, int) and 1 <= epochs <= 20
            and epochs == summary.get("epochs_completed"), stage + " epochs differ")
    require(state.get("global_step") == state.get("global_attempts") == epochs * 4791
            and state.get("next_batch") == 4791 and state.get("amp_overflow_skips") == 0,
            stage + " incomplete data traversal/overflow counters")
    require(summary.get("global_step") == summary.get("global_attempts") == state["global_step"]
            and summary.get("amp_overflow_skips") == 0 and not state.get("blocked_reason"),
            stage + " summary counters or blocked state differ")
    history = state.get("history", [])
    require(len(history) == epochs, stage + " history incomplete")
    for epoch, row in enumerate(history, 1):
        v = row.get("validation", {})
        require(row.get("epoch") == epoch and row.get("train_images") == 76642
                and v.get("images") == 10948 and v.get("precision") == "FP32"
                and math.isfinite(v.get("l1", float("nan"))),
                stage + " full TRAIN/FP32 VAL evidence missing")
        if stage == "diffusion":
            require(v.get("ddim20") is True and v.get("single_generation") is True,
                    "B validation is not single deterministic DDIM20")
    selection = state.get("selection", {})
    require(summary.get("selection") == selection, stage + " selection differs")
    anchor, bad = None, 0
    for row in history:
        value = row["validation"]["l1"]
        if anchor is None or value < anchor - 1e-4:
            anchor, bad = value, 0
        else:
            bad += 1
    require(selection.get("early_stopping_anchor") == anchor
            and selection.get("early_stopping_counter") == bad,
            stage + " early-stopping history differs")
    reason = state.get("stop_reason")
    require(summary.get("stop_reason") == reason
            and ((reason == "early_stopping" and bad >= 5)
                 or (reason == "max_epochs" and epochs == 20)),
            stage + " stop reason is not justified by complete history")
    best_epoch = min(history, key=lambda row: row["validation"]["l1"])["epoch"]
    best_l1 = history[best_epoch - 1]["validation"]["l1"]
    require(selection.get("best_epoch") == best_epoch and selection.get("best_val_l1") == best_l1
            and best["state"].get("epoch") == best_epoch,
            stage + " BEST does not match minimum complete VAL L1")
    require(best["state"].get("phase") in ("epoch_complete", "complete")
            and best["state"].get("next_batch") == 4791,
            stage + " BEST is not a complete-epoch checkpoint")
    return {"epochs_completed": epochs, "best_epoch": best_epoch, "best_val_l1": best_l1}


def verify_assets(config, load_checkpoint=None):
    if load_checkpoint is None:
        import torch
        load_checkpoint = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    require(sha(config["source_manifest"]) == MANIFEST_SHA == config["source_manifest_sha256"],
            "Clean source manifest identity differs")
    require(config["architecture_id"] == ARCHITECTURE, "Wrong sigma architecture")
    require(config["ae"].get("state_key") == config["denoiser"].get("state_key") == "model_state_dict"
            and config.get("statistics_mean_key") == "mean" and config.get("statistics_std_key") == "std",
            "Unexpected generator weights/statistics fields")
    a = config["gate_assets"]
    paths = {key: check_file(rec) for key, rec in a.items()}
    for key in ("ae", "denoiser", "latent_statistics"):
        paths[key] = check_file(config[key])
    protocols = {name: read_json(paths[name + "_protocol"]) for name in ("ae", "b")}
    for name, protocol in protocols.items():
        require(protocol.get("source_manifest_sha256") == MANIFEST_SHA
                and protocol.get("expected_counts") == {"train": 76642, "val": 10948}
                and protocol.get("test_split_used") is False,
                name + " protocol is not the clean TRAIN/VAL authority")
        require(protocol.get("manifest_sha256", {}).get("train") == a["train_manifest"]["sha256"]
                and protocol.get("manifest_sha256", {}).get("val") == a["val_manifest"]["sha256"],
                name + " train/val manifests differ")
        require(protocol.get("config", {}).get("seed") == 20260817,
                name + " generator seed authority differs")
        require(protocol.get("epoch_order_file_sha256") == config["order_sha256"],
                name + " frozen epoch order authority differs")
    ae_best = load_checkpoint(paths["ae"])
    ae_last = load_checkpoint(paths["ae_last"])
    ae_info = verify_stage(read_json(paths["ae_summary"]), ae_best, ae_last,
                          a["ae_protocol"]["sha256"], "ae", config["ae"]["sha256"],
                          a["ae_last"]["sha256"])
    del ae_last
    stats = read_json(paths["latent_statistics"])
    require(stats.get("protocol_sha256") == a["ae_protocol"]["sha256"]
            and stats.get("ae_checkpoint_sha256") == config["ae"]["sha256"]
            and stats.get("source_manifest_sha256") == MANIFEST_SHA
            and stats.get("train_manifest_sha256") == a["train_manifest"]["sha256"]
            and stats.get("test_split_used") is False and stats.get("images") == 76642
            and stats.get("count_values_per_channel") == 76642 * 256 * 256
            and stats.get("latent_shape_per_image") == [4, 256, 256]
            and stats.get("method") == "population_float64_all_train_posterior_mean"
            and stats.get("posterior_encoding_precision") == "FP32", "Latent statistics binding differs")
    values = stats.get("mean", []) + stats.get("std", [])
    require(len(stats.get("mean", [])) == len(stats.get("std", [])) == 4
            and all(math.isfinite(x) for x in values) and min(stats["std"]) > 0,
            "Latent statistics invalid")
    b_best = load_checkpoint(paths["denoiser"])
    b_last = load_checkpoint(paths["b_last"])
    b_info = verify_stage(read_json(paths["b_summary"]), b_best, b_last,
                         a["b_protocol"]["sha256"], "diffusion", config["denoiser"]["sha256"],
                         a["b_last"]["sha256"],
                         {"ae_checkpoint_sha256": config["ae"]["sha256"],
                          "latent_stats_sha256": config["latent_statistics"]["sha256"]})
    del b_last, b_best, ae_best
    # Refuse sources that advanced during the verification itself.
    for rec in list(a.values()) + [config[k] for k in ("ae", "denoiser", "latent_statistics")]:
        check_file(rec)
    return {"passed": True, "ae": ae_info, "b": b_info, "test_accessed": False,
            "generator_generation_index": 0, "ddim_steps": 20,
            "generator_precision": "FP32", "generator_training_reused_not_relabelled": True}


def prepare(args):
    formal, branch = Path(args.formal_root), Path(args.b_root)
    orders = Path(args.orders_dir)
    c = {"architecture_id": ARCHITECTURE, "source_manifest": str(Path(args.source_manifest).resolve()),
         "source_manifest_sha256": MANIFEST_SHA, "dataset_root": str(Path(args.data_root).resolve()),
         "output_dir": str(Path(args.output_dir).resolve()), "seed": 20260817,
         "expected_counts": {"train": 76642, "val": 10948}, "selection_metric": "nll",
         "orders_dir": str(orders.resolve()),
         "order_sha256": {str(e): sha(orders / ("epoch_%03d.json" % e)) for e in range(1, 21)},
         "training": {"epochs": 20, "global_batch_size": 16, "microbatch_size": 1,
                      "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1, "precision": "FP32"},
         "ae": {**descriptor(formal / "ae/checkpoint_best.pt"), "state_key": "model_state_dict"},
         "denoiser": {**descriptor(branch / "diffusion/checkpoint_best.pt"), "state_key": "model_state_dict"},
         "latent_statistics": descriptor(formal / "latent_stats/statistics.json"),
         "statistics_mean_key": "mean", "statistics_std_key": "std",
         "checkpoint_every_steps": 20, "checkpoint_every_seconds": 300, "num_workers": 2,
         "gate_assets": {}}
    for name, path in {"ae_summary": formal / "ae/training_summary.json",
                       "ae_last": formal / "ae/checkpoint_last.pt", "ae_protocol": formal / "protocol.json",
                       "b_summary": branch / "diffusion/training_summary.json",
                       "b_last": branch / "diffusion/checkpoint_last.pt", "b_protocol": branch / "protocol.json",
                       "train_manifest": formal / "train_manifest.json",
                       "val_manifest": formal / "val_manifest.json"}.items():
        c["gate_assets"][name] = descriptor(path)
    c["target_modality"] = args.target
    report = verify_assets(c)
    target = Path(args.config)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"config": str(target), "sha256": sha(target), "gate": report}, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--target", choices=("uv", "plh", "plv"), default="uv")
    p.add_argument("--formal-root")
    p.add_argument("--b-root")
    p.add_argument("--orders-dir")
    p.add_argument("--source-manifest")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    args = p.parse_args()
    if args.prepare:
        if not all(getattr(args, x) for x in ("formal_root", "b_root", "orders_dir", "source_manifest", "data_root", "output_dir")):
            p.error("prepare requires all source roots and a new output directory")
        prepare(args)
    else:
        print(json.dumps(verify_assets(read_json(args.config)), allow_nan=False))


if __name__ == "__main__":
    main()
