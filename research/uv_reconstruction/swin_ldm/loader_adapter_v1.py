"""PREPARATION ONLY: frozen Swin/LDM inference loaders; no CLI or import side effects.

Nothing here has been model-loaded or inference-validated. The future execution
controller must verify training/resource completion, GPU UUIDs, catalog ownership,
input descriptors and masks before calling a loader. This module never launches
workers, reads TEST images, writes files, or changes runtime/scientific settings.
"""

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


PINS = {
    "swin": {
        "schema": "swin_redesign_production_20261024_v1",
        "source": "8dd8a7e20cfcc7228c896ef7880a81e6aaac61af858c3edbd1fe679f3c831314",
        "runner": "5cf078e46344bf4cec685cb09bd467adc23bd10ad5c91f17f75a89706e1450cc",
        "binding": "2836b6045824c29472ae03c9208738c9961ab305047ca3d6db251cffa4004a2b",
        "config": "8073c30b1cd07a04496f8825384d3fce3590c9bbabc5a9fc7815bd1721ae8fcc",
        "checkpoint": "5c9da944dd83f9478fbdcb96e75a4108f1fc2ae35243786b0f05d139fc6aadad",
        "metric": "l1", "count_key": "count",
    },
    "ldm": {
        "schema": "ldm_sigma_production_20261024_v1",
        "source": "2b90dcba5617d2e564097eb5771b2bfbcaaa381fe53de35e9204fe1479b83dd2",
        "runner": "1c0ed598c91fb73e824adadd4333e60d388e82d65a6fb90078d0554114a4a5d0",
        "gate": "87add9d85e780c60cc4fc748aafd5447d3eaeda08df104c3b75e5d369d2bde8d",
        "binding": "da7a95d41b8aa6f68b4981fd17da28dc1ac3e6c6018e1cef13cf88202f982dfe",
        "config": "6661f4c70ebf84ccc626f8302c3300d579445f4ccfbe5951f14557aa65f03e85",
        "metric": "nll", "count_key": "images",
    },
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_sha(value):
    # 沿用训练序列化格式
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class FrozenFile:
    path: str
    sha256: str

    def read(self):
        require(valid_sha(self.sha256), "Missing independently frozen SHA256")
        raw = Path(self.path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.sha256, "File SHA differs: " + self.path)
        return raw

    def json(self):
        def reject_constant(value):
            raise ValueError("Nonfinite JSON value: " + value)
        return json.loads(self.read().decode("utf-8-sig"), parse_constant=reject_constant)


@dataclass(frozen=True)
class TrainingBundle:
    source: FrozenFile
    config: FrozenFile
    binding: FrozenFile
    protocol: FrozenFile
    summary: FrozenFile
    best_checkpoint: FrozenFile


def _history_state(kind, history):
    """Replay only TRAIN/VAL metadata selection; never model/data computation."""
    pin = PINS[kind]
    require(isinstance(history, list) and 1 <= len(history) <= 20, "Incomplete/invalid VAL history")
    best = anchor = None
    bad = 0
    best_epoch = None
    for epoch, row in enumerate(history, 1):
        require(row.get("epoch") == epoch and row.get(pin["count_key"]) == 10948,
                "History must cover every complete VAL epoch")
        for metric in ("l1", "nll"):
            value = row.get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value), "Invalid VAL metric")
        require(row["l1"] >= 0, "Negative VAL L1")
        value = row[pin["metric"]]
        if best is None or value < best:
            best, best_epoch = value, epoch
        if anchor is None or value < anchor - 1e-4:
            anchor, bad = value, 0
        else:
            bad += 1
        require(epoch == len(history) or bad < 5, "History extends beyond frozen early stopping")
    return {
        "epoch": len(history) + 1, "cursor": 0, "updates": len(history) * 4791,
        "best": best, "anchor": anchor, "bad_epochs": bad,
        "complete": len(history) == 20 or bad >= 5,
    }, best_epoch


def inspect_metadata(kind, bundle):
    """Hash and check metadata only. Does not import torch, load weights or construct models."""
    require(kind in PINS, "Only frozen Swin and LDM loaders are supported")
    pin = PINS[kind]
    require(bundle.source.sha256 == pin["source"], "Wrong proposed model source")
    require(valid_sha(bundle.best_checkpoint.sha256), "Best checkpoint needs an independently frozen SHA")
    config, binding = bundle.config.json(), bundle.binding.json()
    protocol, summary = bundle.protocol.json(), bundle.summary.json()
    require(canonical_sha(config) == pin["config"], "Exact production configuration differs")
    require(config == binding.get("config") == protocol.get("config"), "Config copies differ")
    require(binding.get("binding") == protocol.get("binding") == summary.get("binding") == pin["binding"],
            "Training bindings differ")
    payload = {key: value for key, value in binding.items() if key != "binding"}
    require(canonical_sha(payload) == pin["binding"], "Cannot reproduce production binding")
    require(binding.get("schema") == protocol.get("schema") == pin["schema"]
            and binding.get("world_size") == 4, "Wrong training schema/world size")
    require(protocol.get("test_accessed") is False and summary.get("test_accessed") is False,
            "Training metadata does not attest TEST unused")
    if kind == "swin":
        require(binding.get("model_sha256") == pin["source"] and binding.get("runner_sha256") == pin["runner"],
                "Swin training sources differ")
        require(bundle.best_checkpoint.sha256 == pin["checkpoint"], "Not the audited Swin BEST checkpoint")
    else:
        sources = {key: pin[key] for key in ("runner", "gate")}
        sources["model"] = pin["source"]
        require(all(protocol.get(key) == value for key, value in binding.items()),
                "LDM protocol differs from the bound training metadata")
        require(binding.get("source_sha256") == protocol.get("source_sha256") == sources,
                "LDM training sources differ")
        require(summary.get("status") == "complete", "LDM training is not complete")
        require(summary.get("checkpoint_best_sha256") == bundle.best_checkpoint.sha256
                and valid_sha(summary.get("checkpoint_last_sha256")), "LDM final checkpoint pins differ")
    state = summary.get("state", {})
    expected, best_epoch = _history_state(kind, state.get("history"))
    require(expected["complete"] is True and all(state.get(k) == v for k, v in expected.items()),
            "Training summary is not the correct completed state")
    return {"kind": kind, "config": config, "binding": binding["binding"],
            "summary": summary, "best_epoch": best_epoch}


def _verified_source(kind, descriptor):
    # 执行已核验字节
    raw = descriptor.read()
    name = "secondary_test_preparation_" + kind + "_" + descriptor.sha256[:12]
    spec = importlib.util.spec_from_file_location(name, descriptor.path)
    require(spec is not None, "Cannot create source module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, descriptor.path, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_checkpoint(torch, descriptor):
    require(valid_sha(descriptor.sha256), "Invalid checkpoint SHA")
    with Path(descriptor.path).open("rb") as stream:
        before = Path(descriptor.path).stat()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        require(digest.hexdigest() == descriptor.sha256, "Checkpoint SHA differs")
        stream.seek(0)
        # 载入完整训练状态
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        after = Path(descriptor.path).stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Checkpoint changed during read")
    return checkpoint


def _runtime(torch, device):
    require(device.type in ("cpu", "cuda"), "Only CPU reference or CUDA execution is supported")
    if device.type == "cuda":
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                "Caller must disable TF32 before adapter use")
        require(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
                and not torch.backends.cudnn.benchmark, "Caller must configure deterministic CUDA")


def load_adapter(kind, bundle, device):
    """Future validated controller entry point. Calling this DOES load/build the model."""
    info = inspect_metadata(kind, bundle)
    source = _verified_source(kind, bundle.source)
    import torch
    device = torch.device(device)
    _runtime(torch, device)
    checkpoint = _load_checkpoint(torch, bundle.best_checkpoint)
    require(checkpoint.get("schema") == PINS[kind]["schema"]
            and checkpoint.get("binding") == info["binding"] and checkpoint.get("world_size") == 4,
            "Checkpoint training schema/binding/world differs")
    require(isinstance(checkpoint.get("rank_rng"), list) and len(checkpoint["rank_rng"]) == 4,
            "Checkpoint does not retain all four training rank RNG records")
    history = info["summary"]["state"]["history"][:info["best_epoch"]]
    expected, unused_best = _history_state(kind, history)
    state = checkpoint.get("state", {})
    require(state.get("history") == history and all(state.get(k) == v for k, v in expected.items()),
            "BEST checkpoint does not correspond to the selected complete VAL epoch")
    if kind == "ldm":
        require(checkpoint.get("checkpoint_kind") == "posttraining_sigma_head_only",
                "LDM requires the phase-C sigma-head checkpoint")
    weights = checkpoint.get("model" if kind == "swin" else "head_state_dict")
    require(isinstance(weights, dict) and weights, "Missing checkpoint state dictionary")
    for name, tensor in weights.items():
        require(isinstance(tensor, torch.Tensor), "Nontensor state: " + name)
        require(tensor.dtype == torch.float32 or (kind == "swin" and tensor.dtype == torch.int64
                and name.endswith(".attention.relative_position_index")),
                "Checkpoint is not the original FP32 state/known integer buffer: " + name)
        require(bool(torch.isfinite(tensor).all()), "Nonfinite checkpoint state: " + name)
    if kind == "swin":
        args = dict(info["config"]["model"])
        require(args.pop("drop_path") == 0.0, "Frozen drop_path differs")
        for key in ("depths", "num_heads", "decoder_depths"):
            args[key] = tuple(args[key])
        model = source.SwinUVUncertainty(
            **args, activation_checkpointing=info["config"]["activation_checkpointing"])
        model.load_state_dict(weights, strict=True)
    else:
        # 核验并载入生成器
        # 固定DDIM步数与种子
        model = source.build_training_model(info["config"], device)
        model.log_variance_head.load_state_dict(weights, strict=True)
    model = model.float().to(device).eval().requires_grad_(False)
    require(all(not tensor.is_floating_point() or (tensor.dtype == torch.float32
                and bool(torch.isfinite(tensor).all())) for tensor in model.state_dict().values()),
            "Loaded model contains nonfinite or non-FP32 values")
    return PredictionAdapter(kind, model, info)


class PredictionAdapter:
    def __init__(self, kind, model, provenance):
        self.kind, self.model, self.provenance = kind, model, provenance

    def predict(self, rgb, frozen_row):
        """Return the untouched source dictionary from ONE forward. No truth argument.

        Caller must bind frozen_row and rgb to the verified TEST catalog/descriptor;
        accepting a mapping here is not proof of that external binding.
        """
        import torch
        sample_id = frozen_row.get("sample_id")
        require(frozen_row.get("split") == "test" and isinstance(sample_id, str) and bool(sample_id),
                "Use the original nonempty TEST sample_id string, never an index surrogate")
        require(isinstance(rgb, torch.Tensor) and rgb.dtype == torch.float32
                and tuple(rgb.shape) == (1, 3, 1024, 1024), "Expected preprocessed batch-one FP32 RGB1024")
        device = next(self.model.parameters()).device
        require(rgb.device == device and bool(torch.isfinite(rgb).all())
                and bool(((rgb >= 0) & (rgb <= 1)).all()), "RGB device/range/finiteness differs")
        require(not self.model.training and all(not p.requires_grad for p in self.model.parameters()),
                "Model must remain frozen and in eval mode")
        _runtime(torch, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            output = self.model(rgb, [sample_id]) if self.kind == "ldm" else self.model(rgb)
        require(isinstance(output, dict), "Unexpected source output")
        for key in ("mean", "raw_log_variance", "log_variance", "variance", "sigma"):
            value = output.get(key)
            expected = (1, 3, 1024, 1024) if key == "mean" else (1, 1, 1024, 1024)
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                    and value.device == device and tuple(value.shape) == expected
                    and bool(torch.isfinite(value).all()), "Invalid source output: " + key)
        require(bool(((output["mean"] >= 0) & (output["mean"] <= 1)).all())
                and bool((output["variance"] > 0).all()) and bool((output["sigma"] > 0).all()),
                "Source output violates original image/Gaussian domain")
        return output
