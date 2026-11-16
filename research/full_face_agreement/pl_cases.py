"""Replay accepted PL TEST predictions on their original RGB, PL and fractional masks."""
import hashlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path


ARCHIVE = Path("/study/siiba/experiments/pl_test_delivery_20261106_v1")
PROTOCOL = "048e687571af02a86b2024581e3b7ac02879a9f7e92a914ffcb4059e14473070"
TARGETS = ("plh", "plv")
REGIONS = ("forehead", "image_left_cheek", "image_right_cheek", "nose", "chin")
# Independently copied native TEST acceptances. Later completed pairs are bound
# by the same native manifest, training acceptance, checkpoint and pilot checks.
PAIR_PINS = {
    "unet_multi_plh": "979b7aa1dc4f0231a0a9b68b2663b0b44cfea793098825194a538429a346f243",
    "unet_multi_plv": "20b603698ff130c88638c04debe89eaee50ef73ebc765247249cbeb1d8b2d6db",
    "unet_single_plh": "b621254a4dbbdc16d03c8a3fae1eab6c5ff3b68c230839d080e809da63f45faa",
    "unet_single_plv": "83a35e79b6b9be634f751e63ca248a1517b36c49adcfd4a17b16abf1979db304",
    "unetplusplus_plh": "1a7c35ec84c8782808ca0362f9e078355e0c5fe70d1f007013bd2db9076bc269",
    "unetplusplus_plv": "4daaa9b42d8f9dfbd94209a438a6fd84842bba75c7802e80871d08ab84ba08e0",
    "swin_plh": "888318793f3e120e277de0bc94934f057a80f593610048366e11f0f21e5cbf5d",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_json(path, sha=None):
    raw = path.read_bytes()
    if sha is not None:
        _require(hashlib.sha256(raw).hexdigest() == sha, f"Changed archive: {path}")
    return json.loads(raw)


@lru_cache(maxsize=1)
def _modules():
    policy = _checked_json(ARCHIVE / "protocol.json", PROTOCOL)
    modules = []
    for name in ("pl_measure", "pl_models"):
        path = ARCHIVE / (name + ".py")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == policy["sources"][path.name],
                 f"Archived loader digest mismatch: {path}")
        spec = importlib.util.spec_from_file_location("_table3b_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return policy, *modules


@lru_cache(maxsize=1)
def _context():
    return _modules()[1].Context()


@lru_cache(maxsize=10)
def accepted_identity(model, head):
    """Validate the original pair acceptance and return the native model binding."""
    policy, _, models = _modules()
    _require(model in models.ARCHITECTURES and head in TARGETS, "Unknown PL model or target")
    key = "both" if model == "unet_multi" else head
    task_heads = list(TARGETS) if key == "both" else [head]
    pair = model + "_" + head
    folder = Path("reports") / pair
    acc = _checked_json(ARCHIVE / folder / "acceptance.json", PAIR_PINS.get(pair))
    _require((acc["schema"], acc["status"], acc["architecture"], acc["target"],
              acc["records"], acc["geometry_valid"]) ==
             ("PL_TEST_pair_acceptance_v1", "passed", model, head, 21899, 21422),
             f"TEST pair not accepted: {pair}")

    def record(desc, expected):
        _require(ARCHIVE / desc["path"] == ARCHIVE / expected, f"Wrong bound path: {desc}")
        return _checked_json(ARCHIVE / expected, desc["sha256"])

    manifest = record(acc["input_manifest"], folder / "input_manifest.json")
    binding = record(manifest["binding"], Path("bindings") / (model + "_" + key + ".json"))
    identity = binding["model_identity"]
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    _require(manifest["protocol"] == {"path": "protocol.json", "sha256": PROTOCOL}
             and manifest["selection"] == policy["selection"]
             and binding["architecture"] == model and binding["targets"] == task_heads
             and binding["protocol_sha256"] == PROTOCOL
             and manifest["model_identity_sha256"] == binding["model_identity_sha256"] == digest
             and manifest["pilots"] == binding["pilot_acceptances"],
             f"Binding/manifest identity mismatch: {pair}")
    _require(len(manifest["pilots"]) == len(policy["device_uuids"]), f"Incomplete pilots: {pair}")
    for lane, (pin, uuid) in enumerate(zip(manifest["pilots"], policy["device_uuids"])):
        pilot = record(pin, Path("pilots") / (model + "_" + key) / (str(lane) + ".json"))
        _require((pilot["status"], pilot["architecture"], pilot["targets"], pilot["protocol_sha256"],
                  pilot["model_identity_sha256"], pilot["device_uuid"]) ==
                 ("passed", model, task_heads, PROTOCOL, digest, uuid), f"Invalid pilot: {pin['path']}")
    return identity


# Keep one GPU model resident when the driver requests successive small ranges.
_active = None


def _predictor(model, key, heads, device):
    global _active
    tag = (model, key, str(device))
    models = _modules()[2]
    identities = [accepted_identity(model, head) for head in heads]
    _require(all(i == identities[0] for i in identities), "PL heads differ in model identity")
    if _active is None or _active[0] != tag:
        _active = None
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        predictor = models.load_model(model, key, device=device)
        _require(predictor.identity == identities[0], "Loaded weights differ from accepted binding")
        _active = (tag, predictor)
    _require(_active[1].identity == identities[0], "Cached weights differ from accepted binding")
    return _active[1]


def iter_cases(start, stop, step, model, target, device, *, skip_indices=()):
    """Yield one row per TEST group and requested head. stop is exclusive.

    target='both' runs the two joint heads with a single original forward.
    A joint singleton selects the requested output of that same native model.
    skip_indices may contain already committed indices for all requested heads.
    Model/archive failures propagate separately from original technical missingness.
    """
    key = "both" if model == "unet_multi" else target
    heads = TARGETS if target == "both" else (target,)
    _require(set(heads) <= set(TARGETS) and (target != "both" or model == "unet_multi"),
             "Both PL outputs together require unet_multi")
    _require(0 <= start <= stop <= 21899 and step > 0, "Invalid TEST range")
    skip = frozenset(skip_indices)
    predictor = _predictor(model, key, heads, device)
    ctx = _context()
    for index in range(start, stop, step):
        if index in skip:
            continue
        case = ctx.case(index, heads)
        gid = case["group_id"]
        if case["missing_reason"] is not None:
            for head in heads:
                yield dict(index=index, group_id=gid, model=model, target=head,
                           missing_reason=case["missing_reason"])
            continue
        preds = predictor.predict(case)
        _require(set(preds) == set(predictor.identity["targets"]), "Wrong model output keys")
        seeds = {name: case["rois"][name] for name in REGIONS}
        for head in heads:
            yield dict(index=index, group_id=gid, model=model, target=head,
                       f_weights=case["weights"]["F"], old_weights=seeds,
                       truth=case["truths"][head], pred=preds[head].detach().cpu().numpy())
