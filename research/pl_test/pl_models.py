"""PL TEST model bindings. CPU-only readiness checks and frozen batch-one inference.

Input RGB must come from pl_measure's digest-checked original center-crop/Lanczos
pipeline. No target image or region mask is passed to a model.
"""
import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

PROJECT = Path('/study/siiba')
E = PROJECT / 'experiments'
SINGLE = E / 'pl_controls_original_uv_pipeline_20261031_v1'
JOINT_EVAL = E / 'three_region_test_eval_20261025_v1'
PLUS_SIDE = E / 'pl_training_autostart_20261101_v1'
SIDE_RESUME = E / 'pl_side_resume_20261104_v1'
MAIN_RESUME = E / 'pl_main_resume_20261105_v1'
RELEASE = E / 'pl_release_next_20261102_v1'
SWIN = E / 'pl_ten_targets_20261031_v1'
TARGETS = ('plh', 'plv')
ARCHITECTURES = ('unet_single', 'unet_multi', 'unetplusplus', 'swin', 'ldm')

# 协议锁定的哈希
EVAL_CONFIG_SHA = 'd4d7ad6cd3cf6aaf96006723dd70174ead25164bd8ee5f937a2f750663bf4bf9'
RUNNER_SHA = '874bffe20046d7b7e368651906f9d22a4403333d77dbdfc033f2457214d51ff4'
JOINT_BEST_SHA = '204dbffedc15fa2cc6053ce6b6d1aa1464f9978c103c5814f84692d69c80844e'
SWIN_SHA = '8dd8a7e20cfcc7228c896ef7880a81e6aaac61af858c3edbd1fe679f3c831314'
LDM_SHA = '2b90dcba5617d2e564097eb5771b2bfbcaaa381fe53de35e9204fe1479b83dd2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sealed(path):
    data = read(path)
    body = {k: v for k, v in data.items() if k != 'record_sha256'}
    if SIDE_RESUME in Path(path).parents:
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    else:
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    require(hashlib.sha256(raw.encode()).hexdigest() == data['record_sha256'], 'Invalid seal: ' + str(path))
    return data


@lru_cache(maxsize=256)
def file_hash(path, inode, size, modified, changed):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((inode, size, modified, changed) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'File changed: ' + str(path))
    return digest.hexdigest()


def descriptor(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), sha256=file_hash(path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def verify(rec, path=None):
    actual = Path(rec['path'])
    require(path is None or actual == path, 'Wrong bound path: ' + str(actual))
    require(descriptor(actual)['sha256'] == rec['sha256'], 'Changed bytes: ' + str(actual))
    return actual


def pinned(path, digest):
    rec = dict(path=str(path), sha256=digest)
    verify(rec)
    return rec


def task_target(architecture, targets):
    if isinstance(targets, str):
        targets = TARGETS if targets == 'both' else (targets,)
    items = tuple(targets)
    require(architecture in ARCHITECTURES, 'Unknown PL model: ' + str(architecture))
    if architecture == 'unet_multi':
        require(sorted(items) == sorted(TARGETS), 'Joint U-Net must serve both PL heads together')
        return 'both'
    require(len(items) == 1 and items[0] in TARGETS, 'Specify one PL target')
    return items[0]


def final_marker(architecture, target):
    if architecture == 'unet_multi':
        return JOINT_EVAL / 'eval_config.json'
    if architecture == 'unet_single':
        return SINGLE / 'training' / target / 'training_accepted.json'
    if architecture == 'unetplusplus' and target == 'plh':
        return SIDE_RESUME / 'independent_final_acceptance.json'
    if architecture == 'swin':
        return MAIN_RESUME / 'accepted' / ('swin_' + target + '.json')
    suffix = '_sigma' if architecture == 'ldm' else ''
    return RELEASE / 'acceptance' / (architecture + '_' + target + suffix + '.json')


def stage_record(path, folder, identity):
    data = sealed(path)
    require(data['status'] == 'passed', 'Unaccepted training: ' + str(path))
    verify(data['summary'], folder / 'training_summary.json')
    verify(data['BEST'], folder / 'checkpoint_best.pt')
    identity['evidence'][str(path)] = descriptor(path)
    identity['evidence'][str(folder / 'training_summary.json')] = data['summary']
    return data, read(data['summary']['path'])


def native_u_sources(identity, arch):
    config_pin = pinned(JOINT_EVAL / 'eval_config.json', EVAL_CONFIG_SHA)
    runner_pin = pinned(JOINT_EVAL / 'source/tools/uv_three_region_eval_20261025.py', RUNNER_SHA)
    cfg = read(config_pin['path'])
    identity.update(evaluation_config=config_pin, loader_source=runner_pin)
    key = {'unet_single': 'src/uv_uncertainty_unet.py', 'unet_multi': 'src/joint_uv_uncertainty_unet.py',
           'unetplusplus': 'src/uv_uncertainty_unetplusplus.py'}[arch]
    sources = cfg['sources']
    required = ['src/unet_baseline.py', key]
    if arch == 'unetplusplus':
        required.append('src/uv_uncertainty_unet.py')
    for name in required:
        verify(sources[name], JOINT_EVAL / 'source' / name)
    identity['model_source'] = sources[key]
    identity['dependencies'] = {k: sources[k] for k in required}
    return cfg


def u_identity(arch, target, marker, identity):
    cfg = native_u_sources(identity, arch)
    if arch == 'unet_multi':
        assets = cfg['models'][arch]
        for rec in assets.values():
            verify(rec)
        summary = read(assets['training_summary']['path'])
        require(assets['checkpoint']['sha256'] == JOINT_BEST_SHA and summary['training_complete']
                and summary['test_split_used'] is False and summary['selection']['best_epoch'] == 11,
                'Original joint macro-VAL BEST differs')
        identity.update(checkpoint=assets['checkpoint'], training_protocol=assets['training_protocol'],
                        training_summary=assets['training_summary'], selected_epoch=11,
                        selection_metric='full_VAL_equal_modality_macro_L1', output_keys=list(TARGETS))
        return
    if arch == 'unet_single':
        root = SINGLE / 'training' / target
        final = sealed(marker)
        audit_path = verify(final['audit'], SINGLE / 'matched_test/training_audits' / (target + '.json'))
        audit = sealed(audit_path)
        require(audit['target'] == target and audit['original_pipeline'] is True and
                audit['existing_frozen_UV_reference'] is False and audit['VAL_images_per_epoch'] == 10948 and
                final['checkpoint'] == audit['checkpoint'] and final['status'] == audit['status'] == 'passed',
                'Single U-Net final audit or target differs')
        verify(audit['checkpoint'], root / 'checkpoint_best.pt')
        verify(audit['summary'], root / 'training_summary.json')
        request = sealed(verify(audit['request'], SINGLE / 'request.json'))
        native = request['sources']['matched_phase_v2.py']
        verify(native, SINGLE / 'source/matched_phase_v2.py')
        identity.update(native_pl_forward_reference=native, checkpoint=audit['checkpoint'],
                        selected_epoch=audit['best_epoch'])
        identity['evidence'].update(audit=final['audit'], summary=audit['summary'])
        run_config = descriptor(SINGLE / 'configs' / (target + '.json'))
    else:
        key = arch + '_' + target
        side = target == 'plh'
        root = (PLUS_SIDE if side else RELEASE) / 'training' / key
        if side:
            final = sealed(marker)
            require(final['status'] == 'passed' and final['target'] == target and final['world'] == 3
                    and final['native_LAST_and_BEST_verify_checkpoint'] is True, 'Incomplete side final audit')
            audit_path = verify(final['native_audit'], SIDE_RESUME / 'training_acceptance' / (key + '.json'))
        else:
            audit_path = marker
        audit, summary = stage_record(audit_path, root, identity)
        require(audit['VAL_images_each'] == 10948 and audit.get('task', key) == key and
                audit.get('target', target) == target and summary['training_complete'] and
                summary['test_split_used'] is False, 'Incomplete U-Net++ training')
        if side:
            require(final['BEST'] == audit['BEST'], 'Side BEST reports differ')
        identity.update(checkpoint=audit['BEST'], selected_epoch=audit['selected_epoch'])
        request_root = PLUS_SIDE if side else RELEASE
        request = sealed(verify(audit['request'], request_root / 'request.json'))
        run_config = request['configs'][target if side else key]
        verify(run_config, request_root / 'configs' / (key + '.json'))
    protocol = descriptor(root / 'protocol.json')
    config = read(run_config['path'])
    summary = read(root / 'training_summary.json')
    require(config['target_modality'] == target and Path(config['output_dir']) == root
            and summary['checkpoint_best_sha256'] == identity['checkpoint']['sha256']
            and summary['protocol_sha256'] == protocol['sha256']
            and summary['selection']['best_epoch'] == identity['selected_epoch']
            and summary['test_split_used'] is False and summary['training_complete'] is True, 'Wrong target/BEST')
    require(read(protocol['path'])['config'] == config, 'Training config differs from frozen protocol')
    identity.update(training_protocol=protocol, training_config=run_config, selection_metric='full_VAL_target_L1',
                    output_keys=['mean'])


def swin_identity(target, marker, identity):
    key = 'swin_' + target
    root = SWIN / 'training' / key
    audit, summary = stage_record(marker, root, identity)
    req = sealed(SWIN / 'request.json')
    config = req['files']['configs/' + key + '.json']
    cfg = read(verify(config, SWIN / 'configs' / (key + '.json')))
    source = req['original_swin']['model']
    verify(source)
    binding = read(root / 'binding.json')
    payload = {k: v for k, v in binding.items() if k != 'binding'}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    require(source['sha256'] == SWIN_SHA and Path(cfg['output_dir']) == root and cfg == binding['config'] and
            signature == binding['binding'] == summary['binding'] and binding['model_sha256'] == SWIN_SHA and
            binding['world_size'] == 1 and summary['state']['complete'] and summary['test_accessed'] is False and
            audit['VAL_count_each'] == 10948, 'Swin target config, full validation, or binding differs')
    history = summary['state']['history']
    selected = min(history, key=lambda h: h['l1'])
    require(selected['l1'] == summary['state']['best'] and all(h['count'] == 10948 for h in history),
            'Swin full-VAL BEST differs')
    identity.update(model_source=source, checkpoint=audit['BEST'], training_config=config, binding=signature,
                    selected_epoch=selected['epoch'], selection_metric='full_VAL_target_L1', output_keys=['mean'])


def ldm_identity(target, marker, identity):
    key = 'ldm_' + target
    root = RELEASE / 'training' / key
    sigma_root = RELEASE / 'training' / (key + '_sigma')
    audit, summary = stage_record(marker, sigma_root, identity)
    require(audit['target'] == target and audit['VAL_images_each'] == 10948 and summary['status'] == 'complete'
            and summary['state']['complete'] and summary['test_accessed'] is False and
            summary['checkpoint_best_sha256'] == audit['BEST']['sha256'], 'Incomplete target sigma final audit')
    req = sealed(verify(audit['request'], RELEASE / 'request.json'))
    run_config = req['configs'][key]
    generator_cfg = read(verify(run_config, RELEASE / 'configs' / (key + '.json')))
    require(generator_cfg['target_modality'] == target and Path(generator_cfg['output_dir']) == root,
            'Wrong generator target configuration')
    bound_path = RELEASE / 'stage_bindings' / ('sigma_' + target + '.json')
    bound = sealed(bound_path)
    require(bound['target'] == target and bound['status'] == 'prepared_after_target_generator_full_audit',
            'Sigma/generator acceptance missing')
    config = bound['config']
    cfg = read(verify(config, RELEASE / 'configs' / (key + '_sigma.json')))
    require(Path(cfg['output_dir']) == sigma_root and cfg['seed'] == 20260817 and cfg['selection_metric'] == 'nll',
            'Wrong target sigma config')
    for stage, conf, field in (('ae', 'ae', 'AE_acceptance'), ('diffusion', 'denoiser', 'diffusion_acceptance')):
        path = verify(bound[field], RELEASE / 'acceptance' / (key + '_' + stage + '.json'))
        record, stage_summary = stage_record(path, root / stage, identity)
        require(record['stage'] == stage and record['task'] == key and record['VAL_images_each'] == 10948 and
                stage_summary['training_complete'] and stage_summary['test_split_used'] is False and
                record['BEST']['sha256'] == cfg[conf]['sha256'] and cfg[conf]['state_key'] == 'model_state_dict',
                'Stage BEST differs from sigma inputs: ' + stage)
        verify(cfg[conf], root / stage / 'checkpoint_best.pt')
    review_path = RELEASE / 'acceptance' / (key + '_fixed_VAL24_compression_review.json')
    review = sealed(review_path)
    require(review['status'] == 'passed' and review['target'] == target and len(review['rows']) == 24,
            'Missing original fixed-VAL compression review')
    stats = read(verify(cfg['latent_statistics'], root / 'latent_stats/statistics.json'))
    require(stats['images'] == 76642 and stats['test_split_used'] is False and
            stats['ae_checkpoint_sha256'] == cfg['ae']['sha256'], 'Foreign/incomplete TRAIN latent statistics')
    source = req['originals']['sigma_model']
    verify(source)
    require(source['sha256'] == LDM_SHA, 'Wrong native LDM generator source')
    selected = min(summary['state']['history'], key=lambda h: h['nll'])
    require(selected['nll'] == summary['state']['best'], 'Sigma BEST selection differs')
    identity['evidence'].update(stage_binding=descriptor(bound_path), fixed_val24=descriptor(review_path))
    identity.update(checkpoint=audit['BEST'], model_source=source, training_config=config, generator_config=run_config,
                    ae=cfg['ae'], denoiser=cfg['denoiser'], latent_statistics=cfg['latent_statistics'],
                    binding=summary['binding'], selected_epoch=selected['epoch'], selection_metric='full_VAL_sigma_NLL',
                    output_keys=['mean'], sampler=dict(ddim_steps=20, seed=20260817, generation_index=0,
                                                       sample_id_key='sample_id'))


def model_identity(architecture, targets):
    target = task_target(architecture, targets)
    marker = final_marker(architecture, target)
    marker_pin = descriptor(marker)
    result = dict(architecture=architecture, target=target, targets=list(TARGETS) if target == 'both' else [target],
                  acceptance=marker_pin, evidence={}, precision='FP32_no_autocast_no_TF32')
    if architecture.startswith('unet'):
        u_identity(architecture, target, marker, result)
    elif architecture == 'swin':
        swin_identity(target, marker, result)
    else:
        ldm_identity(target, marker, result)
    result['checkpoint_sha256'] = result['checkpoint']['sha256']
    return result


def probe_ready(architecture, targets):
    target = task_target(architecture, targets)
    marker = final_marker(architecture, target)
    if not marker.is_file() and architecture != 'unet_multi':
        return None
    try:
        return model_identity(architecture, target)
    except FileNotFoundError as error:
        raise RuntimeError('Incomplete artifacts for final acceptance ' + str(marker)) from error


def import_source(rec):
    path = verify(rec)
    name = '_frozen_pl_test_' + rec['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def load_model(architecture, target, device='cuda:0'):
    import torch
    info = model_identity(architecture, target)
    device = torch.device(device)
    # 沿用测试精度与确定性
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if architecture.startswith('unet'):
        runner = import_source(info['loader_source'])
        eval_cfg = read(info['evaluation_config']['path'])
        frozen = runner.FrozenSources(eval_cfg['sources'], JOINT_EVAL)
        if architecture == 'unet_multi':
            model = runner.load_model(architecture, eval_cfg['models'][architecture], JOINT_EVAL, frozen, str(device))
        else:
            pin = runner.MODEL_PINS[architecture]
            model = getattr(frozen.load(pin['source']), pin['cls'])(base_channels=32)
            blob = torch.load(verify(info['checkpoint']), map_location='cpu', weights_only=False)
            require(blob['protocol_sha256'] == info['training_protocol']['sha256'] and
                    blob['state']['selection']['best_epoch'] == info['selected_epoch'] and
                    blob['test_split_used'] is False, 'Wrong target BEST checkpoint')
            model.load_state_dict(blob['model_state_dict'], strict=True)
            require(sum(v.numel() for v in model.parameters()) == pin['parameters'], 'Architecture differs')
    else:
        source = import_source(info['model_source'])
        cfg = read(info['training_config']['path'])
        blob = torch.load(verify(info['checkpoint']), map_location='cpu', weights_only=False)
        require(blob['binding'] == info['binding'] and blob['state']['history'][-1]['epoch'] == info['selected_epoch'],
                'BEST is not bound to selected complete validation epoch')
        if architecture == 'swin':
            require(blob['world_size'] == 1, 'Wrong target Swin training world')
            model = source.SwinUVUncertainty(activation_checkpointing=cfg.get('activation_checkpointing', True))
            model.load_state_dict(blob['model'], strict=True)
        else:
            require(blob['checkpoint_kind'] == 'posttraining_sigma_head_only' and blob['world_size'] == 2,
                    'Wrong target LDM stage-C checkpoint')
            model = source.build_training_model(cfg, device)
            model.log_variance_head.load_state_dict(blob['head_state_dict'], strict=True)
    model = model.float().eval().requires_grad_(False).to(device)
    return PLPredictor(info, model, device)


class PLPredictor:
    def __init__(self, identity, model, device):
        self.identity = identity
        self.model = model
        self.device = device

    def predict(self, case):
        import numpy as np
        import torch
        row = case['row']
        require(row['split'] == 'test' and isinstance(row['sample_id'], str) and row['sample_id'] and
                ('group_id' not in case or case['group_id'] == row['sample_id']), 'Unbound frozen TEST sample')
        rgb = case['rgb']
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.float32 and rgb.shape == (3, 1024, 1024) and
                np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, 'Expected original FP32 RGB1024')
        batch = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        arch = self.identity['architecture']
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            output = self.model(batch, [row['sample_id']]) if arch == 'ldm' else self.model(batch)
        means = {}
        for target in self.identity['targets']:
            tensor = output[target if arch == 'unet_multi' else 'mean']
            require(tensor.shape == (1, 3, 1024, 1024) and tensor.dtype == torch.float32 and
                    bool(torch.isfinite(tensor).all()) and bool((tensor >= 0).all()) and bool((tensor <= 1).all()),
                    'Invalid encoded RGB mean: ' + target)
            means[target] = tensor[0]
        return means
