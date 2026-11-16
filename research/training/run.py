"""Select a study model and target, then run its native preparation, preflight or training CLI."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from training import unetplus_two_rank, unetplus_three_rank

TRAINERS = {
    'single_unet': 'train_single_uv_uncertainty_clean_20261014',
    'multi_unet': 'train_joint_uv_uncertainty_clean_20261014',
    'unetplusplus': 'train_unetplusplus_uv_uncertainty_clean_20261015',
    'swin': 'swin_train',
    'ldm': 'train_uv_ldm_clean_multigpu_20261016',
    'sigma': 'ldm_sigma_train',
}


def target_row(row, target):
    path, fingerprint = row[target+'_path'], row[target+'_sha256']
    if not path or len(fingerprint) != 64:
        raise ValueError('Missing target identity in private manifest')
    return dict(row, uv_path=path, uv_sha256=fingerprint)


def remap_frozen(splits, config, target):
    path = Path(config['source_manifest'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['source_manifest_sha256']:
        raise ValueError('Manifest hash differs from the frozen target config')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        subjects = {key: [] for key in ('train', 'val')}
        for row in csv.DictReader(stream):
            if row['split'] in subjects and row['eligible_primary'] == '1':
                subjects[row['split']].append(row)
    result = {}
    for split, original in splits.items():
        if split not in subjects or len(original) != len(subjects[split]):
            raise ValueError('Target mapping changed cohort coverage')
        result[split] = []
        for saved, raw in zip(original, subjects[split]):
            if any(saved[k] != raw[k] for k in ('sample_id','split','rgb_path','uv_path')):
                raise ValueError('Frozen sample ordering or image identity changed')
            mapped = target_row(raw,target)
            result[split].append(dict(saved, uv_path=mapped["uv_path"], uv_sha256=mapped["uv_sha256"]))
    return result


def select_module(arch, target, config):
    filename = TRAINERS[arch]
    if arch == 'unetplusplus' and target != 'uv':
        path = HERE / (filename+'.py')
        transform = unetplus_three_rank.transform if target=='plh' else unetplus_two_rank.transform
        tree, _ = transform(path.read_text(encoding='utf-8'))
        mod = types.ModuleType('training.'+filename+'_'+target)
        mod.__file__ = str(path)
        sys.modules[mod.__name__] = mod
        exec(compile(tree,str(path),'exec'),mod.__dict__)
    else:
        mod = importlib.import_module('training.'+filename)
    if target in ('uv','joint'):
        return mod
    if arch == 'single_unet':
        select = mod.select_training_rows
        def mapped(rows, counts=None):
            return select((target_row(row,target) for row in rows if row.get('split') in ('train','val')),counts)
        mod.select_training_rows = mapped
    elif arch in ('unetplusplus','ldm'):
        load = mod.load_frozen
        def mapped(path):
            c, out, protocol, fingerprint, rows = load(path)
            return c,out,protocol,fingerprint,remap_frozen(rows,c,target)
        mod.load_frozen = mapped
    elif arch == 'swin':
        load = mod.load_model_source
        def mapped(path):
            source = load(path)
            read = source.training_data
            def remap(c):
                return {split:[target_row(r,target) for r in rows] for split,rows in read(c).items()}
            source.training_data = remap
            return source
        mod.load_model_source = mapped
    elif arch == 'sigma':
        read = mod.verified_data
        mod.verified_data = lambda source,c: remap_frozen(read(source,c),c,target)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=sorted(TRAINERS))
    p.add_argument('target', choices=['uv','plh','plv','joint'])
    p.add_argument('--config', type=Path, help='Private run config or the supplied public config')
    parsed,native = p.parse_known_args()
    arch,target = parsed.model,parsed.target
    if (arch=='multi_unet') != (target=='joint'):
        p.error('Use joint for multi_unet and an individual target for all other models')
    name = arch if target=='joint' else arch+'_'+target
    path = parsed.config or HERE/'configs'/(name+'.json')
    cfg = json.loads(path.read_bytes())
    if arch!='multi_unet' and cfg.get('target_modality','uv')!=target:
        p.error('target_modality in the configuration must match the requested target')
    mod = select_module(arch,target,cfg)
    sys.argv = [mod.__file__,'--config',str(path),*native]
    if arch in ('swin','sigma'):
        sys.argv.extend(['--model-source',str(HERE.parent/'models'/('swin_model.py' if arch=='swin' else 'ldm_sigma_model.py'))])
    mod.main()


if __name__ == '__main__':
    main()
