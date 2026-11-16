"""PL TEST input binding and one-forward reconstruction/agreement measurements.

The case adapter reads the original TEST catalog, true-UV domain receipts, and
the archived six anatomy maps. Only the source RGB enters the model. The caller
passes its model's outputs here once, for measurement against the actual PL
images on those same supports. No prediction or new mask is persisted here.
"""
import hashlib
import importlib.util
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


EXPERIMENTS = Path('/study/siiba/experiments')
PL_PIPELINE = 'pl_controls_original_uv_pipeline_20261031_v1'
OBJECTS = 'signal_objects_3cd_20261029_v1'
OLD = 'three_region_test_eval_20261025_v1'
DOMAINS = ('P', 'D', 'O', 'F')
ROIS = ('forehead', 'image_left_cheek', 'image_right_cheek', 'nose', 'chin', 'F_reference')
SIGNALS = ('R', 'G', 'B', 'RGB_mean_secondary')
METRICS = ('l1', 'mse', 'psnr', 'ssim')
GRID = (1024, 1024)
PINS = {
    'request': '3a8655b0ddb272976f73f51a2cec56bf3e70282c4c3b4369af88c76a8541c8c8',
    'catalog': '81fae57d92370e825950fe98c584bce5177a999dcb298cbee447efce7f971d89',
    'objects_protocol': 'd7fd5ad88b042b3bd551d92cd3efd2a67cf52b376e2c7b8e4c9b3716401f93da',
    'audited_inputs': '01f9e286a2cbc868383e34dc4753d9479549442aee855a7544d094d6be9212c5',
    'old_runner': '874bffe20046d7b7e368651906f9d22a4403333d77dbdfc033f2457214d51ff4',
    'eval_config': 'd4d7ad6cd3cf6aaf96006723dd70174ead25164bd8ee5f937a2f750663bf4bf9',
    'mask_contract': '0ab265ad6470a6072e947fb2a69404b125c5abd30305dad386a241712daff067',
}
# 组指标精度容差
# CPU 双精度，CUDA 单精度
ATOL = dict(l1=2e-7, mse=2e-7, psnr=2e-5, ssim=3e-4, intensity=2e-11, area=1e-7)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verified(path, expected):
    path = Path(path)
    require(sha(path.read_bytes()) == expected, 'Frozen file SHA differs: ' + str(path))
    return path


def bound(descriptor):
    return verified(descriptor['path'], descriptor['sha256'])


def sealed(value):
    body = {key: val for key, val in value.items() if key != 'record_sha256'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    require(sha(encoded.encode()) == value['record_sha256'], 'Invalid archived record seal')
    return value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def audited_stages(object_root):
    """Read the historical audit manifest once; keep only the 21,899 stage hashes."""
    audit = object_root / 'report/independent_audit_v1/consumed_files_sha256.jsonl'
    digest = hashlib.sha256()
    hashes = {}
    with audit.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if b'/ready.json"' not in line:
                continue
            item = json.loads(line)
            path = Path(item['path'])
            if path.name != 'ready.json' or path.parent.parent != object_root / 'staged':
                continue
            index = int(path.parent.name)
            require(path.parent.name == '%06d' % index and index not in hashes,
                    'Duplicate or noncanonical staged path in accepted audit')
            hashes[index] = item['sha256']
    require(digest.hexdigest() == PINS['audited_inputs'] and set(hashes) == set(range(21899)),
            'Independent input manifest or its complete stage inventory differs')
    return hashes


class Context:
    def __init__(self, root=EXPERIMENTS):
        self.root = Path(root)
        self.object_root = self.root / OBJECTS
        self.request = sealed(read(verified(self.root / PL_PIPELINE / 'request.json', PINS['request'])))
        self.image = load('_pltest_frozen_image', bound(self.request['sources']['data.py'])).image
        self.old = load('_pltest_frozen_domain',
                        verified(self.root / OLD / 'source/tools/uv_three_region_eval_20261025.py',
                                 PINS['old_runner']))
        self.catalog = read(verified(self.root / OLD / 'catalog.json', PINS['catalog']))
        selection = read(bound(self.request['selection']))
        self.rows = self.catalog['rows']
        self.valid = frozenset(selection['geometry_valid_indices'])
        ids = [row['sample_id'] for row in self.rows]
        require(len(set(ids)) == len(ids) == 21899 and ids == selection['all_group_ids'] and
                self.valid <= set(range(len(ids))) and len(self.valid) == 21422 and
                all(row['split'] == 'test' for row in self.rows) and
                self.catalog['split'] == 'test' and self.catalog['data_root'] == self.request['data_root'],
                'Frozen TEST cohort, order or data root differs')
        verified(self.object_root / 'protocol.json', PINS['objects_protocol'])
        self.stage_hashes = audited_stages(self.object_root)

    def case(self, index, targets):
        if isinstance(targets, str):
            targets = (targets,)
        else:
            targets = tuple(targets)
        require(targets and set(targets) <= {'plh', 'plv'} and len(set(targets)) == len(targets),
                'Requested heads must be distinct PL targets')
        require(type(index) is int and 0 <= index < len(self.rows), 'Invalid TEST index')
        row = self.rows[index]
        ready_path = self.object_root / ('staged/%06d/ready.json' % index)
        raw = ready_path.read_bytes()
        ready_sha = sha(raw)
        require(ready_sha == self.stage_hashes[index], 'Staged case differs from completed independent audit')
        ready = sealed(json.loads(raw))
        gid = row['sample_id']
        require((ready['index'], ready['group_id'], ready['row'], ready['protocol_sha256']) ==
                (index, gid, row, PINS['objects_protocol']), 'Original staged case mismatch')
        mask_path = bound(ready['mask_receipt'])
        receipt = self.old.validate_mask_receipt(read(mask_path), row, index, PINS['catalog'],
                                                PINS['mask_contract'], self.catalog['data_root'])
        ref = self.old.verify_seal(read(bound(ready['mask_ref'])))
        require(ref['receipt'] == ready['mask_receipt'] and ref['status'] == receipt['status'] and
                (ref['index'], ref['sample_id']) == (index, gid), 'Archived mask ref and receipt disagree')
        inputs = dict(ready=dict(path=str(ready_path), sha256=ready_sha),
                      audited_inputs_sha256=PINS['audited_inputs'], mask_ref=ready['mask_ref'],
                      mask_receipt=ready['mask_receipt'], uv=row['uv'], rgb=row['rgb'],
                      targets={t: row[t] for t in targets}, catalog_sha256=PINS['catalog'])
        case = dict(index=index, group_id=gid, row=row, targets=targets, inputs=inputs,
                    rgb=None, truths={}, weights=None, rois=None, missing_reason=None)
        if index not in self.valid:
            require(receipt['status'] in self.old.TECHNICAL, 'Original technical status required')
            self.old.validate_technical_evidence(receipt, mask_path)
            case['missing_reason'] = receipt['status']
            return case
        require(receipt['status'] == 'complete', 'Valid case missing frozen UV geometry')
        with np.load(bound(receipt['npz']), allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        self.old.validate_weights(weights)
        self.old.validate_projection_audit(receipt, mask_path, weights)
        with np.load(bound(ready['geometry']), allow_pickle=False) as archive:
            require(set(archive.files) == set(ROIS) | {'F_coverage'}, 'Anatomical cache fields differ')
            require(np.array_equal(archive['F_coverage'], weights['F']), 'Anatomical F coverage differs')
            rois = {name: archive[name] for name in ROIS}
        require(ready['landmarks'] == receipt['evidence']['geometry/landmarks.json'] and
                ready['roi_audit']['landmarks'] == ready['landmarks'] and
                np.array_equal(rois['F_reference'], weights['F']), 'Frozen landmark or F reference differs')
        for name in ROIS[:-1]:
            a = rois[name]
            require(a.shape == GRID and a.dtype == np.float32 and
                    sha(a.tobytes(order='C')) == ready['roi_audit']['regions'][name]['weight_array_sha256'],
                    'Archived ROI array differs: ' + name)
        rgb = self.image(row, 'rgb', self.catalog['data_root'])
        require(sha(rgb.tobytes(order='C')) == ready['inputs']['rgb']['sha256'],
                'Preprocessed RGB differs from original five-model TEST tensor')
        case.update(rgb=rgb, truths={t: self.image(row, t, self.catalog['data_root']) for t in targets},
                    weights=weights, rois=rois)
        inputs.update(anatomical_geometry=ready['geometry'], landmarks=ready['landmarks'],
                      fractional_masks=receipt['npz'], rgb_tensor_sha256=ready['inputs']['rgb']['sha256'])
        return case


def frozen_gpu(request):
    # 按原名加载依赖
    # 构建后还原模块表
    names = ('chain_common', 'gpu_math')
    original = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules[name] = load('_pltest_' + name, bound(request['sources'][name + '.py']))
        module = load('_pltest_original_gpu', bound(request['sources']['gpu_metrics.py']))
        return module.GPU()
    finally:
        for name, previous in original.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def region_record(values, area, center_area):
    row = dict(zip(METRICS, values))
    row.update(effective_area=area, ssim_center_area=center_area)
    for metric in METRICS:
        v = row[metric]
        reason = None
        if v is None:
            if metric == 'ssim' and not center_area:
                reason = 'no_valid_window_centers_in_region'
            elif not area:
                reason = 'empty_region_after_model_crop'
            elif metric == 'psnr' and row['mse'] == 0:
                reason = 'zero_mse_psnr_positive_infinity'
            else:
                raise ValueError('Unexplained null ' + metric)
        else:
            require(math.isfinite(v), 'Nonfinite ' + metric)
        row[metric + '_reason'] = reason
    return row


def signals(values):
    three = list(values)
    mean = sum(three) / 3 if three[0] is not None else None
    return dict(zip(SIGNALS, three + [mean]))


def anatomy_gpu(truth, pred, roi_matrix, areas):
    import torch
    images = torch.cat((torch.as_tensor(truth, device=pred.device), pred), dim=0)
    # 区域均值双精度累加
    # 空区域无强度值
    sums = (images.flatten(1).double() @ roi_matrix).cpu().numpy()
    result = {}
    for j, name in enumerate(ROIS):
        a = areas[j]
        values = sums[:, j] / a if a else [None] * 6
        result[name] = dict(truth=signals(values[:3]), prediction=signals(values[3:]),
                            effective_area=a, reason=None if a else 'empty_anatomical_roi')
    return result


class Measurement:
    def __init__(self, device='cuda:0', root=EXPERIMENTS):
        import torch
        self.device = torch.device(device)
        require(self.device in (torch.device('cuda:0'), torch.device('cpu')),
                'Bind the designated card as cuda:0 or select CPU for a pilot')
        self.root = Path(root)
        q = sealed(read(verified(self.root / PL_PIPELINE / 'request.json', PINS['request'])))
        self.gpu = frozen_gpu(q) if self.device.type == 'cuda' else None
        self.numeric = None

    def numeric_helpers(self):
        if self.numeric is None:
            path = verified(self.root / OLD / 'eval_config.json', PINS['eval_config'])
            old = load('_pltest_cpu_oracle', verified(self.root / OLD / 'source/tools/uv_three_region_eval_20261025.py',
                                                      PINS['old_runner']))
            self.numeric = old.FrozenSources(read(path)['sources'], path.parent).load(
                'tools/uv_architecture_comparison_metrics_v10.py')
        return self.numeric

    def checked_prediction(self, prediction):
        import torch
        y = torch.as_tensor(prediction, device=self.device)
        if tuple(y.shape) == (1, 3, *GRID):
            y = y[0]
        require(tuple(y.shape) == (3, *GRID) and y.dtype == torch.float32 and not y.requires_grad and
                bool(torch.isfinite(y).all()) and float(y.min()) >= 0 and float(y.max()) <= 1,
                'Frozen prediction must be RGB1024 float32 [0,1]')
        return y

    def measure(self, case, predictions):
        import torch
        if case['missing_reason'] is not None:
            require(not predictions, 'Unexpected forward on technical case')
            return {t: dict(status='unavailable_geometry', missing_reason=case['missing_reason'],
                            regions=None, anatomy=None) for t in case['targets']}
        require(set(predictions) == set(case['targets']) == set(case['truths']), 'PL target/head mismatch')
        roi_matrix = torch.as_tensor(np.stack([case['rois'][k] for k in ROIS]),
                                     device=self.device, dtype=torch.float64).flatten(1).T.contiguous()
        roi_areas = roi_matrix.sum(0).cpu().tolist()
        output = {}
        for target in case['targets']:
            y = self.checked_prediction(predictions[target])
            x = case['truths'][target]
            if self.gpu is not None:
                values, areas, centers = self.gpu.measure(x, y, case['weights'])
                regions = {d: region_record(v, a, c) for d, v, a, c in zip(DOMAINS, values, areas, centers)}
            else:
                regions = cpu_regions(self.numeric_helpers(), x, y.cpu().numpy(), case['weights'])
            output[target] = dict(status='ok', missing_reason=None, regions=regions,
                                  anatomy=anatomy_gpu(x, y, roi_matrix, roi_areas))
        return output

    def check(self, case, predictions, result):
        """CPU scalar/SSIM oracle and independent anatomical sums for a fixed QC group."""
        if case['missing_reason'] is not None:
            require(not predictions and result == self.measure(case, {}), 'Technical parity mismatch')
            return dict(status='passed', metric_cells=0, missing_reason=case['missing_reason'])
        numeric = self.numeric_helpers()
        diagnostics = []
        for t in case['targets']:
            x = numeric.validate_image(case['truths'][t], 'PL truth')
            y = self.checked_prediction(predictions[t]).detach().cpu().numpy().astype(np.float64)
            cpu = cpu_regions(numeric, x, y, case['weights'])
            for d in DOMAINS:
                row = result[t]['regions'][d]
                for name in METRICS + ('effective_area', 'ssim_center_area'):
                    compare(diagnostics, (t, d, name), row[name], cpu[d][name], ATOL.get(name, ATOL['area']))
            for name in ROIS:
                w = case['rois'][name].astype(np.float64)
                a = float(w.sum())
                row = result[t]['anatomy'][name]
                compare(diagnostics, (t, name, 'effective_area'), row['effective_area'], a, ATOL['area'])
                for label, img in [('truth', x), ('prediction', y)]:
                    values = signals([float(np.sum(ch * w) / a) for ch in img] if a else [None]*3)
                    for ch in SIGNALS:
                        compare(diagnostics, (t, name, label, ch), row[label][ch], values[ch], ATOL['intensity'])
        return dict(status='passed', metric_cells=len(diagnostics), diagnostics=diagnostics, absolute_tolerances=ATOL)


def cpu_regions(numeric, truth, prediction, weights):
    x = numeric.validate_image(truth, 'PL truth')
    y = numeric.validate_image(prediction, 'PL prediction')
    delta = y - x
    absolute, squared = np.abs(delta).mean(axis=0), (delta * delta).mean(axis=0)
    ssim = numeric.ssim_map_rgb(x, y)
    radius = numeric.SSIM_RADIUS
    result = {}
    for name in DOMAINS:
        w = weights[name].astype(np.float64)
        c = w[radius:-radius, radius:-radius]
        mse = numeric._weighted_mean(squared, w, 'MSE')
        values = [numeric._weighted_mean(absolute, w, 'L1'), mse, numeric._psnr(mse),
                  numeric._weighted_mean(ssim, c, 'SSIM')]
        result[name] = region_record(values, float(w.sum()), float(c.sum()))
    return result


def compare(diagnostics, labels, actual, expected, tolerance):
    require((actual is None) == (expected is None), 'Undefined CPU/CUDA mismatch ' + '/'.join(labels))
    diff = None if actual is None else abs(actual - expected)
    require(diff is None or diff <= tolerance, 'CPU/CUDA mismatch %s: %r' % ('/'.join(labels), diff))
    diagnostics.append(dict(cell=list(labels), cpu=expected, measured=actual, absolute_difference=diff))
