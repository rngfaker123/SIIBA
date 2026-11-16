"""Versioned FP32 backend agreement; does not redefine any scientific metric.

These engineering limits are frozen before the independent CUDA validation.
Old CPU strict comparisons are always reported separately, including failures.
"""
import math

THRESHOLDS = {
    'schema': 'cuda_fp32_execution_thresholds_v1',
    'rule': 'abs(candidate-reference) <= atol + rtol * abs(reference)',
    'original_cpu_strict': {'atol': 1e-9, 'rtol': 1e-7},
    'tensor': {'atol': 2e-5, 'rtol': 0.0},
    'table3': {k: {'atol': (2e-4 if k == 'psnr' else 2e-6), 'rtol': 0.0}
               for k in ('l1', 'mse', 'mean_sigma', 'ssim', 'psnr')},
    'table4': {'gaussian_nll': {'atol': 2e-4, 'rtol': 2e-5},
               'mean_z2': {'atol': 2e-4, 'rtol': 2e-5},
               'rank_corr': {'atol': 2e-4, 'rtol': 0.0},
               'coverage95': {'atol': 2e-5, 'rtol': 0.0}},
    'regional_prediction': {'atol': 2e-6, 'rtol': 0.0},
    'exact_fields': 'Truth, ROI geometry, areas, counts, null/reason/status fields and all unlisted fields must match exactly.'
}


def normalized_thresholds(value):
    if value is None or value == THRESHOLDS:
        return THRESHOLDS
    expected = {k:v for group in ('table3', 'table4') for k,v in THRESHOLDS[group].items()}
    expected['regional_prediction'] = THRESHOLDS['regional_prediction']
    if (value.get('schema') != 'cuda_fp32_execution_acceptance_v1' or
        value.get('original_cpu_rtol') != 1e-7 or value.get('original_cpu_atol') != 1e-9 or
        value.get('tensor_max_absolute_difference') != 2e-5 or value.get('metrics') != expected or
        value.get('precision') != 'FP32_no_autocast_no_TF32'):
        raise ValueError('Unfrozen backend acceptance definition')
    return THRESHOLDS


def compare_payload(reference, candidate, kind, thresholds=None):
    limits = normalized_thresholds(thresholds)
    if limits != THRESHOLDS or kind not in ('table3', 'table4', 'regional'):
        raise ValueError('Unfrozen backend thresholds or unknown comparison kind')
    violations, strict_violations, maximum, count = [], [], 0.0, 0

    def reject(path, reason):
        violations.append({'path': path, 'reason': reason})
        strict_violations.append({'path': path, 'reason': reason})

    def walk(a, b, path):
        nonlocal maximum, count
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                reject(path, 'mapping_keys_differ')
                return
            for key in sorted(a):
                walk(a[key], b[key], path + '/' + key)
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                reject(path, 'sequence_structure_differs')
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + '/' + str(index))
        elif a is None or isinstance(a, (bool, str, int)):
            if type(a) is not type(b) or a != b:
                reject(path, 'exact_value_or_type_differs')
        elif isinstance(a, float):
            if not isinstance(b, (int, float)) or isinstance(b, bool) or not math.isfinite(a) or not math.isfinite(b):
                reject(path, 'nonfinite_or_numeric_type_differs')
                return
            difference = abs(a-b)
            maximum, count = max(maximum, difference), count+1
            last = path.rsplit('/', 1)[-1]
            allowed = {'atol': 0.0, 'rtol': 0.0}
            if kind in ('table3', 'table4') and last in limits[kind]:
                allowed = limits[kind][last]
            elif kind == 'regional' and '/prediction/' in path:
                allowed = limits['regional_prediction']
            bound = allowed['atol'] + allowed['rtol']*abs(a)
            row = dict(path=path, reference=a, candidate=b, absolute_difference=difference,
                       absolute_limit=bound, **allowed)
            if difference > bound:
                violations.append(row)
            original = limits['original_cpu_strict']
            if not math.isclose(a, b, rel_tol=original['rtol'], abs_tol=original['atol']):
                strict_violations.append(row)
        else:
            reject(path, 'unsupported_field_type')

    walk(reference, candidate, kind)
    return dict(schema='cuda_fp32_comparison_v1', kind=kind,
                status='passed' if not violations else 'failed', numeric_fields_compared=count,
                maximum_absolute_difference=maximum, violation_count=len(violations), violations=violations,
                original_strict=dict(status='passed' if not strict_violations else 'failed',
                    violation_count=len(strict_violations), violations=strict_violations,
                    **limits['original_cpu_strict']))


def compare_tensors(reference, candidate, thresholds=None):
    import numpy as np
    limits = normalized_thresholds(thresholds)
    if limits != THRESHOLDS or set(reference) != {'prediction', 'sigma'} or set(candidate) != set(reference):
        raise ValueError('Unfrozen tensor comparison contract')
    result = {}
    for name in ('prediction', 'sigma'):
        a, b = np.asarray(reference[name]), np.asarray(candidate[name])
        if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Tensor geometry/type/nonfinite mismatch')
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        limit = limits['tensor']['atol'] + limits['tensor']['rtol']*np.abs(a.astype(np.float64))
        strict = limits['original_cpu_strict']
        # Original scalar math.isclose uses max(atol, rtol*max(abs(a),abs(b))).
        strict_limit = np.maximum(strict['atol'], strict['rtol']*np.maximum(np.abs(a),np.abs(b)))
        result[name] = dict(shape=list(a.shape), dtype='float32', exact=bool(np.array_equal(a,b)),
            maximum_absolute_difference=float(difference.max()),
            backend_violation_count=int(np.count_nonzero(difference > limit)),
            original_strict_violation_count=int(np.count_nonzero(difference > strict_limit)))
    return dict(schema='cuda_fp32_tensor_comparison_v1', status='passed' if all(v['backend_violation_count']==0 for v in result.values()) else 'failed',
                fields=result, original_strict_pass=all(v['original_strict_violation_count']==0 for v in result.values()))
