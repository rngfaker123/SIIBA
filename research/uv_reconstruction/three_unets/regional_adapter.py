"""Regional statistics adapter for versioned CPU-preserving CUDA evaluation.

No inference, new masks, calibration fitting, or old-output writes occur here.
The caller supplies the hash-verified original regional/statistics modules.
"""
from collections import Counter
import csv
import html
import io
import json
import math
from pathlib import Path

import numpy as np

REGIONS = ('forehead', 'image_left_cheek', 'image_right_cheek', 'nose', 'chin', 'F_reference')
SIGNALS = ('R', 'G', 'B', 'RGB_mean_secondary')
FIELDS = ('status', 'reason', 'regions', 'roi_audit', 'parity')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_fields(value):
    need(set(value) == set(FIELDS), 'Regional field schema differs')
    if value['status'] == 'unavailable_geometry':
        need(isinstance(value['reason'], str) and bool(value['reason']) and
             value['regions'] is None and value['roi_audit'] is None and value['parity'] is None,
             'Technical regional fields must be explicit and unavailable')
        return value
    need(value['status'] == 'complete' and value['reason'] is None and
         isinstance(value['roi_audit'], dict) and set(value['regions']) == set(REGIONS),
         'Invalid complete regional record')
    need(isinstance(value['roi_audit'].get('landmark_source'), str) and
         bool(value['roi_audit']['landmark_source']), 'Regional landmark source missing')
    for region in REGIONS:
        item = value['regions'][region]
        need(set(item) == {'truth', 'prediction', 'effective_area', 'reason'}, 'Unexpected regional measurement fields')
        area = item['effective_area']
        need(not isinstance(area, bool) and isinstance(area, (float, int)) and math.isfinite(area) and area >= 0,
             'Invalid regional area')
        for method in ('truth', 'prediction'):
            values = item[method]
            need(set(values) == set(SIGNALS), 'Incomplete regional signal set')
            if area == 0:
                need(item['reason'] == 'empty_anatomical_roi' and all(v is None for v in values.values()),
                     'Empty ROI is not an explicit null')
            else:
                need(item['reason'] is None and all(not isinstance(v, bool) and isinstance(v, (float, int)) and
                     math.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in values.values()),
                     'Invalid regional channel measurement')
                need(math.isclose(values['RGB_mean_secondary'], sum(values[k] for k in ('R', 'G', 'B')) / 3,
                     rel_tol=1e-12, abs_tol=1e-14), 'Secondary RGB mean is not the arithmetic channel mean')
    return value


def verify_old_record(r, c, ctx, index, model, path, original_record_path,
                      mask_ref_path, mask_receipt_path, expected_protocol_sha):
    """Reuse original scalars exactly after checking their complete provenance."""
    value = c.checked(r.read(path))
    row = ctx['rows'][index]
    need(value['schema'] == 'regional_signal_agreement_sample_v1' and
         value['index'] == index and value['sample_id'] == row['sample_id'] and
         value['subject_id'] == row['subject_id'] and value['model_id'] == model and
         value['checkpoint_sha256'] == ctx['old'].MODEL_PINS[model]['sha'] and
         value['protocol_sha256'] == expected_protocol_sha and
         value['source_row_sha256'] == c.digest(row) and
         value['original_record'] == r.descriptor(original_record_path) and
         value['mask_ref'] == r.descriptor(mask_ref_path) and
         value['mask_receipt'] == r.descriptor(mask_receipt_path), 'Original regional reuse binding differs')
    prior = c.checked(r.read(original_record_path))
    need(prior['index'] == index and prior['sample_id'] == row['sample_id'] and prior['model_id'] == model and
         prior['checkpoint_sha256'] == value['checkpoint_sha256'] and prior['status'] == value['status'],
         'Regional original combined record differs')
    fields = {k: value[k] for k in FIELDS}
    validate_fields(fields)
    if fields['status'] == 'complete':
        need(isinstance(fields['parity'], dict) and fields['parity'].get('status') == 'passed',
             'Reused regional parity did not pass')
        r.check_desc(fields['roi_audit']['landmarks'])
    return fields


def prepare_geometry(r, c, ctx, row, receipt_path, receipt):
    return r.roi_weights(c, ctx, row, receipt_path, receipt)


def measure_same_forward(r, truth, prediction, rois, audit, parity=None):
    """Use the original intensity function on one already-produced UV output."""
    target, predicted = r.intensities(truth, rois), r.intensities(prediction, rois)
    need(set(target) == set(predicted) == set(REGIONS), 'Anatomical regions differ')
    values = {}
    for region in REGIONS:
        need(target[region]['area'] == predicted[region]['area'] and
             target[region]['reason'] == predicted[region]['reason'], 'Truth/prediction masks differ')
        values[region] = dict(truth=target[region]['values'], prediction=predicted[region]['values'],
                              effective_area=target[region]['area'], reason=target[region]['reason'])
    return validate_fields(dict(status='complete', reason=None, regions=values, roi_audit=audit, parity=parity))


def unavailable(reason):
    return validate_fields(dict(status='unavailable_geometry', reason=reason,
                                regions=None, roi_audit=None, parity=None))


def collect(c, ctx, contract, read_unified, contract_sha):
    """Independently validate regional fields and matched populations in all models."""
    rows, models = ctx['rows'], tuple(contract['models'])
    need(contract['sample_count'] == len(rows) and len(models) == 3 and len(set(models)) == 3,
         'Regional cohort/model inventory differs')
    data = np.full((len(models), len(REGIONS), len(SIGNALS), len(rows), 2), np.nan, dtype=np.float64)
    missing = {(mi, ri): Counter() for mi in range(len(models)) for ri in range(len(REGIONS))}
    statuses = {m: Counter() for m in models}
    provenance = {m: Counter() for m in models}
    landmarks, manifest, reasons = Counter(), [], {}
    for index, row in enumerate(rows):
        same = None
        for mi, model in enumerate(models):
            record = c.checked(read_unified(model, index))
            need(record['schema'] == 'test_gpu_sample_v1' and record['index'] == index and
                 record['sample_id'] == row['sample_id'] and record['subject_id'] == row['subject_id'] and
                 record['model_id'] == model and record['source_row_sha256'] == c.digest(row) and
                 record['contract_sha256'] == contract_sha and
                 record['checkpoint_sha256'] == ctx['old'].MODEL_PINS[model]['sha'],
                 'Unified regional identity/protocol differs')
            fields = validate_fields(record['regional'])
            need(record['status'] == fields['status'], 'Unified/regional status differs')
            origin = record['provenance']['regional']
            need(origin['kind'] in ('gpu_new', 'technical_derived'), 'Main regional cohort must use uniform CUDA outputs; CPU reuse is comparison only')
            need(origin['record'] is None, 'New regional provenance invents an old record')
            need((origin['kind'] == 'technical_derived') == (fields['status'] == 'unavailable_geometry'),
                 'Regional execution provenance/status mismatch')
            for key in ('mask_ref', 'mask_receipt'):
                c.bound(record[key])
            statuses[model][fields['status']] += 1
            provenance[model][origin['kind']] += 1
            manifest.append(dict(index=index, model=model, record_sha256=record['record_sha256'],
                                 provenance=record['provenance'], mask_ref=record['mask_ref'],
                                 mask_receipt=record['mask_receipt']))
            comparison = dict(status=fields['status'], reason=fields['reason'], roi_audit=fields['roi_audit'], regions={})
            if fields['status'] == 'complete':
                c.bound(fields['roi_audit']['landmarks'])
                if mi == 0:
                    landmarks[fields['roi_audit']['landmark_source']] += 1
            for ri, region in enumerate(REGIONS):
                item = fields['regions'][region] if fields['status'] == 'complete' else None
                reason = fields['reason'] if item is None else item['reason']
                reasons[(mi, ri, index)] = reason
                comparison['regions'][region] = None if item is None else {k: item[k] for k in ('truth', 'effective_area', 'reason')}
                if reason:
                    missing[(mi, ri)][reason] += 1
                else:
                    for si, signal in enumerate(SIGNALS):
                        data[mi, ri, si, index] = (item['truth'][signal], item['prediction'][signal])
            if same is None:
                same = comparison
            else:
                need(comparison == same, 'Anatomical ROI/truth/valid sample IDs differ across models')
    return data, missing, manifest, {m: dict(v) for m, v in statuses.items()}, dict(landmarks), provenance, reasons


def finalize(root, c, ctx, contract, read_unified, original_report, stats):
    """Recompute original 72 agreement cells and figures in the new GPU root only."""
    root = Path(root).resolve()
    folder = root / 'regional_report'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = original_report.descriptor(root / 'contract.json')
    need(c.digest(contract) == c.digest(original_report.read(root / 'contract.json')), 'Finalizer contract differs')
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = c.checked(original_report.read(receipt_path))
        need(receipt['status'] == 'verified' and receipt['contract'] == protocol, 'Existing regional report differs')
        for descriptor in receipt['outputs']:
            c.bound(descriptor)
        return original_report.read(folder / 'summary.json')
    data, missing, manifest, statuses, landmarks, provenance, reasons = collect(c, ctx, contract, read_unified, protocol['sha256'])
    rows, models = ctx['rows'], tuple(contract['models'])
    for entry in manifest:
        path = root / 'records' / entry['model'] / ('%06d.json' % entry['index'])
        need(c.checked(original_report.read(path))['record_sha256'] == entry['record_sha256'],
             'Unified regional file differs from validated record')
        entry['unified_record'] = original_report.descriptor(path)
    c.save(folder / 'record_manifest.json', c.seal(dict(schema='gpu_regional_record_manifest_v1', contract=protocol,
        original_regional_protocol=contract['original_regional_protocol'], original_combined_protocol=contract['original_combined_protocol'],
        catalog=contract['catalog'], records=manifest, record_count=len(manifest))))
    cells, checks = [], []
    for mi, model in enumerate(models):
        for ri, region in enumerate(REGIONS):
            for si, signal in enumerate(SIGNALS):
                pairs = data[mi, ri, si]
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
                result = stats.summarize_pairs(pairs[:, 0], pairs[:, 1], bootstrap_repeats=2000, seed=20261026)
                check = original_report.independent_point_check(pairs[:, 0], pairs[:, 1], result)
                cell = dict(model=model, region=region, signal=signal,
                    analysis_role='reference_domain' if region == 'F_reference' else 'anatomical_subroi',
                    signal_role='secondary_diagnostic' if signal == 'RGB_mean_secondary' else 'primary_channel',
                    n_total=len(rows), n_valid=len(pairs), n_undefined=len(rows)-len(pairs),
                    undefined_reasons=dict(sorted(missing[(mi, ri)].items())), statistics=result)
                need(sum(cell['undefined_reasons'].values()) == cell['n_undefined'], 'Undefined regional count differs')
                cells.append(cell)
                checks.append(dict(model=model, region=region, signal=signal, **check))
                print(json.dumps(dict(event='gpu_regional_summary_cell', model=model, region=region, signal=signal, n=len(pairs))), flush=True)
    need(len(cells) == 72, 'Regional final matrix must contain 72 cells')
    c.save(folder / 'independent_audit.json', c.seal(dict(status='passed', cell_count=72, checks=checks,
        same_valid_IDs_across_models_per_region=True, method='Original independent Python sums and full two-column ANOVA')))
    summary = dict(schema='gpu_regional_signal_agreement_report_v1', status='complete_three_models', contract=protocol,
        original_regional_protocol=contract['original_regional_protocol'], sample_count=len(rows),
        model_order=list(models), region_order=list(REGIONS), signal_order=list(SIGNALS), cell_count=72,
        status_counts=statuses, provenance_counts={k: dict(v) for k, v in provenance.items()},
        landmark_source_counts_valid_sample_groups=landmarks,
        record_manifest=original_report.descriptor(folder / 'record_manifest.json'),
        independent_audit=original_report.descriptor(folder / 'independent_audit.json'),
        interpretation='Descriptive regional encoded-UV agreement; not clinical interchangeability, fluorophore concentration, lesion localization or physical irrecoverability.',
        unit_note='Sample groups are not verified independent patients. Paired bootstrap CIs do not account for unknown repeated-patient dependence.',
        ci_note='Original paired percentile bootstrap: 2000 replicates, seed20261026. LoA = bias +/-1.96 difference sample SD(ddof1); no calibration fitting.',
        cells=cells)
    c.save(folder / 'summary.json', summary)
    output = io.StringIO(newline=''); writer = csv.writer(output)
    base_columns = ('model', 'region', 'signal', 'n_total', 'n_valid', 'n_undefined')
    stat_columns = tuple(cells[0]['statistics'])
    writer.writerow(base_columns + stat_columns + ('undefined_reasons',))
    for cell in cells:
        writer.writerow([cell[k] for k in base_columns] + [cell['statistics'][k] for k in stat_columns] + [json.dumps(cell['undefined_reasons'], sort_keys=True)])
    original_report.save_bytes(folder / 'summary.csv', output.getvalue().encode('utf8'))
    output = io.StringIO(newline=''); writer = csv.writer(output)
    writer.writerow(('sample_id', 'sample_index', 'model', 'region', 'signal', 'truth', 'prediction', 'prediction_minus_truth', 'valid_pair', 'undefined_reason'))
    for mi, model in enumerate(models):
        for index, row in enumerate(rows):
            for ri, region in enumerate(REGIONS):
                for si, signal in enumerate(SIGNALS):
                    x, y = data[mi, ri, si, index]
                    good = bool(np.isfinite([x, y]).all())
                    writer.writerow([row['sample_id'], index, model, region, signal, float(x) if good else '',
                                     float(y) if good else '', float(y-x) if good else '', int(good), reasons[(mi, ri, index)] or ''])
    original_report.save_bytes(folder / 'paired_values.csv', output.getvalue().encode('utf8'))
    plots = original_report.make_plots(folder, models, data, cells)
    headings = ('Model', 'Region', 'Signal', 'N', 'Bias (prediction - truth)', 'Difference SD', 'ICC(A,1) [95% CI]', '95% LoA')
    table = []
    fmt = original_report.fmt
    for cell in cells:
        s = cell['statistics']
        table.append([cell['model'], cell['region'], cell['signal'], str(cell['n_valid']), fmt(s['bias']),
            fmt(s['difference_sample_sd_ddof1']), fmt(s['icc_a1'])+' ['+fmt(s['icc_a1_ci95_low'])+', '+fmt(s['icc_a1_ci95_high'])+']',
            '['+fmt(s['loa_low'])+', '+fmt(s['loa_high'])+']'])
    intro = ['Regional UV Signal Intensity Agreement on Clean TEST',
        'All primary results use the same versioned CUDA FP32 execution contract. Earlier CPU results remain read-only comparison evidence and are not mixed into the primary cohort.',
        'Five predefined anatomical sub-ROIs; F is supplementary. R/G/B encoded channel means are primary; arithmetic RGB mean is secondary.',
        summary['unit_note'], summary['ci_note'], summary['interpretation'],
        'Archived landmark sources (valid sample groups): '+json.dumps(landmarks, sort_keys=True)+'. Some original geometry may use UV fallback.',
        'All 72 cells passed independent bias/SD/LoA/ICC recalculation. Undefined quantities retain explicit reasons.']
    original_report.save_bytes(folder / 'report.txt', ('\n\n'.join(intro)+'\n\n'+'\t'.join(headings)+'\n'+'\n'.join('\t'.join(v) for v in table)+'\n').encode('utf8'))
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Regional UV agreement</title><style>body{font:15px Arial;margin:28px}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #bbb}img{max-width:100%}</style>'
    body += '<h1>'+html.escape(intro[0])+'</h1>'+''.join('<p>'+html.escape(v)+'</p>' for v in intro[1:])
    body += '<p><a href="summary.csv">Summary CSV</a> | <a href="paired_values.csv">All paired values</a></p><table><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headings)+'</tr>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table)+'</table>'
    for path in plots:
        if path.suffix == '.png':
            relative = path.relative_to(folder).as_posix()
            body += '<h2>'+html.escape(path.stem)+'</h2><a href="'+html.escape(relative[:-4]+'.pdf')+'">PDF</a><img src="'+html.escape(relative)+'" alt="Regional agreement">'
    original_report.save_bytes(folder / 'report.html', (body+'</html>').encode('utf8'))
    outputs = [folder / p for p in ('record_manifest.json', 'independent_audit.json', 'summary.json', 'summary.csv', 'paired_values.csv', 'report.txt', 'report.html')] + plots
    c.save(receipt_path, c.seal(dict(schema='gpu_regional_agreement_receipt_v1', status='verified', contract=protocol,
        record_count=len(manifest), cell_count=72, outputs=[original_report.descriptor(p) for p in outputs])))
    return summary
