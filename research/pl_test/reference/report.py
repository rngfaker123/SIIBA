"""Verified regional agreement tables and plots from immutable paired records.

No model inference, mask generation, exclusions inferred from missing files,
calibration fit or outcome-based selection occurs here.
"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from agreement_stats import summarize_pairs

REGIONS = ('forehead', 'image_left_cheek', 'image_right_cheek', 'nose', 'chin', 'F_reference')
SIGNALS = ('R', 'G', 'B', 'RGB_mean_secondary')
POINT_METRICS = ('bias', 'difference_sample_sd_ddof1', 'loa_low', 'loa_high', 'icc_a1')
LABELS = {'forehead': 'Forehead', 'image_left_cheek': 'Image-left cheek',
          'image_right_cheek': 'Image-right cheek', 'nose': 'Nose', 'chin': 'Chin',
          'F_reference': 'Full F (reference, not a sub-ROI)'}
MODEL_LABELS = {'unet_single': 'Single-target U-Net', 'unet_multi': 'Multi-target U-Net',
                'unetplusplus': 'U-Net++'}
COLORS = ('#DA9700', '#248D4E', '#347BC2', '#BA4D9E', '#C75C36', '#555555')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked_file(desc):
    p = Path(desc['path'])
    require(p.is_absolute() and sha(p) == desc['sha256'], 'Bound file changed: ' + str(p))
    return p


def save_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Refuse report overwrite: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(str(temp), str(path))
    except FileExistsError:
        require(path.read_bytes() == data, 'Concurrent differing report')
    finally:
        temp.unlink()


def commit_stream(temp, path):
    if path.exists():
        require(sha(path) == sha(temp), 'Refuse differing CSV overwrite: ' + str(path))
        temp.unlink()
        return
    try:
        os.link(str(temp), str(path))
    except FileExistsError:
        require(sha(path) == sha(temp), 'Concurrent differing CSV')
    finally:
        temp.unlink()


def independent_point_check(x, y, reported):
    """Direct full two-column ANOVA, independently from optimized estimator."""
    n = len(x)
    expected = {k: None for k in POINT_METRICS}
    if n:
        d = y - x
        expected['bias'] = float(sum(map(float, d)) / n)
        if n >= 2:
            bias = expected['bias']
            sd = math.sqrt(sum((float(z) - bias) ** 2 for z in d) / (n - 1))
            expected.update(difference_sample_sd_ddof1=sd, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd)
            a = np.column_stack((x, y))
            rows, cols, grand = a.mean(axis=1), a.mean(axis=0), a.mean()
            msr = 2.0 * np.square(rows - grand).sum() / (n - 1)
            msc = n * np.square(cols - grand).sum()
            residual = a - rows[:, None] - cols[None, :] + grand
            mse = np.square(residual).sum() / (n - 1)
            denom = msr + mse + 2.0 * (msc - mse) / n
            constant = bool(np.ptp(a) == 0.0)
            if denom > 0.0 and not constant:
                expected['icc_a1'] = float((msr - mse) / denom)
    differences = []
    for name, value in expected.items():
        actual = reported[name]
        if value is None:
            require(actual is None, 'Independent undefined mismatch: ' + name)
        else:
            require(actual is not None and math.isclose(value, actual, rel_tol=1e-9, abs_tol=1e-11), 'Independent point mismatch: ' + name)
            differences.append(abs(actual - value))
    return {'status': 'passed', 'maximum_absolute_difference': max(differences or [0.0]),
            'rtol': 1e-9, 'atol': 1e-11, 'independent_full_anova': True}


def validate_and_collect(root, c, ctx, protocol):
    models = tuple(c.MODELS)
    rows = ctx['rows']
    n = len(rows)
    require(n == protocol['sample_count'] and protocol['models'] == list(models), 'Cohort/models differ')
    require(protocol['regions'] == list(REGIONS) and protocol['signals'] == list(SIGNALS), 'Region/signal definitions differ')
    ids = [row['sample_id'] for row in rows]
    require(len(set(ids)) == n, 'Duplicate sample groups')
    protocol_desc = descriptor(root / 'protocol.json')
    require(read(root / 'protocol.json') == protocol, 'Supplied protocol differs')
    checked_file(protocol['original_evaluation'])
    for desc in protocol['sources'].values():
        checked_file(desc)
    data = np.full((len(models), len(REGIONS), len(SIGNALS), n, 2), np.nan, dtype=np.float64)
    missing = {(mi, ri): Counter() for mi in range(len(models)) for ri in range(len(REGIONS))}
    manifest = []
    status_counts = {model: Counter() for model in models}
    landmark_sources = Counter()
    original_root = Path(protocol['original_evaluation']['path']).parent
    for i, row in enumerate(rows):
        group = []
        for mi, model in enumerate(models):
            path = root / 'records' / model / ('%06d.json' % i)
            require(path.exists(), 'Missing record is pending, not excluded: ' + str(path))
            v = c.checked(read(path))
            require(v['schema'] == 'regional_signal_agreement_sample_v1' and v['index'] == i and
                    v['sample_id'] == row['sample_id'] and v['subject_id'] == row['subject_id'] and
                    v['model_id'] == model and v['source_row_sha256'] == c.digest(row) and
                    v['protocol_sha256'] == protocol_desc['sha256'] and
                    v['checkpoint_sha256'] == ctx['old'].MODEL_PINS[model]['sha'], 'Sample identity/binding differs')
            prior_path = checked_file(v['original_record'])
            require(prior_path == original_root / 'records' / model / ('%06d.json' % i), 'Wrong original paired record')
            ref_path = checked_file(v['mask_ref'])
            receipt_path = checked_file(v['mask_receipt'])
            require(c.validate_ref(c.checked(read(ref_path)), row, i, ctx['protocol']) == receipt_path,
                    'Mask ref/receipt differs')
            prior = c.verify_record(read(prior_path), ctx, row, i, model, v['mask_receipt']['sha256'])
            require(v['status'] == prior['status'] and v['status'] in ('complete', 'unavailable_geometry'), 'Original status differs')
            status_counts[model][v['status']] += 1
            if v['status'] == 'unavailable_geometry':
                require(v['regions'] is None and v['roi_audit'] is None and v['parity'] is None and
                        isinstance(v['reason'], str) and bool(v['reason']) and
                        v['reason'] == prior['regions']['P']['l1_reason'], 'Technical reason/evidence differs')
                for ri in range(len(REGIONS)):
                    missing[(mi, ri)][v['reason']] += 1
            else:
                require(v['reason'] is None and set(v['regions']) == set(REGIONS) and
                        v['parity']['status'] == 'passed' and v['parity']['rtol'] == c.RTOL and
                        v['parity']['atol'] == c.ATOL, 'Valid/parity declaration differs')
                checked_file(v['roi_audit']['landmarks'])
                for ri, region in enumerate(REGIONS):
                    item = v['regions'][region]
                    require(set(item['truth']) == set(SIGNALS) and set(item['prediction']) == set(SIGNALS), 'Signal set differs')
                    area = item['effective_area']
                    require(isinstance(area, (int, float)) and math.isfinite(area) and area >= 0.0, 'Invalid ROI area')
                    if area == 0.0:
                        require(item['reason'] == 'empty_anatomical_roi' and
                                all(z is None for z in list(item['truth'].values()) + list(item['prediction'].values())), 'Empty ROI has fabricated means')
                        missing[(mi, ri)][item['reason']] += 1
                    else:
                        require(item['reason'] is None, 'Nonempty ROI has exclusion reason')
                        for si, signal in enumerate(SIGNALS):
                            pair = [item['truth'][signal], item['prediction'][signal]]
                            require(all(isinstance(z, (int, float)) and not isinstance(z, bool) and math.isfinite(z) for z in pair), 'Nonfinite paired mean')
                            data[mi, ri, si, i] = pair
            group.append(v)
            manifest.append({'index': i, 'sample_id': row['sample_id'], 'model_id': model,
                             'record': descriptor(path), 'record_seal': v['record_sha256']})
        require(len({v['status'] for v in group}) == 1, 'Models use different technical sample sets')
        if group[0]['status'] == 'complete':
            source = group[0]['roi_audit']['landmark_source']
            require(isinstance(source, str) and bool(source), 'Landmark source must be declared')
            landmark_sources[source] += 1
            require(all(v['roi_audit'] == group[0]['roi_audit'] for v in group[1:]), 'Anatomical audit differs across models')
            for region in REGIONS:
                first = group[0]['regions'][region]
                for v in group[1:]:
                    item = v['regions'][region]
                    require(item['truth'] == first['truth'] and item['effective_area'] == first['effective_area'] and
                            item['reason'] == first['reason'], 'Models use different ROI/truth/valid IDs')
    for ri in range(len(REGIONS)):
        for mi in range(1, len(models)):
            require(np.array_equal(np.isfinite(data[0, ri]), np.isfinite(data[mi, ri])), 'Regional paired model cohort differs')
    return data, missing, manifest, {m: dict(v) for m, v in status_counts.items()}, dict(landmark_sources)


def write_paired_csv(folder, rows, models, data, missing):
    path = folder / 'paired_values.csv'
    temp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temp.open('x', encoding='utf8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'sample_index', 'model', 'region', 'signal', 'truth', 'prediction', 'prediction_minus_truth', 'valid_pair', 'undefined_reason'])
        for mi, model in enumerate(models):
            for i, row in enumerate(rows):
                record = read(folder.parent / 'records' / model / ('%06d.json' % i))
                for ri, region in enumerate(REGIONS):
                    for si, signal in enumerate(SIGNALS):
                        x, y = data[mi, ri, si, i]
                        good = math.isfinite(x) and math.isfinite(y)
                        reason = '' if good else record['reason'] if record['status'] == 'unavailable_geometry' else record['regions'][region]['reason']
                        writer.writerow([row['sample_id'], i, model, region, signal, float(x) if good else '',
                                         float(y) if good else '', float(y-x) if good else '', int(good), reason])
        stream.flush()
        os.fsync(stream.fileno())
    commit_stream(temp, path)


def _figure_bytes(figure, format):
    stream = io.BytesIO()
    metadata = {'Creator': 'SIIBA regional agreement', 'CreationDate': None, 'ModDate': None} if format == 'pdf' else {'Software': 'SIIBA regional agreement'}
    figure.savefig(stream, format=format, dpi=180, bbox_inches='tight', metadata=metadata)
    return stream.getvalue()


def make_plots(folder, models, data, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    bykey = {(v['model'], v['region'], v['signal']): v for v in cells}
    plot_paths = []
    for si, signal in enumerate(SIGNALS):
        finite_data = data[:, :, si]
        finite_data = finite_data[np.isfinite(finite_data).all(axis=-1)]
        limit = max(0.01, float(np.max(np.abs(finite_data[:, 1] - finite_data[:, 0]))) if finite_data.size else .01)
        for cell in cells:
            if cell['signal'] == signal:
                for key in ('loa_low', 'loa_high'):
                    if cell['statistics'][key] is not None:
                        limit = max(limit, abs(cell['statistics'][key]))
        limit *= 1.05
        for mi, model in enumerate(models):
            for mode in ('bland_altman', 'identity_scatter'):
                fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
                for ri, (region, ax) in enumerate(zip(REGIONS, axes.flat)):
                    pairs = data[mi, ri, si]
                    pairs = pairs[np.isfinite(pairs).all(axis=1)]
                    stat = bykey[(model, region, signal)]['statistics']
                    ax.set_title(LABELS[region] + ' | N=' + str(len(pairs)))
                    ax.set_xlim(0, 1)
                    ax.grid(alpha=.15)
                    if mode == 'bland_altman':
                        ax.set_ylim(-limit, limit)
                        if len(pairs):
                            d = pairs[:, 1] - pairs[:, 0]
                            ax.scatter(pairs.mean(axis=1), d, s=3, alpha=.16, c=COLORS[ri], edgecolors='none', rasterized=True)
                            ax.axhline(stat['bias'], color='black', linewidth=1.0, label='Mean bias')
                            if stat['loa_low'] is not None:
                                ax.axhline(stat['loa_low'], color='#AA3333', linestyle='--', linewidth=.8)
                                ax.axhline(stat['loa_high'], color='#AA3333', linestyle='--', linewidth=.8)
                            ax.text(.02, .97, 'Bias %.4f\nSD %s' % (stat['bias'], fmt(stat['difference_sample_sd_ddof1'])),
                                    transform=ax.transAxes, va='top', fontsize=8,
                                    bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none'})
                            # Distribution diagnostic; all differences, no subsampling.
                            inset = ax.inset_axes([.76, .76, .22, .20])
                            inset.hist(d, bins=30, color=COLORS[ri], alpha=.75)
                            inset.set_xticks([]); inset.set_yticks([])
                        ax.set_xlabel('(Prediction + observed UV) / 2')
                        ax.set_ylabel('Prediction - observed UV')
                    else:
                        ax.set_ylim(0, 1)
                        ax.plot([0, 1], [0, 1], '--', color='#555555', linewidth=.8)
                        if len(pairs):
                            ax.scatter(pairs[:, 0], pairs[:, 1], s=3, alpha=.16, c=COLORS[ri], edgecolors='none', rasterized=True)
                        ax.set_xlabel('Observed UV regional mean')
                        ax.set_ylabel('Predicted UV regional mean')
                        ax.text(.02, .97, 'ICC(A,1) ' + fmt(stat['icc_a1']), transform=ax.transAxes, va='top')
                label = 'RGB mean (secondary diagnostic)' if signal == 'RGB_mean_secondary' else signal + ' channel'
                fig.suptitle(MODEL_LABELS.get(model, model) + ' | ' + label + ' | ' + mode.replace('_', ' '), fontsize=15)
                fig.subplots_adjust(top=.90, bottom=.09, hspace=.30, wspace=.22)
                fig.text(.5, .015, 'All valid TEST sample groups; no pixel pooling. Image-left/right refer to display sides. F is a reference domain.\nEncoded UV intensity on [0,1]; agreement limits are descriptive, not clinical acceptance limits.', ha='center', fontsize=9)
                for ext in ('png', 'pdf'):
                    path = folder / 'figures' / (model + '__' + signal + '__' + mode + '.' + ext)
                    save_bytes(path, _figure_bytes(fig, ext))
                    plot_paths.append(path)
                plt.close(fig)
    return plot_paths


def fmt(value):
    return 'NA' if value is None else '%.6f' % value


def finalize(root, c, ctx, protocol):
    root = Path(root).resolve()
    folder = root / 'report'
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        prior = c.checked(read(receipt_path))
        require(prior['status'] == 'verified' and prior['protocol'] == descriptor(root / 'protocol.json'), 'Existing report binding differs')
        for item in prior['outputs']:
            checked_file(item)
        return read(folder / 'summary.json')
    data, missing, manifest, status_counts, landmark_sources = validate_and_collect(root, c, ctx, protocol)
    models, rows = tuple(c.MODELS), ctx['rows']
    record_manifest = c.seal({'schema': 'regional_agreement_record_manifest_v1',
        'protocol': descriptor(root / 'protocol.json'), 'sources': protocol['sources'],
        'sample_count': len(rows), 'record_count': len(manifest), 'records': manifest})
    c.save(folder / 'record_manifest.json', record_manifest)
    cells, checks = [], []
    for mi, model in enumerate(models):
        for ri, region in enumerate(REGIONS):
            for si, signal in enumerate(SIGNALS):
                pairs = data[mi, ri, si]
                pairs = pairs[np.isfinite(pairs).all(axis=1)]
                stat = summarize_pairs(pairs[:, 0], pairs[:, 1], bootstrap_repeats=2000, seed=20261026)
                audit = independent_point_check(pairs[:, 0], pairs[:, 1], stat)
                cell = {'model': model, 'region': region, 'signal': signal,
                        'analysis_role': 'reference_domain' if region == 'F_reference' else 'anatomical_subroi',
                        'signal_role': 'secondary_diagnostic' if signal == 'RGB_mean_secondary' else 'primary_channel',
                        'n_total': len(rows), 'n_valid': len(pairs), 'n_undefined': len(rows)-len(pairs),
                        'undefined_reasons': dict(sorted(missing[(mi, ri)].items())), 'statistics': stat}
                require(sum(cell['undefined_reasons'].values()) == cell['n_undefined'], 'Missing accounting mismatch')
                cells.append(cell)
                checks.append(dict(model=model, region=region, signal=signal, **audit))
                print(json.dumps({'event': 'agreement_summary_cell', 'model': model, 'region': region, 'signal': signal, 'n': len(pairs)}), flush=True)
    require(len(cells) == len(models)*len(REGIONS)*len(SIGNALS), 'Incomplete cell matrix')
    audit = c.seal({'status': 'passed', 'cell_count': len(cells), 'checks': checks,
                   'same_valid_IDs_across_models_per_region': True,
                   'method': 'independent Python sums for bias/SD/LoA; full n-by-two ANOVA for ICC'})
    c.save(folder / 'independent_audit.json', audit)
    summary = {'schema': 'regional_signal_agreement_report_v1', 'status': 'complete_three_models',
        'protocol': descriptor(root / 'protocol.json'), 'sample_count': len(rows),
        'model_order': list(models), 'region_order': list(REGIONS), 'signal_order': list(SIGNALS),
        'cell_count': len(cells), 'status_counts': status_counts,
        'landmark_source_counts_valid_sample_groups': landmark_sources,
        'record_manifest': descriptor(folder / 'record_manifest.json'),
        'independent_audit': descriptor(folder / 'independent_audit.json'),
        'interpretation': 'Descriptive regional encoded-UV intensity agreement only. These results do not establish lesion localization, clinical interchangeability, calibrated fluorophore concentration, or physical irrecoverability.',
        'unit_note': 'Rows are sample groups, not independently verified patients. Paired sample-group bootstrap intervals do not account for unknown within-patient dependence. No training-seed variation is measured.',
        'ci_note': '95% paired percentile bootstrap CIs (2000 replicates, seed20261026). LoA are bias +/-1.96 sample SD; empirical difference percentiles are supplemental distribution checks. Undefined estimates and bootstrap replicates are explicit.',
        'cells': cells}
    c.save(folder / 'summary.json', summary)
    columns = ['model', 'region', 'signal', 'n_total', 'n_valid', 'n_undefined']
    stat_columns = list(cells[0]['statistics'])
    out = io.StringIO(newline='');writer = csv.writer(out)
    writer.writerow(columns + stat_columns + ['undefined_reasons'])
    for cell in cells:
        writer.writerow([cell[k] for k in columns] + [cell['statistics'][k] for k in stat_columns] + [json.dumps(cell['undefined_reasons'], sort_keys=True)])
    save_bytes(folder / 'summary.csv', out.getvalue().encode('utf8'))
    write_paired_csv(folder, rows, models, data, missing)
    plots = make_plots(folder, models, data, cells)
    headings = ['Model', 'Region', 'Signal', 'N', 'Mean bias', 'Difference SD', 'ICC(A,1) [95% CI]', '95% LoA [low, high]']
    table = []
    for cell in cells:
        s = cell['statistics']
        table.append([MODEL_LABELS.get(cell['model'], cell['model']), LABELS[cell['region']], cell['signal'], str(cell['n_valid']),
                      fmt(s['bias']), fmt(s['difference_sample_sd_ddof1']),
                      fmt(s['icc_a1'])+' ['+fmt(s['icc_a1_ci95_low'])+', '+fmt(s['icc_a1_ci95_high'])+']',
                      '['+fmt(s['loa_low'])+', '+fmt(s['loa_high'])+']'])
    intro = ['Regional UV Signal Intensity Agreement on Clean TEST',
             'Models: '+', '.join(MODEL_LABELS.get(m, m) for m in models),
             'Cohort: %d sample groups. Five predefined anatomical sub-ROIs; full F shown separately as reference.' % len(rows),
             'R, G and B are primary encoded channel means; RGB mean is a secondary diagnostic. No physical intensity calibration is inferred.',
             summary['unit_note'], summary['ci_note'], summary['interpretation'],
             'True and predicted images use identical masks and preprocessing; original company module, masks, weights and training are unchanged.',
             'Archived landmark sources (valid sample groups): '+json.dumps(landmark_sources, sort_keys=True)+'. The original geometry may include UV fallback; no claim is made that every ROI was localized from RGB.',
             'Independent point-statistic audit passed for all %d cells. Missing values and reasons are retained in summary.json/csv.' % len(cells)]
    text = '\n\n'.join(intro)+'\n\n'+'\t'.join(headings)+'\n'+'\n'.join('\t'.join(r) for r in table)+'\n'
    save_bytes(folder / 'report.txt', text.encode('utf8'))
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Regional UV Signal Intensity Agreement</title><style>body{font:15px Arial;margin:32px;max-width:1500px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border:1px solid #bbb;text-align:left}th{background:#eaf0f5}img{max-width:100%}p{line-height:1.5}</style><h1>'+html.escape(intro[0])+'</h1>'
    body += ''.join('<p>'+html.escape(x)+'</p>' for x in intro[1:])
    body += '<p><a href="summary.csv">Full summary CSV with all CIs</a> | <a href="paired_values.csv">All paired sample values</a> | <a href="summary.json">Summary JSON</a></p>'
    body += '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headings)+'</tr></thead><tbody>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in table)+'</tbody></table>'
    for path in plots:
        if path.suffix == '.png':
            rel = path.relative_to(folder).as_posix()
            body += '<h2>'+html.escape(path.stem.replace('__', ' | '))+'</h2><a href="'+html.escape(rel[:-4]+'.pdf')+'">PDF</a><img src="'+html.escape(rel)+'" alt="Regional agreement plot">'
    body += '<p>References: <a href="https://doi.org/10.1037/1082-989X.1.1.30">McGraw &amp; Wong (1996)</a>; <a href="https://www-users.york.ac.uk/~mb55/meas/ba.htm">Bland &amp; Altman (1986)</a>.</p></html>'
    save_bytes(folder / 'report.html', body.encode('utf8'))
    paths = [folder / x for x in ('record_manifest.json', 'independent_audit.json', 'summary.json', 'summary.csv', 'paired_values.csv', 'report.txt', 'report.html')] + plots
    receipt = c.seal({'schema': 'regional_signal_agreement_report_receipt_v1', 'status': 'verified',
                     'protocol': descriptor(root / 'protocol.json'), 'cell_count': len(cells),
                     'record_count': len(manifest), 'outputs': [descriptor(p) for p in paths]})
    c.save(receipt_path, receipt)
    return summary
