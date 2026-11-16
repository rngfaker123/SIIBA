"""Finalize CUDA TEST records with explicit program-consistency diagnostics.

Only the versioned GPU task directory is written. Earlier CPU outputs remain
read-only evidence; the original 78-test family and regional estimators survive.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

import regional_adapter
import backend_compare


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_summary(values):
    """Direct sums, independent from combined_eval.statistics.mean/stdev."""
    n = len(values)
    mean = math.fsum(values)/n if n else None
    sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/(n-1)) if n >= 2 else None
    return mean, sd


def check_pair(reference, observed, label):
    if reference is None:
        need(observed is None, 'Independent undefined mismatch: '+label)
        return 0.0
    need(observed is not None and math.isclose(reference, observed, rel_tol=1e-9, abs_tol=1e-11),
         'Independent statistic mismatch: '+label)
    return abs(reference-observed)


def check_primary_record(record):
    need(record['status'] in ('complete', 'unavailable_geometry'), 'Unexpected primary record status')
    expected = 'gpu_new' if record['status'] == 'complete' else 'technical_derived'
    for domain in ('combined', 'regional'):
        source = record['provenance'][domain]
        need(source['kind'] == expected and source['record'] is None,
             'Primary statistics must not mix CPU and CUDA outputs')
    need(record['execution']['forward_count'] == int(record['status'] == 'complete'),
         'Primary record forward count differs')
    if record['status'] == 'complete':
        need(record['execution']['device'] == 'cuda:0', 'Primary finite record is not CUDA')
    continuation = record['continuation']
    need(continuation['status'] in ('computed_program_v1', 'reused_accepted_g2'), 'Unknown CUDA continuation origin')
    reused = continuation['status'] == 'reused_accepted_g2'
    need((continuation['parent_record'] is not None) == reused and
         continuation['new_forward_count'] == (0 if reused else int(record['status'] == 'complete')),
         'CUDA reattestation/forward accounting differs')
    need(continuation['cpu_numeric_gate'] == 'diagnostic_only' and
         continuation['original_cpu_comparison_status'] in ('passed', 'failed', 'unavailable') and
         (continuation['historical_parent_continuation'] is not None) == reused,
         'Numeric diagnostics must not be relabelled scientific acceptance or parity')


def numerical_leaves(reference, candidate, kind):
    """Keep every numeric leaf; only structural/semantic differences are fatal."""
    rows = []
    limits = backend_compare.THRESHOLDS
    def visit(a, b, path):
        if isinstance(a, dict):
            need(isinstance(b, dict) and set(a) == set(b), 'Diagnostic mapping differs: '+path)
            for key in sorted(a): visit(a[key], b[key], path+'/'+key)
        elif isinstance(a, (list, tuple)):
            need(type(a) is type(b) and len(a) == len(b), 'Diagnostic sequence differs: '+path)
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, path+'/'+str(i))
        elif a is None or isinstance(a, (str, bool)):
            need(type(a) is type(b) and a == b, 'Program null/status/identity differs: '+path)
        elif isinstance(a, (int, float)):
            need(isinstance(b, (int, float)) and not isinstance(b, bool) and
                 math.isfinite(a) and math.isfinite(b), 'Nonfinite diagnostic value: '+path)
            key = path.rsplit('/', 1)[-1]
            allowed = limits[kind].get(key) if kind in ('table3', 'table4') else (
                limits['regional_prediction'] if '/prediction/' in path else None)
            scientific_prediction_metric = allowed is not None
            if allowed is None:
                need(type(a) is type(b) and a == b, 'Program area/count/truth/geometry differs: '+path)
                allowed = dict(atol=0.0, rtol=0.0)
            delta = b-a
            historical_limit = allowed['atol'] + allowed['rtol']*abs(a)
            strict = limits['original_cpu_strict']
            rows.append(dict(path=path, cpu=a, gpu=b, signed_delta=delta, absolute_delta=abs(delta),
                prediction_metric=scientific_prediction_metric,
                historical_backend_atol=allowed['atol'], historical_backend_rtol=allowed['rtol'],
                historical_backend_limit=historical_limit, historical_backend_pass=abs(delta)<=historical_limit,
                historical_strict_atol=strict['atol'], historical_strict_rtol=strict['rtol'],
                historical_strict_pass=math.isclose(a,b,rel_tol=strict['rtol'],abs_tol=strict['atol'])))
        else:
            raise ValueError('Unsupported diagnostic field: '+path)
    visit(reference, candidate, kind)
    return rows


def program_case(c, record, source):
    """Independently expose all available CPU/GPU fields, including nonviolations."""
    comparison = record['execution']['cpu_comparison']
    reference = comparison.get('combined_record')
    reference_regional = comparison.get('regional_record')
    pairs = {}
    if reference is not None:
        old = c.checked(c.read(c.bound(reference)))
        need(old['index']==record['index'] and old['model_id']==record['model_id'] and
             old['sample_id']==record['sample_id'] and old['checkpoint_sha256']==record['checkpoint_sha256'],
             'CPU diagnostic record identity differs')
        pairs.update(table3=(old['regions'],record['combined']['regions']),
                     table4=(old['table4'],record['combined']['table4']))
    if reference_regional is not None:
        old = c.checked(c.read(c.bound(reference_regional)))
        need(reference is not None and old['index']==record['index'] and old['model_id']==record['model_id'] and
             old['sample_id']==record['sample_id'] and old['checkpoint_sha256']==record['checkpoint_sha256'],
             'Regional CPU diagnostic record identity differs')
        keys=('status','reason','regions','roi_audit')
        pairs['regional']=({k:old[k] for k in keys},{k:record['regional'][k] for k in keys})
    leaves=[]; statuses={}
    for kind in ('table3','table4','regional'):
        original = comparison.get(kind)
        if kind not in pairs:
            need(original is None, 'Diagnostic comparison invents an absent CPU reference')
            statuses[kind]='unavailable'
            continue
        a,b=pairs[kind]
        leaves.extend(numerical_leaves(a,b,kind))
        observed=backend_compare.compare_payload(a,b,kind)
        need(observed==original, 'Stored historical numeric diagnostic differs from raw CPU/GPU values')
        statuses[kind]=observed['status']
    status='failed' if 'failed' in statuses.values() else ('passed' if pairs else 'unavailable')
    need(record['continuation']['original_cpu_comparison_status']==status,
         'Original diagnostic status must be preserved, including unavailable')
    parent=record['continuation']['parent_record']
    if parent is not None: c.bound(parent)
    failed=record.get('original_failed_record',record['continuation'].get('original_failed_record'))
    if failed is not None: c.bound(failed)
    return dict(index=record['index'],sample_id=record['sample_id'],model_id=record['model_id'],
        status=record['status'],source=source,execution_runtime=record['execution'].get('runtime'),
        continuation=record['continuation'],original_failed_record=failed,
        original_cpu_comparison=comparison,historical_numeric_status=status,
        available_domains=list(pairs),domain_status=statuses,numerical_fields=leaves,
        numerical_differences_are_diagnostic_only=True,parity_claimed=False)


def export_records(root, c, ctx, contract, reader):
    """Revalidate all IDs, export exact metric values; never infer missing rows."""
    rows, models = ctx['rows'], tuple(contract['models'])
    source_manifest, exports, cells, audits, diagnostics = [], [], [], [], []
    status_counts, provenance_counts = {}, {}
    for model in models:
        table3, table4 = [], []
        status_counts[model], provenance_counts[model] = Counter(), Counter()
        for index, row in enumerate(rows):
            record = reader(model,index)
            check_primary_record(record)
            source = c.desc(root/'records'/model/('%06d.json'%index))
            diagnostics.append(program_case(c,record,source))
            source_manifest.append(dict(model=model,index=index,source=source,record_sha256=record['record_sha256']))
            status_counts[model][record['status']] += 1
            provenance_counts[model][record['continuation']['status']] += 1
            table3.append(c.seal(dict(schema='gpu_original_metric_projection_v1',
                **{k:record[k] for k in ('status','index','sample_id','subject_id','model_id','checkpoint_sha256',
                    'contract_sha256','source_row_sha256')},
                regions=record['combined']['regions'], unified_source=source,
                execution_note='Uniform CUDA FP32 cohort; original scientific metric fields only, not a CPU record')))
            c.validate_table4(record['combined']['table4'])
            table4.append(dict(table4=record['combined']['table4']))
        descriptor = ctx['old'].export_jsonl(root/'exports'/(model+'_table3.jsonl'),table3)
        exports.append(dict(model_id=model,checkpoint_sha256=ctx['old'].MODEL_PINS[model]['sha'],records=descriptor))
        for region in c.REGIONS:
            for metric in c.TABLE4_METRICS:
                result = c.summary_cell(table4,region,metric)
                values, missing = [], Counter()
                for record in table4:
                    item=record['table4'][region]
                    if item[metric] is None:
                        missing[item[metric+'_reason']] += 1
                    else:
                        values.append(item[metric])
                mean, sd = independent_summary(values)
                differences=[check_pair(mean,result['mean'],metric+' mean'),check_pair(sd,result['sample_sd'],metric+' SD')]
                need(result['valid_n']==len(values) and result['missing_n']==len(rows)-len(values) and
                     result['missing_reasons']==dict(missing) and result['ddof']==1 and
                     result['sample_sd_reason']==(None if len(values)>=2 else 'fewer_than_two_valid_samples'),
                     'Table4 sample/undefined accounting differs')
                cells.append(dict(model_id=model,region=region,metric=metric,**result))
                audits.append(dict(model_id=model,region=region,metric=metric,status='passed',
                                   maximum_absolute_difference=max(differences)))
    need(len(cells)==36 and len(source_manifest)==len(rows)*3,'Incomplete CUDA statistics inventory')
    unit=dict(name='sample_group',independent_subject_identity_verified=False,
              note='Folder-derived groups are not independently verified biological patients')
    ids=[r['sample_id'] for r in rows]
    order=c.compact_order(ids)
    need(order==contract['sample_order_sha256'],'Frozen sample order differs')
    manifest=dict(schema='uv_three_region_statistics_input_v1',models=exports,model_order=list(models),
        expected_sample_count=len(rows),expected_sample_order_sha256=order,statistical_unit=unit,
        evaluation_protocol=c.desc(root/'contract.json'),execution_note='All-CUDA FP32; earlier CPU results remain historical comparison evidence')
    c.save(root/'table3_stats_manifest.json',manifest)
    c.save(root/'record_manifest.json',c.seal(dict(schema='test_gpu_final_record_manifest_v1',
        contract=c.desc(root/'contract.json'),record_count=len(source_manifest),records=source_manifest)))
    return cells,audits,{k:dict(v) for k,v in status_counts.items()},{k:dict(v) for k,v in provenance_counts.items()},diagnostics


def program_report(root,c,contract,records,report_module):
    folder=root/'program_consistency_report'
    note=('CPU and CUDA use the same frozen model, preprocessing, masks and scientific metric programs. '
          'Their finite numerical outputs may differ. Earlier numeric thresholds are retained as diagnostics only; '
          'no numerical-parity or passed-old-gate claim is made. Primary measurements remain actual CUDA values, '
          'including unchanged reattested G1/G2 outputs. Historical boundary-review flags describe earlier runs '
          'and are not acceptance categories in this program-consistency run. No rounding, CPU substitution, '
          'sample deletion or alteration of scientific definitions is performed.')
    source_counts=Counter(); status_counts=Counter(); domain_counts={k:Counter() for k in ('table3','table4','regional')}
    affected_models=Counter(); history=Counter(); grouped={}; compared_cases=0; failed_cases=0
    for row in records:
        source_counts[row['continuation']['status']]+=1
        status_counts[row['historical_numeric_status']]+=1
        compared_cases+=bool(row['available_domains'])
        failed_cases+=row['historical_numeric_status']=='failed'
        for domain,status in row['domain_status'].items(): domain_counts[domain][status]+=1
        old=row['continuation']['historical_parent_continuation']
        if old is not None:
            history[old.get('status','unspecified')]+=1
            if old.get('review_status')=='accepted_boundary_exception':
                history['historical_boundary_reviewed_case_models']+=1
        if row['original_failed_record'] is not None: history['historical_failed_record_bound']+=1
        if row['historical_numeric_status']=='failed': affected_models[row['model_id']]+=1
        for leaf in row['numerical_fields']:
            key=(row['model_id'],leaf['path']);group=grouped.setdefault(key,dict(n=0,signed=[],absolute=[],backend_failed=0,strict_failed=0))
            group['n']+=1;group['signed'].append(leaf['signed_delta']);group['absolute'].append(leaf['absolute_delta'])
            group['backend_failed']+=not leaf['historical_backend_pass'];group['strict_failed']+=not leaf['historical_strict_pass']
    fields=[]
    for (model,path),v in sorted(grouped.items()):
        mean,sd=independent_summary(v['signed'])
        fields.append(dict(model_id=model,path=path,paired_valid_n=v['n'],mean_signed_delta=mean,
            sample_sd_signed_delta=sd,mean_absolute_delta=math.fsum(v['absolute'])/v['n'],
            maximum_absolute_delta=max(v['absolute']),historical_backend_failure_n=v['backend_failed'],
            historical_strict_failure_n=v['strict_failed'],ddof=1))
    # 保留逐例原始对照

    report_module.save_bytes(folder/'case_diagnostics.jsonl',
        ('\n'.join(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False) for row in records)+'\n').encode('utf8'))
    output=io.StringIO(newline='');writer=csv.writer(output)
    columns=('model_id','path','paired_valid_n','mean_signed_delta','sample_sd_signed_delta','mean_absolute_delta',
             'maximum_absolute_delta','historical_backend_failure_n','historical_strict_failure_n')
    writer.writerow(columns)
    for field in fields:writer.writerow([field[k] for k in columns])
    report_module.save_bytes(folder/'field_summary.csv',output.getvalue().encode('utf8'))
    result=dict(schema='gpu_program_consistency_report_v1',status='verified',contract=c.desc(root/'contract.json'),
        cpu_gpu_numerical_differences_diagnostic_only=True,parity_claimed=False,
        original_numeric_thresholds_changed=False,thresholds_are_run_acceptance_criteria=False,
        scientific_metrics_unchanged=True,numerical_values_replaced=False,
        total_case_models=len(records),diagnostic_case_count=compared_cases,unavailable_cpu_case_count=len(records)-compared_cases,
        historical_backend_failed_case_count=failed_cases,historical_numeric_status_counts=dict(status_counts),
        domain_status_counts={k:dict(v) for k,v in domain_counts.items()},source_counts=dict(source_counts),
        historical_provenance_counts=dict(history),historical_failed_case_count_by_model=dict(affected_models),
        numeric_field_summary=fields,note=note,case_diagnostics=c.desc(folder/'case_diagnostics.jsonl'),
        field_summary=c.desc(folder/'field_summary.csv'))
    c.save(folder/'summary.json',c.seal(result))
    lines=['CPU/CUDA Program Consistency and Numerical Diagnostics',note,
        'Case-models: %d; with archived CPU comparisons: %d; without CPU comparisons: %d; earlier numeric thresholds failed: %d.'%
        (len(records),compared_cases,len(records)-compared_cases,failed_cases),
        'Unavailable CPU comparisons are not counted as passing. Historical gate failures are not relabelled boundary exceptions.',
        json.dumps(result['source_counts'],sort_keys=True),json.dumps(result['historical_provenance_counts'],sort_keys=True),
        '',output.getvalue()]
    report_module.save_bytes(folder/'report.txt',('\n'.join(lines)+'\n').encode('utf8'))
    page='<!doctype html><meta charset="utf-8"><title>CPU/CUDA Program Consistency</title><h1>Program Consistency and Numerical Diagnostics</h1>'
    page+='<p>'+html.escape(note)+'</p><p>'+html.escape(lines[2])+'</p><p><a href="case_diagnostics.jsonl">All case diagnostics and scalar pairs</a> | <a href="field_summary.csv">Field summary CSV</a></p><pre>'+html.escape(output.getvalue())+'</pre>'
    report_module.save_bytes(folder/'report.html',page.encode('utf8'))
    c.save(folder/'receipt.json',c.seal(dict(schema='gpu_program_consistency_receipt_v1',status='verified',
        contract=c.desc(root/'contract.json'),diagnostic_case_count=compared_cases,total_case_models=len(records),
        outputs=[c.desc(folder/name) for name in ('summary.json','case_diagnostics.jsonl','field_summary.csv','report.txt','report.html')])))
    return result


def table3_report(root,c,contract,report_module,program=None):
    source=c.bound(contract['original_stats'])
    stats=load(source,'_frozen_gpu_original_table3_statistics')
    need(stats.FAMILY_SIZE==78 and tuple(stats.MODEL_ORDER)==tuple(contract['models']),'Original Table3 statistical family differs')
    records,unit,provenance=stats.load_manifest(root/'table3_stats_manifest.json')
    summary=stats.summarize_models(records,statistical_unit=unit)
    summary['provenance']=provenance
    summary['execution_contract']=c.desc(root/'contract.json')
    summary['execution_note']='All primary metrics use CUDA FP32; no CPU records are mixed into this cohort.'
    if program is not None:
        summary['program_consistency_report']=c.desc(root/'program_consistency_report/summary.json')
        summary['execution_note']+=' '+program['note']
    count=sum(len(metrics) for regions in summary['models'].values() for metrics in regions.values())
    need(count==42 and len(summary['tests'])==78 and summary['multiplicity']['family_size']==78 and
         summary['sample_count']==contract['sample_count'],'Original Table3 summary/family incomplete')
    folder=root/'table3_report'
    c.save(folder/'summary.json',summary)
    text=stats.report_text(summary)
    page=stats.report_html(summary)
    if program is not None:
        disclosure='\n\nNumerical backend disclosure: '+program['note']+' Case-models with CPU diagnostics: '+str(program['diagnostic_case_count'])+'. See ../program_consistency_report/report.html.\n'
        text+=disclosure
        snippet='<p>'+html.escape(disclosure)+'</p>'
        page=page.replace('</body>',snippet+'</body>') if '</body>' in page else page+snippet
    report_module.save_bytes(folder/'report.txt',text.encode('utf8'))
    report_module.save_bytes(folder/'report.html',page.encode('utf8'))
    c.save(folder/'receipt.json',c.seal(dict(schema='gpu_table3_report_receipt_v1',status='verified',
        contract=c.desc(root/'contract.json'),original_statistics_source=contract['original_stats'],
        descriptive_cells=42,paired_tests=78,sample_count=contract['sample_count'],
        outputs=[c.desc(folder/x) for x in ('summary.json','report.txt','report.html')])))
    return summary


def table4_report(root,c,contract,cells,audits,status_counts,provenance_counts,report_module,program=None):
    folder=root/'table4_report'
    note='Image-equal means and across-sample SD (ddof=1), own valid N/null reasons; not training seeds or patient-population confidence intervals. No new t-test family.'
    result=dict(schema='gpu_table4_report_v1',status='complete_three_models',contract=c.desc(root/'contract.json'),
        original_combined_protocol=contract['original_combined_protocol'],sample_count=contract['sample_count'],
        expected_cells=36,cells=cells,statistical_note=note,model_order=contract['models'],status_counts=status_counts,
        provenance_counts=provenance_counts,independent_recalculation=dict(passed=True,checks=audits,
            maximum_absolute_difference=max(x['maximum_absolute_difference'] for x in audits)),
        metric_notes=dict(gaussian_nll='Original Gaussian constant omitted; no new floor or fitted calibration.',
            rank_corr='Within-image area-weighted Pearson correlation of ordinary ranks of sigma and channel-mean absolute error.',
            mean_z2='Area-weighted standardized squared residual; ideal target1.',
            coverage95='Original channel-pixel marginal abs(z)<=1.959963984540054; target0.95. Coverage68 not calculated.'))
    if program is not None:
        result['program_consistency_report']=c.desc(root/'program_consistency_report/summary.json')
        result['diagnostic_case_count']=program['diagnostic_case_count']
        result['program_consistency_note']=program['note']
    c.save(folder/'report.json',result)
    output=io.StringIO(newline='');writer=csv.writer(output)
    columns=('model_id','region','metric','mean','sample_sd','valid_n','missing_n','sample_sd_reason')
    writer.writerow(columns+('missing_reasons',))
    for cell in cells:
        writer.writerow([cell[k] for k in columns]+[json.dumps(cell['missing_reasons'],sort_keys=True)])
    report_module.save_bytes(folder/'summary.csv',output.getvalue().encode('utf8'))
    lines=['Uncertainty and Calibration on Clean TEST (uniform CUDA FP32)',note,
           'Original CPU outputs remain read-only comparison evidence. '+json.dumps(result['metric_notes'],sort_keys=True),'',
           '\t'.join(columns)]
    for cell in cells:
        lines.append('\t'.join('NA' if cell[k] is None else str(cell[k]) for k in columns))
    if program is not None:
        lines+=['',program['note'],'Case-models with CPU diagnostics: '+str(program['diagnostic_case_count'])+
            '; historical numeric threshold failures: '+str(program['historical_backend_failed_case_count'])+'.']
    report_module.save_bytes(folder/'report.txt',('\n'.join(lines)+'\n').encode('utf8'))
    body='<!doctype html><html lang="en"><meta charset="utf-8"><title>Uncertainty and calibration</title><style>body{font:15px Arial;margin:25px}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}</style><h1>Uncertainty and Calibration</h1>'
    body+='<p>'+html.escape(note)+'</p><p>'+html.escape(lines[2])+'</p><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'
    for cell in cells:
        body+='<tr>'+''.join('<td>'+html.escape('NA' if cell[k] is None else str(cell[k]))+'</td>' for k in columns)+'</tr>'
    if program is not None:
        body+='</table><p>'+html.escape(lines[-2])+'</p><p>'+html.escape(lines[-1])+'</p>'
    else:body+='</table>'
    report_module.save_bytes(folder/'report.html',(body+'</html>').encode('utf8'))
    c.save(folder/'receipt.json',c.seal(dict(schema='gpu_table4_report_receipt_v1',status='verified',
        contract=c.desc(root/'contract.json'),cell_count=36,sample_count=contract['sample_count'],
        outputs=[c.desc(folder/x) for x in ('report.json','summary.csv','report.txt','report.html')])))
    return result


def finalize(root,worker):
    root=Path(root).resolve()
    bundle=worker.load_context(gpu=None)
    c,ctx,r=bundle['combined'],bundle['ctx'],bundle['regional_runner']
    contract=worker.verify_contract(root,bundle)
    need(contract['sample_count']==len(ctx['rows'])==21899 and tuple(contract['models'])==tuple(c.MODELS),
         'Final cohort/models differ from clean TEST')
    with c.lock(root,'aggregate'):
        complete=root/'completion.json'
        if complete.exists():
            value=c.checked(c.read(complete))
            need(value['schema']=='test_gpu_program_consistency_completion_v1' and value['status']=='complete_three_models' and value['contract']==c.desc(root/'contract.json') and
                 value['cpu_gpu_numerical_differences_diagnostic_only'] is True,
                 'Existing CUDA completion differs')
            for descriptor in value['outputs']:
                c.bound(descriptor)
            return value
        rp=c.read(c.bound(contract['original_regional_protocol']))
        report_path=c.bound(rp['sources']['report.py'])
        # 核对统计模块来源

        need(Path(r.stats.__file__).resolve()==c.bound(rp['sources']['agreement_stats.py']).resolve(),
             'Loaded regional estimator differs from frozen source')
        report=load(report_path,'_frozen_gpu_original_regional_report')
        reader=lambda model,index:worker.read_unified(root,bundle,contract,model,index)
        cells,audits,statuses,provenance,diagnostics=export_records(root,c,ctx,contract,reader)
        program=program_report(root,c,contract,diagnostics,report)
        s3=table3_report(root,c,contract,report,program)
        s4=table4_report(root,c,contract,cells,audits,statuses,provenance,report,program)
        sr=regional_adapter.finalize(root,c,ctx,contract,reader,report,r.stats)
        outputs=[c.desc(root/'record_manifest.json'),c.desc(root/'table3_stats_manifest.json')]
        for folder in ('table3_report','table4_report','regional_report','program_consistency_report'):
            receipt=c.checked(c.read(root/folder/'receipt.json'))
            need(receipt['status']=='verified' and receipt['contract']==c.desc(root/'contract.json'),'Final report receipt differs')
            for descriptor in receipt['outputs']:
                c.bound(descriptor)
            outputs += [c.desc(root/folder/'receipt.json')]+receipt['outputs']
        result=c.seal(dict(schema='test_gpu_program_consistency_completion_v1',status='complete_three_models',
            contract=c.desc(root/'contract.json'),sample_count=21899,models=contract['models'],
            table3_descriptive_cells=42,table3_paired_tests=78,table4_cells=36,regional_cells=72,
            same_forward_outputs=True,uniform_cuda_primary_cohort=True,old_cpu_results_modified=False,
            excludes_swin_ldm=True,program_consistency_summary=c.desc(root/'program_consistency_report/summary.json'),
            diagnostic_case_count=program['diagnostic_case_count'],
            cpu_gpu_numerical_differences_diagnostic_only=True,outputs=outputs))
        c.save(complete,result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    import worker
    result=finalize(args.root,worker)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
