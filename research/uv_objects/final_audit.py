"""Independent final 3C/3D coverage, saved-array and point-estimate audit.

Never runs a model or detector. All masks/scalars/windows/Dice are audited;
independent matching objectives/RNG are additionally audited for the fixed QC
sample. Bootstrap confidence intervals are retained, not recomputed here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np

import io_utils as util

MODELS = util.MODELS
SIGNALS = ('P', 'D')
ROIS = ('forehead', 'image_left_cheek', 'image_right_cheek', 'nose', 'chin', 'F_remainder')
SCALARS = ('count', 'area_fraction', 'contrast_burden')
SPATIAL = ('precision', 'recall', 'f1', 'mean_center_error')
CHECKPOINTS = {'unet_single': '6f9e5c2a8745a3f67794f993961bfc6612421f04be178f41288f704c53d3cbfb',
              'unet_multi': '204dbffedc15fa2cc6053ce6b6d1aa1464f9978c103c5814f84692d69c80844e',
              'unetplusplus': '81d7dc5d40f2b36d2ac64a10ca6a5a443af3854d70fc88c75027ca5ecd2dec5f',
              'swin': '5c9da944dd83f9478fbdcb96e75a4108f1fc2ae35243786b0f05d139fc6aadad',
              'ldm': 'e294864b955abf20461dd51c66cd08d8b12508aa8746b2e718e6e11556b76dbf'}
STATE = {}


def require(condition, reason):
    if not condition:
        raise ValueError('Independent final audit: '+reason)


def equal(actual, expected, label, tolerance=1e-11):
    if expected is None:
        require(actual is None, label+' must be null')
    else:
        require(actual is not None and math.isfinite(float(actual)), label+' must be finite')
        require(abs(float(actual)-float(expected)) <= tolerance*max(1., abs(float(expected))), label+' differs')


def loaded(path_or_descriptor, manifest, kind='json', sealed=False):
    descriptor = path_or_descriptor if isinstance(path_or_descriptor, dict) else None
    path = Path(descriptor['path'] if descriptor else path_or_descriptor)
    payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
    if descriptor:
        require(digest == descriptor['sha256'], 'Byte binding '+str(path))
    manifest[str(path)] = {'path': str(path), 'sha256': digest, 'bytes': len(payload)}
    if kind == 'json':
        value = json.loads(payload)
        if sealed:
            util.checked(value)
        return value
    if kind == 'npy':
        return np.load(io.BytesIO(payload), allow_pickle=False)
    if kind == 'npz':
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return payload


def domains_from_geometry(f, roi_arrays):
    domains = {}; used = np.zeros(f.shape, bool)
    for name in ROIS[:-1]:
        a = roi_arrays[name]
        require(a.shape == f.shape and np.isfinite(a).all(), 'ROI shape/finiteness')
        domains[name] = f & (a >= .5) & ~used
        used |= domains[name]
    domains[ROIS[-1]] = f & ~used
    return domains


def contrast_fields(rgb, native_shape):
    """Reproduce frozen encoded contrasts without importing detector code."""
    a = np.asarray(rgb)
    require(a.dtype == np.float32 and a.shape == (3, 1024, 1024) and np.isfinite(a).all(), 'Frozen float tensor')
    r, g, b = a
    p = np.maximum(r-(g+b)*np.float32(.5), np.float32(0))
    scale = 1024./min(map(int, native_shape))
    value = np.maximum(np.maximum(r, g), b)
    smooth = cv2.GaussianBlur(value, (0, 0), 3.*scale)
    d = np.maximum(np.maximum(cv2.GaussianBlur(smooth, (0, 0), 20.*scale)-smooth,
                              cv2.GaussianBlur(smooth, (0, 0), 60.*scale)-smooth), np.float32(0))
    return {'P': p, 'D': d}


def independent_scalars(mask, contrast, f, domains):
    """Independent OpenCV8CC; production uses scipy.ndimage8CC."""
    require(mask.dtype == np.bool_ and mask.shape == f.shape and not np.any(mask & ~f), 'Signal/F mask')
    require(contrast.shape == f.shape and np.isfinite(contrast).all() and (contrast >= 0).all(), 'Contrast field')
    number, labels, stats, xy = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    centres = xy[1:, ::-1]
    require(int(stats[1:, cv2.CC_STAT_AREA].sum()) == int(mask.sum()), '8CC pixel conservation')
    memberships = []
    for cy, cx in centres:
        y, x = min(f.shape[0]-1, math.floor(cy+.5)), min(f.shape[1]-1, math.floor(cx+.5))
        memberships.append(next((name for name in ROIS[:-1] if domains[name][y, x]), ROIS[-1]))
    def one(domain, count):
        n = int(np.count_nonzero(domain)); support = domain & mask
        pixels = int(np.count_nonzero(support))
        return {'count': int(count), 'signal_pixels': pixels, 'valid_pixels': n,
                'area_fraction': pixels/n if n else None,
                'contrast_burden': float(np.sum(contrast[support], dtype=np.float64)/n) if n else None,
                'reason': None if n else 'empty_anatomical_domain'}
    return {'F': one(f, number-1), **{name: one(domain, memberships.count(name)) for name, domain in domains.items()}}


def check_scalar(actual, expected, label):
    require(set(actual) == set(expected), label+' scalar fields')
    for k in ('count', 'signal_pixels', 'valid_pixels', 'reason'):
        require(actual[k] == expected[k], label+'/'+k)
    for k in ('area_fraction', 'contrast_burden'):
        equal(actual[k], expected[k], label+'/'+k)


def independent_windows(t, p, f, width):
    labels, scores, origins, valid_pixels = [], [], [], []
    for y in range(0, f.shape[0], width):
        for x in range(0, f.shape[1], width):
            sl = np.s_[y:y+width, x:x+width]; n = int(np.count_nonzero(f[sl]))
            if 2*n >= width*width:
                labels.append(int(np.any(t[sl]))); scores.append(int(np.count_nonzero(p[sl]))/n)
                origins.append([y, x]); valid_pixels.append(n)
    return {'labels': labels, 'scores': scores, 'window_origin_yx': origins,
            'F_pixel_count': valid_pixels, 'positive_windows': sum(labels),
            'negative_windows': len(labels)-sum(labels), 'window_count': len(labels),
            'prevalence': sum(labels)/len(labels) if labels else None}


def spatial_coherence(sp, t, p, nt, npred):
    denominator = int(np.count_nonzero(t))+int(np.count_nonzero(p))
    expected_dice = 2.*int(np.count_nonzero(t & p))/denominator if denominator else None
    equal(sp['dice'], expected_dice, 'all-selected Dice')
    require(sp['correct_negative'] == (nt+npred == 0), 'Correct negative flag')
    require([v['repeat'] for v in sp['random_repeats']] == list(range(10)), 'Random repetitions')
    for item in [sp]+sp['random_repeats']:
        if item is not sp:
            require('dice' not in item, 'No fictional random Dice')
        require(set(item['by_tolerance']) == {'2', '5', '10'}, 'Spatial tolerances')
        previous = -1
        for gate in ('2', '5', '10'):
            v = item['by_tolerance'][gate]; k = v['tp']
            require(isinstance(k, int) and 0 <= k <= min(nt, npred), 'TP domain')
            require((v['fp'], v['fn']) == (npred-k, nt-k), 'FP/FN counts')
            require(v['truth_objects']==nt and v['prediction_objects']==npred and v['matched_count']==k, 'Matching object totals')
            require(k >= previous, 'Cardinality cannot decrease with tolerance'); previous = k
            equal(v['f1'], 2.*k/(nt+npred) if nt+npred else None, 'F1 null/value')
            equal(v['precision'], k/npred if npred else None, 'Precision null/value')
            equal(v['recall'], k/nt if nt else None, 'Recall null/value')
            if k:
                require(0 <= v['mean_center_error'] <= float(gate), 'Mean distance domain')
                equal(v['center_error_sum'], v['mean_center_error']*k, 'Per-group distance aggregation')
            else:
                equal(v['mean_center_error'], None, 'No-match center NA')
                equal(v['center_error_sum'], None, 'No-match sum NA')


def initialize(root, cpus):
    global STATE
    if cpus and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, cpus)
    cv2.setNumThreads(1)
    root = Path(root); protocol = util.read(root/'protocol.json')
    selection = util.read(util.verify(protocol['selection']))
    STATE = {'root': root, 'protocol': protocol, 'protocol_sha256': util.sha(root/'protocol.json'),
             'selection': selection, 'valid': set(selection['geometry_valid_indices']),
             'spatial': set(selection['spatial_indices']), 'qc': set(selection['qc_indices']),
             'missing': {x['index']: x['reason'] for x in selection['missing']}}


def audit_group(index):
    state = STATE; root = state['root']; protocol = state['protocol']; manifest = {}
    folder = root/('staged/%06d'%index); group = state['selection']['all_group_ids'][index]
    ready = loaded(folder/'ready.json', manifest, sealed=True)
    done = loaded(folder/'cpu_complete.json', manifest, sealed=True)
    require(done['index'] == ready['index'] == index and ready['group_id']==group, 'Stage identity')
    require(ready['row_sha256']==util.digest(ready['row']), 'Row digest')
    require(ready['protocol_sha256']==state['protocol_sha256'], 'Stage protocol')
    records = {}
    for model in MODELS:
        record = loaded(root/('records/%s/%06d.json'%(model, index)), manifest, sealed=True)
        require((record['index'],record['group_id'],record['model'])==(index,group,model), 'Record identity')
        require(record['protocol_sha256']==state['protocol_sha256'], 'Record protocol')
        records[model] = record
    if index not in state['valid']:
        for record in records.values():
            require(record['status']=='unavailable_geometry' and record['missing_reason']==state['missing'][index]
                    and record['forward_count']==0 and record['signals']=={}, 'Frozen missing reason/zero forward')
        return {'index': index, 'valid': False, 'missing_reason': state['missing'][index], 'manifest': list(manifest.values())}
    geometry = loaded(ready['geometry'], manifest, kind='npz')
    landmarks = loaded(ready['landmarks'], manifest)
    loaded(ready['mask_ref'], manifest); loaded(ready['mask_receipt'], manifest)
    truth_float = loaded(ready['arrays']['truth'], manifest, kind='npy')
    require(util.array_desc(truth_float)==ready['inputs']['truth'], 'Truth tensor descriptor')
    td = loaded(folder/'truth_detector.json', manifest, sealed=True)
    true = loaded(td['arrays'], manifest, kind='npz')
    f = true['F']; require(f.dtype == np.bool_ and f.any() and np.array_equal(f,geometry['F_coverage']>=.5), 'Common F')
    require(not np.any(true['P'] & true['D']), 'Truth P/D disjoint')
    roi_arrays = {k: geometry[k] for k in ROIS[:-1]}; domains = domains_from_geometry(f,roi_arrays)
    truth_contrast = contrast_fields(truth_float, ready['row']['native_shape'])
    true_scalar = {}
    for signal in SIGNALS:
        require(np.array_equal(true['contrast_'+signal],truth_contrast[signal]), 'Saved truth contrast independently reproduced')
        true_scalar[signal] = independent_scalars(true[signal], truth_contrast[signal], f, domains)
    facts = {}; full_audit_counts = {}
    for model, record in records.items():
        require(record['status']=='ok' and record['missing_reason'] is None, 'Valid record status')
        require(record['sources']==protocol['sources'] and record['source_manifest']==protocol['original_manifest'], 'Source binding')
        require(record['prediction']==ready['outputs'][model], 'One shared stage prediction binding')
        pred_desc = record['prediction']; runtime = util.checked(pred_desc['runtime'])
        require(pred_desc['forward_count']==1 and pred_desc['checkpoint_sha256']==CHECKPOINTS[model], 'Frozen checkpoint/one forward')
        lane=index%4; item=protocol['lanes'][lane]
        require(runtime['lane']==lane and runtime['uuid']==item['uuid'], 'GPU lane binding')
        require(runtime['identity']['boot_id']==protocol['hosts'][item['host']]['identity']['boot_id'], 'Recorded inference host')
        require(record['input_binding']['path']==str(folder/'ready.json') and record['input_binding']['sha256']==manifest[str(folder/'ready.json')]['sha256'], 'Ready binding')
        require(record['truth_detector']['path']==str(folder/'truth_detector.json') and record['truth_detector']['sha256']==manifest[str(folder/'truth_detector.json')]['sha256'], 'Shared truth binding')
        prediction = loaded(pred_desc['prediction'], manifest, kind='npy')
        require(util.array_desc(prediction)==pred_desc['tensor'], 'Saved prediction tensor binding')
        masks = loaded(record['predicted_signals'], manifest, kind='npz')
        require(not np.any(masks['P'] & masks['D']), 'Prediction P/D disjoint')
        contrast = contrast_fields(prediction,ready['row']['native_shape'])
        facts[model] = {}
        for signal in SIGNALS:
            current=record['signals'][signal]
            ps=independent_scalars(masks[signal],contrast[signal],f,domains)
            for roi in ('F',)+ROIS:
                a=current if roi=='F' else current['rois'][roi]
                check_scalar(a['truth'],true_scalar[signal][roi],model+'/'+signal+'/'+roi+'/truth')
                check_scalar(a['prediction'],ps[roi],model+'/'+signal+'/'+roi+'/prediction')
            windows={}
            for width in (64,128):
                expected=independent_windows(true[signal],masks[signal],f,width)
                require(current['windows'][str(width)]==expected, 'All window values/domains '+str(width))
                windows[str(width)]={'labels':expected['labels'],'scores':expected['scores']}
            require(('spatial' in current)==(index in state['spatial']), 'Fixed spatial subset')
            if index in state['spatial']:
                spatial_coherence(current['spatial'],true[signal],masks[signal],true_scalar[signal]['F']['count'],ps['F']['count'])
            if index in state['qc']:
                from validation_metrics import audit_signal_arrays
                full_audit_counts[model+'_'+signal]=audit_signal_arrays(true[signal],masks[signal],f,roi_arrays,
                    current,index,signal,truth_contrast=truth_contrast[signal],prediction_contrast=contrast[signal],check_matching=True)['checks']
            facts[model][signal]={'truth':true_scalar[signal], 'prediction':ps,'windows':windows,
                                 'spatial':current.get('spatial')}
    return {'index':index,'valid':True,'facts':facts,'full_matching_checked':index in state['qc'],
            'full_matching_checks':full_audit_counts,'manifest':list(manifest.values())}


class PairedMoments:
    """Independent streaming covariance formula for ICC(A,1)."""
    def __init__(self):
        self.total=0; self.n=0; self.mx=0.;self.my=0.;self.xx=0.;self.yy=0.;self.xy=0.;self.absolute=0.;self.bias=0.
    def add(self,x,y):
        self.total+=1
        if x is None or y is None:return
        x=float(x);y=float(y);require(math.isfinite(x) and math.isfinite(y),'Finite summary input')
        self.n+=1;dx=x-self.mx;dy=y-self.my;self.mx+=dx/self.n;self.my+=dy/self.n
        self.xx+=dx*(x-self.mx);self.yy+=dy*(y-self.my);self.xy+=dx*(y-self.my)
        self.absolute+=abs(y-x);self.bias+=y-x
    def result(self):
        n=self.n;icc=None
        if n>1:
            msr=(self.xx+self.yy+2*self.xy)/(2*(n-1));mse=(self.xx+self.yy-2*self.xy)/(2*(n-1))
            msc=n*(self.mx-self.my)**2/2
            den=msr+mse+2*(msc-mse)/n
            if den>0:icc=(msr-mse)/den
        return {'n':n,'n_undefined':self.total-n,'mae':self.absolute/n if n else None,
                'bias':self.bias/n if n else None,'truth_mean':self.mx if n else None,
                'prediction_mean':self.my if n else None,'icc_a1':icc}


def mean_estimate(values):
    vals=[float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in vals),'Finite spatial summary')
    n=len(vals);mean=math.fsum(vals)/n if n else None
    return {'n':n,'mean':mean,'sample_sd':math.sqrt(math.fsum((x-mean)**2 for x in vals)/(n-1)) if n>1 else None}


def window_estimate(parts):
    labels=np.concatenate([np.asarray(p['labels'],np.uint8) for p in parts]) if parts else np.empty(0,np.uint8)
    scores=np.concatenate([np.asarray(p['scores'],np.float64) for p in parts]) if parts else np.empty(0)
    pos=int(labels.sum());neg=len(labels)-pos
    result={'positive_windows':pos,'negative_windows':neg,'window_count':len(labels),
            'prevalence':pos/len(labels) if len(labels) else None,'group_count':len(parts),
            'groups_with_windows':sum(bool(p['labels']) for p in parts),
            'groups_without_windows':sum(not p['labels'] for p in parts),
            'unique_exact_score_count':len(np.unique(scores)),
            'auroc':None,'pr_auc_trapezoid':None,'ap_noninterpolated':None}
    if not pos or not neg:return result
    # 独立排序核验
    order=np.argsort(-scores,kind='mergesort');s=scores[order];y=labels[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    cumulative=np.cumsum(y,dtype=np.int64);tp=cumulative[ends];fp=ends+1-tp
    recall=np.r_[0.,tp/pos];precision=np.r_[1.,tp/(ends+1)];fpr=np.r_[0.,fp/neg]
    result['auroc']=float(np.sum((recall[:-1]+recall[1:])*np.diff(fpr)/2))
    result['pr_auc_trapezoid']=float(np.sum((precision[:-1]+precision[1:])*np.diff(recall)/2))
    result['ap_noninterpolated']=float(np.dot(np.diff(recall),precision[1:]))
    return result


class SummaryAudit:
    def __init__(self):
        self.scalar={};self.windows={};self.spatial={};self.face={};self.valid=0;self.missing={}
    def add(self,result):
        if not result['valid']:
            reason=result['missing_reason'];self.missing[reason]=self.missing.get(reason,0)+1;return
        self.valid+=1
        for model,values in result['facts'].items():
            for signal,v in values.items():
                key=(model,signal);face=self.face.setdefault(key,{'positive':0,'negative':0})
                face['positive' if v['truth']['F']['count'] else 'negative']+=1
                for roi in ('F',)+ROIS:
                    for metric in SCALARS:
                        self.scalar.setdefault(key+(roi,metric),PairedMoments()).add(v['truth'][roi][metric],v['prediction'][roi][metric])
                for width,value in v['windows'].items():self.windows.setdefault(key+(width,),[]).append(value)
                if v['spatial'] is not None:self.spatial.setdefault(key,[]).append(v['spatial'])
    def compare(self,summary,all_count,spatial_count):
        checks=0
        for model in MODELS:
            observed=summary['models'][model]
            require(observed['coverage']=={'all':all_count,'geometry_valid':self.valid,'missing_reasons':self.missing,'spatial_selected':spatial_count},'Summary coverage')
            for signal in SIGNALS:
                key=(model,signal);s=observed[signal]
                require(s['whole_face_presence_descriptive']==self.face[key],'Whole face prevalence')
                for roi in ('F',)+ROIS:
                    for metric in SCALARS:
                        actual=s['F'][metric] if roi=='F' else s['rois_descriptive'][roi][metric]
                        for name,value in self.scalar[key+(roi,metric)].result().items():
                            equal(actual[name],value,'Summary '+str(key+(roi,metric,name)),2e-9);checks+=1
                for width in ('64','128'):
                    for name,value in window_estimate(self.windows[key+(width,)]).items():
                        equal(s['windows'][width][name],value,'Window summary '+name);checks+=1
                entries=self.spatial[key];require(len(entries)==spatial_count,'Spatial coverage')
                for tolerance in ('2','5','10'):
                    sp=s['spatial'][tolerance]
                    for metric in ('dice',)+SPATIAL:
                        vals=[x['dice'] if metric=='dice' else x['by_tolerance'][tolerance][metric] for x in entries]
                        expected=mean_estimate(vals);expected['n_undefined']=spatial_count-expected['n']
                        for name,value in expected.items():equal(sp[metric][name],value,'Spatial summary '+metric+'/'+name);checks+=1
                    for metric in SPATIAL:
                        vals=[]
                        for x in entries:
                            values=[r['by_tolerance'][tolerance][metric] for r in x['random_repeats']]
                            vals.append(mean_estimate(values)['mean'])
                        for name,value in mean_estimate(vals).items():
                            equal(sp['conditional_random_centers'][metric][name],value,'Random summary '+metric+'/'+name);checks+=1
                    require(sp['conditional_random_repeat_counts']==[10],'Random repeat summary')
                    for count in ('tp','fp','fn'):
                        require(sp['tp_fp_fn_pooled_descriptive'][count]==sum(x['by_tolerance'][tolerance][count] for x in entries),'Pooled count')
                    require(sp['correct_negative_groups']==sum(x['correct_negative'] for x in entries),'Correct negative summary')
                    require(sp['selected_groups']==spatial_count,'Spatial selected summary')
        return checks


def run(root=None,workers=24,cpus=None):
    root=Path(root or util.ROOT).resolve();workers=int(workers)
    require(1<=workers<=24,'Bounded 1..24 audit workers')
    require(cv2.__version__=='4.10.0' and np.__version__=='2.3.4','Pinned mask runtime required for contrast reproduction')
    protocol=util.read(root/'protocol.json');protocol_sha=util.sha(root/'protocol.json')
    for descriptor in list(protocol['sources'].values())+list(protocol.get('frozen_sources',{}).values()):util.verify(descriptor)
    selection=util.read(util.verify(protocol['selection']))
    require((len(selection['all_group_ids']),len(selection['geometry_valid_indices']),len(selection['spatial_indices']),len(selection['qc_indices']))==(21899,21422,1000,24),'Frozen audit cohort sizes')
    require(len(set(selection['all_group_ids']))==21899,'Unique catalog group IDs')
    report=root/'report';completion=util.read(report/'completion.json')
    require(completion['status']=='complete' and completion['summary_sha256']==util.sha(report/'summary.json'),'Collector summary completion binding')
    require(completion['record_manifest_sha256']==util.sha(report/'record_manifest.json'),'Collector manifest binding')
    for model in MODELS:
        require(sorted(p.name for p in (root/'records'/model).glob('*.json'))==['%06d.json'%i for i in range(21899)],'Exact record coverage '+model)
    output=report/'independent_audit_v1';output.mkdir(exist_ok=False)
    started=time.monotonic();audit=SummaryAudit();done=0;qc_done=0;full_checks=0
    try:
        with (output/'consumed_files_sha256.jsonl').open('x',encoding='utf8') as manifest_file, \
             (output/'group_audits.jsonl').open('x',encoding='utf8') as groups_file, \
             ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),cpus)) as pool:
            todo=iter(range(21899));pending={}
            for _ in range(workers*2):
                i=next(todo,None)
                if i is not None:pending[pool.submit(audit_group,i)]=i
            while pending:
                finished,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    index=pending.pop(future);result=future.result();require(result['index']==index,'Audit task identity')
                    audit.add(result);done+=1;qc_done+=int(result.get('full_matching_checked',False))
                    full_checks+=sum(v['matching'] for v in result.get('full_matching_checks',{}).values())
                    for entry in result.pop('manifest'):manifest_file.write(json.dumps(entry,sort_keys=True)+'\n')
                    result.pop('facts',None)
                    groups_file.write(json.dumps(result,sort_keys=True)+'\n')
                    if done%100==0:
                        manifest_file.flush();groups_file.flush()
                        util.save(output/'progress.json',{'time':util.now(),'groups':done,'full_matching_groups':qc_done,
                                  'elapsed_seconds':time.monotonic()-started},mutable=True)
                    i=next(todo,None)
                    if i is not None:pending[pool.submit(audit_group,i)]=i
        require(done==21899 and audit.valid==21422 and qc_done==24,'Audit terminal coverage')
        summary=util.read(report/'summary.json');summary_checks=audit.compare(summary,21899,1000)
        require(util.sha(root/'protocol.json')==protocol_sha,'Protocol changed during audit')
        require(util.sha(report/'summary.json')==completion['summary_sha256'],'Summary changed during audit')
        # 逐组核验记录
        for item in util.read(report/'record_manifest.json'):util.verify(item)
        outputs={p.name:util.desc(p) for p in sorted(report.glob('*')) if p.is_file()}
        util.save(output/'report_files_sha256.json',outputs)
        result=util.seal({'schema':'signal_objects_3cd_independent_final_audit_v1','status':'passed','time':util.now(),
            'protocol_sha256':protocol_sha,'groups':done,'geometry_valid_groups':audit.valid,'missing_reasons':audit.missing,
            'model_records':done*5,'saved_mask_scalar_window_audits':21422*5*2,'spatial_Dice_and_coherence_audits':1000*5*2,
            'full_independent_matching_groups':qc_done,'independent_matching_objectives':full_checks,
            'summary_point_estimate_checks':summary_checks,'qc_indices':selection['qc_indices'],'elapsed_seconds':time.monotonic()-started,
            'workers':workers,'cpu_affinity_requested':cpus,'runtime':{'opencv':cv2.__version__,'numpy':np.__version__},
            'methods':['Independent OpenCV8CC versus production SciPy8CC','Independent saved-float P/D contrast fields',
                       'All F/anatomical ROI scalars and explicit-loop64/128 windows','All1000-sample Dice/count/null/monotonicity checks',
                       'Fixed24-group run-length8CC, DP/dense-LAP matching and independent conditional random draws',
                       'Streaming covariance ICC(A,1), MAE and independent exact-score curve point estimates'],
            'limits':['No new model or detector execution; detector correctness rests on prior frozen VAL acceptance.',
                      'Full independent matching objective audit covers fixed24 QC groups; all1000 have complete metric coherence and Dice checks.',
                      'Bootstrap confidence intervals and paired-test p-values are source-tested but not independently recomputed in this audit.',
                      'Algorithm-defined objects are not clinical annotations; sample groups are not verified patients.'],
            'evidence':{name:util.desc(output/name) for name in ('consumed_files_sha256.jsonl','group_audits.jsonl','report_files_sha256.json')}})
        util.save(output/'acceptance.json',result);return result
    except BaseException:
        util.save(output/'failure.json',{'time':util.now(),'completed_groups':done,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--cpus',help='Comma-separated released CPU IDs')
    args=parser.parse_args();print(json.dumps(run(args.root,args.workers,None if not args.cpus else list(map(int,args.cpus.split(','))))))
