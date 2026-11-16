"""Four nonoverlapping lanes; original frozen FP32 inference once/model/group."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
from io_utils import *

def original_science():
    import importlib.util
    manifest = read(OLD/'manifest.json')
    for name in ('worker', 'common', 'loader_adapter_v1', 'measurement_bridge_v1'):
        verify(manifest['sources'][name])
    sys.path.insert(0, str(OLD/'source'))
    spec = importlib.util.spec_from_file_location('_original_secondary_worker', str(OLD/'source/worker.py'))
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    return w, manifest, w.load_science(manifest, None)

def configure_precision(science, uuid):
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    runtime = science['ctx']['old'].runtime_info('cuda:0', 2)
    expected = dict(science['ctx']['protocol']['runtime'], device='cuda:0')
    if runtime != expected:
        raise ValueError('Original inference runtime/precision differs: '+str(runtime))
    return runtime

def run(lane, indices=None, validation=False):
    protocol = load_protocol(); item = protocol['lanes'][lane]
    if host_identity() != protocol['hosts'][item['host']]['identity']:
        raise ValueError('Host boot/container differs')
    os.sched_setaffinity(0, item['cpus']); os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    held = lock(ROOT/('locks/infer_%d.lock'%lane))
    if (ROOT/('lanes/%d/infer_complete.json'%lane)).exists():
        raise ValueError('Lane already complete')
    w, oldmanifest, science = original_science()
    runtime = configure_precision(science, item['uuid'])
    import numpy as np
    import torch
    models = {}
    for mid in MODELS:
        if mid in ('swin','ldm'):
            models[mid] = w.loader.load_adapter(mid, w.training_bundle(oldmanifest,mid), 'cuda:0')
        else:
            ctx = science['ctx']
            models[mid] = ctx['old'].load_model(mid, ctx['config']['models'][mid],
                Path(ctx['protocol']['eval_config']['path']).parent, ctx['frozen'], 'cuda:0')
    runtime_receipt = seal({'time':now(),'identity':identity(),'host_identity':host_identity(),'lane':lane,'uuid':item['uuid'],
        'runtime':runtime,'protocol_sha256':sha(ROOT/'protocol.json'),'device_name':torch.cuda.get_device_name(0),
        'model_memory_bytes':torch.cuda.memory_allocated()})
    runid = 'validation' if validation else 'production'
    save(ROOT/('lanes/%d/%s_runtime.json'%(lane,runid)), runtime_receipt)
    selection = read(verify(protocol['selection']))
    todo = list(range(lane,21899,4)) if indices is None else list(indices)
    if any(i%4 != lane for i in todo):
        raise ValueError('Overlapping lane assignment')
    start=time.monotonic(); done=0; timings=[]
    pending=[p.parent for p in (ROOT/'staged').glob('*/ready.json')
             if int(p.parent.name)%4==lane and not (p.parent/'cpu_complete.json').exists()]
    for index in todo:
        if list((ROOT/'failures').glob('*.json')): raise RuntimeError('Pipeline failure present')
        case_dir=ROOT/('staged/%06d'%index)
        if (case_dir/'ready.json').exists():
            ready=checked(read(case_dir/'ready.json'))
            if ready['protocol_sha256'] != sha(ROOT/'protocol.json'):
                raise ValueError('Existing stage protocol differs')
            continue
        if (case_dir/'started.json').exists():
            raise ValueError('Interrupted staged case requires explicit audited recovery')
        if not validation:
            # 限制待检测队列
            pending=[p for p in pending if not (p/'cpu_complete.json').exists()]
            while len(pending) >= 96:
                if list((ROOT/'failures').glob('*.json')): raise RuntimeError('Pipeline failure present')
                save(ROOT/('lanes/%d/progress.json'%lane),{'time':now(),'phase':'waiting_cpu_queue','done':done},True)
                time.sleep(10)
                pending=[p for p in pending if not (p/'cpu_complete.json').exists()]
        t=time.monotonic(); case=w.prepare_case(science,index,wait=False); row=case['row']
        save(case_dir/'started.json',seal({'index':index,'group_id':row['sample_id'],'time':now(),
            'lane':lane,'identity':identity(),'protocol_sha256':sha(ROOT/'protocol.json')}))
        base={'index':index,'group_id':row['sample_id'],'row':row,'row_sha256':digest(row),
              'protocol_sha256':sha(ROOT/'protocol.json'),'mask_ref':desc(case['ref_path']),
              'mask_receipt':desc(case['receipt_path']),'lane':lane,'source_manifest':protocol['original_manifest']}
        if case['technical']:
            if index in selection['geometry_valid_indices']:
                raise ValueError('Technical status differs from prospective cohort')
            for mid in MODELS:
                save(ROOT/('records/%s/%06d.json'%(mid,index)),seal(dict(base,model=mid,
                    status='unavailable_geometry',missing_reason=case['receipt']['status'],signals={},forward_count=0)))
            save(case_dir/'cpu_complete.json',seal({'index':index,'status':'unavailable_geometry','time':now()}))
        else:
            if index not in selection['geometry_valid_indices']:
                raise ValueError('Geometry state differs from prospective selection')
            arrays={'truth':save_array(case_dir/'truth.npy',case['truth'])}
            geom={k:v for k,v in case['rois'].items()}
            geom['F_coverage']=case['weights']['F']
            np.savez_compressed(str(case_dir/'geometry.npz'), **geom)
            lm=case['receipt']['evidence']['geometry/landmarks.json']; verify(lm)
            base.update(arrays=arrays,geometry=desc(case_dir/'geometry.npz'),landmarks=lm,
                        roi_audit=case['roi_audit'],inputs={k:array_desc(case[k]) for k in ('rgb','truth')})
            outputs={}
            for mid in MODELS:
                ft=time.monotonic()
                if mid in ('swin','ldm'):
                    converter=w.bridge.TorchTensorIO('cuda:0')
                    x=converter.as_input(case['rgb'])
                    with torch.inference_mode(): out=models[mid].predict(x,row)
                    pred,sigma=converter.prediction_pair(out)
                else:
                    pred,sigma=science['ctx']['old'].infer(models[mid],case['rgb'],mid,'cuda:0')
                if pred.dtype!=np.float32 or pred.shape!=(3,1024,1024) or not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1:
                    raise ValueError('Invalid frozen prediction tensor '+mid)
                if not np.isfinite(sigma).all() or np.any(sigma<=0): raise ValueError('Invalid sigma')
                torch.cuda.synchronize()
                outputs[mid]={'prediction':save_array(case_dir/(mid+'.npy'),pred),
                    'tensor':array_desc(pred),'sigma_diagnostic':array_desc(sigma),
                    'forward_seconds':time.monotonic()-ft,'forward_count':1,'runtime':runtime_receipt,
                    'checkpoint_sha256':oldmanifest['training_bundles'][mid]['best_checkpoint']['sha256'] if mid in ('swin','ldm') else science['ctx']['old'].MODEL_PINS[mid]['sha']}
            base['outputs']=outputs
        save(case_dir/'ready.json',seal(base))
        if not case['technical']:pending.append(case_dir)
        done+=1; timings.append(time.monotonic()-t)
        save(ROOT/('lanes/%d/progress.json'%lane),{'time':now(),'phase':'inference','done':done,
             'last_index':index,'elapsed_seconds':time.monotonic()-start,'recent_group_seconds':timings[-20:]},True)
    save(ROOT/('lanes/%d/%s_complete.json'%(lane,runid)),seal({'time':now(),'lane':lane,'indices':todo,
        'processed_this_run':done,'elapsed_seconds':time.monotonic()-start,'timings':timings}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lane',type=int,required=True);ap.add_argument('--indices');ap.add_argument('--validation',action='store_true');a=ap.parse_args()
    try: run(a.lane, None if not a.indices else [int(x) for x in a.indices.split(',')], a.validation)
    except BaseException:
        save(ROOT/('failures/infer_%d_%d.json'%(a.lane,os.getpid())),{'time':now(),'identity':identity(),'traceback':traceback.format_exc()})
        raise
