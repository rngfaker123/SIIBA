"""Original-runtime detector workers; group ownership shared across five models."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode=True
from io_utils import *
DETECTORS={}

def work(index, validate=False, protocol=None, selection=None):
    import cv2
    import numpy as np
    import detector
    import metrics
    from validation_metrics import audit_signal_arrays
    cv2.setNumThreads(1)
    protocol=load_protocol() if protocol is None else protocol
    selection=read(verify(protocol['selection'])) if selection is None else selection
    folder=ROOT/('staged/%06d'%index)
    ready=checked(read(folder/'ready.json'))
    if ready['protocol_sha256']!=sha(ROOT/'protocol.json'):
        raise ValueError('Staged protocol binding differs')
    if (folder/'cpu_complete.json').exists(): return
    held=lock(folder/'cpu.lock')
    if (folder/'cpu_started.json').exists():
        raise ValueError('Partial CPU case needs audited recovery')
    save(folder/'cpu_started.json',seal({'time':now(),'identity':identity(),'index':index,
        'protocol_sha256':sha(ROOT/'protocol.json')}))
    started=now();t=time.monotonic()
    oldpath=folder/'old_F.npy'
    if not oldpath.exists():
        subprocess.run([str(GEOM_PY),str(ROOT/'source/geometry_cache.py'),'--meta',str(folder/'ready.json'),
                        '--output',str(oldpath)],check=True,stdin=subprocess.DEVNULL,timeout=120)
    verify(read(str(oldpath)+'.json')['old_f_coverage'])
    geo=np.load(verify(ready['geometry']),allow_pickle=False)
    rois={k:geo[k] for k in metrics.ROI_ORDER}
    lm=read(verify(ready['landmarks']))
    geometry=detector.prepare_geometry(lm, ready['row']['native_shape'],geo['F_coverage'],
        tools_dir=FROZENROOT/'tools',old_f_coverage=np.load(oldpath,allow_pickle=False))
    shape=tuple(ready['row']['native_shape'])
    if shape not in DETECTORS:DETECTORS[shape]=detector.Detector(shape,tools_dir=FROZENROOT/'tools')
    det=DETECTORS[shape]
    detector_protocol=det.protocol()
    truth=np.load(verify(ready['arrays']['truth']),allow_pickle=False)
    truth_time=time.monotonic();true=det.detect(truth,geometry);truth_seconds=time.monotonic()-truth_time
    np.savez_compressed(str(folder/'truth_signals.npz'),P=true['P'],D=true['D'],F=true['F'],
                        contrast_P=true['contrast_P'],contrast_D=true['contrast_D'])
    save(folder/'truth_detector.json',seal({'index':index,'audit':true['audit'],'geometry':geometry['audit'],
        'arrays':desc(folder/'truth_signals.npz'),'seconds':truth_seconds,'detector_protocol':detector_protocol}))
    spatial=index in selection['spatial_indices'];qc=index in selection['qc_indices']
    model_times={};audits={}
    for mid in MODELS:
        outpath=ROOT/('records/%s/%06d.json'%(mid,index))
        if outpath.exists(): raise ValueError('Partial prior case requires reviewed recovery')
        mt=time.monotonic();pred=np.load(verify(ready['outputs'][mid]['prediction']),allow_pickle=False)
        pd=det.detect(pred,geometry);dt=time.monotonic()-mt
        signals={};metric_seconds=0.;audit_seconds=0.
        for signal in ('P','D'):
            metric_start=time.monotonic()
            signals[signal]=metrics.measure_signal(true[signal],pd[signal],true['contrast_'+signal],
                pd['contrast_'+signal],true['F'],rois,index,signal,spatial=spatial,include_pairs=qc)
            metric_seconds+=time.monotonic()-metric_start
            if validate:
                audit_start=time.monotonic()
                audits[mid+'_'+signal]=audit_signal_arrays(true[signal],pd[signal],true['F'],rois,
                    signals[signal],index,signal,truth_contrast=true['contrast_'+signal],
                    prediction_contrast=pd['contrast_'+signal],check_matching=spatial)
                audit_seconds+=time.monotonic()-audit_start
        arraypath=folder/(mid+'_signals.npz')
        np.savez_compressed(str(arraypath),P=pd['P'],D=pd['D'])
        record=seal({'schema':'signal_objects_3cd_group_v1','time':now(),'index':index,
            'group_id':ready['group_id'],'model':mid,'status':'ok','missing_reason':None,
            'protocol_sha256':sha(ROOT/'protocol.json'),'input_binding':desc(folder/'ready.json'),
            'source_manifest':protocol['original_manifest'],'sources':protocol['sources'],
            'prediction':ready['outputs'][mid], 'truth_detector':desc(folder/'truth_detector.json'),
            'predicted_signals':desc(arraypath),'predicted_detector_audit':pd['audit'],
            'detector_protocol':detector_protocol,
            'geometry_audit':geometry['audit'],'signals':signals,
            'timing':{'prediction_detection_seconds':dt,'metric_seconds':metric_seconds,
                      'independent_audit_seconds':audit_seconds,'total_cpu_seconds':time.monotonic()-mt}})
        save(outpath,record);model_times[mid]=record['timing']
        if qc: render_qc(folder,mid,truth,pred,true,pd)
    if validate: save(folder/'independent_metric_validation.json',seal({'index':index,'checks':audits,'passed':True}))
    save(folder/'cpu_complete.json',seal({'index':index,'time':now(),'started':started,
        'elapsed_seconds':time.monotonic()-t,'truth_detection_seconds':truth_seconds,
        'model_times':model_times,'spatial':spatial,'validated':validate,'identity':identity()}))

def render_qc(folder,model,truth,pred,true,pd):
    import numpy as np
    from PIL import Image,ImageDraw
    images=[]
    for rgb,masks,title in ((truth,true,'True UV'),(pred,pd,model)):
        base=np.rint(np.clip(rgb.transpose(1,2,0),0,1)*255).astype(np.uint8)
        plain=Image.fromarray(base).resize((512,512))
        marked=base.copy();marked[masks['P']]=[255,140,0];marked[masks['D']]=[0,210,240]
        overlay=Image.fromarray(marked).resize((512,512))
        images.extend([plain,overlay])
    board=Image.new('RGB',(2048,552),'white');draw=ImageDraw.Draw(board)
    labels=['True UV','True P/D objects',model+' predicted UV',model+' predicted P/D objects']
    for i,im in enumerate(images):board.paste(im,(512*i,40));draw.text((512*i+8,12),labels[i],fill='black')
    p=ROOT/'qc'/('%s_%s.png'%(folder.name,model));p.parent.mkdir(parents=True,exist_ok=True);board.save(p)

def worker(host,slot,validate=False,indices=None):
    protocol=load_protocol();hostspec=protocol['hosts'][host]
    selection=read(verify(protocol['selection']))
    if host_identity()!=hostspec['identity']:raise ValueError('Host changed')
    os.sched_setaffinity(0,[hostspec['cpu_workers'][slot]])
    os.nice(19-os.getpriority(os.PRIO_PROCESS,0))
    held=lock(ROOT/('locks/cpu_%s_%d.lock'%(host,slot)))
    lanes=[i for i,x in enumerate(protocol['lanes']) if x['host']==host]
    allowed=range(21899) if indices is None else indices
    owned=[i for i in allowed if i%4 in lanes and (i//4)%len(hostspec['cpu_workers'])==slot]
    finished=set();start=time.monotonic()
    while len(finished)<len(owned):
        progress=False
        for index in owned:
            if index in finished:continue
            folder=ROOT/('staged/%06d'%index)
            if (folder/'cpu_complete.json').exists():finished.add(index);continue
            if (folder/'ready.json').exists():
                work(index,validate=validate,protocol=protocol,selection=selection);finished.add(index);progress=True
                save(ROOT/('cpu/%s_%d_progress.json'%(host,slot)),{'time':now(),'done':len(finished),
                    'total':len(owned),'last_index':index,'elapsed_seconds':time.monotonic()-start},True)
                break
        if list((ROOT/'failures').glob('*.json')):raise RuntimeError('Pipeline failure present')
        if not progress:time.sleep(3)
    load_protocol()
    save(ROOT/('cpu/%s_%d_complete.json'%(host,slot)),seal({'time':now(),'count':len(finished),'identity':identity()}))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--host');a.add_argument('--slot',type=int);a.add_argument('--index',type=int);a.add_argument('--validation',action='store_true');x=a.parse_args()
    try:
        if x.index is not None:work(x.index,x.validation)
        else:worker(x.host,x.slot,x.validation)
    except BaseException:
        save(ROOT/('failures/cpu_%d.json'%os.getpid()),{'time':now(),'identity':identity(),'traceback':traceback.format_exc()});raise
