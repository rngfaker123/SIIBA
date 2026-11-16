"""One forward per valid case and frozen architecture. CPU is validation only."""
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
TASKS = [('unet_single', 'plh'), ('unet_single', 'plv'), ('unet_multi', 'both'),
         ('unetplusplus', 'plv'), ('unetplusplus', 'plh'), ('swin', 'plh'),
         ('ldm', 'plh'), ('swin', 'plv'), ('ldm', 'plv')]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def raw(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def publish(path, data, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(data, bytes):
        data = raw(data)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.partial-')
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if mutable:
            os.replace(tmp, str(path))
        else:
            os.link(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def desc(path):
    return {'path': str(path), 'sha256': sha(path)}


def check_source():
    p = read(ROOT/'protocol.json')
    for rel, checksum in p['sources'].items():
        need(sha(ROOT/rel) == checksum, 'Changed program: '+rel)
    need(sha(p['selection']['path']) == p['selection']['sha256'], 'Changed TEST selection')
    return p


def task_targets(key):
    return ['plh', 'plv'] if key == 'both' else [key]


def torch_setup(device):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device != 'cpu':
        need(torch.cuda.device_count() == 1, 'GPU worker must see exactly one allocated UUID')
        torch.cuda.set_device(0)
    return torch


def array(v):
    import numpy as np
    a = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
    need(a.shape in ((3,1024,1024),(1,3,1024,1024)), 'Unexpected image geometry')
    need(a.dtype == np.float32 and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, 'Prediction outside declared domain')
    return np.ascontiguousarray(a.reshape(3,1024,1024))


def run(mode, architecture, key, lane):
    import numpy as np
    import pl_measure
    import pl_models
    policy = check_source()
    name = architecture+'_'+key
    targets = task_targets(key)
    ident = pl_models.model_identity(architecture, key)
    model_sha = hashlib.sha256(raw(ident)).hexdigest()
    device = 'cpu' if mode=='cpupilot' else 'cuda:0'
    torch = torch_setup(device)
    uuid = 'CPU' if device=='cpu' else os.environ['CUDA_VISIBLE_DEVICES']
    model = pl_models.load_model(architecture, key, device=device)
    need(model.identity == ident,'Loaded model differs from admitted model')
    ctx = pl_measure.Context()
    engine = pl_measure.Measurement(device)
    selected = read(policy['selection']['path'])
    ids = selected['all_group_ids']
    valid = set(selected['geometry_valid_indices'])
    protocol_sha = sha(ROOT/'protocol.json')
    cpudir = ROOT/'cpu_reference'/name
    def forward(case):
        with torch.inference_mode():
            prediction = model.predict(case)
        need(set(prediction)==set(targets), 'Unexpected target keys')
        return prediction
    def bound_case(i):
        case=ctx.case(i,targets)
        gid=case.get('sample_id') or case.get('group_id') or case.get('row',{}).get('sample_id')
        need(gid==ids[i] and (case.get('missing_reason') is None)==(i in valid),'Group or eligibility does not match frozen selection')
        return case
    if mode in ('cpupilot','gate'):
        diagnostics = []
        for i in policy['pilot_indices']:
            case = bound_case(i)
            pred = forward(case) if i in valid else {}
            measured = engine.measure(case,pred)
            details = engine.check(case,pred,measured) if pred else None
            comparisons = {}
            if pred:
                again = forward(case)
                for target in targets:
                    first = array(pred[target]); repeat = array(again[target])
                    need(np.array_equal(first, repeat), 'Same-device repeat mismatch')
                    sample = first.ravel()[::policy['pilot_value_stride']].copy()
                    ref = cpudir/(str(i)+'_'+target+'.npy')
                    if mode=='cpupilot':
                        import io
                        b=io.BytesIO(); np.save(b,sample,allow_pickle=False); publish(ref,b.getvalue())
                        comparisons[target]={'float32_sample':desc(ref),'image_sha256':hashlib.sha256(first.tobytes()).hexdigest()}
                    else:
                        saved=read(cpudir/'acceptance.json')
                        need(saved['model_identity_sha256']==model_sha and saved['protocol_sha256']==protocol_sha,'CPU validation binding differs')
                        item=next(x for x in saved['cases'] if x['index']==i)['comparison'][target]
                        need(desc(ref)==item['float32_sample'],'CPU pilot values changed')
                        delta=np.abs(np.load(ref,allow_pickle=False).astype('float64')-sample)
                        comparisons[target]={'value_count':int(delta.size),'mean_abs':float(delta.mean()),'max_abs':float(delta.max()),'cpu_sha256':item['image_sha256'],'gpu_sha256':hashlib.sha256(first.tobytes()).hexdigest()}
            diagnostics.append(dict(index=i,sample_id=ids[i],comparison=comparisons,measurement_check=details,measured=measured))
        path = cpudir/'acceptance.json' if mode=='cpupilot' else ROOT/'pilots'/name/(str(lane)+'.json')
        publish(path,dict(status='passed',time=now(),architecture=architecture,targets=targets,model_identity_sha256=model_sha,protocol_sha256=protocol_sha,device_uuid=uuid,cases=diagnostics))
        return
    need(mode=='worker' and 0<=lane<3,'Invalid production worker')
    binding=read(ROOT/'bindings'/(name+'.json'))
    need(binding['model_identity_sha256']==model_sha and binding['protocol_sha256']==protocol_sha and binding['targets']==targets,'Wrong model binding')
    for pin in binding['pilot_acceptances']:
        need(sha(pin['path'])==pin['sha256'] and read(pin['path'])['status']=='passed','Invalid pilot gate')
    need(read(binding['pilot_acceptances'][lane]['path'])['device_uuid']==uuid,'Unvalidated GPU')
    started=time.monotonic(); ng=0; nf=0
    for start in range(32*lane,len(ids),32*3):
        dst=ROOT/'chunks'/name/('%05d.jsonl.gz'%start)
        need(not dst.exists(),'Existing production chunk; refusing duplicate work')
        chunk=[]
        for i in range(start,min(len(ids),start+32)):
            case=bound_case(i)
            pred=forward(case) if i in valid else {}
            result=engine.measure(case,pred)
            need(set(result)==set(targets),'Missing target measurement')
            row=dict(index=i,sample_id=ids[i],architecture=architecture,model_identity_sha256=model_sha,
                     measurements=result,protocol_sha256=protocol_sha,runtime={'device_uuid':uuid},
                     input_binding=case.get('inputs',{}),forward_count=int(bool(pred)))
            chunk.append(raw(row)+b'\n')
            ng+=1; nf+=int(bool(pred))
        publish(dst,gzip.compress(b''.join(chunk),compresslevel=3,mtime=0))
        publish(ROOT/'progress'/name/('%d.json'%lane),dict(time=now(),last_index=i,groups=ng,forwards=nf,elapsed_sec=time.monotonic()-started),True)
    publish(ROOT/'lanes'/name/('%d.json'%lane),dict(status='complete',time=now(),groups=ng,forwards=nf,device_uuid=uuid,model_identity_sha256=model_sha,protocol_sha256=protocol_sha))


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]))
