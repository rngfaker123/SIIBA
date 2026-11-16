"""Isolated immutable supplement I/O. No historical files are modified."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

B = Path('/study/siiba')
ROOT = B / 'experiments/signal_objects_3cd_20261029_v1'
FROZENROOT = ROOT / 'frozen'
OLD = B / 'experiments/test_resume_eight_gpu_20261028_v1'
INFER_PY = B / '.venv_h200/bin/python'
MASK_PY = B / 'experiments/mi_atlas_six_20261025_v1/runtime_mask/bin/python'
GEOM_PY = B / 'experiments/mi_atlas_six_20261025_v1/runtime_geometry/bin/python'
MODELS = ('unet_single', 'unet_multi', 'unetplusplus', 'swin', 'ldm')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def encoded(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')

def digest(v):
    return hashlib.sha256(encoded(v)).hexdigest()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def desc(p):
    return {'path': str(p), 'sha256': sha(p)}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def verify(d):
    p = Path(d['path'])
    if sha(p) != d['sha256']:
        raise ValueError('File binding mismatch: ' + str(p))
    return p

def seal(v):
    if 'record_sha256' in v:
        raise ValueError('Already sealed')
    return dict(v, record_sha256=digest(v))

def checked(v):
    x = dict(v); s = x.pop('record_sha256')
    if digest(x) != s:
        raise ValueError('Record seal mismatch')
    return v

def save(p, v, mutable=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(v) + b'\n'
    if p.exists() and not mutable:
        if p.read_bytes() != data:
            raise ValueError('Refusing immutable overwrite: ' + str(p))
        return
    fd, name = tempfile.mkstemp(prefix=p.name+'.', dir=str(p.parent))
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if mutable:
        os.replace(name, p)
    else:
        try:
            os.link(name, p)
        finally:
            Path(name).unlink()

def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    p = Path('/proc') / str(pid)
    t = (p/'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'starttick': int(t[19]), 'ppid': int(t[1]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cpus': sorted(os.sched_getaffinity(pid)), 'nice': os.getpriority(os.PRIO_PROCESS, pid)}

def host_identity():
    return {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid1_starttick': identity(1)['starttick']}

def load_protocol():
    p = read(ROOT/'protocol.json')
    for d in p['sources'].values():
        verify(d)
    for d in p.get('frozen_sources',{}).values():
        verify(d)
    verify(p['original_manifest'])
    verify(p['catalog'])
    verify(p['selection'])
    return p

def lock(path):
    import fcntl
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open('a+')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f

def array_desc(a):
    import numpy as np
    return {'shape': list(a.shape), 'dtype': str(a.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}

def save_array(p, a):
    import numpy as np
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise ValueError('Array already exists: '+str(p))
    temp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with temp.open('xb') as f:
        np.save(f, a, allow_pickle=False)
    try:
        os.link(temp, p)
    finally:
        temp.unlink()
    return desc(p)
