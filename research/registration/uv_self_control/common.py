import datetime, gzip, hashlib, io, json, math, os, subprocess, tempfile
from pathlib import Path

E=Path('/study/siiba/experiments')
ROOT=E/'uv_misalignment_sensitivity_20261030_v1'
BASE=E/'signal_objects_3cd_20261029_v1'
PY=Path('/study/siiba/.venv_h200/bin/python')
MASKPY=E/'mi_atlas_six_20261025_v1/runtime_mask/bin/python'
REGIONS=('P','D','O','F')
MAGNITUDES=(0,1,2,3,5)
SIGMAS=(0.,.5,1.,2.,4.)
SHIFTS=[(0,0,0)]+[(r,dx,dy) for r in MAGNITUDES[1:] for dx,dy in [(r,0),(-r,0),(0,r),(0,-r)]]
CONDITIONS=[dict(sigma=s,radius=r,dx=dx,dy=dy) for s in SIGMAS for r,dx,dy in SHIFTS]
MARGIN=26

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def desc(p):return dict(path=str(p),sha256=sha(p))
def verify(d):
    p=Path(d['path']);assert sha(p)==d['sha256'],'Changed source/input '+str(p);return p
def seal(v):return dict(v,record_sha256=hashlib.sha256(encoded(v)).hexdigest())
def checked(v):
    z=dict(v);s=z.pop('record_sha256');assert hashlib.sha256(encoded(z)).hexdigest()==s;return v
def read(p):
    p=Path(p)
    with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:return json.load(f)
def save(p,v,mutable=False,sealed=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=encoded(seal(v) if sealed else v)
    if p.suffix=='.gz':
        out=io.BytesIO()
        with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(data)
        data=out.getvalue()
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=str(p.parent))
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        if mutable:os.replace(tmp,str(p))
        else:os.link(tmp,str(p))
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def identity(pid=None):
    pid=os.getpid() if pid is None else pid;s=Path('/proc/%d/stat'%pid).read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,starttick=int(s[19]),ppid=int(s[1]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(') ',1)[1].split()[19]),affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid))
def inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a[0]),uuid=a[1],name=a[2],memory_MiB=int(a[3])) for a in [list(map(str.strip,x.split(','))) for x in out.strip().splitlines()]]
def request():
    q=checked(read(ROOT/'request.json'))
    for d in q['sources'].values():verify(d)
    verify(q['selection']);verify(q['prior_scientific_completion']);return q
def check_host(q,host):
    v=q['hosts'][host];cur=identity();assert cur['boot_id']==v['boot_id'] and cur['pid1_starttick']==v['pid1_starttick'],'Host/container changed'
    assert inventory()==v['gpus'],'GPU inventory changed';return cur
def no_failures():assert not list((ROOT/'failures').glob('*.json')),'Preserved experiment failure; stop without restart'
def clean(v):
    if isinstance(v,float):return v if math.isfinite(v) else None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
