"""Isolated CPU execution. No writes, signals, or scheduling changes to other jobs."""
import datetime,gzip,hashlib,importlib.util,io,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
E=Path('/study/siiba/experiments')
R=E/'conformal_cpu_full_20261031_v1'
N=E/'pl_controls_original_uv_pipeline_20261031_v1'
BASE=E/'signal_objects_3cd_20261029_v1'
EARLY=E/'conformal_task_early_20261031_v1'
PY=Path('/study/siiba/.venv_h200/bin/python')
MASKPY=E/'mi_atlas_six_20261025_v1/runtime_mask/bin/python'
MODELS=('unet_single','unet_multi','unetplusplus','swin','ldm')
REGIONS=('P','D','O','F')
C=N/'conformal'
CPUS={'unet_single':[72,73,74,75],'unet_multi':[84,85,86,87],'unetplusplus':[88,89,90,91],'swin':[112,113,114,115],'ldm':[116,117,118,119]}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def desc(p):return dict(path=str(p),sha256=sha(p))
def verify(d):
 p=Path(d['path']);assert sha(p)==d['sha256'],str(p);return p
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:return json.load(f)
def checked(v):assert v['record_sha256']==digest({k:x for k,x in v.items() if k!='record_sha256'});return v
def save(p,v,mutable=False):
 p=Path(p);assert R in p.parents or C in p.parents,'Only owned CPU metadata and the Conformal task outputs are writable';p.parent.mkdir(parents=True,exist_ok=True)
 z=dict(v);z['record_sha256']=digest(z);data=encoded(z)
 if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
 fd,t=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.tmp.')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  if mutable:os.replace(t,str(p))
  else:os.link(t,str(p))
 finally:
  if os.path.exists(t):os.unlink(t)
def identity(pid=None):
 pid=os.getpid() if pid is None else pid;a=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,starttick=int(a[19]),ppid=int(a[1]),state=a[0],cpus=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]))
def module(p,name):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def no_failures():assert not list((R/'failures').glob('*.json')),'Owned failure preserved; no automatic restart'
def context():
 q=checked(read(R/'request.json'));cur=identity();assert all(cur[k]==q['host'][k] for k in ('boot','pid1_starttick'))
 for d in q['sources'].values():verify(d)
 for d in q['bindings'].values():verify(d)
 no_failures();return q
def env():return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='1')
def progress(name,phase,**kwargs):save(R/('progress/'+name+'.json'),dict(time=now(),phase=phase,**kwargs),True)
def core_record(mid,index):return BASE/('records/'+mid+'/%06d.json'%index)
def record_path(stage,mid,index):return C/(stage+'/records/'+mid+'/%06d.json.gz'%index)
