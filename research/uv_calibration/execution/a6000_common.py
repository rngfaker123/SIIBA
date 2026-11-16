"""Execution amendment only: preserve accepted CPU records, finish missing Conformal on CUDA."""
import datetime,gzip,hashlib,importlib.util,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
E=Path('/study/siiba/experiments')
A=E/'nontraining_a6000_finish_20261101_v5'
R=E/'conformal_cpu_full_20261031_v1'
N=E/'pl_controls_original_uv_pipeline_20261031_v1'
C=N/'conformal'
BASE=E/'signal_objects_3cd_20261029_v1'
PY=Path('/study/siiba/.venv_h200/bin/python')
MODELS=('unet_single','unet_multi','unetplusplus','swin','ldm')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def desc(p):return dict(path=str(p),sha256=sha(p))
def verify(d):
 p=Path(d['path']);assert sha(p)==d['sha256'],'Changed binding '+str(p);return p
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:return json.load(f)
def checked(v):assert v['record_sha256']==digest({k:x for k,x in v.items() if k!='record_sha256'});return v
def save(p,v,mutable=False):
 p=Path(p);assert A in p.parents or C in p.parents,'Owned execution or Conformal outputs only'
 p.parent.mkdir(parents=True,exist_ok=True);z=dict(v);z['record_sha256']=digest(z);data=encoded(z)
 if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
 fd,tmp=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.tmp.')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  if mutable:os.replace(tmp,str(p))
  else:os.link(tmp,str(p))
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def identity(pid=None):
 pid=os.getpid() if pid is None else pid;a=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,starttick=int(a[19]),ppid=int(a[1]),state=a[0],cpus=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]))
def module(p,name):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def inventory():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(z[0]),uuid=z[1],name=z[2],memory_MiB=int(z[3])) for z in [list(map(str.strip,l.split(','))) for l in out.strip().splitlines()]]
def no_failures():
 assert not list((A/'failures').glob('*.json')),'Owned failure preserved; no automatic restart'
 assert not list((R/'failures').glob('*.json')),'Original Conformal failure preserved'
def context(host=True):
 q=checked(read(A/'request.json'));p=checked(read(verify(q['CPU_request'])))
 for d in q['sources'].values():verify(d)
 for d in q['science_sources'].values():verify(d)
 for d in p['sources'].values():verify(d)
 for d in p['bindings'].values():verify(d)
 if host:
  cur=identity();assert all(cur[k]==q['host'][k] for k in ('boot','pid1_starttick'))
  assert inventory()==q['gpus']
 no_failures();return q,p
def record_path(stage,mid,index):return C/(stage+'/records/'+mid+'/%06d.json.gz'%index)
def env(uuid=None):
 z=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 if uuid is not None:z['CUDA_VISIBLE_DEVICES']=uuid
 return z
def progress(name,phase,**kw):save(A/('progress/'+name+'.json'),dict(time=now(),phase=phase,identity=identity(),**kw),True)
def pending_keys(split,stage,preserved):
 return [(m,i) for i in split['calibration_indices' if stage=='cal' else 'evaluation_indices'] for m in MODELS if str(record_path(stage,m,i)) not in preserved]
def compare(a,b,path=''):
 """Structural/finite/domain validation; actual finite numeric differences are diagnostic."""
 out=[]
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
  for k in a:out.extend(compare(a[k],b[k],path+'/'+k))
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):out.extend(compare(x,y,path+'/'+str(i)))
 elif a is None or isinstance(a,(str,bool)):assert a==b,(path,a,b)
 else:
  assert isinstance(b,(float,int)) and math.isfinite(float(a)) and math.isfinite(float(b)),path
  out.append(dict(metric=path,CPU=float(a),CUDA=float(b),absolute_difference=abs(float(a)-float(b))))
 return out
