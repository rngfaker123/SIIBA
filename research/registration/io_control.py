import os,json,gzip,hashlib,datetime,tempfile,io
from pathlib import Path
ROOT=Path('/study/siiba/experiments/registration_confound_controls_20261031_v1')
OLD=ROOT.parent/'model_translation_before_after_20261030_v1'
N=ROOT.parent/'pl_controls_original_uv_pipeline_20261031_v1'
PY=Path('/study/siiba/.venv_h200/bin/python')
MASKPY=ROOT.parent/'mi_atlas_six_20261025_v1/runtime_mask/bin/python'
MODELS=('unet_single','unet_multi','unetplusplus','swin','ldm');REGIONS=('P','D','O','F');METRICS=('L1','MSE','PSNR_dB','SSIM')
SIGMAS=(0.,.5,1.,2.,4.);RADII=(0,1,2,3,5)
SHIFTS=[(0,0)]+[(x,y) for r in RADII[1:] for x,y in [(r,0),(-r,0),(0,r),(0,-r)]]
GRID=sorted([(x,y) for x in range(-5,6) for y in range(-5,6)],key=lambda v:(v[0]*v[0]+v[1]*v[1],v[1],v[0]))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def desc(p):return dict(path=str(p),sha256=sha(p))
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def checked(v):
 v=dict(v);s=v.pop('record_sha256');assert hashlib.sha256(encoded(v)).hexdigest()==s;return v
def verify(d):p=Path(d['path']);assert sha(p)==d['sha256'],str(p);return p
def clean(v):
 import math
 if isinstance(v,float):return v if math.isfinite(v) else None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v
def save(p,v,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);v=clean(v);v['record_sha256']=hashlib.sha256(encoded(v)).hexdigest();data=encoded(v)
 if p.suffix=='.gz':
  b=io.BytesIO()
  with gzip.GzipFile(fileobj=b,mode='wb',mtime=0) as f:f.write(data)
  data=b.getvalue()
 fd,t=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data)
  if mutable:os.replace(t,str(p))
  else:os.link(t,str(p))
 finally:
  if os.path.exists(t):os.unlink(t)
def claim(p,v):
 try:save(p,v);return True
 except FileExistsError:return False
def identity(pid=None):
 pid=pid or os.getpid();p=Path('/proc/%d'%pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,starttick=int(s[19]),state=s[0],ppid=int(s[1]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),affinity=sorted(os.sched_getaffinity(pid)),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]))
def request():
 q=checked(read(ROOT/'request.json'))
 for d in list(q['sources'].values())+list(q['bindings'].values()):verify(d)
 return q
def no_failures():assert not list((ROOT/'failures').glob('*.json')),'Preserved control failure'
def record_pins(q):
 pins={}
 for v in read(verify(q['bindings']['translation_manifest']))['files']:
  for k in ['translated','zero_reused']:pins[v[k]['path']]=v[k]
 for v in read(verify(q['bindings']['common_blur_manifest']))['files']:pins[v['path']]=v
 return pins
def old_modules():
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parent/'translation'))
 import common,pair_engine
 assert common.ROOT==OLD
 return common,pair_engine
