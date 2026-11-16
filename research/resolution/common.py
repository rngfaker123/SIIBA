"""Isolated, versioned common-cohort MI execution; never writes other roots."""
from pathlib import Path
import functools,hashlib,importlib.util,json,os,subprocess,sys,time
import numpy as np
from PIL import Image

E=Path('/study/siiba/experiments')
ROOT=E/'multimodal_native_resolution_allpairs_common1000_20261031_v1'
OLD=E/'multimodal_resolution_native_mi_20261030_v3'
I=E/'ita_stratified_mi_test_20261026_v1'
P=E/'three_region_test_parallel_20261025_v1'
DETECT=E/'signal_objects_3cd_20261029_v1'
TOOLS=DETECT/'frozen/tools';VENDOR=TOOLS/'porphyrin_module.py';CORE=DETECT/'source/detector.py'
CORE_SHA='c6fb5798e453c0c804180403183d991b9bc2de3ac427f1cf1217003cf967a233'
PY=E/'mi_atlas_six_20261025_v1/runtime_mask/bin/python'
FACTORS=(1,2,4,8);BUDGETS=(100,500,1000);SEEDS=(20261025,20261026,20261027)
DOMAINS=('F','P','D','O');MODS=('rgb','uv','plv','plh')
PAIRS=(('rgb','uv'),('rgb','plh'),('rgb','plv'),('plh','plv'),('uv','plh'),('uv','plv'))

def now():
 import datetime
 return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def desc(p):return dict(path=str(p),sha256=sha(p))
def verify(d):
 p=Path(d['path']);assert p.is_absolute() and sha(p)==d['sha256'],str(p)
 return p
def checked(v):
 assert v['record_sha256']==digest({k:x for k,x in v.items() if k!='record_sha256'}),'New compact seal mismatch'
 return v
def legacy(v):
 b=(json.dumps({k:x for k,x in v.items() if k!='record_sha256'},ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 assert v['record_sha256']==hashlib.sha256(b).hexdigest(),'Original indent2 seal mismatch'
 return v
def save(p,v,replace=False):
 p=Path(p);assert ROOT==p or ROOT in p.parents,'Refuse write outside this experiment'
 p.parent.mkdir(parents=True,exist_ok=True);v=dict(v);v['record_sha256']=digest(v)
 if replace:
  tmp=p.with_name(p.name+'.tmp.%d'%os.getpid())
  with tmp.open('xb') as f:f.write(encoded(v))
  os.replace(tmp,p)
 else:
  with p.open('xb') as f:f.write(encoded(v))
 return v
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def identity(pid=None):
 pid=os.getpid() if pid is None else int(pid);p=Path('/proc')/str(pid);a=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=int(a[1]),starttick=int(a[19]),state=a[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int((Path('/proc/1/stat').read_text().rsplit(')',1)[1].split())[19]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cpus=sorted(os.sched_getaffinity(pid)))
def inventory():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(a[0]),uuid=a[1],pci=a[2],name=a[3],memory_MiB=int(a[4])) for a in ([x.strip() for x in line.split(',')] for line in s.strip().splitlines())]
def gpu_processes():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
def no_failures():assert not list((ROOT/'failures').glob('*.json')),'Preserved failure; no automatic restart'
def state(**v):save(ROOT/'status.json',dict(time=now(),**v),True)
def check_host(q):
 x=identity();assert (x['boot_id'],x['pid1_starttick'])==(q['host']['boot_id'],q['host']['pid1_starttick']);assert inventory()==q['host']['gpus']
def lower(im,f):return im.copy() if f==1 else im.resize((im.width//f,im.height//f),Image.Resampling.LANCZOS)
def binary_project(a,shape):
 h,w=map(int,shape)
 return a.copy() if a.shape==(h,w) else np.asarray(Image.fromarray(a.astype(np.float32)).resize((w,h),Image.Resampling.BOX))>=.5
def sampler(mask,sid,factor,domain):
 candidates=np.flatnonzero(mask.ravel());assert len(candidates)>=1000
 out=[]
 for seed in SEEDS:
  payload=json.dumps(['native_resolution_sampling_v1',seed,sid,factor,domain],separators=(',',':')).encode()
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16],'big'))
  out.append(candidates[rng.choice(len(candidates),1000,replace=False)].astype(np.uint64))
 return np.stack(out)
@functools.lru_cache(None)
def context():
 q=checked(read(ROOT/'request.json'))
 for d in q['sources'].values():verify(d)
 for d in q['bindings'].values():verify(d)
 old=checked(read(verify(q['bindings']['old_request'])))
 runner_path=verify(old['original_runner']);sys.path.insert(0,str(runner_path.parent))
 runner=module(runner_path,'common1000_original_runner');est=module(verify(old['original_estimator']),'common1000_original_estimator')
 cat=read(verify(q['bindings']['catalog']));assert len(cat['rows'])==21899
 assert len({r['sample_id'] for r in cat['rows']})==21899
 oldmanifest=read(verify(q['bindings']['old_table1_input']))
 cached={r['sample_id']:(j,r) for j,r in enumerate(oldmanifest['rows'])}
 return q,runner,est,cat,cached
def source_row(cat,index):
 row=cat['rows'][index];out=dict(sample_id=row['sample_id'],subject_id=row['subject_id'],native_shape=row['native_shape'])
 for m in MODS:
  p=Path(row[m]['path']);out[m]=dict(path=str(p if p.is_absolute() else Path(cat['data_root'])/p),sha256=row[m]['sha256'])
 return out
def candidate_path(index):return ROOT/'screening'/('%06d.json'%index)
def group_path(index):return ROOT/'features'/('%06d'%index)/'receipt.json'
def pool_name(f,seed,d,m):return 'factor%d_seed%d_%s_%s.npy'%(f,seed,d,m)
def task_key(t):return '%s_%s_%s_factor%d_n%d_seed%d'%(t['x'],t['y'],t['domain'],t['factor'],t['budget'],t['seed'])
def storage_check():
 size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
 assert size<=24*1024**3,'New experiment 24GiB storage cap exceeded; preserve and fail closed'
 return size
