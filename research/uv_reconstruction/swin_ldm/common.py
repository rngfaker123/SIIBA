"""Bound I/O and process primitives for the isolated autonomous secondary TEST."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

ROOT_NAME = 'test_resume_eight_gpu_20261028_v1'
ROOT = Path('/study/siiba/experiments') / ROOT_NAME
MODELS = ('swin', 'ldm')
COUNT = 21899
GPU_COUNT = 8
HOST_SLOTS = ('30663', '30965')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def encoded(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))

def desc(path):
    return {'path':Path(path).resolve().as_posix(),'sha256':sha(path)}

def bound(value):
    need(isinstance(value,dict) and set(value)=={'path','sha256'},'Invalid file descriptor')
    p=Path(value['path'])
    need(p.is_absolute() and p.is_file() and sha(p)==value['sha256'],'Bound file changed: '+str(p))
    return p

def seal(value):
    need('record_sha256' not in value,'Already sealed')
    return dict(value,record_sha256=digest(value))

def checked(value):
    need(value.get('record_sha256')==digest({k:v for k,v in value.items() if k!='record_sha256'}),'Record seal differs')
    return value

def save(path,value,mutable=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=encoded(value)
    if p.exists() and not mutable:
        need(p.read_bytes()==data,'Refuse differing immutable overwrite: '+str(p));return
    temp=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        if mutable:
            os.replace(str(temp),str(p))
        else:
            try:os.link(str(temp),str(p))
            except FileExistsError:need(p.read_bytes()==data,'Concurrent different immutable record')
    finally:
        if temp.exists():temp.unlink()

def load_verified(descriptor,name):
    p=bound(descriptor);raw=p.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'Source changed before load')
    spec=importlib.util.spec_from_file_location(name,str(p));need(spec is not None,'Cannot load source')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    try:exec(compile(raw,str(p),'exec'),m.__dict__)
    except BaseException:
        sys.modules.pop(name,None);raise
    return m

@contextmanager
def lock(root,name):
    import fcntl
    need(name and '/' not in name and '\\' not in name,'Invalid lock name')
    p=Path(root)/'locks'/(name+'.lock');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def identity(pid):
    p=Path('/proc')/str(pid);s=p.joinpath('stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'starttick':int(s[19]),'ppid':int(s[1]),'pgid':int(s[2]),'state':s[0],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'command':[] if pid==1 else [v.decode('utf8','replace') for v in p.joinpath('cmdline').read_bytes().split(b'\0') if v],
            'affinity':sorted(os.sched_getaffinity(pid)),'nice':int(s[16])}

def host_identity():
    """Only PID1 stat, never its command line or environment (which may contain secrets)."""
    fields=Path('/proc/1/stat').read_text().split(') ',1)[1].split()
    return {'hostname':socket.gethostname(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid1_starttick':int(fields[19])}

def expected_cpus(global_id):
    need(type(global_id) is int and global_id in range(GPU_COUNT),'Invalid global GPU ID')
    offset=global_id if global_id<7 else 0
    return [80+2*offset,81+2*offset]

def validate_inventory(configuration):
    """Pure topology audit; does not probe devices or initialize CUDA."""
    hosts=configuration['hosts']
    need(set(hosts)==set(HOST_SLOTS),'Exactly the two approved host slots required')
    for slot,value in hosts.items():
        need(set(value)=={'hostname','boot_id','pid1_starttick'} and
             isinstance(value['hostname'],str) and bool(value['hostname']) and
             type(value['pid1_starttick']) is int and value['pid1_starttick']>0,
             'Host identity incomplete: '+slot)
        need(str(uuid.UUID(value['boot_id']))==value['boot_id'],'Invalid host boot ID')
    need(hosts['30663']!=hosts['30965'],'Host slots must represent distinct approved runtimes')
    inventory=configuration['gpu_inventory']
    need(isinstance(inventory,list) and len(inventory)==GPU_COUNT,'Exactly eight physical GPUs required')
    for global_id,item in enumerate(inventory):
        need(set(item)=={'index','host','local_index','uuid','cpus'},'GPU descriptor schema differs')
        need(type(item['index']) is int and item['index']==global_id and
             item['host']==('30663' if global_id<7 else '30965') and
             type(item['local_index']) is int and item['local_index']==(global_id if global_id<7 else 0) and
             item['cpus']==expected_cpus(global_id),'Changed global/local GPU topology or CPU assignment')
        need(isinstance(item['uuid'],str) and item['uuid'].startswith('GPU-') and
             str(uuid.UUID(item['uuid'][4:]))==item['uuid'][4:],'Invalid physical GPU UUID')
    need(len({item['uuid'] for item in inventory})==GPU_COUNT,'Duplicate physical GPU ownership')
    return inventory

def validate_probe(configuration,global_id,actual_host,actual_inventory):
    """Pure runtime/topology comparison, used by tests and the real probe."""
    inventory=validate_inventory(configuration)
    need(type(global_id) is int and global_id in range(GPU_COUNT),'Invalid global GPU ID')
    item=inventory[global_id]
    need(actual_host==configuration['hosts'][item['host']],'Wrong host, boot or container for global GPU')
    expected=[{'index':v['local_index'],'uuid':v['uuid']} for v in inventory if v['host']==item['host']]
    need(actual_inventory==expected,'Physical local GPU inventory differs from approved host topology')
    return item

def runtime_probe(global_id,configuration=None,select_cuda=True):
    """Bind global ID to this exact host/local physical GPU without importing Torch."""
    if configuration is None:
        configuration=read(ROOT/'request.json')
    validate_inventory(configuration)
    actual_host=host_identity()
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    actual_inventory=[]
    for line in raw.splitlines():
        parts=[x.strip() for x in line.split(',')]
        need(len(parts)==2,'Unexpected nvidia-smi GPU inventory')
        actual_inventory.append({'index':int(parts[0]),'uuid':parts[1]})
    actual_inventory.sort(key=lambda value:value['index'])
    item=validate_probe(configuration,global_id,actual_host,actual_inventory)
    if select_cuda:
        os.environ['CUDA_VISIBLE_DEVICES']=item['uuid']
    return {'global_gpu_id':global_id,'host':item['host'],'physical_gpu_id':item['local_index'],
            'uuid':item['uuid'],'host_identity':actual_host,'local_gpu_inventory':actual_inventory,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}

def verify_sources(sources):
    need(isinstance(sources,dict) and sources,'Missing source inventory')
    for d in sources.values():bound(d)

def validate_ownership(value):
    expected=[{'gpu':g,'model':MODELS[g//4],'modulus':4,'remainder':g%4} for g in range(GPU_COUNT)]
    need(value==expected,'Changed or overlapping ownership')
    return expected

def load_manifest(root):
    root=Path(root).resolve();need(root.name==ROOT_NAME,'Wrong isolated execution root')
    m=checked(read(root/'manifest.json'))
    need(m['schema']=='secondary_eight_gpu_manifest_v1' and m['models']==list(MODELS)
         and m['sample_count']==COUNT and m['expected_record_count']==COUNT*2,'TEST scope differs')
    need(m['execution_root']==root.as_posix(),'Manifest root differs')
    validate_inventory(m)
    validate_ownership(m['ownership']);verify_sources(m['sources'])
    for key in ('catalog','original_combined_protocol','original_regional_protocol','original_stats','request','training_gate'):
        bound(m[key])
    request=read(m['request']['path'])
    need(request['schema']=='secondary_eight_gpu_request_v1' and request['hosts']==m['hosts']
         and request['gpu_inventory']==m['gpu_inventory'] and request['ownership']==m['ownership'],
         'Execution topology differs from the exact authorized request')
    gate=checked(read(m['training_gate']['path']))
    need(gate['status']=='ready' and gate['training_bundles']==m['training_bundles'],'Training not finally accepted')
    need(set(m['training_bundles'])==set(MODELS),'Training inventory differs')
    for bundle in m['training_bundles'].values():
        need(set(bundle)=={'source','config','binding','protocol','summary','best_checkpoint'},'Training bundle incomplete')
        # 加载时核对权重哈希
        for k,d in bundle.items():
            if k!='best_checkpoint':bound(d)
    return m
