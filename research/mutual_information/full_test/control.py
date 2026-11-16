"""Exclusive MI GPU handoff and native safe-boundary training pause/resume.

No source/config/scientific formula edits. No duplicate original queue launch.
The original CPU MI results remain immutable; completion is owned by this root.
"""
from pathlib import Path
import os,sys,json,time,hashlib,datetime,subprocess,signal,traceback,secrets,importlib.util,shutil,ctypes
try:import fcntl
except ImportError:fcntl=None  # Windows 仅离线测试
R=Path(__file__).resolve().parent
B=Path('/study/siiba/experiments')
M=B/'mi_table1_fulltest_pixelstd_20261104_v2'
T=B/'pl_side_resume_20261104_v1'
TRAIN=B/'pl_training_autostart_20261101_v1/training/unetplusplus_plh'
PY=B.parent/'.venv_h200/bin/python'
MI_PY=B/'mi_atlas_six_20261025_v1/runtime_mask/bin/python'
sys.dont_write_bytecode=True
def need(v,msg):
    if not v:raise RuntimeError(msg)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def enc(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8388608),b''):h.update(x)
    return h.hexdigest()
def desc(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(v):
    c=dict(v);s=c.pop('record_sha256');need(s==hashlib.sha256(enc(c)).hexdigest(),'Seal changed');return v
def put(p,v,mutable=False):
    p=Path(p);need(R.resolve() in p.resolve().parents,'Writes confined to new root');p.parent.mkdir(parents=True,exist_ok=True)
    v=dict(v,record_sha256=hashlib.sha256(enc(v)).hexdigest());tmp=p.with_name(p.name+'.tmp.'+secrets.token_hex(8))
    with tmp.open('xb') as f:f.write(enc(v));f.flush();os.fsync(f.fileno())
    try:
        if mutable:os.replace(tmp,p)
        else:os.link(tmp,p)
    finally:tmp.unlink(missing_ok=True)
def status(phase,**kw):put(R/'status.json',dict(time=now(),phase=phase,identity=identity(),**kw),True)
def identity(pid=None):
    pid=pid or os.getpid();v=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttick=int(v[19]),state=v[0],ppid=int(v[1]),cpus=sorted(os.sched_getaffinity(pid)),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')',1)[1].split()[19]))
def live(i):
    try:
        a=identity(i['pid']);return all(a[k]==i[k] for k in ('pid','starttick','boot','pid1_starttick')) and a['state']!='Z'
    except FileNotFoundError:return False
def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    need(os.uname().machine=='x86_64','Unsupported syscall architecture')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=libc.syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return int(fd)
def pidfd_signal(fd,sig):
    if hasattr(signal,'pidfd_send_signal'):return signal.pidfd_send_signal(fd,sig)
    need(os.uname().machine=='x86_64','Unsupported syscall architecture')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    ret=libc.syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
    if ret<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
def send(i,sig):
    need(live(i),'Process birth no longer matches '+str(i['pid']));fd=pidfd_open(i['pid'])
    try:
        need(live(i),'Process changed while opening pidfd');pidfd_signal(fd,sig)
    finally:os.close(fd)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def apps():
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    return [dict(pid=int(a),uuid=b.strip()) for a,b in (x.split(',') for x in s.strip().splitlines())]
def req():
    q=checked(read(R/'request.json'));a=identity()
    need(all(a[k]==q['host'][k] for k in ('boot','pid1_starttick')),'Container changed; no automatic recovery')
    for x in q['sources']+q['bindings']:need(sha(x['path'])==x['sha256'],'Frozen source/binding changed '+x['path'])
    return q
def training_env(q):
    e=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(q['rank_uuids']),SIIBA_PL_TARGET='plh',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(T/'source')+os.pathsep+str(B/'pl_training_autostart_20261101_v1/source'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','SIIBA_PL_ROOT_BINDING','SIIBA_GUARD_RECOVERY'):e.pop(k,None)
    return e
def stop_cpu_mi(q):
    items=[q['mi_owner']]+q['mi_lanes']
    need(all(live(i) for i in items),'Original MI ownership changed')
    put(R/'mi_pause_intent.json',dict(time=now(),identities=items))
    for i in items:send(i,signal.SIGSTOP)
    for _ in range(100):
        if all(identity(i['pid'])['state']=='T' for i in items):break
        time.sleep(.1)
    need(all(identity(i['pid'])['state']=='T' for i in items),'MI not fully parked')
    raw=[]
    for p in sorted((M/'raw').glob('*.json')):
        v=checked(read(p));claim=checked(read(M/'claims'/p.name))
        need(v['identity']==claim['identity'] and v['task']==claim['task'],'Original claim differs')
        raw.append(desc(p))
    put(R/'mi_handoff.json',dict(time=now(),identities=items,completed=raw,N_completed=len(raw),active_claims=[desc(p) for p in (M/'claims').glob('*.json') if not (M/'raw'/p.name).exists()],original_request=desc(M/'request.json')))
def restore_cpu_mi(q):
    if (R/'mi_retired.json').exists():return
    for i in q['mi_lanes']+[q['mi_owner']]:
        if live(i) and identity(i['pid'])['state']=='T':send(i,signal.SIGCONT)
    if not (R/'mi_restored.json').exists():put(R/'mi_restored.json',dict(time=now(),original_CPU_queue_resumed=True))
def retire_cpu_mi(q):
    need((R/'gpu_gate.json').exists(),'GPU gate absent')
    for i in q['mi_lanes']+[q['mi_owner']]:
        need(live(i) and identity(i['pid'])['state']=='T','CPU MI owner moved')
    put(R/'mi_retirement_intent.json',dict(time=now(),reason='Explicit user GPU-priority handoff; old results retained; uncompleted tasks reassigned once'))
    for i in q['mi_lanes']+[q['mi_owner']]:send(i,signal.SIGKILL)
    put(R/'mi_retired.json',dict(time=now(),identities=q['mi_lanes']+[q['mi_owner']],historical_owner_not_scientifically_complete=True))
def pause_training(q):
    need(all(live(i) for i in [q['training_owner'],q['torchrun']]+q['ranks']),'Training identity changed')
    need(read(TRAIN/'training_progress.json')['status']=='training','Wait for native training phase, do not interrupt VAL')
    need({a['pid'] for a in apps()}=={i['pid'] for i in q['ranks']},'Unowned GPU computation present')
    put(R/'training_pause_intent.json',dict(time=now(),owner=q['training_owner'],torchrun=q['torchrun'],ranks=q['ranks'],native_signal='SIGTERM rank0 -> any_rank_flag -> full checkpoint -> clean return'))
    send(q['training_owner'],signal.SIGSTOP)
    for _ in range(50):
        if identity(q['training_owner']['pid'])['state']=='T':break
        time.sleep(.1)
    need(identity(q['training_owner']['pid'])['state']=='T','Supervisor not parked')
    send(q['ranks'][0],signal.SIGTERM)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if all(not live(i) for i in q['ranks']) and not live(q['torchrun']) and not apps():break
        time.sleep(2)
    need(all(not live(i) for i in q['ranks']) and not apps(),'Native safe stop did not release GPU; no force kill')
    need(read(TRAIN/'training_progress.json')['status']=='stopped_at_safe_boundary','No native safe-boundary marker')
    check=subprocess.run([str(PY),'-B',str(R/'check_training.py')],env=training_env(q),capture_output=True,text=True,timeout=240)
    need(check.returncode==0,'Native checkpoint audit failed: '+check.stderr[-2000:])
    v=json.loads(check.stdout.strip().splitlines()[-1]);need(v['status']=='passed','Checkpoint gate failed')
    preserved=[]
    for n in ['checkpoint_last.pt','checkpoint_best.pt','checkpoint_commit.json','history.json','training_progress.json']:
        src=TRAIN/n;dst=R/'checkpoint_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b,8388608)
        need(sha(src)==sha(dst),'Snapshot differs');preserved.append(dict(original=desc(src),preserved=desc(dst)))
    put(R/'training_paused.json',dict(time=now(),checkpoint_audit=v,preserved=preserved,original_owner=q['training_owner'],all_GPU_resources_released=True))
def resume_training(wait=True):
    q=req();lock=(R/'training_resume.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    try:
        if not (R/'training_pause_intent.json').exists():return
        receipt=R/'training_resume_receipt.json'
        if receipt.exists():
            r=checked(read(receipt));ident=r['identity'];child=None
            if not live(ident):
                need((TRAIN/'training_summary.json').exists(),'Resumed trainer absent without summary; no blind restart')
        else:
            if not (R/'training_paused.json').exists():
                need(live(q['torchrun']),'Pause incomplete and original trainer absent; diagnose before resume')
                if live(q['training_owner']) and identity(q['training_owner']['pid'])['state']=='T':send(q['training_owner'],signal.SIGCONT)
                return
            p=checked(read(R/'training_paused.json'));a=p['checkpoint_audit']
            for d in p['preserved']:
                need(sha(d['original']['path'])==d['original']['sha256']==sha(d['preserved']['path']),'Checkpoint changed before resume')
            need(not apps(),'GPU resources not released for original training')
            need(not (R/'training_resume_reservation.json').exists(),'Reserved resume lacks receipt; diagnose')
            put(R/'training_resume_reservation.json',dict(time=now(),checkpoint=a['LAST'],command=a['command']))
            log=(R/'training_resume.log').open('xb')
            child=subprocess.Popen(a['command'],env=training_env(q),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,a['training_cpus']))
            log.close();ident=identity(child.pid)
            put(receipt,dict(time=now(),identity=ident,command=a['command'],checkpoint=a['LAST'],same_rank_uuids=q['rank_uuids'],native_resume_auto=True))
        if not wait:return
        while live(ident):
            status('original_training_resumed',training_progress=read(TRAIN/'training_progress.json'),child=ident,MI_complete=(R/'completion.json').exists())
            if child is not None and child.poll() is not None:break
            time.sleep(20)
        if child is not None:need(child.wait()==0,'Resumed original training failed; evidence preserved')
        need((TRAIN/'training_summary.json').exists(),'Training ended without full summary')
        need(not apps(),'Training resources not released')
        if not (R/'training_returned_to_original_auditor.json').exists():
            need(live(q['training_owner']),'Original final auditor absent')
            send(q['training_owner'],signal.SIGCONT)
            put(R/'training_returned_to_original_auditor.json',dict(time=now(),owner=q['training_owner'],additional_resume=desc(receipt)))
        status('training_complete_original_audit_running',MI_complete=(R/'completion.json').exists())
    finally:lock.close()
def cleanup_workers():
    for p in (R/'workers').glob('*_birth.json'):
        i=checked(read(p))['identity']
        if live(i):send(i,signal.SIGTERM)
    for _ in range(150):
        ids=[checked(read(p))['identity'] for p in (R/'workers').glob('*_birth.json')]
        if not any(live(i) for i in ids):return
        time.sleep(1)
    raise RuntimeError('MI child did not exit; preserve GPU ownership, do not collide with training')
def guardian(main):
    q=req()
    while live(main):time.sleep(10)
    if (R/'training_returned_to_original_auditor.json').exists():return
    try:
        cleanup_workers()
        if (R/'mi_pause_intent.json').exists() and not (R/'mi_retirement_intent.json').exists():restore_cpu_mi(q)
        resume_training(True)
    except BaseException:put(R/'guardian_failure.json',dict(time=now(),traceback=traceback.format_exc()))
def run():
    q=req();os.sched_setaffinity(0,{74,75});os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    lock=(R/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(R/'owner_started.json',dict(time=now(),identity=identity()))
    log=(R/'guardian.log').open('xb');g=subprocess.Popen([str(MI_PY),'-B',str(__file__),'guardian',json.dumps(identity())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    put(R/'guardian_receipt.json',dict(time=now(),identity=identity(g.pid)))
    try:
        stop_cpu_mi(q)
        worker=module(R/'mi_work.py','mi_priority_work');worker.prepare(q)
        status('waiting_native_train_boundary')
        while read(TRAIN/'training_progress.json')['status']!='training':
            need(live(q['torchrun']),'Training ended before pause; inspect');time.sleep(10)
        pause_training(q)
        worker.validate(q)
        retire_cpu_mi(q)
        worker.production(q)
    except BaseException:
        put(R/'failure.json',dict(time=now(),traceback=traceback.format_exc()))
        cleanup_workers()
        if not (R/'mi_retirement_intent.json').exists():restore_cpu_mi(q)
    finally:
        resume_training(True)
def configure():
    need(not (R/'request.json').exists(),'Already configured')
    fd=pidfd_open(os.getpid())
    try:pidfd_signal(fd,0)
    finally:os.close(fd)
    need(sha(B/'ksg_stability_20261030_v2/source/ksg_stability_cuda_v2.py')=='e0c2eb1bd73d9712161e402fe6658f63b3dd113751f2215a5a46990a852aa547','Accepted GPU backend changed')
    need(sha(B/'ksg_stability_20261030_v2/source/ksg_stability_kernels_v1.cu')=='6fb580b4ded08cc5b0cec3afa1df000844c02db9758eefdfc2a49283f0cb64c9','Accepted GPU kernels changed')
    old=checked(read(T/'execution_spec.json')) if 'record_sha256' in read(T/'execution_spec.json') else read(T/'execution_spec.json')
    mi_owner=checked(read(M/'owner_started.json'))['identity'];lanes=[checked(read(M/'lanes'/('%d_started.json'%j)))['identity'] for j in range(6)]
    tr=checked(read(T/'launches/train_receipt.json'))['identity'];owner=checked(read(T/'owner_started.json'))['identity']
    rank_specs=[(571,624285911),(572,624285912),(573,624285914)];ranks=[]
    for pid,birth in rank_specs:
        i=identity(pid);need(i['starttick']==birth and i['ppid']==tr['pid'],'Rank birth mismatch');ranks.append(i)
    need(all(live(i) for i in [mi_owner,owner,tr]+lanes+ranks),'Owners not live')
    need(mi_owner['pid']==1961 and mi_owner['starttick']==624347278,'Unexpected MI owner')
    need(owner['pid']==264 and owner['starttick']==624278959,'Unexpected training owner')
    need(not list((M/'failures').glob('*.json')) and not list((T/'failures').glob('*.json')),'Existing failure')
    bind=[M/'request.json',M/'owner.py',M/'protocol.json',M/'baseline_pins.json',M/'resources.json',M/'prepared.json',T/'request.json',T/'execution_spec.json',T/'preflight_acceptance.json',B/'ksg_stability_20261030_v2/source/ksg_stability_cuda_v2.py',B/'ksg_stability_20261030_v2/source/ksg_stability_kernels_v1.cu']
    put(R/'request.json',dict(time=now(),authorization='User explicitly requested pause one training task, accelerate this MI with GPUs, then resume original training; 2026-11-05',host=identity(),mi_owner=mi_owner,mi_lanes=lanes,training_owner=owner,torchrun=tr,ranks=ranks,rank_uuids=old['rank_uuids'],sources=[desc(p) for p in sorted(R.glob('*.py'))],bindings=[desc(p) for p in bind],science='Same 396 task keys, all eligible19651, original pixel seeds, fixed group subsets, original FP64 KSG1; exact CUDA tree query only',GPU_lanes=3,CPU_lanes=3,no_desktop_automation=True))
    print('CONFIGURED_NO_TASK_INTERRUPTED')
def launch():
    req();put(R/'launch_reservation.json',dict(time=now(),one_shot=True))
    log=(R/'owner.log').open('xb');c=subprocess.Popen([str(MI_PY),'-B','-u',str(__file__),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    put(R/'launch_receipt.json',dict(time=now(),identity=identity(c.pid),detached=True));print(json.dumps(dict(launched=True,pid=c.pid)))
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='guardian':guardian(json.loads(sys.argv[2]))
    else:{'configure':configure,'launch':launch,'run':run,'resume':resume_training}[mode]()
