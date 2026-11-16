"""One-shot server-side PL test owner. Existing training retains priority."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

from pl_execution import ROOT, EXP, TASKS, check_source, desc, need, now, publish, read, sha, task_targets
import pl_models


def identity(pid):
    fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttick=int(fields[19]), pgid=int(fields[2]), state=fields[0])


def check_host(p):
    need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == p['boot'] and
         identity(1)['starttick'] == p['pid1_start'], 'Side execution container changed')
    # 解释器共用 inode
    need(Path(sys.prefix) == Path(p['python']).parent.parent, 'Wrong Python environment')
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    cards = [r.split(', ') for r in gpu_lines]
    need([r[1].strip() for r in cards] == p['device_uuids'], 'Side GPU identity/order differs')
    return [int(r[2]) for r in cards]


def training_has_released(p):
    marker = EXP/'pl_side_resume_20261104_v1/supervisor_completion.json'
    handback = EXP/'mi_gpu_priority_20261105_v2/training_returned_to_original_auditor.json'
    if not (marker.is_file() and handback.is_file()):
        return False
    accepted = EXP/'pl_side_resume_20261104_v1/independent_final_acceptance.json'
    item = read(marker)
    need(item['status'] == 'complete_original_PLH_training' and
         item['acceptance']['sha256'] == sha(accepted) and Path(item['acceptance']['path']) == accepted and
         read(accepted)['status'] == 'passed', 'Invalid native training release')
    for step in range(2):
        memory = check_host(p)
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader'],text=True).strip()
        if apps or max(memory) > 300:
            return False
        if step == 0:
            time.sleep(3)
    return True


def previous_test_guard():
    matched = EXP/'pl_controls_original_uv_pipeline_20261031_v1/matched_test'
    for target in ('plh','plv'):
        rec = matched/'records'/target
        need(not rec.exists() or not any(rec.glob('*.json')), 'Other PL TEST outputs exist: '+target)
    for cmdfile in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = cmdfile.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(a.endswith((b'/matched_phase.py', b'/matched_phase_v2.py')) for a in args):
            need(not any(a in (b'test',b'worker',b'evaluate',b'infer') for a in args),
                 'An earlier PL test worker is active')


def child(p, label, args, cores, device=''):
    logfile = ROOT/'logs'/(label+'.log')
    logfile.parent.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=device, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    cmd = ['taskset','-c',cores,p['python'],'-B','-u']+list(map(str,args))
    with logfile.open('xb') as log:
        proc = subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,
                                stderr=subprocess.STDOUT,close_fds=True)
    receipt = dict(time=now(), identity=identity(proc.pid), argv=cmd, device_uuid=device,
                   log=str(logfile), protocol_sha256=sha(ROOT/'protocol.json'))
    publish(ROOT/'children'/(label+'.json'),receipt)
    return proc


def wait_jobs(jobs):
    failed = [(key,code) for key,proc in jobs for code in [proc.wait()] if code]
    need(not failed,'Child process failure(s): '+str(failed))


def run_one(p,name,args,cores='96,97'):
    wait_jobs([(name,child(p,name,args,cores))])


def cuda_phase(p,stage,arch,key):
    jobs=[]
    try:
        for lane,uuid in enumerate(p['device_uuids']):
            label='%s_%s_%s_%d'%(stage,arch,key,lane)
            args=[ROOT/'pl_execution.py',stage,arch,key,str(lane)]
            jobs.append((label,child(p,label,args,'%d,%d'%(96+2*lane,97+2*lane),uuid)))
    finally:
        wait_jobs(jobs)


def gpu_job(p,arch,key):
    name=arch+'_'+key
    targets=task_targets(key)
    previous_test_guard()
    cuda_phase(p,'gate',arch,key)
    pilots=[desc(ROOT/'pilots'/name/(str(lane)+'.json')) for lane in range(3)]
    cpu=read(ROOT/'cpu_reference'/name/'acceptance.json')
    for lane,record in enumerate(pilots):
        v=read(record['path'])
        need(v['status']=='passed' and v['model_identity_sha256']==cpu['model_identity_sha256'] and
             v['protocol_sha256']==sha(ROOT/'protocol.json') and v['device_uuid']==p['device_uuids'][lane],
             'Device validation record inconsistent')
    publish(ROOT/'bindings'/(name+'.json'),dict(time=now(),architecture=arch,targets=targets,
            model_identity=pl_models.model_identity(arch,key),model_identity_sha256=cpu['model_identity_sha256'],
            protocol_sha256=sha(ROOT/'protocol.json'),pilot_acceptances=pilots,
            cpu_reference=desc(ROOT/'cpu_reference'/name/'acceptance.json')))
    cuda_phase(p,'worker',arch,key)
    for target in targets:
        run_one(p,'collect_'+arch+'_'+target,[ROOT/'pl_collect.py','pair','--root',ROOT,
                '--architecture',arch,'--target',target])
        need(read(ROOT/'reports'/(arch+'_'+target)/'acceptance.json')['status']=='passed','Pair acceptance missing')
        run_one(p,'tables_'+arch+'_'+target,[ROOT/'pl_collect.py','tables','--root',ROOT])
        try:
            run_one(p,'contrast_'+arch+'_'+target,[ROOT/'pl_uv_contrasts.py','--root',ROOT,
                    '--architecture',arch,'--target',target])
        except Exception:
            publish(ROOT/'reports'/(arch+'_'+target)/'contrast_unavailable.json',
                    dict(time=now(),trace=traceback.format_exc()))
    publish(ROOT/'finished'/(name+'.json'),dict(time=now(),reports=[
        desc(ROOT/'reports'/(arch+'_'+t)/'acceptance.json') for t in targets]))


def status(phase, **extra):
    publish(ROOT/'status.json',dict(time=now(),phase=phase,**extra),True)


def serve(p,nonce):
    reservation=read(ROOT/'launch_reservation.json')
    need(nonce==reservation['nonce'] and reservation['protocol_sha256']==sha(ROOT/'protocol.json'),
         'Wrong launch reservation')
    lock=(ROOT/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_host(p)
    publish(ROOT/'owner_started.json',dict(time=now(),reservation=desc(ROOT/'launch_reservation.json'),
            identity=identity(os.getpid()),boot=p['boot'],pid1_start=p['pid1_start']))
    run_one(p,'initial_tables',[ROOT/'pl_collect.py','tables','--root',ROOT])
    while True:
        done=[]; failed=[]; cpu=[]; waiting=[]
        for arch,key in TASKS:
            name=arch+'_'+key
            if (ROOT/'finished'/(name+'.json')).exists():
                done.append(name); continue
            if (ROOT/'failure'/(name+'.json')).exists():
                failed.append(name); continue
            try:
                if not (ROOT/'cpu_reference'/name/'acceptance.json').exists():
                    if pl_models.probe_ready(arch,key) is None:
                        waiting.append(name); continue
                    status('cpu_validation',model=name,complete=done)
                    run_one(p,'cpu_'+name,[ROOT/'pl_execution.py','cpupilot',arch,key,'0'])
                cpu.append(name)
                if training_has_released(p):
                    status('gpu_validation_then_test',model=name)
                    gpu_job(p,arch,key)
                    done.append(name)
            except Exception:
                publish(ROOT/'failure'/(name+'.json'),dict(time=now(),model=name,trace=traceback.format_exc()))
                failed.append(name)
        if len(done)+len(failed)==len(TASKS):
            final=dict(time=now(),complete=done,failed=failed,status='partial_failure' if failed else 'complete')
            publish(ROOT/'supervisor_completion.json',final)
            status(final['status'],complete=done,failed=failed)
            return
        status('waiting_for_native_training_or_GPU_release',cpu_validated=cpu,complete=done,
               failed=failed,pending_training=waiting)
        time.sleep(60)


def launch(p):
    check_host(p)
    previous_test_guard()
    nonce=uuid4().hex
    publish(ROOT/'launch_reservation.json',dict(time=now(),nonce=nonce,boot=p['boot'],
            pid1_start=p['pid1_start'],protocol_sha256=sha(ROOT/'protocol.json')))
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    with (ROOT/'owner.log').open('xb') as log:
        proc=subprocess.Popen(['taskset','-c','94,95','nice','-n','12',p['python'],'-B','-u',
                              str(ROOT/'pl_owner.py'),'supervise',nonce],
                              env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,close_fds=True,cwd=str(ROOT))
    for _ in range(100):
        if (ROOT/'owner_started.json').exists():
            print(read(ROOT/'owner_started.json'),flush=True); return
        need(proc.poll() is None,'Owner failed to initialize; inspect owner.log')
        time.sleep(0.1)
    raise RuntimeError('No startup handshake; retain reservation, inspect owner.log')


if __name__=='__main__':
    policy=check_source()
    if sys.argv[1:] == ['launch']:
        launch(policy)
    elif len(sys.argv)==3 and sys.argv[1]=='supervise':
        try:
            serve(policy,sys.argv[2])
        except BaseException:
            publish(ROOT/'supervisor_failure.json',dict(time=now(),trace=traceback.format_exc()))
            raise
    else:
        raise ValueError('Use launch or the reserved internal supervise invocation')
