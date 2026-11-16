"""Detached two-host, eight-GPU execution owner for the frozen contextual study.

No SSH, credential, training restart or retry logic. All work has fixed ownership,
immutable job claims, byte-pinned sources and stage barriers. A failed owner or
job fails closed; another invocation cannot adopt/restart its reserved work.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid

SCHEMA = 'siiba_context_execution_v1'
OWNER_CPUS = [64, 65]
ROOT = None
HEARTBEAT = None


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def enc(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def seal(x):
    need('record_sha256' not in x, 'Already sealed')
    return dict(x, record_sha256=hashlib.sha256(enc(x)).hexdigest())


def checked(x):
    need(x.get('record_sha256') == seal({k: v for k, v in x.items() if k != 'record_sha256'})['record_sha256'], 'Seal differs')
    return x


def desc(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': sha(p)}


def bound(d):
    p = Path(d['path'])
    need(p.is_absolute() and p.is_file() and sha(p) == d['sha256'], 'Pinned bytes changed: ' + str(p))
    return p


def save(p, x, mutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = enc(x)
    if p.exists() and not mutable:
        need(p.read_bytes() == data, 'Refuse differing overwrite: ' + str(p))
        return
    tmp = p.with_name(p.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if mutable:
            os.replace(str(tmp), str(p))
        else:
            os.link(str(tmp), str(p))
    finally:
        if tmp.exists(): tmp.unlink()


def ident(pid, command=True):
    raw = Path('/proc/%d/stat' % pid).read_text()
    v = raw[raw.rfind(')') + 2:].split()
    d = dict(pid=pid, start_ticks=int(v[19]), state=v[0], pgid=int(v[2]), ppid=int(v[1]),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if command and pid != 1:
        d['command'] = [p.decode() for p in Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\0') if p]
        d['cpus'] = sorted(os.sched_getaffinity(pid))
    return d


def same(a, b):
    return all(a.get(k) == b[k] for k in ('pid', 'start_ticks', 'boot_id', 'command')) and a['state'] not in ('Z', 'X')


def system():
    p = ident(1, False)
    return dict(boot_id=p['boot_id'], pid1_starttick=p['start_ticks'], hostname=platform.node())


def gpu_rows():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                       check=True, capture_output=True, text=True, timeout=30)
    return {int(x.split(',')[0].strip()): x.split(',')[1].strip() for x in p.stdout.splitlines() if x.strip()}


def gpu_apps():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                       check=True, capture_output=True, text=True, timeout=30)
    return [{'pid': int(line.split(',')[0].strip()), 'uuid': line.split(',')[1].strip()}
            for line in p.stdout.splitlines() if line.strip()]


def verify_host(req, host, idle=False):
    need(host in req['hosts'] and system() == req['hosts'][host], 'Wrong host/container identity')
    wanted = {g['local_index']: g['uuid'] for g in req['gpu_inventory'] if g['host'] == host}
    need(gpu_rows() == wanted, 'GPU UUID mapping differs')
    need(set(OWNER_CPUS + sum([g['cpus'] for g in req['gpu_inventory'] if g['host'] == host], [])).issubset(os.sched_getaffinity(1)),
         'Requested CPU sets unavailable')
    if idle:
        need(not gpu_apps(), 'Competing GPU compute applications exist')


def prefix(cpus):
    return ['taskset', '-c', ','.join(map(str, cpus)), 'nice', '-n', '19', 'ionice', '-c', '3']


def env(gpu_uuid=None):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid or '', CUBLAS_WORKSPACE_CONFIG=':4096:8',
             PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    e.pop('PYTHONPATH', None); e.pop('PYTHONHOME', None)
    return e


def recipe(root, python):
    """Fixed 74 jobs; request seals their exact commands and ownership."""
    root = Path(root)
    f = str(root / 'source/feature_pipeline.py')
    p = str(root / 'source/probes.py')
    e = str(root / 'source/evaluate.py')
    stages = []
    def stage(name, jobs):
        stages.append(dict(name=name, jobs=jobs))
    def feature(name, slot, phase, output, extra=None):
        return dict(id=name, slot=slot, argv=[python, f, '--root', str(root), '--phase', phase] + (extra or []) +
                    (['--device', 'cuda:0'] if slot != 'cpu' else ['--device', 'cpu']),
                    outputs=[str(root / output)])
    stage('validate', [feature('validate_%d' % i, i, 'validate', 'validation/%d.json' % i, ['--shard', str(i)]) for i in range(8)])
    stage('freeze', [feature('freeze', 'cpu', 'freeze', 'design/selection.json')])
    stage('pca_samples', [feature('pca_samples_%d' % i, i, 'pca_samples', 'pca_samples/%d.json' % i, ['--shard', str(i)]) for i in range(8)])
    stage('fit_pca', [feature('fit_pca', 0, 'fit_pca', 'pca/receipt.json')])
    for split in ('train', 'val', 'test'):
        stage('extract_' + split, [feature('extract_%s_%d' % (split, i), i, 'extract', 'features/%s/%d.json' % (split, i),
              ['--split', split, '--shard', str(i)]) for i in range(8)])
    stage('assemble', [feature('assemble', 'cpu', 'assemble', 'features/completion.json')])
    stage('train_probes', [dict(id='train_%02d' % j, slot=j % 8, argv=[python, p, '--root', str(root), '--job', str(j), '--device', 'cuda:0'],
                               outputs=[str(root / ('probes/%02d/completion.json' % j))]) for j in range(15)])
    stage('predict', [dict(id='predict_%02d' % j, slot=j % 8, argv=[python, e, '--root', str(root), '--phase', 'predict', '--job', str(j), '--device', 'cuda:0'],
                          outputs=[str(root / ('test/%02d.json' % j))]) for j in range(15)])
    stage('aggregate', [dict(id='aggregate', slot='cpu', argv=[python, e, '--root', str(root), '--phase', 'aggregate', '--device', 'cpu'],
                            outputs=[str(root / 'report/summary.json')])])
    return stages


def configure(allocation):
    with role_lock('configure'):
        need(not (ROOT / 'execution_request.json').exists(), 'Execution request exists')
        prior = checked(read(allocation))
        science = checked(read(ROOT / 'request.json'))
        need(science['schema'] == 'siiba_context_request_v1', 'Wrong scientific request')
        sources = {p: desc(ROOT / 'source' / p) for p in ('owner.py', 'feature_pipeline.py', 'probes.py', 'evaluate.py')}
        python = prior['python']
        stages = recipe(ROOT, python)
        need(not any(Path(p).exists() for stage in stages for j in stage['jobs'] for p in j['outputs']), 'A planned output already exists; never adopt/retry')
        req = seal(dict(schema=SCHEMA, root=str(ROOT), previous_allocation=desc(allocation), scientific_request=desc(ROOT / 'request.json'),
                   python=python, python_binary=desc(python), hosts=prior['hosts'], gpu_inventory=prior['gpu_inventory'], sources=sources,
                   owner_cpus=OWNER_CPUS, stages=stages, scientific_rules=dict(amp=False, tf32=False, precision='float32',
                   job_retries=0, gpu_cpu_threads=2, source_mutations=False), created_unix=time.time()))
        save(ROOT / 'execution_request.json', req)
        return req


def request():
    req = checked(read(ROOT / 'execution_request.json'))
    need(req['schema'] == SCHEMA and req['root'] == str(ROOT) and req['stages'] == recipe(ROOT, req['python']), 'Execution recipe differs')
    need(bound(req['scientific_request']) == (ROOT / 'request.json').resolve(), 'Wrong scientific request path')
    need(bound(req['python_binary']) == Path(req['python']).resolve(), 'Runtime symlink target changed')
    bound(req['previous_allocation'])
    for name, d in req['sources'].items():
        actual = ROOT / 'source' / name
        need(not actual.is_symlink() and bound(d) == actual.resolve(), 'Source execution path differs')
    need(bound(req['sources']['owner.py']) == Path(__file__).resolve(), 'Wrong owner source')
    old = checked(read(req['previous_allocation']['path']))
    need(req['hosts'] == old['hosts'] and req['gpu_inventory'] == old['gpu_inventory'], 'Allocation provenance differs')
    need(set(req['hosts']) == {'30663', '30965'} and [g['index'] for g in req['gpu_inventory']] == list(range(8)), 'Eight-slot mapping differs')
    need([g['host'] for g in req['gpu_inventory']] == ['30663'] * 7 + ['30965'], 'Host slot assignment differs')
    need(len({g['uuid'] for g in req['gpu_inventory']}) == 8, 'Duplicate GPU UUID')
    return req


@contextmanager
def role_lock(role, blocking=False):
    import fcntl
    (ROOT / 'owners').mkdir(exist_ok=True, parents=True)
    with (ROOT / 'owners' / (role + '.lock')).open('a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def role_path(role, name):
    return ROOT / 'owners' / role / name


class Heartbeat:
    """Publish process liveness while the main thread hashes large artifacts.

    The lock serializes main-thread state updates and timed publications. It
    does not change job state, complete a job, or excuse missing output checks.
    """
    def __init__(self, role, identity, interval=10.0):
        self.role = role
        self.identity = copy.deepcopy(identity)
        self.interval = interval
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.error = None
        self.state = dict(status='waiting', identity=self.identity)
        self.thread = threading.Thread(target=self._loop, name='owner-status-heartbeat', daemon=True)

    def _publish(self):
        # 锁内复制快照
        save(role_path(self.role, 'status.json'), dict(copy.deepcopy(self.state), updated_unix=time.time()), mutable=True)

    def start(self):
        with self.lock: self._publish()
        self.thread.start()
        return self

    def update(self, value):
        with self.lock:
            need(self.error is None, 'Heartbeat publisher failed: ' + str(self.error))
            self.state = copy.deepcopy(dict(value, identity=self.identity))
            self._publish()

    def _loop(self):
        while not self.done.wait(self.interval):
            try:
                with self.lock: self._publish()
            except BaseException as error:
                self.error = repr(error)
                self.done.set()

    def stop(self):
        self.done.set()
        if self.thread.is_alive(): self.thread.join(timeout=30)
        need(not self.thread.is_alive(), 'Heartbeat thread failed to stop')


def stop_heartbeat():
    global HEARTBEAT
    if HEARTBEAT is not None:
        HEARTBEAT.stop()
        HEARTBEAT = None


def launch(role, host):
    need(role == 'coordinator' or role == 'agent_' + host, 'Wrong role')
    need(role != 'coordinator' or host == '30663', 'Coordinator belongs on30663')
    with role_lock(role):
        req = request(); verify_host(req, host, idle=True)
        need(not role_path(role, 'reservation.json').exists(), 'Role already reserved; no relaunch')
        save(role_path(role, 'reservation.json'), seal(dict(role=role, host=host, request=desc(ROOT / 'execution_request.json'), time=time.time())))
        argv = [req['python'], str(ROOT / 'source/owner.py'), '--root', str(ROOT), 'agent' if role.startswith('agent_') else 'coordinator', '--host', host]
        log = role_path(role, 'owner.log')
        with log.open('xb') as f:
            child = subprocess.Popen(prefix(OWNER_CPUS) + argv, env=env(), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=str(ROOT))
        for _ in range(100):
            need(child.poll() is None, 'Detached owner exited during registration')
            current = ident(child.pid)
            if current['command'] == argv: break
            time.sleep(.05)
        need(current['command'] == argv and current['pgid'] == child.pid and current['cpus'] == OWNER_CPUS, 'Detached owner identity differs')
        receipt = seal(dict(role=role, host=host, identity=current, command=argv, execution_request=desc(ROOT / 'execution_request.json'), time=time.time()))
        save(role_path(role, 'launch.json'), receipt)
        return receipt


def owner_start(role, host):
    global HEARTBEAT
    req = request(); verify_host(req, host)
    value = checked(read(role_path(role, 'launch.json')))
    need(same(ident(os.getpid()), value['identity']) and value['execution_request'] == desc(ROOT / 'execution_request.json'), 'No matching owner receipt')
    need(sorted(os.sched_getaffinity(0)) == OWNER_CPUS and os.getpriority(os.PRIO_PROCESS, 0) == 19, 'Owner resource policy differs')
    need(HEARTBEAT is None, 'A heartbeat already owns this process')
    HEARTBEAT = Heartbeat(role, ident(os.getpid())).start()
    return req


def announce(role, state):
    if HEARTBEAT is not None and HEARTBEAT.role == role:
        HEARTBEAT.update(state)
    else:
        save(role_path(role, 'status.json'), dict(state, updated_unix=time.time()), mutable=True)


def failures():
    return sorted((ROOT / 'failures').glob('*.json'))


def failure(role, error, running=None):
    value = seal(dict(role=role, error_type=type(error).__name__, error=str(error), running_children=running or [],
                      retry_allowed=False, active_children_left_untouched=True, time=time.time()))
    save(ROOT / 'failures' / (role + '.json'), value)
    announce(role, dict(status='failed_closed', failure=desc(ROOT / 'failures' / (role + '.json'))))


def jobpath(job, name):
    return ROOT / 'execution/jobs' / job['id'] / name


def expected_job(req, stage_index, job):
    return seal(dict(stage_index=stage_index, job=job, request=desc(ROOT / 'execution_request.json'),
                sources=req['sources'], owner='coordinator' if job['slot'] == 'cpu' else 'agent_' + req['gpu_inventory'][job['slot']]['host']))


def start_job(req, host, stage_index, job):
    need(not failures(), 'Another owner/job failed; no additional launch')
    request(); verify_host(req, host)
    frozen = checked(read(jobpath(job, 'job.json')))
    need(frozen == expected_job(req, stage_index, job), 'Job descriptor changed')
    need(not jobpath(job, 'claim.json').exists(), 'Job already claimed; no retry/adoption')
    need(not any(Path(p).exists() for p in job['outputs']), 'Output already exists without this job completion')
    slot = job['slot']
    gpu = None if slot == 'cpu' else req['gpu_inventory'][slot]
    if gpu:
        need(gpu['host'] == host and not any(a['uuid'] == gpu['uuid'] for a in gpu_apps()), 'GPU assigned elsewhere or busy')
    cpus = gpu['cpus'] if gpu else OWNER_CPUS
    save(jobpath(job, 'claim.json'), seal(dict(job=desc(jobpath(job, 'job.json')), owner=ident(os.getpid()), host=host,
             gpu=gpu, cpus=cpus, environment={k: env(gpu['uuid'] if gpu else None)[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, time=time.time())))
    with jobpath(job, 'stdout.log').open('xb') as f:
        p = subprocess.Popen(prefix(cpus) + job['argv'], env=env(gpu['uuid'] if gpu else None), stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT))
    for _ in range(100):
        if p.poll() is not None: break
        current = ident(p.pid)
        if current['command'] == job['argv']: break
        time.sleep(.05)
    need(p.poll() is None and current['command'] == job['argv'] and current['cpus'] == cpus and current['pgid'] == p.pid and
         os.getpriority(os.PRIO_PROCESS, p.pid) == 19, 'Worker identity/resource differs')
    save(jobpath(job, 'launch.json'), seal(dict(job=desc(jobpath(job, 'job.json')), identity=current, host=host, gpu=gpu,
                                             scientific_request=req['scientific_request'], time=time.time())))
    return p, current


def verify_direct_descriptors(value):
    # 递归校验产物哈希


    if isinstance(value, dict):
        if set(('path', 'sha256')).issubset(value): bound(value)
        else:
            for v in value.values(): verify_direct_descriptors(v)
    elif isinstance(value, list):
        for v in value: verify_direct_descriptors(v)


def finish_job(req, job, proc, birth):
    code = proc.poll()
    need(code is not None, 'Worker is still running')
    save(jobpath(job, 'exit.json'), seal(dict(returncode=code, identity=birth, time=time.time())))
    need(code == 0, 'Worker returned %s: %s' % (code, job['id']))
    outputs = []
    for path in job['outputs']:
        result = checked(read(path))
        if 'request' in result: need(result['request'] == req['scientific_request'], 'Output scientific request differs')
        verify_direct_descriptors(result)
        outputs.append(desc(path))
    save(jobpath(job, 'completion.json'), seal(dict(status='complete', job=desc(jobpath(job, 'job.json')), outputs=outputs,
         returncode=0, launch=desc(jobpath(job, 'launch.json')), time=time.time())))


def verify_job_done(req, stage_index, job):
    frozen = checked(read(jobpath(job, 'job.json')))
    need(frozen == expected_job(req, stage_index, job), 'Stage job binding changed')
    complete = checked(read(jobpath(job, 'completion.json')))
    need(complete['status'] == 'complete' and complete['returncode'] == 0 and complete['job'] == desc(jobpath(job, 'job.json')), 'Job not successful')
    need([d['path'] for d in complete['outputs']] == job['outputs'], 'Job output inventory differs')
    for d in complete['outputs']: bound(d)
    bound(complete['launch'])
    return desc(jobpath(job, 'completion.json'))


def healthy_agents(req):
    now = time.time()
    for host in req['hosts']:
        role = 'agent_' + host
        launch = checked(read(role_path(role, 'launch.json')))
        state = read(role_path(role, 'status.json'))
        need(state['status'] in ('waiting', 'working') and now - state['updated_unix'] < 120 and
             same(state['identity'], launch['identity']), 'Host agent missing/stale/failed: ' + host)


def setup_signals():
    stop = []
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, lambda sig, frame: stop.append(sig))
    return stop


def agent(host):
    role = 'agent_' + host
    stop = setup_signals()
    with role_lock(role, True):
        running = {}
        try:
            req = owner_start(role, host)
            me = ident(os.getpid())
            while not (ROOT / 'completion.json').exists():
                need(not stop, 'Owner stop requested')
                # 先回收已退出任务
                # 保留同批任务
                for slot, (job, proc, birth) in list(running.items()):
                    if proc.poll() is not None:
                        finish_job(req, job, proc, birth)
                        del running[slot]
                need(not failures(), 'Failure recorded by another owner/job')
                current = ROOT / 'execution/current_stage.json'
                if current.exists():
                    current_value = checked(read(current))
                    si = current_value['index']
                    need(0 <= si < len(req['stages']) and current_value['request'] == desc(ROOT / 'execution_request.json') and
                         current_value['stage'] == req['stages'][si], 'Stage publication differs')
                    stage = req['stages'][si]
                    for job in stage['jobs']:
                        slot = job['slot']
                        if slot == 'cpu' or req['gpu_inventory'][slot]['host'] != host or slot in running:
                            continue
                        if jobpath(job, 'completion.json').exists():
                            verify_job_done(req, si, job)
                            continue
                        proc, birth = start_job(req, host, si, job)
                        running[slot] = (job, proc, birth)
                announce(role, dict(status='working' if running else 'waiting', identity=me,
                         running={str(k): dict(job=v[0]['id'], identity=v[2]) for k, v in running.items()}))
                time.sleep(5)
            need(not running, 'Global completion published with active workers')
            announce(role, dict(status='complete', identity=me, completion=desc(ROOT / 'completion.json')))
        except BaseException as error:
            failure(role, error, [dict(job=j['id'], identity=b) for j, p, b in running.values()])
            raise
        finally:
            stop_heartbeat()


def coordinator(host):
    stop = setup_signals()
    role = 'coordinator'
    with role_lock(role, True):
        active = None
        try:
            req = owner_start(role, host)
            need(host == '30663', 'Wrong coordinator host')
            need(not (ROOT / 'execution/current_stage.json').exists(), 'Stages already published; no coordinator replay')
            healthy_agents(req)
            stage_receipts = []
            for si, stage in enumerate(req['stages']):
                need(not stop and not failures(), 'Stop or failure before stage')
                request(); healthy_agents(req)
                for job in stage['jobs']:
                    need(not jobpath(job, 'claim.json').exists(), 'Stage already claimed; no replay')
                    save(jobpath(job, 'job.json'), expected_job(req, si, job))
                save(ROOT / 'execution/current_stage.json', seal(dict(index=si, stage=stage, request=desc(ROOT / 'execution_request.json'))), mutable=True)
                announce(role, dict(status='running', identity=ident(os.getpid()), stage=stage['name'], stage_index=si))
                for job in stage['jobs']:
                    if job['slot'] != 'cpu': continue
                    p, birth = start_job(req, host, si, job)
                    active = (job, p, birth)
                    while p.poll() is None:
                        need(not stop and not failures(), 'Stop/failure while CPU stage active')
                        announce(role, dict(status='running', identity=ident(os.getpid()), stage=stage['name'], cpu_job=job['id'], child=birth))
                        time.sleep(5)
                    finish_job(req, job, p, birth)
                    active = None
                while not all(jobpath(j, 'completion.json').exists() for j in stage['jobs']):
                    need(not stop and not failures(), 'Stop/failure while stage active')
                    healthy_agents(req)
                    done = sum(jobpath(j, 'completion.json').exists() for j in stage['jobs'])
                    announce(role, dict(status='running', identity=ident(os.getpid()), stage=stage['name'], completed_jobs=done, jobs=len(stage['jobs'])))
                    time.sleep(5)
                outputs = [verify_job_done(req, si, j) for j in stage['jobs']]
                stage_file = ROOT / 'execution/stages' / ('%02d_%s.json' % (si, stage['name']))
                save(stage_file, seal(dict(status='complete', index=si, stage=stage['name'], jobs=outputs, time=time.time())))
                stage_receipts.append(desc(stage_file))
            report = checked(read(ROOT / 'report/summary.json'))
            need(report['status'] == 'complete' and len(report['paired_tests']) == 16 and len(report['summaries']) == 40, 'Scientific report incomplete')
            request()
            save(ROOT / 'completion.json', seal(dict(status='complete', scientific_request=desc(ROOT / 'request.json'),
                 execution_request=desc(ROOT / 'execution_request.json'), stages=stage_receipts, report=desc(ROOT / 'report/summary.json'),
                 job_count=74, independent_delivery_acceptance_required=True, time=time.time())))
            announce(role, dict(status='complete', completion=desc(ROOT / 'completion.json'), identity=ident(os.getpid())))
        except BaseException as error:
            failure(role, error, [] if active is None else [dict(job=active[0]['id'], identity=active[2])])
            raise
        finally:
            stop_heartbeat()


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('action', choices=('configure', 'launch-agent', 'launch-coordinator', 'agent', 'coordinator', 'status'))
    p.add_argument('--host', choices=('30663', '30965'))
    p.add_argument('--allocation', type=Path)
    a = p.parse_args(); ROOT = a.root.resolve()
    need(ROOT.name == 'siiba_context_supplement_20261028_v2', 'Wrong versioned output root')
    if a.action == 'configure':
        need(a.allocation is not None, 'Original allocation request required')
        out = configure(a.allocation)
    elif a.action in ('launch-agent', 'launch-coordinator'):
        need(a.host is not None, 'Host required')
        out = launch('agent_' + a.host if a.action == 'launch-agent' else 'coordinator', a.host)
    elif a.action in ('agent', 'coordinator'):
        need(a.host is not None, 'Host required')
        {'agent': agent, 'coordinator': coordinator}[a.action](a.host); return
    else:
        out = {str(path.relative_to(ROOT)): read(path) for path in sorted((ROOT / 'owners').glob('*/status.json'))}
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
