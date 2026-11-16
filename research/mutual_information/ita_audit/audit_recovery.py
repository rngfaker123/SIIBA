"""Audit-only recovery. No model, extraction, KSG, training, or old owner launch.

Run under the original frozen Python environment, after readonly process checks:
  python audit_recovery.py preflight --resources /path/to/resources.json
  python audit_recovery.py audit --resources /path/to/resources.json
The caller may detach the audit process. It is never restarted automatically.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path('/study/siiba/experiments/mi_rgb_pl_regions_20261104_v1')
ORIGINAL_SOURCE_SHA = 'b099719c17615d9bced1f1bbc8a826e01e24516c4775638d41816e9e330d4855'
ORIGINAL_REQUEST_SHA = '2717d6012b15f4067cb7c948d11605f0f61412c4d036f1facf05fa876437bd82'
OLD_OWNER = dict(pid=29755, starttick=622190653, boot='53ba5d39-a628-4c4f-9607-f11426e90dc6')
COHORT_COUNTS = dict(dark=10, very_light=309, table1=1000, light=2498, brown=2793, intermediate=5979, tan=8056)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def isolated_roots():
    new, old = ROOT.resolve(), ORIGINAL_ROOT.resolve()
    require(new != old and old not in new.parents and new not in old.parents,
            'Recovery and original roots must be disjoint')


def write_new(name, value, mutable=False):
    isolated_roots()
    path = (ROOT / name).resolve()
    require(ROOT.resolve() in path.parents, 'Recovery write outside new root')
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(value, record_sha256=hashlib.sha256(encoded(value)).hexdigest())
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()) + '.' + secrets.token_hex(4))
    with temp.open('xb') as handle:
        handle.write(encoded(doc)); handle.flush(); os.fsync(handle.fileno())
    try:
        if mutable:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def identity(pid=None):
    pid = pid or os.getpid()
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttick=int(fields[19]), ppid=int(fields[1]),
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid1_starttick=int(Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19]),
                cpus=sorted(os.sched_getaffinity(pid)))


def cpus(text):
    result = set()
    for piece in text.strip().split(','):
        ends = piece.split('-')
        result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def reject_old_or_other_audit():
    """Reject exact old identity, any original script process, or another recovery audit."""
    source = str(ORIGINAL_ROOT/'source/owner.py').encode()
    own_source = str(Path(__file__).resolve()).encode()
    seen = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry/'cmdline').read_bytes().split(b'\0')
            ident = identity(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not all(ident.get(k) == v for k, v in OLD_OWNER.items()), 'Original owner identity is still alive')
        require(source not in args, 'Original owner/audit/worker process still exists: PID ' + entry.name)
        require(not (own_source in args and b'audit' in args), 'Another recovery audit is already alive: PID ' + entry.name)
        seen.append(ident['pid'])
    return dict(scanned_process_count=len(seen), old_owner_and_original_script_absent=True,
                other_recovery_audit_absent=True)


def resource_check(resources):
    ident = identity()
    require(all(ident[k] == resources['host'][k] for k in ('boot','pid1_starttick')), 'Recovery host/container mismatch')
    selected = resources['audit_cpus']
    require(selected and len(selected) == len(set(selected)), 'Empty/duplicate CPU allocation')
    require(set(selected).issubset(os.sched_getaffinity(0)), 'Recovery CPUs unavailable')
    protected = set(resources['protected_training_cpus'])
    require(not protected.intersection(selected), 'Audit overlaps training CPU allocation')
    for cpu in selected:
        siblings = cpus(Path('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % cpu).read_text())
        require(not siblings.intersection(protected), 'Audit shares a physical CPU with training')
    for prior in resources.get('protected_processes', []):
        actual = identity(prior['pid'])
        require(all(actual[k] == prior[k] for k in ('starttick','boot','pid1_starttick','cpus')), 'Protected process changed')
    return ident


def immutable_original_inventory():
    paths = list((ORIGINAL_ROOT/'source').glob('*.py'))
    for name in ('protocol.json','baseline_pins.json','resources.json','task_plan.json','request.json',
                 'launch_reservation.json','launch_receipt.json','owner_started.json','pilot_pass.json','status.json'):
        paths.append(ORIGINAL_ROOT/name)
    paths.extend(sorted((ORIGINAL_ROOT/'raw').glob('*.json')))
    paths.extend(sorted((ORIGINAL_ROOT/'summaries').glob('*.json')))
    require(len(list((ORIGINAL_ROOT/'raw').glob('*.json'))) == 126, 'Original raw count is not126')
    require(len(list((ORIGINAL_ROOT/'summaries').glob('*.json'))) == 7, 'Original summary count is not7')
    return [descriptor(path) for path in sorted(set(paths))]


def preflight(resources_path):
    isolated_roots()
    resources = read(resources_path)
    require(resources['operation'] == 'audit_only_no_scientific_executor', 'Wrong recovery operation')
    host = resource_check(resources)
    absent = reject_old_or_other_audit()
    require(sha(ORIGINAL_ROOT/'source/owner.py') == ORIGINAL_SOURCE_SHA, 'Original audit source changed')
    require(sha(ORIGINAL_ROOT/'request.json') == ORIGINAL_REQUEST_SHA, 'Original scientific request changed')
    require(not (ORIGINAL_ROOT/'completion.json').exists(), 'Original completion already exists; inspect without rerunning')
    require(not (ORIGINAL_ROOT/'independent_acceptance.json').exists(), 'Original acceptance already exists; inspect without rerunning')
    require(not list((ORIGINAL_ROOT/'failures').glob('*.json')), 'Original failure requires diagnosis')
    require(not (ROOT/'audit_reservation.json').exists(), 'Recovery already reserved; never automatically retry')
    require(not (ROOT/'recovery_acceptance.json').exists(), 'Recovery acceptance already exists')
    require(not list((ROOT/'failures').glob('*.json')), 'Recovery failure already preserved')
    inventory = immutable_original_inventory()
    return dict(time=now(), operation=resources['operation'], identity=host, process_absence=absent,
                resources=descriptor(resources_path), recovery_source=descriptor(__file__),
                original_source_sha256=ORIGINAL_SOURCE_SHA, original_request_sha256=ORIGINAL_REQUEST_SHA,
                original_inventory=inventory)


def forbidden(*args, **kwargs):
    raise RuntimeError('Scientific execution is forbidden in audit-only recovery')


def bind_original_audit():
    path = ORIGINAL_ROOT/'source/owner.py'
    require(sha(path) == ORIGINAL_SOURCE_SHA, 'Original source changed before import')
    spec = importlib.util.spec_from_file_location('readonly_original_MI_audit', str(path))
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    require(original.ROOT.resolve() == ORIGINAL_ROOT.resolve(), 'Original read root changed')
    load_science = original.load_science

    def guarded_science_import():
        result = load_science()
        original.CTX['est'].ksg_mi = forbidden
        original.CTX['runner'].ksg_mi = forbidden
        return result

    original.load_science = guarded_science_import
    for name in ('configure','pilot','pilot_masks','launch','owner','lane','make_pools','extraction_worker',
                 'extract_pl_arrays','summary_cohort','independent_fixture','publish_bytes','status','fail'):
        setattr(original, name, forbidden)

    pending = {}

    def only_final_acceptance(path, value, mutable=False):
        require(Path(path).resolve() == (ORIGINAL_ROOT/'independent_acceptance.json').resolve() and not mutable,
                'Original audit attempted an unauthorized write')
        require(value['status'] == 'passed' and value['raw_count'] == 126 and value['cells'] == 42,
                'Original audit acceptance scope differs')
        require(not pending, 'Original audit attempted duplicate acceptance')
        pending.update(original_audit_acceptance=value,
                       original_acceptance_destination=str(path), original_destination_written=False,
                       original_audit_source_sha256=ORIGINAL_SOURCE_SHA)

    original.put = only_final_acceptance
    real_verify = original.verify
    progress = dict(verified_files=0, last_update=0.0)

    def counted_verify(descriptor):
        clock = time.monotonic()
        if clock - progress['last_update'] >= 5:
            write_new('status.json',dict(time=now(),phase='original_audit_hash_checks',
                      verified_files=progress['verified_files'],next_path=descriptor['path']),True)
            progress['last_update'] = clock
        result = real_verify(descriptor)
        progress['verified_files'] += 1
        return result

    original.verify = counted_verify
    return original, progress, pending


def audit_only(resources_path):
    import fcntl
    isolated_roots()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'audit_recovery.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # 只读打开原锁
    original_lock = (ORIGINAL_ROOT/'owner.lock').open('rb')
    fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    review = preflight(resources_path)
    resources = read(resources_path)
    require(descriptor(resources_path) == review['resources'], 'Recovery resource binding changed')
    os.sched_setaffinity(0,set(resources['audit_cpus']))
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    # 强制空闲级 I/O
    import subprocess
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True,stdout=subprocess.DEVNULL)
    review['audit_identity'] = identity()
    write_new('audit_reservation.json',review)
    try:
        original, progress, pending = bind_original_audit()
        original.audit()
        require(pending, 'Original audit did not return acceptance')
        require(immutable_original_inventory() == review['original_inventory'], 'Original protected bytes changed during audit')
        require(not (ORIGINAL_ROOT/'completion.json').exists() and not (ORIGINAL_ROOT/'independent_acceptance.json').exists(),
                'Original completion/acceptance unexpectedly appeared')
        require(descriptor(__file__) == review['recovery_source'] and descriptor(resources_path) == review['resources'],
                'Recovery source/resource bytes changed during audit')
        write_new('recovery_acceptance.json', dict(pending, time=now(),
                  original_protected_bytes_unchanged=True))
        write_new('recovery_receipt.json',dict(time=now(),status='original_audit_completed_in_recovery',
                  audit_identity=identity(),reservation=descriptor(ROOT/'audit_reservation.json'),
                  acceptance=descriptor(ROOT/'recovery_acceptance.json'),verified_file_operations=progress['verified_files'],
                  original_protected_bytes_unchanged=True,original_supervisor_completion_created=False,
                  KSG_or_extraction_or_model_executed=False,limits=['Original KSG estimates were not independently rerun.',
                  'Historical owner/completion state is preserved; this is a distinct audit recovery receipt.']))
        write_new('status.json',dict(time=now(),phase='recovery_audit_complete',
                  receipt=descriptor(ROOT/'recovery_receipt.json')),True)
    except BaseException:
        write_new('failures/audit_'+str(os.getpid())+'_'+str(time.time_ns())+'.json',dict(time=now(),traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('preflight','audit'))
    parser.add_argument('--resources',required=True,type=Path)
    args = parser.parse_args()
    if args.command == 'preflight':
        result = preflight(args.resources)
        print(json.dumps(dict(preflight='passed',identity=result['identity'],
                              protected_original_files=len(result['original_inventory'])),sort_keys=True))
    else:
        audit_only(args.resources)


if __name__ == '__main__':
    main()
