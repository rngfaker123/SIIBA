"""Freeze an exclusion-filtered cohort and reuse the original split functions.

Only metadata is written in a new versioned directory. No raw image, old
manifest, training configuration, or checkpoint is changed.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

ROOT = Path('/study/siiba')
VERSION = 'dataset_clean_v1.0_20261013'
SPLIT_VERSION = 'split_clean_v1.0_20261013'
OLD = ROOT / 'data_protocol/dataset_v1.0'
DEST = ROOT / 'data_protocol' / VERSION
MANIFEST_SHA = '45aa180aec1fdb50c4797ef9053be0afc11790035e446b6f76802dcfe5fe9e6a'
CODE_SHA = '4e1766e432e7e2ff6e06697533bd30f9bae65e6f9b35bb05b1b3eb1e4dd1697c'
CATALOG_SHA = '3b4aee2e48c3d74b73a2be6c6a1cbc4b027e3a0b0ada3846f85ab990273086e0'
BROAD_SHA = '9c4c50c7ea0c4d1bf5b595c3cf618a3960be60b06d01996d158291f08572bead'
HIGH_SHA = 'da2699a58da73c892ec8c4452b30725992b94a77bc62a95b71867cdf70928296'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save_json(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def load_split():
    path = Path(__file__).with_name('create_split.py')
    if sha(path) != CODE_SHA:
        raise RuntimeError('Original split code differs')
    spec = importlib.util.spec_from_file_location('original_split', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def source_category(event):
    source = event['payload'].get('source', '')
    if source == 'explicit_user_approved_triage_high_batch':
        return 'user_approved_high_priority_batch'
    if source == 'explicit_human_triage_broad_ui':
        return 'human_broad_review'
    if event['event'] == 'confirmed_seed_import':
        return 'human_confirmed_seed'
    if source == 'same_origin_loopback_review_ui':
        return 'human_original_review'
    raise RuntimeError('Unclassified exclusion event: ' + str(event))


def execute(report_path):
    if DEST.exists():
        raise RuntimeError('Frozen destination already exists; never overwrite')
    report = json.loads(report_path.read_text(encoding='utf8'))
    # 取审查后快照
    snapshot_path = Path(report['snapshot_path'])
    if sha(snapshot_path) != report['snapshot_sha256']:
        raise RuntimeError('Frozen review snapshot checksum differs')
    mod, code_path = load_split()
    old_manifest = OLD / 'manifest_frozen_v1.0.csv'
    if sha(old_manifest) != MANIFEST_SHA:
        raise RuntimeError('Original manifest differs')
    rows = mod.read_csv(old_manifest)
    if len(rows) != 115343:
        raise RuntimeError('Incorrect parent cohort')
    index = {r['sample_id']: r for r in rows}
    if len(index) != len(rows):
        raise RuntimeError('Duplicate sample IDs')
    mod.validate_input_rows(rows)
    reproduced = mod.assign_splits(rows)
    for r in reproduced:
        for field in ('split', 'split_key_sha256', 'split_group_id', 'split_unit'):
            if r[field] != index[r['sample_id']][field]:
                raise RuntimeError('Original split reproduction differs')
    mod.validate_splits(reproduced)
    del reproduced

    conn = sqlite3.connect(snapshot_path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise RuntimeError('Review snapshot integrity failure')
    if conn.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Review snapshot foreign key failure')
    settings = {r[0]: json.loads(r[1]) for r in conn.execute('SELECT key,value FROM settings')}
    if settings['manifest_sha256'] != MANIFEST_SHA or settings['catalog_sha256'] != CATALOG_SHA:
        raise RuntimeError('Review provenance differs')
    catalog = list(conn.execute('SELECT * FROM catalog ORDER BY row_index'))
    if {r['sample_id'] for r in catalog} != set(index):
        raise RuntimeError('Catalog and manifest differ')
    for i, r in enumerate(catalog, 1):
        if r['row_index'] != i:
            raise RuntimeError('Catalog range not contiguous')
        rec = json.loads(r['original_record'])
        original = index[r['sample_id']]
        for key in ('sample_id', 'subject_id', 'split', 'uv_sha256', 'rgb_path', 'uv_path', 'plv_path', 'plh_path'):
            if rec[key] != original[key]:
                raise RuntimeError('Catalog original record differs: ' + key)
    states = {r['sample_id']: dict(r) for r in conn.execute('SELECT * FROM reviews')}
    if set(states) != set(index):
        raise RuntimeError('Missing review rows')
    triage = ROOT / 'output/uv_lightleak_triage_20261013_v1'
    queues = {}
    for name, wanted in [('broad', BROAD_SHA), ('high', HIGH_SHA)]:
        path = triage / ('queue_' + name + '.json')
        if sha(path) != wanted:
            raise RuntimeError('Frozen queue changed')
        queues[name] = {r['sample_id'] for r in json.loads(path.read_text())}
    if len(queues['broad']) != 12442 or len(queues['high']) != 4159:
        raise RuntimeError('Queue cardinality differs')
    if queues['broad'] & queues['high']:
        raise RuntimeError('Queues overlap')
    completed_ids = {r['sample_id'] for r in catalog if r['row_index'] <= 25000} | queues['broad']
    if any(states[s]['decision'] == 'unreviewed' for s in completed_ids):
        raise RuntimeError('Completed review range not normalized')
    if any(states[s]['decision'] != 'exclude_light_leak' for s in queues['high']):
        raise RuntimeError('Approved HIGH exclusions changed; investigate before freeze')

    events = []
    last_event = {}
    for r in conn.execute('SELECT * FROM audit_events ORDER BY audit_id'):
        e = dict(r)
        for key in ('old_state', 'new_state', 'payload'):
            e[key] = json.loads(e[key]) if e[key] else None
        events.append(e)
        if e['sample_id'] and e['new_state'] is not None:
            last_event[e['sample_id']] = e
    cutoff = conn.execute('SELECT MAX(audit_id) FROM audit_events').fetchone()[0]
    conn.close()
    excluded_ids = {s for s, state in states.items() if state['decision'] == 'exclude_light_leak'}
    excluded = []
    source_counts = Counter()
    for sid in sorted(excluded_ids):
        e = last_event.get(sid)
        if not e or e['new_state']['decision'] != 'exclude_light_leak':
            raise RuntimeError('Missing final exclusion event')
        for key in ('decision', 'note', 'version', 'updated_at'):
            if key in e['new_state'] and e['new_state'][key] != states[sid][key]:
                raise RuntimeError('Final exclusion audit/state mismatch: ' + sid)
        category = source_category(e)
        source_counts[category] += 1
        excluded.append(dict(index[sid], eligible_primary='0', exclusion_reason='instrument_light_leakage',
                             exclusion_source=category, review_note=states[sid]['note'],
                             review_at_utc=states[sid]['updated_at'], review_audit_id=e['audit_id'],
                             review_version=states[sid]['version'], exclusion_stage='light_leak_review_20261013'))
    retained = [r for r in rows if r['sample_id'] not in excluded_ids]
    n = len(retained)
    mod.EXPECTED_ELIGIBLE = n
    mod.TRAIN_COUNT, mod.VAL_COUNT = 7 * n // 10, n // 10
    mod.TEST_COUNT = n - mod.TRAIN_COUNT - mod.VAL_COUNT
    mod.EXPECTED_SPLIT_COUNTS = dict(train=mod.TRAIN_COUNT, val=mod.VAL_COUNT, test=mod.TEST_COUNT)
    mod.DATASET_VERSION, mod.SPLIT_VERSION = VERSION, SPLIT_VERSION
    validation = mod.validate_input_rows(retained)
    clean = mod.assign_splits(retained)
    split_validation = mod.validate_splits(clean)
    if clean != mod.assign_splits(list(reversed(retained))):
        raise RuntimeError('Split not deterministic against input ordering')
    transitions = Counter((index[r['sample_id']]['split'], r['split']) for r in clean)
    old_excluded = mod.read_csv(OLD / 'exclusion_log_v1.0.csv')
    old_artifacts = json.loads((OLD / 'artifact_hashes_v1.0.json').read_text())
    historical_exclusion_sha = sha(OLD / 'exclusion_log_v1.0.csv')
    if historical_exclusion_sha != old_artifacts['exclusion_log_v1.0.csv']:
        raise RuntimeError('Historical exclusion checksum differs')
    if len(old_excluded) != 39 or {r['sample_id'] for r in old_excluded} & set(index):
        raise RuntimeError('Earlier technical exclusions differ')
    all_excluded = [dict(r, exclusion_stage='historical_technical_qc') for r in old_excluded] + excluded
    if len({r['sample_id'] for r in all_excluded}) != len(all_excluded):
        raise RuntimeError('Duplicate merged exclusions')
    if n + len(all_excluded) != 115382:
        raise RuntimeError('Cohort reconciliation failed')
    for r in clean:
        original = index[r['sample_id']]
        for k in original:
            if k not in ('dataset_version', 'split_version', 'split') and original[k] != r[k]:
                raise RuntimeError('Original paired metadata altered: ' + k)

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    build = DEST.with_name(DEST.name + '.building_' + datetime.datetime.now().strftime('%H%M%S%f'))
    build.mkdir(exist_ok=False)
    fields = list(rows[0])
    extra = ['exclusion_stage', 'review_note', 'review_at_utc', 'review_audit_id', 'review_version']
    mod.write_csv(build / 'manifest_frozen.csv', clean, fields)
    mod.write_csv(build / 'exclusions_lightleak.csv', excluded, fields + extra)
    mod.write_csv(build / 'exclusions_all.csv', all_excluded, fields + extra)
    (build / 'excluded_original_directories.txt').write_text(''.join(r['sample_directory'] + '\n' for r in sorted(all_excluded, key=lambda r: r['sample_directory'])), encoding='utf8')
    for split in ('train', 'val', 'test'):
        (build / (split + '_sample_ids.txt')).write_text(''.join(r['sample_id'] + '\n' for r in clean if r['split'] == split), encoding='utf8')
    comparison = [dict(sample_id=r['sample_id'], sample_directory=r['sample_directory'],
                       old_split=index[r['sample_id']]['split'], new_split=r['split']) for r in clean]
    mod.write_csv(build / 'split_mapping.csv', comparison, ['sample_id', 'sample_directory', 'old_split', 'new_split'])
    review_rows = [dict(sample_id=r['sample_id'], sample_directory=index[r['sample_id']]['sample_directory'],
                        final_cohort_status='exclude' if r['sample_id'] in excluded_ids else 'retain',
                        review_decision=states[r['sample_id']]['decision'],
                        reviewed_range_or_broad=r['sample_id'] in completed_ids,
                        approved_high_queue=r['sample_id'] in queues['high']) for r in catalog]
    mod.write_csv(build / 'review_status.csv', review_rows, list(review_rows[0]))
    shutil.copy2(snapshot_path, build / 'review_snapshot.sqlite3')
    shutil.copy2(code_path, build / 'original_split.py')
    shutil.copy2(Path(__file__), build / 'freeze_clean_dataset.py')
    shutil.copy2(report_path, build / 'review_completion.json')
    shutil.copy2(Path(__file__).with_name('finalize_uv_review_queue.py'), build / 'finalize_uv_review_queue.py')
    shutil.copy2(Path(__file__).with_name('apply_uv_review_decisions.py'), build / 'apply_uv_review_decisions.py')
    for name in ('broad', 'high'):
        shutil.copy2(triage / ('queue_' + name + '.json'), build / ('queue_' + name + '.json'))
    summary = dict(dataset_version=VERSION, split_version=SPLIT_VERSION, frozen_at_utc=stamp,
                   parent_manifest_sha256=MANIFEST_SHA, original_split_code_sha256=CODE_SHA,
                   historical_exclusion_sha256=historical_exclusion_sha,
                   review_snapshot_sha256=report['snapshot_sha256'], review_cutoff_audit_id=cutoff,
                   original_split_reproduction_mismatches=0, original_eligible=115343,
                   excluded_lightleak=len(excluded), excluded_historical_technical=len(old_excluded),
                   excluded_total_from_115382=len(all_excluded), retained=n,
                   exclusion_sources=dict(source_counts), review_counts=dict(Counter(r['decision'] for r in states.values())),
                   split_ratio=dict(train=.7, val=.1, test=.2), split_counts=split_validation['split_counts'],
                   split_salt=mod.SPLIT_SALT, split_method='original assign_splits function; deterministic SHA256 ordering',
                   split_count_rounding='train=(7*N)//10; val=N//10; test=N-train-val; parameterization of original fixed counts',
                   split_unit='numeric_folder_record', validation=validation, leakage_checks=split_validation['leakage_checks'],
                   old_to_new={a: {b: transitions[a,b] for b in ('train','val','test')} for a in ('train','val','test')},
                   raw_files_deleted=False, raw_files_modified=False, old_manifest_modified=False,
                   training_started=False, v10_changed=False,
                   retention_policy='All current exclude_light_leak removed as paired captures. Completed first25000 and broad unmarked kept. Remaining unreviewed retained by exclusion-only policy, not individually reviewed.',
                   identity_limit='Folder-derived subject_id, not independently verified cross-capture person identity.',
                   old_checkpoint_note='Check old_to_new before evaluation. Old checkpoints are not trained on this cleaned version; new frozen experiments require the agreed retraining protocol.',
                   source_images_rehashed_this_run=False,
                   source_hash_note='Original frozen full-file hashes preserved and cross-split checked; raw image bytes were not re-read during this metadata-only freeze.')
    save_json(build / 'summary.json', summary)
    if sha(old_manifest) != MANIFEST_SHA or sha(snapshot_path) != report['snapshot_sha256']:
        raise RuntimeError('Source changed during freeze')
    if sha(build / 'review_snapshot.sqlite3') != report['snapshot_sha256']:
        raise RuntimeError('Snapshot copy differs')
    # 校验已写入清单
    saved = mod.read_csv(build / 'manifest_frozen.csv')
    if saved != clean:
        raise RuntimeError('Manifest roundtrip mismatch')
    mod.validate_input_rows(saved)
    mod.validate_splits(saved)
    hashes = {p.name: sha(p) for p in sorted(build.iterdir()) if p.is_file()}
    save_json(build / 'SHA256SUMS.json', hashes)
    save_json(build / 'FROZEN.json', dict(version=VERSION, frozen_at_utc=stamp,
                                        hashes_sha256=sha(build / 'SHA256SUMS.json'), files=len(hashes),
                                        changes_require_new_version=True))
    build.rename(DEST)
    archive = DEST.with_name(DEST.name + '.zip')
    if archive.exists():
        raise RuntimeError('Archive already exists')
    with zipfile.ZipFile(str(archive), 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(DEST.iterdir()):
            z.write(str(p), arcname=DEST.name + '/' + p.name)
    print(json.dumps(dict(status='frozen', directory=str(DEST), archive=str(archive), archive_sha256=sha(archive),
                          archive_bytes=archive.stat().st_size, summary=summary), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review-completion-report', type=Path, required=True)
    args = p.parse_args()
    execute(args.review_completion_report)


if __name__ == '__main__':
    main()
