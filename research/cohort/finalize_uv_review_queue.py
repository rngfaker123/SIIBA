"""Finalize explicitly completed reviews without deleting images or exclusions.

The user confirmed negative-only review for catalog rows 1..25000 and the
entire frozen BROAD queue. Only still-unreviewed records in that union become
keep. A consistent post-commit snapshot is the handoff for dataset freezing.
Default execution is read-only; --apply is required for the one-time batch.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time

from apply_uv_review_decisions import audit, encoded, utc, protected_digest, MANIFEST, CATALOG

ROOT = Path('/study/siiba')
REVIEW = ROOT / 'output/uv_dataset_quality_review_20261013'
TRIAGE = ROOT / 'output/uv_lightleak_triage_20261013_v1'
QUEUE_SHA = '9c4c50c7ea0c4d1bf5b595c3cf618a3960be60b06d01996d158291f08572bead'
SNAPSHOT_SHA = '4bfb30ea7f53d13cdcbf5c6d67b092d0109ad60e520d52324416e0c84134574f'
BATCH = 'user_completed_rows1_25000_and_broad12442_20261013'
EVENT = 'human_completed_broad_and_range_default_keep'
AUTH = ('User confirmed catalog rows 1..25000 were reviewed, with unmarked meaning keep; '
        'then confirmed the entire frozen BROAD queue finished, with unmarked and explicit '
        'keeps retained. All existing exclusions and explicit keeps must be preserved.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_candidates():
    queue_path = TRIAGE / 'queue_broad.json'
    snapshot_path = TRIAGE / 'snapshot.json'
    if sha(queue_path) != QUEUE_SHA or sha(snapshot_path) != SNAPSHOT_SHA:
        raise RuntimeError('Frozen BROAD queue/snapshot hash mismatch')
    rows = json.loads(queue_path.read_text(encoding='utf8'))
    snapshot = json.loads(snapshot_path.read_text(encoding='utf8'))
    entries = snapshot['entries']
    by_id = {r['sample_id']: r for r in entries}
    summary = json.loads((TRIAGE / 'summary.json').read_text(encoding='utf8'))
    if len(rows) != 12442 or len({r['sample_id'] for r in rows}) != 12442:
        raise RuntimeError('Wrong BROAD queue size or duplicate IDs')
    if len(entries) != 115343 or len(by_id) != 115343:
        raise RuntimeError('Wrong source snapshot size')
    if summary['broad_additional_candidates'] != 12442 or summary['source_snapshot_sha256'] != SNAPSHOT_SHA:
        raise RuntimeError('Summary provenance mismatch')
    for row in rows:
        original = by_id.get(row['sample_id'])
        if (not original or row['queue'] != 'broad' or row['row_index'] <= 25000
                or row['decision'] != 'unreviewed'
                or not summary['broad_threshold'] <= row['score'] < summary['high_threshold']
                or original['row_index'] != row['row_index']
                or original['decision'] != 'unreviewed'
                or original['uv_sha256'] != row['uv_sha256']):
            raise RuntimeError('Invalid frozen BROAD candidate')
    return rows, dict(queue_sha256=QUEUE_SHA, source_snapshot_sha256=SNAPSHOT_SHA,
                      source_snapshot_at=summary['snapshot_at'], broad_threshold=summary['broad_threshold'],
                      high_threshold=summary['high_threshold'])


def validate_cohort(conn, expected_size=115343):
    for key, expected in [('manifest_sha256', MANIFEST), ('catalog_sha256', CATALOG)]:
        row = conn.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        if not row or json.loads(row[0]) != expected:
            raise RuntimeError('Wrong current catalog provenance')
    bounds = conn.execute('SELECT COUNT(*),COUNT(DISTINCT row_index),MIN(row_index),MAX(row_index) FROM catalog').fetchone()
    if tuple(bounds) != (expected_size, expected_size, 1, expected_size):
        raise RuntimeError('Wrong catalog size or row bounds')
    if conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0] != expected_size:
        raise RuntimeError('Wrong review size')
    if conn.execute('SELECT COUNT(*) FROM catalog JOIN reviews USING(sample_id)').fetchone()[0] != expected_size:
        raise RuntimeError('Catalog/reviews mismatch')
    if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or conn.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Database integrity failure')


def find_completed(conn):
    for row in conn.execute('SELECT payload FROM audit_events WHERE event=? AND sample_id IS NULL ORDER BY audit_id DESC', (EVENT,)):
        result = json.loads(row[0])
        if result.get('batch_id') == BATCH:
            return result
    return None


def prepare(conn, rows):
    for name in ('approved_broad_ids', 'approved_scope_ids', 'bulk_target_ids'):
        conn.execute('DROP TABLE IF EXISTS temp.' + name)
    conn.execute('CREATE TEMP TABLE approved_broad_ids(sample_id TEXT PRIMARY KEY,row_index INTEGER UNIQUE)')
    for row in rows:
        if row['row_index'] <= 25000:
            raise RuntimeError('BROAD must not overlap the initial reviewed range')
    conn.executemany('INSERT INTO approved_broad_ids VALUES(?,?)', [(r['sample_id'], r['row_index']) for r in rows])
    if conn.execute('''SELECT COUNT(*) FROM approved_broad_ids b JOIN catalog c USING(sample_id)
                        JOIN reviews r USING(sample_id) WHERE c.row_index=b.row_index''').fetchone()[0] != len(rows):
        raise RuntimeError('BROAD IDs do not match live catalog/reviews')
    conn.execute('CREATE TEMP TABLE approved_scope_ids(sample_id TEXT PRIMARY KEY,scope TEXT)')
    conn.execute("INSERT INTO approved_scope_ids SELECT sample_id,'completed_initial_25000' FROM catalog WHERE row_index BETWEEN 1 AND 25000")
    conn.execute("INSERT INTO approved_scope_ids SELECT sample_id,'completed_broad_queue' FROM approved_broad_ids")


def counts(conn):
    return dict(conn.execute('SELECT decision,COUNT(*) FROM reviews GROUP BY decision'))


def scope_counts(conn):
    return [dict(r) for r in conn.execute('''SELECT s.scope,r.decision,COUNT(*) AS count FROM approved_scope_ids s
        JOIN reviews r USING(sample_id) GROUP BY s.scope,r.decision ORDER BY s.scope,r.decision''')]


def transaction(conn, rows, meta, backup=None, snapshot_path=None):
    start = time.monotonic()
    conn.execute('BEGIN IMMEDIATE')
    try:
        previous = find_completed(conn)
        if previous:
            conn.execute('COMMIT')
            return dict(previous, already_applied=True)
        prepare(conn, rows)
        targets = conn.execute('''SELECT r.*,c.row_index,s.scope FROM approved_scope_ids s
            JOIN catalog c USING(sample_id) JOIN reviews r USING(sample_id)
            WHERE r.decision='unreviewed' ORDER BY c.row_index''').fetchall()
        conn.execute('CREATE TEMP TABLE bulk_target_ids(sample_id TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO bulk_target_ids VALUES(?)', [(r['sample_id'],) for r in targets])
        protected_before, before_counts, before_scope = protected_digest(conn), counts(conn), scope_counts(conn)
        stamp = utc()
        changed_sources = {'completed_initial_25000': 0, 'completed_broad_queue': 0}
        for row in targets:
            old = {k: row[k] for k in ('decision', 'note', 'version', 'updated_at')}
            new = dict(old, decision='keep', version=old['version'] + 1, updated_at=stamp)
            updated = conn.execute("UPDATE reviews SET decision='keep',version=version+1,updated_at=? WHERE sample_id=? AND decision='unreviewed' AND version=?",
                                   (stamp, row['sample_id'], old['version'])).rowcount
            if updated != 1:
                raise RuntimeError('Conditional update mismatch')
            actual = conn.execute('SELECT decision,note,version,updated_at FROM reviews WHERE sample_id=?', (row['sample_id'],)).fetchone()
            if dict(actual) != new:
                raise RuntimeError('Target state or preserved note differs after update')
            source = ('explicit_user_completed_initial25000_default_keep' if row['scope'] == 'completed_initial_25000'
                      else 'explicit_user_completed_broad_default_keep')
            audit(conn, 'human_decision', row['sample_id'], old, new,
                  dict(meta, source=source, batch_id=BATCH, authorization=AUTH, catalog_index=row['row_index'],
                       individual_button_click=False, review_basis='user_confirmed_negative_only_visual_review',
                       manifest_sha256=MANIFEST, catalog_sha256=CATALOG, original_files_changed=False))
            changed_sources[row['scope']] += 1
        protected_after, after_counts = protected_digest(conn), counts(conn)
        if protected_before != protected_after:
            raise RuntimeError('Protected state changed: rollback')
        if (after_counts.get('exclude_light_leak', 0) != before_counts.get('exclude_light_leak', 0)
                or after_counts.get('keep', 0) != before_counts.get('keep', 0) + len(targets)
                or after_counts.get('unreviewed', 0) != before_counts.get('unreviewed', 0) - len(targets)):
            raise RuntimeError('Global decision reconciliation failed')
        if conn.execute("SELECT COUNT(*) FROM approved_scope_ids JOIN reviews USING(sample_id) WHERE decision='unreviewed'").fetchone()[0]:
            raise RuntimeError('Unreviewed remains in completed scope')
        result = dict(meta, status='complete', batch_id=BATCH, authorization=AUTH, changed=len(targets),
                      changed_by_scope=changed_sources, initial_range=[1, 25000], broad_queue_total=len(rows),
                      all_counts_before=before_counts, all_counts_after=after_counts,
                      scope_counts_before=before_scope, scope_counts_after=scope_counts(conn),
                      changed_ids_sha256=hashlib.sha256(encoded([r['sample_id'] for r in targets]).encode('utf8')).hexdigest(),
                      protected_states_unchanged=True, protected_state_sha256=protected_after, prior_notes_preserved=True,
                      original_files_changed=False, manifest_sha256=MANIFEST, catalog_sha256=CATALOG,
                      backup=backup, snapshot_path=snapshot_path, applied_at_utc=stamp,
                      transaction_seconds_before_commit=time.monotonic() - start)
        audit(conn, EVENT, None, None, None, result)
        result['completion_audit_id'] = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.execute('COMMIT')
        return result
    except Exception:
        conn.execute('ROLLBACK')
        raise


def backup_database(conn, path):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Will not overwrite backup/snapshot')
    dest = sqlite3.connect(str(path))
    try:
        conn.backup(dest)
        if dest.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or dest.execute('PRAGMA foreign_key_check').fetchall():
            raise RuntimeError('Backup/snapshot integrity check failed')
    finally:
        dest.close()
    return sha(path)


def freeze_snapshot(conn, path):
    # Pin a consistent read snapshot. Later browser writes cannot alter this copy.
    conn.execute('BEGIN')
    try:
        max_audit = conn.execute('SELECT COALESCE(MAX(audit_id),0) FROM audit_events').fetchone()[0]
        frozen_counts = counts(conn)
        digest = backup_database(conn, path)
        conn.execute('COMMIT')
        return dict(snapshot_path=str(path), snapshot_sha256=digest, snapshot_max_audit_id=max_audit,
                    snapshot_counts=frozen_counts, snapshot_created_at_utc=utc())
    except Exception:
        conn.execute('ROLLBACK')
        raise


def fixture():
    conn = sqlite3.connect(':memory:', isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript('''CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE catalog(sample_id TEXT PRIMARY KEY,row_index INTEGER UNIQUE);
        CREATE TABLE reviews(sample_id TEXT PRIMARY KEY,decision TEXT,note TEXT,updated_at TEXT,version INTEGER);
        CREATE TABLE audit_events(audit_id INTEGER PRIMARY KEY AUTOINCREMENT,at_utc TEXT,event TEXT,sample_id TEXT,old_state TEXT,new_state TEXT,payload TEXT);''')
    items = [(1, 'unreviewed'), (200, 'exclude_light_leak'), (14001, 'unreviewed'), (25000, 'keep'),
             (25001, 'unreviewed'), (25002, 'keep'), (25003, 'exclude_light_leak'), (25004, 'unreviewed'),
             (115343, 'exclude_light_leak')]
    for index, decision in items:
        sid = 'SMP_%06d' % index
        conn.execute('INSERT INTO catalog VALUES(?,?)', (sid, index))
        conn.execute('INSERT INTO reviews VALUES(?,?,?,?,?)', (sid, decision, 'preserve note', 'old', 7))
    rows = [dict(sample_id='SMP_%06d' % n, row_index=n) for n in (25001, 25002, 25003)]
    return conn, rows


def self_test():
    checks = 0
    conn, rows = fixture()
    result = transaction(conn, rows, {})
    assert result['changed'] == 3 and result['changed_by_scope'] == {'completed_initial_25000': 2, 'completed_broad_queue': 1}
    checks += 1
    states = {r['sample_id']: dict(r) for r in conn.execute('SELECT * FROM reviews')}
    assert states['SMP_025004']['decision'] == 'unreviewed' and states['SMP_025004']['version'] == 7
    checks += 1
    assert all(states['SMP_%06d' % n]['version'] == 7 for n in (200, 25000, 25002, 25003, 115343))
    assert all(r['note'] == 'preserve note' for r in states.values())
    checks += 1
    audit_rows = list(conn.execute("SELECT payload FROM audit_events WHERE event='human_decision' ORDER BY audit_id"))
    assert [json.loads(r[0])['source'] for r in audit_rows] == ['explicit_user_completed_initial25000_default_keep'] * 2 + ['explicit_user_completed_broad_default_keep']
    checks += 1
    assert transaction(conn, rows, {})['already_applied'] and conn.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0] == 4
    checks += 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'snapshot.sqlite3'
        frozen = freeze_snapshot(conn, path)
        conn.execute("UPDATE reviews SET decision='keep' WHERE sample_id='SMP_025004'")
        assert sha(path) == frozen['snapshot_sha256']
        copied = sqlite3.connect(str(path))
        try:
            assert copied.execute("SELECT decision FROM reviews WHERE sample_id='SMP_025004'").fetchone()[0] == 'unreviewed'
        finally:
            copied.close()
        checks += 1
        try:
            freeze_snapshot(conn, path)
            raise AssertionError('Overwrite permitted')
        except RuntimeError:
            checks += 1
    for bad in ([dict(sample_id='SMP_025001', row_index=25002)], [dict(sample_id='SMP_000001', row_index=1)]):
        c, unused = fixture()
        try:
            transaction(c, bad, {})
            raise AssertionError('Invalid candidate accepted')
        except RuntimeError:
            assert c.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0] == 0
            checks += 1
        c.close()
    c, r = fixture()
    c.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON audit_events BEGIN SELECT RAISE(ABORT,'test audit failure'); END")
    try:
        transaction(c, r, {})
        raise AssertionError('Audit failure accepted')
    except sqlite3.IntegrityError:
        assert c.execute("SELECT COUNT(*) FROM reviews WHERE decision='unreviewed'").fetchone()[0] == 4
        assert c.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0] == 0
        checks += 1
    c.close()
    c, r = fixture()
    c.execute("CREATE TRIGGER alter_outside AFTER UPDATE ON reviews WHEN NEW.sample_id='SMP_000001' BEGIN UPDATE reviews SET note='unexpected' WHERE sample_id='SMP_025004'; END")
    try:
        transaction(c, r, {})
        raise AssertionError('Protected record mutation accepted')
    except RuntimeError:
        assert c.execute("SELECT note FROM reviews WHERE sample_id='SMP_025004'").fetchone()[0] == 'preserve note'
        assert c.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0] == 0
        checks += 1
    c.close()
    c, r = fixture()
    c.execute("CREATE TRIGGER alter_note AFTER UPDATE ON reviews WHEN NEW.sample_id='SMP_000001' BEGIN UPDATE reviews SET note='unexpected' WHERE sample_id=NEW.sample_id; END")
    try:
        transaction(c, r, {})
        raise AssertionError('Target note mutation accepted')
    except RuntimeError:
        assert c.execute("SELECT note FROM reviews WHERE sample_id='SMP_000001'").fetchone()[0] == 'preserve note'
        assert c.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0] == 0
        checks += 1
    c.close()
    c, unused = fixture()
    c.executemany('INSERT INTO settings VALUES(?,?)', [('manifest_sha256', encoded(MANIFEST)), ('catalog_sha256', encoded(CATALOG))])
    try:
        validate_cohort(c, expected_size=9)
        raise AssertionError('Noncontiguous catalog range accepted')
    except RuntimeError:
        checks += 1
    c.execute("UPDATE settings SET value='\"wrong\"' WHERE key='manifest_sha256'")
    try:
        validate_cohort(c, expected_size=9)
        raise AssertionError('Wrong provenance accepted')
    except RuntimeError:
        checks += 1
    c.close()
    conn.close()
    print(encoded(dict(self_test='passed', checks=checks, real_database_accessed=False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows, meta = load_candidates()
    db = REVIEW / 'review.sqlite3'
    if not db.is_file() or db.is_symlink():
        raise RuntimeError('Existing regular review database required')
    conn = sqlite3.connect(db.as_uri() + ('?mode=rw' if args.apply else '?mode=ro'), uri=True, isolation_level=None, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=20000')
    validate_cohort(conn)
    previous = find_completed(conn)
    report_path = TRIAGE / (BATCH + '.json')
    if previous:
        # Never replace a frozen snapshot on a rerun, even if new live edits exist.
        if not report_path.is_file():
            raise RuntimeError('Batch committed but final report missing; manual recovery required, no new snapshot made')
        result = json.loads(report_path.read_text(encoding='utf8'))
        if (result.get('batch_id') != BATCH or sha(result['snapshot_path']) != result['snapshot_sha256']
                or result['snapshot_path'] != previous['snapshot_path']):
            raise RuntimeError('Frozen completion report/snapshot mismatch')
        print(encoded(dict(result, already_applied=True, live_database_may_have_later_edits=True)))
        conn.close()
        return
    if not args.apply:
        prepare(conn, rows)
        print(encoded(dict(dry_run=True, initial_range=[1, 25000], broad_queue_total=len(rows),
                           scope_counts=scope_counts(conn), all_counts=counts(conn), metadata=meta)))
        conn.close()
        return
    if report_path.exists():
        raise RuntimeError('Unexpected preexisting report; refusing overwrite')
    conn.execute('PRAGMA synchronous=FULL')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backups = REVIEW / 'recovery_backups'
    backups.mkdir(exist_ok=True)
    snapshots = REVIEW / 'frozen_review_snapshots'
    snapshots.mkdir(exist_ok=True)
    backup = backups / ('review_before_final_broad_' + stamp + '.sqlite3')
    snapshot = snapshots / (BATCH + '_' + stamp + '.sqlite3')
    backup_sha = backup_database(conn, backup)
    result = transaction(conn, rows, meta, str(backup), str(snapshot))
    if result.get('already_applied'):
        raise RuntimeError('Concurrent batch completed; no duplicate snapshot created, rerun read-only')
    result.update(freeze_snapshot(conn, snapshot))
    result.update(backup_sha256=backup_sha, report_path=str(report_path))
    conn.close()
    with report_path.open('x', encoding='utf8') as stream:
        stream.write(encoded(result) + '\n')
    print(encoded(result), flush=True)


if __name__ == '__main__':
    main()
