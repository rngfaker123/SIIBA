"""Apply the user's explicit negative-only review convention to rows1..14000.

Never extends to page141, changes an existing keep/exclusion, deletes files, or
relabels subjects beyond the exact sample range. One auditable transaction.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import time

ROOT=Path('/study/siiba')
OUT=ROOT/'output/uv_dataset_quality_review_20261013'
MANIFEST='45aa180aec1fdb50c4797ef9053be0afc11790035e446b6f76802dcfe5fe9e6a'
CATALOG='3b4aee2e48c3d74b73a2be6c6a1cbc4b027e3a0b0ada3846f85ab990273086e0'
BATCH='user_confirmed_default_keep_rows00001_14000_20261013'
AUTHORIZATION='User confirmed first200 means people; completed page140; unmarked201..14000 are keep; later explicitly instructed unmarked first200 also keep. Existing exclusions and page141 onward stay unchanged.'


def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def audit(conn,event,sid,old,new,payload):
    conn.execute('INSERT INTO audit_events(at_utc,event,sample_id,old_state,new_state,payload) VALUES(?,?,?,?,?,?)',
                 (utc(),event,sid,encoded(old) if old is not None else None,
                  encoded(new) if new is not None else None,encoded(payload)))


def protected_digest(conn):
    h=hashlib.sha256()
    query='''SELECT r.sample_id,r.decision,r.note,r.updated_at,r.version
             FROM reviews r LEFT JOIN bulk_target_ids t USING(sample_id)
             WHERE t.sample_id IS NULL ORDER BY r.sample_id'''
    for row in conn.execute(query):h.update((encoded(list(row))+'\n').encode('utf8'))
    return h.hexdigest()


def apply_transaction(conn,backup=None):
    started=time.monotonic();conn.execute('BEGIN IMMEDIATE')
    try:
        previous=conn.execute("SELECT payload FROM audit_events WHERE event='human_default_keep_range_complete' AND sample_id IS NULL ORDER BY audit_id DESC").fetchall()
        for row in previous:
            result=json.loads(row[0])
            if result.get('batch_id')==BATCH:
                conn.execute('COMMIT')
                return dict(result,already_applied=True)
        rows=conn.execute('''SELECT c.row_index,r.* FROM reviews r JOIN catalog c USING(sample_id)
                             WHERE c.row_index BETWEEN 1 AND 14000 AND r.decision='unreviewed'
                             ORDER BY c.row_index''').fetchall()
        conn.execute('CREATE TEMP TABLE bulk_target_ids(sample_id TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO bulk_target_ids VALUES(?)',[(r['sample_id'],) for r in rows])
        before=protected_digest(conn)
        ids_sha=hashlib.sha256(encoded([r['sample_id'] for r in rows]).encode('utf8')).hexdigest()
        stamp=utc()
        payload=dict(source='explicit_user_chat_authorization_negative_only_review',batch_id=BATCH,
                     from_index=1,through_index=14000,through_page=140,authorization=AUTHORIZATION,
                     manifest_sha256=MANIFEST,catalog_sha256=CATALOG,original_files_changed=False)
        for row in rows:
            old={k:row[k] for k in ('decision','note','updated_at','version')}
            new=dict(old,decision='keep',updated_at=stamp,version=old['version']+1)
            cursor=conn.execute("UPDATE reviews SET decision='keep',updated_at=?,version=version+1 WHERE sample_id=? AND decision='unreviewed' AND version=?",
                                (stamp,row['sample_id'],old['version']))
            if cursor.rowcount!=1:raise RuntimeError('Unexpected concurrent/conditional update difference')
            audit(conn,'human_decision',row['sample_id'],old,new,payload)
        after=protected_digest(conn)
        if before!=after:raise RuntimeError('Protected reviews changed; rolling back')
        counts=dict(conn.execute('''SELECT r.decision,COUNT(*) FROM reviews r JOIN catalog c USING(sample_id)
                                    WHERE c.row_index BETWEEN 1 AND 14000 GROUP BY r.decision'''))
        if counts.get('unreviewed',0):raise RuntimeError('Unreviewed records remain in approved range')
        result=dict(status='complete',batch_id=BATCH,from_index=1,through_index=14000,through_page=140,
                    changed=len(rows),first200_changed=sum(r['row_index']<=200 for r in rows),
                    range_counts=counts,changed_ids_sha256=ids_sha,protected_state_sha256=after,
                    protected_states_unchanged=True,prior_notes_preserved=True,backup=backup,
                    manifest_sha256=MANIFEST,catalog_sha256=CATALOG,authorization=AUTHORIZATION,
                    applied_at_utc=stamp,transaction_seconds_before_commit=time.monotonic()-started)
        audit(conn,'human_default_keep_range_complete',None,None,None,result)
        conn.execute('COMMIT')
        return result
    except Exception:
        conn.execute('ROLLBACK');raise


def self_test():
    conn=sqlite3.connect(':memory:',isolation_level=None);conn.row_factory=sqlite3.Row
    conn.executescript('''CREATE TABLE catalog(sample_id TEXT PRIMARY KEY,row_index INTEGER UNIQUE);
        CREATE TABLE reviews(sample_id TEXT PRIMARY KEY,decision TEXT,note TEXT,updated_at TEXT,version INTEGER);
        CREATE TABLE audit_events(audit_id INTEGER PRIMARY KEY AUTOINCREMENT,at_utc TEXT,event TEXT,sample_id TEXT,old_state TEXT,new_state TEXT,payload TEXT);''')
    for index,decision in [(1,'unreviewed'),(2,'exclude_light_leak'),(200,'unreviewed'),(201,'keep'),(14000,'unreviewed'),(14001,'unreviewed'),(15000,'exclude_light_leak')]:
        sid='SMP_%05d'%index;conn.execute('INSERT INTO catalog VALUES(?,?)',(sid,index))
        conn.execute('INSERT INTO reviews VALUES(?,?,?,?,?)',(sid,decision,'existing note','old timestamp',7))
    result=apply_transaction(conn)
    assert result['changed']==3 and result['first200_changed']==2
    states={r['sample_id']:dict(r) for r in conn.execute('SELECT * FROM reviews')}
    assert states['SMP_14001']['decision']=='unreviewed' and states['SMP_14001']['version']==7
    assert states['SMP_00002']['decision']=='exclude_light_leak' and states['SMP_00002']['version']==7
    assert states['SMP_00201']['decision']=='keep' and states['SMP_00201']['version']==7
    assert states['SMP_14000']['decision']=='keep' and states['SMP_14000']['note']=='existing note'
    assert conn.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0]==4
    assert apply_transaction(conn)['already_applied']
    assert conn.execute('SELECT COUNT(*) FROM audit_events').fetchone()[0]==4
    conn.close();print(encoded(dict(self_test='passed',checks=6,real_database_accessed=False)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test();return
    db=OUT/'review.sqlite3'
    if not db.is_file() or db.is_symlink():raise RuntimeError('Existing regular database required')
    conn=sqlite3.connect(db.as_uri()+'?mode=rw',uri=True,isolation_level=None,timeout=20)
    conn.row_factory=sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON');conn.execute('PRAGMA busy_timeout=20000');conn.execute('PRAGMA synchronous=FULL')
    for key,wanted in [('manifest_sha256',MANIFEST),('catalog_sha256',CATALOG)]:
        value=conn.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
        if not value or json.loads(value[0])!=wanted:raise RuntimeError('Wrong frozen provenance')
    if conn.execute('SELECT COUNT(*) FROM catalog').fetchone()[0]!=115343:raise RuntimeError('Wrong cohort')
    if conn.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise RuntimeError('Database quick check failed')
    if not args.apply:
        print(encoded(dict(dry_run=True,authorized_range=[1,14000],current_unreviewed=conn.execute("SELECT COUNT(*) FROM reviews r JOIN catalog c USING(sample_id) WHERE c.row_index BETWEEN 1 AND 14000 AND r.decision='unreviewed'").fetchone()[0])))
        conn.close();return
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backups=OUT/'recovery_backups';backups.mkdir(exist_ok=True)
    backup=backups/('review_before_default_keep_1_14000_'+stamp+'.sqlite3')
    if backup.exists():raise RuntimeError('Backup already exists')
    with sqlite3.connect(str(backup)) as dest:
        conn.backup(dest)
        if dest.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise RuntimeError('Backup invalid')
    result=apply_transaction(conn,str(backup))
    if conn.execute('PRAGMA foreign_key_check').fetchall():raise RuntimeError('Post-commit foreign key check failed')
    conn.close()
    reports=OUT/'bulk_review_reports';reports.mkdir(exist_ok=True)
    with (reports/(BATCH+'_'+stamp+'.json')).open('x',encoding='utf8') as f:f.write(encoded(result)+'\n')
    print(encoded(result),flush=True)


if __name__=='__main__':main()
