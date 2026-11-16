from pathlib import Path
import json,hashlib,os,sys,datetime,collections
B=Path('/study/siiba/experiments');R=B/'ksg_group_fraction_20261102_v1';I=B/'ita_stratified_mi_test_20261026_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v):
    z=dict(v);h=z.pop('record_sha256');assert hashlib.sha256((json.dumps(z,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest()==h
    return v
def save(name,v):
    p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
req=read(I/'request.json');catalog=read(req['catalog']['path']);assert sha(req['catalog']['path'])==req['catalog']['sha256'];rows=catalog['rows'];assert len(rows)==21899
original={};source_manifests={}
for s in ['very_light','light','intermediate','tan','brown','dark']:
    root=I/'strata'/s;m=read(root/'mi_input.json');p=check(read(root/'mi_run/preparation.json'))
    assert len(m['rows'])==len(p['features']);source_manifests[s]={'manifest_path':str(root/'mi_input.json'),'manifest_sha':sha(root/'mi_input.json'),'preparation_path':str(root/'mi_run/preparation.json'),'preparation_sha':sha(root/'mi_run/preparation.json')}
    for row,feature in zip(m['rows'],p['features']):
        sid=row['sample_id'];assert sid==feature['sample_id'] and sid not in original
        original[sid]=dict(row=row,feature=feature,out=str(root/'mi_run'),stratum=s)
eligible=[];excluded=[];extra=[];reasons=collections.Counter()
for index,row in enumerate(rows):
    o=check(read(I/'outcomes'/('%06d.json'%index)));assert o['index']==index and o['sample_id']==row['sample_id']
    counts=o['counts_native']
    if counts is None and o['reason'].startswith('ITA_unavailable:'):
        ref=B/'three_region_test_parallel_20261025_v1/masks/refs'/('%06d.json'%index)
        if ref.exists():
            v=check(read(ref));p=Path(v['receipt']['path']);assert sha(p)==v['receipt']['sha256'];receipt=check(read(p));counts=receipt.get('counts_native');native_status=receipt['status']
        else:native_status='missing_reference'
    else:native_status='complete' if counts else o['reason']
    if counts and min(counts.get(k,0) for k in ['P','D','O','F'])>=1000:
        v=dict(index=index,sample_id=row['sample_id'],subject_id=row['subject_id'],counts=counts,outcome_sha256=sha(I/'outcomes'/('%06d.json'%index)))
        if row['sample_id'] in original:v['prepared']=original[row['sample_id']]
        else:v['native_without_ITA']=dict(outcome=o,catalog_row=row);extra.append(v)
        eligible.append(v)
    else:
        reason='native_region_below1000' if counts else native_status;reasons[reason]+=1;excluded.append(dict(index=index,sample_id=row['sample_id'],reason=reason,counts=counts))
    if (index+1)%2000==0:print('AUDIT',index+1,len(eligible),flush=True)
assert len({v['sample_id'] for v in eligible})==len(eligible)
save('cohort_audit.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),catalogue=req['catalog'],total=len(rows),eligible_N=len(eligible),prepared=len(original),extra_without_prepared=len(extra),eligible=eligible,excluded=excluded,excluded_reasons=dict(reasons),source_manifests=source_manifests,protocol_sha256=sha(R/'protocol.json')))
print('AUDIT_COMPLETE',len(eligible),'PREPARED',len(original),'EXTRA',len(extra),json.dumps(dict(reasons)),flush=True)
