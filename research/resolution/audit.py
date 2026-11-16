"""Independent saved-artifact acceptance. Does not run detectors, models or KSG."""
import csv,math
from common import *

def run():
 q,runner,est,cat,cached=context();check_host(q);no_failures();os.sched_setaffinity(0,range(48,56))
 c=checked(read(ROOT/'cohort.json'));assert c['N']==1000 and c['status']=='complete'
 selected=[]
 for index,item in enumerate(c['screening']):
  assert item['index']==index
  r=checked(read(verify(item['receipt'])));assert r['request']==desc(ROOT/'request.json')
  for d in r['original_bindings'].values():verify(d)
  if r['eligible']:
   assert all(r['counts'][str(f)][d]>=1000 for f in FACTORS for d in DOMAINS);selected.append(index)
  if len(selected)==1000:assert index==c['screened_prefix']-1
 assert selected==c['selected_indices'] and len(selected)==len(set(selected))==1000
 prep=checked(read(ROOT/'preparation_complete.json'));assert prep['cohort']==desc(ROOT/'cohort.json') and prep['N']==1000
 pools={name:np.load(verify(d),mmap_mode='r',allow_pickle=False) for name,d in prep['pools'].items()}
 assert len(pools)==192 and all(a.shape==(1000,1000,3) and a.dtype==np.float64 for a in pools.values())
 coordinate_memberships=0;pool_group_matches=0
 from prepare import load_mask,partition
 for rank,index in enumerate(selected):
  r=checked(read(group_path(index)));screen=checked(read(verify(r['candidate'])));row=screen['row'];assert row==source_row(cat,index)
  proof=legacy(read(verify(screen['native_proof'])));assert proof['sample_id']==row['sample_id']
  for f in FACTORS:
   masks={d:load_mask(verify(proof['masks'][d] if f==1 else screen['lower'][str(f)]['masks'][d]),expected_count=screen['counts'][str(f)][d]) for d in DOMAINS};partition(masks)
   with np.load(verify(r['feature_files'][str(f)]),allow_pickle=False) as z:
    for d in DOMAINS:
     ids=z['indices_'+d];assert ids.shape==(3,1000) and ids.dtype==np.uint64
     for si,seed in enumerate(SEEDS):
      assert len(np.unique(ids[si]))==1000 and masks[d].ravel()[ids[si]].all();coordinate_memberships+=1000
      payload=([runner.VERSION,seed,row['sample_id'],d] if f==1 else ['native_resolution_sampling_v1',seed,row['sample_id'],f,d])
      h=hashlib.sha256(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()).digest()
      rng=np.random.Generator(np.random.PCG64(int.from_bytes(h[:16],'big')));candidates=np.flatnonzero(masks[d].ravel())
      assert np.array_equal(ids[si],candidates[rng.choice(len(candidates),1000,replace=False)])
      for mi,m in enumerate(MODS):
       assert np.array_equal(pools[pool_name(f,seed,d,m)][rank],z['values_'+d][si,mi]);pool_group_matches+=1
   del masks
  if (rank+1)%25==0:state(phase='independent_saved_artifact_audit',groups=rank+1,total=1000)
 hashes={}
 for name,a in pools.items():
  for n in BUDGETS:hashes[name,n]=array_sha(np.ascontiguousarray(a[:,:n,:].reshape(-1,3)))
 records={}
 for p in (ROOT/'results').glob('*.json'):
  r=checked(read(p));t=r['task'];k=(t['budget'],t['factor'],t['seed'],t['domain'],t['x'],t['y'])
  assert k not in records and r['groups']==1000 and r['observations']==1000*t['budget'] and r['request']==desc(ROOT/'request.json') and r['cohort']==desc(ROOT/'cohort.json')
  d=r['diagnostic'];v=r['result'];assert r['status']=='complete' and not r['reused'] and math.isfinite(v['mi_bits']) and v['k']==5 and v['n']==r['observations']
  assert d['cuda_executed'] and d['cpu_validation']['passed'] and d['cpu_validation']['full_reference_used'] and len(d['cpu_validation']['query_indices'])==512
  assert d['cpu_validation']['knn_distances_exact'] and d['cpu_validation']['x_counts_exact'] and d['cpu_validation']['y_counts_exact']
  for label,m in [('x',t['x']),('y',t['y'])]:assert d['arrays_sha256'][label]==hashes[pool_name(t['factor'],t['seed'],t['domain'],m),t['budget']]
  records[k]=r
 expected={(n,f,s,d,x,y) for n in BUDGETS for f in FACTORS for s in SEEDS for d in DOMAINS for x,y in PAIRS};assert set(records)==expected and len(records)==864
 s=checked(read(ROOT/'summary.json'));assert len(s['cells'])==288
 with (ROOT/'summary.csv').open(newline='',encoding='utf8') as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==288
 for r,cr in zip(s['cells'],csvrows):
  vals=np.array([records[r['budget'],r['factor'],seed,r['domain'],r['x'],r['y']]['result']['mi_bits'] for seed in SEEDS])
  assert np.array_equal(vals,np.array(r['draws'])) and abs(vals.mean()-r['mean_bits'])<2e-15 and abs(vals.std(ddof=1)-r['sampling_sd_bits_ddof1'])<2e-15
  assert float(cr['mean_bits'])==r['mean_bits'] and float(cr['sampling_sd_bits_ddof1'])==r['sampling_sd_bits_ddof1'] and int(cr['N_groups'])==1000
 completion=checked(read(ROOT/'scientific_completion.json'))
 for d in completion['figures'].values():verify(d)
 for d in q['sources'].values():verify(d)
 for d in q['bindings'].values():verify(d)
 save(ROOT/'independent_scientific_acceptance.json',dict(time=now(),passed=True,N=1000,raw_records=864,summary_cells=288,coordinate_memberships=coordinate_memberships,pool_group_matches=pool_group_matches,pool_budget_input_hashes=len(hashes),original_source_and_result_bindings_unchanged=True,cohort=desc(ROOT/'cohort.json'),scientific_completion=desc(ROOT/'scientific_completion.json'),summary=desc(ROOT/'summary.json'),source=desc(__file__),request=desc(ROOT/'request.json'),limitations=['No historical detector or KSG execution replay.','Saved feature/pool/coordinate provenance and exact CPU query checks audited; full original image intensity extraction and keyed dequantization not replayed for every group.','Human visual figure QA and local download SHA verification remain delivery checks.']))
 print('INDEPENDENT_SAVED_ARTIFACT_ACCEPTANCE_PASSED',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  import traceback
  save(ROOT/'failures'/('independent_audit_%d.json'%os.getpid()),dict(time=now(),identity=identity(),traceback=traceback.format_exc()));raise
