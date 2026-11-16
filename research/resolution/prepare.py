"""Ordered screening with unchanged true-UV detector; CPU only, no CUDA imports."""
import gc,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
import cv2
from common import *
import native_detector

def initializer():
 os.sched_setaffinity(0,range(48,56));os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));cv2.setNumThreads(1)
def load_mask(p,shape=None,expected_count=None,array_hash=None):
 with Image.open(p) as im:
  assert im.mode=='L';a=np.asarray(im).copy()
 assert a.dtype==np.uint8 and np.isin(np.unique(a),(0,255)).all()
 if shape is not None:assert list(a.shape)==list(shape)
 if expected_count is not None:assert int(np.count_nonzero(a))==expected_count
 if array_hash is not None:assert array_sha(a)==array_hash
 return a!=0
def partition(m):
 assert not np.any(m['P']&m['D']) and not np.any(m['P']&m['O']) and not np.any(m['D']&m['O'])
 assert np.array_equal(m['P']|m['D']|m['O'],m['F'])
def native_sources(index,row):
 refpath=P/'masks/refs'/('%06d.json'%index);ref=legacy(read(refpath));assert ref['sample_id']==row['sample_id'] and ref['index']==index
 cp=verify(ref['receipt']);c=legacy(read(cp));assert c['sample_id']==row['sample_id'] and c['index']==index
 q=context()[0];assert c['catalog_sha256']==q['bindings']['catalog']['sha256']
 assert all(c['sources'][m]==row[m] for m in MODS)
 return ref,c,dict(ref=desc(refpath),compact=desc(cp))
def geometry(index,row,c,proof):
 p=Path(proof['masks']['F']['path']).parent.parent/'geometry/landmarks.json';land=read(p)
 archived=read(verify(c['evidence']['geometry/landmarks.json']))
 assert land['native_xy']==archived['native_xy'] and land['uv_sha256']==row['uv']['sha256']
 old=p.with_name('old_valid_native.png');assert sha(old)==land['old_valid_file_sha256']
 oldF=load_mask(old,row['native_shape']);return land,oldF,dict(landmarks=desc(p),oldF=desc(old),original_landmarks=c['evidence']['geometry/landmarks.json'])
def existing_lower(index,row,c,proof,cached):
 if row['sample_id'] not in cached:return None
 j,oldrow=cached[row['sample_id']]
 assert all(oldrow[m]==row[m] for m in MODS) and oldrow['native_shape']==row['native_shape']
 assert all(oldrow[d]['sha256']==proof['masks'][d]['sha256'] for d in DOMAINS)
 g=checked(read(OLD/'groups'/('%06d'%j)/'receipt.json'));assert g['sample_id']==row['sample_id']
 assert g['counts']['1']==c['counts_native'] and g['image_sources']=={m:row[m] for m in MODS}
 out={}
 for f in (8,4,2):
  a=checked(read(verify(g['lower'][str(f)]['audit'])))
  assert a['factor']==f and a['originalF']['sha256']==proof['masks']['F']['sha256']
  assert a['audit']['detector_protocol']['original_core']['sha256']==CORE_SHA
  out[f]=dict(audit=g['lower'][str(f)]['audit'],masks=a['mask_files'],counts=a['counts'],shape=a['input_uv_shape'][:2],reused_unchanged=True)
 return out
def screen_one(index):
 q,runner,est,cat,cached=context();row=source_row(cat,index);started=time.monotonic();ref,c,bindings=native_sources(index,row)
 if candidate_path(index).exists():
  prior=checked(read(candidate_path(index)));assert prior['request']==desc(ROOT/'request.json') and prior['row']==row and prior['original_bindings']==bindings
  return prior
 result=dict(time=now(),index=index,sample_id=row['sample_id'],row=row,original_bindings=bindings,request=desc(ROOT/'request.json'),counts={},lower={})
 if c['status']!='complete':
  assert c['status'] in ('geometry_unavailable','invalid_geometry','geometry_failed','no_face'),('Unrecognized original status',c['status'])
  result.update(eligible=False,reason='original_'+c['status'],original_status=c['status']);save(candidate_path(index),result);return result
 result['counts']['1']=c['counts_native']
 if min(c['counts_native'][d] for d in DOMAINS)<1000:
  result.update(eligible=False,reason='native_domain_below_1000');save(candidate_path(index),result);return result
 proofpath=I/'native_sources/sources'/('%06d.json'%index)
 assert proofpath.is_file(),'Retained native reference pending; not a scientific exclusion: '+str(index)
 proof=legacy(read(proofpath));assert proof['index']==index and proof['sample_id']==row['sample_id'] and proof['counts_native']==c['counts_native']
 assert proof['compact_receipt']==ref['receipt'] and proof['native_shape']==row['native_shape']
 assert proof['helper_sha256']==q['bindings']['native_helper']['sha256']
 for d in DOMAINS:assert proof['masks'][d]['sha256']==c['native_mask_file_sha256'][d]
 result.update(native_proof=desc(proofpath),native_masks=proof['masks'])
 old=existing_lower(index,row,c,proof,cached)
 if old:
  for f in (8,4,2):
   a=old[f];shape=(row['native_shape'][0]//f,row['native_shape'][1]//f)
   m={d:load_mask(verify(a['masks'][d]),shape,a['counts'][d]) for d in DOMAINS};partition(m);del m
   result['counts'][str(f)]=a['counts'];result['lower'][str(f)]=a
   if min(a['counts'][d] for d in DOMAINS)<1000:
    result.update(eligible=False,reason='factor%d_domain_below_1000'%f);break
  else:result.update(eligible=True,reason=None)
 else:
  F=load_mask(verify(proof['masks']['F']),row['native_shape'],proof['counts_native']['F'],proof['native_array_sha256']['F'])
  land,oldF,geo=geometry(index,row,c,proof);result['geometry']=geo
  verify(row['uv'])
  with Image.open(row['uv']['path']) as opened:
   uv=opened.convert('RGB');assert list(reversed(uv.size))==row['native_shape']
   for f in (8,4,2):
    lo=lower(uv,f);a=np.asarray(lo);m,audit=native_detector.detect(a,row['native_shape'],land,F,oldF)
    folder=ROOT/'masks'/('%06d'%index)/('factor%d'%f);folder.mkdir(parents=True,exist_ok=False)
    for d in DOMAINS:Image.fromarray(m[d].astype(np.uint8)*255).save(folder/(d+'.png'))
    counts={d:int(m[d].sum()) for d in DOMAINS}
    save(folder/'audit.json',dict(time=now(),index=index,sample_id=row['sample_id'],factor=f,input_uv_shape=list(a.shape),input_uv_array_sha256=array_sha(a),counts=counts,audit=audit,geometry=geo,originalF=proof['masks']['F'],mask_files={d:desc(folder/(d+'.png')) for d in DOMAINS},request=desc(ROOT/'request.json')))
    result['counts'][str(f)]=counts;result['lower'][str(f)]=dict(audit=desc(folder/'audit.json'),masks={d:desc(folder/(d+'.png')) for d in DOMAINS},counts=counts,shape=list(a.shape[:2]),reused_unchanged=False)
    lo.close();del a,m,audit
    if min(counts[d] for d in DOMAINS)<1000:
     result.update(eligible=False,reason='factor%d_domain_below_1000'%f);break
   else:result.update(eligible=True,reason=None)
   uv.close()
  del F,oldF;gc.collect()
 result['seconds']=time.monotonic()-started;save(candidate_path(index),result);return result
def feature_one(index):
 q,runner,est,cat,cached=context();c=checked(read(candidate_path(index)));assert c['eligible'];row=c['row'];started=time.monotonic()
 if group_path(index).exists():
  r=checked(read(group_path(index)));assert r['request']==desc(ROOT/'request.json') and r['candidate']==desc(candidate_path(index))
  for d in r['feature_files'].values():verify(d)
  return index
 root=ROOT/'features'/('%06d'%index);root.mkdir(parents=True,exist_ok=False);work={};native={}
 proof=legacy(read(verify(c['native_proof'])))
 for d in DOMAINS:native[d]=load_mask(verify(proof['masks'][d]),row['native_shape'],proof['counts_native'][d],proof['native_array_sha256'][d])
 partition(native);sampled=runner.sample_indices(native,row['sample_id']);assert sampled.shape==(3,4,1000)
 assert tuple(runner.DOMAINS)==DOMAINS
 for f in FACTORS:
  masks=native if f==1 else {d:load_mask(verify(c['lower'][str(f)]['masks'][d]),expected_count=c['counts'][str(f)][d]) for d in DOMAINS}
  arrays={}
  for di,d in enumerate(DOMAINS):
   ids=sampled[:,di] if f==1 else sampler(masks[d],row['sample_id'],f,d)
   assert ids.shape==(3,1000) and all(len(np.unique(v))==1000 and masks[d].ravel()[v].all() for v in ids)
   arrays['indices_'+d]=ids;arrays['raw_'+d]=np.empty((3,4,1000,3),np.uint8)
  work[f]=arrays
  if f!=1:del masks
 del native,sampled;gc.collect()
 for mi,mod in enumerate(MODS):
  verify(row[mod])
  with Image.open(row[mod]['path']) as opened:
   im=opened.convert('RGB');assert list(reversed(im.size))==row['native_shape']
   for f in FACTORS:
    lo=lower(im,f);a=np.asarray(lo).reshape(-1,3)
    for d in DOMAINS:work[f]['raw_'+d][:,mi]=a[work[f]['indices_'+d]]
    lo.close();del a
   im.close()
 files={}
 for f,arrays in work.items():
  width=row['native_shape'][1]//f
  for d in DOMAINS:
   raw=arrays['raw_'+d];values=np.empty(raw.shape,np.float64)
   for si in range(3):
    for mi,mod in enumerate(MODS):values[si,mi]=runner.dequantize_native(raw[si,mi],arrays['indices_'+d][si],row['sample_id'],width,mod)
   assert np.isfinite(values).all();arrays['values_'+d]=values
  p=root/('factor%d.npz'%f)
  with p.open('xb') as stream:np.savez(stream,**arrays)
  files[str(f)]=desc(p)
 save(group_path(index),dict(time=now(),index=index,sample_id=row['sample_id'],candidate=desc(candidate_path(index)),feature_files=files,seconds=time.monotonic()-started,request=desc(ROOT/'request.json')))
 return index
def ordered_selection(records,target):
 selected=[]
 for r in records:
  if r['eligible']:selected.append(r['index'])
  if len(selected)==target:break
 return selected
def screen():
 q,runner,est,cat,cached=context();start=time.monotonic();selected=[];prefix=[];lookahead=8
 with ProcessPoolExecutor(max_workers=8,initializer=initializer,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures={i:pool.submit(screen_one,i) for i in range(lookahead)};next_submit=lookahead
  for index in range(21899):
   r=futures.pop(index).result();no_failures();prefix.append(dict(index=index,sample_id=r['sample_id'],eligible=r['eligible'],reason=r['reason'],counts=r['counts'],receipt=desc(candidate_path(index))))
   if r['eligible']:selected.append(index)
   if index%5==0 or len(selected)==1000:state(phase='ordered_common1000_screening',screened_prefix=index+1,eligible=len(selected),target=1000,elapsed_seconds=time.monotonic()-start)
   if len(selected)==1000:break
   if next_submit<21899:futures[next_submit]=pool.submit(screen_one,next_submit);next_submit+=1
  # 限七组预取，顺序固定
  for f in futures.values():f.result()
 assert selected==ordered_selection(prefix,1000)
 save(ROOT/'cohort.json',dict(time=now(),status='complete' if len(selected)==1000 else 'catalogue_exhausted_shortfall',selected_indices=selected,N=len(selected),target_N=1000,screened_prefix=len(prefix),screening=prefix,overshoot_preserved_not_selected=sorted(set(int(p.stem) for p in (ROOT/'screening').glob('*.json'))-set(range(len(prefix)))),catalog=q['bindings']['catalog'],all_budgets_all_factors_same_cohort=True,rule='first 1000 in original catalogue order with P/D/O/F >=1000 at factors1/2/4/8; no replacements, duplicates or reduced thresholds',request=desc(ROOT/'request.json')))
 return selected
def features(selected):
 start=time.monotonic()
 with ProcessPoolExecutor(max_workers=8,initializer=initializer,mp_context=multiprocessing.get_context('spawn')) as pool:
  for count,index in enumerate(pool.map(feature_one,selected,chunksize=1),1):
   no_failures()
   if count%5==0:state(phase='four_modality_coordinate_feature_preparation',groups=count,total=len(selected),elapsed_seconds=time.monotonic()-start)
   if count%50==0:storage_check()
 save(ROOT/'features_complete.json',dict(time=now(),N=len(selected),cohort=desc(ROOT/'cohort.json'),groups=[desc(group_path(i)) for i in selected],request=desc(ROOT/'request.json')))
def pools(selected):
 N=len(selected);maps={};artifacts={}
 for f in FACTORS:
  for si,seed in enumerate(SEEDS):
   for d in DOMAINS:
    for mi,m in enumerate(MODS):
     p=ROOT/'pools'/pool_name(f,seed,d,m);maps[f,si,d,mi]=np.lib.format.open_memmap(p,mode='w+',dtype=np.float64,shape=(N,1000,3))
 for rank,index in enumerate(selected):
  r=checked(read(group_path(index)))
  for f in FACTORS:
   with np.load(verify(r['feature_files'][str(f)]),allow_pickle=False) as z:
    for d in DOMAINS:
     v=z['values_'+d];assert v.shape==(3,4,1000,3)
     for si in range(3):
      for mi in range(4):maps[f,si,d,mi][rank]=v[si,mi]
  if (rank+1)%25==0:state(phase='pool_assembly',groups=rank+1,total=N)
 for a in maps.values():a.flush()
 maps.clear();gc.collect()
 for p in (ROOT/'pools').glob('*.npy'):artifacts[p.name]=desc(p)
 assert len(artifacts)==192
 save(ROOT/'preparation_complete.json',dict(time=now(),N=N,cohort=desc(ROOT/'cohort.json'),features=desc(ROOT/'features_complete.json'),pools=artifacts,prefix_budgets=list(BUDGETS),owned_bytes=storage_check(),request=desc(ROOT/'request.json')))
