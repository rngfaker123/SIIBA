"""Prospective SIIBA contextual probes: immutable features, TRAIN-only PCA.

The launch owner supplies a byte-pinned request; this module never starts jobs.
No target or ROI enters the RGB representation functions below.
"""
import argparse
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid
import numpy as np

SEED = 20261028
REGIONS = ('P', 'D', 'O', 'F')
COUNTS = {'train': 76642, 'val': 10948, 'test': 21899}
SHARDS = 8

def need(ok, message):
    if not ok:
        raise ValueError(message)

def encoded(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def desc(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}

def validation_image_binding(original_descriptor, resolved_path):
    """Publish an absolute, checked artifact descriptor; retain original text separately."""
    image = desc(resolved_path)
    need(image['sha256'] == original_descriptor['sha256'], 'Resolved validation image hash changed')
    return dict(image=image, original_image_path=original_descriptor['path'])

def bound(d):
    need(sha(d['path']) == d['sha256'], 'Bound bytes changed: '+d['path'])
    return Path(d['path'])

def sealed(x):
    x = dict(x)
    x['record_sha256'] = hashlib.sha256(encoded(x)).hexdigest()
    return x

def checked(x):
    y = dict(x); expected = y.pop('record_sha256')
    need(hashlib.sha256(encoded(y)).hexdigest() == expected, 'Seal mismatch')
    return x

def publish(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'Refuse changed immutable file: '+str(path)); return
    tmp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    try:
        os.link(str(tmp), str(path))
    finally:
        tmp.unlink()

def save(path, x):
    publish(path, encoded(sealed(x)))

def npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), 'Array exists; resume only through its completed receipt')
    tmp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    try:
        os.link(str(tmp), str(path))
    finally:
        tmp.unlink()
    return desc(path)

def module(d, name):
    path = bound(d)
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def request(root):
    r = checked(read(Path(root)/'request.json'))
    need(r['schema'] == 'siiba_context_request_v1' and r['counts'] == COUNTS and r['shards'] == SHARDS,
         'Unexpected prospective cohort/shard inventory')
    bound(r['prospective_plan'])
    for d in r['sources'].values():
        bound(d)
    return r

def originals(r, science=False):
    secondary = bound(r['secondary_manifest'])
    sys.path.insert(0, str(Path(r['sources']['secondary_worker']['path']).parent))
    worker = module(r['sources']['secondary_worker'], '_context_original_secondary_worker')
    manifest = worker.load_manifest(secondary.parent)
    reader = module(r['sources']['train_reader'], '_context_original_training_reader')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        context = worker.load_science(manifest) if science else None
    finally:
        if visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else: os.environ['CUDA_VISIBLE_DEVICES'] = visible
    return worker, reader, context

def rows_for(r, split):
    if split == 'test':
        rows = read(bound(read(bound(r['secondary_manifest']))['catalog']))['rows']
    else:
        rows = read(bound(r['split_manifests'][split]))['rows']
    need(len(rows) == COUNTS[split], 'Cohort length differs')
    need(len(set(row['sample_id'] for row in rows)) == len(rows), 'Duplicate sample IDs')
    need(all(row['split'] == split for row in rows), 'Split mismatch')
    if split in ('train','val'):
        for row in rows: validate_training_row(row,split)
    return rows

def positive_dimension(value):
    """Parse original manifest dimension metadata; no rounding/coercion or geometry change."""
    if type(value) is int:
        need(value > 0, 'Image dimension must be positive')
        return value
    need(type(value) is str and re.fullmatch(r'[1-9][0-9]*',value) is not None,
         'Image dimension must be a positive integer or canonical decimal string')
    return int(value)

def training_shape(row):
    return [positive_dimension(row['source_height']),positive_dimension(row['source_width'])]

def validate_training_row(row,split):
    need(type(row.get('sample_id')) is str and bool(row['sample_id']) and
         type(row.get('subject_id')) is str and bool(row['subject_id']) and row.get('split')==split,
         'Original training row identity/split types differ')
    training_shape(row)
    for modality in ('rgb','uv'):
        need(type(row.get(modality+'_path')) is str and bool(row[modality+'_path']),
             'Original training image path must be a nonempty string')
        digest=row.get(modality+'_sha256')
        need(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}',digest) is not None,
             'Original image SHA256 must be its exact lowercase hexadecimal string')

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(encoded([SEED]+list(parts))).digest()[:8], 'little')

def uniform_coords(split, sid, count=32, size=1024):
    return np.random.default_rng(seed_for('uniform', split, sid)).integers(0, size, size=(count, 2), dtype=np.int64)

def weighted_coords(sid, region, weights, count=100):
    w = np.asarray(weights, np.float64)
    need(w.shape == (1024,1024) and np.isfinite(w).all() and (w >= 0).all(), 'Bad original weights')
    need(w.sum() > 0, 'Empty original weight support')
    indices = np.random.default_rng(seed_for('weighted_test',sid,region)).choice(w.size, count, replace=True,
                                                                           p=w.reshape(-1)/w.sum())
    return np.stack(np.unravel_index(indices,w.shape),1).astype(np.int64)

def donor_permutation(split, count):
    need(count > 1, 'At least two groups required')
    order = np.random.default_rng(seed_for('donor',split)).permutation(count)
    result = np.empty(count,np.int64); result[order] = np.roll(order,1)
    need(np.all(result != np.arange(count)), 'Self donor')
    return result

def freeze(root):
    r = request(root); sets = {}; cohort = {}
    for split in COUNTS:
        rows = rows_for(r,split); ids = [x['sample_id'] for x in rows]
        sets[split] = set(ids)
        cohort[split] = {'sample_ids_sha256':hashlib.sha256(encoded(ids)).hexdigest(), 'count':len(ids)}
        npz(Path(root)/'design'/('donors_'+split+'.npz'), donor=donor_permutation(split,len(rows)))
    need(not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']),
         'Cross-split sample leakage')
    rows = rows_for(r,'train')
    patches = heapq.nsmallest(65536, ((hashlib.sha256(encoded(['patch',x['sample_id'],j,SEED])).hexdigest(),i,j)
                                     for i,x in enumerate(rows) for j in range(32)))
    groups = heapq.nsmallest(16384, ((hashlib.sha256(encoded(['global',x['sample_id'],SEED])).hexdigest(),i)
                                    for i,x in enumerate(rows)))
    save(Path(root)/'design'/'selection.json', dict(request=desc(Path(root)/'request.json'), cohorts=cohort,
         patch_pairs=[[i,j] for _,i,j in patches], global_indices=[i for _,i in groups],
         donor_files=[desc(Path(root)/'design'/('donors_'+s+'.npz')) for s in COUNTS],
         donor_rule='within-split deranged group; local uses donor uniform slot recipient_ordinal_mod32',
         image_representation='original RGB crop+LANCZOS1024; global exact32x32 block area means'))

def torch_setup(device):
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if str(device).startswith('cuda'):
        need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUBLAS env required')
    return torch

def raw_rgb_features(rgb, coords, device):
    """Only RGB and coordinates; no truth/ROI accepted. Patch order C,Y,X."""
    import torch
    import torch.nn.functional as F
    need(rgb.shape == (3,1024,1024) and rgb.dtype == np.float32 and np.isfinite(rgb).all()
         and rgb.min() >= 0 and rgb.max() <= 1, 'Expected original RGB1024')
    need(coords.ndim == 2 and coords.shape[1] == 2 and coords.min() >= 0 and coords.max() < 1024, 'Coordinates')
    x = torch.as_tensor(rgb, device=device)
    xy = torch.as_tensor(coords, device=device)
    padded = F.pad(x.unsqueeze(0),(8,8,8,8),mode='reflect')[0]
    offsets = torch.arange(17,device=device)
    ys = xy[:,0,None,None]+offsets[None,:,None]
    xs = xy[:,1,None,None]+offsets[None,None,:]
    patches = padded[:,ys,xs].permute(1,0,2,3).reshape(len(coords),867)
    center = x[:,xy[:,0],xy[:,1]].T
    positions = xy[:,[1,0]].to(torch.float32)/1023.0
    global_rgb = F.avg_pool2d(x.unsqueeze(0),32,32)[0].reshape(-1)
    return torch.cat((center,positions),1), patches, global_rgb

def training_image(r, reader, row, key):
    path = Path(r['data_root'])/row[key+'_path']
    need(sha(path) == row[key+'_sha256'], 'Original image bytes changed: '+str(path))
    stat = path.stat()
    a = reader.read_image(path).numpy()
    need((path.stat().st_size,path.stat().st_mtime_ns) == (stat.st_size,stat.st_mtime_ns), 'Image changed during decode')
    return a

def validate(root,shard,device):
    """Real original-reader equivalence plus explicit CPU/CUDA RGB-only operator evidence."""
    root=Path(root);r=request(root);torch_setup(device)
    worker,reader,science=originals(r,science=True);proof=[]
    for split in ('train','val','test'):
        rows=rows_for(r,split);row=rows[shard];sid=row['sample_id']
        for key in ('rgb','uv'):
            if split=='test':
                d=row[key];shape=row['native_shape'];base=science['ctx']['catalog']['data_root']
            else:
                d={'path':row[key+'_path'],'sha256':row[key+'_sha256']}
                shape=training_shape(row);base=r['data_root']
            original=science['ctx']['old'].prepare_image(d,shape,base)
            path=science['ctx']['old'].resolve_descriptor(d,base)
            training=reader.read_image(path).numpy()
            need(np.array_equal(original,training),'Original TRAIN/TEST preprocessing pixels differ')
            item=dict(split=split,sample_id=sid,modality=key,
                      pixels_sha256=hashlib.sha256(original.tobytes()).hexdigest(),preprocessing_exact=True,
                      **validation_image_binding(d,path))
            if key=='rgb':
                xy=uniform_coords(split,sid)
                cpu=raw_rgb_features(original,xy,'cpu');gpu=raw_rgb_features(original,xy,device)
                second=raw_rgb_features(original,xy,device)
                differences=[]
                for left,right,repeat in zip(cpu,gpu,second):
                    ca=left.cpu().numpy();ga=right.cpu().numpy();ra=repeat.cpu().numpy()
                    need(np.isfinite(ga).all() and np.array_equal(ga,ra),'Finite/same-device repeat failure')
                    differences.append(float(np.max(np.abs(ca.astype(np.float64)-ga))))
                item.update(cpu_cuda_max_differences=differences,same_device_repeat_exact=True,
                            differences_diagnostic_only=True)
            proof.append(item)
    statistics=module(r['sources']['statistics'],'_context_frozen_statistics_fixture')
    t=statistics.paired_test([1.,2.,3.,4.],[0.,0.,0.,0.],sample_ids=['a','b','c','d'],
                            left_reasons=[None]*4,right_reasons=[None]*4)
    need(t['n_pairs']==4 and np.isclose(t['t_statistic'],2.5/(np.std([1.,2.,3.,4.],ddof=1)/2),rtol=1e-14),
         'Frozen paired statistic fixture differs')
    h=statistics.holm_adjust([.001]+[None]*15,family_size=16)
    need(h[0]==.016 and all(x is None for x in h[1:]),'Frozen separate Holm16 fixture differs')
    s=statistics.sample_summary([1.,3.,None],[None,None,'explicit_fixture_null'])
    need(s['n_valid']==2 and s['n_undefined']==1 and s['mean']==2 and
         np.isclose(s['sample_std_ddof1'],np.sqrt(2.)), 'Frozen summary fixture differs')
    save(root/'validation'/('%d.json'%shard),dict(status='passed',shard=shard,request=desc(root/'request.json'),
          device=device,proof=proof,statistics_fixture=dict(paired=t,holm16=h,summary=s),no_model_or_test_prediction=True))

def validation_gate(root):
    for shard in range(8):
        v=checked(read(Path(root)/'validation'/('%d.json'%shard)))
        need(v['status']=='passed' and v['shard']==shard and v['request']==desc(Path(root)/'request.json'),
             'Missing/mismatched eight-card preprocessing/feature validation')

def pca_samples(root, shard, device):
    r=request(root);validation_gate(root); selection=checked(read(Path(root)/'design/selection.json')); rows=rows_for(r,'train')
    need(selection['request']==desc(Path(root)/'request.json'),'Design request binding')
    worker,reader,_=originals(r); torch_setup(device)
    pairs=selection['patch_pairs']; groups=selection['global_indices']; chosen={}
    for position,(i,j) in enumerate(pairs):
        if i%8 == shard: chosen.setdefault(i,[]).append((position,j))
    gi={i:pos for pos,i in enumerate(groups) if i%8 == shard}
    patch_rows=[]; patch_values=[]; global_rows=[]; global_values=[]
    for i in sorted(set(chosen)|set(gi)):
        row=rows[i]; rgb=training_image(r,reader,row,'rgb')
        _,patch,g=raw_rgb_features(rgb,uniform_coords('train',row['sample_id']),device)
        for pos,j in chosen.get(i,[]): patch_rows.append(pos); patch_values.append(patch[j].cpu().numpy())
        if i in gi: global_rows.append(gi[i]); global_values.append(g.cpu().numpy())
        if i%100==shard: print('PCA sample group',i,flush=True)
    d=npz(Path(root)/'pca_samples'/('%d.npz'%shard),patch_positions=np.array(patch_rows,np.int64),
          patches=np.array(patch_values,np.float32),global_positions=np.array(global_rows,np.int64),
          globals=np.array(global_values,np.float32))
    save(Path(root)/'pca_samples'/('%d.json'%shard),dict(request=desc(Path(root)/'request.json'),selection=desc(Path(root)/'design/selection.json'),shard=shard,arrays=d))

def fit_components(x, device, dimensions=32):
    import torch
    t=torch.as_tensor(x,dtype=torch.float64,device=device); mean=t.mean(0); t=t-mean
    covariance=(t.T@t)/(len(t)-1)
    vals,vec=torch.linalg.eigh(covariance)
    components=vec[:,-dimensions:].flip(1).T
    # 固定特征向量符号
    pivots=components.abs().argmax(1)
    components=components*torch.sign(components[torch.arange(dimensions,device=device),pivots])[:,None]
    return mean.cpu().numpy(),components.cpu().numpy(),vals[-dimensions:].flip(0).cpu().numpy()

def fit_pca(root,device):
    request(root); torch_setup(device); refs=[]
    x=np.empty((65536,867),np.float32); g=np.empty((16384,3072),np.float32)
    seenx=[]; seeng=[]
    for i in range(8):
        receipt=checked(read(Path(root)/'pca_samples'/('%d.json'%i))); refs.append(desc(Path(root)/'pca_samples'/('%d.json'%i)))
        need(receipt['request']==desc(Path(root)/'request.json') and
             receipt['selection']==desc(Path(root)/'design/selection.json') and receipt['shard']==i,
             'PCA sample request/selection/ownership differs')
        a=np.load(bound(receipt['arrays']),allow_pickle=False)
        seenx.extend(a['patch_positions'].tolist()); seeng.extend(a['global_positions'].tolist())
        x[a['patch_positions']]=a['patches'];g[a['global_positions']]=a['globals']
    need(sorted(seenx)==list(range(len(x))) and sorted(seeng)==list(range(len(g))), 'PCA sample coverage/duplicate')
    lm,lc,le=fit_components(x,device); gm,gc,ge=fit_components(g,device)
    arrays=npz(Path(root)/'pca/components.npz',local_mean=lm,local_components=lc,local_eigenvalues=le,
               global_mean=gm,global_components=gc,global_eigenvalues=ge)
    save(Path(root)/'pca/receipt.json',dict(request=desc(Path(root)/'request.json'),arrays=arrays,sample_receipts=refs,
         fitting_split='train',target_access=False,solver='float64 covariance eigh, descending eigenvectors, canonical signs, no whitening'))

def extraction(root,split,shard,device):
    import torch
    root=Path(root); r=request(root); torch_setup(device)
    pca=checked(read(root/'pca/receipt.json'))
    need(pca['request']==desc(root/'request.json') and pca['fitting_split']=='train' and pca['target_access'] is False,
         'PCA request/TRAIN-only provenance differs')
    a=np.load(bound(pca['arrays']),allow_pickle=False)
    lm,lc,gm,gc=[torch.tensor(a[k],device=device,dtype=torch.float32) for k in ('local_mean','local_components','global_mean','global_components')]
    worker,reader,science=originals(r,science=split=='test'); rows=rows_for(r,split)
    indices=np.arange(shard,len(rows),8,dtype=np.int64); n=len(indices); points=400 if split=='test' else 32
    features=np.zeros((n,points,69),np.float32);truth=np.zeros((n,points,3),np.float32)
    donors=np.zeros((n,32,32),np.float32);coords=np.zeros((n,points,2),np.int16)
    valid=np.zeros((n,4),bool) if split=='test' else np.ones((n,1),bool)
    reasons=[]; receipts=[]
    for out,i in enumerate(indices):
        row=rows[int(i)]; sid=row['sample_id']; uniform=uniform_coords(split,sid)
        if split=='test':
            # 校验图像与掩膜
            case=worker.prepare_case(science,int(i),wait=False)
            rgb=science['ctx']['old'].prepare_image(row['rgb'],row['native_shape'],science['ctx']['catalog']['data_root']) if case['technical'] else case['rgb']
            uv=None if case['technical'] else case['truth']; draws=[]; why=[]
            for ri,region in enumerate(REGIONS):
                if case['technical']: reason='mask_'+case['receipt']['status']
                elif float(np.sum(case['weights'][region],dtype=np.float64)) <= 0: reason='empty_region_after_model_crop'
                else: reason=None
                valid[out,ri]=reason is None;why.append(reason)
                draws.append(weighted_coords(sid,region,case['weights'][region]) if reason is None else uniform_coords(split,sid+'/'+region,100))
            xy=np.concatenate(draws);reasons.append(why);receipts.append(desc(case['receipt_path']))
        else:
            rgb=training_image(r,reader,row,'rgb');uv=training_image(r,reader,row,'uv');xy=uniform
            reasons.append([None])
        base,patch,g=raw_rgb_features(rgb,xy,device)
        projected=(patch-lm)@lc.T;global_projected=(g-gm)@gc.T
        features[out]=torch.cat((base,projected,global_projected[None].expand(points,-1)),1).cpu().numpy()
        _,dp,_=raw_rgb_features(rgb,uniform,device);donors[out]=((dp-lm)@lc.T).cpu().numpy()
        coords[out]=xy
        if uv is not None: truth[out]=uv[:,xy[:,0],xy[:,1]].T
        if out%100==0: print(split,'shard',shard,'groups',out,'/',n,flush=True)
    need(np.isfinite(features).all() and np.isfinite(truth).all() and np.isfinite(donors).all(), 'Nonfinite feature')
    arrays=npz(root/'features'/split/('%d.npz'%shard),indices=indices,features=features,truth=truth,
               donor_local=donors,coords=coords,valid=valid)
    save(root/'features'/split/('%d.json'%shard),dict(request=desc(root/'request.json'),pca=desc(root/'pca/receipt.json'),
        split=split,shard=shard,arrays=arrays,sample_ids=[rows[i]['sample_id'] for i in indices],null_reasons=reasons,
        mask_receipts=receipts,coordinates_frozen_before_any_prediction=True))

def assemble(root):
    root=Path(root);request(root);files=[]
    for split,n in COUNTS.items():
        seen=[]
        for shard in range(8):
            path=root/'features'/split/('%d.json'%shard);v=checked(read(path));bound(v['arrays'])
            need(v['request']==desc(root/'request.json') and v['pca']==desc(root/'pca/receipt.json')
                 and v['split']==split and v['shard']==shard,'Feature binding')
            a=np.load(v['arrays']['path'],allow_pickle=False);seen.extend(a['indices'].tolist());files.append(desc(path))
            need(a['features'].shape[2]==69 and np.isfinite(a['features']).all(),'Feature shape/finite')
            need(np.array_equal(a['indices'],np.arange(shard,n,8)), 'Shard index modulo ownership')
            need(np.array_equal(a['valid'],np.asarray([[x is None for x in row] for row in v['null_reasons']],bool)),
                 'Validity/null reason mismatch')
        need(sorted(seen)==list(range(n)),'Feature ownership coverage')
    save(root/'features/completion.json',dict(status='complete',request=desc(root/'request.json'),files=files,
         test_coordinates_frozen_before_model_fit=True,points={'train':32,'val':32,'test_per_region':100},test_replacement=True))

def load_arrays(root,split):
    root=Path(root); complete=checked(read(root/'features/completion.json'))
    need(complete['request']==desc(root/'request.json'),'Feature completion request differs')
    n=COUNTS[split];p=400 if split=='test' else 32
    x=np.empty((n,p,69),np.float32);y=np.empty((n,p,3),np.float32);dl=np.empty((n,32,32),np.float32)
    valid=np.zeros((n,4 if split=='test' else 1),bool);reasons=[None]*n;ids=[None]*n
    for shard in range(8):
        path=root/'features'/split/('%d.json'%shard)
        need(desc(path) in complete['files'],'Unbound feature receipt')
        v=checked(read(path));a=np.load(bound(v['arrays']),allow_pickle=False);idx=a['indices']
        need(v['request']==desc(root/'request.json') and v['pca']==desc(root/'pca/receipt.json')
             and v['split']==split and v['shard']==shard and np.array_equal(idx,np.arange(shard,n,8)),
             'Feature request/PCA/ownership differs')
        x[idx]=a['features'];y[idx]=a['truth'];dl[idx]=a['donor_local'];valid[idx]=a['valid']
        for j,i in enumerate(idx): reasons[i]=v['null_reasons'][j];ids[i]=v['sample_ids'][j]
    dpath=root/'design'/('donors_'+split+'.npz')
    design=checked(read(root/'design/selection.json'))
    need(design['request']==desc(root/'request.json') and desc(dpath) in design['donor_files'],'Donor/design bytes changed')
    donor=np.load(dpath,allow_pickle=False)['donor']
    need(sorted(donor.tolist())==list(range(n)) and np.all(donor!=np.arange(n)),'Donor permutation')
    return x,y,dl,donor,valid,reasons,ids

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',required=True,choices=['validate','freeze','pca_samples','fit_pca','extract','assemble'])
    p.add_argument('--split',choices=list(COUNTS));p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=8);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    need(a.shards==8,'Fixed eight extraction shards')
    if a.phase in ('validate','pca_samples','extract'):need(a.shard in range(8),'Shard index')
    if a.phase=='validate':validate(a.root,a.shard,a.device)
    elif a.phase=='freeze':freeze(a.root)
    elif a.phase=='pca_samples':pca_samples(a.root,a.shard,a.device)
    elif a.phase=='fit_pca':fit_pca(a.root,a.device)
    elif a.phase=='extract':extraction(a.root,a.split,a.shard,a.device)
    else:assemble(a.root)

if __name__=='__main__':main()
