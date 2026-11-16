"""Frozen TEST execution and separate, prospective sixteen-comparison family."""
import argparse
import csv
import io
from pathlib import Path
import numpy as np
import feature_pipeline as f
import probes

def predict(root,job,device):
    import torch
    root=Path(root);f.request(root);f.torch_setup(device);condition,seed=probes.job_spec(job)
    # 十五项选定后测试
    for j in range(15):
        probes.checked_completion(root,j)
    receipt=probes.checked_completion(root,job)
    checkpoint=torch.load(f.bound(receipt['checkpoint']),map_location='cpu',weights_only=False)
    f.need(checkpoint['condition']==condition and checkpoint['seed']==seed
           and checkpoint['features']==f.desc(root/'features/completion.json')
           and checkpoint['epoch']==receipt['selected_epoch']
           and checkpoint['val_group_mean_mse']==receipt['val_group_mean_mse'],'Checkpoint identity/selection')
    net=probes.model().to(device);net.load_state_dict(checkpoint['state'],strict=True);net.eval()
    x,y,dl,donor,valid,reasons,ids=f.load_arrays(root,'test');z=probes.inputs(x,dl,donor,condition)
    flat=z.reshape(-1,69);pred=np.empty((len(flat),3),np.float32)
    with torch.no_grad():
        for start in range(0,len(flat),4096):
            pred[start:start+4096]=net(torch.as_tensor(flat[start:start+4096],device=device)).cpu().numpy()
    pred=pred.reshape(y.shape);difference=pred.astype(np.float64)-y.astype(np.float64)
    l1=np.abs(difference).reshape(len(x),4,100,3).mean((2,3))
    mse=np.square(difference).reshape(len(x),4,100,3).mean((2,3))
    f.need(np.isfinite(pred).all() and np.isfinite(l1).all() and np.isfinite(mse).all(),'Nonfinite TEST result')
    arrays=f.npz(root/'test'/('%02d.npz'%job),prediction=pred,l1=l1,mse=mse,valid=valid)
    f.save(root/'test'/('%02d.json'%job),dict(status='complete',job=job,condition=condition,seed=seed,
          request=f.desc(root/'request.json'),probe=f.desc(root/'probes'/('%02d'%job)/'completion.json'),
          features=f.desc(root/'features/completion.json'),arrays=arrays,null_reasons=reasons,sample_ids=ids,
          note='Invalid slots are retained for array alignment; valid mask alone controls scientific inclusion.'))

def bootstrap_difference(values,seed,draws=2000):
    if len(values)<2:return None
    rng=np.random.default_rng(seed);means=np.empty(draws,np.float64)
    for i in range(draws):means[i]=values[rng.integers(0,len(values),size=len(values))].mean()
    return np.quantile(means,[.025,.975]).tolist()

def aggregate(root):
    root=Path(root);r=f.request(root);stats=f.module(r['sources']['statistics'],'_context_original_statistics')
    data={};metadata={};inputs=[]
    for job in range(15):
        path=root/'test'/('%02d.json'%job);v=f.checked(f.read(path));a=np.load(f.bound(v['arrays']),allow_pickle=False)
        condition,seed=probes.job_spec(job)
        probes.checked_completion(root,job)
        f.need(v['condition']==condition and v['seed']==seed and v['job']==job
               and v['request']==f.desc(root/'request.json')
               and v['probe']==f.desc(root/'probes'/('%02d'%job)/'completion.json')
               and v['features']==f.desc(root/'features/completion.json'),'TEST binding')
        f.need(a['valid'].shape==(21899,4),'TEST validity shape')
        f.need(np.array_equal(a['valid'],np.asarray([[x is None for x in row] for row in v['null_reasons']],bool)),
               'TEST validity/null reasons differ')
        if metadata:f.need(v['sample_ids']==metadata['sample_ids'] and v['null_reasons']==metadata['null_reasons'],'Unpaired groups/reasons')
        else:metadata=v
        data[(condition,seed)]={k:a[k] for k in ('l1','mse','valid')};inputs.append(f.desc(path))
    ids=metadata['sample_ids'];nulls=metadata['null_reasons'];valid=data[('A',20261028)]['valid']
    f.need(all(np.array_equal(v['valid'],valid) for v in data.values()),'Different evaluation supports')
    summaries=[];means={};csvbuf=io.StringIO(newline='');writer=csv.writer(csvbuf)
    writer.writerow(['sample_id','region','condition','seed','l1','mse','null_reason'])
    for condition in probes.CONDITIONS:
        for metric in ('l1','mse'):
            stack=np.stack([data[(condition,s)][metric] for s in probes.SEEDS]);means[(condition,metric)]=stack.mean(0)
            for ri,region in enumerate(f.REGIONS):
                values=means[(condition,metric)][:,ri];mask=valid[:,ri];vals=[float(x) if ok else None for x,ok in zip(values,mask)]
                reasons=[why[ri] for why in nulls];summary=stats.sample_summary(vals,reasons)
                seed_means=[float(x[mask,ri].mean()) if mask.any() else None for x in stack]
                summaries.append(dict(condition=condition,region=region,metric=metric,**summary,
                    seed_means=seed_means,seed_sd=float(np.std(seed_means,ddof=1)) if mask.any() else None))
        for seed in probes.SEEDS:
            d=data[(condition,seed)]
            for i,sid in enumerate(ids):
                for ri,region in enumerate(f.REGIONS):
                    writer.writerow([sid,region,condition,seed,float(d['l1'][i,ri]) if valid[i,ri] else '',
                                     float(d['mse'][i,ri]) if valid[i,ri] else '',nulls[i][ri] or ''])
    tests=[]
    for left,right in (('B','A'),('C','B')):
        for ri,region in enumerate(f.REGIONS):
            for metric in ('l1','mse'):
                mask=valid[:,ri];lv=means[(left,metric)][:,ri];rv=means[(right,metric)][:,ri]
                why=[x[ri] for x in nulls]
                result=stats.paired_test([float(x) if ok else None for x,ok in zip(lv,mask)],
                    [float(x) if ok else None for x,ok in zip(rv,mask)],sample_ids=ids,left_reasons=why,right_reasons=why)
                result.update(left=left,right=right,region=region,metric=metric,
                    bootstrap_ci95_individual=bootstrap_difference((lv-rv)[mask],20261028),
                    bootstrap_draws=2000,bootstrap_base_seed=20261028)
                tests.append(result)
    adjusted=stats.holm_adjust([x['p_raw_two_sided'] for x in tests],family_size=16)
    for x,p in zip(tests,adjusted):x['p_holm16']=p;x['reject_holm_0_05']=None if p is None else p<=.05
    f.publish(root/'report/per_group_values.csv',csvbuf.getvalue().encode('utf8'))
    f.save(root/'report/summary.json',dict(status='complete',summaries=summaries,paired_tests=tests,inputs=inputs,
          request=f.desc(root/'request.json'),per_group_values=f.desc(root/'report/per_group_values.csv'),
          statistical_unit='sample group, not independently verified patient',
          estimand='100 fixed replacement draws per region; mean across three fitted seeds; no physical information bound',
          uncertainty='Bootstrap across groups conditional on frozen draws/probes; separate seed variability; individual CIs not simultaneous',
          controls='Refitted shuffled local/global contextual correspondence controls reported in summaries; no selective significance tests'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',required=True,choices=['predict','aggregate'])
    p.add_argument('--job',type=int);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.phase=='predict':predict(a.root,a.job,a.device)
    else:aggregate(a.root)

if __name__=='__main__':main()
