"""Frozen split-conformal scores. Units are groups; RGB components have equal weight."""
import math
import numpy as np

METHODS=('Gaussian_1.96_sigma','group_conformal_sigma','group_conformal_absolute')
TASKS=tuple(s+'_'+k for s in ('P','D') for k in ('count','area_fraction','contrast_burden'))


def finite_threshold(values,coverage=.95):
    x=np.asarray(values,dtype=np.float64);assert x.ndim==1 and np.isfinite(x).all() and (x>=0).all()
    n=len(x);rank=math.ceil((n+1)*coverage)
    return dict(n_cal=n,rank=rank,threshold=float(np.sort(x,kind='mergesort')[rank-1]) if rank<=n and n else None,infinite=rank>n or n==0,coverage=coverage)


def threshold(d):
    assert d['infinite']==(d['threshold'] is None)
    return math.inf if d['infinite'] else float(d['threshold'])


def weighted_quantile(values,weights,p=.95):
    x=np.asarray(values,dtype=np.float64).ravel();w=np.asarray(weights,dtype=np.float64).ravel();assert x.shape==w.shape and np.isfinite(x).all() and np.isfinite(w).all() and (w>=0).all()
    keep=w>0
    if not keep.any():return None
    x=x[keep];w=w[keep];o=np.argsort(x,kind='mergesort');cs=np.cumsum(w[o],dtype=np.float64);j=np.searchsorted(cs,p*cs[-1],side='left');return float(x[o[min(j,len(o)-1)]])


def cpu_scores(truth,prediction,sigma,weights):
    err=np.abs(prediction.astype(np.float64)-truth.astype(np.float64));sigma=sigma.astype(np.float64);assert sigma.shape==truth.shape[1:] and (sigma>0).all();z=err/sigma[None];scores=[];absolutes=[];regions={}
    for name,w in weights.items():
        w=w.astype(np.float64);area=float(w.sum());ww=np.broadcast_to(w,err.shape)
        a=weighted_quantile(z,ww);b=weighted_quantile(err,ww)
        regions[name]=dict(normalized_score=a,absolute_score=b,area=area,mean_absolute_error=float((err.mean(0)*w).sum()/area) if area else None,mean_sigma=float((sigma*w).sum()/area) if area else None)
        if a is not None:scores.append(a);absolutes.append(b)
    assert scores
    return dict(normalized=max(scores),absolute=max(absolutes),regions=regions)


def cpu_evaluate(truth,prediction,sigma,weights,cal):
    err=np.abs(prediction.astype(np.float64)-truth.astype(np.float64));sg=sigma.astype(np.float64);z=err/sg[None];qs=(1.959963984540054,threshold(cal['pixel_sigma']),threshold(cal['pixel_absolute']));result={}
    for name,w in weights.items():
        w=w.astype(np.float64);area=w.sum();rows=[]
        for m,q in zip(METHODS,qs):
            if not area:rows.append(dict(method=m,coverage=None,width=None,reason='zero_original_fractional_region_area'));continue
            scaled=m!='group_conformal_absolute';covered=z<=q if scaled else err<=q;cov=float((covered.mean(0)*w).sum()/area);width=float((2*q*sg*w).sum()/area) if scaled and math.isfinite(q) else (2*q if math.isfinite(q) else None)
            rows.append(dict(method=m,coverage=cov,width=width,reason=None,width_infinite=not math.isfinite(q)))
        result[name]=rows
    return result


class GPU:
    def __init__(self):
        import torch
        self.t=torch
    def quantile(self,x,w):
        t=self.t;x=x.flatten();w=w.flatten();keep=w>0
        if not bool(keep.any()):return None
        x=x[keep];w=w[keep];o=t.argsort(x,stable=True);cs=t.cumsum(w[o],0);j=t.searchsorted(cs,.95*cs[-1],right=False).clamp(max=len(o)-1);return float(x[o[j]].item())
    def compute(self,truth,pred,sigma,weights,cal=None):
        t=self.t;err=(t.as_tensor(pred,device='cuda:0').double()-t.as_tensor(truth,device='cuda:0').double()).abs();sg=t.as_tensor(sigma,device='cuda:0').double();z=err/sg[None];assert bool(t.isfinite(z).all()) and bool((sg>0).all());regions={};evaluation={};a=[];b=[]
        for name,raw in weights.items():
            w=t.as_tensor(raw,device='cuda:0').double();area=float(w.sum().item());ww=w.expand_as(err);n=self.quantile(z,ww);ab=self.quantile(err,ww)
            regions[name]=dict(normalized_score=n,absolute_score=ab,area=area,mean_absolute_error=float((err.mean(0)*w).sum().item()/area) if area else None,mean_sigma=float((sg*w).sum().item()/area) if area else None)
            if n is not None:a.append(n);b.append(ab)
            if cal is None:continue
            rows=[]
            for m,q in zip(METHODS,(1.959963984540054,threshold(cal['pixel_sigma']),threshold(cal['pixel_absolute']))):
                if not area:rows.append(dict(method=m,coverage=None,width=None,reason='zero_original_fractional_region_area'));continue
                scaled=m!='group_conformal_absolute';covered=z<=q if scaled else err<=q;cov=float((covered.double().mean(0)*w).sum().item()/area);width=float((2*q*sg*w).sum().item()/area) if scaled and math.isfinite(q) else (2*q if math.isfinite(q) else None)
                rows.append(dict(method=m,coverage=cov,width=width,reason=None,width_infinite=not math.isfinite(q)))
            evaluation[name]=rows
        assert a
        return dict(normalized=max(a),absolute=max(b),regions=regions),evaluation


def task_values(record,scale):
    out={}
    for signal in ('P','D'):
        d=record['signals'][signal];a=d['truth'];b=d['prediction']
        for key in ('count','area_fraction','contrast_burden'):
            name=signal+'_'+key;x=a.get(key);y=b.get(key)
            if x is None or y is None:out[name]=dict(error=None,normalized_error=None,truth=x,prediction=y,reason=dict(truth=a.get('reason'),prediction=b.get('reason')));continue
            error=abs(float(x)-float(y));assert math.isfinite(error) and math.isfinite(scale) and scale>0;out[name]=dict(error=error,normalized_error=error/scale,truth=x,prediction=y,reason=None)
    return out


def task_evaluate(tasks,scale,cal):
    out={}
    for name,d in tasks.items():
        result=[]
        for scaled in (False,True):
            c=cal['tasks'][name]['sigma_F_proxy' if scaled else 'constant_width'];q=threshold(c);score=d['normalized_error'] if scaled else d['error'];width=2*q*(scale if scaled else 1.)
            result.append(dict(method='sigma_F_proxy' if scaled else 'constant_width',coverage=float(score<=q) if score is not None else None,width=width if math.isfinite(width) and score is not None else None,width_infinite=not math.isfinite(width),reason=d['reason']))
        out[name]=result
    return out


def risk_curve(errors,scores):
    e=np.asarray(errors,float);s=np.asarray(scores,float);assert e.shape==s.shape;keep=np.isfinite(e)&np.isfinite(s);e=e[keep];s=s[keep];order=np.argsort(s,kind='mergesort');n=len(e);base=float(e.mean()) if n else None;out=[]
    for fraction in (1.,.9,.8,.7,.6,.5,.4,.3,.2,.1):
        k=max(1,int(math.ceil(n*fraction))) if n else 0;v=float(e[order[:k]].mean()) if k else None
        out.append(dict(retained_fraction=fraction,n_retained=k,n_total=n,risk=v,full_sample_risk=base,absolute_gain=base-v if n else None,relative_gain=(base-v)/base if n and base else None))
    return out
