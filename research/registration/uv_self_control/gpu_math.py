import math
import numpy as np
import torch
import torch.nn.functional as F
from common import CONDITIONS,REGIONS,MARGIN

def kernel(sigma,radius):
    a=np.arange(-radius,radius+1,dtype=np.float64);k=np.exp(-a*a/(2*sigma*sigma));return k/k.sum()

def shift(x,dx,dy):
    h,w=x.shape[-2:];out=torch.zeros_like(x)
    xs,xe=max(0,-dx),min(w,w-dx);ys,ye=max(0,-dy),min(h,h-dy)
    out[...,ys+dy:ye+dy,xs+dx:xe+dx]=x[...,ys:ye,xs:xe]
    return out

class Engine:
    def __init__(self,device='cuda:0',batch=4,margin=MARGIN):
        self.device=device;self.batch=batch;self.margin=margin
        self.k=torch.tensor(kernel(1.5,5),device=device,dtype=torch.float32)
    def smooth(self,x,k=None,reflect=False):
        k=self.k if k is None else k;rad=len(k)//2;c=x.shape[1]
        if reflect:x=F.pad(x,(rad,rad,0,0),mode='reflect')
        x=F.conv2d(x,k.reshape(1,1,1,-1).repeat(c,1,1,1),groups=c)
        if reflect:x=F.pad(x,(0,0,rad,rad),mode='reflect')
        return F.conv2d(x,k.reshape(1,1,-1,1).repeat(c,1,1,1),groups=c)
    def blur(self,x,sigma):
        if sigma==0:return x
        k=torch.tensor(kernel(sigma,int(math.ceil(4*sigma))),device=self.device,dtype=torch.float32)
        return self.smooth(x,k,True)
    def evaluate(self,truth,weights,conditions=CONDITIONS):
        x=torch.as_tensor(truth,device=self.device,dtype=torch.float32).unsqueeze(0)
        m=self.margin;ww=torch.as_tensor(np.stack([weights[k] for k in REGIONS]),device=self.device,dtype=torch.float64)[:,m:-m,m:-m]
        areas=ww.flatten(1).sum(1);flat=ww.flatten(1).T.contiguous();safe=torch.where(areas>0,areas,torch.ones_like(areas))
        mt=self.smooth(x);vt=(self.smooth(x*x)-mt*mt).clamp_min(0)
        out=[];cached={}
        for start in range(0,len(conditions),self.batch):
            cs=conditions[start:start+self.batch];ys=[]
            for c in cs:
                s=c['sigma']
                if s not in cached:cached[s]=self.blur(x,s)
                ys.append(shift(cached[s],c['dx'],c['dy']))
            y=torch.cat(ys);d=y-x
            absolute=d.abs().mean(1)[:,m:-m,m:-m];squared=d.square().mean(1)[:,m:-m,m:-m]
            mp=self.smooth(y);vp=(self.smooth(y*y)-mp*mp).clamp_min(0);cov=self.smooth(x*y)-mt*mp
            sm=(((2*mt*mp+.0001)*(2*cov+.0009))/((mt*mt+mp*mp+.0001)*(vt+vp+.0009))).mean(1)
            sm=sm[:,m-5:-(m-5),m-5:-(m-5)]
            vals=[((a.flatten(1).double()@flat)/safe).cpu().numpy() for a in (absolute,squared,sm)]
            for j in range(len(cs)):
                rows=[]
                for r in range(4):
                    if areas[r].item()==0:rows.append([None]*4);continue
                    l1,mse,ssim=(float(a[j,r]) for a in vals)
                    assert all(math.isfinite(z) for z in (l1,mse,ssim)) and l1>=0 and mse>=0
                    rows.append([l1,mse,10*math.log10(1/mse) if mse else None,ssim])
                out.append(rows)
        return out,areas.cpu().tolist()
