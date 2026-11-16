import math
import numpy as np
from io_control import *
MARGIN=26
def translate(x,dx,dy):
 import torch
 h,w=x.shape[-2:];out=torch.zeros_like(x);xs,xe=max(0,-dx),min(w,w-dx);ys,ye=max(0,-dy),min(h,h-dy)
 out[...,ys+dy:ye+dy,xs+dx:xe+dx]=x[...,ys:ye,xs:xe];return out
def pick(scores):
 """Scores121x5x4; GRID's first exact minimum resolves ties prospectively."""
 return np.argmin(scores[:,:,3],axis=0),np.min(scores,axis=0)
class Engine:
 def __init__(self,device='cuda:0'):
  old_modules();from gpu_math import Engine as Original
  self.e=Original(device=device);self.device=device
 def arrays(self,case):
  import torch
  x=torch.as_tensor(case['truth'],device=self.device).unsqueeze(0);y=torch.as_tensor(case['predictions'],device=self.device);assert x.dtype==y.dtype==torch.float32
  w=torch.as_tensor(np.stack([case['weights'][r] for r in REGIONS]),device=self.device,dtype=torch.float64)[:,MARGIN:-MARGIN,MARGIN:-MARGIN]
  area=w.flatten(1).sum(1);return x,y,w.flatten(1).T.contiguous(),area
 def prep(self,x,y):
  mt=self.e.smooth(x);mp=self.e.smooth(y);vt=(self.e.smooth(x*x)-mt*mt).clamp_min(0);vp=(self.e.smooth(y*y)-mp*mp).clamp_min(0)
  return mt,mp,vt,vp
 def metric(self,x,y,flat,area,prepared=None,shift=(0,0)):
  mt,mp,vt,vp=self.prep(x,y) if prepared is None else prepared;dx,dy=shift
  if dx or dy:y=translate(y,dx,dy);mp=translate(mp,dx,dy);vp=translate(vp,dx,dy)
  cov=self.e.smooth(x*y)-mt*mp
  sm=(((2*mt*mp+.0001)*(2*cov+.0009))/((mt*mt+mp*mp+.0001)*(vt+vp+.0009))).mean(1)
  d=y-x;m=MARGIN
  maps=[d.abs().mean(1)[:,m:-m,m:-m],d.square().mean(1)[:,m:-m,m:-m],sm[:,m-5:-(m-5),m-5:-(m-5)]]
  vv=[((a.flatten(1).double()@flat)/area.clamp_min(1e-300)).cpu().numpy() for a in maps];z=np.empty((len(y),4,4))
  for j in range(len(y)):
   for r in range(4):
    if not area[r].item():z[j,r]=np.nan;continue
    l1,mse,ssim=[float(v[j,r]) for v in vv];assert all(math.isfinite(v) for v in (l1,mse,ssim)) and l1>=0 and mse>=0
    z[j,r]=[l1,mse,10*math.log10(1/mse) if mse else np.nan,ssim]
  return z
 def grid(self,x,y,flat,area):
  # 双精度计算绝对差
  m=MARGIN;xx=x[:,:,m:-m,m:-m].double();scores=[];yd=y.double()
  for dx,dy in GRID:
   yy=yd[:,:,m-dy:y.shape[-2]-m-dy,m-dx:y.shape[-1]-m-dx]
   z=((yy-xx).abs().mean(1).flatten(1)@flat)/area.clamp_min(1e-300);z[:,area==0]=float('inf');scores.append(z.cpu().numpy())
  scores=np.array(scores);assert np.isfinite(scores[:,:,3]).all();ix,minimum=pick(scores);return scores,ix,minimum
 def calculate(self,case,family,reused,limited_conditions=None,do_grid=True):
  x,y,flat,area=self.arrays(case);curves=[];corrected=[];shifts=[];grids=[];mins=[]
  for si,sigma in enumerate(SIGMAS):
   yy=self.e.blur(y,sigma);xx=self.e.blur(x,sigma) if family=='common' else x;prepared=self.prep(xx,yy);values=[]
   for ki,shift in enumerate(SHIFTS):
    if sigma==0:values.append(reused['translation'][ki]);continue
    if ki==0 and family=='common':values.append(reused['common_centers'][si]);continue
    values.append(self.metric(xx,yy,flat,area,prepared,shift))
   curves.append(values)
   if family=='common' and do_grid:
    scores,ix,minimum=self.grid(xx,yy,flat,area);grids.append(scores);mins.append(minimum);shifts.append([GRID[k] for k in ix]);out=[]
    for j,k in enumerate(ix):
     dx,dy=GRID[k];out.append(self.metric(xx,yy[j:j+1],flat,area,tuple(v if n in (0,2) else v[j:j+1] for n,v in enumerate(prepared)),(dx,dy))[0])
    corrected.append(out)
  return dict(curves=np.array(curves),corrected=np.array(corrected),oracle_dxdy=shifts,grid_scores=np.array(grids),regional_min_L1=np.array(mins),areas=area.cpu().tolist())
def cpu_score_grid(case,sigma):
 from scipy import ndimage
 from translation.gaussian_kernel import kernel
 def blur(z):
  z=z.astype(float)
  if not sigma:return z
  k=kernel(sigma,int(math.ceil(4*sigma)));return ndimage.convolve1d(ndimage.convolve1d(z,k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror')
 x=blur(case['truth']);y=blur(case['predictions']);m=MARGIN;w=np.stack([case['weights'][r] for r in REGIONS]).astype(float)[:,m:-m,m:-m];area=w.sum((1,2));out=[]
 for dx,dy in GRID:
  d=np.abs(y[:,:,m-dy:y.shape[-2]-m-dy,m-dx:y.shape[-1]-m-dx]-x[None,:,m:-m,m:-m]).mean(1);z=np.einsum('mhw,rhw->mr',d,w)/np.maximum(area,1e-300);z[:,area==0]=np.inf;out.append(z)
 return np.array(out)
def cpu_metric(case,sigma,dx,dy,family):
 from scipy import ndimage
 old,pe=old_modules();from gpu_math import kernel
 def blur(a):
  a=a.astype(float)
  if not sigma:return a
  k=kernel(sigma,int(math.ceil(4*sigma)));return ndimage.convolve1d(ndimage.convolve1d(a,k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror')
 x=blur(case['truth']) if family=='common' else case['truth'].astype(float);y=blur(case['predictions']);z=np.zeros_like(y);h,w=y.shape[-2:];xs,xe=max(0,-dx),min(w,w-dx);ys,ye=max(0,-dy),min(h,h-dy);z[...,ys+dy:ye+dy,xs+dx:xe+dx]=y[...,ys:ye,xs:xe]
 return np.array(pe.oracle(dict(case,truth=x,predictions=z)),float)
