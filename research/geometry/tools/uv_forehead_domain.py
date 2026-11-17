"""Explicit geometric forehead-cap proposal, not a skin/lesion classifier.

The original MediaPipe domain can cut the upper forehead. Only the upper outer
arc is extrapolated. No RGB pixel values or UV-spot decisions define this cap.
Existing eye/lip holes are preserved; nonblack UV support only clips the canvas.
This is an engineering analysis-domain revision and needs visual validation.
"""
from dataclasses import dataclass,asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class ForeheadDomainConfig:
    upper_arc_fraction:float=.25
    upward_extension_fraction:float=.18
    taper_power:float=1.5
    nonblack_value:int=8


def extend_forehead_domain(uv,valid_native,config=None):
    c=config or ForeheadDomainConfig()
    if not isinstance(c,ForeheadDomainConfig) or any(not np.isfinite(v) for v in asdict(c).values()) or not 0<c.upper_arc_fraction<.5 or not 0<c.upward_extension_fraction<.4 or c.taper_power<=0 or not 0<=c.nonblack_value<255:
        raise ValueError('Invalid forehead configuration')
    if not isinstance(uv,np.ndarray) or uv.dtype!=np.uint8 or uv.ndim!=3 or uv.shape[2]!=3:
        raise ValueError('Expected uint8 UV image')
    if not isinstance(valid_native,np.ndarray) or valid_native.dtype not in (np.dtype('uint8'),np.dtype(bool)) or valid_native.shape!=uv.shape[:2]:
        raise ValueError('Expected matching binary domain')
    valid=valid_native!=0
    contours,_=cv2.findContours(valid.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('Empty facial domain')
    contour=max(contours,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(contour)
    original_outer=np.zeros(valid.shape,np.uint8);cv2.drawContours(original_outer,[contour],-1,1,cv2.FILLED)
    points=contour.astype(np.float32)
    taper=np.clip(1-(points[:,:,1]-y)/(h*c.upper_arc_fraction),0,1)**c.taper_power
    points[:,:,1]-=h*c.upward_extension_fraction*taper
    points[:,:,1]=np.clip(points[:,:,1],0,uv.shape[0]-1)
    extended_outer=np.zeros(valid.shape,np.uint8)
    cv2.drawContours(extended_outer,[np.rint(points).astype(np.int32)],-1,1,cv2.FILLED)
    new_cap=(extended_outer!=0)&(original_outer==0)&~valid&(uv.max(axis=2)>c.nonblack_value)
    new_cap[int(np.ceil(y+h*c.upper_arc_fraction)):]=False
    extended=valid|new_cap
    assert not np.any(valid&~extended)
    return {'valid_native':extended.astype(np.uint8)*255,'added_domain_native':new_cap.astype(np.uint8)*255,
            'config':asdict(c),'metadata':{'old_domain_pixels':int(valid.sum()),'added_pixels':int(new_cap.sum()),
            'old_outer_bbox_xywh':[x,y,w,h],'max_upward_pixels':float(h*c.upward_extension_fraction),
            'meaning':'Geometric extrapolation of upper outer arc; not validated skin ground truth.',
            'risks':['May include hair or acquisition fixtures near hairline; inspect boundary overlay',
                     'Does not prove complete skin coverage; not defined using RGB appearance or selected lesions']}}
