"""Targeted recovery from saved proposals, not global threshold relaxation.

Preserves the previous conservative result. Checks additional frozen old or
company-derived dark proposals with illumination-detrended local evidence.
Company porphyrin rules and pixel masks remain read-only. Development only.
"""
from dataclasses import dataclass,asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    work_height:int=1600
    smoothing_sigma:float=1.5
    core_depth:float=6.0
    relative_core_depth:float=0.055
    min_sector_drop:float=2.5
    min_valid_sectors:int=7
    min_bright_sectors:int=7
    context_noise_ratio:float=0.70
    core_to_texture_ratio:float=3.0
    minimum_component_area:int=10
    ring_sector_pixels:int=5
    minimum_directional_drop:float=0.4
    minimum_basin_overlap:float=0.35


def _mask(x,shape,name):
    if not isinstance(x,np.ndarray) or x.dtype not in (np.dtype(bool),np.dtype('uint8')) or x.shape!=shape:
        raise ValueError(name+' must be bool/uint8 with matching H,W')
    return x!=0


def _plane(values,xx,yy,context):
    design=np.column_stack([xx[context],yy[context],np.ones(int(context.sum()))])
    obs=values[context]
    beta=np.linalg.lstsq(design,obs,rcond=None)[0]
    # Symmetric Huber reweighting reduces influence of isolated neighboring
    # bright/dark marks without deliberately selecting only bright reference.
    for _ in range(3):
        residual=obs-design@beta
        sigma=max(1.,1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weights=np.minimum(1.,1.5*sigma/np.maximum(np.abs(residual),1e-6))
        beta=np.linalg.lstsq(design*np.sqrt(weights)[:,None],obs*np.sqrt(weights),rcond=None)[0]
    return beta[0]*xx+beta[1]*yy+beta[2]


def recover_dark_proposals(image_bgr,proposal_native,valid_native,exclude_native,config=None):
    c=config or RecoveryConfig()
    if not isinstance(c,RecoveryConfig) or any(not np.isfinite(v) or v<=0 for v in asdict(c).values()):
        raise ValueError('Invalid recovery config')
    if not 1<=c.min_bright_sectors<=c.min_valid_sectors<=8:raise ValueError('Invalid sector counts')
    if not isinstance(image_bgr,np.ndarray) or image_bgr.dtype!=np.uint8 or image_bgr.ndim!=3 or image_bgr.shape[2]!=3:
        raise ValueError('image must be uint8 H,W,3 BGR')
    h,w=image_bgr.shape[:2]
    proposal=_mask(proposal_native,(h,w),'proposal')
    valid=_mask(valid_native,(h,w),'valid');excluded=_mask(exclude_native,(h,w),'exclude')
    wh=min(h,c.work_height);ww=round(w*wh/h);size=(ww,wh)
    known_native=valid&~excluded
    known_fraction=cv2.resize(known_native.astype(np.float32),size,interpolation=cv2.INTER_AREA)
    known=known_fraction>=.99
    # Context tolerance must never weaken the no-P/whole-object conflict gate.
    any_conflict=cv2.resize((~known_native).astype(np.float32),size,interpolation=cv2.INTER_AREA)>0
    raw_value=image_bgr.max(axis=2).astype(np.float32)
    value=cv2.resize(raw_value*known_native,size,interpolation=cv2.INTER_AREA)/np.maximum(known_fraction,1e-6)
    smooth=cv2.GaussianBlur(value*known_fraction,(0,0),c.smoothing_sigma)/np.maximum(cv2.GaussianBlur(known_fraction,(0,0),c.smoothing_sigma),1e-6)
    candidate=cv2.resize(proposal.astype(np.float32),size,interpolation=cv2.INTER_AREA)>0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    selected=np.zeros_like(candidate);records=[]
    for lab in range(1,n):
        x,y,bw,bh,area=map(int,stats[lab]);reasons=[]
        rec={'label':lab,'bbox_work_xyxy':[x,y,x+bw,y+bh],'area_work':area,'accepted':False,'reasons':reasons}
        if area<c.minimum_component_area:
            reasons.append('small_work_component');records.append(rec);continue
        r=max(7,min(40,round(np.sqrt(area/np.pi)*.9)))
        pad=3*r
        x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(ww,x+bw+pad),min(wh,y+bh+pad)
        comp=labels[y0:y1,x0:x1]==lab;eligible=known[y0:y1,x0:x1]
        local=smooth[y0:y1,x0:x1]
        ext=np.zeros(comp.shape,np.uint8)
        contours,_=cv2.findContours(comp.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ext,contours,-1,1,cv2.FILLED)
        # Reject an entire proposal if its filled support includes P/red or an
        # excluded anatomical cell; never carve a ring around a red dot.
        if np.any((ext!=0)&any_conflict[y0:y1,x0:x1]):
            reasons.append('whole_proposal_exclusion_conflict');records.append(rec);continue
        near=cv2.dilate(ext,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))!=0
        far=cv2.dilate(ext,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4*r+1,4*r+1)))!=0
        ring=near&(ext==0)&eligible
        context=far&~near&eligible
        if context.sum()<80 or ring.sum()<40:
            reasons.append('incomplete_context');records.append(rec);continue
        yy,xx=np.indices(comp.shape,dtype=np.float32)
        scale=float(max(bw,bh,pad))
        plane=_plane(local,(xx-(x-x0+bw/2))/scale,(yy-(y-y0+bh/2))/scale,context)
        residual=local-plane
        q=float(np.percentile(residual[comp],25));core=comp&(residual<=q)
        cy,cx=float(yy[core].mean()),float(xx[core].mean())
        core_level=float(np.median(residual[core]));background=float(np.median(plane[core]))
        depth=-core_level
        sectors=np.floor((np.arctan2(yy-cy,xx-cx)+np.pi)*4/np.pi).astype(np.int32)%8
        deltas=[];raw_deltas=[]
        raw_core=float(np.median(local[core]))
        for s in range(8):
            sel=ring&(sectors==s)
            if sel.sum()<c.ring_sector_pixels:deltas.append(None);raw_deltas.append(None)
            else:
                deltas.append(float(np.median(residual[sel])-core_level))
                raw_deltas.append(float(np.median(local[sel])-raw_core))
        valid_count=sum(d is not None for d in deltas)
        bright_count=sum(d is not None and d>=c.min_sector_drop for d in deltas)
        variation=float(np.percentile(residual[context],90)-np.percentile(residual[context],10))
        texture=float(np.median(np.abs(value[y0:y1,x0:x1]-local)[ring]))
        # The dark basin must close locally after detrending. A residual dark
        # band that exits the context patch is a crease/shadow, not a local mark.
        basin=(residual<=-max(2.0,.40*depth))&eligible
        _,basin_labels=cv2.connectedComponents(basin.astype(np.uint8),8)
        ids=np.unique(basin_labels[core]);ids=ids[ids>0]
        seed_basin=np.isin(basin_labels,ids) if ids.size else np.zeros_like(basin)
        edge=np.zeros_like(seed_basin);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
        opens=bool(np.any(seed_basin&edge))
        overlap=float(np.count_nonzero(seed_basin&comp)/max(1,area))
        if depth<c.core_depth or depth/max(background,1)<c.relative_core_depth:reasons.append('weak_detrended_core')
        if valid_count<c.min_valid_sectors:reasons.append('missing_directional_context')
        if bright_count<c.min_bright_sectors or any(d is not None and d<c.minimum_directional_drop for d in deltas):reasons.append('open_direction_after_detrending')
        if depth<c.context_noise_ratio*variation:reasons.append('incoherent_context_after_detrending')
        if depth<c.core_to_texture_ratio*texture:reasons.append('texture_dominates_core')
        if opens or overlap<c.minimum_basin_overlap:reasons.append('open_or_unsupported_local_basin')
        rec.update(core_depth=depth,relative_depth=depth/max(background,1),background=background,
                   residual_context_span=variation,texture=texture,valid_sectors=valid_count,
                   bright_sectors=bright_count,sector_deltas=deltas,raw_sector_deltas=raw_deltas,
                   basin_opens=opens,basin_proposal_overlap=overlap)
        rec['accepted']=not reasons
        if not reasons:selected[y0:y1,x0:x1]|=comp
        records.append(rec)
    projected=cv2.resize(selected.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)!=0
    result=proposal&projected&known_native
    assert not np.any(result&(~proposal|excluded|~valid))
    return {'mask_native':result.astype(np.uint8)*255,'records':records,'config':asdict(c),'work_shape_hw':[wh,ww]}
