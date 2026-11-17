"""Native whole-region P priority after ALL dark sources have been merged.

P labels are immutable. A separate guard uses the original retained company
external contour support, not enlarged points or a dilated P mask. Guard-only
pixels are not relabelled as P. Rejected D becomes unassigned/Other, not normal.
"""
import cv2
import numpy as np


def _binary(mask,shape,name):
    if not isinstance(mask,np.ndarray) or mask.dtype not in (np.dtype('uint8'),np.dtype(bool)) or mask.ndim!=2 or mask.shape!=shape:
        raise ValueError(name+' must be matching uint8/bool H,W')
    return mask!=0


def company_p_support(shape,retained_spots,valid):
    valid=_binary(valid,shape,'valid')
    support=np.zeros(shape,np.uint8)
    for spot in retained_spots:
        contour=np.asarray(spot['contour'],np.int32).reshape(-1,1,2)
        if len(contour):cv2.drawContours(support,[contour],-1,255,cv2.FILLED)
    support[~valid]=0
    return support


def enforce_porphyrin_priority(dark,porphyrin,valid,porphyrin_support=None):
    if not isinstance(dark,np.ndarray) or dark.ndim!=2:raise ValueError('dark must be H,W')
    shape=dark.shape
    d=_binary(dark,shape,'dark');p=_binary(porphyrin,shape,'porphyrin');f=_binary(valid,shape,'valid')
    if np.any((d|p)&~f):raise ValueError('D/P outside fixed valid domain')
    support=p if porphyrin_support is None else _binary(porphyrin_support,shape,'porphyrin_support')&f
    guard=p|support
    count,labels,stats,_=cv2.connectedComponentsWithStats(d.astype(np.uint8),8)
    rejected_ids=np.zeros(count,bool);records=[]
    for label in range(1,count):
        x,y,w,h,area=map(int,stats[label]);component=labels[y:y+h,x:x+w]==label
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros(component.shape,np.uint8);cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
        footprint=filled!=0;pp=p[y:y+h,x:x+w];gg=guard[y:y+h,x:x+w]
        direct=int(np.count_nonzero(component&pp))
        enclosed=int(np.count_nonzero(footprint&pp))
        support_overlap=int(np.count_nonzero(footprint&gg))
        reject=support_overlap>0;rejected_ids[label]=reject
        records.append({'component':label,'bbox_xyxy':[x,y,x+w,y+h],'area_native':area,
                        'direct_P_pixels':direct,'filled_footprint_P_pixels':enclosed,
                        'filled_footprint_company_support_pixels':support_overlap,
                        'rejected':reject,'reason':('direct_P_conflict' if direct else 'encloses_frozen_P' if enclosed else 'company_P_contour_support_conflict') if reject else None})
    removed=d&rejected_ids[labels];kept=d&~removed
    assert not np.any(kept&p)
    return {'dark_native':kept.astype(np.uint8)*255,'removed_native':removed.astype(np.uint8)*255,
            'guard_native':guard.astype(np.uint8)*255,'records':records,
            'stats':{'D_before':int(d.sum()),'D_after':int(kept.sum()),'removed_pixels':int(removed.sum()),
                     'components_before':int(count-1),'components_rejected':int(rejected_ids.sum()),
                     'direct_pixel_overlap_before':int(np.count_nonzero(d&p)),
                     'porphyrin_pixels_unchanged':int(p.sum())}}
