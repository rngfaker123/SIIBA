"""Conservative post-filter of FROZEN old UV dark candidates (development only).

The company detector is neither imported nor modified. No new candidates, no
circularity constraint, and no RGB-based clinical classification are introduced.
All scalar thresholds are engineering choices, not learned/validated cutoffs.
"""
from dataclasses import asdict, dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class LocalContrastConfig:
    work_height: int = 1600
    smooth_sigma: float = 1.5
    min_ring_pixels: int = 40
    min_sector_pixels: int = 5
    min_valid_sectors: int = 7
    min_enclosed_sectors: int = 7
    sector_core_drop: float = 3.0
    min_core_drop: float = 6.0
    min_relative_core_drop: float = 0.07
    min_plane_drop: float = 4.0
    min_signal_to_context: float = 0.60
    min_directional_drop: float = 0.25


def _binary(a, shape, name):
    if not isinstance(a, np.ndarray) or a.dtype not in (np.dtype('uint8'), np.dtype('bool')):
        raise TypeError(name + ' must be a uint8/bool array')
    if a.shape != shape:
        raise ValueError(name + ' shape mismatch')
    return a != 0


def filter_local_spots(image_bgr, before_native, valid_native, exclusion_native, config=None):
    """Keep a frozen connected candidate only with enclosed local dark evidence.

    A dark core must stand out from a near ring in >=7 of 8 directions and remain
    dark after fitting the surrounding illumination slope. Disconnected shadows
    are not linked; boundaries are retained exactly for accepted components.
    The surrounding ring includes *all* known skin, not just brighter pixels.
    Returns a subset mask, removed mask, records and processing metadata.
    """
    c = config or LocalContrastConfig()
    if not isinstance(c, LocalContrastConfig):
        raise TypeError('config must be LocalContrastConfig')
    if any(not np.isfinite(v) or v <= 0 for v in asdict(c).values()):
        raise ValueError('Config values must be finite and positive')
    if not 1 <= c.min_enclosed_sectors <= c.min_valid_sectors <= 8:
        raise ValueError('Invalid sector counts')
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('image_bgr must be uint8 H,W,3')
    h, w = image_bgr.shape[:2]
    before = _binary(before_native, (h,w), 'before_native')
    valid = _binary(valid_native, (h,w), 'valid_native')
    exclude = _binary(exclusion_native, (h,w), 'exclusion_native')
    if np.any(before & (~valid | exclude)):
        raise ValueError('Frozen candidates overlap excluded pixels')
    wh = min(h, c.work_height)
    ww = round(w*wh/h)
    size = (ww, wh)
    # AREA occupancy avoids dropping tiny frozen native components during work
    # projection. Accepted support is intersected with native before afterwards.
    candidates = cv2.resize(before.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    known_native = valid & ~exclude
    weights = cv2.resize(known_native.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    known = weights >= 0.99
    val_native = image_bgr.max(axis=2).astype(np.float32)
    num = cv2.resize(val_native*known_native, size, interpolation=cv2.INTER_AREA)
    value = num / np.maximum(weights, 1e-6)
    denominator = cv2.GaussianBlur(weights, (0,0), c.smooth_sigma)
    smooth = cv2.GaussianBlur(value*weights, (0,0), c.smooth_sigma) / np.maximum(denominator, 1e-6)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    selected = np.zeros_like(candidates)
    records = []
    for lab in range(1,n):
        x,y,bw,bh,area = map(int,stats[lab])
        radius = max(6, min(28, round(np.sqrt(area/np.pi)*0.8)))
        pad = 2*radius+2
        x0,y0 = max(0,x-pad),max(0,y-pad)
        x1,y1 = min(ww,x+bw+pad),min(wh,y+bh+pad)
        comp = labels[y0:y1,x0:x1] == lab
        local = smooth[y0:y1,x0:x1]
        support = known[y0:y1,x0:x1]
        # Use a filled exterior only to define a surrounding ring, not to fill
        # holes in the output mask or turn it into a circular footprint.
        ext = np.zeros(comp.shape, np.uint8)
        contours,_ = cv2.findContours(comp.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ext,contours,-1,1,cv2.FILLED)
        near = cv2.dilate(ext,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))) != 0
        far = cv2.dilate(ext,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*4+1,radius*4+1))) != 0
        ring = near & (ext == 0) & support
        context = far & (ext == 0) & support
        core_value = float(np.percentile(local[comp],25))
        core = comp & (local <= core_value)
        yy,xx = np.indices(comp.shape, dtype=np.float32)
        cy,cx = float(np.mean(yy[core])),float(np.mean(xx[core]))
        sectors = np.floor((np.arctan2(yy-cy,xx-cx)+np.pi)*4/np.pi).astype(np.int32)%8
        deltas=[]
        for sector in range(8):
            r = ring & (sectors == sector)
            deltas.append(float(np.median(local[r])-core_value) if r.sum() >= c.min_sector_pixels else None)
        valid_sectors = sum(d is not None for d in deltas)
        enclosed = sum(d is not None and d >= c.sector_core_drop for d in deltas)
        ring_value = float(np.median(local[ring])) if ring.any() else 0.0
        core_drop = ring_value-core_value
        plane_drop,context_variation = 0.0,0.0
        if context.sum() >= c.min_ring_pixels:
            # Surrounding affine trend models a slowly changing illumination
            # slope. A true local depression should survive subtracting it.
            scale = float(max(bw,bh,pad))
            dx,dy = (xx-cx)/scale,(yy-cy)/scale
            design = np.column_stack((dx[context],dy[context],np.ones(int(context.sum()))))
            beta = np.linalg.lstsq(design,local[context],rcond=None)[0]
            residual = local-(beta[0]*dx+beta[1]*dy+beta[2])
            plane_drop = -float(np.median(residual[core]))
            context_variation = float(np.percentile(local[context],90)-np.percentile(local[context],10))
        reasons=[]
        if int(ring.sum()) < c.min_ring_pixels or valid_sectors < c.min_valid_sectors:
            reasons.append('insufficient_surrounding_skin')
        if enclosed < c.min_enclosed_sectors:
            reasons.append('one_sided_or_open_darkness')
        if any(d is not None and d < c.min_directional_drop for d in deltas):
            reasons.append('darkness_continues_into_surroundings')
        if core_drop < c.min_core_drop or core_drop/max(ring_value,1) < c.min_relative_core_drop:
            reasons.append('weak_local_black_core')
        if plane_drop < c.min_plane_drop:
            reasons.append('explained_by_local_illumination_slope')
        if core_drop < c.min_signal_to_context*context_variation:
            reasons.append('surrounding_shadow_gradient_dominates')
        record={'label':lab,'bbox_work_xyxy':[x,y,x+bw,y+bh],'area_work':area,
                'radius':radius,'ring_pixels':int(ring.sum()),'core_value':core_value,
                'ring_value':ring_value,'core_drop':core_drop,
                'relative_drop':core_drop/max(ring_value,1),'plane_drop':plane_drop,
                'context_variation':context_variation,'valid_sectors':valid_sectors,
                'enclosed_sectors':enclosed,'sector_deltas':deltas,
                'accepted':not reasons,'reject_reasons':reasons}
        records.append(record)
        if not reasons:
            selected[y0:y1,x0:x1] |= comp
    projection = cv2.resize(selected.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST) != 0
    after = before & projection
    assert not np.any(after & ~before)
    return {'mask_native':after.astype(np.uint8)*255,
            'removed_native':(before & ~after).astype(np.uint8)*255,
            'records':records,'config':asdict(c),'work_shape_hw':[wh,ww]}
