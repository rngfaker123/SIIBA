"""Dark-only development refinement; frozen company porphyrin is never changed.

No circularity filter. Final candidates are a subset of supplied no-circle
candidates. All thresholds below are engineering development choices, not GT.
"""
from dataclasses import asdict, dataclass
import cv2
import numpy as np
from uv_darkspot_module_v4 import _smooth


@dataclass(frozen=True)
class RefinementConfig:
    coherent_sigma: float = 4.0
    weak_abs: float = 2.0
    weak_texture_factor: float = 1.25
    weak_relative: float = 0.02
    seed_abs: float = 4.0
    seed_texture_factor: float = 3.0
    seed_relative: float = 0.035
    min_area: int = 18
    ring_min_pixels: int = 32
    sector_min_pixels: int = 4
    min_valid_sectors: int = 6
    min_bright_sectors: int = 6
    sector_contrast_abs: float = 1.5
    ring_contrast_abs: float = 3.0
    ring_contrast_relative: float = 0.025


def _binary_mask(value, name, shape=None):
    if not isinstance(value,np.ndarray) or value.dtype not in (np.dtype(bool),np.dtype(np.uint8)):
        raise TypeError(name+' must be a bool/uint8 numpy array')
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise ValueError(name+' must have matching 2D mask shape')
    return value != 0


def refine_darkspots(base_result, before_native, extra_exclusion_native=None, config=None, valid_mask_native=None):
    """Refine frozen v4(no circularity) outputs using local measured contrast.

    base_result: detect_uv_darkspots(... min_circularity=0) return dictionary.
    before_native: frozen no-circle 0/255 mask, same native H,W.
    extra_exclusion_native: optional dark-only veto, NOT company labels.
    valid_mask_native: native facial geometry for scale-specific support checks.
    Returns native mask/removal maps and auditable per-scale component records.
    """
    c = RefinementConfig() if config is None else config
    if not isinstance(c, RefinementConfig):
        raise TypeError('config must be RefinementConfig')
    if any(not np.isfinite(v) or v <= 0 for v in asdict(c).values()):
        raise ValueError('Refinement config values must be finite and positive')
    if not 1 <= c.min_bright_sectors <= c.min_valid_sectors <= 8:
        raise ValueError('Require 1 <= min_bright_sectors <= min_valid_sectors <= 8')
    if c.weak_abs > c.seed_abs or c.weak_relative > c.seed_relative:
        raise ValueError('Weak thresholds must not exceed seed thresholds')
    for field in ('min_area','ring_min_pixels','sector_min_pixels','min_valid_sectors','min_bright_sectors'):
        if not isinstance(getattr(c,field),int):
            raise TypeError(field+' must be an integer')
    before = _binary_mask(before_native,'before_native')
    if before.shape != base_result['mask_native'].shape:
        raise ValueError('native shapes differ')
    if not np.array_equal(before, _binary_mask(base_result['mask_native'],'base mask',before.shape)):
        raise ValueError('base result must reproduce supplied no-circle mask')
    maps = base_result['maps_work']
    value = maps['value']
    if not isinstance(value,np.ndarray) or value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('work value must be a finite 2D numpy array')
    wh, ww = value.shape
    h, w = before.shape
    eligible = _binary_mask(maps['eligible_mask'],'eligible mask',value.shape)
    assessed = _binary_mask(maps['assessment_mask'],'assessment mask',value.shape)
    base_selected = _binary_mask(maps['selected_mask'],'selected mask',value.shape)
    extra_native = np.zeros_like(before) if extra_exclusion_native is None else _binary_mask(extra_exclusion_native,'extra exclusion',before.shape)
    extra_work = cv2.resize(extra_native.astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA) > 0
    known = eligible & ~extra_work
    if valid_mask_native is None:
        geometry = None
    else:
        valid = _binary_mask(valid_mask_native,'valid geometry',before.shape)
        geometry = cv2.resize(valid.astype(np.uint8),
                              (ww,wh),interpolation=cv2.INTER_NEAREST) != 0
    smooth, support = _smooth(value, known, c.coherent_sigma)
    texture = maps['local_texture_magnitude']
    if texture.shape != value.shape or not np.isfinite(texture).all():
        raise ValueError('texture map must match work shape and be finite')
    background_maps = {kernel:maps['background_'+str(kernel)]
                       for kernel in base_result['diagnostics']['config']['median_kernels']}
    if any(bg.shape != value.shape or not np.isfinite(bg).all() for bg in background_maps.values()):
        raise ValueError('background maps must match work shape and be finite')
    recomputed_background = bool(extra_work.any())
    if recomputed_background:
        # New dark-only color exclusions must not contaminate the reference
        # background or texture threshold. Reuse v4's numeric preprocessing
        # recipe, but impute extra excluded cells solely from known neighbors.
        base_config = base_result['diagnostics']['config']
        base_smooth, base_weight = _smooth(value, known, base_config['smooth_sigma'])
        fill, fill_weight = _smooth(value, known, base_config['missing_fill_sigma'])
        fallback = float(np.median(value[known])) if known.any() else 0.0
        fill[fill_weight < 1e-5] = fallback
        base_smooth[base_weight < 1e-5] = fill[base_weight < 1e-5]
        median_input = np.clip(np.rint(np.where(known,base_smooth,fill)),0,255).astype(np.uint8)
        background_maps = {kernel:cv2.medianBlur(median_input,kernel).astype(np.float32)
                           for kernel in background_maps}
        residual = np.abs(value-base_smooth)
        residual_fill, residual_weight = _smooth(residual,known,base_config['missing_fill_sigma'])
        residual_fallback = float(np.median(residual[known])) if known.any() else 0.0
        residual_fill[residual_weight < 1e-5] = residual_fallback
        residual_filled = np.where(known,residual,residual_fill)
        texture = cv2.medianBlur(np.clip(np.rint(residual_filled),0,255).astype(np.uint8),
                                 base_config['texture_kernel']).astype(np.float32)
    weak_floor = np.maximum(c.weak_abs, c.weak_texture_factor * texture)
    seed_floor = np.maximum(c.seed_abs, c.seed_texture_factor * texture)
    selected = np.zeros((wh, ww), bool)
    evidence_any = np.zeros_like(selected)
    no_ring = np.zeros_like(selected)
    records = []
    for kernel in base_result['diagnostics']['config']['median_kernels']:
        background = background_maps[kernel]
        coherent_dark = background - smooth
        raw_dark = background - value
        scale_supported = assessed
        if geometry is not None:
            interior = cv2.erode(geometry.astype(np.uint8),np.ones((kernel,kernel),np.uint8),
                                borderType=cv2.BORDER_CONSTANT,borderValue=0) != 0
            coverage = cv2.boxFilter(known.astype(np.float32),-1,(kernel,kernel),normalize=True,
                                     borderType=cv2.BORDER_CONSTANT)
            scale_supported = assessed & interior & (coverage >= 0.55)
        # Scale-local networks cannot borrow a seed or bridge from another scale.
        weak = (base_selected & scale_supported & known & (support >= 0.55)
                & (coherent_dark >= weak_floor) & (raw_dark >= c.weak_abs)
                & (coherent_dark / np.maximum(background, 1) >= c.weak_relative))
        seed = weak & (coherent_dark >= seed_floor) & (raw_dark >= seed_floor)
        seed &= coherent_dark / np.maximum(background, 1) >= c.seed_relative
        count, labels, stats, centers = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
        seeded = set(map(int, np.unique(labels[seed]))) - {0}
        evidence_any |= weak
        for label in range(1, count):
            x, y, bw, bh, area = map(int, stats[label])
            record = {'kernel': kernel, 'label': label, 'bbox_work_xyxy': [x,y,x+bw,y+bh],
                      'area_work_pixels': area, 'accepted': False, 'reject_reason': []}
            reasons = record['reject_reason']
            if label not in seeded:
                reasons.append('no_coherent_seed')
            if area < c.min_area:
                reasons.append('too_small_after_coherence')
            if reasons:
                records.append(record)
                continue
            radius = max(6, min(24, round(np.sqrt(area / np.pi) * 0.5)))
            x0, y0 = max(0,x-radius), max(0,y-radius)
            x1, y1 = min(ww,x+bw+radius), min(wh,y+bh+radius)
            local = labels[y0:y1,x0:x1] == label
            external = np.zeros(local.shape, np.uint8)
            contours,_ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(external,contours,-1,1,cv2.FILLED)
            expanded = cv2.dilate(external,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))) != 0
            # Include ALL eligible neighbors: excluding weak dark pixels creates
            # a systematically brighter reference and falsely favors texture.
            ring = expanded & (external == 0) & known[y0:y1,x0:x1]
            comp_value = float(np.median(smooth[y0:y1,x0:x1][local]))
            ring_count = int(ring.sum())
            ring_value = float(np.median(smooth[y0:y1,x0:x1][ring])) if ring_count else None
            valid_sectors, bright_sectors, deltas = 0, 0, []
            cy, cx = centers[label][1], centers[label][0]
            yy, xx = np.indices(local.shape)
            sector = np.floor((np.arctan2(yy+y0-cy,xx+x0-cx)+np.pi)*4/np.pi).astype(np.int32) % 8
            for s in range(8):
                points = ring & (sector == s)
                if int(points.sum()) < c.sector_min_pixels:
                    deltas.append(None)
                    continue
                delta = float(np.median(smooth[y0:y1,x0:x1][points])) - comp_value
                deltas.append(delta)
                valid_sectors += 1
                bright_sectors += int(delta >= c.sector_contrast_abs)
            if ring_count < c.ring_min_pixels or valid_sectors < c.min_valid_sectors:
                reasons.append('insufficient_surround_support')
            if ring_value is not None and (ring_value-comp_value < c.ring_contrast_abs or
                    (ring_value-comp_value)/max(ring_value,1) < c.ring_contrast_relative):
                reasons.append('weak_unbiased_ring_contrast')
            if bright_sectors < c.min_bright_sectors:
                reasons.append('one_sided_or_unenclosed_darkness')
            record.update(ring_radius=radius,ring_pixels=ring_count,median_component=comp_value,
                          median_ring=ring_value,valid_sectors=valid_sectors,bright_sectors=bright_sectors,
                          sector_deltas=deltas)
            if not reasons:
                selected[y0:y1,x0:x1] |= local
                record['accepted'] = True
            else:
                no_ring[y0:y1,x0:x1] |= local
            records.append(record)
    def project(mask):
        return cv2.resize(mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST) != 0
    after = before & project(selected) & ~extra_native
    color_removed = before & (extra_native | project(extra_work))
    coherence_removed = before & ~project(evidence_any) & ~color_removed
    surround_removed = before & ~after & ~color_removed & ~coherence_removed
    assert not np.any(after & ~before)
    removed = before & ~after
    assert not np.any(color_removed & coherence_removed)
    assert not np.any(color_removed & surround_removed)
    assert not np.any(coherence_removed & surround_removed)
    assert np.array_equal(removed,color_removed | coherence_removed | surround_removed)
    return {
        'mask_native':after.astype(np.uint8)*255,
        'removed_native':removed.astype(np.uint8)*255,
        'color_removed_native':color_removed.astype(np.uint8)*255,
        'coherence_removed_native':coherence_removed.astype(np.uint8)*255,
        'surround_removed_native':surround_removed.astype(np.uint8)*255,
        'maps_work':{'selected':selected.astype(np.uint8)*255,'coherent_value':smooth,
                     'evidence':evidence_any.astype(np.uint8)*255,'extra_exclusion':extra_work.astype(np.uint8)*255},
        'records':records,
        'diagnostics':{'status':'development_candidate_not_validated','config':asdict(c),
                       'before_pixels':int(before.sum()),'after_pixels':int(after.sum()),
                       'retained_fraction':float(after.sum()/max(1,before.sum())),
                       'accepted_scale_components':sum(r['accepted'] for r in records),
                       'company_porphyrin_unchanged':True,
                       'background_recomputed_excluding_extra_guard':recomputed_background,
                       'scale_specific_geometry_checked':geometry is not None,
                       'limitations':['Engineering subset refinement; reduced selection is not precision proof',
                                      'Coherence and surround tests can miss weak or confluent genuine spots',
                                      'Not a clinical classifier and cannot identify moles or acne']}}
