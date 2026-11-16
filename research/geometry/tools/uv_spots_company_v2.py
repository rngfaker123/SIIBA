"""Native UV dark-point adaptation of the supplied company spot pipeline.

The company porphyrin detector is NOT edited or redefined. Its contour/moments
pipeline and byte-bound original centroid-distance deduplication are reused.
The only new detector evidence is multiscale LOCAL darkness instead of orange
HSV bands. Whole-object conflicts are rejected BEFORE original deduplication.
No v4/v5 detector, circularity, solidity or learned classification is used.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from uv_darkspot_company_based_v1 import load_company_primitives
from uv_darkspot_native_red_guard_v1 import build_native_red_guard


@dataclass(frozen=True)
class UVSpotsConfig:
    smooth_sigma_native: float = 3.0
    background_sigmas_native: tuple = (20.0, 60.0)
    # Weak/more-complete footprint first, so its central strong core does not
    # suppress it. If the weak object conflicts, a separate strong core may pass.
    contrast_tiers: tuple = ((6.0, 0.04), (10.0, 0.07))
    minimum_value: int = 20
    minimum_contour_area: float = 1.0
    min_dist_native: float = 10.0


def _mask(value, shape, name, optional=False):
    if value is None and optional:
        return np.zeros(shape, bool)
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype not in (np.dtype('uint8'), np.dtype(bool)):
        raise ValueError(name + ' must be native [H,W] bool/uint8')
    return value != 0


def _validate_config(c):
    if not isinstance(c, UVSpotsConfig):
        raise TypeError('config must be UVSpotsConfig')
    numbers = [c.smooth_sigma_native, *c.background_sigmas_native,
               *[v for tier in c.contrast_tiers for v in tier],
               c.minimum_value, c.minimum_contour_area, c.min_dist_native]
    if not all(np.isfinite(x) for x in numbers):
        raise ValueError('Configuration must be finite')
    if c.smooth_sigma_native <= 0 or len(c.background_sigmas_native) != 2 or any(s <= c.smooth_sigma_native for s in c.background_sigmas_native):
        raise ValueError('Two backgrounds must be larger than the smoothing scale')
    if len(c.contrast_tiers) != 2 or any(a <= 0 or not 0 < r < 1 for a, r in c.contrast_tiers):
        raise ValueError('Two positive absolute/relative contrast tiers required')
    if not 0 <= c.minimum_value < 255 or c.minimum_contour_area < 1 or c.min_dist_native < 0:
        raise ValueError('Invalid value/area/dedup threshold')


def _object_geometry(contour):
    """Exact company area and integer moments centroid; filled conflict domain."""
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])) if moments['m00'] else (0, 0)
    x, y, w, h = cv2.boundingRect(contour)
    footprint = np.zeros((h, w), np.uint8)
    cv2.drawContours(footprint, [contour - np.array([[[x, y]]])], -1, 1, cv2.FILLED)
    return area, center, (x, y, w, h), footprint != 0


def filter_candidates_before_dedup(raw_spots, band_masks, valid, p_footprint, frozen_p, red_guard, anatomy, min_dist):
    """Reject entire measured object on conflict, then run ORIGINAL dedup.

    A filled external contour is checked, including holes: a dark ring around a
    red point is rejected as a WHOLE, not trimmed into a visually false label.
    This conflict-before-dedup order intentionally differs from the company
    porphyrin detector, where the T/U center gate follows deduplication.
    """
    deduplicate, _, provenance = load_company_primitives()
    shape = valid.shape
    eligible = []
    records = []
    conflict_p = np.zeros(shape, np.uint8)
    conflict_other = np.zeros(shape, np.uint8)
    for spot in raw_spots:
        contour = spot['contour']
        area, center, (x, y, w, h), fp = _object_geometry(contour)
        cx, cy = center
        band = band_masks[spot['band_index']]
        measured = fp & band[y:y+h, x:x+w]
        overlaps = {
            'company_p_footprint': int(np.count_nonzero(fp & p_footprint[y:y+h, x:x+w])),
            'frozen_p_mask': int(np.count_nonzero(fp & frozen_p[y:y+h, x:x+w])),
            'native_reddish_guard': int(np.count_nonzero(fp & red_guard[y:y+h, x:x+w])),
            'anatomy': int(np.count_nonzero(fp & anatomy[y:y+h, x:x+w])),
            'outside_valid': int(np.count_nonzero(fp & ~valid[y:y+h, x:x+w])),
        }
        reasons = [name + '_whole_object_conflict' for name, n in overlaps.items() if n]
        center_inside = 0 <= cx < shape[1] and 0 <= cy < shape[0]
        if not center_inside or not valid[cy, cx] or anatomy[cy, cx]:
            reasons.append('company_style_center_gate_failed')
        if center_inside and (p_footprint[cy, cx] or frozen_p[cy, cx] or red_guard[cy, cx]):
            reasons.append('point_center_porphyrin_or_reddish_conflict')
        record = {
            'id': spot['id'], 'band_index': spot['band_index'], 'contour_index_in_band': spot['contour_index_in_band'],
            'center': list(center), 'contour_area': area, 'bbox_xyxy': [x, y, x+w, y+h],
            'contour': contour.reshape(-1, 2).tolist(), 'measured_pixels': int(measured.sum()),
            'footprint_pixels': int(fp.sum()), 'overlap_pixels': overlaps, 'reasons': reasons,
            'status': 'conflict_rejected' if reasons else 'awaiting_original_dedup', 'accepted': False,
        }
        records.append(record)
        if reasons:
            if any(overlaps[name] for name in ('company_p_footprint', 'frozen_p_mask', 'native_reddish_guard')) or 'point_center_porphyrin_or_reddish_conflict' in reasons:
                conflict_p[y:y+h, x:x+w][fp] = 255
            if overlaps['anatomy'] or overlaps['outside_valid'] or 'company_style_center_gate_failed' in reasons:
                conflict_other[y:y+h, x:x+w][fp] = 255
        else:
            # Do not reorder. The company greedy dedup depends on original order.
            eligible.append({**spot, 'center': center, 'area': area})
    retained = deduplicate(eligible, min_dist=min_dist)
    kept_ids = {s['id'] for s in retained}
    eligible_ids = {s['id'] for s in eligible}
    support = np.zeros(shape, np.uint8)
    footprint_mask = np.zeros(shape, np.uint8)
    for record in records:
        if record['id'] not in eligible_ids:
            continue
        if record['id'] not in kept_ids:
            record.update(status='dedup_suppressed', reasons=['company_original_distance_suppression'])
            continue
        record.update(status='accepted', accepted=True)
        contour = np.asarray(record['contour'], np.int32).reshape(-1, 1, 2)
        _, _, (x, y, w, h), fp = _object_geometry(contour)
        support[y:y+h, x:x+w][fp & band_masks[record['band_index']][y:y+h, x:x+w]] = 255
        footprint_mask[y:y+h, x:x+w][fp] = 255
    forbidden = p_footprint | frozen_p | red_guard | anatomy | ~valid
    assert not np.any((support != 0) & forbidden)
    assert not np.any((footprint_mask != 0) & forbidden)
    return {
        'records': records, 'points': [r for r in records if r['accepted']],
        'accepted_support_native': support, 'accepted_footprint_native': footprint_mask,
        'porphyrin_conflict_footprints_native': conflict_p, 'anatomy_conflict_footprints_native': conflict_other,
        'dedup_provenance': provenance,
        'counts': {'area_qualified': len(raw_spots), 'after_conflict_before_dedup': len(eligible),
                   'accepted': len(retained), 'conflict_rejected': len(raw_spots)-len(eligible),
                   'dedup_suppressed': len(eligible)-len(retained)},
    }


def detect_uv_spots(image_bgr, valid_mask, porphyrin_footprint_native, porphyrin_mask_native,
                    anatomical_exclusion_native=None, config=None):
    """Native BGR UV plus frozen geometry/P -> dark-point candidates, not truth.

    RGB photometry is not used. The caller's fixed geometry may originate from
    RGB landmarks. Porphyrin masks are mandatory, read-only inputs. All rejected
    or unassigned pixels are unresolved, NOT automatically normal skin.
    """
    c = config or UVSpotsConfig()
    _validate_config(c)
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError('image_bgr must be native nonempty uint8 [H,W,3] BGR')
    shape = image_bgr.shape[:2]
    valid = _mask(valid_mask, shape, 'valid_mask')
    p_footprint = _mask(porphyrin_footprint_native, shape, 'porphyrin_footprint_native')
    frozen_p = _mask(porphyrin_mask_native, shape, 'porphyrin_mask_native')
    anatomy = _mask(anatomical_exclusion_native, shape, 'anatomical_exclusion_native', True)
    red = build_native_red_guard(image_bgr, valid, porphyrin_mask_native)
    red_guard = red['guard_native'] != 0
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    smooth = cv2.GaussianBlur(value.astype(np.float32), (0, 0), c.smooth_sigma_native)
    band_masks, pass_details, raw_spots = [], [], []
    # Multi-pass scalar bands replace the company's orange HSV color bands.
    # No global inversion and no native/work resampling is performed.
    for sigma in c.background_sigmas_native:
        background = cv2.GaussianBlur(smooth, (0, 0), sigma)
        difference = background - smooth
        for absolute, relative in c.contrast_tiers:
            mask = (value > c.minimum_value) & (difference >= absolute) & (difference >= relative * np.maximum(background, 1))
            index = len(band_masks)
            band_masks.append(mask)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            qualified = 0
            for ci, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= c.minimum_contour_area:
                    raw_spots.append({'id': len(raw_spots), 'band_index': index, 'contour_index_in_band': ci, 'contour': contour})
                    qualified += 1
            pass_details.append({'band_index': index, 'background_sigma_native': sigma,
                                 'minimum_absolute_darkness': absolute, 'minimum_relative_darkness': relative,
                                 'raw_pixels': int(mask.sum()), 'contours': len(contours), 'area_qualified': qualified})
    result = filter_candidates_before_dedup(raw_spots, band_masks, valid, p_footprint, frozen_p, red_guard, anatomy, c.min_dist_native)
    union = np.logical_or.reduce(band_masks)
    result['candidate_union_native'] = union.astype(np.uint8) * 255
    result['native_red_guard'] = red['guard_native']
    result['unresolved_candidate_native'] = (union & (result['accepted_support_native'] == 0)).astype(np.uint8) * 255
    result['diagnostics'] = {
        'schema': 'uv_spots_company_v2', 'status': 'fixed_development_candidate_not_biological_ground_truth',
        'native_shape_hw': list(shape), 'resampling': 'none', 'config': asdict(c), 'passes': pass_details,
        'company': result.pop('dedup_provenance'), 'counts': result.pop('counts'),
        'unchanged_company_primitives': ['findContours RETR_EXTERNAL / CHAIN_APPROX_SIMPLE and original contour order',
            'contourArea >= 1', 'moments integer centroid', 'AST-executed hash-bound original deduplicate_spots',
            '10 native-pixel centroid distance'],
        'dark_adaptation': ['Gaussian-smoothed native V', 'two local background sizes and two local darkness bands per size',
            'whole filled-object conflict check before original dedup; no trimming/subtraction'],
        'deliberate_order_change': 'Reject full P/red/anatomy/region-conflicting objects BEFORE dedup so a rejected object cannot suppress an eligible point',
        'band_order': 'For each small-to-large background scale, weak/more-complete contour pass precedes the strong/core pass; unchanged greedy dedup otherwise favors whichever center occurs first',
        'porphyrin_semantics': 'Company original and frozen P inputs unchanged. Extra native red guard excludes dark candidates only, not new P truth.',
        'mask_semantics': 'accepted_support is measured band evidence; accepted_footprint includes contour holes and is separately exported. Neither is validated full UV-spot boundary.',
        'normal_skin_semantics': 'Not inferred. Unselected/conflicting candidate pixels are unresolved.',
        'parameter_status': 'fixed engineering settings, not validation-selected, no accuracy/recall claims',
        'red_guard': red['diagnostics'],
        'pixels': {key: int(np.count_nonzero(result[key])) for key in ('accepted_support_native', 'accepted_footprint_native', 'candidate_union_native', 'porphyrin_conflict_footprints_native', 'anatomy_conflict_footprints_native', 'native_red_guard', 'unresolved_candidate_native')},
        'limitations': ['Local contrast can still select hair, shadow, texture, acne or moles',
            'Company P and red guard may miss true porphyrin; zero mask overlap is not zero biological error',
            'Whole-object conflicts deliberately sacrifice possible nearby dark lesions',
            'Different multipass contours with centers >=10px can overlap; point count is not unique lesion count'],
    }
    return result
