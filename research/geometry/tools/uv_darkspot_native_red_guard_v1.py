"""Native-resolution reddish exclusion for DARK detection only.

This does not change company porphyrin rules or label pixels as porphyrin.
The fixed engineering rule prevents color dilution during later resizing.
"""
from dataclasses import asdict, dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class NativeRedGuardConfig:
    strict_hue_low_max: int = 22
    strict_hue_high_min: int = 170
    strict_min_saturation: int = 70
    strict_min_value: int = 25
    strict_channel_ratio: float = 1.10
    weak_hue_low_max: int = 25
    weak_hue_high_min: int = 170
    weak_min_saturation: int = 25
    weak_min_value: int = 25
    weak_min_red_difference: int = 3
    weak_channel_ratio: float = 1.03


def _mask(mask, shape, name, optional=False):
    if mask is None and optional:
        return np.zeros(shape, bool)
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.ndim != 2:
        raise ValueError(f'{name} must have native H,W shape')
    if not (np.issubdtype(mask.dtype, np.number) or mask.dtype == np.bool_) or not np.isfinite(mask).all():
        raise ValueError(f'{name} must be finite numeric or boolean')
    return mask != 0


def build_native_red_guard(image_bgr, valid_mask, porphyrin_mask, config=None):
    """Return a native uint8 0/255 exclusion mask and reproducible diagnostics.

    Strict native red pixels are always guarded. Weaker reddish pixels are
    guarded only when their 8-connected color component contains strict red
    or a frozen porphyrin pixel. There is NO dilation of porphyrin labels.
    Downstream dark detection must exclude guard pixels before image resize
    and background estimation, and intersect them out again at native output.
    """
    c = config or NativeRedGuardConfig()
    if not isinstance(c, NativeRedGuardConfig):
        raise TypeError('config must be NativeRedGuardConfig')
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError('image_bgr must be nonempty native uint8 H,W,3 BGR')
    for key, value in asdict(c).items():
        if not np.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if not (0 <= c.strict_hue_low_max < c.strict_hue_high_min <= 179 and 0 <= c.weak_hue_low_max < c.weak_hue_high_min <= 179):
        raise ValueError('Invalid OpenCV hue bounds')
    if not all(0 <= x <= 255 for x in (c.strict_min_saturation, c.strict_min_value, c.weak_min_saturation, c.weak_min_value, c.weak_min_red_difference)):
        raise ValueError('Invalid color threshold')
    if c.strict_channel_ratio < 1 or c.weak_channel_ratio < 1:
        raise ValueError('Red dominance ratios must be >=1')
    valid = _mask(valid_mask, image_bgr.shape[:2], 'valid_mask')
    frozen_p = _mask(porphyrin_mask, image_bgr.shape[:2], 'porphyrin_mask', True)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., j] for j in range(3))
    r = image_bgr[..., 2].astype(np.int16)
    other = np.maximum(image_bgr[..., 0], image_bgr[..., 1]).astype(np.int16)
    strict = (((h <= c.strict_hue_low_max) | (h >= c.strict_hue_high_min))
              & (s >= c.strict_min_saturation) & (v > c.strict_min_value)
              & (r > other * c.strict_channel_ratio) & valid)
    weak = (((h <= c.weak_hue_low_max) | (h >= c.weak_hue_high_min))
            & (s >= c.weak_min_saturation) & (v > c.weak_min_value)
            & (r - other >= c.weak_min_red_difference)
            & (r > other * c.weak_channel_ratio) & valid)
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    touched = np.zeros(n, bool)
    touched[np.unique(labels[(strict | frozen_p) & weak])] = True
    touched[0] = False
    connected = touched[labels]
    guard = connected | strict
    return {
        'guard_native': guard.astype(np.uint8) * 255,
        'config': asdict(c),
        'diagnostics': {
            'schema': 'native_reddish_dark_only_guard_v1',
            'strict_native_pixels': int(strict.sum()),
            'weak_native_pixels': int(weak.sum()),
            'connected_native_pixels': int(connected.sum()),
            'guard_native_pixels': int(guard.sum()),
            'guard_outside_frozen_p_pixels': int((guard & ~frozen_p).sum()),
            'weak_color_components': int(n - 1),
            'selected_weak_color_components': int(touched.sum()),
            'company_p_unchanged': True,
            'semantics': 'Dark-only reddish exclusion, not additional porphyrin labels',
            'no_dilation': True,
            'limitations': ['Fixed engineering color rules, not biological validation',
                           'Red signals without absolute R dominance may remain',
                           'Color-connected diffuse red can also be excluded',
                           'Does not address non-red texture, shadow or hair false positives'],
        },
    }
