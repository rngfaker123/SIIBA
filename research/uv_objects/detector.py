"""Versioned 1024-grid P/D detector for the approved 3C/3D supplement.

Historical detector files are byte checked and never edited. Isolated modules
execute their original functions with prospectively converted spatial scales.
This endpoint does not reuse truth P/D on the prediction side. Its output masks
are algorithmic signal supports, not clinical annotations or follicle labels.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
from PIL import Image

ADAPTER_SHA = 'db52cafb66d40f3b40592dff11724888ce6c6fbf37731df2cc56e250b4320e0e'
VENDOR_SHA = 'de1c4d52ab781d3440b694afb0d9480eb0af1359a365aa5290ae0d750a31a0e9'
CONNECTIONS_SHA = 'cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87'
GRID = 1024
F_THRESHOLD = 0.5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _positive_length(x, ratio):
    return max(1, int(math.floor(float(x) * ratio + .5)))


def _area(x, ratio):
    # 像素面积向上取整
    return max(1, int(math.ceil(float(x) * ratio * ratio)))


def _odd_kernel(x, ratio):
    # 奇数核等距取大
    return max(3, 2 * int(math.floor(float(x) * ratio / 2.)) + 1)


def scale_spec(native_shape):
    h, w = map(int, native_shape)
    if min(h, w) < 16:
        raise ValueError('Invalid native shape')
    side = min(h, w)
    native_ratio = GRID / float(side)
    original_work_ratio = min(h, 1600) / float(h)
    work_ratio = native_ratio / original_work_ratio
    return {'native_shape_hw': [h, w], 'center_crop_xyxy': [(w-side)//2, (h-side)//2,
            (w-side)//2+side, (h-side)//2+side], 'model_grid': [GRID, GRID],
            'native_to_model_length': native_ratio, 'original_native_to_work_length': original_work_ratio,
            'original_work_to_model_length': work_ratio,
            'native_to_model_area': native_ratio ** 2, 'original_work_to_model_area': work_ratio ** 2,
            'integer_length_rule': 'nearest integer; ties upward; minimum1',
            'integer_area_rule': 'minimum counts ceil(scaled area), maximum counts floor(scaled area); minimum1',
            'median_width_rule': 'nearest odd integer; ties upward; minimum3',
            'intensity_ratio_shape_and_sector_count_parameters': 'unchanged'}


def _sources(tools_dir, vendor_path=None):
    folder = Path(tools_dir).resolve()
    adapter = folder / 'mi_atlas_frozen_mask_adapter_20261025.py'
    if sha256(adapter) != ADAPTER_SHA:
        raise ValueError('Historical pipeline adapter SHA mismatch')
    tree = ast.parse(adapter.read_text(encoding='utf-8-sig'))
    pins = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FROZEN_SHA' for t in node.targets):
            pins = ast.literal_eval(node.value)
    if not pins:
        raise ValueError('Historical source manifest missing')
    for filename, digest in pins.items():
        if sha256(folder / filename) != digest:
            raise ValueError('Historical detector changed: ' + filename)
    connections = folder.parent / 'assets/skin_mi_subject1000_v1/face_mesh_connections.py'
    if sha256(connections) != CONNECTIONS_SHA:
        raise ValueError('Landmark connections changed')
    vendor = Path(vendor_path) if vendor_path else folder / 'porphyrin_module.py'
    if not vendor.exists() and vendor_path is None:
        vendor = folder.parent / 'output/porphyrin_company_review_20261021_v1/source/porphyrin_module.py'
    if sha256(vendor) != VENDOR_SHA:
        raise ValueError('Company source SHA mismatch')
    return folder, vendor, connections, pins


def _named_functions(path, names, scope):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Missing original functions: ' + str(names))
    namespace = dict(scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def project_native_mask(mask, native_shape):
    spec = scale_spec(native_shape)
    source = np.asarray(mask)
    if source.shape != tuple(spec['native_shape_hw']) or source.ndim != 2:
        raise ValueError('Native mask shape mismatch')
    x0, y0, x1, y1 = spec['center_crop_xyxy']
    fraction = np.asarray(Image.fromarray((source[y0:y1, x0:x1] != 0).astype(np.float32), mode='F').resize(
        (GRID, GRID), resample=Image.Resampling.BOX), dtype=np.float32)
    return fraction


def prepare_old_f_native(landmarks, native_shape, *, tools_dir, require_runtime=True):
    """Run once in ORIGINAL geometry runtime, then cache output for mask workers.

    No new face detector is invoked. Only archived normalized landmarks and the
    exact original polygon rasterizer are used. Never silently use mask-runtime
    OpenCV4.10 in place of the geometry-runtime4.11 rasterizer.
    """
    folder, _, connections, _ = _sources(tools_dir)
    if require_runtime and (cv2.__version__ != '4.11.0' or np.__version__ != '1.26.4'):
        raise RuntimeError('oldF rasterization requires OpenCV4.11.0/NumPy1.26.4')
    scope = {'np': np, 'cv2': cv2, 'ast': ast, 'CONNECTIONS': connections}
    fn = _named_functions(folder / 'build_porphyrin_mediapipe_regions_20261021.py',
        ['connection_sets', 'ordered_cycle', 'polygon_indices', 'make_region'], scope)
    h, w = map(int, native_shape)
    original, xy, _, _, _ = fn['make_region'](np.asarray(landmarks['normalized_xyz'], np.float64), w, h, fn['polygon_indices']())
    if 'native_xy' in landmarks and not np.array_equal(xy, np.asarray(landmarks['native_xy'], np.float64)):
        raise ValueError('Archived normalized/native landmarks disagree')
    return original


def _coverage(value, name):
    x = np.asarray(value)
    if x.shape != (GRID, GRID) or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(name + ' must be finite1024 fractional coverage in[0,1]')
    return x.astype(np.float32, copy=False)


def prepare_geometry(landmarks, native_shape, f_coverage, *, tools_dir, old_f_native=None,
                     old_f_coverage=None, require_runtime=True):
    """Project cached geometric domains once per group, independent of UV values.

    The final F is the ORIGINAL archived common fractional coverage, >=0.5 for
    this binary-object endpoint. Native anatomy is built by the original guard
    function in the mask runtime, then center-crop/BOX projected >=0.5. Native
    oldF must be supplied from its pinned rasterizer, or its exact BOX coverage.
    """
    folder, _, connections, _ = _sources(tools_dir)
    if require_runtime and (cv2.__version__ != '4.10.0' or np.__version__ != '2.3.4'):
        raise RuntimeError('Anatomy/detection requires OpenCV4.10.0/NumPy2.3.4')
    f = _coverage(f_coverage, 'F')
    if old_f_coverage is None:
        if old_f_native is None:
            raise ValueError('Supply archived/native oldF from prepare_old_f_native in its pinned runtime')
        old = project_native_mask(old_f_native, native_shape)
    else:
        old = _coverage(old_f_coverage, 'oldF')
    source = folder / 'uv_darkspot_anatomical_exclusions_v1.py'
    scope = {'np': np, 'cv2': cv2, 'ast': ast, 'CONNECTIONS': connections,
        'NOSTRIL_ANCHORS': {'right_nostril_guard': [218,237,44,20,242,99,97,240,219],
                           'left_nostril_guard': [438,457,274,250,462,328,326,460,439]}}
    anatomy_funcs = _named_functions(source, ['official_eye_brow_ids', '_filled_guard', 'build_dark_exclusions'], scope)
    anatomy_native = anatomy_funcs['build_dark_exclusions'](landmarks, tuple(map(int, native_shape)))['mask_native']
    anatomy = project_native_mask(anatomy_native, native_shape)
    fb = f >= F_THRESHOLD
    oldb = (old >= F_THRESHOLD) & fb
    return {'F': fb, 'old_F': oldb, 'anatomy': anatomy >= F_THRESHOLD,
        'audit': {'F_original_fraction_sha256': array_sha256(f), 'F_fraction_sum': float(f.sum(dtype=np.float64)),
                  'F_binary_sha256': array_sha256(fb), 'F_binary_area': int(fb.sum()),
                  'old_F_fraction_sha256': array_sha256(old), 'old_F_pixels': int(oldb.sum()),
                  'old_F_outside_common_F_pixels_clipped': int(((old >= F_THRESHOLD) & ~fb).sum()),
                  'anatomy_native_sha256': array_sha256(anatomy_native), 'anatomy_fraction_sha256': array_sha256(anatomy),
                  'binary_threshold': F_THRESHOLD, 'projection': 'original center square then Pillow float32 BOX',
                  'denominator': 'binary commonF pixel count; no fractional weighting for3CD',
                  'native_shape_hw': list(map(int, native_shape)),
                  'landmarks_xy_sha256': array_sha256(np.asarray(landmarks['native_xy'], np.float64))}}


class Detector:
    """One worker-local detector; reuse by native shape. No truth-side inputs."""
    def __init__(self, native_shape, *, tools_dir, vendor_path=None, require_runtime=True):
        if require_runtime and (cv2.__version__ != '4.10.0' or np.__version__ != '2.3.4'):
            raise RuntimeError('Detection requires original OpenCV4.10.0/NumPy2.3.4 mask runtime')
        self.folder, self.vendor_path, _, self.pins = _sources(tools_dir, vendor_path)
        self.scale = scale_spec(native_shape)
        self.n = self.scale['native_to_model_length']
        self.w = self.scale['original_work_to_model_length']
        self.modules = {}
        self.transforms = {}
        self.configs = {}
        self._initialize()

    def _load(self, name, edits=()):
        if name in self.modules:
            return self.modules[name]
        path = self.folder / (name + '.py')
        src = path.read_text(encoding='utf-8-sig')
        applied = []
        for before, after in edits:
            if src.count(before) != 1:
                raise ValueError('Spatial scale patch does not uniquely bind: ' + name + '/' + before)
            src = src.replace(before, after)
            applied.append({'original': before, 'converted': after})
        tree = ast.parse(src, filename=str(path))
        unique = '_signal3cd_' + str(id(self)) + '_' + name
        module = types.ModuleType(unique)
        module.__file__ = str(path)
        module.__dict__.update(_scale_len=lambda x: _positive_length(x, self.w),
                               _scale_area=lambda x: _area(x, self.w))
        for node in list(tree.body):
            if isinstance(node, ast.ImportFrom) and node.module and node.module + '.py' in self.pins:
                dep = self._load(node.module)
                for alias in node.names:
                    if alias.name == '*':
                        raise ValueError('Unexpected wildcard historical import')
                    module.__dict__[alias.asname or alias.name] = getattr(dep, alias.name)
                tree.body.remove(node)
        sys.modules[unique] = module
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
        self.modules[name] = module
        self.transforms[name] = {'original_sha256': self.pins[name + '.py'],
            'converted_text_sha256': hashlib.sha256(src.encode('utf-8')).hexdigest(), 'spatial_edits': applied,
            'dependency_imports': 'bound to isolated byte-verified modules; no global original module mutation'}
        return module

    def _initialize(self):
        w, n = self.w, self.n
        v4 = self._load('uv_darkspot_module_v4', [('max(3, min(15, round(np.sqrt(area / np.pi) * 0.25)))',
            'max(_scale_len(3), min(_scale_len(15), round(np.sqrt(area / np.pi) * 0.25)))')])
        refinement = self._load('uv_darkspot_refinement_v5', [('max(6, min(24, round(np.sqrt(area / np.pi) * 0.5)))',
            'max(_scale_len(6), min(_scale_len(24), round(np.sqrt(area / np.pi) * 0.5)))')])
        local = self._load('uv_spot_local_contrast_filter_20261024', [
            ('max(6, min(28, round(np.sqrt(area/np.pi)*0.8)))', 'max(_scale_len(6), min(_scale_len(28), round(np.sqrt(area/np.pi)*0.8)))'),
            ('pad = 2*radius+2', 'pad = 2*radius+_scale_len(2)')])
        recovery = self._load('uv_spot_recovery_20261024', [
            ('max(7,min(40,round(np.sqrt(area/np.pi)*.9)))', 'max(_scale_len(7),min(_scale_len(40),round(np.sqrt(area/np.pi)*.9)))'),
            ('context.sum()<80 or ring.sum()<40', 'context.sum()<_scale_area(80) or ring.sum()<_scale_area(40)')])
        primitives = self._load('uv_darkspot_company_based_v1')
        original_primitives = primitives.load_company_primitives
        primitives.load_company_primitives = lambda source_path=None: original_primitives(self.vendor_path if source_path is None else source_path)
        company = self._load('uv_spots_company_v2', [('c.minimum_contour_area < 1', 'c.minimum_contour_area <= 0')])
        self._load('uv_darkspot_native_red_guard_v1')
        self._load('uv_porphyrin_priority_20261024')
        adapter = self._load('porphyrin_company_adapter_20261021')
        self.vendor = adapter.load_vendor(self.vendor_path)
        # 只换算轮廓面积阈值

        vendor_src = self.vendor_path.read_text(encoding='utf-8-sig')
        tree = ast.parse(vendor_src)
        entry = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'detect_orange_spots_m')
        replacements = 0
        for node in ast.walk(entry):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'area' and len(node.ops) == 1 and isinstance(node.ops[0], ast.GtE) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 1:
                node.comparators[0] = ast.Constant(value=n*n)
                replacements += 1
        if replacements != 1:
            raise ValueError('Company contour-area patch not uniquely bound')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[entry], type_ignores=[])), str(self.vendor_path), 'exec'), self.vendor.__dict__)
        self.transforms['company_entry'] = {'original_sha256': VENDOR_SHA, 'area_native': 1., 'area_model': n*n,
            'min_distance_native': 10., 'min_distance_model': 10*n, 'ast_area_replacements': replacements}
        self.support = _named_functions(self.folder / 'build_uv_three_masks_review_20261021.py',
            ['porphyrin_pixel_support'], {'np': np, 'cv2': cv2})['porphyrin_pixel_support']
        self.configs['v4'] = replace(v4.DarkSpotConfig(), max_work_height=GRID,
            median_kernels=tuple(_odd_kernel(k, w) for k in (25,65,121)), texture_kernel=_odd_kernel(25,w),
            smooth_sigma=2.5*w, missing_fill_sigma=18*w, porphyrin_guard_radius=_positive_length(2,w),
            min_area=_area(18,w), max_area=max(1,int(math.floor(60000*w*w))), min_minor_axis=4*w, min_ring_pixels=_area(24,w), min_circularity=0.)
        self.configs['refinement'] = replace(refinement.RefinementConfig(), coherent_sigma=4*w,
            min_area=_area(18,w), ring_min_pixels=_area(32,w), sector_min_pixels=_area(4,w))
        self.configs['local'] = replace(local.LocalContrastConfig(), work_height=GRID, smooth_sigma=1.5*w,
            min_ring_pixels=_area(40,w), min_sector_pixels=_area(5,w))
        self.configs['recovery'] = replace(recovery.RecoveryConfig(), work_height=GRID, smoothing_sigma=1.5*w,
            minimum_component_area=_area(10,w), ring_sector_pixels=_area(5,w))
        self.configs['company'] = replace(company.UVSpotsConfig(), smooth_sigma_native=3*n,
            background_sigmas_native=(20*n,60*n), minimum_contour_area=n*n, min_dist_native=10*n)

    def protocol(self):
        return {'schema': 'signal_objects_detector_3cd_v1', 'scale': self.scale,
            'source_pins': self.pins, 'historical_pipeline_sha256': ADAPTER_SHA,
            'isolated_source_conversion': self.transforms, 'configs': {k: asdict(v) for k,v in self.configs.items()},
            'moderation': {'minimum_area_model': _area(70,self.w), 'minimum_core_depth': 11.},
            'F_binary_threshold': F_THRESHOLD, 'photometric_detection': 'common clip[0,1],round-to-nearest-even(*255) uint8; no image-adaptive scaling',
            'P_burden': 'max(R-(G+B)/2,0) from original float encoded image',
            'D_burden': 'max(Gaussian(smoothedV,20*n)-smoothedV,Gaussian(smoothedV,60*n)-smoothedV,0); smoothedV=Gaussian(max(original float RGB),3*n)',
            'object_centers': 'computed downstream from final8-connected binary masks; original company point_count not used',
            'runtime': {'numpy': np.__version__, 'opencv': cv2.__version__}}

    def _moderate(self, candidate, recovery):
        n, labels, stats, _ = cv2.connectedComponentsWithStats((candidate != 0).astype(np.uint8), 8)
        if len(recovery['records']) != n-1:
            raise ValueError('Recovery component binding changed')
        keep = np.zeros(n, bool)
        for rec in recovery['records']:
            if rec['area_work'] != int(stats[rec['label'], cv2.CC_STAT_AREA]):
                raise ValueError('Recovery area mismatch')
            keep[rec['label']] = rec['accepted'] and rec['area_work'] >= _area(70,self.w) and rec['core_depth'] >= 11.
        return ((recovery['mask_native'] != 0) & keep[labels]).astype(np.uint8)*255

    def detect(self, rgb, geometry, *, stage_hook=None):
        """Detect one image independently. No true P/D masks are accepted here."""
        started = time.monotonic()
        a = np.asarray(rgb)
        if a.shape == (3,GRID,GRID):
            a = a.transpose(1,2,0)
        if a.shape != (GRID,GRID,3) or not np.issubdtype(a.dtype,np.floating) or not np.isfinite(a).all():
            raise ValueError('Expected finite float RGB CHW/HWC1024')
        # 负荷保留越界值
        # 仅检测输入限幅
        af = a.astype(np.float32,copy=False)
        quantized = np.rint(np.clip(af,0,1)*255).astype(np.uint8)
        bgr = np.ascontiguousarray(quantized[...,::-1])
        masks = {}
        for key in ['F','old_F','anatomy']:
            value = np.asarray(geometry[key])
            if value.shape != (GRID,GRID) or value.dtype != bool:
                raise ValueError('Geometry '+key+' must be bool1024')
            masks[key] = value.astype(np.uint8)*255
        f, oldf, anatomy = masks['F'], masks['old_F'], masks['anatomy']
        if not np.any(f) or np.any((oldf != 0) & (f == 0)):
            raise ValueError('EmptyF or oldF outsideF')
        audit = {'stages': {}, 'clip_below0': int((af<0).sum()), 'clip_above1': int((af>1).sum()),
            'input_float_sha256': array_sha256(af), 'detector_uint8_rgb_sha256': array_sha256(quantized),
            'quantization_max_abs_difference': float(np.max(np.abs(quantized.astype(np.float32)/255-af))),
            'geometry': geometry.get('audit',{}), 'native_shape_hw': self.scale['native_shape_hw']}
        def emit(name, mask):
            audit['stages'][name] = {'pixels': int(np.count_nonzero(mask)), 'sha256': array_sha256(mask)}
            if stage_hook is not None:
                stage_hook(name, mask)
        m = self.modules
        got = m['porphyrin_company_adapter_20261021'].detect_uv(self.vendor, bgr, roi=None, min_dist=10*self.n)
        spots, bands = got['spots'], got['hsv_bands']
        oldp,_ = self.support(bgr,oldf,spots,bands)
        p,retained = self.support(bgr,f,spots,bands)
        emit('P',p)
        base = m['uv_darkspot_module_v4'].detect_uv_darkspots(bgr,oldf,oldp,anatomy,self.configs['v4'])
        emit('base_D',base['mask_native'])
        red = m['uv_darkspot_native_red_guard_v1'].build_native_red_guard(bgr,oldf,oldp)['guard_native']
        refined_result = m['uv_darkspot_refinement_v5'].refine_darkspots(base,base['mask_native'],red,self.configs['refinement'],oldf)
        refined = refined_result['mask_native']; emit('refined_D',refined)
        del base
        local_result = m['uv_spot_local_contrast_filter_20261024'].filter_local_spots(bgr,refined,oldf,oldp|anatomy|red,self.configs['local'])
        baseline = local_result['mask_native']; emit('baseline_D',baseline)
        footprint = np.zeros_like(p)
        for spot in retained:
            cv2.drawContours(footprint,[np.asarray(spot['contour'],np.int32).reshape(-1,1,2)],-1,255,cv2.FILLED)
        company = m['uv_spots_company_v2'].detect_uv_spots(bgr,f,footprint,p,anatomy,self.configs['company'])
        proposal = company['accepted_support_native']; emit('company_D',proposal)
        forbidden = p|anatomy|company['native_red_guard']
        recovered, moderated = [],[]
        for tag,candidate in [('old',refined),('company',proposal)]:
            rec = m['uv_spot_recovery_20261024'].recover_dark_proposals(bgr,candidate,f,forbidden,self.configs['recovery'])
            recovered.append(rec['mask_native']); moderated.append(self._moderate(candidate,rec))
            emit('recovery_'+tag,recovered[-1]); emit('moderated_'+tag,moderated[-1])
        union = baseline|recovered[0]|recovered[1]
        union[(f == 0)|(forbidden != 0)] = 0
        before = (baseline|moderated[0]|moderated[1])&union
        priority_module = m['uv_porphyrin_priority_20261024']
        priority = priority_module.enforce_porphyrin_priority(before,p,f,priority_module.company_p_support(f.shape,retained,f))
        d = priority['dark_native']; emit('D',d)
        if np.any((p!=0)&(d!=0)) or np.any(((p!=0)|(d!=0))&(f==0)):
            raise AssertionError('Independent detector P/D/F invariants failed')
        contrast_p = np.maximum(af[...,0]-(af[...,1]+af[...,2])*.5,0)
        v = af.max(axis=2)
        smooth_v = cv2.GaussianBlur(v,(0,0),3*self.n)
        contrast_d = np.maximum(np.maximum(cv2.GaussianBlur(smooth_v,(0,0),20*self.n)-smooth_v,
                                          cv2.GaussianBlur(smooth_v,(0,0),60*self.n)-smooth_v),0)
        audit.update(seconds=time.monotonic()-started, legacy_retained_point_count=len(retained),
            legacy_point_count_is_not_object_count=True, priority=priority['stats'],
            resolution_qc={'model_pixel_size_native':1/self.n, 'P_final_pixels':int((p!=0).sum()),
                           'D_final_pixels':int((d!=0).sum()), 'native_objects_not_assumed_preserved':True},
            new_object_definition='8connected final masks; geometric centroid computed downstream')
        return {'P':p!=0,'D':d!=0,'F':f!=0,'contrast_P':contrast_p,'contrast_D':contrast_d,'audit':audit}


def detect_signals(rgb, geometry, native_shape, *, tools_dir, vendor_path=None, require_runtime=True):
    return Detector(native_shape,tools_dir=tools_dir,vendor_path=vendor_path,require_runtime=require_runtime).detect(rgb,geometry)
