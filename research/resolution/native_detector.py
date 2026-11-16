"""Rectangular lower-grid adaptation; immutable accepted detector core is isolated."""
import copy,hashlib,math,sys,types
import cv2
import numpy as np
from common import *

def detector_module(native_shape,grid_shape):
 assert sha(CORE)==CORE_SHA
 h0,w0=map(int,native_shape);h,w=map(int,grid_shape)
 assert h<=h0 and w<=w0 and abs(h/h0-w/w0)<1e-12
 text=CORE.read_text(encoding='utf8')
 edits=[('GRID = 1024','GRID = GRID_H'),('(3,GRID,GRID)','(3,GRID_H,GRID_W)'),
        ('(GRID,GRID,3)','(GRID_H,GRID_W,3)'),('(GRID,GRID)','(GRID_H,GRID_W)')]
 for a,b in edits:
  assert text.count(a)==1,('Ambiguous rectangular-grid adaptation',a,text.count(a));text=text.replace(a,b)
 name='_native_grid_detector_%dx%d_%dx%d'%(h0,w0,h,w)
 m=types.ModuleType(name);m.__file__=str(CORE);m.__dict__.update(GRID_H=h,GRID_W=w);sys.modules[name]=m
 exec(compile(text,str(CORE), 'exec'),m.__dict__)
 def spec(shape):
  assert tuple(shape)==(h0,w0)
  n=h/h0;work=min(h0,1600)/h0;r=n/work
  return dict(native_shape_hw=[h0,w0],model_grid=[h,w],center_crop_xyxy=None,
     native_to_model_length=n,original_native_to_work_length=work,original_work_to_model_length=r,
     native_to_model_area=n*n,original_work_to_model_area=r*r,
     integer_length_rule='Original accepted nearest integer/ties upward/minimum1',
     integer_area_rule='Original accepted ceil minima/floor maxima/minimum1',
     median_width_rule='Original accepted nearest odd/minimum3',
     intensity_ratio_shape_and_sector_count_parameters='unchanged',
     actual_endpoint='rectangular native lower grid; no image enlargement/crop')
 m.scale_spec=spec
 m.rectangular_adaptation_sha256=hashlib.sha256(text.encode()).hexdigest()
 return m

def geometry(m,landmarks,F,oldF,native_shape,grid_shape):
 h0,w0=map(int,native_shape);h,w=map(int,grid_shape)
 land=copy.deepcopy(landmarks);xy=np.asarray(land['native_xy'],np.float64)*np.array([w/w0,h/h0])
 land['native_xy']=xy.tolist()
 if 'source' in land and 'native_shape_hw' in land['source']:land['source']['native_shape_hw']=[h,w]
 scope={'np':np,'cv2':cv2,'ast':m.ast,'CONNECTIONS':TOOLS.parent/'assets/skin_mi_subject1000_v1/face_mesh_connections.py',
        'NOSTRIL_ANCHORS':{'right_nostril_guard':[218,237,44,20,242,99,97,240,219],
                          'left_nostril_guard':[438,457,274,250,462,328,326,460,439]}}
 fns=m._named_functions(TOOLS/'uv_darkspot_anatomical_exclusions_v1.py',
                       ['official_eye_brow_ids','_filled_guard','build_dark_exclusions'],scope)
 anatomy=fns['build_dark_exclusions'](land,(h,w))['mask_native']!=0
 fp=binary_project(F,(h,w));op=binary_project(oldF,(h,w))&fp
 return dict(F=fp,old_F=op,anatomy=anatomy,audit=dict(native_shape_hw=[h0,w0],grid_shape_hw=[h,w],
     F_projection='Original geometry only: BOX coverage>=.5; P/D are never projected',
     F_array_sha256=array_sha(fp),oldF_array_sha256=array_sha(op),anatomy_sha256=array_sha(anatomy),
     landmarks_original_native_sha256=array_sha(np.asarray(landmarks['native_xy'],np.float64))))

def detect(uv,native_shape,landmarks,F,oldF,*,repeat=False):
 assert uv.dtype==np.uint8 and uv.ndim==3 and uv.shape[2]==3
 m=detector_module(native_shape,uv.shape[:2]);g=geometry(m,landmarks,F,oldF,native_shape,uv.shape[:2])
 before={k:array_sha(g[k]) for k in ('F','old_F','anatomy')}
 d=m.Detector(native_shape,tools_dir=TOOLS,vendor_path=VENDOR)
 result=d.detect(uv.astype(np.float32)/255.,g)
 if repeat:
  again=d.detect(uv.astype(np.float32)/255.,g)
  assert all(np.array_equal(result[k],again[k]) for k in ('P','D','F'))
 assert before=={k:array_sha(g[k]) for k in before},'Geometry mutated'
 masks={k:result[k] for k in ('P','D','F')};masks['O']=masks['F']&~(masks['P']|masks['D'])
 assert not np.any(masks['P']&masks['D'])
 assert np.array_equal(masks['P']|masks['D']|masks['O'],masks['F'])
 assert not np.any((masks['P']|masks['D'])&~masks['F'])
 protocol=d.protocol();protocol['schema']='native_rectangular_resolution_detector_v1'
 protocol.update(original_core=dict(path=str(CORE),sha256=CORE_SHA),rectangular_adaptation_sha256=m.rectangular_adaptation_sha256,
                 image_enlargement=False,center_crop=False,repeat_mask_exact=bool(repeat))
 return masks,dict(detector_protocol=protocol,execution=result['audit'],geometry=g['audit'])
