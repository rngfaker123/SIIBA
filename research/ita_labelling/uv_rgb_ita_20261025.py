"""Independent RGB-only ITA labels; never writes approved masks or model outputs.

Native bilateral cheek patches, not P/D/O-selected pixels. The robust patch
aggregation and QC are this study's declared estimator, not calibrated colorimetry.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw

METHOD = {
    'version': 'rgb_ita_bilateral_cheek_v1',
    'roi_centers_landmark_mean': [[50, 101, 205], [280, 330, 425]],
    'axis_landmarks': [33, 263],
    'ellipse_radius_interoutereye': [0.07, 0.09],
    'min_native_pixels_per_patch': 256,
    'max_exact_0_or_255_channel_clipped_fraction': 0.10,
    'minimum_patch_median_b_star_exclusive': 1.0,
    'aggregation': 'per-patch componentwise median physical Lab; equal-weight mean of two patch Lab vectors; ITA of resulting L,b',
    'color_space': 'embedded ICC to sRGB relative-colorimetric when present; otherwise explicit sRGB assumption; D65 2-degree CIELAB',
    'formula': 'degrees(atan((L_star-50)/b_star))',
    'geometry': 'existing archived RGB_preview landmarks only; no UV fallback; no new face inference',
    'pixel_selection': 'all native pixel centers inside both ellipses; no lightness threshold, trimming, UV mask selection, whitening or equalization',
    'bilateral_ITA_disagreement_warning_degrees': 15.0,
    'qc_failure_policy': 'retain sample ID; null ITA with reason; does not exclude from original model evaluation',
    'strata': {'very_light': '>55', 'light': '(41,55]', 'intermediate': '(28,41]', 'tan': '(10,28]', 'brown': '(-30,10]', 'dark': '<=-30'},
    'interpretation': 'image-estimated facial ITA, not instrument-calibrated constitutive skin color, race or Fitzpatrick phototype',
    'camera_calibration': 'user confirms data suitable; no measured color-target calibration independently established',
    'references': ['https://skinoptics.readthedocs.io/en/latest/07_colors.html', 'https://www.w3.org/Graphics/Color/srgb', 'https://arxiv.org/abs/2308.09640'],
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf8')

def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def compact_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def save(path, value, mutable=False):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data=encoded(value)
    if mutable:
        import os, uuid
        temp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
        with temp.open('xb') as f: f.write(data)
        os.replace(temp, path)
    elif path.exists():
        require(path.read_bytes()==data, 'Refuse to overwrite '+str(path))
    else:
        with path.open('xb') as f: f.write(data)

def descriptor(path):
    return {'path':str(Path(path).resolve()), 'sha256':sha(path)}

def bound(desc):
    path=Path(desc['path'])
    require(path.is_absolute() and sha(path)==desc['sha256'], 'Input SHA mismatch: '+str(path))
    return path

def checked_mask(path):
    rec=read(path)
    require(rec.get('record_sha256')==digest({k:v for k,v in rec.items() if k!='record_sha256'}), 'Mask receipt seal mismatch')
    return rec

def srgb_to_lab(rgb):
    """Float physical CIELAB, sRGB IEC transfer + D65/2-degree tristimulus."""
    x=np.asarray(rgb, dtype=np.float64)
    require(x.shape[-1]==3 and np.isfinite(x).all() and x.min()>=0 and x.max()<=255, 'Bad RGB values')
    x=x/255.0
    linear=np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4)
    matrix=np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
    xyz=linear@matrix.T / np.array([0.95047,1.0,1.08883])
    delta=6/29
    f=np.where(xyz>delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])), axis=-1)

def ita_from_lab(lab):
    lab=np.asarray(lab,dtype=float)
    require(lab.shape==(3,) and np.isfinite(lab).all(), 'Bad physical Lab')
    require(lab[2]>METHOD['minimum_patch_median_b_star_exclusive'], 'unstable_or_nonpositive_b_star')
    return math.degrees(math.atan((float(lab[0])-50)/float(lab[2])))

def stratum(ita):
    require(math.isfinite(ita) and -90<ita<90, 'Bad ITA')
    for cutoff,name in [(55,'very_light'),(41,'light'),(28,'intermediate'),(10,'tan'),(-30,'brown')]:
        if ita>cutoff:return name
    return 'dark'

def make_rois(landmarks, shape):
    xy=np.asarray(landmarks['native_xy'],dtype=float)
    require(xy.ndim==2 and xy.shape[0]>=455 and xy.shape[1]==2 and np.isfinite(xy).all(), 'Bad archived landmarks')
    require(list(shape)==landmarks['source']['native_shape_hw'], 'Geometry dimensions mismatch')
    axis=xy[263]-xy[33]; distance=float(np.linalg.norm(axis))
    require(distance>0, 'Degenerate eye baseline')
    u=axis/distance; v=np.array([-u[1],u[0]])
    rx,ry=np.array(METHOD['ellipse_radius_interoutereye'])*distance
    result=[]
    h,w=shape
    for indices in METHOD['roi_centers_landmark_mean']:
        center=xy[indices].mean(axis=0)
        extent=np.sqrt((rx*u)**2+(ry*v)**2)
        # Integer coordinates denote native pixel centers, matching archived xy.
        low=np.floor(center-extent).astype(int); high=np.ceil(center+extent).astype(int)
        if (low<0).any() or high[0]>=w or high[1]>=h:
            raise ValueError('ita_patch_outside_image')
        result.append({'center_xy':center.tolist(),'axis_u':u.tolist(),'axis_v':v.tolist(),'radii_xy':[float(rx),float(ry)],'bbox_xyxy':[int(low[0]),int(low[1]),int(high[0]+1),int(high[1]+1)],'landmark_indices':indices})
    return result

def patch_pixels(image, roi):
    x0,y0,x1,y1=roi['bbox_xyxy']
    yy,xx=np.mgrid[y0:y1,x0:x1]
    dx=xx-roi['center_xy'][0]; dy=yy-roi['center_xy'][1]
    u,v=roi['axis_u'],roi['axis_v']; rx,ry=roi['radii_xy']
    keep=((dx*u[0]+dy*u[1])/rx)**2+((dx*v[0]+dy*v[1])/ry)**2<=1
    return np.asarray(image.crop((x0,y0,x1,y1)))[keep]

def evaluate_patches(image, rois):
    patches=[]
    for roi in rois:
        pixels=patch_pixels(image,roi)
        n=len(pixels)
        require(n>=METHOD['min_native_pixels_per_patch'], 'ita_patch_too_small')
        clipped=float(np.any((pixels==0)|(pixels==255),axis=1).mean())
        lab=srgb_to_lab(pixels)
        median=np.median(lab,axis=0)
        entry={'roi':roi,'n_native_pixels':n,'exact_clipped_fraction':clipped,'median_lab':median.tolist(),
               'lab_q10':np.percentile(lab,10,axis=0).tolist(),'lab_q90':np.percentile(lab,90,axis=0).tolist(),'ita_degrees':None,'qc_reason':None}
        if clipped>METHOD['max_exact_0_or_255_channel_clipped_fraction']:
            entry['qc_reason']='ita_patch_excessive_channel_clipping'
        elif median[2]<=METHOD['minimum_patch_median_b_star_exclusive']:
            entry['qc_reason']='ita_patch_unstable_or_nonpositive_b_star'
        else:entry['ita_degrees']=ita_from_lab(median)
        patches.append(entry)
    invalid=[p['qc_reason'] for p in patches if p['qc_reason']]
    if invalid:return {'status':'invalid','reason':';'.join(sorted(set(invalid))),'ita_degrees':None,'stratum':None,'patches':patches}
    combined=np.mean([p['median_lab'] for p in patches],axis=0)
    ita=ita_from_lab(combined)
    spread=abs(patches[0]['ita_degrees']-patches[1]['ita_degrees'])
    return {'status':'ok','reason':None,'ita_degrees':ita,'stratum':stratum(ita),'patches':patches,'aggregate_lab':combined.tolist(),
            'bilateral_ita_absolute_difference':spread,'qc_warnings':['bilateral_ita_difference_gt_15'] if spread>15 else []}

def preview(image, rois, destination, sample_id):
    """Scientific ROI audit only; original pixels are never overwritten."""
    w,h=image.size; side=min(w,h); left=(w-side)//2; top=(h-side)//2
    resample=getattr(Image,'Resampling',Image).LANCZOS
    canvas=image.crop((left,top,left+side,top+side)).resize((720,720),resample)
    draw=ImageDraw.Draw(canvas); colors=[(20,255,50),(255,190,20)]
    for i,roi in enumerate(rois):
        c=np.asarray(roi['center_xy']); u=np.asarray(roi['axis_u']); v=np.asarray(roi['axis_v']); rx,ry=roi['radii_xy']
        points=[]
        for a in np.linspace(0,2*np.pi,101):
            p=c+rx*math.cos(a)*u+ry*math.sin(a)*v
            points.append(tuple(((p-[left,top])*720/side).tolist()))
        draw.line(points,fill=colors[i],width=3)
    draw.text((10,10),sample_id,fill='white')
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as f:canvas.save(f,format='PNG')

def process_case(row,index,mask_path,protocol_sha,preview_path=None,source_root=None):
    rec=checked_mask(mask_path)
    require(rec.get('schema')=='uv_three_region_test_mask_case_v1' and rec['sample_id']==row['sample_id'] and rec['subject_id']==row['subject_id'] and rec['index']==index and rec['native_shape']==row['native_shape'] and rec['source_row_sha256']==digest(row) and rec['sources']['rgb']['sha256']==row['rgb']['sha256'], 'Mask/sample identity mismatch')
    expected=Path(row['rgb']['path'])
    if not expected.is_absolute():
        require(source_root is not None,'Relative source requires declared data_root')
        expected=Path(source_root)/expected
    require(rec['sources']['rgb']=={'path':str(expected.resolve()),'sha256':row['rgb']['sha256']},'RGB source descriptor mismatch')
    result={'sample_id':row['sample_id'],'subject_id':row['subject_id'],'index':index,'protocol_sha256':protocol_sha,'mask_receipt':descriptor(mask_path),'rgb':rec['sources']['rgb'],
            'status':'invalid','reason':None,'ita_degrees':None,'stratum':None}
    if rec['status']!='complete':
        require(rec['status'] in ('invalid_geometry','no_unique_face'), 'Unexpected mask failure')
        result['reason']='upstream_'+rec['status'];return result
    geometry=rec['evidence']['geometry/landmarks.json']; lm=read(bound(geometry))
    result['geometry']=geometry;result['landmark_source']=lm['source']['landmark_source']
    if result['landmark_source']!='RGB_preview':
        require(result['landmark_source']=='UV_preview_fallback','Unknown landmark source')
        result['reason']='rgb_geometry_unavailable_uv_fallback_not_used';return result
    try:rois=make_rois(lm,row['native_shape'])
    except ValueError as exc:
        if str(exc)!='ita_patch_outside_image':raise
        result['reason']=str(exc);return result
    rgb_path=bound(rec['sources']['rgb'])
    with Image.open(rgb_path) as original:
        require(original.mode=='RGB' and list(reversed(original.size))==row['native_shape'], 'Native RGB type/shape differs')
        require(original.getexif().get(274,1)==1, 'Unsupported EXIF orientation; no automatic transformation')
        icc=original.info.get('icc_profile')
        result['icc_sha256']=hashlib.sha256(icc).hexdigest() if icc else None
        image=original.copy()
        if icc:
            # Unknown/broken profile is a real error, not a silent sRGB fallback.
            image=ImageCms.profileToProfile(image,ImageCms.ImageCmsProfile(io.BytesIO(icc)),ImageCms.createProfile('sRGB'),renderingIntent=1,outputMode='RGB')
            result['color_interpretation']='embedded_ICC_to_sRGB_relative_colorimetric'
        else:result['color_interpretation']='assumed_sRGB_no_embedded_ICC'
        try:result.update(evaluate_patches(image,rois))
        except ValueError as exc:
            if str(exc)!='ita_patch_too_small':raise
            result.update(status='invalid',reason=str(exc),ita_degrees=None,stratum=None)
        if preview_path is not None:
            preview(image,rois,preview_path,row['sample_id']);result['preview']=descriptor(preview_path)
    return result
