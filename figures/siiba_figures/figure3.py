"""Render Figure 3 from accepted tables and matched UV, PLH and PLV crops."""
from pathlib import Path
from .input_paths import INPUT_ROOT
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
try:
    from . import style
except ImportError:
    import style

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = INPUT_ROOT / 'figure3'
PAIR_TABLE_MODELS = list(style.MODEL_ORDER)
MODEL_NAMES = ('Single-target U-Net', 'Multi-target U-Net', 'U-Net++',
               'Swin Transformer', 'Latent diffusion model')
KEYS = ['unet_single', 'unet_multi', 'unetplusplus', 'swin', 'ldm']
CN = ['单目标 U-Net', '多目标 U-Net', 'U-Net++', 'Swin', 'LDM']
# Wrap at word boundaries while keeping the spelling used in panel a.
WRAPPED_NAMES = [name.replace('target U-Net', 'target\nU-Net')
                 .replace('Swin Transformer', 'Swin\nTransformer')
                 .replace('Latent diffusion model', 'Latent\ndiffusion model')
                 for name in MODEL_NAMES]
REGIONS = list(style.REGION_ORDER)
COLORS = [style.REGION_COLORS[r] for r in REGIONS]
MODEL_MARKERS = list(style.MODEL_MARKERS)
TARGETS = list(style.TARGET_ORDER)
TARGET_COLORS = ['#414953', '#B77B64', '#79668F']
INK, GRID = '#20252A', '#E4E7E9'

def _build(output_dir):
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUT/'source_manifest.json').read_text(encoding='utf8'))
    for name, entry in manifest['packaged_files'].items():
        assert hashlib.sha256((INPUT/name).read_bytes()).hexdigest() == entry['sha256'], name
    a = pd.read_csv(INPUT/'reconstruction.csv',float_precision='round_trip')
    assert a.status.eq('accepted').all()
    a = a.set_index(['model','target','region']).sort_index()
    reference = pd.read_csv(INPUT/'Table3A_L1_reference.csv').set_index(['model','target','region'])
    assert reference.index.is_unique and a.index.is_unique
    assert set(reference.index) == set(a.index)
    for index, row in a.iterrows():
        assert f"{row['mean']:.3f} ± {row['sd']:.3f}" == reference.at[index,'L1_mean_SD']
    pairs = pd.read_csv(INPUT/'paired_full_face.csv',float_precision='round_trip').set_index(['model','PL_target'])
    shift = pd.read_csv(INPUT/'spatial_shift.csv',float_precision='round_trip')
    assert (len(a),len(pairs),len(shift)) == (52,8,75)
    C = pd.read_csv(INPUT/'Table3C.csv')
    D = pd.read_csv(INPUT/'Table3D.csv')
    B = pd.read_csv(INPUT/'Table3B.csv',na_values=['TBD']).set_index(['Model','Anatomical region'])
    assert list(B.count().values)==[30]*3+[30]*3+[18]*3

    plt.rcParams.update({'font.family':style.FONT,'font.size':10,'text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':'#737A80','xtick.color':INK,'ytick.color':INK,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white'})
    row_step = .85
    crop_header_space = .19
    icc_height = 1.30
    icc_row_step = icc_height + .12
    icc_extra_height = 3 * (icc_row_step - .82)
    slanted_tick_space = .62
    shift_title_space = .20
    W, H = 8.5, 12.6 + 2 * row_step + crop_header_space + icc_extra_height + 1.01 + 3*slanted_tick_space + shift_title_space
    fig = plt.figure(figsize=(W,H),facecolor='white')
    section_offset = 0.

    def text(x,y,s,**kw):
        return fig.text(x/W,1-(section_offset+y)/H,s,va='center',**kw)

    def heading(y,letter,s):
        text(.10,y,letter,fontsize=16,weight='bold')
        text(.41,y,s,fontsize=13.5,weight='bold')

    def axis(x,top,w,h,grid='y'):
        ax = fig.add_axes([x/W,1-(section_offset+top+h)/H,w/W,h/H])
        ax.tick_params(pad=2,labelsize=10)
        ax.set_axisbelow(True)
        if grid:
            ax.grid(axis=grid,color=GRID,lw=.55)
        return ax

    def title(ax,s):
        ax.set_title(s,loc='left',fontsize=10.5,weight='bold',pad=4)

    def region_legend(y, regions, x):
        style.add_region_legend(fig, regions, (x/W, 1-(section_offset+y)/H),
                                fontsize=10, loc='center left', borderaxespad=0)

    def legend(y,labels,colors,markers,x=.45,size=10,ncol=None,marker_size=5.6):
        handles = [Line2D([], [], ls='none', marker=marker, ms=marker_size, color=color,
                          markeredgewidth=.9, label=label)
                   for label, color, marker in zip(labels, colors, markers)]
        fig.legend(handles=handles,loc='center left',bbox_to_anchor=(x/W,1-(section_offset+y)/H),
            ncol=ncol or len(labels),frameon=False,fontsize=size,handletextpad=.4,
            columnspacing=.95,handlelength=1.05,borderaxespad=0)

    def slant_model_names(ax):
        for tick in ax.get_xticklabels():
            tick.set(rotation=55, ha='right', va='top', rotation_mode='anchor')
        ax.tick_params(axis='x', pad=4)

    def missing(ax,j):
        ax.text(j,.05,'—',transform=ax.get_xaxis_transform(),ha='center',fontsize=10)

    # Fixed, audited ROI.
    heading(.16,'a','Reconstruction of UV, PLH and PLV in the same nasal region')
    provenance = json.loads((INPUT/'ROI_provenance.json').read_text(encoding='utf8'))
    assert provenance['independent_audit']['status']=='passed'
    with np.load(INPUT/provenance['roi_file'],allow_pickle=False) as bundle:
        roi = {name:bundle[name].copy() for name in bundle.files}
    for name,arr in roi.items():
        assert arr.shape == tuple(provenance['crop_shape'])
        assert str(arr.dtype) == provenance['dtype']
        assert hashlib.sha256(arr.tobytes(order='C')).hexdigest() == provenance['array_sha256'][name]
    columns = ['truth', *['prediction_' + key for key in KEYS]]
    images = {('UV', name): arr for name, arr in roi.items()}
    pl_info = json.loads((INPUT/'polarized_roi_provenance.json').read_text(encoding='utf8'))
    assert pl_info['case_index'] == provenance['case_index'] and pl_info['crop_xyxy'] == provenance['crop_xyxy']
    with np.load(INPUT/'polarized_roi.npz', allow_pickle=False) as bundle:
        assert set(bundle.files) == {target + '_' + name for target in ('plh','plv') for name in columns}
        for key in bundle.files:
            modality, name = key.split('_', 1)
            arr = bundle[key]
            assert arr.shape == tuple(provenance['crop_shape']) and arr.dtype == np.float32
            assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
            assert hashlib.sha256(arr.tobytes(order='C')).hexdigest() == pl_info['array_sha256'][key]
            images[modality.upper(), name] = arr
    crop_labels = ['Ground truth', *MODEL_NAMES]
    for j,label in enumerate(crop_labels):
        label = label.replace('target U-Net','target\nU-Net').replace('diffusion model','diffusion\nmodel')
        text(1.19+j*1.31,.55,label,ha='center',fontsize=10.5,weight='bold' if j==0 else 'normal')
    for i,target in enumerate(TARGETS):
        top = .58 + crop_header_space + i * row_step
        text(.30,top+.395,target,ha='center',fontsize=11,weight='bold')
        for j,name in enumerate(columns):
            ax = axis(.64+j*1.31,top,1.10,.79,grid=None)
            ax.imshow(images[target, name].transpose(1,2,0),interpolation='nearest')
            ax.set_axis_off()

    section_offset = 2 * row_step + crop_header_space
    heading(1.63,'b','Regional reconstruction error')
    region_legend(1.63,REGIONS,x=3.91)
    for ti,target in enumerate(TARGETS):
        ax=axis(.61+ti*2.72,2.00,2.19,.76)
        title(ax,'RGB → '+target)
        maximum,unit=(.16,.04) if ti==0 else (.032,.008)
        for j,key in enumerate(KEYS):
            if (key,target,'F') not in a.index:
                missing(ax,j)
                continue
            for ri,region in enumerate(REGIONS):
                q=a.loc[(key,target,region)]
                assert 0<=q['mean']-q['sd'] and q['mean']+q['sd']<maximum
                point = style.region_line_style(region, scale=1.25)
                point['ls'] = 'none'
                ax.errorbar(j+(ri-1.5)*.19,q['mean'],yerr=q['sd'],fmt='',
                    **point,elinewidth=.7,capsize=1.4,capthick=.65,zorder=3)
        ax.set(xlim=(-.5,4.5),ylim=(0,maximum),yticks=np.arange(5)*unit,
            xticks=range(5),xticklabels=WRAPPED_NAMES)
        slant_model_names(ax)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if ti==0 else '%.3f'))
        if ti==0:
            ax.set_ylabel('L1 error',labelpad=3)

    section_offset += .12 + slanted_tick_space
    heading(3.17,'c','Paired full-face reconstruction error')
    legend(3.17,['UV','UV − PL','PLH','PLV'],[TARGET_COLORS[0],'#CED4D8',*TARGET_COLORS[1:]],['s']*4,x=5.02)
    for ti,target in enumerate(TARGETS[1:]):
        ax=axis(.60+ti*4.02,3.60,3.63,.62)
        title(ax,'RGB → '+target+' vs RGB → UV')
        for j,model in enumerate(PAIR_TABLE_MODELS):
            if (model,target) not in pairs.index:
                missing(ax,j)
                continue
            q=pairs.loc[(model,target)]
            uv,delta=q.UV_reference_mean,q.mean_UV_minus_PL
            pl=a.loc[(KEYS[j],target,'F'),'mean']
            assert q.n_paired==21422 and abs(uv-pl-delta)<1e-8
            assert max(uv,pl) < .07
            ax.bar(j-.25,uv,width=.18,color=TARGET_COLORS[0],zorder=3)
            ax.bar(j,delta,bottom=pl,width=.18,color='#CED4D8',edgecolor='#8A949B',lw=.55,zorder=3)
            ax.bar(j+.25,pl,width=.18,color=TARGET_COLORS[ti+1],zorder=3)
            ax.plot([j-.16,j-.09],[uv,uv],color='#7C858B',lw=.6)
            ax.plot([j+.09,j+.16],[pl,pl],color='#7C858B',lw=.6)
        ax.set(xlim=(-.5,4.5),ylim=(0,.07),yticks=[0,.02,.04,.06],xticks=range(5),xticklabels=WRAPPED_NAMES)
        slant_model_names(ax)
        if ti==0:
            ax.set_ylabel('L1 error',labelpad=3)

    section_offset += .21 + slanted_tick_space
    # Split in the blank space between panels c and d at the same native scale.
    page_break_y = section_offset + 4.44
    heading(4.59,'d','Agreement in regional mean colour intensity')
    legend(4.96,MODEL_NAMES,['#50585E']*5,MODEL_MARKERS,x=.36,marker_size=8.5)
    areas=['Forehead','Image-left cheek','Image-right cheek','Nose','Chin','Full valid face (F)']
    area_labels=['Forehead','Image-left cheek','Image-right cheek','Nose','Chin','Full face']
    arch=['Single-target U-Net','Multi-target U-Net','U-Net++','Swin','LDM']
    counts=[]
    for ri,(ch,name,color) in enumerate(zip('RGB',['Red','Green','Blue'],['#A14A59','#52775F','#416FA3'])):
        y=5.34+ri*icc_row_step
        text(.11,y+icc_height/2,name,rotation=90,weight='bold',fontsize=10.5,ha='center',color=color)
        for ti,target in enumerate(TARGETS):
            ax=axis(1.48+ti*2.34,y,2.02,icc_height,grid='x')
            limits = [(.285,1.014), (.909,1.005), (.85,1.007)][ti]
            ax.set(xlim=limits,ylim=(-.60,5.60))
            for band in (0,2,4):
                ax.axhspan(band-.50,band+.50,color='#F5F6F7',lw=0,zorder=0)
            if ri==0:
                title(ax,'RGB → '+target)
            series=B[target+'_'+ch].unstack(level='Model').loc[areas,arch]
            count=0
            for i,(_,row) in enumerate(series.iterrows()):
                vals=row.to_numpy(dtype=float)
                yy=5-i
                ax.plot([np.nanmin(vals),np.nanmax(vals)],[yy,yy],lw=.7,color=color,alpha=.5)
                # Place nearby symbols vertically in print points; x always remains the measured ICC.
                present = np.flatnonzero(np.isfinite(vals))
                order = present[np.argsort(vals[present], kind='stable')]
                x_points = vals * (2.02 * 72 / (limits[1] - limits[0]))
                offsets = {}
                clearance = 7.3
                for j in order:
                    candidates = [0.]
                    neighbours = []
                    for k, prior_y in offsets.items():
                        dx = abs(x_points[j] - x_points[k])
                        if dx < clearance:
                            gap = np.sqrt(clearance**2 - dx**2)
                            candidates.extend((prior_y-gap, prior_y+gap))
                            neighbours.append((dx, prior_y))
                    offsets[j] = min((yoff for yoff in candidates if all(
                        np.hypot(dx, yoff-prior_y) >= clearance-1e-8
                        for dx, prior_y in neighbours)), key=abs)
                centre = (min(offsets.values()) + max(offsets.values())) / 2
                for j, offset in offsets.items():
                    dy = (offset-centre) * 6.20 / (icc_height*72)
                    ax.scatter(vals[j], yy+dy, marker=MODEL_MARKERS[j], s=43,
                               color=color, edgecolor='white', linewidths=.45, zorder=3)
                    count += 1
            counts.append(count)
            ax.set(yticks=range(6),yticklabels=area_labels[::-1] if ti==0 else [],
                xticks=[[.3,.5,.7,.9,1.], [.92,.94,.96,.98,1.], [.85,.9,.95,1.]][ti])
            ax.tick_params(axis='x',labelbottom=(ri==2),length=2 if ri==2 else 0)
            ax.tick_params(axis='y',length=0,pad=3,labelsize=10)
            ax.spines['left'].set_visible(False)
            if ri==2:
                ax.set_xlabel('ICC(A,1)',labelpad=3)
    assert counts==[30,30,18]*3

    section_offset += icc_extra_height + .28
    heading(8.02,'e','UV counts, area and local detection')
    region_legend(8.02,REGIONS[:2],x=5.62)
    ctitles=['Count MAE','Count ICC','Area MAE\n(pp)','Area ICC','Local\nAUROC','Local\nPR-AUC']
    limits=[(-20,630),(-.03,1.),(-.08,2.55),(-.03,1.),(.49,.82),(.48,.92)]
    ticks=[[0,300,600],[0,.5,1],[0,1,2],[0,.5,1],[.5,.65,.8],[.5,.7,.9]]
    plotted=0
    for i,(name,lim,xt) in enumerate(zip(ctitles,limits,ticks)):
        ax=axis(1.58+i*1.14,8.55,.90,.85,grid='x')
        ax.set_title(name,fontsize=10,weight='bold',pad=5)
        for si,signal in enumerate('PD'):
            row=C.loc[C.iloc[:,0]==signal].iloc[i]
            for j,model in enumerate(CN):
                cell=str(row[model])
                value=float(cell.split()[0]);y=4-j+(.14 if si==0 else -.14)
                if '[' in cell:
                    lo,hi=map(float,re.search(r'\[([^\]]+)\]',cell).group(1).split(','))
                    ax.plot([lo,hi],[y,y],lw=.85,color=COLORS[si])
                ax.plot([value],[y],**style.region_line_style(signal, scale=1.25))
                plotted+=1
        ax.set(ylim=(-.55,4.55),yticks=range(5),yticklabels=MODEL_NAMES[::-1] if i==0 else [],xlim=lim,xticks=xt)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
    assert plotted==60

    section_offset += .30
    heading(9.47,'f','Spatial matching of UV objects')
    region_legend(9.47,REGIONS[:2],x=5.62)
    dtitles=['Mask Dice','Precision','Recall','F1','Centre error (px)','Matched groups (k)']
    for i,name in enumerate(dtitles):
        ax=axis(.60+(i%3)*2.63,9.84 if i<3 else 10.58,2.25,.37)
        title(ax,name)
        for si,signal in enumerate('PD'):
            sub=D[(D.iloc[:,0]==signal)&(D.iloc[:,2]!='端点有效组数 N')]
            assert len(sub)==6
            vals=np.array([float(str(sub.iloc[i][m]).split()[0]) for m in CN])
            if i==5:
                vals/=1000
            ax.plot(range(5),vals,**style.region_line_style(signal, scale=1.25))
        ax.set(xlim=(-.24,4.24),xticks=range(5),xticklabels=WRAPPED_NAMES)
        if i<3:
            ax.tick_params(labelbottom=False)
        else:
            slant_model_names(ax)
        if i<4:
            ax.set(ylim=(0,.51),yticks=[0,.25,.5])
        elif i==4:
            ax.set(ylim=(2.,3.12),yticks=[2.,2.5,3.])
        else:
            ax.set(ylim=(0,23),yticks=[0,10,20])

    section_offset += .10 + slanted_tick_space
    heading(11.36,'g','Sensitivity to spatial shifts')
    region_legend(11.36,REGIONS[:3],x=4.15)
    for j,key in enumerate(KEYS):
        ax=axis(.60+j*1.59,11.75+shift_title_space,1.35,.42)
        title(ax,WRAPPED_NAMES[j])
        for ri,region in enumerate(REGIONS[:3]):
            q=shift[(shift.model==key)&(shift.region==region)].sort_values('shift_px')
            assert list(q.shift_px)==[0,1,2,3,5]
            ax.plot(q.shift_px,q.relative_mean_change_percent,
                    **style.region_line_style(region, scale=1.25))
        ax.set(xlim=(-.18,5.18),ylim=(-.5,10.5),yticks=[0,5,10],xticks=[0,1,2,3,5])
        if j==0:
            ax.set_ylabel('ΔL1 (%)',labelpad=3)
        else:
            ax.tick_params(labelleft=False)
    text(4.48,12.44+shift_title_space,'Predicted UV displacement (px)',ha='center',fontsize=10.5)

    for suffix in ['pdf','svg','png']:
        fig.savefig(OUT/('Figure3.'+suffix),dpi=350)
    with PdfPages(OUT/'Figure3_two_pages.pdf') as pages:
        for bottom, top in [(H-page_break_y, H), (0, H-page_break_y)]:
            pages.savefig(fig, bbox_inches=Bbox.from_extents(0, bottom, W, top),
                          pad_inches=0)
    plt.close(fig)
    with Image.open(OUT/'Figure3.png') as im:
        im.thumbnail((1800,2668),Image.Resampling.LANCZOS)
        im.save(OUT/'Figure3_preview.png')
    provenance['numeric_counts']={'3A_L1':52,'full_face_paired_differences':8,'3B_ICC':234,'3C':60,'3D':60,'spatial_shift':75}
    provenance['polarized_crops'] = pl_info
    (OUT/'Figure3_ROI_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')

    caption_en='''Figure 3. Accuracy of UV and polarized image reconstruction from RGB.
(a) Matched nasal crops from one held-out capture. Rows show UV-excited fluorescence (UV), parallel-polarized light (PLH) and cross-polarized light (PLV). The first column shows the measured target (Ground truth). The next five columns show reconstructions from the same RGB image by Single-target U-Net, Multi-target U-Net, U-Net++, Swin Transformer and Latent diffusion model, respectively.
(b) Regional reconstruction error for each model and target modality. Points show mean absolute error (L1) between reconstructed and measured images. Error bars show one standard deviation across captures. P denotes the porphyrin-signal region, D the UV-spot region, O the remaining valid skin and F the full valid face.
(c) Full-face L1 for UV versus PLH (left) and UV versus PLV (right). For each model, the grey segment shows the mean paired difference in L1, calculated as UV minus the corresponding polarized target.
(d) Agreement between reconstructed and measured mean red, green and blue intensities in each anatomical region and across the full face. Values of the absolute-agreement intraclass correlation coefficient, ICC(A,1), closer to 1 indicate greater agreement. Symbols identify the models. Each horizontal line spans the lowest to highest model estimate for that region.
(e) Porphyrin and UV-spot counts, area fractions and local detection. Mean absolute error (MAE) and ICC quantify error and agreement in counts and area fractions. Area-fraction MAE is expressed in percentage points. AUROC and PR-AUC assess detection of signal presence in 64 × 64-pixel windows.
(f) Spatial agreement of reconstructed and measured UV signals. Dice measures overlap of the signal masks. Individual objects are matched within a 5-pixel centroid tolerance. The plots report precision, recall, F1, distance between matched centroids and the number of captures with at least one match (in thousands).
(g) The reconstructed UV image was shifted by 0, 1, 2, 3 or 5 pixels while the measured UV image and region masks were held fixed. Curves show the percentage change in mean L1 within P, D and O relative to the unshifted reconstruction.'''
    caption_cn='''图 3　从 RGB 重建 UV 和偏振图像的准确性。
（a）比较同一受试者鼻部的实测图像与重建图像。三行依次为紫外激发荧光（UV）、平行偏振光（PLH）和交叉偏振光（PLV）。第一列为实测图像（Ground truth），右侧五列依次为 Single-target U-Net、Multi-target U-Net、U-Net++、Swin Transformer 和 Latent diffusion model 从同一张 RGB 图像得到的重建结果。
（b）各模型在三种目标模态上的区域重建误差。点表示重建与实测图像之间的平均绝对误差（L1），误差线表示采集记录间的一个标准差。P 为卟啉信号区，D 为 UV 斑区，O 为其余有效皮肤，F 为全脸有效区域。
（c）比较同一模型重建 UV 和偏振图像时的全脸 L1 误差。左图比较 UV 与 PLH，右图比较 UV 与 PLV。对每次采集，用 UV 误差减去相应的偏振图像误差。灰色区段表示该差值的平均值。
（d）比较重建图像与实测图像在各面部区域及全脸的平均红、绿、蓝通道强度。横轴为绝对一致性组内相关系数 ICC(A,1)，越接近 1 表示一致性越高。符号代表模型，水平线连接同一区域内各模型的最低值和最高值。
（e）比较卟啉信号和 UV 斑的数量、面积占比及局部检出表现。数量和面积占比均用平均绝对误差（MAE）及 ICC 评价，面积占比误差以百分点计。AUROC 和 PR-AUC 衡量模型辨别 64 × 64 像素局部区域内有无相应信号的能力。
（f）比较重建与实测 UV 图像中信号的位置。Dice 衡量信号掩膜的重叠程度。以 5 像素为质心距离容限匹配单个信号，报告精确率、召回率、F1、匹配后的质心距离，以及至少有一个匹配信号的采集记录数（单位为千）。
（g）固定实测 UV 图像和区域掩膜，将重建 UV 图像平移 0、1、2、3 或 5 像素。曲线表示 P、D、O 三区的平均 L1 误差相对于未平移时的变化百分比。'''
    (OUT/'Figure3_caption_EN.txt').write_text(caption_en,encoding='utf8')
    (OUT/'Figure3_caption_CN.txt').write_text(caption_cn,encoding='utf8')
    return {name: OUT/name for name in ['Figure3.pdf','Figure3_two_pages.pdf','Figure3.svg','Figure3.png','Figure3_preview.png']}


def build(output_dir):
    """Render the accepted Figure 3 from the bundled inputs."""
    with plt.rc_context():
        return _build(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'outputs'/'figure3')
    args = parser.parse_args()
    build(args.output_dir)
