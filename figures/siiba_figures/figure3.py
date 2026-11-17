"""Render Figure 3 from accepted tables and matched UV, PLH and PLV crops."""
from pathlib import Path
from .input_paths import INPUT_ROOT
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image
try:
    from . import style
except ImportError:
    import style

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = INPUT_ROOT / 'figure3'
MODELS = list(style.MODEL_ORDER)
KEYS = ['unet_single', 'unet_multi', 'unetplusplus', 'swin', 'ldm']
CN = ['单目标 U-Net', '多目标 U-Net', 'U-Net++', 'Swin', 'LDM']
TICKS = ['Single\nU-Net', 'Multi\nU-Net', 'U-Net++', 'Swin', 'LDM']
REGIONS = list(style.REGION_ORDER)
COLORS = [style.REGION_COLORS[r] for r in REGIONS]
MARKERS = [style.REGION_MARKERS[r] for r in REGIONS]
MODEL_MARKERS = list(style.MODEL_MARKERS)
REGION_NAMES = [r+'  '+style.REGION_NAMES[r] for r in REGIONS]
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

    plt.rcParams.update({'font.family':style.FONT,'font.size':8.0,'text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':'#737A80','xtick.color':INK,'ytick.color':INK,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white'})
    row_step = .85
    W, H = 8.5, 12.6 + 2 * row_step
    fig = plt.figure(figsize=(W,H),facecolor='white')
    section_offset = 0.

    def text(x,y,s,**kw):
        return fig.text(x/W,1-(section_offset+y)/H,s,va='center',**kw)

    def heading(y,letter,s):
        text(.10,y,letter,fontsize=14,weight='bold')
        text(.37,y,s,fontsize=11,weight='bold')

    def axis(x,top,w,h,grid='y'):
        ax = fig.add_axes([x/W,1-(section_offset+top+h)/H,w/W,h/H])
        ax.tick_params(pad=2,labelsize=8)
        ax.set_axisbelow(True)
        if grid:
            ax.grid(axis=grid,color=GRID,lw=.55)
        return ax

    def title(ax,s):
        ax.set_title(s,loc='left',fontsize=8.5,weight='bold',pad=4)

    def legend(y,labels,colors,markers,x=.45,size=8.2,ncol=None,edge_white=False):
        handles=[]
        for i,(label,color,marker) in enumerate(zip(labels,colors,markers)):
            hollow = edge_white and i==3
            handles.append(Line2D([],[],ls='none',marker=marker,ms=4.5,color=color,
                markerfacecolor='white' if hollow else color,markeredgewidth=.9,label=label))
        fig.legend(handles=handles,loc='center left',bbox_to_anchor=(x/W,1-(section_offset+y)/H),
            ncol=ncol or len(labels),frameon=False,fontsize=size,handletextpad=.4,
            columnspacing=1.35,handlelength=1.05,borderaxespad=0)

    def missing(ax,j):
        ax.text(j,.05,'—',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)

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
    for j,label in enumerate(['Ground truth',*MODELS]):
        text(1.19+j*1.31,.45,label,ha='center',fontsize=8.5,weight='bold' if j==0 else 'normal')
    for i,target in enumerate(TARGETS):
        top = .58 + i * row_step
        text(.30,top+.395,target,ha='center',fontsize=9,weight='bold')
        for j,name in enumerate(columns):
            ax = axis(.64+j*1.31,top,1.10,.79,grid=None)
            ax.imshow(images[target, name].transpose(1,2,0),interpolation='nearest')
            ax.set_axis_off()

    section_offset = 2 * row_step
    heading(1.60,'b','Regional reconstruction error for UV, PLH and PLV')
    legend(1.83,REGION_NAMES,COLORS,MARKERS,x=3.07,edge_white=True)
    for ti,target in enumerate(TARGETS):
        ax=axis(.60+ti*2.63,2.08,2.25,.68)
        title(ax,'RGB → '+target)
        maximum,unit=(.16,.04) if ti==0 else (.032,.008)
        for j,key in enumerate(KEYS):
            if (key,target,'F') not in a.index:
                missing(ax,j)
                continue
            for ri,region in enumerate(REGIONS):
                q=a.loc[(key,target,region)]
                assert 0<=q['mean']-q['sd'] and q['mean']+q['sd']<maximum
                ax.errorbar(j+(ri-1.5)*.19,q['mean'],yerr=q['sd'],fmt=MARKERS[ri],
                    ms=3.8,color=COLORS[ri],mfc='white' if ri==3 else COLORS[ri],
                    mew=.8,lw=.7,capsize=1.4,capthick=.65,zorder=3)
        ax.set(xlim=(-.5,4.5),ylim=(0,maximum),yticks=np.arange(5)*unit,
            xticks=range(5),xticklabels=TICKS)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if ti==0 else '%.3f'))
        if ti==0:
            ax.set_ylabel('L1 error',labelpad=3)

    heading(3.17,'c','Full-face reconstruction error for PLH and PLV compared with UV')
    legend(3.40,['UV','UV − PL','PLH','PLV'],[TARGET_COLORS[0],'#CED4D8',*TARGET_COLORS[1:]],['s']*4,x=4.65)
    for ti,target in enumerate(TARGETS[1:]):
        ax=axis(.60+ti*4.02,3.64,3.63,.56)
        title(ax,'RGB → '+target+' vs RGB → UV')
        for j,model in enumerate(MODELS):
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
        ax.set(xlim=(-.5,4.5),ylim=(0,.07),yticks=[0,.02,.04,.06],xticks=range(5),xticklabels=TICKS)
        if ti==0:
            ax.set_ylabel('L1 error',labelpad=3)

    heading(4.59,'d','Agreement in regional mean colour intensity')
    legend(4.83,MODELS,['#50585E']*5,MODEL_MARKERS,x=1.65)
    areas=['Forehead','Image-left cheek','Image-right cheek','Nose','Chin','Full valid face (F)']
    area_labels=['Forehead','Image-left cheek','Image-right cheek','Nose','Chin','Full face']
    arch=['Single-target U-Net','Multi-target U-Net','U-Net++','Swin','LDM']
    counts=[]
    for ri,(ch,name,color) in enumerate(zip('RGB',['Red','Green','Blue'],['#A14A59','#52775F','#416FA3'])):
        y=5.19+ri*.82
        text(.18,y+.36,name,rotation=90,weight='bold',fontsize=8.2,ha='center',color=color)
        for ti,target in enumerate(TARGETS):
            ax=axis(1.40+ti*2.35,y,2.08,.73,grid='x')
            if ri==0:
                title(ax,'RGB → '+target)
            series=B[target+'_'+ch].unstack(level='Model').loc[areas,arch]
            count=0
            for i,(_,row) in enumerate(series.iterrows()):
                vals=row.to_numpy(dtype=float)
                yy=5-i
                ax.plot([np.nanmin(vals),np.nanmax(vals)],[yy,yy],lw=.7,color=color,alpha=.5)
                span=.724 if ti==0 else .152
                present=np.where(np.isfinite(vals))[0]
                ordered=present[np.argsort(vals[present],kind='stable')]
                clusters=np.split(ordered,np.where(np.diff(vals[ordered])>span*.045)[0]+1)
                for cluster in clusters:
                    group=sorted(cluster)
                    offsets=(np.arange(len(group))-(len(group)-1)/2)*.18
                    for j,dy in zip(group,offsets):
                        ax.scatter(vals[j],yy+dy,marker=MODEL_MARKERS[j],s=19,color=color,
                            edgecolor='white',linewidths=.35,zorder=3)
                        count+=1
            counts.append(count)
            ax.set(ylim=(-.55,5.55),yticks=range(6),yticklabels=area_labels[::-1] if ti==0 else [],
                xlim=(.285,1.009) if ti==0 else (.85,1.002),
                xticks=[.3,.5,.7,.9,1.] if ti==0 else [.85,.9,.95,1.])
            ax.tick_params(axis='x',labelbottom=(ri==2),length=2 if ri==2 else 0)
            ax.tick_params(axis='y',length=0,pad=3,labelsize=8)
            ax.spines['left'].set_visible(False)
            if ri==2:
                ax.set_xlabel('ICC(A,1)',labelpad=3)
    assert counts==[30,30,18]*3

    heading(8.02,'e','UV counts, area and local detection')
    legend(8.02,REGION_NAMES[:2],COLORS[:2],MARKERS[:2],x=5.65)
    ctitles=['Count MAE','Count ICC','Area MAE\n(pp)','Area ICC','Local\nAUROC','Local\nPR-AUC']
    limits=[(-20,630),(-.03,1.),(-.08,2.55),(-.03,1.),(.49,.82),(.48,.92)]
    ticks=[[0,300,600],[0,.5,1],[0,1,2],[0,.5,1],[.5,.65,.8],[.5,.7,.9]]
    plotted=0
    for i,(name,lim,xt) in enumerate(zip(ctitles,limits,ticks)):
        ax=axis(1.20+i*1.20,8.51,1.01,.65,grid='x')
        ax.set_title(name,fontsize=8.2,weight='bold',pad=5)
        for si,signal in enumerate('PD'):
            row=C.loc[C.iloc[:,0]==signal].iloc[i]
            for j,model in enumerate(CN):
                cell=str(row[model])
                value=float(cell.split()[0]);y=4-j+(.14 if si==0 else -.14)
                if '[' in cell:
                    lo,hi=map(float,re.search(r'\[([^\]]+)\]',cell).group(1).split(','))
                    ax.plot([lo,hi],[y,y],lw=.85,color=COLORS[si])
                ax.plot(value,y,marker=MARKERS[si],color=COLORS[si],ms=3.8,lw=0)
                plotted+=1
        ax.set(ylim=(-.55,4.55),yticks=range(5),yticklabels=MODELS[::-1] if i==0 else [],xlim=lim,xticks=xt)
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
    assert plotted==60

    heading(9.47,'f','Spatial matching of UV objects')
    legend(9.47,REGION_NAMES[:2],COLORS[:2],MARKERS[:2],x=5.65)
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
            ax.plot(range(5),vals,marker=MARKERS[si],ms=3.8,lw=1.0,color=COLORS[si])
        ax.set(xlim=(-.24,4.24),xticks=range(5),xticklabels=TICKS)
        if i<3:
            ax.tick_params(labelbottom=False)
        if i<4:
            ax.set(ylim=(0,.51),yticks=[0,.25,.5])
        elif i==4:
            ax.set(ylim=(2.,3.12),yticks=[2.,2.5,3.])
        else:
            ax.set(ylim=(0,23),yticks=[0,10,20])

    heading(11.36,'g','Sensitivity to spatial shifts')
    legend(11.36,REGION_NAMES[:3],COLORS[:3],MARKERS[:3],x=4.47)
    for j,key in enumerate(KEYS):
        ax=axis(.60+j*1.59,11.75,1.35,.42)
        title(ax,MODELS[j])
        for ri,region in enumerate(REGIONS[:3]):
            q=shift[(shift.model==key)&(shift.region==region)].sort_values('shift_px')
            assert list(q.shift_px)==[0,1,2,3,5]
            ax.plot(q.shift_px,q.relative_mean_change_percent,marker=MARKERS[ri],ms=3.3,color=COLORS[ri],lw=1.)
        ax.set(xlim=(-.18,5.18),ylim=(-.5,10.5),yticks=[0,5,10],xticks=[0,1,2,3,5])
        if j==0:
            ax.set_ylabel('ΔL1 (%)',labelpad=3)
        else:
            ax.tick_params(labelleft=False)
    text(4.48,12.44,'Predicted UV displacement (px)',ha='center',fontsize=8.5)

    for suffix in ['pdf','svg','png']:
        fig.savefig(OUT/('Figure3.'+suffix),dpi=350)
    plt.close(fig)
    with Image.open(OUT/'Figure3.png') as im:
        im.thumbnail((1800,2668),Image.Resampling.LANCZOS)
        im.save(OUT/'Figure3_preview.png')
    provenance['numeric_counts']={'3A_L1':52,'full_face_paired_differences':8,'3B_ICC':234,'3C':60,'3D':60,'spatial_shift':75}
    provenance['polarized_crops'] = pl_info
    (OUT/'Figure3_ROI_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')

    caption_en='''Figure 3. Reconstruction from RGB across image, anatomical and UV-signal measurements. The five models are single-target U-Net (Single U-Net), multi-target U-Net (Multi U-Net), U-Net++, Swin Transformer (Swin) and latent diffusion model (LDM). (a) Rows show UV, PLH and PLV from one held-out capture. The ground truth column shows the measured target and the next five columns show reconstructions from its paired RGB image. All panels use the same nasal crop and display range. (b) Regional mean absolute reconstruction error (L1) for UV, parallel-polarized light (PLH) and cross-polarized light (PLV). Error bars show one between-capture standard deviation. P, porphyrin-signal region; D, UV-spot region; O, other valid skin; F, full valid face. (c) Full-face L1 on matched captures. UV and PL bars show the two reconstruction errors; the grey segment shows the paired reduction from UV to the polarized target. (d) Absolute-agreement ICC(A,1) between measured and reconstructed regional mean red, green and blue intensities. Symbols identify models, and horizontal segments connect the lowest and highest model estimates in each anatomical region. Slight vertical offsets separate overlapping symbols. (e) UV object-count and area agreement and local detection. Count MAE measures the absolute difference in object count; count ICC measures count agreement. Area MAE measures the error in occupied area fraction, in percentage points (pp); area ICC measures area-fraction agreement. Local AUROC and PR-AUC assess the detection of signal presence in local windows. (f) Mask Dice, object precision, recall and F1, matched-centroid error, and the number of captures with matched objects, using a 5-pixel centroid tolerance. k denotes thousands of captures. (g) Relative change in regional L1 after shifting the predicted UV image by the indicated displacement, without added blur. A dash marks an unavailable estimate for that panel.'''
    caption_cn='''图3　从图像、解剖区域和UV信号三个尺度评价RGB重建。五种模型分别为单目标U-Net（Single U-Net）、多目标U-Net（Multi U-Net）、U-Net++、Swin Transformer（Swin）和潜扩散模型（LDM）。（a）三行依次为同一测试样本的 UV、PLH 和 PLV。Ground truth 列为各模态的实测图像，右侧五列为五个模型从配对 RGB 图像得到的重建。所有图像均使用相同的鼻部裁剪坐标和显示范围。（b）UV、PLH和PLV各区域的平均绝对重建误差L1，误差线表示记录间的一个标准差。P为卟啉信号区，D为UV斑区，O为其余有效皮肤，F为全脸有效区域。（c）在配对记录上比较全脸L1。UV和PL柱表示各自重建误差，灰色区段表示UV相对于对应偏振模态的配对误差差值。（d）实测与重建图像中各解剖区域平均红、绿、蓝通道强度的绝对一致性ICC(A,1)。不同符号表示模型，水平线连接该区域模型估计的最小值与最大值；轻微纵向错位用于分开重叠符号。（e）UV对象数量、面积及局部存在检测。Count MAE表示对象计数的平均绝对误差，Count ICC表示计数一致性；Area MAE表示面积占比误差，单位为百分点，Area ICC表示面积占比一致性；Local AUROC与PR-AUC评价局部窗口内是否存在目标信号。（f）在质心距离容限为5像素时，评价掩膜Dice、对象匹配精确率、召回率、F1、匹配对象中心误差和存在匹配对象的记录数，k表示千条记录。（g）在不额外模糊的条件下，移动预测UV图像后各区域L1的相对变化。横线表示该子图缺少已验收估计值。'''
    (OUT/'Figure3_caption_EN.txt').write_text(caption_en,encoding='utf8')
    (OUT/'Figure3_caption_CN.txt').write_text(caption_cn,encoding='utf8')
    return {name: OUT/name for name in ['Figure3.pdf','Figure3.svg','Figure3.png','Figure3_preview.png']}


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
