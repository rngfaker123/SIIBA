"""Build Figure 1 from the approved editable SVG template."""
from copy import deepcopy
import base64
import json
from pathlib import Path
from .input_paths import INPUT_ROOT
from textwrap import dedent
from xml.etree import ElementTree as ET
import hashlib

from .style import REGION_COLORS, REGION_NAMES, REGION_ORDER
from .svg_export import export_svg

ROOT = Path(__file__).resolve().parents[1]
SOURCE = INPUT_ROOT / 'figure1/layout_template.svg'
NS = '{http://www.w3.org/2000/svg}'
XL = '{http://www.w3.org/1999/xlink}'
ET.register_namespace('', NS[1:-1])
ET.register_namespace('xlink', XL[1:-1])


def build(output_dir):
    """Write vector and raster outputs."""
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((SOURCE.parent / 'provenance.json').read_text(encoding='utf8'))
    svg = SOURCE.read_bytes()
    for relative in provenance['image_assets']:
        token = b'"' + relative.encode('ascii') + b'"'
        assert svg.count(token) == 1, relative
        image = (SOURCE.parent / relative).read_bytes()
        svg = svg.replace(token, b'"data:image/png;base64,' + base64.b64encode(image) + b'"')
    assert hashlib.sha256(svg).hexdigest() == provenance['sha256'], "Figure 1 layout changed"
    root = ET.fromstring(svg)
    tree = ET.ElementTree(root)
    panels = {el.get('id'): el for el in root.iter(NS+'g') if el.get('id', '').startswith('panel-')}

    def photo_hashes(panel):
        return [hashlib.sha256(el.get(XL+'href', el.get('href', '')).encode()).hexdigest()
                for el in panel.iter(NS+'image')]

    original_photos = {key: photo_hashes(panels[key]) for key in ['panel-a', 'panel-b', 'panel-e']}

    def text(parent, x, y, value, size=13.4, weight='400', anchor='middle', fill='#202126'):
        item = ET.SubElement(parent, NS+'text', {
            'x': str(x), 'y': str(y), 'font-family': 'Arial, Helvetica, sans-serif',
            'font-size': str(size), 'font-weight': weight, 'fill': fill, 'text-anchor': anchor})
        item.text = value
        return item

    # Figure labels.
    changes = {
        '115,382 groups': '115,382 individuals',
        'Excluded: 5,893 groups': 'Excluded: 5,893 individuals',
        'Test groups by skin tone': 'Skin tone of individuals in the test set',
        'Four-mode facial imaging': 'Four-mode facial imaging',
    }
    for key, panel in panels.items():
        for el in panel.iter(NS+'text'):
            if el.text in changes:
                el.text = changes[el.text]
            if el.get('font-family'):
                el.set('font-family', 'Arial, Helvetica, sans-serif')
        for el in panel.findall(NS+'text'):
            size = float(el.get('font-size', '13.4'))
            if size == 17:
                el.set('font-size', '18')
            elif size < 13.4:
                el.set('font-size', '13.4')

    # Mode labels.
    photo_panel = panels['panel-b']
    for x, lines in zip([70.6, 191.2, 311.8, 432.4],
                        [('White light',), ('Ultraviolet',), ('Parallel', 'polarization'), ('Cross', 'polarization')]):
        for i, line in enumerate(lines):
            text(photo_panel, x, 480+i*16, line)

    # ITA labels.
    wheel = panels['panel-c'].find(NS+'svg')
    wheel.set('overflow', 'visible')
    wheel.set('x', '503')
    wheel.set('y', '282')
    wheel.set('width', '235')
    wheel.set('height', str(235*390/415))
    for el in wheel.iter(NS+'text'):
        size = float(el.get('font-size', '13'))
        if size in (16, 15.3):
            el.set('font-size', '20')
        if el.text == '21,415 groups':
            el.set('font-size', '17.5')
            el.set('y', '236')
            el.text = '21,415'
            unit = ET.SubElement(el, NS+'tspan', {'x': el.get('x'), 'dy': '19'})
            unit.text = 'individuals'

    # Region order.
    domains = panels['panel-d']
    original = domains.find(NS+'svg')
    bitmap = deepcopy(original.find(NS+'image'))
    bitmap.set('id', 'signal-domain-master')
    defs = ET.SubElement(domains, NS+'defs')
    defs.append(bitmap)
    domains.remove(original)
    for i, left in enumerate([342, 640, 938, 44]):
        viewport = ET.SubElement(domains, NS+'svg', {
            'x': str(22+112*i), 'y': '550', 'width': '112', 'height': '140',
            'viewBox': f'{left} 0 300 375', 'overflow': 'hidden'})
        ET.SubElement(viewport, NS+'use', {XL+'href': '#signal-domain-master'})
    for el in list(domains):
        if el.tag == NS+'text' and el.get('y') in ['704', '720']:
            domains.remove(el)
    for i, region in enumerate(REGION_ORDER):
        text(domains, 78+112*i, 710, REGION_NAMES[region], fill=REGION_COLORS[region], weight='600')
    eq = next(el for el in domains.findall(NS+'text') if el.text == 'F = P ∪ D ∪ O')
    eq.set('y', '739')
    eq.set('font-size', '15')

    # Anatomy labels.
    anatomy = next(el for el in domains.findall(NS+'svg') if el.get('viewBox') == '0 30 486 278')
    anatomy.set('overflow', 'visible')
    for background in list(anatomy.findall(NS+'rect')):
        if background.get('fill') == 'white':
            anatomy.remove(background)
    for el in anatomy.iter(NS+'text'):
        el.set('font-size', '16.4')

    design = panels['panel-e']
    for el in design.iter(NS+'text'):
        size = float(el.get('font-size', '13.4'))
        if size < 13.4:
            el.set('font-size', '13.4')

    assert all(photo_hashes(panels[key]) == value for key, value in original_photos.items())
    tree.write(OUT/'Figure1.svg', encoding='utf-8')

    caption_en = '''
    Figure 1. Study cohort, facial regions and experimental design.
    (a) Images were acquired in four modalities. The flowchart shows cohort screening, allocation to the training, validation and test sets, and eligibility for analysis stratified by individual typology angle (ITA).
    (b) From left to right, the images show white-light colour (RGB), UV-excited fluorescence (UV), parallel-polarized light (PLH) and cross-polarized light (PLV), all acquired from the same individual in a single session.
    (c) ITA defines six skin-tone strata. The ring shows the number of individuals in each stratum before applying the minimum regional pixel-count requirement. The corresponding ITA thresholds are listed on the right.
    (d) On the left, signals in the measured UV image partition the full valid facial domain (F) into the porphyrin-signal region (P), the UV-spot region (D) and other valid skin (O). The three regions are non-overlapping and together cover F (F = P ∪ D ∪ O). On the right, a separate anatomical partition divides the same domain into the forehead, image-left cheek, image-right cheek, nose and chin for comparison of measured and reconstructed regional intensities.
    (e) Four experiments. (1) Mutual information (MI) was estimated between RGB and each target modality (UV, PLH and PLV) within each signal region and across the full valid face to compare how much information RGB shares with each target. (2) The analysis was repeated within each ITA stratum to test whether the ordering of regional MI varies with skin tone. (3) Five models reconstructed the three target modalities from RGB. Reconstructions were compared with paired measurements across the full face and within anatomical regions. For UV, the counts and locations of porphyrin signals and UV spots were also evaluated. (4) For UV reconstruction, the analysis tested whether predicted uncertainty identifies local reconstruction errors and whether conformally calibrated prediction intervals for porphyrin and UV-spot counts attain the target coverage of 95%.
    '''
    caption_cn = '''
    图 1　研究队列、面部区域划分与实验设计。
    （a）四种成像模式的采集、队列筛选、训练集与验证集及测试集的划分，以及 ITA 分层分析的纳入条件。
    （b）从左至右为同一人在同一次采集中获得的白光彩色图像（RGB）、紫外激发荧光图像（UV）、平行偏振光图像（PLH）和交叉偏振光图像（PLV）。
    （c）根据个体类型角（ITA）将肤色划分为六层。圆环标注区域像素数筛选前的各层人数，右侧列出分层阈值。
    （d）左图按实测 UV 图像中的信号，将全脸有效区域（F）划分为卟啉信号区（P）、UV 斑区（D）和其余有效皮肤区（O）。三区互不重叠，且 F = P ∪ D ∪ O。右图按解剖位置将同一 F 区完整划分为额头、图像左颊、图像右颊、鼻部和下巴，用于比较各部位的实测与重建强度。
    （e）四项实验。① 分别计算 RGB 与 UV、PLH 和 PLV 在各信号区及全脸的互信息，比较 RGB 与三种目标模态共享的信息量。② 在六个 ITA 肤色层内重复上述分析，检验各区域互信息的相对高低是否随肤色改变。③ 用五种模型从 RGB 重建三种目标模态，在全脸和解剖区域与配对实测图像比较；对 UV 进一步评估卟啉信号和 UV 斑的数量及位置能否恢复。④ 仅针对 UV 重建，检验预测不确定性能否提示局部误差，并评估经共形校准的计数预测区间能否达到 95% 的目标覆盖率。
    '''
    (OUT/'Figure1_caption_EN.txt').write_text(dedent(caption_en).lstrip('\n'), encoding='utf8')
    (OUT/'Figure1_caption_CN.txt').write_text(dedent(caption_cn).lstrip('\n'), encoding='utf8')
    export_svg(OUT / "Figure1.svg")
    return OUT / "Figure1.pdf"
