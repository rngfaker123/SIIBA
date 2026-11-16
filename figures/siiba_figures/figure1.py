"""Build Figure 1 from the approved editable SVG template."""
from copy import deepcopy
import base64
import json
from pathlib import Path
from .input_paths import INPUT_ROOT
from textwrap import dedent
from xml.etree import ElementTree as ET
import hashlib

from .style import REGION_COLORS, REGION_ORDER
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
        '115,382 groups': '115,382 records',
        'Excluded: 5,893 groups': 'Excluded: 5,893 records',
        '21,415 groups': '21,415 records',
        'Test groups by skin tone': 'Skin-tone strata (ITA)',
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
        if el.text == '21,415 records':
            el.set('font-size', '17.5')
            el.set('y', '238')

    # Region order.
    domains = panels['panel-d']
    original = domains.find(NS+'svg')
    bitmap = deepcopy(original.find(NS+'image'))
    bitmap.set('id', 'signal-domain-master')
    defs = ET.SubElement(domains, NS+'defs')
    defs.append(bitmap)
    domains.remove(original)
    cols = [REGION_COLORS[key] for key in REGION_ORDER]
    for i, left in enumerate([342, 640, 938, 44]):
        viewport = ET.SubElement(domains, NS+'svg', {
            'x': str(22+112*i), 'y': '550', 'width': '112', 'height': '140',
            'viewBox': f'{left} 0 300 375', 'overflow': 'hidden'})
        ET.SubElement(viewport, NS+'use', {XL+'href': '#signal-domain-master'})
    for el in list(domains):
        if el.tag == NS+'text' and el.get('y') in ['704', '720']:
            domains.remove(el)
    for i, (name, color) in enumerate(zip(['Porphyrin', 'UV spots', 'Other skin', 'Full face'], cols)):
        text(domains, 78+112*i, 710, name, fill=color, weight='600')
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
    Figure 1. Paired skin imaging, analysis domains and experimental design.
    (a) Four-mode facial imaging and cohort selection. Of 115,382 paired capture records, 109,489 records (437,956 images) remained after technical and light-leakage exclusions. The retained records were split into training (76,642), validation (10,948) and held-out test (21,899) sets. Among test records, 21,415 received an ITA classification and 19,645 met the pixel-support requirements for ITA-stratified MI. Each record comprises four images from one acquisition. The acquisition device is illustrated schematically.
    (b) Paired white-light RGB, ultraviolet (UV), parallel-polarized (PLH) and cross-polarized (PLV) images from one validation record. Eye regions are concealed in all photographic examples.
    (c) The six skin-tone strata defined by individual typology angle (ITA), with test-record counts and the corresponding thresholds. Equal ring segments denote categories. The portrait skin colours are illustrative.
    (d) Two complementary regional definitions. Measured UV defines porphyrin-signal regions (P), UV spots (D) and other valid skin (O), which partition the full valid facial domain (F). The same coordinates are used for all imaging modalities. Anatomical evaluation subdivides F into five mutually exclusive regions that together cover F: forehead, image-left cheek, image-right cheek, nose and chin. Each expanded anatomical region contains its original sampling region. The diagrams illustrate these definitions.
    (e) Four linked experiments. Regional MI between RGB and each target modality quantifies shared information and is evaluated across sampling settings and image resolutions. Repeating MI within the six ITA strata assesses skin-tone dependence. Five model families reconstruct UV, PLH and PLV from RGB; comparisons with paired measurements assess image fidelity and regional intensity agreement, with UV-specific analyses of object counts, area and spatial matching. UV uncertainty evaluation examines the relationship between predicted uncertainty and error, uncertainty-based selection, and the coverage and width of calibrated intervals. The uncertainty ranking and count interval are schematic.
    '''
    caption_cn = '''
    图1 配对皮肤成像、分析区域与实验设计。
    （a）四种成像模式的采集与队列筛选。从115,382条配对采集记录中排除技术问题和漏光记录后，保留109,489条记录，共437,956幅图像。训练集、验证集和测试集分别包含76,642、10,948和21,899条记录。测试集中21,415条记录可划分ITA肤色层级，19,645条进一步满足ITA分层互信息分析的区域像素要求。每条记录包含同次采集的四幅图像，设备为采集示意。
    （b）同一条验证记录的白光RGB、紫外UV、平行偏振PLH及交叉偏振PLV图像。所有真实人脸示例均遮挡眼部。
    （c）按个体类型角ITA划分的六个肤色层级，以及各层测试记录数和划分阈值。圆环等长分段表示类别，小人的肤色仅作示意。
    （d）两套互补的区域划分。实测UV定义卟啉信号区P、UV斑区D和其余有效皮肤O，三者构成全有效面部F；各模态使用相同位置。解剖分区将F划为互不重叠且合起来覆盖F的额头、图像左颊、图像右颊、鼻部和下巴，每个扩展区域均包含原来的取样区域。示意图展示两套定义。
    （e）四项相互关联的实验。区域互信息量化RGB与各目标模态共享的信息，并检验其对抽样设置和分辨率的敏感性。在六个ITA层内重复分析以考察肤色影响。五类模型由RGB重建UV、PLH和PLV，与实测图比较图像保真度和区域平均强度一致性，UV另评价对象数量、面积和空间匹配。不确定性实验考察预测不确定性与误差的关系、按不确定性筛选样本，以及校准区间的覆盖率和宽度。排序和计数区间均为示意。
    '''
    (OUT/'Figure1_caption_EN.txt').write_text(dedent(caption_en).lstrip('\n'), encoding='utf8')
    (OUT/'Figure1_caption_CN.txt').write_text(dedent(caption_cn).lstrip('\n'), encoding='utf8')
    export_svg(OUT / "Figure1.svg")
    return OUT / "Figure1.pdf"
