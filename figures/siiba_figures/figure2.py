"""Reproduce Figure 2 from frozen, verified inputs."""
from pathlib import Path
from .input_paths import INPUT_ROOT
import argparse
import csv
import hashlib
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from .style import (FONT, REGION_ORDER, REGION_COLORS, REGION_NAMES,
                    TARGET_ORDER, region_line_style, add_region_legend)

ROOT = Path(__file__).resolve().parents[1]
INPUT = INPUT_ROOT / 'figure2'
TARGETS = [target.lower() for target in TARGET_ORDER]
DOMAINS = list(REGION_ORDER)
NAMES = [f'RGB-{target}' for target in TARGET_ORDER]
COLORS = REGION_COLORS
FACTORS = [1, 2, 4, 8]
FACTOR_OFFSETS = [.27, .09, -.09, -.27]
FACTOR_COLORS = ['#D8E0E8', '#A2B5C8', '#6483A1', '#233F5C']
STRATA = ['very_light', 'light', 'intermediate', 'tan', 'brown', 'dark']
MANIFEST_SHA256 = '824b7bd245eea858aff0f9afc9fc75234f459871169d48660c9d386f32a29965'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs():
    """Check source hashes and values."""
    assert sha(INPUT / 'manifest.json') == MANIFEST_SHA256, 'Figure 2 manifest changed'
    manifest = json.loads((INPUT / 'manifest.json').read_text(encoding='utf8'))
    for relative, expected in manifest['files'].items():
        assert sha(INPUT / relative) == expected, relative
    integers = {'percent', 'points', 'k', 'N_groups', 'factor', 'budget', 'N'}
    floats = {'mean', 'sample_SD', 'mean_bits', 'sampling_sd_bits_ddof1', 'sample_sd_bits'}
    tables = {}
    for name, expected_count in manifest['row_counts'].items():
        with (INPUT / (name + '.csv')).open(encoding='utf8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for field in integers & row.keys():
                row[field] = int(row[field])
            for field in floats & row.keys():
                row[field] = float(row[field])
        assert len(rows) == expected_count, name
        tables[name] = rows

    evidence = INPUT / 'evidence'
    audit = json.loads((evidence / 'main_audit.json').read_text(encoding='utf8'))
    assert audit['status'] == 'passed'
    assert sha(evidence / 'main_summary.json') == audit['summary_sha256']
    summary = json.loads((evidence / 'main_summary.json').read_text(encoding='utf8'))
    fields = ['target', 'domain', 'percent', 'points', 'k']
    expected = {}
    for collection in summary.values():
        if isinstance(collection, list):
            for row in collection:
                if isinstance(row, dict) and all(field in row for field in fields + ['mean', 'sample_SD']):
                    expected[tuple(row[field] for field in fields)] = row
    for name in ['main', 'fraction', 'k']:
        for row in tables[name]:
            original = expected[tuple(row[field] for field in fields)]
            assert row['mean'] == original['mean'] and row['sample_SD'] == original['sample_SD']

    audit = json.loads((evidence / 'resolution_audit.json').read_text(encoding='utf8'))
    assert audit['passed']
    assert sha(evidence / 'resolution_summary.json') == audit['summary']['sha256']
    resolution = json.loads((evidence / 'resolution_summary.json').read_text(encoding='utf8'))
    expected = {(row['x'], row['y'], row['domain'], row['factor']): row
                for row in resolution['cells'] if row['budget'] == 1000}
    for row in tables['resolution']:
        original = expected[(row['x'], row['y'], row['domain'], row['factor'])]
        assert row['mean_bits'] == original['mean_bits']
        assert row['sampling_sd_bits_ddof1'] == original['sampling_sd_bits_ddof1']
        assert np.isclose(np.mean(original['draws']), row['mean_bits'], rtol=0, atol=1e-12)
        assert np.isclose(np.std(original['draws'], ddof=1), row['sampling_sd_bits_ddof1'], rtol=0, atol=1e-12)

    audit = json.loads((evidence / 'ita_audit.json').read_text(encoding='utf8'))
    assert audit['status'] == 'passed'
    payload = json.loads((evidence / 'ita_payload.json').read_text(encoding='utf8'))
    expected = {(row['cohort'], row['target'], row['domain']): row
                for row in payload['cell_bindings'] if row['table'] == 'ITA'}
    for row in tables['ita']:
        original = expected[(row['cohort'], row['target'], row['domain'])]
        assert original['status'] == 'accepted'
        for field in ['mean_bits', 'sample_sd_bits', 'points', 'N']:
            assert row[field] == original[field], field
    for modality in ['rgb', 'uv']:
        for factor in FACTORS:
            relative = f'assets/{modality}_factor{factor}.png'
            with Image.open(INPUT / relative) as image:
                assert hashlib.sha256(image.tobytes()).hexdigest() == manifest['provenance'][relative]['pixel_sha256']
    return tables


def build(output_dir):
    """Write the figure and its source data."""
    OUT = Path(output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'Figure2_source_data').mkdir(exist_ok=True)
    data = read_inputs()
    res = data['resolution']
    ita = data['ita']
    source_hashes = json.loads((INPUT / 'evidence/revision_audit.json').read_text(encoding='utf8'))['source_hashes']
    paired_path = INPUT / 'full_face_paired_mi.json'
    paired = json.loads(paired_path.read_text(encoding='utf8'))
    source_hashes[paired_path.name] = sha(paired_path)
    assert paired['domain'] == 'F' and paired['contrast'] == 'MI(RGB,PLH) minus MI(RGB,UV)'
    assert {row['N_groups'] for row in data['main']} == {paired['n_acquisitions']}
    assert sum(paired[key] for key in ('n_positive', 'n_negative', 'n_ties')) == paired['n_acquisitions']
    log_p = paired['log10_p']
    assert np.isfinite(log_p) and log_p <= 0
    coefficient, exponent = paired['p_scientific'].split('e')
    assert np.isclose(np.log10(float(coefficient)) + int(exponent), log_p)
    assert log_p < -3, 'The paired test must support the displayed threshold'
    plt.rcParams.update({
        'font.family': FONT, 'font.size': 8, 'text.color': '#171717',
        'axes.labelcolor': '#171717', 'axes.edgecolor': '#333333',
        'axes.linewidth': .6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    W, H = 8.5, 9.0
    fig = plt.figure(figsize=(W, H))


    def text(x, y, value, **kwargs):
        return fig.text(x / W, y / H, value, **kwargs)


    def axes(x, y, width, height):
        ax = fig.add_axes([x / W, y / H, width / W, height / H])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(pad=2)
        ax.set_axisbelow(True)
        return ax


    def heading(letter, label, y):
        text(.10, y, letter, fontsize=14, weight='bold', va='top')
        text(.36, y - .017, label, fontsize=11, weight='bold', va='top')


    def region_legend(y):
        add_region_legend(fig, DOMAINS, (8.37 / W, y / H), fontsize=8, loc='center right')


    heading('a', 'Analysis domains and shared information', 8.91)
    asset = INPUT / 'assets/domains.png'
    im = Image.open(asset)
    assert im.size == (1280, 473)
    source_hashes['user_supplied_cartoon.png'] = sha(asset)
    for display_centre, source_centre in zip([195, 491, 790, 1091], [491, 790, 1091, 195]):
        panel_left = .10 + 4.62 * (display_centre - 150) / 1280
        panel_width = 4.62 * 300 / 1280
        pic = fig.add_axes([panel_left / W, 7.23 / H, panel_width / W, (4.62 * 375 / 1280) / H])
        pic.imshow(im)
        pic.set_xlim(source_centre - 150, source_centre + 150)
        pic.set_ylim(374.5, -.5)
        pic.set_axis_off()
    for cx, domain in zip([195, 491, 790, 1091], DOMAINS):
        text(.10 + 4.62 * cx / 1280, 7.16, REGION_NAMES[domain],
             ha='center', va='top', fontsize=8.5, color=COLORS[domain], weight='semibold')
    text(2.43, 6.85, 'F = P ∪ D ∪ O', ha='center', va='center', fontsize=10,
         fontfamily='DejaVu Sans')
    ax = axes(5.32, 7.10, 2.98, 1.49)
    ax.grid(axis='y', color='#E8E9EB', lw=.45)
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(0, 4.68)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel('MI (bits)', labelpad=3)
    centres = np.arange(3)[:, None] + (np.arange(4) - 1.5) * .185
    ax.set_xticks(centres.ravel(), DOMAINS * 3)
    ax.tick_params(axis='x', length=0, pad=2)
    for tick in ax.get_xticklabels():
        tick.set_weight('bold')
    for i, target in enumerate(NAMES):
        ax.annotate(target, (i, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -13), textcoords='offset points',
                    ha='center', va='top', fontsize=8.5)
    ax.set_title('Mutual information with RGB', fontsize=11, weight='bold', pad=6)
    for di, domain in enumerate(DOMAINS):
        rows = [next(r for r in data['main'] if r['target'] == target and r['domain'] == domain)
                for target in TARGETS]
        x = np.arange(3) + (di - 1.5) * .185
        ax.bar(x, [r['mean'] for r in rows], width=.165, color=COLORS[domain],
               edgecolor='#333333', lw=.3, zorder=3)
        errors = ax.errorbar(x, [r['mean'] for r in rows],
                            yerr=[r['sample_SD'] for r in rows], fmt='none',
                            ecolor='#191919', elinewidth=.7, capsize=2.5,
                            capthick=.7, zorder=4)
        for row, segment in zip(rows, errors.lines[2][0].get_segments(), strict=True):
            assert np.allclose(segment[:, 1],
                               [row['mean'] - row['sample_SD'], row['mean'] + row['sample_SD']],
                               rtol=0, atol=1e-14)

    # Anchor the comparison to the full-face bar for each tested modality.
    left = centres[TARGETS.index('uv'), DOMAINS.index('F')]
    right = centres[TARGETS.index('plh'), DOMAINS.index('F')]
    ax.text((left + right) / 2, 4.08, r'$p < 0.001$', ha='center', va='bottom', fontsize=11)

    heading('b', 'Mutual-information stability', 6.43)
    region_legend(6.34)
    lefts = [.55, 3.33, 6.11]
    width = 2.19
    for x, name in zip(lefts, NAMES):
        text(x + width / 2, 6.07, name, ha='center', va='center', fontsize=11, weight='bold')
    for row_index, kind in enumerate(['fraction', 'k']):
        bottom = 5.18 if kind == 'fraction' else 4.10
        for column, target in enumerate(TARGETS):
            ax = axes(lefts[column], bottom, width, .69)
            ticks = [10, 30, 50, 70, 100] if kind == 'fraction' else [1, 3, 5, 10, 20]
            if kind == 'k':
                ax.set_xscale('log')
                pad = np.log(20) * .045
                ax.set_xlim(np.exp(-pad), 20 * np.exp(pad))
            else:
                ax.set_xlim(6, 104)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis='x', color='#ECEDEF', lw=.4)
            if target == 'uv':
                ax.set_ylim(.35, 1.15)
                ax.set_yticks([.4, .6, .8, 1.0])
            else:
                ax.set_ylim(2.8, 3.9)
                ax.set_yticks([2.8, 3.2, 3.6])
            if column == 0:
                ax.set_ylabel('MI (bits)', labelpad=3)
            field = 'percent' if kind == 'fraction' else 'k'
            for domain in DOMAINS:
                rows = sorted([r for r in data[kind] if r['target'] == target and r['domain'] == domain],
                              key=lambda r: r[field])
                x = [r[field] for r in rows]
                y = [r['mean'] for r in rows]
                style = region_line_style(domain)
                ax.errorbar(x, y, yerr=[r['sample_SD'] for r in rows], fmt='none',
                            ecolor=style['color'], capsize=1.2, capthick=.5,
                            elinewidth=.5, zorder=2)
                ax.plot(x, y, color=style['color'], ls=style['ls'], lw=style['lw'], zorder=3)
                ax.plot(x, y, **{**style, 'ls': 'none'}, zorder=6 if domain == 'F' else 5)
        xlabel = ('Percentage of eligible test records (%)' if kind == 'fraction'
                  else 'Nearest neighbours in the MI estimator (k)')
        text(4.43, bottom - .27, xlabel, ha='center', va='center', fontsize=8.5)

    heading('c', 'Resolution sensitivity', 3.55)
    tile_side, tile_gap, tile_left = .32, .012, .42
    for row_index, modality in enumerate(['RGB', 'UV']):
        for factor_index, factor in enumerate(FACTORS):
            with Image.open(INPUT / 'assets' / f'{modality.lower()}_factor{factor}.png') as image:
                tile = image.copy()
            assert tile.size == (96 // factor, 96 // factor)
            x = tile_left + factor_index * (tile_side + tile_gap)
            y = 2.73 - row_index * (tile_side + tile_gap)
            patch = fig.add_axes([x / W, y / H, tile_side / W, tile_side / H])
            patch.imshow(tile, interpolation='nearest')
            patch.set_axis_off()
            if row_index == 0:
                text(x + tile_side / 2, 3.12, ['Native', '1/2', '1/4', '1/8'][factor_index],
                     ha='center', va='center', fontsize=8)
                fig.add_artist(Line2D([(x + tile_side / 2) / W], [2.32 / H],
                    transform=fig.transFigure, marker='o', ms=4,
                    mfc=FACTOR_COLORS[factor_index], mec='#344A60', mew=.5, ls=''))
        text(tile_left - .04, 2.73 + tile_side / 2 - row_index * (tile_side + tile_gap),
             modality, ha='right', va='center', fontsize=8)
    text(1.08, 3.28, 'Spatial downsampling', ha='center', va='center', fontsize=8)
    text(1.08, 2.12, 'Image resolution decreases\nfrom left to right.',
         ha='center', va='center', fontsize=8, linespacing=1.05)
    for column, target in enumerate(TARGETS):
        ax = axes([2.15, 4.25, 6.35][column], 2.35, 1.95, .79)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color='#ECEDEF', lw=.4)
        ax.set_ylim(-.53, 3.53)
        ax.set_yticks([3, 2, 1, 0], DOMAINS)
        ax.tick_params(axis='y', length=0, pad=5)
        for label in ax.get_yticklabels():
            label.set_color(COLORS[label.get_text()])
            label.set_weight('bold')
        if target == 'uv':
            ax.set_xlim(.48, 1.45)
            ax.set_xticks([.6, .8, 1, 1.2, 1.4])
        elif target == 'plh':
            ax.set_xlim(2.8, 5.3)
            ax.set_xticks([3, 3.5, 4, 4.5, 5])
        else:
            ax.set_xlim(2.8, 4.42)
            ax.set_xticks([3, 3.4, 3.8, 4.2])
        ax.set_title(NAMES[column], fontsize=11, weight='bold', pad=5)
        ax.set_xlabel('MI (bits)', labelpad=3)
        for domain_index, domain in enumerate(DOMAINS):
            rows = sorted([r for r in res if r['y'] == target and r['domain'] == domain], key=lambda r: r['factor'])
            assert [r['factor'] for r in rows] == FACTORS
            base = 3 - domain_index
            ax.plot([rows[0]['mean_bits'], rows[-1]['mean_bits']], [base, base],
                    color='#C3CCD5', lw=1.1, zorder=2, solid_capstyle='round')
            for offset, factor_color, record in zip(FACTOR_OFFSETS, FACTOR_COLORS, rows):
                ax.errorbar(record['mean_bits'], base + offset,
                            xerr=record['sampling_sd_bits_ddof1'], fmt='o', ms=3.6,
                            color='#344A60', mfc=factor_color, mec='#344A60', mew=.4,
                            elinewidth=.5, capsize=1.0, capthick=.5, zorder=4)

    heading('d', 'MI across skin-tone strata', 1.91)
    region_legend(1.82)
    tone_labels = ['Very light', 'Light', 'Intermediate', 'Tan', 'Brown', 'Dark']
    for column, target in enumerate(TARGETS):
        ax = axes(lefts[column], .56, width, .82)
        ax.set_xlim(-.18, 5.18)
        ax.set_xticks(np.arange(6), tone_labels, rotation=32, ha='right',
                      rotation_mode='anchor')
        ax.tick_params(axis='x', labelsize=8, pad=3)
        if target == 'uv':
            ax.set_ylim(.25, 1.31)
            ax.set_yticks([.3, .6, .9, 1.2])
        else:
            ax.set_ylim(1.98, 3.85)
            ax.set_yticks([2, 2.5, 3, 3.5])
        ax.set_title(NAMES[column], fontsize=11, weight='bold', pad=5)
        if column == 0:
            ax.set_ylabel('MI (bits)', labelpad=3)
        for domain in DOMAINS:
            rows = [next(r for r in ita if r['target'] == target and r['domain'] == domain and r['cohort'] == stratum)
                    for stratum in STRATA]
            style = region_line_style(domain)
            ax.errorbar(np.arange(6), [r['mean_bits'] for r in rows],
                        yerr=[r['sample_sd_bits'] for r in rows], fmt='none',
                        ecolor=COLORS[domain], elinewidth=.5, capsize=1.2, capthick=.5, zorder=2)
            ax.plot(np.arange(6), [r['mean_bits'] for r in rows],
                    color=COLORS[domain], ls=style['ls'], lw=style['lw'], zorder=3)
            ax.plot(np.arange(6), [r['mean_bits'] for r in rows],
                    **{**style, 'ls': 'none'}, zorder=6 if domain == 'F' else 5)

    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'Figure2.{extension}', dpi=400)
    plt.close(fig)
    with Image.open(OUT / 'Figure2.png') as preview:
        preview.thumbnail((1700, 1800), Image.Resampling.LANCZOS)
        preview.save(OUT / 'Figure2_preview.png')
    for kind in ['main', 'fraction', 'k']:
        rows = data[kind]
        with (OUT / 'Figure2_source_data' / f'{kind}_plotted.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, rows, fields in [
            ('resolution', res, ['x', 'y', 'domain', 'factor', 'N_groups', 'budget', 'mean_bits', 'sampling_sd_bits_ddof1']),
            ('ita', ita, ['cohort', 'target', 'domain', 'N', 'points', 'mean_bits', 'sample_sd_bits'])]:
        with (OUT / 'Figure2_source_data' / f'{name}_plotted.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in rows)
    previous = INPUT
    for name in ['main', 'fraction', 'k', 'resolution', 'ita']:
        filename = name + '_plotted.csv'
        assert (OUT / 'Figure2_source_data' / filename).read_bytes() == (previous / (name + '.csv')).read_bytes(), filename
    (OUT / 'Figure2_source_data' / paired_path.name).write_bytes(paired_path.read_bytes())
    caption_en = (
        'Figure 2. Regional mutual information between RGB and UV, PLH and PLV. '
        '(a) The illustration defines the full valid facial domain (F), porphyrin-signal region (P), '
        'UV-spot region (D) and other valid skin (O). Bars show mutual information (MI) estimated from pixels pooled across acquisitions. '
        f'The p value compares RGB–UV and RGB–PLH MI across the full face using a two-sided exact paired sign test. Each of the {paired["n_acquisitions"]:,} acquisitions '
        'contributed one RGB–UV and one RGB–PLH MI estimate, each averaged over three draws of 1,000 matched full-face pixels. '
        '(b) Stability of MI estimates as the percentage of eligible test records and the number of nearest neighbours '
        'in the MI estimator vary. (c) Paired RGB and UV crops show the same skin patch at native resolution and '
        'after downsampling to 1/2, 1/4 and 1/8 of the original width and height. Dot shades denote these resolutions. '
        '(d) Regional MI across six ITA skin-tone strata. Values are means of three pixel-sampling draws; '
        'error bars show one sample standard deviation across these draws.'
    )
    caption_cn = (
        '图2 RGB与UV、PLH和PLV的区域互信息。'
        '（a）示意全脸有效区域（F）、卟啉区（P）、UV斑区（D）及其余有效皮肤（O）。柱高为合并各次采集的像素后估计的互信息。'
        f'p 值来自比较RGB–UV与RGB–PLH全脸互信息的双侧精确配对符号检验。在{paired["n_acquisitions"]:,}次采集中，每次分别计算RGB–UV和RGB–PLH的互信息，'
        '各取三轮抽样的均值，每轮使用全脸同一位置的1,000个配对像素。'
        '（b）改变符合条件的测试记录抽取比例及互信息估计中的近邻数，检验估计的稳定性。'
        '（c）配对RGB与UV局部图展示同一皮肤区域在原始分辨率以及宽、高分别下采样至原始的1/2、1/4和1/8后的细节。圆点深浅表示对应的分辨率。'
        '（d）六个ITA肤色层级内的区域互信息。数值为三次像素抽样的均值，误差线表示这三次抽样的一个样本标准差。'
    )
    (OUT / 'Figure2_caption_EN.txt').write_text(caption_en + '\n', encoding='utf8')
    (OUT / 'Figure2_caption_CN.txt').write_text(caption_cn + '\n', encoding='utf8')
    (OUT / 'Figure2_source_data' / 'revision_audit.json').write_text(json.dumps({
        'source_hashes': source_hashes,
        'plotted_data_identical_to_previous': True,
        'paired_full_face_test': paired,
        'relative_resolution_definition': 'Image width and height divided by 1, 2, 4 or 8 before crop extraction.',
        'region_order': DOMAINS, 'region_colors': COLORS,
    }, indent=2), encoding='utf8')
    return OUT / 'Figure2.pdf'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    options = parser.parse_args()
    print(build(options.output))
