"""
Summary charts "<metric> vs Capacity" (one line per method), in the style of
the thesis figures: seaborn whitegrid, husl palette, one marker per method,
legend titled "Metoda / Method".

Input is the per-stegobject analysis CSV written by ``Analysis`` (one or more
files - e.g. several runs with different capacities can be combined).

    from lib.Plots import load_results, plot_capacity_charts
    df = load_results(['run1/test_analysis_x.csv', 'run2/test_analysis_y.csv'])
    plot_capacity_charts(df, 'Plots/')

Every chart is also exported as a CSV table (mean, std, n per method and
capacity) next to the PNG, so the plotted numbers can be checked or reused.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.Analysis import NUMERIC_COLUMNS

# Canonical legend order of the thesis figures. With the 'husl' palette these
# seven methods get exactly the colours / markers of the original charts.
CANONICAL_METHOD_ORDER = [
    'PC Xue [v1]',
    'PC Xue [v2]',
    'PC [v1]',
    'PC [v2]',
    'PC [v3]',
    'Wen Et. Al.',
    'Xue et. al.',
]

CANONICAL_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']
EXTRA_MARKERS = ['P', 'X', '*', 'h', 'p', 'H', 'd', '8']

# Colour-vision-deficiency-safe alternative (validated categorical palette).
ACCESSIBLE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
EXTRA_COLORS = ['#4d4d4d', '#8c564b', '#17becf', '#bcbd22', '#7f7f7f', '#000000']

# Variants of a method ('<name> (orig.)', '<name> (legacy)') keep the colour
# and marker of <name> and get their own line style with hollow markers.
VARIANT_LINESTYLES = {'orig.': '--', 'legacy': ':'}
_VARIANT_PATTERN = re.compile(r'^(?P<base>.+) \((?P<tag>[^()]+)\)$')

# metric column -> (title, y label)
CHARTS = {
    'BER [%]': ('Bit Error Rate (BER) vs Capacity', 'BER [%]'),
    'EMB TIME [s]': ('Average Embedding Time vs Capacity', 'Embedding Time [s]'),
    'EXT TIME [s]': ('Average Extraction Time vs Capacity', 'Extraction Time [s]'),
    'MSE': ('Mean squared error (MSE) vs Capacity', 'Mean squared error (MSE)'),
    'PESQ': ('PESQ vs Capacity', 'PESQ'),
    'PSNR [dB]': ('Peak signal-to-noise ratio (PSNR) vs Capacity', 'PSNR [dB]'),
    'SNR [dB]': ('Signal-to-noise ratio (SNR) vs Capacity', 'SNR [dB]'),
    'SEGSNR [dB]': ('Segmental SNR vs Capacity', 'SegSNR [dB]'),
    'STOI': ('STOI vs Capacity', 'STOI'),
}

DEFAULT_METRICS = ['BER [%]', 'EMB TIME [s]', 'EXT TIME [s]', 'MSE', 'PESQ']

LEGEND_TITLE = 'Metoda / Method'
X_COLUMN = 'CAP. [bps]'
X_LABEL = 'Capacity [bps]'


def load_results(paths):
    """
    Loads and concatenates analysis CSV files. ``paths`` may contain files,
    directories (searched recursively for test_analysis_*.csv, grouped files
    excluded) and glob patterns.
    """
    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        if os.path.isdir(path):
            found = glob.glob(os.path.join(path, '**', 'test_analysis_*.csv'), recursive=True)
            files.extend(f for f in found if 'test_analysis_group_' not in os.path.basename(f))
        else:
            files.extend(glob.glob(path) or [path])

    frames = []
    for file in sorted(set(files)):
        if not os.path.isfile(file):
            raise FileNotFoundError(file)
        df = pd.read_csv(file)
        df['SOURCE FILE'] = file
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No analysis CSV files found in {paths}.")

    df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if 'TEST' in df.columns:
        df['TEST'] = df['TEST'].fillna('NONE')
    return df


def method_styles(method_names, palette='husl'):
    """
    Colour and marker per method name. Colours follow the method (entity),
    never its position in a filtered subset, so the same method has the same
    colour in every chart.
    """
    names = list(dict.fromkeys(method_names))
    canonical = [n for n in CANONICAL_METHOD_ORDER]

    if palette == 'accessible':
        base_colors = ACCESSIBLE_COLORS
    else:
        base_colors = sns.color_palette(palette, len(CANONICAL_METHOD_ORDER)).as_hex()

    styles = {}
    for i, name in enumerate(canonical):
        styles[name] = {'color': base_colors[i], 'marker': CANONICAL_MARKERS[i], 'linestyle': '-',
                        'markerfacecolor': base_colors[i]}

    # variants of known methods: same colour / marker, own line style, hollow marker
    variants = {}
    for name in names:
        match = _VARIANT_PATTERN.match(name)
        if name not in styles and match and match.group('base') in styles:
            base = styles[match.group('base')]
            styles[name] = dict(base, linestyle=VARIANT_LINESTYLES.get(match.group('tag'), '-.'),
                                markerfacecolor='white')
            variants.setdefault(match.group('base'), []).append(name)

    extras = sorted(n for n in names if n not in styles)
    extra_colors = (ACCESSIBLE_COLORS[len(canonical):] if palette == 'accessible' else []) + EXTRA_COLORS
    for i, name in enumerate(extras):
        color = extra_colors[i % len(extra_colors)]
        styles[name] = {'color': color, 'marker': EXTRA_MARKERS[i % len(EXTRA_MARKERS)], 'linestyle': '-',
                        'markerfacecolor': color}

    order = []
    for name in canonical:
        if name in names:
            order.append(name)
        order.extend(sorted(variants.get(name, [])))
    order += [n for n in sorted(variants) if n not in canonical for n in sorted(variants[n])]
    order += extras
    return styles, list(dict.fromkeys(order))


def aggregate(df, metric, test='NONE', x_column=X_COLUMN, name_column='NAME'):
    """Mean / std / count / 95% CI of ``metric`` per method and capacity."""
    data = df
    if test is not None and 'TEST' in data.columns:
        data = data[data['TEST'] == test]
    data = data.dropna(subset=[metric, x_column])
    if data.empty:
        return pd.DataFrame(columns=[name_column, x_column, 'mean', 'std', 'n', 'ci95'])

    grouped = data.groupby([name_column, x_column])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.rename(columns={'count': 'n'})
    grouped['std'] = grouped['std'].fillna(0.0)
    grouped['ci95'] = 1.96 * grouped['std'] / np.sqrt(grouped['n'].clip(lower=1))
    return grouped.sort_values([name_column, x_column])


def _apply_style():
    sns.set_theme(style='whitegrid', context='notebook', rc={
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.edgecolor': '#cccccc',
    })


def plot_metric_vs_capacity(df, metric, output_dir, test='NONE', palette='husl', ci=False,
                            title=None, ylabel=None, filename=None, dpi=200, logy=False):
    """
    Draws one '<metric> vs Capacity' chart. Returns the PNG path, or None if
    there is nothing to plot.
    """
    if metric not in df.columns:
        return None

    table = aggregate(df, metric, test=test)
    if table.empty:
        return None

    default_title, default_ylabel = CHARTS.get(metric, (f'{metric} vs Capacity', metric))
    title = title or default_title
    if test not in (None, 'NONE'):
        title = f'{title} ({test})'
    ylabel = ylabel or default_ylabel

    styles, order = method_styles(df['NAME'].dropna().unique(), palette)

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in order:
        rows = table[table['NAME'] == name]
        if rows.empty:
            continue
        style = styles[name]
        ax.plot(rows[X_COLUMN], rows['mean'], label=name, color=style['color'], marker=style['marker'],
                linestyle=style.get('linestyle', '-'), markerfacecolor=style.get('markerfacecolor', style['color']),
                linewidth=1.8, markersize=8)
        if ci and (rows['n'] > 1).any():
            ax.fill_between(rows[X_COLUMN], rows['mean'] - rows['ci95'], rows['mean'] + rows['ci95'],
                            color=style['color'], alpha=0.15, linewidth=0)

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(X_LABEL, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if logy:
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title=LEGEND_TITLE, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11,
              title_fontsize=11)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe = filename or _safe_filename(metric)
    if test not in (None, 'NONE'):
        safe = f'{safe}_{_safe_filename(test)}'
    png_path = os.path.join(output_dir, f'{safe}_vs_capacity.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    table.to_csv(os.path.join(output_dir, f'{safe}_vs_capacity.csv'), index=False)
    return png_path


def _safe_filename(text):
    keep = [c if c.isalnum() else '_' for c in str(text)]
    return '_'.join(filter(None, ''.join(keep).split('_'))).lower()


def plot_capacity_charts(df, output_dir, metrics=None, palette='husl', ci=False, tests=None, dpi=200):
    """
    Draws the standard set of charts. ``tests`` - list of TEST labels to plot
    (default: 'NONE' for all metrics, plus a BER chart for every attack found
    in the data). Returns the list of written PNG files.
    """
    metrics = metrics or DEFAULT_METRICS
    written = []

    for metric in metrics:
        path = plot_metric_vs_capacity(df, metric, output_dir, test='NONE', palette=palette, ci=ci, dpi=dpi)
        if path:
            written.append(path)

    if 'TEST' in df.columns:
        attack_tests = tests if tests is not None else sorted(t for t in df['TEST'].dropna().unique() if t != 'NONE')
        for test in attack_tests:
            path = plot_metric_vs_capacity(df, 'BER [%]', os.path.join(output_dir, 'Attacks'), test=test,
                                           palette=palette, ci=ci, dpi=dpi)
            if path:
                written.append(path)

    return written
