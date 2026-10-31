"""
Steganalysis built into the coder (previously the standalone scripts in
Steganalysis/0_..5_*.py).

Input: a list of *pairs* - one clean stegobject and its carrier::

    {'method': <raw method name>, 'label': <display name>, 'carrier': <carrier name>,
     'carrier_path': <wav>, 'stego_path': <wav>}

The coder writes them to ``<test dir>/steganalysis_pairs.json``; for older
result folders :func:`pairs_from_results_dir` reconstructs the list.

Two independent parts:

1. Statistical tests (script 0): carrier vs stegobject distributions of
   amplitudes, DCT band magnitudes and LSBs - Chi^2, KS, Jensen-Shannon and
   Kullback-Leibler divergences with permutation p-values, Benjamini-Hochberg
   FDR correction stratified by metric. Heatmaps show the FDR-corrected
   q-values (the old script plotted raw p-values under an "FDR" title).

2. Machine-learning detectors (scripts 1-5): RME, DHS, WMC and DMC feature
   sets, SVM (and FastRealBoostBins if the ``frbb`` module is importable).
   Evaluation uses grouped, stratified k-fold cross-validation - all files
   derived from one carrier stay in the same fold, so the detector is never
   tested on a carrier it saw during training. Reported: accuracy, P_E
   (minimum average decision error), AUC, TPR at 5% / 10% FPR; ROC curves
   from out-of-fold scores, accuracy heatmap / point plot, transferability
   matrix (train on method A, test on method B, again grouped by carrier)
   and t-SNE of the feature space. Final models (trained on all pairs) are
   saved for :func:`predict_file`.
"""

import json
import os
import pickle
import random
import re
import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal as sp_signal
from scipy import stats
from scipy.stats import chisquare, ks_2samp

from lib.Carrier import read_audio_as_int16
from lib.Method import ensure_logger

FEATURE_SETS = ('rme', 'dhs', 'wmc', 'dmc')

# Raw method name prefix -> display name (longest prefix first). Used when
# pairs are reconstructed from a results directory without a manifest.
DISPLAY_NAME_PREFIXES = [
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE', 'PC [v3] baseline'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED', 'PC [v3] optimized'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', 'PC [v1]'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', 'PC [v2]'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', 'PC [v3]'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1', 'PC Xue [v1]'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2', 'PC Xue [v2]'),
    ('DIFFERENTIAL_SVD_XUE_MULTI', 'Xue Et. al. [multi]'),
    ('DIFFERENTIAL_SVD_XUE', 'Xue et. al.'),
    ('MDCT_SVD_WEN', 'Wen Et. Al.'),
]

DEFAULT_CONFIG = {
    # --- statistical part ---
    'STATISTICAL': True,
    'NUM_SAMPLES_STATS': None,      # None = all pairs; int = random subset of carriers
    'NUM_SAMPLES_VISUAL': 3,        # carriers with full visualisations
    'SEED': 42,
    'ENABLE': {
        'AMPLITUDE': True, 'DCT_BANDS': True, 'LSB': True,
        'CHISQ': True, 'KS': True, 'PERMUTE_JS': True,
        'KL_PQ': True, 'KL_QP': True, 'PERMUTE_KL': True,
    },
    'BINS_AMPLITUDE': 256,
    'BINS_DCT': 128,
    'EPS_SMOOTH': 1e-12,
    'PERM_B': 200,                  # permutations (for publication: 500-1000)
    'FRAME_LEN_DCT': 1024,
    'DCT_BANDS_HZ': [(0.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0), (3000.0, 4000.0), (4000.0, 8000.0)],
    'ALPHA_FDR': 0.05,
    'FIG_DPI': 200,

    # --- machine-learning part ---
    'ML': True,
    'FEATURE_SETS': list(FEATURE_SETS),
    'CLASSIFIERS': ['SVM', 'FRBB'],  # FRBB is skipped if the frbb module is missing
    'N_SPLITS': 5,
    'MIN_CARRIERS_ML': 4,
    'SVM_C': 10.0,
    'FRBB_T': 16,
    'TRANSFERABILITY': True,
    'TSNE': True,
    'TSNE_MAX_SAMPLES': 200,
    'SAVE_MODELS': True,
}


def merge_config(config=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy (tuples -> lists)
    for key, value in (config or {}).items():
        if key == 'ENABLE' and isinstance(value, dict):
            cfg['ENABLE'].update(value)
        else:
            cfg[key] = value
    return cfg


def display_name_from_raw(raw_name):
    for prefix, label in DISPLAY_NAME_PREFIXES:
        if raw_name.startswith(prefix):
            return label
    return raw_name


# =========================================================================== #
# Pairs
# =========================================================================== #

def save_pairs(pairs, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)
    return path


def load_pairs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pairs_from_results_dir(results_dir, carriers_dir):
    """
    Reconstructs carrier/stegobject pairs from ``<results_dir>/<method>/Stegobjects``.

    Accepted stegobject names: ``<carrier>.wav`` (ONLY_EMBED runs) or
    ``<method>_<secret>_<carrier>.wav`` (normal runs). Attacked copies are
    ignored.
    """
    manifest = os.path.join(results_dir, 'steganalysis_pairs.json')
    if os.path.isfile(manifest):
        return load_pairs(manifest)

    carriers = {os.path.splitext(f)[0]: os.path.join(carriers_dir, f)
                for f in sorted(os.listdir(carriers_dir)) if f.lower().endswith('.wav')}

    pairs = []
    for method in sorted(os.listdir(results_dir)):
        stego_dir = os.path.join(results_dir, method, 'Stegobjects')
        if not os.path.isdir(stego_dir):
            continue
        for file in sorted(os.listdir(stego_dir)):
            stem, ext = os.path.splitext(file)
            if ext.lower() != '.wav':
                continue
            carrier = None
            if stem in carriers:
                carrier = stem
            elif stem.startswith(method + '_'):
                matches = [c for c in carriers if stem.endswith('_' + c)]
                if matches:
                    carrier = max(matches, key=len)
                    middle = stem[len(method) + 1:-(len(carrier) + 1)]
                    if '_' in middle:  # secret names have no '_' - longer middle = attacked copy
                        carrier = None
            if carrier:
                pairs.append({'method': method, 'label': display_name_from_raw(method), 'carrier': carrier,
                              'carrier_path': carriers[carrier], 'stego_path': os.path.join(stego_dir, file)})
    return pairs


# =========================================================================== #
# Statistical steganalysis (script 0)
# =========================================================================== #

def stable_edges(lo, hi, bins):
    if lo == hi:
        lo, hi = float(lo), float(lo + 1.0)
    return np.linspace(lo, hi, bins + 1)


def common_hist(x, y, bins, eps, density=False):
    lo = float(min(x.min(initial=0), y.min(initial=0)))
    hi = float(max(x.max(initial=1), y.max(initial=1)))
    edges = stable_edges(lo, hi, bins)
    h1, _ = np.histogram(x, bins=edges, density=density)
    h2, _ = np.histogram(y, bins=edges, density=density)
    return h1 + eps, h2 + eps, edges


def kl_divergence(p, q):
    """D_KL(P || Q) = sum(P * log(P / Q))."""
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def _normalised_hist(x, edges, eps):
    h, _ = np.histogram(x, bins=edges)
    p = h.astype(np.float64) + eps
    return p / p.sum()


def _divergence(p1, p2, metric):
    if metric == 'JS':
        return js_divergence(p1, p2)
    if metric == 'KL_PQ':
        return kl_divergence(p1, p2)
    if metric == 'KL_QP':
        return kl_divergence(p2, p1)
    raise ValueError(f"Unknown divergence {metric}")


def perm_p_divergence(x, y, edges, B, rng, eps, metric="JS"):
    """
    Divergence statistic (JS / KL) and its permutation p-value
    ``(count(stat_perm >= stat_obs) + 1) / (B + 1)``. ``B <= 0`` -> p is None.
    """
    stat_obs = _divergence(_normalised_hist(x, edges, eps), _normalised_hist(y, edges, eps), metric)
    if B <= 0:
        return stat_obs, None

    xy = np.concatenate([x, y])
    n1 = len(x)
    count = 0
    for _ in range(B):
        perm = rng.permutation(xy)
        stat_perm = _divergence(_normalised_hist(perm[:n1], edges, eps),
                                _normalised_hist(perm[n1:], edges, eps), metric)
        if stat_perm >= stat_obs:
            count += 1
    return stat_obs, (count + 1) / (B + 1)


def lsb_counts(x):
    lsb = np.asarray(x).astype(np.int16) & 1
    return int(np.sum(lsb == 0)), int(np.sum(lsb == 1))


def fdr_bh(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (q-values). None/NaN -> treated as 1."""
    p = np.array([v if (v is not None and np.isfinite(v)) else 1.0 for v in pvals], dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty(n, dtype=float)
    q_adj[order] = np.clip(q_sorted, 0.0, 1.0)
    return q_adj.tolist()


def framewise_dct(x: np.ndarray, frame: int):
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / (np.iinfo(x.dtype).max or 1.0)
    nframes = len(x) // frame
    if nframes == 0:
        return np.empty((0, frame))
    xf = x[:nframes * frame].reshape(nframes, frame)
    k = np.arange(frame).reshape(-1, 1)
    nidx = np.arange(frame).reshape(1, -1)
    basis = np.cos(np.pi * (nidx + 0.5) * k / frame)
    return xf @ basis.T


def dct_band_indices(sr, n, f_lo, f_hi):
    k_lo = max(1, int(np.floor(2 * n * f_lo / sr)))
    k_hi = min(n - 1, int(np.ceil(2 * n * f_hi / sr)))
    if k_lo >= k_hi:
        k_lo = max(1, min(n - 2, k_hi - 1))
    return k_lo, k_hi


def statistical_tests_for_pair(cover, stego, sr, cfg, rng, pair_id):
    """All enabled statistical tests for one carrier/stegobject pair."""
    enable = cfg['ENABLE']
    eps = cfg['EPS_SMOOTH']
    perm_b = cfg['PERM_B']
    rows = []

    def add(domain, metric, value, p):
        rows.append({'id': pair_id, 'domain': domain, 'metric': metric, 'val': float(value),
                     'p': None if p is None else float(p)})

    if enable['AMPLITUDE']:
        h_c, h_s, edges = common_hist(cover, stego, cfg['BINS_AMPLITUDE'], eps)
        if enable['CHISQ']:
            # chisquare needs equal totals - scale the expected histogram
            expected = h_c * (h_s.sum() / h_c.sum())
            stat, p = chisquare(h_s, f_exp=expected)
            add('Amplitude', 'chi2', stat, p)
        if enable['KS']:
            stat, p = ks_2samp(cover, stego)
            add('Amplitude', 'ks', stat, p)
        if enable['PERMUTE_JS']:
            add('Amplitude', 'js_perm', *perm_p_divergence(cover, stego, edges, perm_b, rng, eps, 'JS'))
        n_perm = perm_b if enable['PERMUTE_KL'] else 0
        if enable['KL_PQ']:
            add('Amplitude', 'kl_pq', *perm_p_divergence(cover, stego, edges, n_perm, rng, eps, 'KL_PQ'))
        if enable['KL_QP']:
            add('Amplitude', 'kl_qp', *perm_p_divergence(cover, stego, edges, n_perm, rng, eps, 'KL_QP'))

    if enable['DCT_BANDS']:
        frame = cfg['FRAME_LEN_DCT']
        Dc = framewise_dct(cover, frame)
        Ds = framewise_dct(stego, frame)
        for f_lo, f_hi in cfg['DCT_BANDS_HZ']:
            if f_lo >= sr / 2:
                continue
            k_lo, k_hi = dct_band_indices(sr, frame, f_lo, min(f_hi, sr / 2))
            band_c = np.abs(Dc[:, k_lo:k_hi]).ravel()
            band_s = np.abs(Ds[:, k_lo:k_hi]).ravel()
            if len(band_c) == 0 or len(band_s) == 0:
                continue
            domain = f"DCT {int(f_lo)}-{int(f_hi)}"
            if enable['KS']:
                stat, p = ks_2samp(band_c, band_s)
                add(domain, 'ks', stat, p)
            edges = stable_edges(min(band_c.min(), band_s.min()), max(band_c.max(), band_s.max()), cfg['BINS_DCT'])
            if enable['PERMUTE_JS']:
                add(domain, 'js_perm', *perm_p_divergence(band_c, band_s, edges, perm_b, rng, eps, 'JS'))
            n_perm = perm_b if enable['PERMUTE_KL'] else 0
            if enable['KL_PQ']:
                add(domain, 'kl_pq', *perm_p_divergence(band_c, band_s, edges, n_perm, rng, eps, 'KL_PQ'))
            if enable['KL_QP']:
                add(domain, 'kl_qp', *perm_p_divergence(band_c, band_s, edges, n_perm, rng, eps, 'KL_QP'))

    if enable['LSB']:
        z_c, o_c = lsb_counts(cover)
        z_s, o_s = lsb_counts(stego)
        expected = np.array([z_c + 1, o_c + 1], dtype=float)
        observed = np.array([z_s + 1, o_s + 1], dtype=float)
        stat, p = chisquare(observed, f_exp=expected * observed.sum() / expected.sum())
        add('LSB', 'chi2', stat, p)

    return rows


def plot_advanced_visuals(carrier, stego, sr, label, out_dir, fname_base, dpi=200):
    limit = np.max(np.abs(carrier)) if len(carrier) > 0 else 1
    bins = np.linspace(-limit, limit, 256)
    hc, edges = np.histogram(carrier, bins=bins, density=True)
    hs, _ = np.histogram(stego, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(centers, hc + 1e-9, label="Cover")
    axes[0].plot(centers, hs + 1e-9, label="Stego", ls='--')
    axes[0].set_yscale('log')
    axes[0].set_title(f"Log-Histogram: {label}")
    axes[0].legend()

    carrier_f = np.asarray(carrier, dtype=np.float64)
    stego_f = np.asarray(stego, dtype=np.float64)
    f, t, Sxx = sp_signal.spectrogram(carrier_f, sr)
    axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    axes[1].set_title("Spectrogram (Cover)")

    _, _, Sxx_s = sp_signal.spectrogram(stego_f, sr)
    diff = 10 * np.log10(Sxx_s + 1e-10) - 10 * np.log10(Sxx + 1e-10)
    max_d = np.max(np.abs(diff)) or 1.0
    im = axes[2].pcolormesh(t, f, diff, shading='gouraud', cmap='seismic', vmin=-max_d, vmax=max_d)
    axes[2].set_title("Difference (Stego - Cover)")
    fig.colorbar(im, ax=axes[2], label="dB Delta")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"vis_{fname_base}.png"), dpi=dpi)
    plt.close(fig)


def _domain_sort_key(name):
    if 'Amplitude' in name:
        return -1.0
    if name.startswith('DCT'):
        try:
            return float(name.split()[1].split('-')[0])
        except (IndexError, ValueError):
            return 999.0
    return 1e6


def plot_heatmap(df, metric_key, title, filename, out_dir, value='q', dpi=200):
    """Heatmap of -log10(value) per pair (rows) and domain (columns)."""
    data = df[df['metric'] == metric_key]
    if data.empty:
        return None
    pivot = data.pivot_table(index='id', columns='domain', values=value, aggfunc='first')
    pivot = pivot[sorted(pivot.columns, key=_domain_sort_key)]
    matrix = -np.log10(pivot.fillna(1.0).to_numpy(dtype=float) + 1e-30)

    fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.5 + 2))
    im = ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0, vmax=5)
    fig.colorbar(im, ax=ax, label=f"-log10({'q' if value == 'q' else 'p'}-value)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def run_statistical_steganalysis(pairs, out_dir, config=None, logger=None):
    """
    Runs the statistical tests for every pair. Writes
    ``results_stratified.csv`` (all tests with p and q values),
    ``summary_detection_rate.csv`` (share of pairs with q < ALPHA_FDR per
    method / domain / metric), heatmaps and visualisations.
    Returns the results DataFrame.
    """
    cfg = merge_config(config)
    logger = ensure_logger(logger)
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    rng = np.random.default_rng(cfg['SEED'])
    py_rng = random.Random(cfg['SEED'])

    carriers = sorted({p['carrier'] for p in pairs})
    if cfg['NUM_SAMPLES_STATS'] and len(carriers) > cfg['NUM_SAMPLES_STATS']:
        carriers = sorted(py_rng.sample(carriers, cfg['NUM_SAMPLES_STATS']))
    visual = set(carriers[:cfg['NUM_SAMPLES_VISUAL']])

    rows = []
    cover_cache = {}
    for pair in pairs:
        if pair['carrier'] not in carriers:
            continue
        if pair['carrier'] not in cover_cache:
            cover_cache[pair['carrier']] = read_audio_as_int16(pair['carrier_path'])
        cover, sr = cover_cache[pair['carrier']]
        stego, sr_s = read_audio_as_int16(pair['stego_path'])
        if sr_s != sr:
            logger.warning(f"Skipping {pair['stego_path']}: sampling rate {sr_s} != {sr}.")
            continue
        n = min(len(cover), len(stego))
        cover_t, stego_t = cover[:n], stego[:n]

        pair_id = f"{pair['label']} ({pair['carrier']})"
        logger.info(f"Statistical steganalysis: {pair_id}")
        for row in statistical_tests_for_pair(cover_t, stego_t, sr, cfg, rng, pair_id):
            row.update({'method': pair['method'], 'label': pair['label'], 'carrier': pair['carrier']})
            rows.append(row)

        if pair['carrier'] in visual:
            try:
                base = re.sub(r'[^A-Za-z0-9]+', '_', f"{pair['label']}_{pair['carrier']}").strip('_')
                plot_advanced_visuals(cover_t, stego_t, sr, pair_id, vis_dir, base, dpi=cfg['FIG_DPI'])
            except Exception as e:
                plt.close('all')
                logger.warning(f"Visualisation failed for {pair_id}: {e}")

    df = pd.DataFrame(rows, columns=['id', 'method', 'label', 'carrier', 'domain', 'metric', 'val', 'p'])
    if df.empty:
        logger.warning("Statistical steganalysis: no pairs to analyse.")
        return df

    # FDR correction stratified by metric
    df['q'] = np.nan
    for metric, index in df[df['p'].notna()].groupby('metric').groups.items():
        df.loc[index, 'q'] = fdr_bh(df.loc[index, 'p'].tolist())

    df.to_csv(os.path.join(out_dir, 'results_stratified.csv'), index=False)

    alpha = cfg['ALPHA_FDR']
    detected = df[df['q'].notna()].assign(detected=lambda d: d['q'] < alpha)
    summary = (detected.groupby(['label', 'domain', 'metric'])['detected']
               .agg(['mean', 'sum', 'count']).reset_index()
               .rename(columns={'mean': f'share_q<{alpha}', 'sum': 'detected', 'count': 'pairs'}))
    summary.to_csv(os.path.join(out_dir, 'summary_detection_rate.csv'), index=False)

    titles = {'ks': 'KS test', 'js_perm': 'JS permutation test', 'kl_pq': 'KL Cover||Stego permutation test',
              'kl_qp': 'KL Stego||Cover permutation test', 'chi2': 'Chi-square test'}
    for metric, title in titles.items():
        plot_heatmap(df, metric, f"{title} (q-values, FDR stratified by metric)",
                     f"heatmap_{metric}.png", out_dir, dpi=cfg['FIG_DPI'])

    return df


# =========================================================================== #
# Features (script 1 / utils_features.py)
# =========================================================================== #

def load_float_mono(path):
    """Float32 mono signal in [-1, 1) - same values as librosa.load(sr=None)."""
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    return data.mean(axis=1).astype(np.float32), sr


def extract_rme(y, sr):
    import librosa
    n_fft, hop_length = 1024, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')) ** 2
    n_mels = 20
    rmel_max = 1127 * np.log(1 + (0.5 * sr) / 700.0)
    rmel_pts = np.linspace(0, rmel_max, n_mels + 2)
    freqs = 0.5 * sr - 700 * (np.exp(rmel_pts / 1127.0) - 1)
    freqs = freqs[::-1]
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_l, f_c):
            if f_c != f_l:
                filterbank[m - 1, k] = (k - f_l) / (f_c - f_l)
        for k in range(f_c, f_r):
            if f_r != f_c:
                filterbank[m - 1, k] = (f_r - k) / (f_r - f_c)
    energies = np.log(np.dot(filterbank, S) + 1e-8)
    features = []
    for m in range(n_mels):
        vals = energies[m, :]
        features.extend([np.mean(vals), np.std(vals), stats.skew(vals), stats.kurtosis(vals)])
    return np.nan_to_num(np.array(features, dtype=np.float64))


def extract_dhs(y, sr):
    import librosa
    y_diff2 = np.convolve(y, np.array([1.0, -2.0, 1.0]), mode='same')
    n_fft, hop_length = 1024, 512
    S2 = np.abs(librosa.stft(y_diff2, n_fft=n_fft, hop_length=hop_length, window='hamming')) ** 2
    bin_thresh = int(np.floor((n_fft + 1) * 0.75 * (sr / 2) / sr))
    bin_thresh = min(max(bin_thresh, 0), n_fft // 2)
    high_energy = np.sum(S2[bin_thresh:, :], axis=0)
    return np.nan_to_num(np.array([np.mean(high_energy), np.std(high_energy),
                                   stats.skew(high_energy), stats.kurtosis(high_energy)], dtype=np.float64))


def extract_wmc(y, sr):
    import librosa
    import pywt
    features = []
    n_mfcc = 13
    for sub_signal in pywt.wavedec(y, 'db4', level=2):
        if len(sub_signal) < 2:
            mfcc = np.zeros((n_mfcc, 1))
        else:
            mfcc = librosa.feature.mfcc(y=np.asarray(sub_signal, dtype=np.float32), sr=sr, n_mfcc=n_mfcc,
                                        n_fft=1024, hop_length=512, window='hamming')
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
    return np.nan_to_num(np.array(features, dtype=np.float64))


def extract_dmc(y, sr):
    import librosa
    y_diff2 = np.convolve(y, np.array([1.0, -2.0, 1.0]), mode='same').astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y_diff2, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512, window='hamming')
    return np.nan_to_num(np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(np.float64))


EXTRACTORS = {'rme': extract_rme, 'dhs': extract_dhs, 'wmc': extract_wmc, 'dmc': extract_dmc}


def extract_file_features(path, feature_sets=FEATURE_SETS):
    y, sr = load_float_mono(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {name: EXTRACTORS[name](y, sr) for name in feature_sets}


def extract_features(pairs, feature_sets=FEATURE_SETS, out_dir=None, logger=None):
    """
    Features of every pair. Returns
    ``{method: {feature: {'X', 'y', 'groups', 'filenames', 'label'}}}``
    (y: 0 = cover, 1 = stego; groups = carrier name). Cover features are
    computed once per carrier. If ``out_dir`` is given the data is also saved
    as ``<out_dir>/<method>/features_<feature>.pkl`` (format of script 1).
    """
    logger = ensure_logger(logger)
    cover_cache = {}
    data = {}

    for pair in pairs:
        if pair['carrier'] not in cover_cache:
            cover_cache[pair['carrier']] = extract_file_features(pair['carrier_path'], feature_sets)
        stego_features = extract_file_features(pair['stego_path'], feature_sets)
        logger.info(f"Features: {pair['label']} / {pair['carrier']}")

        method_data = data.setdefault(pair['method'], {})
        for feature in feature_sets:
            store = method_data.setdefault(feature, {'X': [], 'y': [], 'groups': [], 'filenames': [],
                                                     'label': pair['label']})
            for vector, label, kind in ((cover_cache[pair['carrier']][feature], 0, 'cover'),
                                        (stego_features[feature], 1, 'stego')):
                store['X'].append(vector)
                store['y'].append(label)
                store['groups'].append(pair['carrier'])
                store['filenames'].append(f"{pair['carrier']}|{kind}")

    for method, features in data.items():
        for feature, store in features.items():
            store['X'] = np.vstack(store['X'])
            store['y'] = np.asarray(store['y'], dtype=int)
            store['groups'] = np.asarray(store['groups'])
            if out_dir:
                os.makedirs(os.path.join(out_dir, method), exist_ok=True)
                with open(os.path.join(out_dir, method, f'features_{feature}.pkl'), 'wb') as f:
                    pickle.dump({'X': store['X'], 'y': store['y'], 'filenames': store['filenames'],
                                 'groups': store['groups'].tolist(), 'label': store['label']}, f)
    return data


# =========================================================================== #
# Classifiers (scripts 2-5)
# =========================================================================== #

def frbb_available():
    try:
        from frbb import FastRealBoostBins  # noqa: F401
        return True
    except Exception:
        return False


def make_classifier(name, cfg, seed=None):
    if name == 'SVM':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', C=cfg['SVM_C'],
                                                   random_state=seed))
    if name == 'FRBB':
        from frbb import FastRealBoostBins
        return FastRealBoostBins(T=cfg['FRBB_T'])
    raise ValueError(f"Unknown classifier {name}")


def _to01(pred):
    pred = np.asarray(pred).ravel()
    if set(np.unique(pred)) <= {-1, 1}:
        return (pred > 0).astype(int)
    return pred.astype(int)


def classifier_scores(clf, X):
    """Continuous score (higher = more 'stego') for ROC curves."""
    if hasattr(clf, 'decision_function'):
        return np.asarray(clf.decision_function(X), dtype=float).ravel()
    if hasattr(clf, 'predict_proba'):
        return np.asarray(clf.predict_proba(X))[:, 1]
    return _to01(clf.predict(X)).astype(float)


def detection_metrics(y_true, y_pred, y_score):
    from sklearn.metrics import auc, confusion_matrix, roc_curve

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p_fa = fp / (fp + tn) if (fp + tn) else 0.0
    p_md = fn / (fn + tp) if (fn + tp) else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    def tpr_at(target):
        allowed = tpr[fpr <= target + 1e-12]
        return float(allowed.max()) if len(allowed) else 0.0

    return {
        'accuracy': float((tp + tn) / len(y_true)),
        'detection_error': float(0.5 * (p_fa + p_md)),          # at the classifier threshold
        'P_E': float(np.min(0.5 * (fpr + (1 - tpr)))),           # minimum over thresholds
        'AUC': float(roc_auc),
        'TPR_at_FPR_5%': tpr_at(0.05),
        'TPR_at_FPR_10%': tpr_at(0.10),
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }


def grouped_folds(y, groups, n_splits, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def evaluate_detector(X, y, groups, classifier, cfg, seed=None):
    """Grouped CV. Returns metrics computed on the pooled out-of-fold predictions."""
    folds = grouped_folds(y, groups, cfg['N_SPLITS'], seed)
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y), dtype=float)
    fold_acc = []
    fit_time = predict_time = 0.0

    for train_idx, test_idx in folds:
        clf = make_classifier(classifier, cfg, seed)
        t0 = time.perf_counter()
        clf.fit(X[train_idx], y[train_idx])
        fit_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred[test_idx] = _to01(clf.predict(X[test_idx]))
        score[test_idx] = classifier_scores(clf, X[test_idx])
        predict_time += time.perf_counter() - t0
        fold_acc.append(float(np.mean(pred[test_idx] == y[test_idx])))

    metrics = detection_metrics(y, pred, score)
    metrics.update({'folds': len(folds), 'accuracy_fold_std': float(np.std(fold_acc)),
                    'training_time': fit_time, 'prediction_time': predict_time, 'samples': int(len(y))})
    return metrics


def transferability_matrix(features, feature, classifier, cfg, seed=None):
    """
    Accuracy of a detector trained on method A (rows) and tested on method B
    (columns). Folds are made over carriers common to both methods, so a
    detector is never tested on a carrier (cover or stego) used in training.
    """
    methods = sorted(features)
    matrix = np.full((len(methods), len(methods)), np.nan)

    for i, train_method in enumerate(methods):
        train_data = features[train_method][feature]
        for j, test_method in enumerate(methods):
            test_data = features[test_method][feature]
            carriers = np.intersect1d(np.unique(train_data['groups']), np.unique(test_data['groups']))
            if len(carriers) < 2:
                continue
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(carriers)
            k = max(2, min(cfg['N_SPLITS'], len(carriers)))
            accuracies = []
            for fold in np.array_split(shuffled, k):
                train_mask = ~np.isin(train_data['groups'], fold) & np.isin(train_data['groups'], carriers)
                test_mask = np.isin(test_data['groups'], fold)
                if len(np.unique(train_data['y'][train_mask])) < 2 or not test_mask.any():
                    continue
                clf = make_classifier(classifier, cfg, seed)
                clf.fit(train_data['X'][train_mask], train_data['y'][train_mask])
                pred = _to01(clf.predict(test_data['X'][test_mask]))
                accuracies.append(np.mean(pred == test_data['y'][test_mask]))
            if accuracies:
                matrix[i, j] = float(np.mean(accuracies))

    return methods, matrix


def _plot_roc(curves, classifier, feature, out_dir, dpi):
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guess', alpha=0.8)
    for label, metrics in curves:
        ax.plot(metrics['fpr'], metrics['tpr'], lw=2, label=f"{label} (AUC = {metrics['AUC']:.2f})")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.set_title(f'ROC ({classifier}) - {feature.upper()} - out-of-fold', fontsize=16, pad=20)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    path = os.path.join(out_dir, f'roc_{classifier}_{feature}.png')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def _plot_accuracy_heatmap(table, classifier, out_dir, dpi):
    import seaborn as sns
    data = table[table['classifier'] == classifier]
    if data.empty:
        return None
    pivot = data.pivot_table(index='method', columns='feature_set', values='accuracy')
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(pivot) + 2)))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='viridis', linewidths=.5, vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'Accuracy'}, ax=ax)
    ax.set_title(f'Detection accuracy ({classifier}, grouped CV)', fontsize=16, pad=20)
    path = os.path.join(out_dir, f'accuracy_heatmap_{classifier}.png')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def _plot_accuracy_pointplot(table, classifier, out_dir, dpi):
    import seaborn as sns
    data = table[table['classifier'] == classifier]
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * data['method'].nunique() + 3)))
    sns.pointplot(data=data, x='accuracy', y='method', hue='feature_set', linestyle='none', dodge=0.4,
                  palette='deep', ax=ax)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.axvline(0.5, color='red', linestyle=':', label='Random')
    ax.set_title(f'Detection accuracy per method ({classifier}, grouped CV)', fontsize=16)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('')
    ax.set_xlim(0.0, 1.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Feature Set')
    path = os.path.join(out_dir, f'accuracy_pointplot_{classifier}.png')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def _plot_transferability(labels, matrix, classifier, feature, out_dir, dpi):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', xticklabels=labels, yticklabels=labels,
                vmin=0.4, vmax=1.0, cbar_kws={'label': 'Accuracy'}, ax=ax)
    ax.set_title(f'Transferability Matrix - {feature.upper()} ({classifier}, grouped by carrier)', pad=20)
    ax.set_ylabel('Trained on...')
    ax.set_xlabel('Tested on...')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    path = os.path.join(out_dir, f'transferability_{classifier}_{feature}.png')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def _plot_tsne(features, feature, out_dir, cfg, seed, dpi):
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X_all, labels = [], []
    covers_done = set()
    for method in sorted(features):
        store = features[method][feature]
        for cls, name in ((0, 'Cover'), (1, store['label'])):
            idx = np.where(store['y'] == cls)[0]
            if cls == 0:
                # each cover only once
                idx = np.array([i for i in idx if store['groups'][i] not in covers_done], dtype=int)
                covers_done.update(store['groups'][idx])
            if len(idx) > cfg['TSNE_MAX_SAMPLES']:
                idx = rng.choice(idx, cfg['TSNE_MAX_SAMPLES'], replace=False)
            if len(idx):
                X_all.append(store['X'][idx])
                labels.extend([name] * len(idx))

    if not X_all:
        return None
    X = StandardScaler().fit_transform(np.vstack(X_all))
    if len(X) < 5:
        return None
    perplexity = min(30, max(2, (len(X) - 1) // 3))
    embedded = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init='random').fit_transform(X)

    fig, ax = plt.subplots(figsize=(11, 11))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, style=labels, palette='tab10', s=80,
                    alpha=0.8, ax=ax)
    ax.set_title(f't-SNE feature space: {feature.upper()}', fontsize=16)
    ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=3, frameon=False)
    path = os.path.join(out_dir, f'tsne_{feature}.png')
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return path


def run_ml_steganalysis(pairs, out_dir, config=None, logger=None):
    """
    Feature extraction + grouped CV evaluation + plots + final models.
    Returns the summary DataFrame (one row per method / feature set /
    classifier) or an empty DataFrame if there are too few carriers.
    """
    cfg = merge_config(config)
    logger = ensure_logger(logger)
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg['SEED']
    dpi = cfg['FIG_DPI']

    n_carriers = len({p['carrier'] for p in pairs})
    if n_carriers < cfg['MIN_CARRIERS_ML']:
        logger.warning(f"ML steganalysis skipped: {n_carriers} carriers < MIN_CARRIERS_ML={cfg['MIN_CARRIERS_ML']}.")
        return pd.DataFrame()
    if n_carriers < 20:
        logger.warning(f"ML steganalysis on only {n_carriers} carriers - results have a large variance.")

    classifiers = []
    for name in cfg['CLASSIFIERS']:
        if name == 'FRBB' and not frbb_available():
            logger.warning("FRBB skipped: module 'frbb' (FastRealBoostBins) is not importable.")
            continue
        classifiers.append(name)

    feature_sets = [f for f in cfg['FEATURE_SETS'] if f in EXTRACTORS]
    features = extract_features(pairs, feature_sets, out_dir=os.path.join(out_dir, 'features'), logger=logger)

    rows, curves = [], {}
    for method in sorted(features):
        for feature in feature_sets:
            store = features[method][feature]
            for classifier in classifiers:
                try:
                    metrics = evaluate_detector(store['X'], store['y'], store['groups'], classifier, cfg, seed)
                except Exception as e:
                    logger.error(f"Detector {classifier}/{feature} failed for {method}: {e}")
                    continue
                curves.setdefault((classifier, feature), []).append((store['label'], metrics))
                rows.append({'method': store['label'], 'method_raw': method, 'feature_set': feature.upper(),
                             'classifier': classifier,
                             **{k: v for k, v in metrics.items() if k not in ('fpr', 'tpr', 'confusion_matrix')},
                             'confusion_matrix': json.dumps(metrics['confusion_matrix'])})

                if cfg['SAVE_MODELS']:
                    model = make_classifier(classifier, cfg, seed)
                    model.fit(store['X'], store['y'])
                    model_dir = os.path.join(out_dir, 'models', classifier, method)
                    os.makedirs(model_dir, exist_ok=True)
                    with open(os.path.join(model_dir, f'model_{feature}.pkl'), 'wb') as f:
                        pickle.dump(model, f)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.to_csv(os.path.join(out_dir, 'ml_results.csv'), index=False)
    with open(os.path.join(out_dir, 'ml_results.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    for (classifier, feature), method_curves in curves.items():
        _plot_roc(method_curves, classifier, feature, plots_dir, dpi)
    plot_table = table.assign(feature_set=table['feature_set'])
    for classifier in classifiers:
        _plot_accuracy_heatmap(plot_table, classifier, plots_dir, dpi)
        _plot_accuracy_pointplot(plot_table, classifier, plots_dir, dpi)

    for feature in feature_sets:
        if cfg['TRANSFERABILITY'] and len(features) > 1:
            for classifier in classifiers:
                labels_raw, matrix = transferability_matrix(features, feature, classifier, cfg, seed)
                labels = [features[m][feature]['label'] for m in labels_raw]
                pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
                    os.path.join(out_dir, f'transferability_{classifier}_{feature}.csv'))
                _plot_transferability(labels, matrix, classifier, feature, plots_dir, dpi)
        if cfg['TSNE']:
            try:
                _plot_tsne(features, feature, plots_dir, cfg, seed, dpi)
            except Exception as e:
                plt.close('all')
                logger.warning(f"t-SNE failed for {feature}: {e}")

    return table


def run_steganalysis(pairs, out_dir, config=None, logger=None):
    """Runs the enabled parts. Returns ``{'statistical': df, 'ml': df}``."""
    cfg = merge_config(config)
    logger = ensure_logger(logger)
    os.makedirs(out_dir, exist_ok=True)
    save_pairs(pairs, os.path.join(out_dir, 'pairs.json'))

    results = {}
    if cfg['STATISTICAL']:
        results['statistical'] = run_statistical_steganalysis(pairs, os.path.join(out_dir, 'Statistical'), cfg, logger)
    if cfg['ML']:
        results['ml'] = run_ml_steganalysis(pairs, os.path.join(out_dir, 'ML'), cfg, logger)
    return results


# =========================================================================== #
# Prediction for a single file (script 3)
# =========================================================================== #

def predict_file(file_path, models_dir, classifier='SVM', min_votes=2):
    """
    Runs every saved model (``<models_dir>/<classifier>/<method>/model_<feature>.pkl``)
    on one file. Returns ``{method: {'votes': k, 'models': n, 'verdict': ..., 'details': {...}}}``;
    verdict is 'SUSPICIOUS' if at least ``min_votes`` feature sets say stego.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    root = os.path.join(models_dir, classifier)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No models in {root} - run the steganalysis first.")

    features = extract_file_features(file_path, FEATURE_SETS)
    report = {}
    for method in sorted(os.listdir(root)):
        method_dir = os.path.join(root, method)
        if not os.path.isdir(method_dir):
            continue
        details, votes = {}, []
        for feature in FEATURE_SETS:
            model_path = os.path.join(method_dir, f'model_{feature}.pkl')
            if not os.path.isfile(model_path):
                continue
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            prediction = int(_to01(model.predict(features[feature].reshape(1, -1)))[0])
            votes.append(prediction)
            details[feature.upper()] = 'STEGO' if prediction == 1 else 'Cover'
        if votes:
            report[method] = {'label': display_name_from_raw(method), 'votes': int(sum(votes)),
                              'models': len(votes), 'details': details,
                              'verdict': 'SUSPICIOUS' if sum(votes) >= min_votes else 'Clean'}
    return report
