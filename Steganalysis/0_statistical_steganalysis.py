#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Steganalysis Suite v3.0:
- Analiza Time/Frequency (DCT) / LSB
- Testy: KS, Chi2, JS Permutation, KL Divergence (P||Q, Q||P)
- Korekcja FDR: STRATIFIED (liczona osobno dla każdej metryki, aby uniknąć biasu)
- Automatyczne Heatmapy i Raport CSV
"""

import os
import csv
import pathlib
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.stats import chisquare, ks_2samp
from scipy import signal
from typing import List

# ============================
# ========= CONFIG ===========
# ============================

CONFIG = {
    # --- Ścieżki ---
    "CARRIER_DIR": "Carriers",
    "STEGO_ROOT_DIR": "Stegobjects",
    "OUT_DIR": "stego_stratified_results",

    # --- Wybór próbek ---
    "NUM_SAMPLES_STATS":  5,  # None = przeanalizuj wszystkie pliki w folderze
    "NUM_SAMPLES_VISUAL": 3,    # Dla ilu plików wygenerować pełne wykresy (spektrogramy)
    "SEED": 42,

    # --- Mapowanie nazw (Dostosuj do swoich folderów/plików) ---
    "NAME_MAPPING": {
        "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V1",
        "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V2",
        "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_L_8_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V3",
        "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC Xue V1",
        "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC Xue V2",
        "DIFFERENTIAL_SVD_XUE_L_0_A_0.7_B_0.7_VOICED_rVAD_FREQ_L_1500_FREQ_H_2500_FL_20": "Xue et al",
        "MDCT_SVD_WEN_L_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_40": "Wen et al"
    },

    # --- Włączanie modułów ---
    "ENABLE": {
        "AMPLITUDE": True,      # Analiza w dziedzinie czasu
        "DCT_BANDS": True,      # Analiza w dziedzinie częstotliwości
        "LSB": True,            # Analiza bitów najmniej znaczących
        
        # Testy Statystyczne
        "CHISQ": True,
        "KS": True,
        
        # Dywergencje i permutacje
        "JS": True,             # Jensen-Shannon
        "PERMUTE_JS": True,     # Test permutacyjny JS (p-value)
        
        "KL_PQ": True,          # Kullback-Leibler (Cover || Stego)
        "KL_QP": True,          # Kullback-Leibler (Stego || Cover)
        "PERMUTE_KL": True,     # Test permutacyjny KL (wymaga czasu, ale daje p-value)
    },

    # --- Parametry matematyczne ---
    "BINS_AMPLITUDE": 256,
    "BINS_DCT": 128,
    "EPS_SMOOTH": 1e-12,
    "PERM_B": 200,             # Liczba permutacji (dla publikacji: 500-1000)
    
    # --- Konfiguracja pasm DCT ---
    "FRAME_LEN_DCT": 1024,
    "DCT_BANDS_HZ": [
        (0.0, 1000.0), 
        (1000.0, 2000.0), 
        (2000.0, 3000.0), 
        (3000.0, 4000.0),
        (4000.0, 8000.0)
    ],

    # --- Wygląd ---
    "FIG_DPI": 200,
}

# ============================
# ====== MATH & UTILS ========
# ============================

mpl.rcParams.update({
    "figure.dpi": CONFIG["FIG_DPI"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def load_wav_mono(path: str):
    try:
        data, sr = sf.read(path, always_2d=False)
        if data is None:
            return None, None
        if getattr(data, 'ndim', 1) == 2:
            data = data.mean(axis=1)
        # Jeśli float -> przeskaluj do int16 dla zgodności
        if np.issubdtype(getattr(data, 'dtype', None), np.floating):
            x = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype(np.int16)
        else:
            x = data.astype(np.int16)
        return sr, x
    except Exception:
        return None, None

def align_signals(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def stable_edges(lo, hi, bins):
    if lo == hi: lo, hi = float(lo), float(lo + 1.0)
    return np.linspace(lo, hi, bins + 1)

def common_hist(x, y, bins, eps, density=False):
    lo = float(min(x.min(initial=0), y.min(initial=0)))
    hi = float(max(x.max(initial=1), y.max(initial=1)))
    edges = stable_edges(lo, hi, bins)
    h1, _ = np.histogram(x, bins=edges, density=density)
    h2, _ = np.histogram(y, bins=edges, density=density)
    return h1 + eps, h2 + eps, edges

# --- Metryki ---
def kl_divergence(p, q):
    # D_KL(P || Q) = sum( P * log(P/Q) )
    return float(np.sum(p * (np.log(p) - np.log(q))))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def perm_p_divergence(x, y, edges, B, rng, eps, metric="JS"):
    """
    Oblicza statystykę (JS lub KL) oraz p-value metodą permutacji (Bootstrapping).
    metric: "JS", "KL_PQ" (P||Q), "KL_QP" (Q||P)
    """
    # 1. Observed
    h1, _ = np.histogram(x, bins=edges, density=True); p1 = h1 + eps
    h2, _ = np.histogram(y, bins=edges, density=True); p2 = h2 + eps
    p1 /= p1.sum()
    p2 /= p2.sum()
    
    if metric == "JS":
        stat_obs = js_divergence(p1, p2)
    elif metric == "KL_PQ":
        stat_obs = kl_divergence(p1, p2)
    elif metric == "KL_QP":
        stat_obs = kl_divergence(p2, p1)
    else:
        return 0.0, 1.0

    if B <= 0:
        return stat_obs, None

    # 2. Permutations
    xy = np.concatenate([x, y])
    n1 = len(x)
    cnt = 0
    
    for _ in range(B):
        rp = rng.permutation(xy)
        a, b = rp[:n1], rp[n1:]
        
        ha, _ = np.histogram(a, bins=edges, density=True); pa = ha + eps
        hb, _ = np.histogram(b, bins=edges, density=True); pb = hb + eps
        pa /= pa.sum(); pb /= pb.sum()
        
        if metric == "JS":
            stat_perm = js_divergence(pa, pb)
        elif metric == "KL_PQ":
            stat_perm = kl_divergence(pa, pb)
        elif metric == "KL_QP":
            stat_perm = kl_divergence(pb, pa)
            
        if stat_perm >= stat_obs:
            cnt += 1
            
    return stat_obs, (cnt + 1) / (B + 1)

def lsb_counts(x):
    x_int = x.astype(np.int16)
    lsb = x_int & 1
    return np.sum(lsb == 0), np.sum(lsb == 1)

def fdr_bh(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg Correction"""
    p = np.array([v if (v is not None and np.isfinite(v)) else 1.0 for v in pvals])
    n = len(p)
    if n == 0: return []
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty(n, dtype=float)
    q_adj[order] = np.clip(q_sorted, 0.0, 1.0)
    return q_adj.tolist()

# --- DCT Implementation ---
def framewise_dct(x: np.ndarray, frame: int):
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max or 1.0)
    
    nframes = len(x) // frame
    if nframes == 0: return np.empty((0, frame))
    
    xf = x[:nframes*frame].reshape(nframes, frame)
    n = frame
    k = np.arange(n).reshape(-1, 1)
    nidx = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (nidx + 0.5) * k / n)
    return xf @ basis.T

def dct_band_indices(sr, n, f_lo, f_hi):
    k_lo = max(1, int(np.floor(2*n*f_lo/sr)))
    k_hi = min(n-1, int(np.ceil(2*n*f_hi/sr)))
    if k_lo >= k_hi: k_lo = max(1, min(n-2, k_hi-1))
    return k_lo, k_hi

# ============================
# ====== VISUALIZATION =======
# ============================

def plot_advanced_visuals(carrier, stego, sr, label, out_dir, fname_base):
    limit = np.max(np.abs(carrier)) if len(carrier) > 0 else 1
    bins = np.linspace(-limit, limit, 256)
    hc, edges = np.histogram(carrier, bins=bins, density=True)
    hs, _ = np.histogram(stego, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Log Histogram
    axes[0].plot(centers, hc+1e-9, label="Cover")
    axes[0].plot(centers, hs+1e-9, label="Stego", ls='--')
    axes[0].set_yscale('log')
    axes[0].set_title(f"Log-Histogram: {label}")
    axes[0].legend()
    
    # 2. Spectrogram Cover
    f, t, Sxx = signal.spectrogram(carrier, sr)
    axes[1].pcolormesh(t, f, 10*np.log10(Sxx+1e-10), shading='gouraud', cmap='inferno')
    axes[1].set_title("Spectrogram (Cover)")
    
    # 3. Diff Spectrogram
    _, _, Sxx_s = signal.spectrogram(stego, sr)
    diff = 10*np.log10(Sxx_s+1e-10) - 10*np.log10(Sxx+1e-10)
    max_d = np.max(np.abs(diff))
    im = axes[2].pcolormesh(t, f, diff, shading='gouraud', cmap='seismic', vmin=-max_d, vmax=max_d)
    axes[2].set_title("Difference (Stego - Cover)")
    fig.colorbar(im, ax=axes[2], label="dB Delta")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"vis_{fname_base}.png")
    plt.close()

def plot_heatmap(rows, metric_key, title, filename, out_dir):
    # Agregacja p-values
    data = {} # {method: {domain: val}}
    domains = set()
    
    for r in rows:
        if r["metric"] != metric_key: continue
        row_id = r["id"]
        dom = r["domain"]
        val = r["p"] if r["p"] is not None else 1.0
        
        if row_id not in data: data[row_id] = {}
        data[row_id][dom] = val
        domains.add(dom)
        
    if not data: return

    sorted_ids = sorted(data.keys())
    
    def get_freq_sort_key(name):
        if "Amplitude" in name: return -1.0
        if "DCT" in name:
            try:
                range_str = name.split()[1]
                start_freq = range_str.split('-')[0]
                return float(start_freq)
            except: return 999.0
        return 999.0

    sorted_doms = sorted(list(domains), key=get_freq_sort_key)
    
    matrix = np.zeros((len(sorted_ids), len(sorted_doms)))
    for i, rid in enumerate(sorted_ids):
        for j, d in enumerate(sorted_doms):
            matrix[i, j] = data[rid].get(d, np.nan)
            
    # Plot -log10(p)
    log_matrix = -np.log10(matrix + 1e-30)
    
    plt.figure(figsize=(10, len(sorted_ids)*0.5 + 2))
    plt.imshow(log_matrix, aspect='auto', cmap='magma', vmin=0, vmax=5) # vmax=5 -> p=1e-5
    plt.colorbar(label="-log10(p-value)")
    plt.xticks(range(len(sorted_doms)), sorted_doms, rotation=45, ha='right')
    plt.yticks(range(len(sorted_ids)), sorted_ids)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()

# ============================
# ========= MAIN =============
# ============================

def run():
    cfg = CONFIG
    out_dir = pathlib.Path(cfg["OUT_DIR"])
    vis_dir = out_dir / "visualizations"
    ensure_dir(out_dir)
    ensure_dir(vis_dir)
    
    rng = np.random.default_rng(cfg["SEED"])
    random.seed(cfg["SEED"])

    print("--- [Ultimate Steganalysis v3.0: Stratified FDR + KL] ---")

    # 1. Zbieranie plików
    carrier_dir = cfg["CARRIER_DIR"]
    if not os.path.exists(carrier_dir): 
        print(f"Brak katalogu {carrier_dir}!"); return
        
    exts = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    all_carriers = sorted([f for f in os.listdir(carrier_dir) if os.path.splitext(f)[1].lower() in exts])
    
    if cfg["NUM_SAMPLES_STATS"] and len(all_carriers) > cfg["NUM_SAMPLES_STATS"]:
        files_stats = random.sample(all_carriers, cfg["NUM_SAMPLES_STATS"])
    else:
        files_stats = all_carriers
        
    files_vis = set(files_stats[:cfg["NUM_SAMPLES_VISUAL"]])
    
    rows = [] 

    for idx, fname in enumerate(files_stats):
        print(f"Processing [{idx+1}/{len(files_stats)}]: {fname}")
        
        path_c = os.path.join(carrier_dir, fname)
        sr_c, sig_c = load_wav_mono(path_c)
        if sig_c is None: continue

        # Szukamy par stego
        for raw_method, clean_name in cfg["NAME_MAPPING"].items():
            possibilities = [
                os.path.join(cfg["STEGO_ROOT_DIR"], raw_method, "Stegobjects", fname),
                os.path.join(cfg["STEGO_ROOT_DIR"], raw_method, fname),
                os.path.join(cfg["STEGO_ROOT_DIR"], fname.replace(".wav", f"_{raw_method}.wav"))
            ]
            path_s = next((p for p in possibilities if os.path.exists(p)), None)
            
            if not path_s: continue
            
            sr_s, sig_s = load_wav_mono(path_s)
            if sig_s is None: continue
            
            c_trim, s_trim = align_signals(sig_c, sig_s)
            res_id = f"{clean_name} ({fname})"
            
            # === ANALIZA 1: AMPLITUDA ===
            if cfg["ENABLE"]["AMPLITUDE"]:
                h_c, h_s, edges = common_hist(c_trim, s_trim, cfg["BINS_AMPLITUDE"], cfg["EPS_SMOOTH"], density=False)
                
                if cfg["ENABLE"]["CHISQ"]:
                    stat, p = chisquare(h_s, f_exp=h_c)
                    rows.append({"id": res_id, "domain": "Amplitude", "metric": "chi2", "val": stat, "p": p})
                
                if cfg["ENABLE"]["KS"]:
                    stat, p = ks_2samp(c_trim, s_trim)
                    rows.append({"id": res_id, "domain": "Amplitude", "metric": "ks", "val": stat, "p": p})
                
                if cfg["ENABLE"]["PERMUTE_JS"]:
                    val, p = perm_p_divergence(c_trim, s_trim, edges, cfg["PERM_B"], rng, cfg["EPS_SMOOTH"], "JS")
                    rows.append({"id": res_id, "domain": "Amplitude", "metric": "js_perm", "val": val, "p": p})

                if cfg["ENABLE"]["KL_PQ"]:
                    n_perm = cfg["PERM_B"] if cfg["ENABLE"]["PERMUTE_KL"] else 0
                    val, p = perm_p_divergence(c_trim, s_trim, edges, n_perm, rng, cfg["EPS_SMOOTH"], "KL_PQ")
                    rows.append({"id": res_id, "domain": "Amplitude", "metric": "kl_pq", "val": val, "p": p})

                if cfg["ENABLE"]["KL_QP"]:
                    n_perm = cfg["PERM_B"] if cfg["ENABLE"]["PERMUTE_KL"] else 0
                    val, p = perm_p_divergence(c_trim, s_trim, edges, n_perm, rng, cfg["EPS_SMOOTH"], "KL_QP")
                    rows.append({"id": res_id, "domain": "Amplitude", "metric": "kl_qp", "val": val, "p": p})

            # === ANALIZA 2: DCT BANDS ===
            if cfg["ENABLE"]["DCT_BANDS"]:
                Dc = framewise_dct(c_trim, cfg["FRAME_LEN_DCT"])
                Ds = framewise_dct(s_trim, cfg["FRAME_LEN_DCT"])
                
                for (flo, fhi) in cfg["DCT_BANDS_HZ"]:
                    k_lo, k_hi = dct_band_indices(sr_c, cfg["FRAME_LEN_DCT"], flo, fhi)
                    band_c = np.abs(Dc[:, k_lo:k_hi]).ravel()
                    band_s = np.abs(Ds[:, k_lo:k_hi]).ravel()
                    
                    if len(band_c) == 0: continue
                    dom_name = f"DCT {int(flo)}-{int(fhi)}"
                    
                    if cfg["ENABLE"]["KS"]:
                        stat, p = ks_2samp(band_c, band_s)
                        rows.append({"id": res_id, "domain": dom_name, "metric": "ks", "val": stat, "p": p})
                        
                    # Histogram dla dywergencji
                    hlo, hhi = min(band_c.min(), band_s.min()), max(band_c.max(), band_s.max())
                    edges_dct = stable_edges(hlo, hhi, cfg["BINS_DCT"])
                        
                    if cfg["ENABLE"]["PERMUTE_JS"]:
                        val, p = perm_p_divergence(band_c, band_s, edges_dct, cfg["PERM_B"], rng, cfg["EPS_SMOOTH"], "JS")
                        rows.append({"id": res_id, "domain": dom_name, "metric": "js_perm", "val": val, "p": p})

                    if cfg["ENABLE"]["KL_PQ"]:
                        n_perm = cfg["PERM_B"] if cfg["ENABLE"]["PERMUTE_KL"] else 0
                        val, p = perm_p_divergence(band_c, band_s, edges_dct, n_perm, rng, cfg["EPS_SMOOTH"], "KL_PQ")
                        rows.append({"id": res_id, "domain": dom_name, "metric": "kl_pq", "val": val, "p": p})

                    if cfg["ENABLE"]["KL_QP"]:
                        n_perm = cfg["PERM_B"] if cfg["ENABLE"]["PERMUTE_KL"] else 0
                        val, p = perm_p_divergence(band_c, band_s, edges_dct, n_perm, rng, cfg["EPS_SMOOTH"], "KL_QP")
                        rows.append({"id": res_id, "domain": dom_name, "metric": "kl_qp", "val": val, "p": p})

            # === ANALIZA 3: LSB ===
            if cfg["ENABLE"]["LSB"]:
                z_c, o_c = lsb_counts(c_trim)
                z_s, o_s = lsb_counts(s_trim)
                stat, p = chisquare([z_s+1, o_s+1], f_exp=[z_c+1, o_c+1])
                rows.append({"id": res_id, "domain": "LSB", "metric": "chi2", "val": stat, "p": p})

            # === WIZUALIZACJA ===
            if fname in files_vis:
                bname = f"{clean_name.replace(' ', '_')}_{fname.replace('.wav', '')}"
                plot_advanced_visuals(c_trim, s_trim, sr_c, f"{clean_name} {fname}", vis_dir, bname)

    # --- KOREKCJA FDR (Stratified by Metric) ---
    print("Calculating FDR corrections (Stratified by Metric)...")
    
    # Grupowanie indeksów według metryki
    metric_groups = {}
    for i, r in enumerate(rows):
        if r["p"] is None: 
            r["q"] = None
            continue
        m = r["metric"]
        if m not in metric_groups: metric_groups[m] = []
        metric_groups[m].append(i)

    # Liczenie FDR osobno dla każdej grupy
    for m, indices in metric_groups.items():
        p_vals = [rows[i]["p"] for i in indices]
        if not p_vals: continue
        q_vals = fdr_bh(p_vals)
        for idx, q in zip(indices, q_vals):
            rows[idx]["q"] = q

    # --- EKSPORT WYNIKÓW ---
    csv_path = out_dir / "results_ultimate_stratified.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Domain", "Metric", "Value", "p-value", "q-value (FDR)"])
        for r in rows:
            pv = f"{r['p']:.6g}" if r['p'] is not None else ""
            qv = f"{r['q']:.6g}" if r['q'] is not None else ""
            w.writerow([r["id"], r["domain"], r["metric"], f"{r['val']:.6g}", pv, qv])

    # --- GENEROWANIE HEATMAP ---
    # Generujemy tylko jeśli dana metryka była liczona
    if cfg["ENABLE"]["KS"]:
        plot_heatmap(rows, "ks", "KS Test (FDR Stratified)", "heatmap_ks.png", out_dir)
    
    if cfg["ENABLE"]["PERMUTE_JS"]:
        plot_heatmap(rows, "js_perm", "JS Permutation (FDR Stratified)", "heatmap_js_perm.png", out_dir)

    if cfg["ENABLE"]["KL_PQ"] and cfg["ENABLE"]["PERMUTE_KL"]:
        plot_heatmap(rows, "kl_pq", "KL Cover||Stego (FDR Stratified)", "heatmap_kl_pq.png", out_dir)
        
    if cfg["ENABLE"]["KL_QP"] and cfg["ENABLE"]["PERMUTE_KL"]:
        plot_heatmap(rows, "kl_qp", "KL Stego||Cover (FDR Stratified)", "heatmap_kl_qp.png", out_dir)

    print(f"\nZakończono.")
    print(f"CSV: {csv_path}")
    print(f"Heatmapy i spektrogramy w: {out_dir}")

if __name__ == "__main__":
    run()