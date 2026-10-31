# Steganalysis

`lib/Steganalysis.py` replaces the stand-alone scripts `Steganalysis/0_...5_*.py`.
It works on **pairs** (carrier, clean stegobject) and has two independent parts.

## Running it

```bash
# after a run (FOR_CLASSIFIER enables it by default)
python coder.py --test FOR_CLASSIFIER --seed 1
python coder.py --test CAPACITY_SWEEP --steganalysis

# on an existing result folder (uses steganalysis_pairs.json, or reconstructs the
# pairs from <method>/Stegobjects for older runs; attacked copies are skipped)
python coder.py --steganalysis-from ~/Results/<run> --carriers-dir Carriers --perm-b 500

# one suspicious file against the saved models
python coder.py --predict file.wav --models ~/Results/<run>/Steganalysis/ML/models
```

Options are in `lib.Steganalysis.DEFAULT_CONFIG` and can be overridden with
`RUN_SETTINGS['STEGANALYSIS_CONFIG']` in `coder.py` (e.g. `{'PERM_B': 1000,
'N_SPLITS': 10, 'CLASSIFIERS': ['SVM']}`) or `--perm-b`.

| Old script | Now |
|---|---|
| `0_statistical_steganalysis.py` | `run_statistical_steganalysis` → `Steganalysis/Statistical/` |
| `1_extract_features.py` + `utils_features.py` | `extract_features` → `Steganalysis/ML/features/<method>/features_<set>.pkl` |
| `2_train_classifiers.py` | `run_ml_steganalysis` → `ml_results.csv/json`, `models/<CLF>/<method>/model_<set>.pkl` |
| `3_predict_file.py` | `--predict` / `predict_file` |
| `4_thesis_plots.py`, `5_generate_roc_plots.py` | `Steganalysis/ML/plots/` (heatmap, point plot, ROC, transferability, t-SNE) |

## 1. Statistical tests (`Statistical/`)

For every pair (signals aligned to the common length):

| Domain | Tests |
|---|---|
| Amplitude histogram (256 bins) | Chi², KS, Jensen-Shannon and KL (cover‖stego, stego‖cover) with permutation p-values |
| DCT band magnitudes (1024-sample frames; 0-1, 1-2, 2-3, 3-4, 4-8 kHz) | KS, JS, KL with permutation p-values |
| LSB (share of 0 / 1) | Chi² |

Permutation p-value: `(#{stat_perm >= stat_obs} + 1) / (PERM_B + 1)` - with
`PERM_B = 200` the smallest possible p is 0.005; use 500-1000 for publication.
p-values are corrected with Benjamini-Hochberg **separately for every metric**
(q-values).

Chi² (`chi2_homogeneity`) is the two-sample test of homogeneity (2 × K contingency
table on the raw counts of cover and stegobject). Adjacent bins are merged until every
expected count is ≥ `CHI2_MIN_EXPECTED` (5). Until 2026-10-31 both Chi² tests compared
the stegobject with the cover histogram taken as exact expected counts. In the amplitude
test an empty cover bin had the expected count `EPS_SMOOTH = 1e-12`, so a single stego
sample above the cover's peak gave χ² ≈ 10¹³ and p = 0: **every method was "detected"
on every carrier** (amplitude Chi² 100% for all 9 methods in the 50 bps comparison).
Results of the amplitude and LSB Chi² from earlier runs should not be used.

Cover and stegobject are paired: unmodified samples (e.g. unvoiced frames) are
identical, so all two-sample tests here are conservative. A detection says that the
first-order statistics of *this* pair differ. It does not mean that a detector without
the cover could detect the stegobject. That is what the ML part measures.

Output: `results_stratified.csv` (all tests with `p` and `q`),
`summary_detection_rate.csv` (share of pairs with `q < 0.05` per method, domain and
metric), `heatmap_<metric>.png` (-log10 q per pair and domain - the old script plotted
raw p-values under an "FDR" title), `visualizations/` (log histogram, spectrogram,
spectrogram difference for the first carriers).

## 2. Machine-learning detectors (`ML/`)

Feature sets (computed once per carrier for the covers):

| Set | Dimension | Content |
|---|---|---|
| RME | 80 | reversed-mel filter bank log energies: mean, std, skewness, kurtosis of 20 bands |
| DHS | 4 | statistics of the energy above 0.75 · Nyquist of the 2nd-order difference signal |
| WMC | 78 | MFCC (mean, std) of 3 wavelet sub-bands (db4, 2 levels) |
| DMC | 26 | MFCC (mean, std) of the 2nd-order difference signal |

Classifiers: SVM (RBF, C = 10, standardised features) and FastRealBoostBins (`frbb`
module - used only if importable, otherwise skipped with a warning).

**Evaluation: stratified, carrier-grouped k-fold cross-validation** (`N_SPLITS = 5`).
The cover and the stego version of a carrier are always in the same fold, so a detector
is never tested on a carrier it has seen. All metrics are computed from the pooled
out-of-fold predictions:

* `accuracy`, `accuracy_fold_std`, `confusion_matrix`,
* `detection_error` = ½(P_FA + P_MD) at the classifier's threshold,
* `P_E` = min over thresholds of ½(P_FA + P_MD),
* `AUC`, `TPR_at_FPR_5%`, `TPR_at_FPR_10%` (largest TPR with FPR ≤ target),
* training / prediction time.

The old scripts used a single 80/20 split (4 test files with 10 carriers) and the
transferability matrix evaluated each model on its own training data. Now:

* **ROC curves** are drawn from out-of-fold scores;
* **transferability** (train on method A, test on method B) uses folds over the
  carriers shared by both methods - test carriers are never used for training;
* **final models** (trained on all pairs) are saved for `--predict`; a file is
  `SUSPICIOUS` if at least 2 of the 4 feature-set models say stego.

With fewer than `MIN_CARRIERS_ML = 4` carriers the ML part is skipped; below 20 carriers
a warning about the large variance is logged.

Output: `ml_results.csv/json`, `transferability_<CLF>_<set>.csv`, `features/`,
`models/`, and in `plots/`: `accuracy_heatmap_<CLF>.png`, `accuracy_pointplot_<CLF>.png`,
`roc_<CLF>_<set>.png`, `transferability_<CLF>_<set>.png`, `tsne_<set>.png`.
