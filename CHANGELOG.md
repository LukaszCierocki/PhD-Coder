# Changelog

## 2026-10-31 (2) - methods checked against the papers

Checked against Xue et al. (IEEE Access 2019), Wen et al. (MTAP 2021) and the PC paper
(Pejaś & Cierocki, "Adaptive High-Fidelity Speech Steganography Based on DCT and
Multi-Level SVD"). Numbers: 10 CMU ARCTIC carriers, 50 bps, α = β = 0.7.

### Embedding rules (change results - previous behaviour: `LEGACY=True` / `--legacy`)

* **PC methods follow Algorithm 4 (RSLBE) of the PC paper** (`lib/EmbeddingRules.py`,
  equal to Algorithm 3 with the Lemma 1 factors). The previous code differed:
  PC Xue [v1]/[v2] bit 1 (`b' = a + β_A S` instead of `a + β S`), PC [v1] bit 0 (shift
  relative to the current values - BER 2.57% → 0%), PC [v2] (reset to the mean with the
  adjusted instead of the constant factors - MSE 23 067 → 14 702). PC Xue: MSE 39 360 →
  29 317.
* **PC [v3] score of Algorithm 2**: `|a' − b'| / MSE` instead of signed margin / mean
  absolute error (BER 0.60% → 1.14%, MSE 8 219 → 6 835).
* **Xue Eq. (22)**: the symmetric form `b' = a + T2` is kept (the printed `b + T2`
  contradicts the paper's own `d_j ≤ −T_bu,2` and gives ≈ 10% BER at the paper's
  α = 0.2); the printed form is the LEGACY rule.
* `ADAPTIVE` / `REDUCED` only affect the LEGACY rules. Legacy methods are named
  `..._LEGACY` / `... (legacy)`.

### Lemma 1

The guaranteed margins are `γ_α` (= α for the [v1] modes, α/2 for the [v2] modes) for
bit 0 and β/2 for bit 1, not α and β: the no-modification branches were treated as if
the adjusted factors were equal to their constant parts. Corrected statement and proof:
see the separate LaTeX file; tested in `test_suite/test_rules.py` (including a
counter-example to the submitted version and the non-negativity limits α ≤ 1 for PC [v1],
α, β ≤ 2).

### Frequency-band safeguards

* even number of band coefficients (`2M`, as in all three papers) - only 80 bps frames
  change (25 → 24 bins);
* band actually used reported (log, key, CSV `BAND [Hz]`), band energy check (CSV
  `BAND ENERGY [dB]`, warning below −35 dB), warning for a band including DC;
* the receiver must use the band / `LEVELS` / `HANKEL_ROWS` stored in the key -
  otherwise `ValueError` (before: silently wrong bits, e.g. 4% BER with another band);
* `WHERE_TO_HIDE='ALL'` (no VAD) and `LABEL` (display name).

### Metrics, attacks, presets

* PSNR with the full-scale peak of 16-bit PCM, `A = 32767` (as defined in the PC paper).
* Re-quantisation 16 → 8 → 16 bit (was 10 bit, the paper says 8; parameter
  `REQUANTIZE_BITS`), resampling fs → fs/2 → fs with an anti-aliasing filter (was
  decimation without a filter, which aliased 4-8 kHz into the band). All attack parameters
  in `lib.Attacks.DEFAULT_ATTACK_PARAMS`, recorded in the run configuration.
* Preset `BASELINES`: Xue (α = β = 0.2, every frame) and Wen (α = 0.61, 100 Hz - 3 kHz)
  as published, labelled `(orig.)`; charts draw `(orig.)` / `(legacy)` variants in the
  colour of their method with dashed / dotted lines.
* Command line: `--legacy`, `--attacks all|none|NAMES`.
* `EMB TIME` stays without the VAD (the PC paper text says it includes it - to be
  changed in the paper).

## 2026-10-31 - review: correctness, safeguards, steganalysis and charts in the coder

Numbers below: 10 CMU ARCTIC carriers (`arctic_a0001`-`a0010`, 16 kHz), 50 bps,
band 1500-2500 Hz, α = β = 0.7, measured before and after the change.

### Fixes that change published / earlier results

| Problem | Effect before | After |
|---|---|---|
| **MSE computed in int16** - `carrier - stego` and `** 2` overflowed for differences > 181 | MSE far too small and meaningless (Xue, a0001: reported **1 565**, true **22 390**). The earlier "MSE vs Capacity" chart (values ~700-1 500) is invalid; PSNR used it too | float64; Xue at 50 bps ≈ 47 000-50 000 on average |
| **PSNR** used `max(stegobject)` (not the absolute peak) and the wrong MSE | wrong values | `10 log10(max|carrier|² / MSE)`; SNR and SegSNR added |
| **Embedding time dominated by logging** - `logger.debug(f"... {array}")` formats every numpy array even when DEBUG is off | PC v3: ~35 s per 4 s carrier, ~90% of it string formatting; all multi-level methods inflated | PC v3 ≈ 0.9 s per carrier (≈ 0.5-1.7 s from 25 to 100 bps); the old "Average Embedding Time" chart measured logging |
| **Xue: bit 1 rule** `σo' = σo + β σe` instead of `σe + β σe` | bit lost whenever `σo < (1-β) σe`: BER **0.12%** | **0%** (same fix in Xue multi) |
| **Wen: MDCT band** - number of coefficients passed instead of frame length | 1500-2500 Hz configured, 750-1250 Hz used: BER **0.06%** | band as configured, **0%** |
| **PC v3 level map leaked between carriers** - not reset in `set_carrier`, read with a wrapping index | 2nd extraction (e.g. attacked copy) of the 2nd carrier used carrier 1's levels: BER **7.6%** instead of 0.5%; every later carrier affected when attacks are on | explicit extraction key, state reset per carrier |
| **NOISE attack** cast a float signal in [-1, 1] to int16 | the "noised" file was practically silent | noise added in the int16 domain, target SNR reached (±0.5 dB) |
| **BER ignored missing bits** (`zip` stops at the shorter sequence) | attacks that shorten the signal looked better | missing / extra bits count as errors |
| **Stegobject truncated, not rounded**, no clipping (int16 wrap-around possible) | systematic −0.5 LSB error, wrap-around on overflow | rounding + clipping, `CLIPPED` column. PC v1 (non-adaptive): 3.17% → 3.29% (2 knife-edge bits) |
| **WAV reading** via `soundfile` float × 32767, truncated | 99.7% of the samples of every loaded stegobject changed by 1 LSB | exact int16 reading |
| **Diagnostic lists** (`s_even`, `alphas`, ...) accumulated over carriers | per-carrier singular value / α-β charts mixed carriers | reset per carrier and per embedding |

Verified unchanged: the per-frame output of PC Xue v1/v2 and PC v1/v2/v3 (1-8 levels) is
bit-identical to the previous implementation; only the final rounding differs.

### Safeguards

* Parameter validation before embedding (see `docs/METHODS.md`): band vs Nyquist, band
  width, hop vs frame (DCT: equal, Wen: half), integer samples per frame, VAD type,
  secret vs capacity, missing parameters (e.g. `REDUCED` missing in `FOR_CLASSIFIER`
  used to raise `KeyError`), carrier length, no voiced frames.
* rVAD framing is checked against the extractor framing; WebRTC framing no longer
  creates a zero-padded last frame that the extractor never sees.
* Extraction key (JSON) per stegobject - extraction no longer depends on hidden state of
  the method object; a fresh receiver gets the same bits.
* `--config` replay works (was broken: missing secret size, logger, wrong `extraction`
  and `prepare_method_directory` arguments, `MDCT_SVD_WEN` missing from the class map).
* Methods are imported lazily (`lib/Registry.py`) - the deleted legacy methods no
  longer break `coder.py`.
* One failing (method, carrier) pair or attack is logged and listed in `failures`; the
  run continues. Diagnostic plots never abort a run.
* Seeds: `--seed` makes secrets and random attacks reproducible; `Secret` no longer
  re-seeds the global `random` module (same texts for the same seeds as before).
* `HANKEL_ROWS > 2`: detail components beyond the second were dropped on
  reconstruction.
* 145 pytest tests (`test_suite/`).

### New

* Steganalysis built into the coder (`lib/Steganalysis.py`, `--steganalysis`,
  `--steganalysis-from`, `--predict`) with carrier-grouped cross-validation, q-value
  heatmaps, ROC from out-of-fold scores, P_E / AUC / TPR@FPR, leakage-free
  transferability. Fixed on the way: features were never saved
  (`1_extract_features.py`), `3_predict_file.py` looked for models in the wrong folder,
  heatmaps showed raw p-values under an FDR title.
* "*metric* vs Capacity" charts (`lib/Plots.py`, `--plots-from`) in the thesis style,
  per-attack BER charts, CSV tables, optional colour-blind-safe palette and CI bands.
* Presets `CAPACITY_SWEEP` (25-100 bps) and `ROBUSTNESS` (all attacks).
* New result columns: `SNR [dB]`, `SEGSNR [dB]`, `EFF. CAP. [bps]`, `BIT ERRORS`,
  `EXTRACTED [b]`, `CLIPPED`, `FREQ_L`, `FREQ_H`, `FRAME [ms]`, `HOP [ms]`.
* PC v3 BASELINE / OPTIMIZED are thin subclasses of PC v3 (identical stegobjects).

### Removed

* `Steganalysis/*.py` scripts (replaced by `lib/Steganalysis.py`, mapping in
  `docs/STEGANALYSIS.md`).
