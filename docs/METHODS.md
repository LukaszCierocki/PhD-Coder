# Methods

All methods hide **one bit per frame**. The embedding pipeline, the parameter
validation and the extraction loop are shared (`lib/Method.py`); the rules that change
the singular values live in one module, `lib/EmbeddingRules.py`, and are unit-tested
against the papers (`test_suite/test_rules.py`).

Notation: `a = σe` - largest singular value of the matrix built from the *even*
coefficients of the band (`d_1^{L_J}`, `σ11`), `b = σo` - the same for the *odd*
coefficients (`d_1^{ℒ_J}`, `σ21`), `S = S_avg = (a + b) / 2`,
`δ = (a − b) / S = 2(a − b)/(a + b)`, `α = ALPHA`, `β = BETA`. Extraction is the same
for every method:

```
bit = 0 if a > b else 1
```

## Common pipeline

1. **Framing.** Frames of `FRAME_LENGTH` ms every `HOP_DURATION_MS` ms, only full frames;
   the tail that does not fill a frame is copied unchanged. The duration must correspond
   to an integer number of samples (e.g. 12.5 ms at 16 kHz = 200 samples; 10 ms at
   22.05 kHz is rejected).
   * DCT methods concatenate frames without a synthesis window, so
     `HOP_DURATION_MS == FRAME_LENGTH` is required.
   * Wen uses the MDCT with a Kaiser-Bessel-derived window, `FRAME_LENGTH = 2 x HOP`.
2. **Frames that carry bits** (`WHERE_TO_HIDE`):
   * `VOICED` (default) - frames marked as speech by `rVAD` (default) or `webrtc`
     (10/20/30 ms frames, 8/16/32/48 kHz);
   * `ALL` - every frame, no VAD (Xue et al. as published).
3. **Band.** See [Frequency band](#frequency-band) below.
4. **Embedding** in the frame, inverse transform, overlap-add.
5. **Quantisation.** The stegobject is rounded to the nearest integer and clipped to int16
   (`CLIPPED` column); NaN/inf raise an error.
6. **Extraction key** (`Keys/*.key.json`): carrying-frame mask, number of bits,
   frame/hop, sampling rate, band bins and band in Hz, the parameters the receiver must
   share, and for PC v3 the level of every bit.

### Multi-level Hankel SVD (PC family, Xue multi)

For a coefficient vector `x` and `L = LEVELS`:

```
for level 1..L:  H = Hankel(x, HANKEL_ROWS);  U S V^T = svd(H);  x = diagavg(s1 u1 v1^T)
```

The largest singular value of the deepest level is modified, then the decomposition is
inverted (detail components of every level are added back). With unmodified singular
values the reconstruction is exact (tested). After a modification, diagonal averaging
projects the matrix back onto Hankel structure, so the singular values measured by the
receiver differ slightly from the intended ones - this, plus int16 rounding, is the
reconstruction part of the perturbation `ε` in Lemma 1 and the source of the remaining
bit errors of PC v3.

## Frequency band

| Safeguard | Behaviour |
|---|---|
| `FREQ_H` above the Nyquist frequency of the carrier | `ValueError` in `set_carrier` |
| band narrower than the method needs (Xue/Wen: 2, multi-level: `2·(HANKEL_ROWS+1)` coefficients) | `ValueError`, the message shows the band in bins and Hz |
| odd number of coefficients in the band | the last bin is dropped - Xue, Wen and the PC paper assume `2M` coefficients (equal even/odd branches). Example: 12.5 ms frames at 16 kHz, 1.5-2.5 kHz → 24 instead of 25 bins. Not applied with `LEGACY=True` |
| actual band differs from the requested one (bin quantisation) | reported: log, key (`band_bins`, `band_hz`), CSV column `BAND [Hz]` |
| band includes the DC coefficient (DCT methods) | warning - changing it shifts the mean of every frame (clicks at frame borders) |
| band with (almost) no signal energy | warning if the median band/frame energy of the carrying frames is below −35 dB (CSV column `BAND ENERGY [dB]`; speech in 1.5-2.5 kHz ≈ −20 dB on CMU ARCTIC, leakage floor of rectangular DCT frames ≈ −40 dB) and for frames with no energy in the band |
| receiver configured with another band / `LEVELS` / `HANKEL_ROWS` than the sender | `ValueError` in `extraction` (compared with the key) instead of silently wrong bits |

DCT bin `k` of an `N`-sample frame is `k · fs / (2N)` (bins selected by truncation);
MDCT bin `k` of a `2N`-sample frame is centred at `(k + 0.5) · fs / (2N)` (floor/ceil).
For Wen et al. this gives exactly their `2N_k = 116` coefficients for 100 Hz - 3 kHz with
a 320-sample window at 8 kHz (or 640 samples at 16 kHz).

## Versions

### PC paper rules (Pejaś & Cierocki), default

The four single-level modes are implemented exactly as **Algorithm 4 (RSLBE)** of the
paper (`lib.EmbeddingRules.rslbe`). It is equal to Algorithm 3 (GSLBE) with the adjusted
factors of Lemma 1 (checked numerically, max. difference 1e-14):

| Mode | Lemma 1 factors | bit 0 (keep if …; else …) | bit 1 (keep if …; else …) |
|---|---|---|---|
| PC Xue [v1] (`…_MULTI_V1`) | `α_A = α`, `β_A = β + δ` | `a − b ≥ αS`; `a' = b + αS` | `b − a ≥ βS/2`; `b' = a + βS` |
| PC Xue [v2] (`…_MULTI_V2`) | `α_A = α − δ`, `β_A = β + δ` | `a − b ≥ αS/2`; `a' = b + αS` | as v1 |
| PC [v1] (`…_MULTI_PC_V1`) | `α_A = α`, `β_A = β + δ` | `a − b ≥ αS`; `a' = b + αS/2`, `b' = b − αS/2` | `b − a ≥ βS/2`; `a' = S − βS/2`, `b' = S + βS/2` |
| PC [v2] (`…_MULTI_PC_V2`) | `α_A = α − δ`, `β_A = β + δ` | `a − b ≥ αS/2`; `a' = S + αS/2`, `b' = S − αS/2` | as v1 |

`ADAPTIVE` and `REDUCED` have no effect in this mode (the variant fixes the factors;
Algorithm 4 is already the reduced form). The recorded α/β per frame (`alphas_betas`
charts) are the adjusted factors `α_A`, `β_A`.

**Guaranteed decision margins (corrected Lemma 1).** After embedding,
`(a' − b')/S ≥ γ_α` for bit 0 and `(b' − a')/S ≥ β/2` for bit 1, with `γ_α = α` for the
[v1] modes and `α/2` for the [v2] modes. A perturbation `ε = μS` of the difference is
therefore tolerated if `|μ| < min(γ_α, β/2)`; the bound is tight (no-modification
branches). The submitted version of Lemma 1 claimed `|μ| < min(α, β)`; counter-example
(PC [v1], bit 1, β = 0.7): `a = 1`, `b = 1.43` is not modified, and `μ = 0.5 < β` flips
the bit. Non-negative singular values require `α ≤ 1` for PC [v1] and `α, β ≤ 2` for the
reset-to-mean rules (`lib.EmbeddingRules.nonnegative_limits`).

**PC [v3] (`…_MULTI_PC_V3`)** - adaptive multi-level selection, Algorithm 2: the PC [v2]
rule is applied at every depth `1..LEVELS`, the candidate with the highest

```
score_j = |a'_j − b'_j| / (MSE(x, x_j) + ε)
```

is used (ties: the shallowest). The depth of every bit is side information stored in the
key. `PC [v3] baseline` decomposes every candidate depth from scratch, `PC [v3]` /
`PC [v3] optimized` decompose once and use prefixes - identical stegobjects; the variants
exist to compare embedding times.

### Xue et al. - `DIFFERENTIAL_SVD_XUE`

Even/odd coefficients reshaped to near-square matrices (zero padding if needed; the
padded entries are dropped after the inverse SVD), `T1 = α b`, `T2 = β a`:

```
bit 0: if a − b < T1:  a' = b + T1          (Eq. 19)
bit 1: if b − a < T2:  b' = a + T2          (Eq. 22, see below)
```

Eq. (22) is printed as `b' = b + T2`. That form does not reach the threshold (and is
decoded as 0) whenever `b < (1 − β) a`, contradicting the paper's own statement that
`d_j ≤ −T_bu,2` for bit 1; with the paper's `α = β = 0.2` it gives ≈ 10% BER without any
attack (CMU ARCTIC), while the paper reports BER = 0 under scaling and MP3. The symmetric
form is therefore used; `LEGACY=True` reproduces the printed form.

As published (BASELINES preset): `α = β = 0.2`, 1.5-2.5 kHz, **every frame**
(`WHERE_TO_HIDE='ALL'`), TIMIT at 8 kHz. `DIFFERENTIAL_SVD_XUE_MULTI` applies the same
rule on the deepest level of the multi-level Hankel SVD.

### Wen et al. - `MDCT_SVD_WEN`

MDCT of the windowed frame, band, even/odd matrices as in Xue, `S_min = min(a, b)`:

```
bit 0: if a − b < α S_min:  a' = S + α S_min / 2,  b' = S − α S_min / 2
bit 1: if b − a < α S_min:  a' = S − α S_min / 2,  b' = S + α S_min / 2
```

Exactly Algorithm 1 of Wen et al.; `BETA` is not used. Frames without a bit pass through
the analysis and synthesis window (`w²`), so the overlap-add reproduces the carrier; only
the first and the last half frame of the file are attenuated. The KBD shape parameter
(`WINDOW_BETA = 4.0`) is not given in the paper ("β = 320" there is the window length).

As published (BASELINES preset): `α = 0.61`, 100 Hz - 3 kHz, 320-sample window at 8 kHz,
a codec VAD (here: rVAD), TIMIT at 8 kHz.

## LEGACY - previous rules

`LEGACY=True` (or `python coder.py --legacy`) restores the embedding rules of the code
before 2026-10-31 for the methods whose rule changed. Names get `_LEGACY` /
` (legacy)`. Fixes of the shared pipeline (int16 rounding, exact WAV reading, Wen's band,
metrics) are **not** undone.

| Method | Previous rule (differs from the paper) | Effect (10 ARCTIC carriers, 50 bps, α = β = 0.7) |
|---|---|---|
| PC Xue [v1], [v2] | bit 1: `b' = a + β_A S` (adaptation applied twice); `ADAPTIVE`/`REDUCED` as before - `REDUCED=True` equals Algorithm 4 | MSE 39 360 vs 29 317, PESQ 4.201 vs 4.245, BER 0 |
| PC [v1] | bit 0: `a' = a + T S/2`, `b' = b − T S/2` (relative to the current values); with `ADAPTIVE=False` the sign is not guaranteed | BER 2.57% vs 0% |
| PC [v2] | reset to the mean with `α_A`, `β_A` instead of `α`, `β` | MSE 23 067 vs 14 702, PESQ 4.183 vs 4.242 |
| PC [v3] | score = signed margin / mean absolute error | BER 0.60% vs 1.14%, MSE 8 219 vs 6 835 (8 levels) |
| Xue (and Xue multi) | Eq. (22) as printed | see above |
| all | odd band coefficient counts kept | only 80 bps (12.5 ms) frames |

## Side information (non-blind extraction)

The extractor receives the extraction key, i.e. information that a real receiver would
have to obtain separately (the PC paper sends it over a side channel):

* the **carrying-frame mask** of the carrier - running the VAD on the stegobject (or on
  an attacked copy) may give a different mask;
* the **number of embedded bits**;
* for PC v3 the **depth of every bit**: `ceil(log2(LEVELS))` bits per payload bit
  (3 bits per bit for `LEVELS = 8`), i.e. more side information than payload.

## Other known limitations

* **Secret statistics.** The secret is ASCII text (letters and spaces), so the most
  significant bit of every byte is 0 and the bits are not balanced.
* **Matrix padding** (Xue, Wen) - for band sizes that are not a product of two close
  integers, the padded entries are dropped after the modification, which slightly
  changes the singular values.
* **PC v3 error floor.** The level is chosen from the intended singular values; after
  Hankelization and rounding the decoder may measure a different sign (≈ 0.5-1% of the
  bits on ARCTIC). Scoring the candidates with the margin measured on the rounded
  candidate frame removes these errors (tested: 0.60% → 0%), but it is not part of
  Algorithm 2 and not implemented.

## Validation (raised before embedding)

| Check | Error |
|---|---|
| missing / non-numeric / non-positive `ALPHA`, `BETA`, `FRAME_LENGTH`, ... | `ValueError` at construction |
| `FREQ_H <= FREQ_L`, unknown `TYPE_OF_VAD` / `WHERE_TO_HIDE`, `HANKEL_ROWS < 2`, `LEVELS < 1` (multi-level), `LEGACY` for a method without a legacy rule, empty `LABEL` | `ValueError` at construction |
| hop != frame (DCT methods), frame != 2 x hop (Wen) | `ValueError` at construction |
| band problems (see [Frequency band](#frequency-band)), non-integer frame size, carrier shorter than a frame, no carrying frame | `ValueError` in `set_carrier` |
| secret larger than the capacity / empty | `ValueError` in `set_secret` |
| embedding without carrier or secret, extraction without key | `RuntimeError` |
| stegobject with a different sampling rate, key with another band / decomposition, PC v3 key without levels | `ValueError` in `extraction` |
