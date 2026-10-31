# Methods

All methods hide **one bit per voiced frame**. The embedding pipeline, the parameter
validation and the extraction loop are shared (`lib/Method.py`); a method version only
defines how the two largest singular values are changed.

Notation: `σe` - largest singular value of the matrix built from the *even* coefficients
of the band, `σo` - the same for the *odd* coefficients, `davg = (σe + σo) / 2`,
`α = ALPHA`, `β = BETA`. Extraction is the same for every method:

```
bit = 0 if σe > σo else 1
```

## Common pipeline

1. **Framing.** Frames of `FRAME_LENGTH` ms every `HOP_DURATION_MS` ms, only full frames;
   the tail that does not fill a frame is copied unchanged. The duration must correspond
   to an integer number of samples (e.g. 12.5 ms at 16 kHz = 200 samples; 10 ms at
   22.05 kHz is rejected).
   * DCT methods concatenate frames without a synthesis window, so
     `HOP_DURATION_MS == FRAME_LENGTH` is required.
   * Wen uses the MDCT with a Kaiser-Bessel-derived window, `FRAME_LENGTH = 2 x HOP`.
2. **VAD.** `rVAD` (default) or `webrtc` (10/20/30 ms frames, 8/16/32/48 kHz) on the
   carrier. Bits go into voiced frames in order.
3. **Band.** DCT bin `k` of an `N`-sample frame is `k * fs / (2N)` Hz; MDCT bin `k` of a
   `2N`-sample frame is centred at `(k + 0.5) * fs / (2N)` Hz. The band must contain
   enough coefficients (Xue / Wen: 2, multi-level methods: `2 * (HANKEL_ROWS + 1)`).
4. **Embedding** in the frame, inverse transform, overlap-add.
5. **Quantisation.** The stegobject is rounded to the nearest integer and clipped to int16
   (`CLIPPED` column); NaN/inf raise an error.
6. **Extraction key** (`Keys/*.key.json`): voiced mask of the carrier, number of bits,
   frame/hop, sampling rate, and for PC v3 the level of every bit.

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
source of the remaining bit errors of PC v3.

## Versions

### Xue et al. - `DIFFERENTIAL_SVD_XUE`

Even/odd coefficients are reshaped to near-square matrices (zero padding if needed;
the padded entries are dropped after the inverse SVD).

```
bit 0: if σe - σo < α σo:  σe' = σo + α σo
bit 1: if σo - σe < β σe:  σo' = σe + β σe
```

`DIFFERENTIAL_SVD_XUE_MULTI` applies the same rule on the deepest level of the
multi-level Hankel SVD.

### Wen et al. - `MDCT_SVD_WEN`

MDCT of the windowed frame, band, even/odd matrices as in Xue, `s_min = min(σe, σo)`,
`s_avg = davg`:

```
bit 0: if σe - σo < α s_min:  σe' = s_avg + α s_min / 2,  σo' = s_avg - α s_min / 2
bit 1: if σo - σe < α s_min:  σe' = s_avg - α s_min / 2,  σo' = s_avg + α s_min / 2
```

`BETA` is not used. Frames without a bit pass through the analysis and synthesis window
(`w^2`), so the overlap-add reproduces the carrier; only the first and the last half
frame of the file are attenuated (no neighbour to complete the window).

### PC Xue [v1] - `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1`

```
T1 = α + (σo - σe) / davg   if ADAPTIVE else α
T2 = β + (σe - σo) / davg                       (adaptive also when ADAPTIVE = False)
bit 0: if σe - σo < T1 davg:  σe' = σo + T1 davg
bit 1: if σo - σe < T2 davg:  σo' = σe + T2 davg
```

`REDUCED = True`: bit 0 - threshold `α davg`, `σe' = σo + α davg`; bit 1 - threshold
`β davg / 2`, `σo' = σe + β davg`.

### PC Xue [v2] - `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2`

Both thresholds are always adaptive (the `ADAPTIVE` flag has no effect).

```
bit 0: if σe - σo < T1 davg:  σe' = σe + T1 davg
bit 1: if σo - σe < T2 davg:  σo' = σe + T2 davg
```

`REDUCED = True`: as v1, but the bit 0 threshold is `α davg / 2`.

### PC [v1] - `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1`

Symmetric shift relative to the current values (`T1`, `T2` as in PC Xue v1):

```
bit 0: if σe - σo < T1 davg:  σe' = σe + T1 davg / 2,  σo' = σo - T1 davg / 2
bit 1: if σo - σe < T2 davg:  σe' = σe - T2 davg / 2,  σo' = σo + T2 davg / 2
```

With adaptive thresholds the new difference is exactly `α davg` (bit 0) / `β davg`
(bit 1) and `σ' = davg (1 ± α/2) > 0` for `α < 2`. With `ADAPTIVE = False` the bit 0
shift is fixed: if `σo - σe > α davg` the sign is not reached and the bit is lost
**by design** (about 3% of the bits on CMU ARCTIC at 50 bps, growing with capacity).

### PC [v2] - `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2` (Algorithm 4)

Both values are reset to the mean and moved apart; thresholds always adaptive:

```
bit 0: if σe - σo < T1 davg:  σe' = davg + T1 davg / 2,  σo' = davg - T1 davg / 2
bit 1: if σo - σe < T2 davg:  σe' = davg - T2 davg / 2,  σo' = davg + T2 davg / 2
```

### PC [v3] - `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3`

Adaptive multi-level selection: the adaptive PC v1 rule is tried at every depth
`1..LEVELS` and the depth with the highest

```
score = (σ' margin in the direction of the bit) / mean |embedded frame - frame|
```

is used (ties: the shallowest). The thresholds are always adaptive (`ADAPTIVE` is not
used). `PC [v3] baseline` decomposes every candidate depth from scratch,
`PC [v3]` / `PC [v3] optimized` decompose once and use prefixes - all three give
bit-identical stegobjects; the variants exist to compare embedding times.

## Side information (non-blind extraction)

The extractor receives the extraction key, i.e. information that a real receiver would
have to obtain separately:

* the **voiced mask of the carrier** - running the VAD on the stegobject (or on an
  attacked copy) may give a different mask;
* the **number of embedded bits**;
* for PC v3 the **depth of every bit**: `ceil(log2(LEVELS))` bits per payload bit
  (3 bits per bit for `LEVELS = 8`), i.e. more side information than payload.

This should be stated when the methods are compared with blind schemes.

## Other known limitations

* **Secret statistics.** The secret is ASCII text (letters and spaces), so the most
  significant bit of every byte is 0 and the bits are not balanced. Methods whose error
  probability differs for 0 and 1 get a BER that depends on this distribution.
* **Hankel re-projection** (see above) - the margin chosen during embedding is not
  exactly the margin measured during extraction.
* **Matrix padding** (Xue, Wen) - for band sizes that are not a product of two close
  integers, the padded entries are dropped after the modification, which slightly
  changes the singular values.

## Validation (raised before embedding)

| Check | Error |
|---|---|
| missing / non-numeric / non-positive `ALPHA`, `BETA`, `FRAME_LENGTH`, ... | `ValueError` at construction |
| `FREQ_H <= FREQ_L`, unknown `TYPE_OF_VAD` / `WHERE_TO_HIDE`, `HANKEL_ROWS < 2`, `LEVELS < 1` (multi-level) | `ValueError` at construction |
| hop != frame (DCT methods), frame != 2 x hop (Wen) | `ValueError` at construction |
| `FREQ_H` above Nyquist, band too narrow, non-integer frame size, carrier shorter than a frame, no voiced frame | `ValueError` in `set_carrier` |
| secret larger than the capacity / empty | `ValueError` in `set_secret` |
| embedding without carrier or secret, extraction without key | `RuntimeError` |
| stegobject with a different sampling rate, PC v3 key without levels | `ValueError` in `extraction` |
