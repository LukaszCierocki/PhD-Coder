# Tests

```bash
python -m pytest                 # whole suite, ~1 min
python -m pytest test_suite/test_methods.py -k PC_V3 -v
```

The suite does not need `Carriers/`: it generates speech-like signals (harmonic
segments with a varying f0 separated by quiet noise) that rVAD classifies as voiced.

| File | What is checked |
|---|---|
| `test_signal_processing.py` | vectorised diagonal averaging and Hankel matrices equal the original loops; multi-level decomposition reconstructs exactly (1-8 levels, 2-3 rows); MDCT/KBD overlap-add is TDAC-orthogonal; MDCT band selection hits the requested frequency; integer frame sizes; int16 rounding / clipping / NaN; VAD framing = extractor framing |
| `test_io.py` | WAV round trip is bit-exact; float WAV scaling; secrets reproducible, identical to the old implementation for the same seed, no effect on the global `random` state |
| `test_methods.py` | for **every method**: valid int16 stegobject of the carrier length; samples outside embedding frames unchanged; BER within the expected bound; identical bits in memory, after a WAV round trip and for a fresh receiver with only the JSON key; one object reused for several carriers (the PC v3 level-map bug); determinism; PC v3 variants give identical stegobjects; truncated stegobjects report missing bits; Xue rule reaches the threshold; Wen uses the requested band |
| `test_validation.py` | invalid parameters and misuse raise errors (see docs/METHODS.md) |
| `test_analysis.py` | MSE without int16 overflow, SNR/PSNR/SegSNR values, BER with missing/extra bits, attack labels from file names |
| `test_attacks.py` | NOISE attack reaches the target SNR (±0.5 dB) and is reproducible with a seed; other attacks keep length and rate; G.711; a failing attack (no ffmpeg) does not stop the others |
| `test_plots.py` | charts and CSV tables written; aggregation per test; colours follow the method |
| `test_steganalysis.py` | BH-FDR values, permutation test, q ≥ p, feature dimensions, grouped folds never split a carrier, CV metrics on separable data, ML pipeline + prediction, pair reconstruction from a result folder |
| `test_coder.py` | full runs of `coder.py` (embedding, NOISE attack, extraction, CSV, charts, keys), `--config` replay gives identical results, replay of an old-format config, `ONLY_EMBED` + `--steganalysis-from`, `--plots-from` |

Expected BER bounds on the synthetic signal are listed in `METHOD_CASES`
(`test_suite/conftest.py`): 0% for all methods except PC [v1] with `ADAPTIVE = False`
(loses bits by design) and PC [v3] (Hankel re-projection, ≤ 3%).

The folder is called `test_suite/` because `.gitignore` ignores `Tests/*` and Windows /
macOS file systems are case-insensitive.
