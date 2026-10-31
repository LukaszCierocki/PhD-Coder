# PhD Coder: Audio Steganography Toolkit

PhD Coder is a toolkit for experimenting with, testing and comparing steganographic
methods for speech signals. One run embeds a pseudo-random secret into every carrier
with every configured method, optionally attacks the stegobjects, extracts the secret
again and reports transparency (MSE, PSNR, SNR, SegSNR, PESQ, STOI), robustness (BER),
capacity and speed. It can also run statistical and machine-learning steganalysis on the
stegobjects and draw the "*metric* vs Capacity" charts used in the thesis.

## Contents

- [Quick start](#quick-start)
- [What a run does](#what-a-run-does)
- [Presets and settings](#presets-and-settings)
- [Command line](#command-line)
- [Output](#output)
- [Methods](#methods)
- [Adding a method](#adding-a-method)
- [Tests](#tests)
- [Directory structure](#directory-structure)
- Detailed documentation: [docs/METHODS.md](docs/METHODS.md),
  [docs/STEGANALYSIS.md](docs/STEGANALYSIS.md), [docs/PLOTS.md](docs/PLOTS.md),
  [docs/TESTING.md](docs/TESTING.md), [CHANGELOG.md](CHANGELOG.md)

## Quick start

```bash
pip install -r requirements.txt
```

Put the carrier files (mono speech, 16-bit PCM WAV, e.g. CMU ARCTIC at 16 kHz) into
`Carriers/` (the WAV files are git-ignored), then:

```bash
python coder.py                                  # DEBUG preset (Xue et al., 50 bps)
python coder.py --test CAPACITY_SWEEP --seed 1   # all methods, 25-100 bps + charts
python coder.py --test FOR_CLASSIFIER --seed 1   # all methods + steganalysis
```

Requirements: Python 3.9+ (tested with 3.13). `ffmpeg` in `PATH` is needed only for
the MP3 attack. On Python >= 3.13 the G.711 attack uses `audioop-lts` (installed by
`requirements.txt`). If `pip` fails to build `pesq` or `BitVector`, use
`pip install --use-pep517 pesq BitVector`.

## What a run does

For every (method, carrier) pair:

1. **VAD + framing** - the carrier is split into frames (`FRAME_LENGTH`,
   `HOP_DURATION_MS`); rVAD (or WebRTC VAD) marks the voiced frames.
2. **Secret** - pseudo-random ASCII text that fills the capacity (one bit per voiced
   frame, rounded down to whole bytes). With `--seed` the secrets are reproducible.
3. **Embedding** - one bit per voiced frame. The stegobject is rounded and clipped to
   int16 (clipped samples are counted and reported).
4. **Extraction key** - everything the receiver needs besides the stegobject: the voiced
   mask, the number of bits and method-specific side information (PC v3: the
   decomposition level of every bit). It is saved as `Keys/<stegobject>.key.json`.
5. **Attacks** (optional) - noise, filters, requantisation, MP3, jitter, resampling,
   scaling, packet loss, G.711.
6. **Extraction + analysis** of the clean and every attacked stegobject.

Afterwards: CSV files, a grouped summary table in the console, the run configuration
(JSON, can be replayed), summary charts and - if enabled - steganalysis.

Every parameter is validated before anything is embedded (frequency band vs Nyquist,
hop vs frame length, band wide enough for the method, integer number of samples per
frame, secret not larger than the capacity, ...). A failing (method, carrier) pair is
logged and listed under `failures` in the configuration file; the run continues.

## Presets and settings

Presets are defined in `coder.py` (`PRESETS`):

| Preset | Methods | Attacks | Extras |
|---|---|---|---|
| `DEBUG` | Xue et al., 50 bps | none | per-file charts |
| `FOR_CLASSIFIER` | all thesis methods, 50 bps | none | steganalysis |
| `CAPACITY_SWEEP` | all thesis methods at 25, 31.25, 50, 62.5, 80, 100 bps | none | capacity charts |
| `ROBUSTNESS` | all thesis methods, 50 bps | all | BER chart per attack |

The thesis method set (`THESIS_METHODS`) is: Wen et al. (MDCT), Xue et al., PC Xue v1,
PC Xue v2, PC v1, PC v2, PC v3 (8 levels); band 1500-2500 Hz, ALPHA = BETA = 0.7, rVAD.
The nominal capacity is one bit per hop: `bps = 1000 / HOP_DURATION_MS` (Wen uses MDCT
frames twice as long as the hop).

Default settings are in `RUN_SETTINGS` at the top of `coder.py`; command-line options
override them, and a preset's own defaults apply unless the option is given explicitly.

## Command line

```bash
python coder.py --help

# experiments
python coder.py --test ROBUSTNESS --carriers 20 --seed 7 --no-file-plots
python coder.py --test FOR_CLASSIFIER --only-embed --steganalysis --perm-b 500
python coder.py --config ~/Results/<run>/test_config_<time>.json      # replay a run

# charts from one or more finished runs (e.g. several capacity runs)
python coder.py --plots-from ~/Results/<run1> ~/Results/<run2> --out ~/Results/Plots
python coder.py --plots-from ~/Results/<run> --metrics "BER [%]" PESQ --palette accessible --ci

# steganalysis of an existing result folder, prediction for one file
python coder.py --steganalysis-from ~/Results/<run> --perm-b 200
python coder.py --predict file.wav --models ~/Results/<run>/Steganalysis/ML/models

python coder.py --list-methods
```

Results go to `$PHD_CODER_RESULTS` or `~/Results` (`--results-root` overrides). Previous
runs in that folder are moved to `Archive/` when a new run starts.

## Output

```
~/Results/<date>_<id>/
├── test_analysis_<time>.csv         # one row per (method, carrier, test)
├── test_analysis_group_<time>.csv   # means grouped by method / test / parameters
├── test_config_<time>.json          # settings, parameters, secret seeds, keys, failures
├── test_log_<time>.log
├── steganalysis_pairs.json          # carrier / clean stegobject pairs
├── Plots/                           # "<metric> vs Capacity" PNG + CSV, Attacks/ per attack
├── Steganalysis/                    # Statistical/ and ML/ (see docs/STEGANALYSIS.md)
├── Carriers/Charts/                 # carrier waveforms and spectrograms
└── <method name>/
    ├── Stegobjects/                 # clean and attacked stegobjects (16-bit WAV)
    ├── Keys/                        # extraction keys (JSON)
    ├── Secrets/  Extracted/         # embedded and extracted text
    └── Charts/                      # per-stegobject charts (PER_FILE_PLOTS)
```

Main CSV columns: `NAME`, `CARRIER`, `TEST` (`NONE` = clean stegobject, otherwise the
attack), `SEC. SIZE [b]`, `EXTRACTED [b]`, `BIT ERRORS`, `BER [%]` (missing bits count as
errors), `MSE`, `PSNR [dB]` (peak = max |carrier|), `SNR [dB]`, `SEGSNR [dB]` (20 ms
segments, clamped to [-10, 35] dB), `PESQ` (wide-band at 16 kHz, narrow-band at 8 kHz),
`STOI` (extended), `CAP. [bps]` (nominal), `EFF. CAP. [bps]` (embedded bits / duration),
`EMB TIME [s]`, `EXT TIME [s]`, `CLIPPED`.

## Methods

| Name in results | Class | Idea |
|---|---|---|
| Xue et. al. | `DIFFERENTIAL_SVD_XUE` | DCT band, even/odd coefficients as matrices, differential SVD rule |
| Xue Et. al. [multi] | `DIFFERENTIAL_SVD_XUE_MULTI` | Xue rule on the deepest level of a multi-level Hankel SVD |
| Wen Et. Al. | `MDCT_SVD_WEN` | MDCT (50% overlap, KBD window) + SVD |
| PC Xue [v1], [v2] | `DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1/V2` | Xue-style rule on the multi-level Hankel SVD, thresholds scaled by the mean singular value |
| PC [v1], [v2] | `..._MULTI_PC_V1/V2` | symmetric shift / reset-to-mean rules, adaptive thresholds |
| PC [v3] | `..._MULTI_PC_V3` | adaptive selection of the decomposition depth per frame |
| PC [v3] baseline / optimized | `..._PC_V3_BASELINE/OPTIMIZED` | same stegobject as PC [v3]; unoptimised / optimised search (timing comparison) |

Parameters: `ALPHA`, `BETA` (strength), `FREQ_L`, `FREQ_H` (band in Hz), `FRAME_LENGTH`,
`HOP_DURATION_MS` (ms), `LEVELS`, `HANKEL_ROWS`, `ADAPTIVE`, `REDUCED`, `TYPE_OF_VAD`
(`rVAD` / `webrtc`), `WHERE_TO_HIDE` (`VOICED`). Embedding rules, side information and
known limitations of every version are described in [docs/METHODS.md](docs/METHODS.md).

## Adding a method

1. Create `Methods/<NAME>.py` with a class `<NAME>`. For a DCT + multi-level SVD method
   derive from `lib.Method.DCT_MULTI_LEVEL_SVD_METHOD` and implement only
   `modify_singular_values(ro_11, ro_21, bit)`; otherwise derive from
   `DIFFERENTIAL_SVD_XUE_BASED_METHOD` and implement `embedd_in_frame` and
   `extract_from_frame`. Set `self.name`, `self.normal_name` and `calculate_capacity`.
2. If the receiver needs per-bit side information, extend `build_extraction_key` and
   `extract_bit` (see PC v3).
3. Register the class in `lib/Registry.py` and add it to a preset in `coder.py`.
4. Add a case to `METHOD_CASES` in `test_suite/conftest.py` - the generic tests then
   check round trips, untouched frames, key-based extraction and carrier reuse.

## Tests

```bash
python -m pytest            # ~1 minute, no carriers needed (synthetic speech)
```

See [docs/TESTING.md](docs/TESTING.md).

## Directory structure

```
PhD-Coder/
├── coder.py              # test runner and command line
├── Methods/              # steganographic methods
├── lib/
│   ├── Method.py         # base classes, framing + VAD, Hankel/SVD, MDCT, validation
│   ├── Registry.py       # method name -> class (lazy imports)
│   ├── Analysis.py       # metrics, CSV export, per-file charts
│   ├── Attacks.py        # robustness attacks
│   ├── Plots.py          # "<metric> vs Capacity" charts
│   ├── Steganalysis.py   # statistical and ML steganalysis
│   ├── Tools.py Carrier.py Secret.py Logger.py
│   └── speechproc.py     # rVAD
├── scripts/              # stand-alone figures for the thesis
├── test_suite/           # pytest suite
├── docs/                 # documentation
└── Carriers/             # carrier WAV files (git-ignored)
```

## Contact

This project was developed as part of a PhD thesis. For questions or collaboration,
please contact the author.
