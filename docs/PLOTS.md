# Summary charts ("metric vs Capacity")

`lib/Plots.py` draws the comparison charts of the thesis: one line per method, x = the
nominal capacity `CAP. [bps]`, y = the mean over carriers.

| File | Title | Metric column |
|---|---|---|
| `ber_vs_capacity.png` | Bit Error Rate (BER) vs Capacity | `BER [%]` |
| `emb_time_s_vs_capacity.png` | Average Embedding Time vs Capacity | `EMB TIME [s]` |
| `ext_time_s_vs_capacity.png` | Average Extraction Time vs Capacity | `EXT TIME [s]` |
| `mse_vs_capacity.png` | Mean squared error (MSE) vs Capacity | `MSE` |
| `pesq_vs_capacity.png` | PESQ vs Capacity | `PESQ` |
| `psnr_db_…`, `snr_db_…`, `segsnr_db_…`, `stoi_…` | PSNR / SNR / SegSNR / STOI vs Capacity | |
| `Attacks/ber_<attack>_vs_capacity.png` | BER vs Capacity (<attack>) | `BER [%]`, one chart per attack |

Only the clean stegobjects (`TEST == NONE`) are used for the main charts. Every PNG has a
CSV next to it with `mean`, `std`, `n` and `ci95` per method and capacity.

## Producing the data

```bash
python coder.py --test CAPACITY_SWEEP --seed 1          # 25, 31.25, 50, 62.5, 80, 100 bps
```

or several separate runs, combined afterwards:

```bash
python coder.py --plots-from ~/Results/run_25bps ~/Results/run_50bps ~/Results/run_100bps \
                --out ~/Results/Plots
```

`--plots-from` accepts CSV files, glob patterns and directories (searched recursively
for `test_analysis_*.csv`, grouped files are ignored).

## Style

seaborn `whitegrid`, dashed grid, 10 × 6 in at 200 dpi, bold y label, legend titled
"Metoda / Method". Colours and markers follow the method, not its position, so a method
keeps its colour in every chart and when some methods are missing. The seven thesis
methods get exactly the colours of the original figures (`husl` palette, order
PC Xue [v1], PC Xue [v2], PC [v1], PC [v2], PC [v3], Wen Et. Al., Xue et. al.); other
methods get additional colours and markers. Variants named `<method> (orig.)` (BASELINES
preset) and `<method> (legacy)` keep the colour and marker of `<method>` with a dashed /
dotted line and hollow markers, and are listed right after it in the legend:

```bash
python coder.py --plots-from ~/Results/<CAPACITY_SWEEP run> ~/Results/<BASELINES run> --out Plots
```

With more than 9 lines (`MAX_LEGEND_INSIDE`) the legend is placed outside the axes so
that no data is hidden; up to 9 lines it stays inside, as in the thesis figures.

Options:

* `--palette accessible` - colour-blind-safe palette. The `husl` palette fails a
  colour-vision-deficiency check (PC [v1] / PC Xue [v2] and PC [v2] / PC [v3] are hard to
  tell apart); the markers keep the lines distinguishable, but for print or presentation
  the accessible palette is safer.
* `--ci` - shaded 95% confidence band (`1.96 · std / sqrt(n)`).
* `--metrics "BER [%]" PESQ` - only selected charts.

From Python:

```python
from lib.Plots import load_results, plot_capacity_charts, plot_metric_vs_capacity
df = load_results(['~/Results/run1', '~/Results/run2'])
plot_capacity_charts(df, 'Plots', palette='husl', ci=False)
plot_metric_vs_capacity(df, 'BER [%]', 'Plots', test='NOISED', logy=True)
```

Per-stegobject diagnostic charts (waveforms, spectrograms, singular values, adaptive
α/β) are produced by `Analysis` when `PER_FILE_PLOTS` is on (`--no-file-plots` to skip).
