"""Capacity charts."""

import os

import numpy as np
import pandas as pd
import seaborn as sns

from lib import Plots


def results_frame():
    rows = []
    rng = np.random.default_rng(0)
    for name in Plots.CANONICAL_METHOD_ORDER + ['Xue Et. al. [multi]']:
        for bps in (25, 50, 100):
            for carrier in ('a', 'b'):
                for test in ('NONE', 'NOISED'):
                    rows.append({'NAME': name, 'CAP. [bps]': bps, 'CARRIER': carrier, 'TEST': test,
                                 'BER [%]': rng.random(), 'MSE': rng.random() * 1000, 'PESQ': 4 + rng.random() / 2,
                                 'EMB TIME [s]': rng.random(), 'EXT TIME [s]': rng.random()})
    return pd.DataFrame(rows)


def test_capacity_charts_are_written(tmp_path):
    written = Plots.plot_capacity_charts(results_frame(), str(tmp_path), ci=True)
    names = sorted(os.path.basename(p) for p in written)
    assert 'ber_vs_capacity.png' in names and 'pesq_vs_capacity.png' in names
    assert 'ber_noised_vs_capacity.png' in names  # attack chart
    table = pd.read_csv(tmp_path / 'ber_vs_capacity.csv')
    assert set(table.columns) >= {'NAME', 'CAP. [bps]', 'mean', 'std', 'n', 'ci95'}
    assert (table['n'] == 2).all()


def test_aggregate_uses_only_requested_test():
    df = results_frame()
    table = Plots.aggregate(df, 'BER [%]', test='NONE')
    expected = df[(df['TEST'] == 'NONE') & (df['NAME'] == 'PC [v1]') & (df['CAP. [bps]'] == 50)]['BER [%]'].mean()
    row = table[(table['NAME'] == 'PC [v1]') & (table['CAP. [bps]'] == 50)]
    assert row['mean'].iloc[0] == expected


def test_colours_follow_the_method_not_its_rank():
    full, _ = Plots.method_styles(Plots.CANONICAL_METHOD_ORDER)
    subset, order = Plots.method_styles(['Xue et. al.', 'PC [v3]'])
    assert order == ['PC [v3]', 'Xue et. al.']
    assert subset['Xue et. al.'] == full['Xue et. al.']
    # husl colours of the thesis figures
    assert full['PC Xue [v1]']['color'] == sns.color_palette('husl', 7).as_hex()[0]


def test_load_results_reads_directories(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    results_frame().to_csv(run / 'test_analysis_1.csv', index=False)
    results_frame().to_csv(run / 'test_analysis_group_1.csv', index=False)  # must be ignored
    df = Plots.load_results(str(tmp_path))
    assert len(df) == len(results_frame())
