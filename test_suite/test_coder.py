"""Integration: a full run of coder.py on synthetic carriers and its replay."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import coder
from lib.Carrier import read_audio_as_int16

from conftest import synthetic_speech, write_wav


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    carriers = tmp_path / 'Carriers'
    carriers.mkdir()
    for i in range(2):
        write_wav(str(carriers / f'spk_{i}.wav'), synthetic_speech(seed=i, seconds=1.5))
    monkeypatch.setitem(coder.PRESETS, 'TINY', lambda logger: {
        'methods': [coder.create_method('DIFFERENTIAL_SVD_XUE', coder.params(20), logger),
                    coder.create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3',
                                        coder.params(20, ADAPTIVE=True, LEVELS=3), logger),
                    coder.create_method('MDCT_SVD_WEN', coder.params(20, MDCT=True), logger)],
        'attacks': dict(coder.no_attacks(), NOISE=True),
        'settings': {'PER_FILE_PLOTS': False},
    })
    return str(carriers), str(tmp_path / 'Results')


def test_full_run_and_replay(workspace):
    carriers_dir, results_root = workspace
    settings = {'TEST': 'TINY', 'SEED': 3}

    first = coder.run_and_save_test(settings, results_root=results_root, carriers_dir=carriers_dir)
    assert first['failures'] == []
    df = pd.read_csv(first['csv'])
    assert len(df) == 3 * 2 * 2  # methods x carriers x (clean + noised)
    clean = df[df['TEST'] == 'NONE']
    assert (clean[clean['NAME'] != 'PC [v3]']['BER [%]'] == 0).all()
    assert set(df['TEST']) == {'NONE', 'NOISED'}
    assert any(p.endswith('ber_vs_capacity.png') for p in first['charts'])

    with open(first['config']) as f:
        config = json.load(f)
    assert len(config['performed_tests']) == 6
    key_path = os.path.join(first['test_dir'], config['performed_tests'][0]['key'])
    assert os.path.isfile(key_path)

    second = coder.run_test_from_config(first['config'], results_root=results_root, carriers_dir=carriers_dir)
    df2 = pd.read_csv(second['csv'])
    columns = ['NAME', 'CARRIER', 'TEST', 'SECRET', 'BER [%]', 'MSE']
    a = df.sort_values(['NAME', 'CARRIER', 'TEST'])[columns].reset_index(drop=True)
    b = df2.sort_values(['NAME', 'CARRIER', 'TEST'])[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # same secrets, same stegobjects, same seeded noise


def test_replay_of_old_config_format(workspace, tmp_path):
    carriers_dir, results_root = workspace
    old = {
        "attacks": {'NOISE': False, 'HIGHPASS': False, 'LOWPASS': False, 'REQUANTIZE': False, 'MP3': False,
                    'JITTER': False, 'RESAMPLE': False, 'CLIPPED': False},
        "performed_tests": [{
            "method": {"name": "MDCT_SVD_WEN",
                       "params": {'ADAPTIVE': False, 'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD': 'rVAD',
                                  'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 40,
                                  'HOP_DURATION_MS': 20, 'LEVELS': 1, 'HANKEL_ROWS': 2}},
            "carrier": "spk_0", "secret": {"seed": 42, "type": "MAX_CAP"}, "test_name": "x"}],
    }
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(old))
    result = coder.run_test_from_config(str(path), results_root=results_root, carriers_dir=carriers_dir)
    assert result['failures'] == []
    assert pd.read_csv(result['csv'])['BER [%]'].iloc[0] == 0


def test_only_embed_and_steganalysis_from_dir(workspace, tmp_path):
    carriers_dir, results_root = workspace
    result = coder.run_and_save_test({'TEST': 'TINY', 'ONLY_EMBED': True, 'SEED': 1},
                                     results_root=results_root, carriers_dir=carriers_dir)
    pairs = json.load(open(result['pairs']))
    assert len(pairs) == 6
    for pair in pairs:
        stego, _ = read_audio_as_int16(pair['stego_path'])
        cover, _ = read_audio_as_int16(pair['carrier_path'])
        assert len(stego) == len(cover) and not np.array_equal(stego, cover)

    code = coder.main(['--steganalysis-from', result['test_dir'], '--carriers-dir', carriers_dir,
                       '--perm-b', '5', '--out', str(tmp_path / 'sa')])
    assert code == 0
    assert (tmp_path / 'sa' / 'Statistical' / 'results_stratified.csv').exists()


def test_cli_plots_from(workspace, tmp_path):
    carriers_dir, results_root = workspace
    result = coder.run_and_save_test({'TEST': 'TINY', 'SEED': 2, 'SUMMARY_PLOTS': False},
                                     results_root=results_root, carriers_dir=carriers_dir)
    assert 'charts' not in result
    out = tmp_path / 'charts'
    assert coder.main(['--plots-from', result['test_dir'], '--out', str(out), '--metrics', 'BER [%]', 'PESQ']) == 0
    assert (out / 'ber_vs_capacity.png').exists() and (out / 'pesq_vs_capacity.png').exists()


def test_legacy_and_attack_flags(workspace):
    carriers_dir, results_root = workspace
    settings = coder.settings_from_args(coder.parse_args(['--test', 'TINY', '--legacy', '--attacks', 'none',
                                                          '--seed', '4']))
    result = coder.run_and_save_test(settings, results_root=results_root, carriers_dir=carriers_dir)
    df = pd.read_csv(result['csv'])
    assert set(df['TEST']) == {'NONE'}                                 # NOISE of the preset switched off
    assert set(df['NAME']) == {'Xue et. al. (legacy)', 'PC [v3] (legacy)', 'Wen Et. Al.'}
    assert df.set_index('NAME').loc['Wen Et. Al.', 'LEGACY'].all() == False   # no legacy variant for Wen
    with pytest.raises(SystemExit):
        coder.parse_attacks(['NOPE'])


def test_baselines_preset_uses_published_parameters():
    import logging
    methods = coder.PRESETS['BASELINES'](logging.getLogger('x'))['methods']
    assert len(methods) == 2 * len(coder.CAPACITY_SWEEP_BPS)
    xue = [m for m in methods if m.__class__.__name__ == 'DIFFERENTIAL_SVD_XUE']
    wen = [m for m in methods if m.__class__.__name__ == 'MDCT_SVD_WEN']
    assert {(m.alpha, m.where_to_hide, m.normal_name) for m in xue} == {(0.2, 'ALL', 'Xue et. al. (orig.)')}
    assert {(m.alpha, m.FREQ_L, m.FREQ_H, m.normal_name) for m in wen} == {(0.61, 100, 3000, 'Wen Et. Al. (orig.)')}
