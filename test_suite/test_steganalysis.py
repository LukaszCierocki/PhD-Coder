"""Steganalysis: statistics, FDR, grouped cross-validation, features, pairs."""

import os

import numpy as np
import pytest

from lib import Steganalysis as S

from conftest import synthetic_speech, write_wav


def test_fdr_bh_known_values():
    q = S.fdr_bh([0.01, 0.04, 0.03, 0.2])
    # p sorted: 0.01, 0.03, 0.04, 0.2 -> raw q: 0.04, 0.06, 0.0533, 0.2 -> monotone: 0.04, 0.0533, 0.0533, 0.2
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert S.fdr_bh([]) == []
    assert S.fdr_bh([None, float('nan')]) == [1.0, 1.0]


def test_permutation_test_separates_distributions():
    rng = np.random.default_rng(0)
    same_a, same_b = rng.normal(size=2000), rng.normal(size=2000)
    edges = S.stable_edges(-5, 5, 50)
    _, p_same = S.perm_p_divergence(same_a, same_b, edges, 99, rng, 1e-12, 'JS')
    _, p_diff = S.perm_p_divergence(same_a, same_b + 1.0, edges, 99, rng, 1e-12, 'JS')
    assert p_same > 0.05
    assert p_diff == pytest.approx(1 / 100)


def test_divergences():
    p = np.array([0.5, 0.5])
    assert S.kl_divergence(p, p) == 0
    assert S.js_divergence(p, np.array([0.9, 0.1])) > 0


def test_chi2_homogeneity_is_the_contingency_table_test():
    from scipy.stats import chi2_contingency
    a, b = np.array([30, 50, 20]), np.array([25, 45, 40])
    stat, p = S.chi2_homogeneity(a, b)
    ref_stat, ref_p, _, _ = chi2_contingency(np.array([a, b]), correction=False)
    assert stat == pytest.approx(ref_stat)
    assert p == pytest.approx(ref_p)
    # sparse bins are merged; nothing left to compare -> no evidence
    assert S.chi2_homogeneity([3, 0], [2, 1]) == (0.0, 1.0)
    with pytest.raises(ValueError):
        S.chi2_homogeneity([1, 2], [1, 2, 3])


def test_chi2_ignores_a_sample_beyond_the_cover_peak():
    # Regression: an empty cover bin had the expected count 1e-12, so one stego sample
    # above the cover's maximum gave chi2 ~ 1e13 and p = 0 - every method was
    # "detected" on every carrier.
    rng = np.random.default_rng(1)
    cover = np.clip(rng.normal(0, 3000, 64000), -32768, 32767).astype(np.int16)
    stego = cover.copy()
    stego[np.argmax(cover)] = int(cover.max()) + 500
    rows = S.statistical_tests_for_pair(cover, stego, 16000, S.merge_config({'PERM_B': 9}),
                                        np.random.default_rng(0), 'pair')
    chi2 = [r for r in rows if r['domain'] == 'Amplitude' and r['metric'] == 'chi2'][0]
    assert chi2['p'] > 0.5
    lsb = [r for r in rows if r['domain'] == 'LSB'][0]
    assert lsb['p'] > 0.5


def test_chi2_detects_real_changes():
    rng = np.random.default_rng(2)
    cover = np.clip(rng.normal(0, 3000, 64000), -32768, 32767).astype(np.int16)
    louder = np.clip(cover.astype(np.float64) * 1.2, -32768, 32767).astype(np.int16)
    edges = S.stable_edges(-20000, 20000, 256)
    _, p = S.chi2_homogeneity(np.histogram(cover, edges)[0], np.histogram(louder, edges)[0])
    assert p < 1e-6
    # LSB plane: all LSBs cleared vs. untouched
    assert S.chi2_homogeneity(S.lsb_counts(cover), S.lsb_counts(cover & ~1))[1] < 1e-6
    assert S.chi2_homogeneity(S.lsb_counts(cover), S.lsb_counts(cover))[1] == pytest.approx(1.0)


@pytest.fixture
def pairs(tmp_path):
    """4 carriers, stego = carrier + small noise in a band (detectable by statistics)."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(4):
        cover = synthetic_speech(seed=i, seconds=1.0)
        stego = np.clip(cover.astype(int) + rng.integers(-40, 41, len(cover)), -32768, 32767)
        c = write_wav(str(tmp_path / f'c{i}.wav'), cover)
        s = write_wav(str(tmp_path / f's{i}.wav'), stego)
        out.append({'method': 'M', 'label': 'Method', 'carrier': f'c{i}', 'carrier_path': c, 'stego_path': s})
    return out


def test_statistical_steganalysis_writes_q_values(pairs, tmp_path):
    df = S.run_statistical_steganalysis(pairs, str(tmp_path / 'stat'),
                                        {'PERM_B': 9, 'NUM_SAMPLES_VISUAL': 1})
    assert not df.empty
    assert df['q'].between(0, 1).all()
    assert (df['q'] >= df['p'] - 1e-12).all()
    for name in ('results_stratified.csv', 'summary_detection_rate.csv', 'heatmap_ks.png'):
        assert (tmp_path / 'stat' / name).exists()


def test_feature_dimensions(pairs):
    features = S.extract_file_features(pairs[0]['carrier_path'])
    assert {k: len(v) for k, v in features.items()} == {'rme': 80, 'dhs': 4, 'wmc': 78, 'dmc': 26}
    assert all(np.all(np.isfinite(v)) for v in features.values())


def test_grouped_folds_never_split_a_carrier():
    groups = np.repeat(np.arange(10), 2)
    y = np.tile([0, 1], 10)
    for train, test in S.grouped_folds(y, groups, 5, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_detector_on_separable_features():
    rng = np.random.default_rng(0)
    groups = np.repeat([f'c{i}' for i in range(12)], 2)
    y = np.tile([0, 1], 12)
    X = rng.normal(size=(24, 5)) + y[:, None] * 5.0
    metrics = S.evaluate_detector(X, y, groups, 'SVM', S.merge_config(), seed=0)
    assert metrics['accuracy'] == 1.0 and metrics['AUC'] == 1.0 and metrics['P_E'] == 0.0
    assert metrics['TPR_at_FPR_5%'] == 1.0


def test_ml_pipeline_and_prediction(pairs, tmp_path):
    out = str(tmp_path / 'ml')
    table = S.run_ml_steganalysis(pairs, out, {'N_SPLITS': 2, 'MIN_CARRIERS_ML': 4, 'TSNE': False,
                                               'CLASSIFIERS': ['SVM', 'FRBB']})
    assert set(table['feature_set']) == {'RME', 'DHS', 'WMC', 'DMC'}
    assert set(table['classifier']) == {'SVM'} or S.frbb_available()
    assert os.path.exists(os.path.join(out, 'plots', 'roc_SVM_rme.png'))
    report = S.predict_file(pairs[0]['stego_path'], os.path.join(out, 'models'))
    assert report['M']['models'] == 4


def test_too_few_carriers_skip_ml(pairs, tmp_path):
    assert S.run_ml_steganalysis(pairs[:2], str(tmp_path / 'ml')).empty


def test_pairs_from_results_dir(tmp_path):
    carriers = tmp_path / 'Carriers'
    carriers.mkdir()
    write_wav(str(carriers / 'arctic_a0001.wav'), synthetic_speech(seconds=0.3))
    write_wav(str(carriers / 'arctic_a00010.wav'), synthetic_speech(seconds=0.3))
    method = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2_L_1_A_0.7'
    stego_dir = tmp_path / 'run' / method / 'Stegobjects'
    stego_dir.mkdir(parents=True)
    for name in (f'{method}_aB3dE_arctic_a0001', f'noised_30dB_{method}_aB3dE_arctic_a0001',
                 f'{method}_xY12z_arctic_a00010', 'arctic_a0001'):
        write_wav(str(stego_dir / f'{name}.wav'), synthetic_speech(seconds=0.3))
    pairs = S.pairs_from_results_dir(str(tmp_path / 'run'), str(carriers))
    found = sorted((p['carrier'], os.path.basename(p['stego_path'])) for p in pairs)
    assert found == [('arctic_a0001', f'{method}_aB3dE_arctic_a0001.wav'),
                     ('arctic_a0001', 'arctic_a0001.wav'),
                     ('arctic_a00010', f'{method}_xY12z_arctic_a00010.wav')]
    assert {p['label'] for p in pairs} == {'PC [v2]'}
