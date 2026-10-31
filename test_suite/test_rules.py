"""
Embedding rules of the PC paper (Algorithms 3 and 4, corrected Lemma 1) and of
Xue et al., and their wiring into the method classes.
"""

import numpy as np
import pytest

from lib import EmbeddingRules as R
from lib.Registry import create_method

from conftest import BASE

MODES = R.PAPER_MODES
ALPHA = BETA = 0.7


@pytest.fixture(scope='module')
def pairs():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.05, 10, 4000)
    b = a * np.exp(rng.uniform(-1.6, 1.6, 4000))
    return list(zip(a, b))


def margin(bit, new_a, new_b, a, b):
    """Normalised decision margin in the direction of the bit."""
    S = (a + b) / 2
    return ((new_a - new_b) if bit == 0 else (new_b - new_a)) / S


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('bit', [0, 1])
def test_algorithm_4_equals_algorithm_3_with_lemma_factors(mode, bit, pairs):
    for a, b in pairs:
        generic = R.gslbe(mode, bit, a, b, *R.adjusted_factors(mode, a, b, ALPHA, BETA))
        reduced = R.rslbe(mode, bit, a, b, ALPHA, BETA)
        assert np.allclose(generic, reduced, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('bit', [0, 1])
def test_corrected_lemma_margins_hold_and_are_tight(mode, bit, pairs):
    bound = R.guaranteed_margins(mode, ALPHA, BETA)[bit]
    margins = np.array([margin(bit, *R.rslbe(mode, bit, a, b, ALPHA, BETA), a, b) for a, b in pairs])
    assert margins.min() >= bound - 1e-12
    # tight: a pair exactly on the keep threshold attains the bound
    S = 1.0
    if bit == 0:
        a, b = S + bound * S / 2, S - bound * S / 2
    else:
        a, b = S - bound * S / 2, S + bound * S / 2
    assert margin(bit, *R.rslbe(mode, bit, a, b, ALPHA, BETA), a, b) == pytest.approx(bound)


def test_counterexample_to_the_submitted_lemma():
    """PC [v1], bit 1: |mu| < beta, but the bit is decoded wrongly (needs |mu| < beta / 2)."""
    a, b, mu = 1.0, 1.43, 0.5
    new_a, new_b = R.rslbe('PCv1', 1, a, b, ALPHA, BETA)
    assert (new_a, new_b) == (a, b)                      # no modification
    S = (a + b) / 2
    delta = new_a - new_b + mu * S                       # extraction statistic
    assert mu < BETA and delta > 0                       # decoded as '0'
    assert mu > R.guaranteed_margins('PCv1', ALPHA, BETA)[1]


@pytest.mark.parametrize('mode', MODES)
def test_nonnegative_singular_values_within_limits(mode, pairs):
    alpha_max, beta_max = R.nonnegative_limits(mode)
    alpha = alpha_max if alpha_max else 1.9
    beta = beta_max if beta_max else 1.9
    for a, b in pairs:
        for bit in (0, 1):
            assert min(R.rslbe(mode, bit, a, b, alpha, beta)) >= -1e-12


def test_pc_v1_needs_alpha_at_most_one():
    a = 1.0
    b = a * (2 - 1.2) / (2 + 1.2) + 1e-9                 # just inside the modification branch
    assert R.rslbe('PCv1', 0, a, b, 1.2, BETA)[1] < 0


def test_xue_rule_symmetric_and_as_printed():
    # bit 1 with sigma_21 < (1 - beta) sigma_11: Eq. (22) as printed stays decoded as 0
    a, b, alpha = 10.0, 1.0, 0.2
    new_a, new_b = R.xue_rule(a, b, 1, alpha, alpha)
    assert new_b - new_a == pytest.approx(alpha * a)     # d_j = -T_bu,2, as the paper states
    new_a, new_b = R.xue_rule(a, b, 1, alpha, alpha, as_printed=True)
    assert new_a - new_b > 0
    assert R.xue_rule(1.0, 10.0, 0, alpha, alpha)[0] - 10.0 == pytest.approx(alpha * 10.0)


PC_CLASSES = [
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1', 'PCXUEv1'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2', 'PCXUEv2'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', 'PCv1'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', 'PCv2'),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', 'PCv2'),
]


@pytest.mark.parametrize('name,mode', PC_CLASSES)
@pytest.mark.parametrize('adaptive', [False, True])
def test_methods_use_algorithm_4(name, mode, adaptive, pairs):
    method = create_method(name, dict(BASE, ADAPTIVE=adaptive, REDUCED=adaptive))
    assert method.PAPER_MODE == mode
    for a, b in pairs[:500]:
        for bit in (0, 1):
            new_a, new_b, alpha_A, beta_A = method.modify_singular_values(a, b, bit)
            assert (new_a, new_b) == R.rslbe(mode, bit, a, b, ALPHA, BETA)
            assert (alpha_A, beta_A) == R.adjusted_factors(mode, a, b, ALPHA, BETA)


def test_legacy_rules_are_the_previous_ones():
    a, b, S = 1.0, 2.0, 1.5
    pc_v1 = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', dict(BASE, LEGACY=True))
    assert pc_v1.modify_singular_values(a, b, 0)[:2] == pytest.approx((a + 0.7 * S / 2, b - 0.7 * S / 2))
    xue_v1 = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1', dict(BASE, LEGACY=True))
    beta_A = 0.7 + 2 * (2.0 - 1.0) / 3.0
    assert xue_v1.modify_singular_values(2.0, 1.0, 1)[:2] == pytest.approx((2.0, 2.0 + beta_A * S))
    pc_v2 = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', dict(BASE, LEGACY=True))
    alpha_A = 0.7 + 2 * (b - a) / 3.0
    assert pc_v2.modify_singular_values(a, b, 0)[:2] == pytest.approx((S + alpha_A * S / 2, S - alpha_A * S / 2))
    assert pc_v1.name.endswith('_LEGACY') and pc_v1.normal_name == 'PC [v1] (legacy)'


def test_pc_v3_score_follows_algorithm_2():
    method = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', dict(BASE, LEVELS=2))
    frame = np.zeros(8)
    embedded = np.full(8, 2.0)
    stats = {'new_s_even': 1.0, 'new_s_odd': 3.0}
    assert method.candidate_score(frame, embedded, stats, 0) == pytest.approx(2.0 / 4.0)  # |diff| / MSE
    legacy = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', dict(BASE, LEVELS=2, LEGACY=True))
    assert legacy.candidate_score(frame, embedded, stats, 0) == pytest.approx(-2.0 / 2.0)  # signed / MAE
