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


@pytest.mark.parametrize('mode', MODES)
def test_beta_k_fixes_the_counterexample_to_the_submitted_lemma(mode):
    """
    Bit 1, a = 1, b = 1.43, mu = 0.5 < beta: without beta_k (keep if
    b - a >= beta S / 2) the pair was not modified and mu = -0.5 flipped the
    bit. With beta_k (Theorem 3) the pair is modified and any |mu| < beta is
    tolerated.
    """
    a, b, mu = 1.0, 1.43, 0.5
    S = (a + b) / 2
    assert b - a >= BETA * S / 2                         # old keep test: not modified
    new_a, new_b = R.rslbe(mode, 1, a, b, ALPHA, BETA)
    assert (new_a, new_b) != (a, b)
    assert new_a - new_b + mu * S < 0                    # still decoded as '1'
    assert R.guaranteed_margins(mode, ALPHA, BETA)[1] == BETA


def test_beta_k_theorem_3():
    # b <= a: beta_k = gamma;  b > a: beta_k = gamma + 2 (b - a) / (a + b)
    assert R.beta_k(2.0, 1.0, 0.7) == pytest.approx(0.7)
    assert R.beta_k(1.0, 3.0, 0.7) == pytest.approx(0.7 + 1.0)
    # beta_A = beta_k + delta satisfies both Lemma 1 conditions for gamma > mu_max
    mu_max = 0.69
    for a, b in [(1.0, 3.0), (3.0, 1.0), (1.0, 1.0)]:
        delta = R.relative_difference(a, b)
        beta_A = R.beta_k(a, b, 0.7) + delta
        assert beta_A > mu_max and beta_A > mu_max + delta


@pytest.mark.parametrize('mode', MODES)
def test_nonnegative_singular_values_within_limits(mode, pairs):
    alpha_max, beta_max = R.nonnegative_limits(mode)
    alpha = alpha_max if alpha_max else 1.9
    beta = beta_max if beta_max else 1.9
    for a, b in pairs:
        for bit in (0, 1):
            assert min(R.rslbe(mode, bit, a, b, alpha, beta)) >= -1e-12


@pytest.mark.parametrize('mode', ['PCv1', 'PCv2'])
def test_pc_modes_need_beta_at_most_one(mode):
    gamma = 1.2
    b = 1.0
    a = b * (2 - gamma) / (2 + gamma) + 1e-9             # b - a just below gamma S
    assert R.rslbe(mode, 1, a, b, ALPHA, gamma)[0] < 0


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


@pytest.mark.parametrize('mode', MODES)
def test_beta_rule_half_is_the_previous_algorithm_4(mode, pairs):
    """BETA_RULE='HALF': keep if b - a >= beta S / 2, shift by beta S, margin beta / 2."""
    S = 1.0
    a, b = S - BETA * S / 4, S + BETA * S / 4             # b - a = beta S / 2: kept by HALF only
    assert R.rslbe(mode, 1, a, b, ALPHA, BETA, 'HALF') == (a, b)
    assert R.rslbe(mode, 1, a, b, ALPHA, BETA) != (a, b)
    assert R.guaranteed_margins(mode, ALPHA, BETA, 'HALF')[1] == BETA / 2
    for a, b in pairs[:500]:
        generic = R.gslbe(mode, 1, a, b, *R.adjusted_factors(mode, a, b, ALPHA, BETA, 'HALF'))
        assert np.allclose(generic, R.rslbe(mode, 1, a, b, ALPHA, BETA, 'HALF'), rtol=1e-12, atol=1e-12)
        assert R.rslbe(mode, 0, a, b, ALPHA, BETA, 'HALF') == R.rslbe(mode, 0, a, b, ALPHA, BETA)


def test_beta_rule_parameter_of_the_methods():
    half = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', dict(BASE, BETA_RULE='HALF'))
    new = create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', BASE)
    assert half.name == new.name + '_BETA_HALF' and half.normal_name == 'PC [v2] [β/2]'
    a, b = 1.0, 1.43
    assert half.modify_singular_values(a, b, 1)[:2] == R.rslbe('PCv2', 1, a, b, ALPHA, BETA, 'HALF')
    assert new.modify_singular_values(a, b, 1)[:2] == R.rslbe('PCv2', 1, a, b, ALPHA, BETA)
    with pytest.raises(ValueError, match='BETA_RULE'):
        create_method('DIFFERENTIAL_SVD_XUE', dict(BASE, BETA_RULE='HALF'))
    with pytest.raises(ValueError, match='BETA_RULE'):
        create_method('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', dict(BASE, BETA_RULE='x'))


RULE_COMBINATIONS = [(br, ar) for br in R.BETA_RULES for ar in R.ALPHA_RULES]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('beta_rule,alpha_rule', RULE_COMBINATIONS)
def test_algorithm_4_equals_algorithm_3_for_every_rule(mode, beta_rule, alpha_rule, pairs):
    for a, b in pairs[:1000]:
        factors = R.adjusted_factors(mode, a, b, ALPHA, BETA, beta_rule, alpha_rule)
        for bit in (0, 1):
            generic = R.gslbe(mode, bit, a, b, *factors)
            reduced = R.rslbe(mode, bit, a, b, ALPHA, BETA, beta_rule, alpha_rule)
            assert np.allclose(generic, reduced, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_alpha_k_margin_and_scope(mode, pairs):
    """ALPHA_K (default): bit 0 margin alpha in every mode; [v1] modes and bit 1 equal to HALF."""
    assert R.guaranteed_margins(mode, ALPHA, BETA) == (ALPHA, BETA)
    half_margin = ALPHA if mode in ('PCXUEv1', 'PCv1') else ALPHA / 2
    assert R.guaranteed_margins(mode, ALPHA, BETA, 'BETA_K', 'HALF') == (half_margin, BETA)
    margins = [margin(0, *R.rslbe(mode, 0, a, b, ALPHA, BETA), a, b) for a, b in pairs]
    assert min(margins) >= ALPHA - 1e-12
    for a, b in pairs[:500]:
        assert R.rslbe(mode, 1, a, b, ALPHA, BETA, 'BETA_K', 'HALF') == R.rslbe(mode, 1, a, b, ALPHA, BETA)
        if mode in ('PCXUEv1', 'PCv1'):
            assert R.rslbe(mode, 0, a, b, ALPHA, BETA, 'BETA_K', 'HALF') == R.rslbe(mode, 0, a, b, ALPHA, BETA)
    # [v2]: a - b = alpha S / 2 is kept by HALF only (mirror of the beta_k case)
    if mode in ('PCXUEv2', 'PCv2'):
        a, b = 1 + ALPHA / 4, 1 - ALPHA / 4
        assert R.rslbe(mode, 0, a, b, ALPHA, BETA, 'BETA_K', 'HALF') == (a, b)
        assert R.rslbe(mode, 0, a, b, ALPHA, BETA) != (a, b)


def test_alpha_k_needs_alpha_at_most_one_for_pc_v2(pairs):
    assert R.nonnegative_limits('PCv2', 'BETA_K', 'ALPHA_K') == (1.0, 1.0)
    for a, b in pairs:
        assert min(R.rslbe('PCv2', 0, a, b, 1.0, 1.0, 'BETA_K', 'ALPHA_K')) >= -1e-12
    a = 1.0
    b = a * (2 - 1.2) / (2 + 1.2) + 1e-9                 # a - b just below alpha S
    assert R.rslbe('PCv2', 0, a, b, 1.2, BETA, 'BETA_K', 'ALPHA_K')[1] < 0


def test_alpha_rule_parameter_of_the_methods():
    name = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3'
    default = create_method(name, BASE)
    assert default.method_arguments['ALPHA_RULE'] == 'ALPHA_K'
    both = create_method(name, dict(BASE, ALPHA_RULE='HALF', BETA_RULE='HALF'))
    assert both.name == default.name + '_BETA_HALF_ALPHA_HALF'
    assert both.normal_name == 'PC [v3] [β/2] [α/2]'
    a, b = 1.43, 1.0
    assert both.modify_singular_values(a, b, 0)[:2] == R.rslbe('PCv2', 0, a, b, ALPHA, BETA, 'HALF', 'HALF')
    assert default.modify_singular_values(a, b, 0)[:2] == R.rslbe('PCv2', 0, a, b, ALPHA, BETA, 'BETA_K', 'ALPHA_K')
    # [v1] modes: ALPHA_RULE has no effect, so it does not change the name
    v1 = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1'
    assert create_method(v1, dict(BASE, ALPHA_RULE='HALF')).name == create_method(v1, BASE).name
    with pytest.raises(ValueError, match='ALPHA_RULE'):
        create_method('MDCT_SVD_WEN', dict(BASE, FRAME_LENGTH=40, HOP_DURATION_MS=20, ALPHA_RULE='HALF'))
    with pytest.raises(ValueError, match='ALPHA_RULE'):
        create_method(name, dict(BASE, ALPHA_RULE='x'))


def test_rule_variants_are_rejected_with_legacy():
    """The legacy rules ignore BETA_RULE / ALPHA_RULE - such a variant would silently get the same name."""
    name = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2'
    create_method(name, dict(BASE, LEGACY=True))
    for rules in ({'BETA_RULE': 'HALF'}, {'ALPHA_RULE': 'HALF'}):
        with pytest.raises(ValueError, match='LEGACY'):
            create_method(name, dict(BASE, LEGACY=True, **rules))
