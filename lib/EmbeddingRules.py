"""
Singular-value embedding rules - one place, unit-tested.

Notation (all rules work on the two largest singular values of one frame):

    a = sigma_e  - largest singular value of the even branch   (d_1^{L_J},  sigma_11)
    b = sigma_o  - largest singular value of the odd branch    (d_1^{calL_J}, sigma_21)
    S = (a + b) / 2                       (S_avg)
    delta = (a - b) / S = 2 (a - b) / (a + b)

The extractor decides bit = 0 if a > b, else 1.

Pejas & Cierocki, "Adaptive High-Fidelity Speech Steganography Based on DCT and
Multi-Level SVD":

    adjusted_factors   - Lemma 1 / Theorem 3 (alpha_A, beta_A)
    beta_k             - Theorem 3: beta_k = gamma + max(0, -delta)
    alpha_k            - mirror for bit 0 ([v2] modes): alpha + max(0, delta) (default)
    gslbe              - Algorithm 3 (generic four-mode single-level bit embedding)
    rslbe              - Algorithm 4 (reduced four-mode embedding); equals
                         gslbe with the Lemma 1 factors (checked by the tests)
    guaranteed_margins - minimal decision margins (corrected Lemma 1)

Xue et al., "Robust Speech Steganography Using Differential SVD", IEEE Access 2019:

    xue_rule           - Eq. (19)-(22)
"""

PAPER_MODES = ('PCXUEv1', 'PCXUEv2', 'PCv1', 'PCv2')

# Bit 1 of Algorithm 4:
#   'BETA_K' - Theorem 3: beta_k = beta + max(0, -delta)  (current paper, default)
#   'HALF'   - previous version: keep if b - a >= beta S / 2, shift by beta S
BETA_RULES = ('BETA_K', 'HALF')


# Bit 0 of Algorithm 4, [v2] modes (PC XUE [v2], PC [v2], PC [v3]):
#   'ALPHA_K' - mirror of beta_k: alpha_k = alpha + max(0, delta), margin alpha (default)
#   'HALF'    - previous version: keep if a - b >= alpha S / 2, margin alpha / 2
# The [v1] modes already have margin alpha (alpha_A = alpha) - no effect there.
ALPHA_RULES = ('ALPHA_K', 'HALF')


def _check_beta_rule(beta_rule):
    if beta_rule not in BETA_RULES:
        raise ValueError(f"Unknown beta rule {beta_rule!r}. Known: {BETA_RULES}")


def _check_alpha_rule(alpha_rule):
    if alpha_rule not in ALPHA_RULES:
        raise ValueError(f"Unknown alpha rule {alpha_rule!r}. Known: {ALPHA_RULES}")


def _check(mode, bit):
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    if bit not in (0, 1):
        raise ValueError(f"Bit must be 0 or 1, got {bit!r}")


def relative_difference(a, b):
    """delta = 2 (a - b) / (a + b); 0 for an all-zero pair."""
    total = a + b
    return 2.0 * (a - b) / total if total != 0 else 0.0


def adjusted_factors(mode, a, b, alpha, beta, beta_rule='BETA_K', alpha_rule='ALPHA_K'):
    """
    Lemma 1: adjusted scaling factors.

    PC XUE [v1], PC [v1]:  alpha_A = alpha,          beta_A = beta_k + delta
    PC XUE [v2], PC [v2]:  alpha_A = alpha - delta,  beta_A = beta_k + delta

    with beta_k = beta + max(0, -delta) (Theorem 3, ``beta`` is the constant
    part gamma), i.e. beta_A = beta + max(0, delta).
    beta_rule='HALF': beta_A = beta + delta (previous version).
    alpha_rule='ALPHA_K' ([v2] modes): alpha_A = alpha_k - delta
    = alpha + max(0, -delta).
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    _check_beta_rule(beta_rule)
    _check_alpha_rule(alpha_rule)
    delta = relative_difference(a, b)
    beta_A = (beta_k(a, b, beta) if beta_rule == 'BETA_K' else beta) + delta
    if mode in ('PCXUEv1', 'PCv1'):
        return alpha, beta_A
    return (alpha_k(a, b, alpha) if alpha_rule == 'ALPHA_K' else alpha) - delta, beta_A


def alpha_k(a, b, gamma):
    """
    Bit 0, [v2] modes (mirror of beta_k): alpha_k = gamma + max(0, 2 (a - b) / (a + b))
    = gamma + max(0, delta). Keep test a - b >= alpha_k S / 2 <=> a - b >= gamma S.
    """
    return gamma + max(0.0, relative_difference(a, b))


def beta_k(a, b, gamma):
    """
    Theorem 3 / Algorithm 4 (bit 1): beta_k = gamma + max(0, 2 (b - a) / (a + b))
    = gamma + max(0, -delta). With gamma > mu_max both Lemma 1 conditions for
    beta_A = beta_k + delta hold (Corollary: beta_k > beta_k,inf).
    """
    return gamma + max(0.0, -relative_difference(a, b))


def gslbe(mode, bit, a, b, alpha_A, beta_A):
    """Algorithm 3 (generic form, takes the adjusted factors). Returns (a', b')."""
    _check(mode, bit)
    S = (a + b) / 2.0
    if bit == 0:
        if a - b >= alpha_A * S:
            return a, b
        if mode == 'PCXUEv1':
            return b + alpha_A * S, b
        if mode == 'PCXUEv2':
            return a + alpha_A * S, b
        if mode == 'PCv1':
            return b + alpha_A * S / 2.0, b - alpha_A * S / 2.0
        return a + alpha_A * S / 2.0, b - alpha_A * S / 2.0          # PCv2
    if b - a >= beta_A * S:
        return a, b
    if mode in ('PCXUEv1', 'PCXUEv2'):
        return a, b + beta_A * S
    return a - beta_A * S / 2.0, b + beta_A * S / 2.0                   # PCv1, PCv2


def rslbe(mode, bit, a, b, alpha, beta, beta_rule='BETA_K', alpha_rule='ALPHA_K'):
    """
    Algorithm 4 (reduced form, takes the constant parts alpha and
    beta = gamma of the paper).
    Returns (a', b').

    bit 0:  PC XUE [v1], PC [v1]: keep if a - b >= alpha S
                PC XUE [v1]: a' = b + alpha S
                PC [v1]:     a' = b + alpha S / 2,  b' = b - alpha S / 2
            PC XUE [v2], PC [v2]: keep if a - b >= alpha S / 2
                PC XUE [v2]: a' = b + alpha S
                PC [v2]:     a' = S (1 + alpha / 2), b' = S (1 - alpha / 2)
            alpha_rule='ALPHA_K' (default): [v2] modes with alpha replaced by
                alpha_k = alpha + max(0, delta) (keep <=> a - b >= alpha S,
                margin alpha); [v1] modes unchanged. 'HALF': constant alpha
                as written above (previous version, margin alpha / 2).
    bit 1:  beta_k = beta + max(0, 2 (b - a) / (a + b))       (Theorem 3)
            all modes: keep if b - a >= beta_k S / 2   (<=> b - a >= beta S)
                PC XUE [v1], [v2]: b' = a + beta_k S
                PC [v1], [v2]:     a' = S (1 - beta_k / 2), b' = S (1 + beta_k / 2)
            beta_rule='HALF': the same with beta_k replaced by beta (previous
            version, keep if b - a >= beta S / 2, margin beta / 2).
    """
    _check(mode, bit)
    _check_beta_rule(beta_rule)
    _check_alpha_rule(alpha_rule)
    S = (a + b) / 2.0
    if bit == 0:
        if mode in ('PCXUEv1', 'PCv1'):
            if a - b >= alpha * S:
                return a, b
            if mode == 'PCXUEv1':
                return b + alpha * S, b
            return b + alpha * S / 2.0, b - alpha * S / 2.0
        ak = alpha_k(a, b, alpha) if alpha_rule == 'ALPHA_K' else alpha
        if a - b >= ak * S / 2.0:
            return a, b
        if mode == 'PCXUEv2':
            return b + ak * S, b
        return S + ak * S / 2.0, S - ak * S / 2.0
    bk = beta_k(a, b, beta) if beta_rule == 'BETA_K' else beta
    if b - a >= bk * S / 2.0:
        return a, b
    if mode in ('PCXUEv1', 'PCXUEv2'):
        return a, a + bk * S
    return S - bk * S / 2.0, S + bk * S / 2.0


def guaranteed_margins(mode, alpha, beta, beta_rule='BETA_K', alpha_rule='ALPHA_K'):
    """
    Corrected Lemma 1: minimal normalised decision margins after embedding,
    (a' - b') / S for bit 0 and (b' - a') / S for bit 1.

    bit 0: alpha for the [v1] modes, alpha / 2 for the [v2] modes
           (alpha with alpha_rule='ALPHA_K')
    bit 1: beta (= gamma) for every mode (Theorem 3; it was beta / 2
           before beta_k was introduced)
    A perturbation eps = mu * S of the difference is tolerated if
    |mu| < min(margins).
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    _check_beta_rule(beta_rule)
    _check_alpha_rule(alpha_rule)
    m0 = alpha if mode in ('PCXUEv1', 'PCv1') or alpha_rule == 'ALPHA_K' else alpha / 2.0
    return m0, (beta if beta_rule == 'BETA_K' else beta / 2.0)


def nonnegative_limits(mode, beta_rule='BETA_K', alpha_rule='ALPHA_K'):
    """
    Largest (alpha, beta) that keep every modified singular value >= 0
    (corrected Lemma 1). None = no limit.

    PC modes, beta_k / alpha_k: the smaller value is moved by gamma S / 2 and
    only while the difference is below gamma S, so it stays > S (1 - gamma)
    -> gamma <= 1. Reset to the mean with the constant factor: <= 2.
    PC [v1] bit 0: b' = b - alpha S / 2 -> alpha <= 1.
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    _check_beta_rule(beta_rule)
    _check_alpha_rule(alpha_rule)
    if mode not in ('PCv1', 'PCv2'):
        return None, None
    beta_max = 1.0 if beta_rule == 'BETA_K' else 2.0
    if mode == 'PCv1':
        return 1.0, beta_max
    return (1.0 if alpha_rule == 'ALPHA_K' else 2.0), beta_max


def xue_rule(a, b, bit, alpha, beta, as_printed=False):
    """
    Differential SVD rule of Xue et al. (2019), Eq. (19)-(22):
    T1 = alpha * b, T2 = beta * a.

    bit 0: if a - b < T1:  a' = b + T1                         (Eq. 19)
    bit 1: if b - a < T2:  b' = a + T2                         (default)
                           b' = b + T2   (as_printed=True: Eq. 22 as printed)

    Eq. (22) as printed does not reach the threshold (and is decoded as 0)
    whenever b < (1 - beta) a; it contradicts the paper's own statement that
    d_j <= -T2 for bit 1 (Sec. III-B). With alpha = beta = 0.2 (the paper's
    setting) the printed form gives ~10% BER without any attack, while the
    paper reports BER = 0 under scaling and MP3 - so the symmetric form
    b' = a + T2 is used by default. ``as_printed=True`` is the LEGACY
    behaviour of this code base.
    """
    if bit not in (0, 1):
        raise ValueError(f"Bit must be 0 or 1, got {bit!r}")
    T_bu_1 = alpha * b
    T_bu_2 = beta * a
    if bit == 0:
        return (a if a - b >= T_bu_1 else b + T_bu_1), b
    if b - a >= T_bu_2:
        return a, b
    return a, (b + T_bu_2 if as_printed else a + T_bu_2)
