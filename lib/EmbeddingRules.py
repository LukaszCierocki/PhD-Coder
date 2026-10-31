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

    adjusted_factors   - Lemma 1 (alpha_A, beta_A)
    gslbe              - Algorithm 3 (generic four-mode single-level bit embedding)
    rslbe              - Algorithm 4 (reduced four-mode embedding); equals
                         gslbe with the Lemma 1 factors (checked by the tests)
    guaranteed_margins - minimal decision margins (corrected Lemma 1)

Xue et al., "Robust Speech Steganography Using Differential SVD", IEEE Access 2019:

    xue_rule           - Eq. (19)-(22)
"""

PAPER_MODES = ('PCXUEv1', 'PCXUEv2', 'PCv1', 'PCv2')


def _check(mode, bit):
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    if bit not in (0, 1):
        raise ValueError(f"Bit must be 0 or 1, got {bit!r}")


def relative_difference(a, b):
    """delta = 2 (a - b) / (a + b); 0 for an all-zero pair."""
    total = a + b
    return 2.0 * (a - b) / total if total != 0 else 0.0


def adjusted_factors(mode, a, b, alpha, beta):
    """
    Lemma 1: adjusted scaling factors.

    PC XUE [v1], PC [v1]:  alpha_A = alpha,          beta_A = beta + delta
    PC XUE [v2], PC [v2]:  alpha_A = alpha - delta,  beta_A = beta + delta
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    delta = relative_difference(a, b)
    if mode in ('PCXUEv1', 'PCv1'):
        return alpha, beta + delta
    return alpha - delta, beta + delta


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


def rslbe(mode, bit, a, b, alpha, beta):
    """
    Algorithm 4 (reduced form, takes the constant parts alpha, beta).
    Returns (a', b').

    bit 0:  PC XUE [v1], PC [v1]: keep if a - b >= alpha S
                PC XUE [v1]: a' = b + alpha S
                PC [v1]:     a' = b + alpha S / 2,  b' = b - alpha S / 2
            PC XUE [v2], PC [v2]: keep if a - b >= alpha S / 2
                PC XUE [v2]: a' = b + alpha S
                PC [v2]:     a' = S (1 + alpha / 2), b' = S (1 - alpha / 2)
    bit 1:  all modes: keep if b - a >= beta S / 2
                PC XUE [v1], [v2]: b' = a + beta S
                PC [v1], [v2]:     a' = S (1 - beta / 2), b' = S (1 + beta / 2)
    """
    _check(mode, bit)
    S = (a + b) / 2.0
    if bit == 0:
        if mode in ('PCXUEv1', 'PCv1'):
            if a - b >= alpha * S:
                return a, b
            if mode == 'PCXUEv1':
                return b + alpha * S, b
            return b + alpha * S / 2.0, b - alpha * S / 2.0
        if a - b >= alpha * S / 2.0:
            return a, b
        if mode == 'PCXUEv2':
            return b + alpha * S, b
        return S + alpha * S / 2.0, S - alpha * S / 2.0
    if b - a >= beta * S / 2.0:
        return a, b
    if mode in ('PCXUEv1', 'PCXUEv2'):
        return a, a + beta * S
    return S - beta * S / 2.0, S + beta * S / 2.0


def guaranteed_margins(mode, alpha, beta):
    """
    Corrected Lemma 1: minimal normalised decision margins after embedding,
    (a' - b') / S for bit 0 and (b' - a') / S for bit 1.

    bit 0: alpha for the [v1] modes, alpha / 2 for the [v2] modes
    bit 1: beta / 2 for every mode
    A perturbation eps = mu * S of the difference is tolerated if
    |mu| < min(margins).
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    m0 = alpha if mode in ('PCXUEv1', 'PCv1') else alpha / 2.0
    return m0, beta / 2.0


def nonnegative_limits(mode):
    """
    Largest (alpha, beta) that keep every modified singular value >= 0
    (corrected Lemma 1). None = no limit.
    """
    if mode not in PAPER_MODES:
        raise ValueError(f"Unknown embedding mode {mode!r}. Known: {PAPER_MODES}")
    if mode == 'PCv1':
        return 1.0, 2.0
    if mode == 'PCv2':
        return 2.0, 2.0
    return None, None


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
