"""Shared fixtures. Tests use synthetic speech-like signals, so they do not
need the (git-ignored) Carriers/ directory."""

import logging
import os
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.Carrier import Carrier  # noqa: E402

FS = 16000

BASE = {'REDUCED': False, 'ADAPTIVE': False, 'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD': 'rVAD',
        'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20,
        'HOP_DURATION_MS': 20, 'LEVELS': 1, 'HANKEL_ROWS': 2}

# (registry name, parameters, maximal BER [%] on the clean synthetic stegobject)
METHOD_CASES = [
    ('MDCT_SVD_WEN', dict(BASE, FRAME_LENGTH=40, HOP_DURATION_MS=20), 0.0),
    ('DIFFERENTIAL_SVD_XUE', dict(BASE), 0.0),
    ('DIFFERENTIAL_SVD_XUE_MULTI', dict(BASE, LEVELS=2), 0.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1', dict(BASE), 0.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2', dict(BASE, ADAPTIVE=True), 0.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', dict(BASE, ADAPTIVE=True), 0.0),
    # non-adaptive PC v1 loses bits by design (fixed shift may be too small)
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', dict(BASE, ADAPTIVE=False), 15.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', dict(BASE, ADAPTIVE=True), 0.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', dict(BASE, ADAPTIVE=True, LEVELS=4), 3.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE', dict(BASE, ADAPTIVE=True, LEVELS=3), 3.0),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED', dict(BASE, ADAPTIVE=True, LEVELS=3), 3.0),
]


def method_case_id(case):
    name, params, _ = case
    return f"{name}-A{params['ADAPTIVE']}-L{params['LEVELS']}"


def synthetic_speech(fs=FS, seconds=3.0, seed=0, amplitude=12000):
    """Harmonic 'voiced' segments with a varying f0, separated by quiet noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = np.zeros(n)
    segments = [(0.2, 0.8), (1.0, 1.5), (1.7, 2.3), (2.5, 2.9)]
    for k, (a, b) in enumerate(segments):
        b = min(b, seconds)
        mask = (t >= a) & (t < b)
        if not mask.any():
            continue
        f0 = 110 + 25 * k + 10 * np.sin(2 * np.pi * 3 * t[mask]) + 5 * seed
        phase = 2 * np.pi * np.cumsum(f0) / fs
        voiced = sum((0.8 / h) * np.sin(h * phase + rng.uniform(0, 2 * np.pi)) for h in range(1, 30))
        envelope = np.sin(np.pi * (t[mask] - a) / (b - a)) ** 0.5
        x[mask] = voiced * envelope
    x = x / np.max(np.abs(x)) * amplitude + rng.normal(0, 30, n)
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


@pytest.fixture
def logger():
    log = logging.getLogger('phd_coder.tests')
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


@pytest.fixture
def carrier():
    return Carrier({'name': 'synthetic_a', 'fs': FS, 'data_signed_int16': synthetic_speech(seed=0)})


@pytest.fixture
def carrier_b():
    return Carrier({'name': 'synthetic_b', 'fs': FS, 'data_signed_int16': synthetic_speech(seed=1, seconds=2.5)})


def write_wav(path, data, fs=FS):
    from scipy.io import wavfile
    wavfile.write(path, fs, np.asarray(data, dtype=np.int16))
    return path
