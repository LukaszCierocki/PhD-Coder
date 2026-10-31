"""
End-to-end checks of every steganographic method:

* the stegobject is a valid int16 signal of the carrier length,
* samples outside the embedding frames are untouched,
* the bits are recovered (in memory, after a WAV round trip and by a fresh
  receiver that only has the extraction key),
* re-using one method object for several carriers is safe,
* embedding is deterministic.
"""

import json
import os

import numpy as np
import pytest

from lib.Analysis import Analysis
from lib.Method import decode_mask
from lib.Registry import create_method
from lib.Secret import Secret
from lib.Tools import Tools

from conftest import METHOD_CASES, method_case_id


def embed(name, params, carrier, logger, seed=11):
    method = create_method(name, params, logger)
    method.set_carrier(carrier)
    secret = Secret({'size': method.get_capacity(), 'seed': seed, 'only_embed': True}, logger=logger)
    method.set_secret(secret)
    return method, secret, method.embedding()


def ber(secret, extracted):
    return Analysis.count_bit_errors(secret, extracted)[1]


@pytest.fixture(scope='module')
def tools(tmp_path_factory):
    root = tmp_path_factory.mktemp('results')
    return Tools(results_root=str(root), carriers_dir=str(root / 'none'))


@pytest.mark.parametrize('case', METHOD_CASES, ids=method_case_id)
def test_embedding_and_extraction(case, carrier, logger, tools, tmp_path):
    name, params, max_ber = case
    method, secret, stego = embed(name, params, carrier, logger)

    # valid stegobject
    assert stego.dtype == np.int16
    assert len(stego) == len(carrier.data_signed_int16)
    assert method.embedded_bits == len(secret.binary_data)
    assert len(method.s_even) == len(method.new_s_even) == len(secret.binary_data)
    assert not np.array_equal(stego, carrier.data_signed_int16)

    # extraction in memory
    extracted = method.extraction(stego)
    assert len(extracted) == len(secret.binary_data)
    assert ber(secret, extracted) <= max_ber

    # WAV round trip gives exactly the same bits
    tools.save_audio_file(str(tmp_path), {'name': 's.wav', 'content': stego, 'fs': carrier.fs})
    loaded = tools.load_audio_file(str(tmp_path / 's.wav'))
    assert np.array_equal(loaded['content_signed_int16'], stego)
    assert method.extraction(loaded) == extracted

    # a fresh receiver with only the (JSON) key extracts the same bits
    key = json.loads(json.dumps(method.get_extraction_key()))
    receiver = create_method(name, params, logger)
    assert receiver.extraction(loaded, key=key) == extracted


@pytest.mark.parametrize('case', METHOD_CASES, ids=method_case_id)
def test_frames_without_bits_are_untouched(case, carrier, logger):
    name, params, _ = case
    method, secret, stego = embed(name, params, carrier, logger)
    mask = decode_mask(method.get_extraction_key()['voiced_mask'])
    frame, hop = method.frame_samples, method.hop_samples

    # frames that carry a bit
    carrying, bits = set(), 0
    for index, voiced in enumerate(mask):
        if voiced and bits < len(secret.binary_data):
            carrying.add(index)
            bits += 1

    touched = np.zeros(len(stego), dtype=bool)
    for index in carrying:
        touched[index * hop:index * hop + frame] = True
    if method.FRAMING == 'mdct':
        # w^2 pass-through of the first / last half frame is not complemented
        touched[:hop] = True
        touched[(len(mask) - 1) * hop + hop:] = True
        tolerance = 1
    else:
        tolerance = 0

    difference = np.abs(stego.astype(int) - carrier.data_signed_int16.astype(int))
    assert difference[~touched].max(initial=0) <= tolerance


@pytest.mark.parametrize('case', [c for c in METHOD_CASES if 'PC_V3' in c[0] or 'XUE' in c[0]],
                         ids=method_case_id)
def test_one_object_for_many_carriers(case, carrier, carrier_b, logger):
    """The PC v3 level map used to leak between carriers (wrong levels from the
    2nd carrier on, BER ~8% on the 2nd extraction)."""
    name, params, max_ber = case
    method = create_method(name, params, logger)

    for c in (carrier, carrier_b):
        method.set_carrier(c)
        secret = Secret({'size': method.get_capacity(), 'seed': 3, 'only_embed': True}, logger=logger)
        method.set_secret(secret)
        stego = method.embedding()
        first = method.extraction(stego)
        second = method.extraction(stego)  # e.g. an attacked copy
        assert first == second
        assert len(method.s_even) == len(secret.binary_data)

        _, _, fresh_stego = embed(name, params, c, logger, seed=3)
        assert np.array_equal(fresh_stego, stego)
        assert ber(secret, first) <= max_ber


def test_embedding_is_deterministic(carrier, logger):
    name, params, _ = METHOD_CASES[-3]  # PC v3
    _, _, a = embed(name, params, carrier, logger)
    _, _, b = embed(name, params, carrier, logger)
    assert np.array_equal(a, b)


def test_pc_v3_variants_produce_identical_stegobjects(carrier, logger):
    params = dict(METHOD_CASES[-1][1])
    stegos = [embed(name, params, carrier, logger)[2] for name in (
        'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3',
        'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE',
        'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED')]
    assert np.array_equal(stegos[0], stegos[1]) and np.array_equal(stegos[0], stegos[2])


def test_pc_v3_key_contains_one_level_per_bit(carrier, logger):
    name, params, _ = METHOD_CASES[-3]
    method, secret, _ = embed(name, params, carrier, logger)
    levels = method.get_extraction_key()['levels']
    assert len(levels) == len(secret.binary_data)
    assert set(levels) <= set(range(1, params['LEVELS'] + 1))


def test_truncated_stegobject_reports_missing_bits(carrier, logger):
    name, params, _ = METHOD_CASES[1]  # Xue
    method, secret, stego = embed(name, params, carrier, logger)
    extracted = method.extraction(stego[:len(stego) // 2])
    assert 0 < len(extracted) < len(secret.binary_data)
    errors, rate = Analysis.count_bit_errors(secret, extracted)
    assert errors >= len(secret.binary_data) - len(extracted)


def test_xue_bit_one_reaches_threshold():
    """Old rule: ro_21 + beta * ro_11 - fails when ro_21 < (1 - beta) * ro_11."""
    from Methods.DIFFERENTIAL_SVD_XUE import xue_rule
    new_11, new_21 = xue_rule(10.0, 1.0, 1, 0.7, 0.7)
    assert new_21 - new_11 >= 0.7 * 10.0 - 1e-12
    new_11, new_21 = xue_rule(1.0, 10.0, 0, 0.7, 0.7)
    assert new_11 - new_21 >= 0.7 * 10.0 - 1e-12


def test_wen_uses_requested_band(carrier, logger):
    name, params, _ = METHOD_CASES[0]
    method = create_method(name, params, logger)
    method.set_carrier(carrier)
    frame = method.frame_samples
    _, mdct, band, low, high = method._band(np.zeros(frame), carrier.fs)
    assert (low + 0.5) * carrier.fs / frame == pytest.approx(method.FREQ_L, abs=carrier.fs / frame)
    assert high * carrier.fs / frame == pytest.approx(method.FREQ_H, abs=carrier.fs / frame)
