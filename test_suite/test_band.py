"""Frequency-band safeguards."""

import logging

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from lib.Carrier import Carrier
from lib.Registry import create_method
from lib.Secret import Secret

from conftest import BASE, FS, synthetic_speech


def make(name='DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', logger=None, **overrides):
    return create_method(name, dict(BASE, **overrides), logger)


def embedded(method, carrier):
    method.set_carrier(carrier)
    method.set_secret(Secret({'size': method.get_capacity(), 'seed': 2, 'only_embed': True}))
    return method.embedding()


def test_band_has_an_even_number_of_coefficients(carrier):
    """12.5 ms at 16 kHz: 1.5-2.5 kHz covers 25 DCT bins - the papers assume 2M."""
    method = make(FRAME_LENGTH=12.5, HOP_DURATION_MS=12.5)
    method.set_carrier(carrier)
    assert method.band == (37, 61) and (method.band[1] - method.band[0]) % 2 == 0
    assert method.band_frequency == pytest.approx((1480.0, 2440.0))
    legacy = make(FRAME_LENGTH=12.5, HOP_DURATION_MS=12.5, LEGACY=True)
    legacy.set_carrier(carrier)
    assert legacy.band == (37, 62)                        # previous behaviour


def test_wen_band_matches_the_paper(carrier):
    """Wen et al.: 100 Hz - 3 kHz with a 40 ms MDCT frame = 116 = 2 N_k coefficients (N_k = 58)."""
    method = make('MDCT_SVD_WEN', FRAME_LENGTH=40, HOP_DURATION_MS=20, FREQ_L=100, FREQ_H=3000, ALPHA=0.61)
    method.set_carrier(carrier)
    assert method.band[1] - method.band[0] == 116
    assert method.band_frequency == pytest.approx((100.0, 3000.0))


def test_band_is_stored_in_the_key_and_reported(carrier, logger):
    method = make(logger=logger)
    embedded(method, carrier)
    key = method.get_extraction_key()
    assert key['band_bins'] == [60, 100] and key['band_hz'] == [1500.0, 2500.0]
    assert key['params']['FREQ_L'] == 1500 and key['params']['LEGACY'] is False
    assert -40 < method.band_energy_db < 0


def test_receiver_with_another_band_is_rejected(carrier, logger):
    sender = make(logger=logger)
    stego = embedded(sender, carrier)
    key = sender.get_extraction_key()
    for overrides in ({'FREQ_L': 2000, 'FREQ_H': 3000}, {'LEVELS': 2}):
        receiver = make(logger=logger, **overrides)
        with pytest.raises(ValueError):
            receiver.extraction(stego, key=key)
    # the frame length is taken from the key, so a different FRAME_LENGTH at the receiver is harmless
    assert make(logger=logger, FRAME_LENGTH=10, HOP_DURATION_MS=10).extraction(stego, key=key) == \
        sender.extraction(stego)
    # 12.5 ms frames: 24 bins (even band) at the sender, 25 bins with LEGACY at the receiver
    sender_125 = make(logger=logger, FRAME_LENGTH=12.5, HOP_DURATION_MS=12.5)
    stego_125 = embedded(sender_125, carrier)
    with pytest.raises(ValueError, match='bins'):
        make(logger=logger, FRAME_LENGTH=12.5, HOP_DURATION_MS=12.5, LEGACY=True).extraction(
            stego_125, key=sender_125.get_extraction_key())
    # keys written before the check (no band information) are accepted
    old_key = {k: v for k, v in key.items() if k not in ('band_bins', 'band_hz', 'params')}
    assert make(logger=logger).extraction(stego, key=old_key) == sender.extraction(stego)


def test_band_including_dc_warns(carrier, caplog):
    method = make('DIFFERENTIAL_SVD_XUE', logger=logging.getLogger('band-test'), FREQ_L=0, FREQ_H=500)
    with caplog.at_level(logging.WARNING, logger='band-test'):
        method.set_carrier(carrier)
    assert any('DC coefficient' in r.message for r in caplog.records)


def test_band_without_energy_warns(caplog):
    """Carrier low-passed at 600 Hz, band 1.5-2.5 kHz: the band lies outside the signal bandwidth."""
    sos = butter(10, 600 / (FS / 2), output='sos')       # second-order sections: stable at high order
    lowpassed = np.rint(sosfiltfilt(sos, synthetic_speech().astype(float))).astype(np.int16)
    carrier = Carrier({'name': 'lowpassed', 'fs': FS, 'data_signed_int16': lowpassed})
    method = make(logger=logging.getLogger('band-test'))
    with caplog.at_level(logging.WARNING, logger='band-test'):
        method.set_carrier(carrier)
    assert method.band_energy_db < -35          # speech in 1.5-2.5 kHz: about -20 dB
    assert any('almost no energy' in r.message for r in caplog.records)


def test_band_above_nyquist_for_8khz_carrier():
    carrier = Carrier({'name': 'nb', 'fs': 8000, 'data_signed_int16': synthetic_speech(fs=8000)})
    with pytest.raises(ValueError, match='Nyquist'):
        make(FREQ_L=3500, FREQ_H=4500).set_carrier(carrier)
    method = make()                                      # 1.5-2.5 kHz works at 8 kHz as well
    method.set_carrier(carrier)
    assert method.band_frequency == pytest.approx((1500.0, 2500.0))


def test_every_frame_mode_without_vad(carrier, logger):
    method = make('DIFFERENTIAL_SVD_XUE', logger=logger, ALPHA=0.2, BETA=0.2, WHERE_TO_HIDE='ALL')
    stego = embedded(method, carrier)
    n_frames = len(carrier.data_signed_int16) // method.frame_samples
    assert method.capacity == n_frames
    assert set(method.get_extraction_key()['voiced_mask']) == {'1'}
    assert len(method.extraction(stego)) == len(method.secret.binary_data)
