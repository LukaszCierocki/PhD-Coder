"""Quality metrics, BER and test labels."""

import math

import numpy as np
import pytest

from lib.Analysis import Analysis, test_label_from_filename as label_of


@pytest.fixture
def analysis():
    return Analysis(plots=False)


def test_mse_has_no_int16_overflow(analysis):
    a = np.array([30000, -30000, 0], dtype=np.int16)
    b = np.array([-30000, 30000, 0], dtype=np.int16)
    assert analysis.calculate_mean_squared_error(a, b) == pytest.approx((60000 ** 2 * 2) / 3)


def test_mse_aligns_lengths(analysis):
    assert analysis.calculate_mean_squared_error(np.ones(10, dtype=np.int16), np.zeros(4, dtype=np.int16)) == 1.0


def test_snr_and_psnr(analysis):
    x = np.array([1000, -1000, 1000, -1000], dtype=np.int16)
    y = x + np.array([10, -10, 10, -10], dtype=np.int16)
    assert analysis.calculate_snr(x, y) == pytest.approx(40.0)
    assert analysis.calculate_peak_signal_to_noise_ratio(x, y) == pytest.approx(40.0)
    assert math.isinf(analysis.calculate_snr(x, x))


def test_segmental_snr_is_clamped(analysis):
    x = np.tile(np.array([1000, -1000], dtype=np.int16), 100)
    assert analysis.calculate_segmented_snr(x, x, 20) == pytest.approx(35.0)
    assert analysis.calculate_segmented_snr(x, x + 1, 20) == pytest.approx(35.0)
    assert analysis.calculate_segmented_snr(x, np.zeros_like(x), 20) == pytest.approx(0.0)


def test_ber_counts_missing_and_extra_bits():
    secret = [1, 0, 1, 1, 0, 0, 1, 0]
    assert Analysis.count_bit_errors(secret, secret) == (0, 0.0)
    assert Analysis.count_bit_errors(secret, secret[:6]) == (2, 25.0)
    assert Analysis.count_bit_errors(secret, secret + [1, 1]) == (2, 25.0)
    assert Analysis.count_bit_errors(secret, []) == (8, 100.0)
    flipped = [1 - b for b in secret]
    assert Analysis.count_bit_errors(secret, flipped) == (8, 100.0)
    with pytest.raises(ValueError):
        Analysis.count_bit_errors([], [1])


@pytest.mark.parametrize('filename,expected', [
    ('M_abcde_c', 'NONE'),
    ('noised_30dB_M_abcde_c', 'NOISED'),
    ('filtered_low_pass_M_abcde_c', 'FILTERED_LOW_PASS'),
    ('filtered_high_pass_M_abcde_c', 'FILTERED_HIGH_PASS'),
    ('requantized_10bit_M_abcde_c', 'REQUANTIZED'),
    ('mp3_attacked_64k_M_abcde_c', 'MP3_ATTACKED_64k'),
    ('jitter_1_M_abcde_c', 'JITTER'),
    ('packetloss_10_M_abcde_c', 'PACKET_LOSS'),
    ('resample_attack_x2_M_abcde_c', 'RESAMPLE_ATTACK'),
    ('scaled_70_M_abcde_c', 'SCALED_70'),
    ('g711_ulaw_attacked_M_abcde_c', 'G711_ULAW_ATTACKED'),
    ('g711_alaw_attacked_M_abcde_c', 'G711_ALAW_ATTACKED'),
])
def test_attack_labels(filename, expected):
    assert label_of(filename + '.wav', 'M_abcde_c') == expected


def test_carrier_names_never_look_like_attacks():
    test_name = 'METHOD_xY12z_scaled_noised_speaker'
    assert label_of(test_name + '.wav', test_name) == 'NONE'
    assert label_of('jitter_1_' + test_name, test_name) == 'JITTER'
