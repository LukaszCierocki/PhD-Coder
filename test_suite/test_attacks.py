"""Attacks write valid files; the noise attack reaches its target SNR."""

import os

import numpy as np
import pytest

from lib.Analysis import Analysis
from lib.Attacks import Attacks, audioop, no_attacks
from lib.Carrier import read_audio_as_int16

from conftest import FS, synthetic_speech, write_wav


@pytest.fixture
def stego(tmp_path):
    write_wav(str(tmp_path / 'test.wav'), synthetic_speech(seconds=1.0))
    return str(tmp_path), 'test.wav'


def test_noise_attack_reaches_target_snr(stego):
    directory, name = stego
    path = Attacks(seed=1).add_gaussian_noise_with_target_snr(directory, name, target_snr_db=30)
    clean, _ = read_audio_as_int16(os.path.join(directory, name))
    noisy, fs = read_audio_as_int16(path)
    assert fs == FS and len(noisy) == len(clean)
    assert np.std(noisy.astype(float)) > 0.9 * np.std(clean.astype(float))  # not silence (old bug)
    assert Analysis(plots=False).calculate_snr(clean, noisy) == pytest.approx(30, abs=0.5)


def test_noise_attack_is_reproducible_with_seed(stego):
    directory, name = stego
    first = read_audio_as_int16(Attacks(seed=5).add_gaussian_noise_with_target_snr(directory, name, 20))[0]
    second = read_audio_as_int16(Attacks(seed=5).add_gaussian_noise_with_target_snr(directory, name, 20))[0]
    assert np.array_equal(first, second)


@pytest.mark.parametrize('attack,args', [
    ('low_pass_filter', (3000,)), ('high_pass_filter', (500,)), ('requantize_audio', (10,)),
    ('jitter_attack', (1,)), ('resample_attack', (2,)), ('clipping_attack', (1.3,)),
    ('packet_loss_attack', (20, 0.2)),
])
def test_attacks_keep_length_and_rate(stego, attack, args):
    directory, name = stego
    path = getattr(Attacks(seed=0), attack)(directory, name, *args)
    data, fs = read_audio_as_int16(path)
    assert fs == FS
    assert len(data) == len(read_audio_as_int16(os.path.join(directory, name))[0])


@pytest.mark.skipif(audioop is None, reason='audioop not available')
def test_g711_attack(stego):
    directory, name = stego
    path = Attacks().g711_attack(directory, name, law='mu')
    data, fs = read_audio_as_int16(path)
    clean = read_audio_as_int16(os.path.join(directory, name))[0]
    assert fs == FS and len(data) == len(clean)
    assert Analysis(plots=False).calculate_snr(clean, data) > 10


def test_failing_attack_does_not_stop_the_others(stego, monkeypatch):
    directory, name = stego
    monkeypatch.setattr('shutil.which', lambda _: None)  # no ffmpeg
    config = dict(no_attacks(), MP3=True, NOISE=True)
    written = Attacks(seed=0).perform_attacks(directory, 'test', config)
    assert [os.path.basename(p) for p in written] == ['noised_30dB_test.wav']


def _tone(freq, seconds=0.5, amplitude=8000):
    t = np.arange(int(FS * seconds)) / FS
    return np.rint(amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def test_requantization_to_8_bits(tmp_path):
    write_wav(str(tmp_path / 'x.wav'), synthetic_speech(seconds=0.5))
    path = Attacks().requantize_audio(str(tmp_path), 'x.wav', bit_depth=8)
    data, _ = read_audio_as_int16(path)
    assert path.endswith('requantized_8bit_x.wav')
    assert np.all((data.astype(int) % 256 == 0) | (data == 32767))


def test_resampling_uses_an_anti_aliasing_filter(tmp_path):
    """16 kHz -> 8 kHz -> 16 kHz: a 2 kHz tone survives, a 6 kHz tone is removed (not aliased to 2 kHz)."""
    analysis = Analysis(plots=False)
    for freq, survives in ((2000, True), (6000, False)):
        write_wav(str(tmp_path / f'{freq}.wav'), _tone(freq))
        data, _ = read_audio_as_int16(Attacks().resample_attack(str(tmp_path), f'{freq}.wav', 2))
        clean = _tone(freq)
        energy_ratio = np.sum(data[400:-400].astype(float) ** 2) / np.sum(clean[400:-400].astype(float) ** 2)
        if survives:
            assert analysis.calculate_snr(clean[400:-400], data[400:-400]) > 30
        else:
            assert energy_ratio < 1e-3


def test_attack_parameters(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Unknown attack parameters'):
        Attacks(params={'NOISE_DB': 20})
    write_wav(str(tmp_path / 'test.wav'), synthetic_speech(seconds=0.5))
    attacks = Attacks(seed=0, params={'REQUANTIZE_BITS': 10, 'NOISE_SNR_DB': 20})
    written = attacks.perform_attacks(str(tmp_path), 'test', dict(no_attacks(), REQUANTIZE=True, NOISE=True))
    assert sorted(os.path.basename(p) for p in written) == ['noised_20dB_test.wav', 'requantized_10bit_test.wav']
    with pytest.raises(ValueError, match='Nyquist'):
        Attacks().low_pass_filter(str(tmp_path), 'test.wav', cutoff=9000)
