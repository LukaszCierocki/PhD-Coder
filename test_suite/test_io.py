"""Exact audio I/O and reproducible secrets."""

import os
import random

import numpy as np
import pytest
import soundfile as sf

from lib.Carrier import Carrier, read_audio_as_int16
from lib.Secret import Secret
from lib.Tools import Tools

from conftest import FS, synthetic_speech, write_wav


@pytest.fixture
def tools(tmp_path):
    return Tools(results_root=str(tmp_path / 'Results'), carriers_dir=str(tmp_path / 'Carriers'))


def test_wav_round_trip_is_bit_exact(tools, tmp_path):
    data = np.array([-32768, -32767, -1000, -1, 0, 1, 1000, 32766, 32767], dtype=np.int16)
    data = np.concatenate([data, synthetic_speech(seconds=0.5)])
    tools.save_audio_file(str(tmp_path), {'name': 'x.wav', 'content': data, 'fs': FS})
    loaded = tools.load_audio_file(str(tmp_path / 'x.wav'))
    assert loaded['fs'] == FS
    assert loaded['content_signed_int16'].dtype == np.int16
    assert np.array_equal(loaded['content_signed_int16'], data)


def test_float_wav_is_scaled_with_32768_and_rounded(tmp_path):
    path = str(tmp_path / 'f.wav')
    sf.write(path, np.array([0.5, -0.5, 1000.4 / 32768, -1.0]), FS, subtype='FLOAT')
    data, fs = read_audio_as_int16(path)
    assert fs == FS
    assert data.tolist() == [16384, -16384, 1000, -32768]


def test_carrier_from_path_is_exact(tmp_path):
    data = synthetic_speech(seconds=0.5)
    path = write_wav(str(tmp_path / 'c.wav'), data)
    carrier = Carrier({'name': 'c', 'path': path})
    assert np.array_equal(carrier.data_signed_int16, data)
    assert carrier.fs == FS


def test_save_rejects_float_audio(tools, tmp_path):
    with pytest.raises(TypeError):
        tools.save_audio_file(str(tmp_path), {'name': 'x.wav', 'content': np.zeros(10), 'fs': FS})


def test_load_carriers_sorted_and_limited(tools, tmp_path):
    os.makedirs(tools.carriers_dir)
    for name in ('b', 'a', 'c'):
        write_wav(os.path.join(tools.carriers_dir, f'{name}.wav'), synthetic_speech(seconds=0.3))
    assert [c.name for c in tools.load_carriers()] == ['a', 'b', 'c']
    assert [c.name for c in tools.load_carriers(limit=2)] == ['a', 'b']
    with pytest.raises(FileNotFoundError):
        tools.load_carrier('missing')


def test_extracted_secret_with_non_ascii_bytes_is_saved(tools, tmp_path):
    from BitVector import BitVector
    os.makedirs(tmp_path / 'E')
    tools.save_extracted_secret(BitVector(bitlist=[1] * 16), {'Extracted': str(tmp_path / 'E')}, 'x')
    assert (tmp_path / 'E' / 'x.txt').read_text(encoding='utf-8') == '\\xff\\xff'


def test_secret_is_reproducible_and_fits_capacity():
    a = Secret({'size': 101, 'seed': 123, 'only_embed': True})
    b = Secret({'size': 101, 'seed': 123, 'only_embed': True})
    assert a.binary_data == b.binary_data and a.name == b.name
    assert a.size == 96  # 8 * (101 // 8)
    assert Secret({'size': 101, 'seed': 124, 'only_embed': True}).binary_data != a.binary_data


def test_secret_matches_previous_global_random_implementation():
    """Old code: random.seed(seed); random.choices(...) - same text for the same seed."""
    import string
    seed, size = 77, 160
    random.seed(seed)
    text = ""
    while len(text.encode()) < size // 8:
        text += ''.join(random.choices(string.ascii_letters, k=random.randint(1, 10))) + " "
    text = text[:size // 8].strip()
    text += "a" * (size // 8 - len(text))
    assert Secret({'size': size, 'seed': seed, 'only_embed': True}).full_text == text


def test_secret_does_not_touch_global_random_state():
    random.seed(5)
    expected = random.random()
    random.seed(5)
    Secret({'size': 64, 'seed': 1, 'only_embed': True})
    assert random.random() == expected


@pytest.mark.parametrize('size', [None, 0, 7, 3.5])
def test_secret_rejects_invalid_sizes(size):
    with pytest.raises(ValueError):
        Secret({'size': size, 'only_embed': True})
