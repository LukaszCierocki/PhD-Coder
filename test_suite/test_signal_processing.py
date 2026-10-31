"""Building blocks: Hankel/SVD decomposition, MDCT/TDAC, band selection, framing, int16 conversion."""

import numpy as np
import pytest

from lib.Method import (FrameProcessor, MDCTProcessor, MultiLevelSVDProcessor, decode_mask, encode_mask,
                        ms_to_samples, to_int16)


def diagonal_average_reference(matrix):
    """The original double loop."""
    rows, cols = matrix.shape
    result = np.zeros(rows + cols - 1)
    count = np.zeros(rows + cols - 1)
    for i in range(rows):
        for j in range(cols):
            result[i + j] += matrix[i, j]
            count[i + j] += 1
    return result / count


@pytest.mark.parametrize('shape', [(2, 9), (3, 7), (5, 5), (1, 4)])
def test_diagonal_average_matches_original_loop(shape):
    matrix = np.random.default_rng(0).normal(size=shape)
    assert np.array_equal(MultiLevelSVDProcessor.diagonal_average(matrix), diagonal_average_reference(matrix))


@pytest.mark.parametrize('k', [1, 2, 3, 6])
def test_hankel_k_row_matches_definition(k):
    vector = np.arange(10, dtype=float)
    cols = len(vector) - k + 1
    expected = np.vstack([vector[i:i + cols] for i in range(k)])
    assert np.array_equal(MultiLevelSVDProcessor.hankel_k_row(vector, k), expected)


def test_hankel_k_row_rejects_too_many_rows():
    with pytest.raises(ValueError):
        MultiLevelSVDProcessor.hankel_k_row(np.arange(3), 4)


@pytest.mark.parametrize('rows', [2, 3])
@pytest.mark.parametrize('levels', [1, 2, 5, 8])
def test_multilevel_decomposition_reconstructs_exactly(rows, levels):
    processor = MultiLevelSVDProcessor({'HANKEL_ROWS': rows})
    vector = np.random.default_rng(levels).normal(size=20)
    decomposition = processor.multi_level_decompose(vector, levels)
    assert len(decomposition) == levels
    assert np.allclose(processor.reconstruct_from_levels(decomposition), vector, atol=1e-9)


def test_multilevel_decomposition_rejects_short_vectors():
    processor = MultiLevelSVDProcessor({'HANKEL_ROWS': 2})
    with pytest.raises(ValueError):
        processor.multi_level_decompose(np.ones(2), 1)


def test_mdct_is_tdac_orthogonal():
    """Coefficients written into overlapping frames are recovered exactly after
    overlap-add (the property MDCT_SVD_WEN relies on)."""
    mdct = MDCTProcessor()
    frame, hop, n_frames = 64, 32, 8
    window = mdct.kaiser_derived(frame, beta=4.0)
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=(n_frames, hop))

    signal = np.zeros(hop * (n_frames - 1) + frame)
    for i in range(n_frames):
        signal[i * hop:i * hop + frame] += mdct.imdct(coefficients[i]) * window

    for i in range(1, n_frames - 1):  # interior frames have both neighbours
        recovered = mdct.mdct(signal[i * hop:i * hop + frame] * window)
        assert np.allclose(recovered, coefficients[i], atol=1e-9)


def test_mdct_band_selection_hits_the_requested_frequency():
    fs, frame = 16000, 640
    mdct = MDCTProcessor()
    t = np.arange(frame) / fs
    coefficients = mdct.mdct(np.sin(2 * np.pi * 2000 * t) * mdct.kaiser_derived(frame, 4.0))
    _, low, high = mdct.filter_mdct_coeffs(coefficients, fs, frame, 1500, 2500)
    assert low <= int(np.argmax(np.abs(coefficients))) < high
    assert (low + 0.5) * fs / frame == pytest.approx(1500, abs=fs / frame)


def test_frame_sizes_must_be_integer_samples():
    assert ms_to_samples(12.5, 16000) == 200
    with pytest.raises(ValueError):
        ms_to_samples(10, 22050)


def test_to_int16_rounds_and_clips():
    data, clipped = to_int16([0.4, 0.6, -0.6, 40000.0, -40000.0, 32767.4])
    assert data.dtype == np.int16
    assert data.tolist() == [0, 1, -1, 32767, -32768, 32767]
    assert clipped == 2
    with pytest.raises(ValueError):
        to_int16([0.0, np.nan])


def test_mask_encoding_round_trip():
    mask = [True, False, False, True]
    assert encode_mask(mask) == '1001'
    assert decode_mask('1001') == mask
    with pytest.raises(ValueError):
        decode_mask('10x1')


def test_vad_framing_equals_extractor_framing(carrier):
    processor = FrameProcessor()
    voiced, unvoiced, voiced_idx, unvoiced_idx, remaining = processor.divide_into_voiced_unvoiced_frames(
        'rVAD', carrier.data_signed_int16, carrier.fs, 20, 20)
    frames, _, rest = processor.divide_into_frames(carrier.data_signed_int16, 20, carrier.fs, 20)
    assert len(voiced) + len(unvoiced) == len(frames)
    assert len(voiced) > 0
    for frame, index in zip(voiced, voiced_idx):
        assert np.array_equal(frame, frames[index])
    assert np.array_equal(remaining, rest)
