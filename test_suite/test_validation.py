"""Invalid configurations and misuse must fail loudly instead of producing wrong results."""

import numpy as np
import pytest

from lib.Carrier import Carrier
from lib.Registry import create_method, get_method_class
from lib.Secret import Secret

from conftest import BASE, FS


def make(name='DIFFERENTIAL_SVD_XUE', logger=None, **overrides):
    return create_method(name, dict(BASE, **overrides), logger)


@pytest.mark.parametrize('overrides', [
    {'ALPHA': 0}, {'ALPHA': -1}, {'BETA': 'x'}, {'ALPHA': float('nan')},
    {'FREQ_L': 3000, 'FREQ_H': 2000}, {'FREQ_L': -5},
    {'FRAME_LENGTH': 0}, {'HOP_DURATION_MS': 10},           # hop != frame for a DCT method
    {'TYPE_OF_VAD': 'energy'}, {'WHERE_TO_HIDE': 'UNVOICED'},
    {'ADAPTIVE': 'yes'}, {'HANKEL_ROWS': 1}, {'LEGACY': 'yes'}, {'LABEL': ''}, {'LABEL': 3},
])
def test_invalid_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


def test_missing_parameter_is_rejected():
    params = dict(BASE)
    del params['ALPHA']
    with pytest.raises(ValueError):
        get_method_class('DIFFERENTIAL_SVD_XUE')(params)


def test_multilevel_methods_need_at_least_one_level():
    with pytest.raises(ValueError):
        make('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', LEVELS=0)


def test_wen_needs_half_overlap():
    with pytest.raises(ValueError):
        make('MDCT_SVD_WEN', FRAME_LENGTH=20, HOP_DURATION_MS=20)
    make('MDCT_SVD_WEN', FRAME_LENGTH=40, HOP_DURATION_MS=20)


def test_defaults_are_recorded():
    params = dict(BASE)
    params.pop('REDUCED')
    params.pop('HOP_DURATION_MS')
    method = get_method_class('DIFFERENTIAL_SVD_XUE')(params)
    assert method.method_arguments['REDUCED'] is False
    assert method.method_arguments['HOP_DURATION_MS'] == params['FRAME_LENGTH']


def test_band_above_nyquist_is_rejected(carrier):
    method = make(FREQ_L=7000, FREQ_H=9000)
    with pytest.raises(ValueError, match='Nyquist'):
        method.set_carrier(carrier)


def test_too_narrow_band_is_rejected(carrier):
    method = make('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', FREQ_L=1500, FREQ_H=1600)
    with pytest.raises(ValueError, match='coefficients'):
        method.set_carrier(carrier)


def test_non_integer_frame_is_rejected(carrier):
    method = make(FRAME_LENGTH=20.03, HOP_DURATION_MS=20.03)
    with pytest.raises(ValueError, match='integer'):
        method.set_carrier(carrier)


def test_carrier_shorter_than_frame_is_rejected():
    method = make()
    short = Carrier({'name': 'short', 'fs': FS, 'data_signed_int16': np.ones(100, dtype=np.int16)})
    with pytest.raises(ValueError):
        method.set_carrier(short)


def test_silent_carrier_has_no_capacity():
    method = make()
    silent = Carrier({'name': 'silent', 'fs': FS, 'data_signed_int16': np.zeros(FS, dtype=np.int16)})
    with pytest.raises(ValueError):
        method.set_carrier(silent)


def test_secret_larger_than_capacity_is_rejected(carrier, logger):
    method = make(logger=logger)
    method.set_carrier(carrier)
    too_big = Secret({'size': method.get_capacity() + 64, 'seed': 1, 'only_embed': True})
    with pytest.raises(ValueError, match='capacity'):
        method.set_secret(too_big)


def test_embedding_without_carrier_or_secret_fails(carrier):
    method = make()
    with pytest.raises(RuntimeError):
        method.embedding()
    method.set_carrier(carrier)
    with pytest.raises(RuntimeError):
        method.embedding()


def test_extraction_without_key_fails(carrier):
    with pytest.raises(RuntimeError):
        make().extraction({'content_signed_int16': carrier.data_signed_int16, 'fs': FS})


def test_extraction_with_other_sampling_rate_fails(carrier, logger):
    method = make(logger=logger)
    method.set_carrier(carrier)
    method.set_secret(Secret({'size': method.get_capacity(), 'seed': 1, 'only_embed': True}))
    stego = method.embedding()
    with pytest.raises(ValueError, match='sampling rate'):
        method.extraction({'content_signed_int16': stego, 'fs': 8000})


def test_pc_v3_key_without_levels_fails(carrier, logger):
    method = make('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', logger, ADAPTIVE=True, LEVELS=2)
    method.set_carrier(carrier)
    method.set_secret(Secret({'size': method.get_capacity(), 'seed': 1, 'only_embed': True}))
    stego = method.embedding()
    key = dict(method.get_extraction_key())
    del key['levels']
    with pytest.raises(ValueError, match='levels'):
        method.extraction(stego, key=key)


def test_set_carrier_discards_previous_secret(carrier, carrier_b, logger):
    method = make(logger=logger)
    method.set_carrier(carrier)
    method.set_secret(Secret({'size': method.get_capacity(), 'seed': 1, 'only_embed': True}))
    method.embedding()
    method.set_carrier(carrier_b)
    assert method.get_extraction_key() is None
    with pytest.raises(RuntimeError):
        method.embedding()


def test_unknown_method_name():
    with pytest.raises(KeyError):
        get_method_class('NOPE')


def test_legacy_only_where_a_previous_rule_exists():
    with pytest.raises(ValueError, match='LEGACY'):
        make('MDCT_SVD_WEN', FRAME_LENGTH=40, HOP_DURATION_MS=20, LEGACY=True)
    assert make('DIFFERENTIAL_SVD_XUE', LEGACY=True).legacy


def test_label_overrides_the_display_name():
    method = make('DIFFERENTIAL_SVD_XUE', LABEL='Xue et. al. (orig.)')
    assert method.normal_name == 'Xue et. al. (orig.)'
    assert method.name.startswith('DIFFERENTIAL_SVD_XUE_')
