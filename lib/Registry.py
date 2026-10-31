"""
Registry of steganographic methods.

Classes are imported lazily, so a missing or broken method file only fails
when that method is actually requested (previously one missing module in the
import list of coder.py stopped the whole program).
"""

import importlib

# class / config name -> module (the class has the same name as the module)
METHOD_MODULES = {
    'DIFFERENTIAL_SVD_XUE': 'Methods.DIFFERENTIAL_SVD_XUE',
    'DIFFERENTIAL_SVD_XUE_MULTI': 'Methods.DIFFERENTIAL_SVD_XUE_MULTI',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE',
    'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED': 'Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED',
    'MDCT_SVD_WEN': 'Methods.MDCT_SVD_WEN',
}

# Old names that may appear in saved test configurations.
ALIASES = {
    'MDCT_SVD_WEN_MULTI': 'MDCT_SVD_WEN',
}


def canonical_name(name):
    return ALIASES.get(name, name)


def get_method_class(name):
    """Returns the method class for a registry / class name."""
    name = canonical_name(name)
    if name not in METHOD_MODULES:
        raise KeyError(f"Unknown method {name!r}. Known methods: {sorted(METHOD_MODULES)}")
    module = importlib.import_module(METHOD_MODULES[name])
    return getattr(module, name)


def create_method(name, params, logger=None):
    return get_method_class(name)(params, logger)


def available_methods():
    """Names of the methods whose modules import correctly."""
    names = []
    for name in METHOD_MODULES:
        try:
            get_method_class(name)
            names.append(name)
        except Exception:
            pass
    return names
