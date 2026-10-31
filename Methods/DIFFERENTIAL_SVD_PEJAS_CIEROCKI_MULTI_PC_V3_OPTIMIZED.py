from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED(DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3):
    """
    PC [v3] with the single-decomposition search (same algorithm as PC [v3],
    separate name so both can appear side by side in a comparison).
    """

    DECOMPOSE_PER_LEVEL = False

    VARIANT_NAME = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED'
    VARIANT_NORMAL_NAME = 'PC [v3] optimized'
