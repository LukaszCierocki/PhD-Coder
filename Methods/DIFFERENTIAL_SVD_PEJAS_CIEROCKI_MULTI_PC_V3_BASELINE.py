from Methods.DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3 import DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE(DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3):
    """
    PC [v3] with the original (unoptimised) search: every candidate depth is
    decomposed from scratch. Produces the same stegobject as PC [v3]; kept to
    measure the speed-up of the single-decomposition search.
    """

    DECOMPOSE_PER_LEVEL = True

    VARIANT_NAME = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE'
    VARIANT_NORMAL_NAME = 'PC [v3] baseline'
