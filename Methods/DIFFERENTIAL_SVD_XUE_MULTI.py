from lib.EmbeddingRules import xue_rule
from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_XUE_MULTI(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    Xue et al. rule applied on the deepest level of the multi-level Hankel SVD.
    LEGACY=True: Eq. (22) as printed (previous code).
    """

    SUPPORTS_LEGACY = True

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_XUE_MULTI_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.normal_name = 'Xue Et. al. [multi]'

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def modify_singular_values(self, ro_11, ro_21, bit):
        ro_dash_11, ro_dash_21 = xue_rule(ro_11, ro_21, bit, self.alpha, self.beta, as_printed=self.legacy)
        return ro_dash_11, ro_dash_21, self.alpha, self.beta
