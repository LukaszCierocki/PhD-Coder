from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    PC [v2] - Algorithm 4 (RSLBE), mode PCv2: both singular values are reset
    to the mean S_avg and moved apart by alpha S/2 (bit 0) or beta S/2
    (bit 1). Keep thresholds alpha S/2, beta S/2; guaranteed margins alpha/2,
    beta/2.

    LEGACY=True: previous rule - the reset used the adjusted factors
    alpha_A, beta_A instead of alpha, beta (adaptation applied twice).
    """

    PAPER_MODE = 'PCv2'
    SUPPORTS_LEGACY = True

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'PC [v2]'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def modify_singular_values(self, ro_11, ro_21, bit):
        if self.legacy:
            return self.legacy_rule(ro_11, ro_21, bit)
        return self.paper_rule(ro_11, ro_21, bit)

    def legacy_rule(self, ro_11, ro_21, bit):
        """Rule of the code base before 2026-10-31 (adjusted factors used in the reset)."""
        davg = (ro_11 + ro_21) / 2.0
        denom = self.safe_denominator(ro_21 + ro_11)

        # NOTE: identical for ADAPTIVE True/False (as in the original
        # implementation) - see docs/METHODS.md.
        T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / denom
        T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / denom

        ro_dash_11, ro_dash_21 = ro_11, ro_21

        if bit == 0:
            if ro_11 - ro_21 < T_bu_1 * davg:
                # LOGIKA PC [v2] (Algorytm 4): reset do średniej i rozsunięcie o próg
                ro_dash_11 = davg + T_bu_1 * (davg / 2.0)
                ro_dash_21 = davg - T_bu_1 * (davg / 2.0)
        else:
            if ro_21 - ro_11 < T_bu_2 * davg:
                ro_dash_11 = davg - T_bu_2 * (davg / 2.0)
                ro_dash_21 = davg + T_bu_2 * (davg / 2.0)

        return ro_dash_11, ro_dash_21, T_bu_1, T_bu_2
