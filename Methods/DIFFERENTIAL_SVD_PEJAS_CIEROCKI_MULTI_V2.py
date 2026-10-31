from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    PC Xue [v2] - Algorithm 4 (RSLBE), mode PCXUEv2: like PC Xue [v1] but
    with the adjusted factor alpha_A = alpha - delta for bit 0 (keep threshold
    alpha S_avg / 2). ADAPTIVE and REDUCED are used only with LEGACY=True.
    """

    PAPER_MODE = 'PCXUEv2'
    SUPPORTS_LEGACY = True

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}_REDUCED_{self.reduced}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'PC Xue [v2]'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def modify_singular_values(self, ro_11, ro_21, bit):
        if self.legacy:
            return self.legacy_rule(ro_11, ro_21, bit)
        return self.paper_rule(ro_11, ro_21, bit)

    def legacy_rule(self, ro_11, ro_21, bit):
        """Rule of the code base before 2026-10-31 (differs from Algorithm 4 for bit 1)."""
        davg = (ro_11 + ro_21) / 2.0
        ro_dash_11, ro_dash_21 = ro_11, ro_21

        if not self.reduced:
            denom = self.safe_denominator(ro_21 + ro_11)

            # NOTE: identical for ADAPTIVE True/False (as in the original
            # implementation) - see docs/METHODS.md.
            T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / denom
            T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / denom

            if bit == 0:
                if ro_11 - ro_21 < T_bu_1 * davg:
                    ro_dash_11 = ro_11 + T_bu_1 * davg
            else:
                if ro_21 - ro_11 < T_bu_2 * davg:
                    ro_dash_21 = ro_11 + T_bu_2 * davg

            return ro_dash_11, ro_dash_21, T_bu_1, T_bu_2

        # REDUCED variant
        if bit == 0:
            if ro_11 - ro_21 < self.alpha * davg / 2:
                ro_dash_11 = ro_21 + self.alpha * davg
        else:
            if ro_21 - ro_11 < self.beta * davg / 2:
                ro_dash_21 = ro_11 + self.beta * davg

        return ro_dash_11, ro_dash_21, self.alpha, self.beta
