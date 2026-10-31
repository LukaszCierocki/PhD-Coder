from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    PC [v1] - Algorithm 4 (RSLBE), mode PCv1: bit 0 moves both singular values
    symmetrically around the odd one (a' = b + alpha S/2, b' = b - alpha S/2),
    bit 1 resets both to the mean and moves them apart by beta S/2.
    Guaranteed margins: alpha (bit 0), beta/2 (bit 1); alpha <= 1 keeps
    b' >= 0. ADAPTIVE is used only with LEGACY=True.

    LEGACY=True: previous rule - bit 0 shifted relative to the current values
    (a' = a + T S/2); with ADAPTIVE=False the sign is not guaranteed and about
    3% of the bits are lost.
    """

    PAPER_MODE = 'PCv1'
    SUPPORTS_LEGACY = True

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        # PC VERSION 1
        self.normal_name = 'PC [v1]'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def modify_singular_values(self, ro_11, ro_21, bit):
        if self.legacy:
            return self.legacy_rule(ro_11, ro_21, bit)
        return self.paper_rule(ro_11, ro_21, bit)

    def legacy_rule(self, ro_11, ro_21, bit):
        """Rule of the code base before 2026-10-31 (differs from Algorithm 4 for bit 0)."""
        davg = (ro_11 + ro_21) / 2.0
        denom = self.safe_denominator(ro_21 + ro_11)

        if self.adaptive:
            T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / denom
        else:
            T_bu_1 = self.alpha
        T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / denom

        ro_dash_11, ro_dash_21 = ro_11, ro_21

        if bit == 0:
            if ro_11 - ro_21 < T_bu_1 * davg:
                # LOGIKA PC [v1]: Symetryczne rozsunięcie względem obecnych wartości
                ro_dash_11 = ro_11 + T_bu_1 * (davg / 2.0)
                ro_dash_21 = ro_21 - T_bu_1 * (davg / 2.0)
        else:
            if ro_21 - ro_11 < T_bu_2 * davg:
                ro_dash_11 = ro_11 - T_bu_2 * (davg / 2.0)
                ro_dash_21 = ro_21 + T_bu_2 * (davg / 2.0)

        return ro_dash_11, ro_dash_21, T_bu_1, T_bu_2
