from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    PC [v1] - symmetric shift of both singular values relative to their
    current values by ``T * davg / 2``.

    With ADAPTIVE=True the shift always produces the requested sign
    (new difference = ALPHA * davg). With ADAPTIVE=False the shift for bit 0
    is fixed (ALPHA * davg) and may be too small when the odd singular value
    is much larger - such bits are extracted wrongly by design.
    """

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
