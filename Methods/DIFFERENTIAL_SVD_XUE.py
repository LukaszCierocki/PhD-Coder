import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD


def xue_rule(ro_11, ro_21, bit, alpha, beta):
    """
    Differential SVD rule of Xue et al.

    bit 0: require ro_11 - ro_21 >= alpha * ro_21, otherwise ro_11 := ro_21 + alpha * ro_21
    bit 1: require ro_21 - ro_11 >= beta  * ro_11, otherwise ro_21 := ro_11 + beta  * ro_11

    The previous implementation used ``ro_21 + beta * ro_11`` for bit 1, which
    does not reach the threshold (and is decoded as 0) whenever
    ro_21 < (1 - beta) * ro_11.
    """
    T_bu_1 = alpha * ro_21
    T_bu_2 = beta * ro_11

    ro_dash_11, ro_dash_21 = ro_11, ro_21

    if bit == 0:
        if ro_11 - ro_21 < T_bu_1:
            ro_dash_11 = ro_21 + T_bu_1
    else:
        if ro_21 - ro_11 < T_bu_2:
            ro_dash_21 = ro_11 + T_bu_2

    return ro_dash_11, ro_dash_21


class DIFFERENTIAL_SVD_XUE(DIFFERENTIAL_SVD_XUE_BASED_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)
        self.levels = 0

        self.name = f"DIFFERENTIAL_SVD_XUE_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_{self.type_of_vad}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.normal_name = 'Xue et. al.'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def _branch_matrices(self, frame, fs):
        frame_dct = dct(np.asarray(frame, dtype=np.float64), norm='ortho')
        band, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)
        evn = np.asarray(band[::2], dtype=np.float64)
        odd = np.asarray(band[1::2], dtype=np.float64)
        # The same matrix shape is used for embedding and extraction (the old
        # extractor used a zero-padded square matrix instead).
        evn_matrix, _, _ = self.vector_to_rectangular_with_padding(evn)
        odd_matrix, _, _ = self.vector_to_rectangular_with_padding(odd)
        return frame_dct, idx_low, idx_high, evn, odd, evn_matrix, odd_matrix

    def embedd_in_frame(self, frame, bit, fs):

        frame_dct, idx_low, idx_high, evn, odd, evn_matrix, odd_matrix = self._branch_matrices(frame, fs)

        U_evn, s_evn, Vh_evn = svd(evn_matrix, full_matrices=False)
        U_odd, s_odd, Vh_odd = svd(odd_matrix, full_matrices=False)

        i_evn = int(np.argmax(s_evn))
        i_odd = int(np.argmax(s_odd))

        ro_11 = s_evn[i_evn]
        ro_21 = s_odd[i_odd]

        ro_dash_11, ro_dash_21 = xue_rule(ro_11, ro_21, bit, self.alpha, self.beta)

        self.s_even.append(ro_11)
        self.s_odd.append(ro_21)
        self.new_s_even.append(ro_dash_11)
        self.new_s_odd.append(ro_dash_21)
        self.alphas.append(self.alpha)
        self.betas.append(self.beta)

        s_evn = s_evn.copy()
        s_odd = s_odd.copy()
        s_evn[i_evn] = ro_dash_11
        s_odd[i_odd] = ro_dash_21

        # Padding entries (if any) are dropped after the inverse SVD.
        inv_evn = (U_evn @ np.diag(s_evn) @ Vh_evn).flatten()[:len(evn)]
        inv_odd = (U_odd @ np.diag(s_odd) @ Vh_odd).flatten()[:len(odd)]

        merged_coeffs = np.empty(len(inv_evn) + len(inv_odd), dtype=np.float64)
        merged_coeffs[::2] = inv_evn
        merged_coeffs[1::2] = inv_odd

        frame_dct[idx_low:idx_high] = merged_coeffs

        return idct(frame_dct, norm='ortho')

    def extract_from_frame(self, frame, fs):

        _, _, _, _, _, evn_matrix, odd_matrix = self._branch_matrices(frame, fs)

        ro_11 = np.max(svd(evn_matrix, compute_uv=False))
        ro_21 = np.max(svd(odd_matrix, compute_uv=False))

        return 0 if ro_11 - ro_21 > 0 else 1
