import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.EmbeddingRules import xue_rule  # noqa: F401  (re-exported for older imports)
from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD


class DIFFERENTIAL_SVD_XUE(DIFFERENTIAL_SVD_XUE_BASED_METHOD):
    """
    Xue et al. (2019): DCT band, even/odd coefficients reshaped to matrices,
    differential rule on the largest singular values (lib.EmbeddingRules.xue_rule).

    LEGACY=True uses Eq. (22) exactly as printed (b' = b + beta a), which the
    previous code did. As published, Xue et al. embed in every frame
    (WHERE_TO_HIDE='ALL') with alpha = beta = 0.2 (BASELINES preset).
    """

    SUPPORTS_LEGACY = True

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

        ro_dash_11, ro_dash_21 = xue_rule(ro_11, ro_21, bit, self.alpha, self.beta, as_printed=self.legacy)

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
