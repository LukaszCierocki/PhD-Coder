import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd
import random

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'PC [v3]'

        self.random_levels = []

    def calculate_capacity(self, carrier):
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):
        frame_dct = dct(frame, norm='ortho')
        filtered_dct_coeffs, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = filtered_dct_coeffs[::2]
        odd_dct_coeffs = filtered_dct_coeffs[1::2]

        best_level = None
        lowest_diff = float('inf')
        best_score = -np.inf
        best_coefficients = None

        for level in range(1, self.levels + 1):
            levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, level)
            levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, level)

            new_levels_evn = []
            new_levels_odd = []

            for i, ((U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd)) in enumerate(zip(levels_evn, levels_odd)):

                if i == level - 1:
                    s_evn_max_idx = np.argmax(s_evn)
                    s_odd_max_idx = np.argmax(s_odd)
                    old_s_even = s_evn[s_evn_max_idx]
                    old_s_odd = s_odd[s_odd_max_idx]
                    ro_11 = s_evn[s_evn_max_idx]
                    ro_21 = s_odd[s_odd_max_idx]

                    davg = (ro_11 + ro_21) / 2.0
                    T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / (ro_21 + ro_11)
                    T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / (ro_21 + ro_11)

                    qwe = T_bu_1
                    asd = T_bu_2

                    if bit == 0:
                        if ro_11 - ro_21 >= T_bu_1 * davg:
                            ro_dash_11 = ro_11
                            ro_dash_21 = ro_21
                        else:
                            ro_dash_11 = ro_21 + T_bu_1 * davg / 2
                            ro_dash_21 = ro_21 - T_bu_1 * davg / 2
                    else:
                        if ro_21 - ro_11 >= T_bu_2 * davg:
                            ro_dash_11 = ro_11
                            ro_dash_21 = ro_21
                        else:
                            ro_dash_11 = ro_11 - T_bu_2 * davg / 2
                            ro_dash_21 = ro_21 + T_bu_2 * davg / 2

                    s_evn[s_evn_max_idx] = ro_dash_11
                    s_odd[s_odd_max_idx] = ro_dash_21

                    new_s_even = ro_dash_11
                    new_s_odd = ro_dash_21

                new_levels_evn.append((U_evn, s_evn, VT_evn))
                new_levels_odd.append((U_odd, s_odd, VT_odd))

            rec_evn = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_evn)
            rec_odd = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_odd)

            coeffs = []
            for i in range(len(filtered_dct_coeffs)):
                coeffs.append(rec_evn[i // 2] if i % 2 == 0 else rec_odd[i // 2])
            coeffs = np.array(coeffs)

            full_dct = frame_dct.copy()
            full_dct[idx_low:idx_high] = coeffs
            reconstructed = idct(full_dct, norm='ortho')

            diff = np.mean(np.abs(reconstructed - frame))

            if bit == 0:
                robustness_score = (ro_dash_11 - ro_dash_21) / (diff + 1e-8)
            else:
                robustness_score = (ro_dash_21 - ro_dash_11) / (diff + 1e-8)

            if robustness_score > best_score:
                best_score = robustness_score
                best_level = level
                self.s_even.append(old_s_even)
                self.s_odd.append(old_s_odd)
                self.new_s_even.append(new_s_even)
                self.new_s_odd.append(new_s_odd)

                self.alphas.append(T_bu_1)
                self.betas.append(T_bu_2)

                best_coefficients = coeffs

        self.random_levels.append(best_level)

        frame_dct[idx_low:idx_high] = best_coefficients
        embedded_frame = idct(frame_dct, norm='ortho')
        return embedded_frame

    def extract_from_frame(self, frame, fs):
        frame_dct = dct(frame, norm='ortho')
        filtered_coeffs, k_low, k_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = filtered_coeffs[::2]
        odd_dct_coeffs = filtered_coeffs[1::2]

        if len(self.random_levels) != 0:
            level = self.random_levels.pop(0)
        else:
            raise ValueError("Brak zapisanego poziomu do ekstrakcji.")

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, level)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, level)

        Ue, se, VTe = levels_evn[level - 1]
        Uo, so, VTo = levels_odd[level - 1]

        ro_11 = se[np.argmax(se)]
        ro_21 = so[np.argmax(so)]

        return 0 if (ro_11 - ro_21) > 0 else 1