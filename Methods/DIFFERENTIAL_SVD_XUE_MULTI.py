import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_XUE_MULTI(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)
        
        self.name = f"DIFFERENTIAL_SVD_XUE_MULTI_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"
        
        self.normal_name = 'Xue Et. al. [multi]'
        
        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):

        frame_dct = dct(frame, norm='ortho')

        filtered_dct_coeffs, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = []
        odd_dct_coeffs = []

        for i, coeff in enumerate(filtered_dct_coeffs):
            if i % 2 == 0:
                evn_dct_coeffs.append(coeff)
            else:
                odd_dct_coeffs.append(coeff)

        evn_dct_coeffs = np.array(evn_dct_coeffs)
        odd_dct_coeffs = np.array(odd_dct_coeffs)

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        new_levels_evn = []
        new_levels_odd = []

        i = 0

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):
            
            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                self.s_even.append(ro_11)
                self.s_odd.append(ro_21)

                T_bu_1 = self.alpha * ro_21
                T_bu_2 = self.beta * ro_11

                self.alphas.append(self.alpha)
                self.betas.append(self.beta)

                ro_dash_11 = 0
                ro_dash_21 = 0

                if bit == 0:

                    if ro_11 - ro_21 >= T_bu_1:
                        ro_dash_11 = ro_11
                    else:
                        ro_dash_11 = ro_21 + T_bu_1

                    ro_dash_21 = ro_21

                elif bit == 1:

                    if ro_21 - ro_11 >= T_bu_2:
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_21 = ro_21 + T_bu_2

                    ro_dash_11 = ro_11

                self.new_s_even.append(ro_dash_11)
                self.new_s_odd.append(ro_dash_21)

                s_evn[s_evn_max_idx] = ro_dash_11
                s_odd[s_odd_max_idx] = ro_dash_21

            new_levels_evn.append((U_evn, s_evn, VT_evn))
            new_levels_odd.append((U_odd, s_odd, VT_odd))
            i += 1

        evn_dct_coeffs = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_evn)
        odd_dct_coeffs = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_odd)

        coefficents = []

        evn_len = len(evn_dct_coeffs)
        odd_len = len(odd_dct_coeffs)

        assert len(evn_dct_coeffs) == len(odd_dct_coeffs), "Błąd: Nieparzysta liczba współczynników!"

        for i in range(evn_len + odd_len):
            if i % 2 == 0:
                coefficents.append(evn_dct_coeffs[i // 2])
            else:
                coefficents.append(odd_dct_coeffs[i // 2])

        coefficents = np.array(coefficents)
        frame_dct[idx_low:idx_high] = coefficents

        inverse_dct_frame = idct(frame_dct, norm='ortho')
        return inverse_dct_frame

    def extract_from_frame(self, frame, fs):

        frame_dct = dct(frame, norm='ortho')

        filtered_coeffs, k_low, k_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = []
        odd_dct_coeffs = []

        for i, coeff in enumerate(filtered_coeffs):
            if i % 2 == 0:
                evn_dct_coeffs.append(coeff)
            else:
                odd_dct_coeffs.append(coeff)

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        i = 0

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                dj = ro_11 - ro_21

                if dj > 0:
                    return 0
                else:
                    return 1
                
            i += 1