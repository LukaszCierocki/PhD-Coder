import numpy as np
from scipy.fftpack import dct, idct
import random

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'PC Xue [v3]' 

    def calculate_capacity(self, carrier):
        self.logger.info("Calculating capacity for DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V3")
        self.logger.debug(f"Calculated capacity: {len(carrier)}")

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

        level_random = random.randint(1, self.levels)

        self.random_levels.append(level_random)

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, level_random)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, level_random)

        i = 0

        new_levels_evn = []
        new_levels_odd = []

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == level_random - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                self.s_even.append(ro_11)
                self.s_odd.append(ro_21)

                T_bu_1 = self.alpha + 2.0*(ro_21 - ro_11)/(ro_21 + ro_11)
                T_bu_2 = self.beta + 2.0*(ro_11 - ro_21)/(ro_21 + ro_11)

                self.alphas.append(T_bu_1)
                self.betas.append(T_bu_2)

                self.alphas.append(self.alpha)
                self.betas.append(self.beta)

                ro_dash_11 = 0
                ro_dash_21 = 0

                davg = (ro_11 + ro_21) / 2.0

                if bit == 0:

                    if ro_11 - ro_21 >= T_bu_1*davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_21 + T_bu_1*davg
                        ro_dash_21 = ro_21

                elif bit == 1:
                    if (ro_21 - ro_11) >= T_bu_2*davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21 + T_bu_2*davg

                s_evn[s_evn_max_idx] = ro_dash_11
                s_odd[s_odd_max_idx] = ro_dash_21

                self.new_s_even.append(ro_dash_11)
                self.new_s_odd.append(ro_dash_21)

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

        embedded_frame = idct(frame_dct, norm='ortho')

        return embedded_frame

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

        if(len(self.random_levels) != 0):
            random_level = self.random_levels.pop(0)
        else:
            random_level = random.randint(1, self.levels)

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, random_level)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, random_level)

        i = 0

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == random_level - 1:

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