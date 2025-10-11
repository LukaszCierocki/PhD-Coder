import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD

class DIFFERENTIAL_SVD_XUE(DIFFERENTIAL_SVD_XUE_BASED_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)
        self.levels = 0

        self.name = f"DIFFERENTIAL_SVD_XUE_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_{self.type_of_vad}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.normal_name = 'Xue et. al.'


    def calculate_capacity(self, carrier):

        self.logger.debug(f"Calculating capacity for carrier: {self.carrier.name}, length: {len(carrier)}")        
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):

        frame_after_dct = dct(frame, norm='ortho')

        self.logger.debug(f"Frame after DCT: \n{frame_after_dct}")

        filtered_dct_coeffs, idx_low, idx_high = self.filter_dct_coeffs(frame_after_dct, fs, self.FREQ_L, self.FREQ_H)

        self.logger.debug(f"Filtered DCT coefficients: \n{filtered_dct_coeffs}")

        evn_dct_coeffs = []
        odd_dct_coeffs = []

        for i, coeff in enumerate(filtered_dct_coeffs):
            if i % 2 == 0:
                evn_dct_coeffs.append(coeff)
            else:
                odd_dct_coeffs.append(coeff)

        evn_dct_coeffs = np.array(evn_dct_coeffs)
        odd_dct_coeffs = np.array(odd_dct_coeffs)

        self.logger.debug(f"Even DCT coefficients: \n{evn_dct_coeffs}")
        self.logger.debug(f"Odd DCT coefficients: \n{odd_dct_coeffs}")

        evn_matrix, evn_shape = self.vector_to_rectangular(evn_dct_coeffs)
        odd_matrix, odd_shape = self.vector_to_rectangular(odd_dct_coeffs)

        U_evn, s_evn, Vh_evn = svd(evn_matrix, full_matrices=False)
        U_odd, s_odd, Vh_odd = svd(odd_matrix, full_matrices=False)

        self.logger.debug(f"Even SVD singular values: \n{s_evn}")
        self.logger.debug(f"Odd SVD singular values: \n{s_odd}")

        s_evn_max_idx = np.argmax(s_evn)
        s_odd_max_idx = np.argmax(s_odd)

        ro_11 = s_evn[s_evn_max_idx]
        ro_21 = s_odd[s_odd_max_idx]

        T_bu_1 = self.alpha * ro_21
        T_bu_2 = self.beta * ro_11

        self.alphas.append(self.alpha)
        self.betas.append(self.beta)

        self.logger.debug(f"Max even singular value: {ro_11}, Max odd singular value: {ro_21}, T_bu_1: {T_bu_1}, T_bu_2: {T_bu_2}")

        ro_dash_11 = 0
        ro_dash_21 = 0

        self.s_even.append(s_evn[s_evn_max_idx])
        self.s_odd.append(s_odd[s_odd_max_idx])

        if bit == 0:

            self.logger.debug("Embedding bit 0")

            if ro_11 - ro_21 >= T_bu_1:
                self.logger.debug("Condition met for ro_11 - ro_21 >= T_bu_1")
                ro_dash_11 = ro_11
            else:
                self.logger.debug("Condition not met, adjusting ro_dash_11")
                ro_dash_11 = ro_21 + T_bu_1

            ro_dash_21 = ro_21

        elif bit == 1:

            self.logger.debug("Embedding bit 1")

            if ro_21 - ro_11 >= T_bu_2:
                self.logger.debug("Condition met for ro_21 - ro_11 >= T_bu_2")
                ro_dash_21 = ro_21
            else:
                self.logger.debug("Condition not met, adjusting ro_dash_21")
                ro_dash_21 = ro_21 + T_bu_2

            ro_dash_11 = ro_11

        self.new_s_even.append(ro_dash_11)
        self.new_s_odd.append(ro_dash_21)

        s_evn[s_evn_max_idx] = ro_dash_11
        s_odd[s_odd_max_idx] = ro_dash_21

        inverse_svd_evn = U_evn @ np.diag(s_evn) @ Vh_evn
        inverse_svd_odd = U_odd @ np.diag(s_odd) @ Vh_odd

        inverse_svd_evn = inverse_svd_evn.flatten()
        inverse_svd_odd = inverse_svd_odd.flatten()

        coefficents = []

        evn_len = len(inverse_svd_evn)
        odd_len = len(inverse_svd_odd)

        assert len(evn_dct_coeffs) == len(odd_dct_coeffs), "Błąd: Nieparzysta liczba współczynników!"

        for i in range(evn_len + odd_len):  # łącznie 40 elementów
            if i % 2 == 0:
                coefficents.append(inverse_svd_evn[i // 2])
            else:
                coefficents.append(inverse_svd_odd[i // 2])

        coefficents = np.array(coefficents)
        frame_after_dct[idx_low:idx_high] = coefficents

        inverse_dct_frame = idct(frame_after_dct, norm='ortho')

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

        evn_dct_coeffs = np.array(evn_dct_coeffs)
        odd_dct_coeffs = np.array(odd_dct_coeffs)

        evn_matrix, num_padded_evn = self.make_square_matrix_with_padding(evn_dct_coeffs)
        odd_matrix, num_padded_odd = self.make_square_matrix_with_padding(odd_dct_coeffs)

        U_evn, s_evn, Vh_evn = svd(evn_matrix)
        U_odd, s_odd, Vh_odd = svd(odd_matrix)

        s_evn_max_idx = np.argmax(s_evn)
        s_odd_max_idx = np.argmax(s_odd)

        ro_11 = s_evn[s_evn_max_idx]
        ro_21 = s_odd[s_odd_max_idx]

        dj = ro_11 - ro_21

        if dj > 0:
            return 0
        else:
            return 1