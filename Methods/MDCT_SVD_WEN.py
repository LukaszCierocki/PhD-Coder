import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD

class MDCT_SVD_WEN(DIFFERENTIAL_SVD_XUE_BASED_METHOD):
    
    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)
        self.levels = 0

        self.name = f"MDCT_SVD_WEN_L_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'Wen Et. Al.'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        bit : int (0 lub 1) — bit do osadzenia
        fs : int — częstotliwość próbkowania
        """
        # Konwersja do float [-1, 1]
        
        # frame = frame.astype(np.float32) / 32768.0
        # self.logger.debug(f"Frame after normalization: \n{frame}")
        
        window = self.MDCTProcessor.kaiser_derived(len(frame), beta = 4.0)
        frame = frame * window

        self.logger.debug(f"Frame after windowing: \n{frame}")
        
        # MDCT
        frame_mdct = self.MDCTProcessor.mdct(frame)

        self.logger.debug(f"Frame after MDCT: \n{frame_mdct}")

        # Filtracja w paśmie
        filtered_mdct_coeffs, idx_low, idx_high = self.MDCTProcessor.filter_mdct_coeffs(
            frame_mdct, fs, len(frame_mdct), self.FREQ_L, self.FREQ_H
        )

        self.logger.debug(f"Filtered MDCT coefficients: \n{filtered_mdct_coeffs}")

        # Podział na parzyste i nieparzyste
        evn_dct_coeffs = filtered_mdct_coeffs[::2]
        odd_dct_coeffs = filtered_mdct_coeffs[1::2]

        evn_matrix, padded_evn, evn_shape = self.vector_to_rectangular_with_padding(evn_dct_coeffs)
        odd_matrix, padded_odd, odd_shape = self.vector_to_rectangular_with_padding(odd_dct_coeffs)

        self.logger.debug(f"Even matrix: \n{evn_matrix}")
        self.logger.debug(f"Odd matrix: \n{odd_matrix}")

        self.logger.debug(f"Even matrix shape: {evn_shape}, Odd matrix shape: {odd_shape}")
        self.logger.debug(f"Padded even coefficients: \n{padded_evn}")
        self.logger.debug(f"Padded odd coefficients: \n{padded_odd}")

        # SVD
        U_evn, s_evn, Vh_evn = svd(evn_matrix, full_matrices=False)
        U_odd, s_odd, Vh_odd = svd(odd_matrix, full_matrices=False)

        # Największe wartości osobliwe
        s1_idx = np.argmax(s_evn)
        s2_idx = np.argmax(s_odd)

        s1 = s_evn[s1_idx]
        s2 = s_odd[s2_idx]

        self.logger.debug(f"Singular values before modification: s1 = {s1}, s2 = {s2}")
        
        s_avg = (s1 + s2) / 2
        s_min = min(s1, s2)

        self.s_even.append(s1)
        self.s_odd.append(s2)

        self.alphas.append(self.alpha)
        self.betas.append(self.beta)


        # Modyfikacja
        if bit == 0:
            if s1 - s2 < self.alpha * s_min:
                s1 = s_avg + (self.alpha * s_min / 2)
                s2 = s_avg - (self.alpha * s_min / 2)
        elif bit == 1:
            if s2 - s1 < self.alpha * s_min:
                s1 = s_avg - (self.alpha * s_min / 2)
                s2 = s_avg + (self.alpha * s_min / 2)

        self.logger.debug(f"Modified singular values: s1 = {s1}, s2 = {s2}")

        self.new_s_even.append(s1)
        self.new_s_odd.append(s2)

        # Podstawienie
        s_evn[s1_idx] = s1
        s_odd[s2_idx] = s2

        # Rekonstrukcja
        inv_evn = (U_evn @ np.diag(s_evn) @ Vh_evn).flatten()[:len(evn_dct_coeffs)]
        inv_odd = (U_odd @ np.diag(s_odd) @ Vh_odd).flatten()[:len(odd_dct_coeffs)]

        self.logger.debug(f"Inverse even coefficients: \n{inv_evn}")
        self.logger.debug(f"Inverse odd coefficients: \n{inv_odd}")

        inv_evn = inv_evn[:len(evn_dct_coeffs)]
        inv_odd = inv_odd[:len(odd_dct_coeffs)]

        # Scalanie
        merged_coeffs = np.empty(len(inv_evn) + len(inv_odd), dtype=evn_dct_coeffs.dtype)
        merged_coeffs[::2] = inv_evn
        merged_coeffs[1::2] = inv_odd

        self.logger.debug(f"Merged coefficients: \n{merged_coeffs}")

        # Wstawienie
        frame_mdct[idx_low:idx_high] = merged_coeffs

        self.logger.debug(f"Frame MDCT after embedding: \n{frame_mdct}")

        # IMDCT
        embedded_frame = self.MDCTProcessor.imdct(frame_mdct)

        self.logger.debug(f"Embedded frame after IMDCT: \n{embedded_frame}")

        # Okno syntezy
        embedded_frame = embedded_frame * window

        self.logger.debug(f"Embedded frame after windowing: \n{embedded_frame}")

        return embedded_frame

    def extract_from_frame(self, frame, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        fs : int — częstotliwość próbkowania
        """
        # Konwersja do float [-1, 1], jeśli frame jest int16
        # if frame.dtype == np.int16:
        #     frame = frame.astype(np.float32) / 32768.0

        window = self.MDCTProcessor.kaiser_derived(len(frame), beta=4.0)
        frame = frame * window
        
        # MDCT
        frame_mdct = self.MDCTProcessor.mdct(frame)

        # Filtracja w paśmie
        filtered_mdct_coeffs, idx_low, idx_high = self.MDCTProcessor.filter_mdct_coeffs(
            frame_mdct, fs, len(frame_mdct), self.FREQ_L, self.FREQ_H
        )

        # Podział na parzyste i nieparzyste
        evn_dct_coeffs = filtered_mdct_coeffs[::2]
        odd_dct_coeffs = filtered_mdct_coeffs[1::2]

        # Konwersja do macierzy
        evn_matrix, _, _ = self.vector_to_rectangular_with_padding(evn_dct_coeffs)
        odd_matrix, _, _ = self.vector_to_rectangular_with_padding(odd_dct_coeffs)

        # SVD
        U_evn, s_evn, Vh_evn = svd(evn_matrix, full_matrices=False)
        U_odd, s_odd, Vh_odd = svd(odd_matrix, full_matrices=False)

        # Największe wartości osobliwe
        ro_11 = np.max(s_evn)
        ro_21 = np.max(s_odd)

        dj = ro_11 - ro_21

        # Decyzja
        extracted_bit = 0 if dj > 0 else 1

        return extracted_bit