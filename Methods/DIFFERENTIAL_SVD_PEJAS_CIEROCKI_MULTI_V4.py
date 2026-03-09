import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V4(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        # Pobieranie parametru GAMMA (domyślnie 4.0, zgodnie z artykułem)
        self.gamma = method_arguments.get('gamma', 4.0)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V4_L_{self.levels}_A_{self.alpha}_B_{self.beta}_G_{self.gamma}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        # Nazwa zgodna z artykułem (Reduced form / Algorithm 6)
        self.normal_name = 'PC Xue [v4]' 

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

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        i = 0

        new_levels_evn = []
        new_levels_odd = []

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx] # d_L1 (większa, jeśli posortowane naturalnie w ramce, ale tu bierzemy max)
                ro_21 = s_odd[s_odd_max_idx] # d_L2

                # W tej metodzie zakładamy, że pracujemy na "naturalnie" większej i mniejszej wartości
                # Ale w kodzie ro_11 i ro_21 to po prostu wartości z parzystych/nieparzystych
                
                self.s_even.append(ro_11)
                self.s_odd.append(ro_21)
                
                # Zapisujemy parametry do logów
                self.alphas.append(self.alpha)
                self.betas.append(self.beta) # Gamma jest stała dla metody, nie logujemy per ramka (chyba że zrobisz adaptive gamma)

                ro_dash_11 = 0
                ro_dash_21 = 0
                
                # S_avg obliczane zgodnie z Algorytmem 6
                davg = (ro_11 + ro_21) / 2.0

                if bit == 0:
                    # LOGIKA DLA BITU 0 (Zgodna z Algorithm 6 / PC Xue [v1])
                    # Warunek: (d1 - d2) >= alpha * S_avg
                    # Tutaj ro_11 ma być większe (bit 0)
                    
                    # Sprawdzenie czy już jest dobrze
                    if (ro_11 > ro_21) and (ro_11 - ro_21 >= self.alpha * davg):
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        # Modyfikacja zgodnie z Algorytmem 6 (str. 29 PDF):
                        # d_hat_L1 = d_L1 + alpha * S_avg
                        # Uwaga: Jeśli naturalnie ro_21 > ro_11, to musimy "przeskoczyć" ro_21.
                        # Wersja bezpieczna (jak w V1): Baza to konkurent (ro_21)
                        ro_dash_11 = ro_21 + self.alpha * davg
                        ro_dash_21 = ro_21 

                elif bit == 1:
                    # LOGIKA DLA BITU 1 (Z PARAMETREM GAMMA - KLUCZ METODY)
                    # Zgodnie z Algorytmem 6 (str. 29 PDF):
                    # Warunek: (d_L1 - d_L2) >= 2.0 * beta * S_avg / gamma
                    # Tutaj chcemy, żeby ro_21 (odd) było większe.
                    
                    diff = ro_21 - ro_11 # Różnica "na naszą korzyść"
                    
                    # Obliczenie progu z użyciem Gammy
                    threshold = (2.0 * self.beta * davg) / self.gamma
                    
                    if (ro_21 > ro_11) and (diff >= threshold):
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        # Modyfikacja GAMMA (Algorytm 6):
                        # d_hat_L2 = d_L1 + (gamma - 2.0)*(d_L1 - d_L2)/2.0 + beta * S_avg
                        # W naszym przypadku "d_L1" (baza) to ro_11 (bo chcemy je przeskoczyć)
                        # A różnica (d_L1 - d_L2) to tutaj (ro_11 - ro_21) - dystans do pokonania
                        
                        # Ponieważ ro_11 może być > ro_21 (sytuacja przeciwna do bitu), 
                        # to distance = abs(ro_11 - ro_21).
                        # Wzór z artykułu zakłada dL1 > dL2, więc weźmy po prostu ro_11 jako bazę.
                        
                        term_gamma = (self.gamma - 2.0) * (abs(ro_11 - ro_21) / 2.0)
                        term_beta = self.beta * davg
                        
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_11 + term_gamma + term_beta

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