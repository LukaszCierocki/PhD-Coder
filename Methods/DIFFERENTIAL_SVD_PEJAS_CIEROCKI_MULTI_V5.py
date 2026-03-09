import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V5(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        # Parametr Gamma (domyślny punkt startowy dla auto-tuningu lub stała dla trybu non-adaptive)
        self.gamma = method_arguments.get('gamma', 4.0)

        self.name = f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V5_L_{self.levels}_A_{self.alpha}_B_{self.beta}_G_{self.gamma}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        # Nazwa robocza: Algorithm 6 z Auto-Tuningiem
        self.normal_name = 'PC Xue [v5]' 

    def calculate_capacity(self, carrier):
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):

        frame_dct = dct(frame, norm='ortho')

        # Filtrowanie pasma (zgodnie z ustawieniami)
        filtered_dct_coeffs, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = []
        odd_dct_coeffs = []

        for i, coeff in enumerate(filtered_dct_coeffs):
            if i % 2 == 0:
                evn_dct_coeffs.append(coeff)
            else:
                odd_dct_coeffs.append(coeff)

        # Dekompozycja wielopoziomowa
        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        i = 0
        new_levels_evn = []
        new_levels_odd = []

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            # Modyfikujemy tylko ostatni poziom (najgłębszy)
            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                self.s_even.append(ro_11)
                self.s_odd.append(ro_21)
                
                # Do logów zapisujemy bazowe parametry (w auto-tuningu faktyczna gamma się zmienia)
                self.alphas.append(self.alpha)
                self.betas.append(self.beta)

                ro_dash_11 = 0
                ro_dash_21 = 0
                
                # Średnia (potrzebna do wzorów Algorytmu 6)
                davg = (ro_11 + ro_21) / 2.0

                # -----------------------------------------------------------
                # LOGIKA OSADZANIA V5 (SMART ALGORITHM 6)
                # -----------------------------------------------------------

                if bit == 0:
                    # Cel: ro_11 > ro_21
                    
                    # 1. Obliczamy wymagany próg
                    if self.adaptive:
                        # Używamy ulepszenia z V1 (Adaptive Alpha), żeby wyeliminować błędy bitu 0
                        # Wzór z Lemma 1 dla bitu 0
                        T_bu_1 = self.alpha + 2.0*(ro_21 - ro_11)/(ro_21 + ro_11)
                    else:
                        # Tryb sztywny (zgodny z Algorytmem 6 wprost)
                        T_bu_1 = self.alpha

                    # 2. Sprawdzamy czy modyfikacja jest konieczna
                    if (ro_11 > ro_21) and (ro_11 - ro_21 >= T_bu_1 * davg):
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        # Modyfikacja bezpieczna (przeskok nad sąsiadem)
                        # Gwarantuje spełnienie warunku niezależnie od startowych wartości
                        ro_dash_11 = ro_21 + T_bu_1 * davg
                        ro_dash_21 = ro_21

                elif bit == 1:
                    # Cel: ro_21 > ro_11
                    # Tutaj stosujemy AUTO-TUNING GAMMY (jeśli adaptive=True)
                    
                    if self.adaptive:
                        # --- AUTO-TUNING GAMMA START ---
                        # Szukamy najmniejszej gammy, która "załatwi sprawę"
                        
                        start_gamma = 2.1   # Minimalna gamma (musi być > 2.0 dla stabilności)
                        max_gamma = 12.0    # Górny limit (brutalna siła)
                        step_gamma = 0.5    # Krok precyzji
                        
                        current_gamma = start_gamma
                        found_optimal = False
                        
                        # Inicjalizacja bezpieczna
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_11 # placeholder
                        
                        while current_gamma <= max_gamma:
                            # a) Obliczamy nową wartość wg wzoru Algorytmu 6 dla current_gamma
                            # Wzór: d_hat = d_base + (gamma - 2.0)*diff/2 + beta*avg
                            # Bazujemy na ro_11, bo chcemy je przeskoczyć.
                            # Diff to odległość, którą musimy pokonać.
                            
                            term_gamma = (current_gamma - 2.0) * (abs(ro_11 - ro_21) / 2.0)
                            term_beta = self.beta * davg
                            
                            temp_ro_21 = ro_11 + term_gamma + term_beta
                            
                            # b) Sprawdzamy warunek ekstrakcji (Lemma 1)
                            # Czy nowa różnica jest wystarczająca?
                            # Wymagamy marginesu self.beta * davg (lub ciut mniej w zależności od definicji, 
                            # ale bezpieczniej trzymać margines beta).
                            
                            new_diff = temp_ro_21 - ro_11
                            
                            if new_diff >= self.beta * davg:
                                # SUKCES! Znaleźliśmy optymalną, minimalną gammę.
                                ro_dash_11 = ro_11
                                ro_dash_21 = temp_ro_21
                                found_optimal = True
                                # print(f"Auto-Tuning: Found Gamma={current_gamma}") # Debug
                                break
                            
                            current_gamma += step_gamma
                        
                        if not found_optimal:
                            # Jeśli pętla się skończyła i nic, używamy MAX GAMMA (Brute Force)
                            # lub fallback do logiki V1 (przeskok o stałą wartość).
                            # Tutaj dla spójności z Algorytmem 6 weźmy max gammę.
                            term_gamma = (max_gamma - 2.0) * (abs(ro_11 - ro_21) / 2.0)
                            term_beta = self.beta * davg
                            ro_dash_11 = ro_11
                            ro_dash_21 = ro_11 + term_gamma + term_beta
                            
                        # --- AUTO-TUNING GAMMA END ---
                        
                    else:
                        # Tryb "Sztywny" (Standardowy Algorytm 6 ze stałą Gammą)
                        # To zachowuje się jak V4 (adaptive=False)
                        
                        threshold = (2.0 * self.beta * davg) / self.gamma
                        diff = ro_21 - ro_11
                        
                        if (ro_21 > ro_11) and (diff >= threshold):
                            ro_dash_11 = ro_11
                            ro_dash_21 = ro_21
                        else:
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