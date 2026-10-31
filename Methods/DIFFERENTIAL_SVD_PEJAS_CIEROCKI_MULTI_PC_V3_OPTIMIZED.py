import numpy as np
from scipy.fftpack import dct, idct
# import random
# from joblib import Parallel, delayed  # <--- wcześniejsza równoległość

from lib.Method import MULTI_LEVEL_METHOD

class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED(MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):
        super().__init__(method_arguments, logger)

        self.name = (
            f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_OPTIMIZED"
            f"_L_{self.levels}"
            f"_A_{self.alpha}"
            f"_B_{self.beta}"
            f"_{self.where_to_hide}"
            f"_FREQ_L_{self.FREQ_L}"
            f"_FREQ_H_{self.FREQ_H}"
            f"_FL_{self.frame_length}"
            f"_ADAPTIVE_{self.adaptive}"
        )

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'
        self.normal_name = 'PC [v3] optimized'
        self.random_levels = []
        
        self.n_jobs = -1

    def set_carrier(self, carrier):
        super().set_carrier(carrier)
        self.reset_adaptive_state()

    def reset_adaptive_state(self):
        self.random_levels = []
        self._extract_idx = 0

        self.s_even = []
        self.s_odd = []
        self.new_s_even = []
        self.new_s_odd = []
        self.alphas = []
        self.betas = []

    def extraction(self, *args, **kwargs):
        self._extract_idx = 0
        return super().extraction(*args, **kwargs)

    def calculate_capacity(self, carrier):
        return len(carrier)

    def _calculate_score_for_level(self, level, levels_evn_full, levels_odd_full, frame_dct, idx_low, idx_high, frame, bit):
        """
        Oblicza wynik (robustness_score) dla danego poziomu.
        Po przebudowie metoda korzysta z wcześniej policzonej dekompozycji do
        najgłębszego poziomu i bierze tylko prefiks [1..level] z tej samej struktury.
        """
        # OLD:
        # levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, level)
        # levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, level)
        levels_evn = [(U, s.copy(), VT) for U, s, VT in levels_evn_full[:level]]
        levels_odd = [(U, s.copy(), VT) for U, s, VT in levels_odd_full[:level]]

        new_levels_evn = []
        new_levels_odd = []
        
        # Zmienne do przechowywania statystyk (dla późniejszego zapisu w self.s_even itp.)
        stats = {}

        # 2. Modyfikacja współczynników
        for i, ((U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd)) in enumerate(zip(levels_evn, levels_odd)):
            if i == level - 1:
                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)
                
                old_s_even = s_evn[s_evn_max_idx]
                old_s_odd = s_odd[s_odd_max_idx]
                
                ro_11 = old_s_even
                ro_21 = old_s_odd

                davg = (ro_11 + ro_21) / 2.0
                # Zabezpieczenie przed dzieleniem przez zero, choć przy SVD rzadkie
                denom = (ro_21 + ro_11) if (ro_21 + ro_11) != 0 else 1e-10
                
                T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / denom
                T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / denom

                ro_dash_11 = ro_11
                ro_dash_21 = ro_21

                if bit == 0:
                    if ro_11 - ro_21 >= T_bu_1 * davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_11 + T_bu_1 * davg / 2
                        ro_dash_21 = ro_21 - T_bu_1 * davg / 2
                elif bit == 1:
                    if (ro_21 - ro_11) >= T_bu_2 * davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_11 - T_bu_2 * davg / 2
                        ro_dash_21 = ro_21 + T_bu_2 * davg / 2

                s_evn[s_evn_max_idx] = ro_dash_11
                s_odd[s_odd_max_idx] = ro_dash_21
                
                # Zapisujemy statystyki do słownika
                stats['old_s_even'] = old_s_even
                stats['old_s_odd'] = old_s_odd
                stats['new_s_even'] = ro_dash_11
                stats['new_s_odd'] = ro_dash_21
                stats['alpha_val'] = T_bu_1
                stats['beta_val'] = T_bu_2
                stats['ro_dash_11'] = ro_dash_11
                stats['ro_dash_21'] = ro_dash_21

            new_levels_evn.append((U_evn, s_evn, VT_evn))
            new_levels_odd.append((U_odd, s_odd, VT_odd))

        rec_evn = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_evn)
        rec_odd = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_odd)

        # OLD:
        # coeffs = []
        # total_len = len(rec_evn) + len(rec_odd)
        # for k in range(total_len):
        #     coeffs.append(rec_evn[k // 2] if k % 2 == 0 else rec_odd[k // 2])
        total_len = len(rec_evn) + len(rec_odd)
        coeffs = np.empty(total_len, dtype=rec_evn.dtype)
        coeffs[::2] = rec_evn
        coeffs[1::2] = rec_odd

        full_dct = frame_dct.copy()
        full_dct[idx_low:idx_high] = coeffs
        reconstructed = idct(full_dct, norm='ortho')

        diff = np.mean(np.abs(reconstructed - frame))
        
        if bit == 0:
            robustness_score = (stats['ro_dash_11'] - stats['ro_dash_21']) / (diff + 1e-8)
        else:
            robustness_score = (stats['ro_dash_21'] - stats['ro_dash_11']) / (diff + 1e-8)

        return (robustness_score, level, coeffs, stats)

    def embedd_in_frame(self, frame, bit, fs):
        frame_dct = dct(frame, norm='ortho')
        filtered_dct_coeffs, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = filtered_dct_coeffs[::2]
        odd_dct_coeffs = filtered_dct_coeffs[1::2]

        # OLD:
        # results = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self._calculate_score_for_level)(
        #         level, evn_dct_coeffs, odd_dct_coeffs, frame_dct, idx_low, idx_high, frame, bit
        #     ) for level in range(1, self.levels + 1)
        # )
        # Nowa wersja: jedna dekompozycja do najgłębszego poziomu, a potem
        # testowanie kandydatów na tej samej strukturze.
        levels_evn_full = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd_full = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        results = []
        for level in range(1, self.levels + 1):
            results.append(
                self._calculate_score_for_level(
                    level, levels_evn_full, levels_odd_full, frame_dct, idx_low, idx_high, frame, bit
                )
            )

        best_result = max(results, key=lambda x: x[0])
        
        best_score, best_level, best_coefficients, best_stats = best_result

        self.s_even.append(best_stats['old_s_even'])
        self.s_odd.append(best_stats['old_s_odd'])
        self.new_s_even.append(best_stats['new_s_even'])
        self.new_s_odd.append(best_stats['new_s_odd'])
        self.alphas.append(best_stats['alpha_val'])
        self.betas.append(best_stats['beta_val'])
        
        self.random_levels.append(best_level)

        frame_dct[idx_low:idx_high] = best_coefficients
        embedded_frame = idct(frame_dct, norm='ortho')
        
        return embedded_frame

    def extract_from_frame(self, frame, fs):
        frame_dct = dct(frame, norm='ortho')
        filtered_coeffs, k_low, k_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)

        evn_dct_coeffs = filtered_coeffs[::2]
        odd_dct_coeffs = filtered_coeffs[1::2]

        if len(self.random_levels) == 0:
            raise ValueError("Brak zapisanego poziomu do ekstrakcji.")

        if not hasattr(self, '_extract_idx'):
            # initialize on first extraction
            self._extract_idx = 0

        # wrap-around to allow repeated extractions using same sequence
        if self._extract_idx >= len(self.random_levels):
            self._extract_idx = 0

        level = self.random_levels[self._extract_idx]
        self._extract_idx += 1

        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, level)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, level)

        Ue, se, VTe = levels_evn[level - 1]
        Uo, so, VTo = levels_odd[level - 1]

        ro_11 = se[np.argmax(se)]
        ro_21 = so[np.argmax(so)]

        return 0 if (ro_11 - ro_21) > 0 else 1