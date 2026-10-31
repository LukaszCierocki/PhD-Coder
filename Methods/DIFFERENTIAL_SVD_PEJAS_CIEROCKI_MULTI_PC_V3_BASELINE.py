import numpy as np
from scipy.fftpack import dct, idct

from lib.Method import MULTI_LEVEL_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE(
        MULTI_LEVEL_METHOD):

    def __init__(self, method_arguments, logger=None):
        super().__init__(method_arguments, logger)

        self.name = (
            f"DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_BASELINE"
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
        self.normal_name = 'PC [v3] baseline'

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

    def set_carrier(self, carrier):
        super().set_carrier(carrier)

        self.reset_adaptive_state()

        if self.logger:
            self.logger.info(
                f"Reset AMLS state for carrier: {carrier.name}; "
                f"stored_levels={len(self.random_levels)}, "
                f"extract_idx={self._extract_idx}"
            )

    def extraction(self, *args, **kwargs):
        self._extract_idx = 0
        return super().extraction(*args, **kwargs)

    def calculate_capacity(self, carrier):
        return len(carrier)

    def _calculate_score_for_level(
            self,
            level,
            evn_dct_coeffs,
            odd_dct_coeffs,
            frame_dct,
            idx_low,
            idx_high,
            frame,
            bit
    ):
        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(
            evn_dct_coeffs,
            level
        )

        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(
            odd_dct_coeffs,
            level
        )

        new_levels_evn = []
        new_levels_odd = []
        stats = {}

        for i, (
                (U_evn, s_evn, VT_evn),
                (U_odd, s_odd, VT_odd)
        ) in enumerate(zip(levels_evn, levels_odd)):

            if i == level - 1:
                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                old_s_even = s_evn[s_evn_max_idx]
                old_s_odd = s_odd[s_odd_max_idx]

                ro_11 = old_s_even
                ro_21 = old_s_odd

                davg = (ro_11 + ro_21) / 2.0
                denom = ro_21 + ro_11

                if denom == 0:
                    denom = 1e-10

                T_bu_1 = (
                    self.alpha
                    + 2.0 * (ro_21 - ro_11) / denom
                )

                T_bu_2 = (
                    self.beta
                    + 2.0 * (ro_11 - ro_21) / denom
                )

                ro_dash_11 = ro_11
                ro_dash_21 = ro_21

                if bit == 0:
                    if ro_11 - ro_21 < T_bu_1 * davg:
                        ro_dash_11 = (
                            ro_11
                            + T_bu_1 * davg / 2.0
                        )

                        ro_dash_21 = (
                            ro_21
                            - T_bu_1 * davg / 2.0
                        )

                elif bit == 1:
                    if ro_21 - ro_11 < T_bu_2 * davg:
                        ro_dash_11 = (
                            ro_11
                            - T_bu_2 * davg / 2.0
                        )

                        ro_dash_21 = (
                            ro_21
                            + T_bu_2 * davg / 2.0
                        )

                s_evn[s_evn_max_idx] = ro_dash_11
                s_odd[s_odd_max_idx] = ro_dash_21

                stats = {
                    'old_s_even': old_s_even,
                    'old_s_odd': old_s_odd,
                    'new_s_even': ro_dash_11,
                    'new_s_odd': ro_dash_21,
                    'alpha_val': T_bu_1,
                    'beta_val': T_bu_2,
                    'ro_dash_11': ro_dash_11,
                    'ro_dash_21': ro_dash_21
                }

            new_levels_evn.append(
                (U_evn, s_evn, VT_evn)
            )

            new_levels_odd.append(
                (U_odd, s_odd, VT_odd)
            )

        rec_evn = (
            self.MultiLevelSVDProcessor.reconstruct_from_levels(
                new_levels_evn
            )
        )

        rec_odd = (
            self.MultiLevelSVDProcessor.reconstruct_from_levels(
                new_levels_odd
            )
        )

        total_len = len(rec_evn) + len(rec_odd)

        coeffs = np.empty(
            total_len,
            dtype=rec_evn.dtype
        )

        coeffs[::2] = rec_evn
        coeffs[1::2] = rec_odd

        full_dct = frame_dct.copy()
        full_dct[idx_low:idx_high] = coeffs

        reconstructed = idct(
            full_dct,
            norm='ortho'
        )

        diff = np.mean(
            np.abs(reconstructed - frame)
        )

        if bit == 0:
            robustness_score = (
                stats['ro_dash_11']
                - stats['ro_dash_21']
            ) / (diff + 1e-8)

        else:
            robustness_score = (
                stats['ro_dash_21']
                - stats['ro_dash_11']
            ) / (diff + 1e-8)

        return (
            robustness_score,
            level,
            coeffs,
            stats
        )

    def embedd_in_frame(self, frame, bit, fs):
        frame_dct = dct(
            frame,
            norm='ortho'
        )

        filtered_dct_coeffs, idx_low, idx_high = (
            self.filter_dct_coeffs(
                frame_dct,
                fs,
                self.FREQ_L,
                self.FREQ_H
            )
        )

        evn_dct_coeffs = filtered_dct_coeffs[::2]
        odd_dct_coeffs = filtered_dct_coeffs[1::2]

        results = []

        for level in range(1, self.levels + 1):
            result = self._calculate_score_for_level(
                level,
                evn_dct_coeffs,
                odd_dct_coeffs,
                frame_dct,
                idx_low,
                idx_high,
                frame,
                bit
            )

            results.append(result)

        best_score, best_level, best_coefficients, best_stats = max(
            results,
            key=lambda result: result[0]
        )

        self.s_even.append(
            best_stats['old_s_even']
        )

        self.s_odd.append(
            best_stats['old_s_odd']
        )

        self.new_s_even.append(
            best_stats['new_s_even']
        )

        self.new_s_odd.append(
            best_stats['new_s_odd']
        )

        self.alphas.append(
            best_stats['alpha_val']
        )

        self.betas.append(
            best_stats['beta_val']
        )

        self.random_levels.append(best_level)

        frame_dct[idx_low:idx_high] = best_coefficients

        embedded_frame = idct(
            frame_dct,
            norm='ortho'
        )

        return embedded_frame

    def extract_from_frame(self, frame, fs):
        frame_dct = dct(
            frame,
            norm='ortho'
        )

        filtered_coeffs, _, _ = self.filter_dct_coeffs(
            frame_dct,
            fs,
            self.FREQ_L,
            self.FREQ_H
        )

        evn_dct_coeffs = filtered_coeffs[::2]
        odd_dct_coeffs = filtered_coeffs[1::2]

        if not self.random_levels:
            raise ValueError(
                "Brak zapisanego poziomu do ekstrakcji."
            )

        if self._extract_idx >= len(self.random_levels):
            self._extract_idx = 0

        level = self.random_levels[self._extract_idx]
        self._extract_idx += 1

        levels_evn = (
            self.MultiLevelSVDProcessor.multi_level_decompose(
                evn_dct_coeffs,
                level
            )
        )

        levels_odd = (
            self.MultiLevelSVDProcessor.multi_level_decompose(
                odd_dct_coeffs,
                level
            )
        )

        _, s_evn, _ = levels_evn[level - 1]
        _, s_odd, _ = levels_odd[level - 1]

        ro_11 = s_evn[np.argmax(s_evn)]
        ro_21 = s_odd[np.argmax(s_odd)]

        return 0 if ro_11 - ro_21 > 0 else 1