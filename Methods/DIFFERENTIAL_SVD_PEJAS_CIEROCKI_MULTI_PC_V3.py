import numpy as np

from lib.Method import DCT_MULTI_LEVEL_SVD_METHOD


class DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3(DCT_MULTI_LEVEL_SVD_METHOD):
    """
    PC [v3] - adaptive multi-level selection (AMLS).

    For every frame the PC [v1] adaptive rule is tried on each decomposition
    depth 1..LEVELS and the depth with the best robustness score

        score = (new singular value margin) / mean|embedded frame - frame|

    is used. The chosen depth of every bit is side information: it is stored in
    the extraction key (``key['levels']``) and read back during extraction.
    The receiver therefore needs ~log2(LEVELS) bits of side information per
    embedded bit (see docs/METHODS.md).

    Thresholds are always adaptive (the ADAPTIVE flag is not used).
    """

    # True -> decompose separately for every candidate depth (original
    # "baseline" implementation, used to compare embedding times).
    DECOMPOSE_PER_LEVEL = False

    VARIANT_NAME = 'DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3'
    VARIANT_NORMAL_NAME = 'PC [v3]'

    def __init__(self, method_arguments, logger=None):
        super().__init__(method_arguments, logger)

        self.name = (
            f"{self.VARIANT_NAME}_L_{self.levels}_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}"
            f"_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}_ADAPTIVE_{self.adaptive}"
        )
        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'
        self.normal_name = self.VARIANT_NORMAL_NAME

    def calculate_capacity(self, carrier):
        return len(carrier)

    def modify_singular_values(self, ro_11, ro_21, bit):

        davg = (ro_11 + ro_21) / 2.0
        denom = self.safe_denominator(ro_21 + ro_11)

        T_bu_1 = self.alpha + 2.0 * (ro_21 - ro_11) / denom
        T_bu_2 = self.beta + 2.0 * (ro_11 - ro_21) / denom

        ro_dash_11, ro_dash_21 = ro_11, ro_21

        if bit == 0:
            if ro_11 - ro_21 < T_bu_1 * davg:
                ro_dash_11 = ro_11 + T_bu_1 * davg / 2
                ro_dash_21 = ro_21 - T_bu_1 * davg / 2
        else:
            if ro_21 - ro_11 < T_bu_2 * davg:
                ro_dash_11 = ro_11 - T_bu_2 * davg / 2
                ro_dash_21 = ro_21 + T_bu_2 * davg / 2

        return ro_dash_11, ro_dash_21, T_bu_1, T_bu_2

    def _candidate(self, level, frame, frame_dct, idx_low, idx_high, levels_evn, levels_odd, bit):
        embedded_frame, stats = self.embed_in_levels(frame_dct, idx_low, idx_high, levels_evn, levels_odd, bit)

        diff = np.mean(np.abs(embedded_frame - frame))
        if bit == 0:
            margin = stats['new_s_even'] - stats['new_s_odd']
        else:
            margin = stats['new_s_odd'] - stats['new_s_even']

        return margin / (diff + 1e-8), level, embedded_frame, stats

    def embedd_in_frame(self, frame, bit, fs):
        frame = np.asarray(frame, dtype=np.float64)

        if self.DECOMPOSE_PER_LEVEL:
            candidates = []
            for level in range(1, self.levels + 1):
                frame_dct, idx_low, idx_high, levels_evn, levels_odd = self.analyse_frame(frame, fs, level)
                candidates.append(self._candidate(level, frame, frame_dct, idx_low, idx_high,
                                                  levels_evn, levels_odd, bit))
        else:
            # One decomposition to the deepest level; depth ``level`` uses the
            # prefix [1..level] (identical to decomposing to ``level`` levels).
            frame_dct, idx_low, idx_high, levels_evn, levels_odd = self.analyse_frame(frame, fs, self.levels)
            candidates = [
                self._candidate(level, frame, frame_dct, idx_low, idx_high,
                                levels_evn[:level], levels_odd[:level], bit)
                for level in range(1, self.levels + 1)
            ]

        # max() keeps the first (shallowest) candidate on ties - as before.
        _, best_level, best_frame, best_stats = max(candidates, key=lambda c: c[0])

        self.record_stats(best_stats)
        self.random_levels.append(best_level)

        return best_frame

    def build_extraction_key(self, n_bits):
        key = super().build_extraction_key(n_bits)
        if len(self.random_levels) != n_bits:
            raise RuntimeError(f"PC v3 stored {len(self.random_levels)} levels for {n_bits} bits.")
        key['levels'] = [int(level) for level in self.random_levels]
        return key

    def extract_bit(self, frame, fs, bit_index, key):
        levels = key.get('levels')
        if not levels:
            raise ValueError("PC v3 extraction key has no 'levels' (decomposition depth of every bit).")
        if bit_index >= len(levels):
            raise ValueError(f"PC v3 extraction key has {len(levels)} levels, bit {bit_index} requested.")
        level = int(levels[bit_index])
        if not 1 <= level <= self.levels:
            raise ValueError(f"Invalid level {level} in PC v3 extraction key (LEVELS = {self.levels}).")

        ro_11, ro_21 = self.singular_values_at_level(frame, fs, level)
        return self.decide_bit(ro_11, ro_21)

    def extract_from_frame(self, frame, fs):
        raise RuntimeError("PC v3 needs the per-bit level map - use extraction(stegobject, key).")
