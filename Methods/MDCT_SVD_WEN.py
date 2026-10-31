import numpy as np
from scipy.linalg import svd

from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD


class MDCT_SVD_WEN(DIFFERENTIAL_SVD_XUE_BASED_METHOD):
    """
    Wen et al. - MDCT + SVD.

    Frames overlap by 50% (FRAME_LENGTH = 2 x HOP_DURATION_MS) and use a
    Kaiser-Bessel-derived window, so the MDCT is a lapped orthogonal transform
    (TDAC): coefficients modified in one frame are recovered exactly by the
    MDCT of the overlap-added stegobject.

    Frames without a bit are passed through the analysis and synthesis windows
    (w^2) so that the overlap-add stays consistent.

    Rule (Algorithm 1 of Wen et al.): S_avg, S_min = min(s1, s2);
    bit 0: if s1 - s2 < alpha S_min: s1 = S_avg + alpha S_min / 2, s2 = S_avg - alpha S_min / 2
    bit 1: symmetric. Only ALPHA is used (BETA is ignored, as in the paper).
    As published: band 100-3000 Hz, alpha = 0.61 (BASELINES preset).
    """

    FRAMING = 'mdct'

    # Kaiser-Bessel-derived window parameter - used for analysis and synthesis.
    WINDOW_BETA = 4.0

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)
        self.levels = 0

        self.name = f"MDCT_SVD_WEN_L_A_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'Wen Et. Al.'

        self._window_cache = {}

    def calculate_capacity(self, carrier):
        return len(carrier)

    def window(self, length):
        if length not in self._window_cache:
            self._window_cache[length] = self.MDCTProcessor.kaiser_derived(length, beta=self.WINDOW_BETA)
        return self._window_cache[length]

    def coefficient_count(self, frame_samples):
        """A 2N-sample MDCT frame has N coefficients."""
        return frame_samples // 2

    def _band_bins_for(self, n_coeffs, fs, low_freq, high_freq):
        """
        MDCT bin k of a 2N-sample frame is centred at (k + 0.5) * fs / (2N).
        (The old code used fs / N, i.e. it selected FREQ_L/2 .. FREQ_H/2.)
        """
        resolution = fs / (2.0 * n_coeffs)
        k_low = max(0, int(np.floor(low_freq / resolution)))
        k_high = min(n_coeffs, int(np.ceil(high_freq / resolution)))
        return k_low, k_high

    def _mdct(self, frame):
        frame = np.asarray(frame, dtype=np.float64)
        window = self.window(len(frame))
        return window, self.MDCTProcessor.mdct(frame * window)

    def _band(self, frame, fs):
        """Windowed MDCT of ``frame`` and the FREQ_L-FREQ_H band of it."""
        window, frame_mdct = self._mdct(frame)
        idx_low, idx_high = self.band_bins(fs, len(frame_mdct))
        return window, frame_mdct, frame_mdct[idx_low:idx_high], idx_low, idx_high

    def band_energy_ratios_db(self, frames, fs):
        ratios = []
        for frame in np.asarray(frames, dtype=np.float64):
            _, spectrum = self._mdct(frame)
            k_low, k_high = self.band_bins(fs, len(spectrum))
            total = np.sum(spectrum ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios.append(10.0 * np.log10(np.sum(spectrum[k_low:k_high] ** 2) / total))
        return np.array(ratios)

    def embedd_in_frame(self, frame, bit, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        bit : int (0 lub 1) — bit do osadzenia
        fs : int — częstotliwość próbkowania
        """
        window, frame_mdct, band, idx_low, idx_high = self._band(frame, fs)

        evn_coeffs = band[::2]
        odd_coeffs = band[1::2]

        evn_matrix, _, _ = self.vector_to_rectangular_with_padding(evn_coeffs)
        odd_matrix, _, _ = self.vector_to_rectangular_with_padding(odd_coeffs)

        U_evn, s_evn, Vh_evn = svd(evn_matrix, full_matrices=False)
        U_odd, s_odd, Vh_odd = svd(odd_matrix, full_matrices=False)

        s1_idx = int(np.argmax(s_evn))
        s2_idx = int(np.argmax(s_odd))

        s1 = s_evn[s1_idx]
        s2 = s_odd[s2_idx]

        self.s_even.append(s1)
        self.s_odd.append(s2)
        self.alphas.append(self.alpha)
        self.betas.append(self.beta)

        s_avg = (s1 + s2) / 2
        s_min = min(s1, s2)

        if bit == 0:
            if s1 - s2 < self.alpha * s_min:
                s1 = s_avg + (self.alpha * s_min / 2)
                s2 = s_avg - (self.alpha * s_min / 2)
        elif bit == 1:
            if s2 - s1 < self.alpha * s_min:
                s1 = s_avg - (self.alpha * s_min / 2)
                s2 = s_avg + (self.alpha * s_min / 2)

        self.new_s_even.append(s1)
        self.new_s_odd.append(s2)

        s_evn = s_evn.copy()
        s_odd = s_odd.copy()
        s_evn[s1_idx] = s1
        s_odd[s2_idx] = s2

        inv_evn = (U_evn @ np.diag(s_evn) @ Vh_evn).flatten()[:len(evn_coeffs)]
        inv_odd = (U_odd @ np.diag(s_odd) @ Vh_odd).flatten()[:len(odd_coeffs)]

        merged_coeffs = np.empty(len(inv_evn) + len(inv_odd), dtype=np.float64)
        merged_coeffs[::2] = inv_evn
        merged_coeffs[1::2] = inv_odd

        frame_mdct = frame_mdct.copy()
        frame_mdct[idx_low:idx_high] = merged_coeffs

        # IMDCT + synthesis window
        return self.MDCTProcessor.imdct(frame_mdct) * window

    def process_frame_without_embedding(self, frame):
        # Analysis + synthesis window, so the overlap-add of unmodified frames
        # reproduces the carrier (Princen-Bradley condition).
        window = self.window(len(frame))
        return np.asarray(frame, dtype=np.float64) * window * window

    def extract_from_frame(self, frame, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        fs : int — częstotliwość próbkowania
        """
        _, _, band, _, _ = self._band(frame, fs)

        evn_matrix, _, _ = self.vector_to_rectangular_with_padding(band[::2])
        odd_matrix, _, _ = self.vector_to_rectangular_with_padding(band[1::2])

        ro_11 = np.max(svd(evn_matrix, compute_uv=False))
        ro_21 = np.max(svd(odd_matrix, compute_uv=False))

        return 0 if ro_11 - ro_21 > 0 else 1
