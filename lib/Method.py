"""
Base classes and signal-processing helpers shared by all steganographic methods.

Design rules enforced here (see docs/METHODS.md):

* every parameter is validated when the method is created (static checks)
  and again when a carrier is set (checks that need the sampling rate);
* all per-carrier state is reset in ``set_carrier`` and at the start of
  ``embedding`` -- re-using one method object for many carriers is safe;
* everything the receiver needs besides the stegobject (VAD mask, number of
  embedded bits, method specific side information such as the PC v3 level
  map) is stored explicitly in an *extraction key* (``self.extraction_key``),
  which can be saved to JSON and passed back to ``extraction``;
* the stegobject is rounded (not truncated) and clipped to int16, and the
  number of clipped samples is reported;
* debug logging never formats numpy arrays in hot loops (an eagerly evaluated
  f-string used to dominate the measured embedding time).
"""

import logging
import math
from numbers import Real
from timeit import default_timer as timer

import numpy as np
from scipy.fftpack import dct, idct, fft, ifft
from scipy.linalg import svd
from scipy.signal import lfilter
from scipy.signal.windows import kaiser
from BitVector import BitVector

from lib import speechproc

try:  # webrtcvad is optional - only needed for TYPE_OF_VAD = 'webrtc'
    import webrtcvad
except ImportError:  # pragma: no cover - depends on the environment
    webrtcvad = None


INT16_MIN = -32768
INT16_MAX = 32767

SUPPORTED_VADS = ('rVAD', 'webrtc', 'webrtcVAD')
SUPPORTED_HIDING_PLACES = ('VOICED',)

_NULL_LOGGER = logging.getLogger('phd_coder.null')
_NULL_LOGGER.addHandler(logging.NullHandler())
_NULL_LOGGER.propagate = False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def ensure_logger(logger):
    """Returns ``logger`` or a silent logger, so methods never crash on ``None``."""
    return logger if logger is not None else _NULL_LOGGER


def ms_to_samples(duration_ms, fs, what='frame'):
    """
    Converts a duration in milliseconds to an exact number of samples.

    Raises ``ValueError`` when the duration does not correspond to an integer
    number of samples (e.g. 10 ms at 22050 Hz), because the VAD (rVAD uses
    ``fix(fs * t)``) and the extractor would then disagree about frame borders
    and the nominal capacity in bps would be wrong.
    """
    exact = float(fs) * float(duration_ms) / 1000.0
    samples = int(round(exact))
    if samples < 1:
        raise ValueError(f"{what} of {duration_ms} ms at {fs} Hz is shorter than one sample.")
    if abs(exact - samples) > 1e-6:
        raise ValueError(
            f"{what} of {duration_ms} ms at {fs} Hz is {exact:.3f} samples - not an integer. "
            f"Choose a duration that gives an integer number of samples."
        )
    return samples


def to_int16(signal):
    """
    Rounds a floating point signal to the nearest integer and clips it to the
    int16 range.

    Returns ``(int16_signal, number_of_clipped_samples)``. Raises ``ValueError``
    if the signal contains NaN or inf (that always indicates a bug upstream).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        bad = int(np.count_nonzero(~np.isfinite(signal)))
        raise ValueError(f"Signal contains {bad} non-finite samples (NaN/inf).")
    rounded = np.rint(signal)
    clipped = int(np.count_nonzero((rounded < INT16_MIN) | (rounded > INT16_MAX)))
    return np.clip(rounded, INT16_MIN, INT16_MAX).astype(np.int16), clipped


def encode_mask(mask):
    """Encodes a boolean/0-1 sequence as a compact '0101...' string (for JSON keys)."""
    return ''.join('1' if bool(v) else '0' for v in mask)


def decode_mask(mask):
    """Inverse of :func:`encode_mask`; also accepts a list of bools/ints."""
    if isinstance(mask, str):
        if set(mask) - {'0', '1'}:
            raise ValueError("Voiced mask string may contain only '0' and '1'.")
        return [c == '1' for c in mask]
    return [bool(v) for v in mask]


def _require_number(arguments, key, positive=False, non_negative=False):
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Parameter {key} must be a finite number, got {value!r}.")
    if positive and value <= 0:
        raise ValueError(f"Parameter {key} must be > 0, got {value!r}.")
    if non_negative and value < 0:
        raise ValueError(f"Parameter {key} must be >= 0, got {value!r}.")
    return value


def _require_int(arguments, key, minimum):
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"Parameter {key} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _require_bool(arguments, key):
    value = arguments.get(key)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Parameter {key} must be True or False, got {value!r}.")
    return bool(value)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

class Method:

    def __init__(self, method_arguments, logger=None):

        if not isinstance(method_arguments, dict):
            raise TypeError("method_arguments must be a dict.")

        self.logger = ensure_logger(logger)

        # Copy, so the caller's dict is never mutated and defaults are recorded
        # in the saved test configuration.
        self.method_arguments = dict(method_arguments)

        self.name = None
        self.normal_name = None

        self.capacity = None

        self.embedding_time = None
        self.extraction_time = None

        self.domain = None
        self.style = None

        self.method_directories = None

        self.carrier = None
        self.secret = None

        self.extraction_key = None
        self.clipped_samples = 0
        self.embedded_bits = 0

    def set_carrier(self, carrier):
        """
        Sets the carrier for the method.

        Parameters:
        - carrier: Carrier object, the carrier to be used in the method
        """
        self.carrier = carrier

    def set_secret(self, secret):
        """
        Sets the secret for the method.

        Parameters:
        - secret: Secret object, the secret to be used in the method
        """
        if secret is None or getattr(secret, 'binary_data', None) is None:
            raise ValueError("Secret has no binary data.")
        if len(secret.binary_data) == 0:
            raise ValueError("Secret is empty (0 bits) - nothing to embed.")
        if self.capacity is not None and len(secret.binary_data) > self.capacity:
            raise ValueError(
                f"Secret has {len(secret.binary_data)} bits but the carrier capacity of "
                f"{self.name} is only {self.capacity} bits."
            )
        self.secret = secret

    def embedding(self):
        raise NotImplementedError("Embedding method must be implemented by subclasses!")

    def extraction(self, stegobject, key=None):
        raise NotImplementedError("Extraction method must be implemented by subclasses!")

    def calculate_capacity(self, carrier):
        raise NotImplementedError("Calculate capacity method must be implemented by subclasses!")

    def set_method_directories(self, method_directories):
        """
        Sets the directories for the method.

        Parameters:
        - method_directories: dict, contains paths for 'Stegobjects', 'Results', 'Logs', 'Plots'
        """
        self.method_directories = method_directories

    def get_capacity(self):
        """
        Returns the capacity of the method (number of bits that fit into the
        current carrier).
        """
        return self.capacity

    def get_extraction_key(self):
        """Returns the side information needed by the receiver (see module docstring)."""
        return self.extraction_key


# --------------------------------------------------------------------------- #
# Multi-level (Hankel/SSA-like) SVD decomposition
# --------------------------------------------------------------------------- #

class MultiLevelSVDProcessor:

    def __init__(self, method_arguments, logger=None):

        self.how_much_rows_in_hankel = method_arguments['HANKEL_ROWS']
        self.logger = ensure_logger(logger)

    @staticmethod
    def hankel_2row(vector):
        vector = np.asarray(vector)
        return np.vstack((vector[:-1], vector[1:]))

    @staticmethod
    def hankel_k_row(vector, k):
        """
        Builds a Hankel matrix with ``k`` rows from a 1D vector.

        Returns a 2D array of shape (k, len(vector) - k + 1) where row ``i`` is
        ``vector[i : i + cols]``.
        """
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("Hankel input must be a 1D vector.")
        if k < 1 or k > len(vector):
            raise ValueError(
                f"Number of Hankel rows k={k} must be between 1 and the vector length ({len(vector)})."
            )
        cols = len(vector) - k + 1
        return np.lib.stride_tricks.sliding_window_view(vector, cols).copy()

    @staticmethod
    def diagonal_average(matrix):
        """
        Hankelisation: averages the anti-diagonals of ``matrix``.

        Vectorised equivalent of the original double loop (same summation order,
        so the results are bit-identical).
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        rows, cols = matrix.shape
        anti_diagonal = np.add.outer(np.arange(rows), np.arange(cols)).ravel()
        sums = np.bincount(anti_diagonal, weights=matrix.ravel(), minlength=rows + cols - 1)
        counts = np.bincount(anti_diagonal, minlength=rows + cols - 1)
        return sums / counts

    def _check_decomposable(self, signal):
        k = self.how_much_rows_in_hankel
        if len(signal) < k + 1:
            raise ValueError(
                f"Vector of length {len(signal)} is too short for a {k}-row Hankel decomposition "
                f"(need at least {k + 1} values). Widen the FREQ_L-FREQ_H band or use longer frames."
            )

    def multi_level_decompose(self, signal, num_levels):
        """
        Repeatedly: Hankel matrix -> SVD -> keep the rank-1 approximation
        (largest singular value) -> diagonal averaging. Returns the list of
        (U, s, Vh) for every level.
        """
        if num_levels < 1:
            raise ValueError(f"Number of levels must be >= 1, got {num_levels}.")

        levels = []
        current = np.asarray(signal, dtype=np.float64)
        self._check_decomposable(current)

        for _ in range(num_levels):
            H = self.hankel_k_row(current, self.how_much_rows_in_hankel)
            U, s, Vh = svd(H, full_matrices=False)
            levels.append((U, s, Vh))
            Ha = s[0] * np.outer(U[:, 0], Vh[0, :])
            current = self.diagonal_average(Ha)

        return levels

    @staticmethod
    def _detail_matrix(U, s, Vh):
        """All components except the first one (for 2-row Hankel: s1*u1*v1^T)."""
        Hd = s[1] * np.outer(U[:, 1], Vh[1, :])
        for i in range(2, len(s)):
            # Only reached for HANKEL_ROWS > 2. The original code dropped these
            # components, which made the reconstruction lossy.
            Hd = Hd + s[i] * np.outer(U[:, i], Vh[i, :])
        return Hd

    def reconstruct_from_levels(self, levels):
        """
        Inverse of :meth:`multi_level_decompose`. With unmodified singular
        values the original vector is reconstructed exactly (up to float
        rounding).
        """
        if not levels:
            raise ValueError("Cannot reconstruct from an empty list of levels.")

        U, s, Vh = levels[-1]
        if len(s) < 2:
            raise ValueError("Each level needs at least 2 singular values (HANKEL_ROWS >= 2).")

        Ha = s[0] * np.outer(U[:, 0], Vh[0, :])
        Hd = self._detail_matrix(U, s, Vh)
        currentL = self.diagonal_average(Ha + Hd)

        for U, s, Vh in reversed(levels[:-1]):
            Hd = self._detail_matrix(U, s, Vh)
            currentR = self.diagonal_average(Hd)
            Hd_reconstructed = self.hankel_k_row(currentR, self.how_much_rows_in_hankel)
            Ha_reconstructed = self.hankel_k_row(currentL, self.how_much_rows_in_hankel)
            currentL = self.diagonal_average(Ha_reconstructed + Hd_reconstructed)

        return currentL


# --------------------------------------------------------------------------- #
# MDCT
# --------------------------------------------------------------------------- #

class MDCTProcessor:
    def __init__(self):
        pass

    def filter_mdct_coeffs(self, mdct_coeffs, fs, frame_size, f_low, f_high):
        """
        Selects MDCT coefficients in [f_low, f_high].

        ``frame_size`` is the length of the *time-domain* frame (2N for N MDCT
        coefficients). MDCT bin ``k`` is centred at ``(k + 0.5) * fs / frame_size``.
        """
        freq_resolution = fs / frame_size
        idx_low = int(np.floor(f_low / freq_resolution))
        idx_high = int(np.ceil(f_high / freq_resolution))
        idx_low = max(0, idx_low)
        idx_high = min(len(mdct_coeffs), idx_high)
        return mdct_coeffs[idx_low:idx_high], idx_low, idx_high

    def kaiser_derived(self, M, beta):
        if M < 1:
            return np.array([])
        if M % 2:
            raise ValueError("Kaiser Bessel Derived windows are only defined for even number of taps")
        w = np.zeros(M)
        kaiserw = kaiser(M // 2 + 1, beta)
        csum = np.cumsum(kaiserw)
        halfw = np.sqrt(csum[:-1] / csum[-1])
        w[:M//2] = halfw
        w[-M//2:] = halfw[::-1]
        return w

    def mdct(self, x, odd=True):
        return np.real(self.cmdct(x, odd=odd)) * np.sqrt(2)

    def imdct(self, X, odd=True):
        return self.icmdct(X, odd=odd) * np.sqrt(2)

    def cmdct(self, x, odd=True):
        N = len(x) // 2
        n0 = (N + 1) / 2
        if odd:
            outlen = N
            pre_twiddle = np.exp(-1j * np.pi * np.arange(N * 2) / (N * 2))
            offset = 0.5
        else:
            outlen = N + 1
            pre_twiddle = 1.0
            offset = 0.0

        post_twiddle = np.exp(-1j * np.pi * n0 * (np.arange(outlen) + offset) / N)
        X = fft(x * pre_twiddle)[:outlen]

        if not odd:
            X[0] *= np.sqrt(0.5)
            X[-1] *= np.sqrt(0.5)

        return X * post_twiddle * np.sqrt(1 / N)

    def icmdct(self, X, odd=True):
        if not odd and len(X) % 2 == 0:
            raise ValueError("Even inverse CMDCT requires an odd number of coefficients")
        X = X.copy()
        if odd:
            N = len(X)
            n0 = (N + 1) / 2
            post_twiddle = np.exp(1j * np.pi * (np.arange(N * 2) + n0) / (N * 2))
            Y = np.zeros(N * 2, dtype=X.dtype)
            Y[:N] = X
            Y[N:] = -1 * np.conj(X[::-1])
        else:
            N = len(X) - 1
            n0 = (N + 1) / 2
            post_twiddle = 1.0
            X[0] *= np.sqrt(2)
            X[-1] *= np.sqrt(2)
            Y = np.zeros(N * 2, dtype=X.dtype)
            Y[:N+1] = X
            Y[N+1:] = -1 * np.conj(X[-2:0:-1])

        pre_twiddle = np.exp(1j * np.pi * n0 * np.arange(N * 2) / N)
        y = ifft(Y * pre_twiddle)
        return np.real(y * post_twiddle) * np.sqrt(N)

    def process_with_overlap_add(self, signal, frame_size, beta=8):
        """
        Przetwarzanie całego sygnału z overlap-add (50% nakładanie).
        """
        hop_size = frame_size // 2
        window = self.kaiser_derived(frame_size, beta)
        num_frames = (len(signal) - frame_size) // hop_size + 1
        reconstructed = np.zeros(len(signal))

        for i in range(num_frames):
            start = i * hop_size
            frame = signal[start:start+frame_size] * window
            mdct_coeffs = self.mdct(frame)
            frame_rec = self.imdct(mdct_coeffs)
            reconstructed[start:start+frame_size] += frame_rec * window

        return reconstructed


# --------------------------------------------------------------------------- #
# Framing + VAD
# --------------------------------------------------------------------------- #

class FrameProcessor:

    WEBRTC_FRAME_MS = (10, 20, 30)
    WEBRTC_FS = (8000, 16000, 32000, 48000)

    def __init__(self, logger=None):
        """
        Initializes the FrameProcessor with an optional logger.

        Args:
            logger (logging.Logger, optional): Logger for debug information.
        """
        self.logger = ensure_logger(logger)

    @staticmethod
    def _frame_signal(signal, frame_size, hop_size):
        """Full frames only (no zero padding) + the remaining tail samples."""
        if len(signal) < frame_size:
            return np.empty((0, frame_size), dtype=signal.dtype), np.array([], dtype=signal.dtype)
        n_frames = (len(signal) - frame_size) // hop_size + 1
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_size][:n_frames].copy()
        last_frame_end = (n_frames - 1) * hop_size + frame_size
        remaining = signal[last_frame_end:].copy()
        return frames, remaining

    def divide_into_voiced_unvoiced_frames(self, type: str, signal: np.ndarray, fs: int,
                                           frame_duration_ms: float,
                                           hop_duration_ms: float = None,
                                           winlen: float = None, ovrlen: float = None,
                                           ftThres: float = None, vadThres: float = None):
        """
        Wrapper VAD (rVAD lub WebRTC) — dzieli sygnał na ramki dźwięczne/bezdźwięczne.

        Frames are always full frames starting at ``i * hop`` (exactly the same
        framing as :meth:`divide_into_frames`, used by the extractor).

        Returns: (voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices, remaining_samples).
        """
        signal = np.asarray(signal)
        if hop_duration_ms is None:
            hop_duration_ms = frame_duration_ms

        frame_size = ms_to_samples(frame_duration_ms, fs, 'FRAME_LENGTH')
        hop_size = ms_to_samples(hop_duration_ms, fs, 'HOP_DURATION_MS')

        if len(signal) < frame_size:
            raise ValueError(
                f"Signal has {len(signal)} samples - shorter than one frame ({frame_size} samples)."
            )

        frames, remaining_samples = self._frame_signal(signal, frame_size, hop_size)
        n_frames = len(frames)

        if type == 'rVAD':

            self.logger.info("Starting rVAD analysis on signal")
            self.logger.debug("Signal length: %d samples, fs: %d Hz, frame: %s ms, hop: %s ms",
                              len(signal), fs, frame_duration_ms, hop_duration_ms)

            signal_float = signal.astype(np.float64) / 32768.0

            winlen_sec = winlen if winlen is not None else (frame_duration_ms / 1000.0)
            ovrlen_sec = ovrlen if ovrlen is not None else (hop_duration_ms / 1000.0)

            if ftThres is None:
                ftThres = 0.5
            if vadThres is None:
                vadThres = 0.4

            ft, flen, fsh, nfr = speechproc.sflux(signal_float, fs, winlen_sec, ovrlen_sec, 512)

            flen_samples = int(flen)
            fsh_samples = int(fsh)

            if flen_samples != frame_size or fsh_samples != hop_size or int(nfr) != n_frames:
                raise RuntimeError(
                    f"rVAD framing (len={flen_samples}, hop={fsh_samples}, n={nfr}) differs from the "
                    f"extractor framing (len={frame_size}, hop={hop_size}, n={n_frames})."
                )

            pv01 = np.zeros(nfr)
            pv01[ft <= ftThres] = 1
            pitch = np.copy(ft)
            opts = 1
            pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr, opts)

            ENERGYFLOOR = np.exp(-50)
            b = np.array([0.9770, -0.9770], dtype=np.float64)
            a = np.array([1.0000, -0.9540], dtype=np.float64)
            fdata = lfilter(b, a, signal_float)

            noise_samp, _, n_noise = speechproc.snre_highenergy(fdata, nfr, flen_samples, fsh_samples,
                                                               ENERGYFLOOR, pv01, pvblk)
            for j in range(int(n_noise)):
                start = int(noise_samp[j, 0])
                stop = int(noise_samp[j, 1])
                fdata[start: stop + 1] = 0.0

            vad_mask = np.asarray(speechproc.snre_vad(fdata, nfr, flen_samples, fsh_samples,
                                                      ENERGYFLOOR, pv01, pvblk, vadThres)).astype(int)

            if len(vad_mask) != n_frames:
                raise RuntimeError(f"rVAD returned {len(vad_mask)} decisions for {n_frames} frames.")

            self.logger.info("Finished rVAD analysis on signal")

        elif type in ('webrtcVAD', 'webrtc'):

            if webrtcvad is None:
                raise ImportError("webrtcvad is not installed (pip install webrtcvad-wheels).")
            if frame_duration_ms not in self.WEBRTC_FRAME_MS:
                raise ValueError(f"WebRTC VAD supports only {self.WEBRTC_FRAME_MS} ms frames, "
                                 f"got {frame_duration_ms} ms.")
            if fs not in self.WEBRTC_FS:
                raise ValueError(f"WebRTC VAD supports only fs in {self.WEBRTC_FS}, got {fs} Hz.")

            self.logger.info("Starting WebRTC VAD analysis on signal")

            vad = webrtcvad.Vad(3)
            vad_mask = np.array([1 if vad.is_speech(frame.astype(np.int16).tobytes(), fs) else 0
                                 for frame in frames], dtype=int)

            self.logger.info("Finished WebRTC VAD analysis on signal")

        else:
            raise ValueError(f"Unsupported VAD type: {type}. Supported: {SUPPORTED_VADS}")

        frame_numbers = np.arange(n_frames)
        voiced_indices = frame_numbers[vad_mask == 1]
        unvoiced_indices = frame_numbers[vad_mask == 0]

        voiced_frames = frames[vad_mask == 1]
        unvoiced_frames = frames[vad_mask == 0]

        self.logger.debug("Voiced frames: %d, unvoiced frames: %d, remaining samples: %d",
                          len(voiced_frames), len(unvoiced_frames), len(remaining_samples))

        return voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices, remaining_samples

    def divide_into_frames(self, signal, frame_duration_ms, fs, hop_duration_ms=None):
        """
        Dzieli sygnał na ramki o zadanej długości i przesunięciu (hop size),
        bez klasyfikacji (bez VAD).

        Returns:
            tuple: (frames [num_frames x frame_size], frame_indices, remaining samples)
        """
        signal = np.asarray(signal)
        frame_size = ms_to_samples(frame_duration_ms, fs, 'FRAME_LENGTH')
        hop_size = frame_size if hop_duration_ms is None else ms_to_samples(hop_duration_ms, fs, 'HOP_DURATION_MS')

        frames, remaining = self._frame_signal(signal, frame_size, hop_size)
        return frames, list(range(len(frames))), remaining

    def bytes_from_int16(self, audio_samples: np.ndarray) -> bytes:
        """Konwertuje tablicę int16 na bajty (16-bit PCM little-endian)."""
        return audio_samples.astype(np.int16).tobytes()

    def frame_generator(self, frame_duration_ms: int, audio_bytes: bytes, sample_rate: int):
        """Generator dzielący audio na ramki o określonej długości (ms)."""
        frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
        for offset in range(0, len(audio_bytes) - frame_length + 1, frame_length):
            yield audio_bytes[offset: offset + frame_length]

    def vad_process(self, audio_samples: np.ndarray, sample_rate: int = 16000,
                    frame_duration_ms: int = 20, aggressiveness: int = 3):
        """Przetwarza dane audio i zwraca listę ramek zawierających mowę."""
        if frame_duration_ms not in self.WEBRTC_FRAME_MS:
            raise ValueError("Długość ramki musi wynosić 10, 20 lub 30 ms")
        if sample_rate not in self.WEBRTC_FS:
            raise ValueError("Nieobsługiwana częstotliwość próbkowania")
        if webrtcvad is None:
            raise ImportError("webrtcvad is not installed (pip install webrtcvad-wheels).")
        audio_bytes = self.bytes_from_int16(audio_samples)
        vad = webrtcvad.Vad(aggressiveness)
        return [frame for frame in self.frame_generator(frame_duration_ms, audio_bytes, sample_rate)
                if vad.is_speech(frame, sample_rate)]


# --------------------------------------------------------------------------- #
# Frame-based SVD methods (Xue family, PC family, Wen)
# --------------------------------------------------------------------------- #

class DIFFERENTIAL_SVD_XUE_BASED_METHOD(Method):

    # Parameters every frame-based method needs.
    REQUIRED_ARGUMENTS = ('ALPHA', 'BETA', 'FREQ_L', 'FREQ_H', 'FRAME_LENGTH')

    # Defaults filled in when a parameter is missing (recorded in the config).
    DEFAULT_ARGUMENTS = {
        'REDUCED': False,
        'ADAPTIVE': False,
        'TYPE_OF_VAD': 'rVAD',
        'WHERE_TO_HIDE': 'VOICED',
        'HOP_DURATION_MS': None,
        'LEVELS': 1,
        'HANKEL_ROWS': 2,
    }

    # 'rectangular': frames are concatenated without a synthesis window, so
    # HOP_DURATION_MS must equal FRAME_LENGTH (otherwise overlapping samples
    # would be summed twice). 'mdct': 50% overlap with a TDAC window.
    FRAMING = 'rectangular'

    # Minimal number of coefficients in the FREQ_L-FREQ_H band.
    MIN_BAND_COEFFICIENTS = 2

    def __init__(self, method_arguments, logger=None):

        arguments = dict(self.DEFAULT_ARGUMENTS)
        arguments.update({k: v for k, v in dict(method_arguments).items()})

        super().__init__(arguments, logger)

        self._validate_static_arguments(self.method_arguments)

        arguments = self.method_arguments

        self.alpha = arguments['ALPHA']
        self.beta = arguments['BETA']
        self.where_to_hide = arguments['WHERE_TO_HIDE']

        self.FREQ_H = arguments['FREQ_H']
        self.FREQ_L = arguments['FREQ_L']

        self.frame_length = arguments['FRAME_LENGTH']

        self.domain = 'FREQUENCY'
        self.style = 'SELECTIVE'

        self.type_of_vad = arguments['TYPE_OF_VAD']

        self.reduced = arguments['REDUCED']
        self.adaptive = arguments['ADAPTIVE']

        if arguments.get('HOP_DURATION_MS') is not None:
            self.hop_duration_ms = arguments['HOP_DURATION_MS']
        else:
            self.hop_duration_ms = self.frame_length
        # Record the effective hop so saved configs are explicit.
        self.method_arguments['HOP_DURATION_MS'] = self.hop_duration_ms

        self.bits_per_frame = 1  # Domyślnie 1 bit na ramkę

        self.voiced_frames = []
        self.unvoiced_frames = []
        self.voiced_indices = []
        self.unvoiced_indices = []
        self.frames_to_use = []
        self.frames_voice_status = []
        self.remaining_samples = np.array([], dtype=np.int16)

        self.frame_samples = None
        self.hop_samples = None

        self.FrameProcessor = FrameProcessor(self.logger)
        self.MDCTProcessor = MDCTProcessor()
        self.MultiLevelSVDProcessor = MultiLevelSVDProcessor(self.method_arguments, self.logger)

        self.reset_state()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_static_arguments(self, arguments):
        missing = [key for key in self.REQUIRED_ARGUMENTS if key not in arguments]
        if missing:
            raise ValueError(f"{self.__class__.__name__}: missing parameters {missing}.")

        _require_number(arguments, 'ALPHA', positive=True)
        _require_number(arguments, 'BETA', positive=True)
        f_low = _require_number(arguments, 'FREQ_L', non_negative=True)
        f_high = _require_number(arguments, 'FREQ_H', positive=True)
        if f_high <= f_low:
            raise ValueError(f"FREQ_H ({f_high}) must be greater than FREQ_L ({f_low}).")

        frame = _require_number(arguments, 'FRAME_LENGTH', positive=True)
        hop = arguments.get('HOP_DURATION_MS')
        if hop is None:
            hop = frame
        else:
            hop = _require_number(arguments, 'HOP_DURATION_MS', positive=True)

        if self.FRAMING == 'rectangular' and not math.isclose(hop, frame):
            raise ValueError(
                f"{self.__class__.__name__} concatenates frames without a synthesis window, so "
                f"HOP_DURATION_MS ({hop}) must equal FRAME_LENGTH ({frame})."
            )
        if self.FRAMING == 'mdct' and not math.isclose(hop * 2, frame):
            raise ValueError(
                f"{self.__class__.__name__} uses MDCT with 50% overlap, so FRAME_LENGTH ({frame}) "
                f"must be 2 x HOP_DURATION_MS ({hop})."
            )

        if arguments.get('TYPE_OF_VAD') not in SUPPORTED_VADS:
            raise ValueError(f"TYPE_OF_VAD must be one of {SUPPORTED_VADS}, got {arguments.get('TYPE_OF_VAD')!r}.")
        if arguments.get('WHERE_TO_HIDE') not in SUPPORTED_HIDING_PLACES:
            raise ValueError(f"WHERE_TO_HIDE must be one of {SUPPORTED_HIDING_PLACES}, "
                             f"got {arguments.get('WHERE_TO_HIDE')!r}.")

        _require_bool(arguments, 'ADAPTIVE')
        _require_bool(arguments, 'REDUCED')
        _require_int(arguments, 'LEVELS', 0)
        _require_int(arguments, 'HANKEL_ROWS', 2)

    def _validate_carrier(self, carrier):
        fs = carrier.fs
        if not isinstance(fs, (int, np.integer)) or fs <= 0:
            raise ValueError(f"Invalid sampling rate of carrier {carrier.name}: {fs!r}.")
        if self.FREQ_H > fs / 2:
            raise ValueError(f"FREQ_H = {self.FREQ_H} Hz is above the Nyquist frequency "
                             f"({fs / 2} Hz) of carrier {carrier.name}.")

        self.frame_samples = ms_to_samples(self.frame_length, fs, 'FRAME_LENGTH')
        self.hop_samples = ms_to_samples(self.hop_duration_ms, fs, 'HOP_DURATION_MS')

        if self.FRAMING == 'mdct' and self.frame_samples % 2:
            raise ValueError("MDCT frames must have an even number of samples.")

        n_band = self.band_size(fs)
        if n_band < self.MIN_BAND_COEFFICIENTS:
            raise ValueError(
                f"Only {n_band} coefficients fall into {self.FREQ_L}-{self.FREQ_H} Hz for "
                f"{self.frame_length} ms frames at {fs} Hz; {self.name} needs at least "
                f"{self.MIN_BAND_COEFFICIENTS}. Widen the band or use longer frames."
            )

        if len(carrier.data_signed_int16) < self.frame_samples:
            raise ValueError(f"Carrier {carrier.name} is shorter than one frame.")

    def band_size(self, fs):
        """Number of transform coefficients in the FREQ_L-FREQ_H band."""
        dummy = np.zeros(self.frame_samples)
        band, _, _ = self.filter_dct_coeffs(dummy, fs, self.FREQ_L, self.FREQ_H)
        return len(band)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def reset_state(self):
        """
        Clears every per-embedding diagnostic list. Called by ``set_carrier``
        and at the beginning of ``embedding`` so that statistics (and the PC v3
        level map) never leak from one carrier/embedding to the next.
        """
        self.s_even = []
        self.s_odd = []
        self.new_s_even = []
        self.new_s_odd = []
        self.alphas = []
        self.betas = []
        self.extraction_key = None
        self.clipped_samples = 0
        self.embedded_bits = 0

    # ------------------------------------------------------------------ #
    # Sync helpers (not used by the default pipeline)
    # ------------------------------------------------------------------ #

    def prepare_secret_with_sync(self):
        """
        Przygotowuje wiadomość binarną:
        [PREAMBUŁA (7b)] + [LENGTH (11b) zabezpieczony Hammingiem 15,11] + [DANE]
        """
        original = list(self.secret.binary_data)

        if len(original) > 2047:
            raise ValueError("Maksymalna długość wiadomości to 2047 bitów.")

        length_bits = BitVector(intVal=len(original), size=11)
        length_hamming = self.hamming_15_11_encode(length_bits)
        preamble_bits = BitVector(bitlist=[1, 1, 1, 0, 0, 1, 0])  # Barker

        return preamble_bits + length_hamming + BitVector(bitlist=original)

    def find_sync_and_extract(self, extracted_bits):
        """
        Szuka preambuły (z tolerancją 1 błędu), dekoduje Hamming(15,11),
        i wyodrębnia wiadomość (nawet jeśli końcówka jest ucięta).
        """
        def hamming_distance(a, b):
            return sum(x != y for x, y in zip(a, b))

        preamble_ref = [1, 1, 1, 0, 0, 1, 0]
        extracted_bits = list(extracted_bits)
        total_len = len(extracted_bits)

        for i in range(total_len - 22 + 1):  # 7 (preambuła) + 15 (Hamming)

            candidate = extracted_bits[i:i + 7]

            if hamming_distance(candidate, preamble_ref) <= 1:

                hamming_bits = extracted_bits[i + 7:i + 22]

                try:
                    decoded_length_bits = self.hamming_15_11_decode(hamming_bits)
                except Exception as e:
                    self.logger.warning("Hamming decode failed at index %d: %s", i, e)
                    continue

                msg_len = decoded_length_bits.intValue()
                msg_start = i + 22
                msg_end = msg_start + msg_len

                if msg_end <= total_len:
                    return BitVector(bitlist=extracted_bits[msg_start:msg_end])

                self.logger.warning("Message truncated: expected %d bits, but only %d bits available.",
                                    msg_len, total_len - msg_start)
                return BitVector(bitlist=extracted_bits[msg_start:total_len])

        raise ValueError("Nie znaleziono preambuły (nawet z tolerancją 1 błędu).")

    # ------------------------------------------------------------------ #
    # Carrier
    # ------------------------------------------------------------------ #

    def set_carrier(self, carrier):

        self.logger.info("Setting carrier for method %s, carrier name: %s, fs: %s, length: %d samples",
                         self.name, carrier.name, carrier.fs, len(carrier.data_signed_int16))

        self._validate_carrier(carrier)

        super().set_carrier(carrier)

        self.secret = None
        self.reset_state()

        (self.voiced_frames, self.unvoiced_frames, self.voiced_indices,
         self.unvoiced_indices, self.remaining_samples) = self.FrameProcessor.divide_into_voiced_unvoiced_frames(
            self.type_of_vad, self.carrier.data_signed_int16, self.carrier.fs,
            self.frame_length, self.hop_duration_ms)

        total_frames = len(self.voiced_indices) + len(self.unvoiced_indices)
        self.frames_to_use = [None] * total_frames
        self.frames_voice_status = [None] * total_frames
        for frame, idx in zip(self.voiced_frames, self.voiced_indices):
            self.frames_to_use[idx] = frame
            self.frames_voice_status[idx] = True
        for frame, idx in zip(self.unvoiced_frames, self.unvoiced_indices):
            self.frames_to_use[idx] = frame
            self.frames_voice_status[idx] = False

        if any(status is None for status in self.frames_voice_status):
            raise RuntimeError("VAD did not classify every frame.")

        self.logger.info("Carrier set for method %s, total frames: %d, voiced: %d, unvoiced: %d",
                         self.name, total_frames, len(self.voiced_frames), len(self.unvoiced_frames))

        self.capacity = self.calculate_capacity(self.voiced_frames)

        if self.capacity <= 0:
            raise ValueError(f"Carrier {carrier.name} has no voiced frames - capacity is 0 bits.")

    # ------------------------------------------------------------------ #
    # Per-frame hooks
    # ------------------------------------------------------------------ #

    def embedd_in_frame(self, frame, bit, fs):
        raise NotImplementedError("embedd_in_frame method must be implemented by subclasses!")

    def extract_from_frame(self, frame, fs):
        raise NotImplementedError("extract_from_frame method must be implemented by subclasses!")

    def extract_bit(self, frame, fs, bit_index, key):
        """
        Extracts one bit. ``bit_index`` and ``key`` are available for methods
        that need per-bit side information (PC v3: decomposition level).
        """
        return self.extract_from_frame(frame, fs)

    def build_extraction_key(self, n_bits):
        """Side information required by the receiver (JSON serialisable)."""
        return {
            'method': self.__class__.__name__,
            'method_name': self.name,
            'carrier': getattr(self.carrier, 'name', None),
            'fs': int(self.carrier.fs),
            'frame_length_ms': self.frame_length,
            'hop_duration_ms': self.hop_duration_ms,
            'n_bits': int(n_bits),
            'voiced_mask': encode_mask(self.frames_voice_status),
        }

    # ------------------------------------------------------------------ #
    # Matrix helpers
    # ------------------------------------------------------------------ #

    def vector_to_rectangular(self, arr):
        if arr.ndim != 1:
            raise ValueError("Tablica wejściowa musi być jednowymiarowa.")

        size = arr.size
        rows = int(np.floor(np.sqrt(size)))

        while rows > 1 and size % rows != 0:
            rows -= 1

        if rows == 1 and size != 1:
            raise ValueError("Nie można utworzyć prostokątnej macierzy dla tej długości wektora.")

        cols = size // rows
        return arr.reshape((rows, cols)), (rows, cols)

    def make_square_matrix_with_padding(self, coeffs: np.ndarray) -> tuple[np.ndarray, int]:
        n = len(coeffs)
        side = int(np.ceil(np.sqrt(n)))

        square_matrix = np.zeros((side, side))
        square_matrix.flat[:n] = coeffs

        return square_matrix, side * side - n

    def vector_to_rectangular_with_padding(self, coeffs: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int]]:
        """
        Przekształca wektor do najbliższej prostokątnej macierzy z paddingiem.

        Zwraca:
            - macierz prostokątną
            - liczba dodanych elementów
            - kształt macierzy (rows, cols)
        """
        coeffs = np.asarray(coeffs)
        n = len(coeffs)
        if n == 0:
            raise ValueError("Cannot build a matrix from an empty coefficient vector.")
        best_shape = None
        min_diff = float('inf')
        for rows in range(1, n + 1):
            cols = int(np.ceil(n / rows))
            if rows * cols >= n and abs(rows - cols) < min_diff:
                best_shape = (rows, cols)
                min_diff = abs(rows - cols)

        rows, cols = best_shape
        padded_size = rows * cols
        padded_coeffs = np.zeros(padded_size, dtype=coeffs.dtype)
        padded_coeffs[:n] = coeffs

        return padded_coeffs.reshape((rows, cols)), padded_size - n, best_shape

    def filter_dct_coeffs(self, coeffs: np.ndarray, Fs: float, low_freq: float, high_freq: float):
        """
        Zwraca współczynniki DCT tylko z określonego zakresu częstotliwości.
        DCT-II bin ``k`` of an N-point frame corresponds to ``k * Fs / (2N)``.
        """
        N = len(coeffs)

        k_low = int(low_freq / (Fs / 2) * N)
        k_high = int(high_freq / (Fs / 2) * N)

        k_low = max(0, k_low)
        k_high = min(N, k_high)

        return coeffs[k_low:k_high], k_low, k_high

    def hamming_15_11_encode(self, data_bits):
        if len(data_bits) != 11:
            raise ValueError("Expected 11-bit input for Hamming(15,11)")

        code = [0] * 16  # index 0 unused for readability (1-based)

        data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        for pos, bit in zip(data_positions, data_bits):
            code[pos] = bit

        code[1] = (code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11] ^ code[13] ^ code[15])
        code[2] = (code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11] ^ code[14] ^ code[15])
        code[4] = (code[5] ^ code[6] ^ code[7] ^ code[12] ^ code[13] ^ code[14] ^ code[15])
        code[8] = (code[9] ^ code[10] ^ code[11] ^ code[12] ^ code[13] ^ code[14] ^ code[15])

        return BitVector(bitlist=code[1:16])

    def hamming_15_11_decode(self, bits_15):
        if len(bits_15) != 15:
            raise ValueError("Expected 15-bit input for Hamming(15,11)")

        code = [0] + list(bits_15)

        s1 = code[1] ^ code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11] ^ code[13] ^ code[15]
        s2 = code[2] ^ code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11] ^ code[14] ^ code[15]
        s4 = code[4] ^ code[5] ^ code[6] ^ code[7] ^ code[12] ^ code[13] ^ code[14] ^ code[15]
        s8 = code[8] ^ code[9] ^ code[10] ^ code[11] ^ code[12] ^ code[13] ^ code[14] ^ code[15]

        error_pos = s1 + (s2 << 1) + (s4 << 2) + (s8 << 3)

        if error_pos != 0 and error_pos <= 15:
            code[error_pos] ^= 1

        data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        return BitVector(bitlist=[code[i] for i in data_positions])

    # ------------------------------------------------------------------ #
    # Embedding / extraction
    # ------------------------------------------------------------------ #

    def _require_ready(self):
        if self.carrier is None:
            raise RuntimeError(f"{self.name}: call set_carrier() before embedding().")
        if self.secret is None:
            raise RuntimeError(f"{self.name}: call set_secret() before embedding().")

    def _secret_bits(self):
        bits = [int(b) for b in self.secret.binary_data]
        if not bits:
            raise ValueError("Secret is empty - nothing to embed.")
        if any(b not in (0, 1) for b in bits):
            raise ValueError("Secret must contain only 0/1 bits.")
        if len(bits) > self.capacity:
            raise ValueError(f"Secret has {len(bits)} bits but capacity is {self.capacity} bits.")
        return bits

    def process_frame_without_embedding(self, frame):
        """How frames that carry no bit enter the overlap-add (identity here)."""
        return frame

    def embedding(self):

        self._require_ready()
        bits = self._secret_bits()
        self.reset_state()

        self.logger.info("########### Starting embedding process (%s) ###########", self.name)

        start_embedding = timer()

        embedded_frames = []
        i = 0

        for frame, is_voiced in zip(self.frames_to_use, self.frames_voice_status):
            if is_voiced and i < len(bits):
                embedded_frames.append(self.embedd_in_frame(frame, bits[i], self.carrier.fs))
                i += 1
            else:
                embedded_frames.append(self.process_frame_without_embedding(frame))

        if i != len(bits):
            raise RuntimeError(f"Embedded only {i} of {len(bits)} bits.")

        stegobject_float = self.reconstruct_signal_from_frames(embedded_frames)
        stegobject, clipped = to_int16(stegobject_float)

        self.embedding_time = timer() - start_embedding

        if len(stegobject) != len(self.carrier.data_signed_int16):
            raise RuntimeError(f"Stegobject length {len(stegobject)} differs from carrier length "
                               f"{len(self.carrier.data_signed_int16)}.")

        self.clipped_samples = clipped
        if clipped:
            self.logger.warning("%s: %d samples clipped to the int16 range on carrier %s.",
                                self.name, clipped, self.carrier.name)

        self.embedded_bits = i
        self.extraction_key = self.build_extraction_key(i)

        self.logger.info("########### Finished embedding process ###########")

        return stegobject

    def _unpack_stegobject(self, stegobject, key):
        if isinstance(stegobject, dict):
            data = stegobject['content_signed_int16']
            fs = stegobject.get('fs', key['fs'])
        else:
            data, fs = stegobject, key['fs']
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("Stegobject must be a mono (1D) signal.")
        if int(fs) != int(key['fs']):
            raise ValueError(f"Stegobject sampling rate {fs} Hz differs from the embedding rate {key['fs']} Hz.")
        return data, int(fs)

    def extraction(self, stegobject, key=None):
        """
        Extracts the hidden bits.

        Parameters:
        - stegobject: dict with 'content_signed_int16' and 'fs' (as returned by
          Tools.load_audio_file) or a 1D int16 array
        - key: extraction key (dict) - defaults to the key of the last embedding

        Returns a BitVector. If the stegobject is shorter than the carrier (e.g.
        after an attack) fewer bits may be returned; Analysis counts missing
        bits as errors.
        """
        key = key if key is not None else self.extraction_key
        if key is None:
            raise RuntimeError(f"{self.name}: no extraction key - run embedding() first or pass key=...")

        data, fs = self._unpack_stegobject(stegobject, key)
        voiced_mask = decode_mask(key['voiced_mask'])
        n_bits = int(key['n_bits'])

        self.logger.info("########### Starting extraction process (%s) ###########", self.name)

        extraction_start = timer()

        frames, _, _ = self.FrameProcessor.divide_into_frames(
            data, key.get('frame_length_ms', self.frame_length), fs,
            key.get('hop_duration_ms', self.hop_duration_ms))

        if len(frames) < len(voiced_mask):
            self.logger.warning("Stegobject has %d frames, the carrier had %d - some bits will be missing.",
                                len(frames), len(voiced_mask))

        secret = []
        for frame_index, is_voiced in enumerate(voiced_mask):
            if len(secret) >= n_bits or frame_index >= len(frames):
                break
            if is_voiced:
                secret.append(int(self.extract_bit(frames[frame_index], fs, len(secret), key)))

        self.extraction_time = timer() - extraction_start

        self.logger.info("########### Finished extraction process ##########")

        return BitVector(bitlist=secret) if secret else BitVector(size=0)

    def reconstruct_signal_from_frames(self, embedded_frames):
        """Overlap-add of the processed frames + the untouched tail of the carrier."""
        if not embedded_frames:
            raise ValueError("No frames to reconstruct.")

        frame_len = len(embedded_frames[0])
        hop_size = self.hop_samples if self.hop_samples else ms_to_samples(self.hop_duration_ms, self.carrier.fs)

        output_length = hop_size * (len(embedded_frames) - 1) + frame_len
        output = np.zeros(output_length)

        for i, frame in enumerate(embedded_frames):
            if len(frame) != frame_len:
                raise ValueError(f"Frame {i} has {len(frame)} samples, expected {frame_len}.")
            start = i * hop_size
            output[start:start + frame_len] += frame

        carrier_len = len(self.carrier.data_signed_int16)
        output = output[:carrier_len]

        if self.remaining_samples is not None and len(self.remaining_samples):
            output = np.concatenate((output, np.asarray(self.remaining_samples, dtype=np.float64)))

        self.logger.info("Reconstructed signal length after overlap-add: %d samples", len(output))

        return output


class MULTI_LEVEL_METHOD(DIFFERENTIAL_SVD_XUE_BASED_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.levels = _require_int(self.method_arguments, 'LEVELS', 1)

        self.HANKEL_ROWS = self.method_arguments['HANKEL_ROWS']

        self.random_levels = []

    @property
    def MIN_BAND_COEFFICIENTS(self):
        # both the even and the odd branch need HANKEL_ROWS + 1 values
        return 2 * (self.method_arguments['HANKEL_ROWS'] + 1)

    def reset_state(self):
        super().reset_state()
        self.random_levels = []

    def split_even_odd(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        return coefficients[::2].copy(), coefficients[1::2].copy()

    @staticmethod
    def merge_even_odd(even, odd):
        merged = np.empty(len(even) + len(odd), dtype=np.float64)
        merged[::2] = even
        merged[1::2] = odd
        return merged

    def generate_random_integers(self, k, N, seed=None):
        """
        Generuje ciąg N losowych liczb całkowitych z zakresu od 1 do k (włącznie).
        """
        rng = np.random.default_rng(seed)
        return [int(v) for v in rng.integers(1, k + 1, size=N)]


class DCT_MULTI_LEVEL_SVD_METHOD(MULTI_LEVEL_METHOD):
    """
    Template shared by the PC family and the multi-level Xue variant:

        frame -> DCT -> FREQ_L..FREQ_H band -> even / odd coefficients
              -> multi-level Hankel SVD of both branches
              -> modify the largest singular values on the deepest level
                 (``modify_singular_values`` - the only thing a version defines)
              -> inverse decomposition -> merge -> inverse DCT

    Extraction: bit = 0 if sigma_even > sigma_odd on the deepest level, else 1.
    """

    def modify_singular_values(self, ro_11, ro_21, bit):
        """
        Returns ``(new_ro_11, new_ro_21, recorded_alpha, recorded_beta)``.
        ``ro_11`` - largest singular value of the even branch,
        ``ro_21`` - largest singular value of the odd branch.
        """
        raise NotImplementedError

    @staticmethod
    def safe_denominator(value):
        """Guards the adaptive thresholds against a (theoretical) all-zero band."""
        return value if value != 0 else 1e-10

    @staticmethod
    def decide_bit(ro_11, ro_21):
        return 0 if (ro_11 - ro_21) > 0 else 1

    def analyse_frame(self, frame, fs, levels):
        frame_dct = dct(np.asarray(frame, dtype=np.float64), norm='ortho')
        band, idx_low, idx_high = self.filter_dct_coeffs(frame_dct, fs, self.FREQ_L, self.FREQ_H)
        evn, odd = self.split_even_odd(band)
        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn, levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd, levels)
        return frame_dct, idx_low, idx_high, levels_evn, levels_odd

    def embed_in_levels(self, frame_dct, idx_low, idx_high, levels_evn, levels_odd, bit):
        """
        Modifies the deepest level of ``levels_evn``/``levels_odd`` (copies of
        the singular value vectors are used, the input lists are not changed)
        and returns ``(embedded_frame, stats)``.
        """
        levels_evn = [(U, s.copy(), Vh) for U, s, Vh in levels_evn]
        levels_odd = [(U, s.copy(), Vh) for U, s, Vh in levels_odd]

        _, s_evn, _ = levels_evn[-1]
        _, s_odd, _ = levels_odd[-1]

        i_evn = int(np.argmax(s_evn))
        i_odd = int(np.argmax(s_odd))
        ro_11 = s_evn[i_evn]
        ro_21 = s_odd[i_odd]

        new_11, new_21, recorded_alpha, recorded_beta = self.modify_singular_values(ro_11, ro_21, bit)

        s_evn[i_evn] = new_11
        s_odd[i_odd] = new_21

        rec_evn = self.MultiLevelSVDProcessor.reconstruct_from_levels(levels_evn)
        rec_odd = self.MultiLevelSVDProcessor.reconstruct_from_levels(levels_odd)

        full_dct = frame_dct.copy()
        full_dct[idx_low:idx_high] = self.merge_even_odd(rec_evn, rec_odd)

        stats = {
            'old_s_even': ro_11, 'old_s_odd': ro_21,
            'new_s_even': new_11, 'new_s_odd': new_21,
            'alpha_val': recorded_alpha, 'beta_val': recorded_beta,
        }
        return idct(full_dct, norm='ortho'), stats

    def record_stats(self, stats):
        self.s_even.append(stats['old_s_even'])
        self.s_odd.append(stats['old_s_odd'])
        self.new_s_even.append(stats['new_s_even'])
        self.new_s_odd.append(stats['new_s_odd'])
        self.alphas.append(stats['alpha_val'])
        self.betas.append(stats['beta_val'])

    def embedd_in_frame(self, frame, bit, fs):
        frame_dct, idx_low, idx_high, levels_evn, levels_odd = self.analyse_frame(frame, fs, self.levels)
        embedded_frame, stats = self.embed_in_levels(frame_dct, idx_low, idx_high, levels_evn, levels_odd, bit)
        self.record_stats(stats)
        return embedded_frame

    def singular_values_at_level(self, frame, fs, level):
        _, _, _, levels_evn, levels_odd = self.analyse_frame(frame, fs, level)
        _, s_evn, _ = levels_evn[level - 1]
        _, s_odd, _ = levels_odd[level - 1]
        return s_evn[np.argmax(s_evn)], s_odd[np.argmax(s_odd)]

    def extract_from_frame(self, frame, fs):
        ro_11, ro_21 = self.singular_values_at_level(frame, fs, self.levels)
        return self.decide_bit(ro_11, ro_21)
