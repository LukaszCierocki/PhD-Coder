import numpy as np
import webrtcvad
from scipy.fftpack import dct, idct, fft, ifft
from scipy.linalg import svd
from timeit import default_timer as timer
from BitVector import BitVector
from lib import speechproc
from scipy.signal import lfilter
from scipy.signal.windows import kaiser
import random

class Method:
    
    def __init__(self, method_arguments, logger=None):

        self.logger = logger

        self.method_arguments = method_arguments

        self.name = None

        self.capacity = None

        self.embedding_time = None
        self.extraction_time = None

        self.domain = None
        self.style = None

        self.method_directories = None

        self.extraction_mask = None

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
        self.secret = secret

    def embedding(self):
        raise NotImplementedError("Embedding method must be implemented by subclasses!")
    
    def extraction(self):
        raise NotImplementedError("Extraction method must be implemented by subclasses!")
    
    def calculate_capacity(self):
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
        Returns the capacity of the method.
        
        Returns:
        - int, the capacity of the method
        """
        return self.capacity
    
class MultiLevelSVDProcessor:

    def __init__(self, method_arguments, logger=None):

        self.how_much_rows_in_hankel = method_arguments['HANKEL_ROWS']
        self.logger = logger

    def hankel_2row(self, vector):
        vector = np.asarray(vector)
        return np.vstack((vector[:-1], vector[1:]))
    
    def hankel_k_row(self, vector, k):
        """
        Tworzy macierz Hankela o k wierszach z wektora wejściowego.
        
        Parametry:
        - vector: 1D array-like, dane wejściowe
        - k: int, liczba wierszy
        
        Zwraca:
        - 2D numpy array o wymiarach (k, len(vector) - k + 1)
        """
        vector = np.asarray(vector)
        if k > len(vector):
            raise ValueError("Liczba wierszy k nie może być większa niż długość wektora")
        
        cols = len(vector) - k + 1
        return np.vstack([vector[i:i + cols] for i in range(k)])

    def diagonal_average(self, matrix):
        rows, cols = matrix.shape
        result = np.zeros(rows + cols - 1)
        count = np.zeros(rows + cols - 1)
        for i in range(rows):
            for j in range(cols):
                result[i + j] += matrix[i, j]
                count[i + j] += 1
        return result / count

    def multi_level_decompose(self, signal, num_levels):
        levels = []
        current = signal

        for _ in range(num_levels):
            
            self.logger.debug(f"Current signal: \n{current}")
            H = self.hankel_k_row(current, self.how_much_rows_in_hankel)

            self.logger.debug(f"Hankel matrix: \n{H}")

            U, s, Vh = svd(H, full_matrices=False)

            self.logger.debug(f"SVD U: \n{U}, s: \n{s}, Vh: \n{Vh}")
            
            levels.append((U, s, Vh))
            
            Ha = s[0] * np.outer(U[:, 0], Vh[0, :])

            self.logger.debug(f"Ha: \n{Ha}")
            
            current = self.diagonal_average(Ha)

            self.logger.debug(f"Current signal after diagonal average: \n{current}")

        self.logger.debug(f"Final levels: \n{levels}")

        return levels

    def reconstruct_from_levels(self, levels):

        self.logger.debug(f"Reconstructing from levels: {levels}")

        U, s, Vh = levels[-1]

        self.logger.debug(f"Last level U: {U}, s: {s}, Vh: {Vh}")

        Ha = s[0] * np.outer(U[:, 0], Vh[0, :])

        self.logger.debug(f"Ha: {Ha}")

        Hd = s[1] * np.outer(U[:, 1], Vh[1, :])

        self.logger.debug(f"Hd: {Hd}")

        currentL = self.diagonal_average(Ha + Hd)

        self.logger.debug(f"Current level after diagonal average: {currentL}")

        i = 0

        for U, s, Vh in reversed(levels[:-1]):

            self.logger.debug(f"Reconstructing level U: {U}, s: {s}, Vh: {Vh}")

            Hd = s[1] * np.outer(U[:, 1], Vh[1, :])

            self.logger.debug(f"Hd: {Hd}")

            currentR = self.diagonal_average(Hd)

            self.logger.debug(f"Current R: {currentR}")

            Hd_reconstructed = self.hankel_k_row(currentR, self.how_much_rows_in_hankel)

            self.logger.debug(f"Hd_reconstructed: {Hd_reconstructed}")

            Ha_reconstructed = self.hankel_k_row(currentL, self.how_much_rows_in_hankel)

            self.logger.debug(f"Ha_reconstructed: {Ha_reconstructed}")

            H = Ha_reconstructed + Hd_reconstructed

            self.logger.debug(f"Reconstructed Hankel matrix: {H}")

            currentL = self.diagonal_average(H)

            self.logger.debug(f"Current level after reconstruction: {currentL}")

        self.logger.debug(f"Final reconstructed signal: {currentL}")

        return currentL
    
class MDCTProcessor:
    def __init__(self):
        pass

    def filter_mdct_coeffs(self, mdct_coeffs, fs, frame_size, f_low, f_high):
        freq_resolution = fs / frame_size
        idx_low = int(np.floor(f_low / freq_resolution))
        idx_high = int(np.ceil(f_high / freq_resolution))
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

class FrameProcessor:

    def __init__(self, logger=None):
        """
        Initializes the FrameProcessor with an optional logger.
        
        Args:
            logger (logging.Logger, optional): Logger for debug information.
        """
        self.logger = logger

    def divide_into_voiced_unvoiced_frames(self, type: str, signal: np.ndarray, fs: int,
                                            frame_duration_ms: float,
                                            hop_duration_ms: float = None,
                                            winlen: float = None, ovrlen: float = None,
                                            ftThres: float = None, vadThres: float = None):
        """
        Wrapper VAD (rVAD lub WebRTC) — dzieli sygnał na ramki dźwięczne/bezdźwięczne.
        Zwraca: (voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices, remaining_samples).
        """

        remaining_samples = np.array([], dtype=signal.dtype)

        if type == 'rVAD':

            if self.logger is not None:
                self.logger.info("Starting rVAD analysis on signal")
                self.logger.debug(f"Signal length: {len(signal)} samples, fs: {fs} Hz, frame_duration_ms: {frame_duration_ms}, hop_duration_ms: {hop_duration_ms}")

            # ✅ Poprawiona normalizacja
            signal_float = signal.astype(np.float64) / 32768.0

            # Parametry okna
            winlen_sec = winlen if winlen is not None else (frame_duration_ms / 1000.0)
            ovrlen_sec = ovrlen if ovrlen is not None else (hop_duration_ms / 1000.0 if hop_duration_ms is not None else winlen_sec)

            if ftThres is None:
                ftThres = 0.5
            if vadThres is None:
                vadThres = 0.4

            # Funkcja rVAD (speechproc)
            ft, flen, fsh, nfr = speechproc.sflux(signal_float, fs, winlen_sec, ovrlen_sec, 512)

            # ✅ Sprawdzenie jednostek (sekundy czy próbki)
            if flen < 1:
                flen_samples = int(flen * fs)
            else:
                flen_samples = int(flen)

            if fsh < 1:
                fsh_samples = int(fsh * fs)
            else:
                fsh_samples = int(fsh)

            if self.logger is not None:
                self.logger.debug(f"Calculated flen_samples: {flen_samples}, fsh_samples: {fsh_samples}, n_frames: {nfr}")

            pv01 = np.zeros(nfr)
            pv01[ft <= ftThres] = 1
            pitch = np.copy(ft)
            opts = 1
            pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr, opts)

            ENERGYFLOOR = np.exp(-50)
            b = np.array([0.9770, -0.9770], dtype=np.float64)
            a = np.array([1.0000, -0.9540], dtype=np.float64)
            fdata = lfilter(b, a, signal_float)

            noise_samp, _, n_noise = speechproc.snre_highenergy(fdata, nfr, flen_samples, fsh_samples, ENERGYFLOOR, pv01, pvblk)
            for j in range(int(n_noise)):
                start = int(noise_samp[j, 0])
                stop = int(noise_samp[j, 1])
                fdata[start: stop + 1] = 0.0

            vad_mask = speechproc.snre_vad(fdata, nfr, flen_samples, fsh_samples, ENERGYFLOOR, pv01, pvblk, vadThres)

            # ✅ Ramkowanie
            frames = []
            total_len = len(signal)

            for i in range(int(nfr)):
                start = i * fsh_samples
                end = start + flen_samples

                if end <= total_len:
                    frame = signal[start:end]
                else:
                    frame = signal[start:total_len]
                    frame = np.pad(frame, (0, end - total_len), 'constant', constant_values=0)

                frames.append(frame)

            frames = np.vstack(frames)

            last_frame_end = int((nfr - 1) * fsh_samples + flen_samples)
            if last_frame_end < total_len:
                remaining_samples = signal[last_frame_end:].copy()

            frame_numbers = np.arange(len(vad_mask))
            voiced_indices = frame_numbers[vad_mask == 1]
            unvoiced_indices = frame_numbers[vad_mask == 0]

            voiced_frames = frames[vad_mask == 1]
            unvoiced_frames = frames[vad_mask == 0]

            if self.logger is not None:
                self.logger.debug(f"Number of voiced frames: {len(voiced_frames)}, Number of unvoiced frames: {len(unvoiced_frames)}, remaining_samples: {len(remaining_samples)}")
                self.logger.info("Finished rVAD analysis on signal")

        elif type in ('webrtcVAD', 'webrtc'):
            self.logger.info("Starting WebRTC VAD analysis on signal")

            frame_size = int(fs * frame_duration_ms / 1000)
            hop_size = frame_size if hop_duration_ms is None else int(fs * hop_duration_ms / 1000)

            vad_aggressiveness = 3
            vad = webrtcvad.Vad(vad_aggressiveness)
            signal = signal.astype(np.int16)

            frames = []
            total_len = len(signal)
            i = 0
            while i < total_len:
                end = i + frame_size
                if end <= total_len:
                    frame = signal[i:end]
                else:
                    frame = signal[i:total_len]
                    frame = np.pad(frame, (0, end - total_len), 'constant', constant_values=0)
                frames.append(frame)
                i += hop_size

            frames = np.vstack(frames)
            n_frames = len(frames)

            voiced_frames = []
            unvoiced_frames = []
            voiced_indices = []
            unvoiced_indices = []

            for idx in range(n_frames):
                frame_bytes = frames[idx].tobytes()
                if vad.is_speech(frame_bytes, fs):
                    voiced_frames.append(frames[idx])
                    voiced_indices.append(idx)
                else:
                    unvoiced_frames.append(frames[idx])
                    unvoiced_indices.append(idx)

            last_frame_end = (n_frames - 1) * hop_size + frame_size
            if last_frame_end < total_len:
                remaining_samples = signal[last_frame_end:].copy()

            voiced_frames = np.array(voiced_frames)
            unvoiced_frames = np.array(unvoiced_frames)

            self.logger.info("Finished WebRTC VAD analysis on signal")

        else:
            raise ValueError(f"Unsupported VAD type: {type}")

        return voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices, remaining_samples

    def divide_into_frames(self, signal, frame_duration_ms, fs, hop_duration_ms=None):
        """
        Dzieli sygnał na ramki o zadanej długości i przesunięciu (hop size),
        bez klasyfikacji (bez VAD).

        Args:
            signal (np.ndarray): Sygnał 1D (int16 lub float).
            frame_duration_ms (float): Długość ramki w milisekundach.
            fs (int): Częstotliwość próbkowania (Hz).
            hop_duration_ms (float, optional): Przesunięcie ramek w ms. Domyślnie = długość ramki.

        Returns:
            tuple:
                frames (np.ndarray): Tablica ramek (num_frames x frame_size).
                frame_indices (List[int]): Indeksy ramek (kolejność w sygnale).
                remaining (np.ndarray): Pozostałe próbki, które nie zmieściły się w ramki.
        """
        frame_size = int(fs * frame_duration_ms / 1000)
        hop_size = frame_size if hop_duration_ms is None else int(fs * hop_duration_ms / 1000)

        frames = []
        indices = []

        i = 0
        frame_idx = 0
        while i + frame_size <= len(signal):
            frame = signal[i : i + frame_size]
            frames.append(frame)
            indices.append(frame_idx)
            i += hop_size
            frame_idx += 1

        remaining = signal[i:] if i < len(signal) else np.array([], dtype=signal.dtype)

        return np.array(frames), indices, remaining
        
    def bytes_from_int16(self, audio_samples: np.ndarray) -> bytes:
        """Konwertuje tablicę int16 na bajty (16-bit PCM little-endian)."""
        # Upewnij się, że typ to int16
        audio_int16 = audio_samples.astype(np.int16)
        # Konwersja do sekwencji bajtów
        return audio_int16.tobytes()

    def frame_generator(self, frame_duration_ms: int, audio_bytes: bytes, sample_rate: int):
        """Generator dzielący audio na ramki o określonej długości (ms)."""
        # Oblicz rozmiar ramki w bajtach (16-bit = 2 bajty na próbkę)
        frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
        # Iteruj co frame_length bajtów (pomijamy ewentualny niepełny koniec)
        for offset in range(0, len(audio_bytes) - frame_length + 1, frame_length):
            yield audio_bytes[offset : offset + frame_length]

    def vad_process(self, audio_samples: np.ndarray, sample_rate: int = 16000, 
                    frame_duration_ms: int = 20, aggressiveness: int = 3):
        """Przetwarza dane audio i zwraca listę ramek zawierających mowę."""
        assert frame_duration_ms in (10, 20, 30), "Długość ramki musi wynosić 10, 20 lub 30 ms"
        assert sample_rate in (8000, 16000, 32000, 48000), "Nieobsługiwana częstotliwość próbkowania"
        audio_bytes = self.bytes_from_int16(audio_samples)
        vad = webrtcvad.Vad(aggressiveness)
        speech_frames = []
        # Podziel audio na ramki i sprawdź każdą za pomocą VAD
        for frame in self.frame_generator(frame_duration_ms, audio_bytes, sample_rate):
            if vad.is_speech(frame, sample_rate):
                speech_frames.append(frame)
        return speech_frames
class DIFFERENTIAL_SVD_XUE_BASED_METHOD(Method):

    def __init__(self, method_arguments, logger=None):
        
        super().__init__(method_arguments, logger)

        self.alpha = method_arguments['ALPHA']
        self.beta  = method_arguments['BETA']
        self.where_to_hide = method_arguments['WHERE_TO_HIDE']

        self.FREQ_H = method_arguments['FREQ_H']
        self.FREQ_L = method_arguments['FREQ_L']

        self.frame_length = method_arguments['FRAME_LENGTH']

        self.domain = 'FREQUENCY'
        self.style = 'SELECTIVE'

        self.s_even = []
        self.s_odd = []

        self.new_s_even = []
        self.new_s_odd = []

        self.type_of_vad = method_arguments['TYPE_OF_VAD']

        self.voiced_frames = []
        self.unvoiced_frames = []
        self.voiced_indices = []
        self.unvoiced_indices = []

        self.reduced = method_arguments['REDUCED']

        self.frames_to_use = []
        self.frames_voice_status = []

        self.adaptive = method_arguments['ADAPTIVE']

        self.alphas = []
        self.betas = []

        self.remaining_samples = np.array([], dtype=np.int16)

        if method_arguments.get('HOP_DURATION_MS') is not None:
            self.hop_duration_ms = method_arguments['HOP_DURATION_MS']
        else:
            self.hop_duration_ms = self.frame_length

        self.bits_per_frame = 1  # Domyślnie 1 bit na ramkę

        self.FrameProcessor = FrameProcessor(logger)
        self.MDCTProcessor = MDCTProcessor()
        self.MultiLevelSVDProcessor = MultiLevelSVDProcessor(method_arguments, logger)


    def prepare_secret_with_sync(self):
        """
        Przygotowuje wiadomość binarną:
        [PREAMBUŁA (7b)] + [LENGTH (11b) zabezpieczony Hammingiem 15,11] + [DANE]
        """
        original = list(self.secret.binary_data)

        if len(original) > 2047:
            raise ValueError("Maksymalna długość wiadomości to 2047 bitów.")

        self.logger.debug(f"[ENCODE] original length = {len(original)}")
        length_bits = BitVector(intVal=len(original), size=11)

        self.logger.debug(f"[ENCODE] length bits = {length_bits}, length = {length_bits.intValue()} bits")

        length_hamming = self.hamming_15_11_encode(length_bits)

        self.logger.debug(f"[ENCODE] length hamming = {length_hamming}")

        length_hamming_decoded = self.hamming_15_11_decode(length_hamming)

        self.logger.debug(f"[ENCODE] length hamming decoded = {length_hamming_decoded}, length = {length_hamming_decoded.intValue()} bits")

        preamble_bits = BitVector(bitlist=[1,1,1,0,0,1,0])  # Barker

        full = preamble_bits + length_hamming + BitVector(bitlist=original)

        self.logger.debug(f"Prepared secret with sync: preamble {preamble_bits}, length {length_hamming}, data {original[:50]}... (total length: {len(full)})")
        self.logger.debug(f"Full secret length: {len(full)} bits")
        self.logger.debug(f"Decoded length (Hamming): {length_hamming_decoded.intValue()} bits")

        return full

    def find_sync_and_extract(self, extracted_bits):
        """
        Szuka preambuły (z tolerancją 1 błędu), dekoduje Hamming(15,11),
        i wyodrębnia wiadomość (nawet jeśli końcówka jest ucięta).

        Args:
            extracted_bits: lista bitów odczytanych z ramek

        Returns:
            BitVector: wyodrębniona wiadomość binarna
        """
        def hamming_distance(a, b):
            return sum(x != y for x, y in zip(a, b))

        preamble_ref = [1, 1, 1, 0, 0, 1, 0]
        total_len = len(extracted_bits)

        for i in range(total_len - 22):  # 7 (preambuła) + 15 (Hamming)
            
            candidate = extracted_bits[i:i+7]

            self.logger.debug(f"Checking candidate preamble at index {i}: {candidate}")

            if hamming_distance(candidate, preamble_ref) <= 1:

                hamming_bits = extracted_bits[i+7:i+22]

                self.logger.debug(f"Found preamble at index {i}, Hamming bits: {hamming_bits}")

                try:
                    decoded_length_bits = self.hamming_15_11_decode(hamming_bits)

                    self.logger.debug(f"Decoded length bits: {decoded_length_bits}, length = {decoded_length_bits.intValue()} bits")

                except Exception as e:
                    self.logger.warning(f"Hamming decode failed at index {i}: {e}")
                    continue  # próbuj dalej

                msg_len = decoded_length_bits.intValue()
                msg_start = i + 22
                msg_end = msg_start + msg_len

                if msg_end <= total_len:
                    return BitVector(bitlist=extracted_bits[msg_start:msg_end])
                else:
                    available = total_len - msg_start
                    self.logger.warning(
                        f"Message truncated: expected {msg_len} bits, "
                        f"but only {available} bits available."
                    )
                    return BitVector(bitlist=extracted_bits[msg_start:total_len])

        raise ValueError("Nie znaleziono preambuły (nawet z tolerancją 1 błędu).")

    def set_carrier(self, carrier):

        self.logger.info(f"Setting carrier for method {self.name}, carrier name: {carrier.name}, carrier fs: {carrier.fs}, carrier length: {len(carrier.data_signed_int16)} samples")

        super().set_carrier(carrier)
    
        self.voiced_frames, self.unvoiced_frames, self.voiced_indices, self.unvoiced_indices, self.remaining_samples = self.FrameProcessor.divide_into_voiced_unvoiced_frames(self.type_of_vad, self.carrier.data_signed_int16, self.carrier.fs, self.frame_length, self.hop_duration_ms)

        self.frames_to_use = []

        total_frames = len(self.voiced_indices) + len(self.unvoiced_indices)
        self.frames_to_use = [None] * total_frames
        self.frames_voice_status = [None] * total_frames
        for frame, idx in zip(self.voiced_frames, self.voiced_indices):
            self.frames_to_use[idx] = frame
            self.frames_voice_status[idx] = True
        for frame, idx in zip(self.unvoiced_frames, self.unvoiced_indices):
            self.frames_to_use[idx] = frame
            self.frames_voice_status[idx] = False
        
        self.logger.info(f"Carrier set for method {self.name}, total frames: {total_frames}, voiced frames: {len(self.voiced_frames)}, unvoiced frames: {len(self.unvoiced_frames)}")
        
        self.capacity = self.calculate_capacity(self.voiced_frames)
    
    def embedd_in_frame(self):
        raise NotImplementedError("embedd_in_frame method must be implemented by subclasses!")
    
    def extract_from_frame(self):
        raise NotImplementedError("extract_from_frame method must be implemented by subclasses!")

    def vector_to_rectangular(self, arr):
        if arr.ndim != 1:
            raise ValueError("Tablica wejściowa musi być jednowymiarowa.")

        size = arr.size
        rows = int(np.floor(np.sqrt(size)))

        # Znajdź najbliższy dzielnik rozmiaru tablicy
        while rows > 1 and size % rows != 0:
            rows -= 1

        if rows == 1 and size != 1:
            raise ValueError("Nie można utworzyć prostokątnej macierzy dla tej długości wektora.")

        cols = size // rows
        return arr.reshape((rows, cols)), (rows, cols)
        
    def make_square_matrix_with_padding(self, coeffs: np.ndarray) -> tuple[np.ndarray, int]:
        n = len(coeffs)
        side = int(np.ceil(np.sqrt(n)))  # Najmniejszy rozmiar kwadratowy

        square_matrix = np.zeros((side, side))
        square_matrix.flat[:n] = coeffs  # Wypełnienie współczynnikami

        # Obliczenie liczby dodanych elementów
        num_padded_elements = side * side - n

        return square_matrix, num_padded_elements

    def vector_to_rectangular_with_padding(self, coeffs: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int]]:
        """
        Przekształca wektor do najbliższej prostokątnej macierzy z paddingiem.

        Zwraca:
            - macierz prostokątną
            - liczba dodanych elementów
            - kształt macierzy (rows, cols)
        """
        n = len(coeffs)
        # Wybieramy najbliższy rozkład na rzędy i kolumny o jak najmniejszej różnicy
        best_shape = None
        min_diff = float('inf')
        for rows in range(1, n+1):
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

        :param coeffs: numpy array z współczynnikami DCT (1D)
        :param Fs: częstotliwość próbkowania (Hz)
        :param low_freq: dolna granica zakresu (Hz)
        :param high_freq: górna granica zakresu (Hz)
        :return: numpy array zawierający tylko współczynniki z określonego zakresu częstotliwości
        """
        N = len(coeffs)

        # Konwersja zakresu z Hz na indeksy
        k_low = int(low_freq / (Fs / 2) * N)
        k_high = int(high_freq / (Fs / 2) * N)

        # Upewniamy się, że indeksy są w dozwolonym zakresie
        k_low = max(0, k_low)
        k_high = min(N, k_high)

        # Zwracamy tylko wycinek - same istotne współczynniki

        return coeffs[k_low:k_high], k_low, k_high

    def hamming_15_11_encode(self, data_bits):
        if len(data_bits) != 11:
            raise ValueError("Expected 11-bit input for Hamming(15,11)")

        code = [0] * 16  # index 0 unused for readability (1-based)

        # Wstaw dane
        data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        for pos, bit in zip(data_positions, data_bits):
            code[pos] = bit

        # Oblicz bity parzystości
        code[1] = (code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11] ^ code[13] ^ code[15])  # p1
        code[2] = (code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11] ^ code[14] ^ code[15]) # p2
        code[4] = (code[5] ^ code[6] ^ code[7] ^ code[12] ^ code[13] ^ code[14] ^ code[15]) # p4
        code[8] = (code[9] ^ code[10] ^ code[11] ^ code[12] ^ code[13] ^ code[14] ^ code[15])# p8

        return BitVector(bitlist=code[1:16])
    
    def hamming_15_11_decode(self, bits_15):
        if len(bits_15) != 15:
            raise ValueError("Expected 15-bit input for Hamming(15,11)")

        code = [0] + list(bits_15)  # index 0 unused

        # Oblicz syndrom (pozycja błędu)
        s1 = code[1] ^ code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11] ^ code[13] ^ code[15]
        s2 = code[2] ^ code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11] ^ code[14] ^ code[15]
        s4 = code[4] ^ code[5] ^ code[6] ^ code[7] ^ code[12] ^ code[13] ^ code[14] ^ code[15]
        s8 = code[8] ^ code[9] ^ code[10] ^ code[11] ^ code[12] ^ code[13] ^ code[14] ^ code[15]

        error_pos = s1 + (s2 << 1) + (s4 << 2) + (s8 << 3)

        if error_pos != 0 and error_pos <= 15:
            code[error_pos] ^= 1  # poprawienie błędu

        data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        return BitVector(bitlist=[code[i] for i in data_positions])

    

    
    def extraction(self, stegobject):

        self.logger.info("########### Starting extraction process ###########")

        data = stegobject['content_signed_int16']
        fs = stegobject['fs']

        extration_start = timer()
        
        frames, indices, remaining = self.FrameProcessor.divide_into_frames(data, self.frame_length, fs, self.hop_duration_ms)

        secret = []

        for frame, bit_mask in zip(frames, self.frames_voice_status):
        
            if(bit_mask == True):
                self.logger.debug(f"Extracting bit from frame \n{frame}")

                extracted_bit = self.extract_from_frame(frame, fs)
                secret.append(extracted_bit)

                if(len(secret) == len(self.secret.binary_data)):
                    self.logger.debug("Found enough bits for extraction.")
                    break

        secret = BitVector(bitlist=secret)
        
        extraction_stop = timer()

        self.extraction_time = extraction_stop - extration_start

        self.logger.info("########### Finished extraction process ##########")
        
        return secret
    
    def embedding(self):

        self.logger.info("########### Starting embedding process ###########")

        embedded_frames = []
        
        start_embedding = timer()

        i = 0
        index = 0

        for frame, is_voiced in zip(self.frames_to_use, self.frames_voice_status):

            if(is_voiced):

                if i < len(self.secret.binary_data):

                    secret_bit = self.secret.binary_data[i]
                    
                    self.logger.debug(f"Embedding bit {secret_bit} in frame idx: {index} \n{frame}")

                    embedded_frame = self.embedd_in_frame(frame, secret_bit, self.carrier.fs)

                    embedded_frames.append(embedded_frame)

                    self.logger.debug(f"Embedded frame: \n{embedded_frame}")

                    i += 1

                else:
                    self.logger.debug(f"No more bits to embed, using original frame at index {index}")
                    embedded_frames.append(frame)

            else:

                self.logger.debug(f"Using unvoiced frame at index {index} without embedding")
                embedded_frames.append(frame)

            index += 1

        stop_embedding = timer()

        self.logger.info("########### Finished embedding process ###########")

        self.embedding_time = stop_embedding - start_embedding

        stegobject = self.reconstruct_signal_from_frames(embedded_frames)

        return stegobject.astype(np.int16)

    def reconstruct_signal_from_frames(self, embedded_frames):
        self.logger.info("Reconstructing signal from embedded frames (with overlap-add)")

        total_frames = len(self.voiced_indices) + len(self.unvoiced_indices)
        reconstructed = [None] * total_frames

        reconstructed = embedded_frames

        # --- Wyznacz frame_len i hop_size ---
        frame_len = len(reconstructed[0])
        hop_size = int(self.hop_duration_ms * self.carrier.fs / 1000)

        output_length = hop_size * (len(reconstructed) - 1) + frame_len
        output = np.zeros(output_length)

        for i, frame in enumerate(reconstructed):
            start = i * hop_size
            output[start:start + frame_len] += frame

        # Przycinamy do długości nośnika (jeśli trzeba)
        if hasattr(self, "carrier") and hasattr(self.carrier, "data_signed_int16"):
            carrier_len = len(self.carrier.data_signed_int16)
            output = output[:carrier_len]

        # Doklej remaining_samples jeśli masz
        if hasattr(self, "remaining_samples") and self.remaining_samples is not None:
            output = np.concatenate((output, self.remaining_samples))

        self.logger.info(f"Reconstructed signal length after overlap-add: {len(output)} samples")

        return output

class MULTI_LEVEL_METHOD(DIFFERENTIAL_SVD_XUE_BASED_METHOD):

    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.levels = method_arguments['LEVELS']

        self.alpha = method_arguments['ALPHA']
        self.beta  = method_arguments['BETA']

        self.HANKEL_ROWS = method_arguments['HANKEL_ROWS']

        self.MultiLevelSVDProcessor = MultiLevelSVDProcessor(method_arguments, logger)

        self.random_levels = []

    def generate_random_integers(self, k, N):
        """
        Generuje ciąg N losowych liczb całkowitych z zakresu od 1 do k (włącznie).

        :param k: górna granica zakresu liczb
        :param N: długość ciągu
        :return: lista N losowych liczb od 1 do k
        """
        return [random.randint(1, k) for _ in range(N)]