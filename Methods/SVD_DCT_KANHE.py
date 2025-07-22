from lib.Method import Method
from BitVector import BitVector
import numpy as numpy
from timeit import default_timer as timer
from scipy.fftpack import dct, idct
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt

class SVD_DCT_KANHE(Method):
    
    def __init__(self, method_arguments):
        self.alpha = method_arguments['ALPHA']
        self.name = f"SVD_DCT_KANHE_{self.alpha}_{method_arguments['FRAME_LENGTH']}"

        self.frame_length = method_arguments['FRAME_LENGTH']
        
        super().__init__(method_arguments)
        
        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'
        self.U1, self.V1, self.S1 = [], [], []
        self.remaining_data = None
        
        self.embedding_time = 0
        self.extraction_time = 0

        self.capacity = 0
        self.secret = None

        self.charts_directory = method_arguments['CHARTS_DIRECTORY']

    def zerocrossing_count(self, signal):
        """
        Liczy liczbę przejść sygnału przez zero.
        
        Args:
            signal (numpy array): Sygnał wejściowy (np. dźwiękowy w próbkach).
            
        Returns:
            int: Liczba przejść przez zero w sygnale.
        """

        signal = np.where(signal == 0, np.finfo(float).eps, signal)
        
        # Liczenie przejść przez zero
        zero_crossings = np.sum((signal[:-1] * signal[1:]) < 0)
        
        return int(zero_crossings)

    def divide_into_frames(self, signal, frame_duration_ms, fs, hop_duration_ms=None):
        """
        Dzieli sygnał na ramki o zadanej szerokości czasowej.
        
        Args:
            signal (numpy array): Sygnał wejściowy.
            frame_duration_ms (float): Długość ramki w milisekundach.
            fs (int): Częstotliwość próbkowania (Hz).
            hop_duration_ms (float): Opcjonalnie, przesunięcie między ramkami w ms (domyślnie = frame_duration_ms).
        
        Returns:
            numpy array: Tablica ramek o wymiarach (liczba ramek, długość ramki w próbkach).
        """
        # Oblicz liczbę próbek w ramce i przesunięciu
        frame_size = int(fs * (frame_duration_ms / 1000))
        hop_size = frame_size if hop_duration_ms is None else int(fs * (hop_duration_ms / 1000))
        
        # Liczba ramek
        num_frames = (len(signal) - frame_size) // hop_size + 1
        
        # Utwórz tablicę ramek
        frames = np.array([
            signal[i * hop_size : i * hop_size + frame_size]
            for i in range(num_frames)
        ])
        
        return frames

    def calculate_capacity(self, carrier):
        return len(carrier) / 16 * 12 * 1

    def frame_data(self, data):
        framed_data = [np.array(data[i:i+16]).reshape(4, 4) for i in range(0, len(data) - len(data) % 16, 16)]
        self.remaining_data = data[len(data) - len(data) % 16:]
        return framed_data

    def reframe_data(self, data):
        return [cell for frame in data for row in frame for cell in row]

    def create_secret_frames(self, bit_array, frame_size=(4,4)):
        # Usunięcie przekątnej z ramki 4x4 - maska dla pozycji, które nie są na diagonali
        mask = np.ones(frame_size, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Obliczenie liczby bitów potrzebnych na jedną ramkę
        bits_per_frame = np.sum(mask)
        
        # Liczba ramek możliwa do utworzenia
        total_bits = bit_array.size
        num_frames = total_bits // bits_per_frame
        
        # Przekształcenie tablicy wejściowej na wektor dla łatwego przetwarzania
        flat_bits = bit_array
        # Lista ramek
        frames = []
        
        for i in range(num_frames):
            frame = np.zeros(frame_size, dtype=int)  # Inicjalizacja ramki 4x4 zerami
            frame_values = flat_bits[i * bits_per_frame : (i + 1) * bits_per_frame]  # Pobranie bitów
            
            frame[mask] = frame_values  # Wstawienie wartości z pominięciem przekątnej
            frames.append(frame)
        
        return np.array(frames)

    def short_time_energy(self, signal):

        """
        Oblicza Short-Time Energy (STE) dla ramki sygnału z zastosowaniem okna Hamminga.
        
        Args:
            frame (numpy array): Ramka sygnału (jednowymiarowa tablica próbek).
            
        Returns:
            float: Energia ramki po zastosowaniu okna Hamminga.
        """

        signal = signal.astype(np.float32) / 32767.0

        # Tworzenie okna Hamminga dla danej ramki
        hamming_window = np.hamming(len(signal))
        
        # Zastosowanie okna Hamminga do ramki
        windowed_frame = signal * hamming_window
        
        energy = np.sum(windowed_frame**2)

        # Normalizacja energii do długości ramki
        normalized_ste = energy / len(signal)

        return normalized_ste
    
    def create_carrier_frames(self, data_array, frame_size=(4,4)):
        """
        Dzieli jednowymiarową tablicę numpy na ramki 4x4, wypełniając każdą pozycję wartościami z wejściowej tablicy.
        
        :param data_array: Jednowymiarowa tablica numpy z wartościami liczbowymi.
        :param frame_size: Rozmiar pojedynczej ramki (domyślnie 4x4).
        :return: Tablica ramek o kształcie (num_frames, 4, 4).
        """
        # Obliczenie liczby wartości potrzebnych na jedną ramkę
        values_per_frame = frame_size[0] * frame_size[1]
        
        # Liczba ramek możliwa do utworzenia
        total_values = len(data_array)
        num_frames = total_values // values_per_frame
        
        # Przekształcenie tablicy wejściowej na wektor dla łatwego przetwarzania
        flat_data = data_array.flatten()
        
        # Lista ramek
        frames = []
        
        for i in range(num_frames):
            frame = flat_data[i * values_per_frame : (i + 1) * values_per_frame]  # Pobranie wartości do ramki
            frame = frame.reshape(frame_size)  # Przekształcenie na macierz 4x4
            frames.append(frame)
        
        return np.array(frames)

    def embedding(self, carrier, secret):

        data = carrier.data_signed_int16
        fs = carrier.fs

        start_embedding = timer()

        carrier_frames = self.divide_into_frames(data, self.frame_length, fs)

        zerro_crossings = []
        ste_values = []
        labels = []

        for carrier_frame in carrier_frames:
            # Obliczenie liczby zer przekraczających próg
            zero_crossing = self.zerocrossing_count(carrier_frame)
            ste_value = self.short_time_energy(carrier_frame)

            zerro_crossings.append(zero_crossing)
            ste_values.append(ste_value)

        ste_threshold, zcc_threshold = self.compute_thresholds(ste_values, zerro_crossings)

        for ste, zcc in zip(ste_values, zerro_crossings):
            if ste >= ste_threshold and zcc <= zcc_threshold:  # Dźwięczne: wysokie STE i niskie ZCC
                labels.append('VOICED')
            else:  # Bezdźwięczne: umiarkowane STE i wysokie ZCC
                labels.append('UNVOICED')

        voiced_frames = []
        unvoiced_frames = []
        
        for carrier_frame, label in zip(carrier_frames, labels): 
            if(label == 'VOICED'):
                voiced_frames.append(carrier_frame)
            else:
                unvoiced_frames.append(carrier_frame)

        voiced_frames_samples_count = self.count_samples_in_frames(voiced_frames)
        voiced_frames_samples_reframed = np.concatenate(voiced_frames)

        self.capacity = np.round(self.calculate_capacity(voiced_frames_samples_reframed)).astype(int)

        if(secret.type == 'MAX_CAP'):
            secret.create_secret_from_dummy_text(self.capacity)

        self.secret = secret
        
        if(secret.size > self.capacity):
            raise Exception(f"Method doesnt have enough capacity - method capacity {self.capacity} - secret size {secret.size}")
        
        self.save_chart_signal_with_classification(data, fs, labels)

        secret_frames = self.create_secret_frames(secret.binary_data)

        carrier_frames = self.create_carrier_frames(voiced_frames_samples_reframed)

        embedded_frames = [self._embed_frame(c_frame, s_frame) for c_frame, s_frame in zip(carrier_frames, secret_frames)]

        embedded_frames = np.concatenate(embedded_frames).round().astype(int)

        embedded_frames = self.divide_into_frames(embedded_frames, self.frame_length, fs)

        embedded_frames = embedded_frames.tolist()

        stegobject = []

        for label in labels:
            if(label == 'VOICED'):
                stegobject.append(embedded_frames.pop(0))
            else:
                stegobject.append(unvoiced_frames.pop(0))

        stegobject = np.concatenate(stegobject)

        stop_embedding = timer()
        
        self.embedding_time = stop_embedding - start_embedding

        return stegobject.astype(np.int16)

    def save_chart_signal_with_classification(self, signal, fs, labels):
        """
        Zapisuje w katalogu 'charts' dla testu wizualizacje, dla różnych typów ramek.
        
        Args:
            signal (numpy array): Sygnał wejściowy.
            fs (int): Częstotliwość próbkowania.
            labels (list): Lista etykiet ramek ('cisza', 'bezdźwięczne', 'dźwięczne').
            ste_values (list): Lista wartości STE dla każdej ramki.
            ste_threshold (float): Próg dla STE.
        """
        time = np.linspace(0, len(signal) / fs, len(signal))
        frame_duration_samples = len(signal) // len(labels)

        signal = signal.astype(np.float32) / 32767.0

        # Kolory dla różnych etykiet
        colors = {'UNVOICED': 'blue', 'VOICED': 'green'}

        plt.figure(figsize=(15, 6))

        # Wykres sygnału
        plt.plot(time, signal, label='Sygnał', color='black', alpha=0.7)

        # Dodawanie kolorów dla ramek
        for i, label in enumerate(labels):
            start = i * frame_duration_samples
            end = start + frame_duration_samples
            plt.axvspan(time[start], time[end - 1], color=colors[label], alpha=0.3, label=label if i == 0 else None)

        plt.title("Sygnał z klasyfikacją ramek")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")

        filename = self.name + "_signal_with_classification.png"
        filename = filename.replace(".wav", "")

        plt.savefig(f"{self.charts_directory}/{filename}")

    def count_samples_in_frames(self, frames):
        """
        Liczy całkowitą liczbę próbek w podanych ramach.
        
        Args:
            frames (list of numpy arrays): Lista ramek.
            
        Returns:
            int: Liczba próbek we wszystkich ramach.
        """
        return sum(len(frame) for frame in frames)
        
    def compute_thresholds(self, ste_values, zcc_values, ste_factor=0.5, zcc_factor=0.5):
        """
        Oblicza progi dla Short-Time Energy (STE) i Zero Crossing Count (ZCC).
        
        Args:
            ste_values (list or numpy array): Lista lub tablica wartości STE.
            zcc_values (list or numpy array): Lista lub tablica wartości ZCC.
            ste_factor (float): Współczynnik regulacji dla progu STE (domyślnie 1.5).
            zcc_factor (float): Współczynnik regulacji dla progu ZCC (domyślnie 1.0).
            
        Returns:
            tuple: Progi dla STE i ZCC w formacie (ste_threshold, zcc_threshold).
        """
        # Obliczanie średniej i odchylenia standardowego dla STE
        ste_mean = np.mean(ste_values)
        ste_std = np.std(ste_values)
        ste_threshold = ste_mean
        
        # Obliczanie średniej i odchylenia standardowego dla ZCC
        zcc_mean = np.mean(zcc_values)
        zcc_std = np.std(zcc_values)
        zcc_threshold = zcc_mean
        
        return ste_threshold, zcc_threshold

    def _embed_frame(self, carrier_frame, secret_frame):
        A = dct(carrier_frame, norm='ortho')
        U, s, VT = svd(A, full_matrices=True)
        S = np.diag(s)
        modified_S = S + (secret_frame * self.alpha)
        u1, s1, v1 = svd(modified_S, full_matrices=True)
        self.U1.append(u1)
        self.V1.append(v1)
        self.S1.append(S)

        return idct(U.dot(np.diag(s1)).dot(VT), norm='ortho').flatten()

    def extraction(self, stegobject, secret_size):

        data = stegobject['content_signed_int16']
        fs = stegobject['fs']

        extration_start = timer()

        stegobject_frames = self.divide_into_frames(data, self.frame_length, fs)

        zerro_crossings = []
        ste_values = []

        for stegobject_frame in stegobject_frames:
            # Obliczenie liczby zer przekraczających próg
            zero_crossing = self.zerocrossing_count(stegobject_frame)
            ste_value = self.short_time_energy(stegobject_frame)

            zerro_crossings.append(zero_crossing)
            ste_values.append(ste_value)

        ste_threshold, zcc_threshold = self.compute_thresholds(ste_values, zerro_crossings)

        labels = []

        for ste, zcc in zip(ste_values, zerro_crossings):
            if ste >= ste_threshold and zcc <= zcc_threshold:
                labels.append('VOICED')
            else:
                labels.append('UNVOICED')

        voiced_frames = []
        unvoiced_frames = []

        for stegobject_frame, label in zip(stegobject_frames, labels):
            if(label == 'VOICED'):
                voiced_frames.append(stegobject_frame)
            else:
                unvoiced_frames.append(stegobject_frame)

        voiced_frames_samples = self.count_samples_in_frames(voiced_frames)
        voiced_frames_samples_reframed = np.concatenate(voiced_frames)

        stego_frames = self.create_carrier_frames(voiced_frames_samples_reframed)

        extracted_frames = [self._extract_frame(frame, u1, v1, S) for frame, u1, v1, S in zip(stego_frames, self.U1, self.V1, self.S1)]
        secret = self.extract_secret(extracted_frames)

        secret = secret[:secret_size]

        extraction_stop = timer()

        self.extraction_time = extraction_stop - extration_start
        
        return secret

    def _extract_frame(self, frame, u1, v1, S):
        ssv = dct(frame, norm='ortho')
        Us, Ssvn, Vs = svd(ssv, full_matrices=True)
        Lambda = np.diag(Ssvn)
        SSN = u1.dot(Lambda.dot(v1))
        return (SSN - S) / self.alpha

    def extract_secret(self, secret_frames):
        stegobject_bits = []
        max_val = 2 ** 1
        for frame in secret_frames:
            mask = np.ones(frame.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            frame_values = frame[mask].flatten()
            values = np.clip(np.where(frame_values < 0, 0, frame_values.round().astype(int)), 0, max_val - 1)
            stegobject_bits.extend([str(BitVector(intVal=value, size=1)) for value in values])
        return BitVector(bitstring=''.join(stegobject_bits))