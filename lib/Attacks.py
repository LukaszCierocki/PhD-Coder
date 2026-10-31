import numpy
import soundfile as sf
import os
from scipy.signal import butter, filtfilt, resample_poly
import subprocess
import audioop

class Attacks:
    def __init__(self, logger=None):

        self.logger = logger
        pass
    
    def add_gaussian_noise(self, input_path, file_name, noise_level=0.005):

        data, samplerate = sf.read(os.path.join(input_path, file_name))

        noise = numpy.random.normal(0, noise_level * numpy.max(data), len(data))

        noisy_audio = data + noise.astype(numpy.int16)
        
        output_path = os.path.join(input_path, 'noised_' + file_name)

        sf.write(output_path, noisy_audio, samplerate)

    def g711_attack(self, input_path, file_name, law='mu', target_fs=8000):
        """
        Symuluje przejście przez kodek G.711.
        - law: 'mu' / 'u' / 'ulaw' -> mu-law (u-law)
               'a' / 'alaw' -> a-law
        - target_fs: docelowa częstotliwość dla kodeka (domyślnie 8000 Hz)
        Zwraca ścieżkę zapisanego pliku.
        """
        try:
            full_path = os.path.join(input_path, file_name)
            data, fs = sf.read(full_path)  # data: float lub int, shape (N,) lub (N, C)
            orig_fs = fs

            def _resample_1d(arr, src_fs, dst_fs):
                if src_fs == dst_fs:
                    return arr.astype(numpy.int16, copy=False)

                # redukcja ułamka src_fs/dst_fs do małych liczb całkowitych
                # np. 44100 -> 8000
                from math import gcd
                g = gcd(int(src_fs), int(dst_fs))
                up = int(dst_fs // g)
                down = int(src_fs // g)

                resampled = resample_poly(arr.astype(numpy.float64), up, down)

                return numpy.clip(numpy.round(resampled), -32768, 32767).astype(numpy.int16)

            if data.dtype.kind == 'f':
                data_int = (data * 32767.0).round().astype(numpy.int16)
            else:
                data_int = data.astype(numpy.int16)

            if data_int.ndim == 1:
                channels = 1
                channel_arrays = [data_int]
            else:
                channels = data_int.shape[1]
                channel_arrays = [data_int[:, ch].astype(numpy.int16) for ch in range(channels)]

            # Resample to target_fs per channel
            if orig_fs != target_fs:
                channel_rs = [_resample_1d(ch_arr, orig_fs, target_fs) for ch_arr in channel_arrays]
            else:
                channel_rs = channel_arrays

            # Interleave channels into bytes for audioop
            if channels == 1:
                interleaved_bytes = channel_rs[0].tobytes()
            else:
                # stack columns interleaved: create int16 interleaved array
                stacked = numpy.stack(channel_rs, axis=1)
                interleaved_bytes = stacked.tobytes()

            # Apply G.711 encoding and decoding via audioop
            if law in ('mu', 'u', 'ulaw'):
                coded = audioop.lin2ulaw(interleaved_bytes, 2)
                recovered_bytes = audioop.ulaw2lin(coded, 2)
                suffix = 'ulaw'
            else:
                coded = audioop.lin2alaw(interleaved_bytes, 2)
                recovered_bytes = audioop.alaw2lin(coded, 2)
                suffix = 'alaw'

            recovered = numpy.frombuffer(recovered_bytes, dtype=numpy.int16)

            # De-interleave back to channels
            if channels == 1:
                recovered_arr = recovered
            else:
                recovered_arr = recovered.reshape(-1, channels)

            # If we resampled to target_fs, resample back to original fs
            if orig_fs != target_fs:
                if channels == 1:
                    back = _resample_1d(recovered_arr, target_fs, orig_fs)
                    final = back
                else:
                    ch_back = [_resample_1d(recovered_arr[:, ch], target_fs, orig_fs) for ch in range(channels)]
                    # Align lengths (trim to min)
                    minlen = min(len(c) for c in ch_back)
                    ch_back = [c[:minlen] for c in ch_back]
                    final = numpy.stack(ch_back, axis=1)
            else:
                final = recovered_arr

            output_filename = f"g711_{suffix}_attacked_" + file_name
            output_path = os.path.join(input_path, output_filename)

            # Save final signal (int16) with original sampling rate
            sf.write(output_path, final, orig_fs)

            if self.logger:
                self.logger.info(f"G.711 ({suffix}) attack saved: {output_filename}")
            return output_path

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error during g711_attack on {file_name}: {e}")
            raise

    def add_gaussian_noise_with_target_snr(self, input_path, file_name, target_snr_db=20):
        """
        Dodaje szum Gaussa do pliku audio w celu uzyskania zadanego SNR w dB.
        Obsługuje pliki mono i stereo.
        """

        self.logger.debug(f"Adding Gaussian noise with target SNR: {target_snr_db} dB to {file_name}")

        # Wczytaj dane
        data, samplerate = sf.read(os.path.join(input_path, file_name))

        # Konwersja na float64
        data = data.astype(numpy.float64)

        # Obsługa kanałów stereo
        if data.ndim == 1:
            channels = 1
            data = data[:, numpy.newaxis]
        else:
            channels = data.shape[1]

        noisy_audio = numpy.zeros_like(data)

        for ch in range(channels):
            signal = data[:, ch]

            # Moc sygnału
            signal_power = numpy.mean(signal ** 2)

            # Obliczenie mocy szumu z żądanego SNR
            snr_linear = 10 ** (target_snr_db / 10)
            noise_power = signal_power / snr_linear

            # Odchylenie standardowe szumu
            noise_std = numpy.sqrt(noise_power)

            # Generowanie szumu Gaussa
            noise = numpy.random.normal(0, noise_std, size=signal.shape)

            self.logger.debug(f"Channel {ch}: Signal power = {signal_power}, Noise power = {noise_power}, Noise std = {noise_std}")
            self.logger.info(f"Noise: mean = {numpy.mean(noise)}, std = {numpy.std(noise)}, max = {numpy.max(noise)}, min = {numpy.min(noise)}")

            # Dodanie szumu
            noisy_audio[:, ch] = signal + noise

        # Przeskalowanie do typu int16
        noisy_audio = numpy.clip(noisy_audio, -32768, 32767).astype(numpy.int16)

        # Zapis do pliku
        output_file = os.path.join(input_path, f"noised_{target_snr_db:.0f}dB_" + file_name)
        sf.write(output_file, noisy_audio, samplerate)
        
    def low_pass_filter(self, input_path, file_name, cutoff=3000):
        
        # output_path = input_path
        
        data, samplerate = sf.read(os.path.join(input_path, file_name))
        
        # Zdefiniuj filtr dolnoprzepustowy
        nyquist = 0.5 * samplerate
        norm_cutoff = cutoff / nyquist
        b, a = butter(5, norm_cutoff, btype='low', analog=False)
        
        # Przefiltruj sygnał
        filtered_data = filtfilt(b, a, data)
        
        output_path = os.path.join(input_path, 'filtered_low_pass_' + file_name)
        
        # Zapisz przefiltrowany sygnał do pliku
        sf.write(output_path, filtered_data, samplerate)

    def requantize_audio(self, input_path, file_name, bit_depth=8):
        """
        Rekwantyzuje sygnał audio do mniejszej liczby bitów i przywraca do oryginalnej rozdzielczości.
        
        Params:
        - input_path: ścieżka do folderu z plikiem
        - file_name: nazwa pliku WAV
        - bit_depth: docelowa liczba bitów (np. 8)

        Zapisuje nowy plik WAV z przywróconą oryginalną rozdzielczością.
        """
        full_path = os.path.join(input_path, file_name)
        data, samplerate = sf.read(full_path)

        # Jeśli stereo – zrób mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Zakres danych: załóżmy, że są w [-1, 1]
        data_clipped = numpy.clip(data, -1.0, 1.0)

        # Rekwantyzacja: przeskaluj do [0, levels-1], zaokrągl, potem z powrotem do [-1, 1]
        levels = 2 ** bit_depth
        quantized = numpy.round((data_clipped + 1) * (levels / 2 - 1))  # [0, levels-1]
        requantized = (quantized / (levels / 2 - 1)) - 1  # wróć do [-1, 1]

        # Zapisz sygnał po rekwantyzacji i przywróceniu
        output_file = f"requantized_{bit_depth}bit_" + file_name
        sf.write(os.path.join(input_path, output_file), requantized, samplerate)
        
    def high_pass_filter(self, input_path, file_name, cutoff=3000):
        """
        Funkcja implementująca filtr górnoprzepustowy.
        
        Parametry:
        - input_path (str): Ścieżka do katalogu z plikami audio.
        - file_name (str): Nazwa pliku audio, który ma być filtrowany.
        - cutoff (int): Częstotliwość odcięcia filtra (domyślnie 3000 Hz).
        
        Zwraca:
        - Zapisuje przefiltrowany plik audio jako nowy plik z prefiksem 'filtered_high_pass_'.
        """
        
        # Odczytaj dane z pliku audio
        data, samplerate = sf.read(os.path.join(input_path, file_name))
        
        # Zdefiniuj filtr górnoprzepustowy
        nyquist = 0.5 * samplerate  # Częstotliwość Nyquista
        norm_cutoff = cutoff / nyquist  # Znormalizowana częstotliwość odcięcia
        b, a = butter(5, norm_cutoff, btype='high', analog=False)  # Filtr górnoprzepustowy
        
        # Przefiltruj sygnał
        filtered_data = filtfilt(b, a, data)
        
        # Zapisz przefiltrowany sygnał do nowego pliku
        output_path = os.path.join(input_path, 'filtered_high_pass_' + file_name)
        sf.write(output_path, filtered_data, samplerate)
        
        return output_path
    
    def mp3_compression_attack(self, input_path, file_name, bitrate):
        """
        Atak przez stratną kompresję MP3 i konwersję z powrotem do WAV.
        Wymaga zainstalowanego ffmpeg.
        """
        input_full = os.path.join(input_path, file_name)
        mp3_path = os.path.join(input_path, "temp_compressed.mp3")
        recovered_path = os.path.join(input_path, "mp3_attacked_" + bitrate + "_" + file_name)

        subprocess.run(
            ["ffmpeg", "-y", "-i", input_full, "-b:a", bitrate, mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Dekompresja do WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, recovered_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Usunięcie tymczasowego pliku
        os.remove(mp3_path)

    def jitter_attack(self, input_path, file_name, jitter_strength=1):
        """
        Atak przez zaburzenie kolejności próbek w niewielkim zakresie.
        Może zakłócić osadzenie danych opartych na kolejności.
        """
        data, samplerate = sf.read(os.path.join(input_path, file_name))
        indices = numpy.arange(len(data))
        jitter = numpy.random.randint(-jitter_strength, jitter_strength + 1, size=len(data))
        jittered_indices = numpy.clip(indices + jitter, 0, len(data) - 1)
        jittered_data = data[jittered_indices]
        
        output_file = f"jitter_{jitter_strength}_" + file_name
        sf.write(os.path.join(input_path, output_file), jittered_data, samplerate)

    def packet_loss_attack(self, input_path, file_name, frame_ms=20, loss_rate=0.1, replace_with='zero'):
        """
        Symuluje utratę pakietów (packet loss) przez usunięcie (lub zastąpienie) losowych ramek o długości frame_ms milisekund.
        - loss_rate: od 0.0 do 1.0 (np. 0.1 = 10% ramek wyciętych)
        - replace_with: 'zero' lub 'noise' (zastąpienie ramki ciszą lub białym szumem)
        Zwraca ścieżkę zapisanego pliku.
        """
        try:
            full_path = os.path.join(input_path, file_name)
            data, fs = sf.read(full_path)

            # Pracujemy na 2D tablicy kanałów dla wygody
            was_mono = False
            if data.ndim == 1:
                was_mono = True
                data = data[:, numpy.newaxis]

            frame_len = int((frame_ms / 1000.0) * fs)
            if frame_len < 1:
                frame_len = 1

            n_frames = int(numpy.ceil(len(data) / frame_len))
            out = data.copy()

            for i in range(n_frames):
                if numpy.random.rand() < loss_rate:
                    start = i * frame_len
                    end = min(start + frame_len, len(data))
                    if replace_with == 'zero':
                        out[start:end, :] = 0
                    else:
                        # white noise scaled to signal std (per channel)
                        for ch in range(out.shape[1]):
                            sigma = numpy.std(data[:, ch]) if numpy.std(data[:, ch]) > 0 else 1.0
                            out[start:end, ch] = numpy.random.normal(0, sigma * 0.1, size=(end - start))

            final = out[:, 0] if was_mono else out

            output_file = os.path.join(input_path, f"packetloss_{int(loss_rate*100)}_{file_name}")
            sf.write(output_file, final, fs)

            if self.logger:
                self.logger.info(f"Packet loss attack saved: {output_file}")
            return output_file

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error during packet_loss_attack on {file_name}: {e}")
            raise

    def resample_attack(self, input_path, file_name, factor=2):
        """
        Atak przez zmianę częstotliwości próbkowania (downsampling i upsampling).
        Z interpolacją liniową podczas upsamplingu.
        """
        # Wczytanie sygnału
        data, samplerate = sf.read(os.path.join(input_path, file_name))
        
        # Obsługa kanałów stereo/mono
        if data.ndim == 1:
            channels = 1
            data = data[:, numpy.newaxis]
        else:
            channels = data.shape[1]

        # Inicjalizacja bufora na wynikowy sygnał
        upsampled = []

        for ch in range(channels):
            signal = data[:, ch]

            # Downsampling przez wybieranie co 'factor'-tej próbki
            downsampled = signal[::factor]

            # Tworzymy oryginalne i nowe indeksy czasowe
            x_old = numpy.arange(0, len(downsampled)) * factor
            x_new = numpy.arange(0, len(signal))

            # Interpolacja liniowa
            interpolated = numpy.interp(x_new, x_old, downsampled)
            upsampled.append(interpolated)

        # Złożenie kanałów
        upsampled = numpy.stack(upsampled, axis=1) if channels > 1 else upsampled[0]

        # Zapis do pliku
        output_file = f"resample_attack_x{factor}_" + file_name
        sf.write(os.path.join(input_path, output_file), upsampled, samplerate)
    
    def clipping_attack(self, input_path, file_name, threshold):
        """
        Atak przez skalowanie amplitudy sygnału.
        Może zakłócić dane ukryte w sygnale poprzez zmianę jego poziomu energetycznego.

        Parametry:
        - threshold: wartość mnożnika amplitudy (np. 0.7 oznacza 70% amplitudy, 1.3 oznacza wzmocnienie o 30%)
        """

        data, samplerate = sf.read(os.path.join(input_path, file_name))

        scaled = data * threshold

        scaled = numpy.clip(scaled, -1.0, 1.0)

        output_file = f"scaled_{int(threshold * 100)}_" + file_name
        sf.write(os.path.join(input_path, output_file), scaled, samplerate)

    def perfome_attacks(self, method_stegobjects_directory, test_name, test_to_perform):
        """
        Wykonuje ataki na pliki audio w oparciu o zdefiniowane metody i ataki.
        """
        if test_to_perform['NOISE']:
            self.add_gaussian_noise_with_target_snr(method_stegobjects_directory, test_name + '.wav', target_snr_db=30)

        if test_to_perform['HIGHPASS']:
            self.high_pass_filter(method_stegobjects_directory, test_name + '.wav', cutoff=500)
        
        if test_to_perform['LOWPASS']:
            self.low_pass_filter(method_stegobjects_directory, test_name + '.wav', cutoff=6500)
        
        if test_to_perform['REQUANTIZE']:
            self.requantize_audio(method_stegobjects_directory, test_name + '.wav', bit_depth=10)

        if test_to_perform['MP3']:
            for bitrate in ["64k", "128k", "192k"]:
                self.mp3_compression_attack(method_stegobjects_directory, test_name + '.wav', bitrate=bitrate)
        
        if test_to_perform['JITTER']:
            self.jitter_attack(method_stegobjects_directory, test_name + '.wav', 1)

        if test_to_perform['RESAMPLE']:
            self.resample_attack(method_stegobjects_directory, test_name + '.wav', 2)

        if test_to_perform['CLIPPED']:
            for clipping_lvl in [0.7, 1.3]:
                self.clipping_attack(method_stegobjects_directory, test_name + '.wav', threshold=clipping_lvl)
        if test_to_perform.get('PACKET_LOSS'):
            try:
                # domyślnie 20 ms ramki, 10% strat
                self.packet_loss_attack(method_stegobjects_directory, test_name + '.wav', frame_ms=20, loss_rate=0.1, replace_with='zero')
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error performing PACKET_LOSS attack on {test_name}: {e}")
        if test_to_perform.get('G711'):
            try:
                self.g711_attack(method_stegobjects_directory, test_name + '.wav', law='mu', target_fs=8000)
                self.g711_attack(method_stegobjects_directory, test_name + '.wav', law='a', target_fs=8000)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error performing G.711 attack on {test_name}: {e}")
