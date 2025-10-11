import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import math
from scipy.stats import chisquare
from scipy.stats import entropy
from scipy.stats import skew
from scipy.signal import spectrogram
from pesq import pesq
from timeit import default_timer as timer
from pystoi.stoi import stoi
import pandas as pd
from tabulate import tabulate
from collections import defaultdict
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.stats import norm
import itertools




from colorama import Fore, Style, init

from collections import defaultdict

init(autoreset=True)

class Analysis:
    
    def __init__(self, logger):
        
        self.methods_analysis = []
        self.logger = logger
        self.singular_values_per_carrier = []

    def plot_singular_values(self, save_path=None, use_log_scale=False):

        singular_values_per_carrier = self.singular_values_per_carrier

        grouped = defaultdict(list)
        for item in singular_values_per_carrier:
            grouped[item['carrier_name']].append(item)

        num_carriers = len(grouped)
        fig, axes = plt.subplots(num_carriers, 1, figsize=(10, 5 * num_carriers), sharex=True)

        if num_carriers == 1:
            axes = [axes]

        for ax, (carrier_name, methods) in zip(axes, grouped.items()):
            for method in methods:
                s_even = np.array(method['s_even'])
                s_odd = np.array(method['s_odd'])
                new_s_even = np.array(method['new_s_even'])
                new_s_odd = np.array(method['new_s_odd'])
                method_name = method['method_name']

                # Sprawdzamy długości
                if len(s_even) != len(s_odd) or len(new_s_even) != len(new_s_odd):
                    print(f"❗ Nierówne długości dla {method_name} — pomijam.")
                    continue

                mean_orig = (s_even + s_odd) / 2
                mean_new = (new_s_even + new_s_odd) / 2
                indices = np.arange(len(mean_orig))

                ax.plot(indices, mean_orig, 'o-', label=f"{method_name} - mean(orig)", alpha=0.7)
                ax.plot(indices, mean_new, 'x--', label=f"{method_name} - mean(new)", alpha=0.7)

            ax.set_title(f"Carrier: {carrier_name}", fontsize=14, fontweight='bold')
            ax.set_ylabel("Mean singular value", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)

            if use_log_scale:
                ax.set_yscale('log')

            ax.legend(fontsize=9, loc='best')

        axes[-1].set_xlabel("Singular value index", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, 'methods_mean_singular_values_per_carrier.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        
    def plot_audio_waveform(self, data, fs, current_test_charts_directory, filename):
        
        time = numpy.linspace(0, len(data) / fs, len(data))
        
        # Tworzenie wykresu
        
        data = data / numpy.max(numpy.abs(data))
        
        plt.figure(figsize=(14, 5))
        plt.plot(time, data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform of ' + filename)

        filename = filename.replace('.wav', '')
        plt.savefig(os.path.join(current_test_charts_directory, filename + '_waveform.png'))
        plt.close()

    def plot_singular_values_and_delta(self, method, carrier_name, save_path=None, filename=None, normalize=True):
        """
        method — obiekt z atrybutami:
            s_even, s_odd, new_s_even, new_s_odd, name
        """

        s_even = np.array(method.s_even)
        s_odd = np.array(method.s_odd)
        new_s_even = np.array(method.new_s_even)
        new_s_odd = np.array(method.new_s_odd)
        method_name = method.normal_name

        indices = np.arange(len(s_even))

        delta_s_orig = s_even - s_odd
        delta_s_mod = new_s_even - new_s_odd

        # Normalizacja Z-score
        if normalize:
            mean_orig_val = np.mean(delta_s_orig)
            std_orig_val = np.std(delta_s_orig)
            delta_s_orig = (delta_s_orig - mean_orig_val) / std_orig_val

            mean_mod_val = np.mean(delta_s_mod)
            std_mod_val = np.std(delta_s_mod)
            delta_s_mod = (delta_s_mod - mean_mod_val) / std_mod_val

        mean_orig = np.mean(delta_s_orig)
        mean_mod = np.mean(delta_s_mod)

        fig, (ax_s, ax_hist) = plt.subplots(1, 2, figsize=(18, 6))

        # Singular values
        ax_s.plot(indices, s_even, color='tab:blue', marker='.', alpha=0.6, label="s_even orig")
        ax_s.plot(indices, s_odd, color='tab:green', marker='.', alpha=0.6, label="s_odd orig")
        ax_s.plot(indices, new_s_even, color='tab:blue', linestyle='--', marker='x', alpha=0.9, label="s_even mod")
        ax_s.plot(indices, new_s_odd, color='tab:green', linestyle='--', marker='x', alpha=0.9, label="s_odd mod")

        ax_s.set_title(f"{carrier_name} — Singular values", fontsize=14, fontweight='bold')
        ax_s.set_xlabel("Frame index", fontsize=12)
        ax_s.set_ylabel("Singular value", fontsize=12)
        ax_s.grid(True, linestyle='--', alpha=0.5)
        ax_s.legend(fontsize=9)

        # Histogram Δs
        bins = 50
        ax_hist.hist(delta_s_orig, bins=bins, alpha=0.4, color='tab:purple', label="original Δs", density=True)
        ax_hist.hist(delta_s_mod, bins=bins, alpha=0.4, color='tab:orange', label="modified Δs", density=True)

        # Krzywe KDE
        kde_orig = gaussian_kde(delta_s_orig)
        kde_mod = gaussian_kde(delta_s_mod)

        x_vals = np.linspace(min(delta_s_orig.min(), delta_s_mod.min()), 
                            max(delta_s_orig.max(), delta_s_mod.max()), 500)

        ax_hist.plot(x_vals, kde_orig(x_vals), color='tab:purple', linestyle='-', linewidth=2, label="KDE orig Δs")
        ax_hist.plot(x_vals, kde_mod(x_vals), color='tab:orange', linestyle='-', linewidth=2, label="KDE mod Δs")

        # Średnie linie
        ax_hist.axvline(mean_orig, color='tab:purple', linestyle='--', linewidth=2, label="mean orig Δs")
        ax_hist.axvline(mean_mod, color='tab:orange', linestyle='--', linewidth=2, label="mean mod Δs")

        ax_hist.set_title("Histogram of Δs with KDE", fontsize=14)
        ax_hist.set_xlabel("Δs value (s1 - s2)", fontsize=12)
        ax_hist.set_ylabel("Density", fontsize=12)
        ax_hist.grid(True, linestyle='--', alpha=0.5)
        ax_hist.legend(fontsize=9)

        plt.tight_layout()

        if filename:
            filename = filename.replace('.wav', '')
            fname = f"{filename}_singular_values_and_delta.png"
        else:
            fname = f"{carrier_name}_{method_name}_singular_values_and_delta.png"

        if save_path:
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def plot_grouped_singular_values_and_delta(self, save_path=None):
        """
        methods_list — lista dictów z atrybutami:
            s_even, s_odd, new_s_even, new_s_odd, method_name, carrier_name
        """

        methods_list = self.singular_values_per_carrier

        # Grupowanie po carrier_name
        grouped = defaultdict(list)
        for method in methods_list:
            grouped[method['carrier_name']].append(method)

        for carrier_name, methods in grouped.items():
            num_methods = len(methods)
            fig, axs = plt.subplots(num_methods, 2, figsize=(18, 5 * num_methods))

            # Obsługa przypadku gdy jedna metoda
            if num_methods == 1:
                axs = np.expand_dims(axs, axis=0)

            for i, method in enumerate(methods):
                s_even = np.array(method['s_even'])
                s_odd = np.array(method['s_odd'])
                new_s_even = np.array(method['new_s_even'])
                new_s_odd = np.array(method['new_s_odd'])
                method_name = method['method_name']

                indices = np.arange(len(s_even))

                # Δs
                delta_s_orig = s_even - s_odd
                delta_s_mod = new_s_even - new_s_odd

                mean_orig = np.mean(delta_s_orig)
                mean_mod = np.mean(delta_s_mod)

                ax_s, ax_hist = axs[i]

                # Singular values
                ax_s.plot(indices, s_even, 'b.-', alpha=0.6, label="s_even orig")
                ax_s.plot(indices, s_odd, 'g.-', alpha=0.6, label="s_odd orig")
                ax_s.plot(indices, new_s_even, 'bx--', alpha=0.8, label="s_even mod")
                ax_s.plot(indices, new_s_odd, 'gx--', alpha=0.8, label="s_odd mod")

                ax_s.set_title(f"{carrier_name} — {method_name} singular values", fontsize=14, fontweight='bold')
                ax_s.set_xlabel("Frame index", fontsize=12)
                ax_s.set_ylabel("Singular value", fontsize=12)
                ax_s.grid(True, linestyle='--', alpha=0.5)
                ax_s.legend(fontsize=8)

                # Histogram Δs + KDE
                bins = 50
                ax_hist.hist(delta_s_orig, bins=bins, alpha=0.5, label="orig Δs", color='blue', density=True)
                ax_hist.hist(delta_s_mod, bins=bins, alpha=0.5, label="mod Δs", color='red', density=True)

                kde_orig = gaussian_kde(delta_s_orig)
                kde_mod = gaussian_kde(delta_s_mod)

                x_vals = np.linspace(min(delta_s_orig.min(), delta_s_mod.min()), 
                                    max(delta_s_orig.max(), delta_s_mod.max()), 500)

                ax_hist.plot(x_vals, kde_orig(x_vals), color='blue', linestyle='-', linewidth=2, label="KDE orig Δs")
                ax_hist.plot(x_vals, kde_mod(x_vals), color='red', linestyle='--', linewidth=2, label="KDE mod Δs")

                ax_hist.axvline(mean_orig, color='blue', linestyle=':', linewidth=2, label="mean Δs orig")
                ax_hist.axvline(mean_mod, color='red', linestyle=':', linewidth=2, label="mean Δs mod")

                ax_hist.set_title(f"{method_name} Δs histogram + KDE", fontsize=14)
                ax_hist.set_xlabel("Δs value (s1 - s2)", fontsize=12)
                ax_hist.set_ylabel("Density", fontsize=12)
                ax_hist.grid(True, linestyle='--', alpha=0.5)
                ax_hist.legend(fontsize=8)

            plt.tight_layout()

            # Zapis
            if save_path:
                fname = f"{carrier_name}_grouped_singular_values_and_delta.png"
                plt.savefig(os.path.join(os.path.join(save_path, 'Grouped Charts'), fname), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def plot_signal_with_singular_values(self, data, method ,fs, current_test_charts_directory, filename):

        time = numpy.linspace(0, len(data) / fs, len(data))
        
        data = data / numpy.max(numpy.abs(data))

        s_even = method.s_even
        s_odd = method.s_odd

        new_s_even = method.new_s_even
        new_s_odd = method.new_s_odd

        duration = len(data) / fs
        frame_duration = 0.005  # 5 ms
        frame_size = int(fs * frame_duration)

        s_even = numpy.array(s_even)
        s_odd = numpy.array(s_odd)

        new_s_even = numpy.array(new_s_even)
        new_s_odd = numpy.array(new_s_odd)


        s_avg = (s_even + s_odd)/2
        new_s_avg = (new_s_even + new_s_odd)/2

        # Normalizacja osobliwości do [-1, 1]
        # s_even = 2 * (s_even - numpy.min(s_even)) / (numpy.max(s_even) - numpy.min(s_even)) - 1
        # s_odd = 2 * (s_odd - numpy.min(s_odd)) / (numpy.max(s_odd) - numpy.min(s_odd)) - 1

        s_avg = 2 * (s_avg - numpy.min(s_avg)) / (numpy.max(s_avg) - numpy.min(s_avg)) - 1
        new_s_avg = 2 * (new_s_avg - numpy.min(new_s_avg)) / (numpy.max(new_s_avg) - numpy.min(new_s_avg)) - 1

        frame_times_avg = numpy.linspace(0, duration, len(s_avg), endpoint=False)
        # frame_times_odd = numpy.linspace(0, duration, len(s_odd), endpoint=False)

        # Wykres
        plt.figure(figsize=(15, 6))
        plt.plot(time, data, label="Normalized audio signal", alpha=0.6)
        plt.plot(frame_times_avg, s_avg, 'o-', label="singular values (SVD, norm, avg)", markersize=4)
        plt.plot(frame_times_avg, new_s_avg, 'x-', label="new singular values (SVD, norm, avg)", markersize=4)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude / Singular value (z-norm)")
        plt.title("Audio signal and singular values (SVD) for " + filename)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = filename.replace('.wav', '')
        plt.savefig(os.path.join(current_test_charts_directory, filename + '_waveform_with_singular.png'))
        plt.close()

    def plot_signal_difference(self, signal_a, signal_b, fs, output_dir, filename_1, filename_2):
        """
        Creates two separate plots:
        1. Original int16 signals with overlaid difference.
        2. Histogram of amplitude differences.

        Parameters:
        - signal_a (np.ndarray): First audio signal (int16).
        - signal_b (np.ndarray): Second audio signal (int16).
        - fs (int): Sampling frequency (Hz).
        - output_dir (str): Path to save plots.
        - filename_1 (str): Name of the first file.
        - filename_2 (str): Name of the second file.
        """

        # Konwersja do NumPy int16
        signal_a = numpy.asarray(signal_a, dtype=numpy.int16)
        signal_b = numpy.asarray(signal_b, dtype=numpy.int16)

        # Dopasowanie długości sygnałów
        min_length = min(len(signal_a), len(signal_b))
        signal_a = signal_a[:min_length]
        signal_b = signal_b[:min_length]

        # Obliczenie różnicy (zabezpieczenie przed przepełnieniem)
        difference = signal_a.astype(numpy.int32) - signal_b.astype(numpy.int32)

        # Oś czasu
        t = numpy.arange(min_length) / fs

        # ---- Pierwszy wykres: Sygnały A i B z zaznaczoną różnicą ----
        plt.figure(figsize=(12, 5))
        plt.plot(t, signal_a, color='blue', alpha=0.5, linewidth=1, label='Signal A (int16)')
        plt.plot(t, signal_b, color='green', alpha=0.5, linewidth=1, label='Signal B (int16)')
        plt.plot(t, difference, color='red', alpha=0.7, linewidth=0.8, linestyle='dashed', label='Difference')

        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude (int16)')
        plt.title('Comparison of Original Signals with Difference')
        plt.legend()
        plt.grid(True)

        # Zapisywanie wykresu
        filename_1_clean = os.path.splitext(filename_1)[0]
        filename_2_clean = os.path.splitext(filename_2)[0]
        output_path_1 = os.path.join(output_dir, f"{filename_2_clean}_to_carrier_difference.png")
        plt.tight_layout()
        plt.savefig(output_path_1)
        plt.close()

        # ---- Drugi wykres: Histogram różnicy ----
        plt.figure(figsize=(8, 5))
        plt.hist(difference, bins=50, color='red', alpha=0.7)
        plt.xlabel('Difference Values')
        plt.ylabel('Number of Samples')
        plt.title('Histogram of Amplitude Differences')
        plt.grid(True)

        # Zapisywanie histogramu
        output_path_2 = os.path.join(output_dir, f"{filename_2_clean}_to_carrier_histogram.png")
        plt.tight_layout()
        plt.savefig(output_path_2)
        plt.close()

    def plot_alphas_betas(self, method, save_path=None, filename=None):
        """
        method — obiekt z listami alphas i betas.
        """
        alphas = np.array(method.alphas)
        betas = np.array(method.betas)
        indices = np.arange(len(alphas))

        mean_alpha = np.mean(alphas)
        mean_beta = np.mean(betas)

        plt.figure(figsize=(15, 6))
        plt.plot(indices, alphas, 'b.-', label='Alpha', alpha=0.7)
        plt.plot(indices, betas, 'r.-', label='Beta', alpha=0.7)
        plt.hlines(mean_alpha, xmin=0, xmax=len(indices)-1, colors='blue', linestyles='--', label=f'Mean Alpha = {mean_alpha:.4f}')
        plt.hlines(mean_beta, xmin=0, xmax=len(indices)-1, colors='red', linestyles='--', label=f'Mean Beta = {mean_beta:.4f}')

        plt.title(f"Adaptive values α and β per frame in {method.normal_name}" , fontsize=14, fontweight='bold')
        plt.xlabel("Frame index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()

        if filename:
            filename = filename.replace('.wav', '')
            fname = f"{filename}_alphas_betas_plot.png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def calculate_bitrate(self, frame_duration_ms: float, sampling_rate: int, hop_duration_ms: float = None, bits_per_frame: int = 1) -> float:
        """
        Oblicza przepływność w bps na podstawie długości ramki, częstotliwości próbkowania oraz hop size.

        Args:
            frame_duration_ms (float): Długość ramki w milisekundach.
            sampling_rate (int): Częstotliwość próbkowania w Hz.
            hop_duration_ms (float): Przesunięcie między ramkami w milisekundach (jeśli None, przyjmujemy brak nakładania, czyli hop = frame_duration).
            bits_per_frame (int): Liczba bitów osadzanych w jednej ramce.

        Returns:
            float: Przepływność w bitach na sekundę (bps).
        """
        if hop_duration_ms is None:
            hop_duration_ms = frame_duration_ms

        # Czas trwania jednej ramki w sekundach (tylko dla obliczenia próbek w ramce — nie używamy tutaj do bitrate)
        frame_duration_sec = frame_duration_ms / 1000.0

        # Przesunięcie (hop) w sekundach
        hop_duration_sec = hop_duration_ms / 1000.0

        # Liczba ramek na sekundę
        frames_per_second = 1.0 / hop_duration_sec

        # Przepływność
        bitrate_bps = bits_per_frame * frames_per_second

        return bitrate_bps

    def analyze(self, method, carrier, secret, stegobject, extracted_data):

        stegobject_name = stegobject['filename']
        stegobject_fs = stegobject['fs']
        stegobject = stegobject['content_signed_int16']

        current_test_charts_directory = method.method_directories['Charts']
        current_carrier_charts_directory = method.method_directories['Carriers_Charts']

        self.logger.info(f"Analyzing method: {method.normal_name} on carrier: {carrier.name} with stegobject: {stegobject_name}")
        
        if 'filtered_low_pass' in stegobject_name:
            test_name = 'FILTERED_LOW_PASS'
        elif 'filtered_high_pass' in stegobject_name:
            test_name = 'FILTERED_HIGH_PASS'
        elif 'noised' in stegobject_name:
            test_name = 'NOISED'
        elif 'requantized' in stegobject_name:
            test_name = 'REQUANTIZED'
        elif 'mp3_attacked' in stegobject_name:
            parts = stegobject_name.split("_")
            bitrate = parts[2]
            test_name = 'MP3_ATTACKED' + f'_{bitrate}'
        elif 'jitter' in stegobject_name:
            test_name = 'JITTER'
        elif 'resample_attack' in stegobject_name:
            test_name = 'RESAMPLE_ATTACK'
        elif 'scaled' in stegobject_name:
            parts = stegobject_name.split("_")
            scale_factor = parts[1]
            test_name = 'SCALED' + f'_{scale_factor}'
        else:
            test_name = 'NONE'
        
        method_analysis = {
            'NAME' : method.normal_name,
            'TEST NAME' : method.name,
            'LEVELS' : method.levels,
            'FRAMES' : method.where_to_hide,
            'VAD' : method.type_of_vad,
            'ADAPTIVE' : method.adaptive,
            'ALPHA' : method.alpha,
            'BETA' : method.beta,
            'CARRIER' : carrier.name,
            'TEST' : test_name,
            'LEVELS' : method.levels,
            'SECRET' : method.secret.name,
            'SEC. SIZE [b]' : method.secret.size,
            'MSE' : round(self.calculate_mean_squared_error(carrier.data_signed_int16, stegobject), 3),
            'PSNR [dB]' : round(self.calculate_peak_signal_to_noise_ratio(carrier, stegobject), 3),
            'PESQ' : round(self.calculate_perceptual_evaluation_of_speech_quality(carrier.fs, carrier.data_signed_int16, stegobject), 3),
            'STOI' : round(self.calculate_stoi(carrier.data_signed_int16, stegobject, carrier.fs), 3),
            'BER [%]' : round(self.calculate_bit_error_rate(method.secret, extracted_data), 3),
            'CAP. [%]' : round((method.capacity / carrier.carrier_size_in_bits) * 100, 3),
            'CAP. [bps]' : round(self.calculate_bitrate(method.frame_length, carrier.fs, method.hop_duration_ms, method.bits_per_frame), 3),
            'EMB TIME [s]' : round(method.embedding_time, 3),
            'EXT TIME [s]' : round(method.extraction_time, 3),
            'DOMAIN' : method.domain,
            'STYLE' : method.style
        }
        

        self.plot_audio_waveform(carrier.data_signed_int16, carrier.fs, current_carrier_charts_directory, carrier.name)
        
        self.plot_audio_waveform(stegobject, stegobject_fs, current_test_charts_directory, stegobject_name)
        
        self.plot_signal_difference(carrier.data_signed_int16, stegobject, carrier.fs, current_test_charts_directory, carrier.name, stegobject_name)

        self.plot_signal_spectrogram(carrier.data_signed_int16, carrier.fs, current_carrier_charts_directory, carrier.name)
        
        if('DIFFERENTIAL_SVD' or 'MDCT_SVD_WEN' in method.name):
            self.plot_signal_with_singular_values(stegobject, method, stegobject_fs, current_test_charts_directory, stegobject_name)
            self.plot_signal_with_singular_values_and_spectrogram(stegobject, method, stegobject_fs, current_test_charts_directory, stegobject_name)
            self.plot_singular_values_and_delta(method, carrier.name, current_test_charts_directory, stegobject_name)
            self.plot_alphas_betas(method, current_test_charts_directory, stegobject_name)

        self.singular_values_per_carrier.append({
            'carrier_name': carrier.name,
            'method_name': method.normal_name,
            's_even': method.s_even,
            's_odd': method.s_odd,
            'new_s_even': method.new_s_even,
            'new_s_odd': method.new_s_odd
        })

        self.methods_analysis.append(method_analysis)

    def plot_signal_spectrogram(self, data, fs, current_test_charts_directory, filename):

        """
        Rysuje spektrogram sygnału audio i zapisuje go do pliku.
        
        :param data: numpy array, sygnał audio
        :param fs: int, częstotliwość próbkowania (Hz)
        :param current_test_charts_directory: str, ścieżka do katalogu, gdzie zapisać wykres
        :param filename: str, nazwa pliku audio (bez rozszerzenia)
        """
        
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=1024, noverlap=512)

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        plt.colorbar(label='Intensity [dB]')
        plt.title('Spectrogram of ' + filename)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        filename = filename.replace('.wav', '')
        plt.savefig(os.path.join(current_test_charts_directory, filename + '_spectrogram.png'))
        plt.close()

    def calculate_ssim(self, secret, extracted_data):
        
        result = ssim(secret, extracted_data, data_range=secret.max() - secret.min())
        
        return result
    
    def calculate_stoi(self, reference: numpy.ndarray, degraded: numpy.ndarray, fs: int = 16000, extended: bool = True) -> float:
        """
        Oblicza STOI lub Extended STOI między dwoma sygnałami dźwiękowymi.
        
        Parametry:
        - reference: numpy array z czystym (referencyjnym) sygnałem
        - degraded: numpy array ze zniekształconym/przetworzonym sygnałem
        - fs: częstotliwość próbkowania (domyślnie 16000 Hz)
        - extended: czy użyć ESTOI (domyślnie False — klasyczny STOI)

        Zwraca:
        - wartość STOI/ESTOI w zakresie 0–1
        """
        # Upewnij się, że oba sygnały mają ten sam rozmiar
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]

        # Oblicz STOI / ESTOI
        score = stoi(reference, degraded, fs, extended=extended)
        return score
            
    def save_methods_analysis_to_csv(self, filepath):

        fieldnames = [
            'NAME', 'TEST NAME', 'LEVELS', 'VAD', 'ADAPTIVE', 'ALPHA', 'BETA', 'FRAMES', 'CARRIER', 'TEST', 'LEVELS', 'SECRET' , 'SEC. SIZE [b]' , 'MSE', 'PESQ', 'STOI', 'PSNR [dB]','BER [%]' ,'CAP. [%]', 'CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]', 'DOMAIN' ,'STYLE'
        ]

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_analysis_{current_time}.csv'
        path = os.path.join(filepath, filename)

        with open(path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for analysis in self.methods_analysis:
                writer.writerow(analysis)
                        
    def printTable(self, myDict, colList=None):
        """
        Pretty print a list of dictionaries (myDict) as a dynamically sized table.
        If colList is provided, only those columns will be displayed.
        Missing keys in items are replaced with empty strings.
        """
        if not myDict:
            print("No data to display.")
            return

        if not colList:
            colList = list(myDict[0].keys())

        myList = [colList]  # Header row

        for item in myDict:
            row = [str(item.get(col, '')) for col in colList]  # Use .get() to avoid KeyError
            myList.append(row)

        colSize = [max(map(len, col)) for col in zip(*myList)]
        formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
        separator = ['-' * i for i in colSize]

        print(formatStr.format(*colList))
        print(formatStr.format(*separator))
        for row in myList[1:]:
            print(formatStr.format(*row))
        
    def calculate_peak_signal_to_noise_ratio(self, carrier, stegobject):
        """
        Oblicza PSNR (Peak Signal-to-Noise Ratio) między nośnikiem a stegobiektem.
        """

        mse = self.calculate_mean_squared_error(carrier.data_signed_int16, stegobject)
        
        max_signal = numpy.max(stegobject)

        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 *  math.log10(max_signal / math.sqrt(mse))

        return psnr
    
    def calculate_segmented_snr(self, carrier, stegobject, segment_size):
        """
        Oblicza SegmentedSNR (Segmented Signal-to-Noise Ratio) między nośnikiem a stegobiektem.
        """
        num_segments = len(carrier) // segment_size
        segmented_snr = []

        for i in range(num_segments):
            carrier_segment = carrier[i * segment_size:(i + 1) * segment_size]
            stegobject_segment = stegobject[i * segment_size:(i + 1) * segment_size]
            
            mse = self.calculate_mean_squared_error(carrier_segment, stegobject_segment)
            max_signal = numpy.max(stegobject_segment)

            if mse <= 0:
                snr = float('inf')
            else:
                snr = 20 * math.log10(max_signal / math.sqrt(mse))
            
            segmented_snr.append(snr)

        average_snr = numpy.mean(segmented_snr)
        return average_snr
    
    def analyze_sample_difference_histogram(self, carrier_waveform, stego_waveform, current_test_charts_directory, filename):
        """
        Funkcja rysuje histogram różnic między sąsiadującymi próbkami dla nośnika i stegobobiektu.
        
        Parametry:
        - carrier_waveform (numpy.array): Tablica numpy zawierająca próbki sygnału oryginalnego (nośnika).
        - stego_waveform (numpy.array): Tablica numpy zawierająca próbki sygnału podejrzanego o steganografię.
        
        Zwraca:
        - None: Funkcja rysuje dwa histogramy różnic między próbkami dla nośnika i stegobobiektu.
        
        Interpretacja wyników:
        - **Histogram różnic nośnika**: Rozkład różnic między sąsiadującymi próbkami w oryginalnym pliku.
        - **Histogram różnic stegobobiektu**: Rozkład różnic między sąsiadującymi próbkami w podejrzanym pliku.
        - **Analiza różnic**:
        - Zmiany w rozkładzie różnic mogą sugerować obecność ukrytych danych, szczególnie jeśli różnice w stegobiekcie są mniej regularne.
        - Bardzo podobne histogramy mogą wskazywać na brak ukrytych danych.
        """
        
        # Oblicz różnice między sąsiadującymi próbkami
        carrier_diff = numpy.diff(carrier_waveform)
        stego_diff = numpy.diff(stego_waveform)
        
        # Rysowanie histogramu różnic dla nośnika
        plt.figure(figsize=(12, 6))  # Ustaw odpowiedni rozmiar figury
        plt.subplot(1, 2, 1)
        plt.hist(carrier_diff, bins=256, color='blue', alpha=0.7)
        plt.title("Difference Histogram for Carrier")
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')

        # Rysowanie histogramu różnic dla stegobobiektu
        plt.subplot(1, 2, 2)
        plt.hist(stego_diff, bins=256, color='red', alpha=0.7)
        plt.title("Difference Histogram for Stegoobject")
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        
        # Zapisanie wykresów do pliku
        plt.tight_layout()
        plt.savefig(os.path.join(current_test_charts_directory, filename + '_histogram_differences.png'))
        plt.close()

    def calculate_signal_to_noise_ratio(self, method):
    # Oblicz szum jako różnicę między nośnikiem a stegoobiektem
        noise = method.carrier.raw_data - method.stegobject
        
        # Oblicz moc sygnału (nośnika)
        signal_power = numpy.mean(method.carrier.raw_data ** 2)
        
        # Oblicz moc szumu
        noise_power = numpy.mean(noise ** 2)
        
        # Oblicz SNR
        snr = 10 * numpy.log10(signal_power / noise_power)
        
        return snr
    
    def calculate_bit_error_rate(self, secret, extracted_data):
        
        secret = secret.binary_data
        recovered_bits = extracted_data
        
        errors = 0
        secret_length = 0
        for oryginal, recovered in zip(secret, recovered_bits):
            if oryginal != recovered:
                errors += 1
        secret_length += len(secret)

        self.logger.info(f"Calculated BER: {errors} errors in {secret_length} bits")

        return ((errors / secret_length) * 100)
            
    def print_results(self):

        columns_list = ['NAME', 'CARRIER', 'LEVELS', 'VAD', 'ADAPTIVE', 'ALPHA', 'BETA' ,'FRAMES', 'TEST', 'SECRET', 'SEC. SIZE [b]', 'MSE', 'PSNR [dB]', 'PESQ', 'STOI','BER [%]', 'CAP. [%]', 'CAP. [bps]','EMB TIME [s]', 'EXT TIME [s]']
                
        self.printTable(self.methods_analysis, columns_list)

    def calculate_mean_squared_error(self, carrier, stegobject):

        roznica = carrier - stegobject

        # Podnieś różnicę do kwadratu i oblicz średnią
        mse = numpy.mean(roznica ** 2)

        return mse
    
    def calculate_perceptual_evaluation_of_speech_quality(self, fs ,carrier, stegobject):
        
        pesq_score = pesq(fs, carrier, stegobject, 'wb')
        
        return pesq_score
    
    def plot_signal_with_errors(self, signal, sample_rate, frame_duration_ms, secret, extracted_secret):
        """
        Rysuje wykres sygnału z zaznaczonymi ramkami, w których wystąpiły błędy.
        
        :param signal: numpy array, sygnał audio
        :param sample_rate: int, częstotliwość próbkowania (Hz)
        :param frame_duration_ms: int, długość ramki w milisekundach
        :param error_frames: lista, indeksy ramek z błędami
        """

        error_frames = []
        idx = 0

        for bit, extracted_bit in zip(secret, extracted_secret):
            if bit != extracted_bit:
                error_frames.append(idx)
            idx += 1

        time = numpy.linspace(0, len(signal) / sample_rate, len(signal))  # Oś czasu w sekundach
        frame_length = int((frame_duration_ms / 1000) * sample_rate)  # Długość ramki w próbkach
        num_frames = len(signal) // frame_length  # Liczba pełnych ramek
        
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, label="Sygnał audio", color="black", alpha=0.6)
        
        # Oznaczanie błędnych ramek
        for frame_idx in error_frames:
            start_sample = frame_idx * frame_length
            end_sample = start_sample + frame_length
            if end_sample > len(signal):  # Zapobiega wyjściu poza zakres
                end_sample = len(signal)
            
            plt.axvspan(time[start_sample], time[end_sample], color='red', alpha=0.3, label="Błąd w ramce" if frame_idx == error_frames[0] else "")

        plt.xlabel("Czas (s)")
        plt.ylabel("Amplituda")
        plt.title("Wykres sygnału audio z zaznaczonymi błędami")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_signal_with_singular_values_and_spectrogram(self, data, method, fs, current_test_charts_directory, filename):
        time = np.linspace(0, len(data) / fs, len(data))
        data = data / np.max(np.abs(data))  # Normalizacja sygnału

        # Wartości osobliwe
        s_even = np.array(method.s_even)
        s_odd = np.array(method.s_odd)
        new_s_even = np.array(method.new_s_even)
        new_s_odd = np.array(method.new_s_odd)

        s_avg = (s_even + s_odd) / 2
        new_s_avg = (new_s_even + new_s_odd) / 2

        # Normalizacja SVD do [-1, 1]
        s_avg = 2 * (s_avg - np.min(s_avg)) / (np.max(s_avg) - np.min(s_avg)) - 1
        new_s_avg = 2 * (new_s_avg - np.min(new_s_avg)) / (np.max(new_s_avg) - np.min(new_s_avg)) - 1

        duration = len(data) / fs
        frame_times_avg = np.linspace(0, duration, len(s_avg), endpoint=False)

        # Tworzenie wykresu z 2 subplotami
        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # --- Spektrogram ---
        f, t_spec, Sxx = spectrogram(data, fs=fs, nperseg=1024, noverlap=512)
        axs[0].pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        axs[0].set_ylabel('Frequency [Hz]')
        axs[0].set_title("Spectrogram and SVD singular values – " + filename)
        axs[0].set_ylim(0, fs // 2)
        axs[0].grid(True)

        # --- Wartości osobliwe ---
        axs[1].plot(time, data, label="Normalized audio signal", alpha=0.3, color='gray')
        axs[1].plot(frame_times_avg, s_avg, 'o-', label="singular values (SVD, norm, avg)", markersize=3)
        axs[1].plot(frame_times_avg, new_s_avg, 'x-', label="new singular values (SVD, norm, avg)", markersize=3)
        axs[1].set_ylabel("Singular value (norm)")
        axs[1].set_xlabel("Time [s]")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        filename = filename.replace('.wav', '')
        plt.savefig(os.path.join(current_test_charts_directory, filename + '_spectrogram_singular.png'))
        plt.close()

    def analyze_by_multiple_groups(self, group_by_fields, filters=None):
        if not self.methods_analysis:
            return

        df = pd.DataFrame(self.methods_analysis)

        # Filtrowanie po warunkach
        if filters:
            for key, condition in filters.items():
                if key not in df.columns:
                    return
                if callable(condition):
                    df = df[df[key].apply(condition)]
                else:
                    df = df[df[key] == condition]

        if df.empty:
            return

        for field in group_by_fields:
            if field not in df.columns:
                return

        numeric_cols = ['MSE', 'PSNR [dB]', 'PESQ', 'STOI', 'BER [%]',
                        'CAP. [%]', 'CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        grouped = df.groupby(group_by_fields, as_index=False)[numeric_cols].mean()

        best_type = {
            'MSE': 'min',
            'PSNR [dB]': 'max',
            'PESQ': 'max',
            'STOI': 'max',
            'BER [%]': 'min',
            'CAP. [%]': 'max',
            'CAP. [bps]': 'max',
            'EMB TIME [s]': 'min',
            'EXT TIME [s]': 'min'
        }

        best_vals = {}
        worst_vals = {}
        for col in numeric_cols:
            vals = grouped[col].dropna()
            if vals.empty:
                best_vals[col] = worst_vals[col] = None
                continue
            if best_type[col] == 'max':
                best_vals[col] = vals.max()
                worst_vals[col] = vals.min()
            else:
                best_vals[col] = vals.min()
                worst_vals[col] = vals.max()

        formatted_rows = []
        for _, row in grouped.iterrows():
            formatted_row = []
            for col in grouped.columns:
                val = row[col]
                if col in numeric_cols:
                    if pd.isna(val):
                        formatted_row.append('')
                    elif best_vals[col] == worst_vals[col]:
                        formatted_row.append(Fore.YELLOW + Style.BRIGHT + f"{val:.3f}" + Style.RESET_ALL)
                    elif val == best_vals[col]:
                        formatted_row.append(Fore.GREEN + Style.BRIGHT + f"{val:.3f}" + Style.RESET_ALL)
                    elif val == worst_vals[col]:
                        formatted_row.append(Fore.RED + Style.BRIGHT + f"{val:.3f}" + Style.RESET_ALL)
                    else:
                        formatted_row.append(f"{val:.3f}")
                else:
                    formatted_row.append(val)
            formatted_rows.append(formatted_row)

        print(tabulate(formatted_rows, headers=grouped.columns, tablefmt='grid', showindex=False))

    def save_grouped_to_csv(self, group_by_fields, output_path, filters=None):
        if not self.methods_analysis:
            return

        df = pd.DataFrame(self.methods_analysis)

        # Filtrowanie po warunkach
        if filters:
            for key, condition in filters.items():
                if key not in df.columns:
                    return
                if callable(condition):
                    df = df[df[key].apply(condition)]
                else:
                    df = df[df[key] == condition]

        if df.empty:
            return

        for field in group_by_fields:
            if field not in df.columns:
                return

        numeric_cols = ['MSE', 'PSNR [dB]', 'PESQ', 'STOI', 'BER [%]',
                        'CAP. [%]', 'CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        grouped = df.groupby(group_by_fields, as_index=False)[numeric_cols].mean()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_analysis_group_{current_time}.csv'

        grouped.to_csv(os.path.join(output_path, filename), index=False)