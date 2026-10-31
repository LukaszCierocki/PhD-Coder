import numpy
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import math
from scipy.signal import spectrogram, resample_poly
from pesq import pesq
from pystoi.stoi import stoi
import pandas as pd
from tabulate import tabulate
from scipy.stats import gaussian_kde

from colorama import Fore, Style, init

from collections import defaultdict

from lib.Method import ensure_logger

init(autoreset=True)

# Numeric result columns (used for CSV export, grouping and plots).
NUMERIC_COLUMNS = ['MSE', 'RMSE [LSB]', 'PSNR [dB]', 'SNR [dB]', 'SEGSNR [dB]', 'PESQ', 'STOI', 'BER [%]',
                   'CAP. [%]', 'CAP. [bps]', 'EFF. CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]',
                   'BAND ENERGY [dB]']

# Peak value of 16-bit PCM used by the PSNR (full scale).
PSNR_PEAK = 32767.0

BEST_TYPE = {
    'MSE': 'min', 'RMSE [LSB]': 'min', 'PSNR [dB]': 'max', 'SNR [dB]': 'max', 'SEGSNR [dB]': 'max', 'PESQ': 'max',
    'STOI': 'max', 'BER [%]': 'min', 'CAP. [%]': 'max', 'CAP. [bps]': 'max', 'EFF. CAP. [bps]': 'max',
    'EMB TIME [s]': 'min', 'EXT TIME [s]': 'min', 'BAND ENERGY [dB]': 'max',
}

CSV_FIELDNAMES = [
    'NAME', 'TEST NAME', 'METHOD CLASS', 'LEGACY', 'LEVELS', 'VAD', 'ADAPTIVE', 'REDUCED', 'ALPHA', 'BETA',
    'FRAMES', 'FREQ_L', 'FREQ_H', 'BAND [Hz]', 'BAND ENERGY [dB]', 'FRAME [ms]', 'HOP [ms]', 'CARRIER', 'TEST', 'STEGOBJECT', 'SECRET', 'SEC. SIZE [b]',
    'EXTRACTED [b]', 'BIT ERRORS', 'MSE', 'RMSE [LSB]', 'PSNR [dB]', 'SNR [dB]', 'SEGSNR [dB]', 'PESQ', 'STOI', 'BER [%]',
    'CAP. [%]', 'CAP. [bps]', 'EFF. CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]', 'CLIPPED', 'DOMAIN', 'STYLE'
]

# Attack file prefixes written by lib/Attacks.py -> test label
ATTACK_LABELS = [
    ('filtered_low_pass_', 'FILTERED_LOW_PASS'),
    ('filtered_high_pass_', 'FILTERED_HIGH_PASS'),
    ('noised', 'NOISED'),
    ('requantized', 'REQUANTIZED'),
    ('mp3_attacked', 'MP3_ATTACKED'),
    ('jitter', 'JITTER'),
    ('packetloss_', 'PACKET_LOSS'),
    ('resample_attack', 'RESAMPLE_ATTACK'),
    ('scaled', 'SCALED'),
    ('g711_ulaw_attacked', 'G711_ULAW_ATTACKED'),
    ('g711_alaw_attacked', 'G711_ALAW_ATTACKED'),
]


def test_label_from_filename(stegobject_name, test_name=None):
    """
    Maps an (attacked) stegobject file name to a test label ('NONE' for the
    clean stegobject). If ``test_name`` (the clean stegobject name without
    extension) is given, only the prefix in front of it is inspected, so
    method or carrier names can never be mistaken for an attack.
    """
    name = os.path.splitext(os.path.basename(stegobject_name))[0]
    prefix = name
    if test_name:
        if name == test_name:
            return 'NONE'
        if name.endswith(test_name):
            prefix = name[:-len(test_name)]
    for marker, label in ATTACK_LABELS:
        if prefix.startswith(marker) or (not test_name and marker in prefix):
            parts = prefix.split('_')
            if label == 'MP3_ATTACKED' and len(parts) > 2:
                return f'{label}_{parts[2]}'
            if label == 'SCALED' and len(parts) > 1:
                return f'{label}_{parts[1]}'
            return label
    return 'NONE'

class Analysis:
    
    def __init__(self, logger=None, plots=True):

        self.methods_analysis = []
        self.logger = ensure_logger(logger)
        self.singular_values_per_carrier = []
        # Per-stegobject diagnostic charts (waveforms, spectrograms, ...). They
        # are slow for large experiments - disable with plots=False.
        self.plots = plots

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
        
        data = numpy.asarray(data, dtype=numpy.float64)
        data = data / (numpy.max(numpy.abs(data)) or 1.0)

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
            std_orig_val = np.std(delta_s_orig) or 1.0
            delta_s_orig = (delta_s_orig - np.mean(delta_s_orig)) / std_orig_val

            std_mod_val = np.std(delta_s_mod) or 1.0
            delta_s_mod = (delta_s_mod - np.mean(delta_s_mod)) / std_mod_val

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

        data = numpy.asarray(data, dtype=numpy.float64)
        data = data / (numpy.max(numpy.abs(data)) or 1.0)

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

        s_avg = 2 * (s_avg - numpy.min(s_avg)) / ((numpy.max(s_avg) - numpy.min(s_avg)) or 1.0) - 1
        new_s_avg = 2 * (new_s_avg - numpy.min(new_s_avg)) / ((numpy.max(new_s_avg) - numpy.min(new_s_avg)) or 1.0) - 1

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

    def analyze(self, method, carrier, secret, stegobject, extracted_data, test_label=None, plots=None):
        """
        Computes all quality / robustness metrics of one stegobject and stores
        them as one row of ``self.methods_analysis``.

        Every metric aligns the signals to the common length (attacks may change
        it) and works in float64 (int16 arithmetic overflowed before).
        """

        stegobject_name = stegobject['filename']
        stegobject_fs = stegobject['fs']
        stegobject = np.asarray(stegobject['content_signed_int16'])

        plots = self.plots if plots is None else plots

        self.logger.info(f"Analyzing method: {method.normal_name} on carrier: {carrier.name} with stegobject: {stegobject_name}")

        test_name = test_label or test_label_from_filename(stegobject_name)

        bit_errors, ber = self.count_bit_errors(secret, extracted_data)
        mse = self.calculate_mean_squared_error(carrier.data_signed_int16, stegobject)

        method_analysis = {
            'NAME': method.normal_name,
            'TEST NAME': method.name,
            'METHOD CLASS': method.__class__.__name__,
            'LEGACY': getattr(method, 'legacy', False),
            'LEVELS': getattr(method, 'levels', None),
            'FRAMES': method.where_to_hide,
            'VAD': method.type_of_vad,
            'ADAPTIVE': method.adaptive,
            'REDUCED': method.reduced,
            'ALPHA': method.alpha,
            'BETA': method.beta,
            'FREQ_L': method.FREQ_L,
            'FREQ_H': method.FREQ_H,
            'BAND [Hz]': (f"{method.band_frequency[0]:.1f}-{method.band_frequency[1]:.1f}"
                          if getattr(method, 'band_frequency', None) else None),
            'BAND ENERGY [dB]': (round(method.band_energy_db, 2)
                                 if getattr(method, 'band_energy_db', None) is not None else None),
            'FRAME [ms]': method.frame_length,
            'HOP [ms]': method.hop_duration_ms,
            'CARRIER': carrier.name,
            'TEST': test_name,
            'STEGOBJECT': stegobject_name,
            'SECRET': secret.name,
            'SEC. SIZE [b]': len(secret.binary_data),
            'EXTRACTED [b]': len(extracted_data) if extracted_data is not None else 0,
            'BIT ERRORS': bit_errors,
            'MSE': round(mse, 3),
            'RMSE [LSB]': round(self.root_mean_squared_error(mse), 3),
            'PSNR [dB]': round(self.calculate_peak_signal_to_noise_ratio(carrier, stegobject), 3),
            'SNR [dB]': round(self.calculate_snr(carrier.data_signed_int16, stegobject), 3),
            'SEGSNR [dB]': round(self.calculate_segmented_snr(carrier.data_signed_int16, stegobject,
                                                                int(0.02 * carrier.fs)), 3),
            'PESQ': round(self.calculate_perceptual_evaluation_of_speech_quality(carrier.fs, carrier.data_signed_int16, stegobject), 3),
            'STOI': round(self.calculate_stoi(carrier.data_signed_int16, stegobject, carrier.fs), 3),
            'BER [%]': round(ber, 3),
            'CAP. [%]': round((method.capacity / carrier.carrier_size_in_bits) * 100, 3),
            'CAP. [bps]': round(self.calculate_bitrate(method.frame_length, carrier.fs, method.hop_duration_ms, method.bits_per_frame), 3),
            'EFF. CAP. [bps]': round(len(secret.binary_data) / (len(carrier.data_signed_int16) / carrier.fs), 3),
            'EMB TIME [s]': round(method.embedding_time, 4) if method.embedding_time is not None else None,
            'EXT TIME [s]': round(method.extraction_time, 4) if method.extraction_time is not None else None,
            'CLIPPED': getattr(method, 'clipped_samples', 0),
            'DOMAIN': method.domain,
            'STYLE': method.style
        }

        if plots:
            self._safe_plot(self.plot_audio_waveform, carrier.data_signed_int16, carrier.fs,
                            method.method_directories['Carriers_Charts'], carrier.name)
            self._safe_plot(self.plot_signal_spectrogram, carrier.data_signed_int16, carrier.fs,
                            method.method_directories['Carriers_Charts'], carrier.name)
            charts = method.method_directories['Charts']
            self._safe_plot(self.plot_audio_waveform, stegobject, stegobject_fs, charts, stegobject_name)
            self._safe_plot(self.plot_signal_difference, carrier.data_signed_int16, stegobject, carrier.fs,
                            charts, carrier.name, stegobject_name)

            if len(getattr(method, 's_even', [])) > 1:
                self._safe_plot(self.plot_signal_with_singular_values, stegobject, method, stegobject_fs, charts, stegobject_name)
                self._safe_plot(self.plot_signal_with_singular_values_and_spectrogram, stegobject, method, stegobject_fs, charts, stegobject_name)
                self._safe_plot(self.plot_singular_values_and_delta, method, carrier.name, charts, stegobject_name)
                self._safe_plot(self.plot_alphas_betas, method, charts, stegobject_name)

        if getattr(method, 's_even', None):
            self.singular_values_per_carrier.append({
                'carrier_name': carrier.name,
                'method_name': method.normal_name,
                's_even': list(method.s_even),
                's_odd': list(method.s_odd),
                'new_s_even': list(method.new_s_even),
                'new_s_odd': list(method.new_s_odd)
            })

        self.methods_analysis.append(method_analysis)
        return method_analysis

    def _safe_plot(self, function, *args, **kwargs):
        """Diagnostic plots must never abort an experiment."""
        try:
            function(*args, **kwargs)
        except Exception as e:
            plt.close('all')
            self.logger.warning(f"Plot {function.__name__} failed: {e}")

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
        # Avoid log10(0) warnings by adding small epsilon
        eps = 1e-10
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + eps), shading='gouraud', cmap='magma')
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
        Extended STOI (default) or classic STOI between two signals (aligned to
        the common length). Returns NaN if it cannot be computed.
        """
        reference, degraded = self._align(reference, degraded)
        try:
            return float(stoi(reference, degraded, fs, extended=extended))
        except Exception as e:
            self.logger.warning(f"STOI could not be computed: {e}")
            return float('nan')

    def save_methods_analysis_to_csv(self, filepath):

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_analysis_{current_time}.csv'
        path = os.path.join(filepath, filename)

        fieldnames = list(CSV_FIELDNAMES)
        for row in self.methods_analysis:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for analysis in self.methods_analysis:
                writer.writerow(analysis)

        return path

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
        
    @staticmethod
    def _align(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = min(len(a), len(b))
        return a[:n], b[:n]

    def calculate_peak_signal_to_noise_ratio(self, carrier, stegobject, peak=PSNR_PEAK):
        """
        PSNR = 10 log10(A^2 / MSE) with the full-scale peak of 16-bit PCM,
        A = 32767 (the definition in the PC paper; independent of the
        loudness of the carrier). The old version used max(stegobject) and an
        overflowing int16 MSE.
        """
        reference = carrier.data_signed_int16 if hasattr(carrier, 'data_signed_int16') else carrier
        mse = self.calculate_mean_squared_error(reference, stegobject)
        if mse == 0:
            return float('inf')
        return 10 * math.log10(peak ** 2 / mse)

    def calculate_snr(self, carrier, stegobject):
        """SNR [dB] = 10 log10( sum(x^2) / sum((x - y)^2) )."""
        x, y = self._align(carrier, stegobject)
        noise = np.sum((x - y) ** 2)
        signal = np.sum(x ** 2)
        if noise == 0:
            return float('inf')
        if signal == 0:
            return float('-inf')
        return 10 * math.log10(signal / noise)

    def calculate_segmented_snr(self, carrier, stegobject, segment_size, min_db=-10.0, max_db=35.0):
        """
        Segmental SNR [dB]: mean over segments of 10 log10(sum x^2 / sum e^2),
        each segment clamped to [min_db, max_db] (the usual convention for
        speech). Silent segments (sum x^2 == 0) are skipped.
        """
        x, y = self._align(carrier, stegobject)
        segment_size = max(1, int(segment_size))
        num_segments = len(x) // segment_size
        values = []

        for i in range(num_segments):
            xs = x[i * segment_size:(i + 1) * segment_size]
            es = xs - y[i * segment_size:(i + 1) * segment_size]
            signal = np.sum(xs ** 2)
            noise = np.sum(es ** 2)
            if signal == 0:
                continue
            snr = max_db if noise == 0 else 10 * math.log10(signal / noise)
            values.append(min(max(snr, min_db), max_db))

        return float(np.mean(values)) if values else float('nan')

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
        """SNR between the carrier of ``method`` and ``method.stegobject``."""
        return self.calculate_snr(method.carrier.data_signed_int16, method.stegobject)

    @staticmethod
    def count_bit_errors(secret, extracted_data):
        """
        Returns ``(errors, ber_percent)``. Missing bits (extracted shorter than
        the secret, e.g. after an attack that shortens the signal) and extra
        bits are counted as errors - before they were silently ignored.
        """
        reference = [int(b) for b in (secret.binary_data if hasattr(secret, 'binary_data') else secret)]
        recovered = [int(b) for b in extracted_data] if extracted_data is not None else []

        if not reference:
            raise ValueError("Cannot compute BER for an empty secret.")

        errors = sum(o != r for o, r in zip(reference, recovered))
        errors += abs(len(reference) - len(recovered))
        return errors, errors / len(reference) * 100

    def calculate_bit_error_rate(self, secret, extracted_data):

        errors, ber = self.count_bit_errors(secret, extracted_data)

        self.logger.info(f"Calculated BER: {errors} errors in {len(secret.binary_data)} bits")

        return ber

    def print_results(self):

        columns_list = ['NAME', 'CARRIER', 'LEVELS', 'VAD', 'ADAPTIVE', 'REDUCED', 'ALPHA', 'BETA', 'FRAMES', 'TEST',
                        'SECRET', 'SEC. SIZE [b]', 'MSE', 'RMSE [LSB]', 'PSNR [dB]', 'SNR [dB]', 'PESQ', 'STOI', 'BER [%]',
                        'CAP. [%]', 'CAP. [bps]', 'EFF. CAP. [bps]', 'EMB TIME [s]', 'EXT TIME [s]']

        self.printTable(self.methods_analysis, columns_list)

    def calculate_mean_squared_error(self, carrier, stegobject):
        """
        MSE in float64 on the common length. (The old version subtracted and
        squared int16 arrays, which overflowed for |difference| > 181 and gave
        MSE values that were far too small.)
        """
        x, y = self._align(carrier, stegobject)
        if len(x) == 0:
            return float('nan')
        return float(np.mean((x - y) ** 2))

    @staticmethod
    def root_mean_squared_error(mse):
        """
        RMSE = sqrt(MSE): the typical size of the sample error in LSB, on the
        same scale as the int16 samples (MSE is in LSB^2: MSE 12 000 is an error
        of about +/-110 out of +/-32 768). Like MSE it grows with the recording
        level - SNR / SegSNR do not.
        """
        mse = float(mse)
        return float(np.sqrt(mse)) if np.isfinite(mse) and mse >= 0 else float('nan')

    def calculate_perceptual_evaluation_of_speech_quality(self, fs ,carrier, stegobject):
        try:
            # Ensure arrays
            ref = numpy.asarray(carrier)
            deg = numpy.asarray(stegobject)

            # Choose target sampling frequency for PESQ (only 8000 or 16000 allowed)
            if fs == 8000:
                target_fs = 8000
                mode = 'nb'
            elif fs == 16000:
                target_fs = 16000
                mode = 'wb'
            else:
                # Choose nearest target: prefer 16000 for >= 12000, else 8000
                target_fs = 8000 if fs < 12000 else 16000
                mode = 'nb' if target_fs == 8000 else 'wb'

                # Resample signals to target_fs using resample_poly
                from math import gcd
                g = gcd(int(fs), int(target_fs))
                up = int(target_fs // g)
                down = int(fs // g)

                # Convert to float for resampling to avoid integer artifacts
                ref = resample_poly(ref.astype(numpy.float64), up, down)
                deg = resample_poly(deg.astype(numpy.float64), up, down)

                # Convert back to int16 range
                ref = numpy.clip(numpy.round(ref), -32768, 32767).astype(numpy.int16)
                deg = numpy.clip(numpy.round(deg), -32768, 32767).astype(numpy.int16)

            # Truncate to the same length
            n = min(len(ref), len(deg))
            if n == 0:
                return float('nan')
            ref = ref[:n]
            deg = deg[:n]

            pesq_score = pesq(target_fs, ref, deg, mode)
            return pesq_score
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error calculating PESQ: {e}")
            return float('nan')
    
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
        data = np.asarray(data, dtype=np.float64)
        data = data / (np.max(np.abs(data)) or 1.0)  # Normalizacja sygnału

        # Wartości osobliwe
        s_even = np.array(method.s_even)
        s_odd = np.array(method.s_odd)
        new_s_even = np.array(method.new_s_even)
        new_s_odd = np.array(method.new_s_odd)

        s_avg = (s_even + s_odd) / 2
        new_s_avg = (new_s_even + new_s_odd) / 2

        # Normalizacja SVD do [-1, 1]
        s_avg = 2 * (s_avg - np.min(s_avg)) / ((np.max(s_avg) - np.min(s_avg)) or 1.0) - 1
        new_s_avg = 2 * (new_s_avg - np.min(new_s_avg)) / ((np.max(new_s_avg) - np.min(new_s_avg)) or 1.0) - 1

        duration = len(data) / fs
        frame_times_avg = np.linspace(0, duration, len(s_avg), endpoint=False)

        # Tworzenie wykresu z 2 subplotami
        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # --- Spektrogram ---
        f, t_spec, Sxx = spectrogram(data, fs=fs, nperseg=1024, noverlap=512)
        eps = 1e-10
        axs[0].pcolormesh(t_spec, f, 10 * np.log10(Sxx + eps), shading='gouraud', cmap='magma')
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

        numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

        group_by_fields = [f for f in group_by_fields if f in df.columns]
        grouped = df.groupby(group_by_fields, as_index=False, dropna=False)[numeric_cols].mean()

        best_type = BEST_TYPE

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

        numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

        group_by_fields = [f for f in group_by_fields if f in df.columns]
        grouped = df.groupby(group_by_fields, as_index=False, dropna=False)[numeric_cols].mean()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_analysis_group_{current_time}.csv'

        path = os.path.join(output_path, filename)
        grouped.to_csv(path, index=False)
        return path