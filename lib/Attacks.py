"""
Attacks applied to stegobjects (robustness tests).

Every attack reads ``<input_path>/<file_name>`` and writes a new 16-bit PCM
WAV file named ``<prefix><file_name>`` next to it (the prefixes are mapped to
test labels by ``lib.Analysis.test_label_from_filename``).

All random attacks use ``self.rng`` (``numpy.random.Generator``), so a run
with a fixed ``seed`` is reproducible.
"""

import os
import shutil
import subprocess
import tempfile
import warnings
from math import gcd

import numpy
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly

from lib.Carrier import read_audio_as_int16
from lib.Method import ensure_logger, to_int16

try:  # audioop was removed from the standard library in Python 3.13 (pip install audioop-lts)
    with warnings.catch_warnings():
        # Python 3.11-3.12: stdlib audioop emits a DeprecationWarning on import
        warnings.simplefilter('ignore', DeprecationWarning)
        import audioop
except ImportError:  # pragma: no cover - depends on the environment
    audioop = None


# Attack switches understood by perform_attacks (all default to False).
ATTACK_NAMES = ('NOISE', 'HIGHPASS', 'LOWPASS', 'REQUANTIZE', 'MP3', 'JITTER', 'RESAMPLE',
                'CLIPPED', 'G711', 'PACKET_LOSS')


def no_attacks():
    return {name: False for name in ATTACK_NAMES}


class Attacks:
    def __init__(self, logger=None, seed=None):

        self.logger = ensure_logger(logger)
        self.seed = seed
        self.rng = numpy.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_float(path):
        """Float signal in [-1, 1) (soundfile convention, int16 / 32768)."""
        data, samplerate = sf.read(path, dtype='float64')
        return data, samplerate

    @staticmethod
    def _write(path, data, samplerate):
        """Writes 16-bit PCM. Float input in [-1, 1) is clipped by libsndfile."""
        data = numpy.asarray(data)
        if numpy.issubdtype(data.dtype, numpy.floating):
            data = numpy.clip(data, -1.0, 1.0 - 1.0 / 32768)
        sf.write(path, data, samplerate, subtype='PCM_16')
        return path

    # ------------------------------------------------------------------ #
    # attacks
    # ------------------------------------------------------------------ #

    def add_gaussian_noise(self, input_path, file_name, noise_level=0.005):
        """Gaussian noise with std = noise_level * max|x| (int16 domain)."""
        data, samplerate = read_audio_as_int16(os.path.join(input_path, file_name))
        signal = data.astype(numpy.float64)
        noise = self.rng.normal(0, noise_level * numpy.max(numpy.abs(signal)), len(signal))
        noisy, _ = to_int16(signal + noise)
        return self._write(os.path.join(input_path, 'noised_' + file_name), noisy, samplerate)

    def add_gaussian_noise_with_target_snr(self, input_path, file_name, target_snr_db=20):
        """
        Adds white Gaussian noise so that SNR(stegobject, noisy) = target_snr_db.

        Works on the int16 samples. (The previous version cast the float
        [-1, 1] signal directly to int16, i.e. it wrote an almost silent file.)
        """
        data, samplerate = read_audio_as_int16(os.path.join(input_path, file_name))
        signal = data.astype(numpy.float64)

        signal_power = numpy.mean(signal ** 2)
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        noise = self.rng.normal(0, numpy.sqrt(noise_power), size=signal.shape)

        self.logger.debug(f"Noise attack: signal power {signal_power:.2f}, noise std {numpy.sqrt(noise_power):.2f}")

        noisy, clipped = to_int16(signal + noise)
        if clipped:
            self.logger.warning(f"Noise attack clipped {clipped} samples of {file_name}.")

        output_file = os.path.join(input_path, f"noised_{target_snr_db:.0f}dB_" + file_name)
        return self._write(output_file, noisy, samplerate)

    def g711_attack(self, input_path, file_name, law='mu', target_fs=8000):
        """
        Symuluje przejście przez kodek G.711 (resampling do target_fs,
        kodowanie mu-law / A-law, dekodowanie, resampling z powrotem).
        """
        if audioop is None:
            raise ImportError("G.711 attack needs the 'audioop' module (pip install audioop-lts on Python >= 3.13).")

        data_int, orig_fs = read_audio_as_int16(os.path.join(input_path, file_name))

        def _resample(arr, src_fs, dst_fs):
            if src_fs == dst_fs:
                return arr.astype(numpy.int16, copy=False)
            g = gcd(int(src_fs), int(dst_fs))
            resampled = resample_poly(arr.astype(numpy.float64), int(dst_fs // g), int(src_fs // g))
            return to_int16(resampled)[0]

        resampled = _resample(data_int, orig_fs, target_fs)

        if law in ('mu', 'u', 'ulaw'):
            recovered_bytes = audioop.ulaw2lin(audioop.lin2ulaw(resampled.tobytes(), 2), 2)
            suffix = 'ulaw'
        else:
            recovered_bytes = audioop.alaw2lin(audioop.lin2alaw(resampled.tobytes(), 2), 2)
            suffix = 'alaw'

        recovered = numpy.frombuffer(recovered_bytes, dtype=numpy.int16)
        final = _resample(recovered, target_fs, orig_fs)

        # resample_poly may change the length by a sample - keep the original length
        if len(final) >= len(data_int):
            final = final[:len(data_int)]
        else:
            final = numpy.concatenate([final, numpy.zeros(len(data_int) - len(final), dtype=numpy.int16)])

        output_path = os.path.join(input_path, f"g711_{suffix}_attacked_" + file_name)
        self._write(output_path, final, orig_fs)
        self.logger.info(f"G.711 ({suffix}) attack saved: {os.path.basename(output_path)}")
        return output_path

    def low_pass_filter(self, input_path, file_name, cutoff=3000):
        data, samplerate = self._read_float(os.path.join(input_path, file_name))
        b, a = butter(5, cutoff / (0.5 * samplerate), btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return self._write(os.path.join(input_path, 'filtered_low_pass_' + file_name), filtered_data, samplerate)

    def requantize_audio(self, input_path, file_name, bit_depth=8):
        """
        Rekwantyzuje sygnał audio do mniejszej liczby bitów i przywraca do
        oryginalnej rozdzielczości (zapis jako 16-bit PCM).
        """
        data, samplerate = self._read_float(os.path.join(input_path, file_name))

        if data.ndim > 1:
            data = data.mean(axis=1)

        data_clipped = numpy.clip(data, -1.0, 1.0)

        levels = 2 ** bit_depth
        quantized = numpy.round((data_clipped + 1) * (levels / 2 - 1))
        requantized = (quantized / (levels / 2 - 1)) - 1

        output_file = f"requantized_{bit_depth}bit_" + file_name
        return self._write(os.path.join(input_path, output_file), requantized, samplerate)

    def high_pass_filter(self, input_path, file_name, cutoff=3000):
        data, samplerate = self._read_float(os.path.join(input_path, file_name))
        b, a = butter(5, cutoff / (0.5 * samplerate), btype='high', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return self._write(os.path.join(input_path, 'filtered_high_pass_' + file_name), filtered_data, samplerate)

    def mp3_compression_attack(self, input_path, file_name, bitrate):
        """
        Atak przez stratną kompresję MP3 i konwersję z powrotem do WAV
        (mono, oryginalne fs, 16-bit PCM). Wymaga ffmpeg w PATH.

        Note: MP3 encoders add a delay/padding. ffmpeg removes it when decoding
        (gapless info), so the output is sample-aligned; with other decoders a
        misalignment - and a high BER, the methods have no resynchronisation -
        is possible.
        """
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise FileNotFoundError("MP3 attack needs ffmpeg in PATH.")

        input_full = os.path.join(input_path, file_name)
        samplerate = sf.info(input_full).samplerate
        recovered_path = os.path.join(input_path, "mp3_attacked_" + bitrate + "_" + file_name)

        with tempfile.TemporaryDirectory() as tmp:
            mp3_path = os.path.join(tmp, "compressed.mp3")
            for command in (
                    [ffmpeg, "-y", "-loglevel", "error", "-i", input_full, "-b:a", bitrate, mp3_path],
                    [ffmpeg, "-y", "-loglevel", "error", "-i", mp3_path, "-ac", "1", "-ar", str(samplerate),
                     "-c:a", "pcm_s16le", recovered_path]):
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed ({' '.join(command[:4])} ...): {result.stderr.strip()}")

        return recovered_path

    def jitter_attack(self, input_path, file_name, jitter_strength=1):
        """
        Atak przez zaburzenie kolejności próbek w niewielkim zakresie.
        """
        data, samplerate = self._read_float(os.path.join(input_path, file_name))
        indices = numpy.arange(len(data))
        jitter = self.rng.integers(-jitter_strength, jitter_strength + 1, size=len(data))
        jittered_data = data[numpy.clip(indices + jitter, 0, len(data) - 1)]

        output_file = f"jitter_{jitter_strength}_" + file_name
        return self._write(os.path.join(input_path, output_file), jittered_data, samplerate)

    def packet_loss_attack(self, input_path, file_name, frame_ms=20, loss_rate=0.1, replace_with='zero'):
        """
        Symuluje utratę pakietów: losowe ramki frame_ms są zastępowane ciszą
        ('zero') lub szumem ('noise').
        """
        data, fs = self._read_float(os.path.join(input_path, file_name))

        was_mono = data.ndim == 1
        if was_mono:
            data = data[:, numpy.newaxis]

        frame_len = max(1, int((frame_ms / 1000.0) * fs))
        n_frames = int(numpy.ceil(len(data) / frame_len))
        out = data.copy()

        for i in range(n_frames):
            if self.rng.random() < loss_rate:
                start = i * frame_len
                end = min(start + frame_len, len(data))
                if replace_with == 'zero':
                    out[start:end, :] = 0
                else:
                    for ch in range(out.shape[1]):
                        sigma = numpy.std(data[:, ch]) or 1.0
                        out[start:end, ch] = self.rng.normal(0, sigma * 0.1, size=(end - start))

        final = out[:, 0] if was_mono else out

        output_file = os.path.join(input_path, f"packetloss_{int(loss_rate * 100)}_{file_name}")
        self._write(output_file, final, fs)
        self.logger.info(f"Packet loss attack saved: {output_file}")
        return output_file

    def resample_attack(self, input_path, file_name, factor=2):
        """
        Downsampling (co ``factor``-ta próbka) i powrót z interpolacją liniową.
        """
        data, samplerate = self._read_float(os.path.join(input_path, file_name))

        if data.ndim == 1:
            data = data[:, numpy.newaxis]

        upsampled = []
        for ch in range(data.shape[1]):
            signal = data[:, ch]
            downsampled = signal[::factor]
            x_old = numpy.arange(0, len(downsampled)) * factor
            upsampled.append(numpy.interp(numpy.arange(0, len(signal)), x_old, downsampled))

        upsampled = numpy.stack(upsampled, axis=1) if len(upsampled) > 1 else upsampled[0]

        output_file = f"resample_attack_x{factor}_" + file_name
        return self._write(os.path.join(input_path, output_file), upsampled, samplerate)

    def clipping_attack(self, input_path, file_name, threshold):
        """
        Skalowanie amplitudy (np. 0.7 = 70%, 1.3 = +30% z obcięciem do zakresu).
        """
        data, samplerate = self._read_float(os.path.join(input_path, file_name))
        scaled = numpy.clip(data * threshold, -1.0, 1.0)

        output_file = f"scaled_{int(round(threshold * 100))}_" + file_name
        return self._write(os.path.join(input_path, output_file), scaled, samplerate)

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #

    def perform_attacks(self, method_stegobjects_directory, test_name, test_to_perform):
        """
        Runs the enabled attacks on ``<directory>/<test_name>.wav``.

        One failing attack (e.g. missing ffmpeg) is logged and does not stop
        the others. Returns the list of written files.
        """
        file_name = test_name + '.wav'
        directory = method_stegobjects_directory
        enabled = {name: bool(test_to_perform.get(name, False)) for name in ATTACK_NAMES}
        unknown = set(test_to_perform) - set(ATTACK_NAMES)
        if unknown:
            self.logger.warning(f"Unknown attack switches ignored: {sorted(unknown)}")

        jobs = []
        if enabled['NOISE']:
            jobs.append(('NOISE', lambda: self.add_gaussian_noise_with_target_snr(directory, file_name, target_snr_db=30)))
        if enabled['HIGHPASS']:
            jobs.append(('HIGHPASS', lambda: self.high_pass_filter(directory, file_name, cutoff=500)))
        if enabled['LOWPASS']:
            jobs.append(('LOWPASS', lambda: self.low_pass_filter(directory, file_name, cutoff=6500)))
        if enabled['REQUANTIZE']:
            jobs.append(('REQUANTIZE', lambda: self.requantize_audio(directory, file_name, bit_depth=10)))
        if enabled['MP3']:
            for bitrate in ["64k", "128k", "192k"]:
                jobs.append((f'MP3 {bitrate}', lambda b=bitrate: self.mp3_compression_attack(directory, file_name, bitrate=b)))
        if enabled['JITTER']:
            jobs.append(('JITTER', lambda: self.jitter_attack(directory, file_name, 1)))
        if enabled['RESAMPLE']:
            jobs.append(('RESAMPLE', lambda: self.resample_attack(directory, file_name, 2)))
        if enabled['CLIPPED']:
            for level in [0.7, 1.3]:
                jobs.append((f'CLIPPED {level}', lambda lv=level: self.clipping_attack(directory, file_name, threshold=lv)))
        if enabled['PACKET_LOSS']:
            jobs.append(('PACKET_LOSS', lambda: self.packet_loss_attack(directory, file_name, frame_ms=20,
                                                                        loss_rate=0.1, replace_with='zero')))
        if enabled['G711']:
            jobs.append(('G711 mu', lambda: self.g711_attack(directory, file_name, law='mu', target_fs=8000)))
            jobs.append(('G711 A', lambda: self.g711_attack(directory, file_name, law='a', target_fs=8000)))

        written = []
        for label, job in jobs:
            try:
                written.append(job())
            except Exception as e:
                self.logger.error(f"Attack {label} failed for {test_name}: {e}")
        return written

    # Backwards compatible name (typo kept for old scripts).
    perfome_attacks = perform_attacks
