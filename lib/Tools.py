import json
import logging
import os
import random
import shutil
import string
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from BitVector import BitVector

from lib.Carrier import Carrier, read_audio_as_int16
from lib.Logger import Logger
from lib.Secret import Secret

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}


def default_results_root():
    """``$PHD_CODER_RESULTS`` if set, otherwise ``~/Results``."""
    return os.environ.get('PHD_CODER_RESULTS') or os.path.join(os.path.expanduser('~'), 'Results')


class Tools:

    def __init__(self, log_level=logging.DEBUG, results_root=None, archive=True, carriers_dir=None,
                 console=False):
        self.current_working_directory = os.getcwd()
        self.results_root = results_root or default_results_root()
        self.carriers_dir = carriers_dir or os.path.join(os.getcwd(), 'Carriers')

        # Fallback logger until the file logger exists (used by the helpers below).
        self.logger = logging.getLogger('phd_coder.tools')

        self.create_results_directory()
        if archive:
            self.archive_previous_tests(results_root=self.results_root, archive_folder='Archive')
        self.current_test_directory, self.test_id = self.create_current_test_directory()
        self.logger = Logger(log_dir=self.current_test_directory,
                             log_name='test_log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log',
                             log_level=log_level, console=console).get_logger()

    def get_logger(self):
        return self.logger

    def get_current_test_directory(self):
        return self.current_test_directory

    def get_test_id(self):
        return self.test_id

    def get_results_root(self):
        return self.results_root

    def generate_random_id(self, length=4):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    def create_current_test_directory(self):
        test_id = self.generate_random_id(length=7)
        current_test_directory = os.path.join(self.results_root,
                                              datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f'_{test_id}')
        os.makedirs(current_test_directory)
        return current_test_directory, test_id

    def create_hankel_matrix_from_array(self, arr, n):
        """
        Creating n-row Hankel matrix form 1-D NumPy array.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr musi być tablicą NumPy.")

        if arr.ndim != 1:
            raise ValueError("arr musi być jednowymiarową tablicą.")

        if n <= 0 or n > len(arr):
            raise ValueError("n musi być dodatnie i nie większe niż długość arr.")

        m = len(arr) - n + 1
        return np.lib.stride_tricks.sliding_window_view(arr.astype(np.float64), m).copy()

    def create_array_from_hankel_matrix(self, hankel):
        """
        Retrive 1D array from Hankel Matrix.
        """
        if not isinstance(hankel, np.ndarray):
            raise ValueError("Hankel must be a ndarray.")

        if hankel.ndim != 2:
            raise ValueError("Hankel must be 2D array.")

        first_column = hankel[:, 0]
        last_row = hankel[-1, 1:]

        return np.concatenate((first_column, last_row))

    # ------------------------------------------------------------------ #
    # Audio I/O
    # ------------------------------------------------------------------ #

    def load_audio_file(self, filepath):
        """
        Loads an audio file as exact int16 mono samples.

        Returns a dict with 'content_signed_int16', 'content_unsigned_int16',
        'fs', 'filepath', 'filename' and 'loaded_with'.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Plik {filepath} nie istnieje.")

        try:
            try:
                data_int16, fs = read_audio_as_int16(filepath)
                loaded_with = 'soundfile'
            except Exception:
                # Fallback - pydub (mp3 and other formats via ffmpeg)
                from pydub import AudioSegment
                audio = AudioSegment.from_file(filepath).set_sample_width(2)
                fs = audio.frame_rate
                data = np.array(audio.get_array_of_samples(), dtype=np.int16)
                if audio.channels > 1:
                    mono = np.rint(data.reshape((-1, audio.channels)).astype(np.float64).mean(axis=1))
                    data = np.clip(mono, -32768, 32767).astype(np.int16)
                data_int16 = data
                loaded_with = 'pydub'
        except Exception:
            self.logger.exception(f"Error loading audio file {filepath}")
            raise

        data_unsigned_int16 = np.asarray(data_int16.astype(np.int32) + 2**15, dtype=np.uint16)

        return {
            'content_signed_int16': data_int16,
            'content_unsigned_int16': data_unsigned_int16,
            'fs': int(fs),
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'loaded_with': loaded_with
        }

    def save_audio_file(self, path, audio_object):
        """Saves int16 (or uint16 offset-binary) samples as a 16-bit PCM WAV file."""
        try:
            filename = audio_object['name']
            data = np.asarray(audio_object['content'])
            fs = int(audio_object['fs'])
            if data.dtype == np.uint16:
                data = (data.astype(np.int32) - 32768).astype(np.int16)
            elif data.dtype != np.int16:
                raise TypeError(f"Nieobsługiwany typ danych: {data.dtype}. Oczekiwano int16 lub uint16.")
            if data.ndim != 1:
                raise ValueError("Only mono signals can be saved.")
            os.makedirs(path, exist_ok=True)
            wavfile.write(os.path.join(path, filename), fs, data)
        except Exception:
            self.logger.exception(f"Error saving audio file {audio_object.get('name', 'unknown')}")
            raise

    # ------------------------------------------------------------------ #
    # Carriers / secrets
    # ------------------------------------------------------------------ #

    def list_files_in_carrier_directory(self):
        if not os.path.isdir(self.carriers_dir):
            self.logger.error(f"Carrier directory {self.carriers_dir} does not exist.")
            return []
        files = []
        for f in sorted(os.listdir(self.carriers_dir)):
            path = os.path.join(self.carriers_dir, f)
            if os.path.isfile(path) and os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                files.append({'path': path, 'filename': os.path.splitext(f)[0]})
        return files

    def list_files_in_secrets_directory(self):
        try:
            current_directory = os.path.join(os.getcwd(), 'image_secrets')
            return [{'path': os.path.join(current_directory, f), 'filename': os.path.splitext(f)[0]}
                    for f in sorted(os.listdir(current_directory))
                    if os.path.isfile(os.path.join(current_directory, f))]
        except Exception:
            self.logger.exception("Error listing files in image_secrets/")
            return []

    def load_carriers(self, files=None, limit=None):
        """
        Loads carriers from the Carriers/ directory (sorted by name, so runs are
        reproducible). ``files`` - optional list of carrier names to load,
        ``limit`` - optional maximum number of carriers.
        Unreadable files are skipped and logged.
        """
        listed = self.list_files_in_carrier_directory()
        if files:
            wanted = set(files)
            listed = [f for f in listed if f['filename'] in wanted]
        if limit:
            listed = listed[:int(limit)]

        carriers = []
        for file in listed:
            try:
                carriers.append(Carrier({'name': file['filename'], 'path': file['path']}))
                self.logger.info(f"Loaded carrier: {file['filename']}")
            except Exception:
                self.logger.exception(f"Skipping unreadable carrier {file['path']}")
        if not carriers:
            self.logger.warning(f"No carriers loaded from {self.carriers_dir}.")
        return carriers

    def load_carrier(self, filename):
        for file in self.list_files_in_carrier_directory():
            if file['filename'] == filename:
                audio_obj = self.load_audio_file(file['path'])
                return Carrier({'name': file['filename'], 'path': file['path'], 'fs': audio_obj['fs'],
                                'data_signed_int16': audio_obj['content_signed_int16']})
        raise FileNotFoundError(f"Carrier file {filename} not found in {self.carriers_dir}.")

    def generate_bitvector_from_seed(self, seed, num_bits):
        rng = random.Random(seed)
        bits = []
        while len(bits) < num_bits:
            bits.extend(list(BitVector(intVal=rng.getrandbits(32), size=32)))
        return BitVector(bitlist=bits[:num_bits])

    def load_secrets(self, arguments):

        secrets = []

        if arguments['type'] == 'TXT':
            secrets.append(Secret({'name': self.generate_random_id(), 'path': arguments['path'],
                                   'type': arguments['type'], 'size': arguments['size']}))
        elif arguments['type'] == 'IMAGE':
            for file in self.list_files_in_secrets_directory():
                secrets.append(Secret({'name': file['filename'], 'path': file['path'], 'type': 'image'}))
        elif arguments['type'] == 'MAX_CAP':
            secrets.append(Secret({'name': None, 'type': 'MAX_CAP', 'path': None, 'size': arguments.get('size')}))

        return secrets

    # ------------------------------------------------------------------ #
    # Result directories
    # ------------------------------------------------------------------ #

    def prepare_method_directory(self, base_dir: str, method_name: str, only_embed: bool = False) -> dict:
        try:
            method_dir = os.path.join(base_dir, method_name)
            subdirs = {
                'Stegobjects': os.path.join(method_dir, 'Stegobjects'),
                'Extracted': os.path.join(method_dir, 'Extracted'),
                'Charts': os.path.join(method_dir, 'Charts'),
                'Secrets': os.path.join(method_dir, 'Secrets'),
                'Keys': os.path.join(method_dir, 'Keys'),
                'Carriers_Charts': os.path.join(base_dir, 'Carriers', 'Charts'),
                'Base': method_dir
            }
            for name, path in subdirs.items():
                if not only_embed or name in ('Stegobjects', 'Keys', 'Base'):
                    os.makedirs(path, exist_ok=True)
            return subdirs
        except Exception:
            self.logger.exception(f"Error preparing method directory for {method_name}")
            raise

    def archive_previous_tests(self, results_root='Results', archive_folder='Archive'):
        try:
            results_path = os.path.abspath(results_root)
            archive_path = os.path.join(results_path, archive_folder)
            os.makedirs(archive_path, exist_ok=True)
            for name in os.listdir(results_path):
                path = os.path.join(results_path, name)
                if os.path.isdir(path) and name != archive_folder:
                    dest_path = os.path.join(archive_path, name)
                    if not os.path.exists(dest_path):
                        shutil.move(path, dest_path)
        except Exception:
            self.logger.exception("Error archiving previous tests")
            raise

    def save_extracted_secret(self, extracted_data, method_directories, filename):
        """Saves the extracted bits as text (non-printable bytes are escaped)."""
        if extracted_data is None or len(extracted_data) == 0:
            self.logger.warning("No extracted data to save.")
            return

        how_much = len(extracted_data) // 8
        ext_sec = extracted_data[0:how_much * 8]
        raw = bytes(ext_sec[i * 8:(i + 1) * 8].intValue() for i in range(how_much))
        text = raw.decode('ascii', errors='backslashreplace')
        with open(os.path.join(method_directories['Extracted'], filename + '.txt'), 'w', encoding='utf-8') as file:
            file.write(text)

    def save_extraction_key(self, key, method_directories, filename):
        """Saves the extraction key (side information for the receiver) as JSON."""
        path = os.path.join(method_directories['Keys'], filename + '.key.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(key, f, indent=2)
        return path

    @staticmethod
    def load_extraction_key(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_results_directory(self):
        os.makedirs(self.results_root, exist_ok=True)

    def find_files_with_string_in_name(self, directory, search_string, recursive=True):
        matching_files = []
        filenames = []

        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if search_string in file and os.path.splitext(file)[1].lower() in AUDIO_EXTENSIONS:
                    matching_files.append(os.path.join(root, file))
                    filenames.append(os.path.splitext(file)[0])

            if not recursive:
                break

        return matching_files, filenames
