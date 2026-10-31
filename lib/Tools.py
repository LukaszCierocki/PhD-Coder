import numpy as np

from scipy.io import wavfile
from scipy.signal.windows import kaiser

import soundfile as sf
from pydub import AudioSegment

from lib.Carrier import Carrier
from lib.Secret import Secret

from BitVector import BitVector

import os

import random
import string
import shutil

from datetime import datetime
from lib.Logger import Logger

import logging


class Tools:

    def __init__(self, log_level=logging.DEBUG):
        self.current_working_directory = os.getcwd()
        self.results_root = os.path.join(os.path.expanduser('~'), 'Results')

        self.create_results_directory()
        self.archive_previous_tests(results_root=self.results_root, archive_folder='Archive')
        self.current_test_directory, self.test_id = self.create_current_test_directory()
        self.logger = Logger(log_dir = self.current_test_directory, log_name='test_log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log', log_level=log_level).get_logger()

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
        current_test_directory = os.path.join(self.results_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f'_{test_id}')
        os.makedirs(current_test_directory)
        return current_test_directory, test_id

    def create_hankel_matrix_from_array(self, arr, n):
        """
        Creating n-row Hankel matrix form 1-D NumPy array.
        
        Parametry:
        arr (numpy.ndarray): 1-D array.
        n (int): number of rows in Hankel Matrix.

        Zwraca:
        numpy.ndarray: Hankel matrix.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr musi być tablicą NumPy.")
        
        if arr.ndim != 1:
            raise ValueError("arr musi być jednowymiarową tablicą.")

        if n <= 0 or n > len(arr):
            raise ValueError("n musi być dodatnie i nie większe niż długość arr.")

        m = len(arr) - n + 1  # Liczba kolumn
        if m <= 0:
            raise ValueError("Nie można utworzyć macierzy Hankela z podanymi parametrami.")

        hankel = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                hankel[i, j] = arr[i + j]

        return hankel
    
    def create_array_from_hankel_matrix(self, hankel):
        """
        Retrive 1D array from Hankel Matrix.
        
        Parameters:
        hankel (numpy.ndarray): Hankel Matrix.

        Returns:
        numpy.ndarray: 1D array.
        """
        if not isinstance(hankel, np.ndarray):
            raise ValueError("Hankel must be a ndarray.")
        
        if hankel.ndim != 2:
            raise ValueError("Hankel must be 2D array.")
        
        first_column = hankel[:, 0]
        last_row = hankel[-1, 1:]
        
        original_array = np.concatenate((first_column, last_row))  # Połączenie w pierwotną tablicę
        return original_array
    
    def load_audio_file(self, filepath):
        try:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Plik {filepath} nie istnieje.")

            ext = os.path.splitext(filepath)[1].lower()

            # Najpierw spróbuj wczytać przez soundfile (obsługuje WAV, FLAC i inne)
            try:
                data, fs = sf.read(filepath, always_2d=False)
                loaded_with = 'soundfile'
            except Exception:
                # Fallback - użyj pydub (przydatne dla mp3 i innych formatów przez ffmpeg)
                audio = AudioSegment.from_file(filepath)
                fs = audio.frame_rate
                samples = audio.get_array_of_samples()
                data = np.array(samples)
                if audio.channels > 1:
                    data = data.reshape((-1, audio.channels))
                loaded_with = 'pydub'

            # Jeśli wielokanałowe - zredukuj do mono przez średnią (można zmienić jeśli potrzeba)
            if isinstance(data, np.ndarray) and getattr(data, 'ndim', 1) == 2:
                data = data.mean(axis=1)

            # Jeśli dane są float w [-1,1] - przeskaluj do int16
            if np.issubdtype(getattr(data, 'dtype', None), np.floating):
                data_int16 = np.int16(np.clip(data, -1.0, 1.0) * 32767)
            else:
                data_int16 = data.astype(np.int16)

            data_unsigned_int16 = np.asarray(data_int16.astype(np.int32) + 2**15, dtype=np.uint16)

            audio_object = {
                'content_signed_int16':      data_int16,
                'content_unsigned_int16':    data_unsigned_int16,
                'fs':                        fs,
                'filepath':                  filepath,
                'filename':                  os.path.basename(filepath),
                'loaded_with':               loaded_with
            }
            return audio_object
        except Exception as e:
            # loguj i przekaż dalej
            try:
                self.logger.exception(f"Error loading audio file {filepath}")
            except Exception:
                pass
            raise

    def save_audio_file(self, path, audio_object):
        try:
            filename = audio_object['name']
            data = audio_object['content']
            fs = audio_object['fs']
            if data.dtype == np.uint16:
                data = data.astype(np.int32) - 32768
            elif data.dtype == np.int16:
                data = data.astype(np.int16)
            else:
                raise TypeError(f"Nieobsługiwany typ danych: {data.dtype}. Oczekiwano int16 lub uint16.")
            data = np.clip(data, -32768, 32767)
            data = data.astype(np.int16)
            wavfile.write(os.path.join(path, filename), fs, data)
        except Exception as e:
            self.logger.exception(f"Error saving audio file {audio_object.get('name', 'unknown')}")
            raise

    def list_files_in_carrier_directory(self):
        try:
            current_directory = os.path.join(os.getcwd(), 'Carriers')
            exts = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
            files = []
            for f in os.listdir(current_directory):
                path = os.path.join(current_directory, f)
                if os.path.isfile(path):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in exts:
                        files.append({'path': path, 'filename': os.path.splitext(f)[0]})
            return files
        except Exception as e:
            self.logger.exception("Error listing files in Carriers/")
            return []

    def list_files_in_secrets_directory(self):
        try:
            current_directory = os.getcwd()
            current_directory += '/image_secrets/'
            files = [{'path': current_directory + f, 'filename': os.path.splitext(f)[0]} for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]
            return files
        except Exception as e:
            self.logger.exception("Error listing files in image_secrets/")
            return []

    def load_carriers(self,files=None):
        try:
            files = self.list_files_in_carrier_directory()
            carriers = []
            for file in files:
                carriers.append(Carrier({'name': file['filename'], 'path' : file['path']}))
                self.logger.debug(f"Loaded carrier: {file['filename']} from path: {file['path']}")
                self.logger.info(f"Loaded carrier: {file['filename']}")
            return carriers
        except Exception as e:
            self.logger.exception("Error loading carriers")
            return []

    def load_carrier(self, filename):
        try:
            files = self.list_files_in_carrier_directory()
            for file in files:
                if file['filename'] == filename:
                    # Wczytaj audio uniwersalną metodą i przekaż gotowe dane do Carrier
                    audio_obj = self.load_audio_file(file['path'])
                    return Carrier({'name': file['filename'], 'fs': audio_obj['fs'], 'data_signed_int16': audio_obj['content_signed_int16']})
            raise FileNotFoundError(f"Carrier file {filename} not found in Carriers/ directory.")
        except Exception as e:
            self.logger.exception(f"Error loading carrier {filename}")
            return None

    def generate_bitvector_from_seed(self, seed, num_bits):
        random.seed(seed)  # Ustawienie ziarna
        bits = []

        # Wygeneruj potrzebną liczbę bitów
        while len(bits) < num_bits:
            # Wygeneruj losową liczbę 32-bitową
            rand_val = random.getrandbits(32)
            bv = BitVector(intVal=rand_val, size=32)
            bits.extend(list(bv))

        # Przytnij do dokładnej długości
        bits = bits[:num_bits]

        # Stwórz ostateczny BitVector
        return BitVector(bitlist=bits)
    
    def load_secrets(self, arguments):
    
        secrets = []
        
        if(arguments['type'] == 'TXT'):
            secret = Secret({'name': self.generate_random_id(), 'path' : arguments['path'], 'type' : arguments['type'], 'size' : arguments['size']})
            secrets.append(secret)
        elif(arguments['type'] == 'IMAGE'):
            files=self.list_files_in_secrets_directory()
            for file in files:
                secrets.append(Secret({'name': file['filename'], 'path' : file['path'], 'type' : 'image'}))
        elif(arguments['type'] == 'MAX_CAP'):
            secret = Secret({'name': None, 'type' : 'MAX_CAP', 'path' : None, 'size' : None})    
            secrets.append(secret)

        return secrets
    
    def prepare_method_directory(self, base_dir: str, method_name: str, only_embed: bool) -> dict:
        try:
            method_dir = os.path.join(base_dir, method_name)
            subdirs = {
                'Stegobjects': os.path.join(method_dir, 'Stegobjects'),
                'Extracted': os.path.join(method_dir, 'Extracted'),
                'Charts': os.path.join(method_dir, 'Charts'),
                'Secrets': os.path.join(method_dir, 'Secrets'),
                'Carriers_Charts': os.path.join(base_dir, 'Carriers', 'Charts'),
                'Base': method_dir
            }
            for path in subdirs.values():

                if only_embed and path.endswith('Stegobjects'):
                    os.makedirs(path, exist_ok=True)
                elif not only_embed:
                    os.makedirs(path, exist_ok=True)
            return subdirs
        except Exception as e:
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
        except Exception as e:
            self.logger.exception("Error archiving previous tests")
            raise

    def save_extracted_secret(self, extracted_data, method_directories, filename):

        """ Save extracted secret data to a file.
        """ 
        if not extracted_data:
            self.logger.warning("No extracted data to save.")
            return
        
        how_much = len(extracted_data) // 8
        ext_sec = extracted_data[0:how_much * 8]
        with open(os.path.join(method_directories['Extracted'], filename + '.txt'), 'w') as file:
            file.write(ext_sec.get_bitvector_in_ascii())

    def create_results_directory(self):
        try:
            os.makedirs(self.results_root, exist_ok=True)
            if not os.path.exists(self.results_root):
                os.makedirs(self.results_root)
        except Exception as e:
            self.logger.exception("Error creating Results directory")
            raise

    def find_files_with_string_in_name(self, directory, search_string, recursive=True):
        matching_files = []
        filenames = []

        for root, _, files in os.walk(directory):
            for file in files:
                if search_string in file:
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
                    file_name_without_extension, _ = os.path.splitext(file)
                    filenames.append(file_name_without_extension)
                    
            if not recursive:
                break

        return matching_files, filenames