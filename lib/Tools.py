import numpy as np

from scipy.io import wavfile
from scipy.signal.windows import kaiser

from lib.Carrier import Carrier
from lib.Secret import Secret

from BitVector import BitVector

import os

import random
import string
import math
import shutil

from scipy.fftpack import dct, idct, fft, ifft
from datetime import datetime


class Tools:

    def __init__(self):
        self.current_working_directory = os.getcwd()

    def generate_random_id(self, length=4):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    def create_current_test_directory(self):
        test_id = self.generate_random_id(length=7)   
        current_test_directory = os.path.join(self.current_working_directory, 'Results', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f'_{test_id}')
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
        """
        Load an audio file and return a structured audio object.

        Parameters:
        filepath (str): Path to the audio file.

        Returns:
        dict: Audio object with the following keys:
            'content_signed_int16' (np.ndarray): Audio data in signed 16-bit format.
            'content_unsigned_int16' (np.ndarray): Audio data converted to unsigned 16-bit format.
            'fs' (int): Sampling frequency.
            'filepath' (str): Full path to the audio file.
            'filename' (str): Name of the audio file.
        """
        
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Plik {filepath} nie istnieje.")

        # Odczytaj plik WAV
        fs, data = wavfile.read(filepath)

        # Signed int16 (oryginalne dane)
        data_signed_int16 = data.astype(np.int16)

        # Przekształcenie do unsigned int16 (np. do manipulacji bitowej)
        data_unsigned_int16 = np.asarray(data.astype(np.int32) + 2**15, dtype=np.uint16)

        # Utwórz obiekt z danymi
        audio_object = {
            'content_signed_int16':      data_signed_int16,
            'content_unsigned_int16':    data_unsigned_int16,
            'fs':                        fs,
            'filepath':                  filepath,
            'filename':                  os.path.basename(filepath)
        }

        return audio_object
    
    def find_files_with_string_in_name(self, directory, search_string, recursive=True):
        matching_files = []
        filenames = []

        # Przechodzimy przez wszystkie pliki w katalogu (i opcjonalnie podkatalogach)
        for root, _, files in os.walk(directory):
            for file in files:
                if search_string in file:
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
                    file_name_without_extension, _ = os.path.splitext(file)
                    filenames.append(file_name_without_extension)
                    
            # Jeśli nie chcemy przeszukiwać podkatalogów, przerywamy po pierwszym
            if not recursive:
                break

        return matching_files, filenames
    
    def save_audio_file(self, path, audio_object):
        """
        Zapisz dane audio z obiektu stegoobject do pliku WAV.

        Obsługuje zarówno dane typu int16, jak i uint16.

        Parameters:
        - path (str): Ścieżka docelowa do zapisu.
        - stegoobject (dict): Obiekt zawierający dane audio i metadane.
        """
        
        filename = audio_object['name']
        data = audio_object['content']
        fs = audio_object['fs']

        # Upewnij się, że dane są typu int16 (przekształć jeśli trzeba)
        if data.dtype == np.uint16:
            # Przesunięcie zakresu z [0, 65535] -> [-32768, 32767]
            data = data.astype(np.int32) - 32768
        elif data.dtype == np.int16:
            data = data.astype(np.int16)
        else:
            raise TypeError(f"Nieobsługiwany typ danych: {data.dtype}. Oczekiwano int16 lub uint16.")

        data = np.clip(data, -32768, 32767)  # bezpieczeństwo zakresu
        data = data.astype(np.int16)

        # Zapis do pliku WAV
        wavfile.write(os.path.join(path, filename), fs, data)

    def list_files_in_carrier_directory(self):
        current_directory = os.getcwd()
        current_directory += '/Carriers/' # Get the current working directory
        files = [{'path' : current_directory + '/' + f, 'filename': os.path.splitext(f)[0]} for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]
        return files

    def list_files_in_secrets_directory(self):
        current_directory = os.getcwd()
        current_directory += '/image_secrets/'  # Get the current working directory
        files = [{'path': current_directory + f, 'filename': os.path.splitext(f)[0]} for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]
        return files

    def load_carriers(self,files=None):
        
        files = self.list_files_in_carrier_directory()

        carriers = []
        for file in files:
            carriers.append(Carrier({'name': file['filename'], 'path' : file['path']}))
        return carriers

    def load_carrier(self, filename):

        files = self.list_files_in_carrier_directory()

        for file in files:
            if file['filename'] == filename:
                return Carrier({'name': file['filename'], 'path' : file['path']})
        
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
            secret = Secret({'name': self.tools.generate_random_filename(), 'path' : arguments['path'], 'type' : arguments['type'], 'size' : arguments['size']})
            secrets.append(secret)
        elif(arguments['type'] == 'IMAGE'):
            files=self.list_files_in_secrets_directory()
            for file in files:
                secrets.append(Secret({'name': file['filename'], 'path' : file['path'], 'type' : 'image'}))
        elif(arguments['type'] == 'MAX_CAP'):
            secret = Secret({'name': None, 'type' : 'MAX_CAP', 'path' : None, 'size' : None})    
            secrets.append(secret)

        return secrets
    
    def prepare_method_directory(self, base_dir: str, method_name: str) -> dict:
        """
        Tworzy podkatalogi dla danej metody w katalogu testu.

        Args:
            base_dir (str): katalog główny testu (np. 'Results/2025-06-21_12-00-00')
            method_name (str): unikalna nazwa metody

        Returns:
            dict: słownik ze ścieżkami do podkatalogów tej metody
        """
        method_dir = os.path.join(base_dir, method_name)

        subdirs = {
            'Stegobjects': os.path.join(method_dir, 'Stegobjects'),
            'Extracted': os.path.join(method_dir, 'Extracted'),
            'Charts': os.path.join(method_dir, 'Charts'),
            'Secrets': os.path.join(method_dir, 'Secrets'),
            'Carriers_Charts': os.path.join(base_dir, 'Carriers', 'Charts'),
            'Grouped_Charts': os.path.join(base_dir, 'Grouped Charts'),
            'Base': method_dir
        }

        for path in subdirs.values():
            os.makedirs(path, exist_ok=True)

        return subdirs
    
    def archive_previous_tests(self, results_root='Results', archive_folder='Archive'):
        results_path = os.path.abspath(results_root)
        archive_path = os.path.join(results_path, archive_folder)

        os.makedirs(archive_path, exist_ok=True)

        for name in os.listdir(results_path):
            path = os.path.join(results_path, name)
            if os.path.isdir(path) and name != archive_folder:
                dest_path = os.path.join(archive_path, name)
                if not os.path.exists(dest_path):
                    shutil.move(path, dest_path)

    def create_results_directory(self):
        os.makedirs('Results', exist_ok=True)
        if not os.path.exists('Results'):
            os.makedirs('Results')