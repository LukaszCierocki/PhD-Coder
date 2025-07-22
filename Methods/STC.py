from lib.Method import Method
from BitVector import BitVector
import numpy as numpy
from timeit import default_timer as timer
import os
from ctypes import *
from scipy.signal import butter, filtfilt
import pywt

def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

class STC(Method):
        
    def __init__(self, method_arguments):
        
        self.alpha = method_arguments['ALPHA']

        self.cost_function = method_arguments['COST_FUNCTION']

        self.inverse = method_arguments['INVERSE']

        super().__init__(method_arguments)

        self.domain = 'DISTORTION'
        self.style = 'SEQUENTIAL'
        
        self.name = 'STC' + '_' + str(self.alpha) + '_' + self.cost_function + '_' + str(self.inverse)
        
        self.embedding_time = 0
        self.extraction_time = 0
        
        me = os.path.abspath(os.path.dirname(__file__))
        self.lib = cdll.LoadLibrary(os.path.join(me, "lib", "stc.so"))
        
        self.capacity = 0


    def calculate_capacity(self, carrier):
        return round(len(carrier.binary_data) * self.alpha)

    def embedding(self, carrier, secret):
            
        self.capacity = self.calculate_capacity(carrier)
        
        if(secret.size > self.capacity):
            raise Exception(f"Method doesnt have enough capacity - method capacity {self.capacity} - secret size {secret.size}")
        
        start_embedding = timer()
        
        # Konwersja na tablicę ctypes
        cover = (c_int * len(carrier.raw_data))()
        for idx, sample in enumerate(carrier.raw_data):
            cover[idx] = sample

        # Prepare costs

        if self.cost_function == 'HILL':
            costs = self.calculate_hill_cost(carrier, cover)
        elif self.cost_function == 'NONE':
            costs = self.create_none_cost(cover)
        elif self.cost_function == 'WOW':
            costs = self.calculate_wow_cost(carrier, cover)
        elif self.cost_function == 'HUGO':
            costs = self.calculate_hugo_cost(carrier, cover)
        
        m = int(len(carrier.raw_data)*self.alpha)
        
        message = (c_ubyte*m)()
        
        for i in range(m):
            if i<len(secret.binary_data):
                message[i] = secret.binary_data[i]
            else:
                message[i] = 0
                
        stego = (c_int*(len(carrier.raw_data)*1))()
        
        a = self.lib.stc_hide(len(carrier.raw_data), cover, costs, m, message, stego)

        time = numpy.linspace(0, len(carrier.raw_data) / carrier.fs, len(carrier.raw_data))

        data = carrier.raw_data - 32767
        
        stegobject = []
        # Save output message
        idx=0
        for j in range(1):
            for i in range(len(carrier.raw_data)):
                stegobject.append(stego[idx])
                idx += 1
                
        stop_embedding = timer()
        
        self.embedding_time = stop_embedding - start_embedding
                    
        return numpy.asarray(stegobject)
    
    def calculate_hill_cost(self, carrier, cover, cutoff_high=1000, cutoff_low=300):
        
        INF = 2**31-1
        # costs = (c_float*(len(carrier.raw_data)))()

        data = carrier.raw_data
        fs = carrier.fs
        # Normalize and prepare data
        data = data - numpy.mean(data)

        # Apply High-pass and Low-pass filters
        high_passed = high_pass_filter(data, cutoff_high, fs)
        low_passed = low_pass_filter(data, cutoff_low, fs)

        # Combine filtered signals to create a cost map
        cost_map = numpy.abs(high_passed) + numpy.abs(low_passed)

        # Normalize cost map to [0, 1]
        cost_map = cost_map / numpy.max(cost_map)

        cost_map[cost_map == numpy.inf] = 1

        if self.inverse:
            cost_map = self.invert_costs(cost_map, method="inverse_normalized", epsilon=1e-10)

        costs = (c_float*(len(carrier.raw_data)*1*3))()
        idx=0
        
        for j in range(1):
            for i in range(len(carrier.raw_data)):
                if cover[idx]==0:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                elif cover[idx] == 65535:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                idx += 1

        return costs
    
    def calculate_wow_cost(self, carrier, cover):

        INF = 2**31-1
        # costs = (c_float*(len(carrier.raw_data)))()

        data = carrier.raw_data
        fs = carrier.fs
        # Normalize and prepare data
        wavelet='db1'
        level=3

        coeffs = pywt.swt(data, wavelet=wavelet, level=level)
        
        # Obliczanie reszt jako sumy wartości bezwzględnych współczynników
        residuals = [numpy.abs(detail) for _, detail in coeffs]
        
        # Agregacja do funkcji kosztów (np. suma reszt)
        cost_map = sum(residuals)
        
        # Normalizacja do zakresu [0, 1]
        cost_map = cost_map / numpy.max(cost_map)

        if self.inverse:
            cost_map = self.invert_costs(cost_map, method="inverse_normalized", epsilon=1e-10)

        costs = (c_float*(len(carrier.raw_data)*1*3))()
        idx=0
        
        for j in range(1):
            for i in range(len(carrier.raw_data)):
                if cover[idx]==0:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                elif cover[idx] == 65535:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                idx += 1

        return costs
    
    def create_none_cost(self, cover):

        INF = 2**31-1

        costs = (c_float*(len(cover.raw_data)*1*3))()
        idx=0
        
        for j in range(1):
            for i in range(len(cover.raw_data)):
                if cover[idx]==0:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = 0
                elif cover[idx] == 65535:
                    costs[3*idx+0] = 0
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = 0
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = 0
                idx += 1

        return costs
    
    def calculate_hugo_cost(self, carrier, cover):

        INF = 2**31-1
        data = carrier.raw_data
        fs = carrier.fs
        # Normalize and prepare data

        window_size=6

        differences = numpy.abs(numpy.diff(data))
        differences = numpy.append(differences, 0)

        # Obliczanie lokalnych cech (np. średnia różnic w oknie)
        local_features = numpy.convolve(differences, numpy.ones(window_size), mode='same') / window_size

        epsilon = 1e-10  # Zapobiega dzieleniu przez 0
        cost_map = 1 / (local_features + epsilon)
        
        # Normalizacja do zakresu [0, 1]
        cost_map = cost_map / numpy.max(cost_map)

        if self.inverse:
            cost_map = self.invert_costs(cost_map, method="inverse_normalized", epsilon=1e-10)

        costs = (c_float*(len(carrier.raw_data)*1*3))()

        idx=0
        
        for j in range(1):
            for i in range(len(carrier.raw_data)):
                if cover[idx]==0:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                elif cover[idx] == 65535:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = cost_map[i]
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_map[i]
                idx += 1

        return costs
    
    def invert_costs(self, costs, method="reciprocal", epsilon=1e-10):
        """
        Odwraca wartości funkcji kosztów w celu zmiany interpretacji.
        :param costs: Mapa kosztów (np. wynik funkcji WOW)
        :param method: Metoda odwracania ("reciprocal" lub "inverse_normalized")
        :param epsilon: Mała wartość zapobiegająca dzieleniu przez zero
        :return: Odwrócona mapa kosztów
        """
        if method == "reciprocal":
            # Metoda 1: Odwrotność kosztów (1 / koszt)
            inverted_costs = 1 / (costs + epsilon)
        elif method == "inverse_normalized":
            # Metoda 2: Odwracanie i normalizacja do zakresu [0, 1]
            inverted_costs = 1 - costs
        else:
            raise ValueError("Nieznana metoda odwracania: wybierz 'reciprocal' lub 'inverse_normalized'.")
        
        return inverted_costs

    def extraction(self, carrier, secret_size):
        
        extration_start = timer()
        
        stego = (c_int * len(carrier['content']))()
        for idx, sample in enumerate(carrier['content']):
            stego[idx] = sample

        # Extract the message
        n = len(carrier['content'])*1;
        m = int(len(carrier['content'])*1*self.alpha)
        
        extracted_message = (c_ubyte*m)()
        s = self.lib.stc_unhide(n, stego, m, extracted_message)

        bits = []
        for bit in extracted_message:
            bits.append(bit)
        
        extraction_stop = timer()
        
        self.extraction_time = extraction_stop - extration_start
        
        bits = bits[0:secret_size]
        
        return BitVector(bitlist = bits)