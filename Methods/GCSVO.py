from lib.Method import Method
from BitVector import BitVector
import numpy as numpy
from timeit import default_timer as timer

from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray

class GCSVO(Method):
    def __init__(self, method_arguments):
        self.k = method_arguments['K']
        self.n = method_arguments['N']
        self.name = 'GCSVO' + '_' + str(self.k) + '_' + str(self.n)
        super().__init__(method_arguments)
        
        self.style = 'ERROR MINIMIZATION'
        self.domain = 'SPATIAL'
        
        gray_codes = GrayCode(self.n)
        self.gray_codes = list(gray_codes.generate_gray())
        self.binary = self.gen_binary(self.k)
        self.equivalence_classes = self.create_equivalence_classes()

        self.secret = None
        
        self.embedding_time = 0
        self.extraction_time = 0
        
    def calculate_capacity(self, carrier):
        return len(carrier.binary_data) * self.k

    def gen_binary(self, k):
        bins = []
        max_int = '0b' + '1' * k
        for i in range(0, int(max_int, 2) + 1, 1):
            bins.append(str(format(i, 'b').zfill(k)))
        return bins
    
    def findClosest(self, array, value):
        array = numpy.asarray(array)
        idx = (numpy.abs(array - value)).argmin()
        return array[idx]
    
    def embedding(self, carrier, secret):

        
        self.capacity = self.calculate_capacity(carrier)

        if(secret.type == 'MAX_CAP'):
            secret.create_secret_from_dummy_text(self.capacity)

        self.secret = secret
        
        if(self.secret.size > self.capacity):
            raise Exception(f"Method doesnt have enough capacity - method capacity {self.capacity} - secret size {secret.size}")
        
        embedding_start = timer()
        
        stegobject = []
        embeeded_bits = 0
        
        for sample in carrier.binary_data:
            
            if embeeded_bits < self.secret.size:
                if(embeeded_bits + self.k <= secret.size):
                    sample_binary = BitVector(bitlist=sample)
                    secret_tmp = BitVector(bitstring=bin_to_gray(str(self.secret.binary_data[embeeded_bits:embeeded_bits + self.k])))
                    nearest = self.getNearestEquivalenceClass(secret_tmp, sample_binary[-self.n:])
                    sample_binary[-self.n:] = nearest
                    stegobject.append(int(sample_binary))
                    embeeded_bits += self.k
                else:
                    remaining_bits = secret.size - embeeded_bits
                    secret_tmp = BitVector(bitstring=bin_to_gray(str(self.secret.binary_data[embeeded_bits:embeeded_bits + remaining_bits])))
                    secret_tmp = secret_tmp.pad_from_left(self.k - remaining_bits)
                    nearest = self.getNearestEquivalenceClass(secret_tmp, sample_binary[-self.n:])
                    sample_binary[-self.n:] = nearest
                    stegobject.append(int(sample_binary))
                    embeeded_bits += len(remaining_bits)
            else:
                stegobject.append(int(BitVector(bitlist=sample)))
                
        embedding_stop = timer()
        
        self.embedding_time = embedding_stop - embedding_start
                    
        return numpy.asarray(stegobject).astype(numpy.uint16)
    
    def extraction(self, stegobject, secret_size):
        
        extraction_start = timer()
        
        extracted_bits = []
        
        for sample in stegobject['content_unsigned_int16']:
            
            if len(extracted_bits) * self.k == secret_size:
                break

            sample_binary = BitVector(intVal=sample)
            extracted_bits.append(gray_to_bin(str(sample_binary[-self.k:])))
            
        extraction_stop = timer()
        
        self.extraction_time = extraction_stop - extraction_start
        
        return BitVector(bitstring=''.join(extracted_bits))
    
    def create_equivalence_classes(self):
        equivalence_classes = []
        for binary_value in self.binary:
            equivalence_class = {'secretValue': binary_value, 'grayCodes': []}
            for gray_code in self.gray_codes:
                if binary_value == gray_code[-self.k:]:
                    equivalence_class['grayCodes'].append(int(BitVector(bitstring=gray_code)))
            equivalence_class['grayCodes'].sort()
            equivalence_classes.append(equivalence_class)
        return equivalence_classes
    
    def getNearestEquivalenceClass(self, secret, toFind):
        for equivalence_class in self.equivalence_classes:
            if str(secret) == equivalence_class['secretValue']:
                closestValue = self.findClosest(equivalence_class['grayCodes'], int(toFind))
                break
        return BitVector(intVal=closestValue, size=self.n)