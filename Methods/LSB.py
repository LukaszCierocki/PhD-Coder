from lib.Method import Method
from BitVector import BitVector
import numpy as numpy
from timeit import default_timer as timer

class LSB(Method):
    
    """
        Class for Least Significant Bit method.
        
        Arguments:
        method_arguments : dictionary
            K - number of least significant bits used embed in one sample.
    """
        
    def __init__(self, method_arguments):
        
        self.k = method_arguments['K']
        
        super().__init__(method_arguments)

        self.domain = 'SPATIAL'
        self.style = 'SEQUENTIAL'
        
        self.name = 'LSB' + '_' + str(self.k)
        
        self.embedding_time = 0
        self.extraction_time = 0
        
        self.capacity = 0

        self.secret = None
        
        
    def calculate_capacity(self, carrier):
        return len(carrier.binary_data) * self.k

    def embedding(self, carrier, secret):
            
        self.capacity = self.calculate_capacity(carrier)

        if(secret.type == 'MAX_CAP'):
            secret.create_secret_from_dummy_text(self.capacity)
        
        if(secret.size > self.capacity):
            raise Exception(f"Method doesnt have enough capacity - method capacity {self.capacity} - secret size {secret.size}")
        

        self.secret = secret

        start_embedding = timer()
        
        stegobject = []
        embeeded_bits = 0
        
        for sample in carrier.binary_data:
            
            if embeeded_bits < secret.size:
                
                sample_binary = BitVector(bitlist=sample)
                
                if embeeded_bits + self.k <= secret.size:
                    sample_binary[-self.k:] = secret.binary_data[embeeded_bits:embeeded_bits+self.k]
                else:
                    remaining_bits = secret.size - embeeded_bits
                    sample_binary[-remaining_bits:] = secret.binary_data[embeeded_bits:embeeded_bits+remaining_bits]
                
                stegobject.append(int(sample_binary))
                embeeded_bits += self.k
            else:
                stegobject.append(int(BitVector(bitlist=sample)))
                
        stop_embedding = timer()
        
        self.embedding_time = stop_embedding - start_embedding
                    
        return numpy.asarray(stegobject).astype(numpy.uint16)

    def extraction(self, stegobject, secret_size):
        
        extration_start = timer()
        
        extracted_bits = []
        
        for sample in stegobject['content_unsigned_int16']:
            if len(extracted_bits) * self.k == secret_size:
                break
    
            sample_binary = BitVector(intVal=sample)
            extracted_bits.append(str(sample_binary[-self.k:]))
            
        extraction_stop = timer()
        
        self.extraction_time = extraction_stop - extration_start

        return BitVector(bitstring=''.join(extracted_bits))