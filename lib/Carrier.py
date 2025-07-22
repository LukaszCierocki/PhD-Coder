import numpy
from scipy.io import wavfile
from BitVector import BitVector

class Carrier:
    
    def __init__(self, arguments):
        
        self.name = arguments['name']
        self.path = arguments['path']
        
        samplerate, data = wavfile.read(self.path)
        
        self.fs = samplerate

        self.data_unsigned_int16 = numpy.asarray(data.astype(numpy.int32) + 2**15, dtype=numpy.uint16)

        self.data_signed_int16 = data
        self.data_signed_int16 = numpy.asarray(self.data_signed_int16, dtype=numpy.int16)
            
        self.bits_per_sample = data.dtype.itemsize * 8
        
        self.carrier_size_in_bits = len(self.data_signed_int16) * self.bits_per_sample
        
        self.binary_data = numpy.asarray([BitVector(intVal = sample, size = self.bits_per_sample) for sample in self.data_unsigned_int16])

    def to_metadata(self):
        return {
            "name": self.name
        }


class Stegobject(Carrier):
    
    def __init__(self, arguments):
        super().__init__(arguments)