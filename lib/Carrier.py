import numpy
import soundfile as sf
from BitVector import BitVector


class Carrier:

    def __init__(self, arguments):
        """
        Konstruktor Carrier może przyjmować argumenty w dwóch formach:
        - {'name': ..., 'path': ...} -- wtedy plik zostanie wczytany (fallback przez scipy.wavfile)
        - {'name': ..., 'fs': ..., 'data_signed_int16': numpy_array} -- jeśli audio zostało już zdekodowane

        Dzięki temu `Tools.load_carrier` może wczytać plik używając bardziej uniwersalnej funkcji
        (np. soundfile/pydub) i przekazać gotowe dane tutaj, unikając cyklicznych zależności.
        """

        self.name = arguments.get('name')
        self.path = arguments.get('path', None)

        # Jeśli dane audio już dostarczone - użyj ich
        if 'data_signed_int16' in arguments and 'fs' in arguments:
            samplerate = arguments['fs']
            data_signed_int16 = numpy.asarray(arguments['data_signed_int16'], dtype=numpy.int16)
        else:
            # Fallback: wczytaj plik ze ścieżki (stare zachowanie)
            if not self.path:
                raise ValueError("No path or audio data provided for Carrier")
            data, samplerate = sf.read(self.path, always_2d=False)
            if getattr(data, 'ndim', 1) == 2:
                data = data.mean(axis=1)
            if numpy.issubdtype(getattr(data, 'dtype', None), numpy.floating):
                data_signed_int16 = (numpy.clip(data, -1.0, 1.0) * 32767.0).round().astype(numpy.int16)
            else:
                data_signed_int16 = numpy.asarray(data, dtype=numpy.int16)

        self.fs = samplerate

        # Upewnij się, że mamy 1D tablicę próbek (jeśli wielokanałowe -> uśrednij do mono)
        if data_signed_int16.ndim == 2:
            # Konwersja do mono poprzez uśrednienie kanałów
            data_signed_int16 = numpy.mean(data_signed_int16, axis=1).astype(numpy.int16)

        self.data_signed_int16 = numpy.asarray(data_signed_int16, dtype=numpy.int16)
        self.data_unsigned_int16 = numpy.asarray(self.data_signed_int16.astype(numpy.int32) + 2**15,
                                                 dtype=numpy.uint16)

        self.bits_per_sample = self.data_signed_int16.dtype.itemsize * 8

        self.carrier_size_in_bits = len(self.data_signed_int16) * self.bits_per_sample

        # Buduj tablicę BitVector
        self.binary_data = numpy.asarray([
            BitVector(intVal=int(sample), size=self.bits_per_sample) for sample in self.data_unsigned_int16
        ])

    def to_metadata(self):
        return {"name": self.name}



class Stegobject(Carrier):

    def __init__(self, arguments):
        super().__init__(arguments)