import numpy
import soundfile as sf
from BitVector import BitVector


def read_audio_as_int16(path):
    """
    Reads an audio file into an exact int16 mono signal.

    * PCM files are read directly as int16 (no float round trip, so the
      samples are bit-exact - the previous ``float * 32767`` conversion changed
      ~99% of the samples by 1 LSB);
    * float files are scaled by 32768 (the soundfile/libsndfile convention),
      rounded and clipped;
    * multi-channel files are averaged to mono (with rounding).

    Returns ``(data_int16, fs)``.
    """
    info = sf.info(path)
    if info.subtype.startswith('PCM') or info.subtype in ('ULAW', 'ALAW'):
        data, fs = sf.read(path, dtype='int16', always_2d=True)
        if data.shape[1] == 1:
            return data[:, 0].copy(), fs
        mono = numpy.rint(data.astype(numpy.float64).mean(axis=1))
        return numpy.clip(mono, -32768, 32767).astype(numpy.int16), fs

    data, fs = sf.read(path, dtype='float64', always_2d=True)
    mono = data.mean(axis=1) * 32768.0
    return numpy.clip(numpy.rint(mono), -32768, 32767).astype(numpy.int16), fs


class Carrier:

    def __init__(self, arguments):
        """
        Konstruktor Carrier może przyjmować argumenty w dwóch formach:
        - {'name': ..., 'path': ...} -- wtedy plik zostanie wczytany (dokładnie jako int16)
        - {'name': ..., 'fs': ..., 'data_signed_int16': numpy_array} -- jeśli audio zostało już zdekodowane
        """

        self.name = arguments.get('name')
        self.path = arguments.get('path', None)

        if 'data_signed_int16' in arguments and 'fs' in arguments:
            samplerate = arguments['fs']
            data_signed_int16 = numpy.asarray(arguments['data_signed_int16'])
            if data_signed_int16.dtype != numpy.int16:
                if numpy.issubdtype(data_signed_int16.dtype, numpy.integer) and (
                        data_signed_int16.size == 0 or
                        (data_signed_int16.min() >= -32768 and data_signed_int16.max() <= 32767)):
                    data_signed_int16 = data_signed_int16.astype(numpy.int16)
                else:
                    raise ValueError(f"Carrier {self.name}: data must be int16 samples, "
                                     f"got dtype {data_signed_int16.dtype}.")
        else:
            if not self.path:
                raise ValueError("No path or audio data provided for Carrier")
            data_signed_int16, samplerate = read_audio_as_int16(self.path)

        self.fs = int(samplerate)

        if data_signed_int16.ndim == 2:
            mono = numpy.rint(data_signed_int16.astype(numpy.float64).mean(axis=1))
            data_signed_int16 = numpy.clip(mono, -32768, 32767).astype(numpy.int16)

        if data_signed_int16.ndim != 1 or len(data_signed_int16) == 0:
            raise ValueError(f"Carrier {self.name} contains no audio samples.")

        self.data_signed_int16 = numpy.asarray(data_signed_int16, dtype=numpy.int16)
        self.data_unsigned_int16 = numpy.asarray(self.data_signed_int16.astype(numpy.int32) + 2**15,
                                                 dtype=numpy.uint16)

        self.bits_per_sample = self.data_signed_int16.dtype.itemsize * 8

        self.carrier_size_in_bits = len(self.data_signed_int16) * self.bits_per_sample

        self._binary_data = None

    @property
    def duration(self):
        """Carrier duration in seconds."""
        return len(self.data_signed_int16) / self.fs

    @property
    def binary_data(self):
        """
        Per-sample BitVectors (built lazily - only LSB-style methods need them
        and building them for every carrier was slow).
        """
        if self._binary_data is None:
            self._binary_data = numpy.asarray([
                BitVector(intVal=int(sample), size=self.bits_per_sample) for sample in self.data_unsigned_int16
            ])
        return self._binary_data

    def to_metadata(self):
        return {"name": self.name, "fs": self.fs, "samples": len(self.data_signed_int16)}


class Stegobject(Carrier):

    def __init__(self, arguments):
        super().__init__(arguments)
