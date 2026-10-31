import os
import random
import string
from BitVector import BitVector

from lib.Method import ensure_logger


class Secret:
    """
    Pseudo-random text secret.

    ``size`` is the number of bits available (e.g. the method capacity); the
    secret uses ``8 * (size // 8)`` bits of ASCII text. The text is generated
    from ``seed`` with a private ``random.Random`` instance, so it is
    reproducible and does not disturb the global ``random`` state (the
    sequence is identical to the previous ``random.seed(seed)`` version, so
    old seeds still give the same secrets).
    """

    def __init__(self, arguments, logger=None):
        self.logger = ensure_logger(logger)

        self.type = arguments.get('type', 'MAX_CAP')
        size = arguments.get('size')
        if size is None:
            raise ValueError("Secret needs 'size' (number of bits, e.g. method.get_capacity()).")
        if isinstance(size, bool) or int(size) != size or size < 8:
            raise ValueError(f"Secret size must be an integer >= 8 bits, got {size!r}.")
        size = int(size)

        self.save_directory = arguments.get('save_directory')
        self.binary_data = None
        self.full_text = None

        self.only_embed = arguments.get('only_embed', False)

        seed = arguments.get('seed')
        self.seed = int(seed) if seed is not None else random.SystemRandom().randint(0, 2**32 - 1)

        self.create_secret_from_dummy_text(size, self.save_directory)

    def save_as_file(self, method_secret_path):
        if not method_secret_path:
            return
        os.makedirs(method_secret_path, exist_ok=True)
        with open(os.path.join(method_secret_path, self.name + '.txt'), 'w', encoding='ascii') as file:
            file.write(self.full_text)

    def generate_dummy_text(self, bit_length):
        byte_length = int(bit_length // 8)
        dummy_text = ""

        self.logger.debug("Generating dummy text of %d bytes with seed %d", byte_length, self.seed)

        rng = random.Random(self.seed)
        while len(dummy_text.encode()) < byte_length:
            word = ''.join(rng.choices(string.ascii_letters, k=rng.randint(1, 10)))
            dummy_text += word + " "

        dummy_text = dummy_text[:byte_length].strip()

        if len(dummy_text) < byte_length:
            dummy_text += "a" * (byte_length - len(dummy_text))

        return dummy_text

    def generate_random_filename(self):
        rng = random.Random(self.seed)
        return ''.join(rng.choices(string.ascii_letters + string.digits, k=5))

    def create_secret_from_dummy_text(self, size, method_secret_path):

        dummy_text = self.generate_dummy_text(size)

        self.name = self.generate_random_filename()
        self.full_text = dummy_text
        self.binary_data = BitVector(textstring=dummy_text)

        self.size = len(self.binary_data)

        if self.size == 0 or self.size > size:
            raise RuntimeError(f"Generated secret has {self.size} bits for a capacity of {size} bits.")

        if not self.only_embed:
            self.save_as_file(method_secret_path)

    def to_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "seed": self.seed,
            "size": self.size
        }
