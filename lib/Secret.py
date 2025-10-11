import os
import random
import string
from BitVector import BitVector

class Secret:
    
    def __init__(self, arguments, logger=None):
        self.logger = logger

        self.type = arguments['type']
        size = arguments['size']
        self.save_directory = arguments['save_directory']
        self.binary_data = None
        self.full_text = None

        self.seed = arguments.get('seed', random.randint(0, 2**32 - 1))

        self.create_secret_from_dummy_text(size, self.save_directory)


        # if arguments.get('path') is not None:
        #     self.save = True
        #     self.path = arguments['path']
        # else:
        #     self.save = False
        
        # self.binary_data = None
        # self.full_text = None
        # self.size = 0
        
        # if self.type in ['TXT']:
        #     self.full_text = self.generate_dummy_text(arguments['size'])
        #     self.binary_data = BitVector(textstring=self.full_text)
        #     self.size = len(self.binary_data)

        #     if self.save:
        #         self.save_as_file()


    def save_as_file(self, method_secret_path):
        with open(os.path.join(method_secret_path, self.name + '.txt'), 'w') as file:
            file.write(self.full_text)

    def generate_dummy_text(self, bit_length):
        byte_length = int(bit_length // 8)
        dummy_text = ""

        self.logger.info(f"Starting to generate dummy text.")

        self.logger.debug(f"Generating dummy text of size {byte_length} bytes with seed {self.seed}")

        random.seed(self.seed)
        while len(dummy_text.encode()) < byte_length:
            word = ''.join(random.choices(string.ascii_letters, k=random.randint(1, 10)))
            dummy_text += word + " "

        dummy_text = dummy_text[:byte_length].strip()

        if len(dummy_text) < byte_length:
            dummy_text += "a" * (byte_length - len(dummy_text))

        self.logger.debug(f"Generated dummy text: {dummy_text[:50]}... (length: {len(dummy_text)})")
        
        return dummy_text
    
    def generate_random_filename(self):
        random.seed(self.seed)
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        return f"{random_name}"
    
    def create_secret_from_dummy_text(self, size, method_secret_path):
        
        dummy_text = self.generate_dummy_text(size)

        self.name = self.generate_random_filename()
        self.full_text = dummy_text
        self.binary_data = BitVector(textstring = dummy_text)

        self.size = len(self.binary_data)
        
        self.save_as_file(method_secret_path)

    def to_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "seed": self.seed
        }