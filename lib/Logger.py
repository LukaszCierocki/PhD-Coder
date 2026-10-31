import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_name=None, log_level=logging.INFO, console=False, console_level=logging.INFO):
        """
        Args:
            log_dir (str): Katalog, w którym zapisze się plik logu.
            log_name (str): Nazwa pliku logu.
            log_level (int): Poziom logowania do pliku (np. logging.DEBUG, logging.INFO).
            console (bool): Czy włączyć logowanie do konsoli.
            console_level (int): Poziom logowania do konsoli.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if log_name is None:
            log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"

        log_path = os.path.join(log_dir, log_name)

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)  # Logger zbiera wszystko, filtrują handlery
        self.logger.propagate = False

        # Avoid duplicated lines when a logger with the same name is created twice.
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

        # File handler
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

        # Console handler (opcjonalny)
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(console_level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.log_path = log_path

    def get_logger(self):
        return self.logger

    def get_log_path(self):
        return self.log_path
