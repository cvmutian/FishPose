import logging
import os
from typing import Optional, Dict, Any

from torch.utils.tensorboard import SummaryWriter
import csv

from .dist import is_main_process

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        if not hasattr(self, 'initialized'):
            self.writer = None
            self.csv_file = None
            self.csv_writer = None
            self.fieldnames = []
            
            if is_main_process() and log_dir and config:
                self._setup_logging(log_dir)
                if config['logging']['tensorboard']:
                    self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tb'))
                if config['logging']['csv']:
                    self.csv_file_path = os.path.join(log_dir, 'logs.csv')
                    self.csv_file = open(self.csv_file_path, 'w', newline='')

            self.initialized = True

    def _setup_logging(self, log_dir: str):
        log_file = os.path.join(log_dir, 'console.log')
        
        logger = logging.getLogger()
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ''):
        if not is_main_process():
            return

        prefixed_metrics = {f'{prefix}{k}': v for k, v in metrics.items()}

        if self.writer:
            for key, value in prefixed_metrics.items():
                self.writer.add_scalar(key, value, step)
        else:
            logging.warning("TensorBoard writer not initialized.")

        if self.csv_file:
            if not self.csv_writer:
                self.fieldnames = ['step'] + list(prefixed_metrics.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
                self.csv_writer.writeheader()
            
            row = {'step': step, **prefixed_metrics}
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def info(self, message: str):
        if is_main_process():
            logging.info(message)

    def warning(self, message: str):
        if is_main_process():
            logging.warning(message)

    def error(self, message: str):
        if is_main_process():
            logging.error(message)

    def close(self):
        if is_main_process():
            if self.writer:
                self.writer.close()
            if self.csv_file:
                self.csv_file.close()
            self._instance = None
