import logging


class Logger:
  def __init__(self, log_path):
    self.file_handler = logging.FileHandler(log_path, mode='a')
    self.file_handler.setLevel(logging.INFO)
    self.file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    self.logger.addHandler(self.file_handler)

  def log(self, text):
    self.logger.info(text)
    self.file_handler.flush()
    self.file_handler.close()
