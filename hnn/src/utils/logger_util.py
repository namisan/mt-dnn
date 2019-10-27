"""
Logging util
@Author: penhe@microsoft.com
"""

import logging

import pdb
logging.basicConfig(format = '%(asctime)s|%(levelname)s|%(name)s| %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger=None
def set_logger(name, file_log):
    global logger
    if not logger:
      logger = logging.getLogger(name)
    else:
      logger.name = name
      for h in logger.handlers:
        logger.removeHandler(h)
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s| %(message)s', datefmt='%m%d%Y %H:%M:%S')
    fh = logging.FileHandler(file_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def get_logger(name='logging', file_log='/tmp/log.txt'):
  global logger
  if not logger:
    logger = set_logger(name, file_log)
  return logger
