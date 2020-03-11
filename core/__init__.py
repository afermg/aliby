import omero_py
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

file_handler = RotatingFileHandler(filename='core.log',
                                   maxBytes=1e5,
                                   backupCount=1)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                   '- %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info('Set up the loggers.')

