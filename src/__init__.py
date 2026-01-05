"""Face Mask Detection Project - Modularized Components"""

__version__ = "2.0.0"
__author__ = "Miguel"

from . import config
from . import download
from . import parser
from . import converter
from . import train
from . import evaluate
from . import visualize
from . import utils
from . import main

__all__ = [
    'config',
    'download',
    'parser',
    'converter',
    'train',
    'evaluate',
    'visualize',
    'utils',
    'main',
]
