"""Public API for movie_barcodes.

This package provides CLI and library functions to generate movie color barcodes.

Generate a barcode from Python with :func:`create_barcode` (one method) or :func:`create_barcodes` (several methods,
decoding the video once). Both return PIL images.
"""

from . import barcode_generation as barcode_generation
from . import color_extraction
from . import video_processing
from .api import create_barcode as create_barcode
from .api import create_barcodes as create_barcodes
from .cli import main as main
from . import utility

__all__ = [
    "create_barcode",
    "create_barcodes",
    "barcode_generation",
    "color_extraction",
    "video_processing",
    "utility",
    "main",
]
