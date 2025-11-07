"""Public API for movie_barcodes.

This package provides CLI and library functions to generate movie color barcodes.
"""

from . import barcode_generation as barcode
from . import barcode_generation as barcode_generation
from . import color_extraction
from . import video_processing
from .cli import main as main
from . import utility

__all__ = [
    "barcode",
    "barcode_generation",
    "color_extraction",
    "video_processing",
    "utility",
    "main",
]
