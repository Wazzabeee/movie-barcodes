"""Public API for movie_barcodes.

This package provides CLI and library functions to generate movie color barcodes.

Generate a barcode from Python with :func:`create_barcode` (one method) or :func:`create_barcodes` (several methods,
decoding the video once). Both return PIL images.

For several movies, or several renderings of one movie, :func:`iter_barcodes` processes a list of videos one after
another and yields one result per movie (images or error); :func:`extract_movie_colors` and :func:`render_barcode`
split decoding from rendering, and :func:`save_colors` / :func:`load_colors` keep the colors in a ``.npz`` file.
The functions that decode videos accept an ``on_progress(done, total)`` callback and a ``cancel`` event
that stops them with :class:`Cancelled`.
"""

from . import barcode_generation as barcode_generation
from . import color_extraction
from . import video_processing
from .api import create_barcode as create_barcode
from .api import create_barcodes as create_barcodes
from .api import BarcodeResult as BarcodeResult
from .api import Cancelled as Cancelled
from .api import MovieColors as MovieColors
from .api import extract_movie_colors as extract_movie_colors
from .api import iter_barcodes as iter_barcodes
from .api import load_colors as load_colors
from .api import render_barcode as render_barcode
from .api import render_barcodes as render_barcodes
from .api import save_colors as save_colors
from .cli import main as main
from . import utility

__all__ = [
    "create_barcode",
    "create_barcodes",
    "iter_barcodes",
    "extract_movie_colors",
    "render_barcode",
    "render_barcodes",
    "save_colors",
    "load_colors",
    "MovieColors",
    "BarcodeResult",
    "Cancelled",
    "barcode_generation",
    "color_extraction",
    "video_processing",
    "utility",
    "main",
]
