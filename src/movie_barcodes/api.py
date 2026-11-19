"""
Public Python API: generate movie barcodes as PIL images, without going through the command line.

Example::

    from movie_barcodes import create_barcode

    image = create_barcode("movie.mp4", method="smoothed", barcode_type="circular")
    image.save("movie_barcode.png")

When more than one worker is used (the default), frames are decoded in child processes: on macOS and Windows,
call these functions under an ``if __name__ == "__main__":`` guard in scripts.
"""

import logging
import os
from typing import Dict, Optional, Sequence, Union

from PIL import Image

from .barcode_generation import generate_barcode, generate_circular_barcode
from .utility import barcode_to_image, get_dominant_color_function, resolve_workers, validate_inputs
from .video_processing import detect_content_area, extract_colors, load_video, parallel_extract_colors

logger = logging.getLogger(__name__)

#: Color extraction methods, in the order used by ``movie-barcodes --all_methods``.
METHODS = ("avg", "hsv", "bgr", "kmeans", "smoothed")
#: Barcode shapes.
BARCODE_TYPES = ("horizontal", "circular")
#: Default (and maximum) number of workers: the number of CPU cores.
MAX_PROCESSES = os.cpu_count() or 1
#: Minimum number of frames a video must have.
MIN_FRAME_COUNT = 2

PathType = Union[str, "os.PathLike[str]"]


def get_target_frames(frame_count: int, frame_width: int, width: Optional[int] = None) -> int:
    """
    Number of frames to sample from the video.

    For horizontal barcodes it is also the output width in pixels; for circular barcodes it is the number of rings.

    :param frame_count: The number of frames in the video
    :param frame_width: The width of the video frames in pixels
    :param width: The requested number of frames, or None for the default
    :return: The number of frames to sample: width if provided, else min(video width, frame count)
    """
    if width is not None:
        return width
    return max(1, min(frame_width, frame_count))


def create_barcodes(
    video_path: PathType,
    methods: Sequence[str] = METHODS,
    barcode_type: str = "horizontal",
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
) -> Dict[str, Image.Image]:
    """
    Generate one barcode per color extraction method, decoding the video only once.

    :param video_path: Path to the video file (.mp4, .webm or .mkv).
    :param methods: Color extraction methods among "avg", "hsv", "bgr", "kmeans" and "smoothed". Default is all of
        them. Duplicates are ignored.
    :param barcode_type: "horizontal" or "circular".
    :param width: Number of sampled frames: the image width for horizontal barcodes, the number of rings for circular
        barcodes. Default is min(video width, number of frames).
    :param height: Height of horizontal barcodes. Default is the video height. Ignored for circular barcodes, whose
        diameter is the video width.
    :param workers: Number of worker processes. Default is the number of CPU cores; 1 decodes sequentially in the
        current process. Values above the number of CPU cores are capped to it, with a warning.
    :param crop_borders: Detect black borders (letterbox / pillarbox) once and ignore them when extracting colors.
    :param linear: Average colors in linear light (brighter on high-contrast scenes). Applies to the "avg" and
        "smoothed" methods only; the other methods are unaffected.
    :return: {method: barcode image}, keyed by the requested method names (also with ``linear``), in the order of
        ``methods``. Horizontal barcodes are RGB images, circular
        barcodes are RGBA images with a transparent background.
    :raises FileNotFoundError: If the video file does not exist.
    :raises ValueError: If an argument is invalid, the video cannot be opened or has too few frames.
    """
    video_path = os.fspath(video_path)
    methods = list(dict.fromkeys(methods))
    if not methods:
        raise ValueError("At least one color extraction method is required.")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Invalid method: {method!r}. Valid methods: {', '.join(METHODS)}.")
    if barcode_type not in BARCODE_TYPES:
        raise ValueError(f"Invalid barcode type: {barcode_type!r}. Valid types: {', '.join(BARCODE_TYPES)}.")

    # Check the input file exists before trying to open it, to get a clear error message
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The specified input video file '{video_path}' does not exist.")

    video, frame_count, frame_width, frame_height = load_video(video_path)
    video.release()
    validate_inputs(video_path, frame_count, MIN_FRAME_COUNT, width=width, height=height)
    workers = resolve_workers(workers, MAX_PROCESSES)

    # Detect black borders once for the whole video
    crop_rect = None
    if crop_borders:
        crop_rect = detect_content_area(video_path, frame_count)
        _, _, crop_width, crop_height = crop_rect
        if (crop_width, crop_height) == (frame_width, frame_height):
            logger.info("Cropping borders: no black borders detected")
            crop_rect = None
        else:
            logger.info("Cropping borders: %dx%d -> %dx%d", frame_width, frame_height, crop_width, crop_height)

    color_functions = {method: get_dominant_color_function(method, linear) for method in methods}

    # Number of frames to sample: one per output column (horizontal) or ring (circular)
    target_frames = get_target_frames(frame_count, frame_width, width)

    if workers == 1:
        colors_by_method = extract_colors(
            video_path, 0, frame_count - 1, color_functions, target_frames, crop_rect=crop_rect
        )
    else:
        colors_by_method = parallel_extract_colors(
            video_path, frame_count, color_functions, workers or MAX_PROCESSES, target_frames, crop_rect=crop_rect
        )

    barcodes = {}
    for method in methods:
        colors = colors_by_method[method]
        if barcode_type == "circular":
            barcode = generate_circular_barcode(colors, frame_width)
        else:
            barcode_height = height if height is not None else frame_height
            barcode = generate_barcode(colors, barcode_height, frame_count, target_frames)
        barcodes[method] = barcode_to_image(barcode)
    return barcodes


def create_barcode(
    video_path: PathType,
    method: str = "avg",
    barcode_type: str = "horizontal",
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
) -> Image.Image:
    """
    Generate the color barcode of a video.

    :param video_path: Path to the video file (.mp4, .webm or .mkv).
    :param method: Color extraction method: "avg" (average), "kmeans" (K-Means clustering), "hsv" (HSV histogram),
        "bgr" (BGR histogram) or "smoothed" (two-step resize). Default is "avg".
    :param barcode_type: "horizontal" or "circular". Default is "horizontal".
    :param width: Number of sampled frames: the image width for horizontal barcodes, the number of rings for circular
        barcodes. Default is min(video width, number of frames).
    :param height: Height of horizontal barcodes. Default is the video height. Ignored for circular barcodes, whose
        diameter is the video width.
    :param workers: Number of worker processes. Default is the number of CPU cores; 1 decodes sequentially in the
        current process. Values above the number of CPU cores are capped to it, with a warning.
    :param crop_borders: Detect black borders (letterbox / pillarbox) once and ignore them when extracting colors.
    :param linear: Average colors in linear light (brighter on high-contrast scenes). Applies to the "avg" and
        "smoothed" methods only; the other methods are unaffected.
    :return: The barcode: an RGB image for horizontal barcodes, an RGBA image with a transparent background for
        circular barcodes.
    :raises FileNotFoundError: If the video file does not exist.
    :raises ValueError: If an argument is invalid, the video cannot be opened or has too few frames.
    """
    barcodes = create_barcodes(
        video_path,
        [method],
        barcode_type=barcode_type,
        width=width,
        height=height,
        workers=workers,
        crop_borders=crop_borders,
        linear=linear,
    )
    return barcodes[method]
