"""
Public Python API: generate movie barcodes as PIL images, without going through the command line.

Example::

    from movie_barcodes import create_barcode

    image = create_barcode("movie.mp4", method="smoothed", barcode_type="circular")
    image.save("movie_barcode.png")

Decoding is the slow part. To render a movie several ways, extract its colors once with
:func:`extract_movie_colors`, then call :func:`render_barcode` as many times as needed (any method extracted, any
barcode type, any height). The colors can be kept on disk with :func:`save_colors` / :func:`load_colors`.
:func:`iter_barcodes` does all of this for a list of movies, one after another, and reports per-movie errors
instead of stopping.

Long calls accept ``on_progress(done, total)``, called as frames are decoded, and ``cancel``, a
:class:`threading.Event` that stops the decoding (terminating the worker processes) and raises :class:`Cancelled`
once set: e.g. to drive a progress bar and a cancel button from a background thread.

When more than one worker is used (the default), frames are decoded in child processes: on macOS and Windows,
call these functions under an ``if __name__ == "__main__":`` guard in scripts.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .barcode_generation import generate_barcode, generate_circular_barcode
from .utility import barcode_to_image, get_dominant_color_function, resolve_workers, validate_inputs
from .video_processing import (
    Cancelled,
    ProgressCallback,
    _sample_count,
    check_cancel,
    _split_work,
    detect_content_area,
    extract_colors,
    load_video,
    parallel_extract_colors,
)

logger = logging.getLogger(__name__)

#: Color extraction methods, in the order used by ``movie-barcodes --all_methods``.
METHODS = ("avg", "hsv", "bgr", "kmeans", "smoothed")
#: Barcode shapes.
BARCODE_TYPES = ("horizontal", "circular")
#: Default (and maximum) number of workers: the number of CPU cores.
MAX_PROCESSES = os.cpu_count() or 1
#: Minimum number of frames a video must have.
MIN_FRAME_COUNT = 2
#: Extension of the colors files written by :func:`save_colors`.
COLORS_FILE_EXTENSION = ".npz"
#: Version of the colors file format, stored in the file.
COLORS_FILE_VERSION = 1

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


@dataclass
class MovieColors:
    """
    Colors extracted from a video: everything needed to render its barcodes again without decoding it.

    :ivar colors: {method: colors}, one entry per sampled frame, in frame order. Each method keeps the values
        returned by its color extraction function: (N, 3) BGR colors, or (N, height, 1, 3) uint8 columns for
        "smoothed".
    :ivar frame_count: Number of frames of the video.
    :ivar frame_width: Width of the video frames, also the diameter of circular barcodes.
    :ivar frame_height: Height of the video frames, the default height of horizontal barcodes.
    :ivar target_frames: Number of frames requested, the width of horizontal barcodes.
    :ivar frame_indices: Indices of the frames requested for sampling, in order. They are the requested ones: a seek
        can land a frame further, and a video shorter than its reported frame count gives fewer colors.
    :ivar fps: Frame rate of the video (0 if unknown).
    :ivar crop_rect: (x, y, w, h) of the area colors were extracted from (``crop_borders``), or None for the full
        frame.
    :ivar linear: Whether avg and smoothed colors were averaged in linear light.
    :ivar video_name: File name of the video, used to name output files.
    """

    colors: Dict[str, np.ndarray]
    frame_count: int
    frame_width: int
    frame_height: int
    target_frames: int
    frame_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fps: float = 0.0
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    linear: bool = False
    video_name: str = ""

    @property
    def methods(self) -> List[str]:
        """
        Methods whose colors are available, in extraction order.
        """
        return list(self.colors)


def _check_methods(methods: Iterable[str]) -> List[str]:
    """
    Remove duplicates and check the color extraction methods are valid.
    """
    methods = list(dict.fromkeys(methods))
    if not methods:
        raise ValueError("At least one color extraction method is required.")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Invalid method: {method!r}. Valid methods: {', '.join(METHODS)}.")
    return methods


def _check_barcode_types(barcode_types: Iterable[str]) -> List[str]:
    """
    Remove duplicates and check the barcode types are valid.
    """
    barcode_types = list(dict.fromkeys(barcode_types))
    if not barcode_types:
        raise ValueError("At least one barcode type is required.")
    for barcode_type in barcode_types:
        if barcode_type not in BARCODE_TYPES:
            raise ValueError(f"Invalid barcode type: {barcode_type!r}. Valid types: {', '.join(BARCODE_TYPES)}.")
    return barcode_types


def _requested_frame_indices(frame_count: int, target_frames: int, workers: int) -> np.ndarray:
    """
    Indices of the frames sampled by extract_colors / parallel_extract_colors, in order.
    """
    tasks = [(0, frame_count - 1, target_frames)] if workers == 1 else _split_work(frame_count, target_frames, workers)
    indices = []
    for start, end, samples in tasks:
        total = end - start + 1
        n_samples = _sample_count(start, end, samples)
        indices.extend(start + i * total // n_samples for i in range(n_samples))
    return np.asarray(indices, dtype=np.int64)


def extract_movie_colors(
    video_path: PathType,
    methods: Sequence[str] = METHODS,
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
) -> MovieColors:
    """
    Decode a video once and extract the colors of its sampled frames with one or several methods.

    Render them with :func:`render_barcode`, keep them with :func:`save_colors`.

    :param video_path: Path to the video file.
    :param methods: Color extraction methods among "avg", "hsv", "bgr", "kmeans" and "smoothed". Default is all of
        them. Duplicates are ignored.
    :param width: Number of sampled frames. Default is min(video width, number of frames).
    :param height: Only validated here (must be positive), for the barcodes rendered later.
    :param workers: Number of worker processes. Default is the number of CPU cores; 1 decodes sequentially in the
        current process. Values above the number of CPU cores are capped to it, with a warning.
    :param crop_borders: Detect black borders (letterbox / pillarbox) once and ignore them when extracting colors.
    :param linear: Average colors in linear light. Applies to the "avg" and "smoothed" methods only.
    :param on_progress: Called with (frames processed, frames to sample) as frames are decoded, in the calling
        thread.
    :param cancel: Once set (e.g. from another thread), decoding stops, worker processes are terminated and
        :class:`Cancelled` is raised. Checked for each frame (sequential) or every 0.1 s (parallel); starting the
        worker processes and detecting borders are not interruptible.
    :return: The extracted colors and the video properties needed to render them.
    :raises FileNotFoundError: If the video file does not exist.
    :raises ValueError: If an argument is invalid, the video cannot be opened or has too few frames.
    :raises Cancelled: If cancel is set.
    """
    video_path = os.fspath(video_path)
    methods = _check_methods(methods)
    check_cancel(cancel)

    # Check the input file exists before trying to open it, to get a clear error message
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The specified input video file '{video_path}' does not exist.")

    video, frame_count, frame_width, frame_height = load_video(video_path)
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    video.release()
    validate_inputs(video_path, frame_count, MIN_FRAME_COUNT, width=width, height=height)
    workers = resolve_workers(workers, MAX_PROCESSES) or MAX_PROCESSES

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

    check_cancel(cancel)

    color_functions = {method: get_dominant_color_function(method, linear) for method in methods}

    # Number of frames to sample: one per output column (horizontal) or ring (circular)
    target_frames = get_target_frames(frame_count, frame_width, width)

    if workers == 1:
        colors_by_method = extract_colors(
            video_path,
            0,
            frame_count - 1,
            color_functions,
            target_frames,
            crop_rect=crop_rect,
            on_progress=on_progress,
            cancel=cancel,
        )
    else:
        colors_by_method = parallel_extract_colors(
            video_path,
            frame_count,
            color_functions,
            workers,
            target_frames,
            crop_rect=crop_rect,
            on_progress=on_progress,
            cancel=cancel,
        )

    return MovieColors(
        # Stacking keeps each method's own dtype (float64 means, uint8 histograms and smoothed columns)
        colors={method: np.asarray(colors_by_method[method]) for method in methods},
        frame_count=frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
        target_frames=target_frames,
        frame_indices=_requested_frame_indices(frame_count, target_frames, workers),
        fps=fps,
        crop_rect=tuple(int(value) for value in crop_rect) if crop_rect is not None else None,
        linear=linear,
        video_name=os.path.basename(video_path),
    )


def render_barcode(
    movie_colors: MovieColors, method: str = "avg", barcode_type: str = "horizontal", height: Optional[int] = None
) -> Image.Image:
    """
    Render a barcode from extracted colors, without decoding the video.

    :param movie_colors: Colors returned by :func:`extract_movie_colors` or :func:`load_colors`.
    :param method: The color extraction method to render; its colors must have been extracted.
    :param barcode_type: "horizontal" or "circular".
    :param height: Height of horizontal barcodes. Default is the video height. Ignored for circular barcodes, whose
        diameter is the video width.
    :return: An RGB image for horizontal barcodes, an RGBA image with a transparent background for circular ones.
    :raises ValueError: If the method was not extracted, or the barcode type or height is invalid.
    """
    _check_barcode_types([barcode_type])
    if method not in movie_colors.colors:
        available = ", ".join(movie_colors.methods) or "none"
        raise ValueError(f"No colors for method {method!r} (available: {available}).")
    if height is not None and height <= 0:
        raise ValueError("Height must be greater than 0.")

    # A list of per-frame colors, as returned by the extraction
    colors = list(movie_colors.colors[method])
    if barcode_type == "circular":
        barcode = generate_circular_barcode(colors, movie_colors.frame_width)
    else:
        barcode_height = height if height is not None else movie_colors.frame_height
        barcode = generate_barcode(colors, barcode_height, movie_colors.frame_count, movie_colors.target_frames)
    return barcode_to_image(barcode)


def render_barcodes(
    movie_colors: MovieColors,
    methods: Optional[Sequence[str]] = None,
    barcode_types: Sequence[str] = ("horizontal",),
    height: Optional[int] = None,
) -> Dict[Tuple[str, str], Image.Image]:
    """
    Render several barcodes from the same extracted colors.

    :param movie_colors: Colors returned by :func:`extract_movie_colors` or :func:`load_colors`.
    :param methods: Methods to render. Default is every method in movie_colors.
    :param barcode_types: Barcode types to render: "horizontal" and/or "circular".
    :param height: Height of horizontal barcodes. Default is the video height.
    :return: {(method, barcode_type): image}, methods first then types, in the requested order.
    """
    methods = list(dict.fromkeys(methods)) if methods is not None else movie_colors.methods
    barcode_types = _check_barcode_types(barcode_types)
    return {
        (method, barcode_type): render_barcode(movie_colors, method, barcode_type, height)
        for method in methods
        for barcode_type in barcode_types
    }


def save_colors(movie_colors: MovieColors, file_path: PathType) -> None:
    """
    Save extracted colors to a compressed ``.npz`` file (no pickled objects), to render barcodes again later.

    The file holds one array per method (``colors_<method>``), the requested frame indices (``frame_indices``) and
    the metadata as a JSON string (``metadata``): format version, video file name, frame count, size and rate,
    number of sampled frames, crop rectangle and linear flag.

    :param movie_colors: Colors to save.
    :param file_path: Path of the file; ``.npz`` is appended by NumPy if missing. Parent directories are created.
    """
    file_path = os.fspath(file_path)
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    metadata = {
        "version": COLORS_FILE_VERSION,
        "video_name": movie_colors.video_name,
        "methods": movie_colors.methods,
        "frame_count": movie_colors.frame_count,
        "frame_width": movie_colors.frame_width,
        "frame_height": movie_colors.frame_height,
        "target_frames": movie_colors.target_frames,
        "fps": movie_colors.fps,
        "crop_rect": list(movie_colors.crop_rect) if movie_colors.crop_rect is not None else None,
        "linear": movie_colors.linear,
    }
    arrays = {f"colors_{method}": colors for method, colors in movie_colors.colors.items()}
    np.savez_compressed(
        file_path,
        metadata=np.array(json.dumps(metadata)),
        frame_indices=np.asarray(movie_colors.frame_indices, dtype=np.int64),
        **arrays,
    )


def load_colors(file_path: PathType) -> MovieColors:
    """
    Load colors saved by :func:`save_colors`.

    :param file_path: Path of the ``.npz`` file.
    :return: The colors, ready for :func:`render_barcode`.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is not a colors file, or was written by a newer version.
    """
    file_path = os.fspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified colors file '{file_path}' does not exist.")
    try:
        with np.load(file_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            colors = {method: data[f"colors_{method}"] for method in metadata["methods"]}
            frame_indices = data["frame_indices"]
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"'{file_path}' is not a movie-barcodes colors file: {error}") from error
    if metadata.get("version", 0) > COLORS_FILE_VERSION:
        raise ValueError(
            f"'{file_path}' was written by a newer version of movie-barcodes (format {metadata['version']})."
        )
    crop_rect = metadata.get("crop_rect")
    return MovieColors(
        colors=colors,
        frame_count=int(metadata["frame_count"]),
        frame_width=int(metadata["frame_width"]),
        frame_height=int(metadata["frame_height"]),
        target_frames=int(metadata["target_frames"]),
        frame_indices=frame_indices,
        fps=float(metadata.get("fps", 0.0)),
        crop_rect=tuple(crop_rect) if crop_rect is not None else None,
        linear=bool(metadata.get("linear", False)),
        video_name=metadata.get("video_name", ""),
    )


def is_colors_file(path: PathType) -> bool:
    """
    Whether a path names a colors file (by its extension), rather than a video.
    """
    return os.fspath(path).lower().endswith(COLORS_FILE_EXTENSION)


@dataclass
class BarcodeResult:
    """
    Outcome of one input of :func:`iter_barcodes`.

    :ivar path: The input path (video or colors file).
    :ivar status: "ok", "failed" or "skipped".
    :ivar barcodes: {(method, barcode_type): image} when status is "ok".
    :ivar colors: The extracted (or loaded) colors when status is "ok", e.g. to save them with :func:`save_colors`.
    :ivar error: The exception raised when status is "failed".
    :ivar duration: Processing time in seconds.
    """

    path: str
    status: str
    barcodes: Dict[Tuple[str, str], Image.Image] = field(default_factory=dict)
    colors: Optional[MovieColors] = None
    error: Optional[Exception] = None
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        """
        Whether the barcodes were generated.
        """
        return self.status == "ok"


def iter_barcodes(
    paths: Iterable[PathType],
    methods: Optional[Sequence[str]] = None,
    barcode_types: Sequence[str] = ("horizontal",),
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
    skip: Optional[Callable[[str], bool]] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
) -> Iterator[BarcodeResult]:
    """
    Generate the barcodes of several movies, one after another, yielding one result per input.

    Each video is decoded once for all the requested methods and barcode types, using all the workers. An input
    that fails (missing, unreadable, too short...) gives a "failed" result holding the exception, and the next
    inputs are still processed. Inputs ending in ``.npz`` are colors files (see :func:`save_colors`): they are
    rendered without decoding; width, workers, crop_borders and linear don't apply to them.

    Example::

        for result in iter_barcodes(["a.mp4", "b.mkv"], methods=["avg"], barcode_types=["horizontal", "circular"]):
            if result.ok:
                for (method, barcode_type), image in result.barcodes.items():
                    image.save(f"{result.path}_{method}_{barcode_type}.png")
            else:
                print(result.path, result.error)

    :param paths: Video files and/or colors files. Use :func:`movie_barcodes.utility.find_videos` to expand
        directories.
    :param methods: Color extraction methods. Default is "avg" for videos, every stored method for colors files.
    :param barcode_types: Barcode types to render: "horizontal" and/or "circular".
    :param width: Number of sampled frames (videos only). Default is min(video width, number of frames).
    :param height: Height of horizontal barcodes. Default is the video height.
    :param workers: Number of worker processes per video. Default is the number of CPU cores.
    :param crop_borders: Detect and ignore black borders (videos only).
    :param linear: Average avg and smoothed colors in linear light (videos only).
    :param skip: Optional predicate called with each path: True gives a "skipped" result without processing it.
    :param on_progress: Called with (frames processed, frames to sample) of the video being decoded, which is the
        input after the last yielded result. Not called for colors files.
    :param cancel: Once set, the current video stops decoding (see :func:`extract_movie_colors`) and
        :class:`Cancelled` is raised instead of yielding more results. It is not reported as a failed input.
    :return: An iterator of :class:`BarcodeResult`, in input order.
    :raises ValueError: If a method or barcode type is invalid (before processing any input).
    :raises Cancelled: If cancel is set.
    """
    if methods is not None:
        methods = _check_methods(methods)
    barcode_types = _check_barcode_types(barcode_types)

    for path in paths:
        path = os.fspath(path)
        check_cancel(cancel)
        if skip is not None and skip(path):
            yield BarcodeResult(path, "skipped")
            continue
        start_time = time.time()
        try:
            if is_colors_file(path):
                movie_colors = load_colors(path)
            else:
                movie_colors = extract_movie_colors(
                    path,
                    methods if methods is not None else ["avg"],
                    width=width,
                    height=height,
                    workers=workers,
                    crop_borders=crop_borders,
                    linear=linear,
                    on_progress=on_progress,
                    cancel=cancel,
                )
            barcodes = render_barcodes(movie_colors, methods, barcode_types, height)
        except Cancelled:
            raise
        except Exception as error:  # One failing movie must not stop the batch
            yield BarcodeResult(path, "failed", error=error, duration=time.time() - start_time)
            continue
        yield BarcodeResult(path, "ok", barcodes, movie_colors, duration=time.time() - start_time)


def create_barcodes(
    video_path: PathType,
    methods: Sequence[str] = METHODS,
    barcode_type: str = "horizontal",
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
) -> Dict[str, Image.Image]:
    """
    Generate one barcode per color extraction method, decoding the video only once.

    :param video_path: Path to the video file.
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
    :param on_progress: Called with (frames processed, frames to sample) as frames are decoded.
    :param cancel: Once set, decoding stops and :class:`Cancelled` is raised (see :func:`extract_movie_colors`).
    :return: {method: barcode image}, keyed by the requested method names (also with ``linear``), in the order of
        ``methods``. Horizontal barcodes are RGB images, circular
        barcodes are RGBA images with a transparent background.
    :raises FileNotFoundError: If the video file does not exist.
    :raises ValueError: If an argument is invalid, the video cannot be opened or has too few frames.
    :raises Cancelled: If cancel is set.
    """
    methods = _check_methods(methods)
    _check_barcode_types([barcode_type])
    movie_colors = extract_movie_colors(
        video_path,
        methods,
        width=width,
        height=height,
        workers=workers,
        crop_borders=crop_borders,
        linear=linear,
        on_progress=on_progress,
        cancel=cancel,
    )
    return {method: render_barcode(movie_colors, method, barcode_type, height) for method in methods}


def create_barcode(
    video_path: PathType,
    method: str = "avg",
    barcode_type: str = "horizontal",
    width: Optional[int] = None,
    height: Optional[int] = None,
    workers: Optional[int] = None,
    crop_borders: bool = False,
    linear: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
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
    :param on_progress: Called with (frames processed, frames to sample) as frames are decoded.
    :param cancel: Once set, decoding stops and :class:`Cancelled` is raised (see :func:`extract_movie_colors`).
    :return: The barcode: an RGB image for horizontal barcodes, an RGBA image with a transparent background for
        circular barcodes.
    :raises FileNotFoundError: If the video file does not exist.
    :raises ValueError: If an argument is invalid, the video cannot be opened or has too few frames.
    :raises Cancelled: If cancel is set.
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
        on_progress=on_progress,
        cancel=cancel,
    )
    return barcodes[method]
