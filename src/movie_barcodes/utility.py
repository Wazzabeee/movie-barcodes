import logging
import os
from os import path, access, W_OK, makedirs
from typing import Callable, Iterable, List, Optional
import cv2
import numpy as np
from PIL import Image


from .color_extraction import (
    get_dominant_color_mean,
    get_dominant_color_kmeans,
    get_dominant_color_hsv,
    get_dominant_color_bgr,
    get_smoothed_frame,
    get_dominant_color_mean_linear,
    get_smoothed_frame_linear,
)

# Methods that average colors, and can therefore average them in linear light (--linear)
LINEAR_METHODS = ("avg", "smoothed")

# Video file extensions accepted as input, and looked for when an input is a directory
VIDEO_EXTENSIONS = (
    ".mp4",
    ".m4v",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".mts",
    ".ogv",
    ".3gp",
)


def validate_inputs(
    video_path: str,
    frame_count: int,
    min_frame_count: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Validate the input video and the requested barcode dimensions.

    :param str video_path: The path to the input video file.
    :param int frame_count: The number of frames in the video.
    :param int min_frame_count: The minimum number of frames required in the video.
    :param Optional[int] width: The requested number of sampled frames (barcode width), or None for the default.
    :param Optional[int] height: The requested barcode height, or None for the default.
    :return: None
    :raises FileNotFoundError: If the input video file does not exist.
    :raises ValueError: If the video file has an invalid extension, the width or height is invalid, or the frame
        count is too low.
    """
    # Check if input video file exists
    if not path.exists(video_path):
        raise FileNotFoundError(f"The specified input video file '{video_path}' does not exist.")

    if path.splitext(video_path)[1].lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"The specified video file must have a valid video extension ({', '.join(VIDEO_EXTENSIONS)}).")

    if width is not None:
        if width <= 0:
            raise ValueError("Width must be greater than 0.")
        if width > frame_count:
            raise ValueError("Width must be less than or equal to the number of frames.")

    if height is not None:
        if height <= 0:
            raise ValueError("Height must be greater than 0.")

    if frame_count < min_frame_count:
        raise ValueError(f"The video must have at least {min_frame_count} frames.")


def validate_destination_path(destination_path: Optional[str]) -> None:
    """
    Check that an output image path is writable.

    Missing parent directories are created at save time, so the closest existing ancestor is checked (paths are
    relative to the current working directory).

    :param Optional[str] destination_path: The output image path, or None for the default location.
    :return: None
    :raises PermissionError: If the destination path is not writable.
    """
    if destination_path is None:
        return
    destination_dir = path.dirname(path.abspath(destination_path))
    while not path.exists(destination_dir) and path.dirname(destination_dir) != destination_dir:
        destination_dir = path.dirname(destination_dir)
    if not access(destination_dir, W_OK):
        raise PermissionError(f"The specified destination path '{destination_path}' is not writable.")


def resolve_workers(workers: Optional[int], max_processes: int) -> Optional[int]:
    """
    Validate the requested number of workers and cap it to the number of available CPU cores.

    :param Optional[int] workers: The requested number of workers, or None for the default (all CPU cores).
    :param int max_processes: The maximum number of processes to use (number of available CPU cores).
    :return: The number of workers to use (None stays None). A warning is logged when it is capped.
    :raises ValueError: If the number of workers is lower than 1.
    """
    if workers is None:
        return None
    if workers < 1:
        raise ValueError("The number of workers must be greater than or equal to 1.")
    if workers > max_processes:
        logging.warning(
            "Requested %d workers but only %d CPU cores are available; using %d.",
            workers,
            max_processes,
            max_processes,
        )
        return max_processes
    return workers


def get_dominant_color_function(method: str, linear: bool = False) -> Callable:
    """
    Returns the appropriate function to get the dominant color based on the specified method.

    :param str method: The method to use for color extraction ('avg', 'kmeans', 'hsv', 'bgr' or 'smoothed').
    :param bool linear: Average colors in linear light. Only applies to the methods in LINEAR_METHODS ('avg' and
        'smoothed'); the other methods are unaffected.
    :return: Function to get the dominant color.
    :raises ValueError: If the method is invalid.
    """
    if linear and method == "avg":
        return get_dominant_color_mean_linear
    if linear and method == "smoothed":
        return get_smoothed_frame_linear
    if method == "avg":
        return get_dominant_color_mean
    if method == "kmeans":
        return get_dominant_color_kmeans
    if method == "hsv":
        return get_dominant_color_hsv
    if method == "bgr":
        return get_dominant_color_bgr
    if method == "smoothed":
        return get_smoothed_frame

    raise ValueError(f"Invalid method: {method}")


def format_time(seconds: float) -> str:
    """
    Formats time in seconds to a string representation.

    :param int seconds: The time in seconds.
    :return: Formatted time string.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    return f"{int(minutes)}m {int(seconds)}s"


def get_video_properties(video: cv2.VideoCapture, video_path: str) -> tuple:
    """
    Extracts and returns various properties of a video.

    :param cv2.VideoCapture video: The video capture object.
    :param str video_path: The path to the video file.
    :return: Tuple containing total frames, FPS, video duration in seconds, and video size in bytes.
    """
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0  # in seconds
    video_size = path.getsize(video_path)  # in bytes

    return total_frames, fps, video_duration, video_size


def barcode_to_image(barcode: np.ndarray) -> Image.Image:
    """
    Convert a generated barcode (BGR, or BGRA for circular barcodes) to a PIL image (RGB or RGBA).

    :param np.ndarray barcode: The barcode image as a NumPy array.
    :return: The barcode as an RGB or RGBA PIL image.
    """
    if barcode.shape[2] == 4:  # If the image has an alpha channel (BGRA)
        return Image.fromarray(cv2.cvtColor(barcode, cv2.COLOR_BGRA2RGBA), "RGBA")
    return Image.fromarray(cv2.cvtColor(barcode, cv2.COLOR_BGR2RGB), "RGB")


def get_output_path(
    base_name: str,
    method: str,
    barcode_type: str,
    workers: Optional[int] = None,
    destination_path: Optional[str] = None,
    output_name: Optional[str] = None,
    method_suffix: bool = False,
    type_suffix: bool = False,
    output_dir: str = "barcodes",
) -> str:
    """
    Path where the command line saves a barcode image.

    Paths are resolved relative to the current working directory (not the package install location, which would
    be site-packages once installed with pip).

    :param str base_name: The video file name without extension.
    :param str method: The method used for color extraction.
    :param str barcode_type: The type of barcode ("horizontal" or "circular").
    :param Optional[int] workers: The number of workers requested by the user, or None if not specified.
    :param Optional[str] destination_path: The output path chosen by the user, if any.
    :param Optional[str] output_name: The output file name (without extension) chosen by the user, if any.
    :param bool method_suffix: Append the method to destination_path or output_name (before the extension), so that
        the images of several methods saved in one run don't overwrite each other. The generated name always
        contains the method.
    :param bool type_suffix: Same for the barcode type, when several types are saved in one run.
    :param str output_dir: Directory of the generated and output_name paths. Default is ./barcodes.
    :return: destination_path if given, else <output_dir>/<output_name>.png, else
        <output_dir>/<base_name>_<method>_<barcode_type>[_workers_<workers>].png; with method_suffix and/or
        type_suffix, <destination_path stem>[_<method>][_<barcode_type>]<extension> or
        <output_dir>/<output_name>[_<method>][_<barcode_type>].png
    """
    suffix = "".join(f"_{part}" for part, enabled in ((method, method_suffix), (barcode_type, type_suffix)) if enabled)
    if destination_path:
        if suffix:
            root, extension = path.splitext(destination_path)
            return f"{root}{suffix}{extension}"
        return destination_path
    if output_name:
        destination_name = output_name + suffix + ".png"
    else:
        filename_parts = [base_name, method, barcode_type]
        if workers:
            filename_parts.append(f"workers_{str(workers)}")
        destination_name = "_".join(filename_parts) + ".png"
    return path.join(output_dir, destination_name)


def save_barcode_image(image: Image.Image, destination_path: str) -> None:
    """
    Saves a barcode image to a file, creating the parent directories if needed.

    :param Image.Image image: The barcode image.
    :param str destination_path: The path of the image file.
    """
    parent_dir = path.dirname(destination_path)
    if parent_dir:
        ensure_directory(parent_dir)
    image.save(destination_path)
    logging.info("File saved at '%s'", path.abspath(destination_path))


def ensure_directory(directory_name: str) -> None:
    """
    Ensures that a directory exists. Creates it if it doesn't.

    :param str directory_name: The name of the directory to ensure.
    """
    makedirs(directory_name, exist_ok=True)


def read_path_list(list_file: str) -> List[str]:
    """
    Read a list of input paths: one per line, blank lines and lines starting with # are ignored.

    Relative paths are resolved relative to the directory of the list file.

    :param str list_file: Path of the list file (UTF-8 text).
    :return: The paths, in order.
    :raises OSError: If the file cannot be read.
    """
    base_dir = path.dirname(list_file)
    paths = []
    with open(list_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(path.normpath(path.join(base_dir, path.expanduser(line))))
    return paths


def find_videos(paths: Iterable[str], recursive: bool = False) -> List[str]:
    """
    Expand input paths into the list of files to process.

    Directories are replaced by the video files they contain (by extension, see VIDEO_EXTENSIONS), sorted by name,
    including their subdirectories with recursive. Other paths are kept as given, whatever their extension or even
    if they don't exist, so that errors are reported for them. Duplicates are removed.

    :param paths: Files and/or directories.
    :param bool recursive: Also look for videos in the subdirectories of directories.
    :return: The files, in order.
    """
    files: List[str] = []
    for input_path in paths:
        input_path = os.fspath(input_path)
        if not path.isdir(input_path):
            files.append(input_path)
            continue
        found = []
        for directory, subdirectories, file_names in os.walk(input_path):
            subdirectories[:] = sorted(d for d in subdirectories if not d.startswith(".")) if recursive else []
            for file_name in file_names:
                if not file_name.startswith(".") and path.splitext(file_name)[1].lower() in VIDEO_EXTENSIONS:
                    found.append(path.join(directory, file_name))
        files.extend(sorted(found, key=lambda file_path: (path.dirname(file_path), file_path.lower())))
    unique = {}
    for file_path in files:
        unique.setdefault(path.normcase(path.abspath(file_path)), file_path)
    return list(unique.values())
