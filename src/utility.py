import argparse
from os import path, access, W_OK, makedirs
from typing import Callable
import cv2
import numpy as np
from PIL import Image


from .color_extraction import (
    get_dominant_color_mean,
    get_dominant_color_kmeans,
    get_dominant_color_hsv,
    get_dominant_color_bgr,
)


def validate_args(args: argparse.Namespace, frame_count: int, MAX_PROCESSES: int, MIN_FRAME_COUNT: int) -> None:
    """
    Validate command-line arguments for logical errors.

    :param argparse.Namespace args: The command-line arguments.
    :param int frame_count: The number of frames in the video.
    :param int MAX_PROCESSES: The maximum number of processes to use.
    :param int MIN_FRAME_COUNT: The minimum number of frames required in the video.
    :return: None
    :raises FileNotFoundError: If the input video file does not exist.
    :raises ValueError: If the video file has an invalid extension, the destination path is not writable, the number of
        workers is invalid, the width is invalid, the frame count is invalid, or the method is invalid.
    :raises PermissionError: If the destination path is not writable.
    """
    # Check if input video file exists
    if not path.exists(args.input_video_path):
        raise FileNotFoundError(f"The specified input video file '{args.input_video_path}' does not exist.")

    valid_extensions = [".mp4", ".webm"]
    if path.splitext(args.input_video_path)[1].lower() not in valid_extensions:
        raise ValueError("The specified video file must have a valid video extension (e.g., .mp4).")

    # Check if the destination path is writable
    if args.destination_path is not None:
        destination_dir = path.dirname(args.destination_path)
        if not access(destination_dir, W_OK):
            raise PermissionError(f"The specified destination path '{args.destination_path}' is not writable.")

    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("The number of workers must be greater than or equal to 1.")
        if args.workers > MAX_PROCESSES:
            raise ValueError(
                f"The number of workers specified ({args.workers}) exceeds "
                f"the number of available CPU cores ({MAX_PROCESSES})."
            )

    if args.width is not None:
        if args.width <= 0:
            raise ValueError("Width must be greater than 0.")
        if args.width > frame_count:
            raise ValueError(
                f"Specified width ({args.width}) cannot be greater than the number of frames ({frame_count}) in the "
                f"video."
            )

    if frame_count < MIN_FRAME_COUNT:
        raise ValueError(f"The video must have at least {MIN_FRAME_COUNT} frames.")

    if args.all_methods and args.method is not None:
        raise ValueError("The --all_methods flag cannot be used with the --method argument.")


def get_dominant_color_function(method: str) -> Callable:
    """
    Returns the appropriate function to get the dominant color based on the specified method.

    :param str method: The method to use for color extraction ('avg', 'kmeans', 'hsv', or 'bgr').
    :return: Function to get the dominant color.
    :raises ValueError: If the method is invalid.
    """
    if method == "avg":
        # return get_dominant_color_avg
        return get_dominant_color_mean
    if method == "kmeans":
        return get_dominant_color_kmeans
    if method == "hsv":
        return get_dominant_color_hsv
    if method == "bgr":
        return get_dominant_color_bgr

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


def get_video_properties(video: cv2.VideoCapture, args: argparse.Namespace) -> tuple:
    """
    Extracts and returns various properties of a video.

    :param cv2.VideoCapture video: The video capture object.
    :param args: Command line arguments.
    :return: Tuple containing total frames, FPS, video duration in seconds, and video size in bytes.
    """
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0  # in seconds
    video_size = path.getsize(args.input_video_path)  # in bytes

    return total_frames, fps, video_duration, video_size


def save_barcode_image(barcode: np.ndarray, base_name: str, args: argparse.Namespace, method: str) -> None:
    """
    Saves a generated barcode image to a file.

    :param np.ndarray barcode: The barcode image as a NumPy array.
    :param str base_name: The base name of the file to save.
    :param args: Command line arguments.
    :param str method: The method used for color extraction.
    """
    current_dir = path.dirname(path.abspath(__file__))
    project_root = path.dirname(current_dir)  # Go up one directory to get to the project root
    # If destination_path isn't specified, construct one based on the video's name
    if not args.destination_path:
        barcode_dir = path.join(project_root, "barcodes")
        ensure_directory(barcode_dir)

        # If an output_name is provided by the user
        if args.output_name:
            destination_name = args.output_name + ".png"
        else:
            filename_parts = [base_name, method, args.barcode_type]
            if args.workers:
                filename_parts.append(f"workers_{str(args.workers)}")
            destination_name = "_".join(filename_parts) + ".png"

        destination_path = path.join(barcode_dir, destination_name)
    else:
        # In case a destination_path is provided, consider appending the method
        # or managing as per your requirement
        destination_path = path.join(project_root, args.destination_path)

    if barcode.shape[2] == 4:  # If the image has an alpha channel (RGBA)
        image = Image.fromarray(barcode, "RGBA")
    else:  # If the image doesn't have an alpha channel (RGB)
        image = Image.fromarray(barcode, "RGB")

    image.save(destination_path)


def ensure_directory(directory_name: str) -> None:
    """
    Ensures that a directory exists. Creates it if it doesn't.

    :param str directory_name: The name of the directory to ensure.
    """
    if not path.exists(directory_name):
        makedirs(directory_name)
