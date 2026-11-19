import argparse
import logging
import sys
import time

from typing import Callable, Dict, Optional
from os import cpu_count, path

from .barcode_generation import generate_circular_barcode, generate_barcode

from .utility import (
    save_barcode_image,
    get_dominant_color_function,
    format_time,
    get_video_properties,
    validate_args,
)
from .video_processing import (
    load_video,
    extract_colors,
    parallel_extract_colors,
    detect_content_area,
    Rect,
)

MAX_PROCESSES = cpu_count() or 1
MIN_FRAME_COUNT = 2


def get_target_frames(args: argparse.Namespace, frame_count: int, frame_width: int) -> int:
    """
    Number of frames to sample from the video.

    For horizontal barcodes it is also the output width in pixels; for circular barcodes it is the number of rings.

    :param args: argparse.Namespace object containing the command-line arguments
    :param frame_count: The number of frames in the video
    :param frame_width: The width of the video frames in pixels
    :return: The number of frames to sample: --width if provided, else min(video width, frame count)
    """
    if args.width is not None:
        return args.width
    return max(1, min(frame_width, frame_count))


def generate_and_save_barcodes(
    args: argparse.Namespace, color_functions: Dict[str, Callable], crop_rect: Optional[Rect] = None
) -> None:
    """
    Generate and save one barcode image per color extraction method.

    The video is decoded once: every sampled frame goes through all the color extraction functions.

    :param args: argparse.Namespace object containing the command-line arguments
    :param color_functions: {method: function to extract the dominant color from a frame}, one barcode is saved
        per method
    :param crop_rect: (x, y, w, h) of the area to keep in each frame, or None to use the full frame
    :return: None
    """
    start_time = time.time()

    # Get Video Properties
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)
    _, _, video_duration, video_size = get_video_properties(video, args)
    video.release()

    # Number of frames to sample: one per output column (horizontal) or ring (circular).
    # Defaults to the video width, capped by the number of frames.
    target_frames = get_target_frames(args, frame_count, frame_width)

    # If the user specifies the 'workers' argument
    if args.workers is not None:
        if args.workers == 1:
            # If the user explicitly sets 'workers' to 1, use sequential processing
            colors_by_method = extract_colors(
                args.input_video_path,
                0,
                frame_count - 1,
                color_functions,
                target_frames,
                crop_rect=crop_rect,
            )
        else:
            # Perform parallel processing with the user-specified number of workers
            colors_by_method = parallel_extract_colors(
                args.input_video_path,
                frame_count,
                color_functions,
                args.workers,
                target_frames,
                crop_rect=crop_rect,
            )
    else:
        # If 'workers' is not specified, use the maximum number of available CPU cores
        colors_by_method = parallel_extract_colors(
            args.input_video_path,
            frame_count,
            color_functions,
            MAX_PROCESSES,
            target_frames,
            crop_rect=crop_rect,
        )

    base_name = path.basename(args.input_video_path)
    file_name_without_extension = path.splitext(base_name)[0]

    for method, colors in colors_by_method.items():
        # Generate the appropriate type of barcode
        if args.barcode_type == "circular":
            barcode = generate_circular_barcode(colors, frame_width)
        else:
            # Use the specified height if provided, otherwise use the video frame height
            barcode_height = args.height if args.height is not None else frame_height
            barcode = generate_barcode(colors, barcode_height, frame_count, target_frames)

        save_barcode_image(barcode, file_name_without_extension, args, method)

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    # Log the information
    logging.info("Processed File: %s", file_name_without_extension)
    logging.info("Number of Frames: %d", frame_count)
    logging.info("Video Duration: %s", format_time(video_duration))
    logging.info("Video Size: %.2f MB", video_size / (1024 * 1024))
    logging.info("Processing Time: %s", format_time(processing_time))


def main() -> None:
    """
    Main function to generate a barcode from a video file.
    """
    # Logging configuration
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    header_msg = "=" * 40 + " NEW RUN " + "=" * 40
    logging.info("%s", header_msg)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate a color barcode from a video file.")
    parser.add_argument("-i", "--input_video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument(
        "-d",
        "--destination_path",
        type=str,
        nargs="?",
        help="Path to save the output image. If not provided, the image will be saved in a default location.",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--barcode_type",
        choices=["horizontal", "circular"],
        default="horizontal",
        help="Type of barcode to generate: horizontal or circular. Default is horizontal.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["avg", "kmeans", "hsv", "bgr", "smoothed"],
        default="avg",
        help="Method to extract dominant color: avg (average), kmeans (K-Means clustering), hsv (HSV histogram), "
        ",bgr (BGR histogram) or smoothed version (averaging the colors with two-step resize). Default is avg.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing. Default behavior uses all available CPU cores. Setting this "
        "to 1 will use sequential processing. Values above the number of CPU cores are capped to it (with a warning).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Number of frames sampled: the output image width for horizontal barcodes, the number of rings for "
        "circular barcodes. If not provided, defaults to the video width (capped by the number of frames).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the output image. If not provided, the height will be the same as the video",
    )
    parser.add_argument(
        "-n",
        "--output_name",
        type=str,
        nargs="?",
        help="Custom name for the output barcode image. If not provided, a name will be automatically generated.",
        default=None,
    )
    parser.add_argument(
        "-a",
        "--all_methods",
        action="store_true",
        help="If provided, all methods to extract dominant color will be used to create barcodes. Overrides --method "
        "argument.",
    )
    parser.add_argument(
        "--crop-borders",
        action="store_true",
        help="If provided, detect black borders (letterbox / pillarbox) once and ignore them when extracting "
        "colors. Default is False.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Check the input file exists before trying to open it, to get a clear error message
    if not path.exists(args.input_video_path):
        raise FileNotFoundError(f"The specified input video file '{args.input_video_path}' does not exist.")

    # Validate and process video file
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)
    video.release()
    validate_args(args, frame_count, MAX_PROCESSES, MIN_FRAME_COUNT)

    # Detect black borders once for the whole run
    crop_rect = None
    if args.crop_borders:
        crop_rect = detect_content_area(args.input_video_path, frame_count)
        _, _, crop_width, crop_height = crop_rect
        if (crop_width, crop_height) == (frame_width, frame_height):
            logging.info("Cropping borders: no black borders detected")
            crop_rect = None
        else:
            logging.info("Cropping borders: %dx%d -> %dx%d", frame_width, frame_height, crop_width, crop_height)

    # Choose the method to generate barcode
    methods = ["avg", "hsv", "bgr", "kmeans", "smoothed"] if args.all_methods else [args.method]
    generate_and_save_barcodes(args, {method: get_dominant_color_function(method) for method in methods}, crop_rect)


if __name__ == "__main__":
    main()
