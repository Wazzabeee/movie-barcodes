import argparse
import logging
import time

from typing import Callable
from os import cpu_count, path

from .barcode_generation import generate_circular_barcode, generate_barcode

from .utility import (
    save_barcode_image,
    get_dominant_color_function,
    format_time,
    get_video_properties,
    validate_args,
)
from .video_processing import load_video, extract_colors, parallel_extract_colors

MAX_PROCESSES = cpu_count() or 1
MIN_FRAME_COUNT = 2


def generate_and_save_barcode(args: argparse.Namespace, dominant_color_function: Callable, method: str) -> None:
    """
    Generate and save the barcode image based on the specified method.

    :param args: argparse.Namespace object containing the command-line arguments
    :param dominant_color_function: The function to extract the dominant color from a frame
    :param method: The method used to extract the dominant color
    :return: None
    """
    start_time = time.time()

    # Get Video Properties
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)
    _, _, video_duration, video_size = get_video_properties(video, args)

    # If the user specifies the 'workers' argument
    if args.workers is not None:
        if args.workers == 1:
            # If the user explicitly sets 'workers' to 1, use sequential processing
            colors = extract_colors(
                args.input_video_path,
                0,
                frame_count - 1,
                dominant_color_function,
                args.width,
            )
        else:
            # Perform parallel processing with the user-specified number of workers
            colors = parallel_extract_colors(
                args.input_video_path,
                frame_count,
                dominant_color_function,
                args.workers,
                args.width,
            )
    else:
        # If 'workers' is not specified, use the maximum number of available CPU cores
        colors = parallel_extract_colors(
            args.input_video_path,
            frame_count,
            dominant_color_function,
            MAX_PROCESSES,
            args.width,
        )

    # Generate the appropriate type of barcode
    if args.barcode_type == "circular":
        # Assuming image width = video frame width for circular barcodes
        barcode = generate_circular_barcode(colors, frame_width)
    else:
        barcode = generate_barcode(colors, frame_height, frame_count, args.width)

    base_name = path.basename(args.input_video_path)
    file_name_without_extension = path.splitext(base_name)[0]
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

    video.release()


def main() -> None:
    """
    Main function to generate a barcode from a video file.
    """
    # Logging configuration
    logging.basicConfig(
        filename=path.join("..", "logs.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    header_msg = "=" * 40 + " NEW RUN " + "=" * 40
    logging.info("\n%s\n", header_msg)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate a color barcode from a video file.")
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument(
        "--destination_path",
        type=str,
        nargs="?",
        help="Path to save the output image. If not provided, the image will be saved in a default location.",
        default=None,
    )
    parser.add_argument(
        "--barcode_type",
        choices=["horizontal", "circular"],
        default="horizontal",
        help="Type of barcode to generate: horizontal or circular. Default is horizontal.",
    )
    parser.add_argument(
        "--method",
        choices=["avg", "kmeans", "hsv", "bgr"],
        default="avg",
        help="Method to extract dominant color: avg (average), kmeans (K-Means clustering), hsv (HSV histogram), "
        "or bgr (BGR histogram). Default is avg.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing. Default behavior uses all available CPU cores. Setting this "
        "to 1 will use sequential processing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of the output image. If not provided, the width will be the same as the video",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        nargs="?",
        help="Custom name for the output barcode image. If not provided, a name will be automatically generated.",
        default=None,
    )
    parser.add_argument(
        "--all_methods",
        action="store_true",
        help="If provided, all methods to extract dominant color will be used to create barcodes. Overrides --method "
        "argument.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate and process video file
    _, frame_count, _, _ = load_video(args.input_video_path)
    validate_args(args, frame_count, MAX_PROCESSES, MIN_FRAME_COUNT)

    # Choose the method to generate barcode
    methods = ["avg", "hsv", "bgr", "kmeans"]
    if args.all_methods:
        for method in methods:
            dominant_color_function = get_dominant_color_function(method)
            generate_and_save_barcode(args, dominant_color_function, method)
    else:
        dominant_color_function = get_dominant_color_function(args.method)
        generate_and_save_barcode(args, dominant_color_function, args.method)


if __name__ == "__main__":
    main()
