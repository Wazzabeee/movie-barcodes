import argparse
import logging
import sys
import time

from os import path

from .api import MAX_PROCESSES, METHODS, create_barcodes
from .utility import (
    LINEAR_METHODS,
    format_time,
    get_output_path,
    get_video_properties,
    resolve_workers,
    save_barcode_image,
    validate_destination_path,
)
from .video_processing import load_video


def generate_and_save_barcodes(args: argparse.Namespace) -> None:
    """
    Generate and save one barcode image per requested color extraction method.

    With --all_methods, the video is decoded once: every sampled frame goes through all the color extraction
    functions.

    :param args: argparse.Namespace object containing the command-line arguments
    :return: None
    """
    # Fail before decoding the video if the image cannot be saved
    validate_destination_path(args.destination_path)
    # Workers above the number of CPU cores are capped (with a warning); the file name mentions the number of
    # workers actually used, only when the user chose it
    workers = resolve_workers(args.workers, MAX_PROCESSES)

    start_time = time.time()
    methods = list(METHODS) if args.all_methods else [args.method]
    barcodes = create_barcodes(
        args.input_video_path,
        methods,
        barcode_type=args.barcode_type,
        width=args.width,
        height=args.height,
        workers=workers,
        crop_borders=args.crop_borders,
        linear=args.linear,
    )

    file_name_without_extension = path.splitext(path.basename(args.input_video_path))[0]
    for method, image in barcodes.items():
        # With --linear, avg and smoothed barcodes get a distinct name so they don't overwrite the gamma averaged ones
        output_method = f"{method}_linear" if args.linear and method in LINEAR_METHODS else method
        destination_path = get_output_path(
            file_name_without_extension,
            output_method,
            args.barcode_type,
            workers=workers,
            destination_path=args.destination_path,
            output_name=args.output_name,
            # With several methods, -n / -d would give every image the same path: append the method
            method_suffix=len(barcodes) > 1,
        )
        save_barcode_image(image, destination_path)

    processing_time = time.time() - start_time

    video, frame_count, _, _ = load_video(args.input_video_path)
    _, _, video_duration, video_size = get_video_properties(video, args.input_video_path)
    video.release()

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
        help="Path to save the output image. If not provided, the image will be saved in a default location. With "
        "--all_methods, the method is appended to the file name (e.g. out_avg.png, out_kmeans.png).",
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
        help="Custom name for the output barcode image. If not provided, a name will be automatically generated. "
        "With --all_methods, the method is appended (e.g. NAME_avg.png, NAME_kmeans.png).",
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

    parser.add_argument(
        "--linear",
        action="store_true",
        help="Average colors in linear light (perceptually correct, brighter on high-contrast scenes). Applies to "
        "the avg and smoothed methods only; their output files get a '_linear' suffix after the method name. "
        "Default is False.",
    )

    # Parse arguments
    args = parser.parse_args()

    generate_and_save_barcodes(args)


if __name__ == "__main__":
    main()
