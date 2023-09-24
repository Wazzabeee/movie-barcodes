import argparse
import os

from barcode_generation import generate_circular_barcode, generate_barcode

from utility import save_barcode_image, get_dominant_color_function
from video_processing import load_video, extract_colors

MAX_PROCESSES = 4  # You can adjust this based on your machine's capacity.


def generate_and_save_barcode(args, dominant_color_function, method: str):
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)

    # Adjust the frame_count for the barcode width
    adjusted_frame_count = frame_count // args.frame_skip
    colors = extract_colors(args.input_video_path, frame_count, dominant_color_function, args.workers, args.frame_skip)
    # colors = extract_colors(video, frame_count, dominant_color_function, args.workers, args.frame_skip)

    # Generate the appropriate type of barcode
    if args.barcode_type == "circular":
        barcode = generate_circular_barcode(colors,
                                            frame_width)  # Assuming image width = video frame width for circular
        # barcodes
    else:
        barcode = generate_barcode(colors, frame_height, adjusted_frame_count)

    base_name = os.path.basename(args.input_video_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    save_barcode_image(barcode, file_name_without_extension, args, method)

    video.release()


def main(args):
    # Get a list of all available methods
    methods = ['avg', 'hsv', 'bgr']

    # Check if all_methods flag is set
    if args.all_methods:
        for method in methods:
            # Generate barcodes for each method
            dominant_color_function = get_dominant_color_function(method)
            generate_and_save_barcode(args, dominant_color_function, method)
    else:
        # Use the specified method to generate barcode
        dominant_color_function = get_dominant_color_function(args.method)
        generate_and_save_barcode(args, dominant_color_function, args.method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a color barcode from a video file.')
    parser.add_argument('input_video_path',
                        type=str,
                        help='Path to the video file.')
    parser.add_argument('--destination_path',
                        type=str,
                        nargs='?',
                        help='Path to save the output image. If not provided, the image will be saved in a default '
                             'location.',
                        default=None)
    parser.add_argument('--barcode_type',
                        choices=['horizontal', 'circular'],
                        default='horizontal',
                        help='Type of barcode to generate: horizontal or circular. Default is horizontal.')
    parser.add_argument('--method',
                        choices=['avg', 'kmeans', 'hsv', 'bgr'],
                        default='avg',
                        help='Method to extract dominant color: avg (average), kmeans (K-Means clustering), hsv (HSV '
                             'histogram), or bgr (BGR histogram). Default is avg.')
    parser.add_argument('--workers',
                        type=int,
                        default=None,
                        help='Number of workers for parallel processing. If not provided, processing will be '
                             'sequential.')
    parser.add_argument('--frame_skip',
                        type=int,
                        default=1,
                        help='Number of frames to skip between processing. Default is 1 (process every frame).')
    parser.add_argument('--all_methods',
                        type=bool,
                        default=False,
                        help='If provided, all methods to extract dominant color will be used to create barcodes. '
                             'Overrides --method argument.')
    args = parser.parse_args()
    main(args)
