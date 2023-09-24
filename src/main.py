import argparse
import os

from barcode_generation import generate_circular_barcode, generate_barcode
from color_extraction import get_dominant_color_kmeans, get_dominant_color_avg
from utility import save_barcode_image
from video_processing import load_video, extract_colors

MAX_PROCESSES = 2  # You can adjust this based on your machine's capacity.


def main(args):
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)

    if args.method == "kmeans":
        color_extractor = get_dominant_color_kmeans
    else:
        color_extractor = get_dominant_color_avg

    colors = extract_colors(video, frame_count, color_extractor, args.workers, args.frame_skip)

    # Adjust the frame_count for the barcode width
    adjusted_frame_count = frame_count // args.frame_skip

    # Generate the appropriate type of barcode
    if args.barcode_type == "circular":
        barcode = generate_circular_barcode(colors,
                                            frame_width)  # Assuming image width = video frame width for circular
        # barcodes
    else:
        barcode = generate_barcode(colors, frame_height, adjusted_frame_count)

    base_name = os.path.basename(args.input_video_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    save_barcode_image(barcode, file_name_without_extension, args)

    video.release()


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
                        choices=['avg', 'kmeans'],
                        default='avg',
                        help='Method to extract dominant color: avg (average) or kmeans (K-Means clustering). Default '
                             'is avg.')
    parser.add_argument('--workers',
                        type=int,
                        default=None,
                        help='Number of workers for parallel processing. If not provided, processing will be '
                             'sequential.')
    parser.add_argument('--frame_skip',
                        type=int,
                        default=1,
                        help='Number of frames to skip between processing. Default is 1 (process every frame).')

    args = parser.parse_args()
    main(args)
