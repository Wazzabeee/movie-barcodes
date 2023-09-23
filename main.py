import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Callable, Optional
from tqdm import tqdm

MAX_PROCESSES = 2  # You can adjust this based on your machine's capacity.


# Step 2: Load the video
def load_video(video_path: str) -> tuple:
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return video, frame_count, frame_width, frame_height


def process_frame_chunk(chunk_frames, color_extractor):
    colors = []
    for frame in chunk_frames:
        dominant = color_extractor(frame)
        colors.append(dominant)
    return colors


# Step 3: Process each frame
def get_dominant_color_avg(frame: np.ndarray) -> np.ndarray:
    """
    Use simple average to find the most dominant color.
    """
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def get_dominant_color_kmeans(frame: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Use KMeans clustering to find the most dominant color.
    """
    # Reshape the frame to be a list of pixels
    pixels = frame.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # If we are looking only for the most dominant color
    if k == 1:
        return cluster_centers[0]
    else:
        # If we wanted to retrieve more than one color
        # We'd take the center of the largest cluster, assuming it's the dominant color
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_index = labels[np.argmax(counts)]
        return cluster_centers[dominant_index]


def extract_colors(video: cv2.VideoCapture, frame_count: int, color_extractor: Callable,
                   workers: Optional[int] = None, frame_skip: int = 1) -> list:
    colors = []

    if workers:  # Use parallel processing
        CHUNK_SIZE = 10  # Adjust this based on the desired chunk size

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []

            # Load and submit chunks for processing
            for _ in range(0, frame_count, CHUNK_SIZE * frame_skip):
                chunk_frames = []
                for _ in range(CHUNK_SIZE):
                    ret, frame = video.read()
                    if not ret:
                        break

                    # Only process the first frame in the chunk and skip the rest based on frame_skip
                    if _ % frame_skip == 0:
                        chunk_frames.append(frame)
                    else:
                        continue

                if chunk_frames:
                    futures.append(executor.submit(process_frame_chunk, chunk_frames, color_extractor))

            # Gather the results
            for future in tqdm(as_completed(futures), total=len(futures)):
                colors.extend(future.result())

    else:  # Use sequential processing
        for _ in tqdm(range(0, frame_count, frame_skip)):
            ret, frame = video.read()
            if ret:
                dominant_color = color_extractor(frame)
                colors.append(dominant_color)

    return colors


# Step 4: Create the barcode
def generate_barcode(colors: list, frame_height: int, frame_count: int) -> np.ndarray:
    barcode = np.zeros((frame_height, frame_count, 3))
    for i, color in enumerate(colors):
        barcode[:, i] = color
    return barcode


# Step 5: Save the barcode
def save_barcode_image(barcode: np.ndarray, output_path: str = "barcode.png") -> None:
    image = Image.fromarray(barcode.astype(np.uint8))
    image.save(output_path)


def ensure_directory(directory_name: str) -> None:
    """Ensure that a given directory exists. If not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def main(args):
    video, frame_count, frame_width, frame_height = load_video(args.input_video_path)

    if args.method == "kmeans":
        color_extractor = get_dominant_color_kmeans
    else:
        color_extractor = get_dominant_color_avg

    colors = extract_colors(video, frame_count, color_extractor, args.workers, args.frame_skip)

    # Adjust the frame_count for the barcode width
    adjusted_frame_count = frame_count // args.frame_skip

    barcode = generate_barcode(colors, frame_height, adjusted_frame_count)

    # If destination_path isn't specified, construct one based on the video's name
    if not args.destination_path:
        base_name = os.path.basename(args.input_video_path)
        file_name_without_extension = os.path.splitext(base_name)[0]
        ensure_directory("barcodes")
        destination_path = os.path.join("barcodes", f"barcode_{file_name_without_extension}.png")
    else:
        destination_path = args.destination_path

    save_barcode_image(barcode, destination_path)
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
