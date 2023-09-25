from multiprocessing import Pool
from typing import Callable
from utility import format_time

import cv2
import time
import numpy as np
from tqdm import tqdm

CHUNK_SIZE = 10  # Adjust this based on the desired chunk size


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


def process_frames(start_frame: int, end_frame: int, video_path: str, color_extractor: Callable) -> list:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    colors = []
    start_time = time.time()
    print_frequency = 1000  # Adjust this value to your liking

    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        color = color_extractor(frame)
        colors.append(color)

        if (i - start_frame + 1) % print_frequency == 0:
            elapsed_time = time.time() - start_time
            frames_left = end_frame - i
            time_per_frame = elapsed_time / (i - start_frame + 1)
            estimated_time_left = frames_left * time_per_frame
            progress_percentage = ((i - start_frame + 1) / (end_frame - start_frame + 1)) * 100
            formatted_time_left = format_time(estimated_time_left)
            print(f"Progress: {progress_percentage:.2f}% - Estimated time remaining: {formatted_time_left}")

    cap.release()

    return colors


def parallel_extract_colors(video_path: str, frame_count: int, color_extractor, workers=1) -> list:
    frames_per_worker = frame_count // workers

    with Pool(workers) as pool:
        args = [(i * frames_per_worker, (i + 1) * frames_per_worker - 1, video_path, color_extractor) for i in range(workers)]
        if frame_count % workers != 0:
            args[-1] = (args[-1][0], frame_count - 1, video_path, color_extractor)

        results = pool.starmap(process_frames, args)

    # Concatenate results from all workers
    final_colors = [color for colors in results for color in colors]

    return final_colors


def extract_colors(video: cv2.VideoCapture, frame_count: int, color_extractor: Callable, frame_skip: int = 1) -> list:
    colors = []

    for _ in tqdm(range(0, frame_count, frame_skip)):
        ret, frame = video.read()
        if ret:
            # frame = crop_black_borders(frame)
            dominant_color = color_extractor(frame)
            colors.append(dominant_color)

    return colors


def crop_black_borders(frame: np.ndarray, threshold=30) -> np.ndarray:
    """
    Crop out black borders from a frame.

    :param frame: Input frame.
    :param threshold: Threshold below which a pixel is considered 'black'.
    :return: Cropped frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    cropped_frame = frame[y:y + h, x:x + w]

    return cropped_frame
