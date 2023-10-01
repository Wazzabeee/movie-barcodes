from multiprocessing import Pool
from typing import Callable, List
from utility import format_time

import cv2
import time
import numpy as np
from tqdm import tqdm


def load_video(video_path: str) -> tuple:
    """
    Load a video file and return its properties.

    :param str video_path: The path to the video file.
    :return: Tuple containing the video capture object, frame count, frame width, and frame height.
    """
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return video, frame_count, frame_width, frame_height


def process_frames(start_frame: int, end_frame: int, video_path: str, color_extractor: Callable) -> list:
    """
    Process frames in a range to extract dominant colors.

    :param int start_frame: The index of the first frame to process.
    :param int end_frame: The index of the last frame to process.
    :param str video_path: The path to the video file.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :return: List of dominant colors for the frames in the range.
    """
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
    """
    Extract dominant colors from frames in a video file using parallel processing.

    :param str video_path: The path to the video file.
    :param int frame_count: The total number of frames in the video.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :param int workers: Number of parallel workers.
    :return: List of dominant colors for the frames in the video.
    """
    frames_per_worker = frame_count // workers

    with Pool(workers) as pool:
        args = [(i * frames_per_worker, (i + 1) * frames_per_worker - 1, video_path, color_extractor) for i in range(workers)]
        if frame_count % workers != 0:
            args[-1] = (args[-1][0], frame_count - 1, video_path, color_extractor)

        results = pool.starmap(process_frames, args)

    # Concatenate results from all workers
    final_colors = [color for colors in results for color in colors]

    return final_colors


def extract_colors(video: cv2.VideoCapture, frame_count: int, target_frames: int, color_extractor: Callable) -> List:
    """
    Extracts dominant colors from frames in a video file sequentially.

    :param cv2.VideoCapture video: The video capture object.
    :param int frame_count: The total number of frames in the video.
    :param int target_frames: The total number of frames to sample.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :return: List of dominant colors from the sampled frames.
    """
    # Calculate frame_skip based on target_frames
    frame_skip = frame_count // target_frames if target_frames else 1

    # Set a default value for target_frames if it's None
    if target_frames is None:
        target_frames = frame_count

    colors = []

    for _ in tqdm(range(0, target_frames), desc='Processing frames'):
        for _ in range(frame_skip - 1):
            video.grab()  # Skip frames

        ret, frame = video.read()
        if ret:
            dominant_color = color_extractor(frame)
            colors.append(dominant_color)

    video.release()

    return colors


def crop_black_borders(frame: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Crop out black borders from a frame.

    :param np.ndarray frame: Input frame.
    :param int threshold: Threshold below which a pixel is considered 'black'.
    :return np.ndarray: Cropped frame.
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
