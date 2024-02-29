from multiprocessing import Pool
from typing import Callable, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


def load_video(video_path: str) -> tuple:
    """
    Load a video file and return its properties.

    :param str video_path: The path to the video file.
    :return: Tuple containing the video capture object, frame count, frame width, and frame height.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        raise ValueError(f"The video file {video_path} has no frames.")

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"The video file {video_path} has invalid dimensions.")

    return video, frame_count, frame_width, frame_height


def parallel_extract_colors(
    video_path: str,
    frame_count: int,
    color_extractor: Callable,
    workers: int,
    target_frames: Optional[int] = None,
) -> list:
    """
    Extract dominant colors from frames in a video file using parallel processing.

    :param str video_path: The path to the video file.
    :param int frame_count: The total number of frames in the video.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :param int workers: Number of parallel workers.
    :param Optional[int] target_frames: The total number of frames to sample.
    :return: List of dominant colors for the frames in the video.
    """
    if target_frames is None:
        target_frames = frame_count

    frames_per_worker = frame_count // workers
    target_frames_per_worker = target_frames // workers

    with Pool(workers) as pool:
        args = [
            (
                video_path,
                i * frames_per_worker,
                (i + 1) * frames_per_worker - 1,
                color_extractor,
                target_frames_per_worker,
            )
            for i in range(workers)
        ]

        if frame_count % workers != 0 or target_frames % workers != 0:
            args[-1] = (
                video_path,
                args[-1][1],
                frame_count - 1,
                color_extractor,
                target_frames - (workers - 1) * target_frames_per_worker,
            )

        results = pool.starmap(extract_colors, args)

    # Concatenate results from all workers
    final_colors = [color for colors in results for color in colors]

    return final_colors


def extract_colors(
    video_path: str,
    start_frame: int,
    end_frame: int,
    color_extractor: Callable,
    target_frames: Optional[int] = None,
) -> List:
    """
    Extracts dominant colors from frames in a video file.

    :param str video_path: The video capture object.
    :param int start_frame: The index of the first frame to process.
    :param int end_frame: The index of the last frame to process.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :param Optional[int] target_frames: The total number of frames to sample.
    :return: List of dominant colors from the sampled frames.
    """
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Calculate frame_skip based on target_frames
    total_frames = end_frame - start_frame + 1
    if target_frames:
        frame_skip = total_frames // target_frames
    else:
        frame_skip = 1

    colors = []

    for _ in tqdm(range(target_frames or total_frames), desc="Processing frames"):
        ret, frame = video.read()  # Read the first or next frame
        if ret:
            dominant_color = color_extractor(frame)
            colors.append(dominant_color)
        for _ in range(frame_skip - 1):
            video.grab()  # Skip frames

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

    cropped_frame = frame[y : y + h, x : x + w]

    return cropped_frame
