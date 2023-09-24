import cv2
from typing import Callable, Optional
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def extract_colors(video: cv2.VideoCapture, frame_count: int, color_extractor: Callable,
                   workers: Optional[int] = None, frame_skip: int = 1) -> list:
    colors = []

    if workers:  # Use parallel processing

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
