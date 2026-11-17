import logging
from multiprocessing import Pool
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Number of frames sampled over the movie to detect black borders
CROP_DETECTION_SAMPLES = 24
# Share of the movie skipped at the start and at the end when sampling (credits, fades)
CROP_DETECTION_MARGIN = 0.05
# Rows/columns whose brightness (high percentile of the brightest channel) stays at or below this value
# in every sampled frame are considered black borders. Above the usual video black level (16) plus compression noise.
CROP_BRIGHTNESS_THRESHOLD = 24
# Percentile used instead of the max, so isolated compression artifacts in the bars are ignored
CROP_BRIGHTNESS_PERCENTILE = 98
# A detected content area smaller than this share of the frame (on either axis) is considered unreliable
CROP_MIN_RATIO = 0.3

Rect = Tuple[int, int, int, int]

# Below this frame index, workers reach their start frame by grabbing from the beginning instead of seeking:
# FFmpeg seeks are unreliable near the start of some streams (e.g. MPEG-TS lands on the same frame for every
# target up to ~1.3 s, and even a seek to frame 0 skips frames), and decoding a few hundred frames is cheap.
SEEK_FROM_START_FRAMES = 250


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


def find_content_area(frames: Sequence[np.ndarray]) -> Optional[Rect]:
    """
    Find the area of the frames that is not covered by black borders (letterbox or pillarbox).

    A row (or column) belongs to the content if it is brighter than CROP_BRIGHTNESS_THRESHOLD in at least one
    frame. Fully dark frames are ignored.

    :param Sequence[np.ndarray] frames: BGR frames of the same size.
    :return: Optional[Rect]: (x, y, w, h) of the content area, or None if it cannot be detected reliably.
    """
    row_profile = None
    col_profile = None
    for frame in frames:
        brightness = frame.max(axis=2) if frame.ndim == 3 else frame
        rows = np.percentile(brightness, CROP_BRIGHTNESS_PERCENTILE, axis=1)
        cols = np.percentile(brightness, CROP_BRIGHTNESS_PERCENTILE, axis=0)
        if rows.max() <= CROP_BRIGHTNESS_THRESHOLD:
            continue  # Fully dark frame: no information about the borders
        row_profile = rows if row_profile is None else np.maximum(row_profile, rows)
        col_profile = cols if col_profile is None else np.maximum(col_profile, cols)

    if row_profile is None or col_profile is None:
        return None

    content_rows = np.flatnonzero(row_profile > CROP_BRIGHTNESS_THRESHOLD)
    content_cols = np.flatnonzero(col_profile > CROP_BRIGHTNESS_THRESHOLD)
    if content_rows.size == 0 or content_cols.size == 0:
        return None

    y, x = int(content_rows[0]), int(content_cols[0])
    h = int(content_rows[-1]) - y + 1
    w = int(content_cols[-1]) - x + 1
    if h < CROP_MIN_RATIO * row_profile.size or w < CROP_MIN_RATIO * col_profile.size:
        return None

    return x, y, w, h


def detect_content_area(video_path: str, frame_count: int) -> Rect:
    """
    Detect the content area of a video once, by sampling frames spread over the movie.

    :param str video_path: The path to the video file.
    :param int frame_count: The total number of frames in the video.
    :return: Rect: (x, y, w, h) of the content area; the full frame if no reliable border is detected.
    """
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first = int(frame_count * CROP_DETECTION_MARGIN)
    last = max(first, int(frame_count * (1 - CROP_DETECTION_MARGIN)) - 1)
    indices = np.unique(np.linspace(first, last, CROP_DETECTION_SAMPLES).astype(int))

    frames = []
    for index in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ret, frame = video.read()
        if ret:  # The frame count may be over-reported by the container
            frames.append(frame)
    video.release()

    if frames:
        frame_height, frame_width = frames[0].shape[:2]

    return find_content_area(frames) or (0, 0, frame_width, frame_height)


def crop_and_extract(frame: np.ndarray, color_extractor: Callable, crop_rect: Rect):
    """
    Crop a frame to the given area, then extract its color.

    Module-level function so that functools.partial(crop_and_extract, ...) can be sent to worker processes.

    :param np.ndarray frame: The frame as a NumPy array.
    :param Callable color_extractor: A function to extract the dominant color from a frame.
    :param Rect crop_rect: (x, y, w, h) of the area to keep.
    :return: The output of color_extractor on the cropped frame.
    """
    x, y, w, h = crop_rect
    return color_extractor(frame[y : y + h, x : x + w])


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

    # Cap workers to available work to avoid empty tasks
    active_workers = max(1, min(workers, frame_count, target_frames))

    # Evenly distribute frame ranges across active workers
    base_frames = frame_count // active_workers
    remainder_frames = frame_count % active_workers
    frame_ranges = []
    start = 0
    for i in range(active_workers):
        length = base_frames + (1 if i < remainder_frames else 0)
        end = start + length - 1
        frame_ranges.append((start, end))
        start = end + 1

    # Evenly distribute target samples ensuring total equals target_frames
    base_samples = target_frames // active_workers
    remainder_samples = target_frames % active_workers
    samples_per_worker = [base_samples + (1 if i < remainder_samples else 0) for i in range(active_workers)]

    # Build only tasks that have at least one sample (avoid passing 0 -> falsy)
    task_args = []
    for i in range(active_workers):
        if samples_per_worker[i] <= 0:
            continue
        start_frame, end_frame = frame_ranges[i]
        if end_frame < start_frame:
            continue
        task_args.append(
            (
                video_path,
                start_frame,
                end_frame,
                color_extractor,
                samples_per_worker[i],
            )
        )

    with Pool(active_workers) as pool:
        results = pool.starmap(extract_colors, task_args)

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
    if not _seek_exact(video, start_frame):
        # The frame count is only an estimate for some containers: this range starts past the real end
        video.release()
        return []

    total_frames = end_frame - start_frame + 1
    if not target_frames or target_frames > total_frames:
        target_frames = total_frames

    colors = []
    position = 0  # Offset (relative to start_frame) of the next frame read() will return

    for i in tqdm(range(target_frames), desc="Processing frames"):
        # Spread samples evenly over the whole range, so the end of the range is sampled too
        # (an integer step total_frames // target_frames would leave up to ~half the range unsampled)
        target_offset = i * total_frames // target_frames
        while position < target_offset:
            video.grab()  # Skip frames without decoding them
            position += 1
        ret, frame = video.read()
        position += 1
        if not ret:
            break
        colors.append(color_extractor(frame))

    video.release()

    return colors


def _seek_exact(video: cv2.VideoCapture, frame_index: int) -> bool:
    """
    Position a freshly opened capture so that the next read() returns frame_index.

    Seeking can land on a nearby keyframe instead of the requested frame, so the position reported by the backend
    is checked and the remaining frames are grabbed (decoded without conversion) up to frame_index.

    :param cv2.VideoCapture video: A capture that has not read any frame yet.
    :param int frame_index: The index of the frame the next read() must return.
    :return: False if the video ends before frame_index (over-reported frame count), True otherwise.
    """
    position = 0
    if frame_index >= SEEK_FROM_START_FRAMES:
        # Seek to the target, then back off once if the backend overshot it
        for seek_to in (frame_index, frame_index - SEEK_FROM_START_FRAMES):
            video.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
            position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            if position <= frame_index:
                break
        else:
            logging.warning(
                "Seek overshot frame %d (landed on %d), colors may be slightly shifted.", frame_index, position
            )
            return True

    for _ in range(frame_index - position):
        if not video.grab():
            return False
    return True
