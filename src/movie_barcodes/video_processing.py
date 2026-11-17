import functools
import logging
import multiprocessing
import queue
from contextlib import closing
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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

# A single color extractor, or several extractors keyed by method name to run on the same decoded frames
ColorExtractors = Union[Callable, Dict[str, Callable]]

# Below this frame index, workers reach their start frame by grabbing from the beginning instead of seeking:
# FFmpeg seeks are unreliable near the start of some streams (e.g. MPEG-TS lands on the same frame for every
# target up to ~1.3 s, and even a seek to frame 0 skips frames), and decoding a few hundred frames is cheap.
SEEK_FROM_START_FRAMES = 250

# Number of frames to skip above which seeking is used instead of grab()-ing each of them.
# grab() still decodes each frame, so its cost grows with the gap, while a seek costs roughly
# one decode from the previous keyframe to the target, whatever the gap. The crossover therefore
# depends on the keyframe interval: measured on 720p files (FFmpeg backend), seeking wins above
# ~30 frames for MPEG-4 Part 2 (GOP 12), ~60 for H.264 with GOP 48, ~90 for H.264 with scene cuts
# every 2-5 s, and ~150 for H.264/HEVC with GOP 250 and no scene cuts (the worst case).
SEEK_THRESHOLD = 100


# Worker processes are started with "spawn" on every platform. With "fork" (the Linux default before
# Python 3.14), workers inherit the parent's thread pools in an inconsistent state: if the parent already
# ran scikit-learn's K-means (OpenMP) or OpenCV's parallel code, K-means in the workers can deadlock forever.
MP_CONTEXT = multiprocessing.get_context("spawn")


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


def crop_frame(frame: np.ndarray, crop_rect: Optional[Rect]) -> np.ndarray:
    """
    Crop a frame to the given area.

    :param np.ndarray frame: The frame as a NumPy array.
    :param Optional[Rect] crop_rect: (x, y, w, h) of the area to keep, or None to keep the whole frame.
    :return: np.ndarray: A view of the cropped area (no copy).
    """
    if crop_rect is None:
        return frame
    x, y, w, h = crop_rect
    return frame[y : y + h, x : x + w]


def _sample_count(start_frame: int, end_frame: int, n_samples: Optional[int]) -> int:
    """
    Number of frames actually sampled from [start_frame, end_frame]: n_samples, capped by the size of the range
    (every frame of the range if n_samples is None or 0).
    """
    total_frames = end_frame - start_frame + 1
    if not n_samples or n_samples > total_frames:
        return total_frames
    return n_samples


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


def _skip_to(video: cv2.VideoCapture, position: int, target: int) -> int:
    """
    Advance the capture so that the next read() returns frame `target`.

    Skips of up to SEEK_THRESHOLD frames use grab(); longer skips seek with
    CAP_PROP_POS_FRAMES, then grab() the remaining frames if the backend landed before the target.
    Targets below SEEK_FROM_START_FRAMES are always grabbed (seeks are unreliable near the start of
    some streams), and a failed seek falls back to grabbing. A backend that overshoots the target
    cannot be corrected cheaply: the next read() then returns the frame it landed on, and the
    returned position reflects it so later samples stay in order. Grabbing stops if the video ends
    early (over-reported frame count); the next read() then fails.

    :param cv2.VideoCapture video: The video capture object.
    :param int position: Absolute index of the frame the next read() would return.
    :param int target: Absolute index of the frame to read next.
    :return: Absolute index of the frame the next read() will return.
    """
    if (
        target - position > SEEK_THRESHOLD
        and target >= SEEK_FROM_START_FRAMES
        and video.set(cv2.CAP_PROP_POS_FRAMES, target)
    ):
        position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    while position < target:
        if not video.grab():  # Skip frames without returning them
            break
        position += 1
    return position


def iter_sampled_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    n_samples: Optional[int] = None,
    crop_rect: Optional[Rect] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Decode n_samples frames spread evenly over [start_frame, end_frame], in order.

    All color extraction goes through it: it positions the capture exactly on start_frame (_seek_exact),
    reaches each next sample by grabbing or seeking depending on the gap (_skip_to), stops early if the video
    is shorter than its reported frame count, and crops each frame to crop_rect. (detect_content_area samples
    its few frames on its own: approximate seeks are enough there.)
    The capture is released when the iterator is exhausted or closed.

    :param str video_path: The path to the video file.
    :param int start_frame: The index of the first frame of the range.
    :param int end_frame: The index of the last frame of the range.
    :param Optional[int] n_samples: Number of frames to sample; every frame of the range if None, 0 or larger
        than the range.
    :param Optional[Rect] crop_rect: (x, y, w, h) of the area to keep in each frame, or None for the full frame.
    :return: Iterator of (frame index, frame) pairs. The index is the one the capture reports, which can differ
        from the requested one if a seek overshoots.
    """
    total_frames = end_frame - start_frame + 1
    n_samples = _sample_count(start_frame, end_frame, n_samples)

    video = cv2.VideoCapture(video_path)
    try:
        if not _seek_exact(video, start_frame):
            # The frame count is only an estimate for some containers: this range starts past the real end
            return

        position = start_frame  # Index of the frame the next read() will return
        for i in range(n_samples):
            # Spread samples evenly over the whole range, so the end of the range is sampled too
            # (an integer step total_frames // n_samples would leave up to ~half the range unsampled)
            position = _skip_to(video, position, start_frame + i * total_frames // n_samples)
            ret, frame = video.read()
            if not ret:
                return
            yield position, crop_frame(frame, crop_rect)
            position += 1
    finally:
        video.release()


def extract_colors(
    video_path: str,
    start_frame: int,
    end_frame: int,
    color_extractor: ColorExtractors,
    target_frames: Optional[int] = None,
    show_progress: bool = True,
    progress_queue: Optional[Any] = None,
    crop_rect: Optional[Rect] = None,
) -> Union[List, Dict[str, List]]:
    """
    Extracts dominant colors from frames sampled over a range of a video file.

    :param str video_path: The path to the video file.
    :param int start_frame: The index of the first frame to process.
    :param int end_frame: The index of the last frame to process.
    :param color_extractor: A function to extract the dominant color from a frame, or a dict
        {method_name: function} to run several extractors on the same decoded frames.
    :param Optional[int] target_frames: The total number of frames to sample.
    :param bool show_progress: Whether to display a progress bar for this call.
    :param Optional[Any] progress_queue: If given, put 1 on this queue for each processed frame
        (used by parallel_extract_colors to drive a single global progress bar).
    :param Optional[Rect] crop_rect: (x, y, w, h) of the area to keep in each frame, or None for the full frame.
    :return: List of dominant colors from the sampled frames, or {method_name: list of colors}
        when a dict of extractors is given.
    """
    extractors = color_extractor if isinstance(color_extractor, dict) else {None: color_extractor}
    colors: Dict = {name: [] for name in extractors}

    frames = iter_sampled_frames(video_path, start_frame, end_frame, target_frames, crop_rect)
    total = _sample_count(start_frame, end_frame, target_frames)
    with closing(frames):
        for _, frame in tqdm(frames, total=total, desc="Processing frames", disable=not show_progress):
            for name, extractor in extractors.items():
                colors[name].append(extractor(frame))
            if progress_queue is not None:
                progress_queue.put(1)

    return colors if isinstance(color_extractor, dict) else colors[None]


def _split_work(frame_count: int, target_frames: int, workers: int) -> List[Tuple[int, int, int]]:
    """
    Split the video into contiguous frame ranges, one per worker, and the samples evenly between them.

    :param int frame_count: The total number of frames in the video.
    :param int target_frames: The total number of frames to sample.
    :param int workers: Number of parallel workers.
    :return: List of (start_frame, end_frame, n_samples), in frame order, for the workers that have something
        to sample. The n_samples add up to target_frames.
    """
    # Cap workers to available work to avoid empty tasks
    active_workers = max(1, min(workers, frame_count, target_frames))

    tasks = []
    start = 0
    for i in range(active_workers):
        length = frame_count // active_workers + (1 if i < frame_count % active_workers else 0)
        samples = target_frames // active_workers + (1 if i < target_frames % active_workers else 0)
        end = start + length - 1
        if samples > 0 and end >= start:
            tasks.append((start, end, samples))
        start = end + 1
    return tasks


def parallel_extract_colors(
    video_path: str,
    frame_count: int,
    color_extractor: ColorExtractors,
    workers: int,
    target_frames: Optional[int] = None,
    crop_rect: Optional[Rect] = None,
) -> Union[List, Dict[str, List]]:
    """
    Extract dominant colors from frames in a video file using parallel processing.

    Each worker runs extract_colors on its own range of the video; the results are concatenated in frame order.

    :param str video_path: The path to the video file.
    :param int frame_count: The total number of frames in the video.
    :param color_extractor: A function to extract the dominant color from a frame, or a dict
        {method_name: function} to run several extractors on the same decoded frames.
    :param int workers: Number of parallel workers.
    :param Optional[int] target_frames: The total number of frames to sample.
    :param Optional[Rect] crop_rect: (x, y, w, h) of the area to keep in each frame, or None for the full frame.
    :return: List of dominant colors for the frames in the video, or {method_name: list of colors}
        when a dict of extractors is given.
    """
    if target_frames is None:
        target_frames = frame_count

    tasks = _split_work(frame_count, target_frames, workers)
    task_args = [(video_path, start, end, color_extractor, samples) for start, end, samples in tasks]
    total_samples = sum(_sample_count(start, end, samples) for start, end, samples in tasks)

    # Workers don't draw their own bar (N bars on one terminal garble the output): they report each
    # processed frame to a shared queue, and the parent process draws a single global bar.
    with MP_CONTEXT.Manager() as manager, MP_CONTEXT.Pool(max(1, len(tasks))) as pool:
        progress_queue = manager.Queue()
        worker = functools.partial(
            extract_colors, show_progress=False, progress_queue=progress_queue, crop_rect=crop_rect
        )
        async_result = pool.starmap_async(worker, task_args)
        with tqdm(total=total_samples, desc="Processing frames") as progress_bar:
            # Poll ready() rather than blocking on the queue, so a worker that raises can't hang us
            while not async_result.ready():
                try:
                    progress_bar.update(progress_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
            while not progress_queue.empty():
                progress_bar.update(progress_queue.get())
            results = async_result.get()  # Re-raises a worker's exception
            # Workers may return fewer frames than requested (end of video): still end at 100%
            progress_bar.update(total_samples - progress_bar.n)

    # Concatenate results from all workers, in frame order
    if isinstance(color_extractor, dict):
        return {name: [color for colors in results for color in colors[name]] for name in color_extractor}
    return [color for colors in results for color in colors]
