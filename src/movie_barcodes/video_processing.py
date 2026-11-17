from multiprocessing import Pool
from typing import Callable, List, Optional

import cv2
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
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
