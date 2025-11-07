from typing import Optional

import numpy as np
import cv2

from tqdm import tqdm


def generate_circular_barcode(colors: list, img_size: int, scale_factor: int = 10) -> np.ndarray:
    """
    Generate a circular barcode from the list of colors or smoothed frames.

    :param list colors: List of RGB colors or smoothed frames.
    :param int img_size: The size of the square image (both width and height).
    :param int scale_factor: The scale factor to use when generating the barcode. Default is 10.
    :return: np.ndarray: Circular barcode image.
    """
    high_res_img_size = img_size * scale_factor

    # Create an image with an alpha channel (4th channel)
    barcode_high_res = np.zeros((high_res_img_size, high_res_img_size, 4), dtype=np.uint8)
    center = high_res_img_size // 2

    # Compute the radius increment
    total_circles = len(colors)
    max_radius = center  # The largest circle's radius will be half of the image size
    radius_increment = max_radius / total_circles

    for idx, color in tqdm(
        enumerate(colors),
        desc="Generating Barcode",
        total=len(colors),
        unit="%",
        bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
    ):
        radius = (idx + 1) * radius_increment

        # Handle both simple RGB tuples and smoothed frames
        if isinstance(color, np.ndarray) and color.ndim > 1:
            # For smoothed frames, take the average color
            color = np.mean(color, axis=(0, 1)).astype(int)

        # Ensure color is in BGR format for OpenCV
        bgr_color = color[::-1] if len(color) == 3 else color[2::-1]

        cv2.circle(
            barcode_high_res,
            (center, center),
            int(radius),
            tuple(map(int, bgr_color)) + (255,),
            thickness=scale_factor,
        )

    # Down-sample to the desired resolution with antialiasing
    barcode = cv2.resize(barcode_high_res, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return barcode


def generate_barcode(
    colors: list, frame_height: int, frame_count: int, frame_width: Optional[int] = None
) -> np.ndarray:
    """
    Generate a barcode image based on dominant colors or smoothed frames of video frames.
    :param list colors: List of dominant colors or smoothed frames from video frames.
    :param int frame_height: The height of the barcode image.
    :param int frame_count: The total number of frames in the video.
    :param Optional[int] frame_width: The width of the barcode image. If not specified, defaults to frame_count.
    :return: np.ndarray: A barcode image.
    """
    if frame_width is None:
        frame_width = frame_count

    barcode = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    step = max(1, len(colors) // frame_width)
    sampled_colors = [colors[i] for i in range(0, len(colors), step)]
    for i, color in tqdm(
        enumerate(sampled_colors),
        desc="Generating Barcode",
        total=len(sampled_colors),
        unit="%",
        bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
    ):
        if i < frame_width:
            if isinstance(color, np.ndarray) and color.ndim == 3 and color.shape[1] == 1:
                # For smoothed frames
                color_column = cv2.resize(color, (1, frame_height))
            else:
                # For single color values
                color_column = np.full((frame_height, 1, 3), color, dtype=np.uint8)

            # Convert to BGR (which will be interpreted as RGB when saved with PIL)
            bgr_color = cv2.cvtColor(color_column, cv2.COLOR_RGB2BGR)
            barcode[:, i] = bgr_color.reshape(frame_height, 3)

    return barcode
