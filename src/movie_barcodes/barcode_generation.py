from typing import Optional

import numpy as np
import cv2

from tqdm import tqdm

# Upper bound for the side of the supersampled canvas used to draw circular barcodes (BGRA, uint8).
# 4096 x 4096 x 4 bytes = 64 MB, instead of e.g. 19200 x 19200 x 4 = 1.4 GB for a 1920px video with scale 10.
MAX_HIGH_RES_SIZE = 4096


def _to_bgr(color) -> tuple:
    """
    Convert a color (BGR triple or smoothed frame column) to a BGR tuple of ints.

    :param color: BGR triple or smoothed frame (H x 1 x 3).
    :return: tuple: (b, g, r) as ints.
    """
    if isinstance(color, np.ndarray) and color.ndim > 1:
        color = np.mean(color, axis=(0, 1))
    return tuple(int(round(float(c))) for c in color)


def generate_circular_barcode(colors: list, img_size: int, scale_factor: int = 10) -> np.ndarray:
    """
    Generate a circular barcode from the list of colors or smoothed frames.

    The first color is at the center and the last one on the outer ring. Rings are drawn as filled discs from
    the outside in, so they always cover the whole disc without gaps, whatever the number of colors.

    :param list colors: List of BGR colors or smoothed frames.
    :param int img_size: The size of the square image (both width and height).
    :param int scale_factor: The supersampling factor used for antialiasing. Default is 10, capped so that the
        high resolution canvas side does not exceed MAX_HIGH_RES_SIZE.
    :return: np.ndarray: Circular barcode image (BGRA; converted to RGBA when saving).
    """
    if not colors:
        raise ValueError("Cannot generate a barcode without colors.")

    scale_factor = max(1, min(scale_factor, MAX_HIGH_RES_SIZE // img_size))
    high_res_img_size = img_size * scale_factor

    # Create an image with an alpha channel (4th channel), fully transparent
    barcode_high_res = np.zeros((high_res_img_size, high_res_img_size, 4), dtype=np.uint8)
    center = high_res_img_size // 2

    # The largest circle's radius will be half of the image size
    total_circles = len(colors)
    radius_increment = center / total_circles

    for idx in tqdm(reversed(range(total_circles)), desc="Generating Barcode", total=total_circles, unit="it"):
        radius = int(round((idx + 1) * radius_increment))
        if radius < 1:
            continue
        cv2.circle(
            barcode_high_res,
            (center, center),
            radius,
            _to_bgr(colors[idx]) + (255,),
            thickness=-1,
        )

    # Down-sample to the desired resolution with antialiasing
    if scale_factor == 1:
        return barcode_high_res
    return cv2.resize(barcode_high_res, (img_size, img_size), interpolation=cv2.INTER_AREA)


def generate_barcode(
    colors: list, frame_height: int, frame_count: int, frame_width: Optional[int] = None
) -> np.ndarray:
    """
    Generate a barcode image based on dominant colors or smoothed frames of video frames.
    Colors are treated as BGR internally and converted to RGB once at save-time.

    Colors are resampled evenly to the requested width: when there are more colors than columns, colors are
    picked evenly across the whole list (the end of the movie is not dropped); when there are fewer (e.g. the
    container over-reported its frame count), colors are stretched instead of leaving black columns.

    :param list colors: List of dominant BGR colors or smoothed frames from video frames.
    :param int frame_height: The height of the barcode image.
    :param int frame_count: The total number of frames in the video.
    :param Optional[int] frame_width: The width of the barcode image. If not specified, defaults to frame_count.
    :return: np.ndarray: A barcode image (BGR).
    """
    if not colors:
        raise ValueError("Cannot generate a barcode without colors.")

    if frame_width is None:
        frame_width = frame_count

    barcode = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    n_colors = len(colors)
    for i in tqdm(range(frame_width), desc="Generating Barcode", total=frame_width, unit="it"):
        color = colors[i * n_colors // frame_width]
        if isinstance(color, np.ndarray) and color.ndim == 3 and color.shape[1] == 1:
            # For smoothed frames: stretch the column to the barcode height
            color_column = cv2.resize(color, (1, frame_height)).reshape(frame_height, 3)
        else:
            # For single color values
            color_column = np.clip(np.round(np.asarray(color, dtype=np.float64)), 0, 255).astype(np.uint8)

        # Keep BGR internally; perform RGB conversion once at save-time
        barcode[:, i] = color_column

    return barcode
