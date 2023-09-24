import numpy as np
import cv2


def generate_circular_barcode(colors: list, img_size: int) -> np.ndarray:
    """
    Generate a circular barcode from the list of colors.

    :param colors: List of RGB colors.
    :param img_size: The size of the square image (both width and height).
    :return: Circular barcode image.
    """
    barcode = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
    center = img_size // 2  # Center of the image

    # Compute the radius increment
    total_circles = len(colors)
    max_radius = center  # The largest circle's radius will be half of the image size
    radius_increment = max_radius / total_circles

    for idx, color in enumerate(colors):
        radius = (idx + 1) * radius_increment
        cv2.circle(barcode, (center, center), int(radius), tuple(map(int, color)), thickness=-1)  # Filled circle

    return barcode


def generate_barcode(colors: list, frame_height: int, frame_count: int) -> np.ndarray:
    barcode = np.zeros((frame_height, frame_count, 3))
    for i, color in enumerate(colors):
        barcode[:, i] = color
    return barcode
