import numpy as np
import cv2


def generate_circular_barcode(colors: list, img_size: int) -> np.ndarray:
    """
    Generate a circular barcode from the list of colors.

    :param colors: List of RGB colors.
    :param img_size: The size of the square image (both width and height).
    :return: Circular barcode image.
    """
    scale_factor = 10  # Increase this for higher quality
    high_res_img_size = img_size * scale_factor

    # Create an image with an alpha channel (4th channel)
    barcode_high_res = np.zeros((high_res_img_size, high_res_img_size, 4), dtype=np.uint8)
    center = high_res_img_size // 2

    # Compute the radius increment
    total_circles = len(colors)
    max_radius = center  # The largest circle's radius will be half of the image size
    radius_increment = max_radius / total_circles

    for idx, color in enumerate(colors):
        radius = (idx + 1) * radius_increment
        cv2.circle(barcode_high_res,
                   (center, center),
                   int(radius),
                   tuple(map(int, color[::-1])) + (255,),
                   thickness=scale_factor)

    # Down-sample to the desired resolution with antialiasing
    barcode = cv2.resize(barcode_high_res, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return barcode


def generate_barcode(colors: list, frame_height: int, frame_count: int) -> np.ndarray:
    barcode = np.zeros((frame_height, frame_count, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        barcode[:, i] = color
    return barcode
