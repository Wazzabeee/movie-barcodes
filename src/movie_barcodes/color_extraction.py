import math

import numpy as np
from sklearn.cluster import KMeans
import cv2

# Frames are downscaled to this many pixels (approximately) before running K-means
KMEANS_MAX_PIXELS = 64 * 64

# Pixels with saturation or value below this threshold have no meaningful hue (black, white, gray)
HSV_MIN_SATURATION = 40
HSV_MIN_VALUE = 40
# Minimum share of colorful pixels required to pick a dominant hue; otherwise fall back to the mean color
HSV_MIN_COLORFUL_RATIO = 0.05

# Bins per channel of the joint BGR histogram (must be a power of two); 16 gives 16 ** 3 = 4096 color bins
BGR_HIST_BINS = 16
# Frames are subsampled to at most this many pixels (approximately) before building the BGR histogram
BGR_HIST_MAX_PIXELS = 256 * 256


def get_smoothed_frame(frame: np.ndarray) -> np.ndarray:
    """
    Smoothes the given frame with a two-step resize process.

    The first step averages every pixel of the frame (area interpolation), the second step stretches
    that average color back to a column of the original height.

    :param np.ndarray frame: frame to be smoothed.
    :return: np.ndarray: Smoothed frame.
    """
    original_height = frame.shape[0]
    # INTER_AREA averages all pixels; the default INTER_LINEAR would only sample a few pixels around the center
    frame = cv2.resize(frame, (1, 1), interpolation=cv2.INTER_AREA)
    return cv2.resize(frame, (1, original_height)).reshape(original_height, 1, 3).astype(np.uint8)


def get_dominant_color_mean(frame: np.ndarray) -> np.ndarray:
    """
    Gets the dominant color of a frame using OpenCV's mean function.

    :param np.ndarray frame: The frame as a NumPy array.
    :return: Dominant color as a NumPy array.
    """
    return np.array(cv2.mean(frame)[:3])


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """
    Decodes sRGB values in [0, 1] to linear light with the exact sRGB transfer function.

    :param np.ndarray values: sRGB values in [0, 1].
    :return: np.ndarray: Linear light values in [0, 1] (float64).
    """
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """
    Encodes linear light values in [0, 1] to sRGB with the exact sRGB transfer function.

    :param np.ndarray values: Linear light values in [0, 1].
    :return: np.ndarray: sRGB values in [0, 1] (float64).
    """
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


# Linear light value of every 8-bit sRGB code value
SRGB_TO_LINEAR_LUT = srgb_to_linear(np.arange(256) / 255.0)


def get_dominant_color_mean_linear(frame: np.ndarray) -> np.ndarray:
    """
    Gets the mean color of a frame averaged in linear light, re-encoded to sRGB.

    Averaging gamma encoded values (get_dominant_color_mean) makes high-contrast frames too dark: a frame half
    black and half white averages to 128 instead of 188, the sRGB value of the perceived mix of the two.

    The mean of every pixel is computed exactly without building a float image: per channel, a 256-bin
    histogram of the 8-bit values is weighted by the linear value of each bin.

    :param np.ndarray frame: The frame as a NumPy array (BGR, uint8).
    :return: Mean color as a NumPy array (BGR, uint8).
    """
    pixel_count = frame.shape[0] * frame.shape[1]
    linear_mean = np.array(
        [cv2.calcHist([frame], [channel], None, [256], [0, 256]).ravel() @ SRGB_TO_LINEAR_LUT for channel in range(3)]
    )
    return np.round(linear_to_srgb(linear_mean / pixel_count) * 255).astype(np.uint8)


def get_smoothed_frame_linear(frame: np.ndarray) -> np.ndarray:
    """
    Linear light version of get_smoothed_frame: a column of the original height filled with the mean color of the
    frame averaged in linear light (see get_dominant_color_mean_linear).

    Area-resizing the whole frame to a single pixel is exactly the mean of all its pixels, so the column is filled
    with that mean directly.

    :param np.ndarray frame: frame to be smoothed (BGR, uint8).
    :return: np.ndarray: Smoothed frame (original height x 1 x 3, uint8).
    """
    return np.tile(get_dominant_color_mean_linear(frame), (frame.shape[0], 1, 1))


def get_dominant_color_kmeans(frame: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Gets the dominant color of a frame using KMeans clustering.

    The frame is downscaled before clustering: the dominant color does not depend on the resolution,
    and clustering every pixel of a full HD frame is very slow.

    :param np.ndarray frame: The frame as a NumPy array.
    :param int k: Number of clusters for KMeans algorithm. Defaults to 3.
    :return: Dominant color as a NumPy array.
    """
    height, width = frame.shape[:2]
    if height * width > KMEANS_MAX_PIXELS:
        scale = (KMEANS_MAX_PIXELS / (height * width)) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Reshape the frame to be a list of pixels
    pixels = frame.reshape(-1, 3).astype(np.float32)

    # Cannot ask for more clusters than there are distinct pixels
    k = max(1, min(k, len(np.unique(pixels, axis=0))))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=0)
    kmeans.fit(pixels)

    # Get the BGR values of the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # If we are looking only for the most dominant color
    if k == 1:
        return cluster_centers[0]

    # Take the center of the largest cluster, assuming it's the dominant color
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_index = labels[np.argmax(counts)]
    return cluster_centers[dominant_index]


def get_dominant_color_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Gets the dominant color of a frame by converting it to HSV color space and finding the dominant hue.

    Only colorful pixels (enough saturation and value) vote for the hue: black, white and gray pixels have
    no meaningful hue (OpenCV reports 0, i.e. red). If the frame has almost no colorful pixels (fade to black,
    credits, black and white movie), the mean color of the frame is returned instead.

    :param np.ndarray frame: The frame as a NumPy array.
    :return: Dominant color as a NumPy array (BGR, uint8).
    """
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colorful = ((hsv[..., 1] >= HSV_MIN_SATURATION) & (hsv[..., 2] >= HSV_MIN_VALUE)).astype(np.uint8)
    if np.count_nonzero(colorful) < HSV_MIN_COLORFUL_RATIO * colorful.size:
        return np.array(cv2.mean(frame)[:3]).round().astype(np.uint8)

    # Histogram of the Hue channel (OpenCV 8-bit hue range is [0, 180)) over colorful pixels only
    hist = cv2.calcHist([hsv], [0], colorful, [180], [0, 180])

    # Find the Hue with the maximum count
    dominant_hue = np.argmax(hist)

    # Construct a color in the HSV space with maximum Hue count, 100% saturation, and 100% value
    dominant_color_hsv = np.array([[[dominant_hue, 255, 255]]], dtype=np.uint8)

    # Convert the dominant color to BGR color space
    dominant_color_bgr = cv2.cvtColor(dominant_color_hsv, cv2.COLOR_HSV2BGR)[0][0]

    return dominant_color_bgr


def get_dominant_color_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Gets the dominant color of a frame with a joint 3D histogram of its BGR values.

    Each channel is quantized to BGR_HIST_BINS bins, so every pixel falls into one of BGR_HIST_BINS ** 3 color
    bins. The most populated bin wins and the mean color of the pixels in that bin is returned: the result is
    always close to a color actually present in the frame (a single-color frame returns that exact color).

    Large frames are subsampled with a regular stride before counting. Unlike a resize, a stride keeps real
    pixels and never blends neighboring colors into a new one.

    :param np.ndarray frame: The frame as a NumPy array.
    :return: Dominant color as a NumPy array (BGR, uint8).
    """
    height, width = frame.shape[:2]
    if height * width > BGR_HIST_MAX_PIXELS:
        stride = math.ceil(math.sqrt(height * width / BGR_HIST_MAX_PIXELS))
        frame = frame[::stride, ::stride]

    pixels = frame.reshape(-1, 3)

    # Flat index of the 3D histogram bin of each pixel
    shift = 8 - int(math.log2(BGR_HIST_BINS))
    quantized = (pixels >> shift).astype(np.intp)
    bin_indices = (quantized[:, 0] * BGR_HIST_BINS + quantized[:, 1]) * BGR_HIST_BINS + quantized[:, 2]

    counts = np.bincount(bin_indices, minlength=BGR_HIST_BINS**3)
    dominant_bin = np.argmax(counts)

    return pixels[bin_indices == dominant_bin].mean(axis=0).round().astype(np.uint8)
