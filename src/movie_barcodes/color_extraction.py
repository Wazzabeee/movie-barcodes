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
    Gets the dominant color of a frame in the BGR color space by calculating the histograms for each channel.

    Note: each channel is maximized independently, so the resulting color may not be present in the frame.

    :param np.ndarray frame: The frame as a NumPy array.
    :return: Dominant color as a NumPy array.
    """
    # Calculate the histogram for each color channel
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

    # Find the color with the maximum count
    dominant_b = np.argmax(hist_b)
    dominant_g = np.argmax(hist_g)
    dominant_r = np.argmax(hist_r)

    return np.array([dominant_b, dominant_g, dominant_r], dtype=np.uint8)
