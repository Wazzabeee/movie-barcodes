import numpy as np
from sklearn.cluster import KMeans
import cv2


def get_dominant_color_avg(frame: np.ndarray) -> np.ndarray:
    """
    Use simple average to find the most dominant color.
    """
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def get_dominant_color_mean(frame: np.ndarray) -> np.ndarray:
    return np.array(cv2.mean(frame)[:3])


def get_dominant_color_kmeans(frame: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Use KMeans clustering to find the most dominant color.
    """
    # Reshape the frame to be a list of pixels
    pixels = frame.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # If we are looking only for the most dominant color
    if k == 1:
        return cluster_centers[0]
    else:
        # If we wanted to retrieve more than one color
        # We'd take the center of the largest cluster, assuming it's the dominant color
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_index = labels[np.argmax(counts)]
        return cluster_centers[dominant_index]


def get_dominant_color_hsv(frame: np.ndarray) -> np.ndarray:
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of the Hue channel
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])

    # Find the Hue with the maximum count
    dominant_hue = np.argmax(hist)

    # Construct a color in the HSV space with maximum Hue count, 100% saturation, and 100% value
    dominant_color_hsv = np.array([[[dominant_hue, 255, 255]]], dtype=np.uint8)

    # Convert the dominant color to BGR color space
    dominant_color_bgr = cv2.cvtColor(dominant_color_hsv, cv2.COLOR_HSV2BGR)[0][0]

    return dominant_color_bgr


def get_dominant_color_bgr(frame: np.ndarray) -> np.ndarray:
    # Calculate the histogram for each color channel
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

    # Find the color with the maximum count
    dominant_b = np.argmax(hist_b)
    dominant_g = np.argmax(hist_g)
    dominant_r = np.argmax(hist_r)

    return np.array([dominant_b, dominant_g, dominant_r], dtype=np.uint8)
