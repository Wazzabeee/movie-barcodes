import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color_avg(frame: np.ndarray) -> np.ndarray:
    """
    Use simple average to find the most dominant color.
    """
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


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
