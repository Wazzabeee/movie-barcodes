import os

from PIL import Image

from color_extraction import *


def get_dominant_color_function(method: str):
    if method == 'avg':
        # return get_dominant_color_avg
        return get_dominant_color_mean
    elif method == 'kmeans':
        return get_dominant_color_kmeans
    elif method == 'hsv':
        return get_dominant_color_hsv
    elif method == 'bgr':
        return get_dominant_color_bgr
    else:
        raise ValueError(f"Invalid method: {method}")


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(minutes)}m {int(seconds)}s"


def save_barcode_image(barcode: np.ndarray, base_name: str, args, method: str) -> None:
    # If destination_path isn't specified, construct one based on the video's name
    if not args.destination_path:
        ensure_directory("barcodes")

        # Constructing the filename, always including the method
        filename_parts = [base_name, method, args.barcode_type]

        if args.workers:
            filename_parts.append(f"workers_{str(args.workers)}")

        destination_name = "_".join(filename_parts) + ".png"
        destination_path = os.path.join("barcodes", destination_name)
    else:
        # In case a destination_path is provided, consider appending the method
        # or managing as per your requirement
        destination_path = args.destination_path

    if barcode.shape[2] == 4:  # If the image has an alpha channel (RGBA)
        image = Image.fromarray(barcode, 'RGBA')
    else:  # If the image doesn't have an alpha channel (RGB)
        image = Image.fromarray(barcode, 'RGB')

    image.save(destination_path)


def ensure_directory(directory_name: str) -> None:
    """Ensure that a given directory exists. If not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
