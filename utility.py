import os
import numpy as np
from PIL import Image


def save_barcode_image(barcode: np.ndarray, base_name: str, args) -> None:
    # If destination_path isn't specified, construct one based on the video's name
    if not args.destination_path:
        ensure_directory("barcodes")

        # Constructing the filename
        filename_parts = [f"barcode_{base_name}"]

        if args.method != 'avg':
            filename_parts.append(args.method)

        if args.barcode_type != 'horizontal':
            filename_parts.append(args.barcode_type)

        if args.workers:
            filename_parts.append(f"workers_{args.workers}")

        destination_name = "_".join(filename_parts) + ".png"
        destination_path = os.path.join("barcodes", destination_name)
    else:
        destination_path = args.destination_path

    image = Image.fromarray(barcode, 'RGBA')  # Specify 'RGBA' mode for image with alpha channel
    image.save(destination_path)


def ensure_directory(directory_name: str) -> None:
    """Ensure that a given directory exists. If not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
