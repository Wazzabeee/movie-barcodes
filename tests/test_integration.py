import argparse
import unittest
import os
import glob
import sys
from unittest.mock import patch

from PIL import Image

from movie_barcodes import cli as main


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.input_video_path = "tests/sample.mp4"

    def _run_test(self, barcode_type, workers, width=None):
        # Build the command line arguments list
        args = [
            "movie-barcodes",
            "--input_video_path",
            self.input_video_path,
            "--barcode_type",
            barcode_type,
            "--method",
            "avg",
            "--workers",
            str(workers),
        ]
        if width is not None:
            args.extend(["--width", str(width)])

        # Use patch to simulate command line arguments
        with patch.object(sys, "argv", args):
            main.main()  # Call the main function which now internally handles args

    def test_horizontal_1_worker_default_width(self):
        self._run_test("horizontal", 1)

    def test_horizontal_1_worker_defined_width(self):
        self._run_test("horizontal", 1, 90)

    def test_horizontal_2_workers_default_width(self):
        self._run_test("horizontal", 2)

    def test_horizontal_2_workers_defined_width(self):
        self._run_test("horizontal", 2, 90)

    def test_circular_1_worker_default_width(self):
        self._run_test("circular", 1)

    def test_circular_1_worker_defined_width(self):
        self._run_test("circular", 1, 90)

    def test_circular_2_workers_default_width(self):
        self._run_test("circular", 2)

    def test_circular_2_workers_defined_width(self):
        self._run_test("circular", 2, 90)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up by deleting the generated images that start with "sample_"
        after all tests in this class have run.
        """
        # Specify the path to the barcode images that start with "sample_"
        barcode_images_path = "barcodes/sample_*"
        # Use glob.glob to find all matching files
        for img_file in glob.glob(barcode_images_path):
            try:
                os.remove(img_file)  # Remove the file
            except OSError as e:
                print(f"Error: {img_file} : {e.strerror}")

    def test_default_width_is_video_width_capped_by_frame_count(self):
        """
        Without --width, the number of samples (and horizontal width) is min(video width, frame count).
        """
        args = argparse.Namespace(width=None)
        self.assertEqual(main.get_target_frames(args, frame_count=100_000, frame_width=1920), 1920)
        self.assertEqual(main.get_target_frames(args, frame_count=93, frame_width=2048), 93)
        self.assertEqual(main.get_target_frames(argparse.Namespace(width=50), 93, 2048), 50)

    def test_missing_input_file_raises_file_not_found(self):
        with patch.object(sys, "argv", ["movie-barcodes", "-i", "does/not/exist.mp4"]):
            with self.assertRaises(FileNotFoundError):
                main.main()

    def test_horizontal_output_dimensions(self):
        """
        End-to-end: the horizontal barcode has the requested width and height and is saved in ./barcodes.
        """
        args = ["movie-barcodes", "-i", self.input_video_path, "-w", "1", "--width", "40", "--height", "25"]
        args += ["-n", "sample_dimensions_check"]
        with patch.object(sys, "argv", args):
            main.main()
        with Image.open("barcodes/sample_dimensions_check.png") as image:
            self.assertEqual(image.size, (40, 25))


if __name__ == "__main__":
    unittest.main()
