import argparse
import unittest
import os
import glob
import sys
import tempfile
from unittest.mock import patch

import numpy as np
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

    def test_all_methods_decodes_video_once(self):
        """
        With --all_methods, frames are extracted in a single pass and one image is saved per method.
        """
        video_path = os.path.abspath(self.input_video_path)
        args = ["movie-barcodes", "-i", video_path, "-a", "-w", "1", "--width", "20", "--height", "10"]
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                with (
                    patch.object(sys, "argv", args),
                    patch.object(main, "extract_colors", wraps=main.extract_colors) as extract_spy,
                ):
                    main.main()
                saved = sorted(os.listdir("barcodes"))
            finally:
                os.chdir(cwd)

        self.assertEqual(extract_spy.call_count, 1)
        methods = ["avg", "bgr", "hsv", "kmeans", "smoothed"]
        self.assertEqual(saved, [f"sample_{method}_horizontal_workers_1.png" for method in methods])

    def test_linear_all_methods_parallel(self):
        """
        --linear with -a, --crop-borders and 2 workers: the linear extractors survive pickling, only avg and
        smoothed get a '_linear' file, and linear averaging is never darker than gamma averaging.
        """
        video_path = os.path.abspath(self.input_video_path)
        common = ["movie-barcodes", "-i", video_path, "-w", "2", "--width", "20", "--height", "10", "--crop-borders"]
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                with patch.object(sys, "argv", common + ["-a", "--linear"]):
                    main.main()
                with patch.object(sys, "argv", common + ["-m", "avg"]):
                    main.main()
                saved = sorted(os.listdir("barcodes"))
                with Image.open("barcodes/sample_avg_linear_horizontal_workers_2.png") as image:
                    linear = np.asarray(image.convert("RGB"), dtype=float)
                with Image.open("barcodes/sample_avg_horizontal_workers_2.png") as image:
                    gamma = np.asarray(image.convert("RGB"), dtype=float)
            finally:
                os.chdir(cwd)

        methods = ["avg", "avg_linear", "bgr", "hsv", "kmeans", "smoothed_linear"]
        self.assertEqual(saved, [f"sample_{method}_horizontal_workers_2.png" for method in methods])
        self.assertTrue(np.all(linear >= gamma - 1))
        self.assertGreater(linear.mean(), gamma.mean())


if __name__ == "__main__":
    unittest.main()
