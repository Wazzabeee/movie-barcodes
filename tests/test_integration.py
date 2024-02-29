import argparse
import unittest
import os
import glob

from src import main


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case.
        """
        # Recreate the parser with minimal necessary setup
        self.parser = argparse.ArgumentParser(description="Test Parser")
        self.parser.add_argument("--input_video_path", type=str)
        self.parser.add_argument("--destination_path", type=str, nargs="?", default=None)
        self.parser.add_argument("--barcode_type", choices=["horizontal", "circular"], default="horizontal")
        self.parser.add_argument("--method", choices=["avg", "kmeans", "hsv", "bgr"], default="avg")
        self.parser.add_argument("--workers", type=int, default=None)
        self.parser.add_argument("--width", type=int, default=None)
        self.parser.add_argument("--output_name", type=str, nargs="?", default=None)
        self.parser.add_argument("--all_methods", type=bool, default=False)

        self.input_video_path = "tests/sample.mp4"

    def _run_test(self, barcode_type, workers, width=None):
        args = [
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

        # Parse args using the recreated parser
        parsed_args = self.parser.parse_args(args)

        # Execute the program with the parsed Namespace object
        main.main(parsed_args)

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


if __name__ == "__main__":
    unittest.main()
