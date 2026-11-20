import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import movie_barcodes
from movie_barcodes import api

VIDEO_PATH = "tests/sample.mp4"


class TestCreateBarcode(unittest.TestCase):
    """
    Public Python API: create_barcode / create_barcodes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        video, cls.frame_count, cls.frame_width, cls.frame_height = api.load_video(VIDEO_PATH)
        video.release()

    def test_exported_from_package(self) -> None:
        self.assertIs(movie_barcodes.create_barcode, api.create_barcode)
        self.assertIs(movie_barcodes.create_barcodes, api.create_barcodes)

    def test_horizontal_default_size_and_mode(self) -> None:
        image = api.create_barcode(VIDEO_PATH, workers=1)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        expected_width = min(self.frame_width, self.frame_count)
        self.assertEqual(image.size, (expected_width, self.frame_height))

    def test_horizontal_custom_size(self) -> None:
        image = api.create_barcode(pathlib.Path(VIDEO_PATH), method="hsv", width=30, height=12, workers=2)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (30, 12))

    def test_circular_is_rgba_square(self) -> None:
        image = api.create_barcode(VIDEO_PATH, barcode_type="circular", width=20, height=5, workers=1)
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (self.frame_width, self.frame_width))
        # Corners are outside the disc: transparent
        self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_crop_borders(self) -> None:
        image = api.create_barcode(VIDEO_PATH, width=10, height=4, workers=1, crop_borders=True)
        self.assertEqual(image.size, (10, 4))

    def test_linear_keeps_method_names(self) -> None:
        """
        linear only changes avg and smoothed (brighter), the keys stay the requested method names.
        """
        methods = ["avg", "smoothed", "hsv"]
        gamma = api.create_barcodes(VIDEO_PATH, methods, width=10, height=2, workers=1)
        linear = api.create_barcodes(VIDEO_PATH, methods, width=10, height=2, workers=1, linear=True)
        self.assertEqual(list(linear), methods)
        for method in ("avg", "smoothed"):
            self.assertGreater(np.asarray(linear[method], float).mean(), np.asarray(gamma[method], float).mean())
        np.testing.assert_array_equal(np.asarray(linear["hsv"]), np.asarray(gamma["hsv"]))
        single = api.create_barcode(VIDEO_PATH, method="avg", width=10, height=2, workers=1, linear=True)
        np.testing.assert_array_equal(np.asarray(single), np.asarray(linear["avg"]))

    def test_create_barcodes_decodes_once(self) -> None:
        methods = ["smoothed", "avg", "kmeans", "hsv", "bgr"]
        with patch.object(api, "extract_colors", wraps=api.extract_colors) as extract_spy:
            barcodes = api.create_barcodes(VIDEO_PATH, methods, width=20, height=10, workers=1)
        self.assertEqual(extract_spy.call_count, 1)
        self.assertEqual(list(barcodes), methods)
        for image in barcodes.values():
            self.assertEqual((image.mode, image.size), ("RGB", (20, 10)))

    def test_create_barcodes_defaults_to_all_methods(self) -> None:
        barcodes = api.create_barcodes(VIDEO_PATH, width=5, height=2, workers=1)
        self.assertEqual(tuple(barcodes), api.METHODS)

    def test_create_barcode_matches_create_barcodes(self) -> None:
        single = api.create_barcode(VIDEO_PATH, method="bgr", width=15, height=3, workers=1)
        multi = api.create_barcodes(VIDEO_PATH, ["avg", "bgr"], width=15, height=3, workers=1)
        np.testing.assert_array_equal(np.asarray(single), np.asarray(multi["bgr"]))

    def test_default_workers_uses_all_cores(self) -> None:
        with patch.object(api, "parallel_extract_colors", wraps=api.parallel_extract_colors) as parallel_spy:
            api.create_barcode(VIDEO_PATH, width=8, height=2)
        self.assertEqual(parallel_spy.call_args.args[3], api.MAX_PROCESSES)

    def test_workers_capped_with_warning(self) -> None:
        with (
            patch.object(api, "parallel_extract_colors", wraps=api.parallel_extract_colors) as parallel_spy,
            self.assertLogs(level="WARNING"),
        ):
            api.create_barcode(VIDEO_PATH, width=8, height=2, workers=api.MAX_PROCESSES + 1)
        self.assertEqual(parallel_spy.call_args.args[3], api.MAX_PROCESSES)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            api.create_barcode("does/not/exist.mp4")

    def test_invalid_arguments_raise_before_decoding(self) -> None:
        invalid = [
            {"method": "median"},
            {"barcode_type": "square"},
            {"width": 0},
            {"width": self.frame_count + 1},
            {"height": 0},
            {"workers": 0},
        ]
        with (
            patch.object(api, "extract_colors") as extract_spy,
            patch.object(api, "parallel_extract_colors") as parallel_spy,
        ):
            for kwargs in invalid:
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    api.create_barcode(VIDEO_PATH, **kwargs)
            with self.assertRaises(ValueError):
                api.create_barcodes(VIDEO_PATH, [])
        extract_spy.assert_not_called()
        parallel_spy.assert_not_called()

    def test_invalid_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.txt")
            shutil.copy(VIDEO_PATH, video_path)
            with self.assertRaises(ValueError):
                api.create_barcode(video_path)

    def test_does_not_write_files(self) -> None:
        cwd = os.getcwd()
        video_path = os.path.abspath(VIDEO_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                api.create_barcode(video_path, width=5, height=2, workers=1)
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
