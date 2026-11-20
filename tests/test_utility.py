import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock, Mock
import cv2
import numpy as np
from PIL import Image


from movie_barcodes import utility, color_extraction


class TestUtility(unittest.TestCase):
    """
    Test the utility functions.
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        :return: None
        """
        super().setUp()
        self.video_path = "test_video.mp4"
        self.destination_path = "/test/output.png"
        self.width = 200
        self.frame_count = 300
        self.MAX_PROCESSES = 8
        self.MIN_FRAME_COUNT = 100
        # Mock setup for path.exists and access to ensure they return True by default
        patcher_exists = patch("movie_barcodes.utility.path.exists", return_value=True)
        patcher_access = patch("movie_barcodes.utility.access", return_value=True)
        self.addCleanup(patcher_exists.stop)
        self.addCleanup(patcher_access.stop)
        self.patcher_exists = patcher_exists
        self.mock_exists = patcher_exists.start()
        self.mock_access = patcher_access.start()

    def _validate(self, **kwargs) -> None:
        """
        Call validate_inputs with the valid defaults of setUp, overridden by kwargs.
        """
        params = {
            "video_path": self.video_path,
            "frame_count": self.frame_count,
            "min_frame_count": self.MIN_FRAME_COUNT,
            "width": self.width,
            "height": None,
        }
        params.update(kwargs)
        utility.validate_inputs(**params)

    @patch("movie_barcodes.utility.path.exists")
    def test_file_not_found_error(self, mock_exists: MagicMock) -> None:
        """
        Test that validate_inputs raises a FileNotFoundError when the input video file does not exist.
        :param mock_exists: MagicMock object for path.exists function to return False
        :return: None
        """
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self._validate()

    def test_invalid_extension_error(self) -> None:
        """
        Test that validate_inputs raises a ValueError when the input video file has an invalid extension.
        :return: None
        """
        with self.assertRaises(ValueError):
            self._validate(video_path="invalid_extension.txt")

    @patch("movie_barcodes.utility.access")
    def test_destination_path_not_writable(self, mock_access: MagicMock) -> None:
        """
        Test that validate_destination_path raises a PermissionError when the destination path is not writable.
        :param mock_access: MagicMock object for os.access function to return False
        :return: None
        """
        mock_access.return_value = False
        with self.assertRaises(PermissionError):
            utility.validate_destination_path(self.destination_path)

    def test_destination_path_none_is_valid(self) -> None:
        """
        Without destination path (default location), nothing is checked.
        :return: None
        """
        self.mock_access.return_value = False
        utility.validate_destination_path(None)
        self.mock_access.assert_not_called()

    def test_workers_value_error(self) -> None:
        """
        Test that resolve_workers raises a ValueError when the number of workers is lower than 1.
        :return: None
        """
        for workers in (0, -1):
            with self.assertRaises(ValueError):
                utility.resolve_workers(workers, self.MAX_PROCESSES)

    def test_workers_capped_to_max_processes(self) -> None:
        """
        Test that resolve_workers caps workers to MAX_PROCESSES with a warning instead of raising.
        :return: None
        """
        with self.assertLogs(level="WARNING") as logs:
            workers = utility.resolve_workers(self.MAX_PROCESSES * 2, self.MAX_PROCESSES)
        self.assertEqual(workers, self.MAX_PROCESSES)
        self.assertEqual(len(logs.records), 1)
        self.assertIn(
            f"Requested {self.MAX_PROCESSES * 2} workers but only {self.MAX_PROCESSES} CPU cores are available",
            logs.output[0],
        )

    def test_workers_within_limit_unchanged(self) -> None:
        """
        Test that resolve_workers leaves workers untouched and logs nothing when it does not exceed MAX_PROCESSES.
        :return: None
        """
        for workers in (None, 1, self.MAX_PROCESSES):
            with self.assertNoLogs(level="WARNING"):
                self.assertEqual(utility.resolve_workers(workers, self.MAX_PROCESSES), workers)

    def test_invalid_width(self) -> None:
        """
        Test that validate_inputs raises a ValueError when the width is invalid.
        :return: None
        """
        with self.assertRaises(ValueError):
            self._validate(width=0)  # Testing for <= 0

        with self.assertRaises(ValueError):
            self._validate(width=self.frame_count + 1)  # Testing for > frame_count

    def test_invalid_height(self) -> None:
        """
        Test that validate_inputs raises a ValueError when the height is invalid.
        :return: None
        """
        with self.assertRaises(ValueError):
            self._validate(height=0)  # Testing for <= 0

        # Heights greater than frame_count are allowed (no exception expected)
        self._validate(height=self.frame_count + 1)

    def test_frame_count_too_low(self) -> None:
        """
        Test that validate_inputs raises a ValueError when the frame count is too low.
        :return: None
        """
        with self.assertRaises(ValueError):
            self._validate(frame_count=self.MIN_FRAME_COUNT - 1)

    def test_frame_count_too_low_specific_branch(self) -> None:
        """
        Ensure the frame_count < min_frame_count branch in validate_inputs is executed
        by avoiding earlier width/height validations.
        :return: None
        """
        with self.assertRaises(ValueError) as cm:
            self._validate(frame_count=self.MIN_FRAME_COUNT - 1, width=None, height=None)
        self.assertIn(f"at least {self.MIN_FRAME_COUNT} frames", str(cm.exception))

    def test_no_error_raised(self) -> None:
        """
        Test that no error is raised when all arguments are valid.
        :return: None
        """
        self._validate()
        utility.validate_destination_path(self.destination_path)

    def test_destination_in_missing_directory_is_valid(self) -> None:
        """
        A destination inside a directory that does not exist yet is accepted when its closest existing
        ancestor is writable (the directory is created at save time).
        :return: None
        """
        self.patcher_exists.stop()
        with tempfile.TemporaryDirectory() as tmp:
            utility.validate_destination_path(os.path.join(tmp, "new", "nested", "out.png"))
            self.mock_access.assert_called_with(tmp, utility.W_OK)

    @patch("movie_barcodes.utility.makedirs")
    def test_ensure_directory_creates_directory(self, mock_makedirs: MagicMock) -> None:
        """
        Test ensure_directory creates the directory (and does not fail if it already exists).
        :param mock_makedirs: MagicMock object for os.makedirs function
        :return: None
        """
        utility.ensure_directory(self.destination_path)

        mock_makedirs.assert_called_once_with(self.destination_path, exist_ok=True)

    def test_format_time_seconds_only(self) -> None:
        """
        Test that format_time correctly formats times less than 60 seconds.
        :return: None
        """
        time_seconds = 45.0  # 45 seconds
        expected_format = "0m 45s"
        self.assertEqual(utility.format_time(time_seconds), expected_format)

    def test_format_time_minutes_and_seconds(self) -> None:
        """
        Test that format_time correctly formats times between 1 minute and less than 1 hour.
        :return: None
        """
        time_seconds = 95.0  # 1 minute and 35 seconds
        expected_format = "1m 35s"
        self.assertEqual(utility.format_time(time_seconds), expected_format)

    def test_format_time_hours_minutes_seconds(self) -> None:
        """
        Test that format_time correctly formats times with hours, minutes, and seconds.
        :return: None
        """
        time_seconds = 3725.0  # 1 hour, 2 minutes, and 5 seconds
        expected_format = "1h 2m 5s"
        self.assertEqual(utility.format_time(time_seconds), expected_format)

    def test_format_time_rounding(self) -> None:
        """
        Test that format_time correctly rounds seconds.
        :return: None
        """
        time_seconds = 3661.6  # Should round to 1 hour, 1 minute, and 1 seconds
        expected_format = "1h 1m 1s"
        self.assertEqual(utility.format_time(time_seconds), expected_format)

    def test_get_dominant_color_function_linear(self) -> None:
        """
        --linear swaps the avg and smoothed extractors only; the default mapping is unchanged.
        """
        self.assertIs(utility.get_dominant_color_function("avg"), color_extraction.get_dominant_color_mean)
        self.assertIs(utility.get_dominant_color_function("smoothed"), color_extraction.get_smoothed_frame)
        self.assertIs(
            utility.get_dominant_color_function("avg", linear=True), color_extraction.get_dominant_color_mean_linear
        )
        self.assertIs(
            utility.get_dominant_color_function("smoothed", linear=True), color_extraction.get_smoothed_frame_linear
        )
        for method in ("kmeans", "hsv", "bgr"):
            with self.subTest(method=method):
                self.assertIs(
                    utility.get_dominant_color_function(method, linear=True),
                    utility.get_dominant_color_function(method),
                )

    def test_get_dominant_color_function_with_invalid_method(self) -> None:
        """
        Test that get_dominant_color_function raises ValueError for unsupported methods.
        :return: None
        """
        with self.assertRaises(ValueError):
            utility.get_dominant_color_function("unsupported_method")

    def test_get_dominant_color_function_with_valid_methods(self) -> None:
        """
        Test that get_dominant_color_function returns a callable for supported methods.
        :return: None
        """
        valid_methods = ["avg", "kmeans", "hsv", "bgr"]
        for method in valid_methods:
            result = utility.get_dominant_color_function(method)
            self.assertTrue(callable(result), f"Method '{method}' should return a callable.")

    def test_get_dominant_color_function_returns_specific_function(self) -> None:
        """
        Test that get_dominant_color_function returns the specific function associated with a method.
        :return: None
        """
        # Mocking the specific functions to test if the correct one is returned
        utility.get_dominant_color_mean = Mock(name="get_dominant_color_mean")
        utility.get_dominant_color_kmeans = Mock(name="get_dominant_color_kmeans")
        utility.get_dominant_color_hsv = Mock(name="get_dominant_color_hsv")
        utility.get_dominant_color_bgr = Mock(name="get_dominant_color_bgr")
        utility.get_smoothed_frame = Mock(name="get_smoothed_frame")

        self.assertEqual(utility.get_dominant_color_function("avg"), utility.get_dominant_color_mean)
        self.assertEqual(
            utility.get_dominant_color_function("kmeans"),
            utility.get_dominant_color_kmeans,
        )
        self.assertEqual(utility.get_dominant_color_function("hsv"), utility.get_dominant_color_hsv)
        self.assertEqual(utility.get_dominant_color_function("bgr"), utility.get_dominant_color_bgr)
        self.assertEqual(utility.get_dominant_color_function("smoothed"), utility.get_smoothed_frame)

        # Test for invalid method
        with self.assertRaises(ValueError):
            utility.get_dominant_color_function("invalid_method")

    @patch("movie_barcodes.utility.path.getsize")
    def test_get_video_properties(self, mock_getsize: MagicMock) -> None:
        """
        Test that get_video_properties correctly extracts video properties.
        :param mock_getsize: MagicMock object for path.getsize function
        :return: None
        """
        # Setup mock return values
        total_frames = 100
        fps = 25
        video_duration = total_frames / fps
        video_size = 1024000  # in bytes

        # Configure the mock objects
        mock_video = MagicMock()
        mock_video.get.side_effect = [
            total_frames,
            fps,
        ]  # CAP_PROP_FRAME_COUNT, CAP_PROP_FPS
        mock_getsize.return_value = video_size

        # Call the function with the mocked objects/return values
        result = utility.get_video_properties(mock_video, "test/sample.mp4")

        # Assert that the function returns the expected tuple
        expected_result = (total_frames, fps, video_duration, video_size)
        self.assertEqual(result, expected_result)

        # Verify that the mocks were called as expected
        mock_video.get.assert_any_call(cv2.CAP_PROP_FRAME_COUNT)
        mock_video.get.assert_any_call(cv2.CAP_PROP_FPS)
        mock_getsize.assert_called_once_with("test/sample.mp4")

    def _save_in_tmp_cwd(self, image: Image.Image = None, **output_path_kwargs) -> str:
        """
        Save a barcode at get_output_path(...) from a temporary working directory and return that directory.
        """
        if image is None:
            image = Image.new("RGB", (10, 10))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cwd = os.getcwd()
        os.chdir(tmp)
        self.addCleanup(os.chdir, cwd)
        destination_path = utility.get_output_path("video_sample", "avg", "type1", **output_path_kwargs)
        utility.save_barcode_image(image, destination_path)
        return tmp

    def test_save_barcode_image_default_location_is_cwd(self) -> None:
        """
        Without destination_path, the image is saved in ./barcodes relative to the current working directory
        (not relative to the package location, which is site-packages once installed).
        """
        self.patcher_exists.stop()
        tmp = self._save_in_tmp_cwd()
        self.assertTrue(os.path.isfile(os.path.join(tmp, "barcodes", "video_sample_avg_type1.png")))

    def test_save_barcode_image_variations(self) -> None:
        """
        Test file naming variations based on workers and output_name.
        """
        self.patcher_exists.stop()
        tmp = self._save_in_tmp_cwd(workers=4)
        self.assertTrue(os.path.isfile(os.path.join(tmp, "barcodes", "video_sample_avg_type1_workers_4.png")))

        tmp = self._save_in_tmp_cwd(workers=4, output_name="custom_name")
        self.assertTrue(os.path.isfile(os.path.join(tmp, "barcodes", "custom_name.png")))

    def test_get_output_path(self) -> None:
        """
        destination_path wins over output_name, which wins over the generated name.
        """
        self.assertEqual(
            utility.get_output_path("movie", "kmeans", "circular"),
            os.path.join("barcodes", "movie_kmeans_circular.png"),
        )
        self.assertEqual(
            utility.get_output_path("movie", "kmeans", "circular", workers=2),
            os.path.join("barcodes", "movie_kmeans_circular_workers_2.png"),
        )
        self.assertEqual(
            utility.get_output_path("movie", "avg", "horizontal", output_name="x"), os.path.join("barcodes", "x.png")
        )
        self.assertEqual(
            utility.get_output_path("movie", "avg", "horizontal", output_name="x", destination_path="d/out.png"),
            "d/out.png",
        )

    def test_get_output_path_with_method_suffix(self) -> None:
        """
        With several methods (--all_methods), a user-chosen name or path gets the method appended, so the images
        don't overwrite each other; the generated name already contains the method.
        """
        self.assertEqual(
            utility.get_output_path("movie", "kmeans", "horizontal", output_name="x", method_suffix=True),
            os.path.join("barcodes", "x_kmeans.png"),
        )
        self.assertEqual(
            utility.get_output_path("movie", "avg", "horizontal", destination_path="d/out.png", method_suffix=True),
            os.path.join("d", "out_avg.png"),
        )
        self.assertEqual(
            utility.get_output_path("movie", "avg_linear", "horizontal", destination_path="out", method_suffix=True),
            "out_avg_linear",
        )
        self.assertEqual(
            utility.get_output_path("movie", "avg", "circular", method_suffix=True),
            os.path.join("barcodes", "movie_avg_circular.png"),
        )

    def test_save_barcode_image_with_relative_destination_path(self) -> None:
        """
        A relative destination_path is resolved relative to the current working directory, and missing
        parent directories are created.
        """
        self.patcher_exists.stop()
        tmp = self._save_in_tmp_cwd(destination_path="relative/output.png")
        self.assertTrue(os.path.isfile(os.path.join(tmp, "relative", "output.png")))

    def test_barcode_to_image_converts_bgr_to_rgb(self) -> None:
        """
        Barcodes are BGR(A) internally and must become RGB(A) images.
        """
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # pure blue in BGR
        image = utility.barcode_to_image(bgr)
        self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(image)[0, 0], [0, 0, 255])

        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        bgra[..., 0] = 255
        bgra[..., 3] = 255
        image = utility.barcode_to_image(bgra)
        self.assertEqual(image.mode, "RGBA")
        np.testing.assert_array_equal(np.asarray(image)[0, 0], [0, 0, 255, 255])

    def test_save_barcode_image_keeps_pixels(self) -> None:
        """
        The saved file holds exactly the pixels of the image.
        """
        self.patcher_exists.stop()
        image = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
        tmp = self._save_in_tmp_cwd(image, destination_path="out.png")
        with Image.open(os.path.join(tmp, "out.png")) as saved:
            self.assertEqual(saved.mode, "RGBA")
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(image))


if __name__ == "__main__":
    unittest.main()
