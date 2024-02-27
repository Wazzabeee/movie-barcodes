import unittest
import argparse
from unittest.mock import patch, MagicMock, Mock
import cv2
import numpy as np


from src import utility


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
        self.args = argparse.Namespace(
            input_video_path="test_video.mp4",
            destination_path="/test/output.png",
            workers=4,
            width=200,
            all_methods=False,
            method="avg",
        )
        self.frame_count = 300
        self.MAX_PROCESSES = 8
        self.MIN_FRAME_COUNT = 100
        # Mock setup for path.exists and access to ensure they return True by default
        patcher_exists = patch("src.utility.path.exists", return_value=True)
        patcher_access = patch("src.utility.access", return_value=True)
        self.addCleanup(patcher_exists.stop)
        self.addCleanup(patcher_access.stop)
        self.mock_exists = patcher_exists.start()
        self.mock_access = patcher_access.start()

    @patch("src.utility.path.exists")
    def test_file_not_found_error(self, mock_exists: MagicMock) -> None:
        """
        Test that validate_args raises a FileNotFoundError when the input video file does not exist.
        :param mock_exists: MagicMock object for path.exists function to return False
        :return: None
        """
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    def test_invalid_extension_error(self, mock_exists: MagicMock) -> None:
        """
        Test that validate_args raises a ValueError when the input video file has an invalid extension.
        :param mock_exists: MagicMock object for path.exists function to return True
        :return: None
        """
        mock_exists.return_value = True
        self.args.input_video_path = "invalid_extension.txt"
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    @patch("src.utility.access")
    def test_destination_path_not_writable(self, mock_access: MagicMock, mock_exists: MagicMock) -> None:
        """
        Test that validate_args raises a PermissionError when the destination path is not writable.
        :param mock_access: MagicMock object for os.access function to return False
        :param mock_exists: MagicMock object for path.exists function to return True
        :return: None
        """
        mock_exists.return_value = True
        mock_access.return_value = False
        with self.assertRaises(PermissionError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    def test_workers_value_error(self, mock_exists: MagicMock) -> None:
        """
        Test that validate_args raises a ValueError when the number of workers is invalid.
        :param mock_exists: MagicMock object for path.exists function to return True
        :return: None
        """
        mock_exists.return_value = True
        self.args.workers = 0  # Testing for < 1
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

        self.args.workers = self.MAX_PROCESSES + 1  # Testing for > MAX_PROCESSES
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_invalid_width(self) -> None:
        """
        Test that validate_args raises a ValueError when the width is invalid.
        :return: None
        """
        self.args.width = 0  # Testing for <= 0
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

        self.args.width = self.frame_count + 1  # Testing for > frame_count
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_frame_count_too_low(self) -> None:
        """
        Test that validate_args raises a ValueError when the frame count is too low.
        :return: None
        """
        low_frame_count = self.MIN_FRAME_COUNT - 1
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, low_frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_all_methods_and_method_error(self) -> None:
        """
        Test that validate_args raises a ValueError when the --all_methods flag is used with the --method argument.
        :return: None
        """
        self.args.all_methods = True
        self.args.method = "avg"  # Explicitly setting method to simulate conflict
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_no_error_raised(self) -> None:
        """
        Test that no error is raised when all arguments are valid.
        :return: None
        """
        utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.makedirs")
    @patch("src.utility.path.exists")
    def test_ensure_directory_creates_directory(self, mock_exists: MagicMock, mock_makedirs: MagicMock) -> None:
        """
        Test ensure_directory creates the directory when it does not exist.
        :param mock_exists: MagicMock object for path.exists function to return False
        :param mock_makedirs: MagicMock object for os.makedirs function
        :return: None
        """
        mock_exists.return_value = False  # Simulate directory does not exist

        utility.ensure_directory(self.args.destination_path)

        mock_exists.assert_called_once_with(self.args.destination_path)
        mock_makedirs.assert_called_once_with(self.args.destination_path)

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

        self.assertEqual(utility.get_dominant_color_function("avg"), utility.get_dominant_color_mean)
        self.assertEqual(
            utility.get_dominant_color_function("kmeans"),
            utility.get_dominant_color_kmeans,
        )
        self.assertEqual(utility.get_dominant_color_function("hsv"), utility.get_dominant_color_hsv)
        self.assertEqual(utility.get_dominant_color_function("bgr"), utility.get_dominant_color_bgr)

    @patch("src.utility.path.getsize")
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

        args = argparse.Namespace(input_video_path="test/sample.mp4")

        # Call the function with the mocked objects/return values
        result = utility.get_video_properties(mock_video, args)

        # Assert that the function returns the expected tuple
        expected_result = (total_frames, fps, video_duration, video_size)
        self.assertEqual(result, expected_result)

        # Verify that the mocks were called as expected
        mock_video.get.assert_any_call(cv2.CAP_PROP_FRAME_COUNT)
        mock_video.get.assert_any_call(cv2.CAP_PROP_FPS)
        mock_getsize.assert_called_once_with(args.input_video_path)

    @patch("src.utility.path.join")
    @patch("src.utility.Image.fromarray")
    @patch("src.utility.path.dirname")
    @patch("src.utility.path.abspath")
    def test_save_barcode_image_variations(
        self,
        mock_abspath: MagicMock,
        mock_dirname: MagicMock,
        mock_fromarray: MagicMock,
        mock_path_join: MagicMock,
    ) -> None:
        """
        Test variations of save_barcode_image behavior based on different argument conditions.
        :param mock_abspath: MagicMock object for path.abspath function
        :param mock_dirname: MagicMock object for path.dirname function
        :param mock_fromarray: MagicMock object for Image.fromarray function
        :param mock_path_join: MagicMock object for path.join function
        :return: None
        """
        # Mock the directory and path handling
        mock_abspath.return_value = "/fake/dir/module.py"
        mock_dirname.side_effect = lambda x: x.rsplit("/", 1)[0]  # Simulates dirname behavior
        mock_path_join.side_effect = lambda *args: "/".join(args)  # Simulates os.path.join behavior

        # Setup a fake barcode and base name
        barcode = np.zeros((100, 100, 3), dtype=np.uint8)
        base_name = "video_sample"

        # Test without workers and without output name
        args_without_workers = argparse.Namespace(
            destination_path=None, output_name=None, barcode_type="type1", workers=None
        )
        utility.save_barcode_image(barcode, base_name, args_without_workers, "avg")
        self.assertTrue("workers_" not in mock_path_join.call_args[0][-1])

        # Test with workers
        args_with_workers = argparse.Namespace(destination_path=None, output_name=None, barcode_type="type1", workers=4)
        utility.save_barcode_image(barcode, base_name, args_with_workers, "avg")
        self.assertTrue("workers_4" in mock_path_join.call_args[0][-1])

        # Test with output name
        args_with_output_name = argparse.Namespace(
            destination_path=None,
            output_name="custom_name",
            barcode_type="type1",
            workers=None,
        )
        utility.save_barcode_image(barcode, base_name, args_with_output_name, "avg")
        self.assertIn("custom_name.png", mock_path_join.call_args[0][-1])

        # Test file naming without output name
        utility.save_barcode_image(barcode, base_name, args_without_workers, "avg")
        expected_name_parts = [base_name, "avg", "type1"]
        expected_name = "_".join(expected_name_parts) + ".png"
        self.assertIn(expected_name, mock_path_join.call_args[0][-1])

        # Test image saving
        mock_image = mock_fromarray.return_value
        utility.save_barcode_image(barcode, base_name, args_without_workers, "avg")
        mock_image.save.assert_called()  # Ensure the image is attempted to be saved


if __name__ == "__main__":
    unittest.main()
