import unittest
import argparse
from unittest.mock import patch

from src import utility


class TestUtility(unittest.TestCase):
    """
    Test the utility functions.
    """

    def setUp(self):
        """
        Set up the test case.
        :return:
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
    def test_file_not_found_error(self, mock_exists):
        """
        Test that validate_args raises a FileNotFoundError when the input video file does not exist.
        :param mock_exists: False
        """
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    def test_invalid_extension_error(self, mock_exists):
        """
        Test that validate_args raises a ValueError when the input video file has an invalid extension.
        :param mock_exists: True
        """
        mock_exists.return_value = True
        self.args.input_video_path = "invalid_extension.txt"
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    @patch("src.utility.access")
    def test_destination_path_not_writable(self, mock_access, mock_exists):
        """
        Test that validate_args raises a PermissionError when the destination path is not writable.
        :param mock_access: True
        :param mock_exists: True
        """
        mock_exists.return_value = True
        mock_access.return_value = False
        with self.assertRaises(PermissionError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    @patch("src.utility.path.exists")
    def test_workers_value_error(self, mock_exists):
        """
        Test that validate_args raises a ValueError when the number of workers is invalid.
        :param mock_exists: True
        """
        mock_exists.return_value = True
        self.args.workers = 0  # Testing for < 1
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

        self.args.workers = self.MAX_PROCESSES + 1  # Testing for > MAX_PROCESSES
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_invalid_width(self):
        """
        Test that validate_args raises a ValueError when the width is invalid.
        """
        self.args.width = 0  # Testing for <= 0
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

        self.args.width = self.frame_count + 1  # Testing for > frame_count
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_frame_count_too_low(self):
        """
        Test that validate_args raises a ValueError when the frame count is too low.
        """
        low_frame_count = self.MIN_FRAME_COUNT - 1
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, low_frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_all_methods_and_method_error(self):
        """
        Test that validate_args raises a ValueError when the --all_methods flag is used with the --method argument.
        """
        self.args.all_methods = True
        self.args.method = "avg"  # Explicitly setting method to simulate conflict
        with self.assertRaises(ValueError):
            utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)

    def test_no_error_raised(self):
        """
        Test that no error is raised when all arguments are valid.
        """
        utility.validate_args(self.args, self.frame_count, self.MAX_PROCESSES, self.MIN_FRAME_COUNT)


if __name__ == "__main__":
    unittest.main()
