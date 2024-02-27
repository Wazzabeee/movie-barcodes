import unittest
from unittest.mock import patch, MagicMock
from src import video_processing


class TestVideoProcessing(unittest.TestCase):
    """
    Test the video processing functions.
    """

    @patch("src.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_file_not_open(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when the video file cannot be opened.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = False
        with self.assertRaises(ValueError) as context:
            video_processing.load_video("nonexistent_file.mp4")
        self.assertIn(
            "Could not open the video file: nonexistent_file.mp4",
            str(context.exception),
        )

    @patch("src.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_no_frames(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when the video has no frames.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.return_value = 0  # Mocking frame count as 0
        with self.assertRaises(ValueError) as context:
            video_processing.load_video("empty_video.mp4")
        self.assertIn("The video file empty_video.mp4 has no frames.", str(context.exception))

    @patch("src.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_invalid_dimensions(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when video has invalid dimensions.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.side_effect = [
            1,
            0,
            0,
        ]  # Mocking frame count as 1 and dimensions as 0
        with self.assertRaises(ValueError) as context:
            video_processing.load_video("invalid_dimensions.mp4")
        self.assertIn(
            "The video file invalid_dimensions.mp4 has invalid dimensions.",
            str(context.exception),
        )

    @patch("src.video_processing.cv2.VideoCapture")
    def test_load_video_success(self, mock_video: MagicMock) -> None:
        """
        Test load_video successfully returns video properties.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = True
        mock_video.return_value.get.side_effect = [
            100,
            1920,
            1080,
        ]  # Mocking frame count, width, and height
        video, frame_count, frame_width, frame_height = video_processing.load_video("valid_video.mp4")
        self.assertEqual(frame_count, 100)
        self.assertEqual(frame_width, 1920)
        self.assertEqual(frame_height, 1080)


if __name__ == "__main__":
    unittest.main()
