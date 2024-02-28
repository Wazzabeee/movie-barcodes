import unittest
from unittest.mock import patch, MagicMock
from src import video_processing


class TestVideoProcessing(unittest.TestCase):
    """
    Test the video processing functions.
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        :return: None
        """
        self.video_path = "dummy_video.mp4"
        self.start_frame = 0
        self.end_frame = 49
        self.target_frames = 5

    @staticmethod
    def mock_color_extractor(frame):
        """
        Mock color extractor function.
        :param frame:
        :return: frame
        """
        return frame

    @patch("src.video_processing.cv2.VideoCapture")
    def test_load_video_raises_error_on_file_not_open(self, mock_video: MagicMock) -> None:
        """
        Test load_video raises ValueError when the video file cannot be opened.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video.return_value.isOpened.return_value = False
        with self.assertRaises(ValueError) as context:
            video_processing.load_video(self.video_path)
        self.assertIn(
            "Could not open the video file: " + self.video_path,
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
            video_processing.load_video(self.video_path)
        self.assertIn(
            "The video file " + self.video_path + " has no frames.",
            str(context.exception),
        )

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
            video_processing.load_video(self.video_path)
        self.assertIn(
            "The video file " + self.video_path + " has invalid dimensions.",
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
        video, frame_count, frame_width, frame_height = video_processing.load_video(self.video_path)
        self.assertEqual(frame_count, 100)
        self.assertEqual(frame_width, 1920)
        self.assertEqual(frame_height, 1080)

    @patch("cv2.VideoCapture")
    def test_extract_colors(self, mock_video: MagicMock) -> None:
        """
        Test extract_colors returns the expected colors.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Mock the read method of VideoCapture
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, "frame_data")] * self.target_frames  # Simulate frames

        # Mock color_extractor to return a predictable color
        def special_color_extractor(frame):
            return "red" if frame == "frame_data" else "unknown"

        # Expected result
        expected_colors = ["red"] * self.target_frames

        # Run the test
        actual_colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            special_color_extractor,
            self.target_frames,
        )

        # Verify the result
        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skipping_logic(self, mock_video: MagicMock) -> None:
        """
        Test that the frame skipping logic is working as expected.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        mock_video_instance = mock_video.return_value

        # Initialize a counter to keep track of the current frame position
        current_frame = [0]  # Use a list so it can be modified within the side_effect functions

        # Define a side effect for the read method
        def read_side_effect():
            if current_frame[0] < self.end_frame:
                frame_data = (True, f"frame_data_{current_frame[0]}")
                current_frame[0] += 1
                return frame_data
            else:
                return False, None  # Simulate end of video

        # Define a side effect for the grab method to simply increment the counter
        def grab_side_effect():
            if current_frame[0] < self.end_frame:
                current_frame[0] += 1

        mock_video_instance.read.side_effect = read_side_effect
        mock_video_instance.grab.side_effect = grab_side_effect

        expected_colors = [f"frame_data_{i*10}" for i in range(self.target_frames)]

        actual_colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            self.mock_color_extractor,
            self.target_frames,
        )

        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skip_default(self, mock_video: MagicMock) -> None:
        """
        Test that the default frame_skip is 1.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        # Call without specifying target_frames
        colors = video_processing.extract_colors(
            self.video_path, self.start_frame, self.end_frame, self.mock_color_extractor
        )

        # Assert frame_skip was effectively 1 by checking all frames were processed
        self.assertEqual(len(colors), self.end_frame)

    @patch("cv2.VideoCapture")
    def test_frame_skip_with_target_frames(self, mock_video: MagicMock) -> None:
        """
        Test that target_frames is respected when extracting colors.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        colors = video_processing.extract_colors(
            self.video_path,
            self.start_frame,
            self.end_frame,
            self.mock_color_extractor,
            self.target_frames,
        )

        # Assert that we only processed the specified number of target_frames
        self.assertEqual(len(colors), self.target_frames)

    @patch("cv2.VideoCapture")
    def test_len_colors_equals_len_video_no_target_frames(self, mock_video: MagicMock) -> None:
        """
        Test that the number of colors extracted is equal to the number of frames in the video
         when target_frames is not given.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Setup mock to simulate frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(self.end_frame)] + [(False, None)]

        colors = video_processing.extract_colors(
            self.video_path, self.start_frame, self.end_frame, self.mock_color_extractor
        )

        # Assert that all frames were processed since target_frames was not given
        self.assertEqual(len(colors), self.end_frame)


if __name__ == "__main__":
    unittest.main()
