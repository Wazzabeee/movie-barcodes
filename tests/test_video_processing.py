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

    @patch("cv2.VideoCapture")
    def test_extract_colors(self, mock_video: MagicMock) -> None:
        """
        Test extract_colors returns the expected colors.
        :param mock_video: MagicMock object for cv2.VideoCapture
        :return: None
        """
        # Mock the read method of VideoCapture
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, "frame_data")] * 100  # Simulate 100 frames

        # Mock color_extractor to return a predictable color
        def mock_color_extractor(frame):
            return "red" if frame == "frame_data" else "unknown"

        # Test parameters
        video_path = "dummy_video.mp4"
        start_frame = 0
        end_frame = 99  # 100 frames total
        target_frames = 10

        # Expected result
        expected_colors = ["red"] * target_frames

        # Run the test
        actual_colors = video_processing.extract_colors(
            video_path, start_frame, end_frame, mock_color_extractor, target_frames
        )

        # Verify the result
        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skipping_logic(self, mock_video: MagicMock) -> None:
        mock_video_instance = mock_video.return_value

        # Initialize a counter to keep track of the current frame position
        current_frame = [0]  # Use a list so it can be modified within the side_effect functions

        # Define a side effect for the read method
        def read_side_effect():
            if current_frame[0] < 50:  # Assuming there are 50 frames available for simplicity
                frame_data = (True, f"frame_data_{current_frame[0]}")
                current_frame[0] += 1
                return frame_data
            else:
                return False, None  # Simulate end of video

        # Define a side effect for the grab method to simply increment the counter
        def grab_side_effect():
            if current_frame[0] < 50:
                current_frame[0] += 1

        mock_video_instance.read.side_effect = read_side_effect
        mock_video_instance.grab.side_effect = grab_side_effect

        # Mock color_extractor to return the frame data
        def mock_color_extractor(frame):
            return frame  # Here, frame should be the unique frame identifier like 'frame_data_0'

        video_path = "dummy_video.mp4"
        start_frame = 0
        end_frame = 49
        target_frames = 5

        expected_colors = [f"frame_data_{i*10}" for i in range(target_frames)]

        actual_colors = video_processing.extract_colors(
            video_path, start_frame, end_frame, mock_color_extractor, target_frames
        )

        self.assertEqual(actual_colors, expected_colors)

    @patch("cv2.VideoCapture")
    def test_frame_skip_default(self, mock_video):
        # Setup mock to simulate 10 frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(10)] + [(False, None)]

        # Mock color_extractor to simply return the frame data
        def mock_color_extractor(frame):
            return frame

        video_path = "dummy_video.mp4"

        # Call without specifying target_frames
        colors = video_processing.extract_colors(video_path, 0, 9, mock_color_extractor)

        # Assert frame_skip was effectively 1 by checking all frames were processed
        self.assertEqual(len(colors), 10)

    @patch("cv2.VideoCapture")
    def test_frame_skip_with_target_frames(self, mock_video):
        # Setup mock to simulate 50 frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(50)] + [(False, None)]

        # Mock color_extractor to simply return the frame data
        def mock_color_extractor(frame):
            return frame

        video_path = "dummy_video.mp4"
        target_frames = 5

        colors = video_processing.extract_colors(video_path, 0, 49, mock_color_extractor, target_frames)

        # Assert that we only processed the specified number of target_frames
        self.assertEqual(len(colors), target_frames)

    @patch("cv2.VideoCapture")
    def test_len_colors_equals_target_frames(self, mock_video):
        # This test overlaps with test_frame_skip_with_target_frames
        pass  # Already covered by test_frame_skip_with_target_frames

    @patch("cv2.VideoCapture")
    def test_len_colors_equals_len_video_no_target_frames(self, mock_video):
        # Setup mock to simulate 20 frames in the video
        mock_video_instance = mock_video.return_value
        mock_video_instance.read.side_effect = [(True, f"frame_{i}") for i in range(20)] + [(False, None)]

        # Mock color_extractor to simply return the frame data
        def mock_color_extractor(frame):
            return frame

        video_path = "dummy_video.mp4"

        colors = video_processing.extract_colors(video_path, 0, 19, mock_color_extractor)

        # Assert that all frames were processed since target_frames was not given
        self.assertEqual(len(colors), 20)


if __name__ == "__main__":
    unittest.main()
