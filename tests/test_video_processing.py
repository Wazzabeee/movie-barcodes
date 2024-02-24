import unittest
from unittest.mock import Mock
import numpy as np
from src import video_processing


class TestVideoProcessing(unittest.TestCase):
    """
    Test the video processing functions.
    """

    def test_load_video(self):
        """
        Test the load_video function.
        :return:
        """
        video_path = "sample.mp4"
        video, frame_count, frame_width, frame_height = video_processing.load_video(video_path)

        self.assertIsNotNone(video)
        self.assertIsInstance(frame_count, int)
        self.assertIsInstance(frame_width, int)
        self.assertIsInstance(frame_height, int)

    def test_process_frame_chunk(self):
        """
        Test the process_frame_chunk function.
        :return:
        """
        chunk_frames = [np.ones((10, 10, 3), dtype=np.uint8) * color for color in range(2)]

        def color_extractor(frame):
            return np.mean(frame, axis=(0, 1)).mean()

        colors = video_processing.process_frame_chunk(chunk_frames, color_extractor)

        self.assertEqual(len(colors), 2)
        self.assertTrue(all(isinstance(color, float) for color in colors))

    def test_extract_colors(self):
        """
        Test the extract_colors function.
        :return:
        """
        # Setup Mock video object with a read method that returns a tuple
        video = Mock()
        video.read = Mock(return_value=(True, "mock_frame"))

        frame_count = 10
        color_extractor = Mock(return_value=np.array([255, 255, 255]))

        colors = video_processing.extract_colors(video, frame_count, color_extractor)

        self.assertEqual(len(colors), frame_count)


if __name__ == "__main__":
    unittest.main()
