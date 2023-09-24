import unittest
from unittest.mock import Mock
from src import video_processing  # Adjust the import as per your project structure and naming
import numpy as np


class TestVideoProcessing(unittest.TestCase):

    def test_load_video(self):
        video_path = 'sample.mp4'
        video, frame_count, frame_width, frame_height = video_processing.load_video(video_path)

        self.assertIsNotNone(video)
        self.assertIsInstance(frame_count, int)
        self.assertIsInstance(frame_width, int)
        self.assertIsInstance(frame_height, int)

    def test_process_frame_chunk(self):
        chunk_frames = [np.ones((10, 10, 3), dtype=np.uint8) * color for color in range(2)]
        color_extractor = lambda frame: np.mean(frame, axis=(0, 1)).mean()  # Adjusted color_extractor

        colors = video_processing.process_frame_chunk(chunk_frames, color_extractor)

        self.assertEqual(len(colors), 2)
        self.assertTrue(all(isinstance(color, float) for color in colors))  # Adjusted assertion

    def test_extract_colors(self):
        # Setup Mock video object with a read method that returns a tuple
        video = Mock()
        video.read = Mock(return_value=(True, "mock_frame"))

        frame_count = 10
        color_extractor = Mock(return_value=np.array([255, 255, 255]))

        colors = video_processing.extract_colors(video, frame_count, color_extractor)

        self.assertEqual(len(colors), frame_count)


if __name__ == '__main__':
    unittest.main()
