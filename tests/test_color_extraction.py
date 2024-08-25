import unittest
import numpy as np
from src import (
    color_extraction,
)


class TestColorExtraction(unittest.TestCase):
    """
    Test the color extraction functions.
    """

    def setUp(self) -> None:
        self.frame = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)

    def test_get_dominant_color_mean(self) -> None:
        """
        Test the get_dominant_color_mean function.
        :return: None
        """
        # Call your function with the sample frame
        dominant_color = color_extraction.get_dominant_color_mean(self.frame)

        # Assert the expected result
        self.assertEqual(dominant_color.tolist(), [127.5, 127.5, 127.5])

    def test_get_dominant_color_kmeans(self) -> None:
        """
        Test the get_dominant_color_kmeans function.
        :return: None
        """
        dominant_color = color_extraction.get_dominant_color_kmeans(self.frame)

        self.assertIsInstance(dominant_color, np.ndarray)
        self.assertEqual(dominant_color.shape, (3,))  # Should be a 3-element array representing a color

    def test_get_smoothed_frame(self) -> None:
        """
        Test the get_smoothed_frame function.
        :return: None
        """
        # Call the function with the sample frame
        smoothed_frame = color_extraction.get_smoothed_frame(self.frame)

        # Assert the expected shape
        self.assertEqual(smoothed_frame.shape, (2, 1, 3))

        # Assert that the smoothed frame is the average of all pixels, rounded to nearest integer
        expected_color = np.round(np.mean(self.frame, axis=(0, 1))).astype(np.uint8)
        np.testing.assert_array_equal(smoothed_frame[0, 0], expected_color)
        np.testing.assert_array_equal(smoothed_frame[1, 0], expected_color)

        # Also check that all rows are the same
        np.testing.assert_array_equal(smoothed_frame[0, 0], smoothed_frame[1, 0])

    def test_get_smoothed_frame_different_sizes(self) -> None:
        """
        Test the get_smoothed_frame function with different frame sizes.
        :return: None
        """
        # Test with a larger frame
        large_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        smoothed_large = color_extraction.get_smoothed_frame(large_frame)
        self.assertEqual(smoothed_large.shape, (100, 1, 3))

        # Test with a single-pixel frame
        single_pixel = np.array([[[123, 45, 67]]], dtype=np.uint8)
        smoothed_single = color_extraction.get_smoothed_frame(single_pixel)
        self.assertEqual(smoothed_single.shape, (1, 1, 3))
        np.testing.assert_array_equal(smoothed_single[0, 0], single_pixel[0, 0])

    def test_get_smoothed_frame_uniform_color(self) -> None:
        """
        Test the get_smoothed_frame function with a uniform color frame.
        :return: None
        """
        uniform_frame = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)
        smoothed_uniform = color_extraction.get_smoothed_frame(uniform_frame)
        self.assertEqual(smoothed_uniform.shape, (50, 1, 3))
        np.testing.assert_array_equal(smoothed_uniform[0, 0], [100, 150, 200])


if __name__ == "__main__":
    unittest.main()
