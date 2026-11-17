import unittest
import numpy as np
from movie_barcodes import (
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

    def test_get_smoothed_frame_averages_whole_frame(self) -> None:
        """
        The smoothed column must be the mean of the whole frame, not the color of the center pixels.
        """
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        frame[:, :96] = (255, 0, 0)
        frame[:, 96:] = (0, 0, 255)
        frame[50:58, 92:100] = (0, 255, 0)  # small green patch right at the center
        smoothed = color_extraction.get_smoothed_frame(frame)
        np.testing.assert_allclose(smoothed[0, 0], color_extraction.get_dominant_color_mean(frame), atol=1)

    def test_get_dominant_color_hsv_neutral_frames_are_not_red(self) -> None:
        """
        Black, gray and white frames have no hue: they must not be turned into pure red.
        """
        for value in (0, 128, 255):
            frame = np.full((20, 20, 3), value, dtype=np.uint8)
            np.testing.assert_array_equal(color_extraction.get_dominant_color_hsv(frame), [value] * 3)

    def test_get_dominant_color_hsv_ignores_black_borders(self) -> None:
        """
        Letterbox black bars must not vote for the hue: a blue picture with large black borders stays blue.
        """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[30:70] = (200, 50, 0)  # blue in BGR, 40% of the frame
        b, g, r = color_extraction.get_dominant_color_hsv(frame)
        self.assertGreater(b, r)
        self.assertGreater(b, g)

    def test_get_dominant_color_kmeans_large_frame(self) -> None:
        """
        K-means works on full HD frames (downscaled internally) and finds the majority color.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :1400] = (10, 200, 30)
        frame[:, 1400:] = (250, 250, 250)
        np.testing.assert_allclose(color_extraction.get_dominant_color_kmeans(frame), [10, 200, 30], atol=2)

    def test_get_dominant_color_kmeans_uniform_frame(self) -> None:
        """
        A uniform frame has fewer distinct colors than clusters; K-means must still return that color.
        """
        frame = np.full((50, 50, 3), (1, 2, 3), dtype=np.uint8)
        np.testing.assert_allclose(color_extraction.get_dominant_color_kmeans(frame), [1, 2, 3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
