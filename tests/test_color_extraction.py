import unittest
import cv2
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

    def test_get_dominant_color_bgr_output_type(self) -> None:
        """
        The bgr method returns 3 uint8 values in BGR order.
        """
        dominant_color = color_extraction.get_dominant_color_bgr(self.frame)
        self.assertIsInstance(dominant_color, np.ndarray)
        self.assertEqual(dominant_color.shape, (3,))
        self.assertEqual(dominant_color.dtype, np.uint8)

    def test_get_dominant_color_bgr_uniform_frame(self) -> None:
        """
        A single-color frame returns exactly that color.
        """
        frame = np.full((50, 50, 3), (12, 130, 240), dtype=np.uint8)
        np.testing.assert_array_equal(color_extraction.get_dominant_color_bgr(frame), [12, 130, 240])

    def test_get_dominant_color_bgr_does_not_invent_colors(self) -> None:
        """
        A per-channel argmax would combine peaks from different pixels: with 40% blue, 30% green and 30% red,
        each channel is 0 for most pixels and the old method returned black, which is not in the frame.
        """
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:4] = (255, 0, 0)
        frame[4:7] = (0, 255, 0)
        frame[7:] = (0, 0, 255)
        np.testing.assert_array_equal(color_extraction.get_dominant_color_bgr(frame), [255, 0, 0])

    def test_get_dominant_color_bgr_majority_wins(self) -> None:
        """
        The majority color wins over several smaller groups of other colors.
        """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :34] = (30, 90, 160)
        minorities = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (128, 128, 128), (250, 250, 250), (0, 0, 0)]
        for i, color in enumerate(minorities):
            frame[:, 34 + i * 11 : 34 + (i + 1) * 11] = color
        np.testing.assert_array_equal(color_extraction.get_dominant_color_bgr(frame), [30, 90, 160])

    def test_get_dominant_color_bgr_returns_mean_of_bin(self) -> None:
        """
        The result is the mean of the pixels in the winning bin, not the center of the bin.
        """
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:5] = (97, 97, 97)
        frame[5:] = (99, 99, 99)
        np.testing.assert_array_equal(color_extraction.get_dominant_color_bgr(frame), [98, 98, 98])

    def test_get_dominant_color_bgr_large_frame_keeps_real_colors(self) -> None:
        """
        Subsampling a full HD frame must not blend fine patterns: 1px blue/red stripes give blue or red, not purple.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, 0::2] = (255, 0, 0)
        frame[:, 1::2] = (0, 0, 255)
        self.assertIn(color_extraction.get_dominant_color_bgr(frame).tolist(), [[255, 0, 0], [0, 0, 255]])

    def test_get_dominant_color_bgr_large_frame_majority(self) -> None:
        """
        The majority color of a full HD frame is found after subsampling.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :1400] = (10, 200, 30)
        frame[:, 1400:] = (250, 250, 250)
        np.testing.assert_array_equal(color_extraction.get_dominant_color_bgr(frame), [10, 200, 30])


class TestLinearLightAveraging(unittest.TestCase):
    """
    Averaging colors in linear light (--linear) for the avg and smoothed methods.
    """

    def test_srgb_linear_round_trip_is_identity(self) -> None:
        values = np.arange(256)
        decoded = color_extraction.SRGB_TO_LINEAR_LUT[values]
        encoded = np.round(color_extraction.linear_to_srgb(decoded) * 255).astype(int)
        np.testing.assert_array_equal(encoded, values)

    def test_transfer_function_reference_points(self) -> None:
        lut = color_extraction.SRGB_TO_LINEAR_LUT
        self.assertEqual(lut[0], 0.0)
        self.assertAlmostEqual(lut[255], 1.0)
        self.assertAlmostEqual(lut[128], 0.2158605, places=6)
        self.assertTrue(np.all(np.diff(lut) > 0))

    def test_half_black_half_white(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, 100:] = 255
        np.testing.assert_array_equal(color_extraction.get_dominant_color_mean_linear(frame), [188, 188, 188])
        np.testing.assert_array_equal(color_extraction.get_dominant_color_mean(frame), [127.5, 127.5, 127.5])

    def test_one_pixel_checkerboard(self) -> None:
        """
        Fine high-contrast detail must also be averaged in linear light (no gamma averaging before decoding).
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[0::2, 0::2] = 255
        frame[1::2, 1::2] = 255
        np.testing.assert_array_equal(color_extraction.get_dominant_color_mean_linear(frame), [188, 188, 188])

    def test_uniform_frames_unchanged(self) -> None:
        for color in ([0, 0, 0], [255, 255, 255], [100, 150, 200], [1, 128, 254]):
            with self.subTest(color=color):
                frame = np.full((30, 40, 3), color, dtype=np.uint8)
                np.testing.assert_array_equal(color_extraction.get_dominant_color_mean_linear(frame), color)
                smoothed = color_extraction.get_smoothed_frame_linear(frame)
                self.assertEqual(smoothed.shape, (30, 1, 3))
                self.assertEqual(smoothed.dtype, np.uint8)
                np.testing.assert_array_equal(smoothed[:, 0], np.tile(color, (30, 1)))

    def test_matches_float_reference_and_keeps_channel_order(self) -> None:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
        frame[..., 0] //= 4  # Distinct channels, to catch a BGR / RGB swap
        linear = color_extraction.srgb_to_linear(frame / 255.0).reshape(-1, 3).mean(axis=0)
        expected = np.round(color_extraction.linear_to_srgb(linear) * 255)
        np.testing.assert_array_equal(color_extraction.get_dominant_color_mean_linear(frame), expected)

    def test_non_contiguous_frame(self) -> None:
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:, 20:60] = (10, 120, 240)
        cropped = frame[:, 20:60]  # Non-contiguous view, as with --crop-borders
        np.testing.assert_array_equal(color_extraction.get_dominant_color_mean_linear(cropped), [10, 120, 240])

    def test_smoothed_linear_equals_linear_mean(self) -> None:
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
        smoothed = color_extraction.get_smoothed_frame_linear(frame)
        self.assertEqual(smoothed.shape, (72, 1, 3))
        mean = color_extraction.get_dominant_color_mean_linear(frame)
        np.testing.assert_array_equal(smoothed[:, 0], np.tile(mean, (72, 1)))

        # Same result as area-resizing the frame decoded to linear floats, as get_smoothed_frame does in sRGB
        linear_frame = color_extraction.srgb_to_linear(frame / 255.0).astype(np.float32)
        pixel = cv2.resize(linear_frame, (1, 1), interpolation=cv2.INTER_AREA)[0, 0]
        np.testing.assert_allclose(color_extraction.linear_to_srgb(pixel) * 255, mean, atol=0.51)


if __name__ == "__main__":
    unittest.main()
