import unittest
from unittest.mock import patch
import numpy as np
from movie_barcodes import (
    barcode_generation,
)


class TestBarcodeGeneration(unittest.TestCase):
    """
    Test the barcode generation functions.
    """

    def setUp(self) -> None:
        """
        Set up the test case.
        :return: None
        """
        self.colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
        self.frame_height = 2
        self.frame_count = 2
        self.img_size = 100
        self.width = 1

    def test_generate_barcode_default(self) -> None:
        """
        Test the generate_barcode function.
        :return: None
        """
        barcode = barcode_generation.generate_barcode(self.colors, self.frame_height, self.frame_count)

        self.assertIsInstance(barcode, np.ndarray)
        self.assertEqual(
            barcode.shape, (self.frame_height, self.frame_count, 3)
        )  # Should match the input frame dimensions

    def test_generate_barcode_with_width(self) -> None:
        """
        Test the generate_barcode function with a specified width.
        :return: None
        """
        barcode = barcode_generation.generate_barcode(self.colors, self.frame_height, self.frame_count, self.width)

        self.assertIsInstance(barcode, np.ndarray)
        self.assertEqual(barcode.shape, (self.frame_height, self.width, 3))

    def test_generate_circular_barcode(self) -> None:
        """
        Test the generate_circular_barcode function.
        :return: None
        """
        barcode = barcode_generation.generate_circular_barcode(self.colors, self.img_size)

        self.assertIsInstance(barcode, np.ndarray)
        self.assertEqual(barcode.shape, (self.img_size, self.img_size, 4))  # RGBA image

    def test_generate_barcode_keeps_end_of_movie(self) -> None:
        """
        When there are more colors than columns, the last column comes from the end of the list.
        """
        colors = [np.array([i, i, i]) for i in range(150)]
        barcode = barcode_generation.generate_barcode(colors, 1, 150, 100)
        self.assertEqual(barcode.shape, (1, 100, 3))
        self.assertGreaterEqual(int(barcode[0, -1, 0]), 148)

    def test_generate_barcode_stretches_when_fewer_colors(self) -> None:
        """
        When there are fewer colors than columns (frame count over-reported), there are no black columns.
        """
        colors = [np.array([10, 20, 30]), np.array([40, 50, 60])]
        barcode = barcode_generation.generate_barcode(colors, 3, 4, 4)
        self.assertTrue((barcode.reshape(-1, 3).sum(axis=1) > 0).all())
        np.testing.assert_array_equal(barcode[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(barcode[0, -1], [40, 50, 60])

    def test_generate_barcode_with_smoothed_frames(self) -> None:
        """
        Smoothed frames (H x 1 x 3 columns) are stretched to the barcode height.
        """
        column = np.full((5, 1, 3), (7, 8, 9), dtype=np.uint8)
        barcode = barcode_generation.generate_barcode([column, column], 10, 2)
        self.assertEqual(barcode.shape, (10, 2, 3))
        np.testing.assert_array_equal(barcode[9, 1], [7, 8, 9])

    def test_generate_circular_barcode_has_no_gaps(self) -> None:
        """
        With few colors, rings must still cover the whole disc (no transparent gaps between rings).
        """
        barcode = barcode_generation.generate_circular_barcode(self.colors, self.img_size)
        center = self.img_size // 2
        # Every pixel along the radius (except the antialiased outer edge) is opaque
        self.assertTrue((barcode[center, center : self.img_size - 2, 3] == 255).all())
        # Inner disc is the first color, outer ring is the last color
        np.testing.assert_array_equal(barcode[center, center, :3], [255, 0, 0])
        np.testing.assert_array_equal(barcode[center, self.img_size - 5, :3], [0, 255, 0])
        # Corners stay transparent
        self.assertEqual(barcode[0, 0, 3], 0)

    def test_generate_circular_barcode_caps_supersampling(self) -> None:
        """
        The supersampled canvas is capped, so large sizes do not allocate gigabytes of memory.
        """
        with patch("movie_barcodes.barcode_generation.np.zeros", wraps=np.zeros) as mock_zeros:
            barcode = barcode_generation.generate_circular_barcode(self.colors, 1920)
        side = mock_zeros.call_args[0][0][0]
        self.assertLessEqual(side, barcode_generation.MAX_HIGH_RES_SIZE)
        self.assertEqual(barcode.shape, (1920, 1920, 4))


if __name__ == "__main__":
    unittest.main()
