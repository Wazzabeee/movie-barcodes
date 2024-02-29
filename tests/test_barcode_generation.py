import unittest
import numpy as np
from src import (
    barcode_generation,
)  # Adjust the import as per your project structure and naming


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


if __name__ == "__main__":
    unittest.main()
