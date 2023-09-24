import unittest
import numpy as np
from src import barcode_generation  # Adjust the import as per your project structure and naming


class TestBarcodeGeneration(unittest.TestCase):

    def test_generate_barcode(self):
        colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
        frame_height = 2
        frame_count = 2

        barcode = barcode_generation.generate_barcode(colors, frame_height, frame_count)

        self.assertIsInstance(barcode, np.ndarray)
        self.assertEqual(barcode.shape, (frame_height, frame_count, 3))  # Should match the input frame dimensions

    class TestBarcodeGeneration(unittest.TestCase):

        def test_generate_circular_barcode(self):
            colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
            img_size = 100  # Example image size

            barcode = barcode_generation.generate_circular_barcode(colors, img_size)

            self.assertIsInstance(barcode, np.ndarray)
            self.assertEqual(barcode.shape, (img_size, img_size, 4))  # RGBA image


if __name__ == '__main__':
    unittest.main()
