import unittest
import numpy as np
from src import color_extraction  # Adjust the import based on your project structure and naming


class TestColorExtraction(unittest.TestCase):

    def test_get_dominant_color_avg(self):
        # Define a sample frame
        frame = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]]
        ], dtype=np.uint8)  # A 2x2 pixel frame with different colors

        # Call your function with the sample frame
        dominant_color = color_extraction.get_dominant_color_avg(frame)

        # Assert the expected result
        self.assertEqual(dominant_color.tolist(), [127.5, 127.5, 127.5])


if __name__ == '__main__':
    unittest.main()
