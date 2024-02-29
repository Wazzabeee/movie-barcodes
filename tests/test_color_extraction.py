import unittest
import numpy as np
from src import (
    color_extraction,
)  # Adjust the import based on your project structure and naming


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


if __name__ == "__main__":
    unittest.main()
