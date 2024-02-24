import unittest
from os import path, rmdir
from unittest.mock import patch, Mock
import numpy as np

from src import utility


class TestUtility(unittest.TestCase):
    """
    Test the utility functions.
    """

    @patch("src.utility.Image.fromarray")
    def test_save_barcode_image(self, mock_fromarray):
        """
        Test the save_barcode_image function.
        :param mock_fromarray:
        :return:
        """
        mock_image = Mock()
        mock_fromarray.return_value = mock_image

        barcode = np.ones((2, 2, 3), dtype=np.uint8) * 255
        base_name = "test_image"

        # Set up mock args object with string values for properties
        args = Mock(
            destination_path=None,
            method="kmeans",  # Example string value
            barcode_type="circular",  # Example string value
            workers="4",  # Example string value
        )

        utility.save_barcode_image(barcode, base_name, args)

        mock_image.save.assert_called_once()

    def test_ensure_directory(self):
        """
        Test the ensure_directory function.
        :return:
        """
        directory_name = "test_directory"

        # Ensure the directory does not exist before the function call
        if path.exists(directory_name):
            rmdir(directory_name)

        utility.ensure_directory(directory_name)
        self.assertTrue(path.exists(directory_name))

        # Clean up by removing the created directory
        rmdir(directory_name)


if __name__ == "__main__":
    unittest.main()
