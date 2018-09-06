"""Unit tests for isotonic_regression.py."""

import unittest
from generalexam.machine_learning import isotonic_regression

BASE_MODEL_FILE_NAME = 'foo/bar/model.h5'
ISOTONIC_FILE_NAME = 'foo/bar/isotonic_regression_models.p'


class IsotonicRegressionTests(unittest.TestCase):
    """Each method is a unit test for isotonic_regression.py."""

    def test_find_model_file(self):
        """Ensures correct output from find_model_file."""

        this_file_name = isotonic_regression.find_model_file(
            base_model_file_name=BASE_MODEL_FILE_NAME,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == ISOTONIC_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
