"""Unit tests for predictor_io.py"""

import unittest
from generalexam.ge_io import predictor_io

# The following constants are used to test find_file.
TOP_DIRECTORY_NAME = 'stuff'
VALID_TIME_UNIX_SEC = 65804414745  # 040545 UTC 5 Apr 4055
FILE_NAME = 'stuff/405504/era5_processed_4055040504.nc'


class PredictorIoTests(unittest.TestCase):
    """Each method is a unit test for predictor_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = predictor_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME)


if __name__ == '__main__':
    unittest.main()
