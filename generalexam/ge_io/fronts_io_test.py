"""Unit tests for fronts_io.py."""

import unittest
from generalexam.ge_io import fronts_io

TOP_DIRECTORY_NAME = 'narr_fronts'
VALID_TIME_UNIX_SEC = 1519419600  # 2100 UTC 23 Feb 2018

POLYLINE_FILE_NAME = 'narr_fronts/201802/frontal_polylines_2018022321.nc'
GRIDDED_FILE_NAME = 'narr_fronts/201802/frontal_grid_2018022321.nc'


class FrontsIoTests(unittest.TestCase):
    """Each method is a unit test for fronts_io.py."""

    def test_find_polyline_file(self):
        """Ensures correct output from find_polyline_file."""

        this_file_name = fronts_io.find_polyline_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == POLYLINE_FILE_NAME)

    def test_find_gridded_file(self):
        """Ensures correct output from find_gridded_file."""

        this_file_name = fronts_io.find_gridded_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == GRIDDED_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
