"""Unit tests for fronts_io.py."""

import unittest
from generalexam.ge_io import fronts_io

DIRECTORY_NAME = 'front_data'
VALID_TIME_UNIX_SEC = 1519419600  # 2100 UTC 23 Feb 2018
START_TIME_UNIX_SEC = 1519387200  # 1200 UTC 23 Feb 2018
END_TIME_UNIX_SEC = 1519452000  # 0600 UTC 24 Feb 2018

POLYLINE_FILE_NAME_TIME_PERIOD = (
    'front_data/front_locations_2018022312-2018022406.p')
GRIDDED_FILE_NAME_TIME_PERIOD = (
    'front_data/narr_frontal_grids_2018022312-2018022406.p')
POLYLINE_FILE_NAME_ONE_TIME = 'front_data/201802/front_locations_2018022321.p'
GRIDDED_FILE_NAME_ONE_TIME = 'front_data/201802/narr_frontal_grids_2018022321.p'


class FrontsIoTests(unittest.TestCase):
    """Each method is a unit test for fronts_io.py."""

    def test_find_file_for_time_period_polylines(self):
        """Ensures correct output from find_file_for_time_period.

        In this case, looking for file with polylines.
        """

        this_file_name = fronts_io.find_file_for_time_period(
            directory_name=DIRECTORY_NAME,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            start_time_unix_sec=START_TIME_UNIX_SEC,
            end_time_unix_sec=END_TIME_UNIX_SEC,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == POLYLINE_FILE_NAME_TIME_PERIOD)

    def test_find_file_for_time_period_grids(self):
        """Ensures correct output from find_file_for_time_period.

        In this case, looking for file with NARR grids.
        """

        this_file_name = fronts_io.find_file_for_time_period(
            directory_name=DIRECTORY_NAME,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            start_time_unix_sec=START_TIME_UNIX_SEC,
            end_time_unix_sec=END_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == GRIDDED_FILE_NAME_TIME_PERIOD)

    def test_find_file_for_one_time_polylines(self):
        """Ensures correct output from find_file_for_one_time.

        In this case, looking for file with polylines.
        """

        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=DIRECTORY_NAME,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == POLYLINE_FILE_NAME_ONE_TIME)

    def test_find_file_for_one_time_grids(self):
        """Ensures correct output from find_file_for_one_time.

        In this case, looking for file with NARR grids.
        """

        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=DIRECTORY_NAME,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)
        self.assertTrue(this_file_name == GRIDDED_FILE_NAME_ONE_TIME)


if __name__ == '__main__':
    unittest.main()
