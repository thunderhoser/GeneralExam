"""Unit tests for prediction_io.py."""

import unittest
from generalexam.ge_io import prediction_io

TOLERANCE = 1e-6

DIRECTORY_NAME = 'foobar'
FIRST_TIME_UNIX_SEC = 1553385600  # 0000 UTC 24 Mar 2019
LAST_TIME_UNIX_SEC = 1553461200  # 2100 UTC 24 Mar 2019

ONE_TIME_FILE_NAME = 'foobar/predictions_2019032400-2019032400.nc'
MANY_TIMES_FILE_NAME = 'foobar/predictions_2019032400-2019032421.nc'


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_find_file_one_time(self):
        """Ensures correct output from find_file.

        In this case the file contains one time step.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=FIRST_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == ONE_TIME_FILE_NAME)

    def test_find_file_many_times(self):
        """Ensures correct output from find_file.

        In this case the file contains many time steps.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == MANY_TIMES_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
