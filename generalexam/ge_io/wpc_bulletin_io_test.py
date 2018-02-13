"""Unit tests for wpc_bulletin_io.py."""

import unittest
import numpy
from generalexam.ge_io import wpc_bulletin_io

TOLERANCE = 1e-6

LATLNG_STRING_5CHARS = '53113'
LATITUDE_FROM_5CHARS_DEG = 53.
LONGITUDE_FROM_5CHARS_DEG = -113.

LATLNG_STRING_7CHARS = '5331135'
LATITUDE_FROM_7CHARS_DEG = 53.3
LONGITUDE_FROM_7CHARS_DEG = -113.5

LATLNG_STRING_5CHARS_BAD = '53a13'
LATLNG_STRING_7CHARS_BAD = '533113b'
LATLNG_STRING_8CHARS = '53311350'

VALID_TIME_UNIX_SEC = 1518480000  # 0000 UTC 13 Feb 2018
TOP_DIRECTORY_NAME = 'wpc_bulletins'
BULLETIN_FILE_NAME = 'wpc_bulletins/2018/KWBCCODSUS_HIRES_20180213_0000'


class WpcBulletinIoTests(unittest.TestCase):
    """Each method is a unit test for wpc_bulletin_io.py."""

    def test_string_to_latlng_5chars(self):
        """Ensures correct output from _string_to_latlng.

        In this case, lat-long string has 5 characters and is valid.
        """

        this_latitude_deg, this_longitude_deg = (
            wpc_bulletin_io._string_to_latlng(LATLNG_STRING_5CHARS))
        self.assertTrue(numpy.isclose(
            this_latitude_deg, LATITUDE_FROM_5CHARS_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_longitude_deg, LONGITUDE_FROM_5CHARS_DEG, atol=TOLERANCE))

    def test_string_to_latlng_7chars(self):
        """Ensures correct output from _string_to_latlng.

        In this case, lat-long string has 7 characters and is valid.
        """

        this_latitude_deg, this_longitude_deg = (
            wpc_bulletin_io._string_to_latlng(LATLNG_STRING_7CHARS))
        self.assertTrue(numpy.isclose(
            this_latitude_deg, LATITUDE_FROM_7CHARS_DEG, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_longitude_deg, LONGITUDE_FROM_7CHARS_DEG, atol=TOLERANCE))

    def test_string_to_latlng_5chars_invalid(self):
        """Ensures correct output from _string_to_latlng.

        In this case, lat-long string has 5 characters but is *invalid*.
        """

        with self.assertRaises(ValueError):
            wpc_bulletin_io._string_to_latlng(LATLNG_STRING_5CHARS_BAD)

    def test_string_to_latlng_7chars_invalid(self):
        """Ensures correct output from _string_to_latlng.

        In this case, lat-long string has 7 characters but is *invalid*.
        """

        with self.assertRaises(ValueError):
            wpc_bulletin_io._string_to_latlng(LATLNG_STRING_7CHARS_BAD)

    def test_string_to_latlng_8chars(self):
        """Ensures correct output from _string_to_latlng.

        In this case, lat-long string has 8 characters, which is *invalid*.
        """

        with self.assertRaises(ValueError):
            wpc_bulletin_io._string_to_latlng(LATLNG_STRING_8CHARS)

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = wpc_bulletin_io.find_file(
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == BULLETIN_FILE_NAME)

    def test_file_name_to_valid_time(self):
        """Ensures correct output from _file_name_to_valid_time."""

        this_time_unix_sec = wpc_bulletin_io._file_name_to_valid_time(
            BULLETIN_FILE_NAME)
        self.assertTrue(this_time_unix_sec == VALID_TIME_UNIX_SEC)


if __name__ == '__main__':
    unittest.main()
