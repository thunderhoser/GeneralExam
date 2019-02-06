"""Unit tests for narr_netcdf_io.py."""

import unittest
from generalexam.ge_io import narr_netcdf_io
from generalexam.ge_io import processed_narr_io

FAKE_FIELD_NAME = 'poop'

NARR_TIME_HOURS = 1884678  # 0600 UTC 2 Jan 2015
UNIX_TIME_SEC = 1420178400  # 0600 UTC 2 Jan 2015

TOP_DIRECTORY_NAME = 'narr_netcdf'
FIELD_NAME_IN_FILE = processed_narr_io.SPECIFIC_HUMIDITY_NAME
MONTH_STRING = '201802'

PATHLESS_FILE_NAME_ISOBARIC = 'shum.201802.nc'
PATHLESS_FILE_NAME_SURFACE = 'shum.2m.2018.nc'

FILE_NAME_ISOBARIC = 'narr_netcdf/shum.201802.nc'
FILE_NAME_SURFACE = 'narr_netcdf/shum.2m.2018.nc'


class NarrNetcdfIoTests(unittest.TestCase):
    """Each method is a unit test for narr_netcdf_io.py."""

    def test_narr_to_unix_time(self):
        """Ensures correct output from _narr_to_unix_time."""

        this_time_unix_sec = narr_netcdf_io._narr_to_unix_time(NARR_TIME_HOURS)
        self.assertTrue(this_time_unix_sec == UNIX_TIME_SEC)

    def test_unix_to_narr_time(self):
        """Ensures correct output from _unix_to_narr_time."""

        this_time_narr_hours = narr_netcdf_io._unix_to_narr_time(UNIX_TIME_SEC)
        self.assertTrue(this_time_narr_hours == NARR_TIME_HOURS)

    def test_check_field_name_netcdf_valid(self):
        """Ensures correct output from _check_field_name_netcdf.

        In this case, field name is valid.
        """

        narr_netcdf_io._check_field_name_netcdf(
            narr_netcdf_io.TEMPERATURE_NAME_NETCDF)

    def test_check_field_name_netcdf_standard(self):
        """Ensures correct output from _check_field_name_netcdf.

        In this case, field name is in standard format (not NetCDF format).
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io._check_field_name_netcdf(
                processed_narr_io.TEMPERATURE_NAME)

    def test_check_field_name_netcdf_fake(self):
        """Ensures correct output from _check_field_name_netcdf.

        In this case, field name is completely fake.
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io._check_field_name_netcdf(FAKE_FIELD_NAME)

    def test_netcdf_to_std_field_name(self):
        """Ensures correct output from _netcdf_to_std_field_name."""

        this_standard_field_name = narr_netcdf_io._netcdf_to_std_field_name(
            narr_netcdf_io.U_WIND_NAME_NETCDF)

        self.assertTrue(this_standard_field_name ==
                        processed_narr_io.U_WIND_EARTH_RELATIVE_NAME)

    def test_std_to_netcdf_field_name(self):
        """Ensures correct output from _std_to_netcdf_field_name."""

        this_netcdf_field_name = narr_netcdf_io._std_to_netcdf_field_name(
            processed_narr_io.V_WIND_EARTH_RELATIVE_NAME)

        self.assertTrue(this_netcdf_field_name ==
                        narr_netcdf_io.V_WIND_NAME_NETCDF)

    def test_get_pathless_file_name_isobaric(self):
        """Ensures correct output from _get_pathless_file_name.

        In this case the hypothetical file contains isobaric (not surface) data.
        """

        this_pathless_file_name = narr_netcdf_io._get_pathless_file_name(
            field_name=FIELD_NAME_IN_FILE, month_string=MONTH_STRING,
            is_surface=False)

        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_ISOBARIC)

    def test_get_pathless_file_name_surface(self):
        """Ensures correct output from _get_pathless_file_name.

        In this case the hypothetical file contains surface data.
        """

        this_pathless_file_name = narr_netcdf_io._get_pathless_file_name(
            field_name=FIELD_NAME_IN_FILE, month_string=MONTH_STRING,
            is_surface=True)

        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_SURFACE)

    def test_find_file_isobaric(self):
        """Ensures correct output from find_file.

        In this case the hypothetical file contains isobaric (not surface) data.
        """

        this_file_name = narr_netcdf_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            field_name=FIELD_NAME_IN_FILE, month_string=MONTH_STRING,
            is_surface=False, raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME_ISOBARIC)

    def test_find_file_surface(self):
        """Ensures correct output from find_file.

        In this case the hypothetical file contains surface data.
        """

        this_file_name = narr_netcdf_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            field_name=FIELD_NAME_IN_FILE, month_string=MONTH_STRING,
            is_surface=True, raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME_SURFACE)


if __name__ == '__main__':
    unittest.main()
