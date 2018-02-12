"""Unit tests for narr_netcdf_io.py."""

import unittest
from gewittergefahr.general_exam import narr_netcdf_io

FAKE_FIELD_NAME = 'poop'

MONTH_STRING = '201802'
FIELD_NAME_FOR_FILES = narr_netcdf_io.SPECIFIC_HUMIDITY_NAME
PATHLESS_FILE_NAME = 'shum.201802.nc'
TOP_DIRECTORY_NAME = 'narr_netcdf'
FILE_NAME = 'narr_netcdf/shum.201802.nc'


class NarrNetcdfIoTests(unittest.TestCase):
    """Each method is a unit test for narr_netcdf_io.py."""

    def test_check_field_name_orig_valid(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is a valid field name in the original (NetCDF)
        format.
        """

        narr_netcdf_io._check_field_name_orig(
            narr_netcdf_io.TEMPERATURE_NAME_ORIG)

    def test_check_field_name_orig_new(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is a valid field name only in the new
        (GewitterGefahr) format, not original (NetCDF) format.
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io._check_field_name_orig(
                narr_netcdf_io.TEMPERATURE_NAME)

    def test_check_field_name_orig_fake(self):
        """Ensures correct output from _check_field_name_orig.

        In this case, input is a completely fake field name.
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io._check_field_name_orig(FAKE_FIELD_NAME)

    def test_check_field_name_valid(self):
        """Ensures correct output from check_field_name.

        In this case, input is a valid field name in the new (GewitterGefahr)
        format.
        """

        narr_netcdf_io.check_field_name(narr_netcdf_io.TEMPERATURE_NAME)

    def test_check_field_name_orig(self):
        """Ensures correct output from check_field_name.

        In this case, input is a valid field name only in the original (NetCDF)
        format, not the new (GewitterGefahr) format.
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io.check_field_name(
                narr_netcdf_io.TEMPERATURE_NAME_ORIG)

    def test_check_field_name_fake(self):
        """Ensures correct output from check_field_name.

        In this case, input is a completely fake field name.
        """

        with self.assertRaises(ValueError):
            narr_netcdf_io.check_field_name(FAKE_FIELD_NAME)

    def test_field_name_orig_to_new(self):
        """Ensures correct output from _field_name_orig_to_new."""

        self.assertTrue(narr_netcdf_io._field_name_orig_to_new(
            narr_netcdf_io.U_WIND_NAME_ORIG) ==
                        narr_netcdf_io.U_WIND_NAME)

    def test_field_name_new_to_orig(self):
        """Ensures correct output from field_name_new_to_orig."""

        self.assertTrue(narr_netcdf_io.field_name_new_to_orig(
            narr_netcdf_io.U_WIND_NAME) ==
                        narr_netcdf_io.U_WIND_NAME_ORIG)

    def test_get_pathless_file_name(self):
        """Ensures correct output from _get_pathless_file_name."""

        this_pathless_file_name = narr_netcdf_io._get_pathless_file_name(
            month_string=MONTH_STRING, field_name=FIELD_NAME_FOR_FILES)
        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME)

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = narr_netcdf_io.find_file(
            month_string=MONTH_STRING, field_name=FIELD_NAME_FOR_FILES,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == FILE_NAME)


if __name__ == '__main__':
    unittest.main()
