"""Unit tests for era5_io.py"""

import unittest
from generalexam.ge_io import era5_io

FAKE_FIELD_NAME = 'poop'

# The following constants are used to test find_raw_file and
# _raw_file_name_to_year.
TOP_RAW_DIRECTORY_NAME = 'era5_raw'
YEAR = 4055
FIELD_NAME_1000MB = era5_io.HEIGHT_NAME_RAW
FIELD_NAME_SURFACE = era5_io.PRESSURE_NAME_RAW

RAW_FILE_NAME_1000MB = 'era5_raw/ERA5_4055_3hrly_1000mbZ.nc'
RAW_FILE_NAME_SURFACE = 'era5_raw/ERA5_4055_3hrly_0mp.nc'

# The following constants are used to test find_processed_file.
TOP_PROCESSED_DIR_NAME = 'era5_processed'
VALID_TIME_UNIX_SEC = 65804414745  # 040545 UTC 5 Apr 4055
PROCESSED_FILE_NAME = 'era5_processed/405504/era5_processed_4055040504.nc'


class Era5IoTests(unittest.TestCase):
    """Each method is a unit test for era5_io.py."""

    def test_check_raw_field_name_good(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is valid (a raw field name).
        """

        era5_io._check_raw_field_name(era5_io.DEWPOINT_NAME_RAW)

    def test_check_raw_field_name_processed(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is *not* valid (a processed field name).
        """

        with self.assertRaises(ValueError):
            era5_io._check_raw_field_name(era5_io.SPECIFIC_HUMIDITY_NAME)

    def test_check_raw_field_name_fake(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is *not* valid (a fake field name).
        """

        with self.assertRaises(ValueError):
            era5_io._check_raw_field_name(FAKE_FIELD_NAME)

    def test_check_field_name_good(self):
        """Ensures correct output from check_field_name.

        In this case the input is valid (a processed field name).
        """

        era5_io.check_field_name(era5_io.TEMPERATURE_NAME)

    def test_check_field_name_raw(self):
        """Ensures correct output from check_field_name.

        In this case the input is *not* valid (a raw field name).
        """

        with self.assertRaises(ValueError):
            era5_io.check_field_name(era5_io.TEMPERATURE_NAME_RAW)

    def test_check_field_name_fake(self):
        """Ensures correct output from check_field_name.

        In this case the input is *not* valid (a fake field name).
        """

        with self.assertRaises(ValueError):
            era5_io.check_field_name(FAKE_FIELD_NAME)

    def test_field_name_raw_to_processed_earth_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be an Earth-relative wind component.
        """

        this_field_name = era5_io.field_name_raw_to_processed(
            raw_field_name=era5_io.U_WIND_NAME_RAW, earth_relative=True)

        self.assertTrue(this_field_name == era5_io.U_WIND_EARTH_RELATIVE_NAME)

    def test_field_name_raw_to_processed_grid_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be a grid-relative wind component.
        """

        this_field_name = era5_io.field_name_raw_to_processed(
            raw_field_name=era5_io.V_WIND_NAME_RAW, earth_relative=False)

        self.assertTrue(this_field_name == era5_io.V_WIND_GRID_RELATIVE_NAME)

    def test_field_name_processed_to_raw_earth_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is an Earth-relative wind component.
        """

        this_raw_field_name = era5_io.field_name_processed_to_raw(
            era5_io.V_WIND_EARTH_RELATIVE_NAME)

        self.assertTrue(this_raw_field_name == era5_io.V_WIND_NAME_RAW)

    def test_field_name_processed_to_raw_grid_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is a grid-relative wind component.
        """

        this_raw_field_name = era5_io.field_name_processed_to_raw(
            era5_io.U_WIND_GRID_RELATIVE_NAME)

        self.assertTrue(this_raw_field_name == era5_io.U_WIND_NAME_RAW)

    def test_find_raw_file_1000mb(self):
        """Ensures correct output from find_raw_file.

        In this case the pressure level is 1000 mb.
        """

        this_raw_file_name = era5_io.find_raw_file(
            top_directory_name=TOP_RAW_DIRECTORY_NAME, year=YEAR,
            raw_field_name=FIELD_NAME_1000MB, pressure_level_mb=1000,
            raise_error_if_missing=False)

        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_1000MB)

    def test_raw_file_name_to_year_1000mb(self):
        """Ensures correct output from _raw_file_name_to_year.

        In this case the pressure level is 1000 mb.
        """

        this_year = era5_io._raw_file_name_to_year(RAW_FILE_NAME_1000MB)
        self.assertTrue(this_year == YEAR)

    def test_find_raw_file_surface(self):
        """Ensures correct output from find_raw_file.

        In this case the "pressure level" is the surface.
        """

        this_raw_file_name = era5_io.find_raw_file(
            top_directory_name=TOP_RAW_DIRECTORY_NAME, year=YEAR,
            raw_field_name=FIELD_NAME_SURFACE,
            pressure_level_mb=era5_io.DUMMY_SURFACE_PRESSURE_MB,
            raise_error_if_missing=False)

        self.assertTrue(this_raw_file_name == RAW_FILE_NAME_SURFACE)

    def test_raw_file_name_to_year_surface(self):
        """Ensures correct output from _raw_file_name_to_year.

        In this case the "pressure level" is the surface.
        """

        this_year = era5_io._raw_file_name_to_year(RAW_FILE_NAME_SURFACE)
        self.assertTrue(this_year == YEAR)

    def test_find_processed_file(self):
        """Ensures correct output from find_processed_file."""

        this_processed_file_name = era5_io.find_processed_file(
            top_directory_name=TOP_PROCESSED_DIR_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_processed_file_name == PROCESSED_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
