"""Unit tests for era5_input.py"""

import unittest
from generalexam.ge_io import era5_input
from generalexam.ge_utils import predictor_utils

FAKE_FIELD_NAME = 'poop'

# The following constants are used to test find_file, _file_name_to_year,
# _file_name_to_field, and _file_name_to_surface_flag.
TOP_DIRECTORY_NAME = 'era5_raw'
YEAR = 4055
PRESSURE_LEVEL_FIELD_NAME_RAW = era5_input.HEIGHT_NAME_RAW
SURFACE_FIELD_NAME_RAW = era5_input.PRESSURE_NAME_RAW

PRESSURE_LEVEL_FILE_NAME = 'era5_raw/ERA5_4055_3hrly_Z.nc'
SURFACE_FILE_NAME = 'era5_raw/ERA5_4055_3hrly_0m_p.nc'


class Era5IoTests(unittest.TestCase):
    """Each method is a unit test for era5_input.py."""

    def test_check_raw_field_name_good(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is valid (a raw field name).
        """

        era5_input._check_raw_field_name(era5_input.DEWPOINT_NAME_RAW)

    def test_check_raw_field_name_processed(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is *not* valid (a processed field name).
        """

        with self.assertRaises(ValueError):
            era5_input._check_raw_field_name(
                predictor_utils.SPECIFIC_HUMIDITY_NAME)

    def test_check_raw_field_name_fake(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is *not* valid (a fake field name).
        """

        with self.assertRaises(ValueError):
            era5_input._check_raw_field_name(FAKE_FIELD_NAME)

    def test_field_name_raw_to_processed_earth_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be an Earth-relative wind component.
        """

        this_field_name = era5_input.field_name_raw_to_processed(
            raw_field_name=era5_input.U_WIND_NAME_RAW, earth_relative=True)

        self.assertTrue(
            this_field_name == predictor_utils.U_WIND_EARTH_RELATIVE_NAME
        )

    def test_field_name_raw_to_processed_grid_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be a grid-relative wind component.
        """

        this_field_name = era5_input.field_name_raw_to_processed(
            raw_field_name=era5_input.V_WIND_NAME_RAW, earth_relative=False)

        self.assertTrue(
            this_field_name == predictor_utils.V_WIND_GRID_RELATIVE_NAME
        )

    def test_field_name_processed_to_raw_earth_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is an Earth-relative wind component.
        """

        this_raw_field_name = era5_input.field_name_processed_to_raw(
            predictor_utils.V_WIND_EARTH_RELATIVE_NAME)

        self.assertTrue(this_raw_field_name == era5_input.V_WIND_NAME_RAW)

    def test_field_name_processed_to_raw_grid_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is a grid-relative wind component.
        """

        this_raw_field_name = era5_input.field_name_processed_to_raw(
            predictor_utils.U_WIND_GRID_RELATIVE_NAME)

        self.assertTrue(this_raw_field_name == era5_input.U_WIND_NAME_RAW)

    def test_find_file_nonsurface(self):
        """Ensures correct output from find_file.

        In this case the file contains pressure-level data.
        """

        this_raw_file_name = era5_input.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, year=YEAR,
            raw_field_name=PRESSURE_LEVEL_FIELD_NAME_RAW, has_surface_data=False,
            raise_error_if_missing=False)

        self.assertTrue(this_raw_file_name == PRESSURE_LEVEL_FILE_NAME)

    def test_file_name_to_year_nonsurface(self):
        """Ensures correct output from _file_name_to_year.

        In this case the file contains pressure-level data.
        """

        this_year = era5_input._file_name_to_year(PRESSURE_LEVEL_FILE_NAME)
        self.assertTrue(this_year == YEAR)

    def test_raw_file_name_to_surface_no(self):
        """Ensures correct output from _file_name_to_surface_flag.

        In this case the file contains pressure-level data.
        """

        self.assertFalse(
            era5_input._file_name_to_surface_flag(PRESSURE_LEVEL_FILE_NAME)
        )

    def test_file_name_to_field_nonsurface(self):
        """Ensures correct output from _file_name_to_field.

        In this case the file contains pressure-level data.
        """

        this_field_name = era5_input._file_name_to_field(
            PRESSURE_LEVEL_FILE_NAME)

        self.assertTrue(
            this_field_name ==
            era5_input.field_name_raw_to_processed(
                PRESSURE_LEVEL_FIELD_NAME_RAW)
        )

    def test_find_file_surface(self):
        """Ensures correct output from find_file.

        In this case the file contains surface data.
        """

        this_raw_file_name = era5_input.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, year=YEAR,
            raw_field_name=SURFACE_FIELD_NAME_RAW, has_surface_data=True,
            raise_error_if_missing=False)

        self.assertTrue(this_raw_file_name == SURFACE_FILE_NAME)

    def test_file_name_to_year_surface(self):
        """Ensures correct output from _file_name_to_year.

        In this case the file contains surface data.
        """

        this_year = era5_input._file_name_to_year(SURFACE_FILE_NAME)
        self.assertTrue(this_year == YEAR)

    def test_raw_file_name_to_surface_yes(self):
        """Ensures correct output from _file_name_to_surface_flag.

        In this case the file contains surface data.
        """

        self.assertTrue(
            era5_input._file_name_to_surface_flag(SURFACE_FILE_NAME)
        )

    def test_file_name_to_field_surface(self):
        """Ensures correct output from _file_name_to_field.

        In this case the file contains surface data.
        """

        this_field_name = era5_input._file_name_to_field(SURFACE_FILE_NAME)

        self.assertTrue(
            this_field_name ==
            era5_input.field_name_raw_to_processed(SURFACE_FIELD_NAME_RAW)
        )


if __name__ == '__main__':
    unittest.main()