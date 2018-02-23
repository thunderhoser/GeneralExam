"""Unit tests for processed_narr_io.py."""

import unittest
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import narr_netcdf_io

FAKE_FIELD_NAME = 'poop'

DIRECTORY_NAME = 'processed_narr_data'
FIELD_NAME_IN_FILES = processed_narr_io.U_WIND_GRID_RELATIVE_NAME
PRESSURE_LEVEL_MB = 1000
VALID_TIME_UNIX_SEC = 1519419600  # 2100 UTC 23 Feb 2018
START_TIME_UNIX_SEC = 1519387200  # 1200 UTC 23 Feb 2018
END_TIME_UNIX_SEC = 1519452000  # 0600 UTC 24 Feb 2018

PROCESSED_FILE_NAME_TIME_PERIOD = (
    'processed_narr_data/u_wind_grid_relative_1000mb_2018022312-2018022406.p')
PROCESSED_FILE_NAME_ONE_TIME = (
    'processed_narr_data/201802/u_wind_grid_relative_1000mb_2018022321.p')


class ProcessedNarrIoTests(unittest.TestCase):
    """Each method is a unit test for processed_narr_io.py."""

    def test_remove_units_from_field_name(self):
        """Ensures correct output from _remove_units_from_field_name."""

        these_field_names_unitless = []
        for this_field_name in processed_narr_io.FIELD_NAMES:
            these_field_names_unitless.append(
                processed_narr_io._remove_units_from_field_name(
                    this_field_name))

        self.assertTrue(these_field_names_unitless ==
                        processed_narr_io.FIELD_NAMES_UNITLESS)

    def test_check_field_name_any_valid(self):
        """Ensures correct output from check_field_name.

        In this case, input may be any field name (standard or derived) and
        input is a derived field name.
        """

        processed_narr_io.check_field_name(
            processed_narr_io.WET_BULB_THETA_NAME, require_standard=False)

    def test_check_field_name_standard_valid(self):
        """Ensures correct output from check_field_name.

        In this case, input must be a standard field name and *is* a standard
        field name.
        """

        processed_narr_io.check_field_name(
            processed_narr_io.TEMPERATURE_NAME, require_standard=True)

    def test_check_field_name_standard_invalid(self):
        """Ensures correct output from check_field_name.

        In this case, input must be a standard field name and is *not* a
        standard field name.
        """

        with self.assertRaises(ValueError):
            processed_narr_io.check_field_name(
                processed_narr_io.WET_BULB_THETA_NAME, require_standard=True)

    def test_check_field_name_orig(self):
        """Ensures correct output from check_field_name.

        In this case, input is a valid field name only in the original (NetCDF)
        format, not the new (GewitterGefahr) format.
        """

        with self.assertRaises(ValueError):
            processed_narr_io.check_field_name(
                narr_netcdf_io.TEMPERATURE_NAME_ORIG)

    def test_check_field_name_fake(self):
        """Ensures correct output from check_field_name.

        In this case, input is a completely fake field name.
        """

        with self.assertRaises(ValueError):
            processed_narr_io.check_field_name(FAKE_FIELD_NAME)

    def test_find_file_for_time_period(self):
        """Ensures correct output from find_file_for_time_period."""

        this_file_name = processed_narr_io.find_file_for_time_period(
            directory_name=DIRECTORY_NAME, field_name=FIELD_NAME_IN_FILES,
            pressure_level_mb=PRESSURE_LEVEL_MB,
            start_time_unix_sec=START_TIME_UNIX_SEC,
            end_time_unix_sec=END_TIME_UNIX_SEC, raise_error_if_missing=False)

        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_TIME_PERIOD)

    def test_find_file_for_one_time(self):
        """Ensures correct output from find_file_for_one_time."""

        this_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=DIRECTORY_NAME, field_name=FIELD_NAME_IN_FILES,
            pressure_level_mb=PRESSURE_LEVEL_MB,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == PROCESSED_FILE_NAME_ONE_TIME)


if __name__ == '__main__':
    unittest.main()
