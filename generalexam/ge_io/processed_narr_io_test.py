"""Unit tests for processed_narr_io.py."""

import unittest
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import narr_netcdf_io

FAKE_FIELD_NAME = 'poop'


class ProcessedNarrIoTests(unittest.TestCase):
    """Each method is a unit test for processed_narr_io.py."""

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


if __name__ == '__main__':
    unittest.main()
