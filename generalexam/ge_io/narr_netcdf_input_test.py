"""Unit tests for narr_netcdf_input.py."""

import unittest
import numpy
from generalexam.ge_io import narr_netcdf_input
from generalexam.ge_utils import predictor_utils

TOLERANCE = 1e-6
FAKE_FIELD_NAME = 'poop'

# The following constants are used to test _narr_to_unix_time and
# _unix_to_narr_time.
VALID_TIME_NARR_HOURS = 1884678  # 0600 UTC 2 Jan 2015
VALID_TIME_UNIX_SEC = 1420178400  # 0600 UTC 2 Jan 2015

# The following constants are used to test _remove_sentinel_values.
DATA_MATRIX_WITH_SENTINELS = numpy.array([
    [numpy.inf, 9, 10, 1e37],
    [-1e37, -1e36, numpy.nan, 3],
    [numpy.nan, 1e40, -1e40, 2]
])

DATA_MATRIX_NO_SENTINELS = numpy.array([
    [numpy.inf, 9, 10, 1e37],
    [numpy.nan, -1e36, numpy.nan, 3],
    [numpy.nan, 1e40, numpy.nan, 2]
])

# The following constants are used to test file-naming.
TOP_DIRECTORY_NAME = 'narr_netcdf'
MONTH_STRING = '201802'

FIELD_NAME_SURFACE = predictor_utils.PRESSURE_NAME
FIELD_NAME_NONSURFACE = predictor_utils.HEIGHT_NAME

PATHLESS_FILE_NAME_SURFACE = 'pres.sfc.2018.nc'
PATHLESS_FILE_NAME_NONSURFACE = 'hgt.201802.nc'

FILE_NAME_SURFACE = 'narr_netcdf/pres.sfc.2018.nc'
FILE_NAME_NONSURFACE = 'narr_netcdf/hgt.201802.nc'


class NarrNetcdfIoTests(unittest.TestCase):
    """Each method is a unit test for narr_netcdf_input.py."""

    def test_check_raw_field_name_good(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is valid.
        """

        narr_netcdf_input._check_raw_field_name(
            raw_field_name=narr_netcdf_input.PRESSURE_NAME_RAW)

    def test_check_raw_field_name_fake(self):
        """Ensures correct output from _check_raw_field_name.

        In this case the input is bad (fake field name).
        """

        with self.assertRaises(ValueError):
            narr_netcdf_input._check_raw_field_name(
                raw_field_name=FAKE_FIELD_NAME)

    def test_check_raw_field_name_surface_good(self):
        """Ensures correct output from _check_raw_field_name.

        In this case, looking for surface field and the input *is* a surface
        field.
        """

        narr_netcdf_input._check_raw_field_name(
            raw_field_name=narr_netcdf_input.PRESSURE_NAME_RAW, at_surface=True)

    def test_check_raw_field_name_surface_hgt(self):
        """Ensures correct output from _check_raw_field_name.

        In this case, looking for surface height (orography).
        """

        narr_netcdf_input._check_raw_field_name(
            raw_field_name=narr_netcdf_input.HEIGHT_NAME_RAW, at_surface=True)

    def test_check_raw_field_name_nonsurface_good(self):
        """Ensures correct output from _check_raw_field_name.

        In this case, looking for non-surface field and the input *is* a
        non-surface field.
        """

        narr_netcdf_input._check_raw_field_name(
            raw_field_name=narr_netcdf_input.HEIGHT_NAME_RAW, at_surface=False)

    def test_check_raw_field_name_nonsurface_bad(self):
        """Ensures correct output from _check_raw_field_name.

        In this case, looking for non-surface field but the input is a surface
        field
        """

        with self.assertRaises(ValueError):
            narr_netcdf_input._check_raw_field_name(
                raw_field_name=narr_netcdf_input.PRESSURE_NAME_RAW,
                at_surface=False)

    def test_field_name_raw_to_processed_earth_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be an Earth-relative wind component.
        """

        this_field_name = narr_netcdf_input._field_name_raw_to_processed(
            raw_field_name=narr_netcdf_input.U_WIND_NAME_RAW,
            earth_relative=True)

        self.assertTrue(
            this_field_name == predictor_utils.U_WIND_EARTH_RELATIVE_NAME
        )

    def test_field_name_raw_to_processed_grid_relative(self):
        """Ensures correct output from field_name_raw_to_processed.

        In this case the output should be a grid-relative wind component.
        """

        this_field_name = narr_netcdf_input._field_name_raw_to_processed(
            raw_field_name=narr_netcdf_input.V_WIND_NAME_RAW,
            earth_relative=False)

        self.assertTrue(
            this_field_name == predictor_utils.V_WIND_GRID_RELATIVE_NAME
        )

    def test_field_name_processed_to_raw_earth_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is an Earth-relative wind component.
        """

        this_raw_field_name = narr_netcdf_input._field_name_processed_to_raw(
            predictor_utils.V_WIND_EARTH_RELATIVE_NAME)

        self.assertTrue(
            this_raw_field_name == narr_netcdf_input.V_WIND_NAME_RAW
        )

    def test_field_name_processed_to_raw_grid_relative(self):
        """Ensures correct output from field_name_processed_to_raw.

        In this case the input is a grid-relative wind component.
        """

        this_raw_field_name = narr_netcdf_input._field_name_processed_to_raw(
            predictor_utils.U_WIND_GRID_RELATIVE_NAME)

        self.assertTrue(
            this_raw_field_name == narr_netcdf_input.U_WIND_NAME_RAW
        )

    def test_narr_to_unix_time(self):
        """Ensures correct output from _narr_to_unix_time."""

        self.assertTrue(
            narr_netcdf_input._narr_to_unix_time(VALID_TIME_NARR_HOURS) ==
            VALID_TIME_UNIX_SEC
        )

    def test_unix_to_narr_time(self):
        """Ensures correct output from _unix_to_narr_time."""

        self.assertTrue(
            narr_netcdf_input._unix_to_narr_time(VALID_TIME_UNIX_SEC) ==
            VALID_TIME_NARR_HOURS
        )

    def test_remove_sentinel_values(self):
        """Ensures correct output from _remove_sentinel_values."""

        this_matrix = narr_netcdf_input._remove_sentinel_values(
            DATA_MATRIX_WITH_SENTINELS + 0.)

        self.assertTrue(numpy.allclose(
            this_matrix, DATA_MATRIX_NO_SENTINELS, rtol=TOLERANCE,
            equal_nan=True
        ))

    def test_file_name_to_surface_yes(self):
        """Ensures correct output from _file_name_to_surface_flag.

        In this case the file contains surface data.
        """

        self.assertTrue(
            narr_netcdf_input._file_name_to_surface_flag(FILE_NAME_SURFACE)
        )

    def test_file_name_to_surface_no(self):
        """Ensures correct output from _file_name_to_surface_flag.

        In this case the file does *not* contain surface data.
        """

        self.assertFalse(
            narr_netcdf_input._file_name_to_surface_flag(
                FILE_NAME_NONSURFACE)
        )

    def test_file_name_to_field_surface(self):
        """Ensures correct output from _file_name_to_field.

        In this case the file contains surface data.
        """

        self.assertTrue(
            narr_netcdf_input._file_name_to_field(FILE_NAME_SURFACE) ==
            FIELD_NAME_SURFACE
        )

    def test_file_name_to_field_nonsurface(self):
        """Ensures correct output from _file_name_to_field.

        In this case the file does *not* contain surface data.
        """

        self.assertTrue(
            narr_netcdf_input._file_name_to_field(FILE_NAME_NONSURFACE) ==
            FIELD_NAME_NONSURFACE
        )

    def test_get_pathless_file_name_surface(self):
        """Ensures correct output from _get_pathless_file_name.

        In this case the file contains surface data.
        """

        this_pathless_file_name = narr_netcdf_input._get_pathless_file_name(
            field_name=FIELD_NAME_SURFACE, month_string=MONTH_STRING,
            at_surface=True)

        self.assertTrue(this_pathless_file_name == PATHLESS_FILE_NAME_SURFACE)

    def test_get_pathless_file_name_nonsurface(self):
        """Ensures correct output from _get_pathless_file_name.

        In this case the file does *not* contain surface data.
        """

        this_pathless_file_name = narr_netcdf_input._get_pathless_file_name(
            field_name=FIELD_NAME_NONSURFACE, month_string=MONTH_STRING,
            at_surface=False)

        self.assertTrue(
            this_pathless_file_name == PATHLESS_FILE_NAME_NONSURFACE
        )

    def test_find_file_surface(self):
        """Ensures correct output from find_file.

        In this case the file contains surface data.
        """

        this_file_name = narr_netcdf_input.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            field_name=FIELD_NAME_SURFACE, month_string=MONTH_STRING,
            at_surface=True, raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME_SURFACE)

    def test_find_file_nonsurface(self):
        """Ensures correct output from find_file.

        In this case the file does *not* contain surface data.
        """

        this_file_name = narr_netcdf_input.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            field_name=FIELD_NAME_NONSURFACE, month_string=MONTH_STRING,
            at_surface=False, raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME_NONSURFACE)


if __name__ == '__main__':
    unittest.main()
