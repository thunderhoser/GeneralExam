"""IO methods for NARR* data in NetCDF format.

* NARR = North American Regional Reanalysis

Since the NARR is a reanalysis, valid time = initialization time always.  In
other words, all "forecasts" are zero-hour forecasts (analyses).
"""

import os.path
import numpy
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io

HOURS_TO_SECONDS = 3600
NARR_ZERO_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1800-01-01-00', '%Y-%m-%d-%H')

TIME_FORMAT_MONTH = '%Y%m'
NETCDF_FILE_EXTENSION = '.nc'
TOP_ONLINE_DIRECTORY_NAME = 'ftp://ftp.cdc.noaa.gov/Datasets/NARR/pressure'

TEMPERATURE_NAME_ORIG = 'air'
HEIGHT_NAME_ORIG = 'hgt'
VERTICAL_VELOCITY_NAME_ORIG = 'omega'
SPECIFIC_HUMIDITY_NAME_ORIG = 'shum'
U_WIND_NAME_ORIG = 'uwnd'
V_WIND_NAME_ORIG = 'vwnd'

FIELD_NAMES_ORIG = [
    TEMPERATURE_NAME_ORIG, HEIGHT_NAME_ORIG, VERTICAL_VELOCITY_NAME_ORIG,
    SPECIFIC_HUMIDITY_NAME_ORIG, U_WIND_NAME_ORIG, V_WIND_NAME_ORIG]

PRESSURE_LEVEL_NAME_ORIG = 'level'
TIME_NAME_ORIG = 'time'
X_COORD_NAME_ORIG = 'x'
Y_COORD_NAME_ORIG = 'y'


def _time_from_narr_to_unix(narr_time_hours):
    """Converts time from NARR format to Unix format.

    NARR format = hours since 0000 UTC 1 Jan 1800
    Unix format = seconds since 0000 UTC 1 Jan 1970

    :param narr_time_hours: Time in NARR format.
    :return: unix_time_sec: Time in Unix format.
    """

    return NARR_ZERO_TIME_UNIX_SEC + narr_time_hours * HOURS_TO_SECONDS


def _time_from_unix_to_narr(unix_time_sec):
    """Converts time from Unix format to NARR format.

    NARR format = hours since 0000 UTC 1 Jan 1800
    Unix format = seconds since 0000 UTC 1 Jan 1970

    :param unix_time_sec: Time in Unix format.
    :return: narr_time_hours: Time in NARR format.
    """

    return (unix_time_sec - NARR_ZERO_TIME_UNIX_SEC) / HOURS_TO_SECONDS


def _check_field_name_orig(field_name_orig):
    """Ensures that name of model field is recognized.

    :param field_name_orig: Field name in original NetCDF format (not
        GewitterGefahr format).
    :raises: ValueError: if field name is unrecognized.
    """

    error_checking.assert_is_string(field_name_orig)
    if field_name_orig not in FIELD_NAMES_ORIG:
        error_string = (
            '\n\n' + str(FIELD_NAMES_ORIG) +
            '\n\nValid field names (listed above) do not include "' +
            field_name_orig + '".')
        raise ValueError(error_string)


def _field_name_orig_to_new(field_name_orig):
    """Converts field name from orig (NetCDF) to new (GewitterGefahr) format.

    :param field_name_orig: Field name in NetCDF format.
    :return: field_name: Field name in GewitterGefahr format.
    """

    _check_field_name_orig(field_name_orig)
    return processed_narr_io.FIELD_NAMES[
        FIELD_NAMES_ORIG.index(field_name_orig)]


def _get_pathless_file_name(month_string, field_name):
    """Generates pathless name for NetCDF file.

    This file should contain a single variable at all pressure levels for one
    month.

    :param month_string: Month (format "yyyymm").
    :param field_name: Field name in GewitterGefahr format.
    :return: pathless_netcdf_file_name: Pathless name for NetCDF file.
    """

    return '{0:s}.{1:s}{2:s}'.format(
        field_name_new_to_orig(field_name), month_string, NETCDF_FILE_EXTENSION)


def field_name_new_to_orig(field_name):
    """Converts field name from new (GewitterGefahr) to orig (NetCDF) format.

    :param field_name: Field name in GewitterGefahr format.
    :return: field_name_orig: Field name in NetCDF format.
    """

    processed_narr_io.check_field_name(field_name)
    return FIELD_NAMES_ORIG[processed_narr_io.FIELD_NAMES.index(field_name)]


def find_file(month_string, field_name, top_directory_name,
              raise_error_if_missing=True):
    """Finds NetCDF file on local machine.

    This file should contain a single variable at all pressure levels for one
    month.

    :param month_string: Month (format "yyyymm").
    :param field_name: Field name in GewitterGefahr format.
    :param top_directory_name: Name of top-level directory with NetCDF files
        containing NARR data.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: netcdf_file_name: Path to file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    # Error-checking.
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(month_string, TIME_FORMAT_MONTH)

    pathless_netcdf_file_name = _get_pathless_file_name(
        month_string=month_string, field_name=field_name)
    netcdf_file_name = '{0:s}/{1:s}'.format(
        top_directory_name, pathless_netcdf_file_name)

    if raise_error_if_missing and not os.path.isfile(netcdf_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                netcdf_file_name))
        raise ValueError(error_string)

    return netcdf_file_name


def download_file(month_string, field_name, top_local_directory_name,
                  raise_error_if_fails=True):
    """Downloads NetCDF file to local machine.

    This file should contain a single variable at all pressure levels for one
    month.

    :param month_string: Month (format "yyyymm").
    :param field_name: Field name in GewitterGefahr format.
    :param top_local_directory_name: Name of top-level directory (on local
        machine) for NARR files in NetCDF format.
    :param raise_error_if_fails: Boolean flag.  If download fails and
        raise_error_if_fails = True, this method will error out.  If download
        fails and raise_error_if_fails = False, this method will return None.
    :return: local_file_name: Path to downloaded file on local machine.  If
        download failed and raise_error_if_fails = False, this is None.
    """

    local_file_name = find_file(
        month_string=month_string, field_name=field_name,
        top_directory_name=top_local_directory_name,
        raise_error_if_missing=False)

    pathless_netcdf_file_name = _get_pathless_file_name(
        month_string=month_string, field_name=field_name)
    online_file_name = '{0:s}/{1:s}'.format(
        TOP_ONLINE_DIRECTORY_NAME, pathless_netcdf_file_name)

    return downloads.download_files_via_http(
        online_file_names=[online_file_name],
        local_file_names=[local_file_name],
        raise_error_if_fails=raise_error_if_fails)


def read_data_from_file(
        netcdf_file_name, field_name, valid_time_unix_sec, pressure_level_mb,
        raise_error_if_fails=True):
    """Reads data from NetCDF file.

    This file should contain a single variable at all pressure levels for one
    month.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param netcdf_file_name: Path to input file.
    :param field_name: Field name in GewitterGefahr format.  Only this field
        will be read.
    :param valid_time_unix_sec: Field will be read only for this valid time.
    :param pressure_level_mb: Field will be read only for this pressure level
        (integer in millibars).
    :param raise_error_if_fails: Boolean flag.  If file cannot be read and
        raise_error_if_fails = True, this method will error out.  If file cannot
        be read and raise_error_if_fails = False, this method will return None
        for all output variables.
    :return: field_matrix: M-by-N numpy array with values of `field_name`.
    :return: grid_point_x_coords_metres: length-N numpy array with x-coordinates
        of grid points.  grid_point_x_coords_metres[j] is the x-coordinate for
        all points in field_matrix[:, j].
    :return: grid_point_y_coords_metres: length-M numpy array with y-coordinates
        of grid points.  grid_point_y_coords_metres[i] is the y-coordinate for
        all points in field_matrix[i, :].
    """

    field_name_orig = field_name_new_to_orig(field_name)
    valid_time_narr_hours = _time_from_unix_to_narr(valid_time_unix_sec)
    error_checking.assert_is_integer(pressure_level_mb)

    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None, None, None

    all_times_narr_hours = netcdf_dataset.variables[TIME_NAME_ORIG]
    all_pressure_levels_mb = netcdf_dataset.variables[PRESSURE_LEVEL_NAME_ORIG]
    all_times_narr_hours = numpy.array(
        all_times_narr_hours).astype(int).tolist()
    all_pressure_levels_mb = numpy.array(
        all_pressure_levels_mb).astype(int).tolist()

    time_index = all_times_narr_hours.index(valid_time_narr_hours)
    pressure_index = all_pressure_levels_mb.index(pressure_level_mb)
    field_matrix = numpy.array(
        netcdf_dataset.variables[field_name_orig][
            time_index, pressure_index, :, :])

    grid_point_x_coords_metres = numpy.array(
        netcdf_dataset.variables[X_COORD_NAME_ORIG])
    grid_point_y_coords_metres = numpy.array(
        netcdf_dataset.variables[Y_COORD_NAME_ORIG])
    return field_matrix, grid_point_x_coords_metres, grid_point_y_coords_metres
