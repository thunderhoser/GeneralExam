"""IO methods for NARR* data in NetCDF format.

* NARR = North American Regional Reanalysis

Since the NARR is a reanalysis, valid time = initialization time always.  In
other words, all "forecasts" are zero-hour forecasts (analyses).
"""

import os.path
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

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

TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
VERTICAL_VELOCITY_NAME = 'w_wind_pascals_s01'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'

FIELD_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, VERTICAL_VELOCITY_NAME,
    SPECIFIC_HUMIDITY_NAME, U_WIND_NAME, V_WIND_NAME]


def _check_field_name_orig(field_name_orig):
    """Ensures that name of model field is recognized.

    :param field_name_orig: Field name in original NetCDF format (not
        GewitterGefahr format).
    :raises: ValueError: if field name is unrecognized.
    """

    error_checking.assert_is_string(field_name_orig)
    if field_name_orig not in FIELD_NAMES_ORIG:
        error_string = (
            '\n\n' + str(FIELD_NAMES) +
            '\n\nValid field names (listed above) do not include "' +
            field_name_orig + '".')
        raise ValueError(error_string)


def _field_name_orig_to_new(field_name_orig):
    """Converts field name from orig (NetCDF) to new (GewitterGefahr) format.

    :param field_name_orig: Field name in NetCDF format.
    :return: field_name: Field name in GewitterGefahr format.
    """

    _check_field_name_orig(field_name_orig)
    return FIELD_NAMES[FIELD_NAMES_ORIG.index(field_name_orig)]


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


def check_field_name(field_name):
    """Ensures that name of model field is recognized.

    :param field_name: Field name in GewitterGefahr format (not the original
        NetCDF format).
    :raises: ValueError: if field name is unrecognized.
    """

    error_checking.assert_is_string(field_name)
    if field_name not in FIELD_NAMES:
        error_string = (
            '\n\n' + str(FIELD_NAMES) +
            '\n\nValid field names (listed above) do not include "' +
            field_name + '".')
        raise ValueError(error_string)


def field_name_new_to_orig(field_name):
    """Converts field name from new (GewitterGefahr) to orig (NetCDF) format.

    :param field_name: Field name in GewitterGefahr format.
    :return: field_name_orig: Field name in NetCDF format.
    """

    check_field_name(field_name)
    return FIELD_NAMES_ORIG[FIELD_NAMES.index(field_name)]


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
