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

SENTINEL_VALUE = -9e36
HOURS_TO_SECONDS = 3600
NARR_ZERO_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1800-01-01-00', '%Y-%m-%d-%H')

TIME_FORMAT_MONTH = '%Y%m'
NETCDF_FILE_EXTENSION = '.nc'

ONLINE_SURFACE_DIR_NAME = 'ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel'
ONLINE_PRESSURE_LEVEL_DIR_NAME = 'ftp://ftp.cdc.noaa.gov/Datasets/NARR/pressure'

TEMPERATURE_NAME_NETCDF = 'air'
HEIGHT_NAME_NETCDF = 'hgt'
VERTICAL_VELOCITY_NAME_NETCDF = 'omega'
SPECIFIC_HUMIDITY_NAME_NETCDF = 'shum'
U_WIND_NAME_NETCDF = 'uwnd'
V_WIND_NAME_NETCDF = 'vwnd'

VALID_FIELD_NAMES_NETCDF = [
    TEMPERATURE_NAME_NETCDF, HEIGHT_NAME_NETCDF, VERTICAL_VELOCITY_NAME_NETCDF,
    SPECIFIC_HUMIDITY_NAME_NETCDF, U_WIND_NAME_NETCDF, V_WIND_NAME_NETCDF
]

PRESSURE_KEY = 'level'
TIME_KEY = 'time'


def _remove_sentinel_values(data_matrix):
    """Removes sentinel values from field.

    M = number of rows in grid
    N = number of columns in grid

    :param data_matrix: M-by-N numpy array of gridded values.
    :return: data_matrix: Same but with sentinel values changed to NaN.
    """

    data_matrix[data_matrix < SENTINEL_VALUE] = numpy.nan
    return data_matrix


def _narr_to_unix_time(narr_time_hours):
    """Converts NARR time to Unix time.

    :param narr_time_hours: NARR time (integer).
    :return: unix_time_sec: Unix time (integer).
    """

    return NARR_ZERO_TIME_UNIX_SEC + narr_time_hours * HOURS_TO_SECONDS


def _unix_to_narr_time(unix_time_sec):
    """Converts Unix time to NARR time.

    :param unix_time_sec: Unix time (integer).
    :return: narr_time_hours: NARR time (integer).
    """

    return (unix_time_sec - NARR_ZERO_TIME_UNIX_SEC) / HOURS_TO_SECONDS


def _check_field_name_netcdf(field_name_netcdf):
    """Error-checks field name.

    :param field_name_netcdf: Field name in NetCDF format.
    :raises: ValueError: if `field_name_netcdf not in VALID_FIELD_NAMES_NETCDF`.
    """

    error_checking.assert_is_string(field_name_netcdf)

    if field_name_netcdf not in VALID_FIELD_NAMES_NETCDF:
        error_string = (
            '\n{0:s}\nValid field names in NetCDF format (listed above) do not '
            'include "{1:s}".'
        ).format(str(VALID_FIELD_NAMES_NETCDF), field_name_netcdf)

        raise ValueError(error_string)


def _netcdf_to_std_field_name(field_name_netcdf):
    """Converts field name from NetCDF to standard format.

    :param field_name_netcdf: Field name in NetCDF format (must be accepted by
        `_check_field_name_netcdf`).
    :return: field_name: Field name in standard format (accepted by
        `processed_narr_io.check_field_name`).
    """

    _check_field_name_netcdf(field_name_netcdf)

    return processed_narr_io.STANDARD_FIELD_NAMES[
        VALID_FIELD_NAMES_NETCDF.index(field_name_netcdf)
    ]


def _std_to_netcdf_field_name(field_name):
    """Converts field name from standard to NetCDF format.

    :param field_name: Field name in standard format (must be accepted by
        `processed_narr_io.check_field_name`).
    :return: field_name_netcdf: Field name in NetCDF format (accepted by
        `_check_field_name_netcdf`).
    """

    processed_narr_io.check_field_name(
        field_name=field_name, require_standard=True)

    return VALID_FIELD_NAMES_NETCDF[
        processed_narr_io.STANDARD_FIELD_NAMES.index(field_name)
    ]


def _get_pathless_file_name(field_name, month_string, is_surface=False):
    """Returns pathless name for NetCDF file.

    :param field_name: Field name (must be accepted by
        `processed_narr_io.check_field_name`).
    :param month_string: Month (format "yyyymm").
    :param is_surface: Boolean flag.  If True, will assume that the file
        contains surface data.  If False, will assume that it contains isobaric
        data at all pressure levels.
    :return: pathless_netcdf_file_name: Pathless name for NetCDF file.
    """

    if not is_surface:
        return '{0:s}.{1:s}{2:s}'.format(
            _std_to_netcdf_field_name(field_name), month_string,
            NETCDF_FILE_EXTENSION)

    if field_name == processed_narr_io.TEMPERATURE_NAME:
        return 'air.2m.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    if field_name == processed_narr_io.SPECIFIC_HUMIDITY_NAME:
        return 'shum.2m.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    # TODO(thunderhoser): This is a HACK.
    if field_name == processed_narr_io.HEIGHT_NAME:
        return 'pres.sfc.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    if field_name == processed_narr_io.VERTICAL_VELOCITY_NAME:
        return 'vvel.hl1.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    if field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
        return 'uwnd.10m.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    if field_name == processed_narr_io.V_WIND_EARTH_RELATIVE_NAME:
        return 'vwnd.10m.{0:s}{1:s}'.format(
            month_string[:4], NETCDF_FILE_EXTENSION)

    return None


def find_file(top_directory_name, field_name, month_string, is_surface=False,
              raise_error_if_missing=True):
    """Finds NetCDF file on the local machine.

    :param top_directory_name: Name of top-level directory with NetCDF files
        containing NARR data.
    :param field_name: See doc for `_get_pathless_file_name`.
    :param month_string: Same.
    :param is_surface: Same.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: netcdf_file_name: Path to NetCDF file.  If file is missing and
        `raise_error_if_missing = False`, will return the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    time_conversion.string_to_unix_sec(month_string, TIME_FORMAT_MONTH)
    error_checking.assert_is_boolean(is_surface)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_file_name(
        field_name=field_name, month_string=month_string, is_surface=is_surface)
    netcdf_file_name = '{0:s}/{1:s}'.format(
        top_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(netcdf_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            netcdf_file_name)
        raise ValueError(error_string)

    return netcdf_file_name


def download_file(
        top_local_dir_name, field_name, month_string, is_surface=False,
        raise_error_if_fails=False):
    """Downloads NetCDF file to the local machine.

    :param top_local_dir_name: Name of top-level target directory (local
        directory for NetCDF files with NARR data).
    :param field_name: See doc for `_get_pathless_file_name`.
    :param month_string: Same.
    :param is_surface: Same.
    :param raise_error_if_fails: Boolean flag.  If download fails and
        `raise_error_if_fails = True`, this method will error out.
    :return: local_file_name: Path to downloaded file on local machine.  If
        download failed and `raise_error_if_fails = False`, this is None.
    """

    local_file_name = find_file(
        top_directory_name=top_local_dir_name, field_name=field_name,
        month_string=month_string, is_surface=is_surface,
        raise_error_if_missing=False)

    pathless_file_name = _get_pathless_file_name(
        field_name=field_name, month_string=month_string, is_surface=is_surface)

    if is_surface:
        online_dir_name = ONLINE_SURFACE_DIR_NAME + ''
    else:
        online_dir_name = ONLINE_PRESSURE_LEVEL_DIR_NAME + ''

    online_file_name = '{0:s}/{1:s}'.format(online_dir_name, pathless_file_name)

    return downloads.download_files_via_http(
        online_file_names=[online_file_name],
        local_file_names=[local_file_name],
        raise_error_if_fails=raise_error_if_fails)


def read_file(netcdf_file_name, field_name, valid_time_unix_sec,
              pressure_level_mb=None):
    """Reads data from NetCDF file.

    This method will extract one field at one pressure level (or surface) at one
    time.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to input file.
    :param field_name: Field to extract (must be accepted by
        `processed_narr_io.check_field_name`).
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: [used only if file contains isobaric data]
        Pressure level to extract (millibars).
    :return: data_matrix: M-by-N numpy array with values of the given field at
        the given pressure level (or surface).
    """

    field_name_orig = _std_to_netcdf_field_name(field_name)
    valid_time_narr_hours = _unix_to_narr_time(valid_time_unix_sec)

    if pressure_level_mb is None:

        # TODO(thunderhoser): This is a HACK.
        if field_name_orig == HEIGHT_NAME_NETCDF:
            field_name_orig = 'pres'
        if field_name_orig == VERTICAL_VELOCITY_NAME_NETCDF:
            field_name_orig = 'vvel'
    else:
        error_checking.assert_is_integer(pressure_level_mb)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    is_surface = PRESSURE_KEY not in dataset_object.variables

    all_times_narr_hours = numpy.round(
        dataset_object.variables[TIME_KEY]
    ).astype(int)

    time_index = numpy.where(
        all_times_narr_hours == valid_time_narr_hours
    )[0][0]

    if is_surface:
        field_matrix = numpy.array(
            dataset_object.variables[field_name_orig][time_index, ...]
        )
    else:
        all_pressure_levels_mb = numpy.round(
            dataset_object.variables[PRESSURE_KEY]
        ).astype(int)

        pressure_index = numpy.where(
            all_pressure_levels_mb == pressure_level_mb
        )[0][0]

        field_matrix = numpy.array(
            dataset_object.variables[field_name_orig][
                time_index, pressure_index, ...]
        )

    return _remove_sentinel_values(field_matrix)
