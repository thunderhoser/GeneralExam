"""Reading and processing raw NARR data from NetCDF files."""

import os.path
import numpy
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import predictor_utils

SENTINEL_VALUE = -9e36
HOURS_TO_SECONDS = 3600
NARR_ZERO_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1800-01-01-00', '%Y-%m-%d-%H')

TIME_FORMAT_MONTH = '%Y%m'

ONLINE_SURFACE_DIR_NAME = 'ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel'
ONLINE_PRESSURE_LEVEL_DIR_NAME = 'ftp://ftp.cdc.noaa.gov/Datasets/NARR/pressure'

TEMPERATURE_NAME_RAW = 'air'
HEIGHT_NAME_RAW = 'hgt'
PRESSURE_NAME_RAW = 'pres'
SPECIFIC_HUMIDITY_NAME_RAW = 'shum'
U_WIND_NAME_RAW = 'uwnd'
V_WIND_NAME_RAW = 'vwnd'

RAW_FIELD_NAMES = [
    TEMPERATURE_NAME_RAW, HEIGHT_NAME_RAW, PRESSURE_NAME_RAW,
    SPECIFIC_HUMIDITY_NAME_RAW, U_WIND_NAME_RAW, V_WIND_NAME_RAW
]

FIELD_NAME_RAW_TO_PROCESSED = {
    TEMPERATURE_NAME_RAW: predictor_utils.TEMPERATURE_NAME,
    HEIGHT_NAME_RAW: predictor_utils.HEIGHT_NAME,
    PRESSURE_NAME_RAW: predictor_utils.PRESSURE_NAME,
    SPECIFIC_HUMIDITY_NAME_RAW: predictor_utils.SPECIFIC_HUMIDITY_NAME,
    U_WIND_NAME_RAW: predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    V_WIND_NAME_RAW: predictor_utils.V_WIND_GRID_RELATIVE_NAME
}

FIELD_NAME_PROCESSED_TO_RAW = {
    predictor_utils.TEMPERATURE_NAME: TEMPERATURE_NAME_RAW,
    predictor_utils.HEIGHT_NAME: HEIGHT_NAME_RAW,
    predictor_utils.PRESSURE_NAME: PRESSURE_NAME_RAW,
    predictor_utils.SPECIFIC_HUMIDITY_NAME: SPECIFIC_HUMIDITY_NAME_RAW,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: U_WIND_NAME_RAW,
    predictor_utils.U_WIND_EARTH_RELATIVE_NAME: U_WIND_NAME_RAW,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: V_WIND_NAME_RAW,
    predictor_utils.V_WIND_EARTH_RELATIVE_NAME: V_WIND_NAME_RAW
}

RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL = {
    TEMPERATURE_NAME_RAW: 2,
    PRESSURE_NAME_RAW: 0,
    SPECIFIC_HUMIDITY_NAME_RAW: 2,
    U_WIND_NAME_RAW: 10,
    V_WIND_NAME_RAW: 10
}

NETCDF_PRESSURE_LEVEL_KEY = 'level'
NETCDF_TIME_KEY = 'time'


def _check_raw_field_name(raw_field_name, at_surface=None):
    """Error-checks raw field name.

    :param raw_field_name: Field name in raw format (used to name raw files).
    :param at_surface: Boolean flag.  If True, will ensure that field name is
        valid at surface.  If False, will ensure that field is valid at pressure
        level.  If None, will allow any field name.
    :raises: ValueError: if `field_name not in RAW_FIELD_NAMES`
    """

    error_checking.assert_is_string(raw_field_name)
    if at_surface is not None:
        error_checking.assert_is_boolean(at_surface)

    if at_surface is None:
        valid_field_names = RAW_FIELD_NAMES
    elif at_surface:
        valid_field_names = [f for f in RAW_FIELD_NAMES if f != HEIGHT_NAME_RAW]
    else:
        valid_field_names = [
            f for f in RAW_FIELD_NAMES if f != PRESSURE_NAME_RAW
        ]

    if raw_field_name not in valid_field_names:
        error_string = (
            '\n{0:s}\nValid field names (listed above) do not include "{1:s}".'
        ).format(str(valid_field_names), raw_field_name)

        raise ValueError(error_string)


def _field_name_raw_to_processed(raw_field_name, earth_relative=False):
    """Converts field name from raw to processed format.

    :param raw_field_name: Field name in raw format.
    :param earth_relative: Boolean flag.  If raw_field_name is a wind component
        and earth_relative = True, will return equivalent field name for
        Earth-relative wind.  Otherwise, will return equivalent field name for
        grid-relative wind.
    :return: field_name: Field name in processed format.
    """

    _check_raw_field_name(raw_field_name)
    error_checking.assert_is_boolean(earth_relative)

    field_name = FIELD_NAME_RAW_TO_PROCESSED[raw_field_name]
    if earth_relative:
        field_name = field_name.replace('grid_relative', 'earth_relative')

    return field_name


def _field_name_processed_to_raw(field_name):
    """Converts field name from processed to raw format.

    :param field_name: Field name in processed format.
    :return: raw_field_name: Field name in raw format.
    """

    predictor_utils.check_field_name(field_name)
    return FIELD_NAME_PROCESSED_TO_RAW[field_name]


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


def _remove_sentinel_values(data_matrix):
    """Removes sentinel values from field.

    :param data_matrix: numpy array with data values.
    :return: data_matrix: Same as input, except that sentinel values have been
        replaced with NaN's.
    """

    data_matrix[data_matrix < SENTINEL_VALUE] = numpy.nan
    return data_matrix


def _file_name_to_surface_flag(narr_file_name):
    """Determines, based on file name, whether or not it contains surface data.

    :param narr_file_name: See doc for `find_file`.
    :return: at_surface: Boolean flag.
    """

    pathless_file_name = os.path.split(narr_file_name)[-1]
    second_word = pathless_file_name.split('.')[1]

    try:
        int(second_word)
        return False
    except ValueError:
        return True


def _file_name_to_field(narr_file_name):
    """Parses field from file name.

    :param narr_file_name: See doc for `find_file`.
    :return: field_name: Field name in processed format.
    """

    pathless_file_name = os.path.split(narr_file_name)[-1]

    return _field_name_raw_to_processed(
        raw_field_name=pathless_file_name.split('.')[0], earth_relative=True)


def _get_pathless_file_name(field_name, month_string, at_surface):
    """Returns pathless name for NetCDF file (with one variable for one month).

    :param field_name: See doc for `find_file`.
    :param month_string: Same.
    :param at_surface: Same.
    :return: pathless_file_name: Pathless name for NetCDF file.
    """

    raw_field_name = _field_name_processed_to_raw(field_name)
    _check_raw_field_name(raw_field_name=raw_field_name, at_surface=at_surface)

    if not at_surface:
        return '{0:s}.{1:s}.nc'.format(raw_field_name, month_string)

    height_m_agl = RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL[raw_field_name]
    if height_m_agl == 0:
        height_string = 'sfc'
    else:
        height_string = '{0:d}m'.format(height_m_agl)

    return '{0:s}.{1:s}.{2:s}.nc'.format(
        raw_field_name, height_string, month_string[:4]
    )


def find_file(top_directory_name, field_name, month_string, at_surface,
              raise_error_if_missing=True):
    """Finds NetCDF file (with one variable for one month).

    :param top_directory_name: Name of top-level directory for NetCDF files with
        NARR data.
    :param field_name: Field name (must be accepted by
        `predictor_utils.check_field_name`).
    :param month_string: Month (format "yyyymm").
    :param at_surface: Boolean flag.  If True, will look for file with surface
        data.  If False, will look for file with pressure-level data.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: narr_file_name: Path to NetCDF file with NARR data.  If file is
        missing and `raise_error_if_missing = False`, this will be the expected
        path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    time_conversion.string_to_unix_sec(month_string, TIME_FORMAT_MONTH)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_file_name(
        field_name=field_name, month_string=month_string, at_surface=at_surface)

    narr_file_name = '{0:s}/{1:s}'.format(
        top_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(narr_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            narr_file_name)
        raise ValueError(error_string)

    return narr_file_name


def download_file(
        top_local_dir_name, field_name, month_string, at_surface=False,
        raise_error_if_fails=False):
    """Downloads NetCDF file to the local machine.

    :param top_local_dir_name: Name of top-level target directory (local
        directory for NetCDF files with NARR data).
    :param field_name: See doc for `_get_pathless_file_name`.
    :param month_string: Same.
    :param at_surface: Same.
    :param raise_error_if_fails: Boolean flag.  If download fails and
        `raise_error_if_fails = True`, this method will error out.
    :return: local_file_name: Path to downloaded file on local machine.  If
        download failed and `raise_error_if_fails = False`, this is None.
    """

    local_file_name = find_file(
        top_directory_name=top_local_dir_name, field_name=field_name,
        month_string=month_string, at_surface=at_surface,
        raise_error_if_missing=False)

    pathless_file_name = _get_pathless_file_name(
        field_name=field_name, month_string=month_string, at_surface=at_surface)

    if at_surface:
        online_dir_name = ONLINE_SURFACE_DIR_NAME + ''
    else:
        online_dir_name = ONLINE_PRESSURE_LEVEL_DIR_NAME + ''

    online_file_name = '{0:s}/{1:s}'.format(online_dir_name, pathless_file_name)

    return downloads.download_files_via_http(
        online_file_names=[online_file_name],
        local_file_names=[local_file_name],
        raise_error_if_fails=raise_error_if_fails)


def read_file(netcdf_file_name, first_time_unix_sec, last_time_unix_sec,
              pressure_level_mb=None):
    """Reads data from NetCDF file (with one variable for one month).

    If file contains pressure-level data, this method will read only a single
    pressure level.

    :param netcdf_file_name: Path to input file.
    :param first_time_unix_sec: First time step to read.
    :param last_time_unix_sec: Last time step to read.
    :param pressure_level_mb: Pressure level (millibars) to read.  Used only if
        file does *not* contain surface data.
    :return: predictor_dict: See doc for `predictor_utils.check_predictor_dict`.
    """

    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)
    at_surface = _file_name_to_surface_flag(netcdf_file_name)

    if not at_surface:
        error_checking.assert_is_integer(pressure_level_mb)
        error_checking.assert_is_greater(pressure_level_mb, 0)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    field_name = _file_name_to_field(netcdf_file_name)
    raw_field_name = _field_name_processed_to_raw(field_name)

    narr_times_hours = numpy.round(
        dataset_object.variables[NETCDF_TIME_KEY]
    ).astype(int)

    valid_times_unix_sec = numpy.array(
        [_narr_to_unix_time(t) for t in narr_times_hours], dtype=int
    )

    time_indices = numpy.where(numpy.logical_and(
        valid_times_unix_sec >= first_time_unix_sec,
        valid_times_unix_sec <= last_time_unix_sec
    ))[0]

    valid_times_unix_sec = valid_times_unix_sec[time_indices]

    if at_surface:
        data_matrix = numpy.array(
            dataset_object.variables[raw_field_name][time_indices, ...]
        )
    else:
        all_pressure_levels_mb = numpy.round(
            dataset_object.variables[NETCDF_PRESSURE_LEVEL_KEY]
        ).astype(int)

        pressure_index = numpy.where(
            all_pressure_levels_mb == pressure_level_mb
        )[0][0]

        data_matrix = numpy.array(
            dataset_object.variables[raw_field_name][
                time_indices, pressure_index, ...]
        )

    data_matrix = _remove_sentinel_values(data_matrix)
    data_matrix = numpy.expand_dims(data_matrix, axis=-1)

    return {
        predictor_utils.DATA_MATRIX_KEY: data_matrix,
        predictor_utils.VALID_TIMES_KEY: valid_times_unix_sec,
        predictor_utils.LATITUDES_KEY: None,
        predictor_utils.LONGITUDES_KEY: None,
        predictor_utils.PRESSURE_LEVELS_KEY:
            numpy.array([pressure_level_mb], dtype=int),
        predictor_utils.FIELD_NAMES_KEY: [field_name]
    }
