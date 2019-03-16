"""IO methods for ERA5 data.

ERA5 = ECMWF Reanalysis 5
ECMWF = European Centre for Medium-range Weather-forecasting
"""

import copy
import os.path
import numpy
import netCDF4
from scipy.interpolate import griddata
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_IN_LOG_MESSAGES = '%Y-%m-%d-%H'

HOURS_TO_SECONDS = 3600
DUMMY_SURFACE_PRESSURE_MB = 1013

TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
PRESSURE_NAME = 'pressure_pascals'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
U_WIND_EARTH_RELATIVE_NAME = 'u_wind_earth_relative_m_s01'
V_WIND_EARTH_RELATIVE_NAME = 'v_wind_earth_relative_m_s01'
WET_BULB_THETA_NAME = 'wet_bulb_potential_temperature_kelvins'
U_WIND_GRID_RELATIVE_NAME = 'u_wind_grid_relative_m_s01'
V_WIND_GRID_RELATIVE_NAME = 'v_wind_grid_relative_m_s01'
DEWPOINT_NAME = 'dewpoint_kelvins'

STANDARD_FIELD_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, PRESSURE_NAME, SPECIFIC_HUMIDITY_NAME,
    U_WIND_EARTH_RELATIVE_NAME, V_WIND_EARTH_RELATIVE_NAME
]

DERIVED_FIELD_NAMES = [
    WET_BULB_THETA_NAME, U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME,
    DEWPOINT_NAME
]

FIELD_NAMES = STANDARD_FIELD_NAMES + DERIVED_FIELD_NAMES

TEMPERATURE_NAME_RAW = 'T'
HEIGHT_NAME_RAW = 'Z'
PRESSURE_NAME_RAW = 'p'
DEWPOINT_NAME_RAW = 'Td'
U_WIND_NAME_RAW = 'u'
V_WIND_NAME_RAW = 'v'

RAW_FIELD_NAMES = [
    TEMPERATURE_NAME_RAW, HEIGHT_NAME_RAW, PRESSURE_NAME_RAW, DEWPOINT_NAME_RAW,
    U_WIND_NAME_RAW, V_WIND_NAME_RAW
]

FIELD_NAME_RAW_TO_PROCESSED = {
    TEMPERATURE_NAME_RAW: TEMPERATURE_NAME,
    HEIGHT_NAME_RAW: HEIGHT_NAME,
    PRESSURE_NAME_RAW: PRESSURE_NAME,
    DEWPOINT_NAME_RAW: DEWPOINT_NAME,
    U_WIND_NAME_RAW: U_WIND_GRID_RELATIVE_NAME,
    V_WIND_NAME_RAW: V_WIND_GRID_RELATIVE_NAME
}

FIELD_NAME_PROCESSED_TO_RAW = {
    TEMPERATURE_NAME: TEMPERATURE_NAME_RAW,
    HEIGHT_NAME: HEIGHT_NAME_RAW,
    PRESSURE_NAME: PRESSURE_NAME_RAW,
    DEWPOINT_NAME: DEWPOINT_NAME_RAW,
    U_WIND_GRID_RELATIVE_NAME: U_WIND_NAME_RAW,
    U_WIND_EARTH_RELATIVE_NAME: U_WIND_NAME_RAW,
    V_WIND_GRID_RELATIVE_NAME: V_WIND_NAME_RAW,
    V_WIND_EARTH_RELATIVE_NAME: V_WIND_NAME_RAW
}

RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL = {
    TEMPERATURE_NAME_RAW: 2,
    PRESSURE_NAME_RAW: 0,
    DEWPOINT_NAME_RAW: 2,
    U_WIND_NAME_RAW: 10,
    V_WIND_NAME_RAW: 10
}

LATITUDES_KEY_RAW = 'latitude'
LONGITUDES_KEY_RAW = 'longitude'
HOURS_INTO_YEAR_KEY_RAW = 'time'
DATA_MATRIX_KEY_RAW = 'VAR_2D'

DATA_MATRIX_KEY = 'data_matrix'
LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
VALID_TIMES_KEY = 'valid_times_unix_sec'
VALID_TIME_KEY = 'valid_time_unix_sec'
PRESSURE_LEVELS_KEY = 'pressure_levels_mb'
FIELD_NAMES_KEY = 'field_names'

ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
PRESSURE_LEVEL_DIM_KEY = 'pressure_level'
FIELD_DIMENSION_KEY = 'field'
FIELD_NAME_CHAR_DIM_KEY = 'field_name_character'


def _check_raw_field_name(raw_field_name):
    """Error-checks raw field name.

    :param raw_field_name: Field name in raw format (used to name raw files).
    :raises: ValueError: if `field_name not in RAW_FIELD_NAMES`
    """

    error_checking.assert_is_string(raw_field_name)

    if raw_field_name not in RAW_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid field names (listed above) do not include "{1:s}".'
        ).format(str(RAW_FIELD_NAMES), raw_field_name)

        raise ValueError(error_string)


def _raw_file_name_to_year(raw_file_name):
    """Parses year from name of raw file.

    :param raw_file_name: See doc for `find_raw_file`.
    :return: year: Year (integer).
    """

    error_checking.assert_is_string(raw_file_name)
    pathless_file_name = os.path.split(raw_file_name)[-1]
    return int(pathless_file_name.split('_')[1])


def _raw_file_name_to_pressure(raw_file_name):
    """Parses pressure level from name of raw file.

    :param raw_file_name: See doc for `find_raw_file`.
    :return: pressure_level_mb: Pressure level (millibars).
    """

    error_checking.assert_is_string(raw_file_name)
    pathless_file_name = os.path.split(raw_file_name)[-1]

    pressure_level_string = pathless_file_name.split('_')[3]
    if 'mb' in pressure_level_string:
        return int(pressure_level_string.replace('mb', ''))

    return DUMMY_SURFACE_PRESSURE_MB


def _raw_file_name_to_field(raw_file_name):
    """Parses field from name of raw file.

    :param raw_file_name: See doc for `find_raw_file`.
    :return: field_name: Field name in processed format.
    """

    error_checking.assert_is_string(raw_file_name)
    pathless_file_name = os.path.split(raw_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    return field_name_raw_to_processed(
        raw_field_name=extensionless_file_name.split('_')[4],
        earth_relative=True
    )


def _check_era5_data(era5_dict):
    """Error-checks ERA5 data.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (fields)
    P = number of pressure levels

    :param era5_dict: Dictionary with the following keys.
    era5_dict['data_matrix']: T-by-M-by-N-by-P-by-C numpy array of data values.
    era5_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    era5_dict['latitudes_deg']: length-M numpy array of grid-point latitudes
        (deg N).  If data are on the NARR grid, this should be None.
    era5_dict['longitudes_deg']: length-N numpy array of grid-point longitudes
        (deg E).  If data are on the NARR grid, this should be None.
    era5_dict['pressure_levels_mb']: length-P numpy array of pressure levels
        (use 1013 mb to denote surface).
    era5_dict['field_names']: length-C list of field names in processed format.
    """

    error_checking.assert_is_integer_numpy_array(era5_dict[VALID_TIMES_KEY])
    error_checking.assert_is_numpy_array(
        era5_dict[VALID_TIMES_KEY], num_dimensions=1)

    error_checking.assert_is_integer_numpy_array(era5_dict[PRESSURE_LEVELS_KEY])
    error_checking.assert_is_greater_numpy_array(
        era5_dict[PRESSURE_LEVELS_KEY], 0)
    error_checking.assert_is_numpy_array(
        era5_dict[PRESSURE_LEVELS_KEY], num_dimensions=1)

    error_checking.assert_is_numpy_array(
        numpy.array(era5_dict[FIELD_NAMES_KEY]), num_dimensions=1)

    num_fields = len(era5_dict[FIELD_NAMES_KEY])
    for j in range(num_fields):
        check_field_name(era5_dict[FIELD_NAMES_KEY][j])

    on_narr_grid = (
        era5_dict[LATITUDES_KEY] is None and era5_dict[LONGITUDES_KEY] is None
    )

    if not on_narr_grid:
        error_checking.assert_is_valid_lat_numpy_array(era5_dict[LATITUDES_KEY])
        error_checking.assert_is_numpy_array(
            era5_dict[LATITUDES_KEY], num_dimensions=1)

        error_checking.assert_is_valid_lng_numpy_array(
            era5_dict[LONGITUDES_KEY], positive_in_west_flag=True)
        error_checking.assert_is_numpy_array(
            era5_dict[LONGITUDES_KEY], num_dimensions=1)

    # error_checking.assert_is_numpy_array_without_nan(era5_dict[DATA_MATRIX_KEY])

    this_num_dimensions = len(era5_dict[DATA_MATRIX_KEY].shape)
    error_checking.assert_is_geq(this_num_dimensions, 5)
    error_checking.assert_is_leq(this_num_dimensions, 5)

    num_times = len(era5_dict[VALID_TIMES_KEY])
    num_pressure_levels = len(era5_dict[PRESSURE_LEVELS_KEY])

    if on_narr_grid:
        num_grid_rows = era5_dict[DATA_MATRIX_KEY].shape[1]
        num_grid_columns = era5_dict[DATA_MATRIX_KEY].shape[2]
    else:
        num_grid_rows = len(era5_dict[LATITUDES_KEY])
        num_grid_columns = len(era5_dict[LONGITUDES_KEY])

    these_expected_dim = numpy.array(
        [num_times, num_grid_rows, num_grid_columns, num_pressure_levels,
         num_fields],
        dtype=int)

    error_checking.assert_is_numpy_array(
        era5_dict[DATA_MATRIX_KEY], exact_dimensions=these_expected_dim)


def check_field_name(field_name, require_standard=False):
    """Error-checks field name.

    :param field_name: Field name (in processed format, not raw format).
    :param require_standard: Boolean flag.  If True, this method will ensure
        that `field_name in STANDARD_FIELD_NAMES`.  If False, this method will
        ensure that `field_name in FIELD_NAMES`.
    :raises: ValueError:
        if `require_standard and field_name not in STANDARD_FIELD_NAMES`
    :raises: ValueError:
        if `not require_standard and field_name not in FIELD_NAMES`
    """

    error_checking.assert_is_string(field_name)
    error_checking.assert_is_boolean(require_standard)

    if require_standard:
        valid_field_names = STANDARD_FIELD_NAMES
    else:
        valid_field_names = FIELD_NAMES

    if field_name not in valid_field_names:
        error_string = (
            '\n{0:s}\nValid field names (listed above) do not include "{1:s}".'
        ).format(str(valid_field_names), field_name)

        raise ValueError(error_string)


def field_name_raw_to_processed(raw_field_name, earth_relative=False):
    """Converts field name from raw to processed format.

    :param raw_field_name: Field name in raw format.
    :param earth_relative: Boolean flag.  If raw_field_name is a wind component
        and earth_relative = True, will return equivalent field name for
        Earth-relative wind.  Otherwise, will return equivalent field name for
        grid-relative wind.
    :return: field_name: Field in processed format.
    """

    _check_raw_field_name(raw_field_name)
    error_checking.assert_is_boolean(earth_relative)

    field_name = FIELD_NAME_RAW_TO_PROCESSED[raw_field_name]
    if earth_relative:
        field_name = field_name.replace('grid_relative', 'earth_relative')

    return field_name


def field_name_processed_to_raw(field_name):
    """Converts field name from processed to raw format.

    :param field_name: Field name in processed format.
    :return: raw_field_name: Field name in raw format.
    """

    check_field_name(field_name)
    return FIELD_NAME_PROCESSED_TO_RAW[field_name]


def find_raw_file(top_directory_name, year, raw_field_name, pressure_level_mb,
                  raise_error_if_missing=True):
    """Finds raw file (NetCDF with one field at one pressure level for a year).

    :param top_directory_name: Name of top-level directory with raw files.
    :param year: Year (integer).
    :param raw_field_name: Field name in raw format.
    :param pressure_level_mb: Pressure level.  If looking for surface data, make
        this 1013.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: raw_file_name: Path to raw file.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(year)
    _check_raw_field_name(raw_field_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if pressure_level_mb == DUMMY_SURFACE_PRESSURE_MB:
        raw_file_name = '{0:s}/ERA5_{1:04d}_3hrly_{2:d}m_{3:s}.nc'.format(
            top_directory_name, year,
            RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL[raw_field_name],
            raw_field_name
        )
    else:
        error_checking.assert_is_integer(pressure_level_mb)
        error_checking.assert_is_greater(pressure_level_mb, 0)

        raw_file_name = '{0:s}/ERA5_{1:04d}_3hrly_{2:d}mb_{3:s}.nc'.format(
            top_directory_name, year, pressure_level_mb, raw_field_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            raw_file_name)
        raise ValueError(error_string)

    return raw_file_name


def read_raw_file(netcdf_file_name, first_time_unix_sec, last_time_unix_sec):
    """Reads raw file (NetCDF with one field at one pressure level for a year).

    :param netcdf_file_name: Path to input file.
    :param first_time_unix_sec: First time step to read.
    :param last_time_unix_sec: Last time step to read.
    :return: era5_dict: See doc for `_check_era5_data`.
    """

    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    latitudes_deg = numpy.array(dataset_object.variables[LATITUDES_KEY_RAW][:])
    longitudes_deg = numpy.array(
        dataset_object.variables[LONGITUDES_KEY_RAW][:]
    )

    year_as_int = _raw_file_name_to_year(netcdf_file_name)
    year_as_string = '{0:04d}'.format(year_as_int)

    hours_into_year = numpy.round(
        dataset_object.variables[HOURS_INTO_YEAR_KEY_RAW][:]
    ).astype(int)

    valid_times_unix_sec = (
        time_conversion.string_to_unix_sec(year_as_string, '%Y') +
        HOURS_TO_SECONDS * hours_into_year
    )

    good_indices = numpy.where(numpy.logical_and(
        valid_times_unix_sec >= first_time_unix_sec,
        valid_times_unix_sec <= last_time_unix_sec
    ))[0]

    valid_times_unix_sec = valid_times_unix_sec[good_indices]
    data_matrix = None

    # TODO(thunderhoser): All these try-except statements are a HACK.

    try:
        data_matrix = numpy.array(
            dataset_object.variables[DATA_MATRIX_KEY_RAW][good_indices, ...]
        )
    except KeyError:
        pass

    if data_matrix is None:
        try:
            data_matrix = numpy.array(
                dataset_object.variables['t2m'][good_indices, ...]
            )
        except KeyError:
            pass

    if data_matrix is None:
        try:
            data_matrix = numpy.array(
                dataset_object.variables['d2m'][good_indices, ...]
            )
        except KeyError:
            pass

    if data_matrix is None:
        try:
            data_matrix = numpy.array(
                dataset_object.variables['u10'][good_indices, ...]
            )
        except KeyError:
            pass

    if data_matrix is None:
        try:
            data_matrix = numpy.array(
                dataset_object.variables['v10'][good_indices, ...]
            )
        except KeyError:
            pass

    if data_matrix is None:
        try:
            data_matrix = numpy.array(
                dataset_object.variables['sp'][good_indices, ...]
            )
        except KeyError:
            pass

    if data_matrix is None:
        print dataset_object.variables

    data_matrix = numpy.flip(data_matrix, axis=1)
    latitudes_deg = latitudes_deg[::-1]

    data_matrix = numpy.expand_dims(data_matrix, axis=-1)
    data_matrix = numpy.expand_dims(data_matrix, axis=-1)

    pressure_levels_mb = numpy.array(
        [_raw_file_name_to_pressure(netcdf_file_name)], dtype=int
    )
    field_names = [_raw_file_name_to_field(netcdf_file_name)]

    return {
        DATA_MATRIX_KEY: data_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec,
        LATITUDES_KEY: latitudes_deg,
        LONGITUDES_KEY: longitudes_deg,
        PRESSURE_LEVELS_KEY: pressure_levels_mb,
        FIELD_NAMES_KEY: field_names
    }


def interp_to_narr_grid(era5_dict, era5_x_matrix_metres=None,
                        era5_y_matrix_metres=None):
    """Interpolates ERA5 data to NARR (North American Regional Reanalysis) grid.

    M = number of rows in ERA5 grid
    N = number of columns in ERA5 grid

    `era5_x_matrix_metres` and `era5_y_matrix_metres` should be in the x-y space
    defined by the NARR's Lambert conformal projection.  If either
    `era5_x_matrix_metres is None` or `era5_y_matrix_metres is None`, these
    matrices will be created on the fly.

    :param era5_dict: See doc for `_check_era5_data`.
    :param era5_x_matrix_metres: M-by-N numpy array with x-coordinates of ERA5
        grid points.
    :param era5_y_matrix_metres: Same but for y-coordinates.
    :return: era5_dict: Same as input, with 3 exceptions.
    era5_dict['data_matrix']: Different spatial dimensions.
    era5_dict['latitudes_deg']: None
    era5_dict['longitudes_deg']: None
    """

    _check_era5_data(era5_dict)

    narr_x_matrix_metres, narr_y_matrix_metres = (
        nwp_model_utils.get_xy_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    num_narr_grid_points = narr_x_matrix_metres.size
    narr_xy_matrix_metres = numpy.full((num_narr_grid_points, 2), numpy.nan)
    narr_xy_matrix_metres[:, 0] = numpy.ravel(narr_x_matrix_metres)
    narr_xy_matrix_metres[:, 1] = numpy.ravel(narr_y_matrix_metres)

    if era5_x_matrix_metres is None or era5_y_matrix_metres is None:
        era5_latitude_matrix_deg, era5_longitude_matrix_deg = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=era5_dict[LATITUDES_KEY],
                unique_longitudes_deg=era5_dict[LONGITUDES_KEY]
            )
        )

        era5_x_matrix_metres, era5_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                latitudes_deg=era5_latitude_matrix_deg,
                longitudes_deg=era5_longitude_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME)
        )

    num_era5_grid_points = era5_x_matrix_metres.size
    era5_xy_matrix_metres = numpy.full((num_era5_grid_points, 2), numpy.nan)
    era5_xy_matrix_metres[:, 0] = numpy.ravel(era5_x_matrix_metres)
    era5_xy_matrix_metres[:, 1] = numpy.ravel(era5_y_matrix_metres)

    num_era5_rows = len(era5_dict[LATITUDES_KEY])
    num_era5_columns = len(era5_dict[LONGITUDES_KEY])
    these_expected_dim = numpy.array(
        [num_era5_rows, num_era5_columns], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(era5_x_matrix_metres)
    error_checking.assert_is_numpy_array(
        era5_x_matrix_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(era5_y_matrix_metres)
    error_checking.assert_is_numpy_array(
        era5_y_matrix_metres, exact_dimensions=these_expected_dim)

    num_times = era5_dict[DATA_MATRIX_KEY].shape[0]
    num_pressure_levels = era5_dict[DATA_MATRIX_KEY].shape[-2]
    num_fields = era5_dict[DATA_MATRIX_KEY].shape[-1]

    num_narr_rows = narr_x_matrix_metres.shape[0]
    num_narr_columns = narr_x_matrix_metres.shape[1]
    new_data_matrix = numpy.full(
        (num_times, num_narr_rows, num_narr_columns, num_pressure_levels,
         num_fields),
        numpy.nan)

    for i in range(num_times):
        this_time_string = time_conversion.unix_sec_to_string(
            era5_dict[VALID_TIMES_KEY][i], TIME_FORMAT_IN_LOG_MESSAGES)

        for j in range(num_pressure_levels):
            for k in range(num_fields):
                print (
                    'Interpolating field "{0:s}" at {1:d} mb and {2:s}...'
                ).format(
                    era5_dict[FIELD_NAMES_KEY][k],
                    era5_dict[PRESSURE_LEVELS_KEY][j], this_time_string
                )

                this_interp_object = griddata(
                    era5_xy_matrix_metres,
                    numpy.ravel(era5_dict[DATA_MATRIX_KEY][i, ..., j, k]),
                    narr_xy_matrix_metres, method='linear',
                    fill_value=numpy.nan)

                new_data_matrix[i, ..., j, k] = numpy.reshape(
                    this_interp_object.T, (num_narr_rows, num_narr_columns)
                )

                # this_interp_object = interp2d(
                #     era5_x_matrix_metres, era5_y_matrix_metres,
                #     era5_dict[DATA_MATRIX_KEY][i, ..., j, k], kind='linear',
                #     bounds_error=False, fill_value=numpy.nan)
                #
                # new_data_matrix[i, ..., j, k] = this_interp_object(
                #     narr_x_matrix_metres, narr_y_matrix_metres)

                # new_data_matrix[i, ..., j, k] = general_utils.fill_nans(
                #     new_data_matrix[i, ..., j, k]
                # )

    era5_dict[DATA_MATRIX_KEY] = new_data_matrix
    era5_dict[LATITUDES_KEY] = None
    era5_dict[LONGITUDES_KEY] = None

    return era5_dict


def find_processed_file(top_directory_name, valid_time_unix_sec,
                        raise_error_if_missing=True):
    """Finds processed file (NetCDF with all data at one time step).

    :param top_directory_name: Name of top-level directory with processed files.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: processed_file_name: Path to processed file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT)

    processed_file_name = '{0:s}/{1:s}/era5_processed_{2:s}.nc'.format(
        top_directory_name, valid_time_string[:6], valid_time_string)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            processed_file_name)
        raise ValueError(error_string)

    return processed_file_name


def write_processed_file(netcdf_file_name, era5_dict):
    """Writes data to processed file (NetCDF with all data at one time step).

    :param netcdf_file_name: Path to output file.
    :param era5_dict: See doc for `_check_era5_data`.
    :raises: ValueError: if `era5_dict` contains more than one time step.
    """

    num_times = len(era5_dict[VALID_TIMES_KEY])
    if num_times > 1:
        error_string = (
            'Dictionary should contain one time step, not {0:d}.'
        ).format(num_times)

        raise ValueError(error_string)

    _check_era5_data(era5_dict)
    era5_dict[DATA_MATRIX_KEY] = era5_dict[DATA_MATRIX_KEY][0, ...]

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes.
    dataset_object.setncattr(VALID_TIME_KEY, era5_dict[VALID_TIMES_KEY][0])

    # Set dimensions.
    dataset_object.createDimension(
        ROW_DIMENSION_KEY, era5_dict[DATA_MATRIX_KEY].shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, era5_dict[DATA_MATRIX_KEY].shape[1]
    )
    dataset_object.createDimension(
        PRESSURE_LEVEL_DIM_KEY, era5_dict[DATA_MATRIX_KEY].shape[2]
    )
    dataset_object.createDimension(
        FIELD_DIMENSION_KEY, era5_dict[DATA_MATRIX_KEY].shape[3]
    )

    num_field_name_chars = numpy.max(numpy.array(
        [len(f) for f in era5_dict[FIELD_NAMES_KEY]]
    ))

    dataset_object.createDimension(FIELD_NAME_CHAR_DIM_KEY,
                                   num_field_name_chars)

    # Add data matrix.
    dataset_object.createVariable(
        DATA_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
                    PRESSURE_LEVEL_DIM_KEY, FIELD_DIMENSION_KEY)
    )
    dataset_object.variables[DATA_MATRIX_KEY][:] = era5_dict[DATA_MATRIX_KEY]

    # Add latitudes.
    if era5_dict[LATITUDES_KEY] is not None:
        dataset_object.createVariable(
            LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY)
        dataset_object.variables[LATITUDES_KEY][:] = era5_dict[LATITUDES_KEY]

    # Add longitudes.
    if era5_dict[LONGITUDES_KEY] is not None:
        dataset_object.createVariable(
            LONGITUDES_KEY, datatype=numpy.float32,
            dimensions=COLUMN_DIMENSION_KEY)
        dataset_object.variables[LONGITUDES_KEY][:] = era5_dict[LONGITUDES_KEY]

    # Add pressure levels.
    dataset_object.createVariable(
        PRESSURE_LEVELS_KEY, datatype=numpy.int32,
        dimensions=PRESSURE_LEVEL_DIM_KEY)

    dataset_object.variables[PRESSURE_LEVELS_KEY][:] = era5_dict[
        PRESSURE_LEVELS_KEY]

    # Add field names.
    this_string_type = 'S{0:d}'.format(num_field_name_chars)
    field_names_char_array = netCDF4.stringtochar(numpy.array(
        era5_dict[FIELD_NAMES_KEY], dtype=this_string_type
    ))

    dataset_object.createVariable(
        FIELD_NAMES_KEY, datatype='S1',
        dimensions=(FIELD_DIMENSION_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[FIELD_NAMES_KEY][:] = numpy.array(
        field_names_char_array)

    dataset_object.close()


def read_processed_file(
        netcdf_file_name, metadata_only=False, pressure_levels_to_keep_mb=None,
        field_names_to_keep=None):
    """Reads processed file (NetCDF with all data at one time step).

    :param netcdf_file_name: Path to input file.
    :param metadata_only: Boolean flag.  If True, will read only metadata
        (everything except the big data matrix).
    :param pressure_levels_to_keep_mb: [used only if `metadata_only == False`]
        1-D numpy array of pressure levels (millibars) to read.  Use 1013 to
        denote surface.  If you want to read all pressure levels, make this
        None.
    :param field_names_to_keep: [used only if `metadata_only == False`]
        1-D list with names of fields to read (in processed format).  If you
        want to read all fields, make this None.
    :return: era5_dict: See doc for `write_processed_file`.  If
        `metadata_only = True`, this dictionary will not contain "data_matrix".
    """

    error_checking.assert_is_boolean(metadata_only)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    valid_time_unix_sec = int(numpy.round(
        getattr(dataset_object, VALID_TIME_KEY)
    ))
    valid_times_unix_sec = numpy.array([valid_time_unix_sec], dtype=int)

    era5_dict = {
        VALID_TIMES_KEY: valid_times_unix_sec,
        PRESSURE_LEVELS_KEY: numpy.array(
            dataset_object.variables[PRESSURE_LEVELS_KEY][:], dtype=int),
        FIELD_NAMES_KEY: [
            str(f) for f in netCDF4.chartostring(
                dataset_object.variables[FIELD_NAMES_KEY][:])
        ]
    }

    if LATITUDES_KEY in dataset_object.variables:
        era5_dict.update({
            LATITUDES_KEY:
                numpy.array(dataset_object.variables[LATITUDES_KEY][:]),
            LONGITUDES_KEY:
                numpy.array(dataset_object.variables[LONGITUDES_KEY][:])
        })
    else:
        era5_dict.update({
            LATITUDES_KEY: None,
            LONGITUDES_KEY: None
        })

    if metadata_only:
        return era5_dict

    era5_dict[DATA_MATRIX_KEY] = numpy.array(
        dataset_object.variables[DATA_MATRIX_KEY][:]
    )
    era5_dict[DATA_MATRIX_KEY] = numpy.expand_dims(
        era5_dict[DATA_MATRIX_KEY], axis=0)

    if pressure_levels_to_keep_mb is None:
        pressure_levels_to_keep_mb = era5_dict[PRESSURE_LEVELS_KEY] + 0

    pressure_levels_to_keep_mb = numpy.round(
        pressure_levels_to_keep_mb
    ).astype(int)

    error_checking.assert_is_numpy_array(
        pressure_levels_to_keep_mb, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(pressure_levels_to_keep_mb, 0)

    if field_names_to_keep is None:
        field_names_to_keep = copy.deepcopy(era5_dict[FIELD_NAMES_KEY])

    error_checking.assert_is_numpy_array(
        numpy.array(field_names_to_keep), num_dimensions=1)

    pressure_indices = numpy.array([
        numpy.where(era5_dict[PRESSURE_LEVELS_KEY] == p)[0][0]
        for p in pressure_levels_to_keep_mb
    ], dtype=int)

    field_indices = numpy.array([
        era5_dict[FIELD_NAMES_KEY].index(f) for f in field_names_to_keep
    ], dtype=int)

    era5_dict[PRESSURE_LEVELS_KEY] = pressure_levels_to_keep_mb
    era5_dict[FIELD_NAMES_KEY] = field_names_to_keep

    era5_dict[DATA_MATRIX_KEY] = numpy.take(
        era5_dict[DATA_MATRIX_KEY], indices=pressure_indices, axis=-2)
    era5_dict[DATA_MATRIX_KEY] = numpy.take(
        era5_dict[DATA_MATRIX_KEY], indices=field_indices, axis=-1)

    return era5_dict
