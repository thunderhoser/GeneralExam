"""Reading and processing raw ERA5 data.

ERA5 = ECMWF Reanalysis 5
ECMWF = European Centre for Medium-range Weather-forecasting
"""

import os.path
import numpy
from scipy.interpolate import griddata
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import predictor_utils

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

TEMPERATURE_NAME_RAW = 'T'
HEIGHT_NAME_RAW = 'Z'
PRESSURE_NAME_RAW = 'p'
DEWPOINT_NAME_RAW = 'Td'
SPECIFIC_HUMIDITY_NAME_RAW = 'q'
U_WIND_NAME_RAW = 'u'
V_WIND_NAME_RAW = 'v'

RAW_FIELD_NAMES = [
    TEMPERATURE_NAME_RAW, HEIGHT_NAME_RAW, PRESSURE_NAME_RAW, DEWPOINT_NAME_RAW,
    SPECIFIC_HUMIDITY_NAME_RAW, U_WIND_NAME_RAW, V_WIND_NAME_RAW
]

FIELD_NAME_RAW_TO_PROCESSED = {
    TEMPERATURE_NAME_RAW: predictor_utils.TEMPERATURE_NAME,
    HEIGHT_NAME_RAW: predictor_utils.HEIGHT_NAME,
    PRESSURE_NAME_RAW: predictor_utils.PRESSURE_NAME,
    DEWPOINT_NAME_RAW: predictor_utils.DEWPOINT_NAME,
    SPECIFIC_HUMIDITY_NAME_RAW: predictor_utils.SPECIFIC_HUMIDITY_NAME,
    U_WIND_NAME_RAW: predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    V_WIND_NAME_RAW: predictor_utils.V_WIND_GRID_RELATIVE_NAME
}

FIELD_NAME_PROCESSED_TO_RAW = {
    predictor_utils.TEMPERATURE_NAME: TEMPERATURE_NAME_RAW,
    predictor_utils.HEIGHT_NAME: HEIGHT_NAME_RAW,
    predictor_utils.PRESSURE_NAME: PRESSURE_NAME_RAW,
    predictor_utils.DEWPOINT_NAME: DEWPOINT_NAME_RAW,
    predictor_utils.SPECIFIC_HUMIDITY_NAME: SPECIFIC_HUMIDITY_NAME_RAW,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: U_WIND_NAME_RAW,
    predictor_utils.U_WIND_EARTH_RELATIVE_NAME: U_WIND_NAME_RAW,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: V_WIND_NAME_RAW,
    predictor_utils.V_WIND_EARTH_RELATIVE_NAME: V_WIND_NAME_RAW
}

RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL = {
    TEMPERATURE_NAME_RAW: 2,
    PRESSURE_NAME_RAW: 0,
    DEWPOINT_NAME_RAW: 2,
    U_WIND_NAME_RAW: 10,
    V_WIND_NAME_RAW: 10
}

SURFACE_FIELD_NAME_TO_NETCDF_KEY = {
    predictor_utils.TEMPERATURE_NAME: 't2m',
    predictor_utils.HEIGHT_NAME: None,
    predictor_utils.PRESSURE_NAME: 'sp',
    predictor_utils.DEWPOINT_NAME: 'd2m',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: None,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: 'u10',
    predictor_utils.U_WIND_EARTH_RELATIVE_NAME: 'u10',
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: 'v10',
    predictor_utils.V_WIND_EARTH_RELATIVE_NAME: 'v10'
}

ISOBARIC_FIELD_NAME_TO_NETCDF_KEY = {
    predictor_utils.TEMPERATURE_NAME: 't',
    predictor_utils.HEIGHT_NAME: 'z',
    predictor_utils.PRESSURE_NAME: None,
    predictor_utils.DEWPOINT_NAME: 'd2m',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: 'q',
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: 'u',
    predictor_utils.U_WIND_EARTH_RELATIVE_NAME: 'u',
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: 'v',
    predictor_utils.V_WIND_EARTH_RELATIVE_NAME: 'v'
}

NETCDF_LATITUDES_KEY = 'latitude'
NETCDF_LONGITUDES_KEY = 'longitude'
NETCDF_HOURS_INTO_YEAR_KEY = 'time'
NETCDF_PRESSURE_LEVELS_KEY = 'level'


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


def _file_name_to_year(era5_file_name):
    """Parses year from file name.

    :param era5_file_name: See doc for `find_file`.
    :return: year: Year (integer).
    """

    error_checking.assert_is_string(era5_file_name)
    pathless_file_name = os.path.split(era5_file_name)[-1]
    return int(pathless_file_name.split('_')[1])


def _file_name_to_surface_flag(era5_file_name):
    """Determines, based on file name, whether or not it contains surface data.

    :param era5_file_name: See doc for `find_file`.
    :return: has_surface_data: Boolean flag.
    """

    pathless_file_name = os.path.split(era5_file_name)[-1]
    return len(pathless_file_name.split('_')) == 5


def _file_name_to_field(era5_file_name):
    """Parses field from file name.

    :param era5_file_name: See doc for `find_file`.
    :return: field_name: Field name in processed format.
    """

    error_checking.assert_is_string(era5_file_name)
    pathless_file_name = os.path.split(era5_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    return field_name_raw_to_processed(
        raw_field_name=extensionless_file_name.split('_')[-1],
        earth_relative=True
    )


def field_name_raw_to_processed(raw_field_name, earth_relative=False):
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


def field_name_processed_to_raw(field_name):
    """Converts field name from processed to raw format.

    :param field_name: Field name in processed format.
    :return: raw_field_name: Field name in raw format.
    """

    predictor_utils.check_field_name(field_name)
    return FIELD_NAME_PROCESSED_TO_RAW[field_name]


def find_file(top_directory_name, year, raw_field_name, has_surface_data,
              raise_error_if_missing=True):
    """Finds ERA5 file (NetCDF with one field at one pressure level for a year).

    :param top_directory_name: Name of top-level directory with raw files.
    :param year: Year (integer).
    :param raw_field_name: Field name in raw format.
    :param has_surface_data: Boolean flag.  If True, looking for file with
        surface data.  If False, looking for data at one or more pressure
        levels.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: era5_file_name: Path to ERA5 file.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(year)
    _check_raw_field_name(raw_field_name)
    error_checking.assert_is_boolean(has_surface_data)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if has_surface_data:
        era5_file_name = '{0:s}/ERA5_{1:04d}_3hrly_{2:d}m_{3:s}.nc'.format(
            top_directory_name, year,
            RAW_FIELD_NAME_TO_SURFACE_HEIGHT_M_AGL[raw_field_name],
            raw_field_name
        )
    else:
        era5_file_name = '{0:s}/ERA5_{1:04d}_3hrly_{2:s}.nc'.format(
            top_directory_name, year, raw_field_name)

    if raise_error_if_missing and not os.path.isfile(era5_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            era5_file_name)
        raise ValueError(error_string)

    return era5_file_name


def read_file(netcdf_file_name, first_time_unix_sec, last_time_unix_sec,
              pressure_level_mb=None):
    """Reads ERA5 file (NetCDF with one field at one pressure level for a year).

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
    has_surface_data = _file_name_to_surface_flag(netcdf_file_name)

    if not has_surface_data:
        error_checking.assert_is_integer(pressure_level_mb)
        error_checking.assert_is_greater(pressure_level_mb, 0)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    field_name = _file_name_to_field(netcdf_file_name)

    if has_surface_data:
        field_name_key = SURFACE_FIELD_NAME_TO_NETCDF_KEY[field_name]
        pressure_level_mb = predictor_utils.DUMMY_SURFACE_PRESSURE_MB
        pressure_level_index = None
    else:
        field_name_key = ISOBARIC_FIELD_NAME_TO_NETCDF_KEY[field_name]

        all_pressure_levels_mb = numpy.array(
            dataset_object.variables[NETCDF_PRESSURE_LEVELS_KEY][:], dtype=int
        )
        print(all_pressure_levels_mb)
        pressure_level_index = numpy.where(
            all_pressure_levels_mb == pressure_level_mb
        )[0][0]

    latitudes_deg = numpy.array(
        dataset_object.variables[NETCDF_LATITUDES_KEY][:]
    )
    longitudes_deg = numpy.array(
        dataset_object.variables[NETCDF_LONGITUDES_KEY][:]
    )

    year_as_int = _file_name_to_year(netcdf_file_name)
    year_as_string = '{0:04d}'.format(year_as_int)

    hours_into_year = numpy.round(
        dataset_object.variables[NETCDF_HOURS_INTO_YEAR_KEY][:]
    ).astype(int)

    valid_times_unix_sec = (
        time_conversion.string_to_unix_sec(year_as_string, '%Y') +
        HOURS_TO_SECONDS * hours_into_year
    )

    time_indices = numpy.where(numpy.logical_and(
        valid_times_unix_sec >= first_time_unix_sec,
        valid_times_unix_sec <= last_time_unix_sec
    ))[0]

    valid_times_unix_sec = valid_times_unix_sec[time_indices]
    data_matrix = numpy.array(
        dataset_object.variables[field_name_key][time_indices, ...]
    )

    if not has_surface_data:

        # TODO(thunderhoser): This is a HACK to deal with the fact that pressure
        # axis is inconsistent across files.
        if len(data_matrix.shape) > 3:
            if data_matrix.shape[1] > 100:
                data_matrix = data_matrix[..., pressure_level_index]
            else:
                data_matrix = data_matrix[:, pressure_level_index, ...]

    data_matrix = numpy.flip(data_matrix, axis=1)
    latitudes_deg = latitudes_deg[::-1]

    data_matrix = numpy.expand_dims(data_matrix, axis=-1)

    return {
        predictor_utils.DATA_MATRIX_KEY: data_matrix,
        predictor_utils.VALID_TIMES_KEY: valid_times_unix_sec,
        predictor_utils.LATITUDES_KEY: latitudes_deg,
        predictor_utils.LONGITUDES_KEY: longitudes_deg,
        predictor_utils.PRESSURE_LEVELS_KEY:
            numpy.array([pressure_level_mb], dtype=int),
        predictor_utils.FIELD_NAMES_KEY: [field_name]
    }


def interp_to_narr_grid(predictor_dict, grid_name, era5_x_matrix_metres=None,
                        era5_y_matrix_metres=None):
    """Interpolates ERA5 data to NARR (North American Regional Reanalysis) grid.

    M = number of rows in ERA5 grid
    N = number of columns in ERA5 grid

    `era5_x_matrix_metres` and `era5_y_matrix_metres` should be in the x-y space
    defined by the NARR's Lambert conformal projection.  If either
    `era5_x_matrix_metres is None` or `era5_y_matrix_metres is None`, these
    matrices will be created on the fly.

    :param predictor_dict: See doc for `predictor_utils.check_predictor_dict`.
    :param grid_name: Grid name (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param era5_x_matrix_metres: M-by-N numpy array with x-coordinates of ERA5
        grid points.
    :param era5_y_matrix_metres: Same but for y-coordinates.
    :return: predictor_dict: Same as input but with the following exceptions.
    predictor_dict['data_matrix']: Different spatial dimensions.
    predictor_dict['latitudes_deg']: None
    predictor_dict['longitudes_deg']: None
    """

    predictor_utils.check_predictor_dict(predictor_dict)

    narr_x_matrix_metres, narr_y_matrix_metres = (
        nwp_model_utils.get_xy_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name)
    )

    num_narr_grid_points = narr_x_matrix_metres.size
    narr_xy_matrix_metres = numpy.full((num_narr_grid_points, 2), numpy.nan)
    narr_xy_matrix_metres[:, 0] = numpy.ravel(narr_x_matrix_metres)
    narr_xy_matrix_metres[:, 1] = numpy.ravel(narr_y_matrix_metres)

    if era5_x_matrix_metres is None or era5_y_matrix_metres is None:
        era5_latitude_matrix_deg, era5_longitude_matrix_deg = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=predictor_dict[
                    predictor_utils.LATITUDES_KEY],
                unique_longitudes_deg=predictor_dict[
                    predictor_utils.LONGITUDES_KEY]
            )
        )

        era5_x_matrix_metres, era5_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                latitudes_deg=era5_latitude_matrix_deg,
                longitudes_deg=era5_longitude_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name)
        )

    num_era5_grid_points = era5_x_matrix_metres.size
    era5_xy_matrix_metres = numpy.full((num_era5_grid_points, 2), numpy.nan)
    era5_xy_matrix_metres[:, 0] = numpy.ravel(era5_x_matrix_metres)
    era5_xy_matrix_metres[:, 1] = numpy.ravel(era5_y_matrix_metres)

    num_era5_rows = len(predictor_dict[predictor_utils.LATITUDES_KEY])
    num_era5_columns = len(predictor_dict[predictor_utils.LONGITUDES_KEY])
    these_expected_dim = numpy.array(
        [num_era5_rows, num_era5_columns], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(era5_x_matrix_metres)
    error_checking.assert_is_numpy_array(
        era5_x_matrix_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(era5_y_matrix_metres)
    error_checking.assert_is_numpy_array(
        era5_y_matrix_metres, exact_dimensions=these_expected_dim)

    num_times = predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[0]
    num_fields = predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[-1]

    num_narr_rows = narr_x_matrix_metres.shape[0]
    num_narr_columns = narr_x_matrix_metres.shape[1]
    new_data_matrix = numpy.full(
        (num_times, num_narr_rows, num_narr_columns, num_fields), numpy.nan
    )

    for i in range(num_times):
        this_time_string = time_conversion.unix_sec_to_string(
            predictor_dict[predictor_utils.VALID_TIMES_KEY][i], TIME_FORMAT)

        for k in range(num_fields):
            print((
                'Interpolating field "{0:s}" at {1:d} mb and {2:s}...'
            ).format(
                predictor_dict[predictor_utils.FIELD_NAMES_KEY][k],
                predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY][k],
                this_time_string
            ))

            this_interp_object = griddata(
                era5_xy_matrix_metres,
                numpy.ravel(
                    predictor_dict[predictor_utils.DATA_MATRIX_KEY][i, ..., k]
                ),
                narr_xy_matrix_metres, method='linear',
                fill_value=numpy.nan)

            new_data_matrix[i, ..., k] = numpy.reshape(
                this_interp_object.T, (num_narr_rows, num_narr_columns)
            )

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = new_data_matrix
    predictor_dict[predictor_utils.LATITUDES_KEY] = None
    predictor_dict[predictor_utils.LONGITUDES_KEY] = None

    return predictor_dict
