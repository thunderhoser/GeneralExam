"""Methods for handling predictor fields."""

import numpy
from gewittergefahr.gg_utils import error_checking

ERA5_GRID_NAME = 'era5_grid'
DUMMY_SURFACE_PRESSURE_MB = 1013

TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
PRESSURE_NAME = 'pressure_pascals'
DEWPOINT_NAME = 'dewpoint_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
U_WIND_EARTH_RELATIVE_NAME = 'u_wind_earth_relative_m_s01'
V_WIND_EARTH_RELATIVE_NAME = 'v_wind_earth_relative_m_s01'
U_WIND_GRID_RELATIVE_NAME = 'u_wind_grid_relative_m_s01'
V_WIND_GRID_RELATIVE_NAME = 'v_wind_grid_relative_m_s01'
WET_BULB_THETA_NAME = 'wet_bulb_potential_temperature_kelvins'

STANDARD_FIELD_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, PRESSURE_NAME, DEWPOINT_NAME,
    SPECIFIC_HUMIDITY_NAME, U_WIND_EARTH_RELATIVE_NAME,
    V_WIND_EARTH_RELATIVE_NAME
]

DERIVED_FIELD_NAMES = [
    WET_BULB_THETA_NAME, U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME
]

FIELD_NAMES = STANDARD_FIELD_NAMES + DERIVED_FIELD_NAMES

DATA_MATRIX_KEY = 'data_matrix'
LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
VALID_TIMES_KEY = 'valid_times_unix_sec'
PRESSURE_LEVELS_KEY = 'pressure_levels_mb'
FIELD_NAMES_KEY = 'field_names'


def check_predictor_dict(predictor_dict):
    """Error-checks dictionary with predictor fields.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_dict: Dictionary with the following keys.
    predictor_dict['data_matrix']: T-by-M-by-N-by-C numpy array of data values.
    predictor_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    predictor_dict['latitudes_deg']: length-M numpy array of grid-point
        latitudes (deg N).  If data are on the NARR grid, this should be None.
    predictor_dict['longitudes_deg']: length-N numpy array of grid-point
        longitudes (deg E).  If data are on the NARR grid, this should be None.
    predictor_dict['pressure_levels_mb']: length-C numpy array of pressure
        levels (use 1013 mb to denote surface).
    predictor_dict['field_names']: length-C list of field names.
    """

    error_checking.assert_is_integer_numpy_array(
        predictor_dict[VALID_TIMES_KEY])
    error_checking.assert_is_numpy_array(
        predictor_dict[VALID_TIMES_KEY], num_dimensions=1)

    error_checking.assert_is_integer_numpy_array(
        predictor_dict[PRESSURE_LEVELS_KEY])
    error_checking.assert_is_greater_numpy_array(
        predictor_dict[PRESSURE_LEVELS_KEY], 0)
    error_checking.assert_is_numpy_array(
        predictor_dict[PRESSURE_LEVELS_KEY], num_dimensions=1)

    num_predictors = len(predictor_dict[PRESSURE_LEVELS_KEY])

    error_checking.assert_is_numpy_array(
        numpy.array(predictor_dict[FIELD_NAMES_KEY]),
        exact_dimensions=numpy.array([num_predictors], dtype=int)
    )

    for j in range(num_predictors):
        check_field_name(predictor_dict[FIELD_NAMES_KEY][j])

    on_era5_grid = not (
        predictor_dict[LATITUDES_KEY] is None or
        predictor_dict[LONGITUDES_KEY] is None
    )

    if on_era5_grid:
        error_checking.assert_is_valid_lat_numpy_array(
            predictor_dict[LATITUDES_KEY])
        error_checking.assert_is_numpy_array(
            predictor_dict[LATITUDES_KEY], num_dimensions=1)

        error_checking.assert_is_valid_lng_numpy_array(
            predictor_dict[LONGITUDES_KEY], positive_in_west_flag=True)
        error_checking.assert_is_numpy_array(
            predictor_dict[LONGITUDES_KEY], num_dimensions=1)

    this_num_dimensions = len(predictor_dict[DATA_MATRIX_KEY].shape)
    error_checking.assert_is_geq(this_num_dimensions, 4)
    error_checking.assert_is_leq(this_num_dimensions, 4)

    num_times = len(predictor_dict[VALID_TIMES_KEY])

    if on_era5_grid:
        num_grid_rows = predictor_dict[DATA_MATRIX_KEY].shape[1]
        num_grid_columns = predictor_dict[DATA_MATRIX_KEY].shape[2]
    else:
        num_grid_rows = len(predictor_dict[LATITUDES_KEY])
        num_grid_columns = len(predictor_dict[LONGITUDES_KEY])

    these_expected_dim = numpy.array(
        [num_times, num_grid_rows, num_grid_columns, num_predictors], dtype=int
    )

    error_checking.assert_is_numpy_array(
        predictor_dict[DATA_MATRIX_KEY], exact_dimensions=these_expected_dim)


def check_field_name(field_name, require_standard=False):
    """Error-checks field name.

    :param field_name: Field name.
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
