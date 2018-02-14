"""IO methods for processed data from NARR (North American Rgnl Reanalysis)."""

import pickle
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
VERTICAL_VELOCITY_NAME = 'w_wind_pascals_s01'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
U_WIND_NAME = 'u_wind_m_s01'
V_WIND_NAME = 'v_wind_m_s01'

STANDARD_FIELD_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, VERTICAL_VELOCITY_NAME,
    SPECIFIC_HUMIDITY_NAME, U_WIND_NAME, V_WIND_NAME]

WET_BULB_THETA_NAME = 'wet_bulb_potential_temperature_kelvins'
DERIVED_FIELD_NAMES = [WET_BULB_THETA_NAME]
FIELD_NAMES = STANDARD_FIELD_NAMES + DERIVED_FIELD_NAMES


def _check_model_fields(
        field_matrix, field_name, pressure_level_pascals, valid_times_unix_sec):
    """Checks model fields for errors.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    T = number of time steps

    :param field_matrix: T-by-M-by-N numpy array with values of a single field
        (atmospheric variable).
    :param field_name: Field name in GewitterGefahr format.
    :param pressure_level_pascals: Pressure level (integer Pascals).
    :param valid_times_unix_sec: length-T numpy array of valid times.
    """

    check_field_name(field_name, require_standard=False)
    error_checking.assert_is_integer(pressure_level_pascals)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(valid_times_unix_sec, num_dimensions=1)
    num_times = len(valid_times_unix_sec)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        field_matrix, exact_dimensions=numpy.array(
            [num_times, num_grid_rows, num_grid_columns]))


def check_field_name(field_name, require_standard=False):
    """Ensures that name of model field is recognized.

    :param field_name: Field name in GewitterGefahr format (not the original
        NetCDF format).
    :param require_standard: Boolean flag.  If True, `field_name` must be in
        `STANDARD_FIELD_NAMES`.  If False, `field_name` must be in
        `FIELD_NAMES`.
    :raises: ValueError: if field name is unrecognized.
    """

    error_checking.assert_is_string(field_name)
    error_checking.assert_is_boolean(require_standard)

    if require_standard:
        valid_field_names = STANDARD_FIELD_NAMES
    else:
        valid_field_names = FIELD_NAMES

    if field_name not in valid_field_names:
        error_string = (
            '\n\n' + str(valid_field_names) +
            '\n\nValid field names (listed above) do not include "' +
            field_name + '".')
        raise ValueError(error_string)


def write_fields_to_file(
        pickle_file_name, field_matrix, field_name, pressure_level_pascals,
        valid_times_unix_sec):
    """Writes fields (at one or more time steps) to Pickle file.

    :param pickle_file_name: Path to output file.
    :param field_matrix: See documentation for `_check_model_fields`.
    :param field_name: See documentation for `_check_model_fields`.
    :param pressure_level_pascals: See documentation for `_check_model_fields`.
    :param valid_times_unix_sec: See documentation for `_check_model_fields`.
    """

    _check_model_fields(
        field_matrix=field_matrix, field_name=field_name,
        pressure_level_pascals=pressure_level_pascals,
        valid_times_unix_sec=valid_times_unix_sec)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(field_matrix, pickle_file_handle)
    pickle.dump(field_name, pickle_file_handle)
    pickle.dump(pressure_level_pascals, pickle_file_handle)
    pickle.dump(valid_times_unix_sec, pickle_file_handle)
    pickle_file_handle.close()


def read_fields_from_file(pickle_file_name):
    """Reads fields (at one or more time steps) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: field_matrix: See documentation for `_check_model_fields`.
    :return: field_name: See documentation for `_check_model_fields`.
    :return: pressure_level_pascals: See doc for `_check_model_fields`.
    :return: valid_times_unix_sec: See documentation for `_check_model_fields`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    field_matrix = pickle.load(pickle_file_handle)
    field_name = pickle.load(pickle_file_handle)
    pressure_level_pascals = pickle.load(pickle_file_handle)
    valid_times_unix_sec = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    _check_model_fields(
        field_matrix=field_matrix, field_name=field_name,
        pressure_level_pascals=pressure_level_pascals,
        valid_times_unix_sec=valid_times_unix_sec)

    return (field_matrix, field_name, pressure_level_pascals,
            valid_times_unix_sec)
