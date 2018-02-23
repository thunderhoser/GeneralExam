"""IO methods for processed data from NARR (North American Rgnl Reanalysis)."""

import os.path
import pickle
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
VERTICAL_VELOCITY_NAME = 'w_wind_pascals_s01'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
U_WIND_EARTH_RELATIVE_NAME = 'u_wind_earth_relative_m_s01'
V_WIND_EARTH_RELATIVE_NAME = 'v_wind_earth_relative_m_s01'

STANDARD_FIELD_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, VERTICAL_VELOCITY_NAME,
    SPECIFIC_HUMIDITY_NAME, U_WIND_EARTH_RELATIVE_NAME,
    V_WIND_EARTH_RELATIVE_NAME]

WET_BULB_TEMP_NAME = 'wet_bulb_temperature_kelvins'
WET_BULB_THETA_NAME = 'wet_bulb_potential_temperature_kelvins'
U_WIND_GRID_RELATIVE_NAME = 'u_wind_grid_relative_m_s01'
V_WIND_GRID_RELATIVE_NAME = 'v_wind_grid_relative_m_s01'

DERIVED_FIELD_NAMES = [
    WET_BULB_TEMP_NAME, WET_BULB_THETA_NAME, U_WIND_GRID_RELATIVE_NAME,
    V_WIND_GRID_RELATIVE_NAME]
FIELD_NAMES = STANDARD_FIELD_NAMES + DERIVED_FIELD_NAMES

UNIT_STRINGS = ['kelvins', 'm', 'asl', 'pascals', 's01', 'kg', 'kg01']

TEMPERATURE_NAME_UNITLESS = 'temperature'
HEIGHT_NAME_UNITLESS = 'height'
VERTICAL_VELOCITY_NAME_UNITLESS = 'w_wind'
SPECIFIC_HUMIDITY_NAME_UNITLESS = 'specific_humidity'
U_WIND_EARTH_RELATIVE_NAME_UNITLESS = 'u_wind_earth_relative'
V_WIND_EARTH_RELATIVE_NAME_UNITLESS = 'v_wind_earth_relative'

STANDARD_FIELD_NAMES_UNITLESS = [
    TEMPERATURE_NAME_UNITLESS, HEIGHT_NAME_UNITLESS,
    VERTICAL_VELOCITY_NAME_UNITLESS, SPECIFIC_HUMIDITY_NAME_UNITLESS,
    U_WIND_EARTH_RELATIVE_NAME_UNITLESS, V_WIND_EARTH_RELATIVE_NAME_UNITLESS]

WET_BULB_TEMP_NAME_UNITLESS = 'wet_bulb_temperature'
WET_BULB_THETA_NAME_UNITLESS = 'wet_bulb_potential_temperature'
U_WIND_GRID_RELATIVE_NAME_UNITLESS = 'u_wind_grid_relative'
V_WIND_GRID_RELATIVE_NAME_UNITLESS = 'v_wind_grid_relative'

DERIVED_FIELD_NAMES_UNITLESS = [
    WET_BULB_TEMP_NAME_UNITLESS, WET_BULB_THETA_NAME_UNITLESS,
    U_WIND_GRID_RELATIVE_NAME_UNITLESS, V_WIND_GRID_RELATIVE_NAME_UNITLESS]
FIELD_NAMES_UNITLESS = (
    STANDARD_FIELD_NAMES_UNITLESS + DERIVED_FIELD_NAMES_UNITLESS)


def _remove_units_from_field_name(field_name):
    """Removes units from field name.

    :param field_name: Field name in GewitterGefahr format.
    :return: field_name_unitless: Field name in GewitterGefahr format, but
        without units.
    """

    field_name_parts = field_name.split('_')
    field_name_parts = [s for s in field_name_parts if s not in UNIT_STRINGS]
    return '_'.join(field_name_parts)


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


def find_file_for_time_period(
        directory_name, field_name, pressure_level_mb, start_time_unix_sec,
        end_time_unix_sec, raise_error_if_missing=True):
    """Finds file with NARR data for a contiguous time period.

    Specifically, this file should contain grids for one variable, at one
    pressure level, at all 3-hour time steps in the given period.

    :param directory_name: Name of directory.
    :param field_name: Field name in GewitterGefahr format.
    :param pressure_level_mb: Pressure level (integer in millibars).
    :param start_time_unix_sec: Start of contiguous time period.
    :param end_time_unix_sec: End of contiguous time period.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: narr_file_name: Path to file.
    """

    error_checking.assert_is_string(directory_name)
    check_field_name(field_name, require_standard=False)
    error_checking.assert_is_integer(pressure_level_mb)
    error_checking.assert_is_greater(pressure_level_mb, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    narr_file_name = '{0:s}/{1:s}_{2:04d}mb_{3:s}-{4:s}.p'.format(
        directory_name, _remove_units_from_field_name(field_name),
        pressure_level_mb,
        time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES))

    if raise_error_if_missing and not os.path.isfile(narr_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                narr_file_name))
        raise ValueError(error_string)

    return narr_file_name


def find_file_for_one_time(
        top_directory_name, field_name, pressure_level_mb, valid_time_unix_sec,
        raise_error_if_missing=True):
    """Finds file with NARR data for a single time step.

    Specifically, this file should contain the grid for one variable, at one
    pressure level, at one time step.

    :param top_directory_name: Name of top-level directory with processed NARR
        files.
    :param field_name: Field name in GewitterGefahr format.
    :param pressure_level_mb: Pressure level (integer in millibars).
    :param valid_time_unix_sec: Valid time (= initialization time for NARR).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: narr_file_name: Path to file.
    """

    error_checking.assert_is_string(top_directory_name)
    check_field_name(field_name, require_standard=False)
    error_checking.assert_is_integer(pressure_level_mb)
    error_checking.assert_is_greater(pressure_level_mb, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    narr_file_name = '{0:s}/{1:s}/{2:s}_{3:04d}mb_{4:s}.p'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_MONTH),
        _remove_units_from_field_name(field_name),
        pressure_level_mb,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES))

    if raise_error_if_missing and not os.path.isfile(narr_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                narr_file_name))
        raise ValueError(error_string)

    return narr_file_name
