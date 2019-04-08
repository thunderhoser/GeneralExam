"""Computes theta_w grid for each time step.

theta_w = wet-bulb potential temperature
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils
from generalexam.ge_utils import conversions as ge_conversions

# TODO(thunderhoser): This script is a bit sloppy, since it does not allow input
# files with multiple pressure levels.

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800
MB_TO_PASCALS = 100

PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files (containing '
    'temperature, specific humidity, and pressure if `{0:s}` = {1:d}) therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.'
).format(PRESSURE_LEVEL_ARG_NAME, predictor_utils.DUMMY_SURFACE_PRESSURE_MB)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will compute theta_w grids for '
    'all time steps in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Will compute theta_w only at this pressure level (millibars).  If you want'
    ' surface theta_w, leave this argument alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for theta_w grids.  Files be written here by '
    '`predictor_io.write_file`, to exact locations determined by '
    '`predictor_io.find_file`.  To make output directory = input directory, '
    'leave this argument alone.')

TOP_PREDICTOR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_PREDICTOR_DIR_NAME_DEFAULT, help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=predictor_utils.DUMMY_SURFACE_PRESSURE_MB,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_DIR_HELP_STRING)


def _read_inputs_one_time(
        top_input_dir_name, valid_time_unix_sec, pressure_level_mb):
    """Reads input fields for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param top_input_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: See documentation at top of file.
    :return: predictor_dict: Dictionary created by `predictor_io.read_file`.
    :return: temperature_matrix_kelvins: 1-by-M-by-N numpy array of
        temperatures.
    :return: humidity_matrix_kg_kg01: 1-by-M-by-N numpy array of
        specific humidities (kg/kg).
    :return: pressure_matrix_pascals: 1-by-M-by-N numpy array of pressures.
    :raises: ValueError: if the file contains multiple time steps or pressure
        levels.
    :raises: ValueError: if the file contains the wrong pressure level.
    """

    input_file_name = predictor_io.find_file(
        top_directory_name=top_input_dir_name,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    predictor_dict = predictor_io.read_file(netcdf_file_name=input_file_name)

    num_times_in_file = len(predictor_dict[predictor_utils.VALID_TIMES_KEY])
    if num_times_in_file > 1:
        error_string = (
            'File ("{0:s}") should contain 1 time step, not {1:d}.'
        ).format(input_file_name, num_times_in_file)

        raise ValueError(error_string)

    unique_pressures_in_file_mb = numpy.unique(
        predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY]
    )

    if len(unique_pressures_in_file_mb) > 1:
        error_string = (
            'File ("{0:s}") should contain 1 unique pressure level, not {1:d}.'
        ).format(input_file_name, len(unique_pressures_in_file_mb))

        raise ValueError(error_string)

    if unique_pressures_in_file_mb[0] != pressure_level_mb:
        error_string = (
            'Pressure level in file ({0:d} mb) does not match desired pressure '
            'level ({1:d} mb).'
        ).format(unique_pressures_in_file_mb[0], pressure_level_mb)

        raise ValueError(error_string)

    temperature_index = predictor_dict[predictor_utils.FIELD_NAMES_KEY].index(
        predictor_utils.TEMPERATURE_NAME)
    temperature_matrix_kelvins = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY][..., temperature_index]

    humidity_index = predictor_dict[predictor_utils.FIELD_NAMES_KEY].index(
        predictor_utils.SPECIFIC_HUMIDITY_NAME)
    humidity_matrix_kg_kg01 = predictor_dict[predictor_utils.DATA_MATRIX_KEY][
        ..., humidity_index]

    if pressure_level_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
        pressure_index = predictor_dict[predictor_utils.FIELD_NAMES_KEY].index(
            predictor_utils.PRESSURE_NAME)
        pressure_matrix_pascals = predictor_dict[
            predictor_utils.DATA_MATRIX_KEY][..., pressure_index]
    else:
        pressure_matrix_pascals = numpy.full(
            humidity_matrix_kg_kg01.shape, pressure_level_mb * MB_TO_PASCALS)

    return (predictor_dict, temperature_matrix_kelvins, humidity_matrix_kg_kg01,
            pressure_matrix_pascals)


def _write_output_one_time(
        theta_w_matrix_kelvins, predictor_dict, top_output_dir_name):
    """Writes theta_w field for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param theta_w_matrix_kelvins: 1-by-M-by-N numpy array of wet-bulb potential
        temperatures.
    :param predictor_dict: Dictionary created by `predictor_io.read_file`.
    :param top_output_dir_name: See documentation at top of file.
    """

    theta_w_matrix_kelvins = numpy.expand_dims(theta_w_matrix_kelvins, axis=-1)

    predictor_dict[predictor_utils.FIELD_NAMES_KEY].append(
        predictor_utils.WET_BULB_THETA_NAME
    )
    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.concatenate(
        (predictor_dict[predictor_utils.DATA_MATRIX_KEY],
         theta_w_matrix_kelvins),
        axis=-1
    )

    output_file_name = predictor_io.find_file(
        top_directory_name=top_output_dir_name,
        valid_time_unix_sec=predictor_dict[predictor_utils.VALID_TIMES_KEY][0],
        raise_error_if_missing=False)

    print 'Writing theta_w field to: "{0:s}"...'.format(output_file_name)
    predictor_io.write_file(
        netcdf_file_name=output_file_name, predictor_dict=predictor_dict)


def _run(top_predictor_dir_name, first_time_string, last_time_string,
         pressure_level_mb, top_output_dir_name):
    """Computes theta_w grid for each time step.

    This is effectively the main method.

    :param top_predictor_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param top_output_dir_name: Same.
    """

    if top_output_dir_name in ['', 'None']:
        top_output_dir_name = top_predictor_dir_name

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        (this_predictor_dict, this_temperature_matrix_kelvins,
         this_humidity_matrix_kg_kg01, this_pressure_matrix_pascals
        ) = _read_inputs_one_time(
            top_input_dir_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            pressure_level_mb=pressure_level_mb)

        this_dewpoint_matrix_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=this_humidity_matrix_kg_kg01,
                total_pressures_pascals=this_pressure_matrix_pascals)
        )

        this_wb_temp_matrix_kelvins = (
            ge_conversions.dewpoint_to_wet_bulb_temperature(
                dewpoints_kelvins=this_dewpoint_matrix_kelvins,
                temperatures_kelvins=this_temperature_matrix_kelvins,
                total_pressures_pascals=this_pressure_matrix_pascals)
        )

        this_theta_w_matrix_kelvins = (
            temperature_conversions.temperatures_to_potential_temperatures(
                temperatures_kelvins=this_wb_temp_matrix_kelvins,
                total_pressures_pascals=this_pressure_matrix_pascals)
        )

        _write_output_one_time(
            theta_w_matrix_kelvins=this_theta_w_matrix_kelvins,
            predictor_dict=this_predictor_dict,
            top_output_dir_name=top_output_dir_name)

        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
