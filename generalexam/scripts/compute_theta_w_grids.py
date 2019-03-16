"""Computes theta_w grid for each time step.

theta_w = wet-bulb potential temperature
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from generalexam.ge_io import era5_io
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import conversions as ge_conversions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

MB_TO_PASCALS = 100
DUMMY_PRESSURE_LEVEL_MB = 1013

NARR_DIR_ARG_NAME = 'input_narr_dir_name'
ERA5_DIR_ARG_NAME = 'input_era5_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data.  Input files (containing '
    'temperature, specific humidity, and pressure if `{0:s}` = {1:d}) therein '
    'will be found by `processed_narr_io.find_file_for_one_time` and read by '
    '`processed_narr_io.read_fields_from_file`.  To use ERA5 data instead, '
    'leave this argument alone.'
).format(PRESSURE_LEVEL_ARG_NAME, DUMMY_PRESSURE_LEVEL_MB)

ERA5_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data.  Input files (containing '
    'temperature, specific humidity, and pressure if `{0:s}` = {1:d}) therein '
    'will be found by `era5_io.find_processed_file` and read by '
    '`era5_io.read_processed_file`.  To use NARR data instead, leave this '
    'argument alone.'
).format(PRESSURE_LEVEL_ARG_NAME, DUMMY_PRESSURE_LEVEL_MB)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will compute theta_w grids for '
    'all time steps in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Will compute theta_w only at this pressure level (millibars).  If you want'
    ' surface theta_w, leave this argument alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory for theta_w grids.  If NARR (ERA5) '
    'data, files will be written by `processed_narr_io.write_fields_to_file` '
    '(`era5_io.write_processed_file`), to locations in this directory '
    'determined by `processed_narr_io.find_file_for_one_time` '
    '(`era5_io.find_processed_file`).  To make output directory = input '
    'directory, leave this argument alone.')

TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_ERA5_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ERA5_DIR_ARG_NAME, type=str, required=False,
    default=TOP_ERA5_DIR_NAME_DEFAULT, help=ERA5_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DUMMY_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_DIR_HELP_STRING)


def _read_era5_inputs_one_time(
        top_input_dir_name, valid_time_unix_sec, pressure_level_mb):
    """Reads ERA5 input fields for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param top_input_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: See documentation at top of file.
    :return: era5_dict: Dictionary created by `era5_io.read_processed_file`.
    :return: temperature_matrix_kelvins: 1-by-M-by-N numpy array of
        temperatures.
    :return: humidity_matrix_kg_kg01: 1-by-M-by-N numpy array of
        specific humidities (kg/kg).
    :return: pressure_matrix_pascals: 1-by-M-by-N numpy array of pressures.
    :raises: ValueError: if the file contains multiple pressure levels or time
        steps.
    """

    input_file_name = era5_io.find_processed_file(
        top_directory_name=top_input_dir_name,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    era5_dict = era5_io.read_processed_file(
        netcdf_file_name=input_file_name,
        pressure_levels_to_keep_mb=numpy.array([pressure_level_mb])
    )

    num_pressures_in_file = len(era5_dict[era5_io.PRESSURE_LEVELS_KEY])
    if num_pressures_in_file > 1:
        error_string = (
            'File ("{0:s}") should contain 1 pressure level, not {1:d}.'
        ).format(input_file_name, num_pressures_in_file)

        raise ValueError(error_string)

    num_times_in_file = len(era5_dict[era5_io.VALID_TIMES_KEY])
    if num_times_in_file > 1:
        error_string = (
            'File ("{0:s}") should contain 1 time step, not {1:d}.'
        ).format(input_file_name, num_times_in_file)

        raise ValueError(error_string)

    temperature_index = era5_dict[era5_io.FIELD_NAMES_KEY].index(
        era5_io.TEMPERATURE_NAME)
    temperature_matrix_kelvins = era5_dict[era5_io.DATA_MATRIX_KEY][
        ..., 0, temperature_index]

    humidity_index = era5_dict[era5_io.FIELD_NAMES_KEY].index(
        era5_io.SPECIFIC_HUMIDITY_NAME)
    humidity_matrix_kg_kg01 = era5_dict[era5_io.DATA_MATRIX_KEY][
        ..., 0, humidity_index]

    if pressure_level_mb == DUMMY_PRESSURE_LEVEL_MB:
        pressure_index = era5_dict[era5_io.FIELD_NAMES_KEY].index(
            era5_io.PRESSURE_NAME)
        pressure_matrix_pascals = era5_dict[era5_io.DATA_MATRIX_KEY][
            ..., 0, pressure_index]
    else:
        pressure_matrix_pascals = numpy.full(
            humidity_matrix_kg_kg01.shape, pressure_level_mb * MB_TO_PASCALS)

    return (era5_dict, temperature_matrix_kelvins, humidity_matrix_kg_kg01,
            pressure_matrix_pascals)


def _read_narr_inputs_one_time(
        top_input_dir_name, valid_time_unix_sec, pressure_level_mb):
    """Reads NARR input fields for one time.

    :param top_input_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: See documentation at top of file.
    :return: temperature_matrix_kelvins: See doc for
        `_read_era5_inputs_one_time`.
    :return: humidity_matrix_kg_kg01: Same.
    :return: pressure_matrix_pascals: Same.
    """

    temperature_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_input_dir_name,
        field_name=processed_narr_io.TEMPERATURE_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(temperature_file_name)
    temperature_matrix_kelvins = processed_narr_io.read_fields_from_file(
        temperature_file_name
    )[0]

    humidity_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_input_dir_name,
        field_name=processed_narr_io.SPECIFIC_HUMIDITY_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(humidity_file_name)
    humidity_matrix_kg_kg01 = processed_narr_io.read_fields_from_file(
        humidity_file_name
    )[0]

    if pressure_level_mb == DUMMY_PRESSURE_LEVEL_MB:
        pressure_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            field_name=processed_narr_io.HEIGHT_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_time_unix_sec)

        print 'Reading data from: "{0:s}"...'.format(pressure_file_name)
        pressure_matrix_pascals = processed_narr_io.read_fields_from_file(
            pressure_file_name
        )[0]
    else:
        pressure_matrix_pascals = numpy.full(
            humidity_matrix_kg_kg01.shape, pressure_level_mb * MB_TO_PASCALS)

    return (temperature_matrix_kelvins, humidity_matrix_kg_kg01,
            pressure_matrix_pascals)


def _write_era5_output_one_time(
        theta_w_matrix_kelvins, era5_dict, top_output_dir_name):
    """Writes theta_w field on ERA5 grid for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param theta_w_matrix_kelvins: 1-by-M-by-N numpy array of wet-bulb potential
        temperatures.
    :param era5_dict: Dictionary created by `era5_io.read_processed_file`.
    :param top_output_dir_name: See documentation at top of file.
    """

    theta_w_matrix_kelvins = numpy.expand_dims(theta_w_matrix_kelvins, axis=-1)
    theta_w_matrix_kelvins = numpy.expand_dims(theta_w_matrix_kelvins, axis=-1)

    era5_dict[era5_io.FIELD_NAMES_KEY].append(era5_io.WET_BULB_THETA_NAME)
    era5_dict[era5_io.DATA_MATRIX_KEY] = numpy.concatenate(
        (era5_dict[era5_io.DATA_MATRIX_KEY], theta_w_matrix_kelvins), axis=-1
    )

    output_file_name = era5_io.find_processed_file(
        top_directory_name=top_output_dir_name,
        valid_time_unix_sec=era5_dict[era5_io.VALID_TIMES_KEY][0],
        raise_error_if_missing=False)

    print 'Writing theta_w field to: "{0:s}"...'.format(output_file_name)
    era5_io.write_processed_file(
        netcdf_file_name=output_file_name, era5_dict=era5_dict)


def _write_narr_output_one_time(theta_w_matrix_kelvins, top_output_dir_name,
                                pressure_level_mb, valid_time_unix_sec):
    """Writes theta_w field on NARR grid for one time.

    :param theta_w_matrix_kelvins: See doc for `_write_era5_output_one_time`.
    :param top_output_dir_name: See documentation at top of file.
    :param pressure_level_mb: Same.
    :param valid_time_unix_sec: Valid time.
    """

    output_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_output_dir_name,
        field_name=processed_narr_io.WET_BULB_THETA_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec, raise_error_if_missing=False)

    print 'Writing theta_w field to: "{0:s}"...'.format(output_file_name)

    processed_narr_io.write_fields_to_file(
        pickle_file_name=output_file_name,
        field_matrix=theta_w_matrix_kelvins,
        field_name=processed_narr_io.WET_BULB_THETA_NAME,
        pressure_level_pascals=pressure_level_mb * MB_TO_PASCALS,
        valid_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )


def _run(top_narr_input_dir_name, top_era5_input_dir_name, first_time_string,
         last_time_string, pressure_level_mb, top_output_dir_name):
    """Computes theta_w grid for each time step.

    This is effectively the main method.

    :param top_narr_input_dir_name: See documentation at top of file.
    :param top_era5_input_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param top_output_dir_name: Same.
    """

    if top_narr_input_dir_name in ['', 'None']:
        top_narr_input_dir_name = None
    if top_era5_input_dir_name in ['', 'None']:
        top_era5_input_dir_name = None
    if top_output_dir_name in ['', 'None']:
        top_output_dir_name = None

    if top_output_dir_name is None:
        top_output_dir_name = (
            top_era5_input_dir_name if top_narr_input_dir_name is None
            else top_narr_input_dir_name
        )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        this_era5_dict = None

        if top_narr_input_dir_name is None:
            (this_era5_dict, this_temperature_matrix_kelvins,
             this_humidity_matrix_kg_kg01, this_pressure_matrix_pascals
            ) = _read_era5_inputs_one_time(
                top_input_dir_name=top_era5_input_dir_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_level_mb)

        else:
            (this_temperature_matrix_kelvins, this_humidity_matrix_kg_kg01,
             this_pressure_matrix_pascals
            ) = _read_narr_inputs_one_time(
                top_input_dir_name=top_narr_input_dir_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_level_mb)

        print 'Temperatures (K):\n{0:s}\n'.format(str(this_temperature_matrix_kelvins[0, 100:110, 110:110]))
        print 'Specific humidities (kg/kg):\n{0:s}\n'.format(str(this_humidity_matrix_kg_kg01[0, 100:110, 110:110]))
        print 'Pressures (Pa):\n{0:s}\n'.format(str(this_pressure_matrix_pascals[0, 100:110, 110:110]))

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

        if top_narr_input_dir_name is None:
            _write_era5_output_one_time(
                theta_w_matrix_kelvins=this_theta_w_matrix_kelvins,
                era5_dict=this_era5_dict,
                top_output_dir_name=top_output_dir_name)

        else:
            _write_narr_output_one_time(
                theta_w_matrix_kelvins=this_theta_w_matrix_kelvins,
                top_output_dir_name=top_output_dir_name,
                pressure_level_mb=pressure_level_mb,
                valid_time_unix_sec=valid_times_unix_sec[i])

        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_narr_input_dir_name=getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME),
        top_era5_input_dir_name=getattr(INPUT_ARG_OBJECT, ERA5_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
