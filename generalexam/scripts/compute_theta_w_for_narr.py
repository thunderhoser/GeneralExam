"""Computes theta_w (wet-bulb potential temperature) for NARR data."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import conversions as ge_conversions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

MB_TO_PASCALS = 100
DUMMY_PRESSURE_LEVEL_MB = 1013

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input data (temperature and specific-'
    'humidity fields, also pressure fields if surface theta_w is to be '
    'computed).  Files therein will be found by '
    '`processed_narr_io.find_file_for_one_time` and read by '
    '`processed_narr_io.read_fields_from_file`.')

TIME_HELP_STRING = (
    'Valid time (format "yyyymmddHH").  Will compute theta_w field for all '
    'valid times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Pressure level (millibars).  Will compute theta_w only at this pressure '
    'level.  If you want the surface field, leave this argument alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for theta_w fields.  Output will be '
    'written here by `processed_narr_io.write_fields_to_file`, to file '
    'locations determined by `processed_narr_io.find_file_for_one_time`.')

TOP_INPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_OUTPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_INPUT_DIR_NAME_DEFAULT, help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False, default=-1,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_OUTPUT_DIR_NAME_DEFAULT, help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, first_time_string, last_time_string,
         pressure_level_mb, top_output_dir_name):
    """Computes theta_w (wet-bulb potential temperature) for NARR data.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param top_output_dir_name: Same.
    """

    if pressure_level_mb <= 0:
        pressure_level_mb = None

    if pressure_level_mb is None:
        pressure_in_file_name_mb = DUMMY_PRESSURE_LEVEL_MB + 0
    else:
        pressure_in_file_name_mb = pressure_level_mb + 0

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    num_times = len(valid_times_unix_sec)
    this_pressure_matrix_pascals = None

    for i in range(num_times):
        this_temperature_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            field_name=processed_narr_io.TEMPERATURE_NAME,
            pressure_level_mb=pressure_in_file_name_mb,
            valid_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_temperature_file_name)
        this_temperature_matrix_kelvins = (
            processed_narr_io.read_fields_from_file(
                this_temperature_file_name)[0]
        )

        this_humidity_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            field_name=processed_narr_io.SPECIFIC_HUMIDITY_NAME,
            pressure_level_mb=pressure_in_file_name_mb,
            valid_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_humidity_file_name)
        this_humidity_matrix_kg_kg01 = (
            processed_narr_io.read_fields_from_file(this_humidity_file_name)[0]
        )

        if pressure_level_mb is None:
            this_pressure_file_name = processed_narr_io.find_file_for_one_time(
                top_directory_name=top_input_dir_name,
                field_name=processed_narr_io.HEIGHT_NAME,
                pressure_level_mb=pressure_in_file_name_mb,
                valid_time_unix_sec=valid_times_unix_sec[i])

            print 'Reading data from: "{0:s}"...'.format(
                this_pressure_file_name)

            this_pressure_matrix_pascals = (
                processed_narr_io.read_fields_from_file(
                    this_pressure_file_name)[0]
            )

            print this_pressure_matrix_pascals[:5, :5]
        else:
            if this_pressure_matrix_pascals is None:
                this_pressure_matrix_pascals = numpy.full(
                    this_humidity_matrix_kg_kg01.shape,
                    pressure_level_mb * MB_TO_PASCALS)

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

        this_theta_w_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_output_dir_name,
            field_name=processed_narr_io.WET_BULB_THETA_NAME,
            pressure_level_mb=pressure_in_file_name_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing theta_w field to: "{0:s}"...\n'.format(
            this_theta_w_file_name)

        processed_narr_io.write_fields_to_file(
            pickle_file_name=this_theta_w_file_name,
            field_matrix=this_theta_w_matrix_kelvins,
            field_name=processed_narr_io.WET_BULB_THETA_NAME,
            pressure_level_pascals=pressure_in_file_name_mb * MB_TO_PASCALS,
            valid_times_unix_sec=valid_times_unix_sec[[i]]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
