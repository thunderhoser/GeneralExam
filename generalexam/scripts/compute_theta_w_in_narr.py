"""Computes theta_w (wet-bulb potential temperature) in NARR data.

For each time step, this script reads temperature and specific-humidity files,
compute theta_w, and writes a theta_w file.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import conversions as ge_conversions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800
LAST_GRIB_TIME_UNIX_SEC = 1412283600

MB_TO_PASCALS = 100

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
PROCESSED_DIR_ARG_NAME = 'processed_narr_dir_name'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  theta_w will be computed at each time step '
    'from `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'theta_w will be computed at this pressure level (millibars) at each time '
    'step.')

PROCESSED_DIR_HELP_STRING = (
    'Name of directory with processed NARR data.  For each time, temperature '
    'and specific humidity at `{0:s}` will be read from `{1:s}`, then theta_w '
    'at `{0:s}` will be computed and written to `{1:s}`.'
).format(PRESSURE_LEVEL_ARG_NAME, PROCESSED_DIR_ARG_NAME)

DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data'
DEFAULT_PROCESSED_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PROCESSED_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_PROCESSED_DIR_NAME, help=PROCESSED_DIR_HELP_STRING)


def _run(first_time_string, last_time_string, pressure_level_mb,
         top_processed_dir_name):
    """Computes theta_w (wet-bulb potential temperature) in NARR data.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param top_processed_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    num_times = len(valid_times_unix_sec)
    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    pressure_level_pascals = int(numpy.round(pressure_level_mb * MB_TO_PASCALS))
    pressure_matrix_pascals = numpy.full(
        (1, num_grid_rows, num_grid_columns), pressure_level_pascals,
        dtype=float)

    for i in range(num_times):
        this_temperature_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_processed_dir_name,
            field_name=processed_narr_io.TEMPERATURE_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(this_temperature_file_name)
        this_temperature_matrix_kelvins = (
            processed_narr_io.read_fields_from_file(
                this_temperature_file_name)[0]
        )

        this_spfh_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_processed_dir_name,
            field_name=processed_narr_io.SPECIFIC_HUMIDITY_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(this_spfh_file_name)
        this_specific_humidity_matrix_kg_kg01 = (
            processed_narr_io.read_fields_from_file(this_spfh_file_name)[0]
        )

        this_dewpoint_matrix_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=
                this_specific_humidity_matrix_kg_kg01,
                total_pressures_pascals=pressure_matrix_pascals)
        )
        this_wet_bulb_temp_matrix_kelvins = (
            ge_conversions.dewpoint_to_wet_bulb_temperature(
                dewpoints_kelvins=this_dewpoint_matrix_kelvins,
                temperatures_kelvins=this_temperature_matrix_kelvins,
                total_pressures_pascals=pressure_matrix_pascals)
        )
        this_theta_w_matrix_kelvins = (
            temperature_conversions.temperatures_to_potential_temperatures(
                temperatures_kelvins=this_wet_bulb_temp_matrix_kelvins,
                total_pressures_pascals=pressure_matrix_pascals)
        )

        this_theta_w_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_processed_dir_name,
            field_name=processed_narr_io.WET_BULB_THETA_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing data to: "{0:s}"...\n'.format(this_theta_w_file_name)
        processed_narr_io.write_fields_to_file(
            pickle_file_name=this_theta_w_file_name,
            field_matrix=this_theta_w_matrix_kelvins,
            field_name=processed_narr_io.WET_BULB_THETA_NAME,
            pressure_level_pascals=pressure_level_pascals,
            valid_times_unix_sec=valid_times_unix_sec[[i]])


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_processed_dir_name=getattr(INPUT_ARG_OBJECT, PROCESSED_DIR_ARG_NAME)
    )
