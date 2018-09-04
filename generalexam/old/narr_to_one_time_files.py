"""Converts NARR data to one-time files.

The input format is one file for each variable, pressure level, and contiguous
time period (usually one month).

The output format is one file for each variable, pressure level, and single time
step.  This will make it easier to create machine-learning examples on the fly
(less data to hold in memory at once).
"""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import processed_narr_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT_MONTH = '%Y%m'

MB_TO_PASCALS = 100
NARR_TIME_INTERVAL_SECONDS = 10800
MINIMUM_SECONDS_IN_MONTH = 28 * 86400

# NARR_FIELD_NAMES = [
#     processed_narr_io.TEMPERATURE_NAME,
#     processed_narr_io.SPECIFIC_HUMIDITY_NAME,
#     processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
#     processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
#     processed_narr_io.WET_BULB_TEMP_NAME]
# PRESSURE_LEVELS_MB = numpy.array([1000, 900], dtype=int)

NARR_FIELD_NAMES = [processed_narr_io.WET_BULB_TEMP_NAME]
PRESSURE_LEVELS_MB = numpy.array([1000], dtype=int)

FIRST_MONTH_INPUT_ARG = 'first_month_string'
LAST_MONTH_INPUT_ARG = 'last_month_string'
NARR_DIR_INPUT_ARG = 'top_narr_directory_name'

MONTH_HELP_STRING = (
    'Month (format "yyyymm").  NARR data will be converted to one-time files '
    '(from monthly files) for all months in `{0:s}`...`{1:s}`.').format(
        FIRST_MONTH_INPUT_ARG, LAST_MONTH_INPUT_ARG)
NARR_DIR_HELP_STRING = (
    'Name of top-level directory with processed NARR data (files readable by '
    '`processed_narr_io.read_fields_from_file`).')

DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_MONTH_INPUT_ARG, type=str, required=True,
    help=MONTH_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_MONTH_INPUT_ARG, type=str, required=True,
    help=MONTH_HELP_STRING)


def _get_months_in_range(first_month_string, last_month_string):
    """Returns months in contiguous period.

    :param first_month_string: First month (format "yyyymm").
    :param last_month_string: Last month (format "yyyymm").
    :return: month_strings: 1-D list of months (format "yyyymm").
    """

    first_year = int(first_month_string[:4])
    first_month = int(first_month_string[4:])
    last_year = int(last_month_string[:4])
    last_month = int(last_month_string[4:])

    month_strings = []
    for this_year in range(first_year, last_year + 1):
        if first_year == last_year:
            this_first_month = first_month
            this_last_month = last_month
        elif this_year == first_year:
            this_first_month = first_month
            this_last_month = 12
        elif this_year == last_year:
            this_first_month = 1
            this_last_month = last_month
        else:
            this_first_month = 1
            this_last_month = 12

        for this_month in range(this_first_month, this_last_month + 1):
            month_strings.append('{0:04d}{1:02d}'.format(this_year, this_month))

    return month_strings


def _convert_narr_to_one_time_files(
        top_narr_directory_name, first_month_string, last_month_string):
    """Converts NARR data to one-time files.

    :param top_narr_directory_name: Name of top-level directory with processed
        NARR data (files readable by `processed_narr_io.read_fields_from_file`).
    :param first_month_string: Month (format "yyyymm").  NARR data will be
        converted to one-time files (from monthly files) for all months in
        `first_month_string`...`last_month_string`.
    :param last_month_string: See above.
    """

    valid_month_strings = _get_months_in_range(
        first_month_string=first_month_string,
        last_month_string=last_month_string)
    valid_months_unix_sec = numpy.array(
        [time_conversion.string_to_unix_sec(s, TIME_FORMAT_MONTH)
         for s in valid_month_strings])

    num_months = len(valid_month_strings)
    num_fields = len(NARR_FIELD_NAMES)
    num_pressure_levels = len(PRESSURE_LEVELS_MB)

    for i in range(num_months):
        this_start_time_unix_sec, this_end_time_unix_sec = (
            time_conversion.first_and_last_times_in_month(
                valid_months_unix_sec[i]))
        this_start_time_unix_sec += 26 * 86400
        this_end_time_unix_sec += 1 - NARR_TIME_INTERVAL_SECONDS

        this_month_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=this_start_time_unix_sec,
            end_time_unix_sec=this_end_time_unix_sec,
            time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

        for j in range(num_fields):
            for k in range(num_pressure_levels):
                this_input_file_name = (
                    processed_narr_io.find_file_for_time_period(
                        directory_name=top_narr_directory_name,
                        field_name=NARR_FIELD_NAMES[j],
                        pressure_level_mb=PRESSURE_LEVELS_MB[k],
                        start_time_unix_sec=this_start_time_unix_sec,
                        end_time_unix_sec=this_end_time_unix_sec,
                        raise_error_if_missing=True))

                print 'Reading data from monthly file: "{0:s}"...'.format(
                    this_input_file_name)
                (this_field_matrix, this_field_name,
                 this_pressure_level_pascals, these_times_unix_sec) = (
                     processed_narr_io.read_fields_from_file(
                         this_input_file_name))

                this_orig_field_name = copy.deepcopy(this_field_name)

                this_field_name = this_orig_field_name.replace(
                    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
                    processed_narr_io.U_WIND_GRID_RELATIVE_NAME)
                this_field_name = this_field_name.replace(
                    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME,
                    processed_narr_io.V_WIND_GRID_RELATIVE_NAME)
                this_field_name = this_field_name.replace(
                    processed_narr_io.WET_BULB_THETA_NAME,
                    processed_narr_io.WET_BULB_TEMP_NAME)

                assert this_field_name == NARR_FIELD_NAMES[j]
                this_pressure_level_mb = int(numpy.round(
                    this_pressure_level_pascals / MB_TO_PASCALS))
                assert this_pressure_level_mb == PRESSURE_LEVELS_MB[k]
                assert numpy.array_equal(
                    these_times_unix_sec, this_month_times_unix_sec)

                if this_field_name != this_orig_field_name:
                    print ('Rewriting data with proper field name ("{0:s}"): '
                           '"{1:s}"...').format(this_field_name,
                                                this_input_file_name)

                    processed_narr_io.write_fields_to_file(
                        pickle_file_name=this_input_file_name,
                        field_matrix=this_field_matrix,
                        field_name=NARR_FIELD_NAMES[j],
                        pressure_level_pascals=
                        PRESSURE_LEVELS_MB[k] * MB_TO_PASCALS,
                        valid_times_unix_sec=this_month_times_unix_sec)

                for m in range(len(this_month_times_unix_sec)):
                    this_output_file_name = (
                        processed_narr_io.find_file_for_one_time(
                            top_directory_name=top_narr_directory_name,
                            field_name=NARR_FIELD_NAMES[j],
                            pressure_level_mb=PRESSURE_LEVELS_MB[k],
                            valid_time_unix_sec=this_month_times_unix_sec[m],
                            raise_error_if_missing=False))

                    print 'Writing data to one-time file: "{0:s}"...'.format(
                        this_output_file_name)

                    processed_narr_io.write_fields_to_file(
                        pickle_file_name=this_output_file_name,
                        field_matrix=this_field_matrix[[m], :, :],
                        field_name=NARR_FIELD_NAMES[j],
                        pressure_level_pascals=
                        PRESSURE_LEVELS_MB[k] * MB_TO_PASCALS,
                        valid_times_unix_sec=numpy.array(
                            [this_month_times_unix_sec[m]]))

                print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    TOP_NARR_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_INPUT_ARG)
    FIRST_MONTH_STRING = getattr(INPUT_ARG_OBJECT, FIRST_MONTH_INPUT_ARG)
    LAST_MONTH_STRING = getattr(INPUT_ARG_OBJECT, LAST_MONTH_INPUT_ARG)

    _convert_narr_to_one_time_files(
        top_narr_directory_name=TOP_NARR_DIRECTORY_NAME,
        first_month_string=FIRST_MONTH_STRING,
        last_month_string=LAST_MONTH_STRING)
