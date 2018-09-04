"""Extracts warm and cold fronts from WPC bulletins for given time period."""

import os.path
import argparse
import warnings
import pandas
from generalexam.ge_io import wpc_bulletin_io
from generalexam.ge_io import fronts_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods

INPUT_TIME_FORMAT = '%Y%m%d%H'
DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SECONDS = 10800

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_TIME_INPUT_ARG = 'first_time_string'
LAST_TIME_INPUT_ARG = 'last_time_string'
BULLETIN_DIR_INPUT_ARG = 'input_bulletin_dir_name'
FRONT_FILE_INPUT_ARG = 'output_front_file_name'

TIME_HELP_STRING = (
    'Time in format "yyyymmddHH".  This script processes WPC bulletins for all '
    '3-hour time steps from `{0:s}`...`{1:s}`.').format(
        FIRST_TIME_INPUT_ARG, LAST_TIME_INPUT_ARG)
BULLETIN_DIR_HELP_STRING = (
    'Name of top-level input directory, containing WPC bulletins.')
FRONT_FILE_HELP_STRING = (
    'Path to output file, which will be written by '
    '`fronts_io.write_polylines_to_file` and contain all fronts from the given '
    'time period.')

DEFAULT_BULLETIN_DIR_NAME = '/condo/swatwork/ralager/wpc_bulletins/hires'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_INPUT_ARG, type=str, required=True,
    help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_INPUT_ARG, type=str, required=True,
    help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + BULLETIN_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_BULLETIN_DIR_NAME, help=BULLETIN_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_FILE_INPUT_ARG, type=str, required=True,
    help=FRONT_FILE_HELP_STRING)


def _process_wpc_bulletins(
        first_time_string, last_time_string, top_input_bulletin_dir_name,
        output_front_file_name):
    """Extracts warm and cold fronts from WPC bulletins for given time period.

    :param first_time_string: Time in format "yyyymmddHH".  This script
        processes WPC bulletins for all 3-hour time steps from
        `first_time_string`...`last_time_string`.
    :param last_time_string: See above.
    :param top_input_bulletin_dir_name: Name of top-level directory with WPC
        bulletins.
    :param output_front_file_name: Path to output file, which will be written by
        `fronts_io.write_polylines_to_file` and contain all fronts from the
        given time period.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, DEFAULT_TIME_FORMAT) for
        t in valid_times_unix_sec]

    num_times = len(valid_times_unix_sec)
    list_of_front_tables = []

    for i in range(num_times):
        this_bulletin_file_name = wpc_bulletin_io.find_file(
            valid_time_unix_sec=valid_times_unix_sec[i],
            top_directory_name=top_input_bulletin_dir_name,
            raise_error_if_missing=False)

        if not os.path.isfile(this_bulletin_file_name):
            warning_string = (
                'Could not find WPC bulletin for {0:s}.  Expected at: '
                '"{1:s}"').format(
                    valid_time_strings[i], this_bulletin_file_name)
            warnings.warn(warning_string)
            continue

        print 'Extracting fronts from: "{0:s}"...'.format(
            this_bulletin_file_name)

        list_of_front_tables.append(
            wpc_bulletin_io.read_fronts_from_file(this_bulletin_file_name))
        if len(list_of_front_tables) == 1:
            continue

        list_of_front_tables[-1], _ = list_of_front_tables[-1].align(
            list_of_front_tables[0], axis=1)

    print SEPARATOR_STRING
    front_table = pandas.concat(list_of_front_tables, axis=0, ignore_index=True)

    print 'Writing front data to: "{0:s}"...'.format(output_front_file_name)
    fronts_io.write_polylines_to_file(front_table, output_front_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    FIRST_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_TIME_INPUT_ARG)
    LAST_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_TIME_INPUT_ARG)
    TOP_INPUT_BULLETIN_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, BULLETIN_DIR_INPUT_ARG)
    OUTPUT_FRONT_FILE_NAME = getattr(INPUT_ARG_OBJECT, FRONT_FILE_INPUT_ARG)

    _process_wpc_bulletins(
        first_time_string=FIRST_TIME_STRING,
        last_time_string=LAST_TIME_STRING,
        top_input_bulletin_dir_name=TOP_INPUT_BULLETIN_DIR_NAME,
        output_front_file_name=OUTPUT_FRONT_FILE_NAME)
