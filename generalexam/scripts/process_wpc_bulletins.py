"""Turns warm/cold fronts from WPC bulletins into polylines and NARR grids."""

import os.path
import argparse
import warnings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import wpc_bulletin_io
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'
TIME_INTERVAL_SECONDS = 10800

DILATION_DISTANCE_METRES = 1.

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
BULLETIN_DIR_ARG_NAME = 'input_bulletin_dir_name'
POLYLINE_DIR_ARG_NAME = 'output_polyline_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'output_frontal_grid_dir_name'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script turns warm/cold fronts into '
    'polylines and NARR grids for all 3-hour time steps from `{0:s}`'
    '...`{1:s}`.').format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)
BULLETIN_DIR_HELP_STRING = 'Name of top-level directory with WPC bulletins.'
POLYLINE_DIR_HELP_STRING = (
    'Name of top-level directory for Pickle files with frontal polylines.')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory for Pickle files with frontal grids.')

DEFAULT_BULLETIN_DIR_NAME = '/condo/swatwork/ralager/wpc_bulletins/hires'
DEFAULT_POLYLINE_DIR_NAME = '/condo/swatwork/ralager/fronts/new/polylines'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/new/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BULLETIN_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_BULLETIN_DIR_NAME, help=BULLETIN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POLYLINE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_POLYLINE_DIR_NAME, help=POLYLINE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_FRONTAL_GRID_DIR_NAME, help=FRONTAL_GRID_DIR_HELP_STRING)


def _process_wpc_bulletins(
        first_time_string, last_time_string, top_bulletin_dir_name,
        top_polyline_dir_name, top_frontal_grid_dir_name):
    """Turns warm/cold fronts from WPC bulletins into polylines and NARR grids.

    :param first_time_string: Time (format "yyyymmddHH").  This script turns
        warm/cold fronts into polylines and NARR grids for all 3-hour time steps
        from `first_time_string`...`last_time_string`.
    :param last_time_string: See above.
    :param top_bulletin_dir_name: [input] Name of top-level directory with WPC
        bulletins.
    :param top_polyline_dir_name: [output] Name of top-level directory for
        Pickle files with frontal polylines.
    :param top_frontal_grid_dir_name: [output] Name of top-level directory for
        Pickle files with frontal grids.
    """

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
        this_bulletin_file_name = wpc_bulletin_io.find_file(
            valid_time_unix_sec=valid_times_unix_sec[i],
            top_directory_name=top_bulletin_dir_name,
            raise_error_if_missing=False)

        if not os.path.isfile(this_bulletin_file_name):
            warning_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
                this_bulletin_file_name)
            warnings.warn(warning_string)
            continue

        print 'Reading data from: "{0:s}"...'.format(this_bulletin_file_name)
        this_polyline_table = wpc_bulletin_io.read_fronts_from_file(
            this_bulletin_file_name)

        this_polyline_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_polyline_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing polylines to file: "{0:s}"...'.format(
            this_polyline_file_name)
        fronts_io.write_polylines_to_file(
            front_table=this_polyline_table,
            pickle_file_name=this_polyline_file_name)

        print 'Converting polylines to NARR grids...'
        this_frontal_grid_table = front_utils.many_polylines_to_narr_grid(
            polyline_table=this_polyline_table,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        this_gridded_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing NARR grids to file: "{0:s}"...\n'.format(
            this_gridded_file_name)
        fronts_io.write_narr_grids_to_file(
            frontal_grid_table=this_frontal_grid_table,
            pickle_file_name=this_gridded_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    FIRST_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME)
    LAST_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME)
    TOP_BULLETIN_DIR_NAME = getattr(INPUT_ARG_OBJECT, BULLETIN_DIR_ARG_NAME)
    TOP_POLYLINE_DIR_NAME = getattr(INPUT_ARG_OBJECT, POLYLINE_DIR_ARG_NAME)
    TOP_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)

    _process_wpc_bulletins(
        first_time_string=FIRST_TIME_STRING,
        last_time_string=LAST_TIME_STRING,
        top_bulletin_dir_name=TOP_BULLETIN_DIR_NAME,
        top_polyline_dir_name=TOP_POLYLINE_DIR_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME)
