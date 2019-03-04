"""Converts WPC warm and cold fronts to polylines and NARR grids."""

import os.path
import argparse
import warnings
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
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
GRIDDED_DIR_ARG_NAME = 'output_gridded_dir_name'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script processes WPC bulletins for all '
    '3-hour time steps in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

BULLETIN_DIR_HELP_STRING = (
    'Name of top-level input directory with WPC bulletins.  Files therein will '
    'be found by `wpc_bulletin_io.find_file` and read by '
    '`wpc_bulletin_io.read_fronts_from_file`.')

POLYLINE_DIR_HELP_STRING = (
    'Name of top-level output directory for polylines.  Files will be written '
    'by `fronts_io.write_polylines_to_file` to locations therein determined by '
    '`fronts_io.find_polyline_file`.')

GRIDDED_DIR_HELP_STRING = (
    'Name of top-level output directory for gridded front labels.  Files will '
    'be written by `fronts_io.write_grid_to_file` to locations therein '
    'determined by `fronts_io.find_gridded_file`.')

TOP_BULLETIN_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/wpc_bulletins/hires'
TOP_POLYLINE_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/polylines')
TOP_GRIDDED_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BULLETIN_DIR_ARG_NAME, type=str, required=False,
    default=TOP_BULLETIN_DIR_NAME_DEFAULT, help=BULLETIN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POLYLINE_DIR_ARG_NAME, type=str, required=False,
    default=TOP_POLYLINE_DIR_NAME_DEFAULT, help=POLYLINE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDDED_DIR_ARG_NAME, type=str, required=False,
    default=TOP_GRIDDED_DIR_NAME_DEFAULT, help=GRIDDED_DIR_HELP_STRING)


def _run(first_time_string, last_time_string, top_bulletin_dir_name,
         top_polyline_dir_name, top_gridded_dir_name):
    """Converts WPC warm and cold fronts to polylines and NARR grids.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param top_bulletin_dir_name: Same.
    :param top_polyline_dir_name: Same.
    :param top_gridded_dir_name: Same.
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

        this_polyline_file_name = fronts_io.find_polyline_file(
            top_directory_name=top_polyline_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing polylines to file: "{0:s}"...'.format(
            this_polyline_file_name)
        fronts_io.write_polylines_to_file(
            polyline_table=this_polyline_table,
            valid_time_unix_sec=valid_times_unix_sec[i],
            netcdf_file_name=this_polyline_file_name)

        this_gridded_front_table = front_utils.many_polylines_to_narr_grid(
            polyline_table=this_polyline_table,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        print this_gridded_front_table

        this_gridded_front_table = this_gridded_front_table.iloc[[0]]

        this_argument_dict = {
            front_utils.DILATION_DISTANCE_COLUMN:
                numpy.array([DILATION_DISTANCE_METRES]),
            front_utils.MODEL_NAME_COLUMN: [nwp_model_utils.NARR_MODEL_NAME]
        }

        this_gridded_front_table = this_gridded_front_table.assign(
            **this_argument_dict)

        this_gridded_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_gridded_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing NARR grids to file: "{0:s}"...\n'.format(
            this_gridded_file_name)
        fronts_io.write_grid_to_file(
            gridded_label_table=this_gridded_front_table,
            netcdf_file_name=this_gridded_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_bulletin_dir_name=getattr(INPUT_ARG_OBJECT, BULLETIN_DIR_ARG_NAME),
        top_polyline_dir_name=getattr(INPUT_ARG_OBJECT, POLYLINE_DIR_ARG_NAME),
        top_gridded_dir_name=getattr(INPUT_ARG_OBJECT, GRIDDED_DIR_ARG_NAME)
    )
