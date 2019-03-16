"""Masks WPC fronts (polylines) that pass through only masked grid cells."""

import os.path
import argparse
import warnings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

INPUT_DIR_ARG_NAME = 'input_polyline_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_polyline_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with unmasked fronts as polylines.  Files '
    'therein will be found by `fronts_io.find_polyline_file` and read by '
    '`fronts_io.read_polylines_from_file`.')

MASK_FILE_HELP_STRING = (
    'Path to mask file (fronts passing through only masked grid cells will be '
    'removed).  Will be read by `machine_learning_utils.read_narr_mask`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will mask polylines for all time '
    'steps in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for masked fronts.  Files will be written here'
    ' by `fronts_io.write_polylines_to_file`, to exact locations determined by '
    '`fronts_io.find_polyline_file`.')

TOP_INPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts_netcdf/polylines'
TOP_OUTPUT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/polylines/masked_era5')
MASK_FILE_NAME_DEFAULT = '/condo/swatwork/ralager/fronts_netcdf/era5_mask.p'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_INPUT_DIR_NAME_DEFAULT, help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=False,
    default=MASK_FILE_NAME_DEFAULT, help=MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_OUTPUT_DIR_NAME_DEFAULT, help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, mask_file_name, first_time_string,
         last_time_string, top_output_dir_name):
    """Masks WPC fronts (polylines) that pass through only masked grid cells.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param mask_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param top_output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    print 'Reading mask from: "{0:s}"...'.format(mask_file_name)
    mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]

    print SEPARATOR_STRING

    for this_time_unix_sec in valid_times_unix_sec:
        this_unmasked_file_name = fronts_io.find_polyline_file(
            top_directory_name=top_input_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False)

        if not os.path.isfile(this_unmasked_file_name):
            warning_string = (
                'POTENTIAL PROBLEM.  Cannot find file expected at: "{0:s}"'
            ).format(this_unmasked_file_name)

            warnings.warn(warning_string)
            continue

        print 'Reading unmasked polylines from: "{0:s}"...'.format(
            this_unmasked_file_name)
        this_polyline_table = fronts_io.read_polylines_from_file(
            this_unmasked_file_name)

        print MINOR_SEPARATOR_STRING
        this_polyline_table = front_utils.remove_fronts_in_masked_area(
            polyline_table=this_polyline_table, narr_mask_matrix=mask_matrix,
            verbose=True)
        print MINOR_SEPARATOR_STRING

        this_masked_file_name = fronts_io.find_polyline_file(
            top_directory_name=top_output_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False)

        print 'Writing masked polylines to: "{0:s}"...'.format(
            this_masked_file_name)
        fronts_io.write_polylines_to_file(
            netcdf_file_name=this_masked_file_name,
            polyline_table=this_polyline_table,
            valid_time_unix_sec=this_time_unix_sec)

        if this_time_unix_sec != valid_times_unix_sec[-1]:
            print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
