"""Creates mask for NARR grid.

Unmasked grid cells will be those where the WPC typically draws fronts.  Masked
grid cells will be those where they typically do not.
"""

import os.path
import warnings
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
MIN_FRONTS_ARG_NAME = 'min_num_fronts'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded front labels.  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.')

TIME_HELP_STRING = (
    'Format ("yyyymmddHH").  Fronts will be read for all times in the period '
    '`{0:s}`...`{1:s}`, and the number of fronts will be counted at each grid '
    'cell.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance.  Will be applied to each front (warm-frontal and cold-'
    'frontal grid cells) before counting the number at each grid cell.')

MIN_FRONTS_HELP_STRING = (
    'Minimum number of fronts (num warm fronts + num cold fronts).  Any grid '
    'cell with < `{0:s}` fronts over the time period will be masked.  The '
    'others will be unmasked.'
).format(MIN_FRONTS_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`machine_learning_utils.write_narr_mask`.')

TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')
DEFAULT_DILATION_DISTANCE_METRES = 50000.
MIN_NUM_FRONTS_DEFAULT = 100

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=DEFAULT_DILATION_DISTANCE_METRES,
    help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_FRONTS_ARG_NAME, type=int, required=False,
    default=MIN_NUM_FRONTS_DEFAULT, help=MIN_FRONTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_gridded_front_dir_name, first_time_string, last_time_string,
         dilation_distance_metres, min_num_fronts, output_file_name):
    """Creates mask for NARR grid.

    This is effectively the main method.

    :param top_gridded_front_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param dilation_distance_metres: Same.
    :param min_num_fronts: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(min_num_fronts, 0)

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
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_221GRID)

    num_cold_fronts_matrix = None
    num_warm_fronts_matrix = None

    for i in range(num_times):
        this_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_gridded_front_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            warning_string = (
                'POTENTIAL PROBLEM.  Cannot find file: "{0:s}"'
            ).format(this_file_name)

            warnings.warn(warning_string)
            continue

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_gridded_front_table = fronts_io.read_grid_from_file(this_file_name)

        this_gridded_front_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=this_gridded_front_table,
            num_rows_per_image=num_grid_rows,
            num_columns_per_image=num_grid_columns)

        this_gridded_front_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_gridded_front_matrix,
            dilation_distance_metres=dilation_distance_metres, verbose=False
        )[0, ...]

        this_num_cold_fronts_matrix = (
            this_gridded_front_matrix == front_utils.COLD_FRONT_ENUM
        ).astype(int)
        this_num_warm_fronts_matrix = (
            this_gridded_front_matrix == front_utils.WARM_FRONT_ENUM
        ).astype(int)

        if num_cold_fronts_matrix is None:
            num_cold_fronts_matrix = this_num_cold_fronts_matrix + 0
            num_warm_fronts_matrix = this_num_warm_fronts_matrix + 0
        else:
            num_cold_fronts_matrix = (
                num_cold_fronts_matrix + this_num_cold_fronts_matrix
            )
            num_warm_fronts_matrix = (
                num_warm_fronts_matrix + this_num_warm_fronts_matrix
            )

    print SEPARATOR_STRING
    print 'Masking out grid cells with < {0:d} fronts...'.format(
        min_num_fronts)

    mask_matrix = (
        num_warm_fronts_matrix + num_cold_fronts_matrix >= min_num_fronts
    ).astype(int)

    print 'Number of mask grid cells = {0:d} out of {1:d}'.format(
        numpy.sum(mask_matrix == 0), mask_matrix.size
    )

    print 'Writing mask to: "{0:s}"...'.format(output_file_name)
    ml_utils.write_narr_mask(
        mask_matrix=mask_matrix, num_warm_fronts_matrix=num_warm_fronts_matrix,
        num_cold_fronts_matrix=num_cold_fronts_matrix,
        pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        min_num_fronts=getattr(INPUT_ARG_OBJECT, MIN_FRONTS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
