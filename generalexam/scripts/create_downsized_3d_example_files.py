"""Writes downsized 3-D training examples to files."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import training_validation_io as trainval_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
NARR_TIME_STEP_SECONDS = 10800

MAIN_KEYS = [
    trainval_io.PREDICTOR_MATRIX_KEY, trainval_io.TARGET_MATRIX_KEY,
    trainval_io.TARGET_TIMES_KEY, trainval_io.ROW_INDICES_KEY,
    trainval_io.COLUMN_INDICES_KEY
]

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
MAX_EXAMPLES_PER_TIME_ARG_NAME = 'max_num_examples_per_time'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
PREDICTOR_NAMES_ARG_NAME = 'narr_predictor_names'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
CLASS_FRACTIONS_ARG_NAME = 'class_fractions'
NUM_HALF_ROWS_ARG_NAME = 'num_half_rows'
NUM_HALF_COLUMNS_ARG_NAME = 'num_half_columns'
FRONT_DIR_ARG_NAME = 'top_frontal_grid_dir_name'
NARR_DIRECTORY_ARG_NAME = 'top_narr_directory_name'
NARR_MASK_FILE_ARG_NAME = 'narr_mask_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
NUM_TIMES_PER_OUT_FILE_ARG_NAME = 'num_times_per_output_file'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Downsized 3-D training examples will be '
    'printed for all times from `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

MAX_EXAMPLES_PER_TIME_HELP_STRING = 'Max number of examples per time step.'

PRESSURE_LEVEL_HELP_STRING = (
    'All NARR predictors will be taken from this pressure level (millibars).')

PREDICTOR_NAMES_HELP_STRING = (
    'List of NARR predictors.  Each name must be accepted by '
    '`processed_narr_io.check_field_name`.')

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance.  Will be used to dilate WF and CF labels, which '
    'effectively creates a distance buffer around each front, thus accounting '
    'for spatial uncertainty in front placement.')

CLASS_FRACTIONS_HELP_STRING = (
    'List of downsampling fractions.  Must have length 3, where the elements '
    'are (NF, WF, CF).  The sum of all fractions must be 1.0.')

NUM_HALF_ROWS_HELP_STRING = (
    'Number of rows in half-grid for each example.  Actual number of rows will '
    'be 2 * `{0:s}` + 1.'
).format(NUM_HALF_ROWS_ARG_NAME)

NUM_HALF_COLUMNS_HELP_STRING = (
    'Number of columns in half-grid for each example.  Actual number of columns'
    ' will be 2 * `{0:s}` + 1.'
).format(NUM_HALF_COLUMNS_ARG_NAME)

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (full-size target images).'
    '  Files therein will be found by `fronts_io.find_file_for_one_time` and '
    'read by `fronts_io.read_narr_grids_from_file`.')

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with NARR grids (full-size predictor images).'
    '  Files therein will be found by '
    '`processed_narr_io.find_file_for_one_time` and read by '
    '`processed_narr_io.read_fields_from_file`.')

NARR_MASK_FILE_HELP_STRING = (
    'See doc for `machine_learning_utils.read_narr_mask`.  Determines which '
    'grid cells can be used as the center of a downsized example.  If you want '
    'no mask, make this the empty string ("").')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Downsized 3-D examples will be '
    'written here by ``, to locations determined by ``.')

NUM_TIMES_PER_OUT_FILE_HELP_STRING = 'Number of time steps in each output file.'

DEFAULT_MAX_EXAMPLES_PER_TIME = 5000
DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_THETA_NAME,
    processed_narr_io.TEMPERATURE_NAME,
    processed_narr_io.SPECIFIC_HUMIDITY_NAME,
    processed_narr_io.HEIGHT_NAME
]
DEFAULT_DILATION_DISTANCE_METRES = 50000.
DEFAULT_CLASS_FRACTIONS = numpy.array([0.5, 0.25, 0.25])
DEFAULT_NUM_HALF_ROWS = 32
DEFAULT_NUM_HALF_COLUMNS = 32
DEFAULT_TOP_FRONT_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_TOP_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_NARR_MASK_FILE_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p')
DEFAULT_NUM_TIMES_PER_OUT_FILE = 8

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
    default=DEFAULT_MAX_EXAMPLES_PER_TIME,
    help=MAX_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_NARR_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=DEFAULT_DILATION_DISTANCE_METRES,
    help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTIONS_ARG_NAME, type=float, nargs='+', required=False,
    default=DEFAULT_CLASS_FRACTIONS, help=CLASS_FRACTIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_ROWS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_ROWS, help=NUM_HALF_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_COLUMNS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_COLUMNS, help=NUM_HALF_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_FRONT_DIR_NAME, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_NARR_DIR_NAME, help=NARR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_MASK_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_MASK_FILE_NAME, help=NARR_MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_PER_OUT_FILE_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TIMES_PER_OUT_FILE,
    help=NUM_TIMES_PER_OUT_FILE_HELP_STRING)


def _run(first_time_string, last_time_string, max_num_examples_per_time,
         pressure_level_mb, narr_predictor_names, dilation_distance_metres,
         class_fractions, num_half_rows, num_half_columns,
         top_frontal_grid_dir_name, top_narr_directory_name,
         narr_mask_file_name, output_dir_name, num_times_per_output_file):
    """Writes downsized 3-D training examples to files.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param max_num_examples_per_time: Same.
    :param pressure_level_mb: Same.
    :param narr_predictor_names: Same.
    :param dilation_distance_metres: Same.
    :param class_fractions: Same.
    :param num_half_rows: Same.
    :param num_half_columns: Same.
    :param top_frontal_grid_dir_name: Same.
    :param top_narr_directory_name: Same.
    :param narr_mask_file_name: Same.
    :param output_dir_name: Same.
    :param num_times_per_output_file: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_STEP_SECONDS)

    if narr_mask_file_name == '':
        narr_mask_matrix = None
    else:
        print 'Reading NARR mask from: "{0:s}"...'.format(narr_mask_file_name)
        narr_mask_matrix = ml_utils.read_narr_mask(narr_mask_file_name)
        print SEPARATOR_STRING

    error_checking.assert_is_greater(num_times_per_output_file, 0)

    num_target_times = len(target_times_unix_sec)
    this_example_dict = None
    this_first_time_unix_sec = target_times_unix_sec[0]

    for i in range(num_target_times):
        if numpy.mod(i, num_times_per_output_file) == 0 and i != 0:
            this_last_time_unix_sec = target_times_unix_sec[i - 1]
            this_output_file_name = trainval_io.find_downsized_3d_example_file(
                directory_name=output_dir_name,
                first_target_time_unix_sec=this_first_time_unix_sec,
                last_target_time_unix_sec=this_last_time_unix_sec,
                raise_error_if_missing=False)

            print 'Writing data to file: "{0:s}"...'.format(
                this_output_file_name)
            trainval_io.write_downsized_3d_examples(
                netcdf_file_name=this_output_file_name,
                example_dict=this_example_dict,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                narr_mask_matrix=narr_mask_matrix)
            print SEPARATOR_STRING

            this_example_dict = None
            this_first_time_unix_sec = target_times_unix_sec[i]

        this_new_example_dict = trainval_io.prep_downsized_3d_examples_to_write(
            target_time_unix_sec=target_times_unix_sec[i],
            max_num_examples=max_num_examples_per_time,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb,
            dilation_distance_metres=dilation_distance_metres,
            class_fractions=class_fractions,
            num_rows_in_half_grid=num_half_rows,
            num_columns_in_half_grid=num_half_columns,
            narr_mask_matrix=narr_mask_matrix)
        print '\n'

        if this_example_dict is None:
            this_example_dict = copy.deepcopy(this_new_example_dict)
            continue

        for this_key in MAIN_KEYS:
            this_example_dict[this_key] = numpy.concatenate(
                (this_example_dict[this_key],
                 this_new_example_dict[this_key]), axis=0)

    if this_example_dict is not None:
        this_last_time_unix_sec = target_times_unix_sec[-1]
        this_output_file_name = trainval_io.find_downsized_3d_example_file(
            directory_name=output_dir_name,
            first_target_time_unix_sec=this_first_time_unix_sec,
            last_target_time_unix_sec=this_last_time_unix_sec,
            raise_error_if_missing=False)

        print 'Writing data to file: "{0:s}"...'.format(
            this_output_file_name)
        trainval_io.write_downsized_3d_examples(
            netcdf_file_name=this_output_file_name,
            example_dict=this_example_dict,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb,
            dilation_distance_metres=dilation_distance_metres,
            narr_mask_matrix=narr_mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        max_num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, MAX_EXAMPLES_PER_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        narr_predictor_names=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_NAMES_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        class_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTIONS_ARG_NAME)),
        num_half_rows=getattr(INPUT_ARG_OBJECT, NUM_HALF_ROWS_ARG_NAME),
        num_half_columns=getattr(INPUT_ARG_OBJECT, NUM_HALF_COLUMNS_ARG_NAME),
        top_frontal_grid_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        narr_mask_file_name=getattr(INPUT_ARG_OBJECT, NARR_MASK_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        num_times_per_output_file=getattr(
            INPUT_ARG_OBJECT, NUM_TIMES_PER_OUT_FILE_ARG_NAME)
    )
