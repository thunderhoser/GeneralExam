"""Creates example files.

--- NOTATION ---

The following letters will be used throughout this file.

E = number of learning examples
M = number of rows in predictor grid
N = number of columns in predictor grid
C = number of predictor variables
"""

import copy
import os.path
import argparse
import numpy
from keras.utils import to_categorical
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import era5_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import training_validation_io as trainval_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

ERA5_DIR_ARG_NAME = 'input_era5_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
PREDICTOR_NAMES_ARG_NAME = 'predictor_names'
NUM_HALF_ROWS_ARG_NAME = 'num_half_rows'
NUM_HALF_COLUMNS_ARG_NAME = 'num_half_columns'
NORMALIZATION_TYPE_ARG_NAME = 'normalization_type_string'
CLASS_FRACTIONS_ARG_NAME = 'class_fractions'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
MAX_EXAMPLES_PER_TIME_ARG_NAME = 'max_examples_per_time'
NUM_TIMES_PER_FILE_ARG_NAME = 'num_times_per_output_file'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

# TODO(thunderhoser): Need to keep metadata for normalization.
# TODO(thunderhoser): May want to allow different normalization.

ERA5_DIR_HELP_STRING = (
    'Name of top-level directory with ERA5 data (predictors).  Files therein '
    'will be found by `era5_io.find_processed_file` and read by '
    '`era5_io.read_processed_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will create learning examples for'
    ' all time steps in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Will create examples only for this pressure level (millibars).  To use '
    'surface data as predictors, leave this argument alone.')

PREDICTOR_NAMES_HELP_STRING = (
    'Names of predictor variables.  Must be accepted by '
    '`era5_io.check_field_name`.')

NUM_HALF_ROWS_HELP_STRING = (
    'Number of rows in half-grid (on either side of center) for predictors.')

NUM_HALF_COLUMNS_HELP_STRING = (
    'Number of columns in half-grid (on either side of center) for predictors.')

NORMALIZATION_TYPE_HELP_STRING = (
    'Normalization type (must be accepted by '
    '`machine_learning_utils._check_normalization_type`).')

CLASS_FRACTIONS_HELP_STRING = (
    'Downsampling fractions for the 3 classes (no front, warm front, cold '
    'front).  Must sum to 1.0.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded front labels.  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.')

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for gridded warm-front and cold-front labels.')

MASK_FILE_HELP_STRING = (
    'Path to mask file (determines which grid cells can be used as center of '
    'learning example).  Will be read by '
    '`machine_learning_utils.read_narr_mask`.  If you do not want a mask, leave'
    ' this empty.')

MAX_EXAMPLES_PER_TIME_HELP_STRING = (
    'Max number of learning examples per time step.')

NUM_TIMES_PER_FILE_HELP_STRING = (
    'Number of time steps per output (example) file.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Learning examples will be written here by '
    '`training_validation_io.write_downsized_3d_examples`, to exact locations '
    'determined by `training_validation_io.find_downsized_3d_example_file`.')

TOP_ERA5_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'
TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')
DEFAULT_MASK_FILE_NAME = '/condo/swatwork/ralager/fronts_netcdf/era5_mask.p'

DEFAULT_PREDICTOR_NAMES = [
    era5_io.U_WIND_GRID_RELATIVE_NAME,
    era5_io.V_WIND_GRID_RELATIVE_NAME,
    era5_io.WET_BULB_THETA_NAME,
    era5_io.TEMPERATURE_NAME,
    era5_io.SPECIFIC_HUMIDITY_NAME,
    era5_io.HEIGHT_NAME
]

DEFAULT_CLASS_FRACTIONS = numpy.array([0.5, 0.25, 0.25])

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ERA5_DIR_ARG_NAME, type=str, required=False,
    default=TOP_ERA5_DIR_NAME_DEFAULT, help=ERA5_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=era5_io.DUMMY_SURFACE_PRESSURE_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_ROWS_ARG_NAME, type=int, required=False, default=16,
    help=NUM_HALF_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_COLUMNS_ARG_NAME, type=int, required=False, default=16,
    help=NUM_HALF_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_TYPE_ARG_NAME, type=str, required=False,
    default=ml_utils.Z_SCORE_STRING, help=NORMALIZATION_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTIONS_ARG_NAME, type=float, nargs=3, required=False,
    default=DEFAULT_CLASS_FRACTIONS, help=CLASS_FRACTIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=50000, help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_MASK_FILE_NAME, help=MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
    default=5000, help=MAX_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_PER_FILE_ARG_NAME, type=int, required=False, default=8,
    help=NUM_TIMES_PER_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_era5_inputs_one_time(
        top_input_dir_name, valid_time_unix_sec, pressure_level_mb,
        predictor_names):
    """Reads ERA5 input fields for one time.

    J = number of rows in full model grid
    K = number of columns in full model grid

    :param top_input_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: See documentation at top of file.
    :param predictor_names: Same.
    :return: predictor_matrix: 1-by-J-by-K-by-C numpy array of predictor values.
    """

    input_file_name = era5_io.find_processed_file(
        top_directory_name=top_input_dir_name,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    era5_dict = era5_io.read_processed_file(
        netcdf_file_name=input_file_name,
        pressure_levels_to_keep_mb=numpy.array([pressure_level_mb]),
        field_names_to_keep=predictor_names)

    predictor_matrix = era5_dict[era5_io.DATA_MATRIX_KEY][[0], ..., 0, :]
    num_fields = predictor_matrix.shape[-1]

    for k in range(num_fields):
        predictor_matrix[..., k] = ml_utils.fill_nans_in_predictor_images(
            predictor_matrix[..., k]
        )

    return predictor_matrix


def _create_examples_one_time(
        top_era5_dir_name, valid_time_unix_sec, pressure_level_mb,
        predictor_names, num_half_rows, num_half_columns,
        normalization_type_string, class_fractions, top_gridded_front_dir_name,
        dilation_distance_metres, max_num_examples, mask_matrix):
    """Creates learning examples for one time.

    J = number of rows in full model grid
    K = number of columns in full model grid

    :param top_era5_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: See documentation at top of file.
    :param predictor_names: Same.
    :param num_half_rows: Same.
    :param num_half_columns: Same.
    :param normalization_type_string: Same.
    :param class_fractions: Same.
    :param top_gridded_front_dir_name: Same.
    :param dilation_distance_metres: Same.
    :param max_num_examples: Maximum number of examples to create.
    :param mask_matrix: J-by-K numpy array of Boolean flags.  If
        mask_matrix[i, j] = 1, grid cell [i, j] can be used as the center of a
        learning example.
    :return: example_dict: Dictionary with the following keys.
    example_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    example_dict['target_matrix']: E-by-3 numpy array of class labels (all
        0 or 1, but the array type is "float64").
    example_dict['target_times_unix_sec']: length-E numpy array of target times
        (all the same).
    example_dict['row_indices']: length-E numpy array, where row_indices[i] is
        the row index, in the full model grid, at the center of the [i]th
        example.
    example_dict['column_indices']: Same but for columns.
    example_dict['first_normalization_param_matrix']: E-by-C numpy array with
        values of first normalization param (either minimum or mean).
    example_dict['second_normalization_param_matrix']: E-by-C numpy array with
        values of second normalization param (either max or stdev).
    """

    gridded_front_file_name = fronts_io.find_gridded_file(
        top_directory_name=top_gridded_front_dir_name,
        valid_time_unix_sec=valid_time_unix_sec, raise_error_if_missing=False)

    if not os.path.isfile(gridded_front_file_name):
        return None

    predictor_matrix = _read_era5_inputs_one_time(
        top_input_dir_name=top_era5_dir_name,
        valid_time_unix_sec=valid_time_unix_sec,
        pressure_level_mb=pressure_level_mb, predictor_names=predictor_names)

    predictor_matrix, normalization_dict = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix,
        normalization_type_string=normalization_type_string)

    print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
    gridded_front_table = fronts_io.read_grid_from_file(gridded_front_file_name)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2]
    )

    target_matrix = ml_utils.dilate_ternary_target_images(
        target_matrix=target_matrix,
        dilation_distance_metres=dilation_distance_metres, verbose=False)

    sampled_target_point_dict = ml_utils.sample_target_points(
        target_matrix=target_matrix, class_fractions=class_fractions,
        num_points_to_sample=max_num_examples, mask_matrix=mask_matrix)

    if sampled_target_point_dict is None:
        return None

    (predictor_matrix, target_values, time_indices, row_indices, column_indices
    ) = ml_utils.downsize_grids_around_selected_points(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        num_rows_in_half_window=num_half_rows,
        num_columns_in_half_window=num_half_columns,
        target_point_dict=sampled_target_point_dict, verbose=False)

    target_matrix = to_categorical(target_values, 3)
    actual_class_fractions = numpy.sum(target_matrix, axis=0)
    print 'Actual class fractions = {0:s}'.format(str(actual_class_fractions))

    example_dict = {
        trainval_io.PREDICTOR_MATRIX_KEY: predictor_matrix,
        trainval_io.TARGET_MATRIX_KEY: target_matrix,
        trainval_io.TARGET_TIMES_KEY:
            numpy.full(target_matrix.shape[0], valid_time_unix_sec, dtype=int),
        trainval_io.ROW_INDICES_KEY: row_indices,
        trainval_io.COLUMN_INDICES_KEY: column_indices
    }

    if normalization_type_string == ml_utils.MINMAX_STRING:
        first_normalization_param_matrix = normalization_dict[
            ml_utils.MIN_VALUE_MATRIX_KEY
        ][time_indices, ...]
        second_normalization_param_matrix = normalization_dict[
            ml_utils.MAX_VALUE_MATRIX_KEY
        ][time_indices, ...]
    else:
        first_normalization_param_matrix = normalization_dict[
            ml_utils.MEAN_VALUE_MATRIX_KEY
        ][time_indices, ...]
        second_normalization_param_matrix = normalization_dict[
            ml_utils.STDEV_MATRIX_KEY
        ][time_indices, ...]

    example_dict.update({
        trainval_io.FIRST_NORM_PARAM_KEY: first_normalization_param_matrix,
        trainval_io.SECOND_NORM_PARAM_KEY: second_normalization_param_matrix
    })

    return example_dict


def _write_example_file(
        top_output_dir_name, example_dict, first_time_unix_sec,
        last_time_unix_sec, predictor_names, pressure_level_mb,
        dilation_distance_metres, mask_matrix):
    """Writes one set of learning examples to file.

    :param top_output_dir_name: See documentation at top of file.
    :param example_dict: Dictionary with keys documented in
        `_create_examples_one_time`.
    :param first_time_unix_sec: First time in set.
    :param last_time_unix_sec: Last time in set.
    :param predictor_names: See documentation at top of file.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param mask_matrix: Same.
    """

    if example_dict is None:
        return

    output_file_name = trainval_io.find_downsized_3d_example_file(
        top_directory_name=top_output_dir_name,
        first_target_time_unix_sec=first_time_unix_sec,
        last_target_time_unix_sec=last_time_unix_sec,
        raise_error_if_missing=False)

    print 'Writing examples to file: "{0:s}"...'.format(output_file_name)
    trainval_io.write_downsized_3d_examples(
        netcdf_file_name=output_file_name, example_dict=example_dict,
        narr_predictor_names=predictor_names,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_metres=dilation_distance_metres,
        narr_mask_matrix=mask_matrix)


def _run(top_era5_dir_name, first_time_string, last_time_string,
         pressure_level_mb, predictor_names, num_half_rows, num_half_columns,
         normalization_type_string, class_fractions, top_gridded_front_dir_name,
         dilation_distance_metres, mask_file_name, max_examples_per_time,
         num_times_per_output_file, top_output_dir_name):
    """Creates example files.

    This is effectively the main method.

    :param top_era5_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param predictor_names: Same.
    :param num_half_rows: Same.
    :param num_half_columns: Same.
    :param normalization_type_string: Same.
    :param class_fractions: Same.
    :param top_gridded_front_dir_name: Same.
    :param dilation_distance_metres: Same.
    :param mask_file_name: Same.
    :param max_examples_per_time: Same.
    :param num_times_per_output_file: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_greater(num_times_per_output_file, 0)

    if mask_file_name in ['', 'None']:
        mask_file_name = None

    if pressure_level_mb == era5_io.DUMMY_SURFACE_PRESSURE_MB:
        predictor_names = [
            era5_io.PRESSURE_NAME if n == era5_io.HEIGHT_NAME else n
            for n in predictor_names
        ]
    else:
        predictor_names = [
            era5_io.HEIGHT_NAME if n == era5_io.PRESSURE_NAME else n
            for n in predictor_names
        ]

    if mask_file_name is not None:
        print 'Reading mask from: "{0:s}"...\n'.format(mask_file_name)
        mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]
    else:
        mask_matrix = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    this_example_dict = None
    this_first_time_unix_sec = valid_times_unix_sec[0]
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        if numpy.mod(i, num_times_per_output_file) == 0 and i > 0:
            _write_example_file(
                top_output_dir_name=top_output_dir_name,
                example_dict=this_example_dict,
                first_time_unix_sec=this_first_time_unix_sec,
                last_time_unix_sec=valid_times_unix_sec[i - 1],
                predictor_names=predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                mask_matrix=mask_matrix)

            print SEPARATOR_STRING
            this_example_dict = None
            this_first_time_unix_sec = valid_times_unix_sec[i]

        this_new_example_dict = _create_examples_one_time(
            top_era5_dir_name=top_era5_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            pressure_level_mb=pressure_level_mb,
            predictor_names=predictor_names,
            num_half_rows=num_half_rows, num_half_columns=num_half_columns,
            normalization_type_string=normalization_type_string,
            class_fractions=class_fractions,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            dilation_distance_metres=dilation_distance_metres,
            max_num_examples=max_examples_per_time, mask_matrix=mask_matrix)

        print '\n'
        if this_new_example_dict is None:
            continue

        if this_example_dict is None:
            this_example_dict = copy.deepcopy(this_new_example_dict)
            continue

        for this_key in trainval_io.MAIN_KEYS:
            this_example_dict[this_key] = numpy.concatenate(
                (this_example_dict[this_key], this_new_example_dict[this_key]),
                axis=0
            )

    _write_example_file(
        top_output_dir_name=top_output_dir_name,
        example_dict=this_example_dict,
        first_time_unix_sec=this_first_time_unix_sec,
        last_time_unix_sec=valid_times_unix_sec[-1],
        predictor_names=predictor_names,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_metres=dilation_distance_metres,
        mask_matrix=mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_era5_dir_name=getattr(INPUT_ARG_OBJECT, ERA5_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        predictor_names=getattr(INPUT_ARG_OBJECT, PREDICTOR_NAMES_ARG_NAME),
        num_half_rows=getattr(INPUT_ARG_OBJECT, NUM_HALF_ROWS_ARG_NAME),
        num_half_columns=getattr(INPUT_ARG_OBJECT, NUM_HALF_COLUMNS_ARG_NAME),
        normalization_type_string=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_TYPE_ARG_NAME),
        class_fractions=numpy.array(getattr(
            INPUT_ARG_OBJECT, CLASS_FRACTIONS_ARG_NAME
        )),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        max_examples_per_time=getattr(
            INPUT_ARG_OBJECT, MAX_EXAMPLES_PER_TIME_ARG_NAME),
        num_times_per_output_file=getattr(
            INPUT_ARG_OBJECT, NUM_TIMES_PER_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
