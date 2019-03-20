"""Applies trained CNN to full grids."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import machine_learning_utils as ml_utils

RANDOM_SEED = 6695
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
USE_MASK_ARG_NAME = 'use_mask'
USE_ISOTONIC_ARG_NAME = 'use_isotonic_regression'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded front labels.  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will apply the CNN to `{0:s}` '
    'random time steps in the period `{1:s}`...`{2:s}`.'
).format(NUM_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of times to draw randomly from `{0:s}`...`{1:s}`.  To use all times'
    ' in the period, leave this argument alone.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

USE_MASK_HELP_STRING = (
    'Boolean flag.  If 1, the CNN will not be applied to masked grid cells '
    '(using the same mask used for training).  If 0, the CNN will be applied to'
    ' all grid cells.')

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, will use isotonic regression to calibrate CNN '
    'probabilities.  If 0, will use raw CNN probabilities with no calibration.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`machine_learning_utils.write_gridded_predictions`, to exact locations '
    'determined by `machine_learning_utils.find_gridded_prediction_file`.')

TOP_PREDICTOR_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/era5_data/processed/with_theta_w')
TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_PREDICTOR_DIR_NAME_DEFAULT, help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_MASK_ARG_NAME, type=int, required=False, default=0,
    help=USE_MASK_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ISOTONIC_ARG_NAME, type=int, required=False, default=0,
    help=USE_ISOTONIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, top_predictor_dir_name, top_gridded_front_dir_name,
         first_time_string, last_time_string, num_times, use_mask,
         use_isotonic_regression, output_dir_name):
    """Applies trained CNN to full grids.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param use_mask: Same.
    :param use_isotonic_regression: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    if num_times > 0:
        num_times = min([num_times, len(valid_times_unix_sec)])

        numpy.random.seed(RANDOM_SEED)
        numpy.random.shuffle(valid_times_unix_sec)
        valid_times_unix_sec = valid_times_unix_sec[:num_times]

    print 'Reading CNN from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print 'Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    if use_isotonic_regression:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name)

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_model_object_by_class = None

    num_times = len(valid_times_unix_sec)
    print SEPARATOR_STRING

    if use_mask:
        mask_matrix = model_metadata_dict[cnn.MASK_MATRIX_KEY]
    else:
        mask_matrix = None

    for i in range(num_times):
        this_class_probability_matrix, this_target_matrix = (
            cnn.apply_model_to_full_grid(
                model_object=model_object,
                top_predictor_dir_name=top_predictor_dir_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=model_metadata_dict[cnn.PRESSURE_LEVEL_KEY],
                predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
                normalization_type_string=model_metadata_dict[
                    cnn.NORMALIZATION_TYPE_KEY],
                dilation_distance_metres=model_metadata_dict[
                    cnn.DILATION_DISTANCE_KEY],
                isotonic_model_object_by_class=isotonic_model_object_by_class,
                mask_matrix=mask_matrix)
        )

        this_target_matrix[this_target_matrix == -1] = 0
        print MINOR_SEPARATOR_STRING

        this_output_file_name = ml_utils.find_gridded_prediction_file(
            directory_name=output_dir_name,
            first_target_time_unix_sec=valid_times_unix_sec[i],
            last_target_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing gridded probabilities and labels to: "{0:s}"...'.format(
            this_output_file_name)

        ml_utils.write_gridded_predictions(
            pickle_file_name=this_output_file_name,
            class_probability_matrix=this_class_probability_matrix,
            target_times_unix_sec=valid_times_unix_sec[[i]],
            model_file_name=model_file_name,
            used_isotonic_regression=use_isotonic_regression,
            target_matrix=this_target_matrix)

        if i != num_times - 1:
            print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        use_mask=bool(getattr(INPUT_ARG_OBJECT, USE_MASK_ARG_NAME)),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
