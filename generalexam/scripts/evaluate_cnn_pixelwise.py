"""Evaluates probability grids created by a CNN.

In this case, evaluation is done in an pixelwise setting.  If you want to do
object-based evaluation, use evaluate_cnn_object_based.py.
"""

import random
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import evaluation_utils as eval_utils
from generalexam.scripts import model_evaluation_helper as model_eval_helper

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
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
    'Time (format "yyyymmddHH").  This script will evaluate the CNN for times '
    'randomly drawn from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of times to draw randomly from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples (pixels) to use at each time step.  These will be drawn'
    ' randomly from the grid.')

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for gridded warm-front and cold-front labels.  To use '
    'the same dilation distance used for training, leave this argument alone.')

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, will use isotonic regression to calibrate CNN '
    'probabilities.  If 0, will use raw CNN probabilities with no calibration.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be saved here.')

TOP_PREDICTOR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'
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
    '--' + NUM_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=-1, help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ISOTONIC_ARG_NAME, type=int, required=False, default=0,
    help=USE_ISOTONIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, top_predictor_dir_name, top_gridded_front_dir_name,
         first_time_string, last_time_string, num_times, num_examples_per_time,
         dilation_distance_metres, use_isotonic_regression, output_dir_name):
    """Evaluates probability grids created by a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param dilation_distance_metres: Same.
    :param use_isotonic_regression: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True)

    print 'Reading model metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    if dilation_distance_metres < 0:
        dilation_distance_metres = (
            model_metadata_dict[cnn.DILATION_DISTANCE_KEY] + 0.
        )

    if use_isotonic_regression:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name, raise_error_if_missing=True)

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_model_object_by_class = None

    print SEPARATOR_STRING
    print model_metadata_dict[cnn.MASK_MATRIX_KEY]

    class_probability_matrix, observed_labels = (
        eval_utils.create_eval_pairs_for_cnn(
            model_object=model_object,
            top_predictor_dir_name=top_predictor_dir_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec, num_times=num_times,
            num_examples_per_time=num_examples_per_time,
            pressure_level_mb=model_metadata_dict[cnn.PRESSURE_LEVEL_KEY],
            predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            dilation_distance_metres=dilation_distance_metres,
            isotonic_model_object_by_class=isotonic_model_object_by_class,
            mask_matrix=model_metadata_dict[cnn.MASK_MATRIX_KEY]
        )
    )

    print SEPARATOR_STRING

    model_eval_helper.run_evaluation(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, output_dir_name=output_dir_name)


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
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
