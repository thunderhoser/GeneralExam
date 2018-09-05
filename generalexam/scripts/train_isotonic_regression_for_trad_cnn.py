"""Trains isotonic regression to bias-correct a traditional CNN.

A "traditional CNN" is one that does patch classification.
"""

import os.path
import argparse
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import evaluation_utils as eval_utils

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to HDF5 file, containing the trained CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training times will be randomly drawn from '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of training times (to be sampled randomly from `{0:s}`...`{1:s}`).'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples for each target time.  Each "example" consists of a '
    'downsized image and the label (no front, warm front, or cold front) at the'
    ' center pixel.  Examples will be drawn randomly from NARR grid cells.')

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level NARR directory (CNN predictors will be read from here). '
    ' Files therein will be found by `processed_narr_io.find_file_for_one_time`'
    ' and read by `processed_narr_io.read_fields_from_file`.')

FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (target images will be read'
    ' from here).  Files therein will be found by '
    '`fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_narr_grids_from_file`.')

DEFAULT_FIRST_TRAINING_TIME_STRING = '2015010100'
DEFAULT_LAST_TRAINING_TIME_STRING = '2015122421'
DEFAULT_NUM_TRAINING_TIMES = 1000
DEFAULT_NUM_EXAMPLES_PER_TIME = 1000
TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONTAL_GRID_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TRAINING_TIMES, help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_TIME,
    help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONTAL_GRID_DIR_NAME_DEFAULT,
    help=FRONTAL_GRID_DIR_HELP_STRING)


def _run(model_file_name, first_training_time_string, last_training_time_string,
         num_times, num_examples_per_time, top_narr_directory_name,
         top_frontal_grid_dir_name):
    """Trains isotonic regression to bias-correct a traditional CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    """

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, INPUT_TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_dir_name = os.path.split(model_file_name)[0]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(model_dir_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metadata_file_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metadata_file_name)

    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])

    print SEPARATOR_STRING
    (orig_class_probability_matrix, observed_labels
    ) = eval_utils.downsized_examples_to_eval_pairs(
        model_object=model_object,
        first_target_time_unix_sec=first_training_time_unix_sec,
        last_target_time_unix_sec=last_training_time_unix_sec,
        num_target_times_to_sample=num_times,
        num_examples_per_time=num_examples_per_time,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=model_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        pressure_level_mb=model_metadata_dict[
            traditional_cnn.PRESSURE_LEVEL_KEY],
        dilation_distance_for_target_metres=model_metadata_dict[
            traditional_cnn.DILATION_DISTANCE_FOR_TARGET_KEY],
        num_rows_in_half_grid=model_metadata_dict[
            traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
        num_columns_in_half_grid=model_metadata_dict[
            traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
        num_classes=num_classes,
        predictor_time_step_offsets=model_metadata_dict[
            traditional_cnn.PREDICTOR_TIME_STEP_OFFSETS_KEY],
        num_lead_time_steps=model_metadata_dict[
            traditional_cnn.NUM_LEAD_TIME_STEPS_KEY],
        narr_mask_matrix=model_metadata_dict[
            traditional_cnn.NARR_MASK_MATRIX_KEY])
    print SEPARATOR_STRING

    isotonic_model_object_by_class = (
        isotonic_regression.train_model_for_each_class(
            orig_class_probability_matrix=orig_class_probability_matrix,
            observed_labels=observed_labels)
    )

    isotonic_regression_file_name = (
        '{0:s}/isotonic_regression_models.p'
    ).format(model_dir_name)

    print '\nWriting trained models to: "{0:s}"...'.format(
        isotonic_regression_file_name)
    isotonic_regression.write_model_for_each_class(
        model_object_by_class=isotonic_model_object_by_class,
        pickle_file_name=isotonic_regression_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)
    )
