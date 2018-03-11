"""Trains isotonic regression for probability calibration of traditional CNN.

A "traditional CNN" is one for which the output (prediction) is not spatially
explicit.  The opposite is a fully convolutional net (see fcn.py).
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
NUM_TIMES_ARG_NAME = 'num_training_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NARR_DIR_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to model file, containing a trained CNN.  This file should be '
    'readable by `traditional_cnn.read_keras_model`.')
TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training examples will be randomly drawn from'
    ' the time period `{0:s}`...`{1:s}`.  One training example consists of K '
    'predicted (uncalibrated) class probabilities and the true class label '
    '(K = number of classes).').format(FIRST_TRAINING_TIME_ARG_NAME,
                                       LAST_TRAINING_TIME_ARG_NAME)
NUM_TIMES_HELP_STRING = (
    'Number of training times to be drawn randomly from the period '
    '`{0:s}`...`{1:s}`.').format(FIRST_TRAINING_TIME_ARG_NAME,
                                 LAST_TRAINING_TIME_ARG_NAME)
NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of training examples to be randomly drawn from each time step.  '
    'Each training example corresponds to one time and one NARR grid point.')
NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data (one file for each variable, '
    'pressure level, and time step).')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (one per file, indicating '
    'which NARR grid cells are intersected by a front).')

DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_FIRST_TRAINING_TIME_STRING = '2015010100'
DEFAULT_LAST_TRAINING_TIME_STRING = '2015122421'
DEFAULT_NUM_TRAINING_TIMES = 1000
DEFAULT_NUM_EXAMPLES_PER_TIME = 1000

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
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_FRONTAL_GRID_DIR_NAME, help=FRONTAL_GRID_DIR_HELP_STRING)


def _train_isotonic_regression(
        model_file_name, first_training_time_string, last_training_time_string,
        num_training_times, num_examples_per_time, top_narr_directory_name,
        top_frontal_grid_dir_name):
    """Trains iso regression for probability calibration of traditional CNN.

    :param model_file_name: Path to model file, containing a trained CNN.  This
        file should be readable by `traditional_cnn.read_keras_model`.
    :param first_training_time_string: Time (format "yyyymmddHH").  Training
        examples will be randomly drawn from the time period
        `first_training_time_string`...`last_training_time_string`.  One
        training example consists of K predicted (uncalibrated) class
        probabilities and the true class label (K = number of classes).
    :param last_training_time_string: See above.
    :param num_training_times: Number of training times to be drawn randomly
        from the period `first_training_time_string`...
        `last_training_time_string`.
    :param num_examples_per_time: Number of training examples to be randomly
        drawn from each time step.  Each training example corresponds to one
        time and one NARR grid point.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one per file, indicating which NARR grid cells are intersected by
        a front).
    """

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, INPUT_TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        model_directory_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metadata_file_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metadata_file_name)

    print SEPARATOR_STRING
    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])

    orig_class_probability_matrix, observed_labels = (
        eval_utils.downsized_examples_to_eval_pairs(
            model_object=model_object,
            first_target_time_unix_sec=first_training_time_unix_sec,
            last_target_time_unix_sec=last_training_time_unix_sec,
            num_target_times_to_sample=num_training_times,
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
                traditional_cnn.NUM_LEAD_TIME_STEPS_KEY]))
    print SEPARATOR_STRING

    isotonic_model_object_by_class = (
        isotonic_regression.train_model_for_each_class(
            orig_class_probability_matrix=orig_class_probability_matrix,
            observed_labels=observed_labels))

    isotonic_regression_file_name = (
        '{0:s}/isotonic_regression_models.p'.format(model_directory_name))

    print 'Writing trained models to: "{0:s}"...'.format(
        isotonic_regression_file_name)
    isotonic_regression.write_model_for_each_class(
        model_object_by_class=isotonic_model_object_by_class,
        pickle_file_name=isotonic_regression_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    MODEL_FILE_NAME = getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    FIRST_TRAINING_TIME_STRING = getattr(
        INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME)
    LAST_TRAINING_TIME_STRING = getattr(
        INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME)

    NUM_TRAINING_TIMES = getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME)
    NUM_EXAMPLES_PER_TIME = getattr(
        INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME)

    TOP_NARR_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME)
    TOP_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)

    _train_isotonic_regression(
        model_file_name=MODEL_FILE_NAME,
        first_training_time_string=FIRST_TRAINING_TIME_STRING,
        last_training_time_string=LAST_TRAINING_TIME_STRING,
        num_training_times=NUM_TRAINING_TIMES,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME,
        top_narr_directory_name=TOP_NARR_DIRECTORY_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME)
