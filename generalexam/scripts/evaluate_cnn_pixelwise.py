"""Runs pixelwise evaluation for traditional (patch-classification) CNN."""

import random
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import traditional_cnn
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
FIRST_EVAL_TIME_ARG_NAME = 'first_eval_time_string'
LAST_EVAL_TIME_ARG_NAME = 'last_eval_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
USE_ISOTONIC_ARG_NAME = 'use_isotonic_regression'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to HDF5 file, containing the trained CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

EVAL_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Evaluation times will be randomly drawn from '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of evaluation times (to be sampled randomly from '
    '`{0:s}`...`{1:s}`).'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)

NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples for each target time.  Each "example" consists of a '
    'downsized image and the label (no front, warm front, or cold front) at the'
    ' center pixel.  Examples will be drawn randomly from NARR grid cells.')

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance.  Target images will be dilated, which increases the '
    'number of "frontal" pixels and accounts for uncertainty in frontal '
    'placement.  To use the same dilation distance used for training, leave '
    'this alone.')

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, isotonic regression will be used to calibrate '
    'probabilities from the CNN, in which case the evaluation will be based on '
    'calibrated probabilities.  If 0, the evaluation will be based on raw '
    'probabilities.')

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level NARR directory (predictors will be read from here).  '
    'Files therein will be found by `processed_narr_io.find_file_for_one_time` '
    'and read by `processed_narr_io.read_fields_from_file`.')

FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (target images will be read'
    ' from here).  Files therein will be found by '
    '`fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_narr_grids_from_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation results will be saved here.')

TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONTAL_GRID_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

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
    '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONTAL_GRID_DIR_NAME_DEFAULT,
    help=FRONTAL_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, first_eval_time_string, last_eval_time_string,
         num_times, num_examples_per_time, dilation_distance_metres,
         use_isotonic_regression, top_narr_directory_name,
         top_frontal_grid_dir_name, output_dir_name):
    """Evaluates CNN trained by patch classification.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param first_eval_time_string: Same.
    :param last_eval_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param dilation_distance_metres: Same.
    :param use_isotonic_regression: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param output_dir_name: Same.
    """

    first_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        first_eval_time_string, INPUT_TIME_FORMAT)
    last_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        last_eval_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metafile_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metafile_name)

    if dilation_distance_metres < 0:
        dilation_distance_metres = model_metadata_dict[
            traditional_cnn.DILATION_DISTANCE_FOR_TARGET_KEY] + 0.

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

    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])
    print SEPARATOR_STRING

    class_probability_matrix, observed_labels = (
        eval_utils.downsized_examples_to_eval_pairs(
            model_object=model_object,
            first_target_time_unix_sec=first_eval_time_unix_sec,
            last_target_time_unix_sec=last_eval_time_unix_sec,
            num_target_times_to_sample=num_times,
            num_examples_per_time=num_examples_per_time,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=model_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            pressure_level_mb=model_metadata_dict[
                traditional_cnn.PRESSURE_LEVEL_KEY],
            dilation_distance_metres=dilation_distance_metres,
            num_rows_in_half_grid=model_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_columns_in_half_grid=model_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            num_classes=num_classes,
            predictor_time_step_offsets=model_metadata_dict[
                traditional_cnn.PREDICTOR_TIME_STEP_OFFSETS_KEY],
            num_lead_time_steps=model_metadata_dict[
                traditional_cnn.NUM_LEAD_TIME_STEPS_KEY],
            isotonic_model_object_by_class=isotonic_model_object_by_class,
            narr_mask_matrix=model_metadata_dict[
                traditional_cnn.NARR_MASK_MATRIX_KEY]
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
        first_eval_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_EVAL_TIME_ARG_NAME),
        last_eval_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_EVAL_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME)),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
