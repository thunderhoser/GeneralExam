"""Applies traditional CNN to full grids.

A "traditional CNN" is one that does patch classification.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import machine_learning_utils as ml_utils

# TODO(thunderhoser): Allow mask to be used.

NARR_TIME_INTERVAL_SEC = 10800
INPUT_TIME_FORMAT = '%Y%m%d%H'

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RANDOMIZE_TIMES_ARG_NAME = 'randomize_times'
NUM_TIMES_ARG_NAME = 'num_times'
USE_ISOTONIC_ARG_NAME = 'use_isotonic_regression'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to HDF5 file, containing the trained CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Target times will be randomly drawn from the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

RANDOMIZE_TIMES_HELP_STRING = (
    'Boolean flag.  If 1, target times will be sampled randomly from '
    '`{0:s}`...`{1:s}`.  If 0, the first `{2:s}` times from `{0:s}`...`{1:s}` '
    'will be sampled.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME, NUM_TIMES_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    '[used iff {0:s} = 1] Number of target times (to be sampled from '
    '`{1:s}`...`{2:s}`).'
).format(RANDOMIZE_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, isotonic regression will be used to calibrate '
    'probabilities from the CNN, in which case each prediction grid will '
    'contain calibrated probabilities.  If 0, each prediction grid will contain'
    ' raw probabilities.')

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
    'Name of output directory.  For each time step, gridded predictions will be'
    ' written here by `machine_learning_utils.write_gridded_predictions`, to a '
    'location determined by `machine_learning_utils.'
    'find_gridded_prediction_file`.')

TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONTAL_GRID_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RANDOMIZE_TIMES_ARG_NAME, type=int, required=False, default=1,
    help=RANDOMIZE_TIMES_HELP_STRING)

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


def _run(model_file_name, first_time_string, last_time_string, randomize_times,
         num_target_times, use_isotonic_regression, top_narr_directory_name,
         top_frontal_grid_dir_name, output_dir_name):
    """Applies traditional CNN to full grids.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param randomize_times: Same.
    :param num_target_times: Same.
    :param use_isotonic_regression: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SEC, include_endpoint=True)

    if randomize_times:
        error_checking.assert_is_leq(
            num_target_times, len(target_times_unix_sec))
        numpy.random.shuffle(target_times_unix_sec)
        target_times_unix_sec = target_times_unix_sec[:num_target_times]

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metafile_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metafile_name)

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

    if model_metadata_dict[traditional_cnn.NUM_LEAD_TIME_STEPS_KEY] is None:
        num_dimensions = 3
    else:
        num_dimensions = 4

    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])
    num_target_times = len(target_times_unix_sec)
    print SEPARATOR_STRING

    for i in range(num_target_times):
        if num_dimensions == 3:
            (this_class_probability_matrix, this_target_matrix
            ) = traditional_cnn.apply_model_to_3d_example(
                model_object=model_object,
                target_time_unix_sec=target_times_unix_sec[i],
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
                isotonic_model_object_by_class=isotonic_model_object_by_class,
                narr_mask_matrix=model_metadata_dict[
                    traditional_cnn.NARR_MASK_MATRIX_KEY])
        else:
            (this_class_probability_matrix, this_target_matrix
            ) = traditional_cnn.apply_model_to_4d_example(
                model_object=model_object,
                target_time_unix_sec=target_times_unix_sec[i],
                predictor_time_step_offsets=model_metadata_dict[
                    traditional_cnn.PREDICTOR_TIME_STEP_OFFSETS_KEY],
                num_lead_time_steps=model_metadata_dict[
                    traditional_cnn.NUM_LEAD_TIME_STEPS_KEY],
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
                isotonic_model_object_by_class=isotonic_model_object_by_class,
                narr_mask_matrix=model_metadata_dict[
                    traditional_cnn.NARR_MASK_MATRIX_KEY])

        print MINOR_SEPARATOR_STRING

        this_prediction_file_name = ml_utils.find_gridded_prediction_file(
            directory_name=output_dir_name,
            first_target_time_unix_sec=target_times_unix_sec[i],
            last_target_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing gridded predictions to file: "{0:s}"...'.format(
            this_prediction_file_name)

        ml_utils.write_gridded_predictions(
            pickle_file_name=this_prediction_file_name,
            class_probability_matrix=this_class_probability_matrix,
            target_times_unix_sec=target_times_unix_sec[[i]],
            model_file_name=model_file_name,
            used_isotonic_regression=use_isotonic_regression,
            target_matrix=this_target_matrix)

        if i != num_target_times - 1:
            print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        randomize_times=bool(getattr(
            INPUT_ARG_OBJECT, RANDOMIZE_TIMES_ARG_NAME)),
        num_target_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME)),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
