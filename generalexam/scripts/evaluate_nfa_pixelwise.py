"""Runs pixelwise evaluation for NFA (numerical frontal analysis)."""

import os.path
import argparse
import numpy
from keras.utils import to_categorical
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import nfa
from generalexam.ge_utils import pixelwise_evaluation as pixelwise_eval
from generalexam.machine_learning import machine_learning_utils as ml_utils

RANDOM_SEED = 6695

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCORE_NAME_TO_FUNCTION = {
    'gerrity': pixelwise_eval.get_gerrity_score,
    'csi': pixelwise_eval.get_csi
}

NUM_CLASSES = 3
LARGE_INTEGER = int(1e12)
NARR_TIME_INTERVAL_SECONDS = 10800

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
USE_ENSEMBLE_ARG_NAME = 'use_ensembled_predictions'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_PIXELS_PER_TIME_ARG_NAME = 'num_pixels_per_time'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
SCORING_FUNCTION_ARG_NAME = 'scoring_function_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of directory with gridded NFA predictions.  Files therein will be '
    'found by `nfa.find_prediction_file` and read by '
    '`nfa.read_gridded_predictions` or `nfa.read_ensembled_predictions`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with labels (true fronts).  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.')

USE_ENSEMBLE_HELP_STRING = (
    'Boolean flag.  If 1, will evaluate ensembled probabilistic predictions '
    '(from many NFA models).  If 0, will evaluate deterministic predictions '
    'from a single NFA model.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will evaluate predictions in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of evaluation times.  This script will evaluate predictions at '
    '`{0:s}` times selected randomly from the period `{1:s}`...`{2:s}`.  To use'
    ' all times, leave this argument alone.'
).format(NUM_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_PIXELS_PER_TIME_HELP_STRING = (
    'Number of pixels per evaluation time.  Pixels will be sampled randomly '
    'from each evaluation time.')

DILATION_DISTANCE_HELP_STRING = 'Dilation distance for labels (true fronts).'

SCORING_FUNCTION_HELP_STRING = (
    'Scoring function used to choose best determinization threshold.  Must be '
    'in the following list:\n{0:s}'
).format(
    str(SCORE_NAME_TO_FUNCTION.keys())
)

OUTPUT_FILE_HELP_STRING = 'Path to output file (NetCDF).'

DEFAULT_NUM_PIXELS_PER_TIME = 1000
DEFAULT_DILATION_DISTANCE_METRES = 50000
TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ENSEMBLE_ARG_NAME, type=int, required=False, default=0,
    help=USE_ENSEMBLE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=LARGE_INTEGER,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PIXELS_PER_TIME_ARG_NAME, type=int, required=True,
    help=NUM_PIXELS_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=int, required=False,
    default=DEFAULT_DILATION_DISTANCE_METRES,
    help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SCORING_FUNCTION_ARG_NAME, type=str, required=False, default='csi',
    help=SCORING_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_prediction_dir_name, top_gridded_front_dir_name,
         use_ensembled_predictions, first_time_string, last_time_string,
         num_times, num_pixels_per_time, dilation_distance_metres,
         scoring_function_name, output_file_name):
    """Runs pixelwise evaluation for NFA (numerical frontal analysis).

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param top_gridded_front_dir_name: Same.
    :param use_ensembled_predictions: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_pixels_per_time: Same.
    :param dilation_distance_metres: Same.
    :param scoring_function_name: Same.
    :param output_file_name: Same.
    """

    scoring_function = SCORE_NAME_TO_FUNCTION[scoring_function_name]

    error_checking.assert_is_greater(num_times, 0)
    error_checking.assert_is_greater(num_pixels_per_time, 0)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    possible_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.seed(RANDOM_SEED)
    numpy.random.shuffle(possible_times_unix_sec)

    observed_labels = numpy.array([], dtype=int)
    class_probability_matrix = None

    num_times_read = 0
    unmasked_grid_rows = None
    unmasked_grid_columns = None

    this_random_seed = RANDOM_SEED + 0

    for this_time_unix_sec in possible_times_unix_sec:
        this_random_seed += 1

        if num_times_read == num_times:
            break

        this_prediction_file_name = nfa.find_prediction_file(
            directory_name=top_prediction_dir_name,
            first_valid_time_unix_sec=this_time_unix_sec,
            last_valid_time_unix_sec=this_time_unix_sec,
            ensembled=use_ensembled_predictions, raise_error_if_missing=False)

        if not os.path.isfile(this_prediction_file_name):
            continue

        num_times_read += 1
        print('Reading data from: "{0:s}"...'.format(this_prediction_file_name))

        if use_ensembled_predictions:
            this_ensemble_dict = nfa.read_ensembled_predictions(
                this_prediction_file_name)

            this_class_probability_matrix = this_ensemble_dict.pop(
                nfa.CLASS_PROBABILITIES_KEY
            )[0, ...]

            this_metadata_dict = this_ensemble_dict
        else:
            this_predicted_label_matrix, this_metadata_dict = (
                nfa.read_gridded_predictions(this_prediction_file_name)
            )

            this_class_probability_matrix = to_categorical(
                y=this_predicted_label_matrix[0, ...], num_classes=NUM_CLASSES)

        if unmasked_grid_rows is None:
            narr_mask_matrix = this_metadata_dict[nfa.NARR_MASK_KEY]
            unmasked_grid_rows, unmasked_grid_columns = numpy.where(
                narr_mask_matrix == 1)

        if num_pixels_per_time >= len(unmasked_grid_rows):
            these_grid_rows = unmasked_grid_rows + 0
            these_grid_columns = unmasked_grid_columns + 0
        else:
            these_indices = numpy.linspace(
                0, len(unmasked_grid_rows) - 1, num=len(unmasked_grid_rows),
                dtype=int)

            numpy.random.seed(this_random_seed)
            these_indices = numpy.random.choice(
                these_indices, size=num_pixels_per_time, replace=False)

            these_grid_rows = unmasked_grid_rows[these_indices]
            these_grid_columns = unmasked_grid_columns[these_indices]

        this_front_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_gridded_front_dir_name,
            valid_time_unix_sec=this_time_unix_sec)

        print('Reading data from: "{0:s}"...'.format(this_front_file_name))
        this_gridded_front_table = fronts_io.read_grid_from_file(
            this_front_file_name)

        this_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=this_gridded_front_table,
            num_rows_per_image=this_class_probability_matrix.shape[0],
            num_columns_per_image=this_class_probability_matrix.shape[1]
        )

        this_target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_target_matrix,
            dilation_distance_metres=dilation_distance_metres, verbose=False)

        this_class_probability_matrix = this_class_probability_matrix[
            these_grid_rows, these_grid_columns, :]

        if class_probability_matrix is None:
            class_probability_matrix = this_class_probability_matrix + 0.
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_class_probability_matrix),
                axis=0)

        these_observed_labels = this_target_matrix[0, ...][
            these_grid_rows, these_grid_columns]

        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels
        ))

    print(SEPARATOR_STRING)

    best_threshold, best_score, all_thresholds = (
        pixelwise_eval.find_best_determinization_threshold(
            class_probability_matrix=class_probability_matrix,
            observed_labels=observed_labels, scoring_function=scoring_function)
    )

    print((
        '\nBest determinization threshold = {0:.4f} ... corresponding "{1:s}" '
        'score = {2:.4f}'
    ).format(
        best_threshold, scoring_function_name, best_score
    ))
    print(SEPARATOR_STRING)

    climo_counts = numpy.array([
        numpy.sum(observed_labels == k)
        for k in range(pixelwise_eval.NUM_CLASSES)
    ], dtype=int)

    result_table_xarray = pixelwise_eval.run_evaluation(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        best_determinizn_threshold=best_threshold,
        all_determinizn_thresholds=all_thresholds,
        climo_counts=climo_counts, bootstrap_rep_index=0)

    print(SEPARATOR_STRING)
    print('Writing results to: "{0:s}"...'.format(output_file_name))

    pixelwise_eval.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        use_ensembled_predictions=bool(getattr(
            INPUT_ARG_OBJECT, USE_ENSEMBLE_ARG_NAME
        )),
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        num_pixels_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_PIXELS_PER_TIME_ARG_NAME
        ),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME
        ),
        scoring_function_name=getattr(
            INPUT_ARG_OBJECT, SCORING_FUNCTION_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
