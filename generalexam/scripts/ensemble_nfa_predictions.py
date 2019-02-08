"""Ensembles predictions from two or more NFA models.

NFA = numerical frontal analysis
"""

import os.path
import argparse
import numpy
from keras.utils import to_categorical
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import nfa

TOLERANCE = 1e-6
INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_CLASSES = 3
NARR_TIME_INTERVAL_SECONDS = 10800

INPUT_DIRS_ARG_NAME = 'prediction_dir_name_by_model'
WEIGHTS_ARG_NAME = 'model_weights'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIRS_HELP_STRING = (
    'List of input directories (one for each NFA model).  Files in each '
    'directory will be found by `nfa.find_prediction_file` and read by '
    '`nfa.read_gridded_predictions`.')

WEIGHTS_HELP_STRING = (
    'List of weights (one for each model).  These must sum to 1.0.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will ensemble predictions for the'
    ' period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Ensemble predictions will be written here by '
    '`nfa.write_gridded_predictions`, to file locations determined by '
    '`nfa.find_prediction_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WEIGHTS_ARG_NAME, type=float, nargs='+', required=True,
    help=WEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(prediction_dir_name_by_model, model_weights, first_time_string,
         last_time_string, output_prediction_dir_name):
    """Ensembles predictions from two or more NFA models.

    This is effectively the main method.

    :param prediction_dir_name_by_model: See documentation at top of file.
    :param model_weights: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_prediction_dir_name: Same.
    """

    error_checking.assert_is_geq_numpy_array(model_weights, 0.)
    error_checking.assert_is_leq_numpy_array(model_weights, 1.)
    error_checking.assert_is_geq(numpy.sum(model_weights), 1. - TOLERANCE)
    error_checking.assert_is_leq(numpy.sum(model_weights), 1. + TOLERANCE)

    num_models = len(model_weights)
    error_checking.assert_is_geq(num_models, 2)

    these_expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(prediction_dir_name_by_model),
        exact_dimensions=these_expected_dim)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    possible_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    narr_mask_matrix = None

    for this_time_unix_sec in possible_times_unix_sec:
        these_prediction_file_names = [''] * num_models

        for j in range(num_models):
            these_prediction_file_names[j] = nfa.find_prediction_file(
                directory_name=prediction_dir_name_by_model[0],
                first_valid_time_unix_sec=this_time_unix_sec,
                last_valid_time_unix_sec=this_time_unix_sec,
                ensembled=False, raise_error_if_missing=j > 0)

            if not os.path.isfile(these_prediction_file_names[j]):
                break

        if these_prediction_file_names[-1] == '':
            continue

        this_class_probability_matrix = None

        for j in range(num_models):
            print 'Reading data from: "{0:s}"...'.format(
                these_prediction_file_names[j])

            this_predicted_label_matrix, this_metadata_dict = (
                nfa.read_gridded_predictions(these_prediction_file_names[j])
            )

            if narr_mask_matrix is None:
                narr_mask_matrix = this_metadata_dict[nfa.NARR_MASK_KEY] + 0

            new_class_probability_matrix = to_categorical(
                y=this_predicted_label_matrix, num_classes=NUM_CLASSES)

            new_class_probability_matrix = (
                model_weights[j] * new_class_probability_matrix.astype(float)
            )

            if this_class_probability_matrix is None:
                this_class_probability_matrix = new_class_probability_matrix + 0
            else:
                this_class_probability_matrix = (
                    this_class_probability_matrix + new_class_probability_matrix
                )

        this_output_file_name = nfa.find_prediction_file(
            directory_name=output_prediction_dir_name,
            first_valid_time_unix_sec=this_time_unix_sec,
            last_valid_time_unix_sec=this_time_unix_sec, ensembled=True,
            raise_error_if_missing=False)

        print 'Writing ensembled predictions to: "{0:s}"...\n'.format(
            this_output_file_name)

        nfa.write_ensembled_predictions(
            pickle_file_name=this_output_file_name,
            class_probability_matrix=this_class_probability_matrix,
            valid_times_unix_sec=numpy.array([this_time_unix_sec], dtype=int),
            narr_mask_matrix=narr_mask_matrix,
            prediction_dir_name_by_model=prediction_dir_name_by_model,
            model_weights=model_weights)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name_by_model=getattr(
            INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME),
        model_weights=numpy.array(
            getattr(INPUT_ARG_OBJECT, WEIGHTS_ARG_NAME), dtype=float),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
