"""Runs neighbourhood evaluation on probability grids created by CNN."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
THRESHOLD_ARG_NAME = 'binarization_threshold'
NEIGH_DISTANCE_ARG_NAME = 'neigh_distance_metres'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of directory with gridded probabilities.  Files therein will be found'
    ' by `machine_learning_utils.find_gridded_prediction_file` and read by '
    '`machine_learning_utils.read_gridded_predictions`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Neighbourhood evaluation will be run on all '
    'times from `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

THRESHOLD_HELP_STRING = (
    'Binarization threshold.  For each case (i.e., each grid cell at each time '
    'step), if NF probability >= `{0:s}`, the deterministic label will be NF.  '
    'Otherwise, the deterministic label will be the max of WF and CF '
    'probabilities.'
).format(THRESHOLD_ARG_NAME)

NEIGH_DISTANCE_HELP_STRING = 'Neighbourhood distance for evaluation.'

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `neigh_evaluation.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_ARG_NAME, type=float, required=True,
    help=THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEIGH_DISTANCE_ARG_NAME, type=float, required=True,
    help=NEIGH_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_prediction_dir_name, first_time_string, last_time_string,
         binarization_threshold, neigh_distance_metres, output_file_name):
    """Runs neighbourhood evaluation on probability grids created by CNN.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param binarization_threshold: Same.
    :param neigh_distance_metres: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    class_probability_matrix = None
    actual_label_matrix = None
    valid_times_unix_sec = []

    for i in range(len(all_times_unix_sec)):
        this_file_name = ml_utils.find_gridded_prediction_file(
            directory_name=top_prediction_dir_name,
            first_target_time_unix_sec=all_times_unix_sec[i],
            last_target_time_unix_sec=all_times_unix_sec[i],
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_file_name):
            continue

        valid_times_unix_sec.append(all_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_dict = ml_utils.read_gridded_predictions(this_file_name)

        this_class_probability_matrix = this_dict[
            ml_utils.PROBABILITY_MATRIX_KEY]
        this_actual_label_matrix = this_dict[
            ml_utils.TARGET_MATRIX_KEY]

        if class_probability_matrix is None:
            class_probability_matrix = this_class_probability_matrix + 0.
            actual_label_matrix = this_actual_label_matrix + 0
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_class_probability_matrix),
                axis=0
            )
            actual_label_matrix = numpy.concatenate(
                (actual_label_matrix, this_actual_label_matrix), axis=0
            )

    valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)

    print SEPARATOR_STRING
    print (
        'Determinizing probabilities with binarization threshold = {0:.4f}...'
    ).format(binarization_threshold)

    predicted_label_matrix = neigh_evaluation.determinize_predictions(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=binarization_threshold)

    print 'Creating 2-class and 3-class contingency tables...'
    (binary_ct_as_dict, prediction_oriented_ct_matrix,
     actual_oriented_ct_matrix
    ) = neigh_evaluation.make_contingency_tables(
        predicted_label_matrix=predicted_label_matrix,
        actual_label_matrix=actual_label_matrix,
        neigh_distance_metres=neigh_distance_metres)

    print 'Binary (2-class) contingency table:\n'
    print binary_ct_as_dict

    print '\nPrediction-oriented 3-class contingency table:\n'
    print prediction_oriented_ct_matrix

    print '\nActual-oriented 3-class contingency table:\n'
    print actual_oriented_ct_matrix
    print '\n'

    binary_pod = neigh_evaluation.get_binary_pod(binary_ct_as_dict)
    binary_success_ratio = neigh_evaluation.get_binary_success_ratio(
        binary_ct_as_dict)
    binary_csi = neigh_evaluation.get_binary_csi(binary_ct_as_dict)
    binary_frequency_bias = neigh_evaluation.get_binary_frequency_bias(
        binary_ct_as_dict)

    print 'Binary POD = {0:.4f}'.format(binary_pod)
    print 'Binary success ratio = {0:.4f}'.format(binary_success_ratio)
    print 'Binary CSI = {0:.4f}'.format(binary_csi)
    print 'Binary frequency bias = {0:.4f}\n'.format(binary_frequency_bias)

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    neigh_evaluation.write_results(
        pickle_file_name=output_file_name,
        predicted_label_matrix=predicted_label_matrix,
        actual_label_matrix=actual_label_matrix,
        valid_times_unix_sec=valid_times_unix_sec,
        neigh_distance_metres=neigh_distance_metres,
        binary_ct_as_dict=binary_ct_as_dict,
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        binarization_threshold=getattr(INPUT_ARG_OBJECT, THRESHOLD_ARG_NAME),
        neigh_distance_metres=getattr(
            INPUT_ARG_OBJECT, NEIGH_DISTANCE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
