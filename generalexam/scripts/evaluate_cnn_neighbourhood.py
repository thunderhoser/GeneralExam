"""Runs neigh evaluation on gridded deterministic labels from CNN."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io
from generalexam.machine_learning import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SECONDS = 10800

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NEIGH_DISTANCE_ARG_NAME = 'neigh_distance_metres'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of input directory.  Files with gridded deterministic labels will be '
    'found therein by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Neighbourhood evaluation will be done for all'
    ' times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

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
    '--' + NEIGH_DISTANCE_ARG_NAME, type=float, required=True,
    help=NEIGH_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(prediction_dir_name, first_time_string, last_time_string,
         neigh_distance_metres, output_file_name):
    """Runs neigh evaluation on gridded deterministic labels from CNN.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param neigh_distance_metres: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    predicted_label_matrix = None
    actual_label_matrix = None
    prediction_file_names = []

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = prediction_io.find_file(
            directory_name=prediction_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            continue

        prediction_file_names.append(this_file_name)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_prediction_dict = prediction_io.read_file(this_file_name)

        this_predicted_label_matrix = this_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]
        this_actual_label_matrix = this_prediction_dict[
            prediction_io.TARGET_MATRIX_KEY]

        if predicted_label_matrix is None:
            predicted_label_matrix = this_predicted_label_matrix + 0.
            actual_label_matrix = this_actual_label_matrix + 0
        else:
            predicted_label_matrix = numpy.concatenate(
                (predicted_label_matrix, this_predicted_label_matrix),
                axis=0
            )
            actual_label_matrix = numpy.concatenate(
                (actual_label_matrix, this_actual_label_matrix), axis=0
            )

    print SEPARATOR_STRING

    (binary_ct_as_dict, prediction_oriented_ct_matrix, actual_oriented_ct_matrix
    ) = neigh_evaluation.make_contingency_tables(
        predicted_label_matrix=predicted_label_matrix,
        actual_label_matrix=actual_label_matrix,
        neigh_distance_metres=neigh_distance_metres)

    print SEPARATOR_STRING

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
        prediction_file_names=prediction_file_names,
        neigh_distance_metres=neigh_distance_metres,
        binary_ct_as_dict=binary_ct_as_dict,
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        neigh_distance_metres=getattr(
            INPUT_ARG_OBJECT, NEIGH_DISTANCE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
