"""Evaluates frontal objects (skeleton lines) created by a CNN.

In this case, evaluation is done in an object-based setting.  If you want to do
pixelwise evaluation, use eval_traditional_cnn_pixelwise.py.
"""

import argparse
from generalexam.evaluation import object_based_evaluation as object_eval

METRES_TO_KM = 1e-3

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_metres'
EVALUATION_FILE_ARG_NAME = 'output_eval_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`object_based_evaluation.read_predictions_and_obs`.')

MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (or neighbourhood distance).  If actual front f_A and '
    'predicted front f_P, both at time t, are of the same type and within '
    '`{0:s}` of each other, they are considered "matching".'
).format(MATCHING_DISTANCE_ARG_NAME)

EVALUATION_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`object_based_evaluation.write_evaluation_results`.')

DEFAULT_MATCHING_DISTANCE_METRES = 1e5

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=False,
    default=DEFAULT_MATCHING_DISTANCE_METRES,
    help=MATCHING_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_FILE_ARG_NAME, type=str, required=True,
    help=EVALUATION_FILE_HELP_STRING)


def _run(input_prediction_file_name, matching_distance_metres,
         output_eval_file_name):
    """Evaluates frontal objects (skeleton lines) created by a CNN.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param matching_distance_metres: Same.
    :param output_eval_file_name: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(input_prediction_file_name)
    (predicted_region_table, actual_polyline_table
    ) = object_eval.read_predictions_and_obs(input_prediction_file_name)

    binary_contingency_table_as_dict = object_eval.get_binary_contingency_table(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        neigh_distance_metres=matching_distance_metres)

    print (
        'Binary contingency table (matching distance = {0:f} km):\n{1:s}\n'
    ).format(METRES_TO_KM * matching_distance_metres,
             binary_contingency_table_as_dict)

    binary_pod = object_eval.get_binary_pod(binary_contingency_table_as_dict)
    binary_success_ratio = object_eval.get_binary_success_ratio(
        binary_contingency_table_as_dict)
    binary_csi = object_eval.get_binary_csi(binary_contingency_table_as_dict)
    binary_frequency_bias = object_eval.get_binary_frequency_bias(
        binary_contingency_table_as_dict)

    print (
        'Binary POD = {0:.4f} ... success ratio = {1:.4f} ... CSI = {2:.4f} ...'
        ' frequency bias = {3:.4f}\n'
    ).format(binary_pod, binary_success_ratio, binary_csi,
             binary_frequency_bias)

    row_normalized_ct_as_matrix = (
        object_eval.get_row_normalized_contingency_table(
            predicted_region_table=predicted_region_table,
            actual_polyline_table=actual_polyline_table,
            neigh_distance_metres=matching_distance_metres)
    )

    print 'Row-normalized contingency table:\n{0:s}\n'.format(
        row_normalized_ct_as_matrix)

    column_normalized_ct_as_matrix = (
        object_eval.get_column_normalized_contingency_table(
            predicted_region_table=predicted_region_table,
            actual_polyline_table=actual_polyline_table,
            neigh_distance_metres=matching_distance_metres)
    )

    print 'Column-normalized contingency table:\n{0:s}\n'.format(
        column_normalized_ct_as_matrix)

    print 'Writing results to: "{0:s}"...'.format(output_eval_file_name)
    object_eval.write_evaluation_results(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        neigh_distance_metres=matching_distance_metres,
        binary_contingency_table_as_dict=binary_contingency_table_as_dict,
        binary_pod=binary_pod, binary_success_ratio=binary_success_ratio,
        binary_csi=binary_csi, binary_frequency_bias=binary_frequency_bias,
        row_normalized_ct_as_matrix=row_normalized_ct_as_matrix,
        column_normalized_ct_as_matrix=column_normalized_ct_as_matrix,
        pickle_file_name=output_eval_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME),
        matching_distance_metres=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME),
        output_eval_file_name=getattr(
            INPUT_ARG_OBJECT, EVALUATION_FILE_ARG_NAME))
