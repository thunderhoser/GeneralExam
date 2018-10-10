"""Evaluates frontal objects (skeleton lines) created by a CNN.

In this case, evaluation is done in an object-based setting.  If you want to do
pixelwise evaluation, use eval_traditional_cnn_pixelwise.py.
"""

import random
import os.path
import argparse
import numpy
import pandas
from generalexam.ge_utils import front_utils
from generalexam.evaluation import object_based_evaluation as object_eval

random.seed(6695)
numpy.random.seed(6695)

METRES_TO_KM = 1e-3
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_metres'
NUM_REPLICATES_ARG_NAME = 'num_bootstrap_replicates'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
EVALUATION_FILE_ARG_NAME = 'output_eval_file_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`object_based_evaluation.read_predictions_and_obs`.')

MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (or neighbourhood distance).  If actual front f_A and '
    'predicted front f_P, both at time t, are of the same type and within '
    '`{0:s}` of each other, they are considered "matching".'
).format(MATCHING_DISTANCE_ARG_NAME)

NUM_REPLICATES_HELP_STRING = (
    'Number of replicates for bootstrapping.  If -1, bootstrapping will not be '
    'done.')

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for bootstrapping.  If p = confidence level, will form '
    'the p*100% confidence interval for all performance metrics and contingency'
    '-table elements.')

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
    '--' + NUM_REPLICATES_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_REPLICATES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.99,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_FILE_ARG_NAME, type=str, required=True,
    help=EVALUATION_FILE_HELP_STRING)


def _run(input_prediction_file_name, matching_distance_metres,
         num_bootstrap_replicates, confidence_level, output_eval_file_name):
    """Evaluates frontal objects (skeleton lines) created by a CNN.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param matching_distance_metres: Same.
    :param num_bootstrap_replicates: Same.
    :param confidence_level: Same.
    :param output_eval_file_name: Same.
    """

    if num_bootstrap_replicates < 2:
        num_bootstrap_replicates = 0

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

    if num_bootstrap_replicates == 0:
        return

    unique_times_unix_sec = numpy.unique(
        numpy.concatenate((
            predicted_region_table[front_utils.TIME_COLUMN].values,
            actual_polyline_table[front_utils.TIME_COLUMN].values))
    )

    num_unique_times = len(unique_times_unix_sec)
    print (
        '\nNumber of unique times between predicted and actual fronts = {0:d}'
    ).format(num_unique_times)
    print SEPARATOR_STRING

    binary_pod_values = numpy.full(num_bootstrap_replicates, numpy.nan)
    binary_success_ratios = numpy.full(num_bootstrap_replicates, numpy.nan)
    binary_csi_values = numpy.full(num_bootstrap_replicates, numpy.nan)
    binary_frequency_biases = numpy.full(num_bootstrap_replicates, numpy.nan)

    for i in range(num_bootstrap_replicates):
        these_times_unix_sec = numpy.random.choice(
            unique_times_unix_sec, size=len(unique_times_unix_sec),
            replace=True)

        these_predicted_region_tables = []
        these_actual_polyline_tables = []

        for this_time_unix_sec in these_times_unix_sec:
            these_predicted_region_tables.append(
                predicted_region_table.loc[
                    predicted_region_table[front_utils.TIME_COLUMN] ==
                    this_time_unix_sec
                    ]
            )

            these_actual_polyline_tables.append(
                actual_polyline_table.loc[
                    actual_polyline_table[front_utils.TIME_COLUMN] ==
                    this_time_unix_sec
                    ]
            )

            if len(these_predicted_region_tables) == 1:
                continue

            these_predicted_region_tables[-1] = (
                these_predicted_region_tables[-1].align(
                    these_predicted_region_tables[0], axis=1
                )[0]
            )
            these_actual_polyline_tables[-1] = (
                these_actual_polyline_tables[-1].align(
                    these_actual_polyline_tables[0], axis=1
                )[0]
            )

        this_predicted_region_table = pandas.concat(
            these_predicted_region_tables, axis=0, ignore_index=True)
        this_actual_polyline_table = pandas.concat(
            these_actual_polyline_tables, axis=0, ignore_index=True)

        print (
            'Number of unique predicted fronts = {0:d} ... number of '
            'predicted fronts in bootstrap replicate {1:d} of {2:d} = {3:d}'
        ).format(len(predicted_region_table.index),
                 i + 1, num_bootstrap_replicates,
                 len(this_predicted_region_table.index))

        print (
            'Number of unique actual fronts = {0:d} ... number of actual '
            'fronts in bootstrap replicate {1:d} of {2:d} = {3:d}'
        ).format(len(actual_polyline_table.index),
                 i + 1, num_bootstrap_replicates,
                 len(this_actual_polyline_table.index))

        this_binary_ct_as_dict = object_eval.get_binary_contingency_table(
            predicted_region_table=this_predicted_region_table,
            actual_polyline_table=this_actual_polyline_table,
            neigh_distance_metres=matching_distance_metres)

        binary_pod_values[i] = object_eval.get_binary_pod(
            this_binary_ct_as_dict)
        binary_success_ratios[i] = object_eval.get_binary_success_ratio(
            this_binary_ct_as_dict)
        binary_csi_values[i] = object_eval.get_binary_csi(
            this_binary_ct_as_dict)
        binary_frequency_biases[i] = object_eval.get_binary_frequency_bias(
            this_binary_ct_as_dict)

        print (
            'This binary POD = {0:.4f} ... success ratio = {1:.4f} ... '
            'CSI = {2:.4f} ... frequency bias = {3:.4f}'
        ).format(binary_pod_values[i], binary_success_ratios[i],
                 binary_csi_values[i], binary_frequency_biases[i])

        if i == num_bootstrap_replicates - 1:
            print SEPARATOR_STRING
        else:
            print '\n'

    min_percentile_level = 100 * (1. - confidence_level) / 2
    max_percentile_level = min_percentile_level + 100 * confidence_level

    min_binary_pod = numpy.percentile(binary_pod_values, min_percentile_level)
    max_binary_pod = numpy.percentile(binary_pod_values, max_percentile_level)
    print (
        '{0:.2f}th and {1:.2f}th percentiles of binary POD = '
        '[{2:.4f}, {3:.4f}]'
    ).format(min_percentile_level, max_percentile_level, min_binary_pod,
             max_binary_pod)

    min_binary_success_ratio = numpy.percentile(
        binary_success_ratios, min_percentile_level)
    max_binary_success_ratio = numpy.percentile(
        binary_success_ratios, max_percentile_level)
    print (
        '{0:.2f}th and {1:.2f}th percentiles of binary success ratio = '
        '[{2:.4f}, {3:.4f}]'
    ).format(min_percentile_level, max_percentile_level,
             min_binary_success_ratio, max_binary_success_ratio)

    min_binary_csi = numpy.percentile(binary_csi_values, min_percentile_level)
    max_binary_csi = numpy.percentile(binary_csi_values, max_percentile_level)
    print (
        '{0:.2f}th and {1:.2f}th percentiles of binary CSI = '
        '[{2:.4f}, {3:.4f}]'
    ).format(min_percentile_level, max_percentile_level, min_binary_csi,
             max_binary_csi)

    min_binary_frequency_bias = numpy.percentile(
        binary_frequency_biases, min_percentile_level)
    max_binary_frequency_bias = numpy.percentile(
        binary_frequency_biases, max_percentile_level)
    print (
        '{0:.2f}th and {1:.2f}th percentiles of binary frequency bias = '
        '[{2:.4f}, {3:.4f}]'
    ).format(min_percentile_level, max_percentile_level,
             min_binary_frequency_bias, max_binary_frequency_bias)

    output_dir_name, pathless_eval_file_name = os.path.split(
        output_eval_file_name)
    extensionless_eval_file_name = os.path.splitext(pathless_eval_file_name)[0]
    min_percentile_eval_file_name = '{0:s}/{1:s}_percentile={2:09.6f}.p'.format(
        output_dir_name, extensionless_eval_file_name, min_percentile_level)
    max_percentile_eval_file_name = '{0:s}/{1:s}_percentile={2:09.6f}.p'.format(
        output_dir_name, extensionless_eval_file_name, max_percentile_level)

    print 'Writing results to: "{0:s}"...'.format(min_percentile_eval_file_name)
    object_eval.write_evaluation_results(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        neigh_distance_metres=matching_distance_metres,
        binary_contingency_table_as_dict=binary_contingency_table_as_dict,
        binary_pod=min_binary_pod,
        binary_success_ratio=min_binary_success_ratio,
        binary_csi=min_binary_csi,
        binary_frequency_bias=min_binary_frequency_bias,
        row_normalized_ct_as_matrix=row_normalized_ct_as_matrix,
        column_normalized_ct_as_matrix=column_normalized_ct_as_matrix,
        pickle_file_name=min_percentile_eval_file_name)

    print 'Writing results to: "{0:s}"...'.format(max_percentile_eval_file_name)
    object_eval.write_evaluation_results(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        neigh_distance_metres=matching_distance_metres,
        binary_contingency_table_as_dict=binary_contingency_table_as_dict,
        binary_pod=max_binary_pod,
        binary_success_ratio=max_binary_success_ratio,
        binary_csi=max_binary_csi,
        binary_frequency_bias=max_binary_frequency_bias,
        row_normalized_ct_as_matrix=row_normalized_ct_as_matrix,
        column_normalized_ct_as_matrix=column_normalized_ct_as_matrix,
        pickle_file_name=max_percentile_eval_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME),
        matching_distance_metres=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME),
        num_bootstrap_replicates=getattr(
            INPUT_ARG_OBJECT, NUM_REPLICATES_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_eval_file_name=getattr(
            INPUT_ARG_OBJECT, EVALUATION_FILE_ARG_NAME))
