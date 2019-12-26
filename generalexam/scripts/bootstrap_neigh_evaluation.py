"""Bootstraps neighbourhood evaluation."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_CLASSES = 3
FAR_WEIGHT_FOR_CSI = 0.5

PREDICTED_LABELS_KEY = 'predicted_front_enums'
PREDICTED_TO_ACTUAL_FRONTS_KEY = 'predicted_to_actual_front_enums'
ACTUAL_LABELS_KEY = 'actual_front_enums'
ACTUAL_TO_PREDICTED_FRONTS_KEY = 'actual_to_predicted_front_enums'

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
NUM_REPLICATES_ARG_NAME = 'num_replicates'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_eval_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (with non-bootstrapped neighbourhood evaluation).  Will'
    ' be read by `neigh_evaluation.read_nonspatial_results`.'
)
NUM_REPLICATES_HELP_STRING = 'Number of bootstrap replicates.'
CONFIDENCE_LEVEL_HELP_STRING = 'Confidence level for scores (in range 0...1).'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (with bootstrapped neighbourhood evaluation).  Will be'
    ' written by `neigh_evaluation.write_nonspatial_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_REPLICATES_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_REPLICATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _decompose_contingency_tables(
        prediction_oriented_ct_matrix, actual_oriented_ct_matrix):
    """Decomposes 3-class contingency tables into prediction-observation pairs.

    P = number of predicted fronts
    A = number of actual fronts

    :param prediction_oriented_ct_matrix: See doc for
        `neigh_evaluation._check_3class_contingency_tables` with
        `normalize == False`.
    :param actual_oriented_ct_matrix: Same.
    :return: match_dict: Dictionary with the following keys.
    match_dict["predicted_front_enums"]: length-P numpy array of predicted
        labels (integers).
    match_dict["predicted_to_actual_front_enums"]: length-P numpy array of
        corresponding actual labels (integers).
    match_dict["actual_front_enums"]: length-A numpy array of actual labels
        (integers).
    match_dict["actual_to_predicted_front_enums"]: length-A numpy array of
        corresponding predicted labels (integers).
    """

    predicted_front_enums = numpy.array([], dtype=int)
    predicted_to_actual_front_enums = numpy.array([], dtype=int)

    for i in range(1, NUM_CLASSES):
        for j in range(NUM_CLASSES):
            this_num_examples = int(numpy.round(
                prediction_oriented_ct_matrix[i, j]
            ))

            predicted_front_enums = numpy.concatenate((
                predicted_front_enums,
                numpy.full(this_num_examples, i, dtype=int)
            ))

            predicted_to_actual_front_enums = numpy.concatenate((
                predicted_to_actual_front_enums,
                numpy.full(this_num_examples, j, dtype=int)
            ))

    actual_front_enums = numpy.array([], dtype=int)
    actual_to_predicted_front_enums = numpy.array([], dtype=int)

    for i in range(NUM_CLASSES):
        for j in range(1, NUM_CLASSES):
            this_num_examples = int(numpy.round(
                actual_oriented_ct_matrix[i, j]
            ))

            actual_front_enums = numpy.concatenate((
                actual_front_enums,
                numpy.full(this_num_examples, j, dtype=int)
            ))

            actual_to_predicted_front_enums = numpy.concatenate((
                actual_to_predicted_front_enums,
                numpy.full(this_num_examples, i, dtype=int)
            ))

    return {
        PREDICTED_LABELS_KEY: predicted_front_enums,
        PREDICTED_TO_ACTUAL_FRONTS_KEY: predicted_to_actual_front_enums,
        ACTUAL_LABELS_KEY: actual_front_enums,
        ACTUAL_TO_PREDICTED_FRONTS_KEY: actual_to_predicted_front_enums
    }


def _bootstrap_contingency_tables(match_dict, test_mode=False):
    """Makes contingency tables for one bootstrap replicate.

    :param match_dict: Dictionary created by `_decompose_contingency_tables`.
    :param test_mode: Never mind.  Just leave this alone.
    :return: binary_ct_as_dict: See doc for
        `neigh_evaluation.make_nonspatial_contingency_tables`.
    :return: prediction_oriented_ct_matrix: Same.
    :return: actual_oriented_ct_matrix: Same.
    """

    predicted_front_enums = match_dict[PREDICTED_LABELS_KEY]
    predicted_to_actual_front_enums = match_dict[
        PREDICTED_TO_ACTUAL_FRONTS_KEY
    ]
    actual_front_enums = match_dict[ACTUAL_LABELS_KEY]
    actual_to_predicted_front_enums = match_dict[
        ACTUAL_TO_PREDICTED_FRONTS_KEY
    ]

    num_predicted_fronts = len(predicted_front_enums)
    predicted_front_indices = numpy.linspace(
        0, num_predicted_fronts - 1, num=num_predicted_fronts, dtype=int
    )

    num_actual_fronts = len(actual_front_enums)
    actual_front_indices = numpy.linspace(
        0, num_actual_fronts - 1, num=num_actual_fronts, dtype=int
    )

    these_dim = (NUM_CLASSES, NUM_CLASSES)
    prediction_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    actual_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    binary_ct_as_dict = dict()

    if test_mode:
        these_indices = predicted_front_indices + 0
    else:
        these_indices = bootstrapping.draw_sample(predicted_front_indices)[-1]

    for i in range(1, NUM_CLASSES):
        for j in range(NUM_CLASSES):
            prediction_oriented_ct_matrix[i, j] = numpy.sum(numpy.logical_and(
                predicted_front_enums[these_indices] == i,
                predicted_to_actual_front_enums[these_indices] == j
            ))

    binary_ct_as_dict[neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY] = (
        numpy.sum(
            predicted_front_enums[these_indices] ==
            predicted_to_actual_front_enums[these_indices]
        )
    )

    binary_ct_as_dict[neigh_evaluation.NUM_FALSE_POSITIVES_KEY] = numpy.sum(
        predicted_front_enums[these_indices] !=
        predicted_to_actual_front_enums[these_indices]
    )

    if test_mode:
        these_indices = actual_front_indices + 0
    else:
        these_indices = bootstrapping.draw_sample(actual_front_indices)[-1]

    for i in range(NUM_CLASSES):
        for j in range(1, NUM_CLASSES):
            actual_oriented_ct_matrix[i, j] = numpy.sum(numpy.logical_and(
                actual_front_enums[these_indices] == j,
                actual_to_predicted_front_enums[these_indices] == i
            ))

    binary_ct_as_dict[neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY] = numpy.sum(
        actual_front_enums[these_indices] ==
        actual_to_predicted_front_enums[these_indices]
    )

    binary_ct_as_dict[neigh_evaluation.NUM_FALSE_NEGATIVES_KEY] = numpy.sum(
        actual_front_enums[these_indices] !=
        actual_to_predicted_front_enums[these_indices]
    )

    print(binary_ct_as_dict)
    print('\n')
    print(prediction_oriented_ct_matrix)
    print('\n')
    print(actual_oriented_ct_matrix)

    return (
        binary_ct_as_dict, prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix
    )


def _run(input_eval_file_name, num_replicates, confidence_level,
         output_eval_file_name):
    """Bootstraps neighbourhood evaluation.

    This is effectively the main method.

    :param input_eval_file_name: See documentation at top of file.
    :param num_replicates: Same.
    :param confidence_level: Same.
    :param output_eval_file_name: Same.
    :raises: ValueError: if input file already contains bootstrapped results.
    """

    error_checking.assert_is_geq(num_replicates, 100)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    print('Reading non-bootstrapped results from: "{0:s}"...'.format(
        input_eval_file_name
    ))
    input_evaluation_dict = neigh_evaluation.read_nonspatial_results(
        input_eval_file_name
    )

    orig_num_replicates = len(
        input_evaluation_dict[neigh_evaluation.BINARY_TABLES_KEY]
    )
    if orig_num_replicates > 1:
        raise ValueError('Input file already contains bootstrapped results.')

    # Save non-bootstrapped versions of contingency tables.
    main_pred_oriented_ct_matrix = copy.deepcopy(
        input_evaluation_dict[neigh_evaluation.PREDICTION_ORIENTED_CT_KEY][
            0, ...]
    )
    main_actual_oriented_ct_matrix = copy.deepcopy(
        input_evaluation_dict[neigh_evaluation.ACTUAL_ORIENTED_CT_KEY][0, ...]
    )

    # Bootstrap contingency tables.
    match_dict = _decompose_contingency_tables(
        prediction_oriented_ct_matrix=main_pred_oriented_ct_matrix,
        actual_oriented_ct_matrix=main_actual_oriented_ct_matrix
    )

    these_dim = (num_replicates, NUM_CLASSES, NUM_CLASSES)
    prediction_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    actual_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    list_of_binary_ct_dicts = [dict()] * num_replicates

    pod_values = numpy.full(num_replicates, numpy.nan)
    far_values = numpy.full(num_replicates, numpy.nan)
    csi_values = numpy.full(num_replicates, numpy.nan)
    weighted_csi_values = numpy.full(num_replicates, numpy.nan)
    frequency_biases = numpy.full(num_replicates, numpy.nan)

    for k in range(num_replicates):
        (
            list_of_binary_ct_dicts[k],
            prediction_oriented_ct_matrix[k, ...],
            actual_oriented_ct_matrix[k, ...]
        ) = _bootstrap_contingency_tables(match_dict)

        print(SEPARATOR_STRING)

        pod_values[k] = neigh_evaluation.get_pod(list_of_binary_ct_dicts[k])
        far_values[k] = neigh_evaluation.get_far(list_of_binary_ct_dicts[k])
        csi_values[k] = neigh_evaluation.get_csi(
            binary_ct_as_dict=list_of_binary_ct_dicts[k], far_weight=1.
        )
        weighted_csi_values[k] = neigh_evaluation.get_csi(
            binary_ct_as_dict=list_of_binary_ct_dicts[k],
            far_weight=FAR_WEIGHT_FOR_CSI
        )
        frequency_biases[k] = neigh_evaluation.get_frequency_bias(
            list_of_binary_ct_dicts[k]
        )

    min_pod = numpy.percentile(pod_values, 50. * (1 - confidence_level))
    max_pod = numpy.percentile(pod_values, 50. * (1 + confidence_level))
    print((
        '{0:.1f}% confidence interval for POD = [{1:.4f}, {2:.4f}]'
    ).format(
        100 * confidence_level, min_pod, max_pod
    ))

    min_far = numpy.percentile(far_values, 50. * (1 - confidence_level))
    max_far = numpy.percentile(far_values, 50. * (1 + confidence_level))
    print((
        '{0:.1f}% confidence interval for FAR = [{1:.4f}, {2:.4f}]'
    ).format(
        100 * confidence_level, min_far, max_far
    ))

    min_csi = numpy.percentile(csi_values, 50. * (1 - confidence_level))
    max_csi = numpy.percentile(csi_values, 50. * (1 + confidence_level))
    print((
        '{0:.1f}% confidence interval for CSI = [{1:.4f}, {2:.4f}]'
    ).format(
        100 * confidence_level, min_csi, max_csi
    ))

    min_weighted_csi = numpy.percentile(
        weighted_csi_values, 50. * (1 - confidence_level)
    )
    max_weighted_csi = numpy.percentile(
        weighted_csi_values, 50. * (1 + confidence_level)
    )
    print((
        '{0:.1f}% confidence interval for weighted CSI = [{1:.4f}, {2:.4f}]'
    ).format(
        100 * confidence_level, min_weighted_csi, max_weighted_csi
    ))

    min_frequency_bias = numpy.percentile(
        frequency_biases, 50. * (1 - confidence_level)
    )
    max_frequency_bias = numpy.percentile(
        frequency_biases, 50. * (1 + confidence_level)
    )
    print((
        '{0:.1f}% confidence interval for frequency bias = [{1:.4f}, {2:.4f}]'
    ).format(
        100 * confidence_level, min_frequency_bias, max_frequency_bias
    ))

    print('Writing results to: "{0:s}"...'.format(output_eval_file_name))
    neigh_evaluation.write_nonspatial_results(
        pickle_file_name=output_eval_file_name,
        prediction_file_names=input_evaluation_dict[
            neigh_evaluation.PREDICTION_FILES_KEY
        ],
        neigh_distance_metres=input_evaluation_dict[
            neigh_evaluation.NEIGH_DISTANCE_KEY
        ],
        list_of_binary_ct_dicts=list_of_binary_ct_dicts,
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_eval_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_replicates=getattr(INPUT_ARG_OBJECT, NUM_REPLICATES_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_eval_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
