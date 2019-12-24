"""Runs neigh evaluation on gridded deterministic labels from CNN."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import prediction_io
from generalexam.machine_learning import cnn
from generalexam.ge_utils import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

NUM_CLASSES = 3
METRES_TO_KM = 0.001
TIME_INTERVAL_SECONDS = 10800

FAR_WEIGHT_FOR_CSI = 0.5

PREDICTED_LABELS_KEY = 'predicted_front_enums'
PREDICTED_TO_ACTUAL_FRONTS_KEY = 'predicted_to_actual_front_enums'
ACTUAL_LABELS_KEY = 'actual_front_enums'
ACTUAL_TO_PREDICTED_FRONTS_KEY = 'actual_to_predicted_front_enums'

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NEIGH_DISTANCES_ARG_NAME = 'neigh_distances_metres'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of input directory.  Files with gridded deterministic labels will be '
    'found therein by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Neighbourhood evaluation will be done for all'
    ' times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NEIGH_DISTANCES_HELP_STRING = (
    'List of neighbourhood distances.  Evaluation will be done for each.')

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates.  If you do not want bootstrapping, leave '
    'this alone.')

CONFIDENCE_LEVEL_HELP_STRING = (
    '[used only if `{0:s}` > 1] Confidence level for bootstrapping, in range '
    '0...1.'
).format(NUM_BOOTSTRAP_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`neigh_evaluation.write_results`.'
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
    '--' + NEIGH_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=NEIGH_DISTANCES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _handle_one_prediction_file(
        prediction_file_name, neigh_distances_metres, binary_ct_by_neigh,
        prediction_oriented_ct_by_neigh, actual_oriented_ct_by_neigh,
        training_mask_matrix=None):
    """Handles one prediction file.

    D = number of neighbourhood distances

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param neigh_distances_metres: length-D numpy array of distances for
        neighbourhood evaluation.
    :param binary_ct_by_neigh: length-D list of binary contingency tables in
        format produced by `neigh_evaluation.make_contingency_tables`.
    :param prediction_oriented_ct_by_neigh: length-D list of prediction-oriented
        contingency tables in format produced by
        `neigh_evaluation.make_contingency_tables`.
    :param actual_oriented_ct_by_neigh: length-D list of actual-oriented
        contingency tables in format produced by
        `neigh_evaluation.make_contingency_tables`.
    :param training_mask_matrix: See doc for
        `neigh_evaluation.make_contingency_tables`.  If this is None, will be
        read from CNN metadata on the fly.
    :return: binary_ct_by_neigh: Same as input but with different values.
    :return: prediction_oriented_ct_by_neigh: Same as input but with different
        values.
    :return: actual_oriented_ct_by_neigh: Same as input but with different
        values.
    :return: training_mask_matrix: Same as input but cannot be None.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(
        netcdf_file_name=prediction_file_name, read_deterministic=True
    )

    predicted_label_matrix = prediction_dict[prediction_io.PREDICTED_LABELS_KEY]
    actual_label_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]

    if training_mask_matrix is None:
        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = cnn.find_metafile(model_file_name)

        print('Reading training mask from: "{0:s}"...'.format(
            model_metafile_name
        ))
        model_metadata_dict = cnn.read_metadata(model_metafile_name)
        training_mask_matrix = model_metadata_dict[cnn.MASK_MATRIX_KEY]

    num_neigh_distances = len(neigh_distances_metres)

    for k in range(num_neigh_distances):
        this_binary_ct, this_prediction_oriented_ct, this_actual_oriented_ct = (
            neigh_evaluation.make_contingency_tables(
                predicted_label_matrix=predicted_label_matrix,
                actual_label_matrix=actual_label_matrix,
                neigh_distance_metres=neigh_distances_metres[k],
                training_mask_matrix=training_mask_matrix, normalize=False
            )
        )

        if binary_ct_by_neigh[k] is None:
            binary_ct_by_neigh[k] = copy.deepcopy(this_binary_ct)
            prediction_oriented_ct_by_neigh = this_prediction_oriented_ct + 0.
            actual_oriented_ct_by_neigh = this_actual_oriented_ct + 0.

    return (
        binary_ct_by_neigh, prediction_oriented_ct_by_neigh,
        actual_oriented_ct_by_neigh, training_mask_matrix
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
        `neigh_evaluation.make_contingency_tables`.
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

    prediction_oriented_ct_matrix, actual_oriented_ct_matrix = (
        neigh_evaluation.normalize_contingency_tables(
            prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
            actual_oriented_ct_matrix=actual_oriented_ct_matrix)
    )

    print(binary_ct_as_dict)
    print('\n')
    print(prediction_oriented_ct_matrix)
    print('\n')
    print(actual_oriented_ct_matrix)

    return (binary_ct_as_dict, prediction_oriented_ct_matrix,
            actual_oriented_ct_matrix)


def _do_eval_one_neigh_distance(
        binary_ct_as_dict, prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix, neigh_distance_metres, num_bootstrap_reps,
        confidence_level, prediction_file_names, output_file_name):
    """Does evaluation with one neighbourhood distance.

    :param binary_ct_as_dict: Binary contingency table in format produced
        by `neigh_evaluation.make_contingency_tables`.
    :param prediction_oriented_ct_matrix: Prediction-oriented contingency table
        in format produced by `neigh_evaluation.make_contingency_tables`.
    :param actual_oriented_ct_matrix: Actual-oriented contingency table in
        format produced by `neigh_evaluation.make_contingency_tables`.
    :param neigh_distance_metres: Neighbourhood distance.
    :param num_bootstrap_reps: See documentation at top of file.
    :param confidence_level: Same.
    :param prediction_file_names: 1-D list of paths to prediction files (will be
        saved as metadata in output file).
    :param output_file_name: Path to output file (will be written by
       `neigh_evaluation.write_results`).
    """

    # Save non-bootstrapped versions of contingency tables.
    main_binary_ct = binary_ct_as_dict
    main_prediction_oriented_ct = prediction_oriented_ct_matrix + 0.
    main_actual_oriented_ct = actual_oriented_ct_matrix + 0.

    # Bootstrap contingency tables.
    match_dict = _decompose_contingency_tables(
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix)

    these_dim = (num_bootstrap_reps, NUM_CLASSES, NUM_CLASSES)
    prediction_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    actual_oriented_ct_matrix = numpy.full(these_dim, numpy.nan)
    list_of_binary_ct_dicts = [dict()] * num_bootstrap_reps

    pod_values = numpy.full(num_bootstrap_reps, numpy.nan)
    far_values = numpy.full(num_bootstrap_reps, numpy.nan)
    csi_values = numpy.full(num_bootstrap_reps, numpy.nan)
    weighted_csi_values = numpy.full(num_bootstrap_reps, numpy.nan)
    frequency_biases = numpy.full(num_bootstrap_reps, numpy.nan)

    for k in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            list_of_binary_ct_dicts[k] = main_binary_ct
            prediction_oriented_ct_matrix[k, ...] = main_prediction_oriented_ct
            actual_oriented_ct_matrix[k, ...] = main_actual_oriented_ct
        else:
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

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    neigh_evaluation.write_results(
        pickle_file_name=output_file_name,
        prediction_file_names=prediction_file_names,
        neigh_distance_metres=neigh_distance_metres,
        list_of_binary_ct_dicts=list_of_binary_ct_dicts,
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix)


def _run(prediction_dir_name, first_time_string, last_time_string,
         neigh_distances_metres, num_bootstrap_reps, confidence_level,
         output_dir_name):
    """Runs neigh evaluation on gridded deterministic labels from CNN.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param neigh_distances_metres: Same.
    :param num_bootstrap_reps: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    if num_bootstrap_reps > 1:
        error_checking.assert_is_geq(confidence_level, 0.9)
        error_checking.assert_is_less_than(confidence_level, 1.)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT
    )
    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

    # Read predictions and create contingency tables.
    num_neigh_distances = len(neigh_distances_metres)
    binary_ct_by_neigh = [None] * num_neigh_distances
    prediction_oriented_ct_by_neigh = [None] * num_neigh_distances
    actual_oriented_ct_by_neigh = [None] * num_neigh_distances

    training_mask_matrix = None
    prediction_file_names = []

    for this_time_unix_sec in all_times_unix_sec:
        this_prediction_file_name = prediction_io.find_file(
            directory_name=prediction_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False)

        if not os.path.isfile(this_prediction_file_name):
            continue

        prediction_file_names.append(this_prediction_file_name)
        print(MINOR_SEPARATOR_STRING)

        (
            binary_ct_by_neigh, prediction_oriented_ct_by_neigh,
            actual_oriented_ct_by_neigh, training_mask_matrix
        ) = _handle_one_prediction_file(
            prediction_file_name=this_prediction_file_name,
            neigh_distances_metres=neigh_distances_metres,
            binary_ct_by_neigh=binary_ct_by_neigh,
            prediction_oriented_ct_by_neigh=prediction_oriented_ct_by_neigh,
            actual_oriented_ct_by_neigh=actual_oriented_ct_by_neigh,
            training_mask_matrix=training_mask_matrix
        )

    # Run neighbourhood evaluation.
    for k in range(num_neigh_distances):
        print(SEPARATOR_STRING)

        this_output_file_name = (
            '{0:s}/evaluation_neigh-distance-metres={1:06d}.p'
        ).format(
            output_dir_name, int(numpy.round(neigh_distances_metres[k]))
        )

        _do_eval_one_neigh_distance(
            binary_ct_as_dict=binary_ct_by_neigh[k],
            prediction_oriented_ct_matrix=prediction_oriented_ct_by_neigh[k],
            actual_oriented_ct_matrix=actual_oriented_ct_by_neigh[k],
            neigh_distance_metres=neigh_distances_metres[k],
            num_bootstrap_reps=num_bootstrap_reps,
            confidence_level=confidence_level,
            prediction_file_names=prediction_file_names,
            output_file_name=this_output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        neigh_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEIGH_DISTANCES_ARG_NAME), dtype=float
        ),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
