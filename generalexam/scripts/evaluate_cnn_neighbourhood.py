"""Runs neigh evaluation on gridded deterministic labels from CNN."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import prediction_io
from generalexam.machine_learning import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

METRES_TO_KM = 0.001
TIME_INTERVAL_SECONDS = 10800

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


def _do_eval_one_neigh_distance(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        num_bootstrap_reps, confidence_level, prediction_file_names,
        output_file_name):
    """Does evaluation with one neighbourhood distance.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix: T-by-M-by-N numpy array of predicted labels
        (in range 0...2).
    :param actual_label_matrix: Same but for actual labels.
    :param neigh_distance_metres: Neighbourhood distance.
    :param num_bootstrap_reps: See documentation at top of file.
    :param confidence_level: Same.
    :param prediction_file_names: length-T list of paths to input (prediction)
        files.
    :param output_file_name: Path to output file.
    """

    num_times = predicted_label_matrix.shape[0]
    time_indices = numpy.linspace(0, num_times - 1, num=num_times, dtype=int)

    list_of_binary_ct_dicts = [None] * num_bootstrap_reps
    prediction_oriented_ct_matrix = None
    actual_oriented_ct_matrix = None

    pod_values = numpy.full(num_bootstrap_reps, numpy.nan)
    success_ratios = numpy.full(num_bootstrap_reps, numpy.nan)
    csi_values = numpy.full(num_bootstrap_reps, numpy.nan)
    frequency_biases = numpy.full(num_bootstrap_reps, numpy.nan)

    for i in range(num_bootstrap_reps):
        these_time_indices = bootstrapping.draw_sample(time_indices)[-1]

        (list_of_binary_ct_dicts[i], this_prediction_oriented_matrix,
         this_actual_oriented_matrix
        ) = neigh_evaluation.make_contingency_tables(
            predicted_label_matrix=predicted_label_matrix[
                these_time_indices, ...],
            actual_label_matrix=actual_label_matrix[these_time_indices, ...],
            neigh_distance_metres=neigh_distance_metres
        )

        if prediction_oriented_ct_matrix is None:
            dimensions = (
                (num_bootstrap_reps,) + this_prediction_oriented_matrix.shape
            )
            prediction_oriented_ct_matrix = numpy.full(dimensions, numpy.nan)
            actual_oriented_ct_matrix = numpy.full(dimensions, numpy.nan)

        prediction_oriented_ct_matrix[i, ...] = this_prediction_oriented_matrix
        actual_oriented_ct_matrix[i, ...] = this_actual_oriented_matrix

        pod_values[i] = neigh_evaluation.get_binary_pod(
            list_of_binary_ct_dicts[i]
        )
        success_ratios[i] = neigh_evaluation.get_binary_success_ratio(
            list_of_binary_ct_dicts[i]
        )
        csi_values[i] = neigh_evaluation.get_binary_csi(
            list_of_binary_ct_dicts[i]
        )
        frequency_biases[i] = neigh_evaluation.get_binary_frequency_bias(
            list_of_binary_ct_dicts[i]
        )

        print((
            'Scores for {0:d}th bootstrap replicate, {1:.1f}-km neighbourhood: '
            'POD = {2:.4f}, success ratio = {3:.4f}, CSI = {4:.4f}, freq bias ='
            ' {5:.4f}'
        ).format(
            i, neigh_distance_metres * METRES_TO_KM,
            pod_values[i], success_ratios[i], csi_values[i], frequency_biases[i]
        ))

        print(SEPARATOR_STRING)

    min_pod = numpy.nanpercentile(pod_values, 50. * (1 - confidence_level))
    max_pod = numpy.nanpercentile(pod_values, 50. * (1 + confidence_level))
    min_success_ratio = numpy.nanpercentile(
        success_ratios, 50. * (1 - confidence_level)
    )
    max_success_ratio = numpy.nanpercentile(
        success_ratios, 50. * (1 + confidence_level)
    )
    min_csi = numpy.nanpercentile(csi_values, 50. * (1 - confidence_level))
    max_csi = numpy.nanpercentile(csi_values, 50. * (1 + confidence_level))
    min_frequency_bias = numpy.nanpercentile(
        frequency_biases, 50. * (1 - confidence_level)
    )
    max_frequency_bias = numpy.nanpercentile(
        frequency_biases, 50. * (1 + confidence_level)
    )

    print((
        '{0:.1f}% confidence interval for POD = [{1:.4f}, {2:.4f}] ... '
        'success ratio = [{3:.4f}, {4:.4f}] ... CSI = [{5:.4f}, {6:.4f}] ... '
        'frequency bias = [{7:.4f}, {8:.4f}]\n'
    ).format(
        100 * confidence_level, min_pod, max_pod,
        min_success_ratio, max_success_ratio, min_csi, max_csi,
        min_frequency_bias, max_frequency_bias
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

    num_bootstrap_reps = max([num_bootstrap_reps, 1])
    if num_bootstrap_reps > 1:
        error_checking.assert_is_geq(confidence_level, 0.9)
        error_checking.assert_is_less_than(confidence_level, 1.)

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

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_file_name, read_deterministic=True)

        this_predicted_label_matrix = this_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]
        this_actual_label_matrix = this_prediction_dict[
            prediction_io.TARGET_MATRIX_KEY]

        if predicted_label_matrix is None:
            predicted_label_matrix = this_predicted_label_matrix + 0
            actual_label_matrix = this_actual_label_matrix + 0
        else:
            predicted_label_matrix = numpy.concatenate(
                (predicted_label_matrix, this_predicted_label_matrix),
                axis=0
            )
            actual_label_matrix = numpy.concatenate(
                (actual_label_matrix, this_actual_label_matrix), axis=0
            )

    for this_neigh_distance_metres in neigh_distances_metres:
        print(SEPARATOR_STRING)

        this_output_file_name = (
            '{0:s}/evaluation_neigh-distance-metres={1:06d}.p'
        ).format(
            output_dir_name, int(numpy.round(this_neigh_distance_metres))
        )

        _do_eval_one_neigh_distance(
            predicted_label_matrix=predicted_label_matrix,
            actual_label_matrix=actual_label_matrix,
            neigh_distance_metres=this_neigh_distance_metres,
            num_bootstrap_reps=num_bootstrap_reps,
            confidence_level=confidence_level,
            prediction_file_names=prediction_file_names,
            output_file_name=this_output_file_name)


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
