"""Determinizes gridded CNN predictions."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

METRES_TO_KM = 0.001
TIME_INTERVAL_SECONDS = 10800

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NF_THRESHOLD_ARG_NAME = 'nf_prob_threshold'
WF_THRESHOLD_ARG_NAME = 'wf_prob_threshold'
CF_THRESHOLD_ARG_NAME = 'cf_prob_threshold'
MIN_LENGTH_ARG_NAME = 'min_region_length_metres'
BUFFER_DISTANCE_ARG_NAME = 'region_buffer_distance_metres'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files with gridded probabilities will be found '
    'therein by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Determinization will be done for all grids in'
    ' the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NF_THRESHOLD_HELP_STRING = (
    'Threshold for NF probabilities.  If you want to use '
    '`determinize_predictions_2thresholds` instead of '
    '`determinize_predictions_1threshold`, leave this argument alone.')

WF_THRESHOLD_HELP_STRING = (
    'Threshold for WF probabilities.  If you want to use '
    '`determinize_predictions_1threshold` instead of '
    '`determinize_predictions_2thresholds`, leave this argument alone.')

CF_THRESHOLD_HELP_STRING = (
    'Threshold for CF probabilities.  If you want to use '
    '`determinize_predictions_1threshold` instead of '
    '`determinize_predictions_2thresholds`, leave this argument alone.')

MIN_LENGTH_HELP_STRING = (
    'Minimum region length.  Frontal regions (connected regions of either WF or'
    ' CF labels) with smaller major axes will be thrown out.')

BUFFER_DISTANCE_HELP_STRING = (
    'Buffer distance for matching small regions (major axis < `{0:s}`) with '
    'large regions (major axis >= `{0:s}`).  Any small region that can be '
    'matched with a large region, will *not* be thrown out.'
).format(MIN_LENGTH_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files with gridded probabilities *and* '
    'deterministic labels will be written here by '
    '`prediction_io.append_deterministic_labels`, to exact locations determined'
    ' by `prediction_io.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NF_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=NF_THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WF_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=WF_THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CF_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=CF_THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LENGTH_ARG_NAME, type=float, required=False, default=2e5,
    help=MIN_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BUFFER_DISTANCE_ARG_NAME, type=float, required=False, default=2e5,
    help=BUFFER_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _determinize_at_one_time(
        input_file_name, nf_prob_threshold, wf_prob_threshold,
        cf_prob_threshold, min_region_length_metres,
        region_buffer_distance_metres, output_dir_name, valid_time_unix_sec):
    """Determinizes CNN predictions at one time.

    :param input_file_name: Path to input file.  Will be read by
        `prediction_io.read_file`.
    :param nf_prob_threshold: See documentation at top of file.
    :param wf_prob_threshold: Same.
    :param cf_prob_threshold: Same.
    :param min_region_length_metres: Same.
    :param region_buffer_distance_metres: Same.
    :param output_dir_name: Name of output directory.  Probabilities and
        deterministic labels will be written here by
        `prediction_io.write_probabilities` and
        `prediction_io.append_deterministic_labels`.
    :param valid_time_unix_sec: Valid time.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)
    class_probability_matrix = (
        prediction_dict[prediction_io.CLASS_PROBABILITIES_KEY] + 0.
    )

    if numpy.isnan(nf_prob_threshold):
        print((
            'Determinizing probabilities with WF threshold = {0:f}, CF '
            'threshold = {1:f}...'
        ).format(
            wf_prob_threshold, cf_prob_threshold
        ))

        predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_2thresholds(
                class_probability_matrix=class_probability_matrix,
                wf_threshold=wf_prob_threshold, cf_threshold=cf_prob_threshold)
        )
    else:
        print((
            'Determinizing probabilities with NF threshold = {0:f}...'
        ).format(
            nf_prob_threshold
        ))

        predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_1threshold(
                class_probability_matrix=class_probability_matrix,
                nf_threshold=nf_prob_threshold)
        )

    print('Removing frontal regions with major axis < {0:.1f} km...'.format(
        min_region_length_metres * METRES_TO_KM
    ))

    orig_num_frontal_points = numpy.sum(
        predicted_label_matrix > front_utils.NO_FRONT_ENUM
    )

    predicted_label_matrix[0, ...] = (
        neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=predicted_label_matrix[0, ...],
            min_length_metres=min_region_length_metres,
            buffer_distance_metres=region_buffer_distance_metres)
    )

    num_frontal_grid_points = numpy.sum(
        predicted_label_matrix > front_utils.NO_FRONT_ENUM
    )

    print('{0:d} of {1:d} frontal grid points were removed.'.format(
        orig_num_frontal_points - num_frontal_grid_points,
        orig_num_frontal_points
    ))

    output_file_name = prediction_io.find_file(
        directory_name=output_dir_name, first_time_unix_sec=valid_time_unix_sec,
        last_time_unix_sec=valid_time_unix_sec, raise_error_if_missing=False)

    print('Writing deterministic predictions to: "{0:s}"...'.format(
        output_file_name
    ))

    if prediction_io.TARGET_MATRIX_KEY in prediction_dict:
        target_matrix = prediction_dict[
            prediction_io.TARGET_MATRIX_KEY]
        dilation_distance_metres = prediction_dict[
            prediction_io.DILATION_DISTANCE_KEY]
    else:
        target_matrix = None
        dilation_distance_metres = None

    prediction_io.write_probabilities(
        netcdf_file_name=output_file_name,
        class_probability_matrix=prediction_dict[
            prediction_io.CLASS_PROBABILITIES_KEY
        ],
        target_matrix=target_matrix,
        valid_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int),
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        target_dilation_distance_metres=dilation_distance_metres,
        used_isotonic=prediction_dict[prediction_io.USED_ISOTONIC_KEY]
    )

    prediction_io.append_deterministic_labels(
        probability_file_name=output_file_name,
        predicted_label_matrix=predicted_label_matrix,
        prob_threshold_by_class=numpy.array(
            [nf_prob_threshold, wf_prob_threshold, cf_prob_threshold]
        ),
        min_region_length_metres=min_region_length_metres,
        region_buffer_distance_metres=region_buffer_distance_metres)


def _run(input_prediction_dir_name, first_time_string, last_time_string,
         nf_prob_threshold, wf_prob_threshold, cf_prob_threshold,
         min_region_length_metres, region_buffer_distance_metres,
         output_prediction_dir_name):
    """Determinizes gridded CNN predictions.

    This is effectively the main method.

    :param input_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param nf_prob_threshold: Same.
    :param wf_prob_threshold: Same.
    :param cf_prob_threshold: Same.
    :param min_region_length_metres: Same.
    :param region_buffer_distance_metres: Same.
    :param output_prediction_dir_name: Same.
    """

    if nf_prob_threshold < 0:
        nf_prob_threshold = numpy.nan
    else:
        wf_prob_threshold = numpy.nan
        cf_prob_threshold = numpy.nan

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        this_input_file_name = prediction_io.find_file(
            directory_name=input_prediction_dir_name,
            first_time_unix_sec=valid_times_unix_sec[i],
            last_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(this_input_file_name):
            continue

        print('\n')

        _determinize_at_one_time(
            input_file_name=this_input_file_name,
            nf_prob_threshold=nf_prob_threshold,
            wf_prob_threshold=wf_prob_threshold,
            cf_prob_threshold=cf_prob_threshold,
            min_region_length_metres=min_region_length_metres,
            region_buffer_distance_metres=region_buffer_distance_metres,
            output_dir_name=output_prediction_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        nf_prob_threshold=getattr(INPUT_ARG_OBJECT, NF_THRESHOLD_ARG_NAME),
        wf_prob_threshold=getattr(INPUT_ARG_OBJECT, WF_THRESHOLD_ARG_NAME),
        cf_prob_threshold=getattr(INPUT_ARG_OBJECT, CF_THRESHOLD_ARG_NAME),
        min_region_length_metres=getattr(INPUT_ARG_OBJECT, MIN_LENGTH_ARG_NAME),
        region_buffer_distance_metres=getattr(
            INPUT_ARG_OBJECT, BUFFER_DISTANCE_ARG_NAME
        ),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
