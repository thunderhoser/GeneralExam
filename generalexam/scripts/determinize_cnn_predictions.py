"""Determinizes gridded CNN predictions."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import neigh_evaluation
from generalexam.machine_learning import machine_learning_utils as ml_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'
TIME_INTERVAL_SECONDS = 10800

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
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

MASK_FILE_HELP_STRING = (
    'Path to mask file (will be read by `machine_learning_utils.read_narr_mask`'
    ').  The mask will be applied after removing small regions.  If you do not '
    'want a mask, leave this argument alone.')

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
    '--' + MASK_FILE_ARG_NAME, type=str, required=False, default='',
    help=MASK_FILE_HELP_STRING)

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
    '--' + MIN_LENGTH_ARG_NAME, type=float, required=False, default=500000,
    help=MIN_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BUFFER_DISTANCE_ARG_NAME, type=float, required=False, default=200000,
    help=BUFFER_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_prediction_dir_name, mask_file_name, first_time_string,
         last_time_string, nf_prob_threshold, wf_prob_threshold,
         cf_prob_threshold, min_region_length_metres,
         region_buffer_distance_metres, output_prediction_dir_name):
    """Determinizes gridded CNN predictions.

    This is effectively the main method.

    :param input_prediction_dir_name: See documentation at top of file.
    :param mask_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param nf_prob_threshold: Same.
    :param wf_prob_threshold: Same.
    :param cf_prob_threshold: Same.
    :param min_region_length_metres: Same.
    :param region_buffer_distance_metres: Same.
    :param output_prediction_dir_name: Same.
    """

    if mask_file_name in ['', 'None']:
        mask_matrix = None
    else:
        print('Reading mask from: "{0:s}"...'.format(mask_file_name))
        mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]

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
        print(this_input_file_name)

        if not os.path.isfile(this_input_file_name):
            continue

        print('\nReading data from: "{0:s}"...'.format(this_input_file_name))
        this_prediction_dict = prediction_io.read_file(this_input_file_name)

        this_class_probability_matrix = (
            this_prediction_dict[prediction_io.CLASS_PROBABILITIES_KEY] + 0.
        )

        if numpy.isnan(nf_prob_threshold):
            print((
                'Determinizing probabilities with WF threshold = {0:f}, CF '
                'threshold = {1:f}...'
            ).format(wf_prob_threshold, cf_prob_threshold))

            this_predicted_label_matrix = (
                neigh_evaluation.determinize_predictions_2thresholds(
                    class_probability_matrix=this_class_probability_matrix,
                    wf_threshold=wf_prob_threshold,
                    cf_threshold=cf_prob_threshold)
            )
        else:
            print((
                'Determinizing probabilities with NF threshold = {0:f}...'
            ).format(
                nf_prob_threshold
            ))

            this_predicted_label_matrix = (
                neigh_evaluation.determinize_predictions_1threshold(
                    class_probability_matrix=this_class_probability_matrix,
                    binarization_threshold=nf_prob_threshold)
            )

        print((
            'Removing small frontal regions (major axis < {0:f} metres)...'
        ).format(min_region_length_metres))

        this_orig_num_frontal = numpy.sum(
            this_predicted_label_matrix > front_utils.NO_FRONT_ENUM
        )

        this_predicted_label_matrix[0, ...] = (
            neigh_evaluation.remove_small_regions_one_time(
                predicted_label_matrix=this_predicted_label_matrix[0, ...],
                min_region_length_metres=min_region_length_metres,
                buffer_distance_metres=region_buffer_distance_metres)
        )

        this_new_num_frontal = numpy.sum(
            this_predicted_label_matrix > front_utils.NO_FRONT_ENUM
        )

        print('Removed {0:d} of {1:d} frontal grid cells.'.format(
            this_orig_num_frontal - this_new_num_frontal, this_orig_num_frontal
        ))

        if mask_matrix is not None:
            print('Masking out {0:d} of {1:d} grid cells...'.format(
                numpy.sum(mask_matrix == 0), mask_matrix.size
            ))

            this_predicted_label_matrix[0, ...][mask_matrix == 0] = (
                front_utils.NO_FRONT_ENUM
            )

            if prediction_io.TARGET_MATRIX_KEY in this_prediction_dict:
                this_prediction_dict[prediction_io.TARGET_MATRIX_KEY][0, ...][
                    mask_matrix == 0
                ] = front_utils.NO_FRONT_ENUM

        this_output_file_name = prediction_io.find_file(
            directory_name=output_prediction_dir_name,
            first_time_unix_sec=valid_times_unix_sec[i],
            last_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print('Writing deterministic predictions to: "{0:s}"...'.format(
            this_output_file_name))

        if prediction_io.TARGET_MATRIX_KEY in this_prediction_dict:
            target_matrix = this_prediction_dict[
                prediction_io.TARGET_MATRIX_KEY]
            dilation_distance_metres = this_prediction_dict[
                prediction_io.DILATION_DISTANCE_KEY]
        else:
            target_matrix = None
            dilation_distance_metres = None

        prediction_io.write_probabilities(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_prediction_dict[
                prediction_io.CLASS_PROBABILITIES_KEY],
            target_matrix=target_matrix,
            valid_times_unix_sec=valid_times_unix_sec[[i]],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            target_dilation_distance_metres=dilation_distance_metres,
            used_isotonic=this_prediction_dict[prediction_io.USED_ISOTONIC_KEY]
        )

        prediction_io.append_deterministic_labels(
            probability_file_name=this_output_file_name,
            predicted_label_matrix=this_predicted_label_matrix,
            prob_threshold_by_class=numpy.array(
                [nf_prob_threshold, wf_prob_threshold, cf_prob_threshold]
            ),
            min_region_length_metres=min_region_length_metres,
            region_buffer_distance_metres=region_buffer_distance_metres)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        nf_prob_threshold=getattr(INPUT_ARG_OBJECT, NF_THRESHOLD_ARG_NAME),
        wf_prob_threshold=getattr(INPUT_ARG_OBJECT, WF_THRESHOLD_ARG_NAME),
        cf_prob_threshold=getattr(INPUT_ARG_OBJECT, CF_THRESHOLD_ARG_NAME),
        min_region_length_metres=getattr(INPUT_ARG_OBJECT, MIN_LENGTH_ARG_NAME),
        region_buffer_distance_metres=getattr(
            INPUT_ARG_OBJECT, BUFFER_DISTANCE_ARG_NAME),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
