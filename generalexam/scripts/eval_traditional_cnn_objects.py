"""Evaluates CNN trained by patch classification.

In this case, evaluation is done in an object-based setting.  If you want to do
pixelwise evaluation, use eval_traditional_cnn_pixelwise.py.
"""

import random
import os.path
import argparse
import numpy
import pandas
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.evaluation import object_based_evaluation as object_eval

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 1e-3
METRES2_TO_KM2 = 1e-6
NARR_TIME_INTERVAL_SECONDS = 10800

GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME)[0]
NUM_GRID_ROWS, NUM_GRID_COLUMNS = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_EVAL_TIME_ARG_NAME = 'first_eval_time_string'
LAST_EVAL_TIME_ARG_NAME = 'last_eval_time_string'
NUM_EVAL_TIMES_ARG_NAME = 'num_eval_times'
BINARIZATION_THRESHOLD_ARG_NAME = 'binarization_threshold'
MIN_AREA_ARG_NAME = 'min_object_area_metres2'
MIN_LENGTH_ARG_NAME = 'min_object_length_metres'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_metres'
FRONT_LINE_DIR_ARG_NAME = 'input_front_line_dir_name'
EVALUATION_DIR_ARG_NAME = 'output_evaluation_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of input directory, containing gridded predictions.  Files therein '
    'will be found by `machine_learning_utils.find_gridded_prediction_file` and'
    ' read by `machine_learning_utils.read_gridded_predictions`.')

EVAL_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Evaluation times will be randomly drawn from '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)

NUM_EVAL_TIMES_HELP_STRING = (
    'Number of evaluation times (to be drawn randomly from `{0:s}`...`{1:s}`).'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)

BINARIZATION_THRESHOLD_HELP_STRING = (
    'Threshold for discriminating between front and no front.  See doc for '
    '`object_based_evaluation.determinize_probabilities`.')

MIN_AREA_HELP_STRING = (
    'Minimum area for predicted frontal region (before skeletonization).  '
    'Smaller regions will be thrown out.')

MIN_LENGTH_HELP_STRING = (
    'Minimum length for skeleton line (predicted frontal polyline).  Shorter '
    'lines will be thrown out.')

MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (or neighbourhood distance).  If actual front f_A and '
    'predicted front f_P, both at time t, are of the same type and within '
    '`{0:s}` of each other, they are considered "matching".'
).format(MATCHING_DISTANCE_ARG_NAME)

FRONT_LINE_DIR_HELP_STRING = (
    'Name of top-level directory with actual fronts (polylines).  Files therein'
    ' will be found by `fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_polylines_from_file`.')

EVALUATION_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation results will be saved here.')

DEFAULT_MIN_AREA_METRES2 = 5e11  # 0.5 million km^2
DEFAULT_MIN_LENGTH_METRES = 5e5  # 500 km
DEFAULT_MATCHING_DISTANCE_METRES = 1e5
TOP_FRONT_LINE_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts/polylines'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EVAL_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_EVAL_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BINARIZATION_THRESHOLD_ARG_NAME, type=float, required=True,
    help=BINARIZATION_THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_AREA_ARG_NAME, type=float, required=False,
    default=DEFAULT_MIN_AREA_METRES2, help=MIN_AREA_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LENGTH_ARG_NAME, type=float, required=False,
    default=DEFAULT_MIN_LENGTH_METRES, help=MIN_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=False,
    default=DEFAULT_MATCHING_DISTANCE_METRES,
    help=MATCHING_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_LINE_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_LINE_DIR_NAME_DEFAULT, help=FRONT_LINE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_DIR_ARG_NAME, type=str, required=True,
    help=EVALUATION_DIR_HELP_STRING)


def _read_actual_polylines(top_input_dir_name, unix_times_sec):
    """Reads actual fronts (polylines) for each time step.

    :param top_input_dir_name: See documentation at top of file.
    :param unix_times_sec: 1-D numpy array of valid times.
    :return: polyline_table: See doc for `fronts_io.write_polylines_to_file`.
    """

    list_of_polyline_tables = []
    for this_time_unix_sec in unix_times_sec:
        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=this_time_unix_sec)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        list_of_polyline_tables.append(
            fronts_io.read_polylines_from_file(this_file_name)
        )
        if len(list_of_polyline_tables) == 1:
            continue

        list_of_polyline_tables[-1] = list_of_polyline_tables[-1].align(
            list_of_polyline_tables[0], axis=1)[0]

    return pandas.concat(list_of_polyline_tables, axis=0, ignore_index=True)


def _run(input_prediction_dir_name, first_eval_time_string,
         last_eval_time_string, num_eval_times, binarization_threshold,
         min_object_area_metres2, min_object_length_metres,
         matching_distance_metres, top_front_line_dir_name,
         output_eval_dir_name):
    """Evaluates CNN trained by patch classification.

    This is effectively the main method.

    :param input_prediction_dir_name: See documentation at top of file.
    :param first_eval_time_string: Same.
    :param last_eval_time_string: Same.
    :param num_eval_times: Same.
    :param binarization_threshold: Same.
    :param min_object_area_metres2: Same.
    :param min_object_length_metres: Same.
    :param matching_distance_metres: Same.
    :param top_front_line_dir_name: Same.
    :param output_eval_dir_name: Same.
    """

    first_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        first_eval_time_string, INPUT_TIME_FORMAT)
    last_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        last_eval_time_string, INPUT_TIME_FORMAT)
    possible_eval_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_eval_time_unix_sec,
        end_time_unix_sec=last_eval_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(possible_eval_times_unix_sec)
    eval_times_unix_sec = []

    list_of_predicted_region_tables = []
    num_times_done = 0

    for i in range(len(possible_eval_times_unix_sec)):
        if num_times_done == num_eval_times:
            break

        this_prediction_file_name = ml_utils.find_gridded_prediction_file(
            directory_name=input_prediction_dir_name,
            first_target_time_unix_sec=possible_eval_times_unix_sec[i],
            last_target_time_unix_sec=possible_eval_times_unix_sec[i],
            raise_error_if_missing=False)
        if not os.path.isfile(this_prediction_file_name):
            continue

        num_times_done += 1
        eval_times_unix_sec.append(possible_eval_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_prediction_file_name)
        this_prediction_dict = ml_utils.read_gridded_predictions(
            this_prediction_file_name)

        print 'Determinizing probabilities...'
        this_predicted_label_matrix = object_eval.determinize_probabilities(
            class_probability_matrix=this_prediction_dict[
                ml_utils.PROBABILITY_MATRIX_KEY],
            binarization_threshold=binarization_threshold)

        print 'Converting image to frontal regions...'
        list_of_predicted_region_tables.append(
            object_eval.images_to_regions(
                predicted_label_matrix=this_predicted_label_matrix,
                image_times_unix_sec=possible_eval_times_unix_sec[[i]])
        )

        print 'Throwing out frontal regions with area < {0:f} km^2...'.format(
            METRES2_TO_KM2 * min_object_area_metres2)
        list_of_predicted_region_tables[
            -1
        ] = object_eval.discard_regions_with_small_area(
            predicted_region_table=list_of_predicted_region_tables[-1],
            x_grid_spacing_metres=GRID_SPACING_METRES,
            y_grid_spacing_metres=GRID_SPACING_METRES,
            min_area_metres2=min_object_area_metres2)

        print 'Skeletonizing frontal regions...'
        list_of_predicted_region_tables[
            -1
        ] = object_eval.skeletonize_frontal_regions(
            predicted_region_table=list_of_predicted_region_tables[-1],
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        list_of_predicted_region_tables[-1] = object_eval.find_main_skeletons(
            predicted_region_table=list_of_predicted_region_tables[-1],
            class_probability_matrix=this_prediction_dict[
                ml_utils.PROBABILITY_MATRIX_KEY],
            image_times_unix_sec=possible_eval_times_unix_sec[[i]],
            x_grid_spacing_metres=GRID_SPACING_METRES,
            y_grid_spacing_metres=GRID_SPACING_METRES,
            min_length_metres=min_object_length_metres)

        if num_times_done != num_eval_times:
            print '\n'

        if len(list_of_predicted_region_tables) == 1:
            continue

        list_of_predicted_region_tables[-1] = (
            list_of_predicted_region_tables[-1].align(
                list_of_predicted_region_tables[0], axis=1)[0]
        )

    print SEPARATOR_STRING

    eval_times_unix_sec = numpy.array(eval_times_unix_sec, dtype=int)
    predicted_region_table = pandas.concat(
        list_of_predicted_region_tables, axis=0, ignore_index=True)
    predicted_region_table = object_eval.convert_regions_rowcol_to_narr_xy(
        predicted_region_table=predicted_region_table,
        are_predictions_from_fcn=False)

    actual_polyline_table = _read_actual_polylines(
        top_input_dir_name=top_front_line_dir_name,
        unix_times_sec=eval_times_unix_sec)
    print SEPARATOR_STRING

    actual_polyline_table = object_eval.project_polylines_latlng_to_narr(
        actual_polyline_table)

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

    evaluation_file_name = '{0:s}/object_based_evaluation.p'.format(
        output_eval_dir_name)
    print 'Writing results to: "{0:s}"...'.format(evaluation_file_name)

    object_eval.write_evaluation_results(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        neigh_distance_metres=matching_distance_metres,
        binary_contingency_table_as_dict=binary_contingency_table_as_dict,
        binary_pod=binary_pod, binary_success_ratio=binary_success_ratio,
        binary_csi=binary_csi, binary_frequency_bias=binary_frequency_bias,
        row_normalized_ct_as_matrix=row_normalized_ct_as_matrix,
        column_normalized_ct_as_matrix=column_normalized_ct_as_matrix,
        pickle_file_name=evaluation_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_eval_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_EVAL_TIME_ARG_NAME),
        last_eval_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_EVAL_TIME_ARG_NAME),
        num_eval_times=getattr(INPUT_ARG_OBJECT, NUM_EVAL_TIMES_ARG_NAME),
        binarization_threshold=getattr(
            INPUT_ARG_OBJECT, BINARIZATION_THRESHOLD_ARG_NAME),
        min_object_area_metres2=getattr(INPUT_ARG_OBJECT, MIN_AREA_ARG_NAME),
        min_object_length_metres=getattr(INPUT_ARG_OBJECT, MIN_LENGTH_ARG_NAME),
        matching_distance_metres=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME),
        top_front_line_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_LINE_DIR_ARG_NAME),
        output_eval_dir_name=getattr(INPUT_ARG_OBJECT, EVALUATION_DIR_ARG_NAME))
