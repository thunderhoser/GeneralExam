"""Runs object-based eval for traditional CNN, preferably on non-training data.

A "traditional CNN" is one for which the output (prediction) is not spatially
explicit.  The opposite is a fully convolutional net (see fcn.py).
"""

import os.path
import pickle
import argparse
import numpy
import pandas
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.evaluation import object_based_evaluation as object_based_eval

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'

NARR_TIME_INTERVAL_SECONDS = 10800
NARR_GRID_SPACING_METRES, _ = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME)
NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
FIRST_EVAL_TIME_ARG_NAME = 'first_eval_time_string'
LAST_EVAL_TIME_ARG_NAME = 'last_eval_time_string'
NUM_EVAL_TIMES_ARG_NAME = 'num_evaluation_times'
USE_ISOTONIC_REGRESSION_ARG_NAME = 'use_isotonic_regression'
BINARIZATION_THRESHOLD_ARG_NAME = 'binarization_threshold'
MIN_AREA_ARG_NAME = 'min_object_area_metres2'
MIN_LENGTH_ARG_NAME = 'min_object_length_metres'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_metres'
NARR_DIR_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
FRONTAL_POLYLINE_DIR_ARG_NAME = 'input_frontal_polyline_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to model file, containing a trained CNN.  This file should be '
    'readable by `traditional_cnn.read_keras_model`.')
EVAL_TIME_HELP_STRING = (
    'Evaluation time (format "yyyymmddHH").  The model will be evaluated for '
    'random evaluation times from `{0:s}`...`{1:s}`.').format(
        FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)
NUM_EVAL_TIMES_HELP_STRING = (
    'Number of evaluation times to be drawn randomly from the period '
    '`{0:s}`...`{1:s}`.').format(FIRST_EVAL_TIME_ARG_NAME,
                                 LAST_EVAL_TIME_ARG_NAME)
USE_ISOTONIC_REGRESSION_HELP_STRING = (
    'Boolean flag.  If 1, isotonic regression will be used to calibrate '
    'probabilities from the base model.')
BINARIZATION_THRESHOLD_HELP_STRING = (
    'Threshold for discriminating between front and no front.  See '
    'documentation for `object_based_evaluation.determinize_probabilities`.')
MIN_AREA_HELP_STRING = (
    'Minimum area of frontal region (BEFORE thinning into skeleton line).  '
    'Smaller regions will be thrown out.')
MIN_LENGTH_HELP_STRING = (
    'Minimum length of frontal region (AFTER thinning into skeleton line).  '
    'Smaller regions will be thrown out.')
MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (or neighbourhood distance).  If actual front f_A and '
    'predicted front f_P have the same type, occur at the same time, and are '
    'within `{0:s}` of each other, they will be considered "matching".').format(
        MATCHING_DISTANCE_ARG_NAME)
NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data (one file for each variable, '
    'pressure level, and time step).')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (one per time step, '
    'indicating which NARR grid cells are intersected by a front).')
FRONTAL_POLYLINE_DIR_HELP_STRING = (
    'Name of top-level directory with frontal polylines (one per time step).')
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation results will be saved here.')

DEFAULT_MIN_AREA_METRES2 = 2e11  # ~200 grid cells
DEFAULT_MIN_LENGTH_METRES = 5e5  # 500 km
DEFAULT_MATCHING_DISTANCE_METRES = 1e5
DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_FRONTAL_POLYLINE_DIR_NAME = '/condo/swatwork/ralager/fronts/polylines'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

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
    '--' + USE_ISOTONIC_REGRESSION_ARG_NAME, type=int, required=False,
    default=0, help=USE_ISOTONIC_REGRESSION_HELP_STRING)

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
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_FRONTAL_GRID_DIR_NAME, help=FRONTAL_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_POLYLINE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_FRONTAL_POLYLINE_DIR_NAME,
    help=FRONTAL_POLYLINE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _write_probability_image_one_time(
        class_probability_matrix, output_dir_name, valid_time_string):
    """Writes grid of predicted class probabilities at one time to Pickle file.

    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)
    K = number of classes = 3

    :param class_probability_matrix: 1-by-M-by-N-by-K numpy array of predicted
        class probabilities.
    :param output_dir_name: Path to output directory.
    :param valid_time_string: Valid time (format "yyyy-mm-dd-HH").
    """

    pickle_file_name = '{0:s}/class_probability_matrix_{1:s}.p'.format(
        output_dir_name, valid_time_string)
    print (
        'Writing grid of predicted class probabilities to file: "{0:s}"...'
    ).format(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(class_probability_matrix, pickle_file_handle)
    pickle_file_handle.close()


def _write_predicted_regions_one_time(
        predicted_region_table, output_dir_name, valid_time_string):
    """Writes predicted regions at one time step to Pickle file.

    :param predicted_region_table: pandas DataFrame.
    :param output_dir_name: Path to output directory.
    :param valid_time_string: Valid time (format "yyyy-mm-dd-HH").
    """

    pickle_file_name = '{0:s}/predicted_regions_{1:s}.p'.format(
        output_dir_name, valid_time_string)
    print 'Writing predicted regions to file: "{0:s}"...'.format(
        pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(predicted_region_table, pickle_file_handle)
    pickle_file_handle.close()


def _write_predicted_regions_time_period(
        predicted_region_table, output_dir_name, start_time_string,
        end_time_string):
    """Writes predicted regions for contiguous time period to Pickle file.

    :param predicted_region_table: pandas DataFrame.
    :param output_dir_name: Path to output directory.
    :param start_time_string: Start of time period (format "yyyy-mm-dd-HH").
    :param end_time_string: End of time period (format "yyyy-mm-dd-HH").
    """

    pickle_file_name = '{0:s}/predicted_regions_{1:s}_{2:s}.p'.format(
        output_dir_name, start_time_string, end_time_string)
    print 'Writing predicted regions to file: "{0:s}"...'.format(
        pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(predicted_region_table, pickle_file_handle)
    pickle_file_handle.close()


def _read_actual_polylines(top_input_dir_name, valid_times_unix_sec):
    """Reads actual fronts (polylines) for contiguous time period.

    :param top_input_dir_name: Name of top-level input directory.
    :param valid_times_unix_sec: 1-D numpy array of valid times.
    :return: actual_polyline_table: See documentation for
        `fronts_io.write_polylines_to_file`.
    """

    list_of_polyline_tables = []
    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=this_time_unix_sec)

        print 'Reading actual fronts from: "{0:s}"...'.format(this_file_name)
        list_of_polyline_tables.append(
            fronts_io.read_polylines_from_file(this_file_name))
        if len(list_of_polyline_tables) == 1:
            continue

        list_of_polyline_tables[-1], _ = list_of_polyline_tables[-1].align(
            list_of_polyline_tables[0], axis=1)

    return pandas.concat(list_of_polyline_tables, axis=0, ignore_index=True)


def _evaluate_model(
        model_file_name, first_eval_time_string, last_eval_time_string,
        num_evaluation_times, use_isotonic_regression, binarization_threshold,
        min_object_area_metres2, min_object_length_metres,
        matching_distance_metres, top_narr_directory_name,
        top_frontal_grid_dir_name, top_frontal_polyline_dir_name,
        output_dir_name):
    """Object-based eval for traditional CNN, preferably on non-training data.

    :param model_file_name: Path to model file, containing a trained CNN.  This
        file should be readable by `traditional_cnn.read_keras_model`.
    :param first_eval_time_string: Evaluation time (format "yyyymmddHH").  The
        model will be evaluated for random evaluation times from
        `first_eval_time_string`...`last_eval_time_string`.
    :param last_eval_time_string: See above.
    :param num_evaluation_times: Number of evaluation times to be drawn randomly
        from the period `first_eval_time_string`...`last_eval_time_string`.
    :param use_isotonic_regression: Boolean flag.  If 1, isotonic regression
        will be used to calibrate probabilities from the base model.
    :param binarization_threshold: Threshold for discriminating between front
        and no front.  See documentation for
        `object_based_evaluation.determinize_probabilities`.
    :param min_object_area_metres2: Minimum area of frontal region (BEFORE
        thinning into skeleton line).  Smaller regions will be thrown out.
    :param min_object_length_metres: Minimum length of frontal region (AFTER
        thinning into skeleton line).  Smaller regions will be thrown out.
    :param matching_distance_metres: Matching distance (or neighbourhood
        distance).  If actual front f_A and predicted front f_P have the same
        type, occur at the same time, and are within `matching_distance_metres`
        of each other, they will be considered "matching".
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one per time step, indicating which NARR grid cells are
        intersected by a front).
    :param top_frontal_polyline_dir_name: Name of top-level directory with
        frontal polylines (one per time step).
    :param output_dir_name: Name of output directory.  Evaluation results will
        be saved here.
    """

    first_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        first_eval_time_string, INPUT_TIME_FORMAT)
    last_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        last_eval_time_string, INPUT_TIME_FORMAT)

    evaluation_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_eval_time_unix_sec,
        end_time_unix_sec=last_eval_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(evaluation_times_unix_sec)
    evaluation_times_unix_sec = evaluation_times_unix_sec[:num_evaluation_times]
    evaluation_time_strings = [
        time_conversion.unix_sec_to_string(t, DEFAULT_TIME_FORMAT)
        for t in evaluation_times_unix_sec]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        model_directory_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metadata_file_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metadata_file_name)

    if use_isotonic_regression:
        isotonic_regression_file_name = (
            '{0:s}/isotonic_regression_models.p'.format(model_directory_name))

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_regression_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(
                isotonic_regression_file_name))
    else:
        isotonic_model_object_by_class = None

    print SEPARATOR_STRING

    list_of_predicted_region_tables = []
    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])

    for i in range(num_evaluation_times):
        print 'Generating model predictions for {0:s}...\n'.format(
            evaluation_time_strings[i])

        if model_metadata_dict[traditional_cnn.NUM_LEAD_TIME_STEPS_KEY] is None:
            this_class_probability_matrix, _ = (
                traditional_cnn.apply_model_to_3d_example(
                    model_object=model_object,
                    target_time_unix_sec=evaluation_times_unix_sec[i],
                    top_narr_directory_name=top_narr_directory_name,
                    top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                    narr_predictor_names=model_metadata_dict[
                        traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
                    pressure_level_mb=model_metadata_dict[
                        traditional_cnn.PRESSURE_LEVEL_KEY],
                    dilation_distance_for_target_metres=model_metadata_dict[
                        traditional_cnn.DILATION_DISTANCE_FOR_TARGET_KEY],
                    num_rows_in_half_grid=model_metadata_dict[
                        traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
                    num_columns_in_half_grid=model_metadata_dict[
                        traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
                    num_classes=num_classes,
                    isotonic_model_object_by_class=
                    isotonic_model_object_by_class))

        else:
            this_class_probability_matrix, _ = (
                traditional_cnn.apply_model_to_4d_example(
                    model_object=model_object,
                    target_time_unix_sec=evaluation_times_unix_sec[i],
                    predictor_time_step_offsets=model_metadata_dict[
                        traditional_cnn.PREDICTOR_TIME_STEP_OFFSETS_KEY],
                    num_lead_time_steps=model_metadata_dict[
                        traditional_cnn.NUM_LEAD_TIME_STEPS_KEY],
                    top_narr_directory_name=top_narr_directory_name,
                    top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                    narr_predictor_names=model_metadata_dict[
                        traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
                    pressure_level_mb=model_metadata_dict[
                        traditional_cnn.PRESSURE_LEVEL_KEY],
                    dilation_distance_for_target_metres=model_metadata_dict[
                        traditional_cnn.DILATION_DISTANCE_FOR_TARGET_KEY],
                    num_rows_in_half_grid=model_metadata_dict[
                        traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
                    num_columns_in_half_grid=model_metadata_dict[
                        traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
                    num_classes=num_classes,
                    isotonic_model_object_by_class=
                    isotonic_model_object_by_class))

        print SEPARATOR_STRING

        _write_probability_image_one_time(
            class_probability_matrix=this_class_probability_matrix,
            output_dir_name=output_dir_name,
            valid_time_string=evaluation_time_strings[i])

        print 'Determinizing probabilities for {0:s}...'.format(
            evaluation_time_strings[i])
        this_predicted_label_matrix = (
            object_based_eval.determinize_probabilities(
                class_probability_matrix=this_class_probability_matrix,
                binarization_threshold=binarization_threshold))

        print 'Converting image to region for {0:s}...'.format(
            evaluation_time_strings[i])
        this_predicted_region_table = object_based_eval.images_to_regions(
            predicted_label_matrix=this_predicted_label_matrix,
            image_times_unix_sec=evaluation_times_unix_sec[[i]])

        print 'Discarding regions with area < {0:f} m2...'.format(
            min_object_area_metres2)
        this_predicted_region_table = (
            object_based_eval.discard_regions_with_small_area(
                predicted_region_table=this_predicted_region_table,
                x_grid_spacing_metres=NARR_GRID_SPACING_METRES,
                y_grid_spacing_metres=NARR_GRID_SPACING_METRES,
                min_area_metres2=min_object_area_metres2))

        print 'Skeletonizing regions...'
        this_predicted_region_table = (
            object_based_eval.skeletonize_frontal_regions(
                predicted_region_table=this_predicted_region_table,
                num_grid_rows=NUM_ROWS_IN_NARR,
                num_grid_columns=NUM_COLUMNS_IN_NARR))

        this_predicted_region_table = object_based_eval.find_main_skeletons(
            predicted_region_table=this_predicted_region_table,
            class_probability_matrix=this_class_probability_matrix,
            image_times_unix_sec=evaluation_times_unix_sec[[i]])

        _write_predicted_regions_one_time(
            predicted_region_table=this_predicted_region_table,
            output_dir_name=output_dir_name,
            valid_time_string=evaluation_time_strings[i])
        print SEPARATOR_STRING

        list_of_predicted_region_tables.append(this_predicted_region_table)
        if len(list_of_predicted_region_tables) == 1:
            continue

        list_of_predicted_region_tables[-1], _ = (
            list_of_predicted_region_tables[-1].align(
                list_of_predicted_region_tables[0], axis=1))

    print 'Putting predicted regions for all time steps in one table...'
    predicted_region_table = pandas.concat(
        list_of_predicted_region_tables, axis=0, ignore_index=True)

    print 'Converting predicted regions from row-column to x-y coordinates...'
    predicted_region_table = (
        object_based_eval.convert_regions_rowcol_to_narr_xy(
            predicted_region_table, are_predictions_from_fcn=False))

    _write_predicted_regions_time_period(
        predicted_region_table=predicted_region_table,
        output_dir_name=output_dir_name,
        start_time_string=evaluation_time_strings[0],
        end_time_string=evaluation_time_strings[-1])
    print SEPARATOR_STRING

    actual_polyline_table = _read_actual_polylines(
        top_input_dir_name=top_frontal_polyline_dir_name,
        valid_times_unix_sec=evaluation_times_unix_sec)
    print SEPARATOR_STRING

    print 'Projecting actual fronts (polylines) from lat-long to x-y...'
    actual_polyline_table = object_based_eval.project_polylines_latlng_to_narr(
        actual_polyline_table)

    print ('Creating binary contingency table with matching distance of {0:f} '
           'metres...').format(matching_distance_metres)
    binary_contingency_table_as_dict = (
        object_based_eval.get_binary_contingency_table(
            predicted_region_table=predicted_region_table,
            actual_polyline_table=actual_polyline_table,
            neigh_distance_metres=matching_distance_metres))

    print 'Binary contingency table is shown below:\n{0:s}\n'.format(
        binary_contingency_table_as_dict)

    print 'Computing binary performance metrics...'
    binary_pod = object_based_eval.get_binary_pod(
        binary_contingency_table_as_dict)
    binary_success_ratio = object_based_eval.get_binary_success_ratio(
        binary_contingency_table_as_dict)
    binary_csi = object_based_eval.get_binary_csi(
        binary_contingency_table_as_dict)
    binary_frequency_bias = object_based_eval.get_binary_frequency_bias(
        binary_contingency_table_as_dict)

    print ('Binary POD = {0:.4f} ... success ratio = {1:.4f} ... CSI = {2:.4f} '
           '... frequency bias = {3:.4f}\n').format(
               binary_pod, binary_success_ratio, binary_csi,
               binary_frequency_bias)

    print 'Creating row-normalized contingency table...'
    row_normalized_ct_as_matrix = (
        object_based_eval.get_row_normalized_contingency_table(
            predicted_region_table=predicted_region_table,
            actual_polyline_table=actual_polyline_table,
            neigh_distance_metres=matching_distance_metres))

    print 'Row-normalized contingency table is shown below:\n{0:s}\n'.format(
        row_normalized_ct_as_matrix)

    print 'Creating column-normalized contingency table...'
    column_normalized_ct_as_matrix = (
        object_based_eval.get_column_normalized_contingency_table(
            predicted_region_table=predicted_region_table,
            actual_polyline_table=actual_polyline_table,
            neigh_distance_metres=matching_distance_metres))

    print 'Column-normalized contingency table is shown below:\n{0:s}\n'.format(
        row_normalized_ct_as_matrix)

    evaluation_file_name = '{0:s}/object_based_evaluation.p'.format(
        output_dir_name)
    print 'Writing evaluation results to: "{0:s}"...'.format(
        evaluation_file_name)

    object_based_eval.write_evaluation_results(
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
    MODEL_FILE_NAME = getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    FIRST_EVAL_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_EVAL_TIME_ARG_NAME)
    LAST_EVAL_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_EVAL_TIME_ARG_NAME)
    NUM_EVALUATION_TIMES = getattr(INPUT_ARG_OBJECT, NUM_EVAL_TIMES_ARG_NAME)
    USE_ISOTONIC_REGRESSION = bool(
        getattr(INPUT_ARG_OBJECT, USE_ISOTONIC_REGRESSION_ARG_NAME))

    BINARIZATION_THRESHOLD = getattr(
        INPUT_ARG_OBJECT, BINARIZATION_THRESHOLD_ARG_NAME)
    MIN_OBJECT_AREA_METRES2 = getattr(INPUT_ARG_OBJECT, MIN_AREA_ARG_NAME)
    MIN_OBJECT_LENGTH_METRES = getattr(INPUT_ARG_OBJECT, MIN_LENGTH_ARG_NAME)
    MATCHING_DISTANCE_METRES = getattr(
        INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME)

    TOP_NARR_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME)
    TOP_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)
    TOP_FRONTAL_POLYLINE_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_POLYLINE_DIR_ARG_NAME)
    OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)

    _evaluate_model(
        model_file_name=MODEL_FILE_NAME,
        first_eval_time_string=FIRST_EVAL_TIME_STRING,
        last_eval_time_string=LAST_EVAL_TIME_STRING,
        num_evaluation_times=NUM_EVALUATION_TIMES,
        use_isotonic_regression=USE_ISOTONIC_REGRESSION,
        binarization_threshold=BINARIZATION_THRESHOLD,
        min_object_area_metres2=MIN_OBJECT_AREA_METRES2,
        min_object_length_metres=MIN_OBJECT_LENGTH_METRES,
        matching_distance_metres=MATCHING_DISTANCE_METRES,
        top_narr_directory_name=TOP_NARR_DIRECTORY_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME,
        top_frontal_polyline_dir_name=TOP_FRONTAL_POLYLINE_DIR_NAME,
        output_dir_name=OUTPUT_DIR_NAME)
