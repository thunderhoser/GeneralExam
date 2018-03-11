"""Applies traditional CNN to the full NARR grid at one or more target times.

A "traditional CNN" is one for which the output (prediction) is not spatially
explicit.  The opposite is a fully convolutional net (see fcn.py).
"""

import os.path
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_utils import front_utils
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.plotting import narr_plotting
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

DOTS_PER_INCH = 300
FRONTAL_LINE_WIDTH = 4
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

NARR_TIME_INTERVAL_SEC = 10800
INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RANDOMIZE_TIMES_ARG_NAME = 'randomize_times'
NUM_TIMES_ARG_NAME = 'num_times'
USE_ISOTONIC_REGRESSION_ARG_NAME = 'use_isotonic_regression'
NARR_DIR_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
FRONTAL_POLYLINE_DIR_ARG_NAME = 'input_frontal_polyline_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to model file, containing a trained CNN.  This file should be '
    'readable by `traditional_cnn.read_keras_model`.')
TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  The model will be applied to target times in '
    'the period `{0:s}`...`{1:s}`.').format(
        FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)
RANDOMIZE_TIMES_HELP_STRING = (
    'Boolean flag.  If 1, the model will be applied only to random times drawn '
    'from `{0:s}`...`{1:s}`.  Otherwise, the model will be applied to all times'
    ' in order.').format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)
NUM_TIMES_HELP_STRING = (
    '[used only if `{0:s}` = 1] The model will be applied to this many random '
    'times from `{1:s}`...`{2:s}`.').format(
        RANDOMIZE_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)
USE_ISOTONIC_REGRESSION_HELP_STRING = (
    'Boolean flag.  If 1, isotonic regression will be used to calibrate '
    'probabilities from the base model.')
NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data (one file for each variable, '
    'pressure level, and time step).')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (one per file, indicating '
    'which NARR grid cells are intersected by a front).')
FRONTAL_POLYLINE_DIR_HELP_STRING = (
    'Name of top-level directory with frontal polylines (one file per time '
    'step).')

DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_FRONTAL_POLYLINE_DIR_NAME = '/condo/swatwork/ralager/fronts/polylines'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RANDOMIZE_TIMES_ARG_NAME, type=int, required=False, default=1,
    help=RANDOMIZE_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ISOTONIC_REGRESSION_ARG_NAME, type=int, required=False,
    default=0, help=USE_ISOTONIC_REGRESSION_ARG_NAME)

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


def _plot_one_time_step(
        class_probability_matrix, top_frontal_polyline_dir_name,
        valid_time_unix_sec, valid_time_string, output_dir_name):
    """Plots actual fronts and predicted probabilities for one time step.

    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)
    K = number of classes

    :param class_probability_matrix: M-by-N-by-K matrix of predicted
        probabilities.
    :param top_frontal_polyline_dir_name: Name of top-level directory with
        frontal polylines (one file per time step).
    :param valid_time_unix_sec: Valid time.
    :param valid_time_string: Valid time (format "yyyy-mm-dd-HH").
    :param output_dir_name: Path to output directory (figure will be saved
        here).
    """

    polyline_file_name = fronts_io.find_file_for_one_time(
        top_directory_name=top_frontal_polyline_dir_name,
        file_type=fronts_io.POLYLINE_FILE_TYPE,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading frontal polylines from: "{0:s}"...'.format(
        polyline_file_name)
    polyline_table = fronts_io.read_polylines_from_file(polyline_file_name)

    print 'Plotting actual fronts and predicted probabilities...'
    _, axes_object, basemap_object = narr_plotting.init_basemap()

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=plotting_utils.DEFAULT_COUNTRY_COLOUR)
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object)
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object)
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG)

    prediction_plotting.plot_narr_grid(
        probability_matrix=class_probability_matrix[
            ..., front_utils.WARM_FRONT_INTEGER_ID],
        front_string_id=front_utils.WARM_FRONT_STRING_ID,
        axes_object=axes_object, basemap_object=basemap_object)

    prediction_plotting.plot_narr_grid(
        probability_matrix=class_probability_matrix[
            ..., front_utils.COLD_FRONT_INTEGER_ID],
        front_string_id=front_utils.COLD_FRONT_STRING_ID,
        axes_object=axes_object, basemap_object=basemap_object)

    colour_map_object, colour_norm_object, _ = (
        prediction_plotting.get_warm_front_colour_map())

    plotting_utils.add_colour_bar(
        axes_object=axes_object,
        values_to_colour=class_probability_matrix[
            ..., front_utils.WARM_FRONT_INTEGER_ID],
        colour_map=colour_map_object, colour_norm_object=colour_norm_object,
        orientation='vertical', extend_min=False, extend_max=False)

    colour_map_object, colour_norm_object, _ = (
        prediction_plotting.get_cold_front_colour_map())

    plotting_utils.add_colour_bar(
        axes_object=axes_object,
        values_to_colour=class_probability_matrix[
            ..., front_utils.COLD_FRONT_INTEGER_ID],
        colour_map=colour_map_object, colour_norm_object=colour_norm_object,
        orientation='vertical', extend_min=False, extend_max=False)

    num_actual_fronts = len(polyline_table.index)
    for j in range(num_actual_fronts):
        front_plotting.plot_polyline(
            latitudes_deg=polyline_table[
                front_utils.LATITUDES_COLUMN].values[j],
            longitudes_deg=polyline_table[
                front_utils.LONGITUDES_COLUMN].values[j],
            basemap_object=basemap_object, axes_object=axes_object,
            front_type=polyline_table[front_utils.FRONT_TYPE_COLUMN].values[j],
            line_width=FRONTAL_LINE_WIDTH)

    figure_file_name = '{0:s}/predictions_{1:s}.jpg'.format(
        output_dir_name, valid_time_string)
    print 'Saving figure to: "{0:s}"...'.format(figure_file_name)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()


def _apply_model(
        model_file_name, first_time_string, last_time_string, randomize_times,
        num_evaluation_times, use_isotonic_regression, top_narr_directory_name,
        top_frontal_grid_dir_name, top_frontal_polyline_dir_name):
    """Applies traditional CNN to full NARR grid at one or more target times.

    :param model_file_name: Path to model file, containing a trained CNN.  This
        file should be readable by `traditional_cnn.read_keras_model`.
    :param first_time_string: Time (format "yyyymmddHH").  The model will be
        applied to target times in the period `first_time_string`...
        `last_time_string`.
    :param last_time_string: See above.
    :param randomize_times: Boolean flag.  If 1, the model will be applied only
        to random times drawn from `first_time_string`...`last_time_string`.
        Otherwise, the model will be applied to all times in order.
    :param num_evaluation_times: [used only if randomize_times = True]
        The model will be applied to this many random times from
        `first_time_string`...`last_time_string`.
    :param use_isotonic_regression: Boolean flag.  If True, isotonic regression
        will be used to calibrate probabilities from the base model.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one per file, indicating which NARR grid cells are intersected by
        a front).
    :param top_frontal_polyline_dir_name: Name of top-level directory with
        frontal polylines (one file per time step).
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SEC, include_endpoint=True)

    if randomize_times:
        error_checking.assert_is_leq_numpy_array(
            num_evaluation_times, len(target_times_unix_sec))
        numpy.random.shuffle(target_times_unix_sec)
        target_times_unix_sec = target_times_unix_sec[:num_evaluation_times]

    num_target_times = len(target_times_unix_sec)
    target_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in target_times_unix_sec]

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

    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])
    if model_metadata_dict[traditional_cnn.NUM_LEAD_TIME_STEPS_KEY] is None:
        num_dimensions_per_example = 3
    else:
        num_dimensions_per_example = 4

    for i in range(num_target_times):
        if num_dimensions_per_example == 3:
            this_class_probability_matrix, this_target_matrix = (
                traditional_cnn.apply_model_to_3d_example(
                    model_object=model_object,
                    target_time_unix_sec=target_times_unix_sec[i],
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
            this_class_probability_matrix, this_target_matrix = (
                traditional_cnn.apply_model_to_4d_example(
                    model_object=model_object,
                    target_time_unix_sec=target_times_unix_sec[i],
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
        this_class_probability_matrix = this_class_probability_matrix[0, ...]
        this_target_matrix = this_target_matrix[0, ...]

        this_prediction_file_name = '{0:s}/predictions_{1:s}.p'.format(
            model_directory_name, target_time_strings[i])
        print 'Writing predictions to file: "{0:s}"...'.format(
            this_prediction_file_name)

        pickle_file_handle = open(this_prediction_file_name, 'wb')
        pickle.dump(this_class_probability_matrix, pickle_file_handle)
        pickle.dump(this_target_matrix, pickle_file_handle)
        pickle_file_handle.close()

        _plot_one_time_step(
            class_probability_matrix=this_class_probability_matrix,
            top_frontal_polyline_dir_name=top_frontal_polyline_dir_name,
            valid_time_unix_sec=target_times_unix_sec[i],
            valid_time_string=target_time_strings[i],
            output_dir_name=model_directory_name)
        print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    MODEL_FILE_NAME = getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    FIRST_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME)
    LAST_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME)
    RANDOMIZE_TIMES_FLAG = bool(
        getattr(INPUT_ARG_OBJECT, RANDOMIZE_TIMES_ARG_NAME))
    USE_ISOTONIC_REGRESSION = bool(
        getattr(INPUT_ARG_OBJECT, USE_ISOTONIC_REGRESSION_ARG_NAME))

    if RANDOMIZE_TIMES_FLAG:
        NUM_EVALUATION_TIMES = getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME)
    else:
        NUM_EVALUATION_TIMES = None

    TOP_NARR_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME)
    TOP_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)
    TOP_FRONTAL_POLYLINE_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_POLYLINE_DIR_ARG_NAME)

    _apply_model(
        model_file_name=MODEL_FILE_NAME,
        first_time_string=FIRST_TIME_STRING,
        last_time_string=LAST_TIME_STRING,
        randomize_times=RANDOMIZE_TIMES_FLAG,
        num_evaluation_times=NUM_EVALUATION_TIMES,
        use_isotonic_regression=USE_ISOTONIC_REGRESSION,
        top_narr_directory_name=TOP_NARR_DIRECTORY_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME,
        top_frontal_polyline_dir_name=TOP_FRONTAL_POLYLINE_DIR_NAME)
