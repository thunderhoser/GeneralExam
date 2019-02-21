"""Plots predictions on full NARR grid.

This script plots both gridded and object-based predictions, from either a CNN
or an NFA model.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import nfa
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.evaluation import object_based_evaluation as object_eval
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
NARR_TIME_INTERVAL_SEC = 10800

CNN_METHOD_NAME = 'cnn'
NFA_METHOD_NAME = 'nfa'
VALID_METHOD_NAMES = [CNN_METHOD_NAME, NFA_METHOD_NAME]

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.
BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300

GRID_DIR_ARG_NAME = 'input_grid_dir_name'
OBJECT_FILE_ARG_NAME = 'input_object_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
METHOD_ARG_NAME = 'method_name'
USE_ENSEMBLE_ARG_NAME = 'use_nfa_ensemble'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRID_DIR_HELP_STRING = (
    'Name of directory with gridded predictions.  Files therein will be either '
    '(1) found by `machine_learning_utils.find_gridded_prediction_file` and '
    'read by `machine_learning_utils.read_gridded_predictions`; or (2) found by'
    ' `nfa.find_prediction_file` and read by `nfa.read_gridded_predictions` or '
    '`nfa.read_ensembled_predictions`.')

OBJECT_FILE_HELP_STRING = (
    'Name of file with object-based predictions.  Will be read by '
    '`object_based_evaluation.read_predictions_and_obs`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Predictions will be plotted for all times '
    'from `{0:s}`...`{1:s}` that are found in `{2:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME, OBJECT_FILE_ARG_NAME)

METHOD_HELP_STRING = (
    'Method used to generate predictions.  Must be in the following list:'
    '\n{0:s}'
).format(str(VALID_METHOD_NAMES))

USE_ENSEMBLE_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Boolean flag.  If 1, gridded predictions '
    'will be probabilities from an NFA ensemble.  If 0, gridded predictions '
    'will be deterministic labels from a single NFA method.'
).format(METHOD_ARG_NAME, NFA_METHOD_NAME)

FIRST_LETTER_HELP_STRING = (
    'Letter label for first time step.  If this is "a", the label "(a)" will be'
    ' printed at the top left of the figure.  If you do not want labels, leave '
    'this argument alone.')

LETTER_INTERVAL_HELP_STRING = (
    'Interval between letter labels for successive time steps.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_DIR_ARG_NAME, type=str, required=True,
    help=GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OBJECT_FILE_ARG_NAME, type=str, required=True,
    help=OBJECT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + METHOD_ARG_NAME, type=str, required=True, help=METHOD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ENSEMBLE_ARG_NAME, type=int, required=False, default=0,
    help=USE_ENSEMBLE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_LETTER_ARG_NAME, type=str, required=False, default='',
    help=FIRST_LETTER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LETTER_INTERVAL_ARG_NAME, type=int, required=False, default=3,
    help=LETTER_INTERVAL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_time(
        predicted_region_table, title_string, letter_label, output_file_name,
        class_probability_matrix=None, predicted_label_matrix=None,
        plot_wf_colour_bar=True, plot_cf_colour_bar=True):
    """Plots predictions at one time.

    Either `class_probability_matrix` or `predicted_label_matrix` will be
    plotted -- not both.

    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param predicted_region_table: Subset of pandas DataFrame returned by
        `object_eval.read_predictions_and_obs`, containing predicted fronts at
        only one time.
    :param title_string: Title (will be placed above figure).
    :param letter_label: Letter label.  If this is "a", the label "(a)" will be
        printed at the top left of the figure.
    :param output_file_name: Path to output file.
    :param class_probability_matrix: M-by-N-by-3 numpy array of class
        probabilities.
    :param predicted_label_matrix: M-by-N numpy array of predicted labels
        (integers in `front_utils.VALID_INTEGER_IDS`).
    :param plot_wf_colour_bar: Boolean flag.  If True, will plot colour bar for
        warm-front probability.
    :param plot_cf_colour_bar: Boolean flag.  If True, will plot colour bar for
        cold-front probability.
    """

    narr_row_limits, narr_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        first_row_in_full_grid=narr_row_limits[0],
        last_row_in_full_grid=narr_row_limits[1],
        first_column_in_full_grid=narr_column_limits[0],
        last_column_in_full_grid=narr_column_limits[1])

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG)

    if class_probability_matrix is None:
        this_matrix = predicted_label_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1)
        ]

        front_plotting.plot_narr_grid(
            frontal_grid_matrix=this_matrix, axes_object=axes_object,
            basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0], opacity=0.25)
    else:
        this_wf_probability_matrix = class_probability_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.WARM_FRONT_INTEGER_ID
        ]
        this_wf_probability_matrix[numpy.isnan(this_wf_probability_matrix)] = 0.

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_wf_probability_matrix,
            front_string_id=front_utils.WARM_FRONT_STRING_ID,
            axes_object=axes_object, basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0], opacity=0.5)

        this_cf_probability_matrix = class_probability_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.COLD_FRONT_INTEGER_ID
        ]
        this_cf_probability_matrix[numpy.isnan(this_cf_probability_matrix)] = 0.

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_cf_probability_matrix,
            front_string_id=front_utils.COLD_FRONT_STRING_ID,
            axes_object=axes_object, basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0], opacity=0.5)

        if plot_wf_colour_bar:
            this_colour_map_object, this_colour_norm_object = (
                prediction_plotting.get_warm_front_colour_map()[:2]
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=axes_object,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                values_to_colour=this_wf_probability_matrix,
                orientation='horizontal', extend_min=True, extend_max=False,
                fraction_of_axis_length=0.9)

        if plot_cf_colour_bar:
            this_colour_map_object, this_colour_norm_object = (
                prediction_plotting.get_cold_front_colour_map()[:2]
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=axes_object,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                values_to_colour=this_cf_probability_matrix,
                orientation='horizontal', extend_min=True, extend_max=False,
                fraction_of_axis_length=0.9)

    narr_latitude_matrix_deg, narr_longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    num_objects = len(predicted_region_table.index)

    for i in range(num_objects):
        these_rows = predicted_region_table[
            object_eval.ROW_INDICES_COLUMN].values[i]
        these_columns = predicted_region_table[
            object_eval.COLUMN_INDICES_COLUMN].values[i]

        front_plotting.plot_polyline(
            latitudes_deg=narr_latitude_matrix_deg[these_rows, these_columns],
            longitudes_deg=narr_longitude_matrix_deg[these_rows, these_columns],
            axes_object=axes_object, basemap_object=basemap_object,
            front_type=predicted_region_table[
                front_utils.FRONT_TYPE_COLUMN].values[i],
            line_width=4)

    # predicted_object_matrix = object_eval.regions_to_images(
    #     predicted_region_table=predicted_region_table,
    #     num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)
    #
    # this_matrix = predicted_object_matrix[
    #     0,
    #     narr_row_limits[0]:(narr_row_limits[1] + 1),
    #     narr_column_limits[0]:(narr_column_limits[1] + 1)
    # ]
    #
    # front_plotting.plot_narr_grid(
    #     frontal_grid_matrix=this_matrix, axes_object=axes_object,
    #     basemap_object=basemap_object,
    #     first_row_in_narr_grid=narr_row_limits[0],
    #     first_column_in_narr_grid=narr_column_limits[0], opacity=1.)

    pyplot.title(title_string)
    if letter_label is not None:
        plotting_utils.annotate_axes(
            axes_object=axes_object,
            annotation_string='({0:s})'.format(letter_label)
        )

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(input_grid_dir_name, input_object_file_name, first_time_string,
         last_time_string, method_name, use_nfa_ensemble, first_letter_label,
         letter_interval, output_dir_name):
    """Plots predictions on full NARR grid.

    This is effectively the main method.

    :param input_grid_dir_name: See documentation at top of file.
    :param input_object_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param method_name: Same.
    :param use_nfa_ensemble: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `method_name not in VALID_METHOD_NAMES`.
    """

    if first_letter_label in ['', 'None']:
        first_letter_label = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if method_name not in VALID_METHOD_NAMES:
        error_string = (
            '\n{0:s}\nValid method names (listed above) do not include "{1:s}".'
        ).format(str(VALID_METHOD_NAMES), method_name)

        raise ValueError(error_string)

    print 'Reading data from: "{0:s}"...'.format(input_object_file_name)
    predicted_region_table = object_eval.read_predictions_and_obs(
        input_object_file_name
    )[0]

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT)

    # region_times_unix_sec = predicted_region_table[
    #     front_utils.TIME_COLUMN].values
    #
    # good_indices = numpy.where(numpy.logical_and(
    #     region_times_unix_sec >= first_time_unix_sec,
    #     region_times_unix_sec <= last_time_unix_sec
    # ))[0]
    #
    # predicted_region_table = predicted_region_table.iloc[good_indices]

    predicted_region_table = predicted_region_table.loc[
        (predicted_region_table[front_utils.TIME_COLUMN] >= first_time_unix_sec)
        &
        (predicted_region_table[front_utils.TIME_COLUMN] <= last_time_unix_sec)
    ]

    valid_times_unix_sec = numpy.unique(
        predicted_region_table[front_utils.TIME_COLUMN].values
    )

    this_class_probability_matrix = None
    this_label_matrix = None
    this_letter_label = None

    plot_wf_colour_bar = False
    plot_cf_colour_bar = True

    for this_time_unix_sec in valid_times_unix_sec:
        if method_name == CNN_METHOD_NAME:
            this_file_name = ml_utils.find_gridded_prediction_file(
                directory_name=input_grid_dir_name,
                first_target_time_unix_sec=this_time_unix_sec,
                last_target_time_unix_sec=this_time_unix_sec)

            print 'Reading data from: "{0:s}"...'.format(this_file_name)
            this_class_probability_matrix = ml_utils.read_gridded_predictions(
                this_file_name
            )[ml_utils.PROBABILITY_MATRIX_KEY][0, ...]

        else:
            this_file_name = nfa.find_prediction_file(
                directory_name=input_grid_dir_name,
                first_valid_time_unix_sec=this_time_unix_sec,
                last_valid_time_unix_sec=this_time_unix_sec,
                ensembled=use_nfa_ensemble)

            print 'Reading data from: "{0:s}"...'.format(this_file_name)

            if use_nfa_ensemble:
                this_class_probability_matrix = nfa.read_ensembled_predictions(
                    this_file_name
                )[nfa.CLASS_PROBABILITIES_KEY][0, ...]
            else:
                this_label_matrix = nfa.read_gridded_predictions(
                    this_file_name
                )[0][0, ...]

        this_predicted_region_table = predicted_region_table.loc[
            predicted_region_table[front_utils.TIME_COLUMN] ==
            this_time_unix_sec
            ]

        this_title_string = '{0:s} predictions at {1:s}'.format(
            method_name.upper(),
            time_conversion.unix_sec_to_string(
                this_time_unix_sec, NICE_TIME_FORMAT)
        )

        this_output_file_name = '{0:s}/predictions_{1:s}.jpg'.format(
            output_dir_name,
            time_conversion.unix_sec_to_string(
                this_time_unix_sec, DEFAULT_TIME_FORMAT)
        )

        if first_letter_label is not None:
            if this_letter_label is None:
                this_letter_label = first_letter_label
            else:
                this_letter_label = chr(
                    ord(this_letter_label) + letter_interval
                )

        plot_wf_colour_bar = not plot_wf_colour_bar
        plot_cf_colour_bar = not plot_cf_colour_bar

        _plot_one_time(
            predicted_region_table=this_predicted_region_table,
            title_string=this_title_string, letter_label=this_letter_label,
            output_file_name=this_output_file_name,
            class_probability_matrix=this_class_probability_matrix,
            predicted_label_matrix=this_label_matrix,
            plot_wf_colour_bar=plot_wf_colour_bar,
            plot_cf_colour_bar=plot_cf_colour_bar)

        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_grid_dir_name=getattr(INPUT_ARG_OBJECT, GRID_DIR_ARG_NAME),
        input_object_file_name=getattr(INPUT_ARG_OBJECT, OBJECT_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        method_name=getattr(INPUT_ARG_OBJECT, METHOD_ARG_NAME),
        use_nfa_ensemble=bool(getattr(
            INPUT_ARG_OBJECT, USE_ENSEMBLE_ARG_NAME)),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
