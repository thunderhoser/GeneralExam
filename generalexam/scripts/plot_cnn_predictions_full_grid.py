"""Plots CNN predictions on full grid."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import neigh_evaluation
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
TIME_INTERVAL_SECONDS = 10800

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300

PROBABILISTIC_DIR_ARG_NAME = 'input_probabilistic_dir_name'
DETERMINISTIC_FILE_ARG_NAME = 'input_deterministic_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PROBABILISTIC_DIR_HELP_STRING = (
    'Name of directory with gridded probabilities.  Files therein will be found'
    ' by `machine_learning_utils.find_gridded_prediction_file` and read by '
    '`machine_learning_utils.read_gridded_predictions`.  If this is empty, will'
    ' plot deterministic labels instead.')

DETERMINISTIC_FILE_HELP_STRING = (
    'Name of file with gridded deterministic labels.  Will be read by '
    '`neigh_evaluation.read_results`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Predictions will be plotted for all times in '
    'the period `{0:s}`...`{1:s}` that are found in the input file(s).'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

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
    '--' + PROBABILISTIC_DIR_ARG_NAME, type=str, required=False, default='',
    help=PROBABILISTIC_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DETERMINISTIC_FILE_ARG_NAME, type=str, required=False, default='',
    help=DETERMINISTIC_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

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
        title_string, letter_label, output_file_name,
        class_probability_matrix=None, predicted_label_matrix=None,
        plot_wf_colour_bar=True, plot_cf_colour_bar=True):
    """Plots predictions at one time.

    Either `class_probability_matrix` or `predicted_label_matrix` will be
    plotted -- not both.

    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param title_string: Title (will be placed above figure).
    :param letter_label: Letter label.  If this is "a", the label "(a)" will be
        printed at the top left of the figure.
    :param output_file_name: Path to output file.
    :param class_probability_matrix: M-by-N-by-3 numpy array of class
        probabilities.
    :param predicted_label_matrix: M-by-N numpy array of predicted labels
        (integers in `front_utils.VALID_FRONT_TYPE_ENUMS`).
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

    parallel_spacing_deg = numpy.round(
        (MAX_LATITUDE_DEG - MIN_LATITUDE_DEG) / (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = numpy.round(
        (MAX_LONGITUDE_DEG - MIN_LONGITUDE_DEG) / (NUM_MERIDIANS - 1)
    )

    parallel_spacing_deg = max([parallel_spacing_deg, 1.])
    meridian_spacing_deg = max([meridian_spacing_deg, 1.])

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
        parallel_spacing_deg=parallel_spacing_deg)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=meridian_spacing_deg)

    if class_probability_matrix is None:
        this_matrix = predicted_label_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1)
        ]

        front_plotting.plot_narr_grid(
            frontal_grid_matrix=this_matrix, axes_object=axes_object,
            basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0], opacity=1.)
    else:
        this_wf_probability_matrix = class_probability_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.WARM_FRONT_ENUM
        ]

        this_wf_probability_matrix[numpy.isnan(this_wf_probability_matrix)] = 0.

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_wf_probability_matrix,
            front_string_id=front_utils.WARM_FRONT_STRING,
            axes_object=axes_object, basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0], opacity=0.5)

        this_cf_probability_matrix = class_probability_matrix[
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.COLD_FRONT_ENUM
        ]

        this_cf_probability_matrix[numpy.isnan(this_cf_probability_matrix)] = 0.

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_cf_probability_matrix,
            front_string_id=front_utils.COLD_FRONT_STRING,
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

            if plot_wf_colour_bar:
                orientation_string = 'vertical'
            else:
                orientation_string = 'horizontal'

            plotting_utils.add_colour_bar(
                axes_object_or_list=axes_object,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                values_to_colour=this_cf_probability_matrix,
                orientation=orientation_string,
                extend_min=True, extend_max=False, fraction_of_axis_length=0.9)

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


def _run(probabilistic_dir_name, deterministic_file_name, first_time_string,
         last_time_string, first_letter_label, letter_interval,
         output_dir_name):
    """Plots CNN predictions on full grid.

    This is effectively the main method.

    :param probabilistic_dir_name: See documentation at top of file.
    :param deterministic_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    """

    if probabilistic_dir_name in ['', 'None']:
        probabilistic_dir_name = None
    else:
        deterministic_file_name = None

    if first_letter_label in ['', 'None']:
        first_letter_label = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT)

    if deterministic_file_name is None:
        valid_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=first_time_unix_sec,
            end_time_unix_sec=last_time_unix_sec,
            time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

        predicted_label_matrix = None
    else:
        print 'Reading data from: "{0:s}"...\n'.format(deterministic_file_name)
        evaluation_dict = neigh_evaluation.read_results(deterministic_file_name)

        valid_times_unix_sec = evaluation_dict[neigh_evaluation.VALID_TIMES_KEY]
        good_indices = numpy.where(numpy.logical_and(
            valid_times_unix_sec >= first_time_unix_sec,
            valid_times_unix_sec <= last_time_unix_sec
        ))[0]

        valid_times_unix_sec = valid_times_unix_sec[good_indices]
        predicted_label_matrix = evaluation_dict[
            neigh_evaluation.PREDICTED_LABELS_KEY
        ][good_indices, ...]

        del evaluation_dict

    this_class_probability_matrix = None
    this_predicted_label_matrix = None
    this_letter_label = None

    plot_wf_colour_bar = False
    plot_cf_colour_bar = True
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        if deterministic_file_name is None:
            this_file_name = ml_utils.find_gridded_prediction_file(
                directory_name=probabilistic_dir_name,
                first_target_time_unix_sec=valid_times_unix_sec[i],
                last_target_time_unix_sec=valid_times_unix_sec[i],
                raise_error_if_missing=False)

            if not os.path.isfile(this_file_name):
                continue

            print 'Reading data from: "{0:s}"...'.format(this_file_name)
            this_class_probability_matrix = ml_utils.read_gridded_predictions(
                this_file_name
            )[ml_utils.PROBABILITY_MATRIX_KEY][0, ...]
        else:
            this_predicted_label_matrix = predicted_label_matrix[i, ...]

        this_title_string = 'CNN predictions at {0:s}'.format(
            time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], NICE_TIME_FORMAT)
        )

        this_output_file_name = '{0:s}/predictions_{1:s}.jpg'.format(
            output_dir_name,
            time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], DEFAULT_TIME_FORMAT)
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

        # _plot_one_time(
        #     title_string=this_title_string, letter_label=this_letter_label,
        #     output_file_name=this_output_file_name,
        #     class_probability_matrix=this_class_probability_matrix,
        #     predicted_label_matrix=this_predicted_label_matrix,
        #     plot_wf_colour_bar=plot_wf_colour_bar,
        #     plot_cf_colour_bar=plot_cf_colour_bar)

        _plot_one_time(
            title_string=this_title_string, letter_label=this_letter_label,
            output_file_name=this_output_file_name,
            class_probability_matrix=this_class_probability_matrix,
            predicted_label_matrix=this_predicted_label_matrix,
            plot_wf_colour_bar=True, plot_cf_colour_bar=True)

        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        probabilistic_dir_name=getattr(
            INPUT_ARG_OBJECT, PROBABILISTIC_DIR_ARG_NAME),
        deterministic_file_name=getattr(
            INPUT_ARG_OBJECT, DETERMINISTIC_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
