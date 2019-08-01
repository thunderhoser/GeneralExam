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
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
TIME_INTERVAL_SECONDS = 10800

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)

# MIN_LATITUDE_DEG = 20.
# MIN_LONGITUDE_DEG = 220.
# MAX_LATITUDE_DEG = 80.
# MAX_LONGITUDE_DEG = 290.

MIN_LATITUDE_DEG = 5.
MIN_LONGITUDE_DEG = 200.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 310.

FIGURE_RESOLUTION_DPI = 300

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
DETERMINISTIC_ARG_NAME = 'plot_deterministic'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of directory with gridded predictions.  Files therein will be found '
    'by `prediction_io.find_file` and read by `prediction_io.read_file`.')

DETERMINISTIC_HELP_STRING = (
    'Boolean flag.  If 1, deterministic predictions will be plotted.  If 0, '
    'probabilities will be plotted.')

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
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DETERMINISTIC_ARG_NAME, type=int, required=False, default=0,
    help=DETERMINISTIC_HELP_STRING)

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

    if class_probability_matrix is None:
        num_grid_rows = predicted_label_matrix.shape[0]
        num_grid_columns = predicted_label_matrix.shape[1]
    else:
        num_grid_rows = class_probability_matrix.shape[0]
        num_grid_columns = class_probability_matrix.shape[1]

    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=num_grid_rows, num_columns=num_grid_columns)

    full_grid_row_limits, full_grid_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name)
    )

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        last_row_in_full_grid=full_grid_row_limits[1],
        first_column_in_full_grid=full_grid_column_limits[0],
        last_column_in_full_grid=full_grid_column_limits[1]
    )

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
        num_parallels=NUM_PARALLELS)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS)

    if class_probability_matrix is None:
        this_matrix = predicted_label_matrix[
            full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
            full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
        ]

        front_plotting.plot_gridded_labels(
            gridded_front_matrix=this_matrix, axes_object=axes_object,
            basemap_object=basemap_object, full_grid_name=full_grid_name,
            first_row_in_full_grid=full_grid_row_limits[0],
            first_column_in_full_grid=full_grid_column_limits[0], opacity=1.)
    else:
        this_wf_probability_matrix = class_probability_matrix[
            full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
            full_grid_column_limits[0]:(full_grid_column_limits[1] + 1),
            front_utils.WARM_FRONT_ENUM
        ]

        this_wf_probability_matrix[numpy.isnan(this_wf_probability_matrix)] = 0.

        prediction_plotting.plot_gridded_probs(
            probability_matrix=this_wf_probability_matrix,
            front_string_id=front_utils.WARM_FRONT_STRING,
            axes_object=axes_object, basemap_object=basemap_object,
            full_grid_name=full_grid_name,
            first_row_in_full_grid=full_grid_row_limits[0],
            first_column_in_full_grid=full_grid_column_limits[0], opacity=0.5)

        this_cf_probability_matrix = class_probability_matrix[
            full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
            full_grid_column_limits[0]:(full_grid_column_limits[1] + 1),
            front_utils.COLD_FRONT_ENUM
        ]

        this_cf_probability_matrix[numpy.isnan(this_cf_probability_matrix)] = 0.

        prediction_plotting.plot_gridded_probs(
            probability_matrix=this_cf_probability_matrix,
            front_string_id=front_utils.COLD_FRONT_STRING,
            axes_object=axes_object, basemap_object=basemap_object,
            full_grid_name=full_grid_name,
            first_row_in_full_grid=full_grid_row_limits[0],
            first_column_in_full_grid=full_grid_column_limits[0], opacity=0.5)

        if plot_wf_colour_bar:
            this_colour_map_object, this_colour_norm_object = (
                prediction_plotting.get_warm_front_colour_map()[:2]
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=this_wf_probability_matrix,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal',
                extend_min=True, extend_max=False, fraction_of_axis_length=0.9)

        if plot_cf_colour_bar:
            this_colour_map_object, this_colour_norm_object = (
                prediction_plotting.get_cold_front_colour_map()[:2]
            )

            if plot_wf_colour_bar:
                orientation_string = 'vertical'
            else:
                orientation_string = 'horizontal'

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=this_cf_probability_matrix,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string=orientation_string,
                extend_min=True, extend_max=False, fraction_of_axis_length=0.9)

    # pyplot.title(title_string)
    #
    # if letter_label is not None:
    #     plotting_utils.annotate_axes(
    #         axes_object=axes_object,
    #         annotation_string='({0:s})'.format(letter_label)
    #     )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(prediction_dir_name, plot_deterministic, first_time_string,
         last_time_string, first_letter_label, letter_interval,
         output_dir_name):
    """Plots CNN predictions on full grid.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param plot_deterministic: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    """

    if first_letter_label in ['', 'None']:
        first_letter_label = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    this_class_probability_matrix = None
    this_predicted_label_matrix = None
    this_letter_label = None

    plot_wf_colour_bar = False
    plot_cf_colour_bar = True
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        this_file_name = prediction_io.find_file(
            directory_name=prediction_dir_name,
            first_time_unix_sec=valid_times_unix_sec[i],
            last_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_file_name,
            read_deterministic=plot_deterministic)

        if plot_deterministic:
            this_predicted_label_matrix = this_prediction_dict[
                prediction_io.PREDICTED_LABELS_KEY
            ][0, ...]
        else:
            this_class_probability_matrix = this_prediction_dict[
                prediction_io.CLASS_PROBABILITIES_KEY
            ][0, ...]

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

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        plot_deterministic=bool(getattr(
            INPUT_ARG_OBJECT, DETERMINISTIC_ARG_NAME)),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
