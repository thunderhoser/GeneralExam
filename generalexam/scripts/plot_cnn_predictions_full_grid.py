"""Plots CNN predictions on full grid."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
TIME_INTERVAL_SECONDS = 10800

NUM_ROWS_IN_CNN_PATCH = plot_gridded_stats.NUM_ROWS_IN_CNN_PATCH
NUM_COLUMNS_IN_CNN_PATCH = plot_gridded_stats.NUM_COLUMNS_IN_CNN_PATCH

BORDER_COLOUR = numpy.full(3, 0.)
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
    'by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DETERMINISTIC_HELP_STRING = (
    'Boolean flag.  If 1, deterministic predictions will be plotted.  If 0, '
    'probabilities will be plotted.'
)
TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Predictions will be plotted for all times in '
    'the period `{0:s}`...`{1:s}` that are found in the input file(s).'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

FIRST_LETTER_HELP_STRING = (
    'Letter label for first time step.  If this is "a", the label "(a)" will be'
    ' printed at the top left of the figure.  If you do not want labels, leave '
    'this argument alone.'
)
LETTER_INTERVAL_HELP_STRING = (
    'Interval between letter labels for successive time steps.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DETERMINISTIC_ARG_NAME, type=int, required=False, default=0,
    help=DETERMINISTIC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_LETTER_ARG_NAME, type=str, required=False, default='',
    help=FIRST_LETTER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LETTER_INTERVAL_ARG_NAME, type=int, required=False, default=3,
    help=LETTER_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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
        basemap_dict = plot_gridded_stats.plot_basemap(
            data_matrix=predicted_label_matrix, border_colour=BORDER_COLOUR
        )
    else:
        basemap_dict = plot_gridded_stats.plot_basemap(
            data_matrix=class_probability_matrix[..., 0],
            border_colour=BORDER_COLOUR
        )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    if class_probability_matrix is None:
        matrix_to_plot = predicted_label_matrix[
            NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
            NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
        ]

        front_plotting.plot_labels_on_general_grid(
            label_matrix=matrix_to_plot,
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            axes_object=axes_object, basemap_object=basemap_object
        )
    else:
        wf_prob_matrix_to_plot = class_probability_matrix[
            NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
            NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH,
            front_utils.WARM_FRONT_ENUM
        ]

        wf_prob_matrix_to_plot[numpy.isnan(wf_prob_matrix_to_plot)] = 0.

        prediction_plotting.plot_probs_on_general_grid(
            probability_matrix=wf_prob_matrix_to_plot,
            front_string_id=front_utils.WARM_FRONT_STRING,
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            axes_object=axes_object, basemap_object=basemap_object
        )

        cf_prob_matrix_to_plot = class_probability_matrix[
            NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
            NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH,
            front_utils.COLD_FRONT_ENUM
        ]

        cf_prob_matrix_to_plot[numpy.isnan(cf_prob_matrix_to_plot)] = 0.

        prediction_plotting.plot_probs_on_general_grid(
            probability_matrix=cf_prob_matrix_to_plot,
            front_string_id=front_utils.COLD_FRONT_STRING,
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            axes_object=axes_object, basemap_object=basemap_object
        )

        if plot_wf_colour_bar:
            this_colour_map_object, this_colour_norm_object = (
                prediction_plotting.get_warm_front_colour_map()[:2]
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=wf_prob_matrix_to_plot,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', padding=0.05,
                extend_min=True, extend_max=False, fraction_of_axis_length=0.9
            )

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
                data_matrix=cf_prob_matrix_to_plot,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string=orientation_string,
                extend_min=True, extend_max=False, fraction_of_axis_length=0.9
            )

    # axes_object.set_title(title_string)

    if letter_label is not None:
        plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


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
        directory_name=output_dir_name
    )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

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
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_file_name,
            read_deterministic=plot_deterministic
        )

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
