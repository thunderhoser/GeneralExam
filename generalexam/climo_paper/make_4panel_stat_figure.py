"""Makes 4-panel figure with mean WF or CF length in each season."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

METRES_TO_KM = 0.001
MASK_IF_NUM_LABELS_BELOW = 100

FIRST_TIME_UNIX_SEC = time_conversion.first_and_last_times_in_year(1979)[0]
LAST_TIME_UNIX_SEC = (
    time_conversion.first_and_last_times_in_year(2018)[1] - 10799
)

SEASON_ABBREV_TO_VERBOSE_DICT = {
    climo_utils.WINTER_STRING: 'winter',
    climo_utils.SPRING_STRING: 'spring',
    climo_utils.SUMMER_STRING: 'summer',
    climo_utils.FALL_STRING: 'fall'
}

MAX_WF_LENGTH_KM = 1500
MAX_CF_LENGTH_KM = 3000
# COLOUR_MAP_OBJECT = pyplot.get_cmap('PuBuGn')
COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')

TITLE_FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_statistic_dir_name'
FRONT_TYPE_ARG_NAME = 'front_type_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_stats`.'
)

FRONT_TYPE_HELP_STRING = (
    'Front type.  Must be in the following list:\n{0:s}'
).format(
    str(front_utils.VALID_FRONT_TYPE_STRINGS)
)

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_TYPE_ARG_NAME, type=str, required=True,
    help=FRONT_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _plot_one_statistic(
        statistic_matrix, max_colour_value, plot_latitudes, plot_longitudes,
        plot_colour_bar, title_string, letter_label, output_file_name):
    """Plots one statistic for one season.

    :param statistic_matrix: 2-D numpy array with value at each grid cell.
    :param max_colour_value: Max value in colour scheme.
    :param plot_latitudes: Boolean flag.  Determines whether or not numbers will
        be plotted on y-axis.
    :param plot_longitudes: Boolean flag.  Determines whether or not numbers
        will be plotted on x-axis.
    :param plot_colour_bar: Boolean flag.  Determines whether or not colour bar
        will be plotted below.
    :param title_string: Title.
    :param letter_label: Letter label (will appear at top-left of panel).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    basemap_dict = plot_gridded_stats.plot_basemap(data_matrix=statistic_matrix)

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    matrix_to_plot = basemap_dict[plot_gridded_stats.MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=matrix_to_plot,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )

    if not plot_latitudes:
        axes_object.set_yticklabels([])
    if not plot_longitudes:
        axes_object.set_xticklabels([])

    if plot_colour_bar:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
            colour_map_object=COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', padding=0.05,
            extend_min=False, extend_max=True, fraction_of_axis_length=1.
        )

        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)
        colour_bar_object.ax.set_xticklabels(tick_values)

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
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


def _run(statistic_dir_name, front_type_string, output_file_name):
    """Makes 8-panel figure with gridded front statistics.

    This is effectively the main method.

    :param statistic_dir_name: See documentation at top of file.
    :param front_type_string: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    front_utils.check_front_type_string(front_type_string)

    season_strings_abbrev = climo_utils.VALID_SEASON_STRINGS
    season_strings_verbose = [
        SEASON_ABBREV_TO_VERBOSE_DICT[a] for a in season_strings_abbrev
    ]

    output_dir_name, pathless_output_file_name = os.path.split(output_file_name)
    extensionless_output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, os.path.splitext(pathless_output_file_name)[0]
    )

    num_seasons = len(season_strings_abbrev)
    panel_file_names = []
    letter_label = None

    for i in range(num_seasons):
        this_file_name = climo_utils.find_aggregated_file(
            directory_name=statistic_dir_name,
            file_type_string=climo_utils.FRONT_STATS_STRING,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            months=climo_utils.season_to_months(season_strings_abbrev[i]),
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_statistic_dict = climo_utils.read_gridded_stats(this_file_name)

        if front_type_string == front_utils.WARM_FRONT_STRING:
            front_type_abbrev = 'wf'
            max_colour_value = MAX_WF_LENGTH_KM

            this_num_labels_matrix = (
                this_statistic_dict[climo_utils.NUM_WF_LABELS_KEY]
            )
            this_length_matrix_km = (
                this_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY] *
                METRES_TO_KM
            )
        else:
            front_type_abbrev = 'cf'
            max_colour_value = MAX_CF_LENGTH_KM

            this_num_labels_matrix = (
                this_statistic_dict[climo_utils.NUM_CF_LABELS_KEY]
            )
            this_length_matrix_km = (
                this_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY] *
                METRES_TO_KM
            )

        this_length_matrix_km[
            this_num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
        ] = numpy.nan

        this_title_string = 'Mean {0:s} length (km) in {1:s}'.format(
            front_type_abbrev.upper(), season_strings_verbose[i]
        )
        this_output_file_name = '{0:s}_{1:s}_{2:s}.jpg'.format(
            extensionless_output_file_name, front_type_abbrev,
            season_strings_abbrev[i]
        )
        panel_file_names.append(this_output_file_name)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _plot_one_statistic(
            statistic_matrix=this_length_matrix_km,
            max_colour_value=max_colour_value,
            plot_latitudes=True, plot_longitudes=i == num_seasons - 1,
            plot_colour_bar=i >= num_seasons - 2,
            title_string=this_title_string, letter_label=letter_label,
            output_file_name=panel_file_names[-1]
        )

    print('Concatenating panels to: "{0:s}"...'.format(output_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=output_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for this_file_name in panel_file_names:
        print('Removing temporary file "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        statistic_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        front_type_string=getattr(INPUT_ARG_OBJECT, FRONT_TYPE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
