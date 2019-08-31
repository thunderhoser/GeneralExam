"""Makes 8-panel figure with Mann-Kendall results for either freq or length."""

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
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

METRES_TO_KM = 0.001
FRACTION_TO_PERCENT = 100.
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

SIG_MARKER_TYPE = plot_gridded_stats.SIG_MARKER_TYPE
SIG_MARKER_COLOUR = plot_gridded_stats.SIG_MARKER_COLOUR
SIG_MARKER_SIZE = 1.
SIG_MARKER_EDGE_WIDTH = plot_gridded_stats.SIG_MARKER_EDGE_WIDTH

MAX_WF_FREQ_TREND_PERCENT_YEAR01 = 0.06
MAX_CF_FREQ_TREND_PERCENT_YEAR01 = 0.07
MAX_WF_LENGTH_TREND_KM_YEAR01 = 15.
MAX_CF_LENGTH_TREND_KM_YEAR01 = 20.
COLOUR_MAP_OBJECT = pyplot.get_cmap('bwr')

BORDER_COLOUR = numpy.full(3, 152. / 255)
TITLE_FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_dir_name'
PLOT_FREQUENCY_ARG_NAME = 'plot_frequency'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_mann_kendall_file` and read by '
    '`climatology_utils.read_mann_kendall_test`.')

PLOT_FREQUENCY_HELP_STRING = (
    'Boolean flag.  If 1, will plot trends for WF and CF frequency.  If 0, will'
    ' plot trends for WF and CF length.')

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FREQUENCY_ARG_NAME, type=int, required=True,
    help=PLOT_FREQUENCY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _plot_one_trend(
        trend_matrix_year01, significance_matrix, max_colour_value,
        plot_latitudes, plot_longitudes, plot_colour_bar, title_string,
        letter_label, output_file_name):
    """Plots trend for one front type in one season.

    M = number of rows in grid
    N = number of columns in grid

    :param trend_matrix_year01: M-by-N numpy array with linear trend (per year)
        at each grid cell.
    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where trend is significant.
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

    basemap_dict = plot_gridded_stats._plot_basemap(
        trend_matrix_year01, border_colour=BORDER_COLOUR)

    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    full_grid_name = basemap_dict[plot_gridded_stats.FULL_GRID_NAME_KEY]
    full_grid_row_limits = basemap_dict[plot_gridded_stats.FULL_GRID_ROWS_KEY]
    full_grid_column_limits = basemap_dict[
        plot_gridded_stats.FULL_GRID_COLUMNS_KEY]

    trend_matrix_to_plot_year01 = trend_matrix_year01[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    sig_matrix_to_plot = significance_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    colour_norm_object = pyplot.Normalize(
        vmin=-max_colour_value, vmax=max_colour_value)

    prediction_plotting.plot_gridded_counts(
        count_or_frequency_matrix=trend_matrix_to_plot_year01,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object, full_grid_name=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    significant_rows, significant_columns = numpy.where(sig_matrix_to_plot)
    significant_y_coords = (
        (significant_rows + 0.5) / sig_matrix_to_plot.shape[0]
    )
    significant_x_coords = (
        (significant_columns + 0.5) / sig_matrix_to_plot.shape[1]
    )

    axes_object.plot(
        significant_x_coords, significant_y_coords, linestyle='None',
        marker=SIG_MARKER_TYPE, markersize=SIG_MARKER_SIZE,
        markerfacecolor=SIG_MARKER_COLOUR, markeredgecolor=SIG_MARKER_COLOUR,
        markeredgewidth=SIG_MARKER_EDGE_WIDTH, transform=axes_object.transAxes)

    if not plot_latitudes:
        axes_object.set_yticklabels([])
    if not plot_longitudes:
        axes_object.set_xticklabels([])

    if plot_colour_bar:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=trend_matrix_to_plot_year01,
            colour_map_object=COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', extend_min=True, extend_max=True,
            fraction_of_axis_length=0.9)

        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)

        if numpy.all(numpy.absolute(tick_values) < 1):
            tick_strings = ['{0:.3f}'.format(x) for x in tick_values]
        else:
            tick_strings = [
                '{0:d}'.format(int(numpy.round(x))) for x in tick_values
            ]

        colour_bar_object.ax.set_xticklabels(tick_strings)

    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object,
        label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(top_input_dir_name, plot_frequency, output_file_name):
    """Makes 8-panel figure with Mann-Kendall results for either freq or length.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param plot_frequency: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    season_strings_abbrev = climo_utils.VALID_SEASON_STRINGS
    season_strings_verbose = [
        SEASON_ABBREV_TO_VERBOSE_DICT[a] for a in season_strings_abbrev
    ]

    if plot_frequency:
        conversion_ratio = FRACTION_TO_PERCENT
        property_names = [
            climo_utils.WF_FREQ_PROPERTY_NAME, climo_utils.CF_FREQ_PROPERTY_NAME
        ]
    else:
        conversion_ratio = METRES_TO_KM
        property_names = [
            climo_utils.WF_LENGTH_PROPERTY_NAME,
            climo_utils.CF_LENGTH_PROPERTY_NAME
        ]

    front_type_abbrevs = ['wf', 'cf']

    output_dir_name, pathless_output_file_name = os.path.split(output_file_name)
    extensionless_output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, os.path.splitext(pathless_output_file_name)[0]
    )

    num_seasons = len(season_strings_abbrev)
    num_properties = len(property_names)
    panel_file_names = []
    letter_label = None

    for i in range(num_seasons):
        for j in range(num_properties):
            this_input_dir_name = '{0:s}/{1:s}'.format(
                top_input_dir_name, season_strings_verbose[i]
            )

            this_file_name = climo_utils.find_mann_kendall_file(
                directory_name=this_input_dir_name,
                property_name=property_names[j],
                raise_error_if_missing=True)

            print('Reading data from file: "{0:s}"...'.format(this_file_name))
            this_mann_kendall_dict = climo_utils.read_mann_kendall_test(
                this_file_name)

            this_num_labels_matrix = this_mann_kendall_dict[
                climo_utils.NUM_LABELS_MATRIX_KEY]
            this_trend_matrix_year01 = (
                conversion_ratio *
                this_mann_kendall_dict[climo_utils.TREND_MATRIX_KEY]
            )
            this_significance_matrix = this_mann_kendall_dict[
                climo_utils.SIGNIFICANCE_MATRIX_KEY]

            this_trend_matrix_year01[
                this_num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
            ] = numpy.nan
            this_significance_matrix[
                this_num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
            ] = False

            if plot_frequency:
                this_title_string = '{0:s}-frequency trend in {1:s}'.format(
                    front_type_abbrevs[j].upper(), season_strings_verbose[i]
                )
            else:
                this_title_string = (
                    '{0:s}-length trend (km per year) in {1:s}'
                ).format(
                    front_type_abbrevs[j].upper(), season_strings_verbose[i]
                )

            this_output_file_name = '{0:s}_{1:s}_{2:s}.jpg'.format(
                extensionless_output_file_name, front_type_abbrevs[j],
                season_strings_abbrev[i]
            )
            panel_file_names.append(this_output_file_name)

            if plot_frequency:
                if front_type_abbrevs[j] == 'wf':
                    this_max_colour_value = MAX_WF_FREQ_TREND_PERCENT_YEAR01
                else:
                    this_max_colour_value = MAX_CF_FREQ_TREND_PERCENT_YEAR01
            else:
                if front_type_abbrevs[j] == 'wf':
                    this_max_colour_value = MAX_WF_LENGTH_TREND_KM_YEAR01
                else:
                    this_max_colour_value = MAX_CF_LENGTH_TREND_KM_YEAR01

            if letter_label is None:
                letter_label = 'a'
            else:
                letter_label = chr(ord(letter_label) + 1)

            _plot_one_trend(
                trend_matrix_year01=this_trend_matrix_year01,
                significance_matrix=this_significance_matrix,
                max_colour_value=this_max_colour_value,
                plot_latitudes=j == 0, plot_longitudes=i == num_seasons - 1,
                plot_colour_bar=i == num_seasons - 1,
                title_string=this_title_string, letter_label=letter_label,
                output_file_name=panel_file_names[-1]
            )

    print('Concatenating panels to: "{0:s}"...'.format(output_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=output_file_name,
        num_panel_rows=num_seasons, num_panel_columns=num_properties)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    for this_file_name in panel_file_names:
        print('Removing temporary file "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        plot_frequency=bool(getattr(INPUT_ARG_OBJECT, PLOT_FREQUENCY_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
