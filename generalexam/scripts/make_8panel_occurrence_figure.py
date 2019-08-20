"""Makes 8-panel front-occurrence figure."""

import os.path
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

NUM_YEARS = 40
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

MAX_WF_FREQUENCY = 0.08
MAX_CF_FREQUENCY = 0.12
MAX_WF_COUNT = 20
MAX_CF_COUNT = 30

WF_COLOUR_MAP_OBJECT = pyplot.get_cmap('YlOrRd')
CF_COLOUR_MAP_OBJECT = pyplot.get_cmap('YlGnBu')

TITLE_FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_count_dir_name'
FRONT_TYPE_ARG_NAME = 'front_type_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_counts`.')

FRONT_TYPE_HELP_STRING = (
    'Front type.  Must be in the following list:\n{0:s}'
).format(str(front_utils.VALID_FRONT_TYPE_STRINGS))

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_TYPE_ARG_NAME, type=str, required=True,
    help=FRONT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _plot_one_front_type(
        count_or_frequency_matrix, colour_map_object, plot_frequency,
        plot_latitudes, plot_longitudes, plot_colour_bar, title_string,
        output_file_name, max_colour_percentile=None, max_colour_value=None):
    """Plots gridded counts or frequencies for one front type.

    :param count_or_frequency_matrix: 2-D numpy array with number or frequency
        of fronts at each grid cell.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param plot_frequency: Same.
    :param plot_latitudes: Boolean flag.  Determines whether or not numbers will
        be plotted on y-axis.
    :param plot_longitudes: Boolean flag.  Determines whether or not numbers
        will be plotted on x-axis.
    :param plot_colour_bar: Boolean flag.  Determines whether or not colour bar
        will be plotted below.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_percentile: [may be None]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all values in `count_or_frequency_matrix`, where q =
        `max_colour_percentile`.
    :param max_colour_value: [used only if `max_colour_percentile is None`]
        Max value in colour scheme.
    """

    basemap_dict = plot_gridded_stats._plot_basemap(count_or_frequency_matrix)
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    full_grid_name = basemap_dict[plot_gridded_stats.FULL_GRID_NAME_KEY]
    full_grid_row_limits = basemap_dict[plot_gridded_stats.FULL_GRID_ROWS_KEY]
    full_grid_column_limits = basemap_dict[
        plot_gridded_stats.FULL_GRID_COLUMNS_KEY]

    this_matrix = basemap_dict[plot_gridded_stats.SUBGRID_DATA_KEY]

    if max_colour_percentile is not None:
        max_colour_value = numpy.nanpercentile(
            this_matrix, max_colour_percentile)

    colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

    prediction_plotting.plot_gridded_counts(
        count_or_frequency_matrix=this_matrix,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, full_grid_name=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    if not plot_latitudes:
        axes_object.set_yticklabels([])
    if not plot_longitudes:
        axes_object.set_xticklabels([])

    if plot_colour_bar:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=this_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', extend_min=False, extend_max=True,
            fraction_of_axis_length=0.9)

        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)

        if plot_frequency:
            colour_bar_object.ax.set_xticklabels(tick_values)
        else:
            tick_strings = ['{0:.1f}'.format(x) for x in tick_values]
            colour_bar_object.ax.set_xticklabels(tick_strings)

    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(count_dir_name, front_type_string, output_file_name):
    """Makes 8-panel front-occurrence figure.

    This is effectively the main method.

    :param count_dir_name: See documentation at top of file.
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

    for i in range(num_seasons):
        this_file_name = climo_utils.find_aggregated_file(
            directory_name=count_dir_name,
            file_type_string=climo_utils.FRONT_COUNTS_STRING,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            months=climo_utils.season_to_months(season_strings_abbrev[i]),
            raise_error_if_missing=True
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_count_dict = climo_utils.read_gridded_counts(this_file_name)
        this_num_times = len(this_count_dict[climo_utils.PREDICTION_FILES_KEY])

        if front_type_string == front_utils.WARM_FRONT_STRING:
            this_frequency_matrix = (
                this_count_dict[climo_utils.NUM_WF_LABELS_KEY] / this_num_times
            )

            this_max_colour_value = MAX_WF_FREQUENCY
            this_colour_map_object = WF_COLOUR_MAP_OBJECT
            this_title_string = 'WF'
        else:
            this_frequency_matrix = (
                this_count_dict[climo_utils.NUM_CF_LABELS_KEY] / this_num_times
            )

            this_max_colour_value = MAX_CF_FREQUENCY
            this_colour_map_object = CF_COLOUR_MAP_OBJECT
            this_title_string = 'CF'

        this_title_string += ' frequency in {0:s}'.format(
            season_strings_verbose[i]
        )

        this_output_file_name = '{0:s}_frequency_{1:s}.jpg'.format(
            extensionless_output_file_name, season_strings_abbrev[i]
        )
        panel_file_names.append(this_output_file_name)

        _plot_one_front_type(
            count_or_frequency_matrix=this_frequency_matrix,
            colour_map_object=this_colour_map_object, plot_frequency=True,
            plot_latitudes=True, plot_longitudes=i == num_seasons - 1,
            plot_colour_bar=i == num_seasons - 1,
            title_string=this_title_string,
            output_file_name=panel_file_names[-1],
            max_colour_value=this_max_colour_value)

        if front_type_string == front_utils.WARM_FRONT_STRING:
            this_count_matrix = (
                this_count_dict[climo_utils.NUM_UNIQUE_WF_KEY] / NUM_YEARS
            )

            this_max_colour_value = MAX_WF_COUNT
            this_title_string = 'Number of unique WF'
        else:
            this_count_matrix = (
                this_count_dict[climo_utils.NUM_UNIQUE_CF_KEY] / NUM_YEARS
            )

            this_max_colour_value = MAX_CF_COUNT
            this_title_string = 'Number of unique CF'

        this_title_string += ' per year in {0:s}'.format(
            season_strings_verbose[i]
        )

        this_output_file_name = '{0:s}_count_{1:s}.jpg'.format(
            extensionless_output_file_name, season_strings_abbrev[i]
        )
        panel_file_names.append(this_output_file_name)

        _plot_one_front_type(
            count_or_frequency_matrix=this_count_matrix,
            colour_map_object=this_colour_map_object, plot_frequency=False,
            plot_latitudes=False, plot_longitudes=i == num_seasons - 1,
            plot_colour_bar=i == num_seasons - 1,
            title_string=this_title_string,
            output_file_name=panel_file_names[-1],
            max_colour_value=this_max_colour_value)

    print('Concatenating panels to: "{0:s}"...'.format(output_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=output_file_name,
        num_panel_rows=num_seasons, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        front_type_string=getattr(INPUT_ARG_OBJECT, FRONT_TYPE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
