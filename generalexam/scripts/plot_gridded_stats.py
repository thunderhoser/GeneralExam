"""Plots statistics for gridded front properties."""

import argparse
import numpy
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting

METRES_TO_KM = 1e-3
METRES2_TO_KM2 = 1e-6

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
BORDER_COLOUR = numpy.full(3, 0.)

MIN_LATITUDE_DEG = 5.
MIN_LONGITUDE_DEG = 200.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 310.

TITLE_FONT_SIZE = 16
TITLE_TIME_FORMAT = '%Y-%m-%d-%H'
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
LENGTH_CMAP_ARG_NAME = 'length_colour_map_name'
AREA_CMAP_ARG_NAME = 'area_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`climatology_utils.read_gridded_stats`.')

LENGTH_CMAP_HELP_STRING = (
    'Name of colour map for front length.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

AREA_CMAP_HELP_STRING = (
    'Name of colour map for front area.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in colour scheme.  Max value in length '
    'colour scheme will be the [q]th percentile of values at all grid cells, '
    'where q = `{0:s}` -- and likewise for area.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LENGTH_CMAP_ARG_NAME, type=str, required=False, default='PuBuGn',
    help=LENGTH_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + AREA_CMAP_ARG_NAME, type=str, required=False, default='PuBuGn',
    help=AREA_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_statistic(
        statistic_matrix, colour_map_object, max_colour_percentile,
        title_string, output_file_name):
    """Plots one statistic for one front type.

    :param statistic_matrix: 2-D numpy array with statistic at each grid cell.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param max_colour_percentile: See documentation at top of file.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=statistic_matrix.shape[0],
        num_columns=statistic_matrix.shape[1]
    )

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

    this_matrix = statistic_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    colour_norm_object = pyplot.Normalize(
        vmin=0,
        vmax=numpy.percentile(this_matrix, max_colour_percentile)
    )

    prediction_plotting.plot_gridded_counts(
        count_or_frequency_matrix=this_matrix,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, full_grid_name=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=this_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=True,
        fraction_of_axis_length=0.9)

    tick_values = colour_bar_object.ax.get_xticks()
    colour_bar_object.ax.set_xticks(tick_values)
    colour_bar_object.ax.set_xticklabels(tick_values)

    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(input_file_name, length_colour_map_name, area_colour_map_name,
         max_colour_percentile, output_dir_name):
    """Plots statistics for gridded front properties.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param length_colour_map_name: Same.
    :param area_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(max_colour_percentile, 50.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    length_colour_map_object = pyplot.get_cmap(length_colour_map_name)
    area_colour_map_object = pyplot.get_cmap(area_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    front_statistic_dict = climo_utils.read_gridded_stats(input_file_name)

    first_time_string = time_conversion.unix_sec_to_string(
        front_statistic_dict[climo_utils.FIRST_TIME_KEY], TITLE_TIME_FORMAT
    )
    last_time_string = time_conversion.unix_sec_to_string(
        front_statistic_dict[climo_utils.LAST_TIME_KEY], TITLE_TIME_FORMAT
    )

    this_title_string = 'Mean WF length (km) from {0:s} to {1:s}'.format(
        first_time_string, last_time_string)

    hours = front_statistic_dict[climo_utils.HOURS_KEY]
    if hours is not None:
        this_title_string += '; hours {0:s}'.format(
            climo_utils.hours_to_string(hours)
        )

    months = front_statistic_dict[climo_utils.MONTHS_KEY]
    if months is not None:
        this_title_string += '; months {0:s}'.format(
            climo_utils.months_to_string(months)
        )

    this_output_file_name = '{0:s}/mean_wf_length.jpg'.format(output_dir_name)
    _plot_one_statistic(
        statistic_matrix=
        front_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY] * METRES_TO_KM,
        colour_map_object=length_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=this_title_string, output_file_name=this_output_file_name)

    this_title_string = this_title_string.replace('Mean WF length',
                                                  'Mean CF length')
    this_output_file_name = '{0:s}/mean_cf_length.jpg'.format(output_dir_name)

    _plot_one_statistic(
        statistic_matrix=
        front_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY] * METRES_TO_KM,
        colour_map_object=length_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=this_title_string, output_file_name=this_output_file_name)

    this_title_string = this_title_string.replace('Mean CF length',
                                                  'Mean WF area')
    this_output_file_name = '{0:s}/mean_wf_area.jpg'.format(output_dir_name)

    _plot_one_statistic(
        statistic_matrix=
        front_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY] * METRES2_TO_KM2,
        colour_map_object=area_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=this_title_string, output_file_name=this_output_file_name)

    this_title_string = this_title_string.replace('Mean WF area',
                                                  'Mean CF area')
    this_output_file_name = '{0:s}/mean_cf_area.jpg'.format(output_dir_name)

    _plot_one_statistic(
        statistic_matrix=
        front_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY] * METRES2_TO_KM2,
        colour_map_object=area_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=this_title_string, output_file_name=this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        length_colour_map_name=getattr(INPUT_ARG_OBJECT, LENGTH_CMAP_ARG_NAME),
        area_colour_map_name=getattr(INPUT_ARG_OBJECT, AREA_CMAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
