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
METRES2_TO_THOUSAND_KM2 = 1e-9

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
BORDER_COLOUR = numpy.full(3, 0.)

MIN_LATITUDE_DEG = 5.
MIN_LONGITUDE_DEG = 200.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 310.

SIG_MARKER_TYPE = '.'
SIG_MARKER_SIZE = 1
SIG_MARKER_COLOUR = numpy.full(3, 152. / 255)
SIG_MARKER_EDGE_WIDTH = 1

TITLE_FONT_SIZE = 16
TITLE_TIME_FORMAT = '%Y-%m-%d-%H'
FIGURE_RESOLUTION_DPI = 300

AXES_OBJECT_KEY = 'axes_object'
BASEMAP_OBJECT_KEY = 'basemap_object'
FULL_GRID_NAME_KEY = 'full_grid_name'
FULL_GRID_ROWS_KEY = 'full_grid_row_limits'
FULL_GRID_COLUMNS_KEY = 'full_grid_column_limits'
SUBGRID_DATA_KEY = 'subgrid_data_matrix'

STATISTIC_FILE_ARG_NAME = 'input_statistic_file_name'
MONTE_CARLO_FILE_ARG_NAME = 'input_monte_carlo_file_name'
LENGTH_CMAP_ARG_NAME = 'length_colour_map_name'
AREA_CMAP_ARG_NAME = 'area_colour_map_name'
DIFFERENCE_CMAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

STATISTIC_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_gridded_stats`).  If you want to plot results of a'
    ' Monte Carlo significance test, leave this argument alone.')

MONTE_CARLO_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_monte_carlo_test`).  If you just want to plot '
    'statistics, leave this argument alone.')

LENGTH_CMAP_HELP_STRING = (
    'Name of colour map for front length.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

AREA_CMAP_HELP_STRING = (
    'Name of colour map for front area.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

DIFFERENCE_CMAP_HELP_STRING = (
    'Name of colour map for composite difference (used only if `{0:s}` is not '
    'empty).  Must be accepted by `matplotlib.pyplot.get_cmap`.'
).format(MONTE_CARLO_FILE_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in colour scheme.  Max value in length '
    'colour scheme will be the [q]th percentile of values at all grid cells, '
    'where q = `{0:s}` -- and likewise for area.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STATISTIC_FILE_ARG_NAME, type=str, required=False, default='',
    help=STATISTIC_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MONTE_CARLO_FILE_ARG_NAME, type=str, required=False, default='',
    help=MONTE_CARLO_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LENGTH_CMAP_ARG_NAME, type=str, required=False, default='PuBuGn',
    help=LENGTH_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + AREA_CMAP_ARG_NAME, type=str, required=False, default='PuBuGn',
    help=AREA_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DIFFERENCE_CMAP_ARG_NAME, type=str, required=False, default='bwr',
    help=DIFFERENCE_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_basemap(full_data_matrix):
    """Plots basemap.

    "Subgrid" = part of full grid to be plotted.

    :param full_data_matrix: 2-D numpy array of data values.
    :return: basemap_dict: Dictionary with the following keys.
    basemap_dict["axes_object"]: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    basemap_dict["basemap_object"]: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    basemap_dict["full_grid_name"]: Name of full grid.
    basemap_dict["full_grid_row_limits"]: length-2 numpy array mapping subgrid
        to full grid.  If full_grid_row_limits[0] = i and
        full_grid_row_limits[1] = j, first and last rows in subgrid are [i]th
        and [j]th rows in full grid, respectively.
    basemap_dict["full_grid_column_limits"]: Same but for columns.
    basemap_dict["subgrid_data_matrix"]: 2-D numpy array of values to plot.
    """

    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=full_data_matrix.shape[0],
        num_columns=full_data_matrix.shape[1]
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

    subgrid_data_matrix = full_data_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    return {
        AXES_OBJECT_KEY: axes_object,
        BASEMAP_OBJECT_KEY: basemap_object,
        FULL_GRID_NAME_KEY: full_grid_name,
        FULL_GRID_ROWS_KEY: full_grid_row_limits,
        FULL_GRID_COLUMNS_KEY: full_grid_column_limits,
        SUBGRID_DATA_KEY: subgrid_data_matrix
    }


def _plot_one_statistic(
        statistic_matrix, colour_map_object, title_string, output_file_name,
        max_colour_value=None, max_colour_percentile=None):
    """Plots one statistic for one front type.

    :param statistic_matrix: 2-D numpy array with statistic at each grid cell.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_value: Max value in colour scheme.  This may be None.
    :param max_colour_percentile: [used only if `max_colour_value is None`]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all values in `statistic_matrix`, where q =
        `max_colour_percentile`.
    """

    basemap_dict = _plot_basemap(statistic_matrix)
    axes_object = basemap_dict[AXES_OBJECT_KEY]
    basemap_object = basemap_dict[BASEMAP_OBJECT_KEY]
    full_grid_name = basemap_dict[FULL_GRID_NAME_KEY]
    full_grid_row_limits = basemap_dict[FULL_GRID_ROWS_KEY]
    full_grid_column_limits = basemap_dict[FULL_GRID_COLUMNS_KEY]
    this_matrix = basemap_dict[SUBGRID_DATA_KEY]

    if max_colour_value is None:
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


def _plot_monte_carlo_diff(
        difference_matrix, significance_matrix, colour_map_object,
        max_colour_percentile, title_string, output_file_name):
    """Plots diff between two composites, along with Monte Carlo significance.

    M = number of rows in grid
    N = number of columns in grid

    :param difference_matrix: M-by-N numpy array with difference between means
        (trial composite minus baseline composite) at each grid cell.
    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where difference is significant.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param max_colour_percentile: Max value in colour scheme will be [q]th
        percentile of all absolute values in `difference_matrix`, where q =
        `max_colour_percentile`.  Minimum value will be -1 * max value.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    basemap_dict = _plot_basemap(difference_matrix)
    axes_object = basemap_dict[AXES_OBJECT_KEY]
    basemap_object = basemap_dict[BASEMAP_OBJECT_KEY]
    full_grid_name = basemap_dict[FULL_GRID_NAME_KEY]
    full_grid_row_limits = basemap_dict[FULL_GRID_ROWS_KEY]
    full_grid_column_limits = basemap_dict[FULL_GRID_COLUMNS_KEY]

    diff_matrix_to_plot = difference_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    sig_matrix_to_plot = significance_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    max_colour_value = numpy.nanpercentile(
        numpy.absolute(diff_matrix_to_plot), max_colour_percentile
    )

    colour_norm_object = pyplot.Normalize(
        vmin=-max_colour_value, vmax=max_colour_value)

    prediction_plotting.plot_gridded_counts(
        count_or_frequency_matrix=diff_matrix_to_plot,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
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
        marker=SIG_MARKER_TYPE, markerfacecolor=SIG_MARKER_COLOUR,
        markeredgecolor=SIG_MARKER_COLOUR, markersize=SIG_MARKER_SIZE,
        markeredgewidth=SIG_MARKER_EDGE_WIDTH,
        transform=axes_object.transAxes)

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=diff_matrix_to_plot,
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


def _run(statistic_file_name, monte_carlo_file_name, length_colour_map_name,
         area_colour_map_name, diff_colour_map_name, max_colour_percentile,
         output_dir_name):
    """Plots statistics for gridded front properties.

    This is effectively the main method.

    :param statistic_file_name: See documentation at top of file.
    :param monte_carlo_file_name: Same.
    :param length_colour_map_name: Same.
    :param area_colour_map_name: Same.
    :param diff_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    if statistic_file_name in ['', 'None']:
        statistic_file_name = None
    if monte_carlo_file_name in ['', 'None']:
        monte_carlo_file_name = None

    error_checking.assert_is_greater(max_colour_percentile, 50.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    length_colour_map_object = pyplot.get_cmap(length_colour_map_name)
    area_colour_map_object = pyplot.get_cmap(area_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if statistic_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(statistic_file_name))
        front_statistic_dict = climo_utils.read_gridded_stats(
            statistic_file_name)

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
                climo_utils.hours_to_string(hours)[0]
            )

        months = front_statistic_dict[climo_utils.MONTHS_KEY]
        if months is not None:
            this_title_string += '; months {0:s}'.format(
                climo_utils.months_to_string(months)[0]
            )

        this_output_file_name = '{0:s}/mean_wf_length.jpg'.format(
            output_dir_name)

        _plot_one_statistic(
            statistic_matrix=(
                front_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY] *
                METRES_TO_KM
            ),
            colour_map_object=length_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            title_string=this_title_string,
            output_file_name=this_output_file_name)

        this_title_string = this_title_string.replace(
            'Mean WF length', 'Mean CF length')
        this_output_file_name = '{0:s}/mean_cf_length.jpg'.format(
            output_dir_name)

        _plot_one_statistic(
            statistic_matrix=(
                front_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY] *
                METRES_TO_KM
            ),
            colour_map_object=length_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            title_string=this_title_string,
            output_file_name=this_output_file_name)

        this_title_string = this_title_string.replace(
            'Mean CF length (km)', r'Mean WF area ($\times$ 1000 km$^2$)'
        )
        this_output_file_name = '{0:s}/mean_wf_area.jpg'.format(output_dir_name)

        _plot_one_statistic(
            statistic_matrix=(
                front_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY] *
                METRES2_TO_THOUSAND_KM2
            ),
            colour_map_object=area_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            title_string=this_title_string,
            output_file_name=this_output_file_name)

        this_title_string = this_title_string.replace(
            'Mean WF area', 'Mean CF area')
        this_output_file_name = '{0:s}/mean_cf_area.jpg'.format(output_dir_name)

        _plot_one_statistic(
            statistic_matrix=(
                front_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY] *
                METRES2_TO_THOUSAND_KM2
            ),
            colour_map_object=area_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            title_string=this_title_string,
            output_file_name=this_output_file_name)

        return

    print('Reading data from: "{0:s}"...'.format(monte_carlo_file_name))
    monte_carlo_dict = climo_utils.read_monte_carlo_test(monte_carlo_file_name)
    property_name = monte_carlo_dict[climo_utils.PROPERTY_NAME_KEY]

    this_title_string = None
    this_output_file_name = None
    conversion_ratio = None
    colour_map_object = None

    diff_colour_map_object = pyplot.get_cmap(diff_colour_map_name)

    if property_name == climo_utils.WF_LENGTH_PROPERTY_NAME:
        this_title_string = 'Mean WF length (km) for baseline composite'
        this_output_file_name = '{0:s}/wf_length_baseline-mean.jpg'.format(
            output_dir_name)

        conversion_ratio = METRES_TO_KM
        colour_map_object = length_colour_map_object

    elif property_name == climo_utils.WF_AREA_PROPERTY_NAME:
        this_title_string = (
            r'Mean WF area ($\times$ 1000 km$^2$) for baseline composite'
        )
        this_output_file_name = '{0:s}/wf_area_baseline-mean.jpg'.format(
            output_dir_name)

        conversion_ratio = METRES2_TO_THOUSAND_KM2
        colour_map_object = area_colour_map_object

    elif property_name == climo_utils.CF_LENGTH_PROPERTY_NAME:
        this_title_string = 'Mean CF length (km) for baseline composite'
        this_output_file_name = '{0:s}/cf_length_baseline-mean.jpg'.format(
            output_dir_name)

        conversion_ratio = METRES_TO_KM
        colour_map_object = length_colour_map_object

    elif property_name == climo_utils.CF_AREA_PROPERTY_NAME:
        this_title_string = (
            r'Mean CF area ($\times$ 1000 km$^2$) for baseline composite'
        )
        this_output_file_name = '{0:s}/cf_area_baseline-mean.jpg'.format(
            output_dir_name)

        conversion_ratio = METRES2_TO_THOUSAND_KM2
        colour_map_object = area_colour_map_object

    concat_data_matrix = conversion_ratio * numpy.concatenate(
        (
            monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY],
            monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY]
        ), axis=0
    )

    max_colour_value = numpy.nanpercentile(
        concat_data_matrix, max_colour_percentile)

    _plot_one_statistic(
        statistic_matrix=(
            monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY] *
            conversion_ratio
        ),
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        title_string=this_title_string, output_file_name=this_output_file_name)

    this_title_string = this_title_string.replace('baseline', 'trial')
    this_output_file_name = this_output_file_name.replace(
        'baseline-mean.jpg', 'trial-mean.jpg')

    _plot_one_statistic(
        statistic_matrix=(
            monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY] *
            conversion_ratio
        ),
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        title_string=this_title_string, output_file_name=this_output_file_name)

    this_title_string = this_title_string.replace(' for trial composite', '')
    this_title_string = this_title_string.replace(
        'Mean', 'Composite difference (trial minus baseline)'
    )
    this_output_file_name = this_output_file_name.replace(
        'trial-mean.jpg', 'difference.jpg')

    difference_matrix = conversion_ratio * (
        monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY] -
        monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY]
    )

    _plot_monte_carlo_diff(
        difference_matrix=difference_matrix,
        significance_matrix=monte_carlo_dict[
            climo_utils.SIGNIFICANCE_MATRIX_KEY],
        colour_map_object=diff_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=this_title_string, output_file_name=this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        statistic_file_name=getattr(INPUT_ARG_OBJECT, STATISTIC_FILE_ARG_NAME),
        monte_carlo_file_name=getattr(
            INPUT_ARG_OBJECT, MONTE_CARLO_FILE_ARG_NAME),
        length_colour_map_name=getattr(INPUT_ARG_OBJECT, LENGTH_CMAP_ARG_NAME),
        area_colour_map_name=getattr(INPUT_ARG_OBJECT, AREA_CMAP_ARG_NAME),
        diff_colour_map_name=getattr(
            INPUT_ARG_OBJECT, DIFFERENCE_CMAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
