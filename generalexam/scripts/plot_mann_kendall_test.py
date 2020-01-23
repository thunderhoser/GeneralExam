"""Plots results of Mann-Kendall test.

Results include linear trend over time, along with statistical significance (yes
or no), at each grid cell.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

MASK_IF_NUM_LABELS_BELOW = 100
NUM_ROWS_IN_CNN_PATCH = plot_gridded_stats.NUM_ROWS_IN_CNN_PATCH
NUM_COLUMNS_IN_CNN_PATCH = plot_gridded_stats.NUM_COLUMNS_IN_CNN_PATCH

TITLE_FONT_SIZE = 16
FIGURE_RESOLUTION_DPI = 300

FRACTION_TO_PERCENT = 100
METRES_TO_KM = 1e-3
METRES2_TO_THOUSAND_KM2 = 1e-9

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
MAX_FDR_ARG_NAME = 'monte_carlo_max_fdr'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`climatology_utils.read_mann_kendall_test`.'
)
COLOUR_MAP_HELP_STRING = (
    'Name of colour map for linear trend.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Max value in colour map will be [q]th percentile of absolute linear trend '
    'over all grid cells, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

MAX_FDR_HELP_STRING = (
    'Max FDR (false-discovery rate) for field-based version of significance '
    'test.  If you do not want to use field-based version, leave this argument '
    'alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='bwr',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FDR_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_FDR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_trend(
        trend_matrix_year01, significance_matrix, colour_map_object,
        title_string, output_file_name, max_colour_percentile=None,
        max_colour_value=None):
    """Plots linear trend over time, along with Mann-Kendall significance.

    M = number of rows in grid
    N = number of columns in grid

    :param trend_matrix_year01: M-by-N numpy array with linear trend (per year)
        at each grid cell.
    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where difference is significant.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_percentile: [may be None]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all absolute values in `trend_matrix_year01`, where q =
        `max_colour_percentile`.  Minimum value will be -1 * max value.
    :param max_colour_value: [used only if `max_colour_percentile is None`]
        Max value in colour scheme.  Minimum value will be -1 * max value.
    """

    basemap_dict = plot_gridded_stats.plot_basemap(
        data_matrix=trend_matrix_year01
    )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    trend_matrix_to_plot_year01 = trend_matrix_year01[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
    ]
    sig_matrix_to_plot = significance_matrix[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
    ]

    if max_colour_percentile is not None:
        max_colour_value = numpy.nanpercentile(
            numpy.absolute(trend_matrix_to_plot_year01), max_colour_percentile
        )

    colour_norm_object = pyplot.Normalize(
        vmin=-max_colour_value, vmax=max_colour_value
    )

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=trend_matrix_to_plot_year01,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    sig_latitudes_deg = latitude_matrix_deg[sig_matrix_to_plot == True]
    sig_longitudes_deg = longitude_matrix_deg[sig_matrix_to_plot == True]
    sig_x_coords_metres, sig_y_coords_metres = basemap_object(
        sig_longitudes_deg, sig_latitudes_deg
    )

    axes_object.plot(
        sig_x_coords_metres, sig_y_coords_metres, linestyle='None',
        marker=plot_gridded_stats.SIG_MARKER_TYPE,
        markerfacecolor=plot_gridded_stats.SIG_MARKER_COLOUR,
        markeredgecolor=plot_gridded_stats.SIG_MARKER_COLOUR,
        markersize=plot_gridded_stats.SIG_MARKER_SIZE,
        markeredgewidth=plot_gridded_stats.SIG_MARKER_EDGE_WIDTH
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=trend_matrix_to_plot_year01,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.05,
        extend_min=False, extend_max=True, fraction_of_axis_length=1.
    )

    tick_values = colour_bar_object.ax.get_xticks()
    colour_bar_object.ax.set_xticks(tick_values)

    if numpy.all(numpy.absolute(tick_values) < 1):
        tick_strings = ['{0:.3f}'.format(x) for x in tick_values]
    else:
        tick_strings = [
            '{0:d}'.format(int(numpy.round(x))) for x in tick_values
        ]

    colour_bar_object.ax.set_xticklabels(tick_strings)
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(input_file_name, colour_map_name, max_colour_percentile,
         monte_carlo_max_fdr, output_dir_name):
    """Plots results of Mann-Kendall test.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param monte_carlo_max_fdr: Same.
    :param output_dir_name: Same.
    """

    if monte_carlo_max_fdr <= 0:
        monte_carlo_max_fdr = None

    error_checking.assert_is_greater(max_colour_percentile, 50.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    colour_map_object = pyplot.get_cmap(colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    mann_kendall_dict = climo_utils.read_mann_kendall_test(input_file_name)

    property_name = mann_kendall_dict[climo_utils.PROPERTY_NAME_KEY]
    trend_matrix_year01 = mann_kendall_dict[climo_utils.TREND_MATRIX_KEY]
    num_labels_matrix = mann_kendall_dict[climo_utils.NUM_LABELS_MATRIX_KEY]
    p_value_matrix = mann_kendall_dict[climo_utils.P_VALUE_MATRIX_KEY]

    if monte_carlo_max_fdr is None:
        significance_matrix = p_value_matrix <= 0.05
    else:
        significance_matrix = climo_utils.find_sig_grid_points(
            p_value_matrix=p_value_matrix,
            max_false_discovery_rate=monte_carlo_max_fdr
        )

    significance_matrix[num_labels_matrix < MASK_IF_NUM_LABELS_BELOW] = False
    trend_matrix_year01[
        num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
    ] = numpy.nan

    if property_name in [
            climo_utils.WF_LENGTH_PROPERTY_NAME,
            climo_utils.CF_LENGTH_PROPERTY_NAME
    ]:
        trend_matrix_year01 = trend_matrix_year01 * METRES_TO_KM

    if property_name in [
            climo_utils.WF_AREA_PROPERTY_NAME,
            climo_utils.CF_AREA_PROPERTY_NAME
    ]:
        trend_matrix_year01 = trend_matrix_year01 * METRES2_TO_THOUSAND_KM2

    if property_name in [
            climo_utils.WF_FREQ_PROPERTY_NAME,
            climo_utils.CF_FREQ_PROPERTY_NAME
    ]:
        trend_matrix_year01 = trend_matrix_year01 * FRACTION_TO_PERCENT

    if property_name == climo_utils.WF_LENGTH_PROPERTY_NAME:
        title_string = 'Trend in mean WF length (km per year)'
    elif property_name == climo_utils.CF_LENGTH_PROPERTY_NAME:
        title_string = 'Trend in mean CF length (km per year)'
    elif property_name == climo_utils.WF_AREA_PROPERTY_NAME:
        title_string = r'Trend in mean WF area ($\times$ 1000 km$^2$ per year)'
    elif property_name == climo_utils.CF_AREA_PROPERTY_NAME:
        title_string = r'Trend in mean CF area ($\times$ 1000 km$^2$ per year)'
    elif property_name == climo_utils.WF_FREQ_PROPERTY_NAME:
        title_string = 'Trend in WF frequency (percent per year)'
    elif property_name == climo_utils.CF_FREQ_PROPERTY_NAME:
        title_string = 'Trend in CF frequency (percent per year)'

    output_file_name = '{0:s}/{1:s}_trend.jpg'.format(
        output_dir_name, property_name
    )

    _plot_trend(
        trend_matrix_year01=trend_matrix_year01,
        significance_matrix=significance_matrix,
        colour_map_object=colour_map_object, title_string=title_string,
        output_file_name=output_file_name,
        max_colour_percentile=max_colour_percentile
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        monte_carlo_max_fdr=getattr(INPUT_ARG_OBJECT, MAX_FDR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
