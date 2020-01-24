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
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting

# TODO(thunderhoser): Making these constants is a HACK.
MASK_IF_NUM_LABELS_BELOW = 100
NUM_ROWS_IN_CNN_PATCH = 16
NUM_COLUMNS_IN_CNN_PATCH = 16

METRES_TO_KM = 1e-3
METRES2_TO_THOUSAND_KM2 = 1e-9

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
DEFAULT_BORDER_COLOUR = numpy.full(3, 0.)
MONTE_CARLO_BORDER_COLOUR = numpy.full(3, 152. / 255)

MIN_ERA5_LATITUDE_DEG = 3.
MIN_ERA5_LONGITUDE_DEG = 178.
MAX_ERA5_LATITUDE_DEG = 82.
MAX_ERA5_LONGITUDE_DEG = 322.

MIN_NARR_LATITUDE_DEG = 15.
MIN_NARR_LONGITUDE_DEG = 178.
MAX_NARR_LATITUDE_DEG = 82.
MAX_NARR_LONGITUDE_DEG = 322.

SIG_MARKER_TYPE = '.'
SIG_MARKER_SIZE = 0.6
SIG_MARKER_COLOUR = numpy.full(3, 0.)
SIG_MARKER_EDGE_WIDTH = 1

TITLE_FONT_SIZE = 16
TITLE_TIME_FORMAT = '%Y-%m-%d-%H'
FIGURE_RESOLUTION_DPI = 300

FIGURE_OBJECT_KEY = 'figure_object'
AXES_OBJECT_KEY = 'axes_object'
BASEMAP_OBJECT_KEY = 'basemap_object'
MATRIX_TO_PLOT_KEY = 'matrix_to_plot'
LATITUDES_KEY = 'latitude_matrix_deg'
LONGITUDES_KEY = 'longitude_matrix_deg'

STATISTIC_FILE_ARG_NAME = 'input_statistic_file_name'
MONTE_CARLO_FILE_ARG_NAME = 'input_monte_carlo_file_name'
LENGTH_CMAP_ARG_NAME = 'length_colour_map_name'
AREA_CMAP_ARG_NAME = 'area_colour_map_name'
DIFFERENCE_CMAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
WF_COLOUR_MAXIMA_ARG_NAME = 'wf_colour_maxima'
CF_COLOUR_MAXIMA_ARG_NAME = 'cf_colour_maxima'
MC_COLOUR_MAXIMA_ARG_NAME = 'monte_carlo_colour_maxima'
MAX_FDR_ARG_NAME = 'monte_carlo_max_fdr'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

STATISTIC_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_gridded_stats`).  If you want to plot results of a'
    ' Monte Carlo significance test, leave this argument alone.'
)
MONTE_CARLO_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_monte_carlo_test`).  If you just want to plot '
    'statistics, leave this argument alone.'
)
LENGTH_CMAP_HELP_STRING = (
    'Name of colour map for front length.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
AREA_CMAP_HELP_STRING = (
    'Name of colour map for front area.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)

DIFFERENCE_CMAP_HELP_STRING = (
    'Name of colour map for composite difference (used only if `{0:s}` is not '
    'empty).  Must be accepted by `matplotlib.pyplot.get_cmap`.'
).format(MONTE_CARLO_FILE_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in each colour scheme.  If you want to '
    'set max values directly, make this negative and use other input args.'
).format(MAX_PERCENTILE_ARG_NAME)

WF_COLOUR_MAXIMA_HELP_STRING = (
    '[used only if `{0:s}` is negative] List of max values for warm-front maps.'
    '  This list should have 2 elements [length (km), area (x 1000 km^2)].'
)
CF_COLOUR_MAXIMA_HELP_STRING = (
    '[used only if `{0:s}` is negative] List of max values for cold-front maps.'
    '  This list should have 2 elements [length (km), area (x 1000 km^2)].'
)
MC_COLOUR_MAXIMA_HELP_STRING = (
    '[used only if `{0:s}` is negative] List of max values for the two types of'
    ' Monte Carlo maps.  This list should have 2 elements [means, differences].'
)
MAX_FDR_HELP_STRING = (
    'Max FDR (false-discovery rate) for field-based version of Monte Carlo '
    'significance test.  If you do not want to use field-based version, leave '
    'this argument alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STATISTIC_FILE_ARG_NAME, type=str, required=False, default='',
    help=STATISTIC_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MONTE_CARLO_FILE_ARG_NAME, type=str, required=False, default='',
    help=MONTE_CARLO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LENGTH_CMAP_ARG_NAME, type=str, required=False, default='plasma',
    help=LENGTH_CMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + AREA_CMAP_ARG_NAME, type=str, required=False, default='plasma',
    help=AREA_CMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DIFFERENCE_CMAP_ARG_NAME, type=str, required=False, default='bwr',
    help=DIFFERENCE_CMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WF_COLOUR_MAXIMA_ARG_NAME, type=float, nargs=2, required=False,
    default=numpy.full(2, 0.), help=WF_COLOUR_MAXIMA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CF_COLOUR_MAXIMA_ARG_NAME, type=float, nargs=2, required=False,
    default=numpy.full(2, 0.), help=CF_COLOUR_MAXIMA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MC_COLOUR_MAXIMA_ARG_NAME, type=float, nargs=2, required=False,
    default=numpy.full(2, 0.), help=MC_COLOUR_MAXIMA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FDR_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_FDR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_statistic(
        statistic_matrix, colour_map_object, title_string, output_file_name,
        max_colour_value=None, max_colour_percentile=None):
    """Plots one statistic for one front type.

    :param statistic_matrix: 2-D numpy array with statistic at each grid cell.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_percentile: [may be None]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all values in `statistic_matrix`, where q =
        `max_colour_percentile`.
    :param max_colour_value: [used only if `max_colour_percentile is None`]
        Max value in colour scheme.
    """

    basemap_dict = plot_basemap(data_matrix=statistic_matrix)

    figure_object = basemap_dict[FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[AXES_OBJECT_KEY]
    basemap_object = basemap_dict[BASEMAP_OBJECT_KEY]
    matrix_to_plot = basemap_dict[MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[LONGITUDES_KEY]

    if max_colour_percentile is not None:
        max_colour_value = numpy.nanpercentile(
            matrix_to_plot, max_colour_percentile
        )

    colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=matrix_to_plot,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.05,
        extend_min=False, extend_max=True, fraction_of_axis_length=1.
    )

    tick_values = colour_bar_object.ax.get_xticks()
    colour_bar_object.ax.set_xticks(tick_values)
    colour_bar_object.ax.set_xticklabels(tick_values)

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def plot_basemap(data_matrix, border_colour=DEFAULT_BORDER_COLOUR):
    """Plots basemap.

    M = number of grid rows to plot
    N = number of grid columns to plot

    :param data_matrix: 2-D numpy array of data values.
    :param border_colour: Border colour (length-3 numpy array).
    :return: basemap_dict: Dictionary with the following keys.
    basemap_dict["figure_object"]: Figure handle (instance of
        `matplotlib.figure.Figure`).
    basemap_dict["figure_object"]: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    basemap_dict["figure_object"]: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    basemap_dict["matrix_to_plot"]: M-by-N numpy array of data values.
    basemap_dict["latitude_matrix_deg"]: M-by-N numpy array of latitudes
        (deg N).
    basemap_dict["longitude_matrix_deg"]: M-by-N numpy array of longitudes
        (deg E).
    """

    num_grid_rows = data_matrix.shape[0]
    num_grid_columns = data_matrix.shape[1]
    grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    latitude_matrix_deg, longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name)
    )

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=MIN_ERA5_LATITUDE_DEG,
            max_latitude_deg=MAX_ERA5_LATITUDE_DEG,
            min_longitude_deg=MIN_ERA5_LONGITUDE_DEG,
            max_longitude_deg=MAX_ERA5_LONGITUDE_DEG)
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_colour=border_colour
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_colour=border_colour
    )

    matrix_to_plot = data_matrix[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH,
        ...
    ]
    latitude_matrix_deg = latitude_matrix_deg[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH,
        ...
    ]
    longitude_matrix_deg = longitude_matrix_deg[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH,
        ...
    ]

    return {
        FIGURE_OBJECT_KEY: figure_object,
        AXES_OBJECT_KEY: axes_object,
        BASEMAP_OBJECT_KEY: basemap_object,
        MATRIX_TO_PLOT_KEY: matrix_to_plot,
        LATITUDES_KEY: latitude_matrix_deg,
        LONGITUDES_KEY: longitude_matrix_deg
    }


def plot_monte_carlo_diff(
        difference_matrix, significance_matrix, colour_map_object, title_string,
        output_file_name, max_colour_percentile=None, max_colour_value=None):
    """Plots diff between two composites, along with Monte Carlo significance.

    M = number of rows in grid
    N = number of columns in grid

    :param difference_matrix: M-by-N numpy array with difference between means
        (trial composite minus baseline composite) at each grid cell.
    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where difference is significant.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_percentile: [may be None]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all absolute values in `difference_matrix`, where q =
        `max_colour_percentile`.  Minimum value will be -1 * max value.
    :param max_colour_value: [used only if `max_colour_percentile is None`]
        Max value in colour scheme.  Minimum value will be -1 * max value.
    """

    basemap_dict = plot_basemap(
        data_matrix=difference_matrix, border_colour=MONTE_CARLO_BORDER_COLOUR
    )

    figure_object = basemap_dict[FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[AXES_OBJECT_KEY]
    basemap_object = basemap_dict[BASEMAP_OBJECT_KEY]
    diff_matrix_to_plot = basemap_dict[MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[LONGITUDES_KEY]

    sig_matrix_to_plot = significance_matrix[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
    ]

    if max_colour_percentile is not None:
        max_colour_value = numpy.nanpercentile(
            numpy.absolute(diff_matrix_to_plot), max_colour_percentile
        )

    colour_norm_object = pyplot.Normalize(
        vmin=-max_colour_value, vmax=max_colour_value
    )

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=diff_matrix_to_plot,
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
        marker=SIG_MARKER_TYPE, markerfacecolor=SIG_MARKER_COLOUR,
        markeredgecolor=SIG_MARKER_COLOUR, markersize=SIG_MARKER_SIZE * 2,
        markeredgewidth=SIG_MARKER_EDGE_WIDTH
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=diff_matrix_to_plot,
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


def _run(statistic_file_name, monte_carlo_file_name, length_colour_map_name,
         area_colour_map_name, diff_colour_map_name, max_colour_percentile,
         wf_colour_maxima, cf_colour_maxima, monte_carlo_colour_maxima,
         monte_carlo_max_fdr, output_dir_name):
    """Plots statistics for gridded front properties.

    This is effectively the main method.

    :param statistic_file_name: See documentation at top of file.
    :param monte_carlo_file_name: Same.
    :param length_colour_map_name: Same.
    :param area_colour_map_name: Same.
    :param diff_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param wf_colour_maxima: Same.
    :param cf_colour_maxima: Same.
    :param monte_carlo_colour_maxima: Same.
    :param monte_carlo_max_fdr: Same.
    :param output_dir_name: Same.
    """

    if max_colour_percentile <= 0:
        max_colour_percentile = None
    else:
        error_checking.assert_is_greater(max_colour_percentile, 50.)
        error_checking.assert_is_leq(max_colour_percentile, 100.)

    if statistic_file_name in ['', 'None']:
        statistic_file_name = None
    if monte_carlo_file_name in ['', 'None']:
        monte_carlo_file_name = None
    if monte_carlo_max_fdr <= 0:
        monte_carlo_max_fdr = None

    length_colour_map_object = pyplot.get_cmap(length_colour_map_name)
    area_colour_map_object = pyplot.get_cmap(area_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if statistic_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(statistic_file_name))
        front_statistic_dict = climo_utils.read_gridded_stats(
            statistic_file_name
        )

        first_time_string = time_conversion.unix_sec_to_string(
            front_statistic_dict[climo_utils.FIRST_TIME_KEY], TITLE_TIME_FORMAT
        )
        last_time_string = time_conversion.unix_sec_to_string(
            front_statistic_dict[climo_utils.LAST_TIME_KEY], TITLE_TIME_FORMAT
        )
        this_title_string = 'Mean WF length (km) from {0:s} to {1:s}'.format(
            first_time_string, last_time_string
        )

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
            output_dir_name
        )

        num_wf_labels_matrix = front_statistic_dict[
            climo_utils.NUM_WF_LABELS_KEY
        ]
        mean_wf_length_matrix_km = (
            front_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY] * METRES_TO_KM
        )
        mean_wf_length_matrix_km[
            num_wf_labels_matrix < MASK_IF_NUM_LABELS_BELOW
        ] = numpy.nan

        _plot_one_statistic(
            statistic_matrix=mean_wf_length_matrix_km,
            colour_map_object=length_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=wf_colour_maxima[0],
            title_string=this_title_string,
            output_file_name=this_output_file_name
        )

        this_title_string = this_title_string.replace(
            'Mean WF length', 'Mean CF length')
        this_output_file_name = '{0:s}/mean_cf_length.jpg'.format(
            output_dir_name
        )

        num_cf_labels_matrix = front_statistic_dict[
            climo_utils.NUM_CF_LABELS_KEY
        ]
        mean_cf_length_matrix_km = (
            front_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY] * METRES_TO_KM
        )
        mean_cf_length_matrix_km[
            num_cf_labels_matrix < MASK_IF_NUM_LABELS_BELOW
        ] = numpy.nan

        _plot_one_statistic(
            statistic_matrix=mean_cf_length_matrix_km,
            colour_map_object=length_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=cf_colour_maxima[0],
            title_string=this_title_string,
            output_file_name=this_output_file_name
        )

        this_title_string = this_title_string.replace(
            'Mean CF length (km)', r'Mean WF area ($\times$ 1000 km$^2$)'
        )
        this_output_file_name = '{0:s}/mean_wf_area.jpg'.format(output_dir_name)

        mean_wf_area_matrix_thousand_km2 = (
            front_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY] *
            METRES2_TO_THOUSAND_KM2
        )
        mean_wf_area_matrix_thousand_km2[
            num_wf_labels_matrix < MASK_IF_NUM_LABELS_BELOW
        ] = numpy.nan

        _plot_one_statistic(
            statistic_matrix=mean_wf_area_matrix_thousand_km2,
            colour_map_object=area_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=wf_colour_maxima[1],
            title_string=this_title_string,
            output_file_name=this_output_file_name
        )

        this_title_string = this_title_string.replace(
            'Mean WF area', 'Mean CF area'
        )
        this_output_file_name = '{0:s}/mean_cf_area.jpg'.format(output_dir_name)

        mean_cf_area_matrix_thousand_km2 = (
            front_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY] *
            METRES2_TO_THOUSAND_KM2
        )
        mean_cf_area_matrix_thousand_km2[
            num_cf_labels_matrix < MASK_IF_NUM_LABELS_BELOW
        ] = numpy.nan

        _plot_one_statistic(
            statistic_matrix=mean_cf_area_matrix_thousand_km2,
            colour_map_object=area_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=cf_colour_maxima[1],
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
            output_dir_name
        )

        conversion_ratio = METRES_TO_KM
        colour_map_object = length_colour_map_object

    elif property_name == climo_utils.WF_AREA_PROPERTY_NAME:
        this_title_string = (
            r'Mean WF area ($\times$ 1000 km$^2$) for baseline composite'
        )
        this_output_file_name = '{0:s}/wf_area_baseline-mean.jpg'.format(
            output_dir_name
        )

        conversion_ratio = METRES2_TO_THOUSAND_KM2
        colour_map_object = area_colour_map_object

    elif property_name == climo_utils.CF_LENGTH_PROPERTY_NAME:
        this_title_string = 'Mean CF length (km) for baseline composite'
        this_output_file_name = '{0:s}/cf_length_baseline-mean.jpg'.format(
            output_dir_name
        )

        conversion_ratio = METRES_TO_KM
        colour_map_object = length_colour_map_object

    elif property_name == climo_utils.CF_AREA_PROPERTY_NAME:
        this_title_string = (
            r'Mean CF area ($\times$ 1000 km$^2$) for baseline composite'
        )
        this_output_file_name = '{0:s}/cf_area_baseline-mean.jpg'.format(
            output_dir_name
        )

        conversion_ratio = METRES2_TO_THOUSAND_KM2
        colour_map_object = area_colour_map_object

    num_labels_matrix = monte_carlo_dict[climo_utils.NUM_LABELS_MATRIX_KEY]
    baseline_mean_matrix = (
        monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY] * conversion_ratio
    )
    baseline_mean_matrix[
        num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
    ] = numpy.nan

    trial_mean_matrix = (
        monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY] * conversion_ratio
    )
    trial_mean_matrix[num_labels_matrix < MASK_IF_NUM_LABELS_BELOW] = numpy.nan

    p_value_matrix = monte_carlo_dict[climo_utils.P_VALUE_MATRIX_KEY]

    if monte_carlo_max_fdr is None:
        significance_matrix = p_value_matrix <= 0.05
    else:
        significance_matrix = climo_utils.find_sig_grid_points(
            p_value_matrix=p_value_matrix,
            max_false_discovery_rate=monte_carlo_max_fdr
        )

    significance_matrix[num_labels_matrix < MASK_IF_NUM_LABELS_BELOW] = False

    if max_colour_percentile is None:
        max_colour_value = monte_carlo_colour_maxima[0]
    else:
        concat_mean_matrix = numpy.concatenate(
            (baseline_mean_matrix, trial_mean_matrix), axis=0
        )
        max_colour_value = numpy.nanpercentile(
            concat_mean_matrix, max_colour_percentile
        )

    _plot_one_statistic(
        statistic_matrix=baseline_mean_matrix,
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        title_string=this_title_string, output_file_name=this_output_file_name
    )

    this_title_string = this_title_string.replace('baseline', 'trial')
    this_output_file_name = this_output_file_name.replace(
        'baseline-mean.jpg', 'trial-mean.jpg'
    )

    _plot_one_statistic(
        statistic_matrix=trial_mean_matrix,
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        title_string=this_title_string, output_file_name=this_output_file_name
    )

    this_title_string = this_title_string.replace(' for trial composite', '')
    this_title_string = this_title_string.replace(
        'Mean', 'Composite difference (trial minus baseline)'
    )
    this_output_file_name = this_output_file_name.replace(
        'trial-mean.jpg', 'difference.jpg'
    )

    plot_monte_carlo_diff(
        difference_matrix=trial_mean_matrix - baseline_mean_matrix,
        significance_matrix=significance_matrix,
        colour_map_object=diff_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        max_colour_value=monte_carlo_colour_maxima[1],
        title_string=this_title_string, output_file_name=this_output_file_name
    )


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
        wf_colour_maxima=numpy.array(
            getattr(INPUT_ARG_OBJECT, WF_COLOUR_MAXIMA_ARG_NAME), dtype=float
        ),
        cf_colour_maxima=numpy.array(
            getattr(INPUT_ARG_OBJECT, CF_COLOUR_MAXIMA_ARG_NAME), dtype=float
        ),
        monte_carlo_colour_maxima=numpy.array(
            getattr(INPUT_ARG_OBJECT, MC_COLOUR_MAXIMA_ARG_NAME), dtype=float
        ),
        monte_carlo_max_fdr=getattr(INPUT_ARG_OBJECT, MAX_FDR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
