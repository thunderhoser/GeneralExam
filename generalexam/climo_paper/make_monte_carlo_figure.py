"""Makes 8-panel figure with Monte Carlo results for either freq or length."""

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
METRES2_TO_THOUSAND_KM2 = 1e-9
MASK_IF_NUM_LABELS_BELOW = 100

NUM_ROWS_IN_CNN_PATCH = plot_gridded_stats.NUM_ROWS_IN_CNN_PATCH
NUM_COLUMNS_IN_CNN_PATCH = plot_gridded_stats.NUM_COLUMNS_IN_CNN_PATCH

FIRST_TIME_UNIX_SEC = time_conversion.first_and_last_times_in_year(1979)[0]
LAST_TIME_UNIX_SEC = (
    time_conversion.first_and_last_times_in_year(2018)[1] - 10799
)

FREQUENCY_PROPERTY_NAME = 'frequency'
LENGTH_PROPERTY_NAME = 'length'
AREA_PROPERTY_NAME = 'area'
VALID_PROPERTY_NAMES = [
    FREQUENCY_PROPERTY_NAME, LENGTH_PROPERTY_NAME, AREA_PROPERTY_NAME
]

PROPERTY_ABBREV_TO_VERBOSE_DICT = {
    climo_utils.WF_FREQ_PROPERTY_NAME: 'WF frequency',
    climo_utils.CF_FREQ_PROPERTY_NAME: 'CF frequency',
    climo_utils.WF_LENGTH_PROPERTY_NAME: 'WF length (km)',
    climo_utils.CF_LENGTH_PROPERTY_NAME: 'CF length (km)',
    climo_utils.WF_AREA_PROPERTY_NAME: r'WF area ($\times$ 1000 km$^{2}$)',
    climo_utils.CF_AREA_PROPERTY_NAME: r'CF area ($\times$ 1000 km$^{2}$)'
}

VALID_SEASON_NAMES = climo_utils.VALID_SEASON_STRINGS
VALID_COMPOSITE_NAMES_ABBREV = [
    'strong_la_nina', 'la_nina', 'el_nino', 'strong_el_nino'
]

SIG_MARKER_TYPE = plot_gridded_stats.SIG_MARKER_TYPE
SIG_MARKER_COLOUR = plot_gridded_stats.SIG_MARKER_COLOUR
SIG_MARKER_SIZE = 1.
SIG_MARKER_EDGE_WIDTH = plot_gridded_stats.SIG_MARKER_EDGE_WIDTH

PROPERTY_TO_MAX_COLOUR_VALUE_DICT = {
    climo_utils.WF_FREQ_PROPERTY_NAME: 0.015,
    climo_utils.WF_LENGTH_PROPERTY_NAME: 500.,
    climo_utils.WF_AREA_PROPERTY_NAME: 250.,
    climo_utils.CF_FREQ_PROPERTY_NAME: 0.02,
    climo_utils.CF_LENGTH_PROPERTY_NAME: 1000.,
    climo_utils.CF_AREA_PROPERTY_NAME: 400.
}

COLOUR_MAP_OBJECT = pyplot.get_cmap('bwr')

BORDER_COLOUR = numpy.full(3, 152. / 255)
TITLE_FONT_SIZE = 26
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_dir_name'
PROPERTY_ARG_NAME = 'main_property_name'
COMPOSITE_ARG_NAME = 'composite_name'
SEASONS_ARG_NAME = 'season_names'
MAX_FDR_ARG_NAME = 'monte_carlo_max_fdr'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_monte_carlo_file` and read by '
    '`climatology_utils.read_monte_carlo_test`.'
)

PROPERTY_HELP_STRING = (
    'Will plot Monte Carlo tests for this property.  Must be in the following '
    'list:\n{0:s}'
).format(str(VALID_PROPERTY_NAMES))

COMPOSITE_HELP_STRING = (
    'Composite to plot.  Must be in the following list:\n{0:s}'
).format(str(VALID_COMPOSITE_NAMES_ABBREV))

SEASONS_HELP_STRING = (
    'List of seasons to plot.  Each must be in the following list:\n{0:s}'
).format(str(VALID_SEASON_NAMES))

MAX_FDR_HELP_STRING = (
    'Max FDR (false-discovery rate) for field-based version of Monte Carlo '
    'significance test.  If you do not want to use field-based version, leave '
    'this argument alone.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROPERTY_ARG_NAME, type=str, required=True,
    help=PROPERTY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_ARG_NAME, type=str, required=True,
    help=COMPOSITE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEASONS_ARG_NAME, type=str, nargs='+', required=False,
    default=VALID_SEASON_NAMES, help=SEASONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FDR_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_FDR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _plot_one_difference(
        difference_matrix, significance_matrix, sig_marker_size,
        max_colour_value, plot_latitudes, plot_longitudes, plot_colour_bar,
        title_string, letter_label, output_file_name):
    """Plots difference for one composite in one season.

    M = number of rows in grid
    N = number of columns in grid

    :param difference_matrix: M-by-N numpy array with differences (trial period
        minus baseline period).
    :param significance_matrix: M-by-N numpy array of Boolean flags, indicating
        where difference is significant.
    :param sig_marker_size: Marker size for stippling.
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

    basemap_dict = plot_gridded_stats.plot_basemap(
        data_matrix=difference_matrix, border_colour=BORDER_COLOUR
    )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    diff_matrix_to_plot = difference_matrix[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
    ]
    sig_matrix_to_plot = significance_matrix[
        NUM_ROWS_IN_CNN_PATCH:-NUM_ROWS_IN_CNN_PATCH,
        NUM_COLUMNS_IN_CNN_PATCH:-NUM_COLUMNS_IN_CNN_PATCH
    ]

    colour_norm_object = pyplot.Normalize(
        vmin=-max_colour_value, vmax=max_colour_value
    )

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=diff_matrix_to_plot,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )

    sig_latitudes_deg = latitude_matrix_deg[sig_matrix_to_plot == True]
    sig_longitudes_deg = longitude_matrix_deg[sig_matrix_to_plot == True]
    sig_x_coords_metres, sig_y_coords_metres = basemap_object(
        sig_longitudes_deg, sig_latitudes_deg
    )

    axes_object.plot(
        sig_x_coords_metres, sig_y_coords_metres,
        linestyle='None', marker=SIG_MARKER_TYPE,
        markerfacecolor=SIG_MARKER_COLOUR, markeredgecolor=SIG_MARKER_COLOUR,
        markersize=sig_marker_size, markeredgewidth=SIG_MARKER_EDGE_WIDTH
    )

    if not plot_latitudes:
        axes_object.set_yticklabels([])
    if not plot_longitudes:
        axes_object.set_xticklabels([])

    if plot_colour_bar:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=diff_matrix_to_plot,
            colour_map_object=COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', padding=0.05,
            extend_min=True, extend_max=True, fraction_of_axis_length=1.
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


def _run(top_input_dir_name, main_property_name, composite_name_abbrev,
         season_names, monte_carlo_max_fdr, output_file_name):
    """Makes 8-panel figure with Monte Carlo results for either freq or length.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param main_property_name: Same.
    :param composite_name_abbrev: Same.
    :param season_names: Same.
    :param monte_carlo_max_fdr: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if monte_carlo_max_fdr <= 0:
        monte_carlo_max_fdr = None

    conversion_ratio = None
    property_names = None

    if main_property_name == FREQUENCY_PROPERTY_NAME:
        conversion_ratio = 1.

        property_names = [
            climo_utils.WF_FREQ_PROPERTY_NAME, climo_utils.CF_FREQ_PROPERTY_NAME
        ]
    elif main_property_name == LENGTH_PROPERTY_NAME:
        conversion_ratio = METRES_TO_KM

        property_names = [
            climo_utils.WF_LENGTH_PROPERTY_NAME,
            climo_utils.CF_LENGTH_PROPERTY_NAME
        ]
    elif main_property_name == AREA_PROPERTY_NAME:
        conversion_ratio = METRES2_TO_THOUSAND_KM2

        property_names = [
            climo_utils.WF_AREA_PROPERTY_NAME, climo_utils.CF_AREA_PROPERTY_NAME
        ]

    output_dir_name, pathless_output_file_name = os.path.split(output_file_name)
    extensionless_output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, os.path.splitext(pathless_output_file_name)[0]
    )

    num_properties = len(property_names)
    num_seasons = len(season_names)
    panel_file_names = []
    letter_label = None

    sig_marker_size = SIG_MARKER_SIZE * (
        1 + int(monte_carlo_max_fdr is not None)
    )

    for j in range(num_seasons):
        for i in range(num_properties):
            this_max_colour_value = PROPERTY_TO_MAX_COLOUR_VALUE_DICT[
                property_names[i]
            ]

            this_input_dir_name = '{0:s}/{1:s}/{2:s}/stitched'.format(
                top_input_dir_name, composite_name_abbrev, season_names[j]
            )

            this_input_file_name = climo_utils.find_monte_carlo_file(
                directory_name=this_input_dir_name,
                property_name=property_names[i],
                first_grid_row=0, first_grid_column=0,
                raise_error_if_missing=True
            )

            print('Reading data from: "{0:s}"...'.format(this_input_file_name))
            this_monte_carlo_dict = climo_utils.read_monte_carlo_test(
                this_input_file_name
            )

            this_num_labels_matrix = (
                this_monte_carlo_dict[climo_utils.NUM_LABELS_MATRIX_KEY]
            )
            this_difference_matrix = conversion_ratio * (
                this_monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY] -
                this_monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY]
            )
            this_p_value_matrix = (
                this_monte_carlo_dict[climo_utils.P_VALUE_MATRIX_KEY]
            )

            if monte_carlo_max_fdr is None:
                this_significance_matrix = this_p_value_matrix <= 0.05
            else:
                this_significance_matrix = climo_utils.find_sig_grid_points(
                    p_value_matrix=this_p_value_matrix,
                    max_false_discovery_rate=monte_carlo_max_fdr
                )

            this_difference_matrix[
                this_num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
            ] = numpy.nan
            this_significance_matrix[
                this_num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
            ] = False

            this_title_string = (
                'Composite difference for {0:s} in {1:s}'
            ).format(
                PROPERTY_ABBREV_TO_VERBOSE_DICT[property_names[i]],
                season_names[j]
            )

            this_output_file_name = '{0:s}_{1:s}_{2:s}.jpg'.format(
                extensionless_output_file_name, composite_name_abbrev[i],
                season_names[j]
            )
            panel_file_names.append(this_output_file_name)

            if letter_label is None:
                letter_label = 'a'
            else:
                letter_label = chr(ord(letter_label) + 1)

            _plot_one_difference(
                difference_matrix=this_difference_matrix,
                significance_matrix=this_significance_matrix,
                sig_marker_size=sig_marker_size,
                max_colour_value=this_max_colour_value,
                plot_latitudes=i == 0, plot_longitudes=j == num_seasons - 1,
                plot_colour_bar=j == num_seasons - 1,
                title_string=this_title_string, letter_label=letter_label,
                output_file_name=panel_file_names[-1]
            )

    print('Concatenating panels to: "{0:s}"...'.format(output_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=output_file_name,
        num_panel_rows=num_seasons, num_panel_columns=num_properties
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
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        main_property_name=getattr(INPUT_ARG_OBJECT, PROPERTY_ARG_NAME),
        composite_name_abbrev=getattr(INPUT_ARG_OBJECT, COMPOSITE_ARG_NAME),
        season_names=getattr(INPUT_ARG_OBJECT, SEASONS_ARG_NAME),
        monte_carlo_max_fdr=getattr(INPUT_ARG_OBJECT, MAX_FDR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
