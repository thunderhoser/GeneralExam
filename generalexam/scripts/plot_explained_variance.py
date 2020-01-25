"""Plots explained variance in WF or CF frequency at each grid point."""

import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
MAX_VALUE_ARG_NAME = 'max_colour_value'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
TITLE_ARG_NAME = 'title_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`climatology_utils.read_explained_variances`.'
)
MAX_VALUE_HELP_STRING = (
    'Max value in colour scheme.  If you want to specify max percentile '
    'instead, leave this argument alone.'
)

MAX_PERCENTILE_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Determines max value in colour scheme'
    ' ([q]th percentile of all values in grid, where q = `{1:s}`).'
).format(MAX_VALUE_ARG_NAME, MAX_PERCENTILE_ARG_NAME)

TITLE_HELP_STRING = 'Title (will be printed above figure).'
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TITLE_ARG_NAME, type=str, required=False, default='',
    help=TITLE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_name, max_colour_value, max_colour_percentile,
         title_string, output_file_name):
    """Plots explained variance in WF or CF frequency at each grid point.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param max_colour_value: Same.
    :param max_colour_percentile: Same.
    :param title_string: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if max_colour_value <= 0:
        max_colour_value = None

    print('Reading data from file: "{0:s}"...'.format(input_file_name))
    exp_variance_dict = climo_utils.read_explained_variances(input_file_name)
    exp_variance_matrix = exp_variance_dict[climo_utils.EXP_VARIANCE_MATRIX_KEY]

    basemap_dict = plot_gridded_stats.plot_basemap(
        data_matrix=exp_variance_matrix
    )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    matrix_to_plot = basemap_dict[plot_gridded_stats.MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    if max_colour_value is None:
        max_colour_value = numpy.nanpercentile(
            matrix_to_plot, max_colour_percentile
        )

    colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

    prediction_plotting.plot_counts_on_general_grid(
        count_or_frequency_matrix=matrix_to_plot,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.05,
        extend_min=False, extend_max=True, fraction_of_axis_length=1.
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.3f}'.format(x) for x in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_VALUE_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        title_string=getattr(INPUT_ARG_OBJECT, TITLE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
