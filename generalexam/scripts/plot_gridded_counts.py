"""Plots number of WF and CF labels at each grid cell over a time period."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
BORDER_COLOUR = numpy.full(3, 0.)

MIN_LATITUDE_DEG = 5.
MIN_LONGITUDE_DEG = 200.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 310.

FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
WF_COLOUR_MAP_ARG_NAME = 'wf_colour_map_name'
CF_COLOUR_MAP_ARG_NAME = 'cf_colour_map_name'
PLOT_FREQUENCY_ARG_NAME = 'plot_frequency'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

# TODO(thunderhoser): In gridded-count files, may want num fronts before and
# after applying separation time.

# TODO(thunderhoser): Make titles fancier.

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`climatology_utils.read_gridded_counts`.')

WF_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for warm-front counts.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

CF_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for cold-front counts.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.')

PLOT_FREQUENCY_HELP_STRING = (
    'Boolean flag.  If 1, will plot frequency (fraction of time steps with '
    'front).  If 0, will plot raw count (number of fronts).')

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in colour scheme.  Max value in warm-'
    'front colour scheme will be the [q]th percentile of values at all grid '
    'cells, where q = `{0:s}` -- and likewise for cold fronts.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WF_COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=WF_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CF_COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlGnBu',
    help=CF_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FREQUENCY_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_FREQUENCY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_front_type(
        count_or_frequency_matrix, colour_map_object, max_colour_percentile,
        title_string, output_file_name):
    """Plots gridded counts or frequencies for one front type.

    :param count_or_frequency_matrix: 2-D numpy array with number or frequency
        of fronts at each grid cell.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param max_colour_percentile: See documentation at top of file.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=count_or_frequency_matrix.shape[0],
        num_columns=count_or_frequency_matrix.shape[1]
    )

    full_grid_row_limits, full_grid_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name)
    )

    print(full_grid_row_limits)
    print(full_grid_column_limits)

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

    colour_norm_object = pyplot.Normalize(
        vmin=0,
        vmax=numpy.percentile(count_or_frequency_matrix, max_colour_percentile)
    )

    prediction_plotting.plot_gridded_counts(
        count_or_frequency_matrix=count_or_frequency_matrix,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, full_grid_name=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=count_or_frequency_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=True,
        fraction_of_axis_length=0.9)

    pyplot.title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(input_file_name, wf_colour_map_name, cf_colour_map_name,
         plot_frequency, max_colour_percentile, output_dir_name):
    """Plots number of WF and CF labels at each grid cell over a time period.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param wf_colour_map_name: Same.
    :param cf_colour_map_name: Same.
    :param plot_frequency: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(max_colour_percentile, 50.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    wf_colour_map_object = pyplot.get_cmap(wf_colour_map_name)
    cf_colour_map_object = pyplot.get_cmap(cf_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    climo_dict = climo_utils.read_gridded_counts(input_file_name)

    warm_front_matrix = climo_dict[climo_utils.NUM_WARM_FRONTS_KEY]
    cold_front_matrix = climo_dict[climo_utils.NUM_COLD_FRONTS_KEY]

    if plot_frequency:
        num_times = len(climo_dict[climo_utils.PREDICTION_FILES_KEY])
        warm_front_matrix = warm_front_matrix.astype(float) / num_times
        cold_front_matrix = cold_front_matrix.astype(float) / num_times

        wf_title_string = 'Frequency'
    else:
        wf_title_string = 'Number'

    wf_title_string += ' of warm fronts'
    wf_output_file_name = '{0:s}/warm_fronts.jpg'.format(output_dir_name)

    _plot_one_front_type(
        count_or_frequency_matrix=warm_front_matrix,
        colour_map_object=wf_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=wf_title_string, output_file_name=wf_output_file_name)

    cf_title_string = wf_title_string.replace('warm', 'cold')
    cf_output_file_name = '{0:s}/cold_fronts.jpg'.format(output_dir_name)

    _plot_one_front_type(
        count_or_frequency_matrix=cold_front_matrix,
        colour_map_object=cf_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        title_string=cf_title_string, output_file_name=cf_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        wf_colour_map_name=getattr(INPUT_ARG_OBJECT, WF_COLOUR_MAP_ARG_NAME),
        cf_colour_map_name=getattr(INPUT_ARG_OBJECT, CF_COLOUR_MAP_ARG_NAME),
        plot_frequency=bool(getattr(INPUT_ARG_OBJECT, PLOT_FREQUENCY_ARG_NAME)),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
