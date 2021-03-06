"""Creates mask, indicating where human forecasters usually draw fronts.

This mask will be defined over the NARR grid.
"""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import narr_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.
BORDER_COLOUR = numpy.full(3, 0.)
OUTPUT_RESOLUTION_DPI = 600

WARM_FRONT_COLOUR_MAP_OBJECT = pyplot.cm.YlOrRd
COLD_FRONT_COLOUR_MAP_OBJECT = pyplot.cm.YlGnBu
BOTH_FRONTS_COLOUR_MAP_OBJECT = pyplot.cm.winter
MAX_COLOUR_PERCENTILE = 99.

FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
MIN_FRONTS_ARG_NAME = 'min_num_fronts'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids.  Files therein will be '
    'found by `fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_narr_grids_from_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Frontal grids will be read for `{0:s}`...'
    '`{1:s}`, and the number of both warm and cold fronts at each grid cell '
    'will be counted.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'At each time step, both warm and cold fronts will be dilated with this '
    'distance buffer.')

MIN_FRONTS_HELP_STRING = (
    'Masking threshold.  Any grid cell with >= `{0:s}` fronts will be retained.'
    '  All other grid cells will be masked out.'
).format(MIN_FRONTS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Results will be saved here.')

DEFAULT_TOP_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_DILATION_DISTANCE_METRES = 50000.
DEFAULT_MIN_FRONTS = 100

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_FRONTAL_GRID_DIR_NAME,
    help=FRONTAL_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=DEFAULT_DILATION_DISTANCE_METRES,
    help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_FRONTS_ARG_NAME, type=int, required=False,
    default=DEFAULT_MIN_FRONTS,
    help=MIN_FRONTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_front_densities(
        num_fronts_matrix, colour_map_object, title_string, annotation_string,
        output_file_name, mask_matrix=None, add_colour_bar=True):
    """Plots number of fronts at each NARR grid cell.

    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)

    :param num_fronts_matrix: M-by-N numpy array with number of fronts at each
        grid cell.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param output_file_name: Path to output (image) file.  The figure will be
        saved here.
    :param mask_matrix: M-by-N numpy array of integers.  If
        mask_matrix[i, j] = 0, grid cell [i, j] will be masked out in the map.
        If `mask_matrix is None`, there will be no masking.
    :param add_colour_bar: Boolean flag.  If True, will add colour bar.
    """

    _, axes_object, basemap_object = narr_plotting.init_basemap()

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
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG)

    num_fronts_matrix = num_fronts_matrix.astype(float)
    max_colour_value = numpy.percentile(
        num_fronts_matrix, MAX_COLOUR_PERCENTILE)

    if mask_matrix is not None:
        num_fronts_matrix[mask_matrix == 0] = numpy.nan

    narr_plotting.plot_xy_grid(
        data_matrix=num_fronts_matrix, axes_object=axes_object,
        basemap_object=basemap_object, colour_map=colour_map_object,
        colour_minimum=0., colour_maximum=max_colour_value)

    if add_colour_bar:
        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=axes_object, values_to_colour=num_fronts_matrix,
            colour_map=colour_map_object, colour_min=0.,
            colour_max=max_colour_value, orientation='horizontal',
            extend_min=False, extend_max=True)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(top_frontal_grid_dir_name, first_time_string, last_time_string,
         dilation_distance_metres, min_num_fronts, output_dir_name):
    """Creates mask, indicating where human forecasters usually draw fronts.

    This is effectively the main method.

    :param top_frontal_grid_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param dilation_distance_metres: Same.
    :param min_num_fronts: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)
    error_checking.assert_is_greater(min_num_fronts, 0)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    num_times = len(valid_times_unix_sec)
    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    num_cold_fronts_matrix = None
    num_warm_fronts_matrix = None

    for i in range(num_times):
        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)
        if not os.path.isfile(this_file_name):
            warning_string = (
                'POTENTIAL PROBLEM.  Cannot find file: "{0:s}"'
            ).format(this_file_name)
            warnings.warn(warning_string)
            continue

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
            this_file_name)

        this_frontal_grid_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=this_frontal_grid_table,
            num_rows_per_image=num_grid_rows,
            num_columns_per_image=num_grid_columns)

        this_frontal_grid_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_frontal_grid_matrix,
            dilation_distance_metres=dilation_distance_metres, verbose=False)
        this_frontal_grid_matrix = this_frontal_grid_matrix[0, ...]

        this_num_cold_fronts_matrix = (
            this_frontal_grid_matrix == front_utils.COLD_FRONT_INTEGER_ID
        ).astype(int)
        this_num_warm_fronts_matrix = (
            this_frontal_grid_matrix == front_utils.WARM_FRONT_INTEGER_ID
        ).astype(int)

        if num_cold_fronts_matrix is None:
            num_cold_fronts_matrix = this_num_cold_fronts_matrix + 0
            num_warm_fronts_matrix = this_num_warm_fronts_matrix + 0
        else:
            num_cold_fronts_matrix = (
                num_cold_fronts_matrix + this_num_cold_fronts_matrix)
            num_warm_fronts_matrix = (
                num_warm_fronts_matrix + this_num_warm_fronts_matrix)

    print SEPARATOR_STRING

    print 'Masking out grid cells with < {0:d} fronts...'.format(
        min_num_fronts)
    num_both_fronts_matrix = num_warm_fronts_matrix + num_cold_fronts_matrix
    mask_matrix = (num_both_fronts_matrix >= min_num_fronts).astype(int)

    pickle_file_name = '{0:s}/narr_mask.p'.format(output_dir_name)
    print 'Writing mask to: "{0:s}"...'.format(pickle_file_name)
    ml_utils.write_narr_mask(
        mask_matrix=mask_matrix, pickle_file_name=pickle_file_name)

    warm_front_map_file_name = '{0:s}/num_warm_fronts.jpg'.format(
        output_dir_name)
    _plot_front_densities(
        num_fronts_matrix=num_warm_fronts_matrix,
        colour_map_object=WARM_FRONT_COLOUR_MAP_OBJECT,
        title_string='Number of warm fronts', annotation_string='(a)',
        output_file_name=warm_front_map_file_name, mask_matrix=None,
        add_colour_bar=True)

    cold_front_map_file_name = '{0:s}/num_cold_fronts.jpg'.format(
        output_dir_name)
    _plot_front_densities(
        num_fronts_matrix=num_cold_fronts_matrix,
        colour_map_object=COLD_FRONT_COLOUR_MAP_OBJECT,
        title_string='Number of cold fronts', annotation_string='(b)',
        output_file_name=cold_front_map_file_name, mask_matrix=None,
        add_colour_bar=True)

    both_fronts_title_string = 'Grid cells with at least {0:d} fronts'.format(
        min_num_fronts)
    both_fronts_map_file_name = '{0:s}/num_both_fronts.jpg'.format(
        output_dir_name)
    num_both_fronts_matrix[num_both_fronts_matrix > 1] = 1

    _plot_front_densities(
        num_fronts_matrix=num_both_fronts_matrix,
        colour_map_object=BOTH_FRONTS_COLOUR_MAP_OBJECT,
        title_string=both_fronts_title_string, annotation_string='(c)',
        output_file_name=both_fronts_map_file_name,
        mask_matrix=mask_matrix, add_colour_bar=False)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        min_num_fronts=getattr(
            INPUT_ARG_OBJECT, MIN_FRONTS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
