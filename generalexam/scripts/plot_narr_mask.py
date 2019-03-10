"""Plots mask for NARR grid.

For details on what this mask means, see create_narr_mask.py.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import narr_plotting

PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.
BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 600
CONCAT_FIGURE_SIZE_PIXELS = int(1e7)

INPUT_FILE_ARG_NAME = 'input_mask_file_name'
WF_COLOUR_MAP_ARG_NAME = 'wf_colour_map_name'
CF_COLOUR_MAP_ARG_NAME = 'cf_colour_map_name'
MASK_COLOUR_MAP_ARG_NAME = 'mask_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_for_colours'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing NARR mask.  Will be read by '
    '`machine_learning_utils.read_narr_mask`.')

WF_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for number of warm fronts.  This name must be accepted '
    'by pyplot, since the command `pyplot.cm.get_cmap` will be used to convert '
    'it from a name to an object.')

CF_COLOUR_MAP_HELP_STRING = 'Same as `{0:s}` but for cold fronts.'.format(
    WF_COLOUR_MAP_ARG_NAME)

MASK_COLOUR_MAP_HELP_STRING = (
    'Same as `{0:s}` but for the mask itself.  Masked grid cells will be in '
    'white; unmasked grid cells will be treated as the highest possible value, '
    'thus receiving the top colour in the scheme.'
).format(WF_COLOUR_MAP_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile for maps with num warm fronts and num cold fronts.  The '
    'highest value in the colour scheme will be [q]th percentile of all values '
    'in the grid, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

WF_COLOUR_MAP_NAME_DEFAULT = 'YlOrRd'
CF_COLOUR_MAP_NAME_DEFAULT = 'YlGnBu'
MASK_COLOUR_MAP_NAME_DEFAULT = 'winter'
MAX_PERCENTILE_DEFAULT = 99.

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WF_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default=WF_COLOUR_MAP_NAME_DEFAULT, help=WF_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CF_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default=CF_COLOUR_MAP_NAME_DEFAULT, help=CF_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MASK_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default=MASK_COLOUR_MAP_NAME_DEFAULT, help=MASK_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=MAX_PERCENTILE_DEFAULT, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _make_one_plot(
        colour_map_object, title_string, label_string, output_file_name,
        num_fronts_matrix=None, max_percentile_for_colours=None,
        mask_matrix=None):
    """Creates one plot.

    This plot will be a gridded map with one of 3 quantities.  Instructions for
    plotting each type are given below.

    - Number of warm fronts at each grid cell:
      make `num_fronts_matrix` = num warm fronts and leave `mask_matrix` alone
    - Number of cold fronts at each grid cell:
      make `num_fronts_matrix` = num cold fronts and leave `mask_matrix` alone
    - The mask itself: include `mask_matrix` and leave `num_fronts_matrix` alone

    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param title_string: Title (will be printed above figure).
    :param label_string: Label (will be printed at top-left of figure).  This is
        usually a parenthesized letter -- like "(a)" -- which will be useful
        when the plot is a panel in a larger figure.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param num_fronts_matrix: M-by-N numpy array with number of fronts at each
        grid cell.
    :param max_percentile_for_colours:
        [used only if `num_fronts_matrix is not None`]
        See documentation at top of file.
    :param mask_matrix: M-by-N numpy array, where 0 means that the grid cell is
        masked.
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

    if num_fronts_matrix is None:
        matrix_to_plot = (mask_matrix + 0).astype(float)
        matrix_to_plot[matrix_to_plot == 0] = numpy.nan

        narr_plotting.plot_xy_grid(
            data_matrix=matrix_to_plot, axes_object=axes_object,
            basemap_object=basemap_object, colour_map=colour_map_object,
            colour_minimum=0., colour_maximum=1.)
    else:
        num_fronts_matrix = num_fronts_matrix.astype(float)
        max_value_for_colours = numpy.percentile(
            num_fronts_matrix, max_percentile_for_colours)

        narr_plotting.plot_xy_grid(
            data_matrix=num_fronts_matrix, axes_object=axes_object,
            basemap_object=basemap_object, colour_map=colour_map_object,
            colour_minimum=0., colour_maximum=max_value_for_colours)

        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=axes_object, values_to_colour=num_fronts_matrix,
            colour_map=colour_map_object, colour_min=0.,
            colour_max=max_value_for_colours, orientation='horizontal',
            extend_min=False, extend_max=True)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=label_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(input_mask_file_name, wf_colour_map_name, cf_colour_map_name,
         mask_colour_map_name, max_percentile_for_colours, output_dir_name):
    """Plots mask for NARR grid.

    This is effectively the main method.

    :param input_mask_file_name: See documentation at top of file.
    :param wf_colour_map_name: Same.
    :param cf_colour_map_name: Same.
    :param mask_colour_map_name: Same.
    :param max_percentile_for_colours: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading data from: "{0:s}"...'.format(input_mask_file_name)
    mask_matrix, num_warm_fronts_matrix, num_cold_fronts_matrix = (
        ml_utils.read_narr_mask(input_mask_file_name)
    )

    warm_front_file_name = '{0:s}/num_warm_fronts.jpg'.format(output_dir_name)
    _make_one_plot(
        colour_map_object=pyplot.cm.get_cmap(wf_colour_map_name),
        title_string='Number of warm fronts', label_string='(a)',
        output_file_name=warm_front_file_name,
        num_fronts_matrix=num_warm_fronts_matrix,
        max_percentile_for_colours=max_percentile_for_colours)

    cold_front_file_name = '{0:s}/num_cold_fronts.jpg'.format(output_dir_name)
    _make_one_plot(
        colour_map_object=pyplot.cm.get_cmap(cf_colour_map_name),
        title_string='Number of cold fronts', label_string='(b)',
        output_file_name=cold_front_file_name,
        num_fronts_matrix=num_cold_fronts_matrix,
        max_percentile_for_colours=max_percentile_for_colours)

    mask_file_name = '{0:s}/narr_mask.jpg'.format(output_dir_name)
    _make_one_plot(
        colour_map_object=pyplot.cm.get_cmap(mask_colour_map_name),
        title_string='Mask', label_string='(c)',
        output_file_name=mask_file_name, mask_matrix=mask_matrix)

    concat_file_name = '{0:s}/narr_mask_3panel.jpg'.format(output_dir_name)
    panel_file_names = [
        warm_front_file_name, cold_front_file_name, mask_file_name
    ]

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PIXELS)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_mask_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        wf_colour_map_name=getattr(INPUT_ARG_OBJECT, WF_COLOUR_MAP_ARG_NAME),
        cf_colour_map_name=getattr(INPUT_ARG_OBJECT, CF_COLOUR_MAP_ARG_NAME),
        mask_colour_map_name=getattr(
            INPUT_ARG_OBJECT, MASK_COLOUR_MAP_ARG_NAME),
        max_percentile_for_colours=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )