"""Plots dilation of WPC fronts."""

import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting

FRONT_LINE_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/fronts/polylines/201712/'
    'front_locations_2017120100.p')
FRONTAL_GRID_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/fronts/narr_grids/no_dilation/'
    '201712/narr_frontal_grids_2017120100.p')

DILATION_DISTANCE_METRES = 50000.

# MIN_LATITUDE_DEG = 25.
# MIN_LONGITUDE_DEG = 225.
# MAX_LATITUDE_DEG = 80.
# MAX_LONGITUDE_DEG = 295.
# PARALLEL_SPACING_DEG = 10.
# MERIDIAN_SPACING_DEG = 20.

MIN_LATITUDE_DEG = 40.
MIN_LONGITUDE_DEG = 250.
MAX_LATITUDE_DEG = 60.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 5.
MERIDIAN_SPACING_DEG = 10.

BORDER_COLOUR = numpy.full(3, 0.)
FRONT_LINE_WIDTH = 2
FRONT_LINE_OPACITY = 0.5

FIGURE_RESOLUTION_DPI = 600
CONCAT_SIZE_PIXELS = int(1e7)

BEFORE_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'dilation/before_dilation.jpg')
AFTER_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'dilation/after_dilation.jpg')
CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'dilation/dilation.jpg')


def _plot_fronts(front_line_table, ternary_front_matrix, title_string,
                 annotation_string, output_file_name):
    """Plots one set of WPC fronts (either before or after dilation).

    :param front_line_table: See doc for `fronts_io.write_polylines_to_file`.
    :param ternary_front_matrix: numpy array created by
        `machine_learning_utils.dilate_ternary_target_images`.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (narr_row_limits, narr_column_limits
    ) = nwp_plotting.latlng_limits_to_rowcol_limits(
        min_latitude_deg=MIN_LATITUDE_DEG, max_latitude_deg=MAX_LATITUDE_DEG,
        min_longitude_deg=MIN_LONGITUDE_DEG,
        max_longitude_deg=MAX_LONGITUDE_DEG,
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        first_row_in_full_grid=narr_row_limits[0],
        last_row_in_full_grid=narr_row_limits[1],
        first_column_in_full_grid=narr_column_limits[0],
        last_column_in_full_grid=narr_column_limits[1])

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

    this_matrix = ternary_front_matrix[
        0, narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]
    front_plotting.plot_narr_grid(
        frontal_grid_matrix=this_matrix, axes_object=axes_object,
        first_row_in_narr_grid=narr_row_limits[0],
        first_column_in_narr_grid=narr_column_limits[0],
        basemap_object=basemap_object, opacity=FRONT_LINE_OPACITY)

    num_fronts = len(front_line_table.index)
    for i in range(num_fronts):
        front_plotting.plot_polyline(
            latitudes_deg=front_line_table[
                front_utils.LATITUDES_COLUMN].values[i],
            longitudes_deg=front_line_table[
                front_utils.LONGITUDES_COLUMN].values[i],
            basemap_object=basemap_object, axes_object=axes_object,
            front_type=front_line_table[
                front_utils.FRONT_TYPE_COLUMN].values[i],
            line_width=FRONT_LINE_WIDTH)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run():
    """Plots dilation of WPC fronts.

    This is effectively the main method.
    """

    print('Reading data from: "{0:s}"...'.format(FRONT_LINE_FILE_NAME))
    front_line_table = fronts_io.read_polylines_from_file(FRONT_LINE_FILE_NAME)

    print('Reading data from: "{0:s}"...'.format(FRONTAL_GRID_FILE_NAME))
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        FRONTAL_GRID_FILE_NAME)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    ternary_front_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=frontal_grid_table,
        num_rows_per_image=num_grid_rows,
        num_columns_per_image=num_grid_columns)

    _plot_fronts(
        front_line_table=front_line_table,
        ternary_front_matrix=ternary_front_matrix,
        title_string='Observed fronts before dilation', annotation_string='(a)',
        output_file_name=BEFORE_FILE_NAME)

    ternary_front_matrix = ml_utils.dilate_ternary_target_images(
        target_matrix=ternary_front_matrix,
        dilation_distance_metres=DILATION_DISTANCE_METRES, verbose=False)

    _plot_fronts(
        front_line_table=front_line_table,
        ternary_front_matrix=ternary_front_matrix,
        title_string='Observed fronts after dilation', annotation_string='(b)',
        output_file_name=AFTER_FILE_NAME)

    print('Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME))

    imagemagick_utils.concatenate_images(
        input_file_names=[BEFORE_FILE_NAME, AFTER_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=1,
        num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=CONCAT_FILE_NAME, output_file_name=CONCAT_FILE_NAME,
        output_size_pixels=CONCAT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
