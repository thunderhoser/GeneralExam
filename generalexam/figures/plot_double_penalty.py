"""Plots example of double penalty.

The "double penalty" is a problem caused by pixelwise evaluation, where it
unduly punishes the model for predicting the correct object with a small spatial
or temporal offset.  In this case, the model perfectly predicts the morphology
and timing of a front, but the predicted front is one pixel off the observed
front, which causes most pixels in the observed front to be counted as misses,
while most pixels in the predicted front are counted as false alarms.
"""

import copy
import numpy
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

INPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/fronts/narr_grids/no_dilation/'
    '201712/narr_frontal_grids_2017120100.p')

DILATION_DISTANCE_METRES = 50000.

MIN_LATITUDE_DEG = 50.
MIN_LONGITUDE_DEG = 260.
MAX_LATITUDE_DEG = 60.
MAX_LONGITUDE_DEG = 270.
PARALLEL_SPACING_DEG = 2.
MERIDIAN_SPACING_DEG = 2.

BORDER_COLOUR = numpy.full(3, 0.)
BACKGROUND_COLOUR = numpy.full(3, 1.)
NO_FRONT_COLOUR = numpy.full(3, 1.)
ACTUAL_FRONT_COLOUR = numpy.array([247., 129., 191.]) / 255
PREDICTED_FRONT_COLOUR = numpy.array([153., 153., 153.]) / 255

ACTUAL_FRONT_OPACITY = 1.
PREDICTED_FRONT_OPACITY = 0.5
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

NO_DILATION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'double_penalty/no_dilation.jpg')
WITH_DILATION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'double_penalty/with_dilation.jpg')
CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'double_penalty/double_penalty.jpg')


def _get_colour_map(for_actual_fronts):
    """Returns colour map for frontal grid.

    N = number of colours

    :param for_actual_fronts: Boolean flag.  If True, will return colour map for
        actual fronts.  If False, will return colour map for predicted fronts.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    """

    main_colour_list = [NO_FRONT_COLOUR]
    if for_actual_fronts:
        main_colour_list.append(ACTUAL_FRONT_COLOUR)
    else:
        main_colour_list.append(PREDICTED_FRONT_COLOUR)

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(BACKGROUND_COLOUR)
    colour_map_object.set_over(BACKGROUND_COLOUR)

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        numpy.array([0.5, 1.5]), colour_map_object.N)
    return colour_map_object, colour_norm_object


def _plot_fronts(
        actual_binary_matrix, predicted_binary_matrix, annotation_string,
        output_file_name):
    """Plots actual and predicted fronts.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_binary_matrix: M-by-N numpy array.  If
        actual_binary_matrix[i, j] = 1, there is an actual front passing through
        grid cell [i, j].
    :param predicted_binary_matrix: Same but for predicted fronts.
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
        last_column_in_full_grid=narr_column_limits[1], resolution_string='i')

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

    this_colour_map_object, this_colour_norm_object = _get_colour_map(True)
    this_matrix = actual_binary_matrix[
        0, narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]
    nwp_plotting.plot_subgrid(
        field_matrix=this_matrix, model_name=nwp_model_utils.NARR_MODEL_NAME,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map=this_colour_map_object,
        min_value_in_colour_map=this_colour_norm_object.boundaries[0],
        max_value_in_colour_map=this_colour_norm_object.boundaries[-1],
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0],
        opacity=ACTUAL_FRONT_OPACITY)

    this_colour_map_object, this_colour_norm_object = _get_colour_map(False)
    this_matrix = predicted_binary_matrix[
        0, narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]
    nwp_plotting.plot_subgrid(
        field_matrix=this_matrix, model_name=nwp_model_utils.NARR_MODEL_NAME,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map=this_colour_map_object,
        min_value_in_colour_map=this_colour_norm_object.boundaries[0],
        max_value_in_colour_map=this_colour_norm_object.boundaries[-1],
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0],
        opacity=PREDICTED_FRONT_OPACITY)

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run():
    """Plots example of double penalty.

    This is effectively the main method.
    """

    print 'Reading data from: "{0:s}"...'.format(INPUT_FILE_NAME)
    actual_grid_point_table = fronts_io.read_narr_grids_from_file(
        INPUT_FILE_NAME)

    predicted_grid_point_table = copy.deepcopy(actual_grid_point_table)
    predicted_grid_point_table[
        front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN
    ] += 1
    predicted_grid_point_table[
        front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN
    ] += 1
    predicted_grid_point_table[
        front_utils.WARM_FRONT_ROW_INDICES_COLUMN
    ] += 1
    predicted_grid_point_table[
        front_utils.COLD_FRONT_ROW_INDICES_COLUMN
    ] += 1

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    actual_binary_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=actual_grid_point_table,
        num_rows_per_image=num_grid_rows,
        num_columns_per_image=num_grid_columns)
    actual_binary_matrix = ml_utils.binarize_front_images(actual_binary_matrix)

    predicted_binary_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=predicted_grid_point_table,
        num_rows_per_image=num_grid_rows,
        num_columns_per_image=num_grid_columns)
    predicted_binary_matrix = ml_utils.binarize_front_images(
        predicted_binary_matrix)

    _plot_fronts(
        actual_binary_matrix=actual_binary_matrix,
        predicted_binary_matrix=predicted_binary_matrix,
        annotation_string='(a)', output_file_name=NO_DILATION_FILE_NAME)

    actual_binary_matrix = ml_utils.dilate_binary_target_images(
        target_matrix=actual_binary_matrix,
        dilation_distance_metres=DILATION_DISTANCE_METRES, verbose=False)
    predicted_binary_matrix = ml_utils.dilate_binary_target_images(
        target_matrix=predicted_binary_matrix,
        dilation_distance_metres=DILATION_DISTANCE_METRES, verbose=False)

    _plot_fronts(
        actual_binary_matrix=actual_binary_matrix,
        predicted_binary_matrix=predicted_binary_matrix,
        annotation_string='(b)', output_file_name=WITH_DILATION_FILE_NAME)

    print 'Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME)
    imagemagick_utils.concatenate_images(
        input_file_names=[NO_DILATION_FILE_NAME, WITH_DILATION_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=1,
        num_panel_columns=2, output_size_pixels=OUTPUT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
