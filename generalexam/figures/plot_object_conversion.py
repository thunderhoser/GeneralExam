"""Plots conversion of gridded probabilities into objects."""

import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.evaluation import object_based_evaluation as object_eval
from generalexam.plotting import front_plotting

BINARIZATION_THRESHOLD = 0.411
MIN_REGION_AREA_METRES2 = 3e11
MIN_ENDPOINT_LENGTH_METRES = 4e5

PREDICTION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/paper_experiment_1000mb/'
    'quick_training/u-wind-grid-relative-m-s01_v-wind-grid-relative-m-s01_'
    'temperature-kelvins_specific-humidity-kg-kg01_init-num-filters=32_'
    'half-image-size-px=16_num-conv-layer-sets=3_dropout=0.50/gridded_'
    'predictions/testing/gridded_predictions_2017012500-2017012500.p')

VALID_TIME_STRING = '2017012500'
TIME_FORMAT = '%Y%m%d%H'

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 600
FIGURE_SIZE_PIXELS = int(1e7)

ALL_REGIONS_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'object_conversion/all_regions.jpg')
LARGE_REGIONS_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'object_conversion/large_regions.jpg')
ALL_SKELETONS_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'object_conversion/all_skeletons.jpg')
MAIN_SKELETONS_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'object_conversion/main_skeletons.jpg')
CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'object_conversion/object_conversion.jpg')


def _plot_predictions(
        predicted_label_matrix, title_string, annotation_string,
        output_file_name):
    """Plots predicted front locations.

    :param predicted_label_matrix: See doc for `target_matrix` in
        `machine_learning_utils.write_gridded_predictions`.
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

    this_matrix = predicted_label_matrix[
        0, narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]
    front_plotting.plot_narr_grid(
        frontal_grid_matrix=this_matrix, axes_object=axes_object,
        basemap_object=basemap_object,
        first_row_in_narr_grid=narr_row_limits[0],
        first_column_in_narr_grid=narr_column_limits[0], opacity=1.)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run():
    """Plots conversion of gridded probabilities into objects.

    This is effectively the main method.
    """

    prediction_dict = ml_utils.read_gridded_predictions(PREDICTION_FILE_NAME)
    class_probability_matrix = prediction_dict[ml_utils.PROBABILITY_MATRIX_KEY]

    for this_id in front_utils.VALID_INTEGER_IDS:
        if this_id == front_utils.NO_FRONT_INTEGER_ID:
            class_probability_matrix[
                ..., this_id
            ][numpy.isnan(class_probability_matrix[..., this_id])] = 1.
        else:
            class_probability_matrix[
                ..., this_id
            ][numpy.isnan(class_probability_matrix[..., this_id])] = 0.

    predicted_label_matrix = object_eval.determinize_probabilities(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=BINARIZATION_THRESHOLD)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix,
        title_string='All frontal regions', annotation_string='(a)',
        output_file_name=ALL_REGIONS_FILE_NAME)

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        VALID_TIME_STRING, TIME_FORMAT)
    valid_times_unix_sec = numpy.array([valid_time_unix_sec], dtype=int)
    predicted_region_table = object_eval.images_to_regions(
        predicted_label_matrix=predicted_label_matrix,
        image_times_unix_sec=valid_times_unix_sec)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    grid_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME
    )[0]

    predicted_region_table = object_eval.discard_regions_with_small_area(
        predicted_region_table=predicted_region_table,
        x_grid_spacing_metres=grid_spacing_metres,
        y_grid_spacing_metres=grid_spacing_metres,
        min_area_metres2=MIN_REGION_AREA_METRES2)
    predicted_label_matrix = object_eval.regions_to_images(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix,
        title_string='Large frontal regions', annotation_string='(b)',
        output_file_name=LARGE_REGIONS_FILE_NAME)

    predicted_region_table = object_eval.skeletonize_frontal_regions(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)
    predicted_label_matrix = object_eval.regions_to_images(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix,
        title_string='All skeleton lines', annotation_string='(c)',
        output_file_name=ALL_SKELETONS_FILE_NAME)

    predicted_region_table = object_eval.find_main_skeletons(
        predicted_region_table=predicted_region_table,
        image_times_unix_sec=valid_times_unix_sec,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        x_grid_spacing_metres=grid_spacing_metres,
        y_grid_spacing_metres=grid_spacing_metres,
        min_endpoint_length_metres=MIN_ENDPOINT_LENGTH_METRES)
    predicted_label_matrix = object_eval.regions_to_images(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix,
        title_string='Main skeleton lines', annotation_string='(d)',
        output_file_name=MAIN_SKELETONS_FILE_NAME)

    print 'Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME)
    imagemagick_utils.concatenate_images(
        input_file_names=[ALL_REGIONS_FILE_NAME, LARGE_REGIONS_FILE_NAME,
                          ALL_SKELETONS_FILE_NAME, MAIN_SKELETONS_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=2,
        num_panel_columns=2, output_size_pixels=FIGURE_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
