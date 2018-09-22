"""Plots conversion of gridded probabilities into objects."""

import pickle
import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.evaluation import object_based_evaluation as object_eval
from generalexam.plotting import front_plotting

BINARIZATION_THRESHOLD = 0.3
MIN_REGION_AREA_METRES2 = 5e11
MIN_ENDPOINT_LENGTH_METRES = 5e5

PREDICTION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/traditional_cnn_experiment05/'
    'no-front-fraction=0.700_num-examples-per-batch=1024_weight-loss-function=1'
    '/object_based_eval_no-isotonic/class_probability_matrix_2016-01-01-15.p')
VALID_TIME_STRING = '2016-01-01-15'
TIME_FORMAT = '%Y-%m-%d-%H'

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 212.5
MAX_LATITUDE_DEG = 75.
MAX_LONGITUDE_DEG = 360.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

BORDER_COLOUR = numpy.full(3, 0.)
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

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
        predicted_label_matrix, annotation_string, output_file_name):
    """Plots predicted front locations.

    :param predicted_label_matrix: See doc for `target_matrix` in
        `machine_learning_utils.write_gridded_predictions`.
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

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run():
    """Plots conversion of gridded probabilities into objects.

    This is effectively the main method.
    """

    # TODO(thunderhoser): Replace with new prediction file.
    print 'Reading data from: "{0:s}"...'.format(PREDICTION_FILE_NAME)
    pickle_file_handle = open(PREDICTION_FILE_NAME, 'rb')
    class_probability_matrix = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    predicted_label_matrix = object_eval.determinize_probabilities(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=BINARIZATION_THRESHOLD)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix, annotation_string='(a)',
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
        predicted_label_matrix=predicted_label_matrix, annotation_string='(b)',
        output_file_name=LARGE_REGIONS_FILE_NAME)

    predicted_region_table = object_eval.skeletonize_frontal_regions(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)
    predicted_label_matrix = object_eval.regions_to_images(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix, annotation_string='(c)',
        output_file_name=ALL_SKELETONS_FILE_NAME)

    predicted_region_table = object_eval.find_main_skeletons(
        predicted_region_table=predicted_region_table,
        class_probability_matrix=class_probability_matrix,
        image_times_unix_sec=valid_times_unix_sec,
        x_grid_spacing_metres=grid_spacing_metres,
        y_grid_spacing_metres=grid_spacing_metres,
        min_endpoint_length_metres=MIN_ENDPOINT_LENGTH_METRES)
    predicted_label_matrix = object_eval.regions_to_images(
        predicted_region_table=predicted_region_table,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    _plot_predictions(
        predicted_label_matrix=predicted_label_matrix, annotation_string='(d)',
        output_file_name=MAIN_SKELETONS_FILE_NAME)

    print 'Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME)
    imagemagick_utils.concatenate_images(
        input_file_names=[ALL_REGIONS_FILE_NAME, LARGE_REGIONS_FILE_NAME,
                          ALL_SKELETONS_FILE_NAME, MAIN_SKELETONS_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=2,
        num_panel_columns=2, output_size_pixels=OUTPUT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
