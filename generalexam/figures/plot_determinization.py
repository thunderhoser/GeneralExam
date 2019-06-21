"""Plots determinization of front probabilities."""

import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.evaluation import object_based_evaluation as object_eval
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

BINARIZATION_THRESHOLD = 0.411
PREDICTION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/paper_experiment_1000mb/'
    'quick_training/u-wind-grid-relative-m-s01_v-wind-grid-relative-m-s01_'
    'temperature-kelvins_specific-humidity-kg-kg01_init-num-filters=32_'
    'half-image-size-px=16_num-conv-layer-sets=3_dropout=0.50/gridded_'
    'predictions/testing/gridded_predictions_2017012500-2017012500.p')

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

BORDER_COLOUR = numpy.full(3, 0.)
DETERMINISTIC_OPACITY = 1.
PROBABILISTIC_OPACITY = 0.5
AXIS_LENGTH_FRACTION_FOR_CBAR = 0.8

FIGURE_RESOLUTION_DPI = 600
CONCAT_SIZE_PIXELS = int(1e7)

BEFORE_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'determinization/before_determinization.jpg')
AFTER_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'determinization/after_determinization.jpg')
CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'determinization/determinization.jpg')


def _plot_predictions(
        output_file_name, title_string, annotation_string,
        class_probability_matrix=None, predicted_label_matrix=None):
    """Plots predicted front locations or probabilities.

    :param output_file_name: Path to output file (figure will be saved here).
    :param class_probability_matrix: See doc for
        `machine_learning_utils.write_gridded_predictions`.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param predicted_label_matrix: See doc for `target_matrix` in
        `machine_learning_utils.write_gridded_predictions`.
    """

    narr_row_limits, narr_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

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

    if class_probability_matrix is None:
        this_matrix = predicted_label_matrix[
            0, narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1)
        ]

        front_plotting.plot_narr_grid(
            frontal_grid_matrix=this_matrix, axes_object=axes_object,
            basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0],
            opacity=DETERMINISTIC_OPACITY)
    else:
        this_matrix = class_probability_matrix[
            0, narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.WARM_FRONT_INTEGER_ID
        ]

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_matrix,
            front_string_id=front_utils.WARM_FRONT_STRING_ID,
            axes_object=axes_object, basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0],
            opacity=PROBABILISTIC_OPACITY)

        this_matrix = class_probability_matrix[
            0, narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1),
            front_utils.COLD_FRONT_INTEGER_ID
        ]

        prediction_plotting.plot_narr_grid(
            probability_matrix=this_matrix,
            front_string_id=front_utils.COLD_FRONT_STRING_ID,
            axes_object=axes_object, basemap_object=basemap_object,
            first_row_in_narr_grid=narr_row_limits[0],
            first_column_in_narr_grid=narr_column_limits[0],
            opacity=PROBABILISTIC_OPACITY)

        this_colour_map_object, this_colour_norm_object = (
            prediction_plotting.get_warm_front_colour_map()[:2]
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=axes_object, colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            values_to_colour=class_probability_matrix[
                ..., front_utils.WARM_FRONT_INTEGER_ID],
            orientation='vertical', extend_min=True, extend_max=False,
            fraction_of_axis_length=AXIS_LENGTH_FRACTION_FOR_CBAR)

        this_colour_map_object, this_colour_norm_object = (
            prediction_plotting.get_cold_front_colour_map()[:2]
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=axes_object, colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            values_to_colour=class_probability_matrix[
                ..., front_utils.COLD_FRONT_INTEGER_ID],
            orientation='horizontal', extend_min=True, extend_max=False)

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
    """Plots determinization of front probabilities.

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

    _plot_predictions(
        output_file_name=BEFORE_FILE_NAME,
        title_string='Probabilities', annotation_string='(a)',
        class_probability_matrix=class_probability_matrix)

    predicted_label_matrix = object_eval.determinize_probabilities(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=BINARIZATION_THRESHOLD)

    _plot_predictions(
        output_file_name=AFTER_FILE_NAME,
        title_string='Deterministic predictions', annotation_string='(b)',
        predicted_label_matrix=predicted_label_matrix)

    print('Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME))
    
    imagemagick_utils.concatenate_images(
        input_file_names=[BEFORE_FILE_NAME, AFTER_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=2,
        num_panel_columns=1)
    
    imagemagick_utils.resize_image(
        input_file_name=CONCAT_FILE_NAME, output_file_name=CONCAT_FILE_NAME,
        output_size_pixels=CONCAT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
