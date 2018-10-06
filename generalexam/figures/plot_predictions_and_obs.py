"""Plots predicted and observed fronts for one time step."""

import pickle
import argparse
import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import utils
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting

ZERO_CELSIUS_IN_KELVINS = 273.15
TIME_FORMAT = '%Y-%m-%d-%H'

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

PRESSURE_LEVEL_MB = 1000
WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]
NARR_FIELD_NAMES = WIND_FIELD_NAMES + [processed_narr_io.WET_BULB_THETA_NAME]

FRONT_LINE_WIDTH = 8
# BORDER_COLOUR = numpy.full(3, 152. / 255)
BORDER_COLOUR = numpy.full(3, 0.)
WARM_FRONT_COLOUR = numpy.array([217., 95., 2.]) / 255
COLD_FRONT_COLOUR = numpy.array([117., 112., 179.]) / 255

PROBABILISTIC_OPACITY = 0.5
LENGTH_FRACTION_FOR_PROB_COLOUR_BAR = 0.5
LENGTH_FRACTION_FOR_THETA_COLOUR_BAR = 0.75

THERMAL_COLOUR_MAP_OBJECT = pyplot.cm.YlGn
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

WIND_COLOUR_MAP_OBJECT = pyplot.cm.binary
WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.
PLOT_EVERY_KTH_WIND_BARB = 8

OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

TOP_FRONT_DIR_NAME = '/localdata/ryan.lagerquist/general_exam/fronts/polylines'
TOP_NARR_DIRECTORY_NAME = (
    '/localdata/ryan.lagerquist/general_exam/narr_data/processed')
TOP_PREDICTION_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/simple_cnn_experiment_1000mb/'
    'architecture-id=5_l2-weight=-1.000000000/gridded_predictions')

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'predictions_and_obs')
CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'predictions_and_obs/predictions_and_obs.jpg')

VALID_TIMES_ARG_NAME = 'valid_time_strings'
VALID_TIMES_HELP_STRING = (
    'List of two valid times (format "yyyy-mm-dd-HH").  Predictions and '
    'observations will be plotted for each valid time, with the top (bottom) '
    'row of the figure showing the first (second) valid time.')

DEFAULT_VALID_TIME_STRINGS = ['2017-01-01-06', '2017-01-01-09']

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_VALID_TIME_STRINGS, help=VALID_TIMES_HELP_STRING)


def _plot_observations_one_time(
        valid_time_string, annotation_string, output_file_name):
    """Plots observations (NARR predictors and WPC fronts) for one valid time.

    :param valid_time_string: Valid time (format "yyyy-mm-dd-HH").
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

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT)
    front_file_name = fronts_io.find_file_for_one_time(
        top_directory_name=TOP_FRONT_DIR_NAME,
        file_type=fronts_io.POLYLINE_FILE_TYPE,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(front_file_name)
    front_line_table = fronts_io.read_polylines_from_file(front_file_name)

    num_narr_fields = len(NARR_FIELD_NAMES)
    narr_matrix_by_field = [numpy.array([])] * num_narr_fields

    for j in range(num_narr_fields):
        this_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=TOP_NARR_DIRECTORY_NAME,
            field_name=NARR_FIELD_NAMES[j], pressure_level_mb=PRESSURE_LEVEL_MB,
            valid_time_unix_sec=valid_time_unix_sec)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        narr_matrix_by_field[j] = processed_narr_io.read_fields_from_file(
            this_file_name)[0][0, ...]

        narr_matrix_by_field[j] = utils.fill_nans(narr_matrix_by_field[j])
        narr_matrix_by_field[j] = narr_matrix_by_field[j][
            narr_row_limits[0]:(narr_row_limits[1] + 1),
            narr_column_limits[0]:(narr_column_limits[1] + 1)
        ]

        if NARR_FIELD_NAMES[j] == processed_narr_io.WET_BULB_THETA_NAME:
            narr_matrix_by_field[j] = (
                narr_matrix_by_field[j] - ZERO_CELSIUS_IN_KELVINS
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

    for j in range(num_narr_fields):
        if NARR_FIELD_NAMES[j] in WIND_FIELD_NAMES:
            continue

        min_colour_value = numpy.percentile(
            narr_matrix_by_field[j], MIN_COLOUR_PERCENTILE)
        max_colour_value = numpy.percentile(
            narr_matrix_by_field[j], MAX_COLOUR_PERCENTILE)

        nwp_plotting.plot_subgrid(
            field_matrix=narr_matrix_by_field[j],
            model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
            basemap_object=basemap_object, colour_map=THERMAL_COLOUR_MAP_OBJECT,
            min_value_in_colour_map=min_colour_value,
            max_value_in_colour_map=max_colour_value,
            first_row_in_full_grid=narr_row_limits[0],
            first_column_in_full_grid=narr_column_limits[0])

        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=axes_object,
            values_to_colour=narr_matrix_by_field[j],
            colour_map=THERMAL_COLOUR_MAP_OBJECT, colour_min=min_colour_value,
            colour_max=max_colour_value, orientation='vertical',
            extend_min=True, extend_max=True,
            fraction_of_axis_length=LENGTH_FRACTION_FOR_THETA_COLOUR_BAR)

    u_wind_index = NARR_FIELD_NAMES.index(
        processed_narr_io.U_WIND_EARTH_RELATIVE_NAME)
    v_wind_index = NARR_FIELD_NAMES.index(
        processed_narr_io.V_WIND_EARTH_RELATIVE_NAME)

    nwp_plotting.plot_wind_barbs_on_subgrid(
        u_wind_matrix_m_s01=narr_matrix_by_field[u_wind_index],
        v_wind_matrix_m_s01=narr_matrix_by_field[v_wind_index],
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object,
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0],
        plot_every_k_rows=PLOT_EVERY_KTH_WIND_BARB,
        plot_every_k_columns=PLOT_EVERY_KTH_WIND_BARB,
        barb_length=WIND_BARB_LENGTH, empty_barb_radius=EMPTY_WIND_BARB_RADIUS,
        colour_map=WIND_COLOUR_MAP_OBJECT,
        colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
        colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT)

    num_fronts = len(front_line_table.index)
    for i in range(num_fronts):
        this_front_type_string = front_line_table[
            front_utils.FRONT_TYPE_COLUMN].values[i]
        if this_front_type_string == front_utils.WARM_FRONT_STRING_ID:
            this_colour = WARM_FRONT_COLOUR
        else:
            this_colour = COLD_FRONT_COLOUR

        front_plotting.plot_polyline(
            latitudes_deg=front_line_table[
                front_utils.LATITUDES_COLUMN].values[i],
            longitudes_deg=front_line_table[
                front_utils.LONGITUDES_COLUMN].values[i],
            basemap_object=basemap_object, axes_object=axes_object,
            front_type=front_line_table[
                front_utils.FRONT_TYPE_COLUMN].values[i],
            line_width=FRONT_LINE_WIDTH, line_colour=this_colour)

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _plot_predictions(
        class_probability_matrix, annotation_string, output_file_name):
    """Plots predicted front probabilities.

    :param class_probability_matrix: See doc for
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

    (this_colour_map_object, this_colour_norm_object
    ) = prediction_plotting.get_warm_front_colour_map()[:2]
    plotting_utils.add_colour_bar(
        axes_object_or_list=axes_object, colour_map=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        values_to_colour=class_probability_matrix[
            ..., front_utils.WARM_FRONT_INTEGER_ID],
        orientation='vertical', extend_min=True, extend_max=False,
        fraction_of_axis_length=LENGTH_FRACTION_FOR_PROB_COLOUR_BAR)

    (this_colour_map_object, this_colour_norm_object
    ) = prediction_plotting.get_cold_front_colour_map()[:2]
    plotting_utils.add_colour_bar(
        axes_object_or_list=axes_object, colour_map=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        values_to_colour=class_probability_matrix[
            ..., front_utils.COLD_FRONT_INTEGER_ID],
        orientation='vertical', extend_min=True, extend_max=False,
        fraction_of_axis_length=LENGTH_FRACTION_FOR_PROB_COLOUR_BAR)

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(valid_time_strings):
    """Plots predicted and observed fronts for one time step.

    This is effectively the main method.

    :param valid_time_strings: See documentation at top of file.
    """

    num_times = 2
    error_checking.assert_is_numpy_array(
        numpy.array(valid_time_strings),
        exact_dimensions=numpy.array([num_times]))

    valid_times_unix_sec = numpy.array(
        [time_conversion.string_to_unix_sec(s, TIME_FORMAT)
         for s in valid_time_strings],
        dtype=int)

    prediction_annotation_strings = ['(a)', '(c)']
    observation_annotation_strings = ['(b)', '(d)']
    figure_file_names = []

    for i in range(num_times):
        this_prediction_file_name = ml_utils.find_gridded_prediction_file(
            directory_name=TOP_PREDICTION_DIR_NAME,
            first_target_time_unix_sec=valid_times_unix_sec[i],
            last_target_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_prediction_file_name)
        this_prediction_dict = ml_utils.read_gridded_predictions(
            this_prediction_file_name)
        this_probability_matrix = this_prediction_dict[
            ml_utils.PROBABILITY_MATRIX_KEY]
        this_probability_matrix[numpy.isnan(this_probability_matrix)] = 0.

        this_figure_file_name = '{0:s}/predictions_{1:s}.jpg'.format(
            OUTPUT_DIR_NAME, valid_time_strings[i])
        figure_file_names.append(this_figure_file_name)

        _plot_predictions(
            class_probability_matrix=this_probability_matrix,
            annotation_string=prediction_annotation_strings[i],
            output_file_name=this_figure_file_name)

        this_figure_file_name = '{0:s}/observations_{1:s}.jpg'.format(
            OUTPUT_DIR_NAME, valid_time_strings[i])
        figure_file_names.append(this_figure_file_name)

        _plot_observations_one_time(
            valid_time_string=valid_time_strings[i],
            annotation_string=observation_annotation_strings[i],
            output_file_name=this_figure_file_name)
        print '\n'

    print 'Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME)
    imagemagick_utils.concatenate_images(
        input_file_names=figure_file_names, output_file_name=CONCAT_FILE_NAME,
        num_panel_rows=2, num_panel_columns=2,
        output_size_pixels=OUTPUT_SIZE_PIXELS)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    _run(getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME))
