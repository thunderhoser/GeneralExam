"""Plots NFA (numerical frontal analysis) procedure."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import nfa
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import utils as general_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting

ZERO_CELSIUS_IN_KELVINS = 273.15
TFP_MULTIPLIER = 1e10
LOCATING_VAR_MULTIPLIER = 1e9

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

BORDER_COLOUR = numpy.full(3, 0.)

THERMAL_COLOUR_MAP_OBJECT = pyplot.cm.YlOrRd
TFP_COLOUR_MAP_OBJECT = pyplot.cm.PRGn
LOCATING_VAR_COLOUR_MAP_OBJECT = pyplot.cm.RdBu
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.
COLOUR_BAR_LENGTH_FRACTION = 0.8

WIND_COLOUR = numpy.full(3, 152. / 255)
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.

WIND_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap([WIND_COLOUR])
WIND_COLOUR_MAP_OBJECT.set_under(WIND_COLOUR)
WIND_COLOUR_MAP_OBJECT.set_over(WIND_COLOUR)

WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
PLOT_EVERY_KTH_WIND_BARB = 8

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
CONCAT_SIZE_PIXELS = int(1e7)

INPUT_TIME_FORMAT = '%Y%m%d%H'
NARR_PREDICTOR_NAMES = [
    processed_narr_io.WET_BULB_THETA_NAME,
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME
]

VALID_TIME_ARG_NAME = 'valid_time_string'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_pixels'
FRONT_PERCENTILE_ARG_NAME = 'front_percentile'
NUM_CLOSING_ITERS_ARG_NAME = 'num_closing_iters'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
NARR_MASK_FILE_ARG_NAME = 'input_narr_mask_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

VALID_TIME_HELP_STRING = 'Valid time (format "yyyymmddHH").'

SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (standard deviation of Gaussian kernel).  Will be applied'
    ' to all predictors (listed below).\n{0:s}'
).format(str(NARR_PREDICTOR_NAMES))

FRONT_PERCENTILE_HELP_STRING = (
    'Used to locate warm and cold fronts.  See `nfa.get_front_types` for '
    'details.')

NUM_CLOSING_ITERS_HELP_STRING = (
    'Number of binary-closing iterations.  Will be applied to both warm-front '
    'and cold-front labels independently.  More iterations lead to larger '
    'frontal regions.')

PRESSURE_LEVEL_HELP_STRING = (
    'Predictors (listed below) will be taken from this pressure level '
    '(millibars).\n{0:s}'
).format(str(NARR_PREDICTOR_NAMES))

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level NARR directory (predictors will be read from here).  '
    'Files therein will be found by `processed_narr_io.find_file_for_one_time` '
    'and read by `processed_narr_io.read_fields_from_file`.')

NARR_MASK_FILE_HELP_STRING = (
    'Pickle file with NARR mask (will be read by `machine_learning_utils.'
    'read_narr_mask`).  Predictions will not be made for masked grid cells.  If'
    ' you do not want masking, make this empty ("").')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

DEFAULT_SMOOTHING_RADIUS_PIXELS = 1.
DEFAULT_FRONT_PERCENTILE = 96.
DEFAULT_NUM_CLOSING_ITERS = 2
DEFAULT_PRESSURE_LEVEL_MB = 900
TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_NARR_MASK_FILE_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=DEFAULT_SMOOTHING_RADIUS_PIXELS, help=SMOOTHING_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_PERCENTILE_ARG_NAME, type=float, required=False,
    default=DEFAULT_FRONT_PERCENTILE, help=FRONT_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CLOSING_ITERS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_CLOSING_ITERS, help=NUM_CLOSING_ITERS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_MASK_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_MASK_FILE_NAME, help=NARR_MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _init_basemap(border_colour):
    """Initializes basemap.

    :param border_colour: Colour (in any format accepted by matplotlib) of
        political borders.
    :return: narr_row_limits: length-2 numpy array of (min, max) NARR rows to
        plot.
    :return: narr_column_limits: length-2 numpy array of (min, max) NARR columns
        to plot.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
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
        line_colour=border_colour)
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=border_colour)
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG)

    return narr_row_limits, narr_column_limits, axes_object, basemap_object


def _plot_narr_fields(
        wet_bulb_theta_matrix_kelvins, u_wind_matrix_m_s01, v_wind_matrix_m_s01,
        title_string, annotation_string, output_file_name):
    """Plots NARR fields.

    M = number of rows in grid
    N = number of columns in grid

    :param wet_bulb_theta_matrix_kelvins: M-by-N numpy array of wet-bulb
        potential temperatures.
    :param u_wind_matrix_m_s01: M-by-N numpy array of u-wind components (metres
        per second).
    :param v_wind_matrix_m_s01: Same but for v-wind.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (narr_row_limits, narr_column_limits, axes_object, basemap_object
    ) = _init_basemap(BORDER_COLOUR)

    wet_bulb_theta_matrix_to_plot = wet_bulb_theta_matrix_kelvins[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ] - ZERO_CELSIUS_IN_KELVINS
    u_wind_matrix_to_plot = u_wind_matrix_m_s01[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]
    v_wind_matrix_to_plot = v_wind_matrix_m_s01[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]

    nwp_plotting.plot_subgrid(
        field_matrix=wet_bulb_theta_matrix_to_plot,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object, colour_map=THERMAL_COLOUR_MAP_OBJECT,
        min_value_in_colour_map=numpy.nanpercentile(
            wet_bulb_theta_matrix_to_plot, MIN_COLOUR_PERCENTILE),
        max_value_in_colour_map=numpy.nanpercentile(
            wet_bulb_theta_matrix_to_plot, MAX_COLOUR_PERCENTILE),
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0])

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object,
        values_to_colour=wet_bulb_theta_matrix_to_plot,
        colour_map=THERMAL_COLOUR_MAP_OBJECT,
        colour_min=numpy.nanpercentile(
            wet_bulb_theta_matrix_to_plot, MIN_COLOUR_PERCENTILE),
        colour_max=numpy.nanpercentile(
            wet_bulb_theta_matrix_to_plot, MAX_COLOUR_PERCENTILE),
        orientation='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=COLOUR_BAR_LENGTH_FRACTION)

    nwp_plotting.plot_wind_barbs_on_subgrid(
        u_wind_matrix_m_s01=u_wind_matrix_to_plot,
        v_wind_matrix_m_s01=v_wind_matrix_to_plot,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object,
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0],
        plot_every_k_rows=PLOT_EVERY_KTH_WIND_BARB,
        plot_every_k_columns=PLOT_EVERY_KTH_WIND_BARB,
        barb_length=WIND_BARB_LENGTH, empty_barb_radius=EMPTY_WIND_BARB_RADIUS,
        fill_empty_barb=False, colour_map=WIND_COLOUR_MAP_OBJECT,
        colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
        colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _plot_tfp(tfp_matrix_kelvins_m02, title_string, annotation_string,
              output_file_name):
    """Plots TFP (thermal front parameter).

    M = number of rows in grid
    N = number of columns in grid

    :param tfp_matrix_kelvins_m02: M-by-N numpy array of TFP values.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (narr_row_limits, narr_column_limits, axes_object, basemap_object
    ) = _init_basemap(BORDER_COLOUR)

    matrix_to_plot = tfp_matrix_kelvins_m02[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ] * TFP_MULTIPLIER

    max_colour_value = numpy.nanpercentile(
        numpy.absolute(matrix_to_plot), MAX_COLOUR_PERCENTILE)
    min_colour_value = -1 * max_colour_value

    nwp_plotting.plot_subgrid(
        field_matrix=matrix_to_plot,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object, colour_map=TFP_COLOUR_MAP_OBJECT,
        min_value_in_colour_map=min_colour_value,
        max_value_in_colour_map=max_colour_value,
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0])

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object,
        values_to_colour=matrix_to_plot,
        colour_map=TFP_COLOUR_MAP_OBJECT, colour_min=min_colour_value,
        colour_max=max_colour_value, orientation='vertical', extend_min=True,
        extend_max=True, fraction_of_axis_length=COLOUR_BAR_LENGTH_FRACTION)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _plot_locating_variable(
        locating_var_matrix_m01_s01, title_string, annotation_string,
        output_file_name):
    """Plots locating variable.

    M = number of rows in grid
    N = number of columns in grid

    :param locating_var_matrix_m01_s01: M-by-N numpy array with values of
        locating variable.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
    figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (narr_row_limits, narr_column_limits, axes_object, basemap_object
    ) = _init_basemap(BORDER_COLOUR)

    matrix_to_plot = locating_var_matrix_m01_s01[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ] * LOCATING_VAR_MULTIPLIER

    max_colour_value = numpy.nanpercentile(
        numpy.absolute(matrix_to_plot), MAX_COLOUR_PERCENTILE)
    min_colour_value = -1 * max_colour_value

    nwp_plotting.plot_subgrid(
        field_matrix=matrix_to_plot,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object,
        colour_map=LOCATING_VAR_COLOUR_MAP_OBJECT,
        min_value_in_colour_map=min_colour_value,
        max_value_in_colour_map=max_colour_value,
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0])

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object,
        values_to_colour=matrix_to_plot,
        colour_map=LOCATING_VAR_COLOUR_MAP_OBJECT, colour_min=min_colour_value,
        colour_max=max_colour_value, orientation='vertical', extend_min=True,
        extend_max=True, fraction_of_axis_length=COLOUR_BAR_LENGTH_FRACTION)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _plot_front_types(predicted_label_matrix, title_string, annotation_string,
                      output_file_name):
    """Plots front type at each grid cell.

    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix: M-by-N numpy array with predicted front type
        at each grid cell.  Each front type is from the list
        `front_utils.VALID_INTEGER_IDS`.
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
    figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (narr_row_limits, narr_column_limits, axes_object, basemap_object
    ) = _init_basemap(BORDER_COLOUR)

    matrix_to_plot = predicted_label_matrix[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]

    front_plotting.plot_narr_grid(
        frontal_grid_matrix=matrix_to_plot, axes_object=axes_object,
        basemap_object=basemap_object,
        first_row_in_narr_grid=narr_row_limits[0],
        first_column_in_narr_grid=narr_column_limits[0])

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(valid_time_string, smoothing_radius_pixels, front_percentile,
         num_closing_iters, pressure_level_mb, top_narr_directory_name,
         narr_mask_file_name, output_dir_name):
    """Plots NFA (numerical frontal analysis) procedure.

    This is effectively the main method.

    :param valid_time_string: See documentation at top of file.
    :param smoothing_radius_pixels: Same.
    :param front_percentile: Same.
    :param num_closing_iters: Same.
    :param pressure_level_mb: Same.
    :param top_narr_directory_name: Same.
    :param narr_mask_file_name: Same.
    :param output_dir_name: Same.
    """

    cutoff_radius_pixels = 4 * smoothing_radius_pixels
    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, INPUT_TIME_FORMAT)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if narr_mask_file_name == '':
        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
        narr_mask_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), 1, dtype=int)
    else:
        print 'Reading mask from: "{0:s}"...\n'.format(narr_mask_file_name)
        narr_mask_matrix = ml_utils.read_narr_mask(narr_mask_file_name)[0]

    wet_bulb_theta_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_narr_directory_name,
        field_name=processed_narr_io.WET_BULB_THETA_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(wet_bulb_theta_file_name)
    wet_bulb_theta_matrix_kelvins = processed_narr_io.read_fields_from_file(
        wet_bulb_theta_file_name)[0][0, ...]
    wet_bulb_theta_matrix_kelvins = general_utils.fill_nans(
        wet_bulb_theta_matrix_kelvins)

    u_wind_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_narr_directory_name,
        field_name=processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(u_wind_file_name)
    u_wind_matrix_m_s01 = processed_narr_io.read_fields_from_file(
        u_wind_file_name)[0][0, ...]
    u_wind_matrix_m_s01 = general_utils.fill_nans(u_wind_matrix_m_s01)

    v_wind_file_name = processed_narr_io.find_file_for_one_time(
        top_directory_name=top_narr_directory_name,
        field_name=processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=valid_time_unix_sec)

    print 'Reading data from: "{0:s}"...'.format(v_wind_file_name)
    v_wind_matrix_m_s01 = processed_narr_io.read_fields_from_file(
        v_wind_file_name)[0][0, ...]
    v_wind_matrix_m_s01 = general_utils.fill_nans(v_wind_matrix_m_s01)

    unsmoothed_narr_file_name = '{0:s}/unsmoothed_narr_fields.jpg'.format(
        output_dir_name)
    _plot_narr_fields(
        wet_bulb_theta_matrix_kelvins=wet_bulb_theta_matrix_kelvins,
        u_wind_matrix_m_s01=u_wind_matrix_m_s01,
        v_wind_matrix_m_s01=v_wind_matrix_m_s01,
        title_string='Predictors before smoothing', annotation_string='(a)',
        output_file_name=unsmoothed_narr_file_name)

    wet_bulb_theta_matrix_kelvins = nfa.gaussian_smooth_2d_field(
        field_matrix=wet_bulb_theta_matrix_kelvins,
        standard_deviation_pixels=smoothing_radius_pixels,
        cutoff_radius_pixels=cutoff_radius_pixels)

    u_wind_matrix_m_s01 = nfa.gaussian_smooth_2d_field(
        field_matrix=u_wind_matrix_m_s01,
        standard_deviation_pixels=smoothing_radius_pixels,
        cutoff_radius_pixels=cutoff_radius_pixels)

    v_wind_matrix_m_s01 = nfa.gaussian_smooth_2d_field(
        field_matrix=v_wind_matrix_m_s01,
        standard_deviation_pixels=smoothing_radius_pixels,
        cutoff_radius_pixels=cutoff_radius_pixels)

    smoothed_narr_file_name = '{0:s}/smoothed_narr_fields.jpg'.format(
        output_dir_name)
    _plot_narr_fields(
        wet_bulb_theta_matrix_kelvins=wet_bulb_theta_matrix_kelvins,
        u_wind_matrix_m_s01=u_wind_matrix_m_s01,
        v_wind_matrix_m_s01=v_wind_matrix_m_s01,
        title_string='Predictors after smoothing', annotation_string='(b)',
        output_file_name=smoothed_narr_file_name)

    x_spacing_metres, y_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
        thermal_field_matrix_kelvins=wet_bulb_theta_matrix_kelvins,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)
    tfp_matrix_kelvins_m02[narr_mask_matrix == 0] = 0.

    tfp_file_name = '{0:s}/tfp.jpg'.format(output_dir_name)
    tfp_title_string = (
        r'Thermal front parameter ($\times$ 10$^{-10}$ K m$^{-2}$)')
    _plot_tfp(tfp_matrix_kelvins_m02=tfp_matrix_kelvins_m02,
              title_string=tfp_title_string, annotation_string='(c)',
              output_file_name=tfp_file_name)

    proj_velocity_matrix_m_s01 = nfa.project_wind_to_thermal_gradient(
        u_matrix_grid_relative_m_s01=u_wind_matrix_m_s01,
        v_matrix_grid_relative_m_s01=v_wind_matrix_m_s01,
        thermal_field_matrix_kelvins=wet_bulb_theta_matrix_kelvins,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)

    locating_var_matrix_m01_s01 = nfa.get_locating_variable(
        tfp_matrix_kelvins_m02=tfp_matrix_kelvins_m02,
        projected_velocity_matrix_m_s01=proj_velocity_matrix_m_s01)

    locating_var_file_name = '{0:s}/locating_variable.jpg'.format(
        output_dir_name)
    locating_var_title_string = (
        r'Locating variable ($\times$ 10$^{-9}$ K m$^{-1}$ s$^{-1}$)')
    _plot_locating_variable(
        locating_var_matrix_m01_s01=locating_var_matrix_m01_s01,
        title_string=locating_var_title_string, annotation_string='(d)',
        output_file_name=locating_var_file_name)

    predicted_label_matrix = nfa.get_front_types(
        locating_var_matrix_m01_s01=locating_var_matrix_m01_s01,
        warm_front_percentile=front_percentile,
        cold_front_percentile=front_percentile)

    unclosed_fronts_file_name = '{0:s}/unclosed_fronts.jpg'.format(
        output_dir_name)
    _plot_front_types(
        predicted_label_matrix=predicted_label_matrix,
        title_string='Frontal regions before closing', annotation_string='(e)',
        output_file_name=unclosed_fronts_file_name)

    predicted_label_matrix = front_utils.close_frontal_image(
        ternary_image_matrix=predicted_label_matrix,
        num_iterations=num_closing_iters)

    closed_fronts_file_name = '{0:s}/closed_fronts.jpg'.format(output_dir_name)
    _plot_front_types(
        predicted_label_matrix=predicted_label_matrix,
        title_string='Frontal regions after closing', annotation_string='(f)',
        output_file_name=closed_fronts_file_name)

    concat_file_name = '{0:s}/nfa_procedure.jpg'.format(output_dir_name)
    print 'Concatenating figures to: "{0:s}"...'.format(concat_file_name)

    panel_file_names = [
        unsmoothed_narr_file_name, smoothed_narr_file_name, tfp_file_name,
        locating_var_file_name, unclosed_fronts_file_name,
        closed_fronts_file_name
    ]

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name, num_panel_rows=3,
        num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_SIZE_PIXELS)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        smoothing_radius_pixels=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
        front_percentile=getattr(INPUT_ARG_OBJECT, FRONT_PERCENTILE_ARG_NAME),
        num_closing_iters=getattr(INPUT_ARG_OBJECT, NUM_CLOSING_ITERS_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        narr_mask_file_name=getattr(INPUT_ARG_OBJECT, NARR_MASK_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
