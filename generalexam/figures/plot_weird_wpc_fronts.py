"""Plots weird WPC fronts along with the following NARR fields:

- wet-bulb potential temperature
- wind velocity
"""

import numpy
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_io import fronts_io
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import utils
from generalexam.plotting import front_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
ZERO_CELSIUS_IN_KELVINS = 273.15

PRESSURE_LEVEL_MB = 1000
WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]

NARR_FIELD_NAMES = WIND_FIELD_NAMES + [processed_narr_io.WET_BULB_THETA_NAME]

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

FRONT_LINE_WIDTH = 8
BORDER_COLOUR = numpy.full(3, 0.)
WARM_FRONT_COLOUR = numpy.array([30, 120, 180], dtype=float) / 255
COLD_FRONT_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255

THERMAL_COLOUR_MAP_OBJECT = pyplot.cm.YlOrRd
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

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

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

TOP_FRONT_DIR_NAME = '/localdata/ryan.lagerquist/general_exam/fronts/polylines'
TOP_NARR_DIRECTORY_NAME = (
    '/localdata/ryan.lagerquist/general_exam/narr_data/processed')
OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'weird_wpc_fronts')

VALID_TIME_STRINGS = [
    '2017-12-08-00', '2017-12-08-03', '2017-12-08-06',
    '2017-12-09-00', '2017-12-09-03', '2017-12-09-06'
]


def _plot_one_time(valid_time_string, title_string, annotation_string):
    """Plots WPC fronts and NARR fields at one time.

    :param valid_time_string: Valid time ("yyyy-mm-dd-HH").
    :param title_string: Title (will be placed above figure).
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :return: figure_file_name: Path to output file (where the figure was saved).
    """

    narr_row_limits, narr_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, DEFAULT_TIME_FORMAT)
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
            colour_max=max_colour_value, orientation='horizontal',
            extend_min=True, extend_max=True)

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
        fill_empty_barb=False, colour_map=WIND_COLOUR_MAP_OBJECT,
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

        front_plotting.plot_front_with_markers(
            line_latitudes_deg=front_line_table[
                front_utils.LATITUDES_COLUMN].values[i],
            line_longitudes_deg=front_line_table[
                front_utils.LONGITUDES_COLUMN].values[i],
            axes_object=axes_object, basemap_object=basemap_object,
            front_type_string=front_line_table[
                front_utils.FRONT_TYPE_COLUMN].values[i],
            marker_colour=this_colour)

    pyplot.title(title_string)
    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME)
    figure_file_name = '{0:s}/fronts_{1:s}.jpg'.format(
        OUTPUT_DIR_NAME, valid_time_string)

    print 'Saving figure to: "{0:s}"...'.format(figure_file_name)
    pyplot.savefig(figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=figure_file_name,
                                      output_file_name=figure_file_name)
    return figure_file_name


def _run():
    """Plots weird WPC fronts, along with theta_w and wind barbs from NARR.
    This is effectively the main method.
    """

    annotation_strings = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    panel_file_names = []

    for k in range(len(VALID_TIME_STRINGS)):
        this_time_unix_sec = time_conversion.string_to_unix_sec(
            VALID_TIME_STRINGS[k], DEFAULT_TIME_FORMAT)
        this_title_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, NICE_TIME_FORMAT)

        this_file_name = _plot_one_time(
            valid_time_string=VALID_TIME_STRINGS[k],
            title_string=this_title_string,
            annotation_string=annotation_strings[k])

        panel_file_names.append(this_file_name)
        print '\n'

    concat_file_name = '{0:s}/weird_fronts.jpg'.format(OUTPUT_DIR_NAME)

    print 'Concatenating figures to: "{0:s}"...'.format(concat_file_name)

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name, num_panel_rows=2,
        num_panel_columns=3)

    imagemagick_utils.trim_whitespace(input_file_name=concat_file_name,
                                      output_file_name=concat_file_name)

    imagemagick_utils.resize_image(input_file_name=concat_file_name,
                                   output_file_name=concat_file_name,
                                   output_size_pixels=CONCAT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
