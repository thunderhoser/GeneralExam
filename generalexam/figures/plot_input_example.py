"""Plots one input example, which includes the following:

- WPC front
- nearby NARR wind field
- nearby NARR wet-bulb potential temperature
"""

import numpy
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

TIME_FORMAT = '%Y-%m-%d-%H'
ZERO_CELSIUS_IN_KELVINS = 273.15

PRESSURE_LEVEL_MB = 1000
WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]
NARR_FIELD_NAMES = WIND_FIELD_NAMES + [processed_narr_io.WET_BULB_THETA_NAME]

APPROX_FRONT_LATITUDE_DEG = 55.
APPROX_FRONT_LONGITUDE_DEG = 265.
PARALLEL_SPACING_DEG = 2.
MERIDIAN_SPACING_DEG = 5.
NUM_ROWS_IN_HALF_GRID = 16
NUM_COLUMNS_IN_HALF_GRID = 16

FRONT_LINE_WIDTH = 5
# BORDER_COLOUR = numpy.full(3, 152. / 255)
BORDER_COLOUR = numpy.full(3, 0.)
WARM_FRONT_COLOUR = numpy.array([217., 95., 2.]) / 255
COLD_FRONT_COLOUR = numpy.array([117., 112., 179.]) / 255

THERMAL_COLOUR_MAP_OBJECT = pyplot.cm.YlGn
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

WIND_COLOUR_MAP_OBJECT = pyplot.cm.binary
WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.
PLOT_EVERY_KTH_WIND_BARB = 1

VALID_TIME_STRING = '2017-01-12-09'
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'input_example/input_example.jpg')

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


def _find_nearest_front(
        front_line_table, query_latitude_deg, query_longitude_deg):
    """Finds nearest front to the query point.

    :param front_line_table: See doc for `fronts_io.write_polylines_to_file`.
    :param query_latitude_deg: Latitude (deg N) of query point.
    :param query_longitude_deg: Longitude (deg E) of query point.
    :return: front_index: Index of nearest front.  This is a row index into
        `front_line_table`.
    :return: front_centroid_latitude_deg: Latitude (deg N) at centroid of
        nearest front.
    :return: front_centroid_longitude_deg: Longitude (deg E) at centroid of
        nearest front.
    """

    num_fronts = len(front_line_table.index)
    centroid_latitudes_deg = numpy.full(num_fronts, numpy.nan)
    centroid_longitudes_deg = numpy.full(num_fronts, numpy.nan)

    for i in range(num_fronts):
        centroid_latitudes_deg[i] = numpy.mean(
            front_line_table[front_utils.LATITUDES_COLUMN].values[i])
        centroid_longitudes_deg[i] = numpy.mean(
            front_line_table[front_utils.LONGITUDES_COLUMN].values[i])

    latitude_diffs_deg = numpy.absolute(
        centroid_latitudes_deg - query_latitude_deg)
    longitude_diffs_deg = numpy.absolute(
        centroid_longitudes_deg - query_longitude_deg)
    front_index = numpy.argmin(latitude_diffs_deg + longitude_diffs_deg)

    return (front_index, centroid_latitudes_deg[front_index],
            centroid_longitudes_deg[front_index])


def _run():
    """Plots input example.

    This is effectively the main method.

    :return: figure_file_name: Path to output file (where the figure was saved).
    """

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        VALID_TIME_STRING, TIME_FORMAT)
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

        if NARR_FIELD_NAMES[j] == processed_narr_io.WET_BULB_THETA_NAME:
            narr_matrix_by_field[j] = (
                narr_matrix_by_field[j] - ZERO_CELSIUS_IN_KELVINS
            )

    (_, front_centroid_latitude_deg, front_centroid_longitude_deg
    ) = _find_nearest_front(
        front_line_table=front_line_table,
        query_latitude_deg=APPROX_FRONT_LATITUDE_DEG,
        query_longitude_deg=APPROX_FRONT_LONGITUDE_DEG)

    projection_object = nwp_model_utils.init_model_projection(
        nwp_model_utils.NARR_MODEL_NAME)
    these_x_metres, these_y_metres = nwp_model_utils.project_latlng_to_xy(
        latitudes_deg=numpy.array([front_centroid_latitude_deg]),
        longitudes_deg=numpy.array([front_centroid_longitude_deg]),
        projection_object=projection_object,
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    front_centroid_x_metres = these_x_metres[0]
    front_centroid_y_metres = these_y_metres[0]

    grid_spacing_metres, _ = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    center_narr_row_index = int(numpy.round(
        front_centroid_y_metres / grid_spacing_metres))
    center_narr_column_index = int(numpy.round(
        front_centroid_x_metres / grid_spacing_metres))

    first_narr_row_index = center_narr_row_index - NUM_ROWS_IN_HALF_GRID
    last_narr_row_index = center_narr_row_index + NUM_ROWS_IN_HALF_GRID
    first_narr_column_index = (
        center_narr_column_index - NUM_COLUMNS_IN_HALF_GRID)
    last_narr_column_index = center_narr_column_index + NUM_COLUMNS_IN_HALF_GRID

    for j in range(num_narr_fields):
        narr_matrix_by_field[j] = narr_matrix_by_field[j][
            first_narr_row_index:(last_narr_row_index + 1),
            first_narr_column_index:(last_narr_column_index + 1)
        ]

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        first_row_in_full_grid=first_narr_row_index,
        last_row_in_full_grid=last_narr_row_index,
        first_column_in_full_grid=first_narr_column_index,
        last_column_in_full_grid=last_narr_column_index, resolution_string='i')

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
            first_row_in_full_grid=first_narr_row_index,
            first_column_in_full_grid=first_narr_column_index)

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
        first_row_in_full_grid=first_narr_row_index,
        first_column_in_full_grid=first_narr_column_index,
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

    print 'Saving figure to: "{0:s}"...'.format(OUTPUT_FILE_NAME)
    file_system_utils.mkdir_recursive_if_necessary(file_name=OUTPUT_FILE_NAME)
    pyplot.savefig(OUTPUT_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=OUTPUT_FILE_NAME,
                                      output_file_name=OUTPUT_FILE_NAME)


if __name__ == '__main__':
    _run()
