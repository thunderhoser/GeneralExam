"""Plots predictors on full NARR grid."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.ge_io import wpc_bulletin_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import utils
from generalexam.plotting import front_plotting

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
NARR_TIME_INTERVAL_SEC = 10800

KG_TO_GRAMS = 1000.
ZERO_CELSIUS_IN_KELVINS = 273.15

VALID_THERMAL_FIELD_NAMES = [
    processed_narr_io.TEMPERATURE_NAME,
    processed_narr_io.SPECIFIC_HUMIDITY_NAME,
    processed_narr_io.WET_BULB_THETA_NAME
]

WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME
]

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

WIND_COLOUR = numpy.full(3, 152. / 255)
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.

WIND_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap([WIND_COLOUR])
WIND_COLOUR_MAP_OBJECT.set_under(WIND_COLOUR)
WIND_COLOUR_MAP_OBJECT.set_over(WIND_COLOUR)

WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
PLOT_EVERY_KTH_WIND_BARB = 8

PRESSURE_SYSTEM_FONT_SIZE = 50
PRESSURE_SYSTEM_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300

NARR_DIR_ARG_NAME = 'input_narr_dir_name'
FRONT_DIR_ARG_NAME = 'input_front_line_dir_name'
BULLETIN_DIR_ARG_NAME = 'input_wpc_bulletin_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
THERMAL_FIELD_ARG_NAME = 'thermal_field_name'
THERMAL_CMAP_ARG_NAME = 'thermal_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_thermal_prctile_for_colours'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR files (containing predictors).  '
    'Files therein will be found by `processed_narr_io.find_file_for_one_time` '
    'and read by `processed_narr_io.read_fields_from_file`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with fronts (represented as polylines).  Files'
    ' therein will be found by `fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_polylines_from_file`.')

BULLETIN_DIR_HELP_STRING = (
    'Name of top-level directory with WPC bulletins.  Files therein will be '
    'found by `wpc_bulletin_io.find_file` and read by '
    '`wpc_bulletin_io.read_highs_and_lows`.  If you do not want to plot high- '
    'and low-pressure centers, leave this argument alone.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Predictors will be plotted for all NARR times'
    ' in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = 'Pressure level (millibars) for NARR predictors.'

THERMAL_FIELD_HELP_STRING = (
    'Name of thermal field (to be plotted with fronts and wind barbs).  Valid '
    'options are listed below.\n{0:s}'
).format(str(VALID_THERMAL_FIELD_NAMES))

THERMAL_CMAP_HELP_STRING = (
    'Name of colour map for thermal field.  For example, if name is "YlGn", the'
    ' colour map used will be `pyplot.cm.YlGn`.  This argument supports only '
    'pyplot colour maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines min/max values in colour scheme for thermal field.  Max value '
    'at time t will be [q]th percentile of thermal field at time t, where '
    'q = `{0:s}`.  Minimum value will be [100 - q]th percentile.'
).format(MAX_PERCENTILE_ARG_NAME)

FIRST_LETTER_HELP_STRING = (
    'Letter label for first time step.  If this is "a", the label "(a)" will be'
    ' printed at the top left of the figure.  If you do not want labels, leave '
    'this argument alone.')

LETTER_INTERVAL_HELP_STRING = (
    'Interval between letter labels for successive time steps.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts/polylines'
# TOP_BULLETIN_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/wpc_bulletins/hires'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BULLETIN_DIR_ARG_NAME, type=str, required=False, default='',
    help=BULLETIN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False, default=1000,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THERMAL_FIELD_ARG_NAME, type=str, required=False,
    default=processed_narr_io.WET_BULB_THETA_NAME,
    help=THERMAL_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THERMAL_CMAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=THERMAL_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_LETTER_ARG_NAME, type=str, required=False, default='',
    help=FIRST_LETTER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LETTER_INTERVAL_ARG_NAME, type=int, required=False, default=3,
    help=LETTER_INTERVAL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_time(
        predictor_matrix, predictor_names, front_polyline_table, high_low_table,
        thermal_colour_map_object, max_thermal_prctile_for_colours,
        narr_row_limits, narr_column_limits, title_string, letter_label,
        output_file_name):
    """Plots predictors at one time.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param front_polyline_table: pandas DataFrame returned by
        `fronts_io.read_polylines_from_file`.
    :param high_low_table: pandas DataFrame returned by
        `wpc_bulletin_io.read_highs_and_lows`.
    :param thermal_colour_map_object: See documentation at top of file.
    :param max_thermal_prctile_for_colours: Same.
    :param narr_row_limits: length-2 numpy array, indicating the first and last
        NARR rows in `predictor_matrix`.  If narr_row_limits = [i, k],
        `predictor_matrix` spans rows i...k of the full NARR grid.
    :param narr_column_limits: Same but for columns.
    :param title_string: Title (will be placed above figure).
    :param letter_label: Letter label.  If this is "a", the label "(a)" will be
        printed at the top left of the figure.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        first_row_in_full_grid=narr_row_limits[0],
        last_row_in_full_grid=narr_row_limits[1],
        first_column_in_full_grid=narr_column_limits[0],
        last_column_in_full_grid=narr_column_limits[1]
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=PARALLEL_SPACING_DEG
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=MERIDIAN_SPACING_DEG
    )

    num_predictors = len(predictor_names)
    for j in range(num_predictors):
        if predictor_names[j] in WIND_FIELD_NAMES:
            continue

        min_colour_value = numpy.percentile(
            predictor_matrix[..., j], 100. - max_thermal_prctile_for_colours)
        max_colour_value = numpy.percentile(
            predictor_matrix[..., j], max_thermal_prctile_for_colours)

        nwp_plotting.plot_subgrid(
            field_matrix=predictor_matrix[..., j],
            model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
            basemap_object=basemap_object, colour_map=thermal_colour_map_object,
            min_value_in_colour_map=min_colour_value,
            max_value_in_colour_map=max_colour_value,
            first_row_in_full_grid=narr_row_limits[0],
            first_column_in_full_grid=narr_column_limits[0]
        )

        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=axes_object,
            values_to_colour=predictor_matrix[..., j],
            colour_map=thermal_colour_map_object, colour_min=min_colour_value,
            colour_max=max_colour_value, orientation='horizontal',
            extend_min=True, extend_max=True, fraction_of_axis_length=0.9)

    u_wind_index = predictor_names.index(
        processed_narr_io.U_WIND_GRID_RELATIVE_NAME)
    v_wind_index = predictor_names.index(
        processed_narr_io.V_WIND_GRID_RELATIVE_NAME)

    nwp_plotting.plot_wind_barbs_on_subgrid(
        u_wind_matrix_m_s01=predictor_matrix[..., u_wind_index],
        v_wind_matrix_m_s01=predictor_matrix[..., v_wind_index],
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        axes_object=axes_object, basemap_object=basemap_object,
        first_row_in_full_grid=narr_row_limits[0],
        first_column_in_full_grid=narr_column_limits[0],
        plot_every_k_rows=PLOT_EVERY_KTH_WIND_BARB,
        plot_every_k_columns=PLOT_EVERY_KTH_WIND_BARB,
        barb_length=WIND_BARB_LENGTH, empty_barb_radius=EMPTY_WIND_BARB_RADIUS,
        fill_empty_barb=False, colour_map=WIND_COLOUR_MAP_OBJECT,
        colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
        colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT)

    if high_low_table is None:
        num_pressure_systems = 0
    else:
        num_pressure_systems = len(high_low_table.index)

    for i in range(num_pressure_systems):
        print this_system_type_string

        this_system_type_string = high_low_table[
            wpc_bulletin_io.SYSTEM_TYPE_COLUMN].values[i]

        if this_system_type_string == wpc_bulletin_io.HIGH_PRESSURE_STRING:
            this_string = 'H'
        else:
            this_string = 'L'

        this_x_coord_metres, this_y_coord_metres = basemap_object(
            high_low_table[wpc_bulletin_io.LONGITUDE_COLUMN].values[i],
            high_low_table[wpc_bulletin_io.LATITUDE_COLUMN].values[i]
        )

        axes_object.text(
            this_x_coord_metres, this_y_coord_metres, this_string,
            fontsize=PRESSURE_SYSTEM_FONT_SIZE, color=PRESSURE_SYSTEM_COLOUR,
            horizontalalignment='center', verticalalignment='center')

    num_fronts = len(front_polyline_table.index)

    for i in range(num_fronts):
        this_front_type_string = front_polyline_table[
            front_utils.FRONT_TYPE_COLUMN].values[i]

        if this_front_type_string == front_utils.WARM_FRONT_STRING_ID:
            this_colour = WARM_FRONT_COLOUR
        else:
            this_colour = COLD_FRONT_COLOUR

        front_plotting.plot_front_with_markers(
            line_latitudes_deg=front_polyline_table[
                front_utils.LATITUDES_COLUMN].values[i],
            line_longitudes_deg=front_polyline_table[
                front_utils.LONGITUDES_COLUMN].values[i],
            axes_object=axes_object, basemap_object=basemap_object,
            front_type_string=front_polyline_table[
                front_utils.FRONT_TYPE_COLUMN].values[i],
            marker_colour=this_colour)

    pyplot.title(title_string)

    if letter_label is not None:
        plotting_utils.annotate_axes(
            axes_object=axes_object,
            annotation_string='({0:s})'.format(letter_label)
        )

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(top_narr_dir_name, top_front_line_dir_name, top_wpc_bulletin_dir_name,
         first_time_string, last_time_string, pressure_level_mb,
         thermal_field_name, thermal_colour_map_name,
         max_thermal_prctile_for_colours, first_letter_label, letter_interval,
         output_dir_name):
    """Plots predictors on full NARR grid.

    This is effectively the main method.

    :param top_narr_dir_name: See documentation at top of file.
    :param top_front_line_dir_name: Same.
    :param top_wpc_bulletin_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param thermal_field_name: Same.
    :param thermal_colour_map_name: Same.
    :param max_thermal_prctile_for_colours: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if
        `thermal_field_name not in VALID_THERMAL_FIELD_NAMES`.
    """

    # Check input args.
    if top_wpc_bulletin_dir_name in ['', 'None']:
        top_wpc_bulletin_dir_name = None

    if first_letter_label in ['', 'None']:
        first_letter_label = None

    if thermal_field_name not in VALID_THERMAL_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid thermal fields (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_THERMAL_FIELD_NAMES), thermal_field_name)

        raise ValueError(error_string)

    thermal_colour_map_object = pyplot.cm.get_cmap(thermal_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SEC, include_endpoint=True)

    # Read metadata for NARR grid.
    narr_latitude_matrix_deg, narr_longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    narr_rotation_cos_matrix, narr_rotation_sin_matrix = (
        nwp_model_utils.get_wind_rotation_angles(
            latitudes_deg=narr_latitude_matrix_deg,
            longitudes_deg=narr_longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    narr_row_limits, narr_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    narr_rotation_cos_matrix = narr_rotation_cos_matrix[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]

    narr_rotation_sin_matrix = narr_rotation_sin_matrix[
        narr_row_limits[0]:(narr_row_limits[1] + 1),
        narr_column_limits[0]:(narr_column_limits[1] + 1)
    ]

    # Do plotting.
    narr_field_names = [
        processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
        processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
        thermal_field_name
    ]

    this_letter_label = None

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_front_line_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=this_time_unix_sec)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_polyline_table = fronts_io.read_polylines_from_file(this_file_name)

        if top_wpc_bulletin_dir_name is not None:
            this_file_name = wpc_bulletin_io.find_file(
                top_directory_name=top_wpc_bulletin_dir_name,
                valid_time_unix_sec=this_time_unix_sec)

            print 'Reading data from: "{0:s}"...'.format(this_file_name)
            this_high_low_table = wpc_bulletin_io.read_highs_and_lows(
                this_file_name)

        this_predictor_matrix = None

        for this_field_name in narr_field_names:
            this_file_name = processed_narr_io.find_file_for_one_time(
                top_directory_name=top_narr_dir_name,
                field_name=this_field_name,
                pressure_level_mb=pressure_level_mb,
                valid_time_unix_sec=this_time_unix_sec)

            print 'Reading data from: "{0:s}"...'.format(this_file_name)
            this_field_matrix = processed_narr_io.read_fields_from_file(
                this_file_name
            )[0][0, ...]

            this_field_matrix = utils.fill_nans(this_field_matrix)
            this_field_matrix = this_field_matrix[
                narr_row_limits[0]:(narr_row_limits[1] + 1),
                narr_column_limits[0]:(narr_column_limits[1] + 1)
            ]

            if this_field_name in [processed_narr_io.TEMPERATURE_NAME,
                                   processed_narr_io.WET_BULB_THETA_NAME]:
                this_field_matrix -= ZERO_CELSIUS_IN_KELVINS

            if this_field_name == processed_narr_io.SPECIFIC_HUMIDITY_NAME:
                this_field_matrix = this_field_matrix * KG_TO_GRAMS

            this_field_matrix = numpy.expand_dims(this_field_matrix, axis=-1)

            if this_predictor_matrix is None:
                this_predictor_matrix = this_field_matrix + 0.
            else:
                this_predictor_matrix = numpy.concatenate(
                    (this_predictor_matrix, this_field_matrix), axis=-1)

        u_wind_index = narr_field_names.index(
            processed_narr_io.U_WIND_GRID_RELATIVE_NAME)
        v_wind_index = narr_field_names.index(
            processed_narr_io.V_WIND_GRID_RELATIVE_NAME)

        (this_predictor_matrix[..., u_wind_index],
         this_predictor_matrix[..., v_wind_index]
        ) = nwp_model_utils.rotate_winds_to_earth_relative(
            u_winds_grid_relative_m_s01=this_predictor_matrix[
                ..., u_wind_index],
            v_winds_grid_relative_m_s01=this_predictor_matrix[
                ..., v_wind_index],
            rotation_angle_cosines=narr_rotation_cos_matrix,
            rotation_angle_sines=narr_rotation_sin_matrix)

        this_title_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, NICE_TIME_FORMAT)

        if pressure_level_mb == 1013:
            this_title_string += ' at surface'
        else:
            this_title_string += ' at {0:d} mb'.format(pressure_level_mb)

        this_default_time_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, DEFAULT_TIME_FORMAT)

        this_output_file_name = '{0:s}/predictors_{1:s}.jpg'.format(
            output_dir_name, this_default_time_string)

        if first_letter_label is not None:
            if this_letter_label is None:
                this_letter_label = first_letter_label
            else:
                this_letter_label = chr(
                    ord(this_letter_label) + letter_interval
                )

        _plot_one_time(
            predictor_matrix=this_predictor_matrix,
            predictor_names=narr_field_names,
            front_polyline_table=this_polyline_table,
            high_low_table=this_high_low_table,
            thermal_colour_map_object=thermal_colour_map_object,
            max_thermal_prctile_for_colours=max_thermal_prctile_for_colours,
            narr_row_limits=narr_row_limits,
            narr_column_limits=narr_column_limits,
            title_string=this_title_string, letter_label=this_letter_label,
            output_file_name=this_output_file_name)

        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_narr_dir_name=getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME),
        top_front_line_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        top_wpc_bulletin_dir_name=getattr(
            INPUT_ARG_OBJECT, BULLETIN_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        thermal_field_name=getattr(INPUT_ARG_OBJECT, THERMAL_FIELD_ARG_NAME),
        thermal_colour_map_name=getattr(INPUT_ARG_OBJECT,
                                        THERMAL_CMAP_ARG_NAME),
        max_thermal_prctile_for_colours=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
