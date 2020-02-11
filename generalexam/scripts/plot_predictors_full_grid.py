"""Plots predictors on full NARR grid."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import wind_plotting
from generalexam.ge_io import fronts_io
from generalexam.ge_io import predictor_io
from generalexam.ge_io import wpc_bulletin_input
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import predictor_utils
from generalexam.ge_utils import conversions
from generalexam.plotting import front_plotting
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

DEFAULT_TIME_FORMAT = '%Y%m%d%H'
NICE_TIME_FORMAT = '%H00 UTC %-d %b %Y'
TIME_INTERVAL_SECONDS = 10800

MB_TO_PASCALS = 100.
KG_TO_GRAMS = 1000.
ZERO_CELSIUS_IN_KELVINS = 273.15

VALID_THERMAL_FIELD_NAMES = [
    predictor_utils.TEMPERATURE_NAME,
    predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.WET_BULB_THETA_NAME
]

WIND_FIELD_NAMES = [
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]

PREDICTOR_NAME_ABBREV_TO_NICE = {
    predictor_utils.TEMPERATURE_NAME: r'Temperature ($^{\circ}$C)',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: r'Specific humidity (g kg$^{-1}$)',
    predictor_utils.WET_BULB_THETA_NAME:
        r'Wet-bulb potential temperature ($^{\circ}$C)'
}

FRONT_LINE_WIDTH = 8
BORDER_COLOUR = numpy.full(3, 152. / 255)
WARM_FRONT_COLOUR = numpy.array([30, 120, 180], dtype=float) / 255
COLD_FRONT_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255

WIND_BARB_COLOUR = numpy.full(3, 0.)
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.

WIND_COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap([WIND_BARB_COLOUR])
WIND_COLOUR_MAP_OBJECT.set_under(WIND_BARB_COLOUR)
WIND_COLOUR_MAP_OBJECT.set_over(WIND_BARB_COLOUR)

WIND_BARB_WIDTH = 1.5
WIND_BARB_LENGTH = 7
EMPTY_WIND_BARB_RADIUS = 0.1
PLOT_EVERY_KTH_WIND_BARB = 8

PRESSURE_SYSTEM_FONT_SIZE = 50
PRESSURE_SYSTEM_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300

PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_front_line_dir_name'
BULLETIN_DIR_ARG_NAME = 'input_wpc_bulletin_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
THERMAL_FIELD_ARG_NAME = 'thermal_field_name'
THERMAL_CMAP_ARG_NAME = 'thermal_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.'
)
FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with fronts (represented as polylines).  Files'
    ' therein will be found by `fronts_io.find_polyline_file` and read by '
    '`fronts_io.read_polylines_from_file`.'
)
BULLETIN_DIR_HELP_STRING = (
    'Name of top-level directory with WPC bulletins.  Files therein will be '
    'found by `wpc_bulletin_input.find_file` and read by '
    '`wpc_bulletin_input.read_highs_and_lows`.  If you do not want to plot '
    'high- and low-pressure centers, leave this argument alone.'
)
TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Predictors will be plotted for all times in '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = 'Pressure level (millibars) for predictors.'

THERMAL_FIELD_HELP_STRING = (
    'Name of thermal field (to be plotted with fronts and wind barbs).  Valid '
    'options are listed below.\n{0:s}'
).format(str(VALID_THERMAL_FIELD_NAMES))

THERMAL_CMAP_HELP_STRING = (
    'Name of colour map for thermal field.  For example, if name is "YlGn", the'
    ' colour map used will be `pyplot.cm.YlGn`.  This argument supports only '
    'pyplot colour maps.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Determines min/max values in colour scheme for thermal field.  Max value '
    'at time t will be [q]th percentile of thermal field at time t, where '
    'q = `{0:s}`.  Minimum value will be [100 - q]th percentile.'
).format(MAX_PERCENTILE_ARG_NAME)

FIRST_LETTER_HELP_STRING = (
    'Letter label for first time step.  If this is "a", the label "(a)" will be'
    ' printed at the top left of the figure.  If you do not want labels, leave '
    'this argument alone.'
)
LETTER_INTERVAL_HELP_STRING = (
    'Interval between letter labels for successive time steps.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

# TOP_PREDICTOR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'
# TOP_FRONT_DIR_NAME_DEFAULT = (
#     '/condo/swatwork/ralager/fronts_netcdf/polylines'
# )
# TOP_BULLETIN_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/wpc_bulletins/hires'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=True,
    help=FRONT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BULLETIN_DIR_ARG_NAME, type=str, required=False, default='',
    help=BULLETIN_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False, default=1000,
    help=PRESSURE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + THERMAL_FIELD_ARG_NAME, type=str, required=False,
    default=predictor_utils.WET_BULB_THETA_NAME, help=THERMAL_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + THERMAL_CMAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=THERMAL_CMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_LETTER_ARG_NAME, type=str, required=False, default='',
    help=FIRST_LETTER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LETTER_INTERVAL_ARG_NAME, type=int, required=False, default=3,
    help=LETTER_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_one_file(
        top_predictor_dir_name, thermal_field_name, pressure_level_mb,
        valid_time_unix_sec, rotation_cosine_matrix, rotation_sine_matrix):
    """Reads predictors from one file.

    M = number of rows in grid
    N = number of columns in grid

    :param top_predictor_dir_name: See documentation at top of file.
    :param thermal_field_name: Same.
    :param pressure_level_mb: Same.
    :param valid_time_unix_sec: Valid time.
    :param rotation_cosine_matrix: M-by-N numpy array with cosines.  Will be
        used to rotate wind from grid-relative to Earth-relative.
    :param rotation_sine_matrix: Same but for sines.
    :return: predictor_dict: See doc for `predictor_io.read_file`.
    """

    predictor_names = [
        thermal_field_name, predictor_utils.U_WIND_GRID_RELATIVE_NAME,
        predictor_utils.V_WIND_GRID_RELATIVE_NAME
    ]
    pressure_levels_mb = numpy.full(
        len(predictor_names), pressure_level_mb, dtype=int
    )

    predictor_file_name = predictor_io.find_file(
        top_directory_name=top_predictor_dir_name,
        valid_time_unix_sec=valid_time_unix_sec
    )

    print('Reading data from: "{0:s}"...'.format(predictor_file_name))

    try:
        predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_name,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            field_names_to_keep=predictor_names
        )
    except Exception as e:
        if thermal_field_name != predictor_utils.WET_BULB_THETA_NAME:
            raise e

        dummy_predictor_names = [
            predictor_utils.TEMPERATURE_NAME,
            predictor_utils.SPECIFIC_HUMIDITY_NAME,
            predictor_utils.U_WIND_GRID_RELATIVE_NAME,
            predictor_utils.V_WIND_GRID_RELATIVE_NAME
        ]

        if pressure_level_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            dummy_predictor_names.insert(2, predictor_utils.PRESSURE_NAME)

        dummy_pressure_levels_mb = numpy.full(
            len(dummy_predictor_names), pressure_level_mb, dtype=int
        )

        predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_name,
            pressure_levels_to_keep_mb=dummy_pressure_levels_mb,
            field_names_to_keep=dummy_predictor_names
        )

        predictor_matrix = predictor_dict[predictor_utils.DATA_MATRIX_KEY]
        if pressure_level_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            pressure_matrix_pa = predictor_matrix[..., 2]
        else:
            pressure_matrix_pa = numpy.full(
                predictor_matrix.shape[:-1], pressure_level_mb * MB_TO_PASCALS
            )

        dewpoint_matrix_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=predictor_matrix[..., 1],
                total_pressures_pascals=pressure_matrix_pa
            )
        )

        theta_w_matrix_kelvins = conversions.dewpoint_to_wet_bulb_temperature(
            dewpoints_kelvins=dewpoint_matrix_kelvins,
            temperatures_kelvins=predictor_matrix[..., 0],
            total_pressures_pascals=pressure_matrix_pa
        )

        predictor_matrix = predictor_matrix[..., -3:]
        predictor_matrix[..., 0] = theta_w_matrix_kelvins

        predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_matrix
        predictor_dict[predictor_utils.FIELD_NAMES_KEY] = predictor_names
        predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] = pressure_levels_mb

    predictor_matrix = predictor_dict[predictor_utils.DATA_MATRIX_KEY]

    if thermal_field_name in [
            predictor_utils.TEMPERATURE_NAME,
            predictor_utils.WET_BULB_THETA_NAME
    ]:
        predictor_matrix[..., 0] -= ZERO_CELSIUS_IN_KELVINS

    if thermal_field_name == predictor_utils.SPECIFIC_HUMIDITY_NAME:
        predictor_matrix[..., 0] *= KG_TO_GRAMS

    if rotation_cosine_matrix is not None:
        predictor_matrix[0, ..., 1], predictor_matrix[0, ..., 2] = (
            nwp_model_utils.rotate_winds_to_earth_relative(
                u_winds_grid_relative_m_s01=predictor_matrix[0, ..., 1],
                v_winds_grid_relative_m_s01=predictor_matrix[0, ..., 2],
                rotation_angle_cosines=rotation_cosine_matrix,
                rotation_angle_sines=rotation_sine_matrix)
        )

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_matrix

    return predictor_dict


def _plot_one_time(
        predictor_matrix, predictor_names, front_polyline_table, high_low_table,
        thermal_colour_map_object, max_colour_percentile, title_string,
        letter_label, output_file_name):
    """Plots predictors at one time.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param front_polyline_table: pandas DataFrame returned by
        `fronts_io.read_polylines_from_file`.
    :param high_low_table: pandas DataFrame returned by
        `wpc_bulletin_input.read_highs_and_lows`.
    :param thermal_colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param title_string: Title (will be placed above figure).
    :param letter_label: Letter label.  If this is "a", the label "(a)" will be
        printed at the top left of the figure.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    basemap_dict = plot_gridded_stats.plot_basemap(
        data_matrix=predictor_matrix, border_colour=BORDER_COLOUR
    )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    matrix_to_plot = basemap_dict[plot_gridded_stats.MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    num_predictors = len(predictor_names)

    for j in range(num_predictors):
        if predictor_names[j] in WIND_FIELD_NAMES:
            continue

        min_colour_value = numpy.nanpercentile(
            matrix_to_plot[..., j], 100. - max_colour_percentile
        )
        max_colour_value = numpy.nanpercentile(
            matrix_to_plot[..., j], max_colour_percentile
        )
        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

        prediction_plotting.plot_counts_on_general_grid(
            count_or_frequency_matrix=matrix_to_plot[..., j],
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            axes_object=axes_object, basemap_object=basemap_object,
            colour_map_object=thermal_colour_map_object,
            colour_norm_object=colour_norm_object
        )

        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=matrix_to_plot[..., j],
            colour_map_object=thermal_colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string='horizontal', padding=0.05,
            extend_min=True, extend_max=True, fraction_of_axis_length=1.
        )

        colour_bar_object.set_label(
            PREDICTOR_NAME_ABBREV_TO_NICE[predictor_names[j]]
        )

        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)
        colour_bar_object.ax.set_xticklabels(tick_values)

    u_wind_index = predictor_names.index(
        predictor_utils.U_WIND_GRID_RELATIVE_NAME
    )
    v_wind_index = predictor_names.index(
        predictor_utils.V_WIND_GRID_RELATIVE_NAME
    )

    u_wind_matrix_m_s01 = matrix_to_plot[..., u_wind_index][
        ::PLOT_EVERY_KTH_WIND_BARB, ::PLOT_EVERY_KTH_WIND_BARB
    ]
    v_wind_matrix_m_s01 = matrix_to_plot[..., v_wind_index][
        ::PLOT_EVERY_KTH_WIND_BARB, ::PLOT_EVERY_KTH_WIND_BARB
    ]
    latitude_matrix_deg = latitude_matrix_deg[
        ::PLOT_EVERY_KTH_WIND_BARB, ::PLOT_EVERY_KTH_WIND_BARB
    ]
    longitude_matrix_deg = longitude_matrix_deg[
        ::PLOT_EVERY_KTH_WIND_BARB, ::PLOT_EVERY_KTH_WIND_BARB
    ]

    u_winds_m_s01 = numpy.ravel(u_wind_matrix_m_s01)
    v_winds_m_s01 = numpy.ravel(v_wind_matrix_m_s01)
    wind_latitudes_deg = numpy.ravel(latitude_matrix_deg)
    wind_longitudes_deg = numpy.ravel(longitude_matrix_deg)

    nan_flags = numpy.logical_or(
        numpy.isnan(u_winds_m_s01), numpy.isnan(v_winds_m_s01)
    )
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    wind_plotting.plot_wind_barbs(
        basemap_object=basemap_object, axes_object=axes_object,
        latitudes_deg=wind_latitudes_deg[real_indices],
        longitudes_deg=wind_longitudes_deg[real_indices],
        u_winds_m_s01=u_winds_m_s01[real_indices],
        v_winds_m_s01=v_winds_m_s01[real_indices],
        barb_length=WIND_BARB_LENGTH, barb_width=WIND_BARB_WIDTH,
        empty_barb_radius=EMPTY_WIND_BARB_RADIUS,
        fill_empty_barb=False, colour_map=WIND_COLOUR_MAP_OBJECT,
        colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
        colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT
    )

    if high_low_table is None:
        num_pressure_systems = 0
    else:
        num_pressure_systems = len(high_low_table.index)

    for i in range(num_pressure_systems):
        this_system_type_string = (
            high_low_table[wpc_bulletin_input.SYSTEM_TYPE_COLUMN].values[i]
        )

        if this_system_type_string == wpc_bulletin_input.HIGH_PRESSURE_STRING:
            this_string = 'H'
        else:
            this_string = 'L'

        this_x_coord_metres, this_y_coord_metres = basemap_object(
            high_low_table[wpc_bulletin_input.LONGITUDE_COLUMN].values[i],
            high_low_table[wpc_bulletin_input.LATITUDE_COLUMN].values[i]
        )

        axes_object.text(
            this_x_coord_metres, this_y_coord_metres, this_string,
            fontsize=PRESSURE_SYSTEM_FONT_SIZE, color=PRESSURE_SYSTEM_COLOUR,
            fontweight='bold', horizontalalignment='center',
            verticalalignment='center'
        )

    num_fronts = len(front_polyline_table.index)

    for i in range(num_fronts):
        this_front_type_string = (
            front_polyline_table[front_utils.FRONT_TYPE_COLUMN].values[i]
        )

        if this_front_type_string == front_utils.WARM_FRONT_STRING:
            this_colour = WARM_FRONT_COLOUR
        else:
            this_colour = COLD_FRONT_COLOUR

        front_plotting.plot_front_with_markers(
            line_latitudes_deg=
            front_polyline_table[front_utils.LATITUDES_COLUMN].values[i],
            line_longitudes_deg=
            front_polyline_table[front_utils.LONGITUDES_COLUMN].values[i],
            axes_object=axes_object, basemap_object=basemap_object,
            front_type_string=
            front_polyline_table[front_utils.FRONT_TYPE_COLUMN].values[i],
            marker_colour=this_colour
        )

    axes_object.set_title(title_string)

    if letter_label is not None:
        plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_predictor_dir_name, top_front_line_dir_name,
         top_wpc_bulletin_dir_name, first_time_string, last_time_string,
         pressure_level_mb, thermal_field_name, thermal_colour_map_name,
         max_colour_percentile, first_letter_label, letter_interval,
         output_dir_name):
    """Plots predictors on full NARR grid.

    This is effectively the main method.

    :param top_predictor_dir_name: See documentation at top of file.
    :param top_front_line_dir_name: Same.
    :param top_wpc_bulletin_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param pressure_level_mb: Same.
    :param thermal_field_name: Same.
    :param thermal_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if
        `thermal_field_name not in VALID_THERMAL_FIELD_NAMES`.
    """

    # Process input args.
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

    thermal_colour_map_object = pyplot.get_cmap(thermal_colour_map_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, DEFAULT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, DEFAULT_TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

    # Find rotation matrix for wind (used to convert from grid-relative to
    # Earth-relative).
    this_predictor_dict = _read_one_file(
        top_predictor_dir_name=top_predictor_dir_name,
        thermal_field_name=thermal_field_name,
        pressure_level_mb=pressure_level_mb,
        valid_time_unix_sec=first_time_unix_sec,
        rotation_cosine_matrix=None, rotation_sine_matrix=None
    )

    predictor_names = this_predictor_dict[predictor_utils.FIELD_NAMES_KEY]

    num_grid_rows = (
        this_predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[1]
    )
    num_grid_columns = (
        this_predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[2]
    )
    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    latitude_matrix_deg, longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=full_grid_name)
    )

    rotation_cosine_matrix, rotation_sine_matrix = (
        nwp_model_utils.get_wind_rotation_angles(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    # Do plotting.
    this_letter_label = None

    for this_time_unix_sec in valid_times_unix_sec:

        # Read predictors.
        this_predictor_dict = _read_one_file(
            top_predictor_dir_name=top_predictor_dir_name,
            thermal_field_name=thermal_field_name,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=this_time_unix_sec,
            rotation_cosine_matrix=rotation_cosine_matrix,
            rotation_sine_matrix=rotation_sine_matrix
        )

        this_predictor_matrix = (
            this_predictor_dict[predictor_utils.DATA_MATRIX_KEY][0, ...]
        )

        # Read frontal polylines.
        this_file_name = fronts_io.find_polyline_file(
            top_directory_name=top_front_line_dir_name,
            valid_time_unix_sec=this_time_unix_sec
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_polyline_table = fronts_io.read_polylines_from_file(
            this_file_name
        )[0]

        # If applicable, read centers of high- and low-pressure systems.
        if top_wpc_bulletin_dir_name is None:
            this_high_low_table = None
        else:
            this_file_name = wpc_bulletin_input.find_file(
                top_directory_name=top_wpc_bulletin_dir_name,
                valid_time_unix_sec=this_time_unix_sec
            )

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            this_high_low_table = wpc_bulletin_input.read_highs_and_lows(
                this_file_name
            )

        # Set title, output location, and panel label.
        this_title_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, NICE_TIME_FORMAT
        )

        if pressure_level_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            this_title_string += ' at surface'
        else:
            this_title_string += ' at {0:d} mb'.format(pressure_level_mb)

        this_output_file_name = '{0:s}/predictors_{1:s}.jpg'.format(
            output_dir_name,
            time_conversion.unix_sec_to_string(
                this_time_unix_sec, DEFAULT_TIME_FORMAT)
        )

        if first_letter_label is not None:
            if this_letter_label is None:
                this_letter_label = first_letter_label
            else:
                this_letter_label = chr(
                    ord(this_letter_label) + letter_interval
                )

        # Plot.
        _plot_one_time(
            predictor_matrix=this_predictor_matrix,
            predictor_names=predictor_names,
            front_polyline_table=this_polyline_table,
            high_low_table=this_high_low_table,
            thermal_colour_map_object=thermal_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            title_string=this_title_string, letter_label=this_letter_label,
            output_file_name=this_output_file_name
        )

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_front_line_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        top_wpc_bulletin_dir_name=getattr(
            INPUT_ARG_OBJECT, BULLETIN_DIR_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        thermal_field_name=getattr(INPUT_ARG_OBJECT, THERMAL_FIELD_ARG_NAME),
        thermal_colour_map_name=getattr(
            INPUT_ARG_OBJECT, THERMAL_CMAP_ARG_NAME
        ),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
