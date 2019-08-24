"""Plots one or more input examples."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting

RANDOM_SEED = 6695

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_HALF_ROWS = 16
NUM_HALF_COLUMNS = 16
PREDICTOR_NAMES = [
    predictor_utils.WET_BULB_THETA_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]

FRONT_MARKER_SIZE = 20
FRONT_SPACING_METRES = 50000.
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

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_WIDTH = 2
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_example_file_name'
FRONT_DIR_ARG_NAME = 'input_front_line_dir_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
COLOUR_MAP_ARG_NAME = 'thetaw_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `learning_examples_io.read_file`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with frontal polylines.  Files therein will be'
    ' found by `fronts_io.find_polyline_file` and read by '
    '`fronts_io.read_polylines_from_file`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to draw randomly from `{0:s}`.  To plot all examples, '
    'leave this argument alone.  To plot selected examples, use `{1:s}` and '
    'leave this argument alone.'
).format(INPUT_FILE_ARG_NAME, EXAMPLE_INDICES_ARG_NAME)

EXAMPLE_INDICES_HELP_STRING = (
    '[used only if `{0:s}` is default] Indices of examples to draw from '
    '`{1:s}`.  Only these examples will be plotted.'
).format(NUM_EXAMPLES_ARG_NAME, INPUT_FILE_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = 'Will plot this pressure level (millibars).'

COLOUR_MAP_HELP_STRING = (
    'Name of colour map for wet-bulb potential temperature.  For example, if '
    'name is "YlGn", the colour map used will be `pyplot.cm.YlGn`.  This '
    'argument supports only pyplot colour maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in each colour map.  Max theta_w for each example '
    'will be the [q]th percentile of all theta_w values in the grid, where '
    'q = `{0:s}`.  Minimum value will be [100 - q]th percentile.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

TOP_FRONT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts_netcdf/polylines'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=EXAMPLE_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=True,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(example_file_name, top_front_line_dir_name, num_examples,
         example_indices, pressure_level_mb, thetaw_colour_map_name,
         max_colour_percentile, output_dir_name):
    """Plots one or more input examples.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param top_front_line_dir_name: Same.
    :param num_examples: Same.
    :param example_indices: Same.
    :param pressure_level_mb: Same.
    :param thetaw_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    if num_examples <= 0:
        num_examples = None

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
    else:
        error_checking.assert_is_greater(num_examples, 0)

    error_checking.assert_is_geq(max_colour_percentile, 0)
    error_checking.assert_is_leq(max_colour_percentile, 100)
    thetaw_colour_map_object = pyplot.cm.get_cmap(thetaw_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading normalized examples from: "{0:s}"...'.format(
        example_file_name))

    pressure_array_mb = numpy.full(
        len(PREDICTOR_NAMES), pressure_level_mb, dtype=int
    )

    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name, num_half_rows_to_keep=NUM_HALF_ROWS,
        num_half_columns_to_keep=NUM_HALF_COLUMNS,
        predictor_names_to_keep=PREDICTOR_NAMES,
        pressure_levels_to_keep_mb=pressure_array_mb)

    # TODO(thunderhoser): This is a HACK (assuming that normalization method is
    # z-score and not min-max).
    mean_value_matrix = example_dict[examples_io.FIRST_NORM_PARAM_KEY]
    standard_deviation_matrix = example_dict[examples_io.SECOND_NORM_PARAM_KEY]

    normalization_dict = {
        ml_utils.MIN_VALUE_MATRIX_KEY: None,
        ml_utils.MAX_VALUE_MATRIX_KEY: None,
        ml_utils.MEAN_VALUE_MATRIX_KEY: mean_value_matrix,
        ml_utils.STDEV_MATRIX_KEY: standard_deviation_matrix
    }

    example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
        ml_utils.denormalize_predictors(
            predictor_matrix=example_dict[examples_io.PREDICTOR_MATRIX_KEY],
            normalization_dict=normalization_dict)
    )

    latitude_matrix_deg, longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)
    )

    rotation_cosine_matrix, rotation_sine_matrix = (
        nwp_model_utils.get_wind_rotation_angles(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    num_examples_total = len(example_dict[examples_io.VALID_TIMES_KEY])
    example_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int)

    if num_examples is not None:
        num_examples = min([num_examples, num_examples_total])

        numpy.random.seed(RANDOM_SEED)
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False)

    thetaw_index = PREDICTOR_NAMES.index(predictor_utils.WET_BULB_THETA_NAME)
    u_wind_index = PREDICTOR_NAMES.index(
        predictor_utils.U_WIND_GRID_RELATIVE_NAME)
    v_wind_index = PREDICTOR_NAMES.index(
        predictor_utils.V_WIND_GRID_RELATIVE_NAME)

    for i in example_indices:
        this_center_row_index = example_dict[examples_io.ROW_INDICES_KEY][i]
        this_first_row_index = this_center_row_index - NUM_HALF_ROWS
        this_last_row_index = this_center_row_index + NUM_HALF_ROWS

        this_center_column_index = example_dict[
            examples_io.COLUMN_INDICES_KEY][i]
        this_first_column_index = this_center_column_index - NUM_HALF_COLUMNS
        this_last_column_index = this_center_column_index + NUM_HALF_COLUMNS

        this_u_wind_matrix_m_s01 = example_dict[
            examples_io.PREDICTOR_MATRIX_KEY][i, ..., u_wind_index]
        this_v_wind_matrix_m_s01 = example_dict[
            examples_io.PREDICTOR_MATRIX_KEY][i, ..., v_wind_index]

        this_cos_matrix = rotation_cosine_matrix[
            this_first_row_index:(this_last_row_index + 1),
            this_first_column_index:(this_last_column_index + 1)
        ]
        this_sin_matrix = rotation_sine_matrix[
            this_first_row_index:(this_last_row_index + 1),
            this_first_column_index:(this_last_column_index + 1)
        ]

        this_u_wind_matrix_m_s01, this_v_wind_matrix_m_s01 = (
            nwp_model_utils.rotate_winds_to_earth_relative(
                u_winds_grid_relative_m_s01=this_u_wind_matrix_m_s01,
                v_winds_grid_relative_m_s01=this_v_wind_matrix_m_s01,
                rotation_angle_cosines=this_cos_matrix,
                rotation_angle_sines=this_sin_matrix)
        )

        _, axes_object, basemap_object = nwp_plotting.init_basemap(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_id=nwp_model_utils.NAME_OF_221GRID,
            first_row_in_full_grid=this_first_row_index,
            last_row_in_full_grid=this_last_row_index,
            first_column_in_full_grid=this_first_column_index,
            last_column_in_full_grid=this_last_column_index,
            resolution_string='i')

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH)
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH)
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH)

        this_latitude_matrix_deg = latitude_matrix_deg[
            this_first_row_index:(this_last_row_index + 1),
            this_first_column_index:(this_last_column_index + 1)
        ]
        this_longitude_matrix_deg = longitude_matrix_deg[
            this_first_row_index:(this_last_row_index + 1),
            this_first_column_index:(this_last_column_index + 1)
        ]

        plotting_utils.plot_parallels(
            basemap_object=basemap_object, axes_object=axes_object,
            min_latitude_deg=numpy.min(this_latitude_matrix_deg),
            max_latitude_deg=numpy.max(this_latitude_matrix_deg),
            num_parallels=NUM_PARALLELS
        )
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            min_longitude_deg=numpy.min(this_longitude_matrix_deg),
            max_longitude_deg=numpy.max(this_longitude_matrix_deg),
            num_meridians=NUM_MERIDIANS
        )

        this_thetaw_matrix_kelvins = example_dict[
            examples_io.PREDICTOR_MATRIX_KEY
        ][i, ..., thetaw_index]

        this_min_value = numpy.percentile(
            this_thetaw_matrix_kelvins, 100. - max_colour_percentile)
        this_max_value = numpy.percentile(
            this_thetaw_matrix_kelvins, max_colour_percentile)

        nwp_plotting.plot_subgrid(
            field_matrix=this_thetaw_matrix_kelvins,
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_id=nwp_model_utils.NAME_OF_221GRID,
            axes_object=axes_object, basemap_object=basemap_object,
            colour_map_object=thetaw_colour_map_object,
            min_colour_value=this_min_value, max_colour_value=this_max_value,
            first_row_in_full_grid=this_first_row_index,
            first_column_in_full_grid=this_first_column_index)

        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=this_thetaw_matrix_kelvins,
            colour_map_object=thetaw_colour_map_object,
            min_value=this_min_value, max_value=this_max_value,
            orientation_string='horizontal', extend_min=True, extend_max=True,
            fraction_of_axis_length=0.8)

        colour_bar_object.set_label(
            r'Wet-bulb potential temperature ($^{\circ}$C)'
        )

        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)
        colour_bar_object.ax.set_xticklabels(tick_values)

        nwp_plotting.plot_wind_barbs_on_subgrid(
            u_wind_matrix_m_s01=this_u_wind_matrix_m_s01,
            v_wind_matrix_m_s01=this_v_wind_matrix_m_s01,
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_id=nwp_model_utils.NAME_OF_221GRID, axes_object=axes_object,
            basemap_object=basemap_object,
            first_row_in_full_grid=this_first_row_index,
            first_column_in_full_grid=this_first_column_index,
            barb_length=WIND_BARB_LENGTH,
            empty_barb_radius=EMPTY_WIND_BARB_RADIUS, fill_empty_barb=False,
            colour_map=WIND_COLOUR_MAP_OBJECT,
            colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
            colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT)

        this_front_file_name = fronts_io.find_polyline_file(
            top_directory_name=top_front_line_dir_name,
            valid_time_unix_sec=example_dict[examples_io.VALID_TIMES_KEY][i]
        )

        this_polyline_table = fronts_io.read_polylines_from_file(
            this_front_file_name
        )[0]
        this_num_fronts = len(this_polyline_table.index)

        for j in range(this_num_fronts):
            this_front_type_string = this_polyline_table[
                front_utils.FRONT_TYPE_COLUMN].values[j]

            if this_front_type_string == front_utils.WARM_FRONT_STRING:
                this_colour = WARM_FRONT_COLOUR
            else:
                this_colour = COLD_FRONT_COLOUR

            front_plotting.plot_front_with_markers(
                line_latitudes_deg=this_polyline_table[
                    front_utils.LATITUDES_COLUMN].values[j],
                line_longitudes_deg=this_polyline_table[
                    front_utils.LONGITUDES_COLUMN].values[j],
                axes_object=axes_object, basemap_object=basemap_object,
                front_type_string=this_polyline_table[
                    front_utils.FRONT_TYPE_COLUMN].values[j],
                marker_colour=this_colour, marker_size=FRONT_MARKER_SIZE,
                marker_spacing_metres=FRONT_SPACING_METRES
            )

        title_string = 'Predictors at {0:s}'.format(
            'surface'
            if pressure_level_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB
            else '{0:d} mb'.format(pressure_level_mb)
        )

        pyplot.title(title_string)
        this_output_file_name = '{0:s}/example{1:06d}.jpg'.format(
            output_dir_name, i)

        print('Saving figure to: "{0:s}"...'.format(this_output_file_name))
        pyplot.savefig(this_output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                       pad_inches=0, bbox_inches='tight')
        pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_front_line_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        thetaw_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
