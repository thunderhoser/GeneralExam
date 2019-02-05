"""Plots one or more input examples."""

import random
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting

random.seed(6695)
numpy.random.seed(6695)

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_HALF_ROWS = 16
NUM_HALF_COLUMNS = 16
NARR_PREDICTOR_NAMES = [
    processed_narr_io.WET_BULB_THETA_NAME,
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME
]

WARM_FRONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
COLD_FRONT_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

WIND_COLOUR_MAP_OBJECT = pyplot.cm.binary
WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.

PARALLEL_SPACING_DEG = 2.
MERIDIAN_SPACING_DEG = 6.
BORDER_WIDTH = 2
BORDER_COLOUR = numpy.full(3, 152. / 255)
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_example_file_name'
FRONT_DIR_ARG_NAME = 'input_front_line_dir_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
COLOUR_MAP_ARG_NAME = 'thetaw_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'thetaw_max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`training_validation_io.read_downsized_3d_examples`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with frontal polylines.  Files therein will be'
    ' found by `fronts_io.find_file_for_one_time` and read by '
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

COLOUR_MAP_HELP_STRING = (
    'Name of colour map for wet-bulb potential temperature.  For example, if '
    'name is "YlGn", the colour map used will be `pyplot.cm.YlGn`.  This '
    'argument supports only pyplot colour maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in each colour map.  Max wet-bulb potential '
    'temperature (theta_w) for each example will be the [q]th percentile of all'
    ' theta_w values in the grid, where q = `{0:s}`.  Minimum value will be '
    '[100 - q]th percentile.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

TOP_FRONT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts/polylines'

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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlGn',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(example_file_name, top_front_line_dir_name, num_examples,
         example_indices, thetaw_colour_map_name, thetaw_max_colour_percentile,
         output_dir_name):
    """Plots one or more input examples.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param top_front_line_dir_name: Same.
    :param num_examples: Same.
    :param example_indices: Same.
    :param thetaw_colour_map_name: Same.
    :param thetaw_max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    if num_examples <= 0:
        num_examples = None

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
    else:
        error_checking.assert_is_greater(num_examples, 0)

    error_checking.assert_is_geq(thetaw_max_colour_percentile, 0)
    error_checking.assert_is_leq(thetaw_max_colour_percentile, 100)
    thetaw_colour_map_object = pyplot.cm.get_cmap(thetaw_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading normalized examples from: "{0:s}"...'.format(
        example_file_name)

    example_dict = trainval_io.read_downsized_3d_examples(
        netcdf_file_name=example_file_name, num_half_rows_to_keep=NUM_HALF_ROWS,
        num_half_columns_to_keep=NUM_HALF_COLUMNS,
        predictor_names_to_keep=NARR_PREDICTOR_NAMES)

    # TODO(thunderhoser): This is a HACK (assuming that normalization method is
    # z-score and not min-max).
    mean_value_matrix = example_dict[trainval_io.FIRST_NORM_PARAM_KEY]
    standard_deviation_matrix = example_dict[trainval_io.SECOND_NORM_PARAM_KEY]

    normalization_dict = {
        ml_utils.MIN_VALUE_MATRIX_KEY: None,
        ml_utils.MAX_VALUE_MATRIX_KEY: None,
        ml_utils.MEAN_VALUE_MATRIX_KEY: mean_value_matrix,
        ml_utils.STDEV_MATRIX_KEY: standard_deviation_matrix
    }

    example_dict[trainval_io.PREDICTOR_MATRIX_KEY] = (
        ml_utils.denormalize_predictors(
            predictor_matrix=example_dict[trainval_io.PREDICTOR_MATRIX_KEY],
            normalization_dict=normalization_dict)
    )

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

    num_examples_total = len(example_dict[trainval_io.TARGET_TIMES_KEY])
    example_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int)

    if num_examples is not None:
        num_examples = min([num_examples, num_examples_total])
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False)

    thetaw_index = NARR_PREDICTOR_NAMES.index(
        processed_narr_io.WET_BULB_THETA_NAME)
    u_wind_index = NARR_PREDICTOR_NAMES.index(
        processed_narr_io.U_WIND_GRID_RELATIVE_NAME)
    v_wind_index = NARR_PREDICTOR_NAMES.index(
        processed_narr_io.V_WIND_GRID_RELATIVE_NAME)

    for i in example_indices:
        this_center_row_index = example_dict[trainval_io.ROW_INDICES_KEY][i]
        this_first_row_index = this_center_row_index - NUM_HALF_ROWS
        this_last_row_index = this_center_row_index + NUM_HALF_ROWS

        this_center_column_index = example_dict[
            trainval_io.COLUMN_INDICES_KEY][i]
        this_first_column_index = this_center_column_index - NUM_HALF_COLUMNS
        this_last_column_index = this_center_column_index + NUM_HALF_COLUMNS

        this_u_wind_matrix_m_s01 = example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY][i, ..., u_wind_index]
        this_v_wind_matrix_m_s01 = example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY][i, ..., v_wind_index]
        this_cos_matrix = narr_rotation_cos_matrix[
            this_first_row_index:(this_last_row_index + 1),
            this_first_column_index:(this_last_column_index + 1)
        ]
        this_sin_matrix = narr_rotation_sin_matrix[
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
        plotting_utils.plot_parallels(
            basemap_object=basemap_object, axes_object=axes_object,
            bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
            parallel_spacing_deg=PARALLEL_SPACING_DEG)
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            bottom_left_lng_deg=0., upper_right_lng_deg=360.,
            meridian_spacing_deg=MERIDIAN_SPACING_DEG)

        this_thetaw_matrix_kelvins = example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY][i, ..., thetaw_index]

        this_min_value = numpy.percentile(
            this_thetaw_matrix_kelvins, 100. - thetaw_max_colour_percentile)
        this_max_value = numpy.percentile(
            this_thetaw_matrix_kelvins, thetaw_max_colour_percentile)

        nwp_plotting.plot_subgrid(
            field_matrix=this_thetaw_matrix_kelvins,
            model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
            basemap_object=basemap_object, colour_map=thetaw_colour_map_object,
            min_value_in_colour_map=this_min_value,
            max_value_in_colour_map=this_max_value,
            first_row_in_full_grid=this_first_row_index,
            first_column_in_full_grid=this_first_column_index)

        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=axes_object,
            values_to_colour=this_thetaw_matrix_kelvins,
            colour_map=thetaw_colour_map_object, colour_min=this_min_value,
            colour_max=this_max_value, orientation='horizontal',
            extend_min=True, extend_max=True, fraction_of_axis_length=0.8)

        nwp_plotting.plot_wind_barbs_on_subgrid(
            u_wind_matrix_m_s01=this_u_wind_matrix_m_s01,
            v_wind_matrix_m_s01=this_v_wind_matrix_m_s01,
            model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
            basemap_object=basemap_object,
            first_row_in_full_grid=this_first_row_index,
            first_column_in_full_grid=this_first_column_index,
            barb_length=WIND_BARB_LENGTH,
            empty_barb_radius=EMPTY_WIND_BARB_RADIUS,
            colour_map=WIND_COLOUR_MAP_OBJECT,
            colour_minimum_kt=MIN_COLOUR_WIND_SPEED_KT,
            colour_maximum_kt=MAX_COLOUR_WIND_SPEED_KT)

        this_front_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_front_line_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=example_dict[trainval_io.TARGET_TIMES_KEY][i]
        )

        print time_conversion.unix_sec_to_string(
            example_dict[trainval_io.TARGET_TIMES_KEY][i], '%Y-%m-%d-%H')

        this_polyline_table = fronts_io.read_polylines_from_file(
            this_front_file_name)
        this_num_fronts = len(this_polyline_table.index)

        for j in range(this_num_fronts):
            this_front_type_string = this_polyline_table[
                front_utils.FRONT_TYPE_COLUMN].values[j]

            if this_front_type_string == front_utils.WARM_FRONT_STRING_ID:
                this_colour = WARM_FRONT_COLOUR + 0.
            else:
                this_colour = COLD_FRONT_COLOUR + 0.

            front_plotting.plot_polyline(
                latitudes_deg=this_polyline_table[
                    front_utils.LATITUDES_COLUMN].values[j],
                longitudes_deg=this_polyline_table[
                    front_utils.LONGITUDES_COLUMN].values[j],
                basemap_object=basemap_object, axes_object=axes_object,
                front_type=this_front_type_string, line_colour=this_colour,
                line_width=4)

        this_output_file_name = '{0:s}/example{1:06d}.jpg'.format(
            output_dir_name, i)

        print 'Saving figure to: "{0:s}"...'.format(this_output_file_name)
        pyplot.savefig(this_output_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_front_line_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int),
        thetaw_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        thetaw_max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
