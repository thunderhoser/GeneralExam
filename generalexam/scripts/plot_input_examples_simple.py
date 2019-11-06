"""Plots one or more examples."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

PLOT_WIND_KEY = 'plot_wind'
FIRST_NARR_ROW_KEY = 'first_narr_row'
LAST_NARR_ROW_KEY = 'last_narr_row'
FIRST_NARR_COLUMN_KEY = 'first_narr_column'
LAST_NARR_COLUMN_KEY = 'last_narr_column'

PREDICTOR_NAME_TO_FANCY = {
    predictor_utils.TEMPERATURE_NAME: r'Temperature ($^{\circ}$C)',
    predictor_utils.HEIGHT_NAME: 'Height (m)',
    predictor_utils.PRESSURE_NAME: 'Pressure (mb)',
    predictor_utils.DEWPOINT_NAME: r'Dewpoint ($^{\circ}$C)',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: r'Specific humidity (g kg$^{-1}$)',
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: r'$u$-wind (m s$^{-1}$)',
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: r'$v$-wind (m s$^{-1}$)',
    predictor_utils.WET_BULB_THETA_NAME: r'Wet-bulb theta ($^{\circ}$C)'
}

PREDICTOR_NAME_TO_CONV_FACTOR = {
    predictor_utils.TEMPERATURE_NAME: -273.15,
    predictor_utils.HEIGHT_NAME: 1.,
    predictor_utils.PRESSURE_NAME: 0.01,
    predictor_utils.DEWPOINT_NAME: -273.15,
    predictor_utils.SPECIFIC_HUMIDITY_NAME: 1000.,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: 1.,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: 1.,
    predictor_utils.WET_BULB_THETA_NAME: -273.15
}

WIND_NAMES = [
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]

FONT_SIZE = 25
MAX_COLOUR_PERCENTILE = 99.
DEFAULT_WIND_BARB_COLOUR = numpy.full(3, 0.)

WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LARGE_BORDER_WIDTH = 2
SMALL_BORDER_WIDTH = 1
BORDER_COLOUR = numpy.full(3, 152. / 255)
FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
PLOT_BARBS_ARG_NAME = 'plot_wind_as_barbs'
WIND_BARB_COLOUR_ARG_NAME = 'wind_barb_colour'
WIND_CMAP_ARG_NAME = 'wind_colour_map_name'
NON_WIND_CMAP_ARG_NAME = 'non_wind_colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `learning_examples_io.read_file`).'
)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to be plotted (will be drawn randomly from `{0:s}`).  '
    'To plot all examples, leave this argument alone.'
).format(EXAMPLE_FILE_ARG_NAME)

MODEL_FILE_HELP_STRING = (
    'Path to model (will be read by `cnn.read_model`).  Before plotting, '
    'examples will be pre-processed in the same way as for training the model, '
    'except that predictors will not be normalized.')

PLOT_BARBS_HELP_STRING = (
    'Boolean flag.  If 1, will plot wind as barbs.  If 0, will plot wind as two'
    ' colour maps (one for u-wind, one for v-wind).')

WIND_BARB_COLOUR_HELP_STRING = (
    '[used only if `{0:s}` = 1] Wind-barb colour.  Must be length-3 numpy array'
    ' with values in range 0...255.'
).format(PLOT_BARBS_ARG_NAME)

WIND_CMAP_HELP_STRING = (
    '[used only if `{0:s}` = 0] Name of colour map (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).  Will be used for both u-wind and v-wind.'
).format(PLOT_BARBS_ARG_NAME)

NON_WIND_CMAP_HELP_STRING = (
    'Name of colour map for all fields other than wind (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here (one paneled figure '
    'for each example).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BARBS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_BARBS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_BARB_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_WIND_BARB_COLOUR * 255, help=WIND_BARB_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_CMAP_ARG_NAME, type=str, required=False, default='seismic',
    help=WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NON_WIND_CMAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=NON_WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_units(example_dict, example_index):
    """Converts units of predictors.

    :param example_dict: Dictionary returned by
        `learning_examples_io.read_file`.
    :param example_index: Will convert units for the [i]th example, where
        i = `example_index`.
    :return: example_dict: Same but with units converted for the [i]th example.
    """

    predictor_names = example_dict[examples_io.PREDICTOR_NAMES_KEY]
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][
        example_index, ...]

    for k in range(len(predictor_names)):
        this_conv_factor = PREDICTOR_NAME_TO_CONV_FACTOR[predictor_names[k]]

        if this_conv_factor < 0:
            predictor_matrix[..., k] = (
                predictor_matrix[..., k] + this_conv_factor
            )
        else:
            predictor_matrix[..., k] = (
                predictor_matrix[..., k] * this_conv_factor
            )

    example_dict[examples_io.PREDICTOR_MATRIX_KEY][example_index, ...] = (
        predictor_matrix
    )

    return example_dict


def _rotate_winds(example_dict, example_index, narr_cosine_matrix,
                  narr_sine_matrix):
    """Rotates winds from grid-relative to Earth-relative.

    M = number of rows in full NARR grid
    N = number of columns in full NARR grid
    m = number of rows in example grid
    n = number of columns in example grid

    :param example_dict: Dictionary returned by
        `learning_examples_io.read_file`.
    :param example_index: Will rotate winds for the [i]th example, where
        i = `example_index`.
    :param narr_cosine_matrix: M-by-N numpy array with cosines of rotation
        angles (used to convert wind from grid-relative to Earth-relative).
    :param narr_sine_matrix: Same but for sines.
    :return: example_dict: Same but with winds for the [i]th example rotated,
        where i = `example_index`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["plot_wind"]: Boolean flag.  If False, there is no wind data
        to plot.
    metadata_dict["first_narr_row"]: First row in NARR grid occupied by the
        given example.
    metadata_dict["last_narr_row"]: Last row.
    metadata_dict["first_narr_column"]: First column.
    metadata_dict["last_narr_column"]: Last column.
    """

    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][
        example_index, ...]

    num_half_rows = int(numpy.round(
        (predictor_matrix.shape[0] - 1) / 2
    ))
    first_narr_row = (
        example_dict[examples_io.ROW_INDICES_KEY][example_index] - num_half_rows
    )
    last_narr_row = (
        example_dict[examples_io.ROW_INDICES_KEY][example_index] + num_half_rows
    )

    num_half_columns = int(numpy.round(
        (predictor_matrix.shape[1] - 1) / 2
    ))
    first_narr_column = (
        example_dict[examples_io.COLUMN_INDICES_KEY][example_index] -
        num_half_columns
    )
    last_narr_column = (
        example_dict[examples_io.COLUMN_INDICES_KEY][example_index] +
        num_half_columns
    )

    predictor_names = numpy.array(example_dict[examples_io.PREDICTOR_NAMES_KEY])
    plot_wind = (
        predictor_utils.U_WIND_GRID_RELATIVE_NAME in predictor_names or
        predictor_utils.V_WIND_GRID_RELATIVE_NAME in predictor_names
    )

    metadata_dict = {
        PLOT_WIND_KEY: plot_wind,
        FIRST_NARR_ROW_KEY: first_narr_row,
        LAST_NARR_ROW_KEY: last_narr_row,
        FIRST_NARR_COLUMN_KEY: first_narr_column,
        LAST_NARR_COLUMN_KEY: last_narr_column
    }

    if not plot_wind:
        return example_dict, metadata_dict

    cosine_matrix = narr_cosine_matrix[
        first_narr_row:(last_narr_row + 1),
        first_narr_column:(last_narr_column + 1)
    ]
    sine_matrix = narr_sine_matrix[
        first_narr_row:(last_narr_row + 1),
        first_narr_column:(last_narr_column + 1)
    ]

    pressure_levels_mb = example_dict[examples_io.PRESSURE_LEVELS_KEY]

    for this_pressure_level_mb in numpy.unique(pressure_levels_mb):
        this_u_wind_index = numpy.where(numpy.logical_and(
            pressure_levels_mb == this_pressure_level_mb,
            predictor_names == predictor_utils.U_WIND_GRID_RELATIVE_NAME
        ))[0][0]

        this_v_wind_index = numpy.where(numpy.logical_and(
            pressure_levels_mb == this_pressure_level_mb,
            predictor_names == predictor_utils.V_WIND_GRID_RELATIVE_NAME
        ))[0][0]

        (predictor_matrix[..., this_u_wind_index],
         predictor_matrix[..., this_v_wind_index]
        ) = nwp_model_utils.rotate_winds_to_earth_relative(
            u_winds_grid_relative_m_s01=
            predictor_matrix[..., this_u_wind_index],
            v_winds_grid_relative_m_s01=
            predictor_matrix[..., this_v_wind_index],
            rotation_angle_cosines=cosine_matrix,
            rotation_angle_sines=sine_matrix
        )

    example_dict[examples_io.PREDICTOR_MATRIX_KEY][example_index, ...] = (
        predictor_matrix
    )

    return example_dict, metadata_dict


def plot_one_example(
        example_dict, example_index, plot_wind_as_barbs,
        non_wind_colour_map_object, output_dir_name, wind_barb_colour=None,
        wind_colour_map_object=None, narr_cosine_matrix=None,
        narr_sine_matrix=None):
    """Plots one example.

    M = number of rows in full NARR grid
    N = number of columns in full NARR grid

    :param example_dict: Dictionary returned by
        `learning_examples_io.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param plot_wind_as_barbs: See documentation at top of file.
    :param non_wind_colour_map_object: Same.
    :param output_dir_name: Same.
    :param wind_barb_colour: [used only if `plot_wind_as_barbs == True`]
        Wind-barb colour as length-3 numpy array.
    :param wind_colour_map_object: [used only if `plot_wind_as_barbs == False`]
        Colour map for u- and v-wind fields.
    :param narr_cosine_matrix: M-by-N numpy array with cosines of rotation
        angles (used to convert wind from grid-relative to Earth-relative).
        If this is None, it will be created on the fly.
    :param narr_sine_matrix: Same but for sines.
    :return: narr_cosine_matrix: See input doc.
    :return: narr_sine_matrix: See input doc.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    num_examples = len(example_dict[examples_io.VALID_TIMES_KEY])
    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)
    error_checking.assert_is_less_than(example_index, num_examples)

    error_checking.assert_is_boolean(plot_wind_as_barbs)

    if narr_cosine_matrix is None:
        narr_lat_matrix_deg, narr_lng_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_221GRID)
        )

        narr_cosine_matrix, narr_sine_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=narr_lat_matrix_deg,
                longitudes_deg=narr_lng_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME)
        )

    num_narr_rows, num_narr_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_221GRID)

    these_expected_dim = numpy.array(
        [num_narr_rows, num_narr_columns], dtype=int
    )

    error_checking.assert_is_geq_numpy_array(narr_cosine_matrix, -1.)
    error_checking.assert_is_leq_numpy_array(narr_cosine_matrix, 1.)
    error_checking.assert_is_numpy_array(
        narr_cosine_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(narr_sine_matrix, -1.)
    error_checking.assert_is_leq_numpy_array(narr_sine_matrix, 1.)
    error_checking.assert_is_numpy_array(
        narr_sine_matrix, exact_dimensions=these_expected_dim)

    # Do housekeeping.
    example_dict = _convert_units(example_dict=example_dict,
                                  example_index=example_index)

    example_dict, metadata_dict = _rotate_winds(
        example_dict=example_dict, example_index=example_index,
        narr_cosine_matrix=narr_cosine_matrix,
        narr_sine_matrix=narr_sine_matrix)

    first_narr_row = metadata_dict[FIRST_NARR_ROW_KEY]
    last_narr_row = metadata_dict[LAST_NARR_ROW_KEY]
    first_narr_column = metadata_dict[FIRST_NARR_COLUMN_KEY]
    last_narr_column = metadata_dict[LAST_NARR_COLUMN_KEY]

    basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_id=nwp_model_utils.NAME_OF_221GRID,
        first_row_in_full_grid=first_narr_row,
        last_row_in_full_grid=last_narr_row,
        first_column_in_full_grid=first_narr_column,
        last_column_in_full_grid=last_narr_column,
        resolution_string='i'
    )[-1]

    plot_wind = metadata_dict[PLOT_WIND_KEY]
    plot_wind_as_barbs = plot_wind_as_barbs and plot_wind

    if plot_wind_as_barbs:
        wind_barb_cmap_object = matplotlib.colors.ListedColormap(
            [wind_barb_colour]
        )
        wind_barb_cmap_object.set_under(wind_barb_colour)
        wind_barb_cmap_object.set_over(wind_barb_colour)

    predictor_names = numpy.array(example_dict[examples_io.PREDICTOR_NAMES_KEY])
    pressure_levels_mb = numpy.round(
        example_dict[examples_io.PRESSURE_LEVELS_KEY]
    ).astype(int)

    num_predictors = len(predictor_names)
    num_unique_pressure_levels = len(numpy.unique(pressure_levels_mb))

    num_panels_desired = (
        num_predictors -
        2 * num_unique_pressure_levels * int(plot_wind_as_barbs)
    )
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels_desired)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels_desired) / num_panel_rows
    ))
    num_panels = num_panel_rows * num_panel_columns

    # Do plotting.
    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True)

    panel_index_linear = -1
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][
        example_index, ...]

    for k in range(num_predictors):
        if predictor_names[k] in WIND_NAMES and plot_wind_as_barbs:
            continue

        panel_index_linear += 1
        this_panel_row, this_panel_column = numpy.unravel_index(
            panel_index_linear, axes_object_matrix.shape)
        this_axes_object = axes_object_matrix[this_panel_row, this_panel_column]

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=LARGE_BORDER_WIDTH)
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=LARGE_BORDER_WIDTH)
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=SMALL_BORDER_WIDTH)

        if this_panel_column == 0:
            plotting_utils.plot_parallels(
                basemap_object=basemap_object, axes_object=this_axes_object,
                num_parallels=NUM_PARALLELS, font_size=FONT_SIZE, z_order=-1e20)

        if this_panel_row == num_panel_rows - 1:
            plotting_utils.plot_meridians(
                basemap_object=basemap_object, axes_object=this_axes_object,
                num_meridians=NUM_MERIDIANS, font_size=FONT_SIZE, z_order=-1e20)

        same_field_indices = numpy.where(
            predictor_names == predictor_names[k]
        )[0]

        if predictor_names[k] in WIND_NAMES:
            this_colour_map_object = wind_colour_map_object
            this_max_value = numpy.percentile(
                numpy.absolute(predictor_matrix[..., same_field_indices]),
                MAX_COLOUR_PERCENTILE
            )
            this_min_value = -1 * this_max_value
        else:
            this_colour_map_object = non_wind_colour_map_object
            this_min_value = numpy.percentile(
                predictor_matrix[..., same_field_indices],
                100. - MAX_COLOUR_PERCENTILE
            )
            this_max_value = numpy.percentile(
                predictor_matrix[..., same_field_indices], MAX_COLOUR_PERCENTILE
            )

        nwp_plotting.plot_subgrid(
            field_matrix=predictor_matrix[..., k],
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_id=nwp_model_utils.NAME_OF_221GRID,
            axes_object=this_axes_object, basemap_object=basemap_object,
            colour_map_object=this_colour_map_object,
            min_colour_value=this_min_value, max_colour_value=this_max_value,
            first_row_in_full_grid=first_narr_row,
            first_column_in_full_grid=first_narr_column)

        this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=this_axes_object,
            data_matrix=predictor_matrix[..., k],
            colour_map_object=this_colour_map_object,
            min_value=this_min_value, max_value=this_max_value,
            orientation_string='vertical', font_size=FONT_SIZE,
            extend_min=True, extend_max=True, fraction_of_axis_length=0.8)

        these_tick_values = this_colour_bar_object.ax.get_xticks()
        this_colour_bar_object.ax.set_xticks(these_tick_values)
        this_colour_bar_object.ax.set_xticklabels(these_tick_values)

        if pressure_levels_mb[k] == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            this_title_string = 'Surface'
        else:
            this_title_string = '{0:d}-mb'.format(pressure_levels_mb[k])

        this_fancy_name = PREDICTOR_NAME_TO_FANCY[predictor_names[k]]
        this_title_string += ' {0:s}{1:s}'.format(
            this_fancy_name[0].lower(), this_fancy_name[1:]
        )

        this_axes_object.set_title(this_title_string, fontsize=FONT_SIZE)

        if not plot_wind_as_barbs:
            continue

        this_u_wind_index = numpy.where(numpy.logical_and(
            pressure_levels_mb == pressure_levels_mb[k],
            predictor_names == predictor_utils.U_WIND_GRID_RELATIVE_NAME
        ))[0][0]

        this_v_wind_index = numpy.where(numpy.logical_and(
            pressure_levels_mb == pressure_levels_mb[k],
            predictor_names == predictor_utils.V_WIND_GRID_RELATIVE_NAME
        ))[0][0]

        nwp_plotting.plot_wind_barbs_on_subgrid(
            u_wind_matrix_m_s01=predictor_matrix[..., this_u_wind_index],
            v_wind_matrix_m_s01=predictor_matrix[..., this_v_wind_index],
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_id=nwp_model_utils.NAME_OF_221GRID,
            axes_object=this_axes_object, basemap_object=basemap_object,
            first_row_in_full_grid=first_narr_row,
            first_column_in_full_grid=first_narr_column,
            plot_every_k_rows=2, plot_every_k_columns=2,
            barb_length=WIND_BARB_LENGTH,
            empty_barb_radius=EMPTY_WIND_BARB_RADIUS, fill_empty_barb=False,
            colour_map=wind_barb_cmap_object,
            colour_minimum_kt=-1., colour_maximum_kt=0.)

    for k in range(panel_index_linear + 1, num_panels):
        this_panel_row, this_panel_column = numpy.unravel_index(
            panel_index_linear, axes_object_matrix.shape)

        axes_object_matrix[this_panel_row, this_panel_column].axis('off')

    example_id_string = examples_io.create_example_id(
        valid_time_unix_sec=
        example_dict[examples_io.VALID_TIMES_KEY][example_index],
        row_index=example_dict[examples_io.ROW_INDICES_KEY][example_index],
        column_index=example_dict[examples_io.COLUMN_INDICES_KEY][example_index]
    )

    output_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                   pad_inches=0, bbox_inches='tight')
    pyplot.close(figure_object)

    return narr_cosine_matrix, narr_sine_matrix


def _run(example_file_name, num_examples, model_file_name, plot_wind_as_barbs,
         wind_barb_colour, wind_colour_map_name, non_wind_colour_map_name,
         output_dir_name):
    """Plots one or more examples.

    :param example_file_name: See documentation at top of file.
    :param num_examples: Same.
    :param model_file_name: Same.
    :param plot_wind_as_barbs: Same.
    :param wind_barb_colour: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param output_dir_name: Same.
    """

    non_wind_colour_map_object = pyplot.cm.get_cmap(non_wind_colour_map_name)
    if plot_wind_as_barbs:
        wind_colour_map_object = None
    else:
        wind_colour_map_object = pyplot.cm.get_cmap(wind_colour_map_name)

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    # Read predictors.
    print('Reading normalized predictors from: "{0:s}"...'.format(
        example_file_name
    ))

    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)
    predictor_names = model_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    pressure_levels_mb = model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]

    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=predictor_names,
        pressure_levels_to_keep_mb=pressure_levels_mb,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    num_examples_found = len(example_dict[examples_io.VALID_TIMES_KEY])

    if 0 < num_examples < num_examples_found:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)

        example_dict = examples_io.subset_examples(
            example_dict=example_dict, desired_indices=desired_indices)

    # Denormalize predictors.
    print('Denormalizing predictors...')

    # TODO(thunderhoser): All this nonsense should be in a separate method.
    normalization_type_string = example_dict[examples_io.NORMALIZATION_TYPE_KEY]

    normalization_dict = {
        ml_utils.MIN_VALUE_MATRIX_KEY: None,
        ml_utils.MAX_VALUE_MATRIX_KEY: None,
        ml_utils.MEAN_VALUE_MATRIX_KEY: None,
        ml_utils.STDEV_MATRIX_KEY: None
    }

    if normalization_type_string == ml_utils.Z_SCORE_STRING:
        normalization_dict[ml_utils.MEAN_VALUE_MATRIX_KEY] = example_dict[
            examples_io.FIRST_NORM_PARAM_KEY]

        normalization_dict[ml_utils.STDEV_MATRIX_KEY] = example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]
    else:
        normalization_dict[ml_utils.MIN_VALUE_MATRIX_KEY] = example_dict[
            examples_io.FIRST_NORM_PARAM_KEY]

        normalization_dict[ml_utils.MAX_VALUE_MATRIX_KEY] = example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]

    example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
        ml_utils.denormalize_predictors(
            predictor_matrix=example_dict[examples_io.PREDICTOR_MATRIX_KEY],
            normalization_dict=normalization_dict)
    )

    num_examples = len(example_dict[examples_io.VALID_TIMES_KEY])
    narr_cosine_matrix = None
    narr_sine_matrix = None

    for i in range(num_examples):
        narr_cosine_matrix, narr_sine_matrix = plot_one_example(
            example_dict=example_dict, example_index=i,
            plot_wind_as_barbs=plot_wind_as_barbs,
            non_wind_colour_map_object=non_wind_colour_map_object,
            output_dir_name=output_dir_name, wind_barb_colour=wind_barb_colour,
            wind_colour_map_object=wind_colour_map_object,
            narr_cosine_matrix=narr_cosine_matrix,
            narr_sine_matrix=narr_sine_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        plot_wind_as_barbs=bool(getattr(INPUT_ARG_OBJECT, PLOT_BARBS_ARG_NAME)),
        wind_barb_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, WIND_BARB_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        wind_colour_map_name=getattr(INPUT_ARG_OBJECT, WIND_CMAP_ARG_NAME),
        non_wind_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NON_WIND_CMAP_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
