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
from generalexam.plotting import example_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PLOT_WIND_KEY = 'plot_wind'
FIRST_NARR_ROW_KEY = 'first_narr_row'
LAST_NARR_ROW_KEY = 'last_narr_row'
FIRST_NARR_COLUMN_KEY = 'first_narr_column'
LAST_NARR_COLUMN_KEY = 'last_narr_column'

FIGURE_OBJECT_KEY = 'figure_object'
AXES_OBJECTS_KEY = 'axes_object_matrix'
NARR_COSINES_KEY = 'narr_cosine_matrix'
NARR_SINES_KEY = 'narr_sine_matrix'

PREDICTOR_NAME_TO_FANCY = {
    predictor_utils.TEMPERATURE_NAME: r'Temperature ($^{\circ}$C)',
    predictor_utils.HEIGHT_NAME: 'Height (m)',
    predictor_utils.PRESSURE_NAME: 'Pressure (mb)',
    predictor_utils.DEWPOINT_NAME: r'Dewpoint ($^{\circ}$C)',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: r'Specific humidity (g kg$^{-1}$)',
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: r'$u$-wind (m s$^{-1}$)',
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: r'$v$-wind (m s$^{-1}$)',
    predictor_utils.WET_BULB_THETA_NAME: r'$\theta_w$ ($^{\circ}$C)'
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

CELSIUS_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.DEWPOINT_NAME,
    predictor_utils.WET_BULB_THETA_NAME
]

MAX_COLOUR_PERCENTILE = 99.
WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LARGE_BORDER_WIDTH = 2
SMALL_BORDER_WIDTH = 1
BORDER_COLOUR = numpy.full(3, 0.)

DEFAULT_MAIN_FONT_SIZE = 20
DEFAULT_TITLE_FONT_SIZE = 20
DEFAULT_CBAR_FONT_SIZE = 20
DEFAULT_CBAR_LENGTH = 0.8

DEFAULT_WIND_CMAP_NAME = 'seismic'
DEFAULT_NON_WIND_CMAP_NAME = 'YlOrRd'
DEFAULT_WIND_BARB_COLOUR = numpy.full(3, 152. / 255)

DEFAULT_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
MODEL_FILE_ARG_NAME = 'input_model_file_name'
PLOT_BARBS_ARG_NAME = 'plot_wind_as_barbs'
WIND_BARB_COLOUR_ARG_NAME = 'wind_barb_colour'
WIND_CMAP_ARG_NAME = 'wind_colour_map_name'
NON_WIND_CMAP_ARG_NAME = 'non_wind_colour_map_name'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
ADD_TITLES_ARG_NAME = 'add_titles'
CBAR_LENGTH_ARG_NAME = 'colour_bar_length'
MAIN_FONT_SIZE_ARG_NAME = 'main_font_size'
TITLE_FONT_SIZE_ARG_NAME = 'title_font_size'
CBAR_FONT_SIZE_ARG_NAME = 'colour_bar_font_size'
RESOLUTION_ARG_NAME = 'figure_resolution_dpi'
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

NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows in each figure.  If you want this to be auto '
    'determined, leave this argument alone.')

ADD_TITLES_HELP_STRING = (
    'Boolean flag.  If 1, will plot title above each figure.')

CBAR_LENGTH_HELP_STRING = 'Length of colour bars (as fraction of axis length).'

MAIN_FONT_SIZE_HELP_STRING = (
    'Main font size (for everything except title and colour bar).')

TITLE_FONT_SIZE_HELP_STRING = 'Font size for titles.'
CBAR_FONT_SIZE_HELP_STRING = 'Font size for colour bars.'
RESOLUTION_HELP_STRING = 'Resolution of saved images (dots per inch).'

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
    '--' + WIND_CMAP_ARG_NAME, type=str, required=False,
    default=DEFAULT_WIND_CMAP_NAME, help=WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NON_WIND_CMAP_ARG_NAME, type=str, required=False,
    default=DEFAULT_NON_WIND_CMAP_NAME, help=NON_WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False,
    default=DEFAULT_CBAR_LENGTH, help=CBAR_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=DEFAULT_MAIN_FONT_SIZE, help=MAIN_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TITLE_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=DEFAULT_TITLE_FONT_SIZE, help=TITLE_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=DEFAULT_CBAR_FONT_SIZE, help=CBAR_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RESOLUTION_ARG_NAME, type=int, required=False,
    default=DEFAULT_RESOLUTION_DPI, help=RESOLUTION_HELP_STRING)

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
    pressure_levels_mb = example_dict[examples_io.PRESSURE_LEVELS_KEY]
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][
        example_index, ...]

    for k in range(len(predictor_names)):
        is_geopotential = (
            predictor_names[k] == predictor_utils.HEIGHT_NAME and
            pressure_levels_mb[k] != predictor_utils.DUMMY_SURFACE_PRESSURE_MB
        )

        # TODO(thunderhoser): HACK.
        if is_geopotential:
            predictor_matrix[..., k] = predictor_matrix[..., k] / 9.80665
            continue

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


def plot_real_example(
        example_dict, example_index, plot_wind_as_barbs,
        non_wind_colour_map_object, plot_diffs=False, num_panel_rows=None,
        add_titles=True, one_cbar_per_panel=True,
        colour_bar_length=DEFAULT_CBAR_LENGTH,
        main_font_size=DEFAULT_MAIN_FONT_SIZE,
        title_font_size=DEFAULT_TITLE_FONT_SIZE,
        colour_bar_font_size=DEFAULT_CBAR_FONT_SIZE,
        wind_barb_colour=None, wind_colour_map_object=None,
        narr_cosine_matrix=None, narr_sine_matrix=None):
    """Plots one example.

    M = number of rows in full NARR grid
    N = number of columns in full NARR grid
    J = number of panel rows in figure
    K = number of panel columns in figure

    :param example_dict: Dictionary returned by
        `learning_examples_io.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param plot_wind_as_barbs: See documentation at top of file.
    :param non_wind_colour_map_object: Same.
    :param plot_diffs: Boolean flag.  If True, plotting differences rather than
        actual values.
    :param num_panel_rows: Number of panel rows in each figure.  If None, will
        be auto determined.
    :param add_titles: Boolean flag.  If True, will plot title at top of each
        figure.
    :param one_cbar_per_panel: Boolean flag.  If True, will plot one colour bar
        for each panel.  If False, one colour bar for the whole figure.
    :param colour_bar_length: Length of colour bars (as fraction of axis
        length).
    :param main_font_size: Font size for everything except titles and colour
        bars.
    :param title_font_size: Font size for titles.
    :param colour_bar_font_size: Font size for colour bars.
    :param wind_barb_colour: [used only if `plot_wind_as_barbs == True`]
        Wind-barb colour as length-3 numpy array.
    :param wind_colour_map_object: [used only if `plot_wind_as_barbs == False`]
        Colour map for u- and v-wind fields.
    :param narr_cosine_matrix: M-by-N numpy array with cosines of rotation
        angles (used to convert wind from grid-relative to Earth-relative).
        If this is None, it will be created on the fly.
    :param narr_sine_matrix: Same but for sines.
    :return: output_dict: Dictionary with the following keys.
    output_dict["figure_object"]: Figure handle (instance of
        `matplotlib.figure.Figure`).
    output_dict["axes_object_matrix"]: J-by-K numpy array of axes handles (each
        an instance of `matplotlib.axes._subplots.AxesSubplot`).
    output_dict["narr_cosine_matrix"]: See input doc.
    output_dict["narr_sine_matrix"]: See input doc.
    """

    # Check input args.
    error_checking.assert_is_boolean(plot_wind_as_barbs)
    error_checking.assert_is_boolean(plot_diffs)
    error_checking.assert_is_boolean(add_titles)
    error_checking.assert_is_boolean(one_cbar_per_panel)
    error_checking.assert_is_greater(main_font_size, 0.)
    error_checking.assert_is_greater(title_font_size, 0.)
    error_checking.assert_is_greater(colour_bar_font_size, 0.)

    num_examples = len(example_dict[examples_io.VALID_TIMES_KEY])
    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)
    error_checking.assert_is_less_than(example_index, num_examples)

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
        grid_name=nwp_model_utils.NAME_OF_221GRID
    )

    these_expected_dim = numpy.array(
        [num_narr_rows, num_narr_columns], dtype=int
    )

    error_checking.assert_is_geq_numpy_array(narr_cosine_matrix, -1.)
    error_checking.assert_is_leq_numpy_array(narr_cosine_matrix, 1.)
    error_checking.assert_is_numpy_array(
        narr_cosine_matrix, exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_geq_numpy_array(narr_sine_matrix, -1.)
    error_checking.assert_is_leq_numpy_array(narr_sine_matrix, 1.)
    error_checking.assert_is_numpy_array(
        narr_sine_matrix, exact_dimensions=these_expected_dim
    )

    # Do housekeeping.
    example_dict = _convert_units(
        example_dict=example_dict, example_index=example_index
    )
    example_dict, metadata_dict = _rotate_winds(
        example_dict=example_dict, example_index=example_index,
        narr_cosine_matrix=narr_cosine_matrix,
        narr_sine_matrix=narr_sine_matrix
    )

    first_narr_row = metadata_dict[FIRST_NARR_ROW_KEY]
    last_narr_row = metadata_dict[LAST_NARR_ROW_KEY]
    first_narr_column = metadata_dict[FIRST_NARR_COLUMN_KEY]
    last_narr_column = metadata_dict[LAST_NARR_COLUMN_KEY]

    temp_figure_object, _, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_id=nwp_model_utils.NAME_OF_221GRID,
        first_row_in_full_grid=first_narr_row,
        last_row_in_full_grid=last_narr_row,
        first_column_in_full_grid=first_narr_column,
        last_column_in_full_grid=last_narr_column,
        resolution_string='i'
    )

    pyplot.close(temp_figure_object)
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

    if num_panel_rows is None:
        num_panel_rows = int(numpy.round(
            numpy.sqrt(num_panels_desired)
        ))

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_leq(num_panel_rows, num_panels_desired)

    num_panel_columns = int(numpy.ceil(
        float(num_panels_desired) / num_panel_rows
    ))
    num_panels = num_panel_rows * num_panel_columns

    # Do plotting.
    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        horizontal_spacing=0.1, vertical_spacing=0.1,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True)

    panel_index_linear = -1
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][
        example_index, ...]

    if plot_diffs:
        celsius_flags = numpy.array([
            p in CELSIUS_NAMES
            for p in example_dict[examples_io.PREDICTOR_NAMES_KEY]
        ], dtype=bool)

        celsius_indices = numpy.where(celsius_flags)[0]
        predictor_matrix[..., celsius_indices] += 273.15

    for k in range(num_predictors):
        if predictor_names[k] in WIND_NAMES and plot_wind_as_barbs:
            continue

        panel_index_linear += 1
        i, j = numpy.unravel_index(panel_index_linear, axes_object_matrix.shape)
        this_axes_object = axes_object_matrix[i, j]

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=LARGE_BORDER_WIDTH)
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=LARGE_BORDER_WIDTH)
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=this_axes_object,
            line_colour=BORDER_COLOUR, line_width=SMALL_BORDER_WIDTH)

        if j == 0:
            plotting_utils.plot_parallels(
                basemap_object=basemap_object, axes_object=this_axes_object,
                num_parallels=NUM_PARALLELS, font_size=main_font_size)

        if i == num_panel_rows - 1:
            plotting_utils.plot_meridians(
                basemap_object=basemap_object, axes_object=this_axes_object,
                num_meridians=NUM_MERIDIANS, font_size=main_font_size)

        if predictor_names[k] == predictor_utils.HEIGHT_NAME:
            same_field_indices = numpy.array([k], dtype=int)
        else:
            same_field_indices = numpy.where(
                predictor_names == predictor_names[k]
            )[0]

        if predictor_names[k] in WIND_NAMES:
            this_colour_map_object = wind_colour_map_object
        else:
            this_colour_map_object = non_wind_colour_map_object

        if predictor_names[k] in WIND_NAMES or plot_diffs:
            this_max_value = numpy.percentile(
                numpy.absolute(predictor_matrix[..., same_field_indices]),
                MAX_COLOUR_PERCENTILE
            )
            this_min_value = -1 * this_max_value
        else:
            this_min_value = numpy.percentile(
                predictor_matrix[..., same_field_indices],
                100. - MAX_COLOUR_PERCENTILE
            )
            this_max_value = numpy.percentile(
                predictor_matrix[..., same_field_indices],
                MAX_COLOUR_PERCENTILE
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

        if one_cbar_per_panel:
            this_padding = 0.05 if i == num_panel_rows - 1 else None

            this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=this_axes_object,
                data_matrix=predictor_matrix[..., k],
                colour_map_object=this_colour_map_object,
                min_value=this_min_value, max_value=this_max_value,
                orientation_string='horizontal', padding=this_padding,
                font_size=colour_bar_font_size,
                extend_min=True, extend_max=True,
                fraction_of_axis_length=colour_bar_length
            )

            these_tick_values = this_colour_bar_object.get_ticks()
            these_tick_strings = [
                '{0:.1f}'.format(v) for v in these_tick_values
            ]

            this_colour_bar_object.set_ticks(these_tick_values)
            this_colour_bar_object.set_ticklabels(these_tick_strings)

        if pressure_levels_mb[k] == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            this_title_string = 'Surface'
        else:
            this_title_string = '{0:d}-mb'.format(pressure_levels_mb[k])

        this_fancy_name = PREDICTOR_NAME_TO_FANCY[predictor_names[k]]
        this_title_string += ' {0:s}{1:s}'.format(
            this_fancy_name[0].lower(), this_fancy_name[1:]
        )
        this_title_string = this_title_string.replace(
            'Surface height', 'Orographic height'
        )

        if add_titles:
            this_axes_object.set_title(
                this_title_string, fontsize=title_font_size
            )

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
            empty_barb_radius=EMPTY_WIND_BARB_RADIUS, fill_empty_barb=True,
            colour_map=wind_barb_cmap_object,
            colour_minimum_kt=-1., colour_maximum_kt=0.)

    while panel_index_linear < num_panels - 1:
        panel_index_linear += 1
        i, j = numpy.unravel_index(panel_index_linear, axes_object_matrix.shape)
        axes_object_matrix[i, j].axis('off')

    if one_cbar_per_panel:
        return {
            FIGURE_OBJECT_KEY: figure_object,
            AXES_OBJECTS_KEY: axes_object_matrix,
            NARR_COSINES_KEY: narr_cosine_matrix,
            NARR_SINES_KEY: narr_sine_matrix
        }

    if all([p in WIND_NAMES for p in predictor_names]):
        plot_diffs = True
        colour_map_object = wind_colour_map_object
    else:
        colour_map_object = non_wind_colour_map_object

    if plot_diffs:
        max_colour_value = numpy.percentile(
            numpy.absolute(predictor_matrix), MAX_COLOUR_PERCENTILE
        )
        min_colour_value = -1 * max_colour_value
    else:
        min_colour_value = numpy.percentile(
            predictor_matrix, 100. - MAX_COLOUR_PERCENTILE
        )
        max_colour_value = numpy.percentile(
            predictor_matrix, MAX_COLOUR_PERCENTILE
        )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object_matrix, data_matrix=predictor_matrix,
        colour_map_object=colour_map_object,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='horizontal', padding=0.05,
        font_size=colour_bar_font_size, extend_min=True, extend_max=True,
        fraction_of_axis_length=colour_bar_length
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return {
        FIGURE_OBJECT_KEY: figure_object,
        AXES_OBJECTS_KEY: axes_object_matrix,
        NARR_COSINES_KEY: narr_cosine_matrix,
        NARR_SINES_KEY: narr_sine_matrix
    }


def plot_composite_example(
        example_dict, plot_wind_as_barbs, non_wind_colour_map_object,
        plot_diffs=False, num_panel_rows=None, add_titles=True,
        one_cbar_per_panel=True, colour_bar_length=DEFAULT_CBAR_LENGTH,
        main_font_size=DEFAULT_MAIN_FONT_SIZE,
        title_font_size=DEFAULT_TITLE_FONT_SIZE,
        colour_bar_font_size=DEFAULT_CBAR_FONT_SIZE,
        wind_barb_colour=None, wind_colour_map_object=None):
    """Plots composite of many examples.

    m = number of rows in example grid
    n = number of columns in example grid
    C = number of predictors

    :param example_dict: Dictionary with the following keys.
    example_dict["predictor_matrix"]: 1-by-m-by-n-by-C numpy array with
        denormalized predictors.
    example_dict["narr_predictor_names"]: length-C list of predictor names.
    example_dict["pressure_levels_mb"]: length-C numpy array of pressure levels
        (millibars).

    :param plot_wind_as_barbs: See doc for `plot_real_example`.
    :param non_wind_colour_map_object: Same.
    :param plot_diffs: Boolean flag.  If True, plotting differences rather than
        actual values.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param one_cbar_per_panel: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param wind_barb_colour: Same.
    :param wind_colour_map_object: Same.
    :return: output_dict: Dictionary with the following keys.
    output_dict["figure_object"]: See doc for `plot_real_example`.
    output_dict["axes_object_matrix"]: Same.
    """

    # Check input args.
    error_checking.assert_is_boolean(plot_wind_as_barbs)
    error_checking.assert_is_boolean(plot_diffs)
    error_checking.assert_is_boolean(add_titles)
    error_checking.assert_is_boolean(one_cbar_per_panel)
    error_checking.assert_is_greater(main_font_size, 0.)
    error_checking.assert_is_greater(title_font_size, 0.)
    error_checking.assert_is_greater(colour_bar_font_size, 0.)

    # Do housekeeping.
    example_dict = _convert_units(example_dict=example_dict, example_index=0)

    predictor_names = numpy.array(example_dict[examples_io.PREDICTOR_NAMES_KEY])
    plot_wind = (
        predictor_utils.U_WIND_GRID_RELATIVE_NAME in predictor_names or
        predictor_utils.V_WIND_GRID_RELATIVE_NAME in predictor_names
    )
    plot_wind_as_barbs = plot_wind_as_barbs and plot_wind

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

    if num_panel_rows is None:
        num_panel_rows = int(numpy.round(
            numpy.sqrt(num_panels_desired)
        ))

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_leq(num_panel_rows, num_panels_desired)

    num_panel_columns = int(numpy.ceil(
        float(num_panels_desired) / num_panel_rows
    ))
    num_panels = num_panel_rows * num_panel_columns

    # Do plotting.
    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True)

    panel_index_linear = -1
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY][0, ...]

    if plot_diffs:
        celsius_flags = numpy.array([
            p in CELSIUS_NAMES
            for p in example_dict[examples_io.PREDICTOR_NAMES_KEY]
        ], dtype=bool)

        celsius_indices = numpy.where(celsius_flags)[0]
        predictor_matrix[..., celsius_indices] += 273.15

    for k in range(num_predictors):
        if predictor_names[k] in WIND_NAMES and plot_wind_as_barbs:
            continue

        panel_index_linear += 1
        i, j = numpy.unravel_index(panel_index_linear, axes_object_matrix.shape)
        this_axes_object = axes_object_matrix[i, j]

        if predictor_names[k] == predictor_utils.HEIGHT_NAME:
            same_field_indices = numpy.array([k], dtype=int)
        else:
            same_field_indices = numpy.where(
                predictor_names == predictor_names[k]
            )[0]

        if predictor_names[k] in WIND_NAMES:
            this_colour_map_object = wind_colour_map_object
        else:
            this_colour_map_object = non_wind_colour_map_object

        if predictor_names[k] in WIND_NAMES or plot_diffs:
            this_max_value = numpy.percentile(
                numpy.absolute(predictor_matrix[..., same_field_indices]),
                MAX_COLOUR_PERCENTILE
            )
            this_min_value = -1 * this_max_value
        else:
            this_min_value = numpy.percentile(
                predictor_matrix[..., same_field_indices],
                100. - MAX_COLOUR_PERCENTILE
            )
            this_max_value = numpy.percentile(
                predictor_matrix[..., same_field_indices],
                MAX_COLOUR_PERCENTILE
            )

        example_plotting.plot_2d_grid(
            predictor_matrix=predictor_matrix[..., k],
            colour_map_object=this_colour_map_object,
            min_colour_value=this_min_value, max_colour_value=this_max_value,
            axes_object=this_axes_object)

        this_axes_object.set_xlim(0, predictor_matrix.shape[-2])
        this_axes_object.set_ylim(0, predictor_matrix.shape[-3])

        if one_cbar_per_panel:
            this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=this_axes_object,
                data_matrix=predictor_matrix[..., k],
                colour_map_object=this_colour_map_object,
                min_value=this_min_value, max_value=this_max_value,
                orientation_string='horizontal', padding=0.01,
                font_size=colour_bar_font_size,
                extend_min=True, extend_max=True,
                fraction_of_axis_length=colour_bar_length
            )

            these_tick_values = this_colour_bar_object.get_ticks()
            these_tick_strings = [
                '{0:.1f}'.format(v) for v in these_tick_values
            ]

            this_colour_bar_object.set_ticks(these_tick_values)
            this_colour_bar_object.set_ticklabels(these_tick_strings)

        if pressure_levels_mb[k] == predictor_utils.DUMMY_SURFACE_PRESSURE_MB:
            this_title_string = 'Surface'
        else:
            this_title_string = '{0:d}-mb'.format(pressure_levels_mb[k])

        this_fancy_name = PREDICTOR_NAME_TO_FANCY[predictor_names[k]]
        this_title_string += ' {0:s}{1:s}'.format(
            this_fancy_name[0].lower(), this_fancy_name[1:]
        )
        this_title_string = this_title_string.replace(
            'Surface height', 'Orographic height'
        )

        if add_titles:
            this_axes_object.set_title(this_title_string,
                                       fontsize=title_font_size)

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

        example_plotting.plot_wind_barbs(
            u_wind_matrix_m_s01=predictor_matrix[..., this_u_wind_index],
            v_wind_matrix_m_s01=predictor_matrix[..., this_v_wind_index],
            axes_object=this_axes_object, plot_every=2,
            barb_colour=wind_barb_colour, barb_length=WIND_BARB_LENGTH,
            empty_barb_radius=EMPTY_WIND_BARB_RADIUS, fill_empty_barb=True
        )

    while panel_index_linear < num_panels - 1:
        panel_index_linear += 1
        i, j = numpy.unravel_index(panel_index_linear, axes_object_matrix.shape)
        axes_object_matrix[i, j].axis('off')

    if one_cbar_per_panel:
        return {
            FIGURE_OBJECT_KEY: figure_object,
            AXES_OBJECTS_KEY: axes_object_matrix
        }

    if all([p in WIND_NAMES for p in predictor_names]):
        plot_diffs = True
        colour_map_object = wind_colour_map_object
    else:
        colour_map_object = non_wind_colour_map_object

    if plot_diffs:
        max_colour_value = numpy.percentile(
            numpy.absolute(predictor_matrix), MAX_COLOUR_PERCENTILE
        )
        min_colour_value = -1 * max_colour_value
    else:
        min_colour_value = numpy.percentile(
            predictor_matrix, 100. - MAX_COLOUR_PERCENTILE
        )
        max_colour_value = numpy.percentile(
            predictor_matrix, MAX_COLOUR_PERCENTILE
        )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object_matrix, data_matrix=predictor_matrix,
        colour_map_object=colour_map_object,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='horizontal', padding=0.01,
        font_size=colour_bar_font_size, extend_min=True, extend_max=True,
        fraction_of_axis_length=colour_bar_length
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return {
        FIGURE_OBJECT_KEY: figure_object,
        AXES_OBJECTS_KEY: axes_object_matrix
    }


def plot_real_examples(
        example_dict, output_dir_name, num_examples_to_plot=None,
        plot_diffs=False, plot_wind_as_barbs=True,
        wind_barb_colour=DEFAULT_WIND_BARB_COLOUR,
        wind_colour_map_name=DEFAULT_WIND_CMAP_NAME,
        non_wind_colour_map_name=DEFAULT_NON_WIND_CMAP_NAME,
        num_panel_rows=None, add_titles=True, one_cbar_per_panel=True,
        colour_bar_length=DEFAULT_CBAR_LENGTH,
        main_font_size=DEFAULT_MAIN_FONT_SIZE,
        title_font_size=DEFAULT_TITLE_FONT_SIZE,
        colour_bar_font_size=DEFAULT_CBAR_FONT_SIZE,
        figure_resolution_dpi=DEFAULT_RESOLUTION_DPI):
    """Plots one or more examples.

    This method assumes that predictors in `example_dict` are already
    denormalized.

    :param example_dict: Dictionary returned by
        `learning_examples_io.read_file`.
    :param output_dir_name: See documentation at top of file.
    :param num_examples_to_plot: Number of examples to plot.  If None, will plot
        them all.
    :param plot_diffs: Boolean flag.  If True, plotting differences rather than
        actual values.
    :param plot_wind_as_barbs: See doc for `plot_real_example`.
    :param wind_barb_colour: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param one_cbar_per_panel: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    num_examples = len(example_dict[examples_io.VALID_TIMES_KEY])
    if num_examples_to_plot is None:
        num_examples_to_plot = num_examples + 0

    num_examples_to_plot = min([num_examples_to_plot, num_examples])

    non_wind_colour_map_object = pyplot.cm.get_cmap(non_wind_colour_map_name)
    if plot_wind_as_barbs:
        wind_colour_map_object = None
    else:
        wind_colour_map_object = pyplot.cm.get_cmap(wind_colour_map_name)

    example_id_strings = examples_io.create_example_ids(
        valid_times_unix_sec=example_dict[examples_io.VALID_TIMES_KEY],
        row_indices=example_dict[examples_io.ROW_INDICES_KEY],
        column_indices=example_dict[examples_io.COLUMN_INDICES_KEY]
    )

    narr_cosine_matrix = None
    narr_sine_matrix = None

    for i in range(num_examples_to_plot):
        this_dict = plot_real_example(
            example_dict=example_dict, example_index=i, plot_diffs=plot_diffs,
            plot_wind_as_barbs=plot_wind_as_barbs,
            non_wind_colour_map_object=non_wind_colour_map_object,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            one_cbar_per_panel=one_cbar_per_panel,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size,
            wind_barb_colour=wind_barb_colour,
            wind_colour_map_object=wind_colour_map_object,
            narr_cosine_matrix=narr_cosine_matrix,
            narr_sine_matrix=narr_sine_matrix)

        if narr_cosine_matrix is None:
            narr_cosine_matrix = this_dict[NARR_COSINES_KEY]
            narr_sine_matrix = this_dict[NARR_SINES_KEY]

        this_figure_object = this_dict[FIGURE_OBJECT_KEY]
        this_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, example_id_strings[i]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        this_figure_object.savefig(this_file_name, dpi=figure_resolution_dpi,
                                   pad_inches=0, bbox_inches='tight')
        pyplot.close(this_figure_object)


def _run(example_file_name, num_examples_to_plot, model_file_name,
         plot_wind_as_barbs, wind_barb_colour, wind_colour_map_name,
         non_wind_colour_map_name, num_panel_rows, add_titles,
         colour_bar_length, main_font_size, title_font_size,
         colour_bar_font_size, figure_resolution_dpi, output_dir_name):
    """Plots one or more examples.

    :param example_file_name: See documentation at top of file.
    :param num_examples_to_plot: Same.
    :param model_file_name: Same.
    :param plot_wind_as_barbs: Same.
    :param wind_barb_colour: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param output_dir_name: Same.
    """

    if num_panel_rows < 1:
        num_panel_rows = None
    if num_examples_to_plot < 1:
        num_examples_to_plot = None

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

    print('Denormalizing predictors...')
    example_dict = examples_io.denormalize_examples(example_dict)
    print(SEPARATOR_STRING)

    plot_real_examples(
        example_dict=example_dict, num_examples_to_plot=num_examples_to_plot,
        plot_wind_as_barbs=plot_wind_as_barbs,
        wind_barb_colour=wind_barb_colour,
        wind_colour_map_name=wind_colour_map_name,
        non_wind_colour_map_name=non_wind_colour_map_name,
        num_panel_rows=num_panel_rows, add_titles=add_titles,
        colour_bar_length=colour_bar_length, main_font_size=main_font_size,
        title_font_size=title_font_size,
        colour_bar_font_size=colour_bar_font_size,
        figure_resolution_dpi=figure_resolution_dpi,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples_to_plot=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        plot_wind_as_barbs=bool(getattr(INPUT_ARG_OBJECT, PLOT_BARBS_ARG_NAME)),
        wind_barb_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, WIND_BARB_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        wind_colour_map_name=getattr(INPUT_ARG_OBJECT, WIND_CMAP_ARG_NAME),
        non_wind_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NON_WIND_CMAP_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        main_font_size=getattr(INPUT_ARG_OBJECT, MAIN_FONT_SIZE_ARG_NAME),
        title_font_size=getattr(INPUT_ARG_OBJECT, TITLE_FONT_SIZE_ARG_NAME),
        colour_bar_font_size=getattr(INPUT_ARG_OBJECT, CBAR_FONT_SIZE_ARG_NAME),
        figure_resolution_dpi=getattr(INPUT_ARG_OBJECT, RESOLUTION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
