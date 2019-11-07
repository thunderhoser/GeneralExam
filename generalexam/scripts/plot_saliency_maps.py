"""Plots saliency maps."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import saliency_plotting
from generalexam.machine_learning import cnn
from generalexam.machine_learning import saliency_maps
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
SALIENCY_CMAP_ARG_NAME = 'saliency_colour_map_name'
MAX_SALIENCY_ARG_NAME = 'max_saliency'
HALF_NUM_CONTOURS_ARG_NAME = 'half_num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
WIND_CMAP_ARG_NAME = plot_examples.WIND_CMAP_ARG_NAME
NON_WIND_CMAP_ARG_NAME = plot_examples.NON_WIND_CMAP_ARG_NAME
NUM_PANEL_ROWS_ARG_NAME = plot_examples.NUM_PANEL_ROWS_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME
MAIN_FONT_SIZE_ARG_NAME = plot_examples.MAIN_FONT_SIZE_ARG_NAME
TITLE_FONT_SIZE_ARG_NAME = plot_examples.TITLE_FONT_SIZE_ARG_NAME
CBAR_FONT_SIZE_ARG_NAME = plot_examples.CBAR_FONT_SIZE_ARG_NAME
RESOLUTION_ARG_NAME = plot_examples.RESOLUTION_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

SALIENCY_CMAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_SALIENCY_HELP_STRING = (
    'Max saliency value in colour scheme.  Keep in mind that the colour scheme '
    'encodes *absolute* value, with positive values in solid contours and '
    'negative values in dashed contours.')

HALF_NUM_CONTOURS_HELP_STRING = (
    'Number of contours on each side of zero (positive and negative).')

SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth saliency maps, make this non-positive.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_CMAP_ARG_NAME, type=str, required=False, default='binary',
    help=SALIENCY_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SALIENCY_ARG_NAME, type=float, required=False,
    default=1.25, help=MAX_SALIENCY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=2., help=SMOOTHING_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_CMAP_ARG_NAME, type=str, required=False,
    default=plot_examples.DEFAULT_WIND_CMAP_NAME,
    help=plot_examples.WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NON_WIND_CMAP_ARG_NAME, type=str, required=False,
    default=plot_examples.DEFAULT_NON_WIND_CMAP_NAME,
    help=plot_examples.NON_WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=plot_examples.NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_CBAR_LENGTH,
    help=plot_examples.CBAR_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_MAIN_FONT_SIZE,
    help=plot_examples.MAIN_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TITLE_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_TITLE_FONT_SIZE,
    help=plot_examples.TITLE_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_CBAR_FONT_SIZE,
    help=plot_examples.CBAR_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RESOLUTION_ARG_NAME, type=int, required=False,
    default=plot_examples.DEFAULT_RESOLUTION_DPI,
    help=plot_examples.RESOLUTION_HELP_STRING)


def _plot_saliency_one_example(
        saliency_matrix, colour_map_object, max_saliency, half_num_contours,
        colour_bar_length, colour_bar_font_size, figure_resolution_dpi,
        figure_object, axes_object_matrix, output_dir_name,
        example_id_string=None):
    """Plots saliency map for one example.

    m = number of rows in example grid
    n = number of columns in example grid
    C = number of predictors

    :param saliency_matrix: m-by-n-by-C numpy array of saliency values.
    :param colour_map_object: Colour scheme for saliency (instance of
        `matplotlib.pyplot.cm`).
    :param max_saliency: Max value in colour scheme for saliency.
    :param half_num_contours: See documentation at top of file.
    :param colour_bar_length: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object_matrix: Will plot on these axes (2-D numpy array of
        `matplotlib.axes._subplots.AxesSubplot` instances).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param example_id_string: Example ID.  If plotting a composite for many
        examples, leave this as None.
    """

    print(numpy.min(saliency_matrix))
    print(numpy.max(saliency_matrix))

    saliency_plotting.plot_many_2d_grids_with_contours(
        saliency_matrix_3d=saliency_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=colour_map_object,
        max_absolute_contour_level=max_saliency,
        contour_interval=max_saliency / half_num_contours,
        row_major=True)

    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    if num_panel_rows >= num_panel_columns:
        orientation_string = 'horizontal'
    else:
        orientation_string = 'vertical'

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object_matrix[-1, ...],
        data_matrix=saliency_matrix,
        colour_map_object=colour_map_object, min_value=0.,
        max_value=max_saliency, orientation_string=orientation_string,
        fraction_of_axis_length=1., padding=0.1,
        extend_min=False, extend_max=True, font_size=colour_bar_font_size)

    if orientation_string == 'horizontal':
        tick_values = colour_bar_object.ax.get_xticks()
        colour_bar_object.ax.set_xticks(tick_values)
        colour_bar_object.ax.set_xticklabels(tick_values)
    else:
        tick_values = colour_bar_object.ax.get_yticks()
        colour_bar_object.ax.set_yticks(tick_values)
        colour_bar_object.ax.set_yticklabels(tick_values)

    output_file_name = '{0:s}/saliency_{1:s}.jpg'.format(
        output_dir_name,
        'pmm' if example_id_string is None else example_id_string
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(output_file_name, dpi=figure_resolution_dpi,
                          pad_inches=0, bbox_inches='tight')
    pyplot.close(figure_object)


def _run(input_file_name, saliency_colour_map_name, max_saliency,
         half_num_contours, smoothing_radius_grid_cells, output_dir_name,
         wind_colour_map_name, non_wind_colour_map_name, num_panel_rows,
         add_titles, colour_bar_length, main_font_size, title_font_size,
         colour_bar_font_size, figure_resolution_dpi):
    """Plots saliency maps.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param saliency_colour_map_name: Same.
    :param max_saliency: Same.
    :param half_num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    """

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None
    if num_panel_rows <= 0:
        num_panel_rows = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    saliency_colour_map_object = pyplot.cm.get_cmap(saliency_colour_map_name)
    wind_colour_map_object = pyplot.cm.get_cmap(wind_colour_map_name)
    non_wind_colour_map_object = pyplot.cm.get_cmap(non_wind_colour_map_name)

    error_checking.assert_is_greater(max_saliency, 0.)
    error_checking.assert_is_geq(half_num_contours, 5)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    saliency_dict, pmm_flag = saliency_maps.read_file(input_file_name)

    if pmm_flag:
        predictor_matrix = numpy.expand_dims(
            saliency_dict[saliency_maps.MEAN_PREDICTOR_MATRIX_KEY], axis=0
        )
        saliency_matrix = numpy.expand_dims(
            saliency_dict[saliency_maps.MEAN_SALIENCY_MATRIX_KEY], axis=0
        )
    else:
        predictor_matrix = saliency_dict.pop(saliency_maps.PREDICTOR_MATRIX_KEY)
        saliency_matrix = saliency_dict.pop(saliency_maps.SALIENCY_MATRIX_KEY)

    if smoothing_radius_grid_cells is not None:
        print((
            'Smoothing saliency maps with Gaussian filter (e-folding radius of '
            '{0:.1f} grid cells)...'
        ).format(
            smoothing_radius_grid_cells
        ))

        num_channels = saliency_matrix.shape[-1]

        for k in range(num_channels):
            saliency_matrix[..., k] = general_utils.apply_gaussian_filter(
                input_matrix=saliency_matrix[..., k],
                e_folding_radius_grid_cells=smoothing_radius_grid_cells)

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    example_dict = {
        examples_io.PREDICTOR_MATRIX_KEY: predictor_matrix,
        examples_io.PREDICTOR_NAMES_KEY:
            model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        examples_io.PRESSURE_LEVELS_KEY:
            model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    }

    if not pmm_flag:
        valid_times_unix_sec, row_indices, column_indices = (
            examples_io.example_ids_to_metadata(
                saliency_dict[saliency_maps.EXAMPLE_IDS_KEY]
            )
        )

        example_dict[examples_io.VALID_TIMES_KEY] = valid_times_unix_sec
        example_dict[examples_io.ROW_INDICES_KEY] = row_indices
        example_dict[examples_io.COLUMN_INDICES_KEY] = column_indices

    num_examples = example_dict[examples_io.PREDICTOR_MATRIX_KEY].shape[0]
    narr_cosine_matrix = None
    narr_sine_matrix = None

    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if pmm_flag:
            this_dict = plot_examples.plot_composite_example(
                example_dict=example_dict, plot_wind_as_barbs=False,
                wind_colour_map_object=wind_colour_map_object,
                non_wind_colour_map_object=non_wind_colour_map_object,
                num_panel_rows=num_panel_rows, add_titles=add_titles,
                colour_bar_length=colour_bar_length,
                main_font_size=main_font_size, title_font_size=title_font_size,
                colour_bar_font_size=colour_bar_font_size)

            this_example_string = None
        else:
            this_dict = plot_examples.plot_real_example(
                example_dict=example_dict, example_index=i,
                plot_wind_as_barbs=False,
                wind_colour_map_object=wind_colour_map_object,
                non_wind_colour_map_object=non_wind_colour_map_object,
                num_panel_rows=num_panel_rows, add_titles=add_titles,
                colour_bar_length=colour_bar_length,
                main_font_size=main_font_size, title_font_size=title_font_size,
                colour_bar_font_size=colour_bar_font_size,
                narr_cosine_matrix=narr_cosine_matrix,
                narr_sine_matrix=narr_sine_matrix)

            this_example_string = saliency_dict[
                saliency_maps.EXAMPLE_IDS_KEY][i]

            if narr_cosine_matrix is None:
                narr_cosine_matrix = this_dict[plot_examples.NARR_COSINES_KEY]
                narr_sine_matrix = this_dict[plot_examples.NARR_SINES_KEY]

        _plot_saliency_one_example(
            saliency_matrix=saliency_matrix[i, ...],
            colour_map_object=saliency_colour_map_object,
            max_saliency=max_saliency, half_num_contours=half_num_contours,
            colour_bar_length=colour_bar_length,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi,
            figure_object=this_dict[plot_examples.FIGURE_OBJECT_KEY],
            axes_object_matrix=this_dict[plot_examples.AXES_OBJECTS_KEY],
            output_dir_name=output_dir_name,
            example_id_string=this_example_string)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        saliency_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_CMAP_ARG_NAME),
        max_saliency=getattr(INPUT_ARG_OBJECT, MAX_SALIENCY_ARG_NAME),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        wind_colour_map_name=getattr(INPUT_ARG_OBJECT, WIND_CMAP_ARG_NAME),
        non_wind_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NON_WIND_CMAP_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        main_font_size=getattr(INPUT_ARG_OBJECT, MAIN_FONT_SIZE_ARG_NAME),
        title_font_size=getattr(INPUT_ARG_OBJECT, TITLE_FONT_SIZE_ARG_NAME),
        colour_bar_font_size=getattr(INPUT_ARG_OBJECT, CBAR_FONT_SIZE_ARG_NAME),
        figure_resolution_dpi=getattr(INPUT_ARG_OBJECT, RESOLUTION_ARG_NAME)
    )
