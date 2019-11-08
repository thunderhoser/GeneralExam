"""Plots Grad-CAM output (guided and unguided class-activation maps)."""

import os
import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import gradcam
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_UNGUIDED_VALUE_LOG10 = -2.

INPUT_FILE_ARG_NAME = 'input_file_name'
GRADCAM_CMAP_ARG_NAME = 'gradcam_colour_map_name'
MAX_UNGUIDED_VALUE_ARG_NAME = 'max_unguided_value'
NUM_UNGUIDED_CONTOURS_ARG_NAME = 'num_unguided_contours'
MAX_GUIDED_VALUE_ARG_NAME = 'max_guided_value'
HALF_NUM_GUIDED_CONTOURS_ARG_NAME = 'half_num_guided_contours'
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
    'Path to input file.  Will be read by `gradcam.read_file`.')

GRADCAM_CMAP_HELP_STRING = (
    'Name of colour map for class activations.  The same colour map will be '
    'used for all predictors and examples.  This argument supports only pyplot '
    'colour maps (those accepted by `pyplot.get_cmap`).')

MAX_UNGUIDED_VALUE_HELP_STRING = (
    'Max value in colour scheme for unguided CAMs.  Keep in mind that unguided '
    'class activation >= 0 always.')

NUM_UNGUIDED_CONTOURS_HELP_STRING = 'Number of contours for unguided CAMs.'

MAX_GUIDED_VALUE_HELP_STRING = (
    'Max value in colour scheme for guided CAMs.  Keep in mind that the colour '
    'scheme encodes *absolute* value, with positive values in solid contours '
    'and negative values in dashed contours.')

HALF_NUM_GUIDED_CONTOURS_HELP_STRING = (
    'Number of contours on each side of zero for guided CAMs.')

SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth class-activation maps, make this non-positive.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_CMAP_ARG_NAME, type=str, required=False, default='gist_yarg',
    help=GRADCAM_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_UNGUIDED_VALUE_ARG_NAME, type=float, required=False,
    default=10 ** 1.5, help=MAX_UNGUIDED_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_UNGUIDED_CONTOURS_ARG_NAME, type=int, required=False,
    default=15, help=NUM_UNGUIDED_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_GUIDED_VALUE_ARG_NAME, type=float, required=False,
    default=0.5, help=MAX_GUIDED_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_GUIDED_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_GUIDED_CONTOURS_HELP_STRING)

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


def _plot_gradcam_one_example(
        colour_map_object, max_unguided_value, num_unguided_contours,
        max_guided_value, half_num_guided_contours, colour_bar_font_size,
        figure_resolution_dpi, figure_object, axes_object_matrix,
        output_dir_name, guided_class_activn_matrix, class_activn_matrix=None,
        example_id_string=None):
    """Plots class-activation map for one example.

    m = number of rows in example grid
    n = number of columns in example grid
    C = number of predictors

    :param colour_map_object: Colour scheme for class activation (instance of
        `matplotlib.pyplot.cm`).
    :param max_unguided_value: See documentation at top of file.
    :param num_unguided_contours: Same.
    :param max_guided_value: Same.
    :param half_num_guided_contours: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object_matrix: Will plot on these axes (2-D numpy array of
        `matplotlib.axes._subplots.AxesSubplot` instances).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param guided_class_activn_matrix:
        [used only if `class_activn_matrix is None`]
        m-by-n-by-C numpy array of guided class activations.
    :param class_activn_matrix: m-by-n numpy array of class activations.
    :param example_id_string: Example ID.  If plotting a composite for many
        examples, leave this as None.
    """

    this_array = figure_object.get_size_inches()
    figure_width_inches = this_array[0]
    figure_height_inches = this_array[1]

    class_activn_matrix_log10 = None
    max_unguided_value_log10 = None

    if class_activn_matrix is None:
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=guided_class_activn_matrix,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_guided_value,
            contour_interval=max_guided_value / half_num_guided_contours,
            row_major=True)
    else:
        num_channels = guided_class_activn_matrix.shape[-1]

        class_activn_matrix_log10 = numpy.log10(
            numpy.expand_dims(class_activn_matrix, axis=-1)
        )
        class_activn_matrix_log10 = numpy.repeat(
            class_activn_matrix_log10, repeats=num_channels, axis=-1)

        max_unguided_value_log10 = numpy.log10(max_unguided_value)
        contour_interval_log10 = (
            (max_unguided_value_log10 - MIN_UNGUIDED_VALUE_LOG10) /
            (num_unguided_contours - 1)
        )

        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=class_activn_matrix_log10,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            min_contour_level=MIN_UNGUIDED_VALUE_LOG10,
            max_contour_level=max_unguided_value_log10,
            contour_interval=contour_interval_log10, row_major=True)

    output_file_name = '{0:s}/gradcam_{1:s}.jpg'.format(
        output_dir_name,
        'pmm' if example_id_string is None else example_id_string
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(output_file_name, dpi=figure_resolution_dpi,
                          pad_inches=0, bbox_inches='tight')
    pyplot.close(figure_object)

    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    if num_panel_rows >= num_panel_columns:
        orientation_string = 'horizontal'
    else:
        orientation_string = 'vertical'

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    if class_activn_matrix is None:
        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=extra_axes_object,
            data_matrix=guided_class_activn_matrix,
            colour_map_object=colour_map_object, min_value=0.,
            max_value=max_guided_value, orientation_string=orientation_string,
            fraction_of_axis_length=1., extend_min=False, extend_max=True,
            font_size=colour_bar_font_size)

        colour_bar_object.set_label('Guided class activation',
                                    fontsize=colour_bar_font_size)
    else:
        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=extra_axes_object,
            data_matrix=class_activn_matrix_log10,
            colour_map_object=colour_map_object,
            min_value=MIN_UNGUIDED_VALUE_LOG10,
            max_value=max_unguided_value_log10,
            orientation_string=orientation_string,
            fraction_of_axis_length=1., extend_min=True, extend_max=True,
            font_size=colour_bar_font_size)

        colour_bar_object.set_label(r'Class activation (log$_{10}$)',
                                    fontsize=colour_bar_font_size)

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/gradcam_{1:s}_colour-bar.jpg'.format(
        output_dir_name,
        'pmm' if example_id_string is None else example_id_string
    )

    extra_figure_object.savefig(extra_file_name, dpi=figure_resolution_dpi,
                                pad_inches=0, bbox_inches='tight')
    pyplot.close(extra_figure_object)

    if orientation_string == 'horizontal':
        this_num_rows = 2
        this_num_columns = 1
    else:
        this_num_rows = 1
        this_num_columns = 2

    imagemagick_utils.concatenate_images(
        input_file_names=[output_file_name, extra_file_name],
        output_file_name=output_file_name,
        num_panel_rows=this_num_rows, num_panel_columns=this_num_columns,
        extra_args_string='-gravity Center')

    os.remove(extra_file_name)

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _smooth_maps(class_activn_matrix, guided_class_activn_matrix,
                 smoothing_radius_grid_cells):
    """Smooths class-activation maps via Gaussian filter.

    E = number of examples
    m = number of rows in example grid
    n = number of columns in example grid
    C = number of predictors

    :param class_activn_matrix: E-by-m-by-n numpy array of class activations.
    :param guided_class_activn_matrix: E-by-m-by-n-by-C numpy array of guided
        class activations.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: class_activn_matrix: Smoothed version of input.
    :return: guided_class_activn_matrix: Same.
    """

    print((
        'Smoothing saliency maps with Gaussian filter (e-folding radius of '
        '{0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_examples = guided_class_activn_matrix.shape[0]
    num_channels = guided_class_activn_matrix.shape[-1]

    for i in range(num_examples):
        class_activn_matrix[i, ...] = general_utils.apply_gaussian_filter(
            input_matrix=class_activn_matrix[i, ...],
            e_folding_radius_grid_cells=smoothing_radius_grid_cells
        )

        for k in range(num_channels):
            guided_class_activn_matrix[i, ..., k] = (
                general_utils.apply_gaussian_filter(
                    input_matrix=guided_class_activn_matrix[i, ..., k],
                    e_folding_radius_grid_cells=smoothing_radius_grid_cells
                )
            )

    return class_activn_matrix, guided_class_activn_matrix


def _run(input_file_name, gradcam_colour_map_name, max_unguided_value,
         num_unguided_contours, max_guided_value, half_num_guided_contours,
         smoothing_radius_grid_cells, wind_colour_map_name,
         non_wind_colour_map_name, num_panel_rows, add_titles,
         colour_bar_length, main_font_size, title_font_size,
         colour_bar_font_size, figure_resolution_dpi, top_output_dir_name):
    """Plots Grad-CAM output (guided and unguided class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param gradcam_colour_map_name: Same.
    :param max_unguided_value: Same.
    :param num_unguided_contours: Same.
    :param max_guided_value: Same.
    :param half_num_guided_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param top_output_dir_name: Same.
    """

    # Check and process input args.
    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None
    if num_panel_rows <= 0:
        num_panel_rows = None

    unguided_cam_dir_name = '{0:s}/main_gradcam'.format(top_output_dir_name)
    guided_cam_dir_name = '{0:s}/guided_gradcam'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=unguided_cam_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=guided_cam_dir_name)

    gradcam_colour_map_object = pyplot.cm.get_cmap(gradcam_colour_map_name)
    wind_colour_map_object = pyplot.cm.get_cmap(wind_colour_map_name)
    non_wind_colour_map_object = pyplot.cm.get_cmap(non_wind_colour_map_name)

    error_checking.assert_is_greater(max_unguided_value, 0.)
    error_checking.assert_is_greater(max_guided_value, 0.)
    error_checking.assert_is_geq(num_unguided_contours, 10)
    error_checking.assert_is_geq(half_num_guided_contours, 5)

    # Read class-activation maps.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gradcam_dict, pmm_flag = gradcam.read_file(input_file_name)

    if pmm_flag:
        predictor_matrix = numpy.expand_dims(
            gradcam_dict[gradcam.MEAN_PREDICTOR_MATRIX_KEY], axis=0
        )
        class_activn_matrix = numpy.expand_dims(
            gradcam_dict[gradcam.MEAN_ACTIVN_MATRIX_KEY], axis=0
        )
        guided_class_activn_matrix = numpy.expand_dims(
            gradcam_dict[gradcam.MEAN_GUIDED_ACTIVN_MATRIX_KEY], axis=0
        )
    else:
        predictor_matrix = gradcam_dict.pop(gradcam.PREDICTOR_MATRIX_KEY)
        class_activn_matrix = gradcam_dict.pop(gradcam.ACTIVN_MATRIX_KEY)
        guided_class_activn_matrix = gradcam_dict.pop(
            gradcam.GUIDED_ACTIVN_MATRIX_KEY)

    if smoothing_radius_grid_cells is not None:
        class_activn_matrix, guided_class_activn_matrix = _smooth_maps(
            class_activn_matrix=class_activn_matrix,
            guided_class_activn_matrix=guided_class_activn_matrix,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells)

    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
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
                gradcam_dict[gradcam.EXAMPLE_IDS_KEY]
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
        this_orig_predictor_matrix = copy.deepcopy(
            example_dict[examples_io.PREDICTOR_MATRIX_KEY][i, ...]
        )

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

            this_example_string = gradcam_dict[gradcam.EXAMPLE_IDS_KEY][i]

            if narr_cosine_matrix is None:
                narr_cosine_matrix = this_dict[plot_examples.NARR_COSINES_KEY]
                narr_sine_matrix = this_dict[plot_examples.NARR_SINES_KEY]

        example_dict[examples_io.PREDICTOR_MATRIX_KEY][i, ...] = (
            this_orig_predictor_matrix
        )

        _plot_gradcam_one_example(
            class_activn_matrix=class_activn_matrix[i, ...],
            guided_class_activn_matrix=guided_class_activn_matrix[i, ...],
            colour_map_object=gradcam_colour_map_object,
            max_unguided_value=max_unguided_value,
            num_unguided_contours=num_unguided_contours,
            max_guided_value=max_guided_value,
            half_num_guided_contours=half_num_guided_contours,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi,
            figure_object=this_dict[plot_examples.FIGURE_OBJECT_KEY],
            axes_object_matrix=this_dict[plot_examples.AXES_OBJECTS_KEY],
            output_dir_name=unguided_cam_dir_name,
            example_id_string=this_example_string)

        if pmm_flag:
            this_dict = plot_examples.plot_composite_example(
                example_dict=example_dict, plot_wind_as_barbs=False,
                wind_colour_map_object=wind_colour_map_object,
                non_wind_colour_map_object=non_wind_colour_map_object,
                num_panel_rows=num_panel_rows, add_titles=add_titles,
                colour_bar_length=colour_bar_length,
                main_font_size=main_font_size, title_font_size=title_font_size,
                colour_bar_font_size=colour_bar_font_size)
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

        _plot_gradcam_one_example(
            guided_class_activn_matrix=guided_class_activn_matrix[i, ...],
            colour_map_object=gradcam_colour_map_object,
            max_unguided_value=max_unguided_value,
            num_unguided_contours=num_unguided_contours,
            max_guided_value=max_guided_value,
            half_num_guided_contours=half_num_guided_contours,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi,
            figure_object=this_dict[plot_examples.FIGURE_OBJECT_KEY],
            axes_object_matrix=this_dict[plot_examples.AXES_OBJECTS_KEY],
            output_dir_name=guided_cam_dir_name,
            example_id_string=this_example_string)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        gradcam_colour_map_name=getattr(
            INPUT_ARG_OBJECT, GRADCAM_CMAP_ARG_NAME),
        max_unguided_value=getattr(
            INPUT_ARG_OBJECT, MAX_UNGUIDED_VALUE_ARG_NAME),
        num_unguided_contours=getattr(
            INPUT_ARG_OBJECT, NUM_UNGUIDED_CONTOURS_ARG_NAME),
        max_guided_value=getattr(INPUT_ARG_OBJECT, MAX_GUIDED_VALUE_ARG_NAME),
        half_num_guided_contours=getattr(
            INPUT_ARG_OBJECT, HALF_NUM_GUIDED_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
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
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )