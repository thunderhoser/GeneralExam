"""Plots results of backwards optimization."""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import backwards_optimization as backwards_opt
from generalexam.scripts import plot_input_examples_simple as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PLOT_BARBS_ARG_NAME = plot_examples.PLOT_BARBS_ARG_NAME
WIND_BARB_COLOUR_ARG_NAME = plot_examples.WIND_BARB_COLOUR_ARG_NAME
NUM_PANEL_ROWS_ARG_NAME = plot_examples.NUM_PANEL_ROWS_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME
MAIN_FONT_SIZE_ARG_NAME = plot_examples.MAIN_FONT_SIZE_ARG_NAME
TITLE_FONT_SIZE_ARG_NAME = plot_examples.TITLE_FONT_SIZE_ARG_NAME
CBAR_FONT_SIZE_ARG_NAME = plot_examples.CBAR_FONT_SIZE_ARG_NAME
RESOLUTION_ARG_NAME = plot_examples.RESOLUTION_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `backwards_optimization.read_file`).')

COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for differences (must be accepted by '
    '`pyplot.get_cmap`).')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BARBS_ARG_NAME, type=int, required=True, default=1,
    help=plot_examples.PLOT_BARBS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_BARB_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=plot_examples.DEFAULT_WIND_BARB_COLOUR * 255,
    help=plot_examples.WIND_BARB_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=plot_examples.NUM_PANEL_ROWS_HELP_STRING)

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


def _run(input_file_name, diff_colour_map_name, plot_wind_as_barbs,
         wind_barb_colour, num_panel_rows, add_titles, colour_bar_length,
         main_font_size, title_font_size, colour_bar_font_size,
         figure_resolution_dpi, top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param diff_colour_map_name: Same.
    :param plot_wind_as_barbs: Same.
    :param wind_barb_colour: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param top_output_dir_name: Same.
    """

    if num_panel_rows <= 0:
        num_panel_rows = None

    before_dir_name = '{0:s}/before_optimization'.format(top_output_dir_name)
    after_dir_name = '{0:s}/after_optimization'.format(top_output_dir_name)
    difference_dir_name = '{0:s}/difference'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    # Read pre- and post-optimized examples.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    bwo_dictionary, pmm_flag = backwards_opt.read_file(input_file_name)

    if pmm_flag:
        before_matrix = numpy.expand_dims(
            bwo_dictionary[backwards_opt.MEAN_INPUT_MATRIX_KEY], axis=0
        )
        after_matrix = numpy.expand_dims(
            bwo_dictionary[backwards_opt.MEAN_OUTPUT_MATRIX_KEY], axis=0
        )
    else:
        before_matrix = bwo_dictionary.pop(backwards_opt.INPUT_MATRIX_KEY)
        after_matrix = bwo_dictionary.pop(backwards_opt.OUTPUT_MATRIX_KEY)

    # Read model metadata.
    model_file_name = bwo_dictionary[backwards_opt.MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    before_example_dict = {
        examples_io.PREDICTOR_MATRIX_KEY: before_matrix + 0.,
        examples_io.PREDICTOR_NAMES_KEY:
            model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        examples_io.PRESSURE_LEVELS_KEY:
            model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    }

    if not pmm_flag:
        valid_times_unix_sec, row_indices, column_indices = (
            examples_io.example_ids_to_metadata(
                bwo_dictionary[backwards_opt.EXAMPLE_IDS_KEY]
            )
        )

        before_example_dict[examples_io.VALID_TIMES_KEY] = valid_times_unix_sec
        before_example_dict[examples_io.ROW_INDICES_KEY] = row_indices
        before_example_dict[examples_io.COLUMN_INDICES_KEY] = column_indices

    after_example_dict = copy.deepcopy(before_example_dict)
    diff_example_dict = copy.deepcopy(before_example_dict)
    after_example_dict[examples_io.PREDICTOR_MATRIX_KEY] = after_matrix + 0.
    diff_example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
        after_matrix - before_matrix
    )

    wind_colour_map_object = pyplot.get_cmap(
        plot_examples.DEFAULT_WIND_CMAP_NAME)
    non_wind_colour_map_object = pyplot.get_cmap(
        plot_examples.DEFAULT_NON_WIND_CMAP_NAME)

    print(SEPARATOR_STRING)

    if pmm_flag:
        this_dict = plot_examples.plot_composite_example(
            example_dict=before_example_dict, plot_diffs=False,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            wind_colour_map_object=wind_colour_map_object,
            non_wind_colour_map_object=non_wind_colour_map_object,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size)

        before_figure_object = this_dict[plot_examples.FIGURE_OBJECT_KEY]
        before_file_name = '{0:s}/before_optimization_pmm.jpg'.format(
            before_dir_name)

        print('Saving figure to: "{0:s}"...'.format(before_file_name))
        before_figure_object.savefig(
            before_file_name, dpi=figure_resolution_dpi,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(before_figure_object)

        this_dict = plot_examples.plot_composite_example(
            example_dict=after_example_dict, plot_diffs=False,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            wind_colour_map_object=wind_colour_map_object,
            non_wind_colour_map_object=non_wind_colour_map_object,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size)

        after_figure_object = this_dict[plot_examples.FIGURE_OBJECT_KEY]
        after_file_name = '{0:s}/after_optimization_pmm.jpg'.format(
            after_dir_name)

        print('Saving figure to: "{0:s}"...'.format(after_file_name))
        after_figure_object.savefig(
            after_file_name, dpi=figure_resolution_dpi,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(after_figure_object)

        this_dict = plot_examples.plot_composite_example(
            example_dict=diff_example_dict, plot_diffs=True,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            wind_colour_map_object=diff_colour_map_object,
            non_wind_colour_map_object=diff_colour_map_object,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size)

        diff_figure_object = this_dict[plot_examples.FIGURE_OBJECT_KEY]
        diff_file_name = '{0:s}/difference_pmm.jpg'.format(difference_dir_name)

        print('Saving figure to: "{0:s}"...'.format(diff_file_name))
        diff_figure_object.savefig(
            diff_file_name, dpi=figure_resolution_dpi,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(diff_figure_object)
    else:
        plot_examples.plot_real_examples(
            example_dict=before_example_dict,
            output_dir_name=before_dir_name, plot_diffs=False,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi)

        print(SEPARATOR_STRING)

        plot_examples.plot_real_examples(
            example_dict=after_example_dict,
            output_dir_name=after_dir_name, plot_diffs=False,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi)

        print(SEPARATOR_STRING)

        plot_examples.plot_real_examples(
            example_dict=diff_example_dict,
            output_dir_name=difference_dir_name, plot_diffs=True,
            plot_wind_as_barbs=plot_wind_as_barbs,
            wind_barb_colour=wind_barb_colour,
            wind_colour_map_name=diff_colour_map_name,
            non_wind_colour_map_name=diff_colour_map_name,
            num_panel_rows=num_panel_rows, add_titles=add_titles,
            colour_bar_length=colour_bar_length,
            main_font_size=main_font_size, title_font_size=title_font_size,
            colour_bar_font_size=colour_bar_font_size,
            figure_resolution_dpi=figure_resolution_dpi)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        plot_wind_as_barbs=bool(getattr(INPUT_ARG_OBJECT, PLOT_BARBS_ARG_NAME)),
        wind_barb_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, WIND_BARB_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        main_font_size=getattr(INPUT_ARG_OBJECT, MAIN_FONT_SIZE_ARG_NAME),
        title_font_size=getattr(INPUT_ARG_OBJECT, TITLE_FONT_SIZE_ARG_NAME),
        colour_bar_font_size=getattr(INPUT_ARG_OBJECT, CBAR_FONT_SIZE_ARG_NAME),
        figure_resolution_dpi=getattr(INPUT_ARG_OBJECT, RESOLUTION_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
