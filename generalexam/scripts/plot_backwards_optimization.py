"""Plots results of backwards optimization."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import \
    feature_optimization as backwards_opt
from generalexam.machine_learning import traditional_cnn
from generalexam.plotting import example_plotting

FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MIN_PERCENTILE_ARG_NAME = 'min_colour_percentile'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
SAME_COLOUR_MAP_ARG_NAME = 'same_cmap_for_all_predictors'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_opt.read_file`.')

NUM_PANEL_ROWS_HELP_STRING = (
    'Number of rows in each paneled figure.  Each figure corresponds to one '
    'example, and each panel corresponds to one predictor for the given '
    'example.  Two figures will be plotted for each example: the original '
    '(example itself) and optimized versions.  Default value for `{0:s}` is '
    'floor(sqrt(num_predictors)).'
).format(NUM_PANEL_ROWS_ARG_NAME)

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Each predictor will be plotted with the same colour '
    'map.  For example, if name "Oranges", the colour map used will be '
    '`pyplot.cm.Oranges`.  This argument supports only pyplot colour maps.')

MIN_PERCENTILE_HELP_STRING = (
    'Used to set minimum value for each colour map.  If `{0:s}` = 0, the '
    'minimum value in the colour map for example e and predictor p will be the '
    '[q]th percentile of predictor p in example e, where q = `{1:s}`.  If '
    '`{0:s}` = 1, the minimum value in the colour map for example e and '
    'predictor p will be the [q]th percentile of all predictors in example e, '
    'where q = `{1:s}`.'
).format(SAME_COLOUR_MAP_ARG_NAME, MIN_PERCENTILE_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Analogous to `{0:s}`, except for max value in each colour map.'
).format(MIN_PERCENTILE_ARG_NAME)

SAME_COLOUR_MAP_HELP_STRING = (
    'Boolean flag.  If 1, for each example e the colour map for all predictors '
    'will be the same.  If 0, the colour map will be different for each pair of'
    ' example and predictor.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='plasma',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILE_ARG_NAME, type=float, required=False, default=1.,
    help=MIN_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SAME_COLOUR_MAP_ARG_NAME, type=int, required=False, default=0,
    help=SAME_COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, num_panel_rows, colour_map_name,
         min_colour_percentile, max_colour_percentile,
         same_cmap_for_all_predictors, top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_panel_rows: Same.
    :param colour_map_name: Same.
    :param min_colour_percentile: Same.
    :param max_colour_percentile: Same.
    :param same_cmap_for_all_predictors: Same.
    :param top_output_dir_name: Same.
    """

    original_output_dir_name = '{0:s}/original'.format(top_output_dir_name)
    optimized_output_dir_name = '{0:s}/optimized'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=original_output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=optimized_output_dir_name)

    error_checking.assert_is_geq(min_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    error_checking.assert_is_greater(
        max_colour_percentile, min_colour_percentile)

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    this_list, backwards_opt_metadata_dict = (
        backwards_opt.read_file(input_file_name)
    )

    optimized_predictor_matrix = this_list[0]
    num_examples = optimized_predictor_matrix.shape[0]
    del this_list

    original_predictor_matrix = backwards_opt_metadata_dict[
        backwards_opt.INIT_FUNCTION_KEY][0]

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=backwards_opt_metadata_dict[
            backwards_opt.MODEL_FILE_NAME_KEY]
    )

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metafile_name)

    narr_predictor_names = model_metadata_dict[
        traditional_cnn.NARR_PREDICTOR_NAMES_KEY]
    num_predictors = len(narr_predictor_names)
    if num_panel_rows <= 0:
        num_panel_rows = int(numpy.floor(numpy.sqrt(num_predictors)))

    for i in range(num_examples):
        this_combined_matrix = numpy.concatenate(
            (original_predictor_matrix[i, ...],
             optimized_predictor_matrix[i, ...]),
            axis=0)

        if same_cmap_for_all_predictors:
            this_min_colour_value = numpy.percentile(
                this_combined_matrix, min_colour_percentile)
            this_max_colour_value = numpy.percentile(
                this_combined_matrix, max_colour_percentile)

            this_min_cval_by_predictor = numpy.full(
                num_predictors, this_min_colour_value)
            this_max_cval_by_predictor = numpy.full(
                num_predictors, this_max_colour_value)
        else:
            this_min_cval_by_predictor = numpy.full(num_predictors, numpy.nan)
            this_max_cval_by_predictor = this_min_cval_by_predictor + 0.

            for k in range(num_predictors):
                this_min_cval_by_predictor[k] = numpy.percentile(
                    this_combined_matrix[..., k], min_colour_percentile)
                this_max_cval_by_predictor[k] = numpy.percentile(
                    this_combined_matrix[..., k], max_colour_percentile)

        this_figure_file_name = '{0:s}/example{1:d}_original.jpg'.format(
            original_output_dir_name, i)

        example_plotting.plot_many_2d_grids(
            predictor_matrix_3d=original_predictor_matrix[i, ...],
            predictor_names=narr_predictor_names, num_panel_rows=num_panel_rows,
            cmap_object_by_predictor=[colour_map_object] * num_predictors,
            min_colour_value_by_predictor=this_min_cval_by_predictor,
            max_colour_value_by_predictor=this_max_cval_by_predictor)

        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_figure_file_name = '{0:s}/example{1:d}_optimized.jpg'.format(
            optimized_output_dir_name, i)

        example_plotting.plot_many_2d_grids(
            predictor_matrix_3d=optimized_predictor_matrix[i, ...],
            predictor_names=narr_predictor_names, num_panel_rows=num_panel_rows,
            cmap_object_by_predictor=[colour_map_object] * num_predictors,
            min_colour_value_by_predictor=this_min_cval_by_predictor,
            max_colour_value_by_predictor=this_max_cval_by_predictor)

        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        min_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_PERCENTILE_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        same_cmap_for_all_predictors=bool(getattr(
            INPUT_ARG_OBJECT, SAME_COLOUR_MAP_ARG_NAME)),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
