"""Plots saliency maps."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import cnn
from generalexam.machine_learning import saliency_maps
from generalexam.plotting import example_plotting
from generalexam.plotting import saliency_plotting

FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
PREDICTOR_CMAP_ARG_NAME = 'predictor_colour_map_name'
MIN_PREDICTOR_PRCTILE_ARG_NAME = 'min_colour_prctile_for_predictors'
MAX_PREDICTOR_PRCTILE_ARG_NAME = 'max_colour_prctile_for_predictors'
SALIENCY_CMAP_ARG_NAME = 'saliency_colour_map_name'
MAX_SALIENCY_PRCTILE_ARG_NAME = 'max_colour_prctile_for_saliency'
LINE_WIDTH_ARG_NAME = 'saliency_contour_line_width'
NUM_CONTOURS_ARG_NAME = 'num_saliency_contours'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

PREDICTOR_CMAP_HELP_STRING = (
    'Name of colour map.  Each predictor will be plotted with the same colour '
    'map.  For example, if name is "Oranges", the colour map used will be '
    '`pyplot.cm.Oranges`.  This argument supports only pyplot colour maps.')

MIN_PREDICTOR_PRCTILE_HELP_STRING = (
    'Used to set minimum value for each predictor map.  The minimum value for '
    'example e and predictor p will be the [q]th percentile of '
    'predictor p in example e, where q = `{0:s}`.'
).format(MIN_PREDICTOR_PRCTILE_ARG_NAME)

MAX_PREDICTOR_PRCTILE_HELP_STRING = (
    'Analogous to `{0:s}`, except for max value in each predictor map.'
).format(MIN_PREDICTOR_PRCTILE_ARG_NAME)

SALIENCY_CMAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_SALIENCY_PRCTILE_HELP_STRING = (
    'Used to set max absolute value for each saliency map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'saliency values for example e, where q = `{0:s}`.'
).format(MAX_SALIENCY_PRCTILE_ARG_NAME)

LINE_WIDTH_HELP_STRING = 'Width of saliency contours.'

NUM_CONTOURS_HELP_STRING = (
    'Number of saliency contours.  Contour levels will be evenly spaced between'
    ' the min and max values, determined by `{0:s}`.  The min contour value '
    'will be the negative of the max contour value, so that contour levels (and'
    ' the colour scheme) are zero-centered.'
).format(MAX_SALIENCY_PRCTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_CMAP_ARG_NAME, type=str, required=False, default='plasma',
    help=PREDICTOR_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PREDICTOR_PRCTILE_ARG_NAME, type=float, required=False,
    default=1., help=MIN_PREDICTOR_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PREDICTOR_PRCTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PREDICTOR_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_CMAP_ARG_NAME, type=str, required=False, default='Greys',
    help=SALIENCY_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SALIENCY_PRCTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_SALIENCY_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LINE_WIDTH_ARG_NAME, type=float, required=False,
    default=saliency_plotting.DEFAULT_LINE_WIDTH, help=LINE_WIDTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CONTOURS_ARG_NAME, type=int, required=False, default=21,
    help=NUM_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, predictor_colour_map_name,
         min_colour_prctile_for_predictors, max_colour_prctile_for_predictors,
         saliency_colour_map_name, max_colour_prctile_for_saliency,
         saliency_contour_line_width, num_saliency_contours, output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param predictor_colour_map_name: Same.
    :param min_colour_prctile_for_predictors: Same.
    :param max_colour_prctile_for_predictors: Same.
    :param saliency_colour_map_name: Same.
    :param max_colour_prctile_for_saliency: Same.
    :param saliency_contour_line_width: Same.
    :param num_saliency_contours: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    error_checking.assert_is_geq(min_colour_prctile_for_predictors, 0.)
    error_checking.assert_is_leq(max_colour_prctile_for_predictors, 100.)
    error_checking.assert_is_greater(max_colour_prctile_for_predictors,
                                     min_colour_prctile_for_predictors)

    error_checking.assert_is_geq(max_colour_prctile_for_saliency, 0.)
    error_checking.assert_is_leq(max_colour_prctile_for_saliency, 100.)

    error_checking.assert_is_geq(num_saliency_contours, 2)
    num_saliency_contours = 1 + int(
        number_rounding.floor_to_nearest(num_saliency_contours, 2)
    )
    half_num_saliency_contours = (num_saliency_contours - 1) / 2

    predictor_colour_map_object = pyplot.cm.get_cmap(predictor_colour_map_name)
    saliency_colour_map_object = pyplot.cm.get_cmap(saliency_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    predictor_matrix, saliency_matrix, saliency_metadata_dict = (
        saliency_maps.read_file(input_file_name)
    )

    model_metafile_name = cnn.find_metafile(
        model_file_name=saliency_metadata_dict[
            saliency_maps.MODEL_FILE_NAME_KEY]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    predictor_names = model_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    num_predictors = len(predictor_names)
    num_examples = predictor_matrix.shape[0]

    for i in range(num_examples):
        this_min_cval_by_predictor = numpy.full(num_predictors, numpy.nan)
        this_max_cval_by_predictor = this_min_cval_by_predictor + 0.

        for k in range(num_predictors):
            this_min_cval_by_predictor[k] = numpy.percentile(
                predictor_matrix[i, ..., k], min_colour_prctile_for_predictors)
            this_max_cval_by_predictor[k] = numpy.percentile(
                predictor_matrix[i, ..., k], max_colour_prctile_for_predictors)

        _, these_axes_objects = example_plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=predictor_matrix[i, ...],
            predictor_names=predictor_names,
            cmap_object_by_predictor=
            [predictor_colour_map_object] * num_predictors,
            min_colour_value_by_predictor=this_min_cval_by_predictor,
            max_colour_value_by_predictor=this_max_cval_by_predictor)

        this_max_abs_contour_level = numpy.percentile(
            numpy.absolute(saliency_matrix[i, ...]),
            max_colour_prctile_for_saliency)

        this_contour_interval = (
            this_max_abs_contour_level / half_num_saliency_contours
        )

        saliency_plotting.plot_many_2d_grids(
            saliency_matrix_3d=saliency_matrix[i, ...],
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=saliency_colour_map_object,
            max_absolute_contour_level=this_max_abs_contour_level,
            contour_interval=this_contour_interval,
            line_width=saliency_contour_line_width)

        this_figure_file_name = '{0:s}/example{1:06d}_saliency.jpg'.format(
            output_dir_name, i)

        print('Saving figure to: "{0:s}"...'.format(this_figure_file_name))
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        predictor_colour_map_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_CMAP_ARG_NAME),
        min_colour_prctile_for_predictors=getattr(
            INPUT_ARG_OBJECT, MIN_PREDICTOR_PRCTILE_ARG_NAME),
        max_colour_prctile_for_predictors=getattr(
            INPUT_ARG_OBJECT, MAX_PREDICTOR_PRCTILE_ARG_NAME),
        saliency_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_CMAP_ARG_NAME),
        max_colour_prctile_for_saliency=getattr(
            INPUT_ARG_OBJECT, MAX_SALIENCY_PRCTILE_ARG_NAME),
        saliency_contour_line_width=getattr(
            INPUT_ARG_OBJECT, LINE_WIDTH_ARG_NAME),
        num_saliency_contours=getattr(INPUT_ARG_OBJECT, NUM_CONTOURS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
