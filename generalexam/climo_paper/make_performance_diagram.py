"""Plots performance diagram for final model on testing data."""

import argparse
import numpy
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import model_eval_plotting
from generalexam.ge_utils import neigh_evaluation

METRES_TO_KM = 0.001
NEIGH_DISTANCES_METRES = numpy.linspace(50000, 200000, num=4, dtype=int)

FONT_SIZE = 24
MARKER_TYPE = 'o'
MARKER_SIZE = 8
MARKER_EDGE_WIDTH = 0
MARKER_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

ERROR_BAR_CAP_SIZE = 4
ERROR_BAR_LINE_WIDTH = 2

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_dir_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files.  Will be read by '
    '`neigh_evaluation.read_nonspatial_results`'
)
CONFIDENCE_LEVEL_HELP_STRING = 'Confidence level for error bars.'
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _plot_one_neigh_distance(evaluation_dict, confidence_level,
                             neigh_distance_metres, axes_object):
    """Plots marker on performance diagram for one neighbourhood distance.

    :param evaluation_dict: Dictionary returned by
        `neigh_evaluation.read_nonspatial_results`.
    :param confidence_level: See documentation at top of file.
    :param neigh_distance_metres: Neighbourhood distance.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    pod_values = numpy.array([
        neigh_evaluation.get_pod(d) for d in
        evaluation_dict[neigh_evaluation.BINARY_TABLES_KEY]
    ])

    mean_pod = numpy.mean(pod_values)
    min_pod = numpy.percentile(pod_values, 50. * (1 - confidence_level))
    max_pod = numpy.percentile(pod_values, 50. * (1 + confidence_level))

    success_ratios = numpy.array([
        1. - neigh_evaluation.get_far(d) for d in
        evaluation_dict[neigh_evaluation.BINARY_TABLES_KEY]
    ])

    mean_success_ratio = numpy.mean(success_ratios)
    min_success_ratio = numpy.percentile(
        success_ratios, 50. * (1 - confidence_level)
    )
    max_success_ratio = numpy.percentile(
        success_ratios, 50. * (1 + confidence_level)
    )

    success_ratio_errors = numpy.array([
        mean_success_ratio - min_success_ratio,
        max_success_ratio - mean_success_ratio
    ])

    pod_errors = numpy.array([
        mean_pod - min_pod,
        max_pod - mean_pod
    ])

    success_ratio_errors = numpy.reshape(
        success_ratio_errors, (success_ratio_errors.size, 1)
    )
    pod_errors = numpy.reshape(pod_errors, (pod_errors.size, 1))

    success_ratio_errors = numpy.maximum(success_ratio_errors, 0.01)
    pod_errors = numpy.maximum(pod_errors, 0.01)

    axes_object.plot(
        mean_success_ratio, mean_pod, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR
    )

    axes_object.errorbar(
        mean_success_ratio, mean_pod,
        xerr=success_ratio_errors, yerr=pod_errors,
        ecolor=MARKER_COLOUR, elinewidth=ERROR_BAR_LINE_WIDTH,
        capsize=ERROR_BAR_CAP_SIZE, capthick=ERROR_BAR_LINE_WIDTH, zorder=1e6
    )

    label_string = '{0:d} km'.format(
        int(numpy.round(neigh_distance_metres * METRES_TO_KM))
    )

    axes_object.text(
        mean_success_ratio + 0.01, mean_pod - 0.01, label_string,
        fontsize=FONT_SIZE, fontweight='bold', color=MARKER_COLOUR,
        horizontalalignment='left', verticalalignment='top', zorder=1e6
    )


def _run(input_dir_name, confidence_level, output_file_name):
    """Plots performance diagram for final model on testing data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    model_eval_plotting.plot_performance_diagram(
        axes_object=axes_object, pod_by_threshold=numpy.full(2, numpy.nan),
        success_ratio_by_threshold=numpy.full(2, numpy.nan)
    )

    num_neigh_distances = len(NEIGH_DISTANCES_METRES)

    for i in range(num_neigh_distances):
        this_file_name = (
            '{0:s}/evaluation_neigh-distance-metres={1:06d}.p'
        ).format(input_dir_name, NEIGH_DISTANCES_METRES[i])

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_evaluation_dict = neigh_evaluation.read_nonspatial_results(
            this_file_name
        )

        _plot_one_neigh_distance(
            evaluation_dict=this_evaluation_dict,
            confidence_level=confidence_level,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            axes_object=axes_object
        )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
