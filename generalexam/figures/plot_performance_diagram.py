"""Plots performance diagram on testing data."""

import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.plotting import model_eval_plotting
from generalexam.evaluation import object_based_evaluation as object_eval

CNN_METHOD_NAME = 'CNN'
NFA_METHOD_NAME = 'NFA'
CNN_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
NFA_COLOUR = numpy.array([255, 127, 0], dtype=float) / 255

NFA_DIRECTORY_NAME = (
    '/localdata/ryan.lagerquist/general_exam/fronts_baseline_experiment/testing'
)
CNN_DIRECTORY_NAME = (
    '/localdata/ryan.lagerquist/general_exam/paper_experiment_1000mb/'
    'quick_training/'
    'u-wind-grid-relative-m-s01_v-wind-grid-relative-m-s01_temperature-kelvins_'
    'specific-humidity-kg-kg01_init-num-filters=32_half-image-size-px=16_'
    'num-conv-layer-sets=3_dropout=0.50/testing'
)

MATCHING_DIST_BY_DATASET_KM = numpy.array([100, 250, 100, 250], dtype=int)
METHOD_NAME_BY_DATASET = [
    CNN_METHOD_NAME, CNN_METHOD_NAME, NFA_METHOD_NAME, NFA_METHOD_NAME
]

METHOD_NAME_TO_COLOUR = {
    CNN_METHOD_NAME: CNN_COLOUR,
    NFA_METHOD_NAME: NFA_COLOUR
}

METHOD_NAME_TO_DIRECTORY = {
    CNN_METHOD_NAME: CNN_DIRECTORY_NAME,
    NFA_METHOD_NAME: NFA_DIRECTORY_NAME
}

FONT_SIZE = 25
LINE_WIDTH = 2
ERROR_BAR_CAP_SIZE = 2

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _run():
    """Plots performance diagram on testing data.

    This is effectively the main method.
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    model_eval_plotting.plot_performance_diagram(
        axes_object=axes_object, pod_by_threshold=numpy.full(2, numpy.nan),
        success_ratio_by_threshold=numpy.full(2, numpy.nan)
    )

    num_datasets = len(METHOD_NAME_BY_DATASET)

    for i in range(num_datasets):
        this_file_name = '{0:s}/obe_{1:d}km_min.p'.format(
            METHOD_NAME_TO_DIRECTORY[METHOD_NAME_BY_DATASET[i]],
            MATCHING_DIST_BY_DATASET_KM[i]
        )

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_evaluation_dict = object_eval.read_evaluation_results(
            this_file_name)

        this_min_pod = this_evaluation_dict[object_eval.BINARY_POD_KEY]
        this_min_success_ratio = this_evaluation_dict[
            object_eval.BINARY_SUCCESS_RATIO_KEY]

        this_file_name = '{0:s}/obe_{1:d}km_max.p'.format(
            METHOD_NAME_TO_DIRECTORY[METHOD_NAME_BY_DATASET[i]],
            MATCHING_DIST_BY_DATASET_KM[i]
        )

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_evaluation_dict = object_eval.read_evaluation_results(
            this_file_name)

        this_max_pod = this_evaluation_dict[object_eval.BINARY_POD_KEY]
        this_max_success_ratio = this_evaluation_dict[
            object_eval.BINARY_SUCCESS_RATIO_KEY]

        this_file_name = '{0:s}/obe_{1:d}km_mean.p'.format(
            METHOD_NAME_TO_DIRECTORY[METHOD_NAME_BY_DATASET[i]],
            MATCHING_DIST_BY_DATASET_KM[i]
        )

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        this_evaluation_dict = object_eval.read_evaluation_results(
            this_file_name)

        this_mean_pod = this_evaluation_dict[object_eval.BINARY_POD_KEY]
        this_mean_success_ratio = this_evaluation_dict[
            object_eval.BINARY_SUCCESS_RATIO_KEY]

        these_x_errors = numpy.array([
            this_mean_success_ratio - this_min_success_ratio,
            this_max_success_ratio - this_mean_success_ratio
        ])

        these_y_errors = numpy.array([
            this_mean_pod - this_min_pod, this_max_pod - this_mean_pod
        ])

        these_x_errors = numpy.reshape(these_x_errors, (these_x_errors.size, 1))
        these_y_errors = numpy.reshape(these_y_errors, (these_y_errors.size, 1))

        axes_object.errorbar(
            this_mean_success_ratio, this_mean_pod, xerr=these_x_errors,
            yerr=these_y_errors,
            ecolor=METHOD_NAME_TO_COLOUR[METHOD_NAME_BY_DATASET[i]],
            elinewidth=LINE_WIDTH, capsize=ERROR_BAR_CAP_SIZE,
            capthick=LINE_WIDTH)

        this_label_string = '{0:s} ({1:d} km)'.format(
            METHOD_NAME_BY_DATASET[i], MATCHING_DIST_BY_DATASET_KM[i])

        if METHOD_NAME_BY_DATASET[i] == CNN_METHOD_NAME:
            this_y_coord = this_mean_pod + 0.01
            this_vertical_alignment_string = 'bottom'
        else:
            this_y_coord = this_mean_pod - 0.01
            this_vertical_alignment_string = 'top'

        axes_object.text(
            this_mean_success_ratio + 0.01, this_y_coord, this_label_string,
            fontsize=FONT_SIZE, fontweight='bold',
            color=METHOD_NAME_TO_COLOUR[METHOD_NAME_BY_DATASET[i]],
            horizontalalignment='left',
            verticalalignment=this_vertical_alignment_string, zorder=1e6)

    pyplot.show()


if __name__ == '__main__':
    _run()
