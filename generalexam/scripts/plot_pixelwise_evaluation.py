"""Plots results of pixelwise evaluation.

Specifically, this script plots the following figures:

- ROC curve
- performance diagram
- one attributes diagram for each class
"""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import model_eval_plotting
from generalexam.machine_learning import evaluation_utils

BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.5,
    'edgecolor': 'black',
    'linewidth': 2,
    'boxstyle': 'round'
}

MARKER_TYPE = '*'
MARKER_SIZE = 32
MARKER_EDGE_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `evaluation_utils.read_file`.'
)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Level for confidence interval (in range 0...1).  If input file does not '
    'contain bootstrapped scores, no confidence interval will be plotted, '
    'making this arg irrelevant.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_roc_curve(result_table_xarray, output_file_name,
                    confidence_level=None):
    """Plots ROC curve.

    :param result_table_xarray: xarray table produced by
        `evaluation_utils.run_evaluation`.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: See documentation at top of file.
    """

    pod_matrix = result_table_xarray[evaluation_utils.BINARY_POD_KEY].values
    pofd_matrix = result_table_xarray[evaluation_utils.BINARY_POFD_KEY].values
    num_bootstrap_reps = pod_matrix.shape[0]
    num_thresholds = pod_matrix.shape[1]

    print(
        result_table_xarray.coords[evaluation_utils.DETERMINIZN_THRESHOLD_DIM]
    )

    # TODO(thunderhoser): Allow for only one best threshold in file.
    best_threshold = (
        result_table_xarray[evaluation_utils.BEST_THRESHOLD_KEY].values[0]
    )
    # best_threshold_index = numpy.argmin(numpy.absolute(
    #     best_threshold -
    #     result_table_xarray[evaluation_utils.THRES].values[0]
    # ))
    best_threshold_index = 0

    auc_values = (
        result_table_xarray[evaluation_utils.AREA_UNDER_ROCC_KEY].values
    )

    if num_bootstrap_reps > 1:
        min_auc, max_auc = bootstrapping.get_confidence_interval(
            stat_values=auc_values, confidence_level=confidence_level)

        annotation_string = 'Area under curve = [{0:.3f}, {1:.3f}]'.format(
            min_auc, max_auc
        )
    else:
        mean_auc = numpy.nanmean(auc_values)
        annotation_string = 'Area under curve = {0:.3f}'.format(mean_auc)

    print(annotation_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan),
            model_eval.POFD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_thresholds):
            (ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (ci_top_dict[model_eval.POFD_BY_THRESHOLD_KEY][j],
             ci_bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pofd_matrix[:, j], confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pofd_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_roc_curve(
            axes_object=axes_object, ci_bottom_dict=ci_bottom_dict,
            ci_mean_dict=ci_mean_dict, ci_top_dict=ci_top_dict)

        best_x_coord = ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][
            best_threshold_index]
        best_y_coord = ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][
            best_threshold_index]
    else:
        model_eval_plotting.plot_roc_curve(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            pofd_by_threshold=pofd_matrix[0, :]
        )

        best_x_coord = pofd_matrix[0, best_threshold_index]
        best_y_coord = pod_matrix[0, best_threshold_index]

    print((
        'Best determinization threshold = {0:.3f} ... corresponding POD = '
        '{1:.3f} ... POFD = {2:.3f}'
    ).format(
        best_threshold, best_y_coord, best_x_coord
    ))

    marker_colour = model_eval_plotting.ROC_CURVE_COLOUR
    axes_object.plot(
        best_x_coord, best_y_coord, linestyle='None', marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
        markerfacecolor=marker_colour, markeredgecolor=marker_colour
    )

    axes_object.text(
        0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='right', verticalalignment='bottom',
        transform=axes_object.transAxes
    )

    axes_object.set_title('ROC curve')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)', y_coord_normalized=1.025
    )

    axes_object.set_aspect('equal')

    print('Saving ROC curve to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _plot_performance_diagram(result_table_xarray, output_file_name,
                              confidence_level=None):
    """Plots performance diagram.

    :param result_table_xarray: xarray table produced by
        `evaluation_utils.run_evaluation`.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: See documentation at top of file.
    """

    pod_matrix = result_table_xarray[evaluation_utils.BINARY_POD_KEY].values
    success_ratio_matrix = (
        1. - result_table_xarray[evaluation_utils.BINARY_FAR_KEY].values
    )
    num_bootstrap_reps = pod_matrix.shape[0]
    num_thresholds = pod_matrix.shape[1]

    best_threshold = (
        result_table_xarray[evaluation_utils.BEST_THRESHOLD_KEY].values[0]
    )
    # best_threshold_index = numpy.argmin(numpy.absolute(
    #     best_threshold -
    #     result_table_xarray[evaluation_utils.THRES].values[0]
    # ))
    best_threshold_index = 0

    aupd_values = (
        result_table_xarray[evaluation_utils.AREA_UNDER_PD_KEY].values
    )

    if num_bootstrap_reps > 1:
        min_aupd, max_aupd = bootstrapping.get_confidence_interval(
            stat_values=aupd_values, confidence_level=confidence_level)

        annotation_string = 'Area under curve = [{0:.3f}, {1:.3f}]'.format(
            min_aupd, max_aupd
        )
    else:
        mean_aupd = numpy.nanmean(aupd_values)
        annotation_string = 'Area under curve = {0:.3f}'.format(mean_aupd)

    print(annotation_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan),
            model_eval.SR_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_thresholds):
            (ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (ci_bottom_dict[model_eval.SR_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.SR_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=success_ratio_matrix[:, j],
                confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                success_ratio_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_performance_diagram(
            axes_object=axes_object, ci_bottom_dict=ci_bottom_dict,
            ci_mean_dict=ci_mean_dict, ci_top_dict=ci_top_dict)

        best_x_coord = ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][
            best_threshold_index]
        best_y_coord = ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][
            best_threshold_index]
    else:
        model_eval_plotting.plot_performance_diagram(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            success_ratio_by_threshold=success_ratio_matrix[0, :]
        )

        best_x_coord = success_ratio_matrix[0, best_threshold_index]
        best_y_coord = pod_matrix[0, best_threshold_index]

    print((
        'Best determinization threshold = {0:.3f} ... corresponding POD = '
        '{1:.3f} ... success ratio = {2:.3f}'
    ).format(
        best_threshold, best_y_coord, best_x_coord
    ))

    marker_colour = model_eval_plotting.PERF_DIAGRAM_COLOUR
    axes_object.plot(
        best_x_coord, best_y_coord, linestyle='None', marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
        markerfacecolor=marker_colour, markeredgecolor=marker_colour
    )

    axes_object.text(
        0.98, 0.98, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='right', verticalalignment='top',
        transform=axes_object.transAxes
    )

    axes_object.set_title('Performance diagram')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)', y_coord_normalized=1.025
    )

    axes_object.set_aspect('equal')

    print('Saving performance diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _plot_attributes_diagram(
        result_table_xarray, class_index, output_file_name,
        confidence_level=None):
    """Plots attributes diagram for one class.

    Will plot for the [k]th class, where k = `class_index`.

    :param result_table_xarray: xarray table produced by
        `evaluation_utils.run_evaluation`.
    :param class_index: See discussion above.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: See documentation at top of file.
    """

    k = class_index

    probability_matrix = result_table_xarray[
        evaluation_utils.MEAN_PROBABILITY_KEY
    ].values[:, k, :]

    event_frequency_matrix = result_table_xarray[
        evaluation_utils.EVENT_FREQUENCY_KEY
    ].values[:, k, :]

    num_examples_by_bin = numpy.mean(
        result_table_xarray[evaluation_utils.NUM_EXAMPLES_KEY].values[:, k, :],
        axis=0
    )
    num_examples_by_bin = numpy.round(num_examples_by_bin).astype(int)

    num_bootstrap_reps = probability_matrix.shape[0]
    num_bins = probability_matrix.shape[1]

    bss_values = result_table_xarray[evaluation_utils.BSS_KEY].values[:, k]

    if num_bootstrap_reps > 1:
        min_bss, max_bss = bootstrapping.get_confidence_interval(
            stat_values=bss_values, confidence_level=confidence_level)

        annotation_string = 'Brier skill score = [{0:.3f}, {1:.3f}]'.format(
            min_bss, max_bss
        )
    else:
        mean_bss = numpy.nanmean(bss_values)
        annotation_string = 'Brier skill score = {0:.3f}'.format(mean_bss)

    print(annotation_string)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.MEAN_FORECAST_BY_BIN_KEY:
                numpy.full(num_bins, numpy.nan),
            model_eval.EVENT_FREQ_BY_BIN_KEY:
                numpy.full(num_bins, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_bins):
            (ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j],
             ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=probability_matrix[:, j],
                confidence_level=confidence_level
            )

            (ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j],
             ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=event_frequency_matrix[:, j],
                confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j] = (
                numpy.nanmean(probability_matrix[:, j])
            )

            ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j] = numpy.nanmean(
                event_frequency_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            ci_bottom_dict=ci_bottom_dict, ci_mean_dict=ci_mean_dict,
            ci_top_dict=ci_top_dict, num_examples_by_bin=num_examples_by_bin)
    else:
        model_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_forecast_by_bin=probability_matrix[0, :],
            event_frequency_by_bin=event_frequency_matrix[0, :],
            num_examples_by_bin=num_examples_by_bin)

    axes_object.text(
        0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='left', verticalalignment='top',
        transform=axes_object.transAxes
    )

    axes_object.set_title('Attributes diagram for {0:d}th class'.format(k + 1))
    axes_object.set_aspect('equal')

    print('Saving attributes diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(input_file_name, confidence_level, output_dir_name):
    """Plots results of pixelwise evaluation.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = evaluation_utils.read_file(input_file_name)

    _plot_roc_curve(
        result_table_xarray=result_table_xarray,
        output_file_name='{0:s}/roc_curve.jpg'.format(output_dir_name),
        confidence_level=confidence_level
    )

    _plot_performance_diagram(
        result_table_xarray=result_table_xarray,
        output_file_name='{0:s}/performance_diagram.jpg'.format(
            output_dir_name
        ),
        confidence_level=confidence_level
    )

    num_classes = result_table_xarray[evaluation_utils.BSS_KEY].values.shape[1]

    for k in range(num_classes):
        _plot_attributes_diagram(
            result_table_xarray=result_table_xarray, class_index=k,
            output_file_name='{0:s}/attributes_diagram_class{1:d}.jpg'.format(
                output_dir_name, k
            ),
            confidence_level=confidence_level
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
