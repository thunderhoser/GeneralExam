"""High-level methods for model evaluation.

To be used by scripts (i.e., files in the "scripts" package).
"""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_auc_score
from gewittergefahr.gg_utils import model_evaluation as gg_evaluation
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import model_eval_plotting
from generalexam.machine_learning import evaluation_utils as ge_evaluation

SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300


def _plot_roc_curves(class_probability_matrix, observed_labels,
                     output_dir_name):
    """Plots one-versus-all ROC curve for each class.

    K = number of classes

    :param class_probability_matrix: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: auc_by_class: length-K numpy array of AUC (area under ROC curve)
        values computed by GewitterGefahr.
    :return: sklearn_auc_by_class: length-K numpy array of AUC values computed
        by scikit-learn.
    """

    num_classes = class_probability_matrix.shape[1]
    auc_by_class = numpy.full(num_classes, numpy.nan)
    sklearn_auc_by_class = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        print 'Creating ROC curve for class {0:d}...'.format(k)

        this_pofd_by_threshold, this_pod_by_threshold = (
            gg_evaluation.get_points_in_roc_curve(
                forecast_probabilities=class_probability_matrix[:, k],
                observed_labels=(observed_labels == k).astype(int),
                threshold_arg=gg_evaluation.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)
        )

        auc_by_class[k] = gg_evaluation.get_area_under_roc_curve(
            pofd_by_threshold=this_pofd_by_threshold,
            pod_by_threshold=this_pod_by_threshold)

        sklearn_auc_by_class[k] = roc_auc_score(
            y_true=(observed_labels == k).astype(int),
            y_score=class_probability_matrix[:, k]
        )

        _, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        model_eval_plotting.plot_roc_curve(
            axes_object=this_axes_object,
            pod_by_threshold=this_pod_by_threshold,
            pofd_by_threshold=this_pofd_by_threshold)

        this_title_string = (
            'AUC = {0:.4f} ... scikit-learn AUC = {1:.4f}'
        ).format(auc_by_class[k], sklearn_auc_by_class[k])

        print this_title_string
        pyplot.title(this_title_string)

        this_figure_file_name = '{0:s}/roc_curve_class{1:d}.jpg'.format(
            output_dir_name, k)

        print 'Saving figure to: "{0:s}"...\n'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

    return auc_by_class, sklearn_auc_by_class


def _plot_performance_diagrams(class_probability_matrix, observed_labels,
                               output_dir_name):
    """Plots one-versus-all performance diagram for each class.

    K = number of classes

    :param class_probability_matrix: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: aupd_by_class: length-K numpy array with area under performance
        diagram for each class.
    """

    num_classes = class_probability_matrix.shape[1]
    aupd_by_class = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        print 'Creating performance diagram for class {0:d}...'.format(k)

        this_sr_by_threshold, this_pod_by_threshold = (
            gg_evaluation.get_points_in_performance_diagram(
                forecast_probabilities=class_probability_matrix[:, k],
                observed_labels=(observed_labels == k).astype(int),
                threshold_arg=gg_evaluation.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)
        )

        aupd_by_class[k] = gg_evaluation.get_area_under_perf_diagram(
            success_ratio_by_threshold=this_sr_by_threshold,
            pod_by_threshold=this_pod_by_threshold)

        _, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        model_eval_plotting.plot_performance_diagram(
            axes_object=this_axes_object,
            pod_by_threshold=this_pod_by_threshold,
            success_ratio_by_threshold=this_sr_by_threshold)

        this_title_string = 'AUPD = {0:.4f}'.format(aupd_by_class[k])
        print this_title_string
        pyplot.title(this_title_string)

        this_figure_file_name = (
            '{0:s}/performance_diagram_class{1:d}.jpg'
        ).format(output_dir_name, k)

        print 'Saving figure to: "{0:s}"...\n'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

    return aupd_by_class


def _plot_attributes_diagrams(class_probability_matrix, observed_labels,
                              output_dir_name):
    """Plots one-versus-all attributes diagram for each class.

    K = number of classes

    :param class_probability_matrix: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: reliability_by_class: length-K numpy array with reliability for
        each class.
    :return: bss_by_class: length-K numpy array with Brier skill score for each
        class.
    """

    num_classes = class_probability_matrix.shape[1]
    reliability_by_class = numpy.full(num_classes, numpy.nan)
    bss_by_class = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        print 'Creating attributes diagram for class {0:d}...'.format(k)

        (this_mean_forecast_by_bin, this_class_freq_by_bin,
         this_num_examples_by_bin
        ) = gg_evaluation.get_points_in_reliability_curve(
            forecast_probabilities=class_probability_matrix[:, k],
            observed_labels=(observed_labels == k).astype(int)
        )

        this_climatology = numpy.mean(observed_labels == k)
        this_bss_dict = gg_evaluation.get_brier_skill_score(
            mean_forecast_prob_by_bin=this_mean_forecast_by_bin,
            mean_observed_label_by_bin=this_class_freq_by_bin,
            num_examples_by_bin=this_num_examples_by_bin,
            climatology=this_climatology)

        reliability_by_class[k] = this_bss_dict[gg_evaluation.RELIABILITY_KEY]
        bss_by_class[k] = this_bss_dict[gg_evaluation.BRIER_SKILL_SCORE_KEY]

        _, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        model_eval_plotting.plot_reliability_curve(
            axes_object=this_axes_object,
            mean_forecast_prob_by_bin=this_mean_forecast_by_bin,
            mean_observed_label_by_bin=this_class_freq_by_bin)

        this_title_string = (
            'REL = {0:.4f} ... RES = {1:.4f} ... BSS = {2:.4f}'
        ).format(this_bss_dict[gg_evaluation.RELIABILITY_KEY],
                 this_bss_dict[gg_evaluation.RESOLUTION_KEY],
                 this_bss_dict[gg_evaluation.BRIER_SKILL_SCORE_KEY])

        print this_title_string
        pyplot.title(this_title_string)

        this_figure_file_name = (
            '{0:s}/reliability_curve_class{1:d}.jpg'
        ).format(output_dir_name, k)

        print 'Saving figure to: "{0:s}"...\n'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_figure_object, this_axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        model_eval_plotting.plot_attributes_diagram(
            figure_object=this_figure_object, axes_object=this_axes_object,
            mean_forecast_prob_by_bin=this_mean_forecast_by_bin,
            mean_observed_label_by_bin=this_class_freq_by_bin,
            num_examples_by_bin=this_num_examples_by_bin)

        pyplot.title(this_title_string)

        this_figure_file_name = (
            '{0:s}/attributes_diagram_class{1:d}.jpg'
        ).format(output_dir_name, k)

        print 'Saving figure to: "{0:s}"...\n'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

    return reliability_by_class, bss_by_class


def run_evaluation(class_probability_matrix, observed_labels, output_dir_name):
    """Evaluates a set of multiclass probabilistic predictions.

    E = number of examples
    K = number of classes

    :param class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] = probability that the [i]th example
        belongs to the [k]th class.  Classes should be mutually exclusive and
        collectively exhaustive, so that the sum across each row is 1.0.
    :param observed_labels: length-E numpy array of observed labels.  Each label
        must be an integer from 0...(K - 1).
    :param output_dir_name: Name of output directory.  Results will be saved
        here.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Finding best binarization threshold (front vs. no front)...'

    binarization_threshold, best_gerrity_score = (
        ge_evaluation.find_best_binarization_threshold(
            class_probability_matrix=class_probability_matrix,
            observed_labels=observed_labels,
            threshold_arg=gg_evaluation.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            criterion_function=ge_evaluation.get_gerrity_score,
            optimization_direction=ge_evaluation.MAX_OPTIMIZATION_DIRECTION,
            forecast_precision_for_thresholds=FORECAST_PRECISION_FOR_THRESHOLDS)
    )

    print (
        'Best binarization threshold = {0:.4f} ... corresponding Gerrity score '
        '= {1:.4f}'
    ).format(binarization_threshold, best_gerrity_score)

    print 'Determinizing multiclass probabilities...'
    predicted_labels = ge_evaluation.determinize_probabilities(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=binarization_threshold)

    contingency_matrix = ge_evaluation.get_contingency_table(
        predicted_labels=predicted_labels, observed_labels=observed_labels,
        num_classes=class_probability_matrix.shape[1]
    )

    print 'Multiclass contingency table is shown below:\n{0:s}'.format(
        str(contingency_matrix)
    )
    print SEPARATOR_STRING

    accuracy = ge_evaluation.get_accuracy(contingency_matrix)
    peirce_score = ge_evaluation.get_peirce_score(contingency_matrix)
    heidke_score = ge_evaluation.get_heidke_score(contingency_matrix)
    gerrity_score = ge_evaluation.get_gerrity_score(contingency_matrix)

    print (
        'Multiclass accuracy = {0:.4f} ... Peirce score = {1:.4f} ... '
        'Heidke score = {2:.4f} ... Gerrity score = {3:.4f}\n'
    ).format(accuracy, peirce_score, heidke_score, gerrity_score)

    binary_contingency_dict = gg_evaluation.get_contingency_table(
        forecast_labels=(predicted_labels > 0).astype(int),
        observed_labels=(observed_labels > 0).astype(int)
    )

    print 'Binary contingency table is shown below:\n{0:s}'.format(
        str(binary_contingency_dict)
    )
    print SEPARATOR_STRING

    binary_pod = gg_evaluation.get_pod(binary_contingency_dict)
    binary_pofd = gg_evaluation.get_pofd(binary_contingency_dict)
    binary_success_ratio = gg_evaluation.get_success_ratio(
        binary_contingency_dict)
    binary_focn = gg_evaluation.get_focn(binary_contingency_dict)
    binary_accuracy = gg_evaluation.get_accuracy(binary_contingency_dict)
    binary_csi = gg_evaluation.get_csi(binary_contingency_dict)
    binary_frequency_bias = gg_evaluation.get_frequency_bias(
        binary_contingency_dict)

    print (
        'Binary POD = {0:.4f} ... POFD = {1:.4f} ... success ratio = {2:.4f} '
        '... FOCN = {3:.4f} ... accuracy = {4:.4f} ... CSI = {5:.4f} ... '
        'frequency bias = {6:.4f}\n'
    ).format(binary_pod, binary_pofd, binary_success_ratio, binary_focn,
             binary_accuracy, binary_csi, binary_frequency_bias)

    auc_by_class, sklearn_auc_by_class = _plot_roc_curves(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    aupd_by_class = _plot_performance_diagrams(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        output_dir_name=output_dir_name)
    print '\n'

    reliability_by_class, bss_by_class = _plot_attributes_diagrams(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    evaluation_file_name = '{0:s}/model_evaluation.p'.format(output_dir_name)
    print 'Writing results to: "{0:s}"...\n'.format(evaluation_file_name)

    ge_evaluation.write_file(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        binarization_threshold=binarization_threshold, accuracy=accuracy,
        peirce_score=peirce_score, heidke_score=heidke_score,
        gerrity_score=gerrity_score, binary_pod=binary_pod,
        binary_pofd=binary_pofd, binary_success_ratio=binary_success_ratio,
        binary_focn=binary_focn, binary_accuracy=binary_accuracy,
        binary_csi=binary_csi, binary_frequency_bias=binary_frequency_bias,
        auc_by_class=auc_by_class,
        scikit_learn_auc_by_class=sklearn_auc_by_class,
        aupd_by_class=aupd_by_class, reliability_by_class=reliability_by_class,
        bss_by_class=bss_by_class, pickle_file_name=evaluation_file_name)
