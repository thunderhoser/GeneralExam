"""Evaluates a traditional CNN, preferably on non-training data.

A "traditional CNN" is one for which the output (prediction) is not spatially
explicit.  The opposite is a fully convolutional net (see fcn.py).
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_auc_score
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.plotting import model_eval_plotting
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import evaluation_utils as eval_utils
from generalexam.machine_learning import isotonic_regression

INPUT_TIME_FORMAT = '%Y%m%d%H'
FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DOTS_PER_INCH = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

MODEL_FILE_ARG_NAME = 'input_model_file_name'
FIRST_EVAL_TIME_ARG_NAME = 'first_eval_time_string'
LAST_EVAL_TIME_ARG_NAME = 'last_eval_time_string'
NUM_EVAL_TIMES_ARG_NAME = 'num_evaluation_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
USE_ISOTONIC_REGRESSION_ARG_NAME = 'use_isotonic_regression'
NARR_DIR_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to model file, containing a trained CNN.  This file should be '
    'readable by `traditional_cnn.read_keras_model`.')
EVAL_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Evaluation pairs (i.e., forecast-observation '
    'pairs) will be drawn randomly from the time period `{0:s}`...'
    '`{1:s}`.').format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)
NUM_EVAL_TIMES_HELP_STRING = (
    'Number of evaluation times to be drawn randomly from the period `{0:s}`...'
    '`{1:s}`.').format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)
NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of evaluation pairs to be drawn randomly from each time.')
USE_ISOTONIC_REGRESSION_HELP_STRING = (
    'Boolean flag.  If 1, isotonic regression will be used to calibrate '
    'probabilities from the base model.')
NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data (one file for each variable, '
    'pressure level, and time step).')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (one per file, indicating '
    'which NARR grid cells are intersected by a front).')

DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EVAL_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_EVAL_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ISOTONIC_REGRESSION_ARG_NAME, type=int, required=False,
    default=0, help=USE_ISOTONIC_REGRESSION_ARG_NAME)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_FRONTAL_GRID_DIR_NAME, help=FRONTAL_GRID_DIR_HELP_STRING)


def _create_roc_curves(
        class_probability_matrix, observed_labels, model_directory_name):
    """Creates one-vs-all ROC curve for each class.

    P = number of evaluation pairs
    K = number of classes

    :param class_probability_matrix: P-by-K numpy array of floats.
        class_probability_matrix[i, k] is the predicted probability that the
        [i]th example belongs to the [k]th class.
    :param observed_labels: length-P numpy array of integers.  If
        observed_labels[i] = k, the [i]th example truly belongs to the [k]th
        class.
    :param model_directory_name: Path to output directory.
    :return: auc_by_class: length-K numpy array with area under ROC curve (AUC)
        from GewitterGefahr for each class.
    :return: scikit_learn_auc_by_class: Same but with scikit-learn calculations.
    """

    num_classes = class_probability_matrix.shape[1]
    auc_by_class = numpy.full(num_classes, numpy.nan)
    scikit_learn_auc_by_class = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        print 'Creating ROC curve for class {0:d}...'.format(k)
        this_pofd_by_threshold, this_pod_by_threshold = (
            model_eval.get_points_in_roc_curve(
                forecast_probabilities=class_probability_matrix[:, k],
                observed_labels=(observed_labels == k).astype(int),
                threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS))

        auc_by_class[k] = model_eval.get_area_under_roc_curve(
            pofd_by_threshold=this_pofd_by_threshold,
            pod_by_threshold=this_pod_by_threshold)
        scikit_learn_auc_by_class[k] = roc_auc_score(
            y_true=(observed_labels == k).astype(int),
            y_score=class_probability_matrix[:, k])

        print ('AUC estimates from GewitterGefahr and scikit-learn: {0:.4f}, '
               '{1:.4f}').format(auc_by_class[k], scikit_learn_auc_by_class[k])

        this_figure_file_name = '{0:s}/roc_curve_class{1:d}.jpg'.format(
            model_directory_name, k)
        print 'Saving ROC curve to: "{0:s}"...\n'.format(this_figure_file_name)

        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        model_eval_plotting.plot_roc_curve(
            axes_object=axes_object, pod_by_threshold=this_pod_by_threshold,
            pofd_by_threshold=this_pofd_by_threshold)

        title_string = (
            'AUC = {0:.4f} (GewitterGefahr), {1:.4f} (scikit-learn).'.format(
                auc_by_class[k], scikit_learn_auc_by_class[k]))
        pyplot.title(title_string)
        pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
        pyplot.close()

    return auc_by_class, scikit_learn_auc_by_class


def _create_performance_diagrams(
        class_probability_matrix, observed_labels, model_directory_name):
    """Creates one-vs-all performance diagram for each class.

    :param class_probability_matrix: See documentation for `_create_roc_curves`.
    :param observed_labels: Same.
    :param model_directory_name: Same.
    """

    num_classes = class_probability_matrix.shape[1]
    for k in range(num_classes):
        print 'Creating performance diagram for class {0:d}...'.format(k)
        this_sr_by_threshold, this_pod_by_threshold = (
            model_eval.get_points_in_performance_diagram(
                forecast_probabilities=class_probability_matrix[:, k],
                observed_labels=(observed_labels == k).astype(int),
                threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS))

        this_figure_file_name = (
            '{0:s}/performance_diagram_class{1:d}.jpg'.format(
                model_directory_name, k))
        print 'Saving performance diagram to: "{0:s}"...'.format(
            this_figure_file_name)

        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        model_eval_plotting.plot_performance_diagram(
            axes_object=axes_object, pod_by_threshold=this_pod_by_threshold,
            success_ratio_by_threshold=this_sr_by_threshold)

        pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
        pyplot.close()


def _create_attributes_diagrams(
        class_probability_matrix, observed_labels, model_directory_name):
    """Creates one-vs-all attributes diagram for each class.

    :param class_probability_matrix: See documentation for `_create_roc_curves`.
    :param observed_labels: Same.
    :param model_directory_name: Same.
    """

    num_classes = class_probability_matrix.shape[1]
    for k in range(num_classes):
        print 'Creating reliability curve for class {0:d}...'.format(k)
        (this_mean_forecast_by_bin,
         this_class_freq_by_bin,
         this_num_examples_by_bin) = model_eval.get_points_in_reliability_curve(
             forecast_probabilities=class_probability_matrix[:, k],
             observed_labels=(observed_labels == k).astype(int))

        this_figure_file_name = (
            '{0:s}/reliability_curve_class{1:d}.jpg'.format(
                model_directory_name, k))
        print 'Saving reliability curve to: "{0:s}"...'.format(
            this_figure_file_name)

        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        model_eval_plotting.plot_reliability_curve(
            axes_object=axes_object,
            mean_forecast_prob_by_bin=this_mean_forecast_by_bin,
            mean_observed_label_by_bin=this_class_freq_by_bin)

        pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
        pyplot.close()

        this_figure_file_name = (
            '{0:s}/attributes_diagram_class{1:d}.jpg'.format(
                model_directory_name, k))
        print 'Saving attributes diagram to: "{0:s}"...'.format(
            this_figure_file_name)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        model_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_forecast_prob_by_bin=this_mean_forecast_by_bin,
            mean_observed_label_by_bin=this_class_freq_by_bin,
            num_examples_by_bin=this_num_examples_by_bin)

        pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
        pyplot.close()


def _evaluate_model(
        model_file_name, first_eval_time_string, last_eval_time_string,
        num_evaluation_times, num_examples_per_time, use_isotonic_regression,
        top_narr_directory_name, top_frontal_grid_dir_name):
    """Evaluates a traditional CNN, preferably on non-training data.

    :param model_file_name: Path to model file, containing a trained CNN.  This
        file should be readable by `traditional_cnn.read_keras_model`.
    :param first_eval_time_string: Time (format "yyyymmddHH").  Evaluation pairs
        (i.e., forecast-observation pairs) will be drawn randomly from the time
        period `first_eval_time_string`...`last_eval_time_string`.
    :param last_eval_time_string: See above.
    :param num_evaluation_times: Number of evaluation times to be drawn randomly
        from the period `first_eval_time_string`...`last_eval_time_string`.
    :param num_examples_per_time: Number of evaluation pairs to be drawn
        randomly from each time.
    :param use_isotonic_regression: Boolean flag.  If True, isotonic regression
        will be used to calibrate probabilities from the base model.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one per file, indicating which NARR grid cells are intersected by
        a front).
    """

    first_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        first_eval_time_string, INPUT_TIME_FORMAT)
    last_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        last_eval_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        model_directory_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metadata_file_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metadata_file_name)

    if use_isotonic_regression:
        isotonic_regression_file_name = (
            '{0:s}/isotonic_regression_models.p'.format(model_directory_name))

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_regression_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(
                isotonic_regression_file_name))
    else:
        isotonic_model_object_by_class = None

    print SEPARATOR_STRING
    num_classes = len(model_metadata_dict[traditional_cnn.CLASS_FRACTIONS_KEY])

    class_probability_matrix, observed_labels = (
        eval_utils.downsized_examples_to_eval_pairs(
            model_object=model_object,
            first_target_time_unix_sec=first_eval_time_unix_sec,
            last_target_time_unix_sec=last_eval_time_unix_sec,
            num_target_times_to_sample=num_evaluation_times,
            num_examples_per_time=num_examples_per_time,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=model_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            pressure_level_mb=model_metadata_dict[
                traditional_cnn.PRESSURE_LEVEL_KEY],
            dilation_distance_for_target_metres=model_metadata_dict[
                traditional_cnn.DILATION_DISTANCE_FOR_TARGET_KEY],
            num_rows_in_half_grid=model_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_columns_in_half_grid=model_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            num_classes=num_classes,
            num_predictor_time_steps=model_metadata_dict[
                traditional_cnn.NUM_PREDICTOR_TIME_STEPS_KEY],
            num_lead_time_steps=model_metadata_dict[
                traditional_cnn.NUM_LEAD_TIME_STEPS_KEY],
            isotonic_model_object_by_class=isotonic_model_object_by_class))
    print SEPARATOR_STRING

    print 'Finding best binarization threshold (front vs. no front)...'
    binarization_threshold, best_binary_peirce_score = (
        eval_utils.find_best_binarization_threshold(
            class_probability_matrix=class_probability_matrix,
            observed_labels=observed_labels,
            threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            criterion_function=model_eval.get_peirce_score,
            optimization_direction=eval_utils.MAX_OPTIMIZATION_DIRECTION,
            forecast_precision_for_thresholds=
            FORECAST_PRECISION_FOR_THRESHOLDS))

    print ('Best binarization threshold = {0:.4f} (corresponding binary Peirce '
           'score = {1:.4f})\n').format(binarization_threshold,
                                        best_binary_peirce_score)

    print 'Determinizing probabilities...'
    predicted_labels = eval_utils.determinize_probabilities(
        class_probability_matrix=class_probability_matrix,
        binarization_threshold=binarization_threshold)

    print 'Creating contingency table...'
    contingency_table_as_matrix = eval_utils.get_contingency_table(
        predicted_labels=predicted_labels, observed_labels=observed_labels,
        num_classes=num_classes)

    print contingency_table_as_matrix

    print 'Computing non-binary performance metrics...'
    accuracy = eval_utils.get_accuracy(contingency_table_as_matrix)
    peirce_score = eval_utils.get_peirce_score(contingency_table_as_matrix)
    heidke_score = eval_utils.get_heidke_score(contingency_table_as_matrix)
    gerrity_score = eval_utils.get_gerrity_score(contingency_table_as_matrix)

    print ('Accuracy = {0:.4f} ... Peirce score = {1:.4f} ... Heidke score = '
           '{2:.4f} ... Gerrity score = {3:.4f}\n').format(
               accuracy, peirce_score, heidke_score, gerrity_score)

    print 'Creating binary contingency table...'
    binary_contingency_table_as_dict = model_eval.get_contingency_table(
        forecast_labels=(predicted_labels > 0).astype(int),
        observed_labels=(observed_labels > 0).astype(int))

    print 'Computing binary performance metrics...'
    binary_pod = model_eval.get_pod(binary_contingency_table_as_dict)
    binary_pofd = model_eval.get_pofd(binary_contingency_table_as_dict)
    binary_success_ratio = model_eval.get_success_ratio(
        binary_contingency_table_as_dict)
    binary_focn = model_eval.get_focn(binary_contingency_table_as_dict)
    binary_accuracy = model_eval.get_accuracy(binary_contingency_table_as_dict)
    binary_csi = model_eval.get_csi(binary_contingency_table_as_dict)
    binary_frequency_bias = model_eval.get_frequency_bias(
        binary_contingency_table_as_dict)

    print ('Binary POD = {0:.4f} ... POFD = {1:.4f} ... success ratio = {2:.4f}'
           ' ... FOCN = {3:.4f} ... accuracy = {4:.4f} ... CSI = {5:.4f} ... '
           'frequency bias = {6:.4f}\n').format(
               binary_pod, binary_pofd, binary_success_ratio, binary_focn,
               binary_accuracy, binary_csi, binary_frequency_bias)

    auc_by_class, scikit_learn_auc_by_class = _create_roc_curves(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        model_directory_name=model_directory_name)
    print '\n'

    evaluation_file_name = '{0:s}/model_evaluation.p'.format(
        model_directory_name)

    print 'Writing evaluation results to: "{0:s}"...\n'.format(
        evaluation_file_name)
    eval_utils.write_evaluation_results(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        binarization_threshold=binarization_threshold, accuracy=accuracy,
        peirce_score=peirce_score, heidke_score=heidke_score,
        gerrity_score=gerrity_score, binary_pod=binary_pod,
        binary_pofd=binary_pofd, binary_success_ratio=binary_success_ratio,
        binary_focn=binary_focn, binary_accuracy=binary_accuracy,
        binary_csi=binary_csi, binary_frequency_bias=binary_frequency_bias,
        auc_by_class=auc_by_class,
        scikit_learn_auc_by_class=scikit_learn_auc_by_class,
        pickle_file_name=evaluation_file_name)

    _create_performance_diagrams(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        model_directory_name=model_directory_name)
    print '\n'

    _create_attributes_diagrams(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        model_directory_name=model_directory_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    MODEL_FILE_NAME = getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)
    FIRST_EVAL_TIME_STRING = getattr(INPUT_ARG_OBJECT, FIRST_EVAL_TIME_ARG_NAME)
    LAST_EVAL_TIME_STRING = getattr(INPUT_ARG_OBJECT, LAST_EVAL_TIME_ARG_NAME)
    NUM_EVALUATION_TIMES = getattr(INPUT_ARG_OBJECT, NUM_EVAL_TIMES_ARG_NAME)
    NUM_EXAMPLES_PER_TIME = getattr(
        INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME)
    USE_ISOTONIC_REGRESSION = bool(
        getattr(INPUT_ARG_OBJECT, USE_ISOTONIC_REGRESSION_ARG_NAME))

    TOP_NARR_DIRECTORY_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME)
    TOP_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_ARG_NAME)

    _evaluate_model(
        model_file_name=MODEL_FILE_NAME,
        first_eval_time_string=FIRST_EVAL_TIME_STRING,
        last_eval_time_string=LAST_EVAL_TIME_STRING,
        num_evaluation_times=NUM_EVALUATION_TIMES,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME,
        use_isotonic_regression=USE_ISOTONIC_REGRESSION,
        top_narr_directory_name=TOP_NARR_DIRECTORY_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME)
