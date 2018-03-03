"""Methods for evaluating a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples.  Each example is one image or a time sequence of images.
M = number of pixel rows in each image
N = number of pixel columns in each image
T = number of predictor times per example (images per sequence)
C = number of channels (predictor variables) in each image

--- DEFINITIONS ---

"Evaluation pair" = forecast-prediction pair

A "downsized" example covers only a portion of the NARR grid (as opposed to
a full-size example, which covers the entire NARR grid).

For a 3-D example, the dimensions are M x N x C (M rows, N columns, C predictor
variables).

For a 4-D example, the dimensions are M x N x T x C (M rows, N columns, T time
steps, C predictor variables).
"""

import pickle
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import fcn

# TODO(thunderhoser): This file contains a lot of duplicated code.  Should
# combine downsized 3-D and 4-D into one method, full-size 3-D and 4-D into one
# method.

NARR_TIME_INTERVAL_SECONDS = 10800
DEFAULT_FORECAST_PRECISION_FOR_THRESHOLDS = 1e-3
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'

MIN_OPTIMIZATION_DIRECTION = 'min'
MAX_OPTIMIZATION_DIRECTION = 'max'
VALID_OPTIMIZATION_DIRECTIONS = [
    MIN_OPTIMIZATION_DIRECTION, MAX_OPTIMIZATION_DIRECTION]

CLASS_PROBABILITY_MATRIX_KEY = 'class_probability_matrix'
OBSERVED_LABELS_KEY = 'observed_labels'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
ACCURACY_KEY = 'accuracy'
PEIRCE_SCORE_KEY = 'peirce_score'
HEIDKE_SCORE_KEY = 'heidke_score'
GERRITY_SCORE_KEY = 'gerrity_score'
BINARY_POD_KEY = 'binary_pod'
BINARY_POFD_KEY = 'binary_pofd'
BINARY_SUCCESS_RATIO_KEY = 'binary_success_ratio'
BINARY_FOCN_KEY = 'binary_focn'
BINARY_ACCURACY_KEY = 'binary_accuracy'
BINARY_CSI_KEY = 'binary_csi'
BINARY_FREQUENCY_BIAS_KEY = 'binary_frequency_bias'
AUC_BY_CLASS_KEY = 'auc_by_class'
SCIKIT_LEARN_AUC_BY_CLASS_KEY = 'scikit_learn_auc_by_class'

EVALUATION_DICT_KEYS = [
    CLASS_PROBABILITY_MATRIX_KEY, OBSERVED_LABELS_KEY,
    BINARIZATION_THRESHOLD_KEY, ACCURACY_KEY, PEIRCE_SCORE_KEY,
    HEIDKE_SCORE_KEY, GERRITY_SCORE_KEY, BINARY_POD_KEY, BINARY_POFD_KEY,
    BINARY_SUCCESS_RATIO_KEY, BINARY_FOCN_KEY, BINARY_ACCURACY_KEY,
    BINARY_CSI_KEY, BINARY_FREQUENCY_BIAS_KEY, AUC_BY_CLASS_KEY,
    SCIKIT_LEARN_AUC_BY_CLASS_KEY
]

NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)
THESE_ROW_INDICES = numpy.linspace(
    0, NUM_ROWS_IN_NARR - 1, num=NUM_ROWS_IN_NARR, dtype=int)
THESE_COLUMN_INDICES = numpy.linspace(
    0, NUM_COLUMNS_IN_NARR - 1, num=NUM_COLUMNS_IN_NARR, dtype=int)
THIS_COLUMN_INDEX_MATRIX, THIS_ROW_INDEX_MATRIX = grids.xy_vectors_to_matrices(
    x_unique_metres=THESE_COLUMN_INDICES, y_unique_metres=THESE_ROW_INDICES)

ALL_ROW_INDICES_FOR_DOWNSIZED_EXAMPLES = numpy.reshape(
    THIS_ROW_INDEX_MATRIX, THIS_ROW_INDEX_MATRIX.size).astype(int)
ALL_COLUMN_INDICES_FOR_DOWNSIZED_EXAMPLES = numpy.reshape(
    THIS_COLUMN_INDEX_MATRIX, THIS_COLUMN_INDEX_MATRIX.size).astype(int)

THESE_ROW_INDICES = numpy.linspace(
    0, len(ml_utils.NARR_ROWS_FOR_FCN_INPUT) - 1,
    num=len(ml_utils.NARR_ROWS_FOR_FCN_INPUT), dtype=int)
THESE_COLUMN_INDICES = numpy.linspace(
    0, len(ml_utils.NARR_COLUMNS_FOR_FCN_INPUT) - 1,
    num=len(ml_utils.NARR_COLUMNS_FOR_FCN_INPUT), dtype=int)
THIS_COLUMN_INDEX_MATRIX, THIS_ROW_INDEX_MATRIX = grids.xy_vectors_to_matrices(
    x_unique_metres=THESE_COLUMN_INDICES, y_unique_metres=THESE_ROW_INDICES)

ALL_ROW_INDICES_FOR_FULL_SIZE_EXAMPLES = numpy.reshape(
    THIS_ROW_INDEX_MATRIX, THIS_ROW_INDEX_MATRIX.size).astype(int)
ALL_COLUMN_INDICES_FOR_FULL_SIZE_EXAMPLES = numpy.reshape(
    THIS_COLUMN_INDEX_MATRIX, THIS_COLUMN_INDEX_MATRIX.size).astype(int)


def _check_evaluation_pairs(class_probability_matrix, observed_labels):
    """Checks evaluation pairs for errors.

    P = number of evaluation pairs
    K = number of classes

    :param class_probability_matrix: P-by-K numpy array of floats.
        class_probability_matrix[i, k] is the predicted probability that the
        [i]th example belongs to the [k]th class.
    :param observed_labels: length-P numpy array of integers.  If
        observed_labels[i] = k, the [i]th example truly belongs to the [k]th
        class.
    """

    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)

    num_evaluation_pairs = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    error_checking.assert_is_numpy_array(
        observed_labels, exact_dimensions=numpy.array([num_evaluation_pairs]))
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_less_than_numpy_array(observed_labels, num_classes)


def _check_contingency_table(contingency_table_as_matrix):
    """Checks contingency table for errors.

    :param contingency_table_as_matrix: K-by-K numpy array.
        contingency_table_as_matrix[i, j] is the number of examples for which
        the predicted label is i and the true label is j.
    """

    error_checking.assert_is_numpy_array(
        contingency_table_as_matrix, num_dimensions=2)

    num_classes = contingency_table_as_matrix.shape[0]
    error_checking.assert_is_numpy_array(
        contingency_table_as_matrix,
        exact_dimensions=numpy.array([num_classes, num_classes]))

    error_checking.assert_is_integer_numpy_array(contingency_table_as_matrix)
    error_checking.assert_is_geq_numpy_array(contingency_table_as_matrix, 0)


def _non_zero(input_value):
    """Makes input non-zero.

    Specifically, if the input is in [0, epsilon], this method returns epsilon
    (machine limit for representable positive floating-point numbers).  If the
    input is in [-epsilon, 0), this method returns -epsilon.

    :param input_value: Input value.
    :return: output_value: Closest number to input value that is not in
        [-epsilon, epsilon].
    """

    epsilon = numpy.finfo(float).eps
    if input_value >= 0:
        return max([input_value, epsilon])

    return min([input_value, -epsilon])


def _get_num_predictions_in_class(contingency_table_as_matrix, class_index):
    """Returns number of predictions in the [k]th class (class_index = k).

    :param contingency_table_as_matrix: See documentation for
        `get_contingency_table`.
    :param class_index: k in the above discussion.
    :return: num_predictions_in_class: Number of predictions in the [k]th class.
    """

    return numpy.sum(contingency_table_as_matrix[class_index, :])


def _get_num_true_labels_in_class(contingency_table_as_matrix, class_index):
    """Returns number of true labels in the [k]th class (class_index = k).

    :param contingency_table_as_matrix: See documentation for
        `get_contingency_table`.
    :param class_index: k in the above discussion.
    :return: num_true_labels_in_class: Number of true labels in the [k]th class.
    """

    return numpy.sum(contingency_table_as_matrix[:, class_index])


def _get_random_sample_points(num_points, for_downsized_examples):
    """Samples random points from NARR grid.

    P = num_points

    :param num_points: Number of points to sample.
    :param for_downsized_examples: Boolean flag.  If True, this method will
        sample center points for downsized images.  If False, this method will
        sample evaluation points from a full-size image.
    :return: row_indices: length-P numpy array with row indices for sampled
        points.
    :return: column_indices: length-P numpy array with column indices for
        sampled points.
    """

    if for_downsized_examples:
        row_indices = numpy.random.choice(
            ALL_ROW_INDICES_FOR_DOWNSIZED_EXAMPLES, size=num_points,
            replace=False)
        column_indices = numpy.random.choice(
            ALL_COLUMN_INDICES_FOR_DOWNSIZED_EXAMPLES, size=num_points,
            replace=False)

    else:
        row_indices = numpy.random.choice(
            ALL_ROW_INDICES_FOR_FULL_SIZE_EXAMPLES, size=num_points,
            replace=False)
        column_indices = numpy.random.choice(
            ALL_COLUMN_INDICES_FOR_FULL_SIZE_EXAMPLES, size=num_points,
            replace=False)

    return row_indices, column_indices


def _get_a_for_gerrity_score(contingency_table_as_matrix):
    """Returns a-vector for Gerrity score.

    The equation for a is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: a_vector: As advertised.
    """

    num_classes = contingency_table_as_matrix.shape[0]
    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)

    num_examples_by_class = numpy.array(
        [_get_num_true_labels_in_class(contingency_table_as_matrix, i)
         for i in range(num_classes)])
    cumul_frequency_by_class = numpy.cumsum(
        num_examples_by_class.astype(float) / num_evaluation_pairs)

    return (1. - cumul_frequency_by_class) / cumul_frequency_by_class


def _get_s_for_gerrity_score(contingency_table_as_matrix):
    """Returns S-matrix for Gerrity score.

    The equation for S is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: s_matrix: As advertised.
    """

    a_vector = _get_a_for_gerrity_score(contingency_table_as_matrix)
    a_vector_reciprocal = 1. / a_vector

    num_classes = contingency_table_as_matrix.shape[0]
    s_matrix = numpy.full((num_classes, num_classes), numpy.nan)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                s_matrix[i, j] = (
                    numpy.sum(a_vector_reciprocal[:i]) +
                    numpy.sum(a_vector[i:-1]))
                continue

            if i > j:
                s_matrix[i, j] = s_matrix[j, i]
                continue

            s_matrix[i, j] = (
                numpy.sum(a_vector_reciprocal[:i]) - (j - i) +
                numpy.sum(a_vector[j:-1]))

    return s_matrix / (num_classes - 1)


def downsized_examples_to_eval_pairs(
        model_object, first_target_time_unix_sec, last_target_time_unix_sec,
        num_target_times_to_sample, num_examples_per_time,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_rows_in_half_grid,
        num_columns_in_half_grid, num_classes, num_predictor_time_steps=None,
        num_lead_time_steps=None):
    """Creates evaluation pairs from downsized 3-D or 4-D examples.

    M = number of pixel rows in full NARR grid
    N = number of pixel columns in full NARR grid

    m = number of pixel rows in each downsized grid
      = 2 * num_rows_in_half_grid + 1
    n = number of pixel columns in each downsized grid
      = 2 * num_columns_in_half_grid + 1

    P = number of evaluation pairs created by this method
    K = number of classes

    :param model_object: Instance of `keras.models.Model`.  This will be applied
        to each downsized example, creating the prediction for said example.
    :param first_target_time_unix_sec: Target time.  Downsized examples will be
        randomly chosen from the period `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param num_target_times_to_sample: Number of target times to sample (from
        the period `first_target_time_unix_sec`...`last_target_time_unix_sec`).
    :param num_examples_per_time: Number of downsized examples per target time.
        Downsized examples will be randomly drawn from each target time.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one file per time step).
    :param narr_predictor_names: 1-D list of NARR fields to use as predictors.
    :param pressure_level_mb: Pressure level (millibars).
    :param dilation_distance_for_target_metres: Dilation distance for both warm
        and cold fronts.
    :param num_rows_in_half_grid: See general discussion above.
    :param num_columns_in_half_grid: See general discussion above.
    :param num_classes: Number of classes.
    :param num_predictor_time_steps: [needed only if examples are 4-D]
        Number of time steps per example (images per sequence).
    :param num_lead_time_steps: [needed only if examples are 4-D]
        Number of time steps between latest predictor time (last image in the
        sequence) and target time.
    :return: class_probability_matrix: See documentation for
        `_check_evaluation_pairs`.
    :return: observed_labels: See doc for `_check_evaluation_pairs`.
    """

    error_checking.assert_is_integer(num_target_times_to_sample)
    error_checking.assert_is_greater(num_target_times_to_sample, 0)
    error_checking.assert_is_integer(num_examples_per_time)
    error_checking.assert_is_greater(num_examples_per_time, 0)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    if num_predictor_time_steps is None:
        num_dimensions_per_example = 3
    else:
        num_dimensions_per_example = 4

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(target_times_unix_sec)
    target_times_unix_sec = target_times_unix_sec[:num_target_times_to_sample]
    target_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in target_times_unix_sec]

    class_probability_matrix = numpy.full(
        (num_target_times_to_sample, num_examples_per_time, num_classes),
        numpy.nan)
    observed_labels = numpy.full(
        (num_target_times_to_sample, num_examples_per_time), -1, dtype=int)

    for i in range(num_target_times_to_sample):
        print 'Drawing evaluation pairs from {0:s}...'.format(
            target_time_strings[i])

        these_center_row_indices, these_center_column_indices = (
            _get_random_sample_points(
                num_points=num_examples_per_time, for_downsized_examples=True))

        if num_dimensions_per_example == 3:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             _, _) = testing_io.create_downsized_3d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 target_time_unix_sec=target_times_unix_sec[i],
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_distance_for_target_metres=
                 dilation_distance_for_target_metres,
                 num_classes=num_classes)

        else:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             _, _) = testing_io.create_downsized_4d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 target_time_unix_sec=target_times_unix_sec[i],
                 num_predictor_time_steps=num_predictor_time_steps,
                 num_lead_time_steps=num_lead_time_steps,
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_distance_for_target_metres=
                 dilation_distance_for_target_metres,
                 num_classes=num_classes)

        class_probability_matrix[i, ...] = model_object.predict(
            this_downsized_predictor_matrix, batch_size=num_examples_per_time)

    new_dimensions = (
        num_target_times_to_sample * num_examples_per_time, num_classes)
    class_probability_matrix = numpy.reshape(
        class_probability_matrix, new_dimensions)
    observed_labels = numpy.reshape(observed_labels, observed_labels.size)

    return class_probability_matrix, observed_labels


def full_size_examples_to_eval_pairs(
        model_object, first_target_time_unix_sec, last_target_time_unix_sec,
        num_target_times_to_sample, num_points_per_time,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_classes,
        num_predictor_time_steps=None, num_lead_time_steps=None):
    """Creates evaluation pairs from full-size 3-D or 4-D examples.

    P = number of evaluation pairs created by this method
    K = number of classes

    :param model_object: See documentation for
        `downsized_examples_to_eval_pairs`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param num_target_times_to_sample: Same.
    :param num_points_per_time: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: Same.
    :param num_classes: Same.
    :param num_predictor_time_steps: Same.
    :param num_lead_time_steps: Same.
    :return: class_probability_matrix: Same.
    :return: observed_labels: Same.
    """

    error_checking.assert_is_integer(num_target_times_to_sample)
    error_checking.assert_is_greater(num_target_times_to_sample, 0)
    error_checking.assert_is_integer(num_points_per_time)
    error_checking.assert_is_greater(num_points_per_time, 0)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    if num_predictor_time_steps is None:
        num_dimensions_per_example = 3
    else:
        num_dimensions_per_example = 4

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(target_times_unix_sec)
    target_times_unix_sec = target_times_unix_sec[:num_target_times_to_sample]
    target_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in target_times_unix_sec]

    class_probability_matrix = numpy.full(
        (num_target_times_to_sample, num_points_per_time, num_classes),
        numpy.nan)
    observed_labels = numpy.full(
        (num_target_times_to_sample, num_points_per_time), -1, dtype=int)

    for i in range(num_target_times_to_sample):
        print 'Drawing evaluation pairs from {0:s}...'.format(
            target_time_strings[i])

        if num_dimensions_per_example == 3:
            this_class_probability_matrix, this_actual_target_matrix = (
                fcn.apply_model_to_3d_example(
                    model_object=model_object,
                    target_time_unix_sec=target_times_unix_sec[i],
                    top_narr_directory_name=top_narr_directory_name,
                    top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                    narr_predictor_names=narr_predictor_names,
                    pressure_level_mb=pressure_level_mb,
                    dilation_distance_for_target_metres=
                    dilation_distance_for_target_metres,
                    num_classes=num_classes))
        else:
            this_class_probability_matrix, this_actual_target_matrix = (
                fcn.apply_model_to_4d_example(
                    model_object=model_object,
                    target_time_unix_sec=target_times_unix_sec[i],
                    num_predictor_time_steps=num_predictor_time_steps,
                    num_lead_time_steps=num_lead_time_steps,
                    top_narr_directory_name=top_narr_directory_name,
                    top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                    narr_predictor_names=narr_predictor_names,
                    pressure_level_mb=pressure_level_mb,
                    dilation_distance_for_target_metres=
                    dilation_distance_for_target_metres,
                    num_classes=num_classes))

        these_row_indices, these_column_indices = _get_random_sample_points(
            num_points=num_points_per_time, for_downsized_examples=False)

        class_probability_matrix[i, ...] = this_class_probability_matrix[
            0, these_row_indices, these_column_indices, ...]
        this_actual_target_matrix = this_actual_target_matrix[
            0, these_row_indices, these_column_indices]
        observed_labels[i, :] = numpy.reshape(
            this_actual_target_matrix, this_actual_target_matrix.size)

    new_dimensions = (
        num_target_times_to_sample * num_points_per_time, num_classes)
    class_probability_matrix = numpy.reshape(
        class_probability_matrix, new_dimensions)
    observed_labels = numpy.reshape(observed_labels, observed_labels.size)

    return class_probability_matrix, observed_labels


def find_best_binarization_threshold(
        class_probability_matrix, observed_labels, threshold_arg,
        criterion_function=model_eval.get_peirce_score,
        optimization_direction=MAX_OPTIMIZATION_DIRECTION,
        forecast_precision_for_thresholds=
        DEFAULT_FORECAST_PRECISION_FOR_THRESHOLDS):
    """Finds the best binarization threshold.

    A "binarization threshold" is used to determinize probabilistic (either
    binary or multi-class) predictions, using the following procedure.
    f* = binarization threshold, and f_0 is the forecast probability of class 0
    (no front).

    [1] If f_0 >= f*, predict no front.
    [2] If f_0 < f*, predict a front.  In multi-class problems, frontal type
        (warm or cold) is determined by whichever of the non-zero classes has
        the highest predicted probability.

    In the following definitions, P = number of evaluation pairs and K = number
    of classes.

    :param class_probability_matrix: See documentation for
        `_check_evaluation_pairs`.
    :param observed_labels: See doc for `_check_evaluation_pairs`.
    :param threshold_arg: See documentation for
        `model_evaluation.get_binarization_thresholds`.  Determines which
        thresholds will be tried.
    :param criterion_function: Criterion to be either minimized or maximized.
        This must be a function that takes input `contingency_table_as_dict` and
        returns a single float.  See `model_evaluation.get_csi` for an example.
    :param optimization_direction: Direction in which criterion function is
        optimized.  Options are "min" and "max".
    :param forecast_precision_for_thresholds: See documentation for
        `model_evaluation.get_binarization_thresholds`.  Determines which
        thresholds will be tried.
    :return: best_threshold: Best binarization threshold.
    :return: best_criterion_value: Value of criterion function at said
        threshold.
    """

    _check_evaluation_pairs(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels)

    error_checking.assert_is_string(optimization_direction)
    if optimization_direction not in VALID_OPTIMIZATION_DIRECTIONS:
        error_string = (
            '\n\n{0:s}\nValid optimization directions (listed above) do not '
            'include "{1:s}".').format(VALID_OPTIMIZATION_DIRECTIONS,
                                       optimization_direction)
        raise ValueError(error_string)

    possible_thresholds = model_eval.get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=class_probability_matrix[:, 0],
        unique_forecast_precision=forecast_precision_for_thresholds)

    num_thresholds = len(possible_thresholds)
    criterion_values = numpy.full(num_thresholds, numpy.nan)

    for i in range(num_thresholds):
        these_predicted_labels = model_eval.binarize_forecast_probs(
            forecast_probabilities=class_probability_matrix[:, 0],
            binarization_threshold=possible_thresholds[i])

        these_predicted_labels = numpy.invert(
            these_predicted_labels.astype(bool)).astype(int)

        this_contingency_table_as_dict = model_eval.get_contingency_table(
            forecast_labels=these_predicted_labels,
            observed_labels=(observed_labels > 0).astype(int))

        criterion_values[i] = criterion_function(this_contingency_table_as_dict)

    if optimization_direction == MAX_OPTIMIZATION_DIRECTION:
        best_criterion_value = numpy.nanmax(criterion_values)
        best_probability_threshold = possible_thresholds[
            numpy.nanargmax(criterion_values)]
    else:
        best_criterion_value = numpy.nanmin(criterion_values)
        best_probability_threshold = possible_thresholds[
            numpy.nanargmin(criterion_values)]

    return best_probability_threshold, best_criterion_value


def determinize_probabilities(class_probability_matrix, binarization_threshold):
    """Determinizes probabilistic predictions.

    P = number of evaluation pairs

    :param class_probability_matrix: See documentation for
        `_check_evaluation_pairs`.
    :param binarization_threshold: See documentation for
        `find_best_binarization_threshold`.
    :return: predicted_labels: length-P numpy array of predicted class labels
        (integers).
    """

    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)
    error_checking.assert_is_geq(binarization_threshold, 0.)
    error_checking.assert_is_leq(binarization_threshold, 1.)

    num_evaluation_pairs = class_probability_matrix.shape[0]
    predicted_labels = numpy.full(num_evaluation_pairs, -1, dtype=int)

    for i in range(num_evaluation_pairs):
        if class_probability_matrix[i, 0] >= binarization_threshold:
            predicted_labels[i] = 0
            continue

        predicted_labels[i] = 1 + numpy.argmax(class_probability_matrix[i, 1:])

    return predicted_labels


def get_contingency_table(predicted_labels, observed_labels, num_classes):
    """Creates either binary or multi-class contingency table.

    P = number of evaluation pairs
    K = number of classes

    :param predicted_labels: length-P numpy array of predicted class labels
        (integers).
    :param observed_labels: length-P numpy array of true class labels
        (integers).
    :param num_classes: Number of classes.
    :return: contingency_table_as_matrix: K-by-K numpy array.
        contingency_table_as_matrix[i, j] is the number of examples for which
        the predicted label is i and the true label is j.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_greater(num_classes, 2)

    error_checking.assert_is_numpy_array(predicted_labels, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(predicted_labels)
    error_checking.assert_is_geq_numpy_array(predicted_labels, 0)
    error_checking.assert_is_less_than_numpy_array(
        predicted_labels, num_classes)

    num_evaluation_pairs = len(predicted_labels)
    error_checking.assert_is_numpy_array(
        observed_labels, exact_dimensions=numpy.array([num_evaluation_pairs]))
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_less_than_numpy_array(observed_labels, num_classes)

    contingency_table_as_matrix = numpy.full(
        (num_classes, num_classes), -1, dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            contingency_table_as_matrix[i, j] = numpy.sum(
                numpy.logical_and(predicted_labels == i, observed_labels == j))

    return contingency_table_as_matrix


def get_accuracy(contingency_table_as_matrix):
    """Computes accuracy (either binary or multi-class).

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: accuracy: Accuracy (float in range 0...1).
    """

    error_checking.assert_is_numpy_array(
        contingency_table_as_matrix, num_dimensions=2)

    num_classes = contingency_table_as_matrix.shape[0]
    error_checking.assert_is_numpy_array(
        contingency_table_as_matrix,
        exact_dimensions=numpy.array([num_classes, num_classes]))

    error_checking.assert_is_integer_numpy_array(contingency_table_as_matrix)
    error_checking.assert_is_geq_numpy_array(contingency_table_as_matrix, 0)

    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)
    num_correct_pairs = numpy.trace(contingency_table_as_matrix)
    return float(num_correct_pairs) / num_evaluation_pairs


def get_peirce_score(contingency_table_as_matrix):
    """Computes Peirce score (either binary or multi-class).

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: peirce_score: Peirce score (float in range -1...1).
    """

    _check_contingency_table(contingency_table_as_matrix)

    num_classes = contingency_table_as_matrix.shape[0]
    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)

    first_numerator_term = 0
    second_numerator_term = 0
    denominator_term = 0

    for i in range(num_classes):
        first_numerator_term += contingency_table_as_matrix[i, i]

        second_numerator_term += (
            _get_num_predictions_in_class(contingency_table_as_matrix, i) *
            _get_num_true_labels_in_class(contingency_table_as_matrix, i))

        denominator_term += _get_num_true_labels_in_class(
            contingency_table_as_matrix, i)**2

    first_numerator_term = float(first_numerator_term) / num_evaluation_pairs
    second_numerator_term = float(
        second_numerator_term) / num_evaluation_pairs**2
    denominator = _non_zero(
        1. - float(denominator_term) / num_evaluation_pairs**2)

    return (first_numerator_term - second_numerator_term) / denominator


def get_heidke_score(contingency_table_as_matrix):
    """Computes Heidke score (either binary or multi-class).

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: heidke_score: Heidke score (float in range -inf...1).
    """

    _check_contingency_table(contingency_table_as_matrix)

    num_classes = contingency_table_as_matrix.shape[0]
    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)

    first_numerator_term = 0
    second_numerator_term = 0

    for i in range(num_classes):
        first_numerator_term += contingency_table_as_matrix[i, i]
        second_numerator_term += (
            _get_num_predictions_in_class(contingency_table_as_matrix, i) *
            _get_num_true_labels_in_class(contingency_table_as_matrix, i))

    first_numerator_term = float(first_numerator_term) / num_evaluation_pairs
    second_numerator_term = float(
        second_numerator_term) / num_evaluation_pairs**2
    denominator = _non_zero(1. - second_numerator_term)

    return (first_numerator_term - second_numerator_term) / denominator


def get_gerrity_score(contingency_table_as_matrix):
    """Computes Gerrity score (either binary or multi-class).

    The full equations are here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_table_as_matrix: See documentation for
        `_check_contingency_table`.
    :return: gerrity_score: Gerrity score (float in range -1...1).
    """

    s_matrix = _get_s_for_gerrity_score(contingency_table_as_matrix)
    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)
    return numpy.sum(
        contingency_table_as_matrix * s_matrix) / num_evaluation_pairs


def write_evaluation_results(
        class_probability_matrix, observed_labels, binarization_threshold,
        accuracy, peirce_score, heidke_score, gerrity_score, binary_pod,
        binary_pofd, binary_success_ratio, binary_focn, binary_accuracy,
        binary_csi, binary_frequency_bias, auc_by_class,
        scikit_learn_auc_by_class, pickle_file_name):
    """Writes evaluation results to Pickle file.

    P = number of evaluation pairs
    K = number of classes

    :param class_probability_matrix: P-by-K numpy array of floats.
        class_probability_matrix[i, k] is the predicted probability that the
        [i]th example belongs to the [k]th class.
    :param observed_labels: length-P numpy array of integers.  If
        observed_labels[i] = k, the [i]th example truly belongs to the [k]th
        class.
    :param binarization_threshold: Best threshold for discriminating between
        front and no front.  For details, see
        `find_best_binarization_threshold`.
    :param accuracy: Accuracy.
    :param peirce_score: Peirce score.
    :param heidke_score: Heidke score.
    :param gerrity_score: Gerrity score.
    :param binary_pod: Binary (front vs. no front) probability of detection.
    :param binary_pofd: Binary probability of false detection.
    :param binary_success_ratio: Binary success ratio.
    :param binary_focn: Binary frequency of correct nulls.
    :param binary_accuracy: Binary accuracy.
    :param binary_csi: Binary critical success index.
    :param binary_frequency_bias: Binary frequency bias.
    :param auc_by_class: length-K numpy array with area under one-vs-all ROC
        curve for each class (calculated by GewitterGefahr).
    :param scikit_learn_auc_by_class: Same but calculated by scikit-learn.
    :param pickle_file_name: Path to output file.
    """

    evaluation_dict = {
        CLASS_PROBABILITY_MATRIX_KEY: class_probability_matrix,
        OBSERVED_LABELS_KEY: observed_labels,
        BINARIZATION_THRESHOLD_KEY: binarization_threshold,
        ACCURACY_KEY: accuracy,
        PEIRCE_SCORE_KEY: peirce_score,
        HEIDKE_SCORE_KEY: heidke_score,
        GERRITY_SCORE_KEY: gerrity_score,
        BINARY_POD_KEY: binary_pod,
        BINARY_POFD_KEY: binary_pofd,
        BINARY_SUCCESS_RATIO_KEY: binary_success_ratio,
        BINARY_FOCN_KEY: binary_focn,
        BINARY_ACCURACY_KEY: binary_accuracy,
        BINARY_CSI_KEY: binary_csi,
        BINARY_FREQUENCY_BIAS_KEY: binary_frequency_bias,
        AUC_BY_CLASS_KEY: auc_by_class,
        SCIKIT_LEARN_AUC_BY_CLASS_KEY: scikit_learn_auc_by_class
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(evaluation_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_evaluation_results(pickle_file_name):
    """Reads evaluation results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: evaluation_dict: Dictionary with all keys in the list
        `EVALUATION_DICT_KEYS`.
    :raises: ValueError: if dictionary does not contain all keys in the list
        `EVALUATION_DICT_KEYS`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    evaluation_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    expected_keys_as_set = set(EVALUATION_DICT_KEYS)
    actual_keys_as_set = set(evaluation_dict.keys())
    if not set(expected_keys_as_set).issubset(actual_keys_as_set):
        error_string = (
            '\n\n{0:s}\nExpected keys are listed above.  Keys found in file '
            '("{1:s}") are listed below.  Some expected keys were not found.'
            '\n{2:s}\n').format(EVALUATION_DICT_KEYS, pickle_file_name,
                                evaluation_dict.keys())

        raise ValueError(error_string)

    return evaluation_dict
