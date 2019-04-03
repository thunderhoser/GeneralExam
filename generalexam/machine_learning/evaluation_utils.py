"""Methods for evaluating probabilistic binary or multiclass classification."""

import pickle
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression

NARR_TIME_INTERVAL_SECONDS = 10800
DEFAULT_FORECAST_PRECISION = 1e-3
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

MIN_OPTIMIZATION_DIRECTION = 'min'
MAX_OPTIMIZATION_DIRECTION = 'max'
VALID_OPTIMIZATION_DIRECTIONS = [
    MIN_OPTIMIZATION_DIRECTION, MAX_OPTIMIZATION_DIRECTION
]

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
AUPD_BY_CLASS_KEY = 'aupd_by_class'
RELIABILITY_BY_CLASS_KEY = 'reliability_by_class'
BSS_BY_CLASS_KEY = 'bss_by_class'

REQUIRED_KEYS = [
    CLASS_PROBABILITY_MATRIX_KEY, OBSERVED_LABELS_KEY,
    BINARIZATION_THRESHOLD_KEY, ACCURACY_KEY, PEIRCE_SCORE_KEY,
    HEIDKE_SCORE_KEY, GERRITY_SCORE_KEY, BINARY_POD_KEY, BINARY_POFD_KEY,
    BINARY_SUCCESS_RATIO_KEY, BINARY_FOCN_KEY, BINARY_ACCURACY_KEY,
    BINARY_CSI_KEY, BINARY_FREQUENCY_BIAS_KEY, AUC_BY_CLASS_KEY,
    SCIKIT_LEARN_AUC_BY_CLASS_KEY, AUPD_BY_CLASS_KEY, RELIABILITY_BY_CLASS_KEY,
    BSS_BY_CLASS_KEY
]

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


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


def _get_random_sample_points(
        num_points, for_downsized_examples, narr_mask_matrix=None,
        random_seed=None):
    """Samples random points from NARR grid.

    M = number of rows in NARR grid
    N = number of columns in NARR grid
    P = number of points sampled

    :param num_points: Number of points to sample.
    :param for_downsized_examples: Boolean flag.  If True, this method will
        sample center points for downsized images.  If False, will sample
        evaluation points from a full-size image.
    :param narr_mask_matrix: M-by-N numpy array of integers (0 or 1).  If
        narr_mask_matrix[i, j] = 0, cell [i, j] in the full grid will never be
        sampled.  If `narr_mask_matrix is None`, any grid cell can be sampled.
    :param random_seed: Seed (input to `numpy.random.seed`).
    :return: row_indices: length-P numpy array with row indices of sampled
        points.
    :return: column_indices: length-P numpy array with column indices of sampled
        points.
    """

    if for_downsized_examples:
        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    else:
        num_grid_rows = (
            ml_utils.LAST_NARR_ROW_FOR_FCN_INPUT -
            ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT + 1
        )
        num_grid_columns = (
            ml_utils.LAST_NARR_COLUMN_FOR_FCN_INPUT -
            ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT + 1
        )
        narr_mask_matrix = None

    if narr_mask_matrix is None:
        num_grid_cells = num_grid_rows * num_grid_columns
        possible_linear_indices = numpy.linspace(
            0, num_grid_cells - 1, num=num_grid_cells, dtype=int)
    else:
        error_checking.assert_is_integer_numpy_array(narr_mask_matrix)
        error_checking.assert_is_geq_numpy_array(narr_mask_matrix, 0)
        error_checking.assert_is_leq_numpy_array(narr_mask_matrix, 1)

        these_expected_dim = numpy.array(
            [num_grid_rows, num_grid_columns], dtype=int)
        error_checking.assert_is_numpy_array(
            narr_mask_matrix, exact_dimensions=these_expected_dim)

        possible_linear_indices = numpy.where(
            numpy.ravel(narr_mask_matrix) == 1
        )[0]

    if random_seed is not None:
        numpy.random.seed(random_seed)

    linear_indices = numpy.random.choice(
        possible_linear_indices, size=num_points, replace=False)

    return numpy.unravel_index(
        linear_indices, (num_grid_rows, num_grid_columns)
    )


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


def check_evaluation_pairs(class_probability_matrix, observed_labels):
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


def create_eval_pairs_for_cnn(
        model_object, top_predictor_dir_name, top_gridded_front_dir_name,
        first_time_unix_sec, last_time_unix_sec, num_times,
        num_examples_per_time, pressure_levels_mb, predictor_names,
        normalization_type_string, dilation_distance_metres,
        isotonic_model_object_by_class=None, mask_matrix=None,
        random_seed=None):
    """Creates evaluation pairs for a CNN (convolutional neural net).

    An "evaluation pair" is a forecast-observation pair.  Keep in mind that a
    CNN does patch classification (as opposed to an FCN, which does semantic
    segmentation), so a CNN works on downsized examples.

    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param top_predictor_dir_name: See doc for
        `testing_io.create_downsized_examples`.
    :param top_gridded_front_dir_name: Same.
    :param first_time_unix_sec: First time in period.  Evaluation pairs will be
        drawn randomly from the period
        `first_time_unix_sec`...`last_time_unix_sec`.
    :param last_time_unix_sec: See above.
    :param num_times: Number of times to draw randomly from the period.
    :param num_examples_per_time: Number of examples (grid cells) to draw
        randomly from each time.
    :param pressure_levels_mb: See doc for
        `testing_io.create_downsized_examples`.
    :param predictor_names: Same.
    :param normalization_type_string: Same.
    :param dilation_distance_metres: Same.
    :param isotonic_model_object_by_class: See doc for
        `cnn.apply_model_to_full_grid`.
    :param mask_matrix: Same.
    :param random_seed: Seed (input to `numpy.random.seed`).
    :return: class_probability_matrix: See documentation for
        `check_evaluation_pairs`.
    :return: observed_labels: Same.
    """

    error_checking.assert_is_integer(num_times)
    error_checking.assert_is_greater(num_times, 0)
    error_checking.assert_is_integer(num_examples_per_time)
    error_checking.assert_is_greater(num_examples_per_time, 0)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    if random_seed is not None:
        numpy.random.seed(random_seed)

    numpy.random.shuffle(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[:num_times]

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, LOG_MESSAGE_TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    num_classes = cnn.model_to_num_classes(model_object)
    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    class_probability_matrix = numpy.full(
        (num_times, num_examples_per_time, num_classes), numpy.nan
    )
    observed_label_matrix = numpy.full(
        (num_times, num_examples_per_time), -1, dtype=int
    )

    this_random_seed = random_seed + 0

    for i in range(num_times):
        print 'Creating {0:d} evaluation pairs for {1:s}...'.format(
            num_examples_per_time, valid_time_strings[i])

        this_random_seed += 1
        these_row_indices, these_column_indices = _get_random_sample_points(
            num_points=num_examples_per_time, for_downsized_examples=True,
            narr_mask_matrix=mask_matrix, random_seed=this_random_seed)

        this_dict = testing_io.create_downsized_examples(
            center_row_indices=these_row_indices,
            center_column_indices=these_column_indices,
            num_half_rows=num_half_rows, num_half_columns=num_half_columns,
            top_predictor_dir_name=top_predictor_dir_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            pressure_levels_mb=pressure_levels_mb,
            predictor_names=predictor_names,
            normalization_type_string=normalization_type_string,
            dilation_distance_metres=dilation_distance_metres,
            num_classes=num_classes)

        this_predictor_matrix = this_dict[testing_io.PREDICTOR_MATRIX_KEY]
        observed_label_matrix[i, :] = this_dict[testing_io.TARGET_VALUES_KEY]
        class_probability_matrix[i, ...] = model_object.predict(
            this_predictor_matrix, batch_size=num_examples_per_time)

    class_probability_matrix = numpy.reshape(
        class_probability_matrix,
        (num_times * num_examples_per_time, num_classes)
    )
    observed_labels = numpy.reshape(
        observed_label_matrix, observed_label_matrix.size)

    if isotonic_model_object_by_class is not None:
        class_probability_matrix = (
            isotonic_regression.apply_model_for_each_class(
                orig_class_probability_matrix=class_probability_matrix,
                observed_labels=observed_labels,
                model_object_by_class=isotonic_model_object_by_class)
        )

    return class_probability_matrix, observed_labels


def find_best_binarization_threshold(
        class_probability_matrix, observed_labels, threshold_arg,
        criterion_function, optimization_direction=MAX_OPTIMIZATION_DIRECTION,
        forecast_precision_for_thresholds=
        DEFAULT_FORECAST_PRECISION):
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
        `check_evaluation_pairs`.
    :param observed_labels: See doc for `check_evaluation_pairs`.
    :param threshold_arg: See documentation for
        `model_evaluation.get_binarization_thresholds`.  Determines which
        thresholds will be tried.
    :param criterion_function: Criterion to be either minimized or maximized.
        This must be a function that takes input `contingency_table_as_matrix`
        and returns a single float.  See `get_gerrity_score` in this module for
        an example.
    :param optimization_direction: Direction in which criterion function is
        optimized.  Options are "min" and "max".
    :param forecast_precision_for_thresholds: See documentation for
        `model_evaluation.get_binarization_thresholds`.  Determines which
        thresholds will be tried.
    :return: best_threshold: Best binarization threshold.
    :return: best_criterion_value: Value of criterion function at said
        threshold.
    """

    check_evaluation_pairs(
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

    # for i in range(num_thresholds):
    #     these_predicted_labels = model_eval.binarize_forecast_probs(
    #         forecast_probabilities=class_probability_matrix[:, 0],
    #         binarization_threshold=possible_thresholds[i])
    #
    #     these_predicted_labels = numpy.invert(
    #         these_predicted_labels.astype(bool)).astype(int)
    #
    #     this_contingency_table_as_dict = model_eval.get_contingency_table(
    #         forecast_labels=these_predicted_labels,
    #         observed_labels=(observed_labels > 0).astype(int))
    #
    #     criterion_values[i] = criterion_function(
    #         this_contingency_table_as_dict)

    for i in range(num_thresholds):
        print i

        these_predicted_labels = determinize_probabilities(
            class_probability_matrix=class_probability_matrix,
            binarization_threshold=possible_thresholds[i])

        this_contingency_table_as_matrix = get_contingency_table(
            predicted_labels=these_predicted_labels,
            observed_labels=observed_labels,
            num_classes=class_probability_matrix.shape[1])

        criterion_values[i] = criterion_function(
            this_contingency_table_as_matrix)

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
        `check_evaluation_pairs`.
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
    error_checking.assert_is_leq(binarization_threshold, 1.01)

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
        observed_labels,
        exact_dimensions=numpy.array([num_evaluation_pairs], dtype=int)
    )

    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_less_than_numpy_array(observed_labels, num_classes)

    contingency_table_as_matrix = numpy.full(
        (num_classes, num_classes), -1, dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            contingency_table_as_matrix[i, j] = numpy.sum(
                numpy.logical_and(predicted_labels == i, observed_labels == j)
            )

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
        contingency_table_as_matrix * s_matrix
    ) / num_evaluation_pairs


def get_multiclass_csi(contingency_table_as_matrix):
    """Computes multiclass critical success index.

    This works for binary classification as well.  In the multiclass setting,
    "correct nulls" are evaluation pairs where both forecast and observed class
     are 0.

    :param contingency_table_as_matrix: See doc for `_check_contingency_table`.
    :return: multiclass_csi: Multiclass CSI.
    """

    num_correct_nulls = contingency_table_as_matrix[0, 0]
    num_correct_forecasts = numpy.trace(contingency_table_as_matrix)
    num_evaluation_pairs = numpy.sum(contingency_table_as_matrix)

    return (
        float(num_correct_forecasts - num_correct_nulls) /
        float(num_evaluation_pairs - num_correct_nulls)
    )


def write_file(
        class_probability_matrix, observed_labels, binarization_threshold,
        accuracy, peirce_score, heidke_score, gerrity_score, binary_pod,
        binary_pofd, binary_success_ratio, binary_focn, binary_accuracy,
        binary_csi, binary_frequency_bias, auc_by_class,
        scikit_learn_auc_by_class, aupd_by_class, reliability_by_class,
        bss_by_class, pickle_file_name):
    """Writes results to Pickle file.

    P = number of evaluation pairs (forecast-observation pairs)
    K = number of classes

    :param class_probability_matrix: P-by-K numpy array of class probabilities.
    :param observed_labels: length-P numpy array of class labels (integers in
        0...[K - 1]).
    :param binarization_threshold: Probability threshold (on no-front
        probability) used to discriminate between front and no front.
    :param accuracy: Accuracy.
    :param peirce_score: Peirce score.
    :param heidke_score: Heidke score.
    :param gerrity_score: Gerrity score.
    :param binary_pod: Binary probability of detection.
    :param binary_pofd: Binary probability of false detection.
    :param binary_success_ratio: Binary success ratio.
    :param binary_focn: Binary frequency of correct nulls.
    :param binary_accuracy: Binary accuracy.
    :param binary_csi: Binary CSI.
    :param binary_frequency_bias: Binary frequency bias.
    :param auc_by_class: length-K numpy array with AUC (area under ROC curve)
        for each class.
    :param scikit_learn_auc_by_class: Same but according to scikit-learn.
    :param aupd_by_class: length-K numpy array with AUPD (area under performance
        diagram) for each class.
    :param reliability_by_class: length-K numpy array of reliabilities.
    :param bss_by_class: length-K numpy array of Brier skill scores.
    :param pickle_file_name: Path to output file.
    """

    result_dict = {
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
        SCIKIT_LEARN_AUC_BY_CLASS_KEY: scikit_learn_auc_by_class,
        AUPD_BY_CLASS_KEY: aupd_by_class,
        RELIABILITY_BY_CLASS_KEY: reliability_by_class,
        BSS_BY_CLASS_KEY: bss_by_class
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(result_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: result_dict: Dictionary with all keys listed in `write_file`.
    :raises: ValueError: if dictionary does not contain all keys in
        `REQUIRED_KEYS`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    result_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(REQUIRED_KEYS) - set(result_dict.keys()))
    if len(missing_keys) == 0:
        return result_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_label_strings,
        y_tick_label_strings, colour_map_object=pyplot.cm.plasma,
        axes_object=None):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_label_strings: length-N list of labels for x-axis.
    :param y_tick_label_strings: length-M list of labels for y-axis.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If `axes_object is None`,
        will create new axes.
    :return: axes_object: See input doc.
    """

    error_checking.assert_is_real_numpy_array(score_matrix)
    error_checking.assert_is_numpy_array(score_matrix, num_dimensions=2)
    error_checking.assert_is_greater(max_colour_value, min_colour_value)

    num_grid_rows = score_matrix.shape[0]
    num_grid_columns = score_matrix.shape[1]

    error_checking.assert_is_string_list(x_tick_label_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(x_tick_label_strings),
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    error_checking.assert_is_string_list(y_tick_label_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(y_tick_label_strings),
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    score_matrix_to_plot = numpy.ma.masked_where(
        numpy.isnan(score_matrix), score_matrix)

    pyplot.imshow(
        score_matrix_to_plot, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value)

    x_tick_values = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float)
    y_tick_values = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float)

    pyplot.xticks(x_tick_values, x_tick_label_strings, rotation=90.)
    pyplot.yticks(y_tick_values, y_tick_label_strings)

    return axes_object
