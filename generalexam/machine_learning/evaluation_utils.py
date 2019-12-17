"""Methods for evaluating probabilistic binary or multiclass classification."""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_CLASSES = 3
NUM_DETERMINIZATION_THRESHOLDS = 1001

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

TICK_LABEL_FONT_SIZE = 35
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _check_contingency_table(contingency_matrix):
    """Checks contingency table for errors.

    :param contingency_matrix: 3-by-3 numpy array.  contingency_matrix[i, j] is
        the number of examples where the predicted label is i and correct label
        is j.
    """

    expected_dim = numpy.array([NUM_CLASSES, NUM_CLASSES], dtype=int)
    error_checking.assert_is_numpy_array(
        contingency_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_integer_numpy_array(contingency_matrix)
    error_checking.assert_is_geq_numpy_array(contingency_matrix, 0)


def _non_zero(input_value):
    """Makes input non-zero.

    :param input_value: Input.
    :return: output_value: Closest number to input that is outside of
        [-epsilon, epsilon], where epsilon is the machine limit for
        floating-point numbers.
    """

    epsilon = numpy.finfo(float).eps
    if input_value >= 0:
        return max([input_value, epsilon])

    return min([input_value, -epsilon])


def _num_examples_with_predicted_class(contingency_matrix, class_index):
    """Returns number of examples where a given class is predicted.

    This method returns number of examples where [k]th class is predicted, k
    being `class_index`.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :param class_index: See above.
    :return: num_examples: See above.
    """

    return numpy.sum(contingency_matrix[class_index, :])


def _num_examples_with_observed_class(contingency_matrix, class_index):
    """Returns number of examples where a given class is observed.

    This method returns number of examples where [k]th class is observed, k
    being `class_index`.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :param class_index: See above.
    :return: num_examples: See above.
    """

    return numpy.sum(contingency_matrix[:, class_index])


def _get_a_for_gerrity_score(contingency_matrix):
    """Returns vector a for Gerrity score.

    The equation for a is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: a_vector: See above.
    """

    num_classes = contingency_matrix.shape[0]
    num_examples = numpy.sum(contingency_matrix)

    num_examples_by_class = numpy.array([
        _num_examples_with_observed_class(contingency_matrix, i)
        for i in range(num_classes)
    ])
    cumulative_freq_by_class = numpy.cumsum(
        num_examples_by_class.astype(float) / num_examples
    )

    return (1. - cumulative_freq_by_class) / cumulative_freq_by_class


def _get_s_for_gerrity_score(contingency_matrix):
    """Returns matrix S for Gerrity score.

    The equation for S is here: http://www.bom.gov.au/wmo/lrfvs/gerrity.shtml

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: s_matrix: See above.
    """

    a_vector = _get_a_for_gerrity_score(contingency_matrix)
    a_vector_reciprocal = 1. / a_vector

    num_classes = contingency_matrix.shape[0]
    s_matrix = numpy.full((num_classes, num_classes), numpy.nan)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                s_matrix[i, j] = (
                    numpy.sum(a_vector_reciprocal[:i]) +
                    numpy.sum(a_vector[i:-1])
                )
                continue

            if i > j:
                s_matrix[i, j] = s_matrix[j, i]
                continue

            s_matrix[i, j] = (
                numpy.sum(a_vector_reciprocal[:i]) - (j - i) +
                numpy.sum(a_vector[j:-1])
            )

    return s_matrix / (num_classes - 1)


def _check_probabilities(class_probability_matrix, num_examples=None):
    """Error-checks probabilities.

    E = number of examples

    :param class_probability_matrix: E-by-3 numpy array of class probabilities.
    :param num_examples: Expected number of examples.
    """

    # TODO(thunderhoser): Verify that probabilities are MECE?

    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)

    if num_examples is None:
        error_checking.assert_is_numpy_array(
            class_probability_matrix, num_dimensions=2
        )
        num_examples = class_probability_matrix.shape[0]

    expected_dim = numpy.array([num_examples, NUM_CLASSES], dtype=int)
    error_checking.assert_is_numpy_array(
        class_probability_matrix, exact_dimensions=expected_dim
    )


def check_predictions_and_obs(observed_labels, class_probability_matrix=None,
                              predicted_labels=None):
    """Error-checks predictions and observations.

    E = number of examples

    :param observed_labels: length-E numpy array of observed labels (classes in
        0...2).
    :param class_probability_matrix: E-by-3 numpy array of class probabilities.
    :param predicted_labels: length-E numpy array of predicted labels (classes
        in 0...2).
    """

    error_checking.assert_is_numpy_array(observed_labels, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_less_than_numpy_array(observed_labels, NUM_CLASSES)

    num_examples = len(observed_labels)

    if class_probability_matrix is not None:
        _check_probabilities(class_probability_matrix=class_probability_matrix,
                             num_examples=num_examples)

    if predicted_labels is not None:
        error_checking.assert_is_integer_numpy_array(predicted_labels)
        error_checking.assert_is_geq_numpy_array(predicted_labels, 0)
        error_checking.assert_is_less_than_numpy_array(
            predicted_labels, NUM_CLASSES
        )

        expected_dim = numpy.array([num_examples], dtype=int)
        error_checking.assert_is_numpy_array(
            predicted_labels, exact_dimensions=expected_dim
        )


def determinize_predictions(class_probability_matrix, threshold):
    """Determinizes predictions.

    In other words, convert from probabilities to deterministic labels.

    E = number of examples
    K = number of classes

    :param class_probability_matrix: See doc for `check_predictions_and_obs`.
    :param threshold: Determinization threshold.  This is a probability
        threshold p* such that:

        If no-front probability >= p*, deterministic label = no front.

        If no-front probability < p*, deterministic label = max of other two
        probabilities.

    :return: predicted_labels: length-E numpy array of predicted class labels
        (integers in 0...[K - 1]).
    """

    _check_probabilities(
        class_probability_matrix=class_probability_matrix, num_examples=None
    )
    error_checking.assert_is_geq(threshold, 0.)
    error_checking.assert_is_leq(threshold, 1.01)

    num_examples = class_probability_matrix.shape[0]
    predicted_labels = numpy.full(num_examples, -1, dtype=int)

    for i in range(num_examples):
        if class_probability_matrix[i, 0] >= threshold:
            predicted_labels[i] = 0
            continue

        predicted_labels[i] = 1 + numpy.argmax(class_probability_matrix[i, 1:])

    return predicted_labels


def get_contingency_table(predicted_labels, observed_labels):
    """Creates contingency table.

    :param predicted_labels: See doc for `check_predictions_and_obs`.
    :param observed_labels: Same.
    :return: contingency_matrix: See doc for `_check_contingency_table`.
    """

    check_predictions_and_obs(observed_labels=observed_labels,
                              predicted_labels=predicted_labels)

    contingency_matrix = numpy.full(
        (NUM_CLASSES, NUM_CLASSES), -1, dtype=int
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            contingency_matrix[i, j] = numpy.sum(numpy.logical_and(
                predicted_labels == i, observed_labels == j
            ))

    return contingency_matrix


def find_best_determinization_threshold(
        class_probability_matrix, observed_labels, scoring_function):
    """Finds best probability threshold for determinizing predictions.

    The "best probability threshold" is that which maximizes the scoring
    function.

    :param class_probability_matrix: See doc for `check_predictions_and_obs`.
    :param observed_labels: Same.
    :param scoring_function: Scoring function.  Must have one input (contingency
        table as 3 x 3 numpy array) and one output (scalar value to be
        maximized).  `get_gerrity_score` in this file is an example of a valid
        scoring function.

    :return: best_threshold: Best probability threshold.
    :return: best_score: Score at best probability threshold.
    """

    check_predictions_and_obs(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels)

    thresholds = numpy.linspace(0., 1., num=NUM_DETERMINIZATION_THRESHOLDS)
    score_by_threshold = numpy.full(NUM_DETERMINIZATION_THRESHOLDS, numpy.nan)

    for k in range(NUM_DETERMINIZATION_THRESHOLDS):
        if numpy.mod(k, 10) == 0:
            print((
                'Have tried {0:d} of {1:d} determinization thresholds...'
            ).format(
                k, NUM_DETERMINIZATION_THRESHOLDS
            ))

        these_predicted_labels = determinize_predictions(
            class_probability_matrix=class_probability_matrix,
            threshold=thresholds[k]
        )

        this_contingency_matrix = get_contingency_table(
            predicted_labels=these_predicted_labels,
            observed_labels=observed_labels)

        score_by_threshold[k] = scoring_function(this_contingency_matrix)

    print('Have tried all {0:d} determinization thresholds!'.format(
        NUM_DETERMINIZATION_THRESHOLDS
    ))

    best_score = numpy.nanmax(score_by_threshold)
    best_threshold = thresholds[numpy.nanargmax(score_by_threshold)]

    return best_threshold, best_score


def get_accuracy(contingency_matrix):
    """Computes accuracy.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: accuracy: Accuracy (range 0...1).
    """

    _check_contingency_table(contingency_matrix)

    num_examples = numpy.sum(contingency_matrix)
    num_correct = numpy.trace(contingency_matrix)
    return float(num_correct) / num_examples


def get_peirce_score(contingency_matrix):
    """Computes Peirce score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: peirce_score: Peirce score (range -1...1).
    """

    _check_contingency_table(contingency_matrix)

    first_numerator_term = 0
    second_numerator_term = 0
    denominator_term = 0

    for i in range(NUM_CLASSES):
        first_numerator_term += contingency_matrix[i, i]

        second_numerator_term += (
            _num_examples_with_predicted_class(contingency_matrix, i) *
            _num_examples_with_observed_class(contingency_matrix, i)
        )

        denominator_term += (
            _num_examples_with_observed_class(contingency_matrix, i) ** 2
        )

    num_examples = numpy.sum(contingency_matrix)

    first_numerator_term = float(first_numerator_term) / num_examples
    second_numerator_term = float(second_numerator_term) / num_examples ** 2
    denominator = _non_zero(1. - float(denominator_term) / num_examples ** 2)

    return (first_numerator_term - second_numerator_term) / denominator


def get_heidke_score(contingency_matrix):
    """Computes Heidke score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: heidke_score: Heidke score (range -inf...1).
    """

    _check_contingency_table(contingency_matrix)

    first_numerator_term = 0
    second_numerator_term = 0

    for i in range(NUM_CLASSES):
        first_numerator_term += contingency_matrix[i, i]

        second_numerator_term += (
            _num_examples_with_predicted_class(contingency_matrix, i) *
            _num_examples_with_observed_class(contingency_matrix, i)
        )

    num_examples = numpy.sum(contingency_matrix)

    first_numerator_term = float(first_numerator_term) / num_examples
    second_numerator_term = (float(second_numerator_term) / num_examples**2)
    denominator = _non_zero(1. - second_numerator_term)

    return (first_numerator_term - second_numerator_term) / denominator


def get_gerrity_score(contingency_matrix):
    """Computes Gerrity score.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: gerrity_score: Gerrity score (range -1...1).
    """

    s_matrix = _get_s_for_gerrity_score(contingency_matrix)
    num_examples = numpy.sum(contingency_matrix)

    return numpy.sum(contingency_matrix * s_matrix) / num_examples


def get_binary_pod(contingency_matrix):
    """Computes binary probability of detection (POD).

    Binary POD = fraction of frontal examples that are correctly labeled.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: binary_pod: Binary POD.
    """

    _check_contingency_table(contingency_matrix)

    numerator = contingency_matrix[1, 1] + contingency_matrix[2, 2]
    denominator = (
        _num_examples_with_observed_class(contingency_matrix, 1) +
        _num_examples_with_observed_class(contingency_matrix, 2)
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_pofd(contingency_matrix):
    """Computes binary probability of false detection (POFD).

    Binary POFD = fraction of non-frontal examples that are labeled frontal.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: binary_pofd: Binary POFD.
    """

    _check_contingency_table(contingency_matrix)

    denominator = _num_examples_with_observed_class(contingency_matrix, 0)
    numerator = denominator - contingency_matrix[0, 0]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_far(contingency_matrix):
    """Computes binary false-alarm ratio (FAR).

    Binary FAR = fraction of examples labeled frontal that are incorrect.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: binary_far: Binary FAR.
    """

    _check_contingency_table(contingency_matrix)

    denominator = (
        _num_examples_with_predicted_class(contingency_matrix, 1) +
        _num_examples_with_predicted_class(contingency_matrix, 2)
    )
    numerator = denominator - (
        contingency_matrix[1, 1] + contingency_matrix[2, 2]
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_frequency_bias(contingency_matrix):
    """Computes binary frequency bias.

    Frequency bias = ratio of examples labeled frontal to ones that are actually
    frontal.

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: binary_freq_bias: Binary frequency bias.
    """

    _check_contingency_table(contingency_matrix)

    numerator = (
        _num_examples_with_predicted_class(contingency_matrix, 1) +
        _num_examples_with_predicted_class(contingency_matrix, 2)
    )
    denominator = (
        _num_examples_with_observed_class(contingency_matrix, 1) +
        _num_examples_with_observed_class(contingency_matrix, 2)
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_csi(contingency_matrix):
    """Computes critical success index (CSI).

    :param contingency_matrix: See doc for `_check_contingency_table`.
    :return: csi: Critical success index.
    """

    _check_contingency_table(contingency_matrix)

    numerator = contingency_matrix[1, 1] + contingency_matrix[2, 2]
    denominator = numpy.sum(contingency_matrix) - contingency_matrix[0, 0]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


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
    result_dict = pickle.load(pickle_file_handle, encoding='latin1')
    pickle_file_handle.close()

    missing_keys = list(set(REQUIRED_KEYS) - set(result_dict.keys()))
    if len(missing_keys) == 0:
        return result_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
