"""Methods for evaluating probabilistic binary or multiclass classification."""

import numpy
import xarray
from gewittergefahr.gg_utils import model_evaluation as gg_evaluation
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_CLASSES = 3
NUM_BINS_FOR_RELIABILITY = 20
NUM_DETERMINIZATION_THRESHOLDS = 1001

BOOTSTRAP_REPLICATE_DIM = 'bootstrap_replicate'
DETERMINIZN_THRESHOLD_DIM = 'determinizn_threshold'
PROBABILITY_BIN_DIM = 'probability_bin'
PREDICTED_CLASS_DIM = 'predicted_class'
OBSERVED_CLASS_DIM = 'observed_class'
EXAMPLE_DIM = 'example'

ACCURACY_KEY = 'accuracy'
PEIRCE_SCORE_KEY = 'peirce_score'
HEIDKE_SCORE_KEY = 'heidke_score'
GERRITY_SCORE_KEY = 'gerrity_score'
BINARY_POD_KEY = 'binary_pod'
BINARY_POFD_KEY = 'binary_pofd'
BINARY_FAR_KEY = 'binary_far'
BINARY_FREQ_BIAS_KEY = 'binary_freq_bias'
CSI_KEY = 'critical_success_index'

AREA_UNDER_ROCC_KEY = 'area_under_roc_curve'
AREA_UNDER_PD_KEY = 'area_under_perf_diagram'

MEAN_PROBABILITY_KEY = 'mean_probability'
EVENT_FREQUENCY_KEY = 'event_frequency'
NUM_EXAMPLES_KEY = 'num_examples'
RESOLUTION_KEY = 'resolution'
RELIABILITY_KEY = 'reliability'
BSS_KEY = 'brier_skill_score'

BEST_THRESHOLD_KEY = 'best_determinization_threshold'
CONTINGENCY_MATRIX_KEY = 'contingency_matrix'

CLASS_PROBABILITY_KEY = 'class_probability'
OBSERVED_LABEL_KEY = 'observed_label'
CLIMO_COUNT_KEY = 'climo_count'


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
    :return: all_thresholds: 1-D numpy array with all thresholds attempted.
    """

    check_predictions_and_obs(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels)

    all_thresholds = gg_evaluation.get_binarization_thresholds(
        threshold_arg=NUM_DETERMINIZATION_THRESHOLDS
    )
    num_thresholds = len(all_thresholds)
    score_by_threshold = numpy.full(num_thresholds, numpy.nan)

    for i in range(num_thresholds):
        if numpy.mod(i, 10) == 0:
            print((
                'Have tried {0:d} of {1:d} determinization thresholds...'
            ).format(
                i, num_thresholds
            ))

        these_predicted_labels = determinize_predictions(
            class_probability_matrix=class_probability_matrix,
            threshold=all_thresholds[i]
        )

        this_contingency_matrix = get_contingency_table(
            predicted_labels=these_predicted_labels,
            observed_labels=observed_labels)

        score_by_threshold[i] = scoring_function(this_contingency_matrix)

    print('Have tried all {0:d} determinization thresholds!'.format(
        num_thresholds
    ))

    best_score = numpy.nanmax(score_by_threshold)
    best_threshold = all_thresholds[numpy.nanargmax(score_by_threshold)]

    return best_threshold, best_score, all_thresholds


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


def run_evaluation(
        class_probability_matrix, observed_labels, best_determinizn_threshold,
        all_determinizn_thresholds, climo_counts, bootstrap_rep_index):
    """Runs full model evaluation.

    T = number of determinization thresholds

    :param class_probability_matrix: See doc for `check_predictions_and_obs`.
    :param observed_labels: Same.
    :param best_determinizn_threshold: Best determinization threshold.
    :param all_determinizn_thresholds: length-T numpy array of determinization
        thresholds.
    :param climo_counts: length-3 numpy array with climatological count for each
        class (used in attributes diagrams).
    :param bootstrap_rep_index: Index for this bootstrap replicate.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    check_predictions_and_obs(class_probability_matrix=class_probability_matrix,
                              observed_labels=observed_labels)

    error_checking.assert_is_numpy_array(
        all_determinizn_thresholds, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(all_determinizn_thresholds, 0.)
    error_checking.assert_is_leq_numpy_array(all_determinizn_thresholds, 1.01)
    error_checking.assert_is_geq(best_determinizn_threshold, 0.)
    error_checking.assert_is_leq(best_determinizn_threshold, 1.01)

    error_checking.assert_is_numpy_array(
        climo_counts, exact_dimensions=numpy.array([NUM_CLASSES], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(climo_counts)
    error_checking.assert_is_greater_numpy_array(climo_counts, 0)

    error_checking.assert_is_integer(bootstrap_rep_index)
    error_checking.assert_is_geq(bootstrap_rep_index, 0)

    climo_frequencies = climo_counts.astype(float) / numpy.sum(climo_counts)
    num_thresholds = len(all_determinizn_thresholds)

    # Do calculations.
    these_dim = (1, num_thresholds)
    accuracy_matrix = numpy.full(these_dim, numpy.nan)
    peirce_score_matrix = numpy.full(these_dim, numpy.nan)
    heidke_score_matrix = numpy.full(these_dim, numpy.nan)
    gerrity_score_matrix = numpy.full(these_dim, numpy.nan)
    binary_pod_matrix = numpy.full(these_dim, numpy.nan)
    binary_pofd_matrix = numpy.full(these_dim, numpy.nan)
    binary_far_matrix = numpy.full(these_dim, numpy.nan)
    binary_freq_bias_matrix = numpy.full(these_dim, numpy.nan)
    csi_matrix = numpy.full(these_dim, numpy.nan)
    contingency_table_matrix = numpy.full(
        (1, num_thresholds, NUM_CLASSES, NUM_CLASSES), -1, dtype=int
    )

    for i in range(num_thresholds):
        if numpy.mod(i, 10) == 0:
            print((
                'Have computed scores for {0:d} of {1:d} thresholds...'
            ).format(
                i, num_thresholds
            ))

        these_predicted_labels = determinize_predictions(
            class_probability_matrix=class_probability_matrix,
            threshold=all_determinizn_thresholds[i]
        )

        contingency_table_matrix[0, i, ...] = get_contingency_table(
            predicted_labels=these_predicted_labels,
            observed_labels=observed_labels)

        this_ct_matrix = contingency_table_matrix[0, i, ...]

        accuracy_matrix[0, i] = get_accuracy(this_ct_matrix)
        peirce_score_matrix[0, i] = get_peirce_score(this_ct_matrix)
        heidke_score_matrix[0, i] = get_heidke_score(this_ct_matrix)
        gerrity_score_matrix[0, i] = get_gerrity_score(this_ct_matrix)
        binary_pod_matrix[0, i] = get_binary_pod(this_ct_matrix)
        binary_pofd_matrix[0, i] = get_binary_pofd(this_ct_matrix)
        binary_far_matrix[0, i] = get_binary_far(this_ct_matrix)
        binary_freq_bias_matrix[0, i] = get_binary_frequency_bias(
            this_ct_matrix
        )
        csi_matrix[0, i] = get_csi(this_ct_matrix)

    print('Have computed scores for all {0:d} thresholds!'.format(
        num_thresholds
    ))

    auc = gg_evaluation.get_area_under_roc_curve(
        pod_by_threshold=binary_pod_matrix[0, :],
        pofd_by_threshold=binary_pofd_matrix[0, :]
    )
    aupd = gg_evaluation.get_area_under_perf_diagram(
        pod_by_threshold=binary_pod_matrix[0, :],
        success_ratio_by_threshold=1 - binary_far_matrix[0, :]
    )

    these_dim = (1, NUM_CLASSES, NUM_BINS_FOR_RELIABILITY)
    mean_probability_matrix = numpy.full(these_dim, numpy.nan)
    event_frequency_matrix = numpy.full(these_dim, numpy.nan)
    num_examples_matrix = numpy.full(these_dim, -1, dtype=int)

    these_dim = (1, NUM_CLASSES)
    reliability_matrix = numpy.full(these_dim, numpy.nan)
    resolution_matrix = numpy.full(these_dim, numpy.nan)
    bss_matrix = numpy.full(these_dim, numpy.nan)

    for k in range(NUM_CLASSES):
        these_mean_probs, these_event_freqs, these_num_examples = (
            gg_evaluation.get_points_in_reliability_curve(
                forecast_probabilities=class_probability_matrix[:, k],
                observed_labels=(observed_labels == k).astype(int),
                num_forecast_bins=NUM_BINS_FOR_RELIABILITY
            )
        )

        mean_probability_matrix[0, k, :] = these_mean_probs
        event_frequency_matrix[0, k, :] = these_event_freqs
        num_examples_matrix[0, k, :] = these_num_examples

        this_bss_dict = gg_evaluation.get_brier_skill_score(
            mean_forecast_prob_by_bin=these_mean_probs,
            mean_observed_label_by_bin=these_event_freqs,
            num_examples_by_bin=these_num_examples,
            climatology=climo_frequencies[k]
        )

        reliability_matrix[0, k] = this_bss_dict[gg_evaluation.RELIABILITY_KEY]
        resolution_matrix[0, k] = this_bss_dict[gg_evaluation.RESOLUTION_KEY]
        bss_matrix[0, k] = this_bss_dict[gg_evaluation.BSS_KEY]

    # Add results to xarray table.
    these_dim = (BOOTSTRAP_REPLICATE_DIM, DETERMINIZN_THRESHOLD_DIM)
    main_data_dict = {
        ACCURACY_KEY: (these_dim, accuracy_matrix),
        PEIRCE_SCORE_KEY: (these_dim, peirce_score_matrix),
        HEIDKE_SCORE_KEY: (these_dim, heidke_score_matrix),
        GERRITY_SCORE_KEY: (these_dim, gerrity_score_matrix),
        BINARY_POD_KEY: (these_dim, binary_pod_matrix),
        BINARY_POFD_KEY: (these_dim, binary_pofd_matrix),
        BINARY_FAR_KEY: (these_dim, binary_far_matrix),
        BINARY_FREQ_BIAS_KEY: (these_dim, binary_freq_bias_matrix),
        CSI_KEY: (these_dim, csi_matrix)
    }

    new_dict = {
        AREA_UNDER_ROCC_KEY: (BOOTSTRAP_REPLICATE_DIM, numpy.array([auc])),
        AREA_UNDER_PD_KEY: (BOOTSTRAP_REPLICATE_DIM, numpy.array([aupd])),
        BEST_THRESHOLD_KEY: (
            BOOTSTRAP_REPLICATE_DIM, numpy.array([best_determinizn_threshold])
        )
    }
    main_data_dict.update(new_dict)

    these_dim = (BOOTSTRAP_REPLICATE_DIM, OBSERVED_CLASS_DIM)
    new_dict = {
        RESOLUTION_KEY: (these_dim, resolution_matrix),
        RELIABILITY_KEY: (these_dim, reliability_matrix),
        BSS_KEY: (these_dim, bss_matrix)
    }
    main_data_dict.update(new_dict)

    these_dim = (
        BOOTSTRAP_REPLICATE_DIM, OBSERVED_CLASS_DIM, PROBABILITY_BIN_DIM
    )
    new_dict = {
        MEAN_PROBABILITY_KEY: (these_dim, mean_probability_matrix),
        EVENT_FREQUENCY_KEY: (these_dim, event_frequency_matrix),
        NUM_EXAMPLES_KEY: (these_dim, num_examples_matrix)
    }
    main_data_dict.update(new_dict)

    these_dim = (
        BOOTSTRAP_REPLICATE_DIM, DETERMINIZN_THRESHOLD_DIM,
        PREDICTED_CLASS_DIM, OBSERVED_CLASS_DIM
    )
    new_dict = {
        CONTINGENCY_MATRIX_KEY: (these_dim, contingency_table_matrix)
    }
    main_data_dict.update(new_dict)

    new_dict = {
        CLASS_PROBABILITY_KEY: (
            (EXAMPLE_DIM, OBSERVED_CLASS_DIM), class_probability_matrix
        ),
        OBSERVED_LABEL_KEY: (EXAMPLE_DIM, observed_labels),
        CLIMO_COUNT_KEY: (OBSERVED_CLASS_DIM, climo_counts)
    }
    main_data_dict.update(new_dict)

    num_examples_total = len(observed_labels)
    example_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )
    class_indices = numpy.linspace(
        0, NUM_CLASSES - 1, num=NUM_CLASSES, dtype=int
    )
    bin_indices = numpy.linspace(
        0, NUM_BINS_FOR_RELIABILITY - 1, num=NUM_BINS_FOR_RELIABILITY, dtype=int
    )
    bootstrap_replicate_indices = numpy.array([bootstrap_rep_index], dtype=int)

    metadata_dict = {
        BOOTSTRAP_REPLICATE_DIM: bootstrap_replicate_indices,
        DETERMINIZN_THRESHOLD_DIM: all_determinizn_thresholds,
        PROBABILITY_BIN_DIM: bin_indices,
        PREDICTED_CLASS_DIM: class_indices,
        OBSERVED_CLASS_DIM: class_indices,
        EXAMPLE_DIM: example_indices
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)


def write_file(result_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param result_table_xarray: xarray table produced by `run_evaluation`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    # result_table_xarray.to_netcdf(
    #     path=netcdf_file_name, mode='w', format='NETCDF3_64BIT_OFFSET')

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT')


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table produced by `run_evaluation`.
    """

    return xarray.open_dataset(netcdf_file_name)
