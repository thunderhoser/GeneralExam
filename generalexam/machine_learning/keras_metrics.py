"""Performance metrics used to monitor Keras model during training.

Keep in mind that these metrics have the following properties, which many users
may find undesirable:

[1] Used only to monitor the Keras model, not to serve as loss functions.
[2] Assume that the problem is binary classification (have not yet been
    generalized for multi-class problems).
[3] Do not binarize the forecast class probabilities.  Thus, the contingency-
    table elements used to compute all the metrics are generally non-integers.

--- DEFINITIONS ---

"Forecast probability" = predicted probability, from the Keras model, that the
true label is 1.

"Inverse forecast probability" = predicted probability, from the Keras model,
that the true label is 0.

--- NOTATION ---

Throughout this module, I will use the following letters to denote elements in
the contingency table.

a = number of true positives = sum of forecast probabilities for examples with
true label of 1

b = number of false positives = sum of forecast probabilities for examples with
true label of 0

c = number of false negatives = sum of inverse forecast probabilities for
examples with true label of 1

d = number of true negatives = sum of inverse forecast probabilities for
examples with true label of 0
"""

import keras.backend as K


def _get_num_true_positives(true_labels, forecast_probabilities):
    """Returns number of true positives (defined in docstring).

    Number of true positives = `a` in contingency table

    :param true_labels: tensor of true labels (integers).
    :param forecast_probabilities: equivalent-sized tensor of forecast
        probabilities (in range 0...1).
    :return: num_true_positives: Number of true positives.
    """

    num_tensor_dimensions = len(K.int_shape(true_labels))
    if num_tensor_dimensions == 2:
        return K.sum(K.clip(
            true_labels[:, 1] * forecast_probabilities[:, 1], 0., 1.))

    return K.sum(K.clip(
        true_labels * forecast_probabilities, 0., 1.))


def _get_num_false_positives(true_labels, forecast_probabilities):
    """Returns number of false positives (defined in docstring).

    Number of false positives = `b` in contingency table

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: num_false_positives: Number of false positives.
    """

    num_tensor_dimensions = len(K.int_shape(true_labels))
    if num_tensor_dimensions == 2:
        return K.sum(K.clip(
            (1 - true_labels[:, 1]) * forecast_probabilities[:, 1], 0., 1.))

    return K.sum(K.clip(
        (1 - true_labels) * forecast_probabilities, 0., 1.))


def _get_num_false_negatives(true_labels, forecast_probabilities):
    """Returns number of false negatives (defined in docstring).

    Number of false negatives = `c` in contingency table

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: num_false_negatives: Number of false negatives.
    """

    num_tensor_dimensions = len(K.int_shape(true_labels))
    if num_tensor_dimensions == 2:
        return K.sum(K.clip(
            true_labels[:, 1] * (1. - forecast_probabilities[:, 1]), 0., 1.))

    return K.sum(K.clip(
        true_labels * (1. - forecast_probabilities), 0., 1.))


def _get_num_true_negatives(true_labels, forecast_probabilities):
    """Returns number of true negatives (defined in docstring).

    Number of true negatives = `d` in contingency table

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: num_true_negatives: Number of true negatives.
    """

    num_tensor_dimensions = len(K.int_shape(true_labels))
    if num_tensor_dimensions == 2:
        return K.sum(K.clip(
            (1 - true_labels[:, 1]) *
            (1. - forecast_probabilities[:, 1]), 0., 1.))

    return K.sum(K.clip(
        (1 - true_labels) * (1. - forecast_probabilities), 0., 1.))


def accuracy(true_labels, forecast_probabilities):
    """Computes accuracy ([a + d] / [a + b + c + d]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: accuracy: Accuracy.
    """

    num_true_positives = _get_num_true_positives(
        true_labels, forecast_probabilities)
    num_false_positives = _get_num_false_positives(
        true_labels, forecast_probabilities)
    num_false_negatives = _get_num_false_negatives(
        true_labels, forecast_probabilities)
    num_true_negatives = _get_num_true_negatives(
        true_labels, forecast_probabilities)

    return (num_true_positives + num_true_negatives) / (
        num_true_positives + num_false_positives + num_false_negatives +
        num_true_negatives + K.epsilon())


def csi(true_labels, forecast_probabilities):
    """Computes critical success index (a / [a + b + c]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: critical_success_index: Critical success index.
    """

    num_true_positives = _get_num_true_positives(
        true_labels, forecast_probabilities)
    num_false_positives = _get_num_false_positives(
        true_labels, forecast_probabilities)
    num_false_negatives = _get_num_false_negatives(
        true_labels, forecast_probabilities)

    return num_true_positives / (
        num_true_positives + num_false_positives + num_false_negatives +
        K.epsilon())


def frequency_bias(true_labels, forecast_probabilities):
    """Computes frequency bias ([a + b] / [a + c]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: frequency_bias: Frequency bias.
    """

    num_true_positives = _get_num_true_positives(
        true_labels, forecast_probabilities)
    num_false_positives = _get_num_false_positives(
        true_labels, forecast_probabilities)
    num_false_negatives = _get_num_false_negatives(
        true_labels, forecast_probabilities)

    return (num_true_positives + num_false_positives) / (
        num_true_positives + num_false_negatives + K.epsilon())


def pod(true_labels, forecast_probabilities):
    """Computes probability of detection (a / [a + c]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: probability_of_detection: Probability of detection.
    """

    num_true_positives = _get_num_true_positives(
        true_labels, forecast_probabilities)
    num_false_negatives = _get_num_false_negatives(
        true_labels, forecast_probabilities)

    return num_true_positives / (
        num_true_positives + num_false_negatives + K.epsilon())


def fom(true_labels, forecast_probabilities):
    """Computes frequency of misses (c / [a + c]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: frequency_of_misses: Frequency of misses.
    """

    return 1. - pod(true_labels, forecast_probabilities)


def pofd(true_labels, forecast_probabilities):
    """Computes probability of false detection (b / [b + d]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: probability_of_false_detection: Probability of false detection.
    """

    num_false_positives = _get_num_false_positives(
        true_labels, forecast_probabilities)
    num_true_negatives = _get_num_true_negatives(
        true_labels, forecast_probabilities)

    return num_false_positives / (
        num_false_positives + num_true_negatives + K.epsilon())


def npv(true_labels, forecast_probabilities):
    """Computes negative predictive value (d / [b + d]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: negative_predictive_value: Negative predictive value.
    """

    return 1. - pofd(true_labels, forecast_probabilities)


def success_ratio(true_labels, forecast_probabilities):
    """Computes success ratio (a / [a + b]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: success_ratio: Success ratio.
    """

    num_true_positives = _get_num_true_positives(
        true_labels, forecast_probabilities)
    num_false_positives = _get_num_false_positives(
        true_labels, forecast_probabilities)

    return num_true_positives / (
        num_true_positives + num_false_positives + K.epsilon())


def far(true_labels, forecast_probabilities):
    """Computes false-alarm rate (b / [a + b]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: false_alarm_rate: False-alarm rate.
    """

    return 1. - success_ratio(true_labels, forecast_probabilities)


def dfr(true_labels, forecast_probabilities):
    """Computes detection-failure ratio (c / [c + d]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: detection_failure_ratio: Detection-failure ratio.
    """

    num_false_negatives = _get_num_false_negatives(
        true_labels, forecast_probabilities)
    num_true_negatives = _get_num_true_negatives(
        true_labels, forecast_probabilities)

    return num_false_negatives / (
        num_false_negatives + num_true_negatives + K.epsilon())


def focn(true_labels, forecast_probabilities):
    """Computes frequency of correct nulls (d / [c + d]).

    :param true_labels: See documentation for `_get_num_true_positives`.
    :param forecast_probabilities: Same.
    :return: freq_of_correct_nulls: Frequency of correct nulls.
    """

    return 1. - dfr(true_labels, forecast_probabilities)
