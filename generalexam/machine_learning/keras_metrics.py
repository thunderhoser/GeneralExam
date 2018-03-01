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
true label != 0

b = number of false positives = sum of forecast probabilities for examples with
true label of 0

c = number of false negatives = sum of inverse forecast probabilities for
examples with true label of 1

d = number of true negatives = sum of inverse forecast probabilities for
examples with true label != 0

--- INPUT FORMATS ---

Inputs to all methods must be in one of two formats.

E = number of examples
M = number of pixel rows
N = number of pixel columns
K = number of classes (possible values of target variable)

FORMAT 1:

true_label_tensor: E-by-K tensor of true labels.  If true_label_tensor[i, m]
    = 1, the [i]th example belongs to the [m]th class.

predicted_probability_tensor: E-by-K tensor of predicted probabilities.
    predicted_probability_tensor[i, m] is the estimated probability that the
    [i]th example belongs to the [m]th class.

FORMAT 2:

true_label_tensor: E-by-M-by-N-by-K tensor of true labels.  If
    true_label_tensor[i, j, k, m] = 1, pixel [j, k] in the [i]th example belongs
    to the [m]th class.

predicted_probability_tensor: E-by-M-by-N-by-K tensor of predicted
    probabilities.  predicted_probability_tensor[i, j, k, m] is the estimated
    probability that pixel [j, k] in the [i]th example belongs to the [m]th
    class.
"""

import keras.backend as K


def _get_num_true_positives(true_label_tensor, predicted_probability_tensor):
    """Returns number of true positives (defined in docstring).

    Number of true positives = `a` in contingency table

    :param true_label_tensor: See docstring.
    :param predicted_probability_tensor: See docstring.
    :return: num_true_positives: Number of true positives.
    """

    return K.sum(K.clip(
        (1 - true_label_tensor[..., 0]) *
        (1. - predicted_probability_tensor[..., 0]), 0., 1.))


def _get_num_false_positives(true_label_tensor, predicted_probability_tensor):
    """Returns number of false positives (defined in docstring).

    Number of false positives = `b` in contingency table

    :param true_label_tensor: See docstring.
    :param predicted_probability_tensor: See docstring.
    :return: num_false_positives: Number of false positives.
    """

    return K.sum(K.clip(
        true_label_tensor[..., 0] *
        (1. - predicted_probability_tensor[..., 0]), 0., 1.))


def _get_num_false_negatives(true_label_tensor, predicted_probability_tensor):
    """Returns number of false negatives (defined in docstring).

    Number of false negatives = `c` in contingency table

    :param true_label_tensor: See docstring.
    :param predicted_probability_tensor: See docstring.
    :return: num_false_negatives: Number of false negatives.
    """

    return K.sum(K.clip(
        (1 - true_label_tensor[..., 0]) *
        predicted_probability_tensor[..., 0], 0., 1.))


def _get_num_true_negatives(true_label_tensor, predicted_probability_tensor):
    """Returns number of true negatives (defined in docstring).

    Number of true negatives = `d` in contingency table

    :param true_label_tensor: See docstring.
    :param predicted_probability_tensor: See docstring.
    :return: num_true_negatives: Number of true negatives.
    """

    return K.sum(K.clip(
        true_label_tensor[..., 0] *
        predicted_probability_tensor[..., 0], 0., 1.))


def accuracy(true_label_tensor, predicted_probability_tensor):
    """Computes accuracy.

    :param true_label_tensor: E-by-K tensor of true labels.  If
        true_label_tensor[i, j] = 1, the [i]th example belongs to the [j]th
        class.
    :param predicted_probability_tensor: E-by-K by tensor of predicted
        probabilities.  predicted_probability_tensor[i, j] is the probability of
        the [i]th example falling in the [j]th class.
    :return: accuracy: Accuracy.
    """

    return K.mean(K.clip(
        true_label_tensor * predicted_probability_tensor, 0., 1.))


def binary_accuracy(true_label_tensor, predicted_probability_tensor):
    """Computes binary accuracy ([a + d] / [a + b + c + d]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_accuracy: Binary accuracy.
    """

    num_true_positives = _get_num_true_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_positives = _get_num_false_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_negatives = _get_num_false_negatives(
        true_label_tensor, predicted_probability_tensor)
    num_true_negatives = _get_num_true_negatives(
        true_label_tensor, predicted_probability_tensor)

    return (num_true_positives + num_true_negatives) / (
        num_true_positives + num_false_positives + num_false_negatives +
        num_true_negatives + K.epsilon())


def binary_csi(true_label_tensor, predicted_probability_tensor):
    """Computes binary critical success index (a / [a + b + c]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_csi: Binary CSI.
    """

    num_true_positives = _get_num_true_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_positives = _get_num_false_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_negatives = _get_num_false_negatives(
        true_label_tensor, predicted_probability_tensor)

    return num_true_positives / (
        num_true_positives + num_false_positives + num_false_negatives +
        K.epsilon())


def binary_frequency_bias(true_label_tensor, predicted_probability_tensor):
    """Computes binary frequency bias ([a + b] / [a + c]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_frequency_bias: Binary frequency bias.
    """

    num_true_positives = _get_num_true_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_positives = _get_num_false_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_negatives = _get_num_false_negatives(
        true_label_tensor, predicted_probability_tensor)

    return (num_true_positives + num_false_positives) / (
        num_true_positives + num_false_negatives + K.epsilon())


def binary_pod(true_label_tensor, predicted_probability_tensor):
    """Computes binary probability of detection (a / [a + c]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_pod: Binary POD.
    """

    num_true_positives = _get_num_true_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_negatives = _get_num_false_negatives(
        true_label_tensor, predicted_probability_tensor)

    return num_true_positives / (
        num_true_positives + num_false_negatives + K.epsilon())


def binary_fom(true_label_tensor, predicted_probability_tensor):
    """Computes binary frequency of misses (c / [a + c]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_fom: Binary FOM.
    """

    return 1. - binary_pod(true_label_tensor, predicted_probability_tensor)


def binary_pofd(true_label_tensor, predicted_probability_tensor):
    """Computes binary probability of false detection (b / [b + d]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_pofd: Binary POFD.
    """

    num_false_positives = _get_num_false_positives(
        true_label_tensor, predicted_probability_tensor)
    num_true_negatives = _get_num_true_negatives(
        true_label_tensor, predicted_probability_tensor)

    return num_false_positives / (
        num_false_positives + num_true_negatives + K.epsilon())


def binary_npv(true_label_tensor, predicted_probability_tensor):
    """Computes binary negative predictive value (d / [b + d]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_npv: Binary NPV.
    """

    return 1. - binary_pofd(true_label_tensor, predicted_probability_tensor)


def binary_success_ratio(true_label_tensor, predicted_probability_tensor):
    """Computes binary success ratio (a / [a + b]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_success_ratio: Binary success ratio.
    """

    num_true_positives = _get_num_true_positives(
        true_label_tensor, predicted_probability_tensor)
    num_false_positives = _get_num_false_positives(
        true_label_tensor, predicted_probability_tensor)

    return num_true_positives / (
        num_true_positives + num_false_positives + K.epsilon())


def binary_far(true_label_tensor, predicted_probability_tensor):
    """Computes binary false-alarm rate (b / [a + b]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_far: Binary FAR.
    """

    return 1. - binary_success_ratio(true_label_tensor,
                                     predicted_probability_tensor)


def binary_dfr(true_label_tensor, predicted_probability_tensor):
    """Computes binary detection-failure ratio (c / [c + d]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_dfr: Binary DFR.
    """

    num_false_negatives = _get_num_false_negatives(
        true_label_tensor, predicted_probability_tensor)
    num_true_negatives = _get_num_true_negatives(
        true_label_tensor, predicted_probability_tensor)

    return num_false_negatives / (
        num_false_negatives + num_true_negatives + K.epsilon())


def binary_focn(true_label_tensor, predicted_probability_tensor):
    """Computes binary frequency of correct nulls (d / [c + d]).

    :param true_label_tensor: See documentation for `accuracy`.
    :param predicted_probability_tensor: Same.
    :return: binary_focn: Binary FOCN.
    """

    return 1. - binary_dfr(true_label_tensor, predicted_probability_tensor)
