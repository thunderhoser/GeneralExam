"""Custom loss functions for Keras models.

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
import tensorflow


def weighted_cross_entropy(class_weights):
    """Computes weighted binary cross-entropy.

    :param class_weights: length-K numpy array of class weights.
    :return: loss: Loss (weighted binary cross-entropy).
    """

    def loss(true_label_tensor, predicted_probability_tensor):
        """Computes weighted binary cross-entropy.

        :param true_label_tensor: See docstring.
        :param predicted_probability_tensor: See docstring.
        :return: loss: Loss (weighted binary cross-entropy).
        """

        these_weights = tensorflow.convert_to_tensor(
            class_weights, dtype='float32')
        these_weights = K.reshape(these_weights, (class_weights.size, 1))

        sample_weight_matrix = K.dot(true_label_tensor, these_weights)
        sample_weight_matrix = K.reshape(
            sample_weight_matrix, K.shape(sample_weight_matrix)[:-1])

        return K.mean(
            sample_weight_matrix *
            K.categorical_crossentropy(
                true_label_tensor, predicted_probability_tensor))

    return loss
