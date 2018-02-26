"""Custom loss functions for Keras models.

--- DEFINITIONS ---

"Forecast probability" = predicted probability, from the Keras model, that the
true label is 1.

"Inverse forecast probability" = predicted probability, from the Keras model,
that the true label is 0.
"""

import keras.backend as K


def weighted_cross_entropy(positive_class_weight):
    """Computes weighted binary cross-entropy.

    :param positive_class_weight: Weight for positive class.  Weight for
        negative class = `1 - positive_class_weight`.
    :return: loss: Loss (weighted binary cross-entropy).
    """

    def loss(true_labels, forecast_probabilities):
        """Computes weighted binary cross-entropy.

        :param true_labels: tensor of true labels (integers).
        :param forecast_probabilities: equivalent-sized tensor of forecast
            probabilities.
        :return: loss: Loss (weighted binary cross-entropy).
        """

        sample_weights = (
            positive_class_weight * true_labels +
            (1 - positive_class_weight) * (1 - true_labels))

        return K.mean(
            sample_weights *
            K.binary_crossentropy(true_labels, forecast_probabilities),
            axis=-1)

    return loss
