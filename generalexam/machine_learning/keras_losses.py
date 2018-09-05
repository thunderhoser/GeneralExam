"""Custom loss functions for Keras models.

--- NOTATION ---

E = number of examples
M = number of rows in grid
N = number of columns in grid
K = number of classes (possible values of target variable)

--- FORMAT 1: BINARY PATCH CLASSIFICATION ---

target_tensor: length-E tensor of target values (observed classes).  If
    target_tensor[i] = k, the [i]th example belongs to the [k]th class.

forecast_probability_tensor: length-E tensor of forecast probabilities.
    forecast_probability_tensor[i] = forecast probability that the [i]th example
    belongs to class 1 (as opposed to 0).

--- FORMAT 2: NON-BINARY PATCH CLASSIFICATION ---

target_tensor: E-by-K tensor of target values (observed classes).  If
    target_tensor[i, k] = 1, the [i]th example belongs to the [k]th class.

forecast_probability_tensor: E-by-K tensor of forecast probabilities.
    forecast_probability_tensor[i, k] = forecast probability that the [i]th
    example belongs to the [k]th class.

--- FORMAT 3: SEMANTIC SEGMENTATION ---

target_tensor: E-by-M-by-N-by-K tensor of target values (observed classes).  If
    target_tensor[i, m, n, k] = 1, grid cell [m, n] in the [i]th example belongs
    to the [k]th class.

forecast_probability_tensor: E-by-M-by-N-by-K tensor of forecast probabilities.
    forecast_probability_tensor[i, m, n, k] = forecast probability that grid
    cell [m, n] in the [i]th example belongs to the [k]th class.
"""

import numpy
import keras.utils
import keras.backend as K
import tensorflow
from gewittergefahr.gg_utils import error_checking


def _get_num_tensor_dimensions(input_tensor):
    """Returns number of dimensions in tensor.

    :param input_tensor: Keras tensor.
    :return: num_dimensions: Number of dimensions.
    """

    return len(input_tensor.get_shape().as_list())


def weighted_cross_entropy(class_weights):
    """Computes weighted cross-entropy.

    The weight for each example is based on its true class.

    :param class_weights: length-K numpy array of class weights.
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, forecast_probability_tensor):
        """Computes weighted cross-entropy.

        :param target_tensor: See docstring for the 3 possible formats.
        :param forecast_probability_tensor: Same.
        :return: loss: Weighted cross-entropy.
        """

        error_checking.assert_is_greater_numpy_array(class_weights, 0.)

        num_dimensions = _get_num_tensor_dimensions(target_tensor)
        if num_dimensions == 1:
            error_checking.assert_is_numpy_array(
                class_weights, exact_dimensions=numpy.array([2]))
        else:
            error_checking.assert_is_numpy_array(
                class_weights, num_dimensions=1)

        num_classes = len(class_weights)
        class_weight_tensor = tensorflow.convert_to_tensor(
            class_weights, dtype='float32')
        class_weight_tensor = K.reshape(class_weight_tensor, (num_classes, 1))

        if num_dimensions == 1:
            example_weight_tensor = K.dot(
                keras.utils.to_categorical(target_tensor, num_classes),
                class_weight_tensor)
        else:
            example_weight_tensor = K.dot(target_tensor, class_weight_tensor)

        example_weight_tensor = K.reshape(
            example_weight_tensor, K.shape(example_weight_tensor)[:-1])

        return K.mean(
            example_weight_tensor *
            K.categorical_crossentropy(
                target_tensor, forecast_probability_tensor)
        )

    return loss
