"""Helper methods for a CNN (convolutional neural network).

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples (images)
M = number of image rows (unique y-coordinates at pixel centers)
N = number of image columns (unique x-coordinates at pixel centers)
C = number of image channels (predictor variables)
"""

import keras
from keras.utils import plot_model
import keras.initializers
import keras.optimizers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Dropout, Flatten, Dense
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

MAX_POOLING_TYPE = 'max'
MEAN_POOLING_TYPE = 'mean'

DEFAULT_MIN_INIT_WEIGHT = -0.05
DEFAULT_MAX_INIT_WEIGHT = 0.05
DEFAULT_DROPOUT_FRACTION = 0.5
EPSILON_FOR_BATCH_NORMALIZATION = 1e-3
DEFAULT_MOMENTUM_FOR_BATCH_NORMALIZATION = 0.99

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MOMENTUM_FOR_SGD = 0.9
DEFAULT_LEARNING_RATE_DECAY_RATE = 2e-5


def get_random_uniform_initializer(
        min_value=DEFAULT_MIN_INIT_WEIGHT, max_value=DEFAULT_MAX_INIT_WEIGHT):
    """Creates weight-initializer with random uniform distribution.

    :param min_value: Minimum value in random uniform distribution.
    :param max_value: Max value in random uniform distribution.
    :return: initializer_object: Instance of `keras.initializers.Initializer`.
    """

    error_checking.assert_is_geq(min_value, -1.)
    error_checking.assert_is_leq(max_value, 1.)
    error_checking.assert_is_greater(max_value, min_value)

    return keras.initializers.RandomUniform(minval=min_value, maxval=max_value)


def get_sgd_optimizer(
        learning_rate=DEFAULT_LEARNING_RATE, momentum=DEFAULT_MOMENTUM_FOR_SGD,
        learning_rate_decay_rate=DEFAULT_LEARNING_RATE_DECAY_RATE):
    """Creates SGD (stochastic gradient descent) optimizer.

    :param learning_rate: Learning rate.
    :param momentum: Momentum (see documentation for `keras.optimizers.SGD`).
    :param learning_rate_decay_rate: Decay rate for learning rate.  After each
        iteration of SGD, the following equation will be applied.

    new_learning_rate = learning_rate / (
        1 + learning_rate_decay_rate * num_iterations_done)

    :return: Instance of `keras.optimizers.SGD`.
    """

    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_geq(momentum, 0.)
    error_checking.assert_is_less_than(momentum, 1.)
    error_checking.assert_is_geq(learning_rate_decay_rate, 0.)
    error_checking.assert_is_less_than(learning_rate_decay_rate, 1.)

    return keras.optimizers.SGD(
        lr=learning_rate, momentum=momentum, decay=learning_rate_decay_rate,
        nesterov=False)


def get_convolutional_layer(
        num_filters, num_rows_in_kernel, num_columns_in_kernel,
        stride_length_in_rows, stride_length_in_columns,
        kernel_weight_initializer=None, bias_initializer=None,
        activation_function='relu', is_first_layer=False,
        num_rows_per_image=None, num_columns_per_image=None,
        num_channels_per_image=None):
    """Creates convolutional layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here (maybe).
    layer_object = get_convolutional_layer(...)
    model_object.add(layer_object)

    :param num_filters: Number of filters.
    :param num_rows_in_kernel: Number of rows in kernel.  If the kernel is
        M x N, it will span M pixels in the y-direction by N pixels in the
        x-direction.
    :param num_columns_in_kernel: Number of columns in kernel.
    :param stride_length_in_rows: Stride length between adjacent kernels in
        y-direction.
    :param stride_length_in_columns: Stride length between adjacent kernels in
        x-direction.
    :param kernel_weight_initializer: Instance of
        `keras.initializers.Initializer`, which will be used to initialize
        kernel weights.
    :param bias_initializer: Instance of `keras.initializers.Initializer`, which
        will be used to initialize the bias weight(s?).
    :param activation_function: Activation function (string).
    :param is_first_layer: Boolean flag (indicates whether or not this is the
        first layer in the network).
    :param num_rows_per_image: [used only if is_first_layer = True]
        Number of pixel rows in each example (input image).
    :param num_columns_per_image: [used only if is_first_layer = True]
        Number of pixel columns in each example (input image).
    :param num_channels_per_image: [used only if is_first_layer = True]
        Number of channels (predictor variables) in each example (input image).
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    # TODO(thunderhoser): Add regularization.

    error_checking.assert_is_integer(num_filters)
    error_checking.assert_is_geq(num_filters, 2)
    error_checking.assert_is_integer(num_rows_in_kernel)
    error_checking.assert_is_geq(num_rows_in_kernel, 3)
    error_checking.assert_is_integer(num_columns_in_kernel)
    error_checking.assert_is_geq(num_columns_in_kernel, 3)

    error_checking.assert_is_integer(stride_length_in_rows)
    error_checking.assert_is_greater(stride_length_in_rows, 0)
    error_checking.assert_is_less_than(
        stride_length_in_rows, num_rows_in_kernel)

    error_checking.assert_is_integer(stride_length_in_columns)
    error_checking.assert_is_greater(stride_length_in_columns, 0)
    error_checking.assert_is_less_than(
        stride_length_in_columns, num_columns_in_kernel)

    error_checking.assert_is_string(activation_function)
    error_checking.assert_is_boolean(is_first_layer)

    if is_first_layer:
        error_checking.assert_is_integer(num_rows_per_image)
        error_checking.assert_is_greater(num_rows_per_image, 0)
        error_checking.assert_is_integer(num_columns_per_image)
        error_checking.assert_is_greater(num_columns_per_image, 0)
        error_checking.assert_is_integer(num_channels_per_image)
        error_checking.assert_is_greater(num_channels_per_image, 0)

    if kernel_weight_initializer is None:
        kernel_weight_initializer = 'glorot_uniform'
    if bias_initializer is None:
        bias_initializer = 'zeros'

    if is_first_layer:
        return Conv2D(
            filters=num_filters,
            kernel_size=(num_rows_in_kernel, num_columns_in_kernel),
            strides=(stride_length_in_rows, stride_length_in_columns),
            padding='valid', data_format='channels_last', dilation_rate=(1, 1),
            activation=activation_function, use_bias=True,
            kernel_initializer=kernel_weight_initializer,
            bias_initializer=bias_initializer,
            input_shape=(num_rows_per_image, num_columns_per_image,
                         num_channels_per_image))

    return Conv2D(
        filters=num_filters,
        kernel_size=(num_rows_in_kernel, num_columns_in_kernel),
        strides=(stride_length_in_rows, stride_length_in_columns),
        padding='valid', data_format='channels_last', dilation_rate=(1, 1),
        activation=activation_function, use_bias=True,
        kernel_initializer=kernel_weight_initializer,
        bias_initializer=bias_initializer)


def get_dropout_layer(
        predictor_dimensions_one_batch=None,
        dropout_fraction=DEFAULT_DROPOUT_FRACTION):
    """Creates dropout layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    ... # Add convolutional layer here.
    layer_object = get_dropout_layer(...)
    model_object.add(layer_object)

    :param predictor_dimensions_one_batch: 1-D numpy array with dimensions of
        predictor data for one batch.  If batch size (number of examples per
        batch) = E, number of pixel rows = M, number of pixel columns = N, and
        number of channels = C, this should be (E, M, N, C).
    :param dropout_fraction: Fraction of input units to drop.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    # TODO(thunderhoser): Start using input arg `predictor_dimensions_one_batch`
    # again.

    # error_checking.assert_is_integer_numpy_array(predictor_dimensions_one_batch)
    # error_checking.assert_is_greater_numpy_array(
    #     predictor_dimensions_one_batch, 0)
    # error_checking.assert_is_numpy_array(
    #     predictor_dimensions_one_batch, num_dimensions=1)

    error_checking.assert_is_greater(dropout_fraction, 0.)
    error_checking.assert_is_less_than(dropout_fraction, 1.)

    return Dropout(rate=dropout_fraction, noise_shape=None)


def get_pooling_layer(
        num_rows_in_window, num_columns_in_window,
        pooling_type=MAX_POOLING_TYPE, stride_length_in_rows=None,
        stride_length_in_columns=None):
    """Creates max-pooling or mean-pooling layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    layer_object = get_pooling_layer(...)
    model_object.add(layer_object)

    :param num_rows_in_window: Number of rows in pooling window.  If the window
        is M x N, it will span M pixels in the y-direction by N pixels in the
        x-direction.
    :param num_columns_in_window: Number of columns in pooling window.
    :param pooling_type: Pooling type (either "max" or "mean").
    :param stride_length_in_rows: Stride length between adjacent windows in
        y-direction.  Default is `num_rows_in_window`, which means that windows
        will be abutting but non-overlapping.
    :param stride_length_in_columns: Stride length between adjacent windows in
        x-direction.  Default is `num_columns_in_window`, which means that
        windows will be abutting but non-overlapping.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    error_checking.assert_is_integer(num_rows_in_window)
    error_checking.assert_is_geq(num_rows_in_window, 2)
    error_checking.assert_is_integer(num_columns_in_window)
    error_checking.assert_is_geq(num_columns_in_window, 2)
    error_checking.assert_is_string(pooling_type)

    if stride_length_in_rows is None or stride_length_in_columns is None:
        stride_length_in_rows = num_rows_in_window
        stride_length_in_columns = num_columns_in_window

    error_checking.assert_is_integer(stride_length_in_rows)
    error_checking.assert_is_greater(stride_length_in_rows, 0)
    error_checking.assert_is_leq(stride_length_in_rows, num_rows_in_window)

    error_checking.assert_is_integer(stride_length_in_columns)
    error_checking.assert_is_greater(stride_length_in_columns, 0)
    error_checking.assert_is_leq(
        stride_length_in_columns, num_columns_in_window)

    if pooling_type == MAX_POOLING_TYPE:
        return MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(stride_length_in_rows, stride_length_in_columns),
            padding='valid', data_format='channels_last')

    if pooling_type == MEAN_POOLING_TYPE:
        return AveragePooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(stride_length_in_rows, stride_length_in_columns),
            padding='valid', data_format='channels_last')

    return None


def get_batch_normalization_layer(
        momentum=DEFAULT_MOMENTUM_FOR_BATCH_NORMALIZATION, scale_data=True):
    """Creates batch-normalization layer.

    :param momentum: Momentum parameter (see documentation for
        `keras.layers.BatchNormalization`).
    :param scale_data: Boolean flag.  If True, layer inputs will be multiplied
        by gamma (which is a parameter learned by the network -- for more on
        gamma, see the second equation in Section 3 of Ioffe and Szegedy
        [2015]).  This should always be true, unless the following layer has a
        linear or ReLU (rectified linear) activation function.
    :return: layer_object: Instance of `keras.layers.BatchNormalization`.
    """

    # TODO(thunderhoser): allow for weight initialization and regularization.

    error_checking.assert_is_geq(momentum, 0.)
    error_checking.assert_is_less_than(momentum, 1.)
    error_checking.assert_is_boolean(scale_data)

    return BatchNormalization(
        axis=-1, momentum=momentum, epsilon=EPSILON_FOR_BATCH_NORMALIZATION,
        center=True, scale=scale_data)


def get_fully_connected_layer(
        num_output_units, activation_function, kernel_weight_initializer=None,
        bias_initializer=None):
    """Creates fully connected (traditional neural-net) layer.

    :param num_output_units: Number of output units.
    :param activation_function: Activation function (string).
    :param kernel_weight_initializer: Instance of
        `keras.initializers.Initializer`, which will be used to initialize
        kernel weights.
    :param bias_initializer: Instance of `keras.initializers.Initializer`, which
        will be used to initialize the bias weight(s?).
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    error_checking.assert_is_integer(num_output_units)
    error_checking.assert_is_greater(num_output_units, 0)
    error_checking.assert_is_string(activation_function)

    if kernel_weight_initializer is None:
        kernel_weight_initializer = 'glorot_uniform'
    if bias_initializer is None:
        bias_initializer = 'zeros'

    return Dense(
        num_output_units, activation=activation_function, use_bias=True,
        kernel_initializer=kernel_weight_initializer,
        bias_initializer=bias_initializer)


def get_flattening_layer():
    """Creates flattening layer.

    A "flattening layer" turns its input into a 1-D vector.

    :return: layer_object: Instance of `keras.layers.Flatten`.
    """

    return Flatten()


def visualize_architecture(model_object, output_image_file_name):
    """Creates and saves figure showing model architecture.

    :param model_object: Instance of `keras.models.Sequential`.
    :param output_image_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_image_file_name)

    plot_model(model_object, show_shapes=True, show_layer_names=False,
               to_file=output_image_file_name)
