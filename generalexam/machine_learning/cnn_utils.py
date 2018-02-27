"""Helper methods for a CNN (convolutional neural network).

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples (images)
M = number of image rows (unique y-coordinates at pixel centers)
N = number of image columns (unique x-coordinates at pixel centers)
C = number of image channels (predictor variables)

--- REFERENCES ---

Ioffe, S., and C. Szegedy (2015): "Batch normalization: Accelerating deep
    network training by reducing internal covariate shift". International
    Conference on Machine Learning, 448-456.
"""

import keras
from keras.utils import plot_model
import keras.initializers
import keras.optimizers
import keras.layers
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

MAX_POOLING_TYPE = 'max'
MEAN_POOLING_TYPE = 'mean'
VALID_POOLING_TYPES = [MAX_POOLING_TYPE, MEAN_POOLING_TYPE]

DEFAULT_MIN_INIT_WEIGHT = -0.05
DEFAULT_MAX_INIT_WEIGHT = 0.05

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MOMENTUM_FOR_SGD = 0.9
DEFAULT_LEARNING_RATE_DECAY_RATE = 2e-5

DEFAULT_DROPOUT_FRACTION = 0.5

EPSILON_FOR_BATCH_NORMALIZATION = 1e-3
DEFAULT_MOMENTUM_FOR_BATCH_NORMALIZATION = 0.99


def _check_input_args_for_conv_layer(
        num_filters, num_kernel_rows, num_kernel_columns, stride_length_in_rows,
        stride_length_in_columns, activation_function, is_first_layer,
        num_kernel_time_steps=None, stride_length_in_time_steps=None,
        num_input_rows=None, num_input_columns=None, num_input_time_steps=None,
        num_input_channels=None):
    """Checks input arguments for convolutional layer.

    :param num_filters: See documentation for `get_3d_convolution_layer`.
    :param num_kernel_rows: Same.
    :param num_kernel_columns: Same.
    :param stride_length_in_rows: Same.
    :param stride_length_in_columns: Same.
    :param activation_function: Same.
    :param is_first_layer: Same.
    :param num_kernel_time_steps: Same.
    :param stride_length_in_time_steps: Same.
    :param num_input_rows: Same.
    :param num_input_columns: Same.
    :param num_input_time_steps: Same.
    :param num_input_channels: Same.
    """

    error_checking.assert_is_integer(num_filters)
    error_checking.assert_is_geq(num_filters, 2)
    error_checking.assert_is_integer(num_kernel_rows)
    error_checking.assert_is_geq(num_kernel_rows, 3)
    error_checking.assert_is_integer(num_kernel_columns)
    error_checking.assert_is_geq(num_kernel_columns, 3)

    error_checking.assert_is_integer(stride_length_in_rows)
    error_checking.assert_is_greater(stride_length_in_rows, 0)
    error_checking.assert_is_less_than(stride_length_in_rows, num_kernel_rows)

    error_checking.assert_is_integer(stride_length_in_columns)
    error_checking.assert_is_greater(stride_length_in_columns, 0)
    error_checking.assert_is_less_than(
        stride_length_in_columns, num_kernel_columns)

    if num_kernel_time_steps is None:
        num_dimensions = 2
    else:
        num_dimensions = 3
        error_checking.assert_is_integer(num_kernel_time_steps)
        error_checking.assert_is_geq(num_kernel_time_steps, 3)

        error_checking.assert_is_integer(stride_length_in_time_steps)
        error_checking.assert_is_greater(stride_length_in_time_steps, 0)
        error_checking.assert_is_less_than(
            stride_length_in_time_steps, num_kernel_time_steps)

    error_checking.assert_is_string(activation_function)
    error_checking.assert_is_boolean(is_first_layer)

    if is_first_layer:
        error_checking.assert_is_integer(num_input_rows)
        error_checking.assert_is_greater(num_input_rows, 0)
        error_checking.assert_is_integer(num_input_columns)
        error_checking.assert_is_greater(num_input_columns, 0)
        error_checking.assert_is_integer(num_input_channels)
        error_checking.assert_is_greater(num_input_channels, 0)

        if num_dimensions == 3:
            error_checking.assert_is_integer(num_input_time_steps)
            error_checking.assert_is_greater(num_input_time_steps, 0)


def _check_input_args_for_pooling_layer(
        num_rows_in_window, num_columns_in_window, pooling_type,
        stride_length_in_rows, stride_length_in_columns,
        num_time_steps_in_window=None, stride_length_in_time_steps=None):
    """Checks input arguments for pooling layer.

    :param num_rows_in_window: See documentation for `get_3d_pooling_layer`.
    :param num_columns_in_window: Same.
    :param pooling_type: Same.
    :param stride_length_in_rows: Same.
    :param stride_length_in_columns: Same.
    :param num_time_steps_in_window: Same.
    :param stride_length_in_time_steps: Same.
    :return: stride_length_in_rows: See doc for `get_3d_pooling_layer`.
    :return: stride_length_in_columns: Same.
    :return: stride_length_in_time_steps: Same.
    :raises: ValueError: if `pooling_type not in VALID_POOLING_TYPES`.
    """

    error_checking.assert_is_integer(num_rows_in_window)
    error_checking.assert_is_geq(num_rows_in_window, 2)
    error_checking.assert_is_integer(num_columns_in_window)
    error_checking.assert_is_geq(num_columns_in_window, 2)

    error_checking.assert_is_string(pooling_type)
    if pooling_type not in VALID_POOLING_TYPES:
        error_string = (
            '\n\n{0:s}\nValid pooling types (listed above) do not include '
            '"{1:s}".').format(VALID_POOLING_TYPES, pooling_type)
        raise ValueError(error_string)

    if num_time_steps_in_window is None:
        num_dimensions = 2
    else:
        num_dimensions = 3
        error_checking.assert_is_integer(num_time_steps_in_window)
        error_checking.assert_is_geq(num_time_steps_in_window, 2)

    if (stride_length_in_rows is None or stride_length_in_columns is None or
            stride_length_in_time_steps is None):
        stride_length_in_rows = num_rows_in_window + 0
        stride_length_in_columns = num_columns_in_window + 0

        if num_dimensions == 3:
            stride_length_in_time_steps = num_time_steps_in_window + 0

    error_checking.assert_is_integer(stride_length_in_rows)
    error_checking.assert_is_greater(stride_length_in_rows, 0)
    error_checking.assert_is_leq(stride_length_in_rows, num_rows_in_window)

    error_checking.assert_is_integer(stride_length_in_columns)
    error_checking.assert_is_greater(stride_length_in_columns, 0)
    error_checking.assert_is_leq(
        stride_length_in_columns, num_columns_in_window)

    if num_dimensions == 3:
        error_checking.assert_is_integer(stride_length_in_time_steps)
        error_checking.assert_is_greater(stride_length_in_time_steps, 0)
        error_checking.assert_is_leq(
            stride_length_in_time_steps, num_time_steps_in_window)

    return (stride_length_in_rows, stride_length_in_columns,
            stride_length_in_time_steps)


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


def get_2d_convolution_layer(
        num_filters, num_kernel_rows, num_kernel_columns, stride_length_in_rows,
        stride_length_in_columns, kernel_weight_initializer='glorot_uniform',
        bias_initializer='zeros', activation_function='relu',
        is_first_layer=False, num_input_rows=None, num_input_columns=None,
        num_input_channels=None):
    """Creates 2-D-convolution layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here (maybe).
    layer_object = get_2d_convolution_layer(...)
    model_object.add(layer_object)

    :param num_filters: Number of filters.
    :param num_kernel_rows: Number of pixel rows in kernel.
    :param num_kernel_columns: Number of pixel columns in kernel.
    :param stride_length_in_rows: Stride length in the y-direction (number of
        pixel rows).
    :param stride_length_in_columns: Stride length in the x-direction (number of
        pixel columns).
    :param kernel_weight_initializer: Either string or instance of
        `keras.initializers.Initializer`.  Will be used to init kernel weights.
    :param bias_initializer: Either string or instance of
        `keras.initializers.Initializer`.  Will be used to init bias weights.
    :param activation_function: Activation function (string).
    :param is_first_layer: Boolean flag.  If True, this is the first layer in
        the network, which means that the input dimensions (dimensions of each
        predictor image) must be known.
    :param num_input_rows: [used only if is_first_layer = True]
        Number of pixel rows in each predictor image.
    :param num_input_columns: [used only if is_first_layer = True]
        Number of pixel columns in each predictor image.
    :param num_input_channels: [used only if is_first_layer = True]
        Number of channels (predictor variables) in each predictor image.
    :return: layer_object: Instance of `keras.layers.Conv2D`.
    """

    # TODO(thunderhoser): Add regularization.

    _check_input_args_for_conv_layer(
        num_filters=num_filters, num_kernel_rows=num_kernel_rows,
        num_kernel_columns=num_kernel_columns,
        stride_length_in_rows=stride_length_in_rows,
        stride_length_in_columns=stride_length_in_columns,
        activation_function=activation_function, is_first_layer=is_first_layer,
        num_input_rows=num_input_rows, num_input_columns=num_input_columns,
        num_input_channels=num_input_channels)

    if is_first_layer:
        return keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(num_kernel_rows, num_kernel_columns),
            strides=(stride_length_in_rows, stride_length_in_columns),
            padding='valid', data_format='channels_last', dilation_rate=(1, 1),
            activation=activation_function, use_bias=True,
            kernel_initializer=kernel_weight_initializer,
            bias_initializer=bias_initializer,
            input_shape=(num_input_rows, num_input_columns, num_input_channels))

    return keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns),
        strides=(stride_length_in_rows, stride_length_in_columns),
        padding='valid', data_format='channels_last', dilation_rate=(1, 1),
        activation=activation_function, use_bias=True,
        kernel_initializer=kernel_weight_initializer,
        bias_initializer=bias_initializer)


def get_3d_convolution_layer(
        num_filters, num_kernel_rows, num_kernel_columns, num_kernel_time_steps,
        stride_length_in_rows, stride_length_in_columns,
        stride_length_in_time_steps, kernel_weight_initializer='glorot_uniform',
        bias_initializer='zeros', activation_function='relu',
        is_first_layer=False, num_input_rows=None, num_input_columns=None,
        num_input_time_steps=None, num_input_channels=None):
    """Creates 3-D-convolution layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here (maybe).
    layer_object = get_3d_convolution_layer(...)
    model_object.add(layer_object)

    :param num_filters: Number of filters.
    :param num_kernel_rows: Number of pixel rows in kernel.
    :param num_kernel_columns: Number of pixel columns in kernel.
    :param num_kernel_time_steps: Number of time steps in kernel.
    :param stride_length_in_rows: Stride length in the y-direction (number of
        pixel rows).
    :param stride_length_in_columns: Stride length in the x-direction (number of
        pixel columns).
    :param stride_length_in_time_steps: Stride length in the t-direction (number
        of time steps).
    :param kernel_weight_initializer: See documentation for
        `get_2d_convolution_layer`.
    :param bias_initializer: Same.
    :param activation_function: Same.
    :param is_first_layer: Same.
    :param num_input_rows: [used only if is_first_layer = True]
        Number of pixel rows in each predictor image.
    :param num_input_columns: [used only if is_first_layer = True]
        Number of pixel columns in each predictor image.
    :param num_input_time_steps: [used only if is_first_layer = True]
        Number of time steps (predictor images) in each example.
    :param num_input_channels: [used only if is_first_layer = True]
        Number of channels (predictor variables) in each predictor image.
    :return: layer_object: Instance of `keras.layers.Conv3D`.
    """

    _check_input_args_for_conv_layer(
        num_filters=num_filters, num_kernel_rows=num_kernel_rows,
        num_kernel_columns=num_kernel_columns,
        num_kernel_time_steps=num_kernel_time_steps,
        stride_length_in_rows=stride_length_in_rows,
        stride_length_in_columns=stride_length_in_columns,
        stride_length_in_time_steps=stride_length_in_time_steps,
        activation_function=activation_function,
        is_first_layer=is_first_layer, num_input_rows=num_input_rows,
        num_input_columns=num_input_columns,
        num_input_time_steps=num_input_time_steps,
        num_input_channels=num_input_channels)

    if is_first_layer:
        return keras.layers.Conv3D(
            filters=num_filters,
            kernel_size=(num_kernel_rows, num_kernel_columns,
                         num_kernel_time_steps),
            strides=(stride_length_in_rows, stride_length_in_columns,
                     stride_length_in_time_steps),
            padding='valid', data_format='channels_last',
            dilation_rate=(1, 1, 1), activation=activation_function,
            use_bias=True, kernel_initializer=kernel_weight_initializer,
            bias_initializer=bias_initializer,
            input_shape=(num_input_rows, num_input_columns,
                         num_input_time_steps, num_input_channels))

    return keras.layers.Conv3D(
        filters=num_filters,
        kernel_size=(num_kernel_rows, num_kernel_columns,
                     num_kernel_time_steps),
        strides=(stride_length_in_rows, stride_length_in_columns,
                 stride_length_in_time_steps),
        padding='valid', data_format='channels_last', dilation_rate=(1, 1, 1),
        activation=activation_function, use_bias=True,
        kernel_initializer=kernel_weight_initializer,
        bias_initializer=bias_initializer)


def get_dropout_layer(dropout_fraction=DEFAULT_DROPOUT_FRACTION):
    """Creates dropout layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    ... # Add convolutional layer here.
    layer_object = get_dropout_layer(...)
    model_object.add(layer_object)

    :param dropout_fraction: Fraction of input units to drop.
    :return: layer_object: Instance of `keras.layers.Dropout`.
    """

    error_checking.assert_is_greater(dropout_fraction, 0.)
    error_checking.assert_is_less_than(dropout_fraction, 1.)
    return keras.layers.Dropout(rate=dropout_fraction)


def get_2d_pooling_layer(
        num_rows_in_window, num_columns_in_window,
        pooling_type=MAX_POOLING_TYPE, stride_length_in_rows=None,
        stride_length_in_columns=None):
    """Creates 2-D max-pooling or mean-pooling layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    layer_object = get_2d_pooling_layer(...)
    model_object.add(layer_object)

    :param num_rows_in_window: Number of pixel rows in pooling window.
    :param num_columns_in_window: Number of pixel columns in pooling window.
    :param pooling_type: Pooling type (either "max" or "mean").
    :param stride_length_in_rows: Stride length in the y-direction (number of
        pixel rows between adjacent windows).  Default is
        `num_rows_in_window`, which means that windows are abutting but
        non-overlapping.
    :param stride_length_in_columns: Same as above, except for columns.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    stride_length_in_rows, stride_length_in_columns, _ = (
        _check_input_args_for_pooling_layer(
            num_rows_in_window=num_rows_in_window,
            num_columns_in_window=num_columns_in_window,
            pooling_type=pooling_type,
            stride_length_in_rows=stride_length_in_rows,
            stride_length_in_columns=stride_length_in_columns))

    if pooling_type == MAX_POOLING_TYPE:
        return keras.layers.MaxPooling2D(
            pool_size=(num_rows_in_window, num_columns_in_window),
            strides=(stride_length_in_rows, stride_length_in_columns),
            padding='valid', data_format='channels_last')

    return keras.layers.AveragePooling2D(
        pool_size=(num_rows_in_window, num_columns_in_window),
        strides=(stride_length_in_rows, stride_length_in_columns),
        padding='valid', data_format='channels_last')


def get_3d_pooling_layer(
        num_rows_in_window, num_columns_in_window, num_time_steps_in_window,
        pooling_type=MAX_POOLING_TYPE, stride_length_in_rows=None,
        stride_length_in_columns=None, stride_length_in_time_steps=None):
    """Creates 3-D max-pooling or mean-pooling layer.

    However, you still need to add this layer to the model.  For example:

    model_object = keras.models.Sequential()
    ... # Add other layers here.
    layer_object = get_3d_pooling_layer(...)
    model_object.add(layer_object)

    :param num_rows_in_window: Number of pixel rows in pooling window.
    :param num_columns_in_window: Number of pixel columns in pooling window.
    :param num_time_steps_in_window: Number of time steps in pooling window.
    :param pooling_type: Pooling type (either "max" or "mean").
    :param stride_length_in_rows: Stride length in the y-direction (number of
        pixel rows between adjacent windows).  Default is
        `num_rows_in_window`, which means that windows are abutting but
        non-overlapping.
    :param stride_length_in_columns: Same as above, except for columns.
    :param stride_length_in_time_steps: Same as above, except for time steps.
    :return: layer_object: Instance of `keras.layers.MaxPooling3D` or
        `keras.layers.AveragePooling3D`.
    """

    (stride_length_in_rows, stride_length_in_columns,
     stride_length_in_time_steps) = _check_input_args_for_pooling_layer(
         num_rows_in_window=num_rows_in_window,
         num_columns_in_window=num_columns_in_window,
         num_time_steps_in_window=num_time_steps_in_window,
         pooling_type=pooling_type,
         stride_length_in_rows=stride_length_in_rows,
         stride_length_in_columns=stride_length_in_columns,
         stride_length_in_time_steps=stride_length_in_time_steps)

    if pooling_type == MAX_POOLING_TYPE:
        return keras.layers.MaxPooling3D(
            pool_size=(num_rows_in_window, num_columns_in_window,
                       num_time_steps_in_window),
            strides=(stride_length_in_rows, stride_length_in_columns,
                     stride_length_in_time_steps),
            padding='valid', data_format='channels_last')

    return keras.layers.AveragePooling3D(
        pool_size=(num_rows_in_window, num_columns_in_window,
                   num_time_steps_in_window),
        strides=(stride_length_in_rows, stride_length_in_columns,
                 stride_length_in_time_steps),
        padding='valid', data_format='channels_last')


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

    # TODO(thunderhoser): Add regularization.

    error_checking.assert_is_geq(momentum, 0.)
    error_checking.assert_is_less_than(momentum, 1.)
    error_checking.assert_is_boolean(scale_data)

    return keras.layers.BatchNormalization(
        axis=-1, momentum=momentum, epsilon=EPSILON_FOR_BATCH_NORMALIZATION,
        center=True, scale=scale_data)


def get_fully_connected_layer(
        num_output_units, activation_function,
        kernel_weight_initializer='glorot_uniform', bias_initializer='zeros'):
    """Creates fully connected (traditional neural-net) layer.

    :param num_output_units: Number of output units.
    :param activation_function: Activation function (string).
    :param kernel_weight_initializer: Either string or instance of
        `keras.initializers.Initializer`.  Will be used to init kernel weights.
    :param bias_initializer: Either string or instance of
        `keras.initializers.Initializer`.  Will be used to init bias weights.
    :return: layer_object: Instance of `keras.layers.Dense`.
    """

    # TODO(thunderhoser): Add regularization.

    error_checking.assert_is_integer(num_output_units)
    error_checking.assert_is_greater(num_output_units, 0)
    error_checking.assert_is_string(activation_function)

    return keras.layers.Dense(
        num_output_units, activation=activation_function, use_bias=True,
        kernel_initializer=kernel_weight_initializer,
        bias_initializer=bias_initializer)


def get_flattening_layer():
    """Creates flattening layer.

    A "flattening layer" turns its input into a 1-D vector.

    :return: layer_object: Instance of `keras.layers.Flatten`.
    """

    return keras.layers.Flatten()


def visualize_architecture(model_object, output_image_file_name):
    """Creates and saves figure showing model architecture.

    :param model_object: Instance of `keras.models.Sequential`.
    :param output_image_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_image_file_name)

    plot_model(model_object, show_shapes=True, show_layer_names=False,
               to_file=output_image_file_name)
