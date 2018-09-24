"""Methods for creating a CNN (building the architecture)."""

import keras.layers
import keras.models
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import keras_metrics

DEFAULT_METRIC_FUNCTION_LIST = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

NUM_CLASSES = 3
INIT_NUM_FILTERS = 32


def _get_output_layer_and_loss_function(num_classes):
    """Creates output layer and loss function.

    :param num_classes: Number of classes.
    :return: dense_layer_object: Instance of `keras.layers.Dense`, with no
        activation.
    :return: activation_layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    :return: loss_function: Instance of `keras.losses.binary_crossentropy` or
        `keras.losses.categorical_crossentropy`.
    """

    if num_classes == 2:
        num_output_units = 1
        loss_function = keras.losses.binary_crossentropy
        activation_function_string = architecture_utils.SIGMOID_FUNCTION_STRING
    else:
        num_output_units = num_classes
        loss_function = keras.losses.categorical_crossentropy
        activation_function_string = architecture_utils.SOFTMAX_FUNCTION_STRING

    dense_layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=num_output_units)
    activation_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=activation_function_string)

    return dense_layer_object, activation_layer_object, loss_function


def get_first_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the simplest architecture.

    :param num_rows: Number of pixel rows per image.
    :param num_columns: Number of pixel columns per image.
    :param num_channels: Number of channels (predictor variables) per image.
    :return: model_object: Instance of `keras.models` with the aforementioned
        architecture.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_second_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the second-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method uses 5-by-5, rather than 3-by-3, kernels.

    :param num_rows: See doc for `get_first_architecture`.
    :param num_columns: Same.
    :param num_channels: Same.
    :return: model_object: Same.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=5,
        num_kernel_columns=5, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=5,
        num_kernel_columns=5, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_third_architecture(num_rows, num_columns, num_channels, l2_weight):
    """Creates 2-D CNN with the third-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method uses L2 regularization.

    :param num_rows: See doc for `get_first_architecture`.
    :param num_columns: Same.
    :param num_channels: Same.
    :param l2_weight: Weight for L2 regularization.
    :return: model_object: Same.
    """

    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_penalty=0, l2_penalty=l2_weight)

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=regularizer_object)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_fourth_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the fourth-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method uses batch normalization.

    :param num_rows: See doc for `get_first_architecture`.
    :param num_columns: Same.
    :param num_channels: Same.
    :return: model_object: Same.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))
    model_object.add(architecture_utils.get_batch_normalization_layer())

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))
    model_object.add(architecture_utils.get_batch_normalization_layer())

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_fifth_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the fifth-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method does not use dropout for conv layers.

    :param num_rows: Number of pixel rows per image.
    :param num_columns: Number of pixel columns per image.
    :param num_channels: Number of channels (predictor variables) per image.
    :return: model_object: Instance of `keras.models` with the aforementioned
        architecture.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_sixth_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the sixth-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method uses the Adam, instead of Adadelta, optimizer.

    :param num_rows: Number of pixel rows per image.
    :param num_columns: Number of pixel columns per image.
    :param num_channels: Number of channels (predictor variables) per image.
    :return: model_object: Instance of `keras.models` with the aforementioned
        architecture.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_seventh_architecture(num_rows, num_columns, num_channels):
    """Creates 2-D CNN with the seventh-simplest architecture.

    The only difference between this method and `get_first_architecture` is that
    this method uses padding, so convolution does not change the size of the
    feature maps.

    :param num_rows: Number of pixel rows per image.
    :param num_columns: Number of pixel columns per image.
    :param num_channels: Number of channels (predictor variables) per image.
    :return: model_object: Instance of `keras.models` with the aforementioned
        architecture.
    """

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.YES_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object


def get_eighth_architecture(
        num_rows, num_columns, num_channels, num_conv_layer_sets):
    """Creates 2-D CNN with the eighth-simplest architecture.

    Differences between this method and `get_first_architecture`:

    - This method does not use dropout for conv layers.
    - This method uses multiple conv layers.

    :param num_rows: Number of pixel rows per image.
    :param num_columns: Number of pixel columns per image.
    :param num_channels: Number of channels (predictor variables) per image.
    :param num_conv_layer_sets: Number of sets of conv layers.
    :return: model_object: Instance of `keras.models` with the aforementioned
        architecture.
    """

    error_checking.assert_is_integer(num_conv_layer_sets)
    error_checking.assert_is_greater(num_conv_layer_sets, 1)

    model_object = keras.models.Sequential()
    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None, is_first_layer=True,
        num_input_rows=num_rows, num_input_columns=num_columns,
        num_input_channels=num_channels)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_conv_layer(
        num_output_filters=2 * INIT_NUM_FILTERS, num_kernel_rows=3,
        num_kernel_columns=3, num_rows_per_stride=1, num_columns_per_stride=1,
        padding_type=architecture_utils.NO_PADDING_TYPE,
        kernel_weight_regularizer=None)
    model_object.add(layer_object)

    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_2d_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=architecture_utils.MAX_POOLING_TYPE, num_rows_per_stride=2,
        num_columns_per_stride=2)
    model_object.add(layer_object)

    this_num_output_filters = 4 * INIT_NUM_FILTERS
    for i in range(1, num_conv_layer_sets):
        if i != 1:
            this_num_output_filters *= 2

        layer_object = architecture_utils.get_2d_conv_layer(
            num_output_filters=this_num_output_filters, num_kernel_rows=3,
            num_kernel_columns=3, num_rows_per_stride=1,
            num_columns_per_stride=1,
            padding_type=architecture_utils.NO_PADDING_TYPE,
            kernel_weight_regularizer=None, is_first_layer=False)
        model_object.add(layer_object)

        model_object.add(architecture_utils.get_activation_layer(
            activation_function_string='relu'))

        layer_object = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=architecture_utils.MAX_POOLING_TYPE,
            num_rows_per_stride=2,
            num_columns_per_stride=2)
        model_object.add(layer_object)

    layer_object = architecture_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = architecture_utils.get_fully_connected_layer(
        num_output_units=128)
    model_object.add(layer_object)
    model_object.add(architecture_utils.get_activation_layer(
        activation_function_string='relu'))

    layer_object = architecture_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    (dense_layer_object, activation_layer_object, loss_function
    ) = _get_output_layer_and_loss_function(NUM_CLASSES)
    model_object.add(dense_layer_object)
    model_object.add(activation_layer_object)

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adadelta(),
        metrics=DEFAULT_METRIC_FUNCTION_LIST)

    model_object.summary()
    return model_object
