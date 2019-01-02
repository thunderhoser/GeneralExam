"""Methods for setting up, training, and applying upconvolution networks."""

import pickle
from random import shuffle
import numpy
import keras
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import upconvnet as gg_upconvnet
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import training_validation_io as trainval_io

# TODO(thunderhoser): This code contains a lot of hacks, including constants
# that shouldn't really be constants.

L1_WEIGHT = 0.
L2_WEIGHT = 0.001
SLOPE_FOR_RELU = 0.2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3

LARGE_INTEGER = int(1e10)
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.005
NUM_EPOCHS_FOR_EARLY_STOP = 5

TRAINING_DIR_NAME_KEY = 'top_training_dir_name'
FIRST_TRAINING_TIME_KEY = 'first_training_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'last_training_time_unix_sec'
CNN_FILE_NAME_KEY = 'cnn_model_file_name'
CNN_LAYER_NAME_KEY = 'cnn_feature_layer_name'
NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_DIR_NAME_KEY = 'top_validation_dir_name'
FIRST_VALIDATION_TIME_KEY = 'first_validation_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'last_validation_time_unix_sec'

MODEL_METADATA_KEYS = [
    TRAINING_DIR_NAME_KEY, FIRST_TRAINING_TIME_KEY, LAST_TRAINING_TIME_KEY,
    CNN_FILE_NAME_KEY, CNN_LAYER_NAME_KEY, NUM_EPOCHS_KEY,
    NUM_EXAMPLES_PER_BATCH_KEY, NUM_TRAINING_BATCHES_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_DIR_NAME_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY
]


def _trainval_generator(
        top_input_dir_name, first_time_unix_sec, last_time_unix_sec,
        narr_predictor_names, num_half_rows, num_half_columns,
        num_examples_per_batch, cnn_model_object, cnn_feature_layer_name):
    """Generates training or validation examples for upconvnet on the fly.

    E = number of examples
    M = number of rows in each grid
    N = number of columns in each grid
    C = number of channels (predictor variables)
    Z = number of scalar features (neurons in layer `cnn_feature_layer_name` of
        the CNN specified by `cnn_model_object`)

    :param top_input_dir_name: Name of top-level directory with downsized 3-D
        examples (two spatial dimensions).  Files therein will be found by
        `training_validation_io.find_downsized_3d_example_file` (with
        `shuffled = True`) and read by
        `training_validation_io.read_downsized_3d_examples`.
    :param first_time_unix_sec: First valid time.  Only examples with valid time
        in `first_time_unix_sec`...`last_time_unix_sec` will be kept.
    :param last_time_unix_sec: See above.
    :param narr_predictor_names: See doc for
        `training_validation_io.read_downsized_3d_examples`.
    :param num_half_rows: See doc for
        `training_validation_io.read_downsized_3d_examples`.
    :param num_half_columns: Same.
    :param num_examples_per_batch: Number of examples in each batch.
    :param cnn_model_object: Trained CNN model (instance of
        `keras.models.Model`).  This will be used to turn images stored in
        `top_input_dir_name` into scalar features.
    :param cnn_feature_layer_name: The "scalar features" will be the set of
        activations from this layer.
    :return: feature_matrix: E-by-Z numpy array of scalar features.  These are
        the "predictors" for the upconv network.
    :return: target_matrix: E-by-M-by-N-by-C numpy array of target images.
        These are the predictors for the CNN and the targets for the upconv
        network.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)

    partial_cnn_model_object = cnn.model_to_feature_generator(
        model_object=cnn_model_object, output_layer_name=cnn_feature_layer_name)

    example_file_names = trainval_io.find_downsized_3d_example_files(
        top_directory_name=top_input_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=LARGE_INTEGER)
    shuffle(example_file_names)

    num_files = len(example_file_names)
    file_index = 0
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    num_examples_in_memory = 0
    full_target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])

            this_example_dict = trainval_io.read_downsized_3d_examples(
                netcdf_file_name=example_file_names[file_index],
                predictor_names_to_keep=narr_predictor_names,
                num_half_rows_to_keep=num_half_rows,
                num_half_columns_to_keep=num_half_columns,
                first_time_to_keep_unix_sec=first_time_unix_sec,
                last_time_to_keep_unix_sec=last_time_unix_sec)

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            this_num_examples = len(
                this_example_dict[trainval_io.TARGET_TIMES_KEY]
            )
            if this_num_examples == 0:
                continue

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_target_matrix = (
                    this_example_dict[trainval_io.PREDICTOR_MATRIX_KEY] + 0.
                )
            else:
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix,
                     this_example_dict[trainval_io.PREDICTOR_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        target_matrix = full_target_matrix[batch_indices, ...].astype('float32')
        feature_matrix = partial_cnn_model_object.predict(
            target_matrix, batch_size=num_examples_per_batch)

        num_examples_in_memory = 0
        full_target_matrix = None

        yield (feature_matrix, target_matrix)


def create_net(
        num_input_features, first_num_rows, first_num_columns,
        upsampling_factors, num_output_channels,
        use_activation_for_out_layer=False, use_bn_for_out_layer=True,
        use_transposed_conv=False, use_conv_for_out_layer=True,
        smoothing_radius_px=None):
    """Creates (but does not train) upconvnet.

    L = number of conv or deconv layers

    :param num_input_features: Number of input features.
    :param first_num_rows: Number of rows in input to first deconv layer.  The
        input features will be reshaped into a grid with this many rows.
    :param first_num_columns: Same but for columns.
    :param upsampling_factors: length-L numpy array of upsampling factors.  Must
        all be positive integers.
    :param num_output_channels: Number of channels in output images.
    :param use_activation_for_out_layer: Boolean flag.  If True, activation will
        be applied to output layer.
    :param use_bn_for_out_layer: Boolean flag.  If True, batch normalization
        will be applied to output layer.
    :param use_transposed_conv: Boolean flag.  If True, upsampling will be done
        with transposed-convolution layers.  If False, each upsampling will be
        done with an upsampling layer followed by a conv layer.
    :param use_conv_for_out_layer: Boolean flag.  If True, will do normal (not
        transposed) convolution for output layer, after zero-padding.  If False,
        will just do zero-padding.
    :param smoothing_radius_px: Smoothing radius (pixels).  Gaussian smoothing
        with this e-folding radius will be done after each upsampling.  If
        `smoothing_radius_px is None`, no smoothing will be done.
    :return: ucn_model_object: Untrained instance of `keras.models.Model`.
    """

    error_checking.assert_is_integer(num_input_features)
    error_checking.assert_is_integer(first_num_rows)
    error_checking.assert_is_integer(first_num_columns)
    error_checking.assert_is_integer(num_output_channels)

    error_checking.assert_is_greater(num_input_features, 0)
    error_checking.assert_is_greater(first_num_rows, 0)
    error_checking.assert_is_greater(first_num_columns, 0)
    error_checking.assert_is_greater(num_output_channels, 0)

    error_checking.assert_is_integer_numpy_array(upsampling_factors)
    error_checking.assert_is_numpy_array(upsampling_factors, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(upsampling_factors, 1)

    error_checking.assert_is_boolean(use_activation_for_out_layer)
    error_checking.assert_is_boolean(use_bn_for_out_layer)
    error_checking.assert_is_boolean(use_transposed_conv)
    error_checking.assert_is_boolean(use_conv_for_out_layer)

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)
    input_layer_object = keras.layers.Input(shape=(num_input_features,))

    current_num_filters = int(numpy.round(
        num_input_features / (first_num_rows * first_num_columns)
    ))

    layer_object = keras.layers.Reshape(
        target_shape=(first_num_rows, first_num_columns, current_num_filters)
    )(input_layer_object)

    upsampling_factors = numpy.concatenate((
        upsampling_factors, numpy.array([-1], dtype=int)
    ))
    num_main_layers = len(upsampling_factors)

    for i in range(num_main_layers):
        this_upsampling_factor = upsampling_factors[i]
        do_smoothing_here = smoothing_radius_px is not None

        if i == num_main_layers - 2:
            current_num_filters = num_output_channels + 0
        elif this_upsampling_factor == 1:
            current_num_filters = int(numpy.round(current_num_filters / 2))

        if i == num_main_layers - 1:
            layer_object = keras.layers.ZeroPadding2D(
                padding=((1, 0), (1, 0)), data_format='channels_last'
            )(layer_object)

            if use_conv_for_out_layer:
                layer_object = keras.layers.Conv2D(
                    filters=current_num_filters,
                    kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                    strides=(1, 1), padding='same', data_format='channels_last',
                    dilation_rate=(1, 1), activation=None, use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=regularizer_object
                )(layer_object)
            else:
                do_smoothing_here = False

        elif use_transposed_conv:
            if this_upsampling_factor > 1:
                this_padding_arg = 'same'
            else:
                this_padding_arg = 'valid'

            layer_object = keras.layers.Conv2DTranspose(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(this_upsampling_factor, this_upsampling_factor),
                padding=this_padding_arg, data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

        else:
            if this_upsampling_factor > 1:
                try:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last', interpolation='nearest'
                    )(layer_object)
                except:
                    layer_object = keras.layers.UpSampling2D(
                        size=(this_upsampling_factor, this_upsampling_factor),
                        data_format='channels_last'
                    )(layer_object)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(layer_object)

            if this_upsampling_factor == 1:
                layer_object = keras.layers.ZeroPadding2D(
                    padding=(1, 1), data_format='channels_last'
                )(layer_object)

        if do_smoothing_here:
            this_weight_matrix = gg_upconvnet.create_smoothing_filter(
                smoothing_radius_px=smoothing_radius_px,
                num_channels=current_num_filters)

            this_bias_vector = numpy.zeros(current_num_filters)

            layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=this_weight_matrix.shape[:2],
                strides=(1, 1), padding='same', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object, trainable=False,
                weights=[this_weight_matrix, this_bias_vector]
            )(layer_object)

        if i < num_main_layers - 1 or use_activation_for_out_layer:
            layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(layer_object)

        if i < num_main_layers - 1 or use_bn_for_out_layer:
            layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(layer_object)

    ucn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object)
    ucn_model_object.compile(
        loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())

    ucn_model_object.summary()
    return ucn_model_object


def train_upconvnet(
        ucn_model_object, top_training_dir_name, first_training_time_unix_sec,
        last_training_time_unix_sec, cnn_model_object, cnn_feature_layer_name,
        cnn_metadata_dict, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name,
        num_validation_batches_per_epoch=None, top_validation_dir_name=None,
        first_validation_time_unix_sec=None, last_validation_time_unix_sec=None):
    """Trains upconvnet.

    :param ucn_model_object: Untrained instance of `keras.models.Model`,
        representing the upconv network.
    :param top_training_dir_name: Training data will be found here.  See doc for
        input `top_input_dir_name` to method `training_generator`.
    :param first_training_time_unix_sec: Determines training period.  See doc
        for input `first_time_unix_sec` to method `training_generator`.
    :param last_training_time_unix_sec: Determines training period.  See doc for
        input `last_time_unix_sec` to method `training_generator`.
    :param cnn_model_object: See doc for `training_generator`.
    :param cnn_feature_layer_name: Same.
    :param cnn_metadata_dict: Dictionary returned by
        `traditional_cnn.read_model_metadata`.
    :param num_examples_per_batch: Number of examples in each training or
        validation batch.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param num_validation_batches_per_epoch: Number of validation batches
        furnished to model in each epoch.  If
        `num_validation_batches_per_epoch is None`, there will be no on-the-fly
        validation.
    :param top_validation_dir_name:
        [used only if `num_validation_batches_per_epoch is not None`]
        Validation data will be found here.  See doc for input
        `top_input_dir_name` to method `training_generator`.
    :param first_validation_time_unix_sec:
        [used only if `num_validation_batches_per_epoch is not None`]
        Determines validation period.  See doc for input `first_time_unix_sec`
        to method `training_generator`.
    :param last_validation_time_unix_sec:
        [used only if `num_validation_batches_per_epoch is not None`]
        Determines validation period.  See doc for input `last_time_unix_sec`
        to method `training_generator`.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_model_file_name)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            output_model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min', period=1)
    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            output_model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

    list_of_callback_objects = [checkpoint_object]

    training_generator = _trainval_generator(
        top_input_dir_name=top_training_dir_name,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec,
        narr_predictor_names=cnn_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        num_half_rows=cnn_metadata_dict[
            traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
        num_half_columns=cnn_metadata_dict[
            traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
        num_examples_per_batch=num_examples_per_batch,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    if num_validation_batches_per_epoch is None:
        ucn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOP, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)

    validation_generator = _trainval_generator(
        top_input_dir_name=top_validation_dir_name,
        first_time_unix_sec=first_validation_time_unix_sec,
        last_time_unix_sec=last_validation_time_unix_sec,
        narr_predictor_names=cnn_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        num_half_rows=cnn_metadata_dict[
            traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
        num_half_columns=cnn_metadata_dict[
            traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
        num_examples_per_batch=num_examples_per_batch,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name)

    ucn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)


def write_model_metadata(
        pickle_file_name, top_training_dir_name, first_training_time_unix_sec,
        last_training_time_unix_sec, cnn_model_file_name,
        cnn_feature_layer_name, num_epochs, num_examples_per_batch,
        num_training_batches_per_epoch, num_validation_batches_per_epoch=None,
        top_validation_dir_name=None, first_validation_time_unix_sec=None,
        last_validation_time_unix_sec=None):
    """Writes metadata for upconvnet to Pickle file.

    :param pickle_file_name: Path to output file.
    :param top_training_dir_name: See doc for `train_upconvnet`.
    :param first_training_time_unix_sec: Same.
    :param last_training_time_unix_sec: Same.
    :param cnn_model_file_name: Same.
    :param cnn_feature_layer_name: Same.
    :param num_epochs: Same.
    :param num_examples_per_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_unix_sec: Same.
    :param last_validation_time_unix_sec: Same.
    """

    model_metadata_dict = {
        TRAINING_DIR_NAME_KEY: top_training_dir_name,
        FIRST_TRAINING_TIME_KEY: first_training_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_training_time_unix_sec,
        CNN_FILE_NAME_KEY: cnn_model_file_name,
        CNN_LAYER_NAME_KEY: cnn_feature_layer_name,
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_DIR_NAME_KEY: top_validation_dir_name,
        FIRST_VALIDATION_TIME_KEY: first_validation_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validation_time_unix_sec
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata for upconvnet from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_metadata_dict: Dictionary with the following keys.
    model_metadata_dict['top_training_dir_name']: See doc for
        `write_model_metadata`.
    model_metadata_dict['first_training_time_unix_sec']: Same.
    model_metadata_dict['last_training_time_unix_sec']: Same.
    model_metadata_dict['cnn_model_file_name']: Same.
    model_metadata_dict['cnn_feature_layer_name']: Same.
    model_metadata_dict['num_epochs']: Same.
    model_metadata_dict['num_examples_per_batch']: Same.
    model_metadata_dict['num_training_batches_per_epoch']: Same.
    model_metadata_dict['num_validation_batches_per_epoch']: Same.
    model_metadata_dict['top_validation_dir_name']: Same.
    model_metadata_dict['first_validation_time_unix_sec']: Same.
    model_metadata_dict['last_validation_time_unix_sec']: Same.

    :raises: ValueError: if any expected key is not found.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(
        set(MODEL_METADATA_KEYS) - set(model_metadata_dict.keys())
    )

    if len(missing_keys) == 0:
        return model_metadata_dict

    error_string = 'Cannot find the following expected keys.\n{0:s}'.format(
        str(missing_keys))
    raise ValueError(error_string)
