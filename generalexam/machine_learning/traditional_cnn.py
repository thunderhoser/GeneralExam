"""Training and testing methods for traditional CNN (convolutional neural net).

A "traditional CNN" is one for which the output (prediction) is not spatially
explicit.  The opposite is a fully convolutional net (FCN; see fcn.py).

For a traditional CNN, there is one output (prediction) per image, rather than
one per pixel.  Thus, a traditional CNN may predict whether or not the image
contains some feature (e.g., atmospheric front), but it may *not* predict where
said features are in the image.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples.  Each example is one image or a time sequence of images.
M = number of pixel rows in each image
N = number of pixel columns in each image
T = number of predictor times per example (images per sequence)
C = number of channels (predictor variables) in each image
"""

import numpy
import keras.losses
import keras.optimizers
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import cnn_utils
from generalexam.machine_learning import training_validation_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import keras_metrics

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

DEFAULT_ASSUMED_POSITIVE_FRACTION = 0.935

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.csi, keras_metrics.frequency_bias,
    keras_metrics.pod, keras_metrics.pofd, keras_metrics.success_ratio,
    keras_metrics.focn]

NUM_CLASSES = 2
DEFAULT_NUM_PIXEL_ROWS = 65
DEFAULT_NUM_PIXEL_COLUMNS = 65

DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_TEMP_NAME]

NUM_NARR_ROWS_WITHOUT_NAN, _ = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)
NUM_NARR_COLUMNS_WITHOUT_NAN = len(ml_utils.NARR_COLUMNS_WITHOUT_NAN)


def get_cnn_with_mnist_architecture(
        narr_predictor_names=DEFAULT_NARR_PREDICTOR_NAMES,
        num_dimensions_per_example=3, num_predictor_times_per_example=None):
    """Creates CNN with architecture used in the following example.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    Said architecture was used to classify handwritten digits from the MNIST
    (Modified National Institute of Standards and Technology) dataset.

    :param narr_predictor_names: length-C list of NARR fields to use as
        predictors.
    :param num_dimensions_per_example: Number of dimensions per training example
        (either 3 or 4).  If 3, the CNN will do 2-D convolution (over the x- and
        y-dimensions).  If 4, the CNN will do 3-D convolution (over dimensions
        x, y, t).
    :param num_predictor_times_per_example: Number of predictor times per
        example (images per sequence).
    :return: model_object: Instance of `keras.models.Sequential`, with the
        aforementioned architecture.
    """

    error_checking.assert_is_integer(num_dimensions_per_example)
    error_checking.assert_is_geq(num_dimensions_per_example, 3)
    error_checking.assert_is_leq(num_dimensions_per_example, 4)

    error_checking.assert_is_string_list(narr_predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(narr_predictor_names), num_dimensions=1)
    num_predictors = len(narr_predictor_names)

    model_object = Sequential()

    if num_dimensions_per_example == 3:
        layer_object = cnn_utils.get_2d_convolution_layer(
            num_filters=32, num_kernel_rows=3, num_kernel_columns=3,
            stride_length_in_rows=1, stride_length_in_columns=1,
            activation_function='relu', is_first_layer=True,
            num_input_rows=DEFAULT_NUM_PIXEL_ROWS,
            num_input_columns=DEFAULT_NUM_PIXEL_COLUMNS,
            num_input_channels=num_predictors)
    else:
        error_checking.assert_is_integer(num_predictor_times_per_example)
        error_checking.assert_is_greater(num_predictor_times_per_example, 0)

        layer_object = cnn_utils.get_3d_convolution_layer(
            num_filters=32, num_kernel_rows=3, num_kernel_columns=3,
            num_kernel_time_steps=3, stride_length_in_rows=1,
            stride_length_in_columns=1, stride_length_in_time_steps=1,
            activation_function='relu', is_first_layer=True,
            num_input_rows=DEFAULT_NUM_PIXEL_ROWS,
            num_input_columns=DEFAULT_NUM_PIXEL_COLUMNS,
            num_input_time_steps=num_predictor_times_per_example,
            num_input_channels=num_predictors)

    model_object.add(layer_object)

    if num_dimensions_per_example == 3:
        layer_object = cnn_utils.get_2d_convolution_layer(
            num_filters=64, num_kernel_rows=3, num_kernel_columns=3,
            stride_length_in_rows=1, stride_length_in_columns=1,
            activation_function='relu')
    else:
        layer_object = cnn_utils.get_3d_convolution_layer(
            num_filters=64, num_kernel_rows=3, num_kernel_columns=3,
            num_kernel_time_steps=3, stride_length_in_rows=1,
            stride_length_in_columns=1, stride_length_in_time_steps=1,
            activation_function='relu')

    model_object.add(layer_object)

    if num_dimensions_per_example == 3:
        layer_object = cnn_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            pooling_type=cnn_utils.MAX_POOLING_TYPE)
    else:
        layer_object = cnn_utils.get_3d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_time_steps_in_window=2, pooling_type=cnn_utils.MAX_POOLING_TYPE)

    model_object.add(layer_object)

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.25)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_flattening_layer()
    model_object.add(layer_object)

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=128, activation_function='relu')
    model_object.add(layer_object)

    layer_object = cnn_utils.get_dropout_layer(dropout_fraction=0.5)
    model_object.add(layer_object)

    layer_object = cnn_utils.get_fully_connected_layer(
        num_output_units=NUM_CLASSES, activation_function='softmax')
    model_object.add(layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), metrics=LIST_OF_METRIC_FUNCTIONS)
    return model_object


def train_with_3d_examples_from_files(
        model_object, output_file_name, num_examples_per_batch,
        training_file_pattern, num_epochs, num_training_batches_per_epoch,
        assumed_positive_fraction=DEFAULT_ASSUMED_POSITIVE_FRACTION,
        validation_file_pattern=None, num_validation_batches_per_epoch=None):
    """Trains CNN, using 3-D examples read from pre-existing files.

    :param model_object: Instance of `keras.models.Sequential`.
    :param output_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param training_file_pattern: Glob pattern for training files (example:
        "downsized_examples/20170*/*.p").  All files matching this pattern will
        be used for training data.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param assumed_positive_fraction: Assumed fraction of positive cases.  This
        will be used to weight classes in the loss function.
    :param validation_file_pattern: Glob pattern for validation files (example:
        "downsized_examples/20171*/*.p").  All files matching this pattern will
        be used for validation data.  If you want no validation data, leave this
        as None.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    class_frequencies = numpy.array(
        [1. - assumed_positive_fraction, assumed_positive_fraction])
    class_weight_dict = ml_utils.get_class_weight_dict(class_frequencies)

    if validation_file_pattern is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=
            training_validation_io.downsized_3d_example_generator_from_files(
                input_file_pattern=training_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object])
    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=
            training_validation_io.downsized_3d_example_generator_from_files(
                input_file_pattern=training_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object],
            validation_data=
            training_validation_io.downsized_3d_example_generator_from_files(
                input_file_pattern=validation_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            validation_steps=num_validation_batches_per_epoch)


def train_with_3d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_examples_per_target_time,
        training_start_time_unix_sec, training_end_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        positive_fraction, num_rows_in_half_grid, num_columns_in_half_grid,
        num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains CNN, using 3-D examples created on the fly.

    :param model_object: Instance of `keras.models.Sequential`.
    :param output_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_examples_per_target_time: See documentation for
        `training_validation_io.downsized_3d_example_generator`.
    :param training_start_time_unix_sec: Same.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :param positive_fraction: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_start_time_unix_sec: See documentation for
        `training_validation_io.downsized_3d_example_generator`.
    :param validation_end_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    class_frequencies = numpy.array([1. - positive_fraction, positive_fraction])
    class_weight_dict = ml_utils.get_class_weight_dict(class_frequencies)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object],
            validation_data=
            training_validation_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            validation_steps=num_validation_batches_per_epoch)


def train_with_4d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_examples_per_target_time,
        num_predictor_time_steps, num_lead_time_steps,
        training_start_time_unix_sec, training_end_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        positive_fraction, num_rows_in_half_grid, num_columns_in_half_grid,
        num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains CNN, using 4-D examples created on the fly.

    :param model_object: See documentation for `train_with_3d_examples`.
    :param output_file_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_examples_per_target_time: Same.
    :param num_predictor_time_steps: Number of predictor times per example.
    :param num_lead_time_steps: Number of time steps separating latest predictor
        time from target time.
    :param training_start_time_unix_sec: See documentation for
        `train_with_3d_examples`.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :param positive_fraction: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_start_time_unix_sec: Same.
    :param validation_end_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    class_frequencies = numpy.array([1. - positive_fraction, positive_fraction])
    class_weight_dict = ml_utils.get_class_weight_dict(class_frequencies)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=training_validation_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object],
            validation_data=
            training_validation_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target,
                positive_fraction=positive_fraction,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            validation_steps=num_validation_batches_per_epoch)


def apply_model_to_3d_example(
        model_object, target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target, num_rows_in_half_grid,
        num_columns_in_half_grid):
    """Applies trained CNN to one 3-D example.

    :param model_object: Instance of `keras.models.Sequential`.
    :param target_time_unix_sec: See documentation for
        `testing_io.create_downsized_3d_examples`.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :return: predicted_target_matrix: 1-by-M-by-N numpy array of predicted
        targets on the NARR grid.
    :return: actual_target_matrix: 1-by-M-by-N numpy array of actual targets on
        the NARR grid.
    """

    predicted_target_matrix = numpy.full(
        (1, NUM_NARR_ROWS_WITHOUT_NAN, NUM_NARR_COLUMNS_WITHOUT_NAN), -1,
        dtype=float)
    actual_target_matrix = numpy.full(
        (1, NUM_NARR_ROWS_WITHOUT_NAN, NUM_NARR_COLUMNS_WITHOUT_NAN), -1,
        dtype=int)

    full_predictor_matrix = None
    full_target_matrix = None

    for i in range(NUM_NARR_ROWS_WITHOUT_NAN):
        these_center_row_indices = numpy.linspace(
            i, i, num=NUM_NARR_COLUMNS_WITHOUT_NAN, dtype=int)
        these_center_column_indices = numpy.linspace(
            0, NUM_NARR_COLUMNS_WITHOUT_NAN - 1,
            num=NUM_NARR_COLUMNS_WITHOUT_NAN, dtype=int)

        if i == 0:
            (this_downsized_predictor_matrix,
             actual_target_matrix[:, i, :],
             full_predictor_matrix,
             full_target_matrix) = testing_io.create_downsized_3d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 target_time_unix_sec=target_time_unix_sec,
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_half_width_for_target=dilation_half_width_for_target)

        else:
            (this_downsized_predictor_matrix,
             actual_target_matrix[:, i, :],
             _, _) = testing_io.create_downsized_3d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 full_predictor_matrix=full_predictor_matrix,
                 full_target_matrix=full_target_matrix)

        this_prediction_matrix = model_object.predict(
            this_downsized_predictor_matrix,
            batch_size=NUM_NARR_COLUMNS_WITHOUT_NAN)
        predicted_target_matrix[:, i, :] = this_prediction_matrix[:, 1]

    return predicted_target_matrix, actual_target_matrix


def apply_model_to_4d_example(
        model_object, target_time_unix_sec, num_predictor_time_steps,
        num_lead_time_steps, top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        num_rows_in_half_grid, num_columns_in_half_grid):
    """Applies trained CNN to one 4-D example.

    :param model_object: Instance of `keras.models.Sequential`.
    :param target_time_unix_sec: See documentation for
        `testing_io.create_downsized_4d_examples`.
    :param num_predictor_time_steps: Same.
    :param num_lead_time_steps: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :return: predicted_target_matrix: 1-by-M-by-N numpy array of predicted
        targets on the NARR grid.
    :return: actual_target_matrix: 1-by-M-by-N numpy array of actual targets on
        the NARR grid.
    """

    predicted_target_matrix = numpy.full(
        (1, NUM_NARR_ROWS_WITHOUT_NAN, NUM_NARR_COLUMNS_WITHOUT_NAN), -1,
        dtype=float)
    actual_target_matrix = numpy.full(
        (1, NUM_NARR_ROWS_WITHOUT_NAN, NUM_NARR_COLUMNS_WITHOUT_NAN), -1,
        dtype=int)

    full_predictor_matrix = None
    full_target_matrix = None

    for i in range(NUM_NARR_ROWS_WITHOUT_NAN):
        these_center_row_indices = numpy.linspace(
            i, i, num=NUM_NARR_COLUMNS_WITHOUT_NAN, dtype=int)
        these_center_column_indices = numpy.linspace(
            0, NUM_NARR_COLUMNS_WITHOUT_NAN - 1,
            num=NUM_NARR_COLUMNS_WITHOUT_NAN, dtype=int)

        if i == 0:
            (this_downsized_predictor_matrix,
             actual_target_matrix[:, i, :],
             full_predictor_matrix,
             full_target_matrix) = testing_io.create_downsized_4d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 target_time_unix_sec=target_time_unix_sec,
                 num_predictor_time_steps=num_predictor_time_steps,
                 num_lead_time_steps=num_lead_time_steps,
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_half_width_for_target=dilation_half_width_for_target)

        else:
            (this_downsized_predictor_matrix,
             actual_target_matrix[:, i, :],
             _, _) = testing_io.create_downsized_4d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 full_predictor_matrix=full_predictor_matrix,
                 full_target_matrix=full_target_matrix)

        this_prediction_matrix = model_object.predict(
            this_downsized_predictor_matrix,
            batch_size=NUM_NARR_COLUMNS_WITHOUT_NAN)
        predicted_target_matrix[:, i, :] = this_prediction_matrix[:, 1]

    return predicted_target_matrix, actual_target_matrix
