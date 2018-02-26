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

E = number of examples (images)
M = number of pixel rows in each image
N = number of pixel columns in each image
C = number of channels (predictor variables) in each image
"""

import numpy
import keras
from keras.models import Sequential
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import cnn_utils
from generalexam.machine_learning import machine_learning_io as ml_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import testing_io

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


def get_cnn_with_mnist_architecture():
    """Creates CNN with architecture used in the following example.

    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    Said architecture was used to classify handwritten digits from the MNIST
    (Modified National Institute of Standards and Technology) dataset.

    :return: model_object: Instance of `keras.models.Sequential`, with the
        aforementioned architecture.
    """

    model_object = Sequential()

    layer_object = cnn_utils.get_convolutional_layer(
        num_filters=32, num_rows_in_kernel=3, num_columns_in_kernel=3,
        stride_length_in_rows=1, stride_length_in_columns=1,
        activation_function='relu', is_first_layer=True,
        num_rows_per_image=DEFAULT_NUM_PIXEL_ROWS,
        num_columns_per_image=DEFAULT_NUM_PIXEL_COLUMNS,
        num_channels_per_image=len(DEFAULT_NARR_PREDICTOR_NAMES))
    model_object.add(layer_object)

    layer_object = cnn_utils.get_convolutional_layer(
        num_filters=64, num_rows_in_kernel=3, num_columns_in_kernel=3,
        stride_length_in_rows=1, stride_length_in_columns=1,
        activation_function='relu')
    model_object.add(layer_object)

    layer_object = cnn_utils.get_pooling_layer(
        num_rows_in_window=2, num_columns_in_window=2,
        pooling_type=cnn_utils.MAX_POOLING_TYPE)
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
        optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return model_object


def train_model_from_files(
        model_object, num_examples_per_batch, training_file_pattern, num_epochs,
        num_training_batches_per_epoch, validation_file_pattern=None,
        num_validation_batches_per_epoch=None):
    """Trains CNN, using examples read from pre-existing files.

    :param model_object: Instance of `keras.models.Sequential`.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param training_file_pattern: Glob pattern for training files (example:
        "downsized_examples/20170*/*.p").  All files matching this pattern will
        be used for training data.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
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

    if validation_file_pattern is None:
        model_object.fit_generator(
            generator=ml_io.downsized_3d_example_generator_from_files(
                input_file_pattern=training_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1)
    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        model_object.fit_generator(
            generator=ml_io.downsized_3d_example_generator_from_files(
                input_file_pattern=training_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1,
            validation_data=ml_io.downsized_3d_example_generator_from_files(
                input_file_pattern=validation_file_pattern,
                num_examples_per_batch=num_examples_per_batch),
            validation_steps=num_validation_batches_per_epoch)


def train_model_from_on_the_fly_examples(
        model_object, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_examples_per_time,
        training_start_time_unix_sec, training_end_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        positive_fraction, num_rows_in_half_grid, num_columns_in_half_grid,
        num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains CNN, using examples generated on the fly.

    :param model_object: Instance of `keras.models.Sequential`.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_examples_per_time: See documentation for
        `machine_learning_io.downsized_3d_example_generator`.
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
        `machine_learning_io.downsized_3d_example_generator`.
    :param validation_end_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)

    if num_validation_batches_per_epoch is None:
        model_object.fit_generator(
            generator=ml_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_time=num_examples_per_time,
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
            verbose=1)

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        model_object.fit_generator(
            generator=ml_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_time=num_examples_per_time,
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
            verbose=1,
            validation_data=ml_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_time=num_examples_per_time,
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


def apply_model_one_target_time(
        model_object, target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target, num_rows_in_half_grid,
        num_columns_in_half_grid):
    """Applies CNN to one target time.

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
        if i == 0:
            (this_downsized_predictor_matrix,
             actual_target_matrix[:, i, :],
             full_predictor_matrix,
             full_target_matrix) = testing_io.create_downsized_3d_examples(
                 narr_row_index=i, num_rows_in_half_grid=num_rows_in_half_grid,
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
                 narr_row_index=i,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 full_predictor_matrix=full_predictor_matrix,
                 full_target_matrix=full_target_matrix)

        this_prediction_matrix = model_object.predict(
            this_downsized_predictor_matrix,
            batch_size=NUM_NARR_COLUMNS_WITHOUT_NAN)
        predicted_target_matrix[:, i, :] = this_prediction_matrix[:, 1]

    return predicted_target_matrix, actual_target_matrix
