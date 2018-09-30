"""Training and deployment methods for a traditional CNN.

A "traditional CNN" is a convolutional neural net with only one target variable,
whereas an FCN (fully convolutional net) has one target variable at each pixel.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of downsized examples
K = number of classes (possible target values).  See below for the definition of
    "target value".
M = number of spatial rows per example
N = number of spatial columns per example
T = number of predictor times per example (number of images per sequence)
C = number of channels (predictor variables) per example

--- DEFINITIONS ---

A "downsized" example covers only a subset of the NARR grid, while a full-size
example covers the entire NARR grid.

The dimensions of a 3-D example are M x N x C (only one predictor time).

The dimensions of a 4-D example are M x N x T x C.

NF = no front
WF = warm front
CF = cold front

Target variable = label at one pixel.  For a downsized example, there is only
one target variable (the label at the center pixel).
"""

import pickle
import os.path
import numpy
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import keras_metrics

NUM_EPOCHS_KEY = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TARGET_TIME_KEY = 'num_examples_per_target_time'
NUM_TRAINING_BATCHES_PER_EPOCH_KEY = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_PER_EPOCH_KEY = 'num_validation_batches_per_epoch'
NUM_ROWS_IN_HALF_GRID_KEY = 'num_rows_in_half_grid'
NUM_COLUMNS_IN_HALF_GRID_KEY = 'num_columns_in_half_grid'
DILATION_DISTANCE_FOR_TARGET_KEY = 'dilation_distance_for_target_metres'
CLASS_FRACTIONS_KEY = 'class_fractions'
WEIGHT_LOSS_FUNCTION_KEY = 'weight_loss_function'
NARR_PREDICTOR_NAMES_KEY = 'narr_predictor_names'
PRESSURE_LEVEL_KEY = 'pressure_level_mb'
TRAINING_START_TIME_KEY = 'training_start_time_unix_sec'
TRAINING_END_TIME_KEY = 'training_end_time_unix_sec'
VALIDATION_START_TIME_KEY = 'validation_start_time_unix_sec'
VALIDATION_END_TIME_KEY = 'validation_end_time_unix_sec'
NUM_PREDICTOR_TIME_STEPS_KEY = 'num_predictor_time_steps'
PREDICTOR_TIME_STEP_OFFSETS_KEY = 'predictor_time_step_offsets'
NUM_LEAD_TIME_STEPS_KEY = 'num_lead_time_steps'
NARR_MASK_MATRIX_KEY = 'narr_mask_matrix'

MODEL_METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_EXAMPLES_PER_BATCH_KEY,
    NUM_EXAMPLES_PER_TARGET_TIME_KEY, NUM_TRAINING_BATCHES_PER_EPOCH_KEY,
    NUM_VALIDATION_BATCHES_PER_EPOCH_KEY, NUM_ROWS_IN_HALF_GRID_KEY,
    NUM_COLUMNS_IN_HALF_GRID_KEY, DILATION_DISTANCE_FOR_TARGET_KEY,
    CLASS_FRACTIONS_KEY, WEIGHT_LOSS_FUNCTION_KEY, NARR_PREDICTOR_NAMES_KEY,
    PRESSURE_LEVEL_KEY, TRAINING_START_TIME_KEY, TRAINING_END_TIME_KEY,
    VALIDATION_START_TIME_KEY, VALIDATION_END_TIME_KEY,
    NUM_PREDICTOR_TIME_STEPS_KEY, PREDICTOR_TIME_STEP_OFFSETS_KEY,
    NUM_LEAD_TIME_STEPS_KEY, NARR_MASK_MATRIX_KEY
]

CUSTOM_OBJECT_DICT_FOR_READING_MODEL = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile (should be written by `write_model_metadata`).

    :param model_file_name: Path to HDF5 file, containing the trained model.
    :param raise_error_if_missing: Boolean flag.  If metafile is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: metafile_name: Path to metafile.  If file is missing and
        `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if metafile is missing and
        `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])
    if not os.path.isfile(metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name)
        raise ValueError(error_string)

    return metafile_name


def write_model_metadata(
        pickle_file_name, num_epochs, num_examples_per_batch,
        num_examples_per_target_time, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, num_rows_in_half_grid,
        num_columns_in_half_grid, dilation_distance_metres, class_fractions,
        weight_loss_function, narr_predictor_names, pressure_level_mb,
        training_start_time_unix_sec, training_end_time_unix_sec,
        validation_start_time_unix_sec, validation_end_time_unix_sec,
        num_lead_time_steps=None, predictor_time_step_offsets=None,
        narr_mask_matrix=None):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_with_3d_examples`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_target_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param dilation_distance_metres: Same.
    :param class_fractions: Same.
    :param weight_loss_function: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param training_start_time_unix_sec: Same.
    :param training_end_time_unix_sec: Same.
    :param validation_start_time_unix_sec: Same.
    :param validation_end_time_unix_sec: Same.
    :param num_lead_time_steps: See doc for `train_with_4d_examples`.
    :param predictor_time_step_offsets: Same.
    :param narr_mask_matrix: See doc for `train_with_3d_examples`.
    """

    model_metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_EXAMPLES_PER_TARGET_TIME_KEY: num_examples_per_target_time,
        NUM_TRAINING_BATCHES_PER_EPOCH_KEY: num_training_batches_per_epoch,
        NUM_VALIDATION_BATCHES_PER_EPOCH_KEY: num_validation_batches_per_epoch,
        NUM_ROWS_IN_HALF_GRID_KEY: num_rows_in_half_grid,
        NUM_COLUMNS_IN_HALF_GRID_KEY: num_columns_in_half_grid,
        DILATION_DISTANCE_FOR_TARGET_KEY: dilation_distance_metres,
        CLASS_FRACTIONS_KEY: class_fractions,
        WEIGHT_LOSS_FUNCTION_KEY: weight_loss_function,
        NARR_PREDICTOR_NAMES_KEY: narr_predictor_names,
        PRESSURE_LEVEL_KEY: pressure_level_mb,
        TRAINING_START_TIME_KEY: training_start_time_unix_sec,
        TRAINING_END_TIME_KEY: training_end_time_unix_sec,
        VALIDATION_START_TIME_KEY: validation_start_time_unix_sec,
        VALIDATION_END_TIME_KEY: validation_end_time_unix_sec,
        PREDICTOR_TIME_STEP_OFFSETS_KEY: predictor_time_step_offsets,
        NUM_LEAD_TIME_STEPS_KEY: num_lead_time_steps,
        NARR_MASK_MATRIX_KEY: narr_mask_matrix
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_metadata_dict: Dictionary with all keys in the list
        `MODEL_METADATA_KEYS`.
    :raises: ValueError: if dictionary does not contain all keys in the list
        `MODEL_METADATA_KEYS`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if PREDICTOR_TIME_STEP_OFFSETS_KEY not in model_metadata_dict:
        model_metadata_dict.update({PREDICTOR_TIME_STEP_OFFSETS_KEY: None})
    if NUM_PREDICTOR_TIME_STEPS_KEY not in model_metadata_dict:
        model_metadata_dict.update({NUM_PREDICTOR_TIME_STEPS_KEY: None})
    if NARR_MASK_MATRIX_KEY not in model_metadata_dict:
        model_metadata_dict.update({NARR_MASK_MATRIX_KEY: None})

    expected_keys_as_set = set(MODEL_METADATA_KEYS)
    actual_keys_as_set = set(model_metadata_dict.keys())
    if not set(expected_keys_as_set).issubset(actual_keys_as_set):
        error_string = (
            '\n\n{0:s}\nExpected keys are listed above.  Keys found in file '
            '("{1:s}") are listed below.  Some expected keys were not found.'
            '\n{2:s}\n').format(MODEL_METADATA_KEYS, pickle_file_name,
                                model_metadata_dict.keys())

        raise ValueError(error_string)

    return model_metadata_dict


def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: keras_model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return load_model(
        hdf5_file_name, custom_objects=CUSTOM_OBJECT_DICT_FOR_READING_MODEL)


def train_with_3d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_examples_per_target_time,
        training_start_time_unix_sec, training_end_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        class_fractions, num_rows_in_half_grid, num_columns_in_half_grid,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None,
        narr_mask_matrix=None):
    """Trains CNN, using 3-D examples created on the fly.

    :param model_object: Instance of `keras.models.Sequential`.
    :param output_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_examples_per_target_time: See doc for
        `training_validation_io.downsized_3d_example_generator`.
    :param training_start_time_unix_sec: Same.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param class_fractions: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param weight_loss_function: Boolean flag.  If True, classes will be
        weighted differently in the loss function (class weights inversely
        proportional to `class_fractions`).
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_start_time_unix_sec: See doc for
        `training_validation_io.downsized_3d_example_generator`.
    :param validation_end_time_unix_sec: Same.
    :param narr_mask_matrix: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)
    error_checking.assert_is_boolean(weight_loss_function)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if weight_loss_function:
        class_weight_dict = ml_utils.get_class_weight_dict(class_fractions)
    else:
        class_weight_dict = None

    if num_validation_batches_per_epoch is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
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
            generator=trainval_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object],
            validation_data=
            trainval_io.downsized_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
            validation_steps=num_validation_batches_per_epoch)


def quick_train_3d(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, training_start_time_unix_sec,
        training_end_time_unix_sec, top_input_dir_name, narr_predictor_names,
        num_classes, num_rows_in_half_grid, num_columns_in_half_grid,
        num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains CNN with 3-D examples stored in processed files.

    These "processed files" are created by
    `training_validation_io.write_downsized_3d_examples`.

    :param model_object: See doc for `train_with_3d_examples`.
    :param output_file_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_start_time_unix_sec: See doc for
        `training_validation_io.quick_downsized_3d_example_gen`.
    :param training_end_time_unix_sec: Same.
    :param top_input_dir_name: Same.
    :param narr_predictor_names: Same.
    :param num_classes: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_start_time_unix_sec: See doc for
        `training_validation_io.quick_downsized_3d_example_gen`.
    :param validation_end_time_unix_sec: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if num_validation_batches_per_epoch is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.quick_downsized_3d_example_gen(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_input_dir_name=top_input_dir_name,
                narr_predictor_names=narr_predictor_names,
                num_classes=num_classes,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=None, callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.quick_downsized_3d_example_gen(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_input_dir_name=top_input_dir_name,
                narr_predictor_names=narr_predictor_names,
                num_classes=num_classes,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=None, callbacks=[checkpoint_object],
            validation_data=trainval_io.quick_downsized_3d_example_gen(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                top_input_dir_name=top_input_dir_name,
                narr_predictor_names=narr_predictor_names,
                num_classes=num_classes,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid),
            validation_steps=num_validation_batches_per_epoch)


def train_with_4d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_examples_per_target_time,
        predictor_time_step_offsets, num_lead_time_steps,
        training_start_time_unix_sec, training_end_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        class_fractions, num_rows_in_half_grid, num_columns_in_half_grid,
        weight_loss_function=True, num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None,
        narr_mask_matrix=None):
    """Trains CNN, using 4-D examples created on the fly.

    :param model_object: See doc for `train_with_3d_examples`.
    :param output_file_name: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_examples_per_target_time: Same.
    :param predictor_time_step_offsets: length-T numpy array of offsets between
        predictor times and (target time - lead time).
    :param num_lead_time_steps: Number of time steps separating latest predictor
        time from target time.
    :param training_start_time_unix_sec: See doc for `train_with_3d_examples`.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param class_fractions: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param weight_loss_function: Boolean flag.  If True, classes will be
        weighted differently in the loss function (class weights inversely
        proportional to `class_fractions`).
    :param num_validation_batches_per_epoch: See doc for
        `train_with_3d_examples`.
    :param validation_start_time_unix_sec: Same.
    :param validation_end_time_unix_sec: Same.
    :param narr_mask_matrix: Same.
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 1)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 1)
    error_checking.assert_is_boolean(weight_loss_function)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if weight_loss_function:
        class_weight_dict = ml_utils.get_class_weight_dict(class_fractions)
    else:
        class_weight_dict = None

    if num_validation_batches_per_epoch is None:
        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                predictor_time_step_offsets=predictor_time_step_offsets,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
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
            generator=trainval_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                predictor_time_step_offsets=predictor_time_step_offsets,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, class_weight=class_weight_dict,
            callbacks=[checkpoint_object],
            validation_data=
            trainval_io.downsized_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                num_examples_per_target_time=num_examples_per_target_time,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                predictor_time_step_offsets=predictor_time_step_offsets,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                class_fractions=class_fractions,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                narr_mask_matrix=narr_mask_matrix),
            validation_steps=num_validation_batches_per_epoch)


def apply_model_to_3d_example(
        model_object, target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_rows_in_half_grid,
        num_columns_in_half_grid, num_classes,
        isotonic_model_object_by_class=None, narr_mask_matrix=None):
    """Applies trained CNN to a 3-D example.

    :param model_object: Trained instance of `keras.models.Sequential`.
    :param target_time_unix_sec: See doc for
        `testing_io.create_downsized_3d_examples`.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_classes: Same.
    :param isotonic_model_object_by_class: length-K list with trained instances
        of `sklearn.isotonic.IsotonicRegression`.  If
        `isotonic_model_object_by_class is None`, will not use isotonic
        regression.
    :param narr_mask_matrix: M-by-N numpy array of integers (0 or 1).  If
        narr_mask_matrix[i, j] = 0, the model will not be applied to grid cell
        [i, j].  If `narr_mask_matrix is None`, the model will be applied to all
        grid cells.
    :return: class_probability_matrix: 1-by-M-by-N-by-K numpy array of predicted
        class probabilities.  If grid cell [i, j] is masked out (due to
        `narr_mask_matrix`), class_probability_matrix[0, i, j, :] = NaN.
    :return: target_matrix: 1-by-M-by-N numpy array with actual target classes.
        If grid cell [i, j] is masked out (due to `narr_mask_matrix`),
        target_matrix[0, i, j] = -1.
    """

    if narr_mask_matrix is None:
        narr_mask_matrix = numpy.full(
            (NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR), 1, dtype=int)

    ml_utils.check_narr_mask(narr_mask_matrix)

    class_probability_matrix = numpy.full(
        (1, NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR, num_classes), numpy.nan)
    target_matrix = numpy.full(
        (1, NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR), -1, dtype=int)

    full_predictor_matrix = None
    full_target_matrix = None

    for i in range(NUM_ROWS_IN_NARR):
        these_column_indices = numpy.where(narr_mask_matrix[i, :] == 1)[0]
        if len(these_column_indices) == 0:
            continue

        these_row_indices = numpy.full(len(these_column_indices), i, dtype=int)

        if full_predictor_matrix is None:
            (this_downsized_predictor_matrix,
             target_matrix[:, these_row_indices, these_column_indices],
             full_predictor_matrix, full_target_matrix
            ) = testing_io.create_downsized_3d_examples(
                center_row_indices=these_row_indices,
                center_column_indices=these_column_indices,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                target_time_unix_sec=target_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                num_classes=num_classes)

        else:
            (this_downsized_predictor_matrix,
             target_matrix[:, these_row_indices, these_column_indices]
            ) = testing_io.create_downsized_3d_examples(
                center_row_indices=these_row_indices,
                center_column_indices=these_column_indices,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                full_predictor_matrix=full_predictor_matrix,
                full_target_matrix=full_target_matrix, num_classes=num_classes
            )[:2]

        class_probability_matrix[
            0, these_row_indices, these_column_indices, ...
        ] = model_object.predict(
            this_downsized_predictor_matrix, batch_size=len(these_row_indices))

    if isotonic_model_object_by_class is not None:
        these_row_indices, these_column_indices = numpy.where(
            narr_mask_matrix == 1)

        class_probability_matrix[
            0, these_row_indices, these_column_indices, ...
        ] = isotonic_regression.apply_model_for_each_class(
            orig_class_probability_matrix=class_probability_matrix[
                0, these_row_indices, these_column_indices, ...],
            observed_labels=target_matrix[
                0, these_row_indices, these_column_indices],
            model_object_by_class=isotonic_model_object_by_class)

    return class_probability_matrix, target_matrix


def apply_model_to_4d_example(
        model_object, target_time_unix_sec, num_lead_time_steps,
        predictor_time_step_offsets, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_rows_in_half_grid,
        num_columns_in_half_grid, num_classes,
        isotonic_model_object_by_class=None, narr_mask_matrix=None):
    """Applies trained CNN to a 4-D example.

    :param model_object: Trained instance of `keras.models.Sequential`.
    :param target_time_unix_sec: See doc for
        `testing_io.create_downsized_4d_examples`.
        :param num_lead_time_steps: Same.
    :param predictor_time_step_offsets: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_classes: Same.
    :param isotonic_model_object_by_class: See doc for
        `apply_model_to_3d_example`.
    :param narr_mask_matrix: Same.
    :return: class_probability_matrix: Same.
    :return: target_matrix: Same.
    """

    if narr_mask_matrix is None:
        narr_mask_matrix = numpy.full(
            (NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR), 1, dtype=int)

    ml_utils.check_narr_mask(narr_mask_matrix)

    class_probability_matrix = numpy.full(
        (1, NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR, num_classes), numpy.nan)
    target_matrix = numpy.full(
        (1, NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR), -1, dtype=int)

    full_predictor_matrix = None
    full_target_matrix = None

    for i in range(NUM_ROWS_IN_NARR):
        these_column_indices = numpy.where(narr_mask_matrix[i, :] == 1)[0]
        if len(these_column_indices) == 0:
            continue

        these_row_indices = numpy.full(len(these_column_indices), i, dtype=int)

        if full_predictor_matrix is None:
            (this_downsized_predictor_matrix,
             target_matrix[:, these_row_indices, these_column_indices],
             full_predictor_matrix, full_target_matrix
            ) = testing_io.create_downsized_4d_examples(
                center_row_indices=these_row_indices,
                center_column_indices=these_column_indices,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                target_time_unix_sec=target_time_unix_sec,
                predictor_time_step_offsets=predictor_time_step_offsets,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                num_classes=num_classes)

        else:
            (this_downsized_predictor_matrix,
             target_matrix[:, these_row_indices, these_column_indices]
            ) = testing_io.create_downsized_4d_examples(
                center_row_indices=these_row_indices,
                center_column_indices=these_column_indices,
                num_rows_in_half_grid=num_rows_in_half_grid,
                num_columns_in_half_grid=num_columns_in_half_grid,
                full_predictor_matrix=full_predictor_matrix,
                full_target_matrix=full_target_matrix, num_classes=num_classes
            )[:2]

        class_probability_matrix[
            0, these_row_indices, these_column_indices, ...
        ] = model_object.predict(
            this_downsized_predictor_matrix, batch_size=len(these_row_indices))

    if isotonic_model_object_by_class is not None:
        these_row_indices, these_column_indices = numpy.where(
            narr_mask_matrix == 1)

        class_probability_matrix[
            0, these_row_indices, these_column_indices, ...
        ] = isotonic_regression.apply_model_for_each_class(
            orig_class_probability_matrix=class_probability_matrix[
                0, these_row_indices, these_column_indices, ...],
            observed_labels=target_matrix[
                0, these_row_indices, these_column_indices],
            model_object_by_class=isotonic_model_object_by_class)

    return class_probability_matrix, target_matrix
