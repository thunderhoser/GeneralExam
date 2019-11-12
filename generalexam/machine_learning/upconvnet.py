"""Methods for training and applying upconvolutional neural nets."""

import os.path
import pickle
import numpy
import keras
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn as gg_cnn
from generalexam.machine_learning import cnn as ge_cnn
from generalexam.machine_learning import training_validation_io as trainval_io

TIME_FORMAT = '%Y%m%d%H'
PATHLESS_FILE_NAME_PREFIX = 'upconvnet_predictions'

PLATEAU_PATIENCE_EPOCHS = 3
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 15
MSE_PATIENCE = 0.005

CNN_FILE_KEY = 'cnn_model_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'
FIRST_TRAINING_TIME_KEY = 'first_training_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'last_training_time_unix_sec'
NUM_EX_PER_TRAIN_BATCH_KEY = 'num_ex_per_train_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
FIRST_VALIDATION_TIME_KEY = 'first_validation_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'last_validation_time_unix_sec'
NUM_EX_PER_VALIDN_BATCH_KEY = 'num_ex_per_validn_batch'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
NUM_EPOCHS_KEY = 'num_epochs'

MODEL_METADATA_KEYS = [
    CNN_FILE_KEY, CNN_FEATURE_LAYER_KEY,
    FIRST_TRAINING_TIME_KEY, LAST_TRAINING_TIME_KEY,
    NUM_EX_PER_TRAIN_BATCH_KEY, NUM_TRAINING_BATCHES_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY,
    NUM_EX_PER_VALIDN_BATCH_KEY, NUM_VALIDATION_BATCHES_KEY, NUM_EPOCHS_KEY
]

RECON_IMAGE_MATRIX_KEY = 'reconstructed_image_matrix'
MEAN_SQUARED_ERRORS_KEY = 'mse_by_example'
EXAMPLE_IDS_KEY = 'example_id_strings'
UPCONVNET_FILE_KEY = 'upconvnet_file_name'

EXAMPLE_DIMENSION_KEY = 'example'
ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
CHANNEL_DIMENSION_KEY = 'channel'
ID_CHAR_DIMENSION_KEY = 'example_id_char'


def _generator_from_example_files(
        cnn_model_object, cnn_metadata_dict, cnn_feature_layer_name,
        top_example_dir_name, first_time_unix_sec, last_time_unix_sec,
        num_examples_per_batch):
    """Generates training or validation examples for upconvnet.

    E = number of examples per batch
    Z = number of features in vector

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).  Will be used to convert images to feature
        vectors, while the upconvnet will attempt to convert feature vectors
        back to original images.
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    :param cnn_feature_layer_name: Name of feature-generating layer in CNN.
        Feature vectors (inputs to upconvnet) will be outputs from this layer.
    :param top_example_dir_name: Name of top-level directory with example files.
        Shuffled files therein will be found by `learning_examples_io.find_file`
        and read by `learning_examples_io.read_file`.
    :param first_time_unix_sec: First time in period.  Will generate only
        examples from `first_time_unix_sec`...`last_time_unix_sec`.
    :param last_time_unix_sec: See above.
    :param num_examples_per_batch: Number of examples per batch.
    :return: feature_matrix: E-by-Z numpy array of scalar features.  Each row is
        one feature vector.
    :return: image_matrix: numpy array of images.  The first axis has length E.
        These are CNN inputs and upconvnet targets.
    """

    num_half_rows, num_half_columns = ge_cnn.model_to_grid_dimensions(
        cnn_model_object)
    num_classes = ge_cnn.model_to_num_classes(cnn_model_object)

    partial_cnn_model_object = gg_cnn.model_to_feature_generator(
        model_object=cnn_model_object,
        feature_layer_name=cnn_feature_layer_name)

    cnn_generator = trainval_io.downsized_generator_from_example_files(
        top_input_dir_name=top_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        predictor_names=cnn_metadata_dict[ge_cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_mb=cnn_metadata_dict[ge_cnn.PRESSURE_LEVELS_KEY],
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_classes=num_classes, num_examples_per_batch=num_examples_per_batch,
        augmentation_dict=None)

    while True:
        try:
            # TODO(thunderhoser): For old upconvnets, target was departure from
            # mean in each image.  Might want to do this again.
            image_matrix = next(cnn_generator)[0]
        except StopIteration:
            break

        feature_matrix = partial_cnn_model_object.predict(
            image_matrix, batch_size=image_matrix.shape[0]
        )

        yield (feature_matrix, image_matrix)


def train_upconvnet(
        ucn_model_object, cnn_model_object, cnn_metadata_dict,
        cnn_feature_layer_name, top_training_dir_name,
        first_training_time_unix_sec, last_training_time_unix_sec,
        num_ex_per_train_batch, num_training_batches_per_epoch,
        top_validation_dir_name, first_validation_time_unix_sec,
        last_validation_time_unix_sec, num_ex_per_validn_batch,
        num_validation_batches_per_epoch, num_epochs, output_dir_name):
    """Trains upconvnet.

    :param ucn_model_object: Untrained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param cnn_model_object: See doc for `_generator_from_example_files`.
    :param cnn_metadata_dict: Same.
    :param cnn_feature_layer_name: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_unix_sec: Same.
    :param last_training_time_unix_sec: Same.
    :param num_ex_per_train_batch: Same.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param top_validation_dir_name: See doc for `_generator_from_example_files`.
    :param first_validation_time_unix_sec: Same.
    :param last_validation_time_unix_sec: Same.
    :param num_ex_per_validn_batch: Same.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param num_epochs: Number of epochs.
    :param output_dir_name: Name of output directory.  Trained upconvnet and
        history file will be saved here.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    upconvnet_file_name = '{0:s}/upconvnet_model.h5'.format(output_dir_name)
    history_file_name = '{0:s}/upconvnet_model_history.csv'.format(
        output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False)

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=upconvnet_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MSE_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min')

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=MSE_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS)

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    training_generator = _generator_from_example_files(
        cnn_model_object=cnn_model_object, cnn_metadata_dict=cnn_metadata_dict,
        cnn_feature_layer_name=cnn_feature_layer_name,
        top_example_dir_name=top_training_dir_name,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec,
        num_examples_per_batch=num_ex_per_train_batch)

    validation_generator = _generator_from_example_files(
        cnn_model_object=cnn_model_object, cnn_metadata_dict=cnn_metadata_dict,
        cnn_feature_layer_name=cnn_feature_layer_name,
        top_example_dir_name=top_validation_dir_name,
        first_time_unix_sec=first_validation_time_unix_sec,
        last_time_unix_sec=last_validation_time_unix_sec,
        num_examples_per_batch=num_ex_per_validn_batch)

    ucn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)


def apply_upconvnet(
        image_matrix, cnn_model_object, cnn_feature_layer_name,
        ucn_model_object, num_examples_per_batch=1000, verbose=True):
    """Applies upconvnet to new images.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors (channels)

    :param image_matrix: E-by-M-by-N-by-C numpy array of original images (inputs
        to CNN).
    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).  Will be used to convert images to feature
        vectors.
    :param cnn_feature_layer_name: Name of feature-generating layer in CNN.
        Feature vectors (inputs to upconvnet) will be outputs from this layer.
    :param ucn_model_object: Trained upconvnet (instance of `keras.models.Model`
        or `keras.models.Sequential`).  Will be used to convert feature vectors
        back to images (ideally back to original images).
    :param num_examples_per_batch: Number of examples per batch.
    :param verbose: Boolean flag.  If True, will print progress messages to
        command window.
    :return: reconstructed_image_matrix: E-by-M-by-N-by-C numpy array of
        reconstructed images.
    """

    partial_cnn_model_object = gg_cnn.model_to_feature_generator(
        model_object=cnn_model_object,
        feature_layer_name=cnn_feature_layer_name)

    error_checking.assert_is_boolean(verbose)
    error_checking.assert_is_numpy_array_without_nan(image_matrix)
    error_checking.assert_is_numpy_array(image_matrix, num_dimensions=4)

    num_examples = image_matrix.shape[0]
    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_greater(num_examples_per_batch, 0)
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    reconstructed_image_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        j = i
        k = min([i + num_examples_per_batch - 1, num_examples - 1])
        these_example_indices = numpy.linspace(j, k, num=k - j + 1, dtype=int)

        if verbose:
            print((
                'Applying upconvnet to examples {0:d}-{1:d} of {2:d}...'
            ).format(
                numpy.min(these_example_indices) + 1,
                numpy.max(these_example_indices) + 1,
                num_examples
            ))

        this_feature_matrix = partial_cnn_model_object.predict(
            image_matrix[these_example_indices, ...],
            batch_size=len(these_example_indices)
        )

        this_reconstructed_matrix = ucn_model_object.predict(
            this_feature_matrix, batch_size=len(these_example_indices)
        )

        if reconstructed_image_matrix is None:
            dimensions = numpy.array(
                (num_examples,) + this_reconstructed_matrix.shape[1:], dtype=int
            )
            reconstructed_image_matrix = numpy.full(dimensions, numpy.nan)

        reconstructed_image_matrix[
            these_example_indices, ...] = this_reconstructed_matrix

    print('Have applied upconvnet to all {0:d} examples!'.format(num_examples))
    return reconstructed_image_matrix


def write_model_metadata(
        pickle_file_name, cnn_model_file_name, cnn_feature_layer_name,
        first_training_time_unix_sec, last_training_time_unix_sec,
        num_ex_per_train_batch, num_training_batches_per_epoch,
        first_validation_time_unix_sec, last_validation_time_unix_sec,
        num_ex_per_validn_batch, num_validation_batches_per_epoch, num_epochs):
    """Writes metadata for upconvnet to Pickle file.

    :param pickle_file_name: Path to output file.
    :param cnn_model_file_name: Path to trained CNN, used to turn images into
        feature vectors.  Must be readable by `cnn.read_model`.
    :param cnn_feature_layer_name: See doc for `train_upconvnet`.
    :param first_training_time_unix_sec: Same.
    :param last_training_time_unix_sec: Same.
    :param num_ex_per_train_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param first_validation_time_unix_sec: Same.
    :param last_validation_time_unix_sec: Same.
    :param num_ex_per_validn_batch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param num_epochs: Same.
    """

    model_metadata_dict = {
        CNN_FILE_KEY: cnn_model_file_name,
        CNN_FEATURE_LAYER_KEY: cnn_feature_layer_name,
        FIRST_TRAINING_TIME_KEY: first_training_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_training_time_unix_sec,
        NUM_EX_PER_TRAIN_BATCH_KEY: num_ex_per_train_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        FIRST_VALIDATION_TIME_KEY: first_validation_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validation_time_unix_sec,
        NUM_EX_PER_VALIDN_BATCH_KEY: num_ex_per_validn_batch,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        NUM_EPOCHS_KEY: num_epochs
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_model_metadata(pickle_file_name):
    """Reads metadata for upconvnet from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_metadata_dict: Dictionary with the following keys.
    model_metadata_dict["cnn_model_file_name"]: See doc for
        `write_model_metadata`.
    model_metadata_dict["cnn_feature_layer_name"]: Same.
    model_metadata_dict["first_training_time_unix_sec"]: Same.
    model_metadata_dict["last_training_time_unix_sec"]: Same.
    model_metadata_dict["num_ex_per_train_batch"]: Same.
    model_metadata_dict["num_training_batches_per_epoch"]: Same.
    model_metadata_dict["first_validation_time_unix_sec"]: Same.
    model_metadata_dict["last_validation_time_unix_sec"]: Same.
    model_metadata_dict["num_ex_per_validn_batch"]: Same.
    model_metadata_dict["num_validation_batches_per_epoch"]: Same.
    model_metadata_dict["num_epochs"]: Same.

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


def find_prediction_file(top_directory_name, valid_time_unix_sec,
                         raise_error_if_missing=False):
    """Finds file with upconvnet predictions (reconstructed images).

    :param top_directory_name: Name of top-level directory with upconvnet
        predictions.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    year_string = time_conversion.unix_sec_to_string(valid_time_unix_sec, '%Y')

    prediction_file_name = (
        '{0:s}/{1:s}/{2:s}_{3:s}.nc'
    ).format(
        top_directory_name, year_string, PATHLESS_FILE_NAME_PREFIX,
        time_conversion.unix_sec_to_string(valid_time_unix_sec, TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_predictions(
        netcdf_file_name, reconstructed_image_matrix, example_id_strings,
        mse_by_example, upconvnet_file_name):
    """Writes predictions (reconstructed images) to NetCDF file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors (channels)

    :param netcdf_file_name: Path to output file.
    :param reconstructed_image_matrix: E-by-M-by-N-by-C numpy array of
        reconstructed images.
    :param example_id_strings: length-E list of example IDs.
    :param mse_by_example: length-E numpy array of mean squared errors (in
        normalized, not physical, units).
    :param upconvnet_file_name: Path to upconvnet that generated the
        reconstructed images (readable by `cnn.read_model`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(reconstructed_image_matrix)
    error_checking.assert_is_numpy_array(reconstructed_image_matrix,
                                         num_dimensions=4)

    num_examples = reconstructed_image_matrix.shape[0]
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_geq_numpy_array(mse_by_example, 0.)
    error_checking.assert_is_numpy_array(
        mse_by_example, exact_dimensions=these_expected_dim)

    error_checking.assert_is_string(netcdf_file_name)
    error_checking.assert_is_string(upconvnet_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(UPCONVNET_FILE_KEY, upconvnet_file_name)

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        ROW_DIMENSION_KEY, reconstructed_image_matrix.shape[1]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, reconstructed_image_matrix.shape[2]
    )
    dataset_object.createDimension(
        CHANNEL_DIMENSION_KEY, reconstructed_image_matrix.shape[3]
    )

    num_id_characters = numpy.max(numpy.array([
        len(id) for id in example_id_strings
    ]))

    dataset_object.createDimension(ID_CHAR_DIMENSION_KEY, num_id_characters)

    # Add reconstructed images.
    dataset_object.createVariable(
        RECON_IMAGE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY,
                    COLUMN_DIMENSION_KEY, CHANNEL_DIMENSION_KEY)
    )
    dataset_object.variables[RECON_IMAGE_MATRIX_KEY][:] = (
        reconstructed_image_matrix
    )

    # Add mean squared errors.
    dataset_object.createVariable(
        MEAN_SQUARED_ERRORS_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[MEAN_SQUARED_ERRORS_KEY][:] = mse_by_example

    # Add example IDs.
    this_string_format = 'S{0:d}'.format(num_id_characters)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, ID_CHAR_DIMENSION_KEY)
    )
    dataset_object.variables[EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array)

    dataset_object.close()


def read_predictions(netcdf_file_name):
    """Reads predictions (reconstructed images) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict["reconstructed_image_matrix"]: See doc for
        `write_predictions`.
    prediction_dict["example_id_strings"]: Same.
    prediction_dict["mse_by_example"]: Same.
    prediction_dict["upconvnet_file_name"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        RECON_IMAGE_MATRIX_KEY: numpy.array(
            dataset_object.variables[RECON_IMAGE_MATRIX_KEY][:]
        ),
        EXAMPLE_IDS_KEY: [
            str(s) for s in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MEAN_SQUARED_ERRORS_KEY: numpy.array(
            dataset_object.variables[MEAN_SQUARED_ERRORS_KEY][:]
        ),
        UPCONVNET_FILE_KEY: str(getattr(dataset_object, UPCONVNET_FILE_KEY))
    }

    dataset_object.close()
    return prediction_dict
