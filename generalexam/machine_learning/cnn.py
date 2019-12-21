"""Methods for training and applying a CNN (convolutional neural network).

CNNs are used for patch classification (where each grid cell [i, j] is
classified independently, based on a subset of the grid centered at [i, j]).  If
you want to do semantic segmentation instead (where all grid cells are
classified at the same time), see fcn.py, which implements fully convolutional
nets.
"""

import pickle
import os.path
import numpy
import keras.callbacks
from keras.models import load_model
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn as gg_cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import keras_metrics

PLATEAU_PATIENCE_EPOCHS = 3
PLATEAU_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 15
CROSS_ENTROPY_PATIENCE = 0.005

NUM_EPOCHS_KEY = 'num_epochs'
NUM_EX_PER_TRAIN_BATCH_KEY = 'num_ex_per_train_batch'
NUM_EX_PER_VALIDN_BATCH_KEY = 'num_ex_per_validn_batch'
NUM_EXAMPLES_PER_TIME_KEY = 'num_examples_per_target_time'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
DILATION_DISTANCE_KEY = 'dilation_distance_for_target_metres'
CLASS_FRACTIONS_KEY = 'class_fractions'
WEIGHT_LOSS_KEY = 'weight_loss_function'
PREDICTOR_NAMES_KEY = 'narr_predictor_names'
PRESSURE_LEVELS_KEY = 'pressure_levels_mb'
NUM_HALF_ROWS_KEY = 'num_rows_in_half_grid'
NUM_HALF_COLUMNS_KEY = 'num_columns_in_half_grid'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
FIRST_TRAINING_TIME_KEY = 'training_start_time_unix_sec'
LAST_TRAINING_TIME_KEY = 'training_end_time_unix_sec'
FIRST_VALIDATION_TIME_KEY = 'validation_start_time_unix_sec'
LAST_VALIDATION_TIME_KEY = 'validation_end_time_unix_sec'
MASK_MATRIX_KEY = 'narr_mask_matrix'
AUGMENTATION_DICT_KEY = 'augmentation_dict'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_EX_PER_TRAIN_BATCH_KEY, NUM_EX_PER_VALIDN_BATCH_KEY,
    NUM_EXAMPLES_PER_TIME_KEY, NUM_TRAINING_BATCHES_KEY,
    NUM_VALIDATION_BATCHES_KEY, DILATION_DISTANCE_KEY, CLASS_FRACTIONS_KEY,
    WEIGHT_LOSS_KEY, PREDICTOR_NAMES_KEY, PRESSURE_LEVELS_KEY,
    NUM_HALF_ROWS_KEY, NUM_HALF_COLUMNS_KEY, NORMALIZATION_FILE_KEY,
    NORMALIZATION_TYPE_KEY, FIRST_TRAINING_TIME_KEY, LAST_TRAINING_TIME_KEY,
    FIRST_VALIDATION_TIME_KEY, LAST_VALIDATION_TIME_KEY,
    MASK_MATRIX_KEY, AUGMENTATION_DICT_KEY
]

PERFORMANCE_METRIC_DICT = {
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


def model_to_num_classes(model_object):
    """Returns number of classes predicted by model.

    :param model_object: CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: num_classes: Number of classes.
    """

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    num_classes = num_output_neurons if num_output_neurons > 1 else 2
    return num_classes


def model_to_grid_dimensions(model_object):
    """Returns dimensions of predictor grid used by model.

    :param model_object: CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: num_half_rows: Number of half-rows in predictor grid (on either
        side of center).
    :return: num_half_columns: Same but for columns.
    """

    input_dimensions = model_object.input.get_shape().as_list()
    num_half_rows = int(numpy.round((input_dimensions[1] - 1) / 2))
    num_half_columns = int(numpy.round((input_dimensions[1] - 1) / 2))

    return num_half_rows, num_half_columns


def get_flattening_layer(model_object):
    """Finds flattening layer in CNN.

    If there are several flattening layers, this method returns the first
    (shallowest).

    :param model_object: CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: layer_name: Name of flattening layer.
    :raises: TypeError: if flattening layer cannot be found.
    """

    layer_names = [str(lyr.name) for lyr in model_object.layers]

    flattening_flags = numpy.array(
        ['flatten' in n for n in layer_names], dtype=bool
    )
    flattening_indices = numpy.where(flattening_flags)[0]

    if len(flattening_indices) == 0:
        error_string = (
            'Cannot find flattening layer in model.  Layer names are listed '
            'below.\n{0:s}'
        ).format(str(layer_names))

        raise TypeError(error_string)

    return layer_names[flattening_indices[0]]


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile written by `write_metadata`.

    :param model_file_name: Path to model itself (see doc for `read_model`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: metafile_name: Path to metafile.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    if not os.path.isfile(metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name)
        raise ValueError(error_string)

    return metafile_name


def write_metadata(
        pickle_file_name, num_epochs,
        num_ex_per_train_batch, num_ex_per_validn_batch, num_examples_per_time,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        predictor_names, pressure_levels_mb, num_half_rows, num_half_columns,
        normalization_file_name, normalization_type_string,
        dilation_distance_metres, class_fractions, weight_loss_function,
        first_training_time_unix_sec, last_training_time_unix_sec,
        first_validation_time_unix_sec, last_validation_time_unix_sec,
        mask_matrix=None, augmentation_dict=None):
    """Writes CNN metadata to Pickle file.

    In this context "validation" means on-the-fly validation (monitoring during
    training).

    C = number of predictors

    :param pickle_file_name: Path to output file.
    :param num_epochs: Number of training epochs.
    :param num_ex_per_train_batch: Number of examples per training batch.
    :param num_ex_per_validn_batch: Number of examples per validation batch.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param predictor_names: length-C list of predictor names (each must be
        accepted by `predictor_utils.check_field_name`).
    :param pressure_levels_mb: length-C numpy array of pressure levels
        (millibars).
    :param num_half_rows: Number of rows in half-grid (on either side of center)
        for predictors.
    :param num_half_columns: Same but for columns.
    :param normalization_file_name: Path to normalization file used for
        `machine_learning_utils.normalize_predictors_global`.
    :param normalization_type_string: Method used for
        `machine_learning_utils.normalize_predictors_nonglobal`.
    :param dilation_distance_metres: Dilation distance for gridded warm-front
        and cold-front labels.
    :param class_fractions: 1-D numpy array with sampling fraction used for each
        class.  Order must be (no front, warm front, cold front) or
        (no front, yes front).
    :param weight_loss_function: Boolean flag.  If True, loss function for each
        class was weighted by reciprocal of its frequency in training data.
    :param first_training_time_unix_sec: First time in training period.
    :param last_training_time_unix_sec: Last time in training period.
    :param first_validation_time_unix_sec: First time in validation period.
    :param last_validation_time_unix_sec: Last time in validation period.
    :param mask_matrix: J-by-K numpy array of zeros and ones, where J and K are
        numbers of rows and columns in NARR grid, respectively.  If
        mask_matrix[j, k] = 0, grid cell [j, k] could not be used as center of
        training or validation example.  If there was no mask, leave this as
        None.
    :param augmentation_dict: See doc for generators in
        training_validation_io.py.  If data augmentation was used, leave this as
        None.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_EX_PER_TRAIN_BATCH_KEY: num_ex_per_train_batch,
        NUM_EX_PER_VALIDN_BATCH_KEY: num_ex_per_validn_batch,
        NUM_EXAMPLES_PER_TIME_KEY: num_examples_per_time,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        PREDICTOR_NAMES_KEY: predictor_names,
        PRESSURE_LEVELS_KEY: pressure_levels_mb,
        NUM_HALF_ROWS_KEY: num_half_rows,
        NUM_HALF_COLUMNS_KEY: num_half_columns,
        NORMALIZATION_FILE_KEY: normalization_file_name,
        NORMALIZATION_TYPE_KEY: normalization_type_string,
        DILATION_DISTANCE_KEY: dilation_distance_metres,
        CLASS_FRACTIONS_KEY: class_fractions,
        WEIGHT_LOSS_KEY: weight_loss_function,
        FIRST_TRAINING_TIME_KEY: first_training_time_unix_sec,
        LAST_TRAINING_TIME_KEY: last_training_time_unix_sec,
        FIRST_VALIDATION_TIME_KEY: first_validation_time_unix_sec,
        LAST_VALIDATION_TIME_KEY: last_validation_time_unix_sec,
        MASK_MATRIX_KEY: mask_matrix,
        AUGMENTATION_DICT_KEY: augmentation_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_metadata(pickle_file_name):
    """Reads CNN metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with keys listed in `write_metadata`.
    :raises: ValueError: if any of these keys are missing from the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle, encoding='latin1')
    pickle_file_handle.close()

    if NUM_EX_PER_TRAIN_BATCH_KEY not in metadata_dict:
        metadata_dict[NUM_EX_PER_TRAIN_BATCH_KEY] = metadata_dict[
            'num_examples_per_batch'
        ]

    if NUM_EX_PER_VALIDN_BATCH_KEY not in metadata_dict:
        metadata_dict[NUM_EX_PER_VALIDN_BATCH_KEY] = metadata_dict[
            'num_examples_per_batch'
        ]

    if AUGMENTATION_DICT_KEY not in metadata_dict:
        metadata_dict[AUGMENTATION_DICT_KEY] = None

    if NORMALIZATION_FILE_KEY not in metadata_dict:
        metadata_dict[NORMALIZATION_FILE_KEY] = None

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def read_model(hdf5_file_name):
    """Reads CNN from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    """

    error_checking.assert_file_exists(hdf5_file_name)
    return load_model(hdf5_file_name, custom_objects=PERFORMANCE_METRIC_DICT)


def train_cnn(
        model_object, output_model_file_name, num_epochs,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        training_generator, validation_generator, weight_loss_function=False,
        class_fractions=None):
    """Trains new CNN.

    In this context "validation" means on-the-fly validation (monitoring during
    training).

    :param model_object: Untrained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param output_model_file_name: Path to output file (will be in HDF5 format,
        so extension should be ".h5").
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param training_generator: Training generator (see
        `training_validation_io.downsized_generator_from_scratch` and
        `training_validation_io.downsized_generator_from_example_files` for
        examples of a compliant generator).
    :param validation_generator: Validation generator (same note).
    :param weight_loss_function: Boolean flag.  If True, loss function for each
        class was weighted by reciprocal of its frequency in training data.
    :param class_fractions: [used only if `weight_loss_function == True`]
        1-D numpy array with sampling fraction used by training generator for
        each class.  Order must be (no front, warm front, cold front) or
        (no front, yes front).
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_boolean(weight_loss_function)

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_model_file_name)

    if weight_loss_function:
        class_weight_dict = ml_utils.get_class_weight_dict(class_fractions)
    else:
        class_weight_dict = None

    history_file_name = '{0:s}/history.csv'.format(
        os.path.split(output_model_file_name)[0]
    )
    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=False
    )

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        output_model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=CROSS_ENTROPY_PATIENCE,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min')

    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=PLATEAU_LEARNING_RATE_MULTIPLIER,
        patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
        min_delta=CROSS_ENTROPY_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS)

    list_of_callback_objects = [
        history_object, checkpoint_object, early_stopping_object, plateau_object
    ]

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, class_weight=class_weight_dict,
        callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)


def apply_model(model_object, predictor_matrix, num_examples_per_batch=1000,
                verbose=False, return_features=False, feature_layer_name=None):
    """Applies trained model to new examples.

    E = number of examples
    M = number of rows in example grid
    N = number of columns in example grid
    C = number of predictors (channels)

    K = number of classes
    Z = number of features

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param num_examples_per_batch: Number of examples per batch.
    :param verbose: Boolean flag.  If True, will print progress message after
        each batch.
    :param return_features: Boolean flag.  If True, will return feature values
        (outputs of intermediate layer) instead of predictions.
    :param feature_layer_name: [used only if `return_features == True`]
        Name of intermediate layer.

    If return_features = True...

    :return: feature_matrix: E-by-Z numpy array of feature values.

    If return_features = False...

    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_examples = predictor_matrix.shape[0]

    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0
    else:
        error_checking.assert_is_integer(num_examples_per_batch)
        error_checking.assert_is_greater(num_examples_per_batch, 0)

    num_examples_per_batch = min([num_examples_per_batch, num_examples])
    error_checking.assert_is_boolean(verbose)
    error_checking.assert_is_boolean(return_features)

    if return_features:
        model_object_to_use = gg_cnn.model_to_feature_generator(
            model_object=model_object, feature_layer_name=feature_layer_name)
    else:
        model_object_to_use = model_object

    output_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print((
                'Applying model to examples {0:d}-{1:d} of {2:d}...'
            ).format(
                this_first_index + 1, this_last_index + 1, num_examples
            ))

        these_outputs = model_object_to_use.predict(
            predictor_matrix[this_first_index:(this_last_index + 1), ...],
            batch_size=this_last_index - this_first_index + 1
        )

        if output_matrix is None:
            output_matrix = these_outputs + 0.
        else:
            output_matrix = numpy.concatenate(
                (output_matrix, these_outputs), axis=0
            )

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    if return_features:
        return output_matrix

    if len(output_matrix.shape) == 1:
        output_matrix = dl_utils.event_probs_to_multiclass(output_matrix)

    return output_matrix


def apply_model_to_full_grid(
        model_object, top_predictor_dir_name, valid_time_unix_sec,
        pressure_levels_mb, predictor_names, normalization_file_name=None,
        normalization_type_string=None, top_gridded_front_dir_name=None,
        dilation_distance_metres=None, isotonic_model_object_by_class=None,
        mask_matrix=None):
    """Applies CNN independently to each grid cell in a full grid.

    M = number of rows in full grid
    N = number of columns in full grid
    C = number of predictors
    K = number of classes

    If `top_gridded_front_dir_name is None`, this method will return None for
    `target_matrix`.

    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param top_predictor_dir_name: Name of top-level directory with predictors.
        Files therein will be found by `predictor_io.find_file` and read by
        `predictor_io.read_file`.
    :param valid_time_unix_sec: Valid time.
    :param pressure_levels_mb: length-C numpy array of pressure levels
        (millibars).
    :param predictor_names: length-C list of predictor names (each must be
        accepted by `predictor_utils.check_field_name`).
    :param normalization_file_name: Path to normalization file used for
        `machine_learning_utils.normalize_predictors_global`.
    :param normalization_type_string:
        [used only if `normalization_file_name is None`]
        Method used for `machine_learning_utils.normalize_predictors_nonglobal`.
    :param top_gridded_front_dir_name: Name of top-level directory with gridded
        front labels.  Files therein will be found by
        `fronts_io.find_gridded_file` and read by
        `fronts_io.read_grid_from_file`.
    :param dilation_distance_metres: Dilation distance for gridded warm-front
        and cold-front labels.
    :param isotonic_model_object_by_class: length-K list of trained isotonic-
        regression models (instances of `sklearn.isotonic.IsotonicRegression`).
        These will be used to calibrate raw CNN probabilities.  If
        `isotonic_model_object_by_class is None`, there will be no calibration.
    :param mask_matrix: See doc for `write_metadata`.
    :return: class_probability_matrix: 1-by-M-by-N-by-K numpy array of predicted
        probabilities.  If grid cell [i, j] is masked out,
        class_probability_matrix[0, i, j, :] = NaN.
    :return: target_matrix: 1-by-M-by-N numpy array of true labels.  If grid
        cell [i, j] is masked out, target_matrix[0, i, j] = -1.
    """

    return_targets = top_gridded_front_dir_name is not None

    predictor_file_name = predictor_io.find_file(
        top_directory_name=top_predictor_dir_name,
        valid_time_unix_sec=valid_time_unix_sec)

    predictor_dict = predictor_io.read_file(
        netcdf_file_name=predictor_file_name, metadata_only=False)

    num_full_grid_rows = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY].shape[1]
    num_full_grid_columns = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY].shape[2]

    if mask_matrix is None:
        mask_matrix = numpy.full(
            (num_full_grid_rows, num_full_grid_columns), 1, dtype=int
        )

    error_checking.assert_is_integer_numpy_array(mask_matrix)
    error_checking.assert_is_geq_numpy_array(mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(mask_matrix, 1)

    these_expected_dim = numpy.array(
        [num_full_grid_rows, num_full_grid_columns], dtype=int
    )
    error_checking.assert_is_numpy_array(
        mask_matrix, exact_dimensions=these_expected_dim)

    num_classes = model_to_num_classes(model_object)
    num_half_patch_rows, num_half_patch_columns = model_to_grid_dimensions(
        model_object)

    full_size_predictor_matrix = None
    class_probability_matrix = numpy.full(
        (1, num_full_grid_rows, num_full_grid_columns, num_classes), numpy.nan
    )

    if return_targets:
        full_size_target_matrix = None
        target_matrix = numpy.full(
            (1, num_full_grid_rows, num_full_grid_columns), -1, dtype=int
        )
    else:
        target_matrix = None

    for i in range(num_full_grid_rows):
        these_column_indices = numpy.where(mask_matrix[i, :] == 1)[0]
        if len(these_column_indices) == 0:
            continue

        these_row_indices = numpy.full(len(these_column_indices), i, dtype=int)

        if full_size_predictor_matrix is None:
            if return_targets:
                this_dict = testing_io.create_downsized_examples_with_targets(
                    center_row_indices=these_row_indices,
                    center_column_indices=these_column_indices,
                    num_half_rows=num_half_patch_rows,
                    num_half_columns=num_half_patch_columns,
                    top_predictor_dir_name=top_predictor_dir_name,
                    top_gridded_front_dir_name=top_gridded_front_dir_name,
                    valid_time_unix_sec=valid_time_unix_sec,
                    pressure_levels_mb=pressure_levels_mb,
                    predictor_names=predictor_names,
                    normalization_file_name=normalization_file_name,
                    normalization_type_string=normalization_type_string,
                    dilation_distance_metres=dilation_distance_metres,
                    num_classes=num_classes)
            else:
                this_dict = testing_io.create_downsized_examples_no_targets(
                    center_row_indices=these_row_indices,
                    center_column_indices=these_column_indices,
                    num_half_rows=num_half_patch_rows,
                    num_half_columns=num_half_patch_columns,
                    top_predictor_dir_name=top_predictor_dir_name,
                    valid_time_unix_sec=valid_time_unix_sec,
                    pressure_levels_mb=pressure_levels_mb,
                    predictor_names=predictor_names,
                    normalization_file_name=normalization_file_name,
                    normalization_type_string=normalization_type_string)

            full_size_predictor_matrix = this_dict[
                testing_io.FULL_PREDICTOR_MATRIX_KEY]

            if return_targets:
                full_size_target_matrix = this_dict[
                    testing_io.FULL_TARGET_MATRIX_KEY]

        else:
            if return_targets:
                this_dict = testing_io.create_downsized_examples_with_targets(
                    center_row_indices=these_row_indices,
                    center_column_indices=these_column_indices,
                    num_half_rows=num_half_patch_rows,
                    num_half_columns=num_half_patch_columns,
                    full_size_predictor_matrix=full_size_predictor_matrix,
                    full_size_target_matrix=full_size_target_matrix)
            else:
                this_dict = testing_io.create_downsized_examples_no_targets(
                    center_row_indices=these_row_indices,
                    center_column_indices=these_column_indices,
                    num_half_rows=num_half_patch_rows,
                    num_half_columns=num_half_patch_columns,
                    full_size_predictor_matrix=full_size_predictor_matrix)

        this_predictor_matrix = this_dict[testing_io.PREDICTOR_MATRIX_KEY]

        if return_targets:
            target_matrix[:, these_row_indices, these_column_indices] = (
                this_dict[testing_io.TARGET_VALUES_KEY]
            )

        this_prob_matrix = model_object.predict(
            this_predictor_matrix, batch_size=len(these_row_indices)
        )
        print(numpy.percentile(this_prob_matrix[..., 1]))
        print(numpy.percentile(this_prob_matrix[..., 2]))

        class_probability_matrix[
            0, these_row_indices, these_column_indices, ...
        ] = this_prob_matrix

    if isotonic_model_object_by_class is None:
        return class_probability_matrix, target_matrix

    these_row_indices, these_column_indices = numpy.where(mask_matrix == 1)

    class_probability_matrix[
        0, these_row_indices, these_column_indices, ...
    ] = isotonic_regression.apply_model_for_each_class(
        orig_class_probability_matrix=class_probability_matrix[
            0, these_row_indices, these_column_indices, ...],
        observed_labels=target_matrix[
            0, these_row_indices, these_column_indices],
        model_object_by_class=isotonic_model_object_by_class)

    return class_probability_matrix, target_matrix
