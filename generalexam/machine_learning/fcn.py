"""Training and deployment methods for an FCN (fully convolutional net).

An FCN has one target variable at each pixel, whereas a traditional CNN has only
one target variable.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of full-size examples
K = number of classes (possible target values).  See below for the definition of
    "target value".
M = number of spatial rows per example
N = number of spatial columns per example
T = number of predictor times per example (number of images per sequence)
C = number of channels (predictor variables) per example

--- DEFINITIONS ---

A "full-size" example covers the entire NARR grid, while a downsized example
covers only a subset of the NARR grid.

The dimensions of a 3-D example are M x N x C (only one predictor time).

The dimensions of a 4-D example are M x N x T x C.

NF = no front
WF = warm front
CF = cold front

Target variable = label at one pixel.  For a full-size example,
there are M*N target variables (the label at each pixel).

--- REFERENCES ---

Ronneberger, O., P. Fischer, and T. Brox (2015): "U-net: Convolutional networks
    for biomedical image segmentation". International Conference on Medical
    Image Computing and Computer-assisted Intervention, 234-241.
"""

import numpy
import keras.models
import keras.layers
from keras.callbacks import ModelCheckpoint
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import keras_metrics
from generalexam.machine_learning import keras_losses

CUSTOM_OBJECT_DICT_FOR_READING_MODEL = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

LEARNING_RATE_FOR_U_NET = 1e-4
LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_success_ratio, keras_metrics.binary_focn
]


def _check_unet_input_args(
        num_predictors, weight_loss_function, convolve_over_time,
        assumed_class_frequencies=None, num_classes=None,
        num_predictor_time_steps=None):
    """Checks input arguments for U-net.

    K = number of classes

    :param num_predictors: See documentation for
        `get_unet_with_2d_convolution`.
    :param weight_loss_function: Same.
    :param convolve_over_time: Same.
    :param assumed_class_frequencies: [used only if weight_loss_function = True]
        Same.
    :param num_classes: [used only if weight_loss_function = False]
        Same.
    :param num_predictor_time_steps: [used only if convolve_over_time = True]
        Same.
    :return: class_weights: length-K numpy array of class weights for loss
        function.
    """

    error_checking.assert_is_integer(num_predictors)
    error_checking.assert_is_geq(num_predictors, 1)
    error_checking.assert_is_boolean(weight_loss_function)
    error_checking.assert_is_boolean(convolve_over_time)

    if weight_loss_function:
        class_weight_dict = ml_utils.get_class_weight_dict(
            assumed_class_frequencies)
        class_weights = numpy.array(class_weight_dict.values())
        num_classes = len(class_weights)

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    if not weight_loss_function:
        class_weights = numpy.array(num_classes, 1. / num_classes)

    if convolve_over_time:
        error_checking.assert_is_integer(num_predictor_time_steps)
        error_checking.assert_is_geq(num_predictor_time_steps, 6)

    return class_weights


def read_keras_model(hdf5_file_name, assumed_class_frequencies):
    """Reads Keras model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :param assumed_class_frequencies: See documentation for
        `get_unet_with_2d_convolution`.
    :return: keras_model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    class_weight_dict = ml_utils.get_class_weight_dict(
        assumed_class_frequencies)
    class_weights = numpy.array(class_weight_dict.values())
    class_weights = numpy.reshape(class_weights, (class_weights.size, 1))

    CUSTOM_OBJECT_DICT_FOR_READING_MODEL.update(
        {'loss': keras_losses.weighted_cross_entropy(class_weights)})
    return keras.models.load_model(
        hdf5_file_name, custom_objects=CUSTOM_OBJECT_DICT_FOR_READING_MODEL)


def get_unet_with_2d_convolution(
        weight_loss_function, num_predictors=3, assumed_class_frequencies=None,
        num_classes=None):
    """Creates U-net with architecture used in the following example.

    https://github.com/zhixuhao/unet/blob/master/unet.py

    For more on U-nets in general, see Ronneberger et al. (2015).

    :param weight_loss_function: Boolean flag.  If True, the loss function will
        weight each class by the inverse of its assumed frequency (see
        `assumed_class_frequencies`).
    :param num_predictors: Number of predictor variables (image channels).
    :param assumed_class_frequencies: [used only if weight_loss_function = True]
        1-D numpy array, where the [k]th element is the estimated frequency of
        the [k]th class.
    :param num_classes: [used only if weight_loss_function = False]
        Number of classes.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    class_weights = _check_unet_input_args(
        num_predictors=num_predictors,
        weight_loss_function=weight_loss_function, convolve_over_time=False,
        assumed_class_frequencies=assumed_class_frequencies,
        num_classes=num_classes)
    num_classes = len(class_weights)

    num_grid_rows = (
        ml_utils.LAST_NARR_ROW_FOR_FCN_INPUT -
        ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT + 1
    )
    num_grid_columns = (
        ml_utils.LAST_NARR_COLUMN_FOR_FCN_INPUT -
        ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT + 1
    )

    input_dimensions = (num_grid_rows, num_grid_columns, num_predictors)
    input_layer_object = keras.layers.Input(shape=input_dimensions)

    conv_layer1_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(input_layer_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer1_object.shape)

    conv_layer1_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer1_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer1_object.shape)

    pooling_layer1_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer1_object)
    print 'Shape of pooling layer: {0:s}'.format(pooling_layer1_object.shape)

    conv_layer2_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer1_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer2_object.shape)

    conv_layer2_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer2_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer2_object.shape)

    pooling_layer2_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer2_object)
    print 'Shape of pooling layer: {0:s}'.format(pooling_layer2_object.shape)

    conv_layer3_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer2_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer3_object.shape)

    conv_layer3_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer3_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer3_object.shape)

    pooling_layer3_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer3_object)
    print 'Shape of pooling layer: {0:s}'.format(pooling_layer3_object.shape)

    conv_layer4_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer3_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer4_object.shape)

    conv_layer4_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer4_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer4_object.shape)

    dropout_layer4_object = keras.layers.Dropout(rate=0.5)(conv_layer4_object)
    pooling_layer4_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(dropout_layer4_object)
    print 'Shape of pooling layer: {0:s}'.format(pooling_layer4_object.shape)

    conv_layer5_object = keras.layers.Conv2D(
        filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer4_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer5_object.shape)

    conv_layer5_object = keras.layers.Conv2D(
        filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer5_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer5_object.shape)

    dropout_layer5_object = keras.layers.Dropout(rate=0.5)(conv_layer5_object)

    upsampling_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(dropout_layer5_object))
    print 'Shape of upsampling layer: {0:s}'.format(
        upsampling_layer6_object.shape)

    merged_layer6_object = keras.layers.merge(
        [dropout_layer4_object, upsampling_layer6_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer: {0:s}'.format(merged_layer6_object.shape)

    conv_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer6_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer6_object.shape)

    conv_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer6_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer6_object.shape)

    upsampling_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer6_object))
    print 'Shape of upsampling layer: {0:s}'.format(
        upsampling_layer7_object.shape)

    merged_layer7_object = keras.layers.merge(
        [conv_layer3_object, upsampling_layer7_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer: {0:s}'.format(merged_layer7_object.shape)

    conv_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer7_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer7_object.shape)

    conv_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer7_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer7_object.shape)

    upsampling_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer7_object))
    print 'Shape of upsampling layer: {0:s}'.format(
        upsampling_layer8_object.shape)

    merged_layer8_object = keras.layers.merge(
        [conv_layer2_object, upsampling_layer8_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer: {0:s}'.format(merged_layer8_object.shape)

    conv_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer8_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer8_object.shape)

    conv_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer8_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer8_object.shape)

    upsampling_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer8_object))
    print 'Shape of upsampling layer: {0:s}'.format(
        upsampling_layer9_object.shape)

    merged_layer9_object = keras.layers.merge(
        [conv_layer1_object, upsampling_layer9_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer: {0:s}'.format(merged_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer9_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer9_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=2*num_classes, kernel_size=(3, 3), activation='relu',
        padding='same', kernel_initializer='he_normal')(conv_layer9_object)
    print 'Shape of convolutional layer: {0:s}'.format(conv_layer9_object.shape)

    # conv_layer10_object = keras.layers.Conv2D(
    #     filters=1, kernel_size=(1, 1), activation='sigmoid')(
    #         conv_layer9_object)

    conv_layer10_object = keras.layers.Conv2D(
        filters=num_classes, kernel_size=(1, 1), activation='softmax')(
            conv_layer9_object)
    print 'Shape of convolutional layer: {0:s}'.format(
        conv_layer10_object.shape)

    model_object = keras.models.Model(
        input=input_layer_object, output=conv_layer10_object)

    model_object.compile(
        loss=keras_losses.weighted_cross_entropy(class_weights),
        optimizer=keras.optimizers.Adam(lr=LEARNING_RATE_FOR_U_NET),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    return model_object


def train_model_with_3d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, training_start_time_unix_sec,
        training_end_time_unix_sec, top_narr_directory_name,
        top_gridded_front_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes,
        num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains FCN, using 3-D examples generated on the fly.

    :param model_object: Instance of `keras.models.Model`.
    :param output_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_start_time_unix_sec: See documentation for
        `machine_learning_io.full_size_3d_example_generator`.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Same.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_start_time_unix_sec: See documentation for
        `machine_learning_io.full_size_3d_example_generator`.
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
            generator=trainval_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                num_classes=num_classes),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=trainval_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                num_classes=num_classes),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=[checkpoint_object],
            validation_data=
            trainval_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_distance_metres=dilation_distance_metres,
                num_classes=num_classes),
            validation_steps=num_validation_batches_per_epoch)


def apply_model_to_3d_example(
        model_object, target_time_unix_sec, top_narr_directory_name,
        top_gridded_front_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes,
        isotonic_model_object_by_class=None):
    """Applies FCN to one 3-D example.

    K = number of classes (possible values of target label)

    :param model_object: Instance of `keras.models.Model`.
    :param target_time_unix_sec: See doc for
        `testing_io.create_full_size_example`.
    :param top_narr_directory_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Number of classes.  This is K in the above discussion.
    :param isotonic_model_object_by_class: length-K list with trained instances
        of `sklearn.isotonic.IsotonicRegression`.  If None, will omit isotonic
        regression.
    :return: class_probability_matrix: 1-by-M-by-N-by-K numpy array of predicted
        class probabilities.
    :return: actual_target_matrix: 1-by-M-by-N numpy array of actual targets on
        the NARR grid.
    """

    predictor_matrix, actual_target_matrix = (
        testing_io.create_full_size_example(
            top_predictor_dir_name=top_narr_directory_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            valid_time_unix_sec=target_time_unix_sec,
            pressure_level_mb=pressure_level_mb,
            predictor_names=narr_predictor_names,
            normalization_type_string=ml_utils.Z_SCORE_STRING,
            dilation_distance_metres=dilation_distance_metres,
            num_classes=num_classes)
    )

    class_probability_matrix = model_object.predict(
        predictor_matrix, batch_size=1)
    actual_target_matrix = actual_target_matrix[..., 0]

    if isotonic_model_object_by_class is not None:
        num_grid_rows = class_probability_matrix.shape[1]
        num_grid_columns = class_probability_matrix.shape[2]

        this_class_probability_matrix = numpy.reshape(
            class_probability_matrix[0, ...],
            (num_grid_rows * num_grid_columns, num_classes)
        )

        these_observed_labels = numpy.reshape(
            actual_target_matrix[0, ...], num_grid_rows * num_grid_columns)

        this_class_probability_matrix = (
            isotonic_regression.apply_model_for_each_class(
                orig_class_probability_matrix=this_class_probability_matrix,
                observed_labels=these_observed_labels,
                model_object_by_class=isotonic_model_object_by_class)
        )

        this_class_probability_matrix = numpy.reshape(
            this_class_probability_matrix,
            (num_grid_rows, num_grid_columns, num_classes)
        )

        class_probability_matrix[0, ...] = this_class_probability_matrix

    return class_probability_matrix, actual_target_matrix
