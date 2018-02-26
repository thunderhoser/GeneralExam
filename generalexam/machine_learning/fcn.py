"""Training/testing methods for FCN (fully convolutional net).

A fully convolutional net (or "spatially dense" CNN) is one for which the output
(prediction) is spatially explicit.  The opposite is a "traditional CNN" (see
traditional_cnn.py).

For a traditional CNN, there is one output (prediction) per image.  For an FCN,
there is one output per pixel.  Thus, a traditional CNN can predict *whether or
not* a feature exists in some image, but only an FCN can predict *where* said
feature exists in the image.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples (images)
M = number of pixel rows in each image
N = number of pixel columns in each image
T = number of time steps per example (i.e., number of images in each sequence)
C = number of channels (predictor variables) in each image

For a 3-D example, the dimensions are M x N x C (M rows, N columns, C predictor
variables).

For a 4-D example, the dimensions are M x N x T x C (M rows, N columns, T time
steps, C predictor variables).

--- REFERENCES ---

Ronneberger, O., P. Fischer, and T. Brox (2015): "U-net: Convolutional networks
    for biomedical image segmentation". International Conference on Medical
    Image Computing and Computer-assisted Intervention, 234-241.
"""

import keras.models
import keras.layers
from keras.callbacks import ModelCheckpoint
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import machine_learning_io as ml_io
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import keras_metrics
from generalexam.machine_learning import keras_losses

DEFAULT_POSITIVE_CLASS_WEIGHT = 0.935

LEARNING_RATE_FOR_UNET = 1e-4
LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.csi, keras_metrics.frequency_bias,
    keras_metrics.pod, keras_metrics.pofd, keras_metrics.success_ratio,
    keras_metrics.focn]

NUM_CLASSES = 2
DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_TEMP_NAME]

# TODO(thunderhoser): This is a hack.  Use cropping and padding.
NUM_NARR_GRID_ROWS_TO_USE = 272
NUM_NARR_GRID_COLUMNS_TO_USE = 256


def get_unet(positive_class_weight=DEFAULT_POSITIVE_CLASS_WEIGHT):
    """Creates U-net with architecture used in the following example.

    https://github.com/zhixuhao/unet/blob/master/unet.py

    For more on U-nets in general, see Ronneberger et al. (2015).

    :param positive_class_weight: Weight for positive class in loss function.
        This should be (1 - frequency of positive class).  With dilation half-
        width of 8 pixels, frequency of positive class (fraction of pixels
        labeled as frontal) is ~0.065.  This is why default weight is 0.935.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    input_layer_object = keras.layers.Input(
        shape=(NUM_NARR_GRID_ROWS_TO_USE,
               NUM_NARR_GRID_COLUMNS_TO_USE,
               len(DEFAULT_NARR_PREDICTOR_NAMES)))

    conv_layer1_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(input_layer_object)
    print 'Shape of convolutional layer 1: {0:s}'.format(
        conv_layer1_object.shape)

    conv_layer1_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer1_object)
    print 'Shape of convolutional layer 1: {0:s}'.format(
        conv_layer1_object.shape)

    pooling_layer1_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer1_object)
    print 'Shape of pooling layer 1: {0:s}'.format(pooling_layer1_object.shape)

    conv_layer2_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer1_object)
    print 'Shape of convolutional layer 2: {0:s}'.format(
        conv_layer2_object.shape)

    conv_layer2_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer2_object)
    print 'Shape of convolutional layer 2: {0:s}'.format(
        conv_layer2_object.shape)

    pooling_layer2_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer2_object)
    print 'Shape of pooling layer 2: {0:s}'.format(pooling_layer2_object.shape)

    conv_layer3_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer2_object)
    print 'Shape of convolutional layer 3: {0:s}'.format(
        conv_layer3_object.shape)

    conv_layer3_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer3_object)
    print 'Shape of convolutional layer 3: {0:s}'.format(
        conv_layer3_object.shape)

    pooling_layer3_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(conv_layer3_object)
    print 'Shape of pooling layer 3: {0:s}'.format(pooling_layer3_object.shape)

    conv_layer4_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer3_object)
    print 'Shape of convolutional layer 4: {0:s}'.format(
        conv_layer4_object.shape)

    conv_layer4_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer4_object)
    print 'Shape of convolutional layer 4: {0:s}'.format(
        conv_layer4_object.shape)

    dropout_layer4_object = keras.layers.Dropout(rate=0.5)(conv_layer4_object)
    pooling_layer4_object = keras.layers.MaxPooling2D(
        pool_size=(2, 2))(dropout_layer4_object)
    print 'Shape of pooling layer 4: {0:s}'.format(pooling_layer4_object.shape)

    conv_layer5_object = keras.layers.Conv2D(
        filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(pooling_layer4_object)
    print 'Shape of convolutional layer 5: {0:s}'.format(
        conv_layer5_object.shape)

    conv_layer5_object = keras.layers.Conv2D(
        filters=1024, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer5_object)
    print 'Shape of convolutional layer 5: {0:s}'.format(
        conv_layer5_object.shape)

    dropout_layer5_object = keras.layers.Dropout(rate=0.5)(conv_layer5_object)

    upsampling_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(dropout_layer5_object))
    print 'Shape of upsampling layer 6: {0:s}'.format(
        upsampling_layer6_object.shape)

    merged_layer6_object = keras.layers.merge(
        [dropout_layer4_object, upsampling_layer6_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer 6: {0:s}'.format(merged_layer6_object.shape)

    conv_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer6_object)
    print 'Shape of convolutional layer 6: {0:s}'.format(
        conv_layer6_object.shape)

    conv_layer6_object = keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer6_object)
    print 'Shape of convolutional layer 6: {0:s}'.format(
        conv_layer6_object.shape)

    upsampling_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer6_object))
    print 'Shape of upsampling layer 7: {0:s}'.format(
        upsampling_layer7_object.shape)

    merged_layer7_object = keras.layers.merge(
        [conv_layer3_object, upsampling_layer7_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer 7: {0:s}'.format(merged_layer7_object.shape)

    conv_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer7_object)
    print 'Shape of convolutional layer 7: {0:s}'.format(
        conv_layer7_object.shape)

    conv_layer7_object = keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer7_object)
    print 'Shape of convolutional layer 7: {0:s}'.format(
        conv_layer7_object.shape)

    upsampling_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer7_object))
    print 'Shape of upsampling layer 8: {0:s}'.format(
        upsampling_layer8_object.shape)

    merged_layer8_object = keras.layers.merge(
        [conv_layer2_object, upsampling_layer8_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer 8: {0:s}'.format(merged_layer8_object.shape)

    conv_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer8_object)
    print 'Shape of convolutional layer 8: {0:s}'.format(
        conv_layer8_object.shape)

    conv_layer8_object = keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer8_object)
    print 'Shape of convolutional layer 8: {0:s}'.format(
        conv_layer8_object.shape)

    upsampling_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(2, 2), activation='relu', padding='same',
        kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 2))(conv_layer8_object))
    print 'Shape of upsampling layer 9: {0:s}'.format(
        upsampling_layer9_object.shape)

    merged_layer9_object = keras.layers.merge(
        [conv_layer1_object, upsampling_layer9_object], mode='concat',
        concat_axis=3)
    print 'Shape of merged layer 9: {0:s}'.format(merged_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(merged_layer9_object)
    print 'Shape of convolutional layer 9: {0:s}'.format(
        conv_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer9_object)
    print 'Shape of convolutional layer 9: {0:s}'.format(
        conv_layer9_object.shape)

    conv_layer9_object = keras.layers.Conv2D(
        filters=2, kernel_size=(3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal')(conv_layer9_object)
    print 'Shape of convolutional layer 9: {0:s}'.format(
        conv_layer9_object.shape)

    conv_layer10_object = keras.layers.Conv2D(
        filters=1, kernel_size=(1, 1), activation='sigmoid')(conv_layer9_object)
    print 'Shape of convolutional layer 10: {0:s}'.format(
        conv_layer10_object.shape)

    model_object = keras.models.Model(
        input=input_layer_object, output=conv_layer10_object)

    model_object.compile(
        loss=keras_losses.weighted_cross_entropy(
            positive_class_weight=positive_class_weight),
        optimizer=keras.optimizers.Adam(lr=LEARNING_RATE_FOR_UNET),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    # model_object.compile(
    #     loss=keras.losses.binary_crossentropy,
    #     optimizer=keras.optimizers.Adam(lr=LEARNING_RATE_FOR_UNET),
    #     metrics=LIST_OF_METRIC_FUNCTIONS)

    return model_object


def train_model_with_3d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, training_start_time_unix_sec,
        training_end_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target, num_validation_batches_per_epoch=None,
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
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
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
            generator=ml_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1,
            callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=ml_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1,
            callbacks=[checkpoint_object],
            validation_data=ml_io.full_size_3d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            validation_steps=num_validation_batches_per_epoch)


def train_model_with_4d_examples(
        model_object, output_file_name, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, num_predictor_time_steps,
        num_lead_time_steps, training_start_time_unix_sec,
        training_end_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target, num_validation_batches_per_epoch=None,
        validation_start_time_unix_sec=None, validation_end_time_unix_sec=None):
    """Trains FCN, using 4-D examples generated on the fly.

    :param model_object: Instance of `keras.models.Model`.
    :param output_file_name: Path to output file (HDF5 format).  The model will
        be saved here after every epoch.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_predictor_time_steps: See documentation for
        `machine_learning_io.full_size_4d_example_generator`.
    :param num_lead_time_steps: Same.
    :param training_start_time_unix_sec: Same.
    :param training_end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
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
            generator=ml_io.full_size_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=[checkpoint_object])

    else:
        error_checking.assert_is_integer(num_validation_batches_per_epoch)
        error_checking.assert_is_geq(num_validation_batches_per_epoch, 1)

        checkpoint_object = ModelCheckpoint(
            output_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        model_object.fit_generator(
            generator=ml_io.full_size_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=training_start_time_unix_sec,
                last_target_time_unix_sec=training_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=[checkpoint_object],
            validation_data=ml_io.full_size_4d_example_generator(
                num_examples_per_batch=num_examples_per_batch,
                first_target_time_unix_sec=validation_start_time_unix_sec,
                last_target_time_unix_sec=validation_end_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb,
                dilation_half_width_for_target=dilation_half_width_for_target),
            validation_steps=num_validation_batches_per_epoch)


def apply_model_to_3d_example(
        model_object, target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target):
    """Applies FCN to one 3-D example.

    :param model_object: Instance of `keras.models.Model`.
    :param target_time_unix_sec: See documentation for
        `testing_io.create_full_size_3d_example`.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :return: predicted_target_matrix: M-by-N numpy array of predicted targets on
        the NARR grid.
    :return: actual_target_matrix: M-by-N numpy array of actual targets on the
        NARR grid.
    """

    predictor_matrix, actual_target_matrix = (
        testing_io.create_full_size_3d_example(
            target_time_unix_sec=target_time_unix_sec,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb,
            dilation_half_width_for_target=dilation_half_width_for_target))

    predicted_target_matrix = model_object.predict(
        predictor_matrix, batch_size=1)

    return (predicted_target_matrix[0, ..., 0].astype('float'),
            actual_target_matrix[0, ..., 0])


def apply_model_to_4d_example(
        model_object, target_time_unix_sec, num_predictor_time_steps,
        num_lead_time_steps, top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target):
    """Applies FCN to one 4-D example.

    :param model_object: Instance of `keras.models.Model`.
    :param target_time_unix_sec: See documentation for
        `testing_io.create_full_size_4d_example`.
    :param num_predictor_time_steps: Same.
    :param num_lead_time_steps: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :return: predicted_target_matrix: M-by-N numpy array of predicted targets on
        the NARR grid.
    :return: actual_target_matrix: M-by-N numpy array of actual targets on the
        NARR grid.
    """

    predictor_matrix, actual_target_matrix = (
        testing_io.create_full_size_4d_example(
            target_time_unix_sec=target_time_unix_sec,
            num_predictor_time_steps=num_predictor_time_steps,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb,
            dilation_half_width_for_target=dilation_half_width_for_target))

    predicted_target_matrix = model_object.predict(
        predictor_matrix, batch_size=1)

    return (predicted_target_matrix[0, ..., 0].astype('float'),
            actual_target_matrix[0, ..., 0])
