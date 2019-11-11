"""Trains CNN with pre-processed example files."""

import argparse
import numpy
import keras
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.scripts import machine_learning_helper as ml_helper

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = ml_helper.TIME_FORMAT
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_BATCH_NUMBER = 0
LAST_BATCH_NUMBER = int(1e12)

WEIGHT_LOSS_FUNCTION = False
NUM_EXAMPLES_PER_TIME_DUMMY = 8
CLASS_FRACTIONS_DUMMY = numpy.array([0.5, 0.25, 0.25])

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_helper.add_input_args(
    argument_parser=INPUT_ARG_PARSER, use_downsized_files=True)


def _run(input_model_file_name, predictor_names, pressure_levels_mb,
         x_translations_pixels, y_translations_pixels, ccw_rotation_angles_deg,
         noise_standard_deviation, num_noisings, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         num_ex_per_train_batch, top_validation_dir_name,
         first_validation_time_string, last_validation_time_string,
         num_ex_per_validn_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains CNN with pre-processed example files.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param predictor_names: Same.
    :param pressure_levels_mb: Same.
    :param x_translations_pixels: Same.
    :param y_translations_pixels: Same.
    :param ccw_rotation_angles_deg: Same.
    :param noise_standard_deviation: Same.
    :param num_noisings: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_ex_per_train_batch: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_ex_per_validn_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    """

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    translate_flag = (
        len(x_translations_pixels) > 1
        or x_translations_pixels[0] != 0 or y_translations_pixels[0] != 0
    )

    if not translate_flag:
        x_translations_pixels = None
        y_translations_pixels = None

    if len(ccw_rotation_angles_deg) == 1 and ccw_rotation_angles_deg[0] == 0:
        ccw_rotation_angles_deg = None

    if num_noisings <= 0:
        num_noisings = 0
        noise_standard_deviation = None

    augmentation_dict = {
        trainval_io.X_TRANSLATIONS_KEY: x_translations_pixels,
        trainval_io.Y_TRANSLATIONS_KEY: y_translations_pixels,
        trainval_io.ROTATION_ANGLES_KEY: ccw_rotation_angles_deg,
        trainval_io.NUM_NOISINGS_KEY: num_noisings,
        trainval_io.NOISE_STDEV_KEY: noise_standard_deviation
    }

    # Read architecture.
    print('Reading architecture from: "{0:s}"...'.format(input_model_file_name))
    model_object = cnn.read_model(input_model_file_name)
    # model_object = keras.models.clone_model(model_object)

    # TODO(thunderhoser): This is a HACK.
    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=cnn.LIST_OF_METRIC_FUNCTIONS
    )

    print(SEPARATOR_STRING)
    model_object.summary()
    print(SEPARATOR_STRING)

    # Read metadata from one training file.
    training_file_names = examples_io.find_many_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=FIRST_BATCH_NUMBER,
        last_batch_number=LAST_BATCH_NUMBER)

    print('Reading metadata from: "{0:s}"...'.format(training_file_names[0]))
    this_example_dict = examples_io.read_file(
        netcdf_file_name=training_file_names[0], metadata_only=True
    )

    normalization_type_string = this_example_dict[
        examples_io.NORMALIZATION_TYPE_KEY]
    dilation_distance_metres = this_example_dict[
        examples_io.DILATION_DISTANCE_KEY]
    mask_matrix = this_example_dict[examples_io.MASK_MATRIX_KEY]

    # Write metadata for CNN.
    model_metafile_name = cnn.find_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(model_metafile_name))

    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    cnn.write_metadata(
        pickle_file_name=model_metafile_name, num_epochs=num_epochs,
        num_ex_per_train_batch=num_ex_per_train_batch,
        num_ex_per_validn_batch=num_ex_per_validn_batch,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME_DUMMY,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        normalization_type_string=normalization_type_string,
        dilation_distance_metres=dilation_distance_metres,
        class_fractions=CLASS_FRACTIONS_DUMMY,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec,
        mask_matrix=mask_matrix, augmentation_dict=augmentation_dict)

    print(SEPARATOR_STRING)
    num_classes = len(CLASS_FRACTIONS_DUMMY)

    training_generator = trainval_io.downsized_generator_from_example_files(
        top_input_dir_name=top_training_dir_name,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec,
        num_examples_per_batch=num_ex_per_train_batch,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_classes=num_classes, augmentation_dict=augmentation_dict)

    validation_generator = trainval_io.downsized_generator_from_example_files(
        top_input_dir_name=top_validation_dir_name,
        first_time_unix_sec=first_validation_time_unix_sec,
        last_time_unix_sec=last_validation_time_unix_sec,
        num_examples_per_batch=num_ex_per_validn_batch,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_classes=num_classes)

    cnn.train_cnn(
        model_object=model_object,
        output_model_file_name=output_model_file_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        training_generator=training_generator,
        validation_generator=validation_generator,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        class_fractions=CLASS_FRACTIONS_DUMMY)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.INPUT_MODEL_FILE_ARG_NAME),
        predictor_names=getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_NAMES_ARG_NAME),
        pressure_levels_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, ml_helper.PRESSURE_LEVELS_ARG_NAME),
            dtype=int
        ),
        x_translations_pixels=numpy.array(
            getattr(INPUT_ARG_OBJECT, ml_helper.X_TRANSLATIONS_ARG_NAME),
            dtype=int),
        y_translations_pixels=numpy.array(
            getattr(INPUT_ARG_OBJECT, ml_helper.Y_TRANSLATIONS_ARG_NAME),
            dtype=int),
        ccw_rotation_angles_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, ml_helper.ROTATION_ANGLES_ARG_NAME),
            dtype=float),
        noise_standard_deviation=getattr(
            INPUT_ARG_OBJECT, ml_helper.NOISE_STDEV_ARG_NAME),
        num_noisings=getattr(INPUT_ARG_OBJECT, ml_helper.NUM_NOISINGS_ARG_NAME),
        top_training_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.LAST_TRAINING_TIME_ARG_NAME),
        num_ex_per_train_batch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EX_PER_TRAIN_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.LAST_VALIDATION_TIME_ARG_NAME),
        num_ex_per_validn_batch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EX_PER_VALIDN_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, ml_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_VALIDATION_BATCHES_ARG_NAME),
        output_model_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.OUTPUT_FILE_ARG_NAME)
    )
