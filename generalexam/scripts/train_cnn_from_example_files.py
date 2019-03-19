"""Trains CNN with example files.

"Example files" are pre-processed files that contain downsized examples,
readable by `learning_examples_io.read_file`.
"""

import argparse
import numpy
import keras
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import traditional_cnn
from generalexam.scripts import machine_learning_helper as ml_helper

TIME_FORMAT = ml_helper.TIME_FORMAT
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_CLASSES = 3
WEIGHT_LOSS_FLAG = False
NUM_EXAMPLES_PER_TIME_DUMMY = 8
DILATION_DISTANCE_METRES = 50000.
CLASS_FRACTIONS = numpy.array([0.5, 0.25, 0.25])
NARR_MASK_FILE_NAME = ml_helper.DEFAULT_MASK_FILE_NAME

# TODO(thunderhoser): Fix this HACK.
PRESSURE_LEVEL_MB = 1000

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_helper.add_input_args(
    argument_parser=INPUT_ARG_PARSER, use_downsized_files=True)


def _run(input_model_file_name, narr_predictor_names,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, top_validation_dir_name,
         first_validation_time_string, last_validation_time_string,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains CNN with example files.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param narr_predictor_names: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_batch: Same.
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

    # Read architecture.
    print 'Reading architecture from: "{0:s}"...'.format(input_model_file_name)
    model_object = traditional_cnn.read_keras_model(input_model_file_name)
    model_object = keras.models.clone_model(model_object)

    # TODO(thunderhoser): This is a HACK.
    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=traditional_cnn.LIST_OF_METRIC_FUNCTIONS)

    print SEPARATOR_STRING
    model_object.summary()
    print SEPARATOR_STRING

    # Write metadata.
    input_tensor = model_object.input
    num_grid_rows = input_tensor.get_shape().as_list()[1]
    num_grid_columns = input_tensor.get_shape().as_list()[2]

    num_half_rows = int(numpy.round((num_grid_rows - 1) / 2))
    num_half_columns = int(numpy.round((num_grid_columns - 1) / 2))

    print 'Reading NARR mask from: "{0:s}"...'.format(NARR_MASK_FILE_NAME)
    narr_mask_matrix = ml_utils.read_narr_mask(NARR_MASK_FILE_NAME)[0]

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)
    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)

    traditional_cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=NUM_EXAMPLES_PER_TIME_DUMMY,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_rows_in_half_grid=num_half_rows,
        num_columns_in_half_grid=num_half_columns,
        dilation_distance_metres=DILATION_DISTANCE_METRES,
        class_fractions=CLASS_FRACTIONS,
        weight_loss_function=WEIGHT_LOSS_FLAG,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=PRESSURE_LEVEL_MB,
        training_start_time_unix_sec=first_training_time_unix_sec,
        training_end_time_unix_sec=last_training_time_unix_sec,
        validation_start_time_unix_sec=first_validation_time_unix_sec,
        validation_end_time_unix_sec=last_validation_time_unix_sec,
        num_lead_time_steps=None, predictor_time_step_offsets=None,
        narr_mask_matrix=narr_mask_matrix)

    print SEPARATOR_STRING

    traditional_cnn.quick_train_3d(
        model_object=model_object, output_file_name=output_model_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_start_time_unix_sec=first_training_time_unix_sec,
        training_end_time_unix_sec=last_training_time_unix_sec,
        top_training_dir_name=top_training_dir_name,
        top_validation_dir_name=top_validation_dir_name,
        narr_predictor_names=narr_predictor_names, num_classes=NUM_CLASSES,
        num_rows_in_half_grid=num_half_rows,
        num_columns_in_half_grid=num_half_columns,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=first_validation_time_unix_sec,
        validation_end_time_unix_sec=last_validation_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.INPUT_MODEL_FILE_ARG_NAME),
        narr_predictor_names=getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_NAMES_ARG_NAME),
        top_training_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.LAST_TRAINING_TIME_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.LAST_VALIDATION_TIME_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EX_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, ml_helper.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_VALIDATION_BATCHES_ARG_NAME),
        output_model_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.OUTPUT_FILE_ARG_NAME)
    )
