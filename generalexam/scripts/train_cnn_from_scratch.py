"""Trains CNN from scratch.

"From scratch" means that the net is trained with examples created on the fly
from raw NARR data and gridded front labels, rather than pre-processed files
readable by `training_validation_io.read_downsized_3d_examples`.
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

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_helper.add_input_args(
    argument_parser=INPUT_ARG_PARSER, use_downsized_files=False)


def _run(input_model_file_name, predictor_names, pressure_level_mb,
         dilation_distance_metres, num_lead_time_steps,
         predictor_time_step_offsets, num_examples_per_time,
         weight_loss_function, class_fractions, top_predictor_dir_name,
         top_gridded_front_dir_name, mask_file_name,
         first_training_time_string, last_training_time_string,
         first_validation_time_string, last_validation_time_string,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains CNN from scratch.

    This is effectively the main method.

    :param input_model_file_name: See documentation at top of file.
    :param predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_lead_time_steps: Same.
    :param predictor_time_step_offsets: Same.
    :param num_examples_per_time: Same.
    :param weight_loss_function: Same.
    :param class_fractions: Same.
    :param top_predictor_dir_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param mask_file_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    :raises: ValueError: if `num_lead_time_steps > 1`.
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

    if mask_file_name in ['', 'None']:
        mask_file_name = None
        mask_matrix = None

    if num_lead_time_steps <= 1:
        num_lead_time_steps = None
        predictor_time_step_offsets = None
    else:
        error_string = (
            'This script cannot yet handle num_lead_time_steps > 1 '
            '(specifically {0:d}).'
        ).format(num_lead_time_steps)

        raise ValueError(error_string)

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

    if mask_file_name is None:
        mask_matrix = None
    else:
        print 'Reading mask from: "{0:s}"...'.format(mask_file_name)
        mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)
    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)

    traditional_cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=num_examples_per_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_rows_in_half_grid=num_half_rows,
        num_columns_in_half_grid=num_half_columns,
        dilation_distance_metres=dilation_distance_metres,
        class_fractions=class_fractions,
        weight_loss_function=weight_loss_function,
        narr_predictor_names=predictor_names,
        pressure_level_mb=pressure_level_mb,
        training_start_time_unix_sec=first_training_time_unix_sec,
        training_end_time_unix_sec=last_training_time_unix_sec,
        validation_start_time_unix_sec=first_validation_time_unix_sec,
        validation_end_time_unix_sec=last_validation_time_unix_sec,
        num_lead_time_steps=num_lead_time_steps,
        predictor_time_step_offsets=predictor_time_step_offsets,
        narr_mask_matrix=mask_matrix)

    print SEPARATOR_STRING

    traditional_cnn.train_with_3d_examples(
        model_object=model_object, output_file_name=output_model_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_examples_per_target_time=num_examples_per_time,
        training_start_time_unix_sec=first_training_time_unix_sec,
        training_end_time_unix_sec=last_training_time_unix_sec,
        top_narr_directory_name=top_predictor_dir_name,
        top_gridded_front_dir_name=top_gridded_front_dir_name,
        narr_predictor_names=predictor_names,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_metres=dilation_distance_metres,
        class_fractions=class_fractions,
        num_rows_in_half_grid=num_half_rows,
        num_columns_in_half_grid=num_half_columns,
        weight_loss_function=weight_loss_function,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=first_validation_time_unix_sec,
        validation_end_time_unix_sec=last_validation_time_unix_sec,
        narr_mask_matrix=mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.INPUT_MODEL_FILE_ARG_NAME),
        predictor_names=getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_NAMES_ARG_NAME),
        pressure_level_mb=getattr(
            INPUT_ARG_OBJECT, ml_helper.PRESSURE_LEVEL_ARG_NAME),
        dilation_distance_metres=float(getattr(
            INPUT_ARG_OBJECT, ml_helper.DILATION_DISTANCE_ARG_NAME)),
        num_lead_time_steps=getattr(
            INPUT_ARG_OBJECT, ml_helper.LEAD_TIME_ARG_NAME),
        predictor_time_step_offsets=numpy.array(getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_TIMES_ARG_NAME), dtype=int),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EX_PER_TIME_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, ml_helper.WEIGHT_LOSS_ARG_NAME)),
        class_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT,
                    ml_helper.DOWNSAMPLING_FRACTIONS_ARG_NAME),
            dtype=float),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_DIR_ARG_NAME),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.FRONT_DIR_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, ml_helper.MASK_FILE_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.LAST_TRAINING_TIME_ARG_NAME),
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
