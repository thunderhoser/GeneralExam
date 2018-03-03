"""Trains convolutional neural net with MNIST architecture.

Said architecture was used to classify handwritten digits from the MNIST
(Modified National Institute of Standards and Technology) dataset.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import traditional_cnn
from generalexam.scripts import machine_learning as ml_script_helper

# TODO(thunderhoser): Explore regularization and batch normalization.

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_TEMP_NAME]

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_script_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_downsized_examples=True)


def _train_cnn(
        num_epochs, num_examples_per_batch, num_examples_per_time,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        num_rows_in_half_grid, num_columns_in_half_grid,
        dilation_distance_for_target_metres, class_fractions,
        weight_loss_function, pressure_level_mb, training_start_time_string,
        training_end_time_string, validation_start_time_string,
        validation_end_time_string, top_narr_dir_name,
        top_frontal_grid_dir_name, output_file_name):
    """Trains convolutional neural net with MNIST architecture.

    :param num_epochs: Number of training epochs.
    :param num_examples_per_batch: Number of examples (downsized images) per
        batch.
    :param num_examples_per_time: Number of examples (downsized images) sampled
        from each time step.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param num_rows_in_half_grid: Number of rows in half-grid for each downsized
        image.  Total number of rows will be 2 * `num_rows_in_half_grid` + 1.
    :param num_columns_in_half_grid: Number of columns in half-grid for each
        downsized image.  Total number of columns will be
        2 * `num_columns_in_half_grid` + 1.
    :param dilation_distance_for_target_metres: Dilation distance.  Target
        images will be dilated, which increases the number of pixels labeled as
        frontal.  This accounts for uncertainty in the placement of fronts.
    :param class_fractions: 1-D numpy array with fraction of examples in each
        class.  Data will be sampled according to these fractions for both
        training and validation.
    :param weight_loss_function: Boolean flag.  If 1, classes will be weighted
        differently in loss function (class weights inversely proportional to
        `class_fractions`).
    :param pressure_level_mb: NARR predictors will be taken from this pressure
        level (millibars).
    :param training_start_time_string: Time (format "yyyymmddHH").  Training
        examples will be taken randomly from the time period
        `training_start_time_string`...`training_end_time_string`.
    :param training_end_time_string: See above.
    :param validation_start_time_string: Time (format "yyyymmddHH").  Validation
        examples will be taken randomly from the time period
        `validation_start_time_string`...`validation_end_time_string`.
    :param validation_end_time_string: See above.
    :param top_narr_dir_name: Name of top-level directory with NARR data (one
        file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one per file, indicating which NARR grid cells are intersected by
        a front).
    :param output_file_name: Path to output file (HDF5 format) for trained
        model.
    """

    class_fractions = numpy.array(class_fractions)
    print 'Class fractions = {0:s} ... weight loss function? {1:d}'.format(
        str(class_fractions), weight_loss_function)

    training_start_time_unix_sec = time_conversion.string_to_unix_sec(
        training_start_time_string, INPUT_TIME_FORMAT)
    training_end_time_unix_sec = time_conversion.string_to_unix_sec(
        training_end_time_string, INPUT_TIME_FORMAT)

    validation_start_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_start_time_string, INPUT_TIME_FORMAT)
    validation_end_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_end_time_string, INPUT_TIME_FORMAT)

    print 'Initializing model...'
    model_object = traditional_cnn.get_cnn_with_mnist_architecture(
        num_classes=len(class_fractions),
        num_predictors=len(NARR_PREDICTOR_NAMES), convolve_over_time=False)
    print SEPARATOR_STRING

    model_dir_name, _ = os.path.split(output_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_dir_name)
    print 'Writing metadata to: "{0:s}"...'.format(metadata_file_name)

    traditional_cnn.write_model_metadata(
        num_epochs=num_epochs, num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=num_examples_per_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        dilation_distance_for_target_metres=dilation_distance_for_target_metres,
        class_fractions=class_fractions,
        weight_loss_function=weight_loss_function,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        pickle_file_name=metadata_file_name)

    traditional_cnn.train_with_3d_examples(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_examples_per_target_time=num_examples_per_time,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_narr_directory_name=top_narr_dir_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_for_target_metres=dilation_distance_for_target_metres,
        class_fractions=class_fractions,
        weight_loss_function=weight_loss_function,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_cnn(
        num_epochs=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_EXAMPLES_PER_TIME_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT,
            ml_script_helper.NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT,
            ml_script_helper.NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME),
        num_rows_in_half_grid=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_ROWS_IN_HALF_GRID_ARG_NAME),
        num_columns_in_half_grid=getattr(
            INPUT_ARG_OBJECT,
            ml_script_helper.NUM_COLUMNS_IN_HALF_GRID_ARG_NAME),
        dilation_distance_for_target_metres=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.DILATION_DISTANCE_ARG_NAME),
        class_fractions=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.CLASS_FRACTIONS_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, ml_script_helper.WEIGHT_LOSS_FUNCTION_ARG_NAME)),
        pressure_level_mb=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.PRESSURE_LEVEL_ARG_NAME),
        training_start_time_string=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.TRAINING_START_TIME_ARG_NAME),
        training_end_time_string=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.TRAINING_END_TIME_ARG_NAME),
        validation_start_time_string=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.VALIDATION_START_TIME_ARG_NAME),
        validation_end_time_string=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.VALIDATION_END_TIME_ARG_NAME),
        top_narr_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.TOP_NARR_DIR_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.TOP_FRONTAL_GRID_DIR_ARG_NAME),
        output_file_name=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.OUTPUT_FILE_ARG_NAME))
