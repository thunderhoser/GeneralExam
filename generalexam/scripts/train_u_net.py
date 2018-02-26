"""Trains U-net with architecture used in the following example.

https://github.com/zhixuhao/unet/blob/master/unet.py

For more on U-nets in general, see Ronneberger et al. (2015).

--- REFERENCES ---

Ronneberger, O., P. Fischer, and T. Brox (2015): "U-net: Convolutional networks
    for biomedical image segmentation". International Conference on Medical
    Image Computing and Computer-assisted Intervention, 234-241.
"""

import argparse
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import fcn
from generalexam.scripts import machine_learning as ml_script_helper

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_TEMP_NAME]

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_script_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_downsized_examples=True)


def _train_u_net(
        num_epochs, num_examples_per_batch, num_training_batches_per_epoch,
        num_validation_batches_per_epoch, dilation_half_width_for_target,
        positive_class_weight, pressure_level_mb, training_start_time_string,
        training_end_time_string, validation_start_time_string,
        validation_end_time_string, top_narr_dir_name,
        top_frontal_grid_dir_name, output_file_name):
    """Trains U-net with certain architecture.

    :param num_epochs: Number of training epochs.
    :param num_examples_per_batch: Number of examples (images over the full NARR
        grid) per batch.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param dilation_half_width_for_target: Half-width of dilation window (number
        of pixels).  Target images will be dilated, which increases the number
        of pixels labeled as frontal.  This accounts for uncertainty in the
        placement of fronts.
    :param positive_class_weight: Weight for positive class in loss function.
        This should be (1 - frequency of positive class).
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

    training_start_time_unix_sec = time_conversion.string_to_unix_sec(
        training_start_time_string, INPUT_TIME_FORMAT)
    training_end_time_unix_sec = time_conversion.string_to_unix_sec(
        training_end_time_string, INPUT_TIME_FORMAT)

    validation_start_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_start_time_string, INPUT_TIME_FORMAT)
    validation_end_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_end_time_string, INPUT_TIME_FORMAT)

    print 'Initializing model...'
    model_object = fcn.get_u_net(positive_class_weight=positive_class_weight)
    print SEPARATOR_STRING

    fcn.train_model_with_3d_examples(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_narr_directory_name=top_narr_dir_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        dilation_half_width_for_target=dilation_half_width_for_target,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_u_net(
        num_epochs=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT,
            ml_script_helper.NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT,
            ml_script_helper.NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME),
        dilation_half_width_for_target=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.DILATION_HALF_WIDTH_ARG_NAME),
        positive_class_weight=getattr(
            INPUT_ARG_OBJECT, ml_script_helper.POSITIVE_CLASS_WEIGHT_ARG_NAME),
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
