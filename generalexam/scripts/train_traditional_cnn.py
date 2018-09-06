"""Trains CNN for patch classification.

Each input to the CNN is a downsized image.  The CNN predicts the class (no
front, warm front, or cold front) of the center pixel in the downsized image.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn_architecture
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.scripts import machine_learning_helper as ml_helper

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_downsized_examples=True)


def _run(num_epochs, num_examples_per_batch, num_examples_per_time,
         num_training_batches_per_epoch, num_validation_batches_per_epoch,
         num_rows_in_half_grid, num_columns_in_half_grid,
         dilation_distance_for_target_metres, weight_loss_function,
         class_fractions, num_classes, num_lead_time_steps,
         predictor_time_step_offsets, pressure_level_mb, narr_predictor_names,
         training_start_time_string, training_end_time_string,
         validation_start_time_string, validation_end_time_string,
         top_narr_dir_name, top_frontal_grid_dir_name, narr_mask_file_name,
         num_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
         conv_activation_function_string, alpha_for_elu, alpha_for_relu,
         use_batch_normalization, init_num_filters, conv_layer_dropout_fraction,
         dense_layer_dropout_fraction, l2_weight, output_file_name):
    """Trains CNN for patch classification.

    This is effectively the main method.

    :param num_epochs: See documentation at top of machine_learning_helper.py.
    :param num_examples_per_batch: Same.
    :param num_examples_per_time: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param dilation_distance_for_target_metres: Same.
    :param weight_loss_function: Same.
    :param class_fractions: Same.
    :param num_classes: Same.
    :param num_lead_time_steps: Same.
    :param predictor_time_step_offsets: Same.
    :param pressure_level_mb: Same.
    :param narr_predictor_names: Same.
    :param training_start_time_string: Same.
    :param training_end_time_string: Same.
    :param validation_start_time_string: Same.
    :param validation_end_time_string: Same.
    :param top_narr_dir_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_mask_file_name: Same.
    :param num_conv_layer_sets: Same.
    :param num_conv_layers_per_set: Same.
    :param pooling_type_string: Same.
    :param conv_activation_function_string: Same.
    :param alpha_for_elu: Same.
    :param alpha_for_relu: Same.
    :param use_batch_normalization: Same.
    :param init_num_filters: Same.
    :param conv_layer_dropout_fraction: Same.
    :param dense_layer_dropout_fraction: Same.
    :param l2_weight: Same.
    :param output_file_name: Same.
    :raises: ValueError: if num_lead_time_steps > 1.  This script cannot yet
        handle convolution over time.
    """

    if conv_layer_dropout_fraction <= 0:
        conv_layer_dropout_fraction = None
    if dense_layer_dropout_fraction <= 0:
        dense_layer_dropout_fraction = None
    if l2_weight <= 0:
        l2_weight = None
    if narr_mask_file_name == '':
        narr_mask_file_name = None

    training_start_time_unix_sec = time_conversion.string_to_unix_sec(
        training_start_time_string, INPUT_TIME_FORMAT)
    training_end_time_unix_sec = time_conversion.string_to_unix_sec(
        training_end_time_string, INPUT_TIME_FORMAT)

    validation_start_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_start_time_string, INPUT_TIME_FORMAT)
    validation_end_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_end_time_string, INPUT_TIME_FORMAT)

    if num_lead_time_steps <= 1:
        num_lead_time_steps = None
        predictor_time_step_offsets = None
    else:
        error_string = (
            'num_lead_time_steps > 1 (specifically {0:d}), but this script '
            'cannot yet handle convolution over time.'
        ).format(num_lead_time_steps)
        raise ValueError(error_string)

    if narr_mask_file_name is None:
        narr_mask_matrix = None
    else:
        print 'Reading NARR mask from: "{0:s}"...'.format(narr_mask_file_name)
        narr_mask_matrix = ml_utils.read_narr_mask(narr_mask_file_name)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_file_name, raise_error_if_missing=True)
    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)

    traditional_cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=num_examples_per_time,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        dilation_distance_for_target_metres=dilation_distance_for_target_metres,
        class_fractions=class_fractions,
        weight_loss_function=weight_loss_function,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        num_lead_time_steps=num_lead_time_steps,
        predictor_time_step_offsets=predictor_time_step_offsets,
        narr_mask_matrix=narr_mask_matrix)

    num_rows_in_grid = 2 * num_rows_in_half_grid + 1
    num_columns_in_grid = 2 * num_columns_in_half_grid + 1
    num_predictor_fields = len(narr_predictor_names)

    model_object = cnn_architecture.get_2d_swirlnet_architecture(
        num_radar_rows=num_rows_in_grid, num_radar_columns=num_columns_in_grid,
        num_radar_channels=num_predictor_fields,
        num_radar_conv_layer_sets=num_conv_layer_sets,
        num_conv_layers_per_set=num_conv_layers_per_set,
        pooling_type_string=pooling_type_string, num_classes=num_classes,
        conv_activation_function_string=conv_activation_function_string,
        alpha_for_elu=alpha_for_elu, alpha_for_relu=alpha_for_relu,
        use_batch_normalization=use_batch_normalization,
        init_num_radar_filters=init_num_filters,
        conv_layer_dropout_fraction=conv_layer_dropout_fraction,
        dense_layer_dropout_fraction=dense_layer_dropout_fraction,
        l2_weight=l2_weight,
        list_of_metric_functions=traditional_cnn.LIST_OF_METRIC_FUNCTIONS)

    traditional_cnn.train_with_3d_examples(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_examples_per_target_time=num_examples_per_time,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_narr_directory_name=top_narr_dir_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_for_target_metres=dilation_distance_for_target_metres,
        class_fractions=class_fractions,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        weight_loss_function=weight_loss_function,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        narr_mask_matrix=narr_mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        num_epochs=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EXAMPLES_PER_TIME_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        num_rows_in_half_grid=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_HALF_ROWS_ARG_NAME),
        num_columns_in_half_grid=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_HALF_COLUMNS_ARG_NAME),
        dilation_distance_for_target_metres=getattr(
            INPUT_ARG_OBJECT, ml_helper.DILATION_DISTANCE_ARG_NAME),
        weight_loss_function=bool(getattr(
            INPUT_ARG_OBJECT, ml_helper.WEIGHT_LOSS_ARG_NAME)),
        class_fractions=numpy.array(getattr(
            INPUT_ARG_OBJECT, ml_helper.CLASS_FRACTIONS_ARG_NAME)),
        num_classes=getattr(INPUT_ARG_OBJECT, ml_helper.NUM_CLASSES_ARG_NAME),
        num_lead_time_steps=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_LEAD_TIME_STEPS_ARG_NAME),
        predictor_time_step_offsets=numpy.array(getattr(
            INPUT_ARG_OBJECT, ml_helper.PREDICTOR_TIMES_ARG_NAME), dtype=int),
        pressure_level_mb=getattr(
            INPUT_ARG_OBJECT, ml_helper.PRESSURE_LEVEL_ARG_NAME),
        narr_predictor_names=getattr(
            INPUT_ARG_OBJECT, ml_helper.NARR_PREDICTORS_ARG_NAME),
        training_start_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.TRAINING_START_TIME_ARG_NAME),
        training_end_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.TRAINING_END_TIME_ARG_NAME),
        validation_start_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.VALIDN_START_TIME_ARG_NAME),
        validation_end_time_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.VALIDN_END_TIME_ARG_NAME),
        top_narr_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.NARR_DIRECTORY_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.FRONTAL_GRID_DIR_ARG_NAME),
        narr_mask_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.NARR_MASK_FILE_ARG_NAME),
        num_conv_layer_sets=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_CONV_LAYER_SETS_ARG_NAME),
        num_conv_layers_per_set=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_LAYERS_PER_SET_ARG_NAME),
        pooling_type_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.POOLING_TYPE_ARG_NAME),
        conv_activation_function_string=getattr(
            INPUT_ARG_OBJECT, ml_helper.ACTIVATION_FUNCTION_ARG_NAME),
        alpha_for_elu=getattr(
            INPUT_ARG_OBJECT, ml_helper.ALPHA_FOR_ELU_ARG_NAME),
        alpha_for_relu=getattr(
            INPUT_ARG_OBJECT, ml_helper.ALPHA_FOR_RELU_ARG_NAME),
        use_batch_normalization=bool(getattr(
            INPUT_ARG_OBJECT, ml_helper.USE_BATCH_NORM_ARG_NAME)),
        init_num_filters=getattr(
            INPUT_ARG_OBJECT, ml_helper.INIT_NUM_FILTERS_ARG_NAME),
        conv_layer_dropout_fraction=getattr(
            INPUT_ARG_OBJECT, ml_helper.CONV_LAYER_DROPOUT_ARG_NAME),
        dense_layer_dropout_fraction=getattr(
            INPUT_ARG_OBJECT, ml_helper.DENSE_LAYER_DROPOUT_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, ml_helper.L2_WEIGHT_ARG_NAME),
        output_file_name=getattr(
            INPUT_ARG_OBJECT, ml_helper.OUTPUT_FILE_ARG_NAME))
