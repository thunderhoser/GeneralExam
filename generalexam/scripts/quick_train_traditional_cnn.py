"""Trains CNN for patch classification, using downsized 3-D images.

What makes this script "quick" is that uses
`training_validation_io.quick_downsized_3d_example_gen`, rather than
`training_validation_io.downsized_3d_example_generator`.  The former generates
examples from processed files (created by
`training_validation_io.write_downsized_3d_examples`), whereas the latter
generates examples from raw NARR files.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn_architecture
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import traditional_cnn
from generalexam.scripts import machine_learning_helper as ml_helper

INPUT_TIME_FORMAT = '%Y%m%d%H'

NUM_EXAMPLES_PER_TIME = 8
DILATION_DISTANCE_METRES = 50000.
CLASS_FRACTIONS = numpy.array([0.5, 0.25, 0.25])
WEIGHT_LOSS_FUNCTION = False
PRESSURE_LEVEL_MB = 1000

NARR_MASK_FILE_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p')
TOP_INPUT_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/shuffled')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = ml_helper.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_downsized_examples=True,
    use_quick_generator=True)


def _run(num_epochs, num_examples_per_batch, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, num_rows_in_half_grid,
         num_columns_in_half_grid, num_classes, narr_predictor_names,
         training_start_time_string, training_end_time_string,
         validation_start_time_string, validation_end_time_string,
         num_conv_layer_sets, num_conv_layers_per_set, pooling_type_string,
         conv_activation_function_string, alpha_for_elu, alpha_for_relu,
         use_batch_normalization, init_num_filters, conv_layer_dropout_fraction,
         dense_layer_dropout_fraction, l2_weight, output_file_name):
    """Trains CNN for patch classification, using downsized 3-D images.

    This is effectively the main method.

    :param num_epochs: See doc at top of machine_learning_helper.py.
    :param num_examples_per_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param num_classes: Same.
    :param narr_predictor_names: Same.
    :param training_start_time_string: Same.
    :param training_end_time_string: Same.
    :param validation_start_time_string: Same.
    :param validation_end_time_string: Same.
    :param num_conv_layer_sets: See doc at top of machine_learning_helper.py.
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
    """

    if conv_layer_dropout_fraction <= 0:
        conv_layer_dropout_fraction = None
    if dense_layer_dropout_fraction <= 0:
        dense_layer_dropout_fraction = None
    if l2_weight <= 0:
        l2_weight = None

    training_start_time_unix_sec = time_conversion.string_to_unix_sec(
        training_start_time_string, INPUT_TIME_FORMAT)
    training_end_time_unix_sec = time_conversion.string_to_unix_sec(
        training_end_time_string, INPUT_TIME_FORMAT)

    validation_start_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_start_time_string, INPUT_TIME_FORMAT)
    validation_end_time_unix_sec = time_conversion.string_to_unix_sec(
        validation_end_time_string, INPUT_TIME_FORMAT)

    print 'Reading NARR mask from: "{0:s}"...'.format(NARR_MASK_FILE_NAME)
    narr_mask_matrix = ml_utils.read_narr_mask(NARR_MASK_FILE_NAME)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_file_name, raise_error_if_missing=False)
    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)

    traditional_cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=NUM_EXAMPLES_PER_TIME,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        dilation_distance_metres=DILATION_DISTANCE_METRES,
        class_fractions=CLASS_FRACTIONS,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=PRESSURE_LEVEL_MB,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        num_lead_time_steps=None,
        predictor_time_step_offsets=None,
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

    traditional_cnn.quick_train_3d(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_input_dir_name=TOP_INPUT_DIR_NAME,
        narr_predictor_names=narr_predictor_names, num_classes=num_classes,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        num_epochs=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_TRAIN_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_VALIDN_BATCHES_ARG_NAME),
        num_rows_in_half_grid=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_HALF_ROWS_ARG_NAME),
        num_columns_in_half_grid=getattr(
            INPUT_ARG_OBJECT, ml_helper.NUM_HALF_COLUMNS_ARG_NAME),
        num_classes=getattr(INPUT_ARG_OBJECT, ml_helper.NUM_CLASSES_ARG_NAME),
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
