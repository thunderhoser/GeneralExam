"""Trains CNN with simple architecture for patch classification.

Each input to the CNN is a downsized image.  The CNN predicts the class (no
front, warm front, or cold front) of the center pixel in the downsized image.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import cnn_architecture
from generalexam.machine_learning import machine_learning_utils as ml_utils

NUM_EPOCHS = 200
NUM_EXAMPLES_PER_TIME = 8
NUM_TRAINING_BATCHES_PER_EPOCH = 32
NUM_VALIDATION_BATCHES_PER_EPOCH = 16
DILATION_DISTANCE_METRES = 50000.
WEIGHT_LOSS_FUNCTION = False
CLASS_FRACTIONS = numpy.array([0.5, 0.25, 0.25])
NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_THETA_NAME
]

INPUT_TIME_FORMAT = '%Y%m%d%H'
TRAINING_START_TIME_STRING = '2008110515'
TRAINING_END_TIME_STRING = '2014122421'
VALIDATION_START_TIME_STRING = '2015010100'
VALIDATION_END_TIME_STRING = '2015122421'
TOP_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
NARR_MASK_FILE_NAME = '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p'

NUM_LEAD_TIME_STEPS = None
PREDICTOR_TIME_STEP_OFFSETS = None

NUM_HALF_ROWS_ARG_NAME = 'num_rows_in_half_grid'
NUM_HALF_COLUMNS_ARG_NAME = 'num_columns_in_half_grid'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
ARCHITECTURE_ID_ARG_NAME = 'architecture_id'
L2_WEIGHT_ARG_NAME = 'l2_weight'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

NUM_HALF_ROWS_HELP_STRING = (
    'Number of rows in half-grid for each downsized image.  Total number of '
    'rows will be 1 + 2 * `{0:s}`.'
).format(NUM_HALF_ROWS_ARG_NAME)

NUM_HALF_COLUMNS_HELP_STRING = (
    'Number of columns in half-grid for each downsized image.  Total number of '
    'columns will be 1 + 2 * `{0:s}`.'
).format(NUM_HALF_COLUMNS_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Will train the CNN with the following fields at this pressure level '
    '(millibars).\n{0:s}'
).format(str(NARR_PREDICTOR_NAMES))

BATCH_SIZE_HELP_STRING = (
    'Number of examples in each training or validation batch.')

ARCHITECTURE_ID_HELP_STRING = (
    'Integer ID for CNN architecture.  For example, if `{0:s}` = 1, the '
    'architecture will be created by `cnn_architecture.get_first_architecture`.'
)

L2_WEIGHT_HELP_STRING = (
    'L2-regularization weight (will be applied to each conv layer).  If you '
    'want no L2 regularization, leave this alone.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format), which will contain the trained model.')

DEFAULT_NUM_HALF_ROWS = 16
DEFAULT_NUM_HALF_COLUMNS = 16
DEFAULT_L2_WEIGHT = 0.001
NUM_EXAMPLES_PER_BATCH_DEFAULT = 1024

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_ROWS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_ROWS, help=NUM_HALF_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_COLUMNS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_COLUMNS, help=NUM_HALF_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=True,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BATCH_SIZE_ARG_NAME, type=int, required=False,
    default=NUM_EXAMPLES_PER_BATCH_DEFAULT, help=BATCH_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ARCHITECTURE_ID_ARG_NAME, type=int, required=True,
    help=ARCHITECTURE_ID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
    default=DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(num_rows_in_half_grid, num_columns_in_half_grid, pressure_level_mb,
         num_examples_per_batch, architecture_id, l2_weight, output_file_name):
    """Trains CNN with simple architecture for patch classification.

    This is effectively the main method.

    :param num_rows_in_half_grid: See documentation at top of file.
    :param num_columns_in_half_grid: Same.
    :param pressure_level_mb: Same.
    :param num_examples_per_batch: Same.
    :param architecture_id: Same.
    :param l2_weight: Same.
    :param output_file_name: Same.
    """

    training_start_time_unix_sec = time_conversion.string_to_unix_sec(
        TRAINING_START_TIME_STRING, INPUT_TIME_FORMAT)
    training_end_time_unix_sec = time_conversion.string_to_unix_sec(
        TRAINING_END_TIME_STRING, INPUT_TIME_FORMAT)

    validation_start_time_unix_sec = time_conversion.string_to_unix_sec(
        VALIDATION_START_TIME_STRING, INPUT_TIME_FORMAT)
    validation_end_time_unix_sec = time_conversion.string_to_unix_sec(
        VALIDATION_END_TIME_STRING, INPUT_TIME_FORMAT)

    print 'Reading NARR mask from: "{0:s}"...'.format(NARR_MASK_FILE_NAME)
    narr_mask_matrix = ml_utils.read_narr_mask(NARR_MASK_FILE_NAME)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_file_name, raise_error_if_missing=False)
    print 'Writing metadata to: "{0:s}"...'.format(model_metafile_name)

    traditional_cnn.write_model_metadata(
        pickle_file_name=model_metafile_name, num_epochs=NUM_EPOCHS,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_target_time=NUM_EXAMPLES_PER_TIME,
        num_training_batches_per_epoch=NUM_TRAINING_BATCHES_PER_EPOCH,
        num_validation_batches_per_epoch=NUM_VALIDATION_BATCHES_PER_EPOCH,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        dilation_distance_for_target_metres=DILATION_DISTANCE_METRES,
        class_fractions=CLASS_FRACTIONS,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        num_lead_time_steps=NUM_LEAD_TIME_STEPS,
        predictor_time_step_offsets=PREDICTOR_TIME_STEP_OFFSETS,
        narr_mask_matrix=narr_mask_matrix)

    num_rows_in_grid = 2 * num_rows_in_half_grid + 1
    num_columns_in_grid = 2 * num_columns_in_half_grid + 1
    num_channels = len(NARR_PREDICTOR_NAMES)

    if architecture_id == 1:
        model_object = cnn_architecture.get_first_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    elif architecture_id == 2:
        model_object = cnn_architecture.get_second_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    elif architecture_id == 3:
        model_object = cnn_architecture.get_third_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels, l2_weight=l2_weight)
    elif architecture_id == 4:
        model_object = cnn_architecture.get_fourth_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    elif architecture_id == 5:
        model_object = cnn_architecture.get_fifth_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    elif architecture_id == 6:
        model_object = cnn_architecture.get_sixth_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    elif architecture_id == 7:
        model_object = cnn_architecture.get_seventh_architecture(
            num_rows=num_rows_in_grid, num_columns=num_columns_in_grid,
            num_channels=num_channels)
    else:
        model_object = None

    traditional_cnn.train_with_3d_examples(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=NUM_EPOCHS,
        num_training_batches_per_epoch=NUM_TRAINING_BATCHES_PER_EPOCH,
        num_examples_per_target_time=NUM_EXAMPLES_PER_TIME,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_narr_directory_name=TOP_NARR_DIR_NAME,
        top_frontal_grid_dir_name=TOP_FRONTAL_GRID_DIR_NAME,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        dilation_distance_for_target_metres=DILATION_DISTANCE_METRES,
        class_fractions=CLASS_FRACTIONS,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        num_validation_batches_per_epoch=NUM_VALIDATION_BATCHES_PER_EPOCH,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec,
        narr_mask_matrix=narr_mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        num_rows_in_half_grid=getattr(
            INPUT_ARG_OBJECT, NUM_HALF_ROWS_ARG_NAME),
        num_columns_in_half_grid=getattr(
            INPUT_ARG_OBJECT, NUM_HALF_COLUMNS_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        num_examples_per_batch=getattr(INPUT_ARG_OBJECT, BATCH_SIZE_ARG_NAME),
        architecture_id=getattr(INPUT_ARG_OBJECT, ARCHITECTURE_ID_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, L2_WEIGHT_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
