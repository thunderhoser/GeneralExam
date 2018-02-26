"""Trains convolutional neural net with MNIST architecture.

Said architecture was used to classify handwritten digits from the MNIST
(Modified National Institute of Standards and Technology) dataset.
"""

import argparse
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import processed_narr_io
from generalexam.machine_learning import traditional_cnn

# TODO(thunderhoser): Allow two versions of `positive_fraction`, one for
# training and one for validation?

# TODO(thunderhoser): Still need to extend downsized approach for future
# prediction.

# TODO(thunderhoser): Explore regularization and batch normalization.

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_TEMP_NAME]

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_ROWS_IN_HALF_GRID_ARG_NAME = 'num_rows_in_half_grid'
NUM_COLUMNS_IN_HALF_GRID_ARG_NAME = 'num_columns_in_half_grid'
DILATION_HALF_WIDTH_ARG_NAME = 'dilation_half_width_for_target'
POSITIVE_FRACTION_ARG_NAME = 'positive_fraction'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
TRAINING_START_TIME_ARG_NAME = 'training_start_time_string'
TRAINING_END_TIME_ARG_NAME = 'training_end_time_string'
VALIDATION_START_TIME_ARG_NAME = 'validation_start_time_string'
VALIDATION_END_TIME_ARG_NAME = 'validation_end_time_string'
TOP_NARR_DIR_ARG_NAME = 'top_narr_dir_name'
TOP_FRONTAL_GRID_DIR_ARG_NAME = 'top_frontal_grid_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples (downsized images) per batch.')
NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples (downsized images) sampled from each time step.')
NUM_TRAIN_BATCHES_PER_EPOCH_HELP_STRING = (
    'Number of training batches per epoch.')
NUM_VALIDN_BATCHES_PER_EPOCH_HELP_STRING = (
    'Number of validation batches per epoch.')
NUM_ROWS_IN_HALF_GRID_HELP_STRING = (
    'Number of rows in half-grid for each downsized image.  Total number of '
    'rows will be 2 * `{0:s}` + 1.').format(NUM_ROWS_IN_HALF_GRID_ARG_NAME)
NUM_COLUMNS_IN_HALF_GRID_HELP_STRING = (
    'Number of columns in half-grid for each downsized image.  Total number of '
    'columns will be 2 * `{0:s}` + 1.').format(
        NUM_COLUMNS_IN_HALF_GRID_ARG_NAME)
DILATION_HALF_WIDTH_HELP_STRING = (
    'Half-width of dilation window (number of pixels).  Target images will be '
    'dilated, which increases the number of pixels labeled as frontal.  This '
    'accounts for uncertainty in the placement of fronts.')
POSITIVE_FRACTION_HELP_STRING = (
    'Fraction of positive examples in both training and validation sets.  A '
    '"positive example" is an image with a front passing within `{0:s}` pixels '
    'of the center.').format(DILATION_HALF_WIDTH_ARG_NAME)
PRESSURE_LEVEL_HELP_STRING = (
    'NARR predictors will be taken from this pressure level (millibars).')
TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training examples will be taken randomly from'
    ' the time period `{0:s}`...`{1:s}`.').format(TRAINING_START_TIME_ARG_NAME,
                                                  TRAINING_END_TIME_ARG_NAME)
VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Validation examples will be taken randomly '
    'from the time period `{0:s}`...`{1:s}`.').format(
        VALIDATION_START_TIME_ARG_NAME, VALIDATION_END_TIME_ARG_NAME)
TOP_NARR_DIR_HELP_STRING = (
    'Name of top-level directory with NARR data (one file for each variable, '
    'pressure level, and time step).')
TOP_FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (one per file, indicating '
    'which NARR grid cells are intersected by a front).')
OUTPUT_FILE_HELP_STRING = 'Path to output file (HDF5 format) for trained model.'

DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EXAMPLES_PER_TIME = 256
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16
DEFAULT_NUM_ROWS_IN_HALF_GRID = 32
DEFAULT_NUM_COLUMNS_IN_HALF_GRID = 32
DEFAULT_DILATION_HALF_WIDTH_IN_PIXELS = 8
DEFAULT_POSITIVE_FRACTION = 0.1
DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_TOP_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_TOP_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_TOP_OUTPUT_DIR_NAME = (
    '/condo/swatwork/ralager/ml_models/downsized_3d_examples')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
    help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_TIME,
    help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
    help=NUM_TRAIN_BATCHES_PER_EPOCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
    help=NUM_VALIDN_BATCHES_PER_EPOCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_IN_HALF_GRID_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_ROWS_IN_HALF_GRID,
    help=NUM_ROWS_IN_HALF_GRID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_IN_HALF_GRID_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_COLUMNS_IN_HALF_GRID,
    help=NUM_COLUMNS_IN_HALF_GRID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_HALF_WIDTH_ARG_NAME, type=int, required=False,
    default=DEFAULT_DILATION_HALF_WIDTH_IN_PIXELS,
    help=DILATION_HALF_WIDTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POSITIVE_FRACTION_ARG_NAME, type=float, required=False,
    default=DEFAULT_POSITIVE_FRACTION, help=POSITIVE_FRACTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_START_TIME_ARG_NAME, type=str, required=True,
    help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_END_TIME_ARG_NAME, type=str, required=True,
    help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_START_TIME_ARG_NAME, type=str, required=True,
    help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_END_TIME_ARG_NAME, type=str, required=True,
    help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TOP_NARR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_NARR_DIR_NAME, help=TOP_NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TOP_FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_FRONTAL_GRID_DIR_NAME,
    help=TOP_FRONTAL_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _train_cnn(
        num_epochs, num_examples_per_batch, num_examples_per_time,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        num_rows_in_half_grid, num_columns_in_half_grid,
        dilation_half_width_for_target, positive_fraction, pressure_level_mb,
        training_start_time_string, training_end_time_string,
        validation_start_time_string, validation_end_time_string,
        top_narr_dir_name, top_frontal_grid_dir_name, output_file_name):
    """Trains convolutional neural net with MNIST architecture.

    :param num_epochs: Number of examples (downsized images) per batch.
    :param num_examples_per_batch: Number of examples (downsized images) sampled
        from each time step.
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
    :param dilation_half_width_for_target: Half-width of dilation window (number
        of pixels).  Target images will be dilated, which increases the number
        of pixels labeled as frontal.  This accounts for uncertainty in the
        placement of fronts.
    :param positive_fraction: Fraction of positive examples in both training and
        validation sets.  A "positive example" is an image with a front passing
        within `dilation_half_width_for_target` pixels of the center.
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
    model_object = traditional_cnn.get_cnn_with_mnist_architecture()
    print SEPARATOR_STRING

    traditional_cnn.train_model_from_on_the_fly_examples(
        model_object=model_object, output_file_name=output_file_name,
        num_examples_per_batch=num_examples_per_batch, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_examples_per_time=num_examples_per_time,
        training_start_time_unix_sec=training_start_time_unix_sec,
        training_end_time_unix_sec=training_end_time_unix_sec,
        top_narr_directory_name=top_narr_dir_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=pressure_level_mb,
        dilation_half_width_for_target=dilation_half_width_for_target,
        positive_fraction=positive_fraction,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_start_time_unix_sec=validation_start_time_unix_sec,
        validation_end_time_unix_sec=validation_end_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _train_cnn(
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_BATCH_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME),
        num_rows_in_half_grid=getattr(
            INPUT_ARG_OBJECT, NUM_ROWS_IN_HALF_GRID_ARG_NAME),
        num_columns_in_half_grid=getattr(
            INPUT_ARG_OBJECT, NUM_COLUMNS_IN_HALF_GRID_ARG_NAME),
        dilation_half_width_for_target=getattr(
            INPUT_ARG_OBJECT, DILATION_HALF_WIDTH_ARG_NAME),
        positive_fraction=getattr(INPUT_ARG_OBJECT, POSITIVE_FRACTION_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        training_start_time_string=getattr(
            INPUT_ARG_OBJECT, TRAINING_START_TIME_ARG_NAME),
        training_end_time_string=getattr(
            INPUT_ARG_OBJECT, TRAINING_END_TIME_ARG_NAME),
        validation_start_time_string=getattr(
            INPUT_ARG_OBJECT, VALIDATION_START_TIME_ARG_NAME),
        validation_end_time_string=getattr(
            INPUT_ARG_OBJECT, VALIDATION_END_TIME_ARG_NAME),
        top_narr_dir_name=getattr(INPUT_ARG_OBJECT, TOP_NARR_DIR_ARG_NAME),
        top_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, TOP_FRONTAL_GRID_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
