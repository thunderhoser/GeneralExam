"""Helper methods for machine-learning scripts."""

from gewittergefahr.gg_utils import error_checking

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_ROWS_IN_HALF_GRID_ARG_NAME = 'num_rows_in_half_grid'
NUM_COLUMNS_IN_HALF_GRID_ARG_NAME = 'num_columns_in_half_grid'
DILATION_HALF_WIDTH_ARG_NAME = 'dilation_half_width_for_target'
POSITIVE_FRACTION_ARG_NAME = 'positive_fraction'
POSITIVE_CLASS_WEIGHT_ARG_NAME = 'positive_class_weight'
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
    'Number of examples per batch.')
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
POSITIVE_CLASS_WEIGHT_HELP_STRING = (
    'Weight for positive class in loss function.  This should be (1 - frequency'
    ' of positive class).')
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
DEFAULT_POSITIVE_CLASS_WEIGHT = 0.935
DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_TOP_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_TOP_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')


def add_input_arguments(argument_parser_object, use_downsized_examples):
    """Adds input args for machine-learning scripts to ArgumentParser object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :param use_downsized_examples: Boolean flag.  If True, machine-learning
        model will be trained with downsized examples (each covering only a
        portion of the NARR grid).  If False, model will be trained with
        examples over the full NARR grid.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    error_checking.assert_is_boolean(use_downsized_examples)

    argument_parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
        help=NUM_TRAIN_BATCHES_PER_EPOCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
        help=NUM_VALIDN_BATCHES_PER_EPOCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + DILATION_HALF_WIDTH_ARG_NAME, type=int, required=False,
        default=DEFAULT_DILATION_HALF_WIDTH_IN_PIXELS,
        help=DILATION_HALF_WIDTH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
        default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRAINING_START_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TRAINING_END_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDATION_START_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDATION_END_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TOP_NARR_DIR_ARG_NAME, type=str, required=False,
        default=DEFAULT_TOP_NARR_DIR_NAME, help=TOP_NARR_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + TOP_FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
        default=DEFAULT_TOP_FRONTAL_GRID_DIR_NAME,
        help=TOP_FRONTAL_GRID_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
        help=OUTPUT_FILE_HELP_STRING)

    if use_downsized_examples:
        argument_parser_object.add_argument(
            '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
            default=DEFAULT_NUM_EXAMPLES_PER_TIME,
            help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + NUM_ROWS_IN_HALF_GRID_ARG_NAME, type=int, required=False,
            default=DEFAULT_NUM_ROWS_IN_HALF_GRID,
            help=NUM_ROWS_IN_HALF_GRID_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + NUM_COLUMNS_IN_HALF_GRID_ARG_NAME, type=int, required=False,
            default=DEFAULT_NUM_COLUMNS_IN_HALF_GRID,
            help=NUM_COLUMNS_IN_HALF_GRID_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + POSITIVE_FRACTION_ARG_NAME, type=float, required=False,
            default=DEFAULT_POSITIVE_FRACTION,
            help=POSITIVE_FRACTION_HELP_STRING)

    else:
        argument_parser_object.add_argument(
            '--' + POSITIVE_CLASS_WEIGHT_ARG_NAME, type=float, required=False,
            default=DEFAULT_POSITIVE_CLASS_WEIGHT,
            help=POSITIVE_CLASS_WEIGHT_HELP_STRING)

    return argument_parser_object
