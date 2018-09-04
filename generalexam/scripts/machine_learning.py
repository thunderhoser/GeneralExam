"""Helper methods for machine-learning scripts."""

import numpy
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io

NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NUM_TRAIN_BATCHES_PER_EPOCH_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_PER_EPOCH_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_ROWS_IN_HALF_GRID_ARG_NAME = 'num_rows_in_half_grid'
NUM_COLUMNS_IN_HALF_GRID_ARG_NAME = 'num_columns_in_half_grid'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_for_target_metres'
WEIGHT_LOSS_FUNCTION_ARG_NAME = 'weight_loss_function'
CLASS_FRACTIONS_ARG_NAME = 'class_fractions'
NUM_CLASSES_ARG_NAME = 'num_classes'
NUM_LEAD_TIME_STEPS_ARG_NAME = 'num_lead_time_steps'
PREDICTOR_TIME_STEP_OFFSETS_ARG_NAME = 'predictor_time_step_offsets'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
NARR_PREDICTORS_ARG_NAME = 'narr_predictor_names'
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
DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance.  Target images will be dilated, which increases the '
    'number of pixels labeled as frontal.  This accounts for uncertainty in the'
    ' placement of fronts.')
WEIGHT_LOSS_FUNCTION_HELP_STRING = (
    'Boolean flag.  If 1, classes will be weighted differently in loss function'
    ' (class weights inversely proportional to `{0:s}`).').format(
        CLASS_FRACTIONS_ARG_NAME)
NUM_CLASSES_HELP_STRING = 'Number of classes.'
NUM_LEAD_TIME_STEPS_HELP_STRING = (
    'Number of time steps (3 hours each) between target time and last possible '
    'predictor time.')
PREDICTOR_TIME_STEP_OFFSETS_HELP_STRING = (
    'List of offsets between last possible predictor time and actual predictor '
    'times.  For example, if this is [0, 2, 4], the model will be trained with '
    'predictor images from [0, 6, 12] + 3 * `{0:s}` hours before the target '
    'time.').format(NUM_LEAD_TIME_STEPS_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'NARR predictors will be taken from this pressure level (millibars).')
NARR_PREDICTORS_HELP_STRING = (
    'Names of NARR predictor variables (must be in the list '
    '`processed_narr_io.FIELD_NAMES`).')
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
DEFAULT_DILATION_DISTANCE_METRES = float(1e5)
DEFAULT_CLASS_FRACTIONS = numpy.array([0.9, 0.05, 0.05])
DEFAULT_NUM_CLASSES = len(DEFAULT_CLASS_FRACTIONS)
DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_THETA_NAME
]

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
        '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
        default=DEFAULT_DILATION_DISTANCE_METRES,
        help=DILATION_DISTANCE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + WEIGHT_LOSS_FUNCTION_ARG_NAME, type=int, required=False,
        default=1, help=WEIGHT_LOSS_FUNCTION_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_CLASSES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_CLASSES, help=NUM_CLASSES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_LEAD_TIME_STEPS_ARG_NAME, type=int, required=False,
        default=-1, help=NUM_LEAD_TIME_STEPS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PREDICTOR_TIME_STEP_OFFSETS_ARG_NAME, type=int, nargs='+',
        required=False, default=[-1],
        help=PREDICTOR_TIME_STEP_OFFSETS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
        default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NARR_PREDICTORS_ARG_NAME, type=str, nargs='+',
        required=False, default=DEFAULT_NARR_PREDICTOR_NAMES,
        help=NARR_PREDICTORS_HELP_STRING)

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

        class_fractions_help_string = (
            'Fraction of examples in each class.  Data will be sampled '
            'according to these fractions for both training and validation.')

    else:
        class_fractions_help_string = (
            'Assumed fraction of examples in each class.  These fractions will '
            'be used to create weights for the loss function.  Said weights '
            'will be inversely proportional to the fractions.')

    argument_parser_object.add_argument(
        '--' + CLASS_FRACTIONS_ARG_NAME, type=float, nargs='+',
        required=False, default=DEFAULT_CLASS_FRACTIONS,
        help=class_fractions_help_string)

    return argument_parser_object
