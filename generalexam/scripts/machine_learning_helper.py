"""Handles input args for machine-learning scripts."""

import numpy
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from generalexam.ge_io import processed_narr_io

USE_QUICK_GENERATOR_ARG_NAME = 'use_quick_generator'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_EXAMPLES_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
NUM_TRAIN_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_HALF_ROWS_ARG_NAME = 'num_rows_in_half_grid'
NUM_HALF_COLUMNS_ARG_NAME = 'num_columns_in_half_grid'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_for_target_metres'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'
CLASS_FRACTIONS_ARG_NAME = 'class_fractions'
NUM_CLASSES_ARG_NAME = 'num_classes'
NUM_LEAD_TIME_STEPS_ARG_NAME = 'num_lead_time_steps'
PREDICTOR_TIMES_ARG_NAME = 'predictor_time_step_offsets'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
NARR_PREDICTORS_ARG_NAME = 'narr_predictor_names'
TRAINING_START_TIME_ARG_NAME = 'training_start_time_string'
TRAINING_END_TIME_ARG_NAME = 'training_end_time_string'
VALIDN_START_TIME_ARG_NAME = 'validation_start_time_string'
VALIDN_END_TIME_ARG_NAME = 'validation_end_time_string'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
FRONTAL_GRID_DIR_ARG_NAME = 'input_frontal_grid_dir_name'
NARR_MASK_FILE_ARG_NAME = 'input_narr_mask_file_name'
NUM_CONV_LAYER_SETS_ARG_NAME = 'num_conv_layer_sets'
NUM_LAYERS_PER_SET_ARG_NAME = 'num_conv_layers_per_set'
POOLING_TYPE_ARG_NAME = 'pooling_type_string'
ACTIVATION_FUNCTION_ARG_NAME = 'conv_activation_function_string'
ALPHA_FOR_ELU_ARG_NAME = 'alpha_for_elu'
ALPHA_FOR_RELU_ARG_NAME = 'alpha_for_relu'
USE_BATCH_NORM_ARG_NAME = 'use_batch_normalization'
INIT_NUM_FILTERS_ARG_NAME = 'init_num_filters'
CONV_LAYER_DROPOUT_ARG_NAME = 'conv_layer_dropout_fraction'
DENSE_LAYER_DROPOUT_ARG_NAME = 'dense_layer_dropout_fraction'
L2_WEIGHT_ARG_NAME = 'l2_weight'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'
NUM_EXAMPLES_PER_BATCH_HELP_STRING = (
    'Number of examples (downsized images) in each batch.')
NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples (downsized images) for each target time.')
NUM_TRAIN_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDN_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

NUM_HALF_ROWS_HELP_STRING = (
    'Number of rows in half-grid for each downsized image.  Total number of '
    'rows will be 1 + 2 * `{0:s}`.'
).format(NUM_HALF_ROWS_ARG_NAME)

NUM_HALF_COLUMNS_HELP_STRING = (
    'Number of columns in half-grid for each downsized image.  Total number of '
    'columns will be 1 + 2 * `{0:s}`.'
).format(NUM_HALF_COLUMNS_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance.  Target images will be dilated, which increases the '
    'number of "frontal" pixels and accounts for uncertainty in frontal '
    'placement.')

WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 0, all classes will be weighted equally in the loss '
    'function.  If 1, class weights will be inversely proportional to the '
    'sampling fractions in `{0:s}`.'
).format(CLASS_FRACTIONS_ARG_NAME)

NUM_CLASSES_HELP_STRING = 'Number of classes.'

NUM_LEAD_TIME_STEPS_HELP_STRING = (
    'Number of time steps (3 hours each) between the target time and last '
    'possible predictor time.')

PREDICTOR_TIMES_HELP_STRING = (
    'List of offsets between the last possible predictor time and actual '
    'predictor times.  For example, if this is [0, 2, 4], predictors will be '
    'taken from [0, 6, 12] + 3 * `{0:s}` hours before the target time.'
).format(NUM_LEAD_TIME_STEPS_ARG_NAME)

PRESSURE_LEVEL_HELP_STRING = (
    'Predictors will be taken from this pressure level (millibars).')

NARR_PREDICTORS_HELP_STRING = (
    'Names of predictor fields.  Each must belong to the following list.\n{0:s}'
).format(str(processed_narr_io.FIELD_NAMES))

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Target times of training examples will be '
    'taken randomly from the period `{0:s}`...`{1:s}`.'
).format(TRAINING_START_TIME_ARG_NAME, TRAINING_END_TIME_ARG_NAME)

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Validation times of training examples will be '
    'taken randomly from the period `{0:s}`...`{1:s}`.'
).format(VALIDN_START_TIME_ARG_NAME, VALIDN_END_TIME_ARG_NAME)

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level NARR directory (predictors will be read from here).  '
    'Files therein will be found by `processed_narr_io.find_file_for_one_time` '
    'and read by `processed_narr_io.read_fields_from_file`.')

FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with frontal grids (target images will be read'
    ' from here).  Files therein will be found by '
    '`fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_narr_grids_from_file`.')

NARR_MASK_FILE_HELP_STRING = (
    'See doc for `machine_learning_utils.read_narr_mask`.  Determines which '
    'grid cells can be used as the center of a downsized grid.  If you do not '
    'want a mask, make this the empty string ("").')

NUM_CONV_LAYER_SETS_HELP_STRING = (
    'Number of sets of convolutional layers.  Each successive conv-layer set '
    'will halve the dimensions of the predictor images.  Conv layers in the '
    'same set will *not* affect the dimensions.')

NUM_LAYERS_PER_SET_HELP_STRING = 'Number of convolutional layers in each set.'

POOLING_TYPE_HELP_STRING = (
    'Pooling type.  Must belong to the following list.\n{0:s}'
).format(str(architecture_utils.VALID_POOLING_TYPES))

ACTIVATION_FUNCTION_HELP_STRING = (
    'Activation function (will be used for each convolutional layer).  Must '
    'belong to the following list.\n{0:s}'
).format(str(architecture_utils.VALID_CONV_LAYER_ACTIV_FUNC_STRINGS))

ALPHA_FOR_ELU_HELP_STRING = (
    'Slope for negative inputs to eLU (exponential linear unit) activation '
    'function.')

ALPHA_FOR_RELU_HELP_STRING = (
    'Slope for negative inputs to ReLU (rectified linear unit) activation '
    'function.')

USE_BATCH_NORM_HELP_STRING = (
    'Boolean flag.  If 1, the net will include a batch-normalization layer '
    'after each conv layer and each dense ("fully connected") layer.')

INIT_NUM_FILTERS_HELP_STRING = (
    'Initial number of filters (in the first conv-layer set).  Each successive '
    'conv-layer set will double the number of filters.')

CONV_LAYER_DROPOUT_HELP_STRING = (
    'Dropout fraction (will be applied to each conv layer).  If you want no '
    'dropout after conv layers, make this negative.')

DENSE_LAYER_DROPOUT_HELP_STRING = (
    'Dropout fraction (will be applied to each dense ["fully connected"] '
    'layer).  If you want no dropout after dense layers, make this negative.')

L2_WEIGHT_HELP_STRING = (
    'L2-regularization weight (will be applied to each conv layer).  If you '
    'want no L2 regularization, leave this alone.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format), which will contain the trained model.')

DEFAULT_NUM_EPOCHS = 25
DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EXAMPLES_PER_TIME = 256
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16
DEFAULT_NUM_HALF_ROWS = 16
DEFAULT_NUM_HALF_COLUMNS = 16
DEFAULT_DILATION_DISTANCE_METRES = 50000.
DEFAULT_CLASS_FRACTIONS = numpy.array([0.9, 0.05, 0.05])
DEFAULT_NUM_CLASSES = len(DEFAULT_CLASS_FRACTIONS)
DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.WET_BULB_THETA_NAME
]
DEFAULT_NARR_MASK_FILE_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p')
DEFAULT_NUM_CONV_LAYER_SETS = 3
DEFAULT_NUM_LAYERS_PER_SET = 1
DEFAULT_POOLING_TYPE_STRING = architecture_utils.MAX_POOLING_TYPE
DEFAULT_ACTIVATION_FUNCTION_STRING = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_ALPHA_FOR_ELU = architecture_utils.DEFAULT_ALPHA_FOR_ELU
DEFAULT_ALPHA_FOR_RELU = architecture_utils.DEFAULT_ALPHA_FOR_RELU
DEFAULT_USE_BATCH_NORM_FLAG = 0
DEFAULT_INIT_NUM_FILTERS = 16
DEFAULT_CONV_LAYER_DROPOUT_FRACTION = -1.
DEFAULT_DENSE_LAYER_DROPOUT_FRACTION = 0.25
DEFAULT_L2_WEIGHT = 0.001

TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
TOP_FRONTAL_GRID_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')


def add_input_arguments(argument_parser_object, use_downsized_examples,
                        use_quick_generator=False):
    """Adds input args for machine learning to the ArgumentParser object.

    :param argument_parser_object: Instance of `argparse.ArgumentParser`, which
        may alrwady contain input args.
    :param use_downsized_examples: Boolean flag.  If True, the model will be
        trained with downsized examples.  If False, with full-size examples.
    :param use_quick_generator: [used iff `use_downsized_examples == True`]
        Boolean flag.  If True, the model will be trained with
        `training_validation_io.quick_downsized_3d_example_gen`.  If False, will
        be trained with `training_validation_io.downsized_3d_example_generator`.
    :return: argument_parser_object: Same as input object, but containing more
        input args.
    """

    error_checking.assert_is_boolean(use_downsized_examples)
    if not use_downsized_examples:
        use_quick_generator = False
    error_checking.assert_is_boolean(use_quick_generator)

    argument_parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_EXAMPLES_PER_BATCH_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_EXAMPLES_PER_BATCH,
        help=NUM_EXAMPLES_PER_BATCH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_TRAIN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
        help=NUM_TRAIN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
        help=NUM_VALIDN_BATCHES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_CLASSES_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_CLASSES, help=NUM_CLASSES_HELP_STRING)

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
        '--' + VALIDN_START_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + VALIDN_END_TIME_ARG_NAME, type=str, required=True,
        help=VALIDATION_TIME_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_CONV_LAYER_SETS_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_CONV_LAYER_SETS,
        help=NUM_CONV_LAYER_SETS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_LAYERS_PER_SET_ARG_NAME, type=int, required=False,
        default=DEFAULT_NUM_LAYERS_PER_SET, help=NUM_LAYERS_PER_SET_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + POOLING_TYPE_ARG_NAME, type=str, required=False,
        default=DEFAULT_POOLING_TYPE_STRING, help=POOLING_TYPE_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ACTIVATION_FUNCTION_ARG_NAME, type=str, required=False,
        default=DEFAULT_ACTIVATION_FUNCTION_STRING,
        help=ACTIVATION_FUNCTION_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ALPHA_FOR_ELU_ARG_NAME, type=float, required=False,
        default=DEFAULT_ALPHA_FOR_ELU, help=ALPHA_FOR_ELU_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + ALPHA_FOR_RELU_ARG_NAME, type=float, required=False,
        default=DEFAULT_ALPHA_FOR_RELU, help=ALPHA_FOR_RELU_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + USE_BATCH_NORM_ARG_NAME, type=int, required=False,
        default=DEFAULT_USE_BATCH_NORM_FLAG, help=USE_BATCH_NORM_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + INIT_NUM_FILTERS_ARG_NAME, type=int, required=False,
        default=DEFAULT_INIT_NUM_FILTERS, help=INIT_NUM_FILTERS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + CONV_LAYER_DROPOUT_ARG_NAME, type=float, required=False,
        default=DEFAULT_CONV_LAYER_DROPOUT_FRACTION,
        help=CONV_LAYER_DROPOUT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + DENSE_LAYER_DROPOUT_ARG_NAME, type=float, required=False,
        default=DEFAULT_DENSE_LAYER_DROPOUT_FRACTION,
        help=DENSE_LAYER_DROPOUT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
        default=DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
        help=OUTPUT_FILE_HELP_STRING)

    if use_downsized_examples:
        argument_parser_object.add_argument(
            '--' + NUM_HALF_ROWS_ARG_NAME, type=int, required=False,
            default=DEFAULT_NUM_HALF_ROWS, help=NUM_HALF_ROWS_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + NUM_HALF_COLUMNS_ARG_NAME, type=int, required=False,
            default=DEFAULT_NUM_HALF_COLUMNS, help=NUM_HALF_COLUMNS_HELP_STRING)

        if not use_quick_generator:
            argument_parser_object.add_argument(
                '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=False,
                default=DEFAULT_NUM_EXAMPLES_PER_TIME,
                help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

            argument_parser_object.add_argument(
                '--' + NARR_MASK_FILE_ARG_NAME, type=str, required=False,
                default=DEFAULT_NARR_MASK_FILE_NAME,
                help=NARR_MASK_FILE_HELP_STRING)

        class_fractions_help_string = (
            'List of sampling fractions (one for each class).  Determines the '
            'proportion of samples for both training and validation.')

    else:
        class_fractions_help_string = (
            '[used iff {0:s} = 1] Assumed fraction of examples in each class, '
            'used to create weights for the loss function.'
        ).format(WEIGHT_LOSS_ARG_NAME)

    if not use_quick_generator:
        argument_parser_object.add_argument(
            '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
            default=DEFAULT_DILATION_DISTANCE_METRES,
            help=DILATION_DISTANCE_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
            default=1, help=WEIGHT_LOSS_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + NUM_LEAD_TIME_STEPS_ARG_NAME, type=int, required=False,
            default=-1, help=NUM_LEAD_TIME_STEPS_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + PREDICTOR_TIMES_ARG_NAME, type=int, nargs='+',
            required=False, default=[-1],
            help=PREDICTOR_TIMES_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
            default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
            default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIRECTORY_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + FRONTAL_GRID_DIR_ARG_NAME, type=str, required=False,
            default=TOP_FRONTAL_GRID_DIR_NAME_DEFAULT,
            help=FRONTAL_GRID_DIR_HELP_STRING)

        argument_parser_object.add_argument(
            '--' + CLASS_FRACTIONS_ARG_NAME, type=float, nargs='+',
            required=False, default=DEFAULT_CLASS_FRACTIONS,
            help=class_fractions_help_string)

    return argument_parser_object
