"""Handles input args for training models."""

import numpy
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import predictor_utils

TIME_FORMAT = '%Y%m%d%H'

INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_NAMES_ARG_NAME = 'predictor_names'
PRESSURE_LEVELS_ARG_NAME = 'pressure_levels_mb'

X_TRANSLATIONS_ARG_NAME = 'x_translations_px'
Y_TRANSLATIONS_ARG_NAME = 'y_translations_px'
ROTATION_ANGLES_ARG_NAME = 'ccw_rotation_angles_deg'
NOISE_STDEV_ARG_NAME = 'noise_standard_deviation'
NUM_NOISINGS_ARG_NAME = 'num_noisings'

NORMALIZATION_FILE_ARG_NAME = 'normalization_file_name'

# The following args are used only when training from scratch (creating examples
# on the fly from raw data).
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
NUM_EX_PER_TIME_ARG_NAME = 'num_examples_per_time'
WEIGHT_LOSS_ARG_NAME = 'weight_loss_function'
DOWNSAMPLING_FRACTIONS_ARG_NAME = 'downsampling_fractions'

# The following args are used only when training from example files.
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'

FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
FIRST_VALIDATION_TIME_ARG_NAME = 'first_validation_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
LAST_VALIDATION_TIME_ARG_NAME = 'last_validation_time_string'
NUM_EX_PER_TRAIN_ARG_NAME = 'num_ex_per_train_batch'
NUM_EX_PER_VALIDN_ARG_NAME = 'num_ex_per_validn_batch'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_MODEL_FILE_HELP_STRING = (
    'Path to input file (containing either trained or untrained CNN).  Will be '
    'read by `traditional_cnn.read_keras_model`.  The architecture of this CNN '
    'will be copied.'
)
PREDICTOR_NAMES_HELP_STRING = (
    'List of predictor variables (channels).  Each must be accepted by '
    '`predictor_utils.check_field_name`.'
)
PRESSURE_LEVELS_HELP_STRING = (
    'List of pressure levels (millibars).  Must have same length as `{0:s}`.'
).format(PREDICTOR_NAMES_ARG_NAME)

X_TRANSLATIONS_HELP_STRING = (
    'x-translations for data augmentation (pixel units).  See doc for '
    '`training_validation_io._do_data_augmentation`.  If you do not want '
    'translation augmentation, leave this alone.'
)
Y_TRANSLATIONS_HELP_STRING = (
    'y-translations for data augmentation (pixel units).  See doc for '
    '`training_validation_io._do_data_augmentation`.  If you do not want '
    'translation augmentation, leave this alone.'
)
ROTATION_ANGLES_HELP_STRING = (
    'Counterclockwise rotation angles for data augmentation.  See doc for '
    '`training_validation_io._do_data_augmentation`.  If you do not want '
    'rotation augmentation, leave this alone.'
)
NOISE_STDEV_HELP_STRING = (
    'Standard deviation for Gaussian noise.  See doc for '
    '`training_validation_io._do_data_augmentation`.  If you do not want '
    'noising augmentation, leave this alone.'
)
NUM_NOISINGS_HELP_STRING = (
    'Number of times to replicate each example with noise.  See doc for '
    '`training_validation_io._do_data_augmentation`.  If you do not want '
    'noising augmentation, leave this alone.'
)

NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with global normalization params (readable by '
    '`predictor_io.read_normalization_params`).  If you want to use default '
    'normalization, leave this alone.'
)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for target variable (front label).  Each warm-frontal or'
    ' cold-frontal grid cell will be dilated by this amount.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.'
)
FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded front labels.  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.'
)
MASK_FILE_HELP_STRING = (
    'Path to mask file (determines which grid cells can be used as center of '
    'learning example).  Will be read by '
    '`machine_learning_utils.read_narr_mask`.  If you do not want a mask, leave'
    ' this empty.'
)
NUM_EX_PER_TIME_HELP_STRING = (
    'Average number of training examples for each target time.  This constraint'
    ' is applied to each batch separately.'
)
WEIGHT_LOSS_HELP_STRING = (
    'Boolean flag.  If 1, each class in the loss function will be weighted by '
    'the inverse of its frequency in training data.  If 0, no such weighting '
    'will be done.'
)
DOWNSAMPLING_FRACTIONS_HELP_STRING = (
    'List of downsampling fractions.  The [k]th value is the fraction for the '
    '[k]th class.  Fractions must add up to 1.0.  If you do not want '
    'downsampling, make this a one-item list.'
)

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training data.  Files therein (containing'
    ' downsized 3-D examples, with 2 spatial dimensions) will be found by '
    '`learning_examples_io.find_file` (with shuffled = True) and read by '
    '`learning_examples_io.read_file`.'
)
VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for on-the-fly validation.'
).format(TRAINING_DIR_ARG_NAME)

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for validation.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME)

NUM_EX_PER_TRAIN_HELP_STRING = 'Number of examples per training batch.'
NUM_EX_PER_VALIDN_HELP_STRING = 'Number of examples per validation batch.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format).  The trained CNN model will be saved '
    'here.'
)

DEFAULT_PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME,
    predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]
DEFAULT_PRESSURE_LEVELS_MB = numpy.full(
    len(DEFAULT_PREDICTOR_NAMES), 1000, dtype=int
)
DEFAULT_PREDICTOR_DIR_NAME = '/condo/swatwork/ralager/era5_data/processed'
DEFAULT_FRONT_DIR_NAME = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation'
)
DEFAULT_MASK_FILE_NAME = '/condo/swatwork/ralager/fronts_netcdf/era5_mask.p'


def add_input_args(argument_parser, use_downsized_files):
    """Adds input args to ArgumentParser object.

    :param argument_parser: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :param use_downsized_files: Boolean flag.  If True, the net will be trained
        with pre-processed files that contain downsized examples, readable by
        `learning_examples_io.read_file`.  If False, the net will be trained
        with examples created on the fly from raw predictors and gridded front
        labels.
    :return: argument_parser: Same as input but with new args added.
    """

    error_checking.assert_is_boolean(use_downsized_files)

    argument_parser.add_argument(
        '--' + INPUT_MODEL_FILE_ARG_NAME, type=str, required=True,
        help=INPUT_MODEL_FILE_HELP_STRING)

    argument_parser.add_argument(
        '--' + PREDICTOR_NAMES_ARG_NAME, type=str, nargs='+', required=False,
        default=DEFAULT_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

    argument_parser.add_argument(
        '--' + PRESSURE_LEVELS_ARG_NAME, type=int, nargs='+', required=False,
        default=DEFAULT_PRESSURE_LEVELS_MB, help=PRESSURE_LEVELS_HELP_STRING)

    argument_parser.add_argument(
        '--' + X_TRANSLATIONS_ARG_NAME, type=int, nargs='+', required=False,
        default=[0], help=X_TRANSLATIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + Y_TRANSLATIONS_ARG_NAME, type=int, nargs='+', required=False,
        default=[0], help=Y_TRANSLATIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + ROTATION_ANGLES_ARG_NAME, type=float, nargs='+', required=False,
        default=[0], help=ROTATION_ANGLES_HELP_STRING)

    argument_parser.add_argument(
        '--' + NOISE_STDEV_ARG_NAME, type=float, required=False,
        default=0.01, help=NOISE_STDEV_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_NOISINGS_ARG_NAME, type=int, required=False,
        default=0, help=NUM_NOISINGS_HELP_STRING)

    argument_parser.add_argument(
        '--' + NORMALIZATION_FILE_ARG_NAME, type=int, required=False,
        default='', help=NORMALIZATION_FILE_HELP_STRING)

    if use_downsized_files:
        argument_parser.add_argument(
            '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
            help=TRAINING_DIR_HELP_STRING)

        argument_parser.add_argument(
            '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
            help=VALIDATION_DIR_HELP_STRING)
    else:
        argument_parser.add_argument(
            '--' + DILATION_DISTANCE_ARG_NAME, type=int, required=False,
            default=50000, help=DILATION_DISTANCE_HELP_STRING)

        argument_parser.add_argument(
            '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=False,
            default=DEFAULT_PREDICTOR_DIR_NAME, help=PREDICTOR_DIR_HELP_STRING)

        argument_parser.add_argument(
            '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
            default=DEFAULT_FRONT_DIR_NAME, help=FRONT_DIR_HELP_STRING)

        argument_parser.add_argument(
            '--' + MASK_FILE_ARG_NAME, type=str, required=False,
            default=DEFAULT_MASK_FILE_NAME, help=MASK_FILE_HELP_STRING)

        argument_parser.add_argument(
            '--' + NUM_EX_PER_TIME_ARG_NAME, type=int, required=False,
            default=8, help=NUM_EX_PER_TIME_HELP_STRING)

        argument_parser.add_argument(
            '--' + WEIGHT_LOSS_ARG_NAME, type=int, required=False,
            default=0, help=WEIGHT_LOSS_HELP_STRING)

        argument_parser.add_argument(
            '--' + DOWNSAMPLING_FRACTIONS_ARG_NAME, type=float, nargs='+',
            required=False, default=[0.5, 0.25, 0.25],
            help=DOWNSAMPLING_FRACTIONS_HELP_STRING)

    argument_parser.add_argument(
        '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=True,
        help=TRAINING_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
        default='', help=VALIDATION_TIME_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_EX_PER_TRAIN_ARG_NAME, type=int, required=False,
        default=1024, help=NUM_EX_PER_TRAIN_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_EX_PER_VALIDN_ARG_NAME, type=int, required=False,
        default=1024, help=NUM_EX_PER_VALIDN_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING)

    argument_parser.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
        default=100, help=NUM_EPOCHS_HELP_STRING)

    argument_parser.add_argument(
        '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
        help=OUTPUT_FILE_HELP_STRING)

    return argument_parser
