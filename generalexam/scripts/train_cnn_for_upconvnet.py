"""Trains CNN for use with upconvnet.

In other words, the CNN will be the encoder and the upconvnet will be the
decoder.  However, the two are trained separately, and the upconvnet is
*not* trained by this script.

The CNN is trained to optimize front detection (specifically, to minimize
cross-entropy between the true labels {no front, warm front, cold front} and
predicted probabilities).

The upconvnet will be trained to optimize reconstruction of the original image
input to the CNN (specifically, to minimize pixelwise mean squared error).
"""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import cnn_architecture
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WEIGHT_LOSS_FUNCTION = False
NUM_EXAMPLES_PER_TIME_DUMMY = 8
CLASS_FRACTIONS_DUMMY = numpy.array([0.5, 0.25, 0.25])

NUM_HALF_ROWS_ARG_NAME = 'num_half_rows'
NUM_HALF_COLUMNS_ARG_NAME = 'num_half_columns'
PREDICTOR_NAMES_ARG_NAME = 'predictor_names'
PRESSURE_LEVELS_ARG_NAME = 'pressure_levels_mb'
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
FIRST_VALIDATION_TIME_ARG_NAME = 'first_validation_time_string'
LAST_VALIDATION_TIME_ARG_NAME = 'last_validation_time_string'
NUM_EX_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

NUM_HALF_ROWS_HELP_STRING = (
    'Number of rows in half-grid for each example.  Total number of rows = '
    '2 * `{0:s}` + 1.'
).format(NUM_HALF_ROWS_ARG_NAME)

NUM_HALF_COLUMNS_HELP_STRING = (
    'Number of columns in half-grid for each example.  Total number of columns '
    '= 2 * `{0:s}` + 1.'
).format(NUM_HALF_COLUMNS_ARG_NAME)

PREDICTOR_NAMES_HELP_STRING = (
    'List of predictor variables (channels).  Each must be accepted by '
    '`predictor_utils.check_field_name`.')

PRESSURE_LEVELS_HELP_STRING = (
    'List of pressure levels (millibars).  Must have same length as `{0:s}`.'
).format(PREDICTOR_NAMES_ARG_NAME)

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training data.  Files therein (containing'
    ' downsized 3-D examples, with 2 spatial dimensions) will be found by '
    '`learning_examples_io.find_file` (with shuffled = True) and read by '
    '`learning_examples_io.read_file`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for training.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for on-the-fly validation.'
).format(TRAINING_DIR_ARG_NAME)

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Only examples from the period '
    '`{0:s}`...`{1:s}` will be used for validation.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME)

NUM_EX_PER_BATCH_HELP_STRING = (
    'Number of examples in each training or validation batch.')

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'

NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches in each epoch.'

NUM_VALIDATION_BATCHES_HELP_STRING = (
    'Number of validation batches in each epoch.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (HDF5 format).  The trained CNN model will be saved '
    'here.')

DEFAULT_NUM_HALF_ROWS = 16
DEFAULT_NUM_HALF_COLUMNS = 16
DEFAULT_PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME,
    predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]

DEFAULT_PRESSURE_LEVELS_MB = numpy.full(
    len(DEFAULT_PREDICTOR_NAMES), 1000, dtype=int
)

DEFAULT_TOP_TRAINING_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/z_normalized/'
    'shuffled/training')
DEFAULT_FIRST_TRAINING_TIME_STRING = '2008110515'
DEFAULT_LAST_TRAINING_TIME_STRING = '2014122421'

DEFAULT_TOP_VALIDATION_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/z_normalized/'
    'shuffled/validation')
DEFAULT_FIRST_VALIDN_TIME_STRING = '2015010100'
DEFAULT_LAST_VALIDN_TIME_STRING = '2015122421'

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_ROWS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_ROWS, help=NUM_HALF_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HALF_COLUMNS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_HALF_COLUMNS, help=NUM_HALF_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVELS_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_PRESSURE_LEVELS_MB, help=PRESSURE_LEVELS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_TRAINING_DIR_NAME, help=TRAINING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_TRAINING_TIME_STRING, help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_VALIDATION_DIR_NAME, help=VALIDATION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_VALIDN_TIME_STRING, help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_VALIDN_TIME_STRING, help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EX_PER_BATCH_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_BATCH, help=NUM_EX_PER_BATCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EPOCHS, help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH,
    help=NUM_TRAINING_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH,
    help=NUM_VALIDATION_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(num_half_rows, num_half_columns, predictor_names, pressure_levels_mb,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, top_validation_dir_name,
         first_validation_time_string, last_validation_time_string,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains CNN for use with upconvnet.

    This is effectively the main method.

    :param num_half_rows: See documentation at top of file.
    :param num_half_columns: Same.
    :param predictor_names: Same.
    :param pressure_levels_mb: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param output_model_file_name: Same.
    """

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    # Read metadata from one training file.
    training_file_names = examples_io.find_many_files(
        top_directory_name=top_training_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=int(1e11)
    )

    print('Reading metadata from: "{0:s}"...'.format(training_file_names[0]))
    this_example_dict = examples_io.read_file(
        netcdf_file_name=training_file_names[0], metadata_only=True)

    normalization_type_string = this_example_dict[
        examples_io.NORMALIZATION_TYPE_KEY]
    dilation_distance_metres = this_example_dict[
        examples_io.DILATION_DISTANCE_KEY]
    mask_matrix = this_example_dict[examples_io.MASK_MATRIX_KEY]

    # Write metadata for CNN.
    output_metafile_name = cnn.find_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)
    print('Writing metadata to: "{0:s}"...'.format(output_metafile_name))

    cnn.write_metadata(
        pickle_file_name=output_metafile_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME_DUMMY,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        normalization_type_string=normalization_type_string,
        dilation_distance_metres=dilation_distance_metres,
        class_fractions=CLASS_FRACTIONS_DUMMY,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec,
        mask_matrix=mask_matrix)
    print(SEPARATOR_STRING)

    model_object = cnn_architecture.create_cnn(
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_channels=len(predictor_names)
    )
    print(SEPARATOR_STRING)

    training_generator = trainval_io.downsized_generator_from_example_files(
        top_input_dir_name=top_training_dir_name,
        first_time_unix_sec=first_training_time_unix_sec,
        last_time_unix_sec=last_training_time_unix_sec,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_classes=len(CLASS_FRACTIONS_DUMMY),
        num_examples_per_batch=num_examples_per_batch)

    validation_generator = trainval_io.downsized_generator_from_example_files(
        top_input_dir_name=top_validation_dir_name,
        first_time_unix_sec=first_validation_time_unix_sec,
        last_time_unix_sec=last_validation_time_unix_sec,
        predictor_names=predictor_names, pressure_levels_mb=pressure_levels_mb,
        num_half_rows=num_half_rows, num_half_columns=num_half_columns,
        num_classes=len(CLASS_FRACTIONS_DUMMY),
        num_examples_per_batch=num_examples_per_batch)

    cnn.train_cnn(
        model_object=model_object,
        output_model_file_name=output_model_file_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        training_generator=training_generator,
        validation_generator=validation_generator,
        weight_loss_function=WEIGHT_LOSS_FUNCTION,
        class_fractions=CLASS_FRACTIONS_DUMMY)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        num_half_rows=getattr(INPUT_ARG_OBJECT, NUM_HALF_ROWS_ARG_NAME),
        num_half_columns=getattr(INPUT_ARG_OBJECT, NUM_HALF_COLUMNS_ARG_NAME),
        predictor_names=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_NAMES_ARG_NAME),
        pressure_levels_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, PREDICTOR_NAMES_ARG_NAME), dtype=int
        ),
        top_training_dir_name=getattr(INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME),
        top_validation_dir_name=getattr(
            INPUT_ARG_OBJECT, VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDATION_TIME_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_EX_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME),
        num_validation_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDATION_BATCHES_ARG_NAME),
        output_model_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
