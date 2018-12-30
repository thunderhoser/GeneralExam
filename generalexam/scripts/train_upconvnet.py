"""Trains upconvnet."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import upconvnet

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UPSAMPLING_FACTORS = numpy.array([2, 1, 1, 2, 1, 1], dtype=int)

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
USE_BATCH_NORM_ARG_NAME = 'use_batch_norm_for_out_layer'
USE_TRANSPOSED_CONV_ARG_NAME = 'use_transposed_conv'
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

CNN_FILE_HELP_STRING = (
    'Path to file with trained CNN (will be read by '
    '`traditional_cnn.read_keras_model`).  Upconvnet predictors will be outputs'
    ' from the CNN''s flattening layer, and upconvnet targets will be CNN '
    'predictors (input images).')

USE_BATCH_NORM_HELP_STRING = (
    'Boolean flag.  If 1, will use batch normalization after output layer.')

USE_TRANSPOSED_CONV_HELP_STRING = (
    'Boolean flag.  If 1, upsampling will be done with transposed-convolution '
    'layers.  If False, each upsampling will be done with an upsampling layer '
    'followed by a conv layer.')

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training data.  Files therein (containing'
    ' downsized 3-D examples, with 2 spatial dimensions) will be found by '
    '`training_validation_io.find_downsized_3d_example_file` (with shuffled = '
    'True) and read by `training_validation_io.read_downsized_3d_examples`.')

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
    'Path to output file (HDF5 format).  The trained UCN model will be saved '
    'here.')

DEFAULT_USE_BATCH_NORM_FLAG = 1
DEFAULT_TRANSPOSED_CONV_FLAG = 0
DEFAULT_TOP_TRAINING_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/shuffled/training')
DEFAULT_FIRST_TRAINING_TIME_STRING = '2008110515'
DEFAULT_LAST_TRAINING_TIME_STRING = '2014122421'

DEFAULT_TOP_VALIDATION_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/shuffled/'
    'validation')
DEFAULT_FIRST_VALIDN_TIME_STRING = '2015010100'
DEFAULT_LAST_VALIDN_TIME_STRING = '2015122421'

DEFAULT_NUM_EXAMPLES_PER_BATCH = 1024
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_TRAINING_BATCHES_PER_EPOCH = 32
DEFAULT_NUM_VALIDATION_BATCHES_PER_EPOCH = 16

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_BATCH_NORM_ARG_NAME, type=int, required=False,
    default=DEFAULT_USE_BATCH_NORM_FLAG, help=USE_BATCH_NORM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_TRANSPOSED_CONV_ARG_NAME, type=int, required=False,
    default=DEFAULT_TRANSPOSED_CONV_FLAG, help=USE_TRANSPOSED_CONV_HELP_STRING)

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


def _run(input_cnn_file_name, use_batch_norm_for_out_layer, use_transposed_conv,
         top_training_dir_name, first_training_time_string,
         last_training_time_string, top_validation_dir_name,
         first_validation_time_string, last_validation_time_string,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validation_batches_per_epoch, output_model_file_name):
    """Trains upconvnet.

    This is effectively the main method.

    :param input_cnn_file_name: See documentation at top of file.
    :param use_batch_norm_for_out_layer: Same.
    :param use_transposed_conv: Same.
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

    print 'Reading trained CNN from: "{0:s}"...'.format(input_cnn_file_name)
    cnn_model_object = traditional_cnn.read_keras_model(input_cnn_file_name)

    cnn_metafile_name = traditional_cnn.find_metafile(
        model_file_name=input_cnn_file_name, raise_error_if_missing=True)

    print 'Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name)
    cnn_metadata_dict = traditional_cnn.read_model_metadata(
        cnn_metafile_name)

    cnn_feature_layer_name = traditional_cnn.get_flattening_layer(
        cnn_model_object)

    cnn_feature_layer_object = cnn_model_object.get_layer(
        name=cnn_feature_layer_name)
    cnn_feature_dimensions = numpy.array(
        cnn_feature_layer_object.input.shape[1:], dtype=int)

    num_input_features = numpy.prod(cnn_feature_dimensions)
    first_num_rows = cnn_feature_dimensions[0]
    first_num_columns = cnn_feature_dimensions[1]
    num_output_channels = numpy.array(
        cnn_model_object.input.shape[1:], dtype=int
    )[-1]

    ucn_metafile_name = traditional_cnn.find_metafile(
        model_file_name=output_model_file_name, raise_error_if_missing=False)

    print 'Writing upconvnet metadata to: "{0:s}"...'.format(ucn_metafile_name)
    upconvnet.write_model_metadata(
        pickle_file_name=ucn_metafile_name,
        top_training_dir_name=top_training_dir_name,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        cnn_model_file_name=input_cnn_file_name,
        cnn_feature_layer_name=cnn_feature_layer_name, num_epochs=num_epochs,
        num_examples_per_batch=num_examples_per_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        top_validation_dir_name=top_validation_dir_name,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec)
    print SEPARATOR_STRING

    ucn_model_object = upconvnet.create_net(
        num_input_features=num_input_features, first_num_rows=first_num_rows,
        first_num_columns=first_num_columns,
        upsampling_factors=UPSAMPLING_FACTORS,
        num_output_channels=num_output_channels,
        use_activation_for_out_layer=False,
        use_bn_for_out_layer=use_batch_norm_for_out_layer,
        use_transposed_conv=use_transposed_conv)
    print SEPARATOR_STRING

    upconvnet.train_upconvnet(
        ucn_model_object=ucn_model_object,
        top_training_dir_name=top_training_dir_name,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name,
        cnn_metadata_dict=cnn_metadata_dict,
        num_examples_per_batch=num_examples_per_batch,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        output_model_file_name=output_model_file_name,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        top_validation_dir_name=top_validation_dir_name,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        use_batch_norm_for_out_layer=bool(
            getattr(INPUT_ARG_OBJECT, USE_BATCH_NORM_ARG_NAME)),
        use_transposed_conv=bool(
            getattr(INPUT_ARG_OBJECT, USE_TRANSPOSED_CONV_ARG_NAME)),
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
