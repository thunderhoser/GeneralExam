"""Trains upconvolutional neural net."""

import argparse
import keras
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import cnn
from generalexam.machine_learning import upconvnet

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=7, inter_op_parallelism_threads=7
)))

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
NUM_EX_PER_TRAIN_ARG_NAME = 'num_ex_per_train_batch'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
FIRST_VALIDATION_TIME_ARG_NAME = 'first_validation_time_string'
LAST_VALIDATION_TIME_ARG_NAME = 'last_validation_time_string'
NUM_EX_PER_VALIDN_ARG_NAME = 'num_ex_per_validn_batch'
NUM_VALIDATION_BATCHES_ARG_NAME = 'num_validation_batches_per_epoch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

CNN_FILE_HELP_STRING = (
    'Path to trained CNN (used to convert original images to feature vectors).'
    '  Will be read by `cnn.read_model`.')

UPCONVNET_FILE_HELP_STRING = (
    'Path to untrained upconvnet (will be trained to convert feature vectors '
    'back to original images).  This file will also be read by '
    '`cnn.read_model`.')

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training files.  Shuffled files therein '
    'will be found by `learning_examples_io.find_file` and read by '
    '`learning_examples_io.read_file`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training period will be `{0:s}`...`{1:s}`.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_EX_PER_TRAIN_HELP_STRING = 'Number of examples per training batch.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'

VALIDATION_DIR_HELP_STRING = (
    'Name of top-level directory with validation files.  Shuffled files therein '
    'will be found by `learning_examples_io.find_file` and read by '
    '`learning_examples_io.read_file`.')

VALIDATION_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Validation period will be `{0:s}`...`{1:s}`.'
).format(FIRST_VALIDATION_TIME_ARG_NAME, LAST_VALIDATION_TIME_ARG_NAME)

NUM_EX_PER_VALIDN_HELP_STRING = 'Number of examples per validation batch.'
NUM_VALIDATION_BATCHES_HELP_STRING = 'Number of validation batches per epoch.'
NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Trained upconvnet will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAINING_TIME_ARG_NAME, type=str, required=True,
    help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAINING_TIME_ARG_NAME, type=str, required=True,
    help=TRAINING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EX_PER_TRAIN_ARG_NAME, type=int, required=False, default=1024,
    help=NUM_EX_PER_TRAIN_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False, default=32,
    help=NUM_TRAINING_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
    help=VALIDATION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDATION_TIME_ARG_NAME, type=str, required=True,
    help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDATION_TIME_ARG_NAME, type=str, required=True,
    help=VALIDATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EX_PER_VALIDN_ARG_NAME, type=int, required=False, default=1024,
    help=NUM_EX_PER_VALIDN_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDATION_BATCHES_ARG_NAME, type=int, required=False,
    default=16, help=NUM_VALIDATION_BATCHES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(cnn_file_name, upconvnet_file_name, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         num_ex_per_train_batch, num_training_batches_per_epoch,
         top_validation_dir_name, first_validation_time_string,
         last_validation_time_string, num_ex_per_validn_batch,
         num_validation_batches_per_epoch, num_epochs, output_dir_name):
    """Trains upconvolutional neural net.

    This is effectively the main method.

    :param cnn_file_name: See documentation at top of file.
    :param upconvnet_file_name: Same.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_ex_per_train_batch: Same.
    :param num_training_batches_per_epoch: Same.
    :param top_validation_dir_name: Same.
    :param first_validation_time_string: Same.
    :param last_validation_time_string: Same.
    :param num_ex_per_validn_batch: Same.
    :param num_validation_batches_per_epoch: Same.
    :param num_epochs: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT)
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT)

    first_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validation_time_string, TIME_FORMAT)
    last_validation_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validation_time_string, TIME_FORMAT)

    # Read trained CNN.
    print('Reading trained CNN from: "{0:s}"...'.format(cnn_file_name))
    cnn_model_object = cnn.read_model(cnn_file_name)
    cnn_feature_layer_name = cnn.get_flattening_layer(cnn_model_object)
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)

    # Read upconvnet architecture.
    print('Reading upconvnet architecture from: "{0:s}"...'.format(
        upconvnet_file_name
    ))
    ucn_model_object = cnn.read_model(upconvnet_file_name)
    # ucn_model_object = keras.models.clone_model(ucn_model_object)

    # TODO(thunderhoser): This is a HACK.
    ucn_model_object.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam()
    )

    print(SEPARATOR_STRING)
    ucn_model_object.summary()
    print(SEPARATOR_STRING)

    # Write upconvnet metadata to file.
    ucn_metafile_name = cnn.find_metafile(
        model_file_name='{0:s}/foo.h5'.format(output_dir_name),
        raise_error_if_missing=False
    )
    print('Writing upconvnet metadata to: "{0:s}"...'.format(ucn_metafile_name))

    upconvnet.write_model_metadata(
        pickle_file_name=ucn_metafile_name, cnn_model_file_name=cnn_file_name,
        cnn_feature_layer_name=cnn_feature_layer_name,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        num_ex_per_train_batch=num_ex_per_train_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec,
        num_ex_per_validn_batch=num_ex_per_validn_batch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_epochs=num_epochs)
    print(SEPARATOR_STRING)

    upconvnet.train_upconvnet(
        ucn_model_object=ucn_model_object, cnn_model_object=cnn_model_object,
        cnn_metadata_dict=cnn_metadata_dict,
        cnn_feature_layer_name=cnn_feature_layer_name,
        top_training_dir_name=top_training_dir_name,
        first_training_time_unix_sec=first_training_time_unix_sec,
        last_training_time_unix_sec=last_training_time_unix_sec,
        num_ex_per_train_batch=num_ex_per_train_batch,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        top_validation_dir_name=top_validation_dir_name,
        first_validation_time_unix_sec=first_validation_time_unix_sec,
        last_validation_time_unix_sec=last_validation_time_unix_sec,
        num_ex_per_validn_batch=num_ex_per_validn_batch,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        num_epochs=num_epochs, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        top_training_dir_name=getattr(INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME),
        first_training_time_string=
        getattr(INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=
        getattr(INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME),
        num_ex_per_train_batch=
        getattr(INPUT_ARG_OBJECT, NUM_EX_PER_TRAIN_ARG_NAME),
        num_training_batches_per_epoch=
        getattr(INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME),
        top_validation_dir_name=
        getattr(INPUT_ARG_OBJECT, VALIDATION_DIR_ARG_NAME),
        first_validation_time_string=
        getattr(INPUT_ARG_OBJECT, FIRST_VALIDATION_TIME_ARG_NAME),
        last_validation_time_string=
        getattr(INPUT_ARG_OBJECT, LAST_VALIDATION_TIME_ARG_NAME),
        num_ex_per_validn_batch=
        getattr(INPUT_ARG_OBJECT, NUM_EX_PER_VALIDN_ARG_NAME),
        num_validation_batches_per_epoch=
        getattr(INPUT_ARG_OBJECT, NUM_VALIDATION_BATCHES_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
