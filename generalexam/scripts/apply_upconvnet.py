"""Applies trained upconvnet to pre-processed examples."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import cnn
from generalexam.machine_learning import upconvnet
from generalexam.machine_learning import learning_examples_io as examples_io

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet (will be read by `cnn.read_model`).')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be found by'
    ' `learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  The upconvnet will be applied to examples '
    'from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'The upconvnet will be applied to this many times chosen randomly from the '
    'period `{0:s}`...`{1:s}`.  To choose all times, leave this argument alone.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Reconstructed images will be written by '
    '`upconvnet.write_predictions`, to locations therein determined by '
    '`upconvnet.find_prediction_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _apply_upconvnet_one_time(
        example_file_name, upconvnet_model_object, cnn_model_object,
        cnn_metadata_dict, cnn_feature_layer_name, upconvnet_file_name,
        top_output_dir_name):
    """Applies trained upconvnet to examples at one time.

    :param example_file_name: Path to input file (will be read by
        `learning_examples_io.read_file`).
    :param upconvnet_model_object: Trained upconvnet (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param cnn_model_object: Trained CNN (instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    :param cnn_feature_layer_name: Name of CNN layer whose output is the feature
        vector, which is the input to the upconvnet.
    :param upconvnet_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(example_file_name))

    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=cnn_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_to_keep_mb=cnn_metadata_dict[cnn.PRESSURE_LEVELS_KEY],
        num_half_rows_to_keep=cnn_metadata_dict[cnn.NUM_HALF_ROWS_KEY],
        num_half_columns_to_keep=cnn_metadata_dict[cnn.NUM_HALF_COLUMNS_KEY]
    )

    example_id_strings = examples_io.create_example_ids(
        valid_times_unix_sec=example_dict[examples_io.VALID_TIMES_KEY],
        row_indices=example_dict[examples_io.ROW_INDICES_KEY],
        column_indices=example_dict[examples_io.COLUMN_INDICES_KEY]
    )

    image_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]

    reconstructed_image_matrix = upconvnet.apply_upconvnet(
        image_matrix=image_matrix, cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name,
        ucn_model_object=upconvnet_model_object, verbose=True)

    mse_by_example = numpy.mean(
        (image_matrix - reconstructed_image_matrix) ** 2, axis=(1, 2, 3)
    )

    print('Mean sqaured error = {0:.3e}'.format(numpy.mean(mse_by_example)))

    output_file_name = upconvnet.find_prediction_file(
        top_directory_name=top_output_dir_name,
        valid_time_unix_sec=example_dict[examples_io.VALID_TIMES_KEY][0],
        raise_error_if_missing=False
    )

    print('Writing predictions to: "{0:s}"...'.format(output_file_name))

    upconvnet.write_predictions(
        netcdf_file_name=output_file_name,
        reconstructed_image_matrix=reconstructed_image_matrix,
        example_id_strings=example_id_strings,
        mse_by_example=mse_by_example, upconvnet_file_name=upconvnet_file_name)


def _run(upconvnet_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_times, top_output_dir_name):
    """Applies trained upconvnet to pre-processed examples.

    This is effectively the main method.

    :param upconvnet_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if no examples are found in the given time period.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    print('Reading upconvnet from: "{0:s}"...'.format(upconvnet_file_name))
    upconvnet_model_object = cnn.read_model(upconvnet_file_name)
    upconvnet_metafile_name = cnn.find_metafile(upconvnet_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        upconvnet_metafile_name
    ))
    upconvnet_metadata_dict = upconvnet.read_model_metadata(
        upconvnet_metafile_name
    )
    cnn_file_name = upconvnet_metadata_dict[upconvnet.CNN_FILE_KEY]

    print('Reading CNN from: "{0:s}"...'.format(cnn_file_name))
    cnn_model_object = cnn.read_model(cnn_file_name)
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)

    example_file_names = examples_io.find_many_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_valid_time_unix_sec=first_time_unix_sec,
        last_valid_time_unix_sec=last_time_unix_sec)

    if len(example_file_names) == 0:
        error_string = (
            'Cannot find any example files from {0:s} to {1:s} in directory: '
            '"{2:s}"'
        ).format(first_time_string, last_time_string, top_example_dir_name)

        raise ValueError(error_string)

    if len(example_file_names) > num_times:
        numpy.random.seed(RANDOM_SEED)
        example_file_names = numpy.array(example_file_names)
        numpy.random.shuffle(example_file_names)
        example_file_names = example_file_names[:num_times].tolist()

    print(SEPARATOR_STRING)

    for this_file_name in example_file_names:
        _apply_upconvnet_one_time(
            example_file_name=this_file_name,
            upconvnet_model_object=upconvnet_model_object,
            cnn_model_object=cnn_model_object,
            cnn_metadata_dict=cnn_metadata_dict,
            cnn_feature_layer_name=
            upconvnet_metadata_dict[upconvnet.CNN_FEATURE_LAYER_KEY],
            top_output_dir_name=top_output_dir_name,
            upconvnet_file_name=upconvnet_file_name
        )

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
