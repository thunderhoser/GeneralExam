"""Randomly shuffles downsized 3-D examples among files."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import training_validation_io as trainval_io

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_EXAMPLES_PER_CHUNK_ARG_NAME = 'num_examples_per_chunk'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME = 'num_examples_per_out_file'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`training_validation_io.find_downsized_3d_example_file` and read by '
    '`training_validation_io.read_downsized_3d_examples`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Examples will be shuffled for all times from '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_PER_CHUNK_HELP_STRING = (
    'Examples will be shuffled in chunks of `{0:s}`.  The smaller the chunk '
    'size, the more examples are shuffled.  The larger the chunk size, the less'
    ' computing time the shuffling takes.'
).format(NUM_EXAMPLES_PER_CHUNK_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written by '
    '`training_validation_io.write_downsized_3d_examples` to locations therein,'
    ' determined by `training_validation_io.find_downsized_3d_example_file`.')

NUM_EXAMPLES_PER_OUT_FILE_HELP_STRING = (
    'Number of examples in each randomly shuffled output file.')

DEFAULT_NUM_EXAMPLES_PER_CHUNK = 8
DEFAULT_NUM_EXAMPLES_PER_OUT_FILE = 1024

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_CHUNK_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_CHUNK,
    help=NUM_EXAMPLES_PER_CHUNK_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_OUT_FILE,
    help=NUM_EXAMPLES_PER_OUT_FILE_HELP_STRING)


def _run(input_dir_name, first_time_string, last_time_string,
         num_examples_per_chunk, output_dir_name, num_examples_per_out_file):
    """Randomly shuffles downsized 3-D examples among files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_examples_per_chunk: Same.
    :param output_dir_name: Same.
    :param num_examples_per_out_file: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)
    error_checking.assert_is_geq(num_examples_per_chunk, 1)
    error_checking.assert_is_geq(num_examples_per_out_file, 100)

    # Find input (non-shuffled) files.
    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    input_file_names = trainval_io.find_downsized_3d_example_files(
        directory_name=input_dir_name, shuffled=False,
        first_target_time_unix_sec=first_time_unix_sec,
        last_target_time_unix_sec=last_time_unix_sec)

    # Determine total number of examples.
    num_input_files = len(input_file_names)
    num_examples_total = 0

    for i in range(num_input_files):
        print 'Reading data from: "{0:s}"...'.format(input_file_names[i])
        this_example_dict = trainval_io.read_downsized_3d_examples(
            netcdf_file_name=input_file_names[i],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec, metadata_only=True)

        num_examples_total += len(
            this_example_dict[trainval_io.TARGET_TIMES_KEY])

    print SEPARATOR_STRING

    # Determine number of output files.
    num_output_files = int(
        numpy.ceil(float(num_examples_total) / num_examples_per_out_file)
    )
    print (
        'Number of examples = {0:d} ... number of examples per output file = '
        '{1:d} ... number of output files = {2:d}'
    ).format(num_examples_total, num_examples_per_out_file, num_output_files)

    output_file_names = [
        trainval_io.find_downsized_3d_example_file(
            directory_name=output_dir_name, shuffled=True, batch_number=i,
            raise_error_if_missing=False
        ) for i in range(num_output_files)
    ]

    # # Shuffle examples among output files.
    # for i in range(num_input_files):
    #     print 'Reading data from: "{0:s}"...'.format(input_file_names[i])
    #     this_example_dict = trainval_io.read_downsized_3d_examples(
    #         netcdf_file_name=input_file_names[i],
    #         first_time_to_keep_unix_sec=first_time_unix_sec,
    #         last_time_to_keep_unix_sec=last_time_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_examples_per_chunk=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_CHUNK_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        num_examples_per_out_file=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME)
    )
