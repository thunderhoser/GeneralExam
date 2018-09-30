"""Randomly shuffles downsized 3-D examples among files."""

import os
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
OUTPUT_DIR_ARG_NAME = 'top_output_dir_name'
FIRST_BATCH_NUM_ARG_NAME = 'first_batch_number'
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
    'Name of top-level output directory.  Files will be written by '
    '`training_validation_io.write_downsized_3d_examples` to locations therein,'
    ' determined by `training_validation_io.find_downsized_3d_example_file`.')

FIRST_BATCH_NUM_HELP_STRING = (
    'Batch number for first output file.  This is used only to create the file '
    'name.')

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
    '--' + FIRST_BATCH_NUM_ARG_NAME, type=int, required=True,
    help=FIRST_BATCH_NUM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_OUT_FILE,
    help=NUM_EXAMPLES_PER_OUT_FILE_HELP_STRING)


def _find_input_files(input_dir_name, first_time_unix_sec, last_time_unix_sec):
    """Finds input files.

    :param input_dir_name: See documentation at top of file.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: input_file_names: 1-D list of paths to input files.
    :return: num_examples_total: Number of examples among all input files.
    """

    input_file_names = trainval_io.find_downsized_3d_example_files(
        top_directory_name=input_dir_name, shuffled=False,
        first_target_time_unix_sec=first_time_unix_sec,
        last_target_time_unix_sec=last_time_unix_sec)

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

    return input_file_names, num_examples_total


def _set_output_locations(
        top_output_dir_name, num_examples_total, num_examples_per_out_file,
        first_batch_number):
    """Determines locations of output files.

    :param top_output_dir_name: See documentation at top of file.
    :param num_examples_total: Number of examples among all input files.
    :param num_examples_per_out_file: See documentation at top of file.
    :param first_batch_number: Same.
    :return: output_file_names: 1-D list of paths to output files.
    """

    num_output_files = int(
        numpy.ceil(float(num_examples_total) / num_examples_per_out_file)
    )

    print (
        'Number of examples = {0:d} ... number of examples per output file = '
        '{1:d} ... number of output files = {2:d}'
    ).format(num_examples_total, num_examples_per_out_file, num_output_files)

    output_file_names = [
        trainval_io.find_downsized_3d_example_file(
            top_directory_name=top_output_dir_name, shuffled=True,
            batch_number=first_batch_number + i, raise_error_if_missing=False
        ) for i in range(num_output_files)
    ]

    for this_file_name in output_file_names:
        if not os.path.isfile(this_file_name):
            continue
        print 'Deleting output file: "{0:s}"...'.format(this_file_name)
        os.remove(this_file_name)

    return output_file_names


def _shuffle_one_input_file(
        input_file_name, first_time_unix_sec, last_time_unix_sec,
        num_examples_per_chunk, output_file_names):
    """Shuffles examples in one input file.

    :param input_file_name: Path to input file.
    :param first_time_unix_sec: See documentation at top of file.
    :param last_time_unix_sec: Same.
    :param num_examples_per_chunk: Same.
    :param output_file_names: 1-D list of paths to output files.
    """

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    example_dict = trainval_io.read_downsized_3d_examples(
        netcdf_file_name=input_file_name,
        first_time_to_keep_unix_sec=first_time_unix_sec,
        last_time_to_keep_unix_sec=last_time_unix_sec)

    num_examples = len(example_dict[trainval_io.TARGET_TIMES_KEY])
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)
    numpy.random.shuffle(example_indices)
    for this_key in trainval_io.MAIN_KEYS:
        example_dict[this_key] = example_dict[this_key][example_indices, ...]

    num_output_files = len(output_file_names)
    output_file_indices = numpy.random.random_integers(
        low=0, high=num_output_files - 1, size=num_examples)

    for j in xrange(0, num_examples, num_examples_per_chunk):
        this_first_index = j
        this_last_index = min(
            [j + num_examples_per_chunk - 1, num_examples - 1]
        )
        these_example_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        this_example_dict = {}
        for this_key in trainval_io.MAIN_KEYS:
            this_example_dict.update({
                this_key: example_dict[this_key][these_example_indices, ...]
            })

        this_output_file_name = output_file_names[output_file_indices[j]]
        print 'Writing shuffled examples to: "{0:s}"...'.format(
            this_output_file_name)

        trainval_io.write_downsized_3d_examples(
            netcdf_file_name=this_output_file_name,
            example_dict=this_example_dict,
            narr_predictor_names=example_dict[trainval_io.PREDICTOR_NAMES_KEY],
            pressure_level_mb=example_dict[trainval_io.PRESSURE_LEVEL_KEY],
            dilation_distance_metres=example_dict[
                trainval_io.DILATION_DISTANCE_KEY],
            narr_mask_matrix=example_dict[trainval_io.NARR_MASK_KEY],
            append_to_file=os.path.isfile(this_output_file_name))


def _run(input_dir_name, first_time_string, last_time_string,
         num_examples_per_chunk, top_output_dir_name, first_batch_number,
         num_examples_per_out_file):
    """Randomly shuffles downsized 3-D examples among files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_examples_per_chunk: Same.
    :param top_output_dir_name: Same.
    :param first_batch_number: Same.
    :param num_examples_per_out_file: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=top_output_dir_name)
    error_checking.assert_is_geq(num_examples_per_chunk, 1)
    error_checking.assert_is_geq(num_examples_per_out_file, 100)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    input_file_names, num_examples_total = _find_input_files(
        input_dir_name=input_dir_name, first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)
    num_input_files = len(input_file_names)
    print SEPARATOR_STRING

    output_file_names = _set_output_locations(
        top_output_dir_name=top_output_dir_name,
        num_examples_total=num_examples_total,
        num_examples_per_out_file=num_examples_per_out_file,
        first_batch_number=first_batch_number)
    print SEPARATOR_STRING

    for i in range(num_input_files):
        _shuffle_one_input_file(
            input_file_name=input_file_names[i],
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec,
            num_examples_per_chunk=num_examples_per_chunk,
            output_file_names=output_file_names)
        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_examples_per_chunk=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_CHUNK_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        first_batch_number=getattr(INPUT_ARG_OBJECT, FIRST_BATCH_NUM_ARG_NAME),
        num_examples_per_out_file=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME)
    )
