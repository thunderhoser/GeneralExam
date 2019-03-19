"""Counts number of training examples."""

import argparse
from generalexam.machine_learning import learning_examples_io as examples_io

TOP_EXAMPLE_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples/shuffled/training')

FIRST_BATCH_NUM_ARG_NAME = 'first_batch_number'
LAST_BATCH_NUM_ARG_NAME = 'last_batch_number'

BATCH_NUM_HELP_STRING = (
    'Batch number.  This script will count training examples over batch numbers'
    ' `{0:s}`...`{1:s}`.'
).format(FIRST_BATCH_NUM_ARG_NAME, LAST_BATCH_NUM_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_BATCH_NUM_ARG_NAME, type=int, required=True,
    help=BATCH_NUM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_BATCH_NUM_ARG_NAME, type=int, required=True,
    help=BATCH_NUM_HELP_STRING)


def _run(first_batch_number, last_batch_number):
    """Counts number of training examples.

    This is effectively the main method.

    :param first_batch_number: See documentation at top of file.
    :param last_batch_number: Same.
    """

    example_file_names = examples_io.find_many_files(
        top_directory_name=TOP_EXAMPLE_DIR_NAME, shuffled=True,
        first_batch_number=first_batch_number,
        last_batch_number=last_batch_number)

    num_examples = 0

    for this_file_name in example_file_names:
        print 'Reading data from: "{0:s}"...'.format(this_file_name)

        this_example_dict = examples_io.read_file(
            netcdf_file_name=this_file_name, metadata_only=True)

        this_num_examples = len(this_example_dict[examples_io.VALID_TIMES_KEY])
        num_examples += this_num_examples

        print 'Number of examples so far = {0:d}\n'.format(num_examples)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_batch_number=getattr(INPUT_ARG_OBJECT, FIRST_BATCH_NUM_ARG_NAME),
        last_batch_number=getattr(INPUT_ARG_OBJECT, LAST_BATCH_NUM_ARG_NAME)
    )
