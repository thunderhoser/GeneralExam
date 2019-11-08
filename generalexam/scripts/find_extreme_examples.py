"""Finds extreme examples.

There are 12 types of extreme examples:

- actual cold fronts (CF) with highest CF probabilities
- actual CF with highest WF probabilities
- actual CF with highest NF probabilities
- actual WF with highest {CF, WF, NF} probabilities
- actual NF with highest {CF, WF, NF} probabilities
- overall examples with highest {CF, WF, NF} probabilities
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from generalexam.ge_io import ungridded_prediction_io
from generalexam.machine_learning import learning_examples_io as examples_io

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_set'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with ungridded predictions from the CNN.  '
    'Files therein will be found by `ungridded_prediction_io.find_file` and '
    'read by `ungridded_prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  All predictions in the period '
    '`{0:s}`...`{1:s}` will be read.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of examples in each of the 12 sets.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`learning_examples_io.write_example_ids`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_conditional_extremes(class_probability_matrix, observed_labels,
                              num_examples_per_set):
    """Finds conditional extremes.

    There are 9 sets of conditional extremes:

    - actual cold fronts (CF) with highest CF probabilities
    - actual CF with highest WF probabilities
    - actual CF with highest NF probabilities
    - actual WF with highest {CF, WF, NF} probabilities
    - actual NF with highest {CF, WF, NF} probabilities

    E = total number of examples
    K = number of classes
    e = number of examples in each set

    :param class_probability_matrix: E-by-K numpy array of class probabilities.
    :param observed_labels: length-E numpy array of observed classes (integers
        in range 0...[K - 1]).
    :param num_examples_per_set: Number of examples in each set.
    :return: extreme_index_matrix: K-by-K-by-e numpy array of indices.
        extreme_index_matrix[i, j, n] is the array index of the [n + 1]th
        example in the set where actual class = j and class with high
        prediction = i.

        More concretely, extreme_index_matrix[0, 1, 25] is the index of the
        26th WF example with a high NF probability.

        Missing examples are denoted by -1.
    """

    num_classes = class_probability_matrix.shape[1]
    extreme_index_matrix = numpy.full(
        (num_classes, num_classes, num_examples_per_set), -1, dtype=int
    )

    for j in range(num_classes):
        this_class_indices = numpy.where(observed_labels == j)[0]

        for i in range(num_classes):
            these_sort_indices = numpy.argsort(
                -1 * class_probability_matrix[this_class_indices, i]
            )
            these_sort_indices = these_sort_indices[:num_examples_per_set]
            these_sort_indices = this_class_indices[these_sort_indices]

            extreme_index_matrix[i, j, :len(these_sort_indices)] = (
                these_sort_indices
            )

    return extreme_index_matrix


def _get_unconditional_extremes(class_probability_matrix, num_examples_per_set):
    """Finds unconditional extremes.

    There are 3 sets of unconditional extremes:

    - examples with highest CF probabilities
    - highest WF probabilities
    - highest NF probabilities

    K = number of classes
    e = number of examples in each set

    :param class_probability_matrix: See doc for `_get_conditional_extremes`.
    :param num_examples_per_set: Same.
    :return: example_index_matrix: K-by-e numpy array of indices.
        extreme_index_matrix[k, n] is the array index of the [n + 1]th example
        with high probabilities for class k.

        More concretely, extreme_index_matrix[2, 50] is the index of the
        51st example with a high CF probability.

        Missing examples are denoted by -1.
    """

    num_classes = class_probability_matrix.shape[1]
    extreme_index_matrix = numpy.full(
        (num_classes, num_examples_per_set), -1, dtype=int
    )

    for k in range(num_classes):
        these_sort_indices = numpy.argsort(-1 * class_probability_matrix[:, k])
        these_sort_indices = these_sort_indices[:num_examples_per_set]
        extreme_index_matrix[k, :len(these_sort_indices)] = these_sort_indices

    return extreme_index_matrix


def _run(top_prediction_dir_name, first_time_string, last_time_string,
         num_examples_per_set, output_dir_name):
    """Finds extreme examples.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_examples_per_set: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    prediction_file_names = ungridded_prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    class_probability_matrix = None
    observed_labels = numpy.array([], dtype=int)
    example_id_strings = []

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = ungridded_prediction_io.read_file(this_file_name)

        example_id_strings += this_prediction_dict[
            ungridded_prediction_io.EXAMPLE_IDS_KEY]

        observed_labels = numpy.concatenate((
            observed_labels,
            this_prediction_dict[ungridded_prediction_io.OBSERVED_LABELS_KEY]
        ))

        if class_probability_matrix is None:
            class_probability_matrix = (
                this_prediction_dict[
                    ungridded_prediction_io.CLASS_PROBABILITIES_KEY] + 0.
            )
        else:
            class_probability_matrix = numpy.concatenate((
                class_probability_matrix,
                this_prediction_dict[
                    ungridded_prediction_io.CLASS_PROBABILITIES_KEY]
            ), axis=0)

    print(SEPARATOR_STRING)

    print('Finding conditional extremes...')
    example_index_matrix = _get_conditional_extremes(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        num_examples_per_set=num_examples_per_set)

    num_classes = class_probability_matrix.shape[1]

    for i in range(num_classes):
        for j in range(num_classes):
            these_indices = example_index_matrix[i, j, ...]
            these_indices = these_indices[these_indices >= 0]
            these_id_strings = [example_id_strings[k] for k in these_indices]

            print((
                'Found {0:d} extreme examples with actual class = {1:d} and '
                'high prediction (mean probability = {2:.3f}) for class {3:d}'
            ).format(
                len(these_id_strings), j,
                numpy.mean(class_probability_matrix[these_indices, i]), i
            ))

            this_output_file_name = (
                '{0:s}/extreme_examples_actual{1:d}_predicted{2:d}.nc'
            ).format(output_dir_name, j, i)

            print('Writing extreme-example IDs to: "{0:s}"...'.format(
                this_output_file_name
            ))

            examples_io.write_example_ids(
                netcdf_file_name=this_output_file_name,
                example_id_strings=these_id_strings)

    print('\nFinding unconditional extremes...')
    example_index_matrix = _get_unconditional_extremes(
        class_probability_matrix=class_probability_matrix,
        num_examples_per_set=num_examples_per_set)

    for k in range(num_classes):
        these_indices = example_index_matrix[k, ...]
        these_indices = these_indices[these_indices >= 0]
        these_id_strings = [example_id_strings[k] for k in these_indices]

        print((
            'Mean of {0:d} highest probabilities for class {1:d} = {2:.3f}'
        ).format(
            len(these_id_strings), k,
            numpy.mean(class_probability_matrix[these_indices, k])
        ))

        this_output_file_name = (
            '{0:s}/extreme_examples_predicted{1:d}.nc'
        ).format(output_dir_name, k)

        print('Writing extreme-example IDs to: "{0:s}"...'.format(
            this_output_file_name
        ))

        examples_io.write_example_ids(
            netcdf_file_name=this_output_file_name,
            example_id_strings=these_id_strings)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_examples_per_set=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
