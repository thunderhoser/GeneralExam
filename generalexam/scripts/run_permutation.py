"""Runs permutation test for predictor importance.

--- NOTATION ---

The following letters will be used in this module.

E = number of examples
M = number of rows in grid
N = number of columns in grid
C = number of channels (predictors)
K = number of target classes
"""

# TODO(thunderhoser): Need to deal with predictor-pressure pairs.

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_EXAMPLES_PER_TIME_ARG_NAME = 'num_examples_per_time'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with learning examples.  Files therein will be found by '
    '`learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Valid times of learning examples will be '
    'randomly drawn from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of valid times (will be sampled randomly from `{0:s}`...`{1:s}`).'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_PER_TIME_HELP_STRING = (
    'Number of examples for each valid time.  The total number of examples used'
    ' will be `{0:s} * {1:s}`.'
).format(NUM_TIMES_ARG_NAME, NUM_EXAMPLES_PER_TIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`permutation_importance.write_results`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _prediction_function(model_object, predictor_matrix_in_list):
    """Prediction function.

    This function satisfies the requirements for the input `prediction_function`
    to `permutation.run_permutation_test`.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param predictor_matrix_in_list: List with only one item (predictor matrix
        as E-by-M-by-N-by-C numpy array).
    :return: class_probability_matrix: E-by-K numpy array, where
        class_probability_matrix[i, k] is the probability that the [i]th example
        belongs to the [k]th class.
    """

    return model_object.predict(
        predictor_matrix_in_list[0],
        batch_size=predictor_matrix_in_list[0].shape[0]
    )


def _read_examples(
        top_example_dir_name, first_time_string, last_time_string, num_times,
        num_examples_per_time, model_object, model_metadata_dict):
    """Reads learning examples.

    These and the trained model are the main inputs to the permutation test.

    :param top_example_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary with metadata for trained model
        (created by `cnn.read_metadata`).
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values
        (images).
    :return: target_values: length-E numpy array of target values (integer
        class labels).
    """

    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    error_checking.assert_is_greater(num_times, 0)
    error_checking.assert_is_geq(num_examples_per_time, 10)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    example_file_names = examples_io.find_many_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_valid_time_unix_sec=first_time_unix_sec,
        last_valid_time_unix_sec=last_time_unix_sec)

    example_file_names = numpy.array(example_file_names)
    numpy.random.seed(RANDOM_SEED)
    numpy.random.shuffle(example_file_names)

    num_times = min([num_times, len(example_file_names)])
    example_file_names = example_file_names[:num_times].tolist()

    predictor_matrix = None
    target_matrix = None
    this_random_seed = RANDOM_SEED + 0

    for i in range(num_times):
        print 'Reading data from: "{0:s}"...'.format(example_file_names[i])

        this_example_dict = examples_io.read_file(
            netcdf_file_name=example_file_names[i],
            predictor_names_to_keep=model_metadata_dict[
                cnn.PREDICTOR_NAMES_KEY],
            pressure_levels_to_keep_mb=model_metadata_dict[
                cnn.PRESSURE_LEVELS_KEY],
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns,
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec)

        this_num_examples_total = this_example_dict[
            examples_io.PREDICTOR_MATRIX_KEY].shape[0]
        this_num_examples_to_keep = min(
            [num_examples_per_time, this_num_examples_total]
        )

        these_example_indices = numpy.linspace(
            0, this_num_examples_total - 1, num=this_num_examples_total,
            dtype=int)

        this_random_seed += 1
        numpy.random.seed(this_random_seed)
        these_example_indices = numpy.random.choice(
            these_example_indices, size=this_num_examples_to_keep,
            replace=False)

        this_predictor_matrix = this_example_dict[
            examples_io.PREDICTOR_MATRIX_KEY][these_example_indices, ...]
        this_target_matrix = this_example_dict[
            examples_io.TARGET_MATRIX_KEY][these_example_indices, ...]

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
            target_matrix = this_target_matrix + 0
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=0)
            target_matrix = numpy.concatenate(
                (target_matrix, this_target_matrix), axis=0)

        num_examples_by_class = numpy.sum(target_matrix, axis=0)
        print 'Number of examples in each class: {0:s}\n'.format(
            str(num_examples_by_class))

    return predictor_matrix, numpy.argmax(target_matrix, axis=1)


def _run(model_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_times, num_examples_per_time, output_file_name):
    """Runs permutation test for predictor importance.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param output_file_name: Same.
    """

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print 'Reading model metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    print SEPARATOR_STRING
    predictor_matrix, target_values = _read_examples(
        top_example_dir_name=top_example_dir_name,
        first_time_string=first_time_string, last_time_string=last_time_string,
        num_times=num_times, num_examples_per_time=num_examples_per_time,
        model_object=model_object, model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    predictor_names = model_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    result_dict = permutation.run_permutation_test(
        model_object=model_object, list_of_input_matrices=[predictor_matrix],
        predictor_names_by_matrix=[predictor_names],
        target_values=target_values, prediction_function=_prediction_function,
        cost_function=permutation.cross_entropy_function)

    print SEPARATOR_STRING
    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    permutation.write_results(
        result_dict=result_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        num_examples_per_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
