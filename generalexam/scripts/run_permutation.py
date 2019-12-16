"""Runs permutation test for predictor importance."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import permutation
from generalexam.machine_learning import correlation

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
NUM_EX_PER_TIME_ARG_NAME = 'num_examples_per_time'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN (will be read by `cnn.read_model`).')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be found by'
    ' `learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  The CNN will be applied to examples in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'The CNN will be applied to this many times chosen randomly from the period'
    ' `{0:s}`...`{1:s}`.  To use all times, leave this argument alone.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EX_PER_TIME_HELP_STRING = (
    'The CNN will be applied to this many examples chosen randomly at each '
    'time.  To use all examples, leave this argument alone.')

DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards test.  If 0, will run forward '
    'test.')

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates (used to compute the cost function after '
    'each permutation).  If you do not want bootstrapping, make this <= 1.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`permutation_utils.write_results`.')

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
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EX_PER_TIME_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EX_PER_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _read_examples(
        top_example_dir_name, first_time_string, last_time_string, num_times,
        num_examples_per_time, model_object, model_metadata_dict):
    """Reads examples for input to permutation test.

    E = number of examples
    M = number of rows in example grid
    N = number of columns in example grid
    C = number of predictors (channels)
    K = number of classes

    :param top_example_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`, corresponding to `model_object`.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: observed_labels: length-E numpy array of observed classes (integers
        in range 0...[K - 1]).
    """

    # Check input args.
    error_checking.assert_is_greater(num_times, 0)
    error_checking.assert_is_geq(num_examples_per_time, 10)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    # Find example files.
    example_file_names = examples_io.find_many_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_valid_time_unix_sec=first_time_unix_sec,
        last_valid_time_unix_sec=last_time_unix_sec)

    if len(example_file_names) > num_times:
        numpy.random.seed(RANDOM_SEED)
        these_indices = numpy.random.permutation(
            len(example_file_names)
        )[:num_times]

        example_file_names = [example_file_names[k] for k in these_indices]

    num_times = len(example_file_names)

    # Read examples.
    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    predictor_matrix = None
    target_matrix = None
    this_random_seed = RANDOM_SEED + 0

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))

        this_example_dict = examples_io.read_file(
            netcdf_file_name=example_file_names[i],
            predictor_names_to_keep=model_metadata_dict[
                cnn.PREDICTOR_NAMES_KEY
            ],
            pressure_levels_to_keep_mb=model_metadata_dict[
                cnn.PRESSURE_LEVELS_KEY
            ],
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns,
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec,
            normalization_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_KEY
            ]
        )

        this_num_examples = this_example_dict[
            examples_io.PREDICTOR_MATRIX_KEY].shape[0]

        if this_num_examples > num_examples_per_time:
            this_random_seed += 1
            numpy.random.seed(this_random_seed)

            these_indices = numpy.random.permutation(
                this_num_examples
            )[:num_examples_per_time]

            this_example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
                this_example_dict[examples_io.PREDICTOR_MATRIX_KEY][
                    these_indices, ...]
            )
            this_example_dict[examples_io.TARGET_MATRIX_KEY] = (
                this_example_dict[examples_io.TARGET_MATRIX_KEY][
                    these_indices, ...]
            )

        this_predictor_matrix = this_example_dict[
            examples_io.PREDICTOR_MATRIX_KEY]
        this_target_matrix = this_example_dict[
            examples_io.TARGET_MATRIX_KEY]

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
            target_matrix = this_target_matrix + 0
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=0
            )
            target_matrix = numpy.concatenate(
                (target_matrix, this_target_matrix), axis=0
            )

        num_examples_by_class = numpy.sum(target_matrix, axis=0)
        print('Number of examples in each class: {0:s}\n'.format(
            str(num_examples_by_class)
        ))

    return predictor_matrix, numpy.argmax(target_matrix, axis=1)


def _run(model_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_times, num_examples_per_time, do_backwards_test,
         num_bootstrap_reps, output_file_name):
    """Runs permutation test for predictor importance.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param num_examples_per_time: Same.
    :param do_backwards_test: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    predictor_matrix, observed_labels = _read_examples(
        top_example_dir_name=top_example_dir_name,
        first_time_string=first_time_string, last_time_string=last_time_string,
        num_times=num_times, num_examples_per_time=num_examples_per_time,
        model_object=model_object, model_metadata_dict=model_metadata_dict)
    print(SEPARATOR_STRING)

    correlation_matrix = correlation.get_pearson_correlations(predictor_matrix)
    nice_predictor_names = permutation.get_nice_predictor_names(
        predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    )

    num_predictors = len(nice_predictor_names)

    for i in range(num_predictors):
        for j in range(i, num_predictors):
            print((
                'Pearson correlation between "{0:s}" and "{1:s}" = {2:.3f}'
            ).format(
                nice_predictor_names[i], nice_predictor_names[j],
                correlation_matrix[i, j]
            ))

    print(SEPARATOR_STRING)

    if do_backwards_test:
        result_dict = permutation.run_backwards_test(
            model_object=model_object, predictor_matrix=predictor_matrix,
            observed_labels=observed_labels,
            model_metadata_dict=model_metadata_dict,
            cost_function=permutation.negative_auc_function,
            num_bootstrap_reps=num_bootstrap_reps)
    else:
        result_dict = permutation.run_forward_test(
            model_object=model_object, predictor_matrix=predictor_matrix,
            observed_labels=observed_labels,
            model_metadata_dict=model_metadata_dict,
            cost_function=permutation.negative_auc_function,
            num_bootstrap_reps=num_bootstrap_reps)

    print(SEPARATOR_STRING)

    result_dict[permutation_utils.MODEL_FILE_KEY] = model_file_name
    result_dict[permutation_utils.TARGET_VALUES_KEY] = observed_labels

    # TODO(thunderhoser): Maybe allow example IDs to be saved here as well.

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    permutation_utils.write_results(
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
            INPUT_ARG_OBJECT, NUM_EX_PER_TIME_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
