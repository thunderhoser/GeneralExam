"""Evaluates pixelwise probabilities created by a CNN.

In this case, evaluation is done in an pixelwise setting.  If you want to do
object-based evaluation, use evaluate_cnn_object_based.py.
"""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.machine_learning import evaluation_utils as eval_utils
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import model_evaluation_helper as model_eval_helper

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NAME_TO_CRITERION_FUNCTION_DICT = {
    'gerrity': eval_utils.get_gerrity_score,
    'csi': eval_utils.get_multiclass_csi
}

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
USE_ISOTONIC_ARG_NAME = 'use_isotonic_regression'
CRITERION_FUNCTION_ARG_NAME = 'criterion_function_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with learning examples (not temporally '
    'shuffled).  Files therein will be found by '
    '`learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will evaluate the CNN for times '
    'randomly drawn from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of times to draw randomly from the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for gridded warm-front and cold-front labels.  To use '
    'the same dilation distance used for training, leave this argument alone.')

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, will use isotonic regression to calibrate CNN '
    'probabilities.  If 0, will use raw CNN probabilities with no calibration.')

CRITERION_FUNCTION_HELP_STRING = (
    'Name of criterion function used to determine best binarization threshold.'
    '  Must be in the following list:\n{0:s}'
).format(str(NAME_TO_CRITERION_FUNCTION_DICT.keys()))

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be saved here.')

TOP_PREDICTOR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'
TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')

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
    '--' + USE_ISOTONIC_ARG_NAME, type=int, required=False, default=0,
    help=USE_ISOTONIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CRITERION_FUNCTION_ARG_NAME, type=str, required=False,
    default='gerrity', help=CRITERION_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _create_evaluation_pairs(
        model_object, model_metadata_dict, example_file_names,
        isotonic_model_object_by_class, first_time_unix_sec,
        last_time_unix_sec):
    """Creates evaluation pairs (prediction-observation pairs).

    E = number of evaluation pairs created

    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    :param example_file_names: 1-D list of paths to input files.
    :param isotonic_model_object_by_class: See doc for
        `isotonic_regression.read_model_for_each_class`.
    :param first_time_unix_sec: First time to use.
    :param last_time_unix_sec: Last time to use.
    :return: class_probability_matrix: E-by-3 numpy array of predicted
        probabilities.
    :return: observed_labels: length-E numpy array of observed labels (integers
        in 0...2).
    """

    class_probability_matrix = None
    observed_labels = numpy.array([], dtype=int)

    for this_file_name in example_file_names:
        print 'Reading data from: "{0:s}"...'.format(this_file_name)

        this_example_dict = examples_io.read_file(
            netcdf_file_name=this_file_name,
            predictor_names_to_keep=model_metadata_dict[
                cnn.PREDICTOR_NAMES_KEY],
            pressure_levels_to_keep_mb=model_metadata_dict[
                cnn.PRESSURE_LEVELS_KEY],
            num_half_rows_to_keep=model_metadata_dict[cnn.NUM_HALF_ROWS_KEY],
            num_half_columns_to_keep=model_metadata_dict[
                cnn.NUM_HALF_COLUMNS_KEY],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec
        )

        these_observed_labels = numpy.argmax(
            this_example_dict[examples_io.TARGET_MATRIX_KEY], axis=1
        )
        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels
        ))

        this_num_examples = len(these_observed_labels)
        print 'Applying CNN to {0:d} examples...\n'.format(this_num_examples)

        this_class_probability_matrix = model_object.predict(
            this_example_dict[examples_io.PREDICTOR_MATRIX_KEY],
            batch_size=this_num_examples)

        if class_probability_matrix is None:
            class_probability_matrix = this_class_probability_matrix + 0.
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_class_probability_matrix),
                axis=0
            )

    if isotonic_model_object_by_class is not None:
        class_probability_matrix = (
            isotonic_regression.apply_model_for_each_class(
                orig_class_probability_matrix=class_probability_matrix,
                observed_labels=observed_labels,
                model_object_by_class=isotonic_model_object_by_class)
        )

    return class_probability_matrix, observed_labels


def _run(model_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_times, use_isotonic_regression,
         criterion_function_name, output_dir_name):
    """Evaluates pixelwise probabilities created by a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param use_isotonic_regression: Same.
    :param criterion_function_name: Same.
    :param output_dir_name: Same.
    """

    criterion_function = NAME_TO_CRITERION_FUNCTION_DICT[
        criterion_function_name]

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True)

    print 'Reading model metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    if use_isotonic_regression:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name, raise_error_if_missing=True)

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_model_object_by_class = None

    print SEPARATOR_STRING

    example_file_names = examples_io.find_many_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_valid_time_unix_sec=first_time_unix_sec,
        last_valid_time_unix_sec=last_time_unix_sec)

    if len(example_file_names) > num_times:
        numpy.random.seed(RANDOM_SEED)
        example_file_names = numpy.array(example_file_names)
        numpy.random.shuffle(example_file_names)
        example_file_names = example_file_names[:num_times].tolist()

    class_probability_matrix, observed_labels = _create_evaluation_pairs(
        model_object=model_object, model_metadata_dict=model_metadata_dict,
        example_file_names=example_file_names,
        isotonic_model_object_by_class=isotonic_model_object_by_class,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    print SEPARATOR_STRING

    model_eval_helper.run_evaluation(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, criterion_function=criterion_function,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME
        )),
        criterion_function_name=getattr(
            INPUT_ARG_OBJECT, CRITERION_FUNCTION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
