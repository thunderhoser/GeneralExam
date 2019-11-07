"""Applies trained CNN to pre-processed examples."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import ungridded_prediction_io
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression
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
USE_ISOTONIC_ARG_NAME = 'use_isotonic'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN (will be read by `cnn.read_model`).')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be found by'
    ' `learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  The CNN will applied to examples in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'The CNN will be applied to this many times chosen randomly from the period'
    ' `{0:s}`...`{1:s}`.  To choose all times, leave this argument alone.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, CNN predictions will be calibrated with isotonic '
    'regression.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written by '
    '`prediction_io.write_ungridded_predictions`, to a location therein '
    'determined by `prediction_io.find_ungridded_file`.')

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
    '--' + USE_ISOTONIC_ARG_NAME, type=int, required=False, default=0,
    help=USE_ISOTONIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _apply_cnn_one_time(
        example_file_name, model_object, model_metadata_dict,
        top_output_dir_name, top_example_dir_name, model_file_name,
        isotonic_object_by_class=None):
    """Applies trained CNN to examples at one time.

    K = number of classes

    :param example_file_name: Path to input file (will be read by
        `learning_examples_io.read_file`).
    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    :param top_output_dir_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param model_file_name: Same.
    :param isotonic_object_by_class: length-K list of isotonic-regression models
        (instances of `sklearn.isotonic.IsotonicRegression`).  If None, isotonic
        regression will not be done.
    """

    print('Reading data from: "{0:s}"...'.format(example_file_name))

    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=model_metadata_dict[
            cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_to_keep_mb=model_metadata_dict[
            cnn.PRESSURE_LEVELS_KEY],
        num_half_rows_to_keep=model_metadata_dict[cnn.NUM_HALF_ROWS_KEY],
        num_half_columns_to_keep=model_metadata_dict[
            cnn.NUM_HALF_COLUMNS_KEY]
    )

    example_id_strings = examples_io.create_example_ids(
        valid_times_unix_sec=example_dict[examples_io.VALID_TIMES_KEY],
        row_indices=example_dict[examples_io.ROW_INDICES_KEY],
        column_indices=example_dict[examples_io.COLUMN_INDICES_KEY]
    )

    observed_labels = numpy.argmax(
        example_dict[examples_io.TARGET_MATRIX_KEY], axis=1
    )

    num_examples = len(observed_labels)
    use_isotonic = isotonic_object_by_class is not None

    print('Applying CNN to {0:d} examples...'.format(num_examples))
    class_probability_matrix = model_object.predict(
        example_dict[examples_io.PREDICTOR_MATRIX_KEY], batch_size=num_examples
    )

    if use_isotonic:
        class_probability_matrix = (
            isotonic_regression.apply_model_for_each_class(
                orig_class_probability_matrix=class_probability_matrix,
                observed_labels=observed_labels,
                model_object_by_class=isotonic_object_by_class)
        )

    print(SEPARATOR_STRING)

    output_file_name = ungridded_prediction_io.find_file(
        top_directory_name=top_output_dir_name,
        valid_time_unix_sec=example_dict[examples_io.VALID_TIMES_KEY][0],
        raise_error_if_missing=False
    )

    print('Writing predictions to: "{0:s}"...'.format(output_file_name))

    ungridded_prediction_io.write_file(
        netcdf_file_name=output_file_name,
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        example_id_strings=example_id_strings,
        top_example_dir_name=top_example_dir_name,
        model_file_name=model_file_name, used_isotonic=use_isotonic)


def _run(model_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_times, use_isotonic, top_output_dir_name):
    """Applies trained CNN to pre-processed examples.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param use_isotonic: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if no examples are found in the given time period.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    if use_isotonic:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name, raise_error_if_missing=True)

        print('Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name
        ))
        isotonic_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_object_by_class = None

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
        _apply_cnn_one_time(
            example_file_name=this_file_name, model_object=model_object,
            model_metadata_dict=model_metadata_dict,
            top_output_dir_name=top_output_dir_name,
            top_example_dir_name=top_example_dir_name,
            model_file_name=model_file_name,
            isotonic_object_by_class=isotonic_object_by_class)

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        use_isotonic=bool(getattr(INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME)),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
