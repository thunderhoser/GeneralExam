"""Runs sequential forward selection.

--- NOTATION ---

The following letters will be used in this module.

E = number of examples
M = number of rows in grid
N = number of columns in grid
C = number of channels (predictors)
"""

import random
import argparse
import numpy
from keras import backend as K
from keras.models import Model
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import sequential_selection
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import training_validation_io as trainval_io

# TODO(thunderhoser): Stopping criteria should be input args to the script.
# TODO(thunderhoser): Predictors and pressure level should also be input args to
# the script.

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_TRAINING_EXAMPLES_PER_BATCH = 512
NUM_EPOCHS = 10

ORIG_MODEL_FILE_ARG_NAME = 'orig_model_file_name'
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
NUM_TRAINING_TIMES_ARG_NAME = 'num_training_times'
NUM_EXAMPLES_PER_TTIME_ARG_NAME = 'num_ex_per_training_time'
VALIDN_DIR_ARG_NAME = 'input_validn_dir_name'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'
NUM_VALIDN_TIMES_ARG_NAME = 'num_validn_times'
NUM_EXAMPLES_PER_VTIME_ARG_NAME = 'num_ex_per_validn_time'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ORIG_MODEL_FILE_HELP_STRING = (
    'Path to file containing original CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.  At each step of sequential selection,'
    ' the architecture of the new CNN will be based on this original CNN.  The '
    'only difference is that the number of filters will be adjusted to account '
    'for different numbers of input channels (predictors).')

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training examples.  Files therein will be'
    ' found by `training_validation_io.find_downsized_3d_example_files` and '
    'read by `training_validation_io.read_downsized_3d_examples`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training times will be randomly drawn from '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_TRAINING_TIMES_HELP_STRING = (
    'Number of training times (will be sampled randomly from '
    '`{0:s}`...`{1:s}`).'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_EXAMPLES_PER_TTIME_HELP_STRING = (
    'Number of examples for each training time.  The total number of training '
    'examples will be `{0:s} * {1:s}`.'
).format(NUM_TRAINING_TIMES_ARG_NAME, NUM_EXAMPLES_PER_TTIME_ARG_NAME)

VALIDN_DIR_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    TRAINING_DIR_ARG_NAME)

FIRST_VALIDN_TIME_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    FIRST_TRAINING_TIME_ARG_NAME)

LAST_VALIDN_TIME_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    LAST_TRAINING_TIME_ARG_NAME)

NUM_VALIDN_TIMES_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    NUM_TRAINING_TIMES_ARG_NAME)

NUM_EXAMPLES_PER_VTIME_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    NUM_EXAMPLES_PER_TTIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`sequential_selection.write_results`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ORIG_MODEL_FILE_ARG_NAME, type=str, required=True,
    help=ORIG_MODEL_FILE_HELP_STRING)

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
    '--' + NUM_TRAINING_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TRAINING_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_TTIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_TTIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALIDN_DIR_ARG_NAME, type=str, required=True,
    help=VALIDN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDN_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_VALIDN_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDN_TIME_ARG_NAME, type=str, required=True,
    help=LAST_VALIDN_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDN_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_VALIDN_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_VTIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_VTIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _create_model_builder(orig_model_object):
    """Creates function (see below).

    :param orig_model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :return: model_builder: Function (see below).
    """

    num_channels_orig = (
        orig_model_object.layers[0].input.get_shape().as_list()[-1]
    )

    def model_builder(predictor_matrix_in_list):
        """Creates architecture for a new CNN.

        The new CNN architecture will be the same as the original (specified by
        `orig_model_object`), except that the number of filters in each
        layer will be multiplied by (num_channels_new / num_channels_orig).

        This function satisfies the requirements of the input `model_builder` to
        `sequential_selection.run_sfs`.

        :param predictor_matrix_in_list: List with only one item (predictor
            matrix as E-by-M-by-N-by-C numpy array).
        :return: model_object: Untrained instance of `keras.models.Model` or
            `keras.models.Sequential`.
        """

        num_channels_new = predictor_matrix_in_list[0].shape[-1]
        multiplier = float(num_channels_new) / num_channels_orig

        model_dict = orig_model_object.get_config()

        for this_layer_dict in model_dict['layers']:
            try:
                this_config_dict = this_layer_dict['config']
            except KeyError:
                this_config_dict = None

            if this_config_dict is None:
                continue

            try:
                this_config_dict['batch_input_shape'] = (
                    this_config_dict['batch_input_shape'][:-1] +
                    (num_channels_new,)
                )
            except KeyError:
                pass

            try:
                this_config_dict['filters'] = int(numpy.round(
                    multiplier * this_config_dict['filters']
                ))
            except KeyError:
                pass

            try:
                if this_config_dict['units'] > 3:
                    this_config_dict['units'] = int(numpy.round(
                        multiplier * this_config_dict['units']
                    ))
            except KeyError:
                pass

            this_layer_dict['config'] = this_config_dict

        model_object = Model.from_config(model_dict)
        model_object.summary()
        return model_object

    return model_builder


def _read_examples(top_example_dir_name, first_time_string, last_time_string,
                   num_times, num_examples_per_time, model_metadata_dict):
    """Reads learning examples for either training or validation.

    :param top_example_dir_name: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param first_time_string: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param last_time_string: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param num_times: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param num_examples_per_time: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param model_metadata_dict: Dictionary (created by
        `traditional_cnn.read_model_metadata`) for original model, whose
        architecture will be mostly copied to train the new models.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values
        (images).
    :return: target_values: length-E numpy array of target values (integer
        class labels).
    """

    error_checking.assert_is_greater(num_times, 0)
    error_checking.assert_is_geq(num_examples_per_time, 10)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    example_file_names = trainval_io.find_downsized_3d_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_target_time_unix_sec=first_time_unix_sec,
        last_target_time_unix_sec=last_time_unix_sec)

    num_times = min([num_times, len(example_file_names)])
    random.shuffle(example_file_names)
    example_file_names = example_file_names[:num_times]

    predictor_matrix = None
    target_matrix = None

    for i in range(num_times):
        print 'Reading data from: "{0:s}"...'.format(example_file_names[i])

        this_example_dict = trainval_io.read_downsized_3d_examples(
            netcdf_file_name=example_file_names[i],
            predictor_names_to_keep=model_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            num_half_rows_to_keep=model_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_half_columns_to_keep=model_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec)

        this_num_examples_total = this_example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY].shape[0]
        this_num_examples_to_keep = min(
            [num_examples_per_time, this_num_examples_total]
        )

        these_example_indices = numpy.linspace(
            0, this_num_examples_total - 1, num=this_num_examples_total,
            dtype=int)
        these_example_indices = numpy.random.choice(
            these_example_indices, size=this_num_examples_to_keep,
            replace=False)

        this_predictor_matrix = this_example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY][these_example_indices, ...]
        this_target_matrix = this_example_dict[
            trainval_io.TARGET_MATRIX_KEY][these_example_indices, ...]

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


def _run(orig_model_file_name, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         num_training_times, num_ex_per_training_time, top_validn_dir_name,
         first_validn_time_string, last_validn_time_string, num_validn_times,
         num_ex_per_validn_time, output_file_name):
    """Runs sequential forward selection.

    This is effectively the main method.

    :param orig_model_file_name: See documentation at top of file.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_training_times: Same.
    :param num_ex_per_training_time: Same.
    :param top_validn_dir_name: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param num_validn_times: Same.
    :param num_ex_per_validn_time: Same.
    :param output_file_name: Same.
    """

    print 'Reading original model from: "{0:s}"...'.format(orig_model_file_name)
    orig_model_object = traditional_cnn.read_keras_model(orig_model_file_name)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=orig_model_file_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metafile_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metafile_name)

    print SEPARATOR_STRING
    training_predictor_matrix, training_target_values = _read_examples(
        top_example_dir_name=top_training_dir_name,
        first_time_string=first_training_time_string,
        last_time_string=last_training_time_string,
        num_times=num_training_times,
        num_examples_per_time=num_ex_per_training_time,
        model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    validn_predictor_matrix, validn_target_values = _read_examples(
        top_example_dir_name=top_validn_dir_name,
        first_time_string=first_validn_time_string,
        last_time_string=last_validn_time_string,
        num_times=num_validn_times,
        num_examples_per_time=num_ex_per_validn_time,
        model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    narr_predictor_names = model_metadata_dict[
        traditional_cnn.NARR_PREDICTOR_NAMES_KEY]

    # TODO(thunderhoser): These things should be input args to the script.
    training_function = sequential_selection.create_training_function(
        num_training_examples_per_batch=NUM_TRAINING_EXAMPLES_PER_BATCH,
        num_epochs=NUM_EPOCHS)

    result_dict = sequential_selection.run_sfs(
        list_of_training_matrices=[training_predictor_matrix],
        training_target_values=training_target_values,
        list_of_validation_matrices=[validn_predictor_matrix],
        validation_target_values=validn_target_values,
        predictor_names_by_matrix=[narr_predictor_names],
        model_builder=_create_model_builder(orig_model_object),
        training_function=training_function,
        min_loss_decrease=0.01)
    print SEPARATOR_STRING

    result_dict.update({
        ORIG_MODEL_FILE_ARG_NAME: orig_model_file_name,
        TRAINING_DIR_ARG_NAME: top_training_dir_name,
        FIRST_TRAINING_TIME_ARG_NAME: first_training_time_string,
        LAST_TRAINING_TIME_ARG_NAME: last_training_time_string,
        NUM_TRAINING_TIMES_ARG_NAME: num_training_times,
        NUM_EXAMPLES_PER_TTIME_ARG_NAME: num_ex_per_training_time,
        VALIDN_DIR_ARG_NAME: top_validn_dir_name,
        FIRST_VALIDN_TIME_ARG_NAME: first_validn_time_string,
        LAST_VALIDN_TIME_ARG_NAME: last_validn_time_string,
        NUM_VALIDN_TIMES_ARG_NAME: num_validn_times,
        NUM_EXAMPLES_PER_VTIME_ARG_NAME: num_ex_per_validn_time
    })

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    sequential_selection.write_results(
        result_dict=result_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        orig_model_file_name=getattr(INPUT_ARG_OBJECT, ORIG_MODEL_FILE_ARG_NAME),
        top_training_dir_name=getattr(INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAINING_TIME_ARG_NAME),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TRAINING_TIME_ARG_NAME),
        num_training_times=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_TIMES_ARG_NAME),
        num_ex_per_training_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_TTIME_ARG_NAME),
        top_validn_dir_name=getattr(INPUT_ARG_OBJECT, VALIDN_DIR_ARG_NAME),
        first_validn_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDN_TIME_ARG_NAME),
        last_validn_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDN_TIME_ARG_NAME),
        num_validn_times=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDN_TIMES_ARG_NAME),
        num_ex_per_validn_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_VTIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
