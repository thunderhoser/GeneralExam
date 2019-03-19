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
from generalexam.machine_learning import learning_examples_io as examples_io

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

LARGE_INTEGER = int(1e10)
INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ORIG_MODEL_FILE_ARG_NAME = 'orig_model_file_name'
TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
FIRST_TRAINING_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAINING_TIME_ARG_NAME = 'last_training_time_string'
NUM_TRAINING_EXAMPLES_ARG_NAME = 'num_training_examples'
VALIDN_DIR_ARG_NAME = 'input_validn_dir_name'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'
NUM_VALIDN_EXAMPLES_ARG_NAME = 'num_validn_examples'
NARR_PREDICTORS_ARG_NAME = 'narr_predictor_names'
NUM_TRAIN_EX_PER_BATCH_ARG_NAME = 'num_training_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
MIN_LOSS_DECREASE_ARG_NAME = 'min_loss_decrease'
MIN_PERCENT_DECREASE_ARG_NAME = 'min_percentage_loss_decrease'
NUM_STEPS_FOR_DECREASE_ARG_NAME = 'num_steps_for_loss_decrease'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ORIG_MODEL_FILE_HELP_STRING = (
    'Path to file containing original CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.  At each step of sequential selection,'
    ' the architecture of the new CNN will be based on this original CNN.  The '
    'only difference is that the number of filters will be adjusted to account '
    'for different numbers of input channels (predictors).')

TRAINING_DIR_HELP_STRING = (
    'Name of top-level directory with training examples.  Files therein will be'
    ' found by `learning_examples_io.find_many_files` and read by '
    '`learning_examples_io.read_file`.')

TRAINING_TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Training times will be randomly drawn from '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_TRAINING_TIME_ARG_NAME, LAST_TRAINING_TIME_ARG_NAME)

NUM_TRAINING_EXAMPLES_HELP_STRING = (
    'Number of training examples (will be sampled randomly from `{0:s}`).'
).format(TRAINING_DIR_ARG_NAME)

VALIDN_DIR_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    TRAINING_DIR_ARG_NAME)

FIRST_VALIDN_TIME_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    FIRST_TRAINING_TIME_ARG_NAME)

LAST_VALIDN_TIME_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    LAST_TRAINING_TIME_ARG_NAME)

NUM_VALIDN_EXAMPLES_HELP_STRING = 'Same as `{0:s}` but for validation.'.format(
    NUM_TRAINING_EXAMPLES_ARG_NAME)

NARR_PREDICTORS_HELP_STRING = (
    'List of predictor variables to test.  Each must be accepted by '
    '`processed_narr_io.check_field_name`.  To test only predictors used in the'
    ' original model (represented by `{0:s}`), leave this argument alone.'
).format(ORIG_MODEL_FILE_ARG_NAME)

NUM_TRAIN_EX_PER_BATCH_HELP_STRING = 'Number of training examples per batch.'

NUM_EPOCHS_HELP_STRING = 'Number of training epochs.'

MIN_LOSS_DECREASE_HELP_STRING = (
    'Used to determine stopping criterion.  For details, see doc for '
    '`sequential_selection.run_sfs`.  If you want to use `{0:s}` instead, make '
    'this negative.'
).format(MIN_PERCENT_DECREASE_ARG_NAME)

MIN_PERCENT_DECREASE_HELP_STRING = (
    'Used to determine stopping criterion.  For details, see doc for '
    '`sequential_selection.run_sfs`.  If you want to use `{0:s}` instead, make '
    'this negative.'
).format(MIN_LOSS_DECREASE_ARG_NAME)

NUM_STEPS_FOR_DECREASE_HELP_STRING = (
    'Used to determine stopping criterion.  For details, see doc for '
    '`sequential_selection.run_sfs`.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`sequential_selection.write_results`.')

NUM_TRAINING_EXAMPLES_DEFAULT = 5120
NUM_VALIDN_EXAMPLES_DEFAULT = 5120
NUM_TRAIN_EX_PER_BATCH_DEFAULT = 256
NUM_EPOCHS_DEFAULT = 10
MIN_LOSS_DECREASE_DEFAULT = -1.
MIN_PERCENT_DECREASE_DEFAULT = 1.
NUM_STEPS_FOR_DECREASE_DEFAULT = 10

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
    '--' + NUM_TRAINING_EXAMPLES_ARG_NAME, type=int, required=False,
    default=NUM_TRAINING_EXAMPLES_DEFAULT,
    help=NUM_TRAINING_EXAMPLES_HELP_STRING)

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
    '--' + NUM_VALIDN_EXAMPLES_ARG_NAME, type=int, required=False,
    default=NUM_TRAINING_EXAMPLES_DEFAULT, help=NUM_VALIDN_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_PREDICTORS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=NARR_PREDICTORS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAIN_EX_PER_BATCH_ARG_NAME, type=int, required=False,
    default=NUM_TRAIN_EX_PER_BATCH_DEFAULT,
    help=NUM_VALIDN_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False,
    default=NUM_EPOCHS_DEFAULT, help=NUM_EPOCHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LOSS_DECREASE_ARG_NAME, type=float, required=False,
    default=MIN_LOSS_DECREASE_DEFAULT, help=MIN_LOSS_DECREASE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENT_DECREASE_ARG_NAME, type=float, required=False,
    default=MIN_PERCENT_DECREASE_DEFAULT, help=MIN_PERCENT_DECREASE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_STEPS_FOR_DECREASE_ARG_NAME, type=int, required=False,
    default=NUM_STEPS_FOR_DECREASE_DEFAULT,
    help=NUM_STEPS_FOR_DECREASE_HELP_STRING)

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
        model_object.compile(
            loss=orig_model_object.loss_functions,
            optimizer=orig_model_object.optimizer,
            metrics=orig_model_object.metrics)

        model_object.summary()
        return model_object

    return model_builder


def _read_examples(top_example_dir_name, first_time_string, last_time_string,
                   num_examples, model_metadata_dict):
    """Reads learning examples for either training or validation.

    :param top_example_dir_name: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param first_time_string: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param last_time_string: See doc for either `top_training_dir_name` or
        `top_validn_dir_name` at top of file.
    :param num_examples: See doc for either `num_training_examples` or
        `num_validn_examples` at top of file.
    :param model_metadata_dict: Dictionary (created by
        `traditional_cnn.read_model_metadata`) for original model, whose
        architecture will be mostly copied to train the new models.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values
        (images).
    :return: target_values: length-E numpy array of target values (integer
        class labels).
    """

    error_checking.assert_is_geq(num_examples, 100)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    example_file_names = examples_io.find_many_files(
        top_directory_name=top_example_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=LARGE_INTEGER)
    random.shuffle(example_file_names)

    predictor_matrix = None
    target_matrix = None

    for this_example_file_name in example_file_names:
        print 'Reading data from: "{0:s}"...'.format(this_example_file_name)

        this_example_dict = examples_io.read_file(
            netcdf_file_name=this_example_file_name,
            predictor_names_to_keep=model_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            num_half_rows_to_keep=model_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_half_columns_to_keep=model_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec)

        this_predictor_matrix = this_example_dict[
            examples_io.PREDICTOR_MATRIX_KEY]
        this_target_matrix = this_example_dict[
            examples_io.TARGET_MATRIX_KEY]

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
            target_matrix = this_target_matrix + 0
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=0)
            target_matrix = numpy.concatenate(
                (target_matrix, this_target_matrix), axis=0)

        if predictor_matrix.shape[0] > num_examples:
            predictor_matrix = predictor_matrix[:num_examples, ...]
            target_matrix = target_matrix[:num_examples, ...]

        num_examples_by_class = numpy.sum(target_matrix, axis=0)
        print 'Number of examples in each class: {0:s}\n'.format(
            str(num_examples_by_class))

        if predictor_matrix.shape[0] >= num_examples:
            break

    return predictor_matrix, numpy.argmax(target_matrix, axis=1)


def _run(orig_model_file_name, top_training_dir_name,
         first_training_time_string, last_training_time_string,
         num_training_examples, top_validn_dir_name, first_validn_time_string,
         last_validn_time_string, num_validn_examples, narr_predictor_names,
         num_training_examples_per_batch, num_epochs, min_loss_decrease,
         min_percentage_loss_decrease, num_steps_for_loss_decrease,
         output_file_name):
    """Runs sequential forward selection.

    This is effectively the main method.

    :param orig_model_file_name: See documentation at top of file.
    :param top_training_dir_name: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param num_training_examples: Same.
    :param top_validn_dir_name: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param num_validn_examples: Same.
    :param narr_predictor_names: Same.
    :param num_training_examples_per_batch: Same.
    :param num_epochs: Same.
    :param min_loss_decrease: Same.
    :param min_percentage_loss_decrease: Same.
    :param num_steps_for_loss_decrease: Same.
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
        num_examples=num_training_examples,
        model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    validn_predictor_matrix, validn_target_values = _read_examples(
        top_example_dir_name=top_validn_dir_name,
        first_time_string=first_validn_time_string,
        last_time_string=last_validn_time_string,
        num_examples=num_validn_examples,
        model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    # TODO(thunderhoser): I could make the code more efficient by making
    # `narr_predictor_names` an input arg to `_read_examples`.
    if narr_predictor_names[0] in ['', 'None']:
        narr_predictor_names = model_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY]

    training_function = sequential_selection.create_training_function(
        num_training_examples_per_batch=num_training_examples_per_batch,
        num_epochs=num_epochs)

    result_dict = sequential_selection.run_sfs(
        list_of_training_matrices=[training_predictor_matrix],
        training_target_values=training_target_values,
        list_of_validation_matrices=[validn_predictor_matrix],
        validation_target_values=validn_target_values,
        predictor_names_by_matrix=[narr_predictor_names],
        model_builder=_create_model_builder(orig_model_object),
        training_function=training_function,
        min_loss_decrease=min_loss_decrease,
        min_percentage_loss_decrease=min_percentage_loss_decrease,
        num_steps_for_loss_decrease=num_steps_for_loss_decrease)
    print SEPARATOR_STRING

    result_dict.update({
        ORIG_MODEL_FILE_ARG_NAME: orig_model_file_name,
        TRAINING_DIR_ARG_NAME: top_training_dir_name,
        FIRST_TRAINING_TIME_ARG_NAME: first_training_time_string,
        LAST_TRAINING_TIME_ARG_NAME: last_training_time_string,
        NUM_TRAINING_EXAMPLES_ARG_NAME: num_training_examples,
        VALIDN_DIR_ARG_NAME: top_validn_dir_name,
        FIRST_VALIDN_TIME_ARG_NAME: first_validn_time_string,
        LAST_VALIDN_TIME_ARG_NAME: last_validn_time_string,
        NUM_VALIDN_EXAMPLES_ARG_NAME: num_validn_examples
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
        num_training_examples=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_EXAMPLES_ARG_NAME),
        top_validn_dir_name=getattr(INPUT_ARG_OBJECT, VALIDN_DIR_ARG_NAME),
        first_validn_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDN_TIME_ARG_NAME),
        last_validn_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDN_TIME_ARG_NAME),
        num_validn_examples=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDN_EXAMPLES_ARG_NAME),
        narr_predictor_names=getattr(
            INPUT_ARG_OBJECT, NARR_PREDICTORS_ARG_NAME),
        num_training_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAIN_EX_PER_BATCH_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        min_loss_decrease=getattr(INPUT_ARG_OBJECT, MIN_LOSS_DECREASE_ARG_NAME),
        min_percentage_loss_decrease=getattr(
            INPUT_ARG_OBJECT, MIN_PERCENT_DECREASE_ARG_NAME),
        num_steps_for_loss_decrease=getattr(
            INPUT_ARG_OBJECT, NUM_STEPS_FOR_DECREASE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
