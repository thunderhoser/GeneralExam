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

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

MODEL_FILE_ARG_NAME = 'input_model_file_name'
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

MODEL_FILE_HELP_STRING = (
    'Path to file containing a trained CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.  At each step of sequential selection,'
    ' the model architecture will be based on this CNN.  The only architechure '
    'parameter that will change, is the number of filters produced by each '
    'layer.')

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

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

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


def _run(model_file_name, top_training_dir_name, first_training_time_string,
         last_training_time_string, num_training_times,
         num_ex_per_training_time, top_validn_dir_name,
         first_validn_time_string, last_validn_time_string, num_validn_times,
         num_ex_per_validn_time):
    """Runs sequential forward selection.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
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
    """

    # TODO(thunderhoser): Write this method.


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
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
    )
