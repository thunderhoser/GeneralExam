"""Evaluates CNN-generated front probabilities pixel by pixel."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import ungridded_prediction_io
from generalexam.machine_learning import evaluation_utils as eval_utils
from generalexam.scripts import model_evaluation_helper as model_eval_helper

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CRITERION_NAME_TO_FUNCTION = {
    'gerrity': eval_utils.get_gerrity_score,
    'csi': eval_utils.get_multiclass_csi
}

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
CRITERION_FUNCTION_ARG_NAME = 'criterion_function_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with ungridded predictions from the CNN.  '
    'Files therein will be found by `ungridded_prediction_io.find_file` and '
    'read by `ungridded_prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  All predictions in the period '
    '`{0:s}`...`{1:s}` will be evaluated.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

CRITERION_FUNCTION_HELP_STRING = (
    'Criterion used to determine best binarization threshold (for converting '
    'probabilities to yes-or-no front predictions).  Must be in the following '
    'list:\n{0:s}'
).format(
    str(list(CRITERION_NAME_TO_FUNCTION.keys()))
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CRITERION_FUNCTION_ARG_NAME, type=str, required=False,
    default='gerrity', help=CRITERION_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_prediction_dir_name, first_time_string, last_time_string,
         criterion_function_name, output_dir_name):
    """Evaluates CNN-generated front probabilities pixel by pixel.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param criterion_function_name: Same.
    :param output_dir_name: Same.
    """

    criterion_function = CRITERION_NAME_TO_FUNCTION[criterion_function_name]

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

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = ungridded_prediction_io.read_file(this_file_name)

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

    model_eval_helper.run_evaluation(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels, criterion_function=criterion_function,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        criterion_function_name=getattr(
            INPUT_ARG_OBJECT, CRITERION_FUNCTION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
