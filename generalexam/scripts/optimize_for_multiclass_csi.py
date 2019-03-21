"""Optimizes forecast-binarization threshold for multiclass CSI.

CSI = critical success index
"""

import argparse
from generalexam.machine_learning import evaluation_utils
from generalexam.scripts import model_evaluation_helper as model_eval_helper

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (containing evaluation results for one model).  Will be'
    ' read by `evaluation_utils.read_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output file.  New results (after optimizing for multiclass CSI) '
    'will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_eval_file_name, output_dir_name):
    """Optimizes forecast-binarization threshold for multiclass CSI.

    This is effectively the main method.

    :param input_eval_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(input_eval_file_name)
    input_evaluation_dict = evaluation_utils.read_file(input_eval_file_name)

    print SEPARATOR_STRING

    model_eval_helper.run_evaluation(
        class_probability_matrix=input_evaluation_dict[
            evaluation_utils.CLASS_PROBABILITY_MATRIX_KEY],
        observed_labels=input_evaluation_dict[
            evaluation_utils.OBSERVED_LABELS_KEY],
        criterion_function=evaluation_utils.get_multiclass_csi,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_eval_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
