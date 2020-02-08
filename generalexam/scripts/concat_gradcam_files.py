"""Concatenates files with Grad-CAM results for different examples."""

import copy
import argparse
import numpy
from generalexam.machine_learning import gradcam

METADATA_KEYS = [
    gradcam.MODEL_FILE_KEY, gradcam.TARGET_CLASS_KEY, gradcam.TARGET_LAYER_KEY
]
MAIN_DATA_KEYS = [
    gradcam.PREDICTOR_MATRIX_KEY, gradcam.ACTIVN_MATRIX_KEY,
    gradcam.GUIDED_ACTIVN_MATRIX_KEY, gradcam.EXAMPLE_IDS_KEY
]

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of files to concatenate.  Each should contain Grad-CAM results for '
    'different examples but with the same metadata otherwise.  These files will'
    ' be read by `gradcam.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `gradcam.write_standard_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, output_file_name):
    """Concatenates files with Grad-CAM results for different examples.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    gradcam_dict = None

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        new_gradcam_dict, new_pmm_flag = gradcam.read_file(this_file_name)

        assert not new_pmm_flag

        if gradcam_dict is None:
            gradcam_dict = copy.deepcopy(new_gradcam_dict)
            continue

        for this_key in METADATA_KEYS:
            assert gradcam_dict[this_key] == new_gradcam_dict[this_key]

        for this_key in MAIN_DATA_KEYS:
            if this_key == gradcam.EXAMPLE_IDS_KEY:
                gradcam_dict[this_key] += new_gradcam_dict[this_key]
            else:
                gradcam_dict[this_key] = numpy.concatenate((
                    gradcam_dict[this_key], new_gradcam_dict[this_key]
                ), axis=0)

    example_id_strings = gradcam_dict[gradcam.EXAMPLE_IDS_KEY]
    num_examples = len(example_id_strings)
    assert len(numpy.unique(numpy.array(example_id_strings))) == num_examples

    print((
        'Writing concatenated results (for {0:d} examples) to: "{1:s}"...'
    ).format(
        num_examples, output_file_name
    ))

    gradcam.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_predictor_matrix=gradcam_dict[gradcam.PREDICTOR_MATRIX_KEY],
        class_activn_matrix=gradcam_dict[gradcam.ACTIVN_MATRIX_KEY],
        guided_class_activn_matrix=
        gradcam_dict[gradcam.GUIDED_ACTIVN_MATRIX_KEY],
        example_id_strings=gradcam_dict[gradcam.EXAMPLE_IDS_KEY],
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        target_class=gradcam_dict[gradcam.TARGET_CLASS_KEY],
        target_layer_name=gradcam_dict[gradcam.TARGET_LAYER_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
