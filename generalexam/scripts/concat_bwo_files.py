"""Concatenates files with backwards-optimzn results for different examples."""

import copy
import argparse
import numpy
from gewittergefahr.deep_learning import \
    backwards_optimization as gg_backwards_opt
from generalexam.machine_learning import backwards_optimization as backwards_opt

METADATA_KEYS = [
    backwards_opt.MODEL_FILE_KEY, backwards_opt.NUM_ITERATIONS_KEY,
    backwards_opt.LEARNING_RATE_KEY, backwards_opt.L2_WEIGHT_KEY,
    backwards_opt.COMPONENT_TYPE_KEY, backwards_opt.TARGET_CLASS_KEY,
    backwards_opt.LAYER_NAME_KEY, backwards_opt.IDEAL_ACTIVATION_KEY,
    backwards_opt.NEURON_INDICES_KEY, backwards_opt.CHANNEL_INDEX_KEY
]
MAIN_DATA_KEYS = [
    backwards_opt.INPUT_MATRIX_KEY, backwards_opt.OUTPUT_MATRIX_KEY,
    backwards_opt.INITIAL_ACTIVATIONS_KEY, backwards_opt.FINAL_ACTIVATIONS_KEY,
    backwards_opt.EXAMPLE_IDS_KEY
]

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of files to concatenate.  Each should contain backwards-optimization '
    'results for different examples but with the same metadata otherwise.  '
    'These files will be read by `backwards_optimization.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`backwards_optimization.write_standard_file`.'
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
    """Concatenates files with backwards-optimzn results for different examples.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    bwo_dictionary = None

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        new_bwo_dictionary, new_pmm_flag = backwards_opt.read_file(
            this_file_name
        )

        assert not new_pmm_flag

        if bwo_dictionary is None:
            bwo_dictionary = copy.deepcopy(new_bwo_dictionary)
            continue

        for this_key in METADATA_KEYS:
            try:
                assert bwo_dictionary[this_key] == new_bwo_dictionary[this_key]
            except:
                if this_key == backwards_opt.NEURON_INDICES_KEY:
                    assert numpy.array_equal(
                        bwo_dictionary[this_key], new_bwo_dictionary[this_key]
                    )
                else:
                    assert numpy.isclose(
                        bwo_dictionary[this_key], new_bwo_dictionary[this_key],
                        atol=1e-6
                    )

        for this_key in MAIN_DATA_KEYS:
            if this_key == backwards_opt.EXAMPLE_IDS_KEY:
                bwo_dictionary[this_key] += new_bwo_dictionary[this_key]
            else:
                bwo_dictionary[this_key] = numpy.concatenate((
                    bwo_dictionary[this_key], new_bwo_dictionary[this_key]
                ), axis=0)

    example_id_strings = bwo_dictionary[backwards_opt.EXAMPLE_IDS_KEY]
    num_examples = len(example_id_strings)
    assert len(numpy.unique(numpy.array(example_id_strings))) == num_examples

    print((
        'Writing concatenated results (for {0:d} examples) to: "{1:s}"...'
    ).format(
        num_examples, output_file_name
    ))

    metadata_dict = gg_backwards_opt.check_metadata(
        component_type_string=bwo_dictionary[backwards_opt.COMPONENT_TYPE_KEY],
        num_iterations=bwo_dictionary[backwards_opt.NUM_ITERATIONS_KEY],
        learning_rate=bwo_dictionary[backwards_opt.LEARNING_RATE_KEY],
        target_class=bwo_dictionary[backwards_opt.TARGET_CLASS_KEY],
        layer_name=bwo_dictionary[backwards_opt.LAYER_NAME_KEY],
        ideal_activation=bwo_dictionary[backwards_opt.IDEAL_ACTIVATION_KEY],
        neuron_indices=bwo_dictionary[backwards_opt.NEURON_INDICES_KEY],
        channel_index=bwo_dictionary[backwards_opt.CHANNEL_INDEX_KEY],
        l2_weight=bwo_dictionary[backwards_opt.L2_WEIGHT_KEY]
    )

    backwards_opt.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_input_matrix=bwo_dictionary[backwards_opt.INPUT_MATRIX_KEY],
        denorm_output_matrix=bwo_dictionary[backwards_opt.OUTPUT_MATRIX_KEY],
        initial_activations=
        bwo_dictionary[backwards_opt.INITIAL_ACTIVATIONS_KEY],
        final_activations=bwo_dictionary[backwards_opt.FINAL_ACTIVATIONS_KEY],
        example_id_strings=bwo_dictionary[backwards_opt.EXAMPLE_IDS_KEY],
        model_file_name=bwo_dictionary[backwards_opt.MODEL_FILE_KEY],
        metadata_dict=metadata_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
