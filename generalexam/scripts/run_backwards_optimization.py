"""Runs backwards optimization on a trained CNN."""

import random
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import \
    feature_optimization as backwards_opt
from gewittergefahr.deep_learning import model_interpretation
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import training_validation_io as trainval_io

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDEX_ARG_NAME = 'channel_index'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file, containing input examples for the CNN.  Will be read'
    ' by `training_validation_io.read_downsized_3d_examples`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to draw randomly from `{0:s}`.  Each example will be '
    'used as an initialization point.  If you want to select the examples, use '
    '`{1:s}` and leave this argument alone.'
).format(EXAMPLE_FILE_ARG_NAME, EXAMPLE_INDICES_ARG_NAME)

EXAMPLE_INDICES_HELP_STRING = (
    '[used only if `{0:s}` is left as default] Indices of examples to draw from'
    ' `{1:s}`.  Each example will be used as an initialization point.'
).format(NUM_EXAMPLES_ARG_NAME, EXAMPLE_FILE_ARG_NAME)

COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Images may be optimized for for one class probability, '
    'one neuron, or one channel.  Valid options are listed below.\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))

TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Images will be optimized for the '
    'probability of class k, where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer containing neuron '
    'or channel for which images will be optimized.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] See doc for '
    '`backwards_opt.optimize_input_for_neuron_activation` or '
    '`backwards_opt.optimize_input_for_channel_activation`.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Images will be optimized for neuron '
    'with the given indices.  For example, to optimize images for neuron '
    '(0, 0, 2), this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Images will be optimized for channel with '
    'the given index.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

NUM_ITERATIONS_HELP_STRING = 'Number of iterations for backwards optimization.'

LEARNING_RATE_HELP_STRING = 'Learning rate for backwards optimization.'

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `backwards_opt.write_file`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=EXAMPLE_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPONENT_TYPE_ARG_NAME, type=str, required=True,
    help=COMPONENT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=-1,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=LAYER_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=backwards_opt.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=CHANNEL_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=backwards_opt.DEFAULT_NUM_ITERATIONS,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=backwards_opt.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, example_file_name, num_examples, example_indices,
         component_type_string, target_class, layer_name, ideal_activation,
         neuron_indices, channel_index, num_iterations, learning_rate,
         output_file_name):
    """Runs backwards optimization on a trained CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_indices: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param output_file_name: Same.
    """

    if num_examples <= 0:
        num_examples = None

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = traditional_cnn.read_keras_model(model_file_name)

    model_metafile_name = traditional_cnn.find_metafile(
        model_file_name=model_file_name)

    print 'Reading model metadata from: "{0:s}"...'.format(
        model_metafile_name)
    model_metadata_dict = traditional_cnn.read_model_metadata(
        model_metafile_name)

    print 'Reading normalized examples from: "{0:s}"...'.format(
        example_file_name)
    example_dict = trainval_io.read_downsized_3d_examples(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=model_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        num_half_rows_to_keep=model_metadata_dict[
            traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
        num_half_columns_to_keep=model_metadata_dict[
            traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY])

    predictor_matrix = example_dict[trainval_io.PREDICTOR_MATRIX_KEY]

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
        num_examples = len(example_indices)
    else:
        error_checking.assert_is_greater(num_examples, 0)

        num_examples_total = predictor_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int)

        num_examples = min([num_examples, num_examples_total])
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False)

    predictor_matrix = predictor_matrix[example_indices, ...]
    optimized_predictor_matrix = numpy.full(predictor_matrix.shape, numpy.nan)
    print SEPARATOR_STRING

    for i in range(num_examples):
        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print (
                'Optimizing {0:d}th of {1:d} images for target class {2:d}...'
            ).format(i + 1, num_examples, target_class)

            optimized_predictor_matrix[i, ...] = (
                backwards_opt.optimize_input_for_class(
                    model_object=model_object, target_class=target_class,
                    init_function_or_matrices=[predictor_matrix[[i], ...]],
                    num_iterations=num_iterations, learning_rate=learning_rate
                )[0]
            )

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            print (
                'Optimizing {0:d}th of {1:d} images for neuron {2:s} in layer '
                '"{3:s}"...'
            ).format(i + 1, num_examples, str(neuron_indices), layer_name)

            optimized_predictor_matrix[i, ...] = (
                backwards_opt.optimize_input_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_indices,
                    init_function_or_matrices=[predictor_matrix[[i], ...]],
                    num_iterations=num_iterations, learning_rate=learning_rate,
                    ideal_activation=ideal_activation
                )[0]
            )

        else:
            print (
                'Optimizing {0:d}th of {1:d} images for channel {2:d} in layer '
                '"{3:s}"...'
            ).format(i + 1, num_examples, channel_index, layer_name)

            optimized_predictor_matrix[i, ...] = (
                backwards_opt.optimize_input_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=channel_index,
                    init_function_or_matrices=[predictor_matrix[[i], ...]],
                    stat_function_for_neuron_activations=K.max,
                    num_iterations=num_iterations, learning_rate=learning_rate,
                    ideal_activation=ideal_activation
                )[0]
            )

        print SEPARATOR_STRING

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    backwards_opt.write_file(
        pickle_file_name=output_file_name,
        list_of_optimized_input_matrices=[optimized_predictor_matrix],
        model_file_name=model_file_name,
        init_function_name_or_matrices=[predictor_matrix],
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=numpy.expand_dims(neuron_indices, axis=0),
        channel_indices=numpy.array([channel_index], dtype=int)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_index=getattr(INPUT_ARG_OBJECT, CHANNEL_INDEX_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
