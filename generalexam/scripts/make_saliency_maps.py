"""Creates saliency map for each example, based on the same CNN."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as gg_saliency_maps
from gewittergefahr.deep_learning import model_interpretation
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import saliency_maps as ge_saliency_maps

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

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
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file, containing input examples for the CNN.  Will be read'
    ' by `learning_examples_io.read_file`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to draw randomly from `{0:s}`.  If you want to select '
    'the examples, use `{1:s}` and leave this argument alone.'
).format(EXAMPLE_FILE_ARG_NAME, EXAMPLE_INDICES_ARG_NAME)

EXAMPLE_INDICES_HELP_STRING = (
    '[used only if `{0:s}` is left as default] Indices of examples to draw from'
    ' `{1:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, EXAMPLE_FILE_ARG_NAME)

COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Saliency maps may be computed for one class probability, '
    'one neuron, or one channel.  Valid options are listed below.\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))

TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for the '
    'probability of class k, where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer containing neuron '
    'or channel for which saliency maps will be computed.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] See doc for '
    '`gg_saliency_maps.get_saliency_maps_for_neuron_activation` or '
    '`gg_saliency_maps.get_saliency_maps_for_channel_activation`.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for neuron '
    'with the given indices.  For example, to compute saliency maps for neuron '
    '(0, 0, 2), this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for channel '
    'with the given index.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `ge_saliency_maps.write_file`).')

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
    default=gg_saliency_maps.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=CHANNEL_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, example_file_name, num_examples, example_indices,
         component_type_string, target_class, layer_name, ideal_activation,
         neuron_indices, channel_index, output_file_name):
    """Creates saliency map for each example, based on the same CNN.

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
    :param output_file_name: Same.
    """

    if num_examples <= 0:
        num_examples = None

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
    else:
        error_checking.assert_is_greater(num_examples, 0)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    print('Reading normalized examples from: "{0:s}"...'.format(
        example_file_name))
    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_to_keep_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY],
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]
    if num_examples is not None:
        num_examples_total = predictor_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int)

        num_examples = min([num_examples, num_examples_total])

        numpy.random.seed(RANDOM_SEED)
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False)

    predictor_matrix = predictor_matrix[example_indices, ...]

    if component_type_string == CLASS_COMPONENT_TYPE_STRING:
        print('Computing saliency maps for target class {0:d}...'.format(
            target_class))

        saliency_matrix = (
            gg_saliency_maps.get_saliency_maps_for_class_activation(
                model_object=model_object, target_class=target_class,
                list_of_input_matrices=[predictor_matrix]
            )[0]
        )

    elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
        print((
            'Computing saliency maps for neuron {0:s} in layer "{1:s}"...'
        ).format(str(neuron_indices), layer_name))

        saliency_matrix = (
            gg_saliency_maps.get_saliency_maps_for_neuron_activation(
                model_object=model_object, layer_name=layer_name,
                neuron_indices=neuron_indices,
                list_of_input_matrices=[predictor_matrix],
                ideal_activation=ideal_activation
            )[0]
        )

    else:
        print((
            'Computing saliency maps for channel {0:d} in layer "{1:s}"...'
        ).format(channel_index, layer_name))

        saliency_matrix = (
            gg_saliency_maps.get_saliency_maps_for_channel_activation(
                model_object=model_object, layer_name=layer_name,
                channel_index=channel_index,
                list_of_input_matrices=[predictor_matrix],
                stat_function_for_neuron_activations=K.max,
                ideal_activation=ideal_activation
            )[0]
        )

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ge_saliency_maps.write_file(
        pickle_file_name=output_file_name,
        normalized_predictor_matrix=predictor_matrix,
        saliency_matrix=saliency_matrix, model_file_name=model_file_name,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_indices=neuron_indices, channel_index=channel_index)


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
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
