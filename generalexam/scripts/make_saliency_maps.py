"""Creates saliency map for each example and each CNN component.

CNN = convolutional neural network
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
import keras.models
from keras import backend as K
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import saliency_maps as gg_saliency_maps
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import saliency_maps as ge_saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CONV_LAYER_TYPE_STRINGS = ['Conv1D', 'Conv2D', 'Conv3D']
DENSE_LAYER_TYPE_STRINGS = ['Dense']

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
ID_FILE_ARG_NAME = 'input_example_id_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDEX_ARG_NAME = 'channel_index'
RANDOMIZE_ARG_NAME = 'randomize_weights'
CASCADING_ARG_NAME = 'cascading_random'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `learning_examples_io.read_file`).  '
    'Will use examples in this file.  If you want to use specific examples, '
    'leave this argument alone and specify `{0:s}` and `{1:s}`, instead.'
).format(EXAMPLE_DIR_ARG_NAME, ID_FILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    '[used only if `{0:s}` is unspecified] Name of top-level directory with '
    'examples to use.  Examples will be read from here by `'
    'learning_examples_io.read_specific_examples_many_files`.'
).format(EXAMPLE_FILE_ARG_NAME)

ID_FILE_HELP_STRING = (
    '[used only if `{0:s}` is unspecified] Path to file with IDs of examples to'
    ' use.  This file will be read by `learning_examples_io.read_example_ids`, '
    'and examples themselves will be read from `{1:s}`.'
).format(EXAMPLE_FILE_ARG_NAME, EXAMPLE_DIR_ARG_NAME)

COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Saliency maps may be computed for one class, one/many '
    'neurons, or one/many channels.  Valid options are listed below.\n{0:s}'
).format(
    str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS)
)

TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for class k,'
    ' where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels for which saliency maps will be computed.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CHANNEL_COMPONENT_TYPE_STRING, IDEAL_ACTIVATION_ARG_NAME)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices of neuron whose saliency map is to '
    'be computed.  For example, to compute saliency maps for neuron (0, 0, 2), '
    'this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index of channel whose saliency map is to '
    'be computed.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

RANDOMIZE_HELP_STRING = (
    'Boolean flag.  If 1, will randomize weights in each convolutional and '
    'dense layer before producing saliency maps.  This allows the '
    'model-parameter-randomization test from Adebayo et al. (2018) to be '
    'carried out.')

CASCADING_HELP_STRING = (
    '[used only if `{0:s}` = 1] Boolean flag.  If 1, will randomize weights in '
    'a cascading manner, going from the deepest to shallowest layer.  In this '
    'case, when weights for layer L are randomized, weights for all deeper '
    'layers are randomized as well.  If 0, will do non-cascading randomization,'
    ' where weights for only one layer are randomized at a time.'
).format(RANDOMIZE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`saliency_maps.write_standard_file` in GeneralExam library).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ID_FILE_ARG_NAME, type=str, required=False, default='',
    help=ID_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPONENT_TYPE_ARG_NAME, type=str, required=True,
    help=COMPONENT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=1,
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
    '--' + RANDOMIZE_ARG_NAME, type=int, required=False, default=0,
    help=RANDOMIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CASCADING_ARG_NAME, type=int, required=False, default=0,
    help=CASCADING_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _find_conv_and_dense_layers(model_object):
    """Finds convolutional and dense layers in model object.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :return: layer_names: 1-D list with names of convolutional and dense layers.
    """

    layer_names = [l.name for l in model_object.layers]
    layer_type_strings = [type(l).__name__ for l in model_object.layers]

    conv_or_dense_flags = numpy.array([
        t in CONV_LAYER_TYPE_STRINGS + DENSE_LAYER_TYPE_STRINGS
        for t in layer_type_strings
    ], dtype=bool)

    conv_or_dense_indices = numpy.where(conv_or_dense_flags)[0]
    return [layer_names[k] for k in conv_or_dense_indices]


def _reset_weights_in_layer(model_object, layer_name):
    """Resets (or "reinitializes" or "randomizes") weights in one layer.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Name of layer in which to reset weights.
    """

    session_object = K.get_session()
    layer_object = model_object.get_layer(name=layer_name)
    layer_object.kernel.initializer.run(session=session_object)


def _run(model_file_name, example_file_name, top_example_dir_name,
         example_id_file_name, component_type_string, target_class, layer_name,
         ideal_activation, neuron_indices, channel_index, randomize_weights,
         cascading_random, output_file_name):
    """Creates saliency map for each example and each CNN component.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param top_example_dir_name: Same.
    :param example_id_file_name: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param randomize_weights: Same.
    :param cascading_random: Same.
    :param output_file_name: Same.
    """

    if example_file_name in ['', 'None']:
        example_file_name = None
    else:
        top_example_dir_name = None
        example_id_file_name = None

    # Check input args.
    saliency_metadata_dict = gg_saliency_maps.check_metadata(
        component_type_string=component_type_string,
        target_class=target_class, layer_name=layer_name,
        ideal_activation=ideal_activation, neuron_indices=neuron_indices,
        channel_index=channel_index)

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    output_dir_name, pathless_output_file_name = os.path.split(output_file_name)
    extensionless_output_file_name, output_file_extension = os.path.splitext(
        pathless_output_file_name)

    if randomize_weights:
        conv_dense_layer_names = _find_conv_and_dense_layers(model_object)
        conv_dense_layer_names.reverse()
        num_sets = len(conv_dense_layer_names)
    else:
        conv_dense_layer_names = []
        num_sets = 1

    print(SEPARATOR_STRING)

    # Read predictors.
    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)
    predictor_names = model_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    pressure_levels_mb = model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]

    if example_file_name is None:
        print('Reading example IDs from: "{0:s}"...'.format(
            example_id_file_name
        ))
        example_id_strings = examples_io.read_example_ids(example_id_file_name)

        example_dict = examples_io.read_specific_examples_many_files(
            top_example_dir_name=top_example_dir_name,
            example_id_strings=example_id_strings,
            predictor_names_to_keep=predictor_names,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns)
    else:
        print('Reading normalized predictors from: "{0:s}"...'.format(
            example_file_name
        ))

        example_dict = examples_io.read_file(
            netcdf_file_name=example_file_name,
            predictor_names_to_keep=predictor_names,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns)

        example_id_strings = examples_io.create_example_ids(
            valid_times_unix_sec=example_dict[examples_io.VALID_TIMES_KEY],
            row_indices=example_dict[examples_io.ROW_INDICES_KEY],
            column_indices=example_dict[examples_io.COLUMN_INDICES_KEY]
        )

    # Denormalize predictors.
    print('Denormalizing predictors...')

    # TODO(thunderhoser): All this nonsense should be in a separate method.
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]
    normalization_type_string = example_dict[examples_io.NORMALIZATION_TYPE_KEY]

    normalization_dict = {
        ml_utils.MIN_VALUE_MATRIX_KEY: None,
        ml_utils.MAX_VALUE_MATRIX_KEY: None,
        ml_utils.MEAN_VALUE_MATRIX_KEY: None,
        ml_utils.STDEV_MATRIX_KEY: None
    }

    if normalization_type_string == ml_utils.Z_SCORE_STRING:
        normalization_dict[ml_utils.MEAN_VALUE_MATRIX_KEY] = example_dict[
            examples_io.FIRST_NORM_PARAM_KEY]

        normalization_dict[ml_utils.STDEV_MATRIX_KEY] = example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]
    else:
        normalization_dict[ml_utils.MIN_VALUE_MATRIX_KEY] = example_dict[
            examples_io.FIRST_NORM_PARAM_KEY]

        normalization_dict[ml_utils.MAX_VALUE_MATRIX_KEY] = example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]

    denorm_predictor_matrix = ml_utils.denormalize_predictors_nonglobal(
        predictor_matrix=predictor_matrix + 0.,
        normalization_dict=normalization_dict
    )

    print(SEPARATOR_STRING)

    for k in range(num_sets):
        if randomize_weights:
            if cascading_random:
                _reset_weights_in_layer(
                    model_object=model_object,
                    layer_name=conv_dense_layer_names[k]
                )

                this_model_object = model_object

                this_output_file_name = (
                    '{0:s}/{1:s}_cascading-random_{2:s}{3:s}'
                ).format(
                    output_dir_name, extensionless_output_file_name,
                    conv_dense_layer_names[k].replace('_', '-'),
                    output_file_extension
                )
            else:
                this_model_object = keras.models.Model.from_config(
                    model_object.get_config()
                )
                this_model_object.set_weights(model_object.get_weights())

                _reset_weights_in_layer(
                    model_object=this_model_object,
                    layer_name=conv_dense_layer_names[k]
                )

                this_output_file_name = '{0:s}/{1:s}_random_{2:s}{3:s}'.format(
                    output_dir_name, extensionless_output_file_name,
                    conv_dense_layer_names[k].replace('_', '-'),
                    output_file_extension
                )
        else:
            this_model_object = model_object
            this_output_file_name = output_file_name

        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print('Computing saliency maps for target class {0:d}...'.format(
                target_class
            ))

            this_saliency_matrix = (
                gg_saliency_maps.get_saliency_maps_for_class_activation(
                    model_object=this_model_object, target_class=target_class,
                    list_of_input_matrices=[predictor_matrix]
                )[0]
            )

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            print((
                'Computing saliency maps for neuron {0:s} in layer "{1:s}"...'
            ).format(
                str(neuron_indices), layer_name
            ))

            this_saliency_matrix = (
                gg_saliency_maps.get_saliency_maps_for_neuron_activation(
                    model_object=this_model_object, layer_name=layer_name,
                    neuron_indices=neuron_indices,
                    list_of_input_matrices=[predictor_matrix],
                    ideal_activation=ideal_activation
                )[0]
            )

        else:
            print((
                'Computing saliency maps for channel {0:d} in layer "{1:s}"...'
            ).format(
                channel_index, layer_name
            ))

            this_saliency_matrix = (
                gg_saliency_maps.get_saliency_maps_for_channel_activation(
                    model_object=this_model_object, layer_name=layer_name,
                    channel_index=channel_index,
                    list_of_input_matrices=[predictor_matrix],
                    stat_function_for_neuron_activations=K.max,
                    ideal_activation=ideal_activation
                )[0]
            )

        print('Writing saliency maps to file: "{0:s}"...'.format(
            this_output_file_name
        ))

        ge_saliency_maps.write_standard_file(
            pickle_file_name=this_output_file_name,
            denorm_predictor_matrix=denorm_predictor_matrix,
            saliency_matrix=this_saliency_matrix,
            example_id_strings=example_id_strings,
            model_file_name=model_file_name,
            metadata_dict=saliency_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(INPUT_ARG_OBJECT, ID_FILE_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_index=getattr(INPUT_ARG_OBJECT, CHANNEL_INDEX_ARG_NAME),
        randomize_weights=bool(getattr(INPUT_ARG_OBJECT, RANDOMIZE_ARG_NAME)),
        cascading_random=bool(getattr(INPUT_ARG_OBJECT, CASCADING_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
