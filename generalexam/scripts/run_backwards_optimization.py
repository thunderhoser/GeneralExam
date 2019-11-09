"""Uses CNN to backwards-optimize one or more examples.

CNN = convolutional neural network
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.deep_learning import \
    backwards_optimization as gg_backwards_opt
from gewittergefahr.deep_learning import model_interpretation
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import \
    backwards_optimization as ge_backwards_opt
from generalexam.scripts import make_saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
ID_FILE_ARG_NAME = make_saliency_maps.ID_FILE_ARG_NAME
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
L2_WEIGHT_ARG_NAME = 'l2_weight'
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

EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
ID_FILE_HELP_STRING = make_saliency_maps.ID_FILE_HELP_STRING
NUM_ITERATIONS_HELP_STRING = 'Number of iterations for backwards optimization.'
LEARNING_RATE_HELP_STRING = 'Learning rate for backwards optimization.'
L2_WEIGHT_HELP_STRING = 'L2 weight for backwards optimization.'

COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Examples will be optimized for one class probability, '
    'one/many neuron activations, or one channel activation.  Valid options '
    'are:\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))

TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Examples will be optimized for class k, '
    'where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Examples will be optimized for '
    'activation of one/many neurons or one channel in this layer.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CHANNEL_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Ideal activation used to define '
    'loss function.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CHANNEL_COMPONENT_TYPE_STRING)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Examples will be optimized for activation '
    'of this neuron.  For example, to optimize for neuron (0, 0, 2) in the '
    'given layer, this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Examples will be optimized for activation '
    'of [k]th channel in the given layer, where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING,
         CHANNEL_INDEX_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`backwards_optimization.write_standard_file` in GeneralExam library).')

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
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=gg_backwards_opt.DEFAULT_NUM_ITERATIONS,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=gg_backwards_opt.DEFAULT_LEARNING_RATE,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
    default=gg_backwards_opt.DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING)

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
    default=gg_backwards_opt.DEFAULT_IDEAL_ACTIVATION,
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


def _run(model_file_name, example_file_name, top_example_dir_name,
         example_id_file_name, num_iterations, learning_rate, l2_weight,
         component_type_string, target_class, layer_name, ideal_activation,
         neuron_indices, channel_index, output_file_name):
    """Uses CNN to backwards-optimize one or more examples.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param top_example_dir_name: Same.
    :param example_id_file_name: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param output_file_name: Same.
    """

    if example_file_name in ['', 'None']:
        example_file_name = None
    else:
        top_example_dir_name = None
        example_id_file_name = None

    if l2_weight <= 0:
        l2_weight = None
    if ideal_activation <= 0:
        ideal_activation = None

    # Check input args.
    bwo_metadata_dict = gg_backwards_opt.check_metadata(
        component_type_string=component_type_string,
        num_iterations=num_iterations, learning_rate=learning_rate,
        target_class=target_class, layer_name=layer_name,
        ideal_activation=ideal_activation, neuron_indices=neuron_indices,
        channel_index=channel_index, l2_weight=l2_weight)

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

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
        print('Reading pre-optimized examples from: "{0:s}"...'.format(
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
    print('Denormalizing pre-optimized examples...')

    # TODO(thunderhoser): All this nonsense should be in a separate method.
    input_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]
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

    denorm_input_matrix = ml_utils.denormalize_predictors(
        predictor_matrix=input_matrix + 0.,
        normalization_dict=normalization_dict)

    print(SEPARATOR_STRING)

    num_examples = input_matrix.shape[0]
    initial_activations = numpy.full(num_examples, numpy.nan)
    final_activations = numpy.full(num_examples, numpy.nan)
    output_matrix = numpy.full(input_matrix.shape, numpy.nan)

    for i in range(num_examples):
        this_input_matrix = [input_matrix[[i], ...]]

        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print((
                '\nOptimizing {0:d}th of {1:d} images for target class {2:d}...'
            ).format(
                i + 1, num_examples, target_class
            ))

            this_result_dict = gg_backwards_opt.optimize_input_for_class(
                model_object=model_object, target_class=target_class,
                init_function_or_matrices=this_input_matrix,
                num_iterations=num_iterations, learning_rate=learning_rate,
                l2_weight=l2_weight)

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            print((
                '\nOptimizing {0:d}th of {1:d} images for neuron {2:s} in layer'
                ' "{3:s}"...'
            ).format(
                i + 1, num_examples, str(neuron_indices), layer_name
            ))

            this_result_dict = gg_backwards_opt.optimize_input_for_neuron(
                model_object=model_object, layer_name=layer_name,
                neuron_indices=neuron_indices,
                init_function_or_matrices=this_input_matrix,
                num_iterations=num_iterations, learning_rate=learning_rate,
                l2_weight=l2_weight, ideal_activation=ideal_activation)

        else:
            print((
                '\nOptimizing {0:d}th of {1:d} images for channel {2:d} in '
                'layer "{3:s}"...'
            ).format(
                i + 1, num_examples, channel_index, layer_name
            ))

            this_result_dict = gg_backwards_opt.optimize_input_for_channel(
                model_object=model_object, layer_name=layer_name,
                channel_index=channel_index,
                init_function_or_matrices=this_input_matrix,
                stat_function_for_neuron_activations=K.max,
                num_iterations=num_iterations, learning_rate=learning_rate,
                l2_weight=l2_weight, ideal_activation=ideal_activation)

        initial_activations[i] = this_result_dict[
            gg_backwards_opt.INITIAL_ACTIVATION_KEY]
        final_activations[i] = this_result_dict[
            gg_backwards_opt.FINAL_ACTIVATION_KEY]
        output_matrix[i, ...] = this_result_dict[
            gg_backwards_opt.NORM_OUTPUT_MATRICES_KEY
        ][0][0, ...]

    print(SEPARATOR_STRING)

    print('Denormalizing optimized examples...')
    denorm_output_matrix = ml_utils.denormalize_predictors(
        predictor_matrix=output_matrix + 0.,
        normalization_dict=normalization_dict)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ge_backwards_opt.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_input_matrix=denorm_input_matrix,
        denorm_output_matrix=denorm_output_matrix,
        initial_activations=initial_activations,
        final_activations=final_activations,
        example_id_strings=example_id_strings,
        model_file_name=model_file_name, metadata_dict=bwo_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(INPUT_ARG_OBJECT, ID_FILE_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, L2_WEIGHT_ARG_NAME),
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
