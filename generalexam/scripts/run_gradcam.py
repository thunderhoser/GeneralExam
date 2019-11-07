"""Runs Grad-CAM (gradient-weighted class-activation maps)."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import gradcam as gg_gradcam
from generalexam.machine_learning import cnn
from generalexam.machine_learning import gradcam as ge_gradcam
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import learning_examples_io as examples_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'model_file_name'
TARGET_CLASS_ARG_NAME = 'target_class'
TARGET_LAYER_ARG_NAME = 'target_layer_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')

TARGET_CLASS_HELP_STRING = (
    'Activation maps will be created for this class.  Must be in 0...(K - 1), '
    'where K = number of classes.')

TARGET_LAYER_HELP_STRING = (
    'Name of target layer.  Neuron-importance weights will be based on '
    'activations in this layer.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `learning_examples_io.read_file`).  '
    'Class-activation maps will be created for examples in this file.')

NUM_EXAMPLES_HELP_STRING = (
    'Class-activation maps will be created for this many examples, drawn '
    'randomly from `{0:s}`.  To use all examples, leave this argument alone.'
).format(EXAMPLE_FILE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `gradcam.write_standard_file` in '
    'GeneralExam library).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=True,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LAYER_ARG_NAME, type=str, required=True,
    help=TARGET_LAYER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, target_class, target_layer_name, example_file_name,
         num_examples, output_file_name):
    """Runs Grad-CAM (gradient-weighted class-activation maps).

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param target_class: Same.
    :param target_layer_name: Same.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    # Read predictors.
    print('Reading normalized predictors from: "{0:s}"...'.format(
        example_file_name
    ))

    num_half_rows, num_half_columns = cnn.model_to_grid_dimensions(model_object)

    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_to_keep_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY],
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    num_examples_found = len(example_dict[examples_io.VALID_TIMES_KEY])

    if 0 < num_examples < num_examples_found:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)

        example_dict = examples_io.subset_examples(
            example_dict=example_dict, desired_indices=desired_indices)

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

    denorm_predictor_matrix = ml_utils.denormalize_predictors(
        predictor_matrix=predictor_matrix + 0.,
        normalization_dict=normalization_dict
    )

    class_activn_matrix = None
    guided_class_activn_matrix = None
    new_model_object = None
    num_examples = len(example_id_strings)

    print(SEPARATOR_STRING)

    for i in range(num_examples):
        print('Running Grad-CAM for example {0:d} of {1:d}...'.format(
            i + 1, num_examples
        ))

        this_activn_matrix = gg_gradcam.run_gradcam(
            model_object=model_object,
            list_of_input_matrices=[predictor_matrix[[i], ...]],
            target_class=target_class, target_layer_name=target_layer_name
        )[0]

        print('Running guided Grad-CAM for example {0:d} of {1:d}...'.format(
            i + 1, num_examples
        ))

        these_matrices, new_model_object = gg_gradcam.run_guided_gradcam(
            orig_model_object=model_object,
            list_of_input_matrices=[predictor_matrix[[i], ...]],
            target_layer_name=target_layer_name,
            list_of_cam_matrices=[this_activn_matrix],
            new_model_object=new_model_object)

        this_guided_activn_matrix = these_matrices[0]

        if class_activn_matrix is None:
            class_activn_matrix = this_activn_matrix + 0.
            guided_class_activn_matrix = this_guided_activn_matrix + 0.
        else:
            class_activn_matrix = numpy.concatenate(
                (class_activn_matrix, this_activn_matrix), axis=0
            )
            guided_class_activn_matrix = numpy.concatenate(
                (guided_class_activn_matrix, this_guided_activn_matrix), axis=0
            )

    print(SEPARATOR_STRING)
    print('Writing class-activation maps to file: "{0:s}"...'.format(
        output_file_name
    ))

    ge_gradcam.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_predictor_matrix=denorm_predictor_matrix,
        class_activn_matrix=class_activn_matrix,
        guided_class_activn_matrix=guided_class_activn_matrix,
        example_id_strings=example_id_strings,
        model_file_name=model_file_name,
        target_class=target_class, target_layer_name=target_layer_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        target_layer_name=getattr(INPUT_ARG_OBJECT, TARGET_LAYER_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
