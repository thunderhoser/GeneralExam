"""Runs Grad-CAM (gradient-weighted class-activation maps)."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.deep_learning import gradcam as gg_gradcam
from generalexam.machine_learning import cnn
from generalexam.machine_learning import gradcam as ge_gradcam
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import make_saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'model_file_name'
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
ID_FILE_ARG_NAME = make_saliency_maps.ID_FILE_ARG_NAME
TARGET_CLASS_ARG_NAME = 'target_class'
TARGET_LAYER_ARG_NAME = 'target_layer_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')

EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
ID_FILE_HELP_STRING = make_saliency_maps.ID_FILE_HELP_STRING

TARGET_CLASS_HELP_STRING = (
    'Activation maps will be created for this class.  Must be in 0...(K - 1), '
    'where K = number of classes.')

TARGET_LAYER_HELP_STRING = (
    'Name of target layer.  Neuron-importance weights will be based on '
    'activations in this layer.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `gradcam.write_standard_file` in '
    'GeneralExam library).')

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
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=True,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_LAYER_ARG_NAME, type=str, required=True,
    help=TARGET_LAYER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, example_file_name, top_example_dir_name,
         example_id_file_name, target_class, target_layer_name,
         output_file_name):
    """Runs Grad-CAM (gradient-weighted class-activation maps).

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param top_example_dir_name: Same.
    :param example_id_file_name: Same.
    :param target_class: Same.
    :param target_layer_name: Same.
    :param output_file_name: Same.
    """

    if example_file_name in ['', 'None']:
        example_file_name = None
    else:
        top_example_dir_name = None
        example_id_file_name = None

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
    normalization_file_name = model_metadata_dict[cnn.NORMALIZATION_FILE_KEY]

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
            num_half_columns_to_keep=num_half_columns,
            normalization_file_name=normalization_file_name)
    else:
        print('Reading normalized predictors from: "{0:s}"...'.format(
            example_file_name
        ))

        example_dict = examples_io.read_file(
            netcdf_file_name=example_file_name,
            predictor_names_to_keep=predictor_names,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns,
            normalization_file_name=normalization_file_name)

        example_id_strings = examples_io.create_example_ids(
            valid_times_unix_sec=example_dict[examples_io.VALID_TIMES_KEY],
            row_indices=example_dict[examples_io.ROW_INDICES_KEY],
            column_indices=example_dict[examples_io.COLUMN_INDICES_KEY]
        )

    print('Denormalizing predictors...')
    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY] + 0.
    example_dict = examples_io.denormalize_examples(example_dict)
    denorm_predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]

    print(SEPARATOR_STRING)

    class_activn_matrix = None
    guided_class_activn_matrix = None
    new_model_object = None
    num_examples = len(example_id_strings)

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
            cam_dimensions = numpy.array(
                (num_examples,) + this_activn_matrix.shape[1:], dtype=int
            )
            class_activn_matrix = numpy.full(cam_dimensions, numpy.nan)

            guided_cam_dimensions = numpy.array(
                (num_examples,) + this_guided_activn_matrix.shape[1:], dtype=int
            )
            guided_class_activn_matrix = numpy.full(
                guided_cam_dimensions, numpy.nan)

        class_activn_matrix[i, ...] = this_activn_matrix[0, ...]
        guided_class_activn_matrix[i, ...] = this_guided_activn_matrix[0, ...]

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
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(INPUT_ARG_OBJECT, ID_FILE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        target_layer_name=getattr(INPUT_ARG_OBJECT, TARGET_LAYER_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
