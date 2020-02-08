"""Creates dummy saliency map for each example.

In these dummy saliency maps, the "saliency map" is actually just the filter
produced by an edge-detector with no learned weights.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.deep_learning import standalone_utils
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import saliency_maps as gg_saliency_maps
from generalexam.machine_learning import cnn
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import saliency_maps as ge_saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

EDGE_DETECTOR_MATRIX = numpy.array([
    [0.25, 0.5, 0.25],
    [0.5, -3, 0.5],
    [0.25, 0.5, 0.25]
])

MODEL_FILE_ARG_NAME = 'model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
ID_FILE_ARG_NAME = 'input_example_id_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.'
)

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `learning_examples_io.read_file`).  '
    'Will use all examples in this file.  If you want to use specific examples,'
    ' leave this argument alone and specify `{0:s}` and `{1:s}`, instead.'
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

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`saliency_maps.write_standard_file` in GeneralExam library).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ID_FILE_ARG_NAME, type=str, required=False, default='',
    help=ID_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, top_example_dir_name,
         example_id_file_name, output_file_name):
    """Creates dummy saliency map for each example.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param top_example_dir_name: Same.
    :param example_id_file_name: Same.
    :param output_file_name: Same.
    """

    if example_file_name in ['', 'None']:
        example_file_name = None
    else:
        top_example_dir_name = None
        example_id_file_name = None

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

    predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY] + 0.
    num_examples = predictor_matrix.shape[0]
    num_channels = predictor_matrix.shape[-1]

    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX, axis=-1)
    kernel_matrix = numpy.repeat(kernel_matrix, num_channels, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)
    kernel_matrix = numpy.repeat(kernel_matrix, num_channels, axis=-1)

    saliency_matrix = numpy.full(predictor_matrix.shape, numpy.nan)
    print(SEPARATOR_STRING)

    for i in range(num_examples):
        if numpy.mod(i, 25) == 0:
            print('Have made {0:d} of {1:d} dummy saliency maps...'.format(
                i, num_examples
            ))

        saliency_matrix[i, ...] = standalone_utils.do_2d_convolution(
            feature_matrix=predictor_matrix[i, ...],
            kernel_matrix=kernel_matrix, pad_edges=True, stride_length_px=1
        )[0, ...]

    print('Have all {0:d} dummy saliency maps!'.format(num_examples))
    print(SEPARATOR_STRING)

    print('Denormalizing predictors...')
    example_dict = examples_io.denormalize_examples(example_dict)
    denorm_predictor_matrix = example_dict[examples_io.PREDICTOR_MATRIX_KEY]

    print('Writing saliency maps to file: "{0:s}"...'.format(output_file_name))

    saliency_metadata_dict = gg_saliency_maps.check_metadata(
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=1
    )

    ge_saliency_maps.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_predictor_matrix=denorm_predictor_matrix,
        saliency_matrix=saliency_matrix, example_id_strings=example_id_strings,
        model_file_name=model_file_name, metadata_dict=saliency_metadata_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(INPUT_ARG_OBJECT, ID_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
