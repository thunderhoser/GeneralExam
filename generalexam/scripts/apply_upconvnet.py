"""Applies upconvnet to one or more examples.

--- NOTATION ---

The following letters are used throughout this script.

E = number of examples
M = number of rows in each grid
N = number of columns in each grid
C = number of channels (predictor variables)
"""

import random
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import upconvnet
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.plotting import example_plotting

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

COLOUR_MAP_OBJECT = pyplot.cm.plasma
FIGURE_RESOLUTION_DPI = 300

UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to file with input data (images).  Will be read by '
    '`training_validation_io.read_downsized_3d_examples`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to draw randomly from `{0:s}`.  If you want to select '
    'the examples, use `{1:s}` and leave this argument alone.'
).format(EXAMPLE_FILE_ARG_NAME, EXAMPLE_INDICES_ARG_NAME)

EXAMPLE_INDICES_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Indices of examples to draw from '
    '`{1:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, EXAMPLE_FILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Reconstructed examples (images) will be plotted'
    ' and saved here.')

DEFAULT_NUM_EXAMPLES = 50

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_input_examples(example_file_name, cnn_metadata_dict, num_examples,
                         example_indices):
    """Reads input examples (images to be reconstructed).

    :param example_file_name: See documentation at top of file.
    :param cnn_metadata_dict: Dictionary returned by
        `traditional_cnn.read_model_metadata`.
    :param num_examples: See documentation at top of file.
    :param example_indices: Same.
    :return: actual_image_matrix: E-by-M-by-N-by-C numpy array with actual
        images (input examples to CNN).
    """

    print 'Reading input examples (images) from: "{0:s}"...'.format(
        example_file_name)
    example_dict = trainval_io.read_downsized_3d_examples(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=cnn_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        num_half_rows_to_keep=cnn_metadata_dict[
            traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
        num_half_columns_to_keep=cnn_metadata_dict[
            traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY]
    )

    actual_image_matrix = example_dict[trainval_io.PREDICTOR_MATRIX_KEY]

    if num_examples is not None:
        num_examples_total = actual_image_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int)

        num_examples = min([num_examples, num_examples_total])
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False)

    return actual_image_matrix[example_indices, ...]


def _plot_examples(actual_image_matrix, reconstructed_image_matrix,
                   narr_predictor_names, top_output_dir_name):
    """Plots actual and reconstructed examples.

    :param actual_image_matrix: E-by-M-by-N-by-C numpy array with actual images
        (input examples to CNN).
    :param reconstructed_image_matrix: E-by-M-by-N-by-C numpy array with
        reconstructed images (output of upconvnet).
    :param narr_predictor_names: length-C list of predictor names.
    :param top_output_dir_name: Name of top-level output directory.  Figures
        will be saved here.
    """

    actual_image_dir_name = '{0:s}/actual_images'.format(top_output_dir_name)
    reconstructed_image_dir_name = '{0:s}/reconstructed_images'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=actual_image_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=reconstructed_image_dir_name)

    try:
        example_plotting.get_wind_indices(narr_predictor_names)
        plot_wind_barbs = True
    except ValueError:
        plot_wind_barbs = False

    num_examples = actual_image_matrix.shape[0]
    num_predictors = len(narr_predictor_names)

    for i in range(num_examples):
        this_combined_matrix = numpy.concatenate(
            (actual_image_matrix[i, ...], reconstructed_image_matrix[i, ...]),
            axis=0)

        these_min_colour_values = numpy.array([
            numpy.percentile(this_combined_matrix[..., k], 1)
            for k in range(num_predictors)
        ])

        these_max_colour_values = numpy.array([
            numpy.percentile(this_combined_matrix[..., k], 99)
            for k in range(num_predictors)
        ])

        this_figure_file_name = '{0:s}/example{1:06d}_actual.jpg'.format(
            actual_image_dir_name, i)

        if plot_wind_barbs:
            example_plotting.plot_many_predictors_with_barbs(
                predictor_matrix=actual_image_matrix[i, ...],
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=[COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)
        else:
            example_plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=reconstructed_image_matrix[i, ...],
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=[COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)

        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        this_figure_file_name = '{0:s}/example{1:06d}_reconstructed.jpg'.format(
            reconstructed_image_dir_name, i)

        if plot_wind_barbs:
            example_plotting.plot_many_predictors_with_barbs(
                predictor_matrix=reconstructed_image_matrix[i, ...],
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=[COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)
        else:
            example_plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=reconstructed_image_matrix[i, ...],
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=[COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)

        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(upconvnet_file_name, example_file_name, num_examples, example_indices,
         top_output_dir_name):
    """Applies upconvnet to one or more examples.

    This is effectively the main method.

    :param upconvnet_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_indices: Same.
    :param top_output_dir_name: Same.
    """

    # Check input args.
    if num_examples <= 0:
        num_examples = None

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
    else:
        error_checking.assert_is_greater(num_examples, 0)

    # Read upconvnet and metadata.
    ucn_metafile_name = traditional_cnn.find_metafile(
        model_file_name=upconvnet_file_name, raise_error_if_missing=True)

    print('Reading trained upconvnet from: "{0:s}"...'.format(
        upconvnet_file_name))
    ucn_model_object = traditional_cnn.read_keras_model(upconvnet_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        ucn_metafile_name))
    ucn_metadata_dict = upconvnet.read_model_metadata(ucn_metafile_name)

    # Read CNN and metadata.
    cnn_file_name = ucn_metadata_dict[upconvnet.CNN_FILE_NAME_KEY]
    cnn_metafile_name = traditional_cnn.find_metafile(
        model_file_name=cnn_file_name, raise_error_if_missing=True)

    print 'Reading trained CNN from: "{0:s}"...'.format(cnn_file_name)
    cnn_model_object = traditional_cnn.read_keras_model(cnn_file_name)

    print 'Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name)
    cnn_metadata_dict = traditional_cnn.read_model_metadata(cnn_metafile_name)
    print SEPARATOR_STRING

    actual_image_matrix = _read_input_examples(
        example_file_name=example_file_name,
        cnn_metadata_dict=cnn_metadata_dict, num_examples=num_examples,
        example_indices=example_indices)
    print SEPARATOR_STRING

    reconstructed_image_matrix = upconvnet.apply_upconvnet(
        actual_image_matrix=actual_image_matrix,
        cnn_model_object=cnn_model_object, ucn_model_object=ucn_model_object)
    print SEPARATOR_STRING

    _plot_examples(
        actual_image_matrix=actual_image_matrix,
        reconstructed_image_matrix=reconstructed_image_matrix,
        narr_predictor_names=cnn_metadata_dict[
            traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
        top_output_dir_name=top_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
