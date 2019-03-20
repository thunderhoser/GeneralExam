"""Plots feature maps for each example and CNN layer."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn as gg_cnn
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import feature_map_plotting
from generalexam.machine_learning import cnn as ge_cnn
from generalexam.machine_learning import learning_examples_io as examples_io

RANDOM_SEED = 6695

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
LAYER_NAMES_ARG_NAME = 'layer_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`ge_cnn.read_model`.')

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

LAYER_NAMES_HELP_STRING = (
    'Layer names.  Feature maps will be plotted for each pair of layer and '
    'example.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here (one '
    'subdirectory per layer).')

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
    '--' + LAYER_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAYER_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_feature_maps_one_layer(feature_matrix, layer_name, output_dir_name):
    """Plots all feature maps for one layer.

    E = number of examples
    M = number of spatial rows
    N = number of spatial columns
    C = number of channels

    :param feature_matrix: E-by-M-by-N-by-C numpy array of feature values.
    :param layer_name: Name of layer that output feature values.
    :param output_dir_name: Name of output directory for this layer.
    """

    num_examples = feature_matrix.shape[0]
    num_channels = feature_matrix.shape[-1]

    num_panel_rows = int(numpy.round(numpy.sqrt(num_channels)))
    annotation_string_by_channel = [None] * num_channels

    # annotation_string_by_channel = [
    #     'Filter {0:d}'.format(c + 1) for c in range(num_channels)
    # ]

    max_colour_value = numpy.percentile(numpy.absolute(feature_matrix), 99)
    min_colour_value = -1 * max_colour_value

    for i in range(num_examples):
        _, these_axes_objects = (
            feature_map_plotting.plot_many_2d_feature_maps(
                feature_matrix=feature_matrix[i, 2:-2, 2:-2, ...],
                annotation_string_by_panel=annotation_string_by_channel,
                num_panel_rows=num_panel_rows,
                colour_map_object=pyplot.cm.seismic,
                min_colour_value=min_colour_value,
                max_colour_value=max_colour_value)
        )

        plotting_utils.add_linear_colour_bar(
            axes_object_or_list=these_axes_objects,
            values_to_colour=feature_matrix[i, ...],
            colour_map=pyplot.cm.seismic, colour_min=min_colour_value,
            colour_max=max_colour_value, orientation='horizontal',
            extend_min=True, extend_max=True)

        # this_title_string = 'Layer "{0:s}", example {1:d}'.format(layer_name, i)
        # pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_figure_file_name = '{0:s}/example{1:06d}.jpg'.format(
            output_dir_name, i)

        print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(model_file_name, example_file_name, num_examples, example_indices,
         layer_names, top_output_dir_name):
    """Plots feature maps for each example and CNN layer.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_indices: Same.
    :param layer_names: Same.
    :param top_output_dir_name: Same.
    """

    if num_examples <= 0:
        num_examples = None

    if num_examples is None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
    else:
        error_checking.assert_is_greater(num_examples, 0)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = ge_cnn.read_model(model_file_name)
    num_half_rows, num_half_columns = ge_cnn.model_to_grid_dimensions(
        model_object)

    model_metafile_name = ge_cnn.find_metafile(model_file_name=model_file_name)
    print 'Reading model metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = ge_cnn.read_metadata(model_metafile_name)

    print 'Reading normalized examples from: "{0:s}"...'.format(
        example_file_name)
    example_dict = examples_io.read_file(
        netcdf_file_name=example_file_name,
        predictor_names_to_keep=model_metadata_dict[ge_cnn.PREDICTOR_NAMES_KEY],
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    print SEPARATOR_STRING
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
    num_examples = predictor_matrix.shape[0]

    num_layers = len(layer_names)
    feature_matrix_by_layer = [None] * num_layers

    for k in range(num_layers):
        print 'Creating feature maps for layer "{0:s}"...'.format(
            layer_names[k])

        this_partial_model_object = gg_cnn.model_to_feature_generator(
            model_object=model_object, feature_layer_name=layer_names[k])

        feature_matrix_by_layer[k] = this_partial_model_object.predict(
            predictor_matrix, batch_size=num_examples)

    print SEPARATOR_STRING

    for k in range(num_layers):
        this_output_dir_name = '{0:s}/{1:s}'.format(
            top_output_dir_name, layer_names[k])
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name)

        _plot_feature_maps_one_layer(
            feature_matrix=feature_matrix_by_layer[k],
            layer_name=layer_names[k], output_dir_name=this_output_dir_name)
        print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int),
        layer_names=getattr(INPUT_ARG_OBJECT, LAYER_NAMES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
