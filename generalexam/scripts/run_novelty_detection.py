"""Runs novelty detection.

--- NOTATION ---

The following letters are used throughout this script.

B = number of baseline examples
T = number of test examples
M = number of rows in each grid
N = number of columns in each grid
C = number of channels (predictor variables)
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import novelty_detection
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import traditional_cnn
from generalexam.machine_learning import upconvnet
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.plotting import example_plotting

# TODO(thunderhoser): Allow different criteria for test set.  The current
# criterion is highest cold-front probability.  Could be highest no-front
# probability, lowest warm-front probability, etc. etc.

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAIN_COLOUR_MAP_OBJECT = pyplot.cm.plasma
NOVELTY_COLOUR_MAP_OBJECT = pyplot.cm.bwr
FIGURE_RESOLUTION_DPI = 300

UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_BASELINE_EX_ARG_NAME = 'num_baseline_examples'
NUM_TEST_EX_ARG_NAME = 'num_test_examples'
NUM_SVD_MODES_ARG_NAME = 'num_svd_modes_to_keep'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet.  Will be read by '
    '`traditional_cnn.read_keras_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input data.  Files therein (containing'
    ' downsized 3-D examples, with 2 spatial dimensions) will be found by '
    '`training_validation_io.find_downsized_3d_example_file` (with shuffled = '
    'False) and read by `training_validation_io.read_downsized_3d_examples`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Only examples in the period `{0:s}`...`{1:s}`'
    ' will be considered for the baseline and testing sets.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_BASELINE_EX_HELP_STRING = (
    'Number of baseline examples.  `{0:s}` examples will be randomly selected '
    'from the period `{1:s}`...`{2:s}` and used to create the original SVD '
    '(singular-value decomposition) model.  The novelty of each test example '
    'will be computed relative to these baseline examples.'
).format(NUM_BASELINE_EX_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TEST_EX_HELP_STRING = (
    'Number of testing examples.  The `{0:s}` examples with the greatest '
    'predicted cold-front probability will be selected from the period '
    '`{1:s}`...`{2:s}` and ranked by their novelty with respect to baseline '
    'examples.'
).format(NUM_TEST_EX_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_SVD_MODES_HELP_STRING = (
    'Number of modes (top eigenvectors) to retain in the SVD (singular-value '
    'decomposition) model.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  The dictionary created by '
    '`_novelty_detection.do_novelty_detection`, as well as plots, will be saved'
    ' here.')

DEFAULT_TOP_EXAMPLE_DIR_NAME = (
    '/condo/swatwork/ralager/narr_data/downsized_3d_examples')
DEFAULT_NUM_BASELINE_EXAMPLES = 100
DEFAULT_NUM_TEST_EXAMPLES = 100

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_EXAMPLE_DIR_NAME, help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BASELINE_EX_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_BASELINE_EXAMPLES, help=NUM_BASELINE_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TEST_EX_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_TEST_EXAMPLES, help=NUM_TEST_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SVD_MODES_ARG_NAME, type=int, required=True,
    help=NUM_SVD_MODES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_cnn_predictions(cnn_model_object, predictor_matrix, target_class,
                         verbose=True):
    """Returns CNN predictions (probabilities) for the given class.

    E = number of examples

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param target_class: Will return probabilities of the [k]th class, where
        k = `target_class`.
    :param verbose: Boolean flag.  If True, progress messages will be printed.
    :return: forecast_probabilities: length-E numpy array of probabilities (that
        class = `target_class`).
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 1000

    forecast_probabilities = numpy.array([], dtype=float)

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print (
                'Generating CNN predictions for examples {0:d}-{1:d} of '
                '{2:d}...'
            ).format(this_first_index, this_last_index, num_examples)

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        these_probabilities = cnn_model_object.predict(
            predictor_matrix[these_indices, ...],
            batch_size=num_examples_per_batch
        )[:, target_class]

        forecast_probabilities = numpy.concatenate(
            (forecast_probabilities, these_probabilities), axis=0)

    if verbose:
        print 'Generated CNN predictions for all {0:d} examples!'.format(
            num_examples)

    return forecast_probabilities


def _find_baseline_and_test_examples(
        top_example_dir_name, first_time_string, last_time_string,
        num_baseline_examples, num_test_examples, cnn_model_object,
        cnn_metadata_dict):
    """Finds examples for baseline and test sets.

    :param top_example_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_baseline_examples: Same.
    :param num_test_examples: Same.
    :param cnn_model_object:
    :param cnn_metadata_dict:
    :return: baseline_image_matrix: B-by-M-by-N-by-C numpy array of baseline
        images (input examples for the CNN).
    :return: test_image_matrix: B-by-M-by-N-by-C numpy array of test images
        (input examples for the CNN).
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT)

    example_file_names = trainval_io.find_downsized_3d_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_target_time_unix_sec=first_time_unix_sec,
        last_target_time_unix_sec=last_time_unix_sec)

    file_indices = numpy.array([], dtype=int)
    file_position_indices = numpy.array([], dtype=int)
    cold_front_probabilities = numpy.array([], dtype=float)

    for k in range(len(example_file_names)):
        print 'Reading data from: "{0:s}"...'.format(example_file_names[k])
        this_example_dict = trainval_io.read_downsized_3d_examples(
            netcdf_file_name=example_file_names[k], metadata_only=False,
            predictor_names_to_keep=cnn_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            num_half_rows_to_keep=cnn_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_half_columns_to_keep=cnn_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec)

        this_num_examples = len(this_example_dict[trainval_io.TARGET_TIMES_KEY])
        if this_num_examples == 0:
            continue

        these_file_indices = numpy.full(this_num_examples, k, dtype=int)
        these_position_indices = numpy.linspace(
            0, this_num_examples - 1, num=this_num_examples, dtype=int)

        these_cold_front_probs = _get_cnn_predictions(
            cnn_model_object=cnn_model_object,
            predictor_matrix=this_example_dict[
                trainval_io.PREDICTOR_MATRIX_KEY],
            target_class=front_utils.COLD_FRONT_INTEGER_ID, verbose=True)
        print '\n'

        file_indices = numpy.concatenate((file_indices, these_file_indices))
        file_position_indices = numpy.concatenate((
            file_position_indices, these_position_indices))
        cold_front_probabilities = numpy.concatenate((
            cold_front_probabilities, these_cold_front_probs))

    print SEPARATOR_STRING

    # Find test set.
    test_indices = numpy.argsort(
        -1 * cold_front_probabilities)[:num_test_examples]
    file_indices_for_test = file_indices[test_indices]
    file_position_indices_for_test = file_position_indices[test_indices]

    print 'Cold-front probabilities for the {0:d} test examples are:'.format(
        num_test_examples)
    for i in test_indices:
        print cold_front_probabilities[i]
    print SEPARATOR_STRING

    # Find baseline set.
    baseline_indices = numpy.linspace(
        0, num_baseline_examples - 1, num=num_baseline_examples, dtype=int)

    baseline_indices = (
        set(baseline_indices.tolist()) - set(test_indices.tolist())
    )
    baseline_indices = numpy.array(list(baseline_indices), dtype=int)
    baseline_indices = numpy.random.choice(
        baseline_indices, size=num_baseline_examples, replace=False)

    file_indices_for_baseline = file_indices[baseline_indices]
    file_position_indices_for_baseline = file_position_indices[baseline_indices]

    print (
        'Cold-front probabilities for the {0:d} baseline examples are:'
    ).format(num_baseline_examples)
    for i in baseline_indices:
        print cold_front_probabilities[i]
    print SEPARATOR_STRING

    # Read test and baseline sets.
    baseline_image_matrix = None
    test_image_matrix = None

    for k in range(len(example_file_names)):
        if not (k in file_indices_for_test or k in file_indices_for_baseline):
            continue

        print 'Reading data from: "{0:s}"...'.format(example_file_names[k])
        this_example_dict = trainval_io.read_downsized_3d_examples(
            netcdf_file_name=example_file_names[k], metadata_only=False,
            predictor_names_to_keep=cnn_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            num_half_rows_to_keep=cnn_metadata_dict[
                traditional_cnn.NUM_ROWS_IN_HALF_GRID_KEY],
            num_half_columns_to_keep=cnn_metadata_dict[
                traditional_cnn.NUM_COLUMNS_IN_HALF_GRID_KEY],
            first_time_to_keep_unix_sec=first_time_unix_sec,
            last_time_to_keep_unix_sec=last_time_unix_sec)

        this_predictor_matrix = this_example_dict[
            trainval_io.PREDICTOR_MATRIX_KEY]

        if baseline_image_matrix is None:
            baseline_image_matrix = numpy.full(
                (num_baseline_examples,) + this_predictor_matrix.shape[1:],
                numpy.nan)
            test_image_matrix = numpy.full(
                (num_test_examples,) + this_predictor_matrix.shape[1:],
                numpy.nan)

        these_baseline_indices = numpy.where(file_indices_for_baseline == k)[0]
        if len(these_baseline_indices) > 0:
            baseline_image_matrix[these_baseline_indices, ...] = (
                this_predictor_matrix[
                    file_position_indices_for_baseline[these_baseline_indices],
                    ...
                ]
            )

        these_test_indices = numpy.where(file_indices_for_test == k)[0]
        if len(these_test_indices) > 0:
            test_image_matrix[these_test_indices, ...] = (
                this_predictor_matrix[
                    file_position_indices_for_test[these_test_indices], ...
                ]
            )

    return baseline_image_matrix, test_image_matrix


def _plot_results(novelty_dict, narr_predictor_names, test_index,
                  top_output_dir_name):
    """Plots results of novelty detection.

    :param novelty_dict: Dictionary created by
        `novelty_detection.do_novelty_detection`.
    :param narr_predictor_names: length-C list of predictor names.
    :param test_index: Array index.  The [i]th-most novel test example will be
        plotted, where i = `test_index`.
    :param top_output_dir_name: Name of top-level output directory.  Figures
        will be saved here.
    """

    num_predictors = len(narr_predictor_names)

    try:
        example_plotting.get_wind_indices(narr_predictor_names)
        plot_wind_barbs = True
    except ValueError:
        plot_wind_barbs = False

    image_matrix_actual = novelty_dict[
        novelty_detection.NOVEL_IMAGES_ACTUAL_KEY][test_index, ...]
    image_matrix_upconv = novelty_dict[
        novelty_detection.NOVEL_IMAGES_UPCONV_KEY][test_index, ...]
    this_combined_matrix = numpy.concatenate(
        (image_matrix_actual, image_matrix_upconv), axis=0)

    these_min_colour_values = numpy.array([
        numpy.percentile(this_combined_matrix[..., k], 1)
        for k in range(num_predictors)
    ])

    these_max_colour_values = numpy.array([
        numpy.percentile(this_combined_matrix[..., k], 99)
        for k in range(num_predictors)
    ])

    if plot_wind_barbs:
        this_figure_object, _ = example_plotting.plot_many_predictors_with_barbs(
            predictor_matrix=image_matrix_actual,
            predictor_names=narr_predictor_names,
            cmap_object_by_predictor=[MAIN_COLOUR_MAP_OBJECT] * num_predictors,
            min_colour_value_by_predictor=these_min_colour_values,
            max_colour_value_by_predictor=these_max_colour_values)
    else:
        this_figure_object, _ = example_plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=image_matrix_actual,
            predictor_names=narr_predictor_names,
            cmap_object_by_predictor=[MAIN_COLOUR_MAP_OBJECT] * num_predictors,
            min_colour_value_by_predictor=these_min_colour_values,
            max_colour_value_by_predictor=these_max_colour_values)

    base_title_string = '{0:d}th-most novel example'.format(test_index + 1)
    this_title_string = '{0:s}: actual'.format(base_title_string)
    this_figure_object.suptitle(this_title_string)

    this_file_name = '{0:s}/actual_images/actual_image{1:04d}.jpg'.format(
        top_output_dir_name, test_index)
    file_system_utils.mkdir_recursive_if_necessary(file_name=this_file_name)

    print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    if plot_wind_barbs:
        this_figure_object, _ = example_plotting.plot_many_predictors_with_barbs(
            predictor_matrix=image_matrix_upconv,
            predictor_names=narr_predictor_names,
            cmap_object_by_predictor=[MAIN_COLOUR_MAP_OBJECT] * num_predictors,
            min_colour_value_by_predictor=these_min_colour_values,
            max_colour_value_by_predictor=these_max_colour_values)
    else:
        this_figure_object, _ = example_plotting.plot_many_predictors_sans_barbs(
            predictor_matrix=image_matrix_upconv,
            predictor_names=narr_predictor_names,
            cmap_object_by_predictor=[MAIN_COLOUR_MAP_OBJECT] * num_predictors,
            min_colour_value_by_predictor=these_min_colour_values,
            max_colour_value_by_predictor=these_max_colour_values)

    this_title_string = r'{0:s}: upconvnet reconstruction'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up}$)'
    this_figure_object.suptitle(this_title_string)

    this_file_name = '{0:s}/upconv_images/upconv_image{1:04d}.jpg'.format(
        top_output_dir_name, test_index)
    file_system_utils.mkdir_recursive_if_necessary(file_name=this_file_name)

    print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    novelty_matrix = (
        image_matrix_upconv -
        novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY][
            test_index, ...]
    )

    these_max_colour_values = numpy.array([
        numpy.percentile(numpy.absolute(image_matrix_upconv[..., k]), 99)
        for k in range(num_predictors)
    ])

    these_min_colour_values = -1 * these_max_colour_values

    if plot_wind_barbs:
        this_figure_object, _ = (
            example_plotting.plot_many_predictors_with_barbs(
                predictor_matrix=novelty_matrix,
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=
                [NOVELTY_COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)
        )
    else:
        this_figure_object, _ = (
            example_plotting.plot_many_predictors_sans_barbs(
                predictor_matrix=novelty_matrix,
                predictor_names=narr_predictor_names,
                cmap_object_by_predictor=
                [NOVELTY_COLOUR_MAP_OBJECT] * num_predictors,
                min_colour_value_by_predictor=these_min_colour_values,
                max_colour_value_by_predictor=these_max_colour_values)
        )

    this_title_string = r'{0:s}: novelty'.format(
        base_title_string)
    this_title_string += r' ($\mathbf{X}_{up} - \mathbf{X}_{up,svd}$)'
    this_figure_object.suptitle(this_title_string)

    this_file_name = '{0:s}/novelty_images/novelty_image{1:04d}.jpg'.format(
        top_output_dir_name, test_index)
    file_system_utils.mkdir_recursive_if_necessary(file_name=this_file_name)

    print 'Saving figure to file: "{0:s}"...'.format(this_file_name)
    pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _run(upconvnet_file_name, top_example_dir_name, first_time_string,
         last_time_string, num_baseline_examples, num_test_examples,
         num_svd_modes_to_keep, top_output_dir_name):
    """Runs novelty detection.

    :param upconvnet_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_baseline_examples: Same.
    :param num_test_examples: Same.
    :param num_svd_modes_to_keep: Same.
    :param top_output_dir_name: Same.
    """

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

    baseline_image_matrix, test_image_matrix = _find_baseline_and_test_examples(
        top_example_dir_name=top_example_dir_name,
        first_time_string=first_time_string, last_time_string=last_time_string,
        num_baseline_examples=num_baseline_examples,
        num_test_examples=num_test_examples, cnn_model_object=cnn_model_object,
        cnn_metadata_dict=cnn_metadata_dict)
    print SEPARATOR_STRING

    novelty_dict = novelty_detection.do_novelty_detection(
        baseline_image_matrix=baseline_image_matrix,
        test_image_matrix=test_image_matrix, cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=traditional_cnn.get_flattening_layer(
            cnn_model_object),
        ucn_model_object=ucn_model_object,
        num_novel_test_images=num_test_examples,
        norm_function=None, denorm_function=None,
        num_svd_modes_to_keep=num_svd_modes_to_keep)
    print SEPARATOR_STRING

    novelty_dict[novelty_detection.UCN_FILE_NAME_KEY] = upconvnet_file_name
    novelty_file_name = '{0:s}/novelty_results.p'.format(top_output_dir_name)

    print 'Writing results to: "{0:s}"...\n'.format(novelty_file_name)
    novelty_detection.write_results(novelty_dict=novelty_dict,
                                    pickle_file_name=novelty_file_name)

    for i in range(num_test_examples):
        _plot_results(
            novelty_dict=novelty_dict,
            narr_predictor_names=cnn_metadata_dict[
                traditional_cnn.NARR_PREDICTOR_NAMES_KEY],
            test_index=i, top_output_dir_name=top_output_dir_name)
        print '\n'


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_baseline_examples=getattr(
            INPUT_ARG_OBJECT, NUM_BASELINE_EX_ARG_NAME),
        num_test_examples=getattr(INPUT_ARG_OBJECT, NUM_TEST_EX_ARG_NAME),
        num_svd_modes_to_keep=getattr(INPUT_ARG_OBJECT, NUM_SVD_MODES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
