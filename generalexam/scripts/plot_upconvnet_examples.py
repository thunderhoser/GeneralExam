"""Plots one or more examples and their upconvnet reconstructions."""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from generalexam.machine_learning import cnn
from generalexam.machine_learning import upconvnet
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'top_output_dir_name'
NUM_EXAMPLES_ARG_NAME = plot_examples.NUM_EXAMPLES_ARG_NAME
PLOT_BARBS_ARG_NAME = plot_examples.PLOT_BARBS_ARG_NAME
WIND_BARB_COLOUR_ARG_NAME = plot_examples.WIND_BARB_COLOUR_ARG_NAME
WIND_CMAP_ARG_NAME = plot_examples.WIND_CMAP_ARG_NAME
NON_WIND_CMAP_ARG_NAME = plot_examples.NON_WIND_CMAP_ARG_NAME
NUM_PANEL_ROWS_ARG_NAME = plot_examples.NUM_PANEL_ROWS_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME
MAIN_FONT_SIZE_ARG_NAME = plot_examples.MAIN_FONT_SIZE_ARG_NAME
TITLE_FONT_SIZE_ARG_NAME = plot_examples.TITLE_FONT_SIZE_ARG_NAME
CBAR_FONT_SIZE_ARG_NAME = plot_examples.CBAR_FONT_SIZE_ARG_NAME
RESOLUTION_ARG_NAME = plot_examples.RESOLUTION_ARG_NAME

PREDICTION_FILE_HELP_STRING = (
    'Path to file with upconvnet predictions (reconstructed images).  Will be '
    'read by `upconvnet.read_predictions`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with actual (non-reconstructed) images.  Files'
    ' therein will be found by `learning_examples_io.find_file` and read by '
    '`learning_examples_io.read_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=plot_examples.NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BARBS_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.PLOT_BARBS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_BARB_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=plot_examples.DEFAULT_WIND_BARB_COLOUR * 255,
    help=plot_examples.WIND_BARB_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WIND_CMAP_ARG_NAME, type=str, required=False,
    default=plot_examples.DEFAULT_WIND_CMAP_NAME,
    help=plot_examples.WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NON_WIND_CMAP_ARG_NAME, type=str, required=False,
    default=plot_examples.DEFAULT_NON_WIND_CMAP_NAME,
    help=plot_examples.NON_WIND_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=plot_examples.NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_CBAR_LENGTH,
    help=plot_examples.CBAR_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_MAIN_FONT_SIZE,
    help=plot_examples.MAIN_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TITLE_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_TITLE_FONT_SIZE,
    help=plot_examples.TITLE_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_FONT_SIZE_ARG_NAME, type=float, required=False,
    default=plot_examples.DEFAULT_CBAR_FONT_SIZE,
    help=plot_examples.CBAR_FONT_SIZE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RESOLUTION_ARG_NAME, type=int, required=False,
    default=plot_examples.DEFAULT_RESOLUTION_DPI,
    help=plot_examples.RESOLUTION_HELP_STRING)


def _run(prediction_file_name, top_example_dir_name, num_examples_to_plot,
         plot_wind_as_barbs, wind_barb_colour, wind_colour_map_name,
         non_wind_colour_map_name, num_panel_rows, add_titles,
         colour_bar_length, main_font_size, title_font_size,
         colour_bar_font_size, figure_resolution_dpi, top_output_dir_name):
    """Plots one or more examples and their upconvnet reconstructions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param num_examples_to_plot: Same.
    :param plot_wind_as_barbs: Same.
    :param wind_barb_colour: Same.
    :param wind_colour_map_name: Same.
    :param non_wind_colour_map_name: Same.
    :param num_panel_rows: Same.
    :param add_titles: Same.
    :param colour_bar_length: Same.
    :param main_font_size: Same.
    :param title_font_size: Same.
    :param colour_bar_font_size: Same.
    :param figure_resolution_dpi: Same.
    :param top_output_dir_name: Same.
    """

    # Read data.
    print('Reading reconstructed images from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = upconvnet.read_predictions(prediction_file_name)

    reconstructed_image_matrix = prediction_dict[
        upconvnet.RECON_IMAGE_MATRIX_KEY]
    example_id_strings = prediction_dict[upconvnet.EXAMPLE_IDS_KEY]
    upconvnet_file_name = prediction_dict[upconvnet.UPCONVNET_FILE_KEY]
    upconvnet_metafile_name = cnn.find_metafile(upconvnet_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        upconvnet_metafile_name
    ))
    upconvnet_metadata_dict = upconvnet.read_model_metadata(
        upconvnet_metafile_name)
    cnn_file_name = upconvnet_metadata_dict[upconvnet.CNN_FILE_KEY]
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)
    predictor_names = cnn_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    pressure_levels_mb = cnn_metadata_dict[cnn.PRESSURE_LEVELS_KEY]

    print(SEPARATOR_STRING)
    actual_example_dict = examples_io.read_specific_examples_many_files(
        top_example_dir_name=top_example_dir_name,
        example_id_strings=example_id_strings,
        predictor_names_to_keep=predictor_names,
        pressure_levels_to_keep_mb=pressure_levels_mb,
        num_half_rows_to_keep=cnn_metadata_dict[cnn.NUM_HALF_ROWS_KEY],
        num_half_columns_to_keep=cnn_metadata_dict[cnn.NUM_HALF_COLUMNS_KEY]
    )
    print(SEPARATOR_STRING)

    # Denormalize data.
    print('Denormalizing predictors...')

    normalization_type_string = actual_example_dict[
        examples_io.NORMALIZATION_TYPE_KEY]

    normalization_dict = {
        ml_utils.MIN_VALUE_MATRIX_KEY: None,
        ml_utils.MAX_VALUE_MATRIX_KEY: None,
        ml_utils.MEAN_VALUE_MATRIX_KEY: None,
        ml_utils.STDEV_MATRIX_KEY: None
    }

    if normalization_type_string == ml_utils.Z_SCORE_STRING:
        normalization_dict[ml_utils.MEAN_VALUE_MATRIX_KEY] = (
            actual_example_dict[examples_io.FIRST_NORM_PARAM_KEY]
        )

        normalization_dict[ml_utils.STDEV_MATRIX_KEY] = actual_example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]
    else:
        normalization_dict[ml_utils.MIN_VALUE_MATRIX_KEY] = actual_example_dict[
            examples_io.FIRST_NORM_PARAM_KEY]

        normalization_dict[ml_utils.MAX_VALUE_MATRIX_KEY] = actual_example_dict[
            examples_io.SECOND_NORM_PARAM_KEY]

    actual_image_matrix = ml_utils.denormalize_predictors(
        predictor_matrix=actual_example_dict[examples_io.PREDICTOR_MATRIX_KEY],
        normalization_dict=normalization_dict)

    reconstructed_image_matrix = ml_utils.denormalize_predictors(
        predictor_matrix=reconstructed_image_matrix,
        normalization_dict=normalization_dict)

    actual_example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
        actual_image_matrix + 0.
    )

    reconstructed_example_dict = {
        examples_io.PREDICTOR_MATRIX_KEY: reconstructed_image_matrix + 0.,
        examples_io.PREDICTOR_NAMES_KEY: predictor_names,
        examples_io.PRESSURE_LEVELS_KEY: pressure_levels_mb
    }

    valid_times_unix_sec, row_indices, column_indices = (
        examples_io.example_ids_to_metadata(example_id_strings)
    )

    reconstructed_example_dict[examples_io.VALID_TIMES_KEY] = (
        valid_times_unix_sec
    )
    reconstructed_example_dict[examples_io.ROW_INDICES_KEY] = row_indices
    reconstructed_example_dict[examples_io.COLUMN_INDICES_KEY] = column_indices

    diff_example_dict = copy.deepcopy(actual_example_dict)
    diff_example_dict[examples_io.PREDICTOR_MATRIX_KEY] = (
        reconstructed_image_matrix - actual_image_matrix
    )

    # Do plotting.
    print(SEPARATOR_STRING)

    plot_examples.plot_real_examples(
        example_dict=actual_example_dict,
        output_dir_name='{0:s}/actual_images'.format(top_output_dir_name),
        num_examples_to_plot=num_examples_to_plot,
        plot_diffs=False, plot_wind_as_barbs=plot_wind_as_barbs,
        wind_barb_colour=wind_barb_colour,
        wind_colour_map_name=wind_colour_map_name,
        non_wind_colour_map_name=non_wind_colour_map_name,
        num_panel_rows=num_panel_rows, add_titles=add_titles,
        colour_bar_length=colour_bar_length,
        main_font_size=main_font_size, title_font_size=title_font_size,
        colour_bar_font_size=colour_bar_font_size,
        figure_resolution_dpi=figure_resolution_dpi)

    print(SEPARATOR_STRING)

    plot_examples.plot_real_examples(
        example_dict=reconstructed_example_dict,
        output_dir_name=
        '{0:s}/reconstructed_images'.format(top_output_dir_name),
        num_examples_to_plot=num_examples_to_plot,
        plot_diffs=False, plot_wind_as_barbs=plot_wind_as_barbs,
        wind_barb_colour=wind_barb_colour,
        wind_colour_map_name=wind_colour_map_name,
        non_wind_colour_map_name=non_wind_colour_map_name,
        num_panel_rows=num_panel_rows, add_titles=add_titles,
        colour_bar_length=colour_bar_length,
        main_font_size=main_font_size, title_font_size=title_font_size,
        colour_bar_font_size=colour_bar_font_size,
        figure_resolution_dpi=figure_resolution_dpi)

    print(SEPARATOR_STRING)

    plot_examples.plot_real_examples(
        example_dict=diff_example_dict,
        output_dir_name='{0:s}/differences'.format(top_output_dir_name),
        num_examples_to_plot=num_examples_to_plot,
        plot_diffs=True, plot_wind_as_barbs=plot_wind_as_barbs,
        wind_barb_colour=wind_barb_colour,
        wind_colour_map_name=wind_colour_map_name,
        non_wind_colour_map_name=non_wind_colour_map_name,
        num_panel_rows=num_panel_rows, add_titles=add_titles,
        colour_bar_length=colour_bar_length,
        main_font_size=main_font_size, title_font_size=title_font_size,
        colour_bar_font_size=colour_bar_font_size,
        figure_resolution_dpi=figure_resolution_dpi)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        num_examples_to_plot=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        plot_wind_as_barbs=bool(getattr(INPUT_ARG_OBJECT, PLOT_BARBS_ARG_NAME)),
        wind_barb_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, WIND_BARB_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        wind_colour_map_name=getattr(INPUT_ARG_OBJECT, WIND_CMAP_ARG_NAME),
        non_wind_colour_map_name=getattr(
            INPUT_ARG_OBJECT, NON_WIND_CMAP_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        main_font_size=getattr(INPUT_ARG_OBJECT, MAIN_FONT_SIZE_ARG_NAME),
        title_font_size=getattr(INPUT_ARG_OBJECT, TITLE_FONT_SIZE_ARG_NAME),
        colour_bar_font_size=getattr(INPUT_ARG_OBJECT, CBAR_FONT_SIZE_ARG_NAME),
        figure_resolution_dpi=getattr(INPUT_ARG_OBJECT, RESOLUTION_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
