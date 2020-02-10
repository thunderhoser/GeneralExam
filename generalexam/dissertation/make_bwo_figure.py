"""Makes nice figure with backwards-optimization results."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import backwards_optimization as backwards_opt
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_SURFACE_PRESSURE_MB = predictor_utils.DUMMY_SURFACE_PRESSURE_MB

WIND_FIELD_NAMES = [
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]
SCALAR_FIELD_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.WET_BULB_THETA_NAME,
    predictor_utils.PRESSURE_NAME, predictor_utils.HEIGHT_NAME
]

MEAN_INPUT_MATRIX_KEY = backwards_opt.MEAN_INPUT_MATRIX_KEY
MEAN_OUTPUT_MATRIX_KEY = backwards_opt.MEAN_OUTPUT_MATRIX_KEY
MODEL_FILE_KEY = backwards_opt.MODEL_FILE_KEY

MAX_DIFF_PERCENTILE = 99.
WIND_BARB_COLOUR = numpy.full(3, 152. / 255)
DIFF_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
NON_WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('YlOrRd')

AXES_TITLE_FONT_SIZE = 25
COLOUR_BAR_FONT_SIZE = 35
COLOUR_BAR_LENGTH = 0.8

CONVERT_EXE_NAME = '/usr/bin/convert'
FIGURE_TITLE_FONT_SIZE = 100
FIGURE_TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILE_ARG_NAME = 'input_bwo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `backwards_optimization.read_file`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_bwo_file(bwo_file_name):
    """Reads backwards-optimization results from file.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictor variables)

    :param bwo_file_name: Path to input file (will be read by
        `backwards_optimization.read_file`).
    :return: mean_before_matrix: 1-by-M-by-N-by-C numpy array with denormalized
        inputs (predictors before optimization).
    :return: mean_after_matrix: Same but with denormalized outputs (after
        optimization).
    :return: cnn_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(bwo_file_name))
    bwo_dictionary = backwards_opt.read_file(bwo_file_name)[0]

    mean_before_matrix = numpy.expand_dims(
        bwo_dictionary[MEAN_INPUT_MATRIX_KEY], axis=0
    )
    mean_after_matrix = numpy.expand_dims(
        bwo_dictionary[MEAN_OUTPUT_MATRIX_KEY], axis=0
    )

    cnn_file_name = bwo_dictionary[MODEL_FILE_KEY]
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)

    return mean_before_matrix, mean_after_matrix, cnn_metadata_dict


def _overlay_text(
        image_file_name, x_offset_from_center_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_center_px: Center-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -gravity north -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name,
        FIGURE_TITLE_FONT_SIZE, FIGURE_TITLE_FONT_NAME,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_one_composite(bwo_file_name, composite_name_abbrev, output_dir_name):
    """Plots one composite -- either diff ("before minus after") or "after".

    :param bwo_file_name: Path to input file.  Will be read by `_read_bwo_file`.
    :param composite_name_abbrev: Name of composite being plotted ("before", "after",
        or "difference").
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: main_figure_file_name: Path to main image file created by this
        method.
    """

    mean_before_matrix, mean_after_matrix, cnn_metadata_dict = (
        _read_bwo_file(bwo_file_name)
    )

    if composite_name_abbrev == 'difference':
        matrix_to_plot = mean_after_matrix - mean_before_matrix
        composite_name_verbose = '(c) Difference'
    elif composite_name_abbrev == 'after':
        matrix_to_plot = mean_after_matrix
        composite_name_verbose = '(b) Synthetic image'
    else:
        matrix_to_plot = mean_before_matrix
        composite_name_verbose = '(a) Original image'

    predictor_names = cnn_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    pressure_levels_mb = cnn_metadata_dict[cnn.PRESSURE_LEVELS_KEY]

    wind_flags = numpy.array(
        [n in WIND_FIELD_NAMES for n in predictor_names], dtype=bool
    )
    wind_indices = numpy.where(wind_flags)[0]

    panel_file_names = []

    for this_field_name in SCALAR_FIELD_NAMES:
        one_cbar_per_panel = False

        if this_field_name == predictor_utils.PRESSURE_NAME:
            gph_flags = numpy.logical_and(
                numpy.array(predictor_names) == predictor_utils.HEIGHT_NAME,
                pressure_levels_mb != DUMMY_SURFACE_PRESSURE_MB
            )

            pressure_flags = numpy.logical_and(
                numpy.array(predictor_names) == predictor_utils.PRESSURE_NAME,
                pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
            )

            scalar_field_indices = numpy.where(
                numpy.logical_or(gph_flags, pressure_flags)
            )[0]

            one_cbar_per_panel = True

        elif this_field_name == predictor_utils.HEIGHT_NAME:
            scalar_field_indices = numpy.where(numpy.logical_and(
                numpy.array(predictor_names) == predictor_utils.HEIGHT_NAME,
                pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
            ))[0]

        elif this_field_name == predictor_utils.WET_BULB_THETA_NAME:
            plot_theta_w = not (
                predictor_utils.TEMPERATURE_NAME in predictor_names and
                predictor_utils.SPECIFIC_HUMIDITY_NAME in predictor_names
            )

            if plot_theta_w:
                scalar_field_indices = numpy.where(
                    numpy.array(predictor_names) == this_field_name
                )[0]
            else:
                scalar_field_indices = numpy.array([], dtype=int)

        else:
            scalar_field_indices = numpy.where(
                numpy.array(predictor_names) == this_field_name
            )[0]

        if len(scalar_field_indices) == 0:
            continue

        channel_indices = numpy.concatenate((
            scalar_field_indices, wind_indices
        ))

        example_dict = {
            examples_io.PREDICTOR_MATRIX_KEY:
                matrix_to_plot[..., channel_indices],
            examples_io.PREDICTOR_NAMES_KEY: [
                predictor_names[k] for k in channel_indices
            ],
            examples_io.PRESSURE_LEVELS_KEY: pressure_levels_mb[channel_indices]
        }

        num_panel_rows = len(scalar_field_indices)
        colour_map_object = (
            DIFF_COLOUR_MAP_OBJECT if composite_name_abbrev == 'difference'
            else NON_WIND_COLOUR_MAP_OBJECT
        )

        handle_dict = plot_examples.plot_composite_example(
            example_dict=example_dict, plot_wind_as_barbs=True,
            non_wind_colour_map_object=colour_map_object,
            add_titles=True, plot_diffs=composite_name_abbrev == 'difference',
            num_panel_rows=num_panel_rows,
            one_cbar_per_panel=one_cbar_per_panel,
            colour_bar_length=COLOUR_BAR_LENGTH / num_panel_rows,
            title_font_size=AXES_TITLE_FONT_SIZE,
            colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
            wind_barb_colour=WIND_BARB_COLOUR
        )

        figure_object = handle_dict[plot_examples.FIGURE_OBJECT_KEY]

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, composite_name_abbrev,
            this_field_name.replace('_', '-')
        )
        panel_file_names.append(output_file_name)

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    figure_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, composite_name_abbrev
    )
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=len(panel_file_names),
        border_width_pixels=50
    )
    imagemagick_utils.resize_image(
        input_file_name=figure_file_name,
        output_file_name=figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name,
        output_file_name=figure_file_name,
        border_width_pixels=FIGURE_TITLE_FONT_SIZE + 25
    )
    _overlay_text(
        image_file_name=figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=composite_name_verbose
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name,
        output_file_name=figure_file_name,
        border_width_pixels=10
    )

    return figure_file_name


def _run(bwo_file_name, output_dir_name):
    """Makes nice figure with backwards-optimization results.

    This is effectively the main method.

    :param bwo_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    before_file_name = _plot_one_composite(
        bwo_file_name=bwo_file_name, composite_name_abbrev='before',
        output_dir_name=output_dir_name
    )
    print('\n')

    after_file_name = _plot_one_composite(
        bwo_file_name=bwo_file_name, composite_name_abbrev='after',
        output_dir_name=output_dir_name
    )
    print('\n')

    difference_file_name = _plot_one_composite(
        bwo_file_name=bwo_file_name, composite_name_abbrev='difference',
        output_dir_name=output_dir_name
    )
    print('\n')

    figure_file_name = '{0:s}/bwo_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[
            before_file_name, after_file_name, difference_file_name
        ],
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        bwo_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
