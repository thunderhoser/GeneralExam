"""Makes figure with PMM composite of extreme examples.

PMM = probability-matched means

"Extreme examples" include best hits, best correct nulls, worst misses, worst
false alarms, high-probability examples (regardless of true label), and
low-probability examples (regardless of true label).
"""

import os
import copy
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import saliency_maps
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

MEAN_PREDICTOR_MATRIX_KEY = saliency_maps.MEAN_PREDICTOR_MATRIX_KEY
MODEL_FILE_KEY = saliency_maps.MODEL_FILE_KEY

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

MAIN_FONT_SIZE = 25
AXES_TITLE_FONT_SIZE = 25
COLOUR_BAR_FONT_SIZE = 32
COLOUR_BAR_LENGTH = 0.8

WIND_BARB_COLOUR = numpy.full(3, 152. / 255)
NON_WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('YlOrRd')

CONVERT_EXE_NAME = '/usr/bin/convert'
FIGURE_TITLE_FONT_SIZE = 100
FIGURE_TITLE_FONT_TYPE = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILES_ARG_NAME = 'input_composite_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
ADD_FIGURE_TITLES_ARG_NAME = 'add_figure_titles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each should contain a PMM composite over '
    'many examples.  Specifically, each should be a Pickle file with one '
    'dictionary, containing the keys "{0:s}" and "{1:s}".'
).format(MEAN_PREDICTOR_MATRIX_KEY, MODEL_FILE_KEY)

COMPOSITE_NAMES_HELP_STRING = (
    'List of PMM-composite names (one per input file).  The list should be '
    'space-separated.  In each list item, underscores will be replaced with '
    'spaces.'
)
ADD_FIGURE_TITLES_HELP_STRING = (
    'Boolean flag.  If 1, will add large title above the figure for each '
    'composite, using ImageMagick.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=COMPOSITE_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ADD_FIGURE_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_FIGURE_TITLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_composite(pickle_file_name):
    """Reads PMM composite of examples from Pickle file.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param pickle_file_name: Path to input file.
    :return: mean_predictor_matrix: M-by-N-by-C numpy array of mean predictor
        values.
    :return: cnn_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(pickle_file_name))
    file_handle = open(pickle_file_name, 'rb')
    composite_dict = pickle.load(file_handle)
    file_handle.close()

    mean_predictor_matrix = composite_dict[MEAN_PREDICTOR_MATRIX_KEY]
    cnn_file_name = composite_dict[MODEL_FILE_KEY]
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)

    return mean_predictor_matrix, cnn_metadata_dict


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
        FIGURE_TITLE_FONT_SIZE, FIGURE_TITLE_FONT_TYPE,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_composite(
        composite_file_name, composite_name_abbrev, composite_name_verbose,
        add_figure_title, output_dir_name):
    """Plot one composite.

    :param composite_file_name: Path to input file.  Will be read by
        `_read_composite`.
    :param composite_name_abbrev: Abbreviated name for composite.  Will be used
        in names of output files.
    :param composite_name_verbose: Verbose name for composite.  Will be used as
        figure title.
    :param add_figure_title: See documentation at top of file.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    :return: figure_file_name: Path to image file created by this method.
    """

    mean_predictor_matrix, cnn_metadata_dict = _read_composite(
        composite_file_name
    )

    predictor_names = copy.deepcopy(cnn_metadata_dict[cnn.PREDICTOR_NAMES_KEY])
    pressure_levels_mb = copy.deepcopy(
        cnn_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    )

    wind_flags = numpy.array(
        [n in WIND_FIELD_NAMES for n in predictor_names], dtype=bool
    )
    wind_indices = numpy.where(wind_flags)[0]

    panel_file_names = []

    for scalar_field_name in SCALAR_FIELD_NAMES:
        one_cbar_per_panel = False

        if scalar_field_name == predictor_utils.PRESSURE_NAME:
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

        elif scalar_field_name == predictor_utils.HEIGHT_NAME:
            scalar_field_indices = numpy.where(numpy.logical_and(
                numpy.array(predictor_names) == predictor_utils.HEIGHT_NAME,
                pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
            ))[0]

        elif scalar_field_name == predictor_utils.WET_BULB_THETA_NAME:
            plot_theta_w = not (
                predictor_utils.TEMPERATURE_NAME in predictor_names and
                predictor_utils.SPECIFIC_HUMIDITY_NAME in predictor_names
            )

            if plot_theta_w:
                scalar_field_indices = numpy.where(
                    numpy.array(predictor_names) == scalar_field_name
                )[0]
            else:
                scalar_field_indices = numpy.array([], dtype=int)

        else:
            scalar_field_indices = numpy.where(
                numpy.array(predictor_names) == scalar_field_name
            )[0]

        if len(scalar_field_indices) == 0:
            continue

        channel_indices = numpy.concatenate((
            wind_indices, scalar_field_indices
        ))

        example_dict = {
            examples_io.PREDICTOR_MATRIX_KEY: numpy.expand_dims(
                mean_predictor_matrix[..., channel_indices], axis=0
            ),
            examples_io.PREDICTOR_NAMES_KEY: [
                predictor_names[k] for k in channel_indices
            ],
            examples_io.PRESSURE_LEVELS_KEY: pressure_levels_mb[channel_indices]
        }

        handle_dict = plot_examples.plot_composite_example(
            example_dict=example_dict, plot_wind_as_barbs=True,
            non_wind_colour_map_object=NON_WIND_COLOUR_MAP_OBJECT,
            num_panel_rows=len(scalar_field_indices), add_titles=True,
            one_cbar_per_panel=one_cbar_per_panel,
            colour_bar_length=COLOUR_BAR_LENGTH / len(scalar_field_indices),
            main_font_size=MAIN_FONT_SIZE,
            title_font_size=AXES_TITLE_FONT_SIZE,
            colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
            wind_barb_colour=WIND_BARB_COLOUR
        )

        figure_object = handle_dict[plot_examples.FIGURE_OBJECT_KEY]

        output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, composite_name_abbrev,
            scalar_field_name.replace('_', '-')
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

    if not add_figure_title:
        return figure_file_name

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


def _run(composite_file_names, composite_names, add_figure_titles,
         output_dir_name):
    """Makes figure with PMM composite of extreme examples.

    This is effectively the main method.

    :param composite_file_names: See documentation at top of file.
    :param composite_names: Same.
    :param add_figure_titles: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_composites = len(composite_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )

    composite_names_abbrev = [
        n.replace('_', '-').lower() for n in composite_names
    ]
    composite_names_verbose = [n.replace('_', ' ') for n in composite_names]

    panel_file_names = [None] * num_composites

    for i in range(num_composites):
        panel_file_names[i] = _plot_composite(
            composite_file_name=composite_file_names[i],
            composite_name_abbrev=composite_names_abbrev[i],
            composite_name_verbose=composite_names_verbose[i],
            add_figure_title=add_figure_titles, output_dir_name=output_dir_name
        )

        print('\n')

    figure_file_name = '{0:s}/extreme_examples_concat.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=figure_file_name,
        num_panel_rows=num_composites, num_panel_columns=1,
        border_width_pixels=100
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        composite_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        add_figure_titles=bool(
            getattr(INPUT_ARG_OBJECT, ADD_FIGURE_TITLES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
