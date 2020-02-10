"""Makes nice figure with saliency maps."""

import os
import copy
import pickle
import argparse
import numpy
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import saliency_maps
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

MEAN_PREDICTOR_MATRIX_KEY = saliency_maps.MEAN_PREDICTOR_MATRIX_KEY
MEAN_SALIENCY_MATRIX_KEY = saliency_maps.MEAN_SALIENCY_MATRIX_KEY
MODEL_FILE_KEY = saliency_maps.MODEL_FILE_KEY

NONE_STRINGS = ['none', 'None']
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

AXES_TITLE_FONT_SIZE = 25
PREDICTOR_CBAR_FONT_SIZE = 35
SALIENCY_CBAR_FONT_SIZE = 25
COLOUR_BAR_LENGTH = 0.8

WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
ACTUAL_WIND_BARB_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
NON_WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('YlOrRd')

CONVERT_EXE_NAME = '/usr/bin/convert'
FIGURE_TITLE_FONT_SIZE = 100
FIGURE_TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

SALIENCY_FILES_ARG_NAME = 'input_saliency_file_names'
MC_FILES_ARG_NAME = 'input_monte_carlo_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_VALUES_ARG_NAME = 'max_colour_values'
HALF_NUM_CONTOURS_ARG_NAME = 'half_num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILES_HELP_STRING = (
    'List of saliency files (each will be read by `saliency_maps.read_file`).'
)
MC_FILES_HELP_STRING = (
    'List of files with Monte Carlo significance (one per saliency file).  Each'
    ' will be read by `_read_monte_carlo_test`.  If you do not want to plot '
    'significance at all, leave this argument alone.  If you do not want to '
    'plot significance for the [i]th composite, make the [i]th string "None".'
)
COMPOSITE_NAMES_HELP_STRING = (
    'List of composite names (one for each saliency file).  This list must be '
    'space-separated, but after reading the list, underscores within each item '
    'will be replaced by spaces.'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for saliency.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MAX_VALUES_HELP_STRING = (
    'Max absolute saliency in each colour scheme (one per file).'
)
HALF_NUM_CONTOURS_HELP_STRING = (
    'Number of saliency contours on either side of zero (positive and '
    'negative).'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth saliency maps, make this negative.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=SALIENCY_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MC_FILES_ARG_NAME, type=str, nargs='+', required=False, default=[''],
    help=MC_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=COMPOSITE_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=1., help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_one_composite(saliency_file_name, smoothing_radius_grid_cells,
                        monte_carlo_file_name=None):
    """Reads predictor and saliency maps for one composite.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictor variables)

    :param saliency_file_name: Path to input file (will be read by
        `saliency_maps.read_file`).
    :param smoothing_radius_grid_cells: Radius for Gaussian smoother, used only
        for saliency map.
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :return: mean_predictor_matrix: 1-by-M-by-N-by-C numpy array with
        denormalized predictors.
    :return: mean_saliency_matrix: 1-by-M-by-N-by-C numpy array with saliency
        values.
    :return: cnn_metadata_dict: Dictionary returned by `cnn.read_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency_maps.read_file(saliency_file_name)[0]

    mean_predictor_matrix = numpy.expand_dims(
        saliency_dict[MEAN_PREDICTOR_MATRIX_KEY], axis=0
    )
    mean_saliency_matrix = numpy.expand_dims(
        saliency_dict[MEAN_SALIENCY_MATRIX_KEY], axis=0
    )

    cnn_file_name = saliency_dict[MODEL_FILE_KEY]
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)
    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)

    if monte_carlo_file_name is None:
        significance_matrix = numpy.full(
            mean_saliency_matrix.shape, False, dtype=bool
        )
    else:
        print('Reading Monte Carlo test from: "{0:s}"...'.format(
            monte_carlo_file_name
        ))

        this_file_handle = open(monte_carlo_file_name, 'rb')
        monte_carlo_dict = pickle.load(this_file_handle)
        this_file_handle.close()

        significance_matrix = numpy.logical_or(
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0] <
            monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][0],
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0] >
            monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][0]
        )
        significance_matrix = numpy.expand_dims(significance_matrix, axis=0)

    print('Fraction of significant differences: {0:.4f}'.format(
        numpy.mean(significance_matrix.astype(float))
    ))

    if smoothing_radius_grid_cells is None:
        return (
            mean_predictor_matrix, mean_saliency_matrix, significance_matrix,
            cnn_metadata_dict
        )

    print((
        'Smoothing saliency maps with Gaussian filter (e-folding radius of '
        '{0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_channels = mean_saliency_matrix.shape[-1]

    for k in range(num_channels):
        mean_saliency_matrix[0, ..., k] = general_utils.apply_gaussian_filter(
            input_matrix=mean_saliency_matrix[0, ..., k],
            e_folding_radius_grid_cells=smoothing_radius_grid_cells
        )

    return (
        mean_predictor_matrix, mean_saliency_matrix, significance_matrix,
        cnn_metadata_dict
    )


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


def _plot_saliency_with_wind_barbs(
        u_wind_saliency_matrix, v_wind_saliency_matrix, axes_object,
        colour_map_object, max_colour_value):
    """Plots 2-D saliency map with wind barbs.

    M = number of rows in grid
    N = number of columns in grid

    :param u_wind_saliency_matrix: M-by-N numpy array of saliency values for
        u-wind.
    :param v_wind_saliency_matrix: M-by-N numpy array of saliency values for
        v-wind.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param max_colour_value: Max saliency in colour scheme.
    """

    num_grid_rows = u_wind_saliency_matrix.shape[0]
    num_grid_columns = v_wind_saliency_matrix.shape[1]

    x_coords_unique = 0.5 + numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )
    y_coords_unique = 0.5 + numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    x_coord_matrix, y_coord_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=x_coords_unique, y_unique_metres=y_coords_unique
    )

    x_coords = numpy.ravel(x_coord_matrix[::2, ::2])
    y_coords = numpy.ravel(y_coord_matrix[::2, ::2])
    u_saliency_values = numpy.ravel(u_wind_saliency_matrix[::2, ::2])
    v_saliency_values = numpy.ravel(v_wind_saliency_matrix[::2, ::2])

    saliency_magnitudes = numpy.sqrt(
        u_saliency_values ** 2 + v_saliency_values ** 2
    )

    size_dict = {
        'emptybarb': EMPTY_WIND_BARB_RADIUS
    }
    increment_dict = {
        'half': 1e-6,
        'full': 2 * numpy.max(saliency_magnitudes),
        'flag': 10 * numpy.max(saliency_magnitudes)
    }

    axes_object.barbs(
        x_coords, y_coords,
        u_saliency_values, v_saliency_values, saliency_magnitudes,
        sizes=size_dict, barb_increments=increment_dict,
        length=WIND_BARB_LENGTH, linewidth=2, fill_empty=True, rounding=True,
        cmap=colour_map_object, clim=numpy.array([0., max_colour_value])
    )


def _plot_one_composite(
        saliency_file_name, plot_significance, composite_name_abbrev,
        composite_name_verbose, colour_map_object, max_colour_value,
        half_num_contours, smoothing_radius_grid_cells, output_dir_name,
        monte_carlo_file_name=None):
    """Plots one composite.

    :param saliency_file_name: Path to input file.  Will be read by
        `_read_one_composite`.
    :param plot_significance: Boolean flag.  If True, will plot significance.
    :param composite_name_abbrev: Abbreviated name for composite.  Will be used
        in names of output files.
    :param composite_name_verbose: Verbose name for composite.  Will be used as
        figure title.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :return: main_figure_file_name: Path to main image file created by this
        method.
    """

    (
        mean_predictor_matrix, mean_saliency_matrix, significance_matrix,
        cnn_metadata_dict
    ) = (
        _read_one_composite(
            saliency_file_name=saliency_file_name,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells,
            monte_carlo_file_name=monte_carlo_file_name)
    )

    predictor_names = cnn_metadata_dict[cnn.PREDICTOR_NAMES_KEY]
    pressure_levels_mb = cnn_metadata_dict[cnn.PRESSURE_LEVELS_KEY]

    wind_flags = numpy.array(
        [n in WIND_FIELD_NAMES for n in predictor_names], dtype=bool
    )
    wind_indices = numpy.where(wind_flags)[0]

    panel_file_names = []
    last_panel_file_name = None

    if plot_significance:
        all_field_names = SCALAR_FIELD_NAMES + WIND_FIELD_NAMES
    else:
        all_field_names = SCALAR_FIELD_NAMES

    for this_field_name in all_field_names:
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

            main_field_indices = numpy.where(
                numpy.logical_or(gph_flags, pressure_flags)
            )[0]

            one_cbar_per_panel = True

        elif this_field_name == predictor_utils.HEIGHT_NAME:
            main_field_indices = numpy.where(numpy.logical_and(
                numpy.array(predictor_names) == predictor_utils.HEIGHT_NAME,
                pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
            ))[0]

        elif this_field_name == predictor_utils.WET_BULB_THETA_NAME:
            plot_theta_w = not (
                predictor_utils.TEMPERATURE_NAME in predictor_names and
                predictor_utils.SPECIFIC_HUMIDITY_NAME in predictor_names
            )

            if plot_theta_w:
                main_field_indices = numpy.where(
                    numpy.array(predictor_names) == this_field_name
                )[0]
            else:
                main_field_indices = numpy.array([], dtype=int)

        else:
            main_field_indices = numpy.where(
                numpy.array(predictor_names) == this_field_name
            )[0]

        if len(main_field_indices) == 0:
            continue

        if plot_significance:
            channel_indices = main_field_indices + 0
        else:
            channel_indices = numpy.concatenate((
                main_field_indices, wind_indices
            ))

        example_dict = {
            examples_io.PREDICTOR_MATRIX_KEY:
                mean_predictor_matrix[..., channel_indices],
            examples_io.PREDICTOR_NAMES_KEY: [
                predictor_names[k] for k in channel_indices
            ],
            examples_io.PRESSURE_LEVELS_KEY: pressure_levels_mb[channel_indices]
        }

        num_panel_rows = len(main_field_indices)

        handle_dict = plot_examples.plot_composite_example(
            example_dict=copy.deepcopy(example_dict),
            plot_wind_as_barbs=not plot_significance,
            non_wind_colour_map_object=NON_WIND_COLOUR_MAP_OBJECT,
            num_panel_rows=num_panel_rows, add_titles=True,
            one_cbar_per_panel=one_cbar_per_panel,
            colour_bar_length=COLOUR_BAR_LENGTH / num_panel_rows,
            colour_bar_font_size=PREDICTOR_CBAR_FONT_SIZE,
            title_font_size=AXES_TITLE_FONT_SIZE,
            wind_barb_colour=ACTUAL_WIND_BARB_COLOUR,
            wind_colour_map_object=WIND_COLOUR_MAP_OBJECT
        )

        axes_object_matrix = handle_dict[plot_examples.AXES_OBJECTS_KEY]
        figure_object = handle_dict[plot_examples.FIGURE_OBJECT_KEY]

        this_matrix = (
            mean_saliency_matrix[0, ...][..., main_field_indices]
        )
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=this_matrix,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_colour_value,
            contour_interval=max_colour_value / half_num_contours
        )

        if plot_significance:
            this_matrix = (
                significance_matrix[0, ...][..., main_field_indices]
            )
            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=this_matrix,
                axes_object_matrix=axes_object_matrix, marker_size=8
            )

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

        if last_panel_file_name is not None or plot_significance:
            continue

        main_example_dict = {
            examples_io.PREDICTOR_MATRIX_KEY:
                mean_predictor_matrix[..., main_field_indices],
            examples_io.PREDICTOR_NAMES_KEY: [
                predictor_names[k] for k in main_field_indices
            ],
            examples_io.PRESSURE_LEVELS_KEY:
                pressure_levels_mb[main_field_indices]
        }

        handle_dict = plot_examples.plot_composite_example(
            example_dict=copy.deepcopy(main_example_dict),
            plot_wind_as_barbs=False,
            non_wind_colour_map_object=NON_WIND_COLOUR_MAP_OBJECT,
            num_panel_rows=num_panel_rows, add_titles=True,
            one_cbar_per_panel=one_cbar_per_panel,
            colour_bar_length=COLOUR_BAR_LENGTH / num_panel_rows,
            colour_bar_font_size=PREDICTOR_CBAR_FONT_SIZE,
            title_font_size=AXES_TITLE_FONT_SIZE
        )

        axes_object_matrix = handle_dict[plot_examples.AXES_OBJECTS_KEY]
        figure_object = handle_dict[plot_examples.FIGURE_OBJECT_KEY]

        for i in range(len(main_field_indices)):
            this_u_wind_index = numpy.where(numpy.logical_and(
                example_dict[examples_io.PRESSURE_LEVELS_KEY] ==
                example_dict[examples_io.PRESSURE_LEVELS_KEY][i],
                numpy.array(example_dict[examples_io.PREDICTOR_NAMES_KEY]) ==
                predictor_utils.U_WIND_GRID_RELATIVE_NAME
            ))[0][0]

            this_v_wind_index = numpy.where(numpy.logical_and(
                example_dict[examples_io.PRESSURE_LEVELS_KEY] ==
                example_dict[examples_io.PRESSURE_LEVELS_KEY][i],
                numpy.array(example_dict[examples_io.PREDICTOR_NAMES_KEY]) ==
                predictor_utils.V_WIND_GRID_RELATIVE_NAME
            ))[0][0]

            u_wind_saliency_matrix = (
                mean_saliency_matrix[0, ...][..., this_u_wind_index]
            )
            v_wind_saliency_matrix = (
                mean_saliency_matrix[0, ...][..., this_v_wind_index]
            )

            _plot_saliency_with_wind_barbs(
                u_wind_saliency_matrix=u_wind_saliency_matrix,
                v_wind_saliency_matrix=v_wind_saliency_matrix,
                axes_object=axes_object_matrix[i, 0],
                colour_map_object=colour_map_object,
                max_colour_value=max_colour_value
            )

        last_panel_file_name = '{0:s}/{1:s}_wind.jpg'.format(
            output_dir_name, composite_name_abbrev
        )

        print('Saving figure to: "{0:s}"...'.format(last_panel_file_name))
        figure_object.savefig(
            last_panel_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    if last_panel_file_name is not None and not plot_significance:
        panel_file_names.append(last_panel_file_name)

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


def _add_colour_bar(figure_file_name, colour_map_object, max_colour_value,
                    temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param max_colour_value: Max value in colour scheme.
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    dummy_values = numpy.array([0., max_colour_value])

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        min_value=0., max_value=max_colour_value,
        orientation_string='vertical', fraction_of_axis_length=1.25,
        extend_min=False, extend_max=True, font_size=SALIENCY_CBAR_FONT_SIZE
    )

    colour_bar_object.set_label(
        'Absolute saliency', fontsize=SALIENCY_CBAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()

    if max_colour_value <= 0.0005:
        tick_strings = ['{0:.5f}'.format(v) for v in tick_values]
    elif max_colour_value <= 0.005:
        tick_strings = ['{0:.4f}'.format(v) for v in tick_values]
    elif max_colour_value <= 0.05:
        tick_strings = ['{0:.3f}'.format(v) for v in tick_values]
    else:
        tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/saliency_colour-bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(saliency_file_names, monte_carlo_file_names, composite_names,
         colour_map_name, max_colour_values, half_num_contours,
         smoothing_radius_grid_cells, output_dir_name):
    """Makes nice figure with saliency maps.

    This is effectively the main method.

    :param saliency_file_names: See documentation at top of file.
    :param monte_carlo_file_names: Same.
    :param composite_names: Same.
    :param colour_map_name: Same.
    :param max_colour_values: Same.
    :param half_num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)
    error_checking.assert_is_geq(half_num_contours, 5)

    num_composites = len(saliency_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )

    if len(monte_carlo_file_names) == 1 and monte_carlo_file_names[0] == '':
        monte_carlo_file_names = [NONE_STRINGS[0]] * num_composites

    error_checking.assert_is_numpy_array(
        numpy.array(monte_carlo_file_names), exact_dimensions=expected_dim
    )
    monte_carlo_file_names = [
        None if f in NONE_STRINGS else f for f in monte_carlo_file_names
    ]

    plot_significance = any([f is not None for f in monte_carlo_file_names])

    error_checking.assert_is_greater_numpy_array(max_colour_values, 0.)
    error_checking.assert_is_numpy_array(
        max_colour_values, exact_dimensions=expected_dim
    )

    composite_names_abbrev = [
        n.replace('_', '-').lower() for n in composite_names
    ]
    composite_names_verbose = [
        '({0:s}) {1:s}'.format(
            chr(ord('a') + i), composite_names[i].replace('_', ' ')
        )
        for i in range(num_composites)
    ]

    panel_file_names = [None] * num_composites

    for i in range(num_composites):
        panel_file_names[i] = _plot_one_composite(
            saliency_file_name=saliency_file_names[i],
            monte_carlo_file_name=monte_carlo_file_names[i],
            plot_significance=plot_significance,
            composite_name_abbrev=composite_names_abbrev[i],
            composite_name_verbose=composite_names_verbose[i],
            colour_map_object=colour_map_object,
            max_colour_value=max_colour_values[i],
            half_num_contours=half_num_contours,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells,
            output_dir_name=output_dir_name
        )

        _add_colour_bar(
            figure_file_name=panel_file_names[i],
            colour_map_object=colour_map_object,
            max_colour_value=max_colour_values[i],
            temporary_dir_name=output_dir_name
        )

        print('\n')

    figure_file_name = '{0:s}/saliency_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    num_panel_columns = int(numpy.floor(
        numpy.sqrt(num_composites)
    ))
    num_panel_rows = int(numpy.ceil(
        float(num_composites) / num_panel_columns
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_names=getattr(INPUT_ARG_OBJECT, SALIENCY_FILES_ARG_NAME),
        monte_carlo_file_names=getattr(INPUT_ARG_OBJECT, MC_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
