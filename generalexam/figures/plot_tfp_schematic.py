"""Creates TFP (thermal front parameter) schematic."""

import numpy
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils

NUM_GRID_ROWS = 100
NUM_GRID_COLUMNS = 100
FIRST_ROW_IN_FRONT = 39
LAST_ROW_IN_FRONT = 59
MIN_TEMPERATURE_KELVINS = 260.
FRONT_GRADIENT_KELVINS_PER_ROW = 1.
NON_FRONT_GRADIENT_KELVINS_PER_ROW = 0.1

X_OFFSET_LEFT = 10
X_OFFSET_MIDDLE = 50
X_OFFSET_RIGHT = 80
Y_OFFSET_BOTTOM = 20
Y_OFFSET_TOP = 80
ARROW_LENGTH = 10
ARROW_WIDTH = 4
ARROW_HEAD_WIDTH = 2
ARROW_HEAD_LENGTH = 3
OVERLAY_COLOUR = numpy.full(3, 0.)
OVERLAY_FONT_SIZE = 50
COLOUR_MAP_OPACITY = 0.75

THETA_GRADIENT_STRING = r'$\vec{\nabla}\theta$'
GRADIENT_GRADIENT_STRING = (
    r'$\vec{\nabla} \left \| \left \| \vec{\nabla}\theta \right \| \right \|$')
TFP_POSITIVE_STRING = r'TFP $>$ 0'
TFP_NEGATIVE_STRING = r'TFP $<$ 0'
TFP_MIN_STRING = 'TFP = min'
TFP_MAX_STRING = 'TFP = max'

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'tfp_schematic/tfp_schematic.jpg')


def _create_temperature_grid():
    """Creates temperature grid.

    M = number of rows in grid
    N = number of columns in grid

    :return: temperature_matrix_kelvins: M-by-N numpy array of temperatures.
    """

    temperature_matrix_kelvins = numpy.full(
        (NUM_GRID_ROWS, NUM_GRID_COLUMNS), MIN_TEMPERATURE_KELVINS)

    for i in range(1, NUM_GRID_ROWS):
        if FIRST_ROW_IN_FRONT < i <= LAST_ROW_IN_FRONT:
            this_diff_kelvins = FRONT_GRADIENT_KELVINS_PER_ROW + 0.
        else:
            this_diff_kelvins = NON_FRONT_GRADIENT_KELVINS_PER_ROW + 0.

        temperature_matrix_kelvins[i, :] = (
            temperature_matrix_kelvins[i - 1, :] + this_diff_kelvins
        )

    return numpy.flipud(temperature_matrix_kelvins)


def _add_transparency(foreground_rgb_colour, background_rgb_colour, opacity):
    """Adds transparency to foreground colour.

    :param foreground_rgb_colour: Foreground colour (length-3 numpy array).
    :param background_rgb_colour: Background colour (length-3 numpy array).
    :param opacity: Opacity (in range 0...1).
    :return: foreground_rgb_colour: New foreground colour (length-3 numpy array
        with transparency built in).
    """

    return (
        foreground_rgb_colour * opacity + background_rgb_colour * (1. - opacity)
    )


def _create_colour_scheme(temperature_matrix_kelvins):
    """Creates colour scheme for temperature map.

    M = number of rows in grid
    N = number of columns in grid

    :param temperature_matrix_kelvins: M-by-N numpy array of temperatures.
    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    """

    min_colour_value = numpy.min(temperature_matrix_kelvins)
    max_colour_value = numpy.max(temperature_matrix_kelvins)
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value)

    orig_colour_map_object = pyplot.cm.rainbow
    num_colours = orig_colour_map_object.N
    test_values = numpy.linspace(
        min_colour_value, max_colour_value, num=num_colours)

    rgb_matrix = orig_colour_map_object(colour_norm_object(test_values))
    rgb_matrix = rgb_matrix[..., :-1]
    for i in range(num_colours):
        rgb_matrix[i, :] = _add_transparency(
            foreground_rgb_colour=rgb_matrix[i, :],
            background_rgb_colour=numpy.full(3, 1.), opacity=COLOUR_MAP_OPACITY)

    return matplotlib.colors.ListedColormap(rgb_matrix)


def _plot_temperature_grid(temperature_matrix_kelvins):
    """Plots temperature grid as colour map.

    M = number of rows in grid
    N = number of columns in grid

    :param temperature_matrix_kelvins: M-by-N numpy array of temperatures.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    """

    (edge_temperature_matrix_kelvins, edge_x_coords_metres, edge_y_coords_metres
    ) = grids.xy_field_grid_points_to_edges(
        field_matrix=temperature_matrix_kelvins, x_min_metres=0.,
        y_min_metres=0., x_spacing_metres=1., y_spacing_metres=1.)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    pyplot.pcolormesh(
        edge_x_coords_metres, edge_y_coords_metres,
        edge_temperature_matrix_kelvins,
        cmap=_create_colour_scheme(temperature_matrix_kelvins),
        vmin=numpy.min(temperature_matrix_kelvins),
        vmax=numpy.max(temperature_matrix_kelvins), shading='flat',
        edgecolors='None', axes=axes_object)

    pyplot.xticks([], [])
    pyplot.yticks([], [])
    return axes_object


def _overlay_text(axes_object):
    """Overlays text on colour map of temperature.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    """

    axes_object.arrow(
        X_OFFSET_LEFT, Y_OFFSET_BOTTOM, 0, ARROW_LENGTH, linewidth=ARROW_WIDTH,
        head_length=ARROW_HEAD_LENGTH, head_width=ARROW_HEAD_WIDTH,
        facecolor=OVERLAY_COLOUR, edgecolor=OVERLAY_COLOUR)
    axes_object.text(
        X_OFFSET_LEFT, Y_OFFSET_BOTTOM - 2.5, GRADIENT_GRADIENT_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='top')

    axes_object.arrow(
        X_OFFSET_LEFT, Y_OFFSET_TOP, 0, -ARROW_LENGTH, linewidth=ARROW_WIDTH,
        head_length=ARROW_HEAD_LENGTH, head_width=ARROW_HEAD_WIDTH,
        facecolor=OVERLAY_COLOUR, edgecolor=OVERLAY_COLOUR)
    axes_object.text(
        X_OFFSET_LEFT, Y_OFFSET_TOP, GRADIENT_GRADIENT_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='bottom')

    axes_object.arrow(
        X_OFFSET_MIDDLE, Y_OFFSET_BOTTOM, 0, -ARROW_LENGTH,
        linewidth=ARROW_WIDTH, head_length=ARROW_HEAD_LENGTH,
        head_width=ARROW_HEAD_WIDTH, facecolor=OVERLAY_COLOUR,
        edgecolor=OVERLAY_COLOUR)
    axes_object.text(
        X_OFFSET_MIDDLE, Y_OFFSET_BOTTOM, THETA_GRADIENT_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='bottom')

    axes_object.arrow(
        X_OFFSET_MIDDLE, Y_OFFSET_TOP, 0, -ARROW_LENGTH, linewidth=ARROW_WIDTH,
        head_length=ARROW_HEAD_LENGTH, head_width=ARROW_HEAD_WIDTH,
        facecolor=OVERLAY_COLOUR, edgecolor=OVERLAY_COLOUR)
    axes_object.text(
        X_OFFSET_MIDDLE, Y_OFFSET_TOP, THETA_GRADIENT_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='bottom')

    axes_object.text(
        X_OFFSET_RIGHT, Y_OFFSET_BOTTOM - ARROW_LENGTH, TFP_POSITIVE_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='bottom')

    axes_object.text(
        X_OFFSET_RIGHT, Y_OFFSET_TOP, TFP_NEGATIVE_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='bottom')

    axes_object.text(
        X_OFFSET_RIGHT, FIRST_ROW_IN_FRONT + 1, TFP_MAX_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='center')

    axes_object.text(
        X_OFFSET_RIGHT, LAST_ROW_IN_FRONT, TFP_MIN_STRING,
        fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_COLOUR,
        horizontalalignment='center', verticalalignment='center')


if __name__ == '__main__':
    TEMPERATURE_MATRIX_KELVINS = _create_temperature_grid()
    AXES_OBJECT = _plot_temperature_grid(TEMPERATURE_MATRIX_KELVINS)
    _overlay_text(AXES_OBJECT)

    print 'Saving figure to: "{0:s}"...'.format(OUTPUT_FILE_NAME)
    file_system_utils.mkdir_recursive_if_necessary(file_name=OUTPUT_FILE_NAME)
    pyplot.savefig(OUTPUT_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=OUTPUT_FILE_NAME, output_file_name=OUTPUT_FILE_NAME)
