"""Plots contingency tables for pixelwise evaluation (PWE)."""

import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

OVERLAY_FONT_SIZE = 50
MAIN_COLOUR = numpy.array([55, 126, 184], dtype=float) / 255
ANNOTATION_COLOUR = numpy.array([77, 175, 74], dtype=float) / 255

LARGE_NUMBER = 1e10
COLOUR_MAP_OBJECT = pyplot.cm.binary

FONT_SIZE = 30
LINE_WIDTH = 4

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('axes', linewidth=LINE_WIDTH)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

BINARY_VARIABLE_NAME_MATRIX = numpy.array([['$a$', '$b$'],
                                           ['$c$', '$d$']], dtype=object)
BINARY_VARIABLE_NAME_MATRIX = numpy.transpose(BINARY_VARIABLE_NAME_MATRIX)

TERNARY_VARIABLE_NAME_MATRIX = numpy.array(
    [['$n_{11}$', '$n_{12}$', '$n_{13}$'],
     ['$n_{21}$', '$n_{22}$', '$n_{23}$'],
     ['$n_{31}$', '$n_{32}$', '$n_{33}$']], dtype=object)
TERNARY_VARIABLE_NAME_MATRIX = numpy.transpose(TERNARY_VARIABLE_NAME_MATRIX)

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'contingency_tables')
BINARY_CT_FILE_NAME = '{0:s}/binary_contingency_table.jpg'.format(
    OUTPUT_DIR_NAME)
TERNARY_CT_FILE_NAME = '{0:s}/ternary_contingency_table.jpg'.format(
    OUTPUT_DIR_NAME)
CONCAT_FILE_NAME = '{0:s}/contingency_tables.jpg'.format(OUTPUT_DIR_NAME)


def _plot_binary_table():
    """Plots binary contingency table."""

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        numpy.zeros((2, 2)), cmap=COLOUR_MAP_OBJECT, vmin=LARGE_NUMBER,
        vmax=LARGE_NUMBER + 1, axes=axes_object, origin='upper')

    tick_locations = numpy.array([0, 1], dtype=int)
    tick_labels = ['Yes', 'No']
    pyplot.xticks(tick_locations, tick_labels)
    pyplot.xlabel('Actual')
    pyplot.yticks(tick_locations, tick_labels)
    pyplot.ylabel('Predicted')

    for i in range(2):
        for j in range(2):
            axes_object.text(
                i, j, str(BINARY_VARIABLE_NAME_MATRIX[i, j]),
                fontsize=OVERLAY_FONT_SIZE, color=MAIN_COLOUR,
                horizontalalignment='center', verticalalignment='center')

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string='(a)',
        font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(BINARY_CT_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=BINARY_CT_FILE_NAME)
    pyplot.savefig(BINARY_CT_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=BINARY_CT_FILE_NAME,
                                      output_file_name=BINARY_CT_FILE_NAME)


def _plot_ternary_table():
    """Plots ternary contingency table."""

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        numpy.zeros((3, 3)), cmap=COLOUR_MAP_OBJECT, vmin=LARGE_NUMBER,
        vmax=LARGE_NUMBER + 1, axes=axes_object, origin='upper')

    tick_locations = numpy.array([0, 1, 2], dtype=int)
    tick_labels = ['NF', 'WF', 'CF']
    pyplot.xticks(tick_locations, tick_labels)
    pyplot.xlabel('Actual')
    pyplot.yticks(tick_locations, tick_labels)
    pyplot.ylabel('Predicted')

    for i in range(3):
        for j in range(3):
            axes_object.text(
                i, j, str(TERNARY_VARIABLE_NAME_MATRIX[i, j]),
                fontsize=OVERLAY_FONT_SIZE, color=MAIN_COLOUR,
                horizontalalignment='center', verticalalignment='center')

    # plotting_utils.annotate_axes(
    #     axes_object=axes_object, annotation_string='(b)',
    #     font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(TERNARY_CT_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=TERNARY_CT_FILE_NAME)
    pyplot.savefig(TERNARY_CT_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=TERNARY_CT_FILE_NAME,
                                      output_file_name=TERNARY_CT_FILE_NAME)


if __name__ == '__main__':
    _plot_binary_table()
    _plot_ternary_table()

    print('Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME))
    imagemagick_utils.concatenate_images(
        input_file_names=[BINARY_CT_FILE_NAME, TERNARY_CT_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=1,
        num_panel_columns=2, output_size_pixels=OUTPUT_SIZE_PIXELS)
