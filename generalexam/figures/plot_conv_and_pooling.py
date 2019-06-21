"""Plots schematics of convolution and pooling."""

import numpy
from keras import backend as K
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

LARGE_NUMBER = 1e10

NUM_ROWS_BEFORE_POOLING = 4
NUM_COLUMNS_BEFORE_POOLING = 6
NUM_ROWS_IN_KERNEL = 3
NUM_COLUMNS_IN_KERNEL = 3
NUM_ROWS_AFTER_POOLING = 2
NUM_COLUMNS_AFTER_POOLING = 3

MIN_FEATURE_VALUE = -5
MAX_FEATURE_VALUE = 5
POSSIBLE_KERNEL_VALUES = numpy.array([-1, -0.5, 0, 0.5, 1])
COLOUR_MAP_OBJECT = pyplot.cm.spring
HIGHLIGHTED_VALUE = 0.

LINE_WIDTH = 4
LINE_COLOUR = numpy.full(3, 0.)
OVERLAY_FONT_SIZE = 50
MAIN_COLOUR = numpy.array([55, 126, 184], dtype=float) / 255
ANNOTATION_COLOUR = numpy.array([77, 175, 74], dtype=float) / 255
SPECIAL_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

FONT_SIZE = 30
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

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'conv_and_pooling')
BEFORE_CONV_FILE_NAME = '{0:s}/before_convolution.jpg'.format(OUTPUT_DIR_NAME)
KERNEL_FILE_NAME = '{0:s}/kernel.jpg'.format(OUTPUT_DIR_NAME)
AFTER_CONV_FILE_NAME = '{0:s}/after_convolution.jpg'.format(OUTPUT_DIR_NAME)
AFTER_POOLING_FILE_NAME = '{0:s}/after_pooling.jpg'.format(OUTPUT_DIR_NAME)
CONCAT_FILE_NAME = '{0:s}/conv_and_pooling.jpg'.format(OUTPUT_DIR_NAME)


def _plot_feature_map_before_conv():
    """Plots original feature map (before convolution).

    M = number of rows in grid
    N = number of columns in grid

    :return: feature_matrix: Feature map as M-by-N numpy array.
    """

    feature_matrix = numpy.random.random_integers(
        low=MIN_FEATURE_VALUE, high=MAX_FEATURE_VALUE,
        size=(NUM_ROWS_BEFORE_POOLING, NUM_COLUMNS_BEFORE_POOLING)
    )

    dummy_matrix = numpy.full(
        (NUM_ROWS_BEFORE_POOLING, NUM_COLUMNS_BEFORE_POOLING), numpy.nan)
    dummy_matrix[:3, :3] = HIGHLIGHTED_VALUE
    dummy_matrix = numpy.ma.masked_where(
        numpy.isnan(dummy_matrix), dummy_matrix)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=HIGHLIGHTED_VALUE - 1,
        vmax=HIGHLIGHTED_VALUE, axes=axes_object, origin='upper')
    pyplot.xticks([], [])
    pyplot.yticks([], [])

    for i in range(feature_matrix.shape[1]):
        for j in range(feature_matrix.shape[0]):
            if i == j == 1:
                this_colour = SPECIAL_COLOUR + 0.
            else:
                this_colour = MAIN_COLOUR + 0.

            axes_object.text(
                i, j, '{0:d}'.format(feature_matrix[j, i]),
                fontsize=OVERLAY_FONT_SIZE, color=this_colour,
                horizontalalignment='center', verticalalignment='center')

    # polygon_x_coords = numpy.array([0, 3, 3, 0, 0], dtype=float) - 0.5
    # polygon_y_coords = numpy.array([3, 3, 0, 0, 3], dtype=float) - 0.5
    # axes_object.plot(
    #     polygon_x_coords, polygon_y_coords, color=LINE_COLOUR,
    #     linewidth=LINE_WIDTH)

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string='(a)',
        font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(BEFORE_CONV_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=BEFORE_CONV_FILE_NAME)
    pyplot.savefig(BEFORE_CONV_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=BEFORE_CONV_FILE_NAME,
                                      output_file_name=BEFORE_CONV_FILE_NAME)

    return feature_matrix


def _plot_kernel():
    """Plots convolutional kernel.

    J = number of rows in kernel
    K = number of columns in kernel

    :return: kernel_matrix: Kernel as J-by-K numpy array.
    """

    kernel_matrix = numpy.random.choice(
        a=POSSIBLE_KERNEL_VALUES,
        size=(NUM_ROWS_IN_KERNEL, NUM_COLUMNS_IN_KERNEL), replace=True)

    dummy_matrix = numpy.full(
        (NUM_ROWS_IN_KERNEL, NUM_COLUMNS_IN_KERNEL), HIGHLIGHTED_VALUE)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=HIGHLIGHTED_VALUE - 1,
        vmax=HIGHLIGHTED_VALUE, axes=axes_object, origin='upper')
    pyplot.xticks([], [])
    pyplot.yticks([], [])

    for i in range(kernel_matrix.shape[1]):
        for j in range(kernel_matrix.shape[0]):
            axes_object.text(
                i, j, '{0:.1f}'.format(kernel_matrix[j, i]),
                fontsize=OVERLAY_FONT_SIZE, color=MAIN_COLOUR,
                horizontalalignment='center', verticalalignment='center')

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string='(b)',
        font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(KERNEL_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(file_name=KERNEL_FILE_NAME)
    pyplot.savefig(KERNEL_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=KERNEL_FILE_NAME,
                                      output_file_name=KERNEL_FILE_NAME)

    return kernel_matrix


def _do_convolution(feature_matrix, kernel_matrix):
    """Convolves 2-D feature map with 2-D kernel.

    M = number of rows in feature map
    N = number of columns in feature map
    J = number of rows in kernel
    K = number of columns in kernel

    :param feature_matrix: Feature map as M-by-N numpy array.
    :param kernel_matrix: Kernel as J-by-K numpy array.
    :return: feature_matrix: New feature map (also M x N).
    """

    this_feature_matrix = numpy.expand_dims(feature_matrix, axis=0)
    this_feature_matrix = numpy.expand_dims(this_feature_matrix, axis=-1)
    this_kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)
    this_kernel_matrix = numpy.expand_dims(this_kernel_matrix, axis=-1)

    feature_tensor = K.conv2d(
        x=K.variable(this_feature_matrix),
        kernel=K.variable(this_kernel_matrix), strides=(1, 1), padding='same',
        data_format='channels_last')
    return feature_tensor.eval(session=K.get_session())[0, ..., 0]


def _plot_feature_map_after_conv(feature_matrix):
    """Plots new feature map (after convolution).

    M = number of rows in grid
    N = number of columns in grid

    :param feature_matrix: Feature map as M-by-N numpy array.
    """

    dummy_matrix = numpy.full(feature_matrix.shape, numpy.nan)
    dummy_matrix[:2, :2] = HIGHLIGHTED_VALUE

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=HIGHLIGHTED_VALUE - 1,
        vmax=HIGHLIGHTED_VALUE, axes=axes_object, origin='upper')
    pyplot.xticks([], [])
    pyplot.yticks([], [])

    for i in range(feature_matrix.shape[1]):
        for j in range(feature_matrix.shape[0]):
            if i == j == 1:
                this_colour = SPECIAL_COLOUR + 0.
            else:
                this_colour = MAIN_COLOUR + 0.

            axes_object.text(
                i, j, '{0:.1f}'.format(feature_matrix[j, i]),
                fontsize=OVERLAY_FONT_SIZE, color=this_colour,
                horizontalalignment='center', verticalalignment='center')

    # polygon_x_coords = numpy.array([0, 2, 2, 0, 0], dtype=float) - 0.5
    # polygon_y_coords = numpy.array([2, 2, 0, 0, 2], dtype=float) - 0.5
    # axes_object.plot(
    #     polygon_x_coords, polygon_y_coords, color=LINE_COLOUR,
    #     linewidth=LINE_WIDTH)

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string='(c)',
        font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(AFTER_CONV_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=AFTER_CONV_FILE_NAME)
    pyplot.savefig(AFTER_CONV_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=AFTER_CONV_FILE_NAME,
                                      output_file_name=AFTER_CONV_FILE_NAME)

    return feature_matrix


def _do_max_pooling(feature_matrix):
    """Performs max-pooling on 2-D feature map.

    M = number of rows before pooling
    N = number of columns before pooling
    m = number of rows after pooling
    n = number of columns after pooling

    :param feature_matrix: Feature map as M-by-N numpy array.
    :return: feature_matrix: New feature map (m-by-n).
    """

    this_feature_matrix = numpy.expand_dims(feature_matrix, axis=0)
    this_feature_matrix = numpy.expand_dims(this_feature_matrix, axis=-1)

    feature_tensor = K.pool2d(
        x=K.variable(this_feature_matrix), pool_mode='max', pool_size=(2, 2),
        strides=(2, 2), padding='valid', data_format='channels_last')
    return feature_tensor.eval(session=K.get_session())[0, ..., 0]


def _plot_feature_map_after_pooling(feature_matrix):
    """Plots new feature map (after pooling).

    m = number of rows in grid
    n = number of columns in grid

    :param feature_matrix: Feature map as m-by-n numpy array.
    """

    dummy_matrix = numpy.full(feature_matrix.shape, numpy.nan)
    dummy_matrix[0, 0] = HIGHLIGHTED_VALUE

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    pyplot.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=HIGHLIGHTED_VALUE - 1,
        vmax=HIGHLIGHTED_VALUE, axes=axes_object, origin='upper')
    pyplot.xticks([], [])
    pyplot.yticks([], [])

    for i in range(feature_matrix.shape[1]):
        for j in range(feature_matrix.shape[0]):
            if i == j == 0:
                this_colour = SPECIAL_COLOUR + 0.
            else:
                this_colour = MAIN_COLOUR + 0.

            axes_object.text(
                i, j, '{0:.1f}'.format(feature_matrix[j, i]),
                fontsize=OVERLAY_FONT_SIZE, color=this_colour,
                horizontalalignment='center', verticalalignment='center')

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string='(d)',
        font_colour=ANNOTATION_COLOUR)

    print('Saving figure to: "{0:s}"...'.format(AFTER_POOLING_FILE_NAME))
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=AFTER_POOLING_FILE_NAME)
    pyplot.savefig(AFTER_POOLING_FILE_NAME, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=AFTER_POOLING_FILE_NAME,
                                      output_file_name=AFTER_POOLING_FILE_NAME)

    return feature_matrix


if __name__ == '__main__':
    FEATURE_MATRIX = _plot_feature_map_before_conv()
    KERNEL_MATRIX = _plot_kernel()

    FEATURE_MATRIX = _do_convolution(
        feature_matrix=FEATURE_MATRIX, kernel_matrix=KERNEL_MATRIX)
    _plot_feature_map_after_conv(FEATURE_MATRIX)

    FEATURE_MATRIX = _do_max_pooling(FEATURE_MATRIX)
    _plot_feature_map_after_pooling(FEATURE_MATRIX)

    print('Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME))
    imagemagick_utils.concatenate_images(
        input_file_names=[BEFORE_CONV_FILE_NAME, KERNEL_FILE_NAME,
                          AFTER_POOLING_FILE_NAME, AFTER_CONV_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=2,
        num_panel_columns=2, output_size_pixels=OUTPUT_SIZE_PIXELS)
