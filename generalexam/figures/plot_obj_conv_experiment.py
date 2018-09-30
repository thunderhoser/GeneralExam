"""Plots results of object-conversion experiment."""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.evaluation import object_based_evaluation as object_eval

METRES_TO_KM = 1e-3
METRES2_TO_MILLION_KM2 = 1e-12

UNIQUE_BINARIZATION_THRESHOLDS = numpy.linspace(-0.2, 0.2, num=9) + 0.553
UNIQUE_MIN_AREAS_METRES2 = (numpy.linspace(0.1, 1., num=10) * 1e12).astype(int)
UNIQUE_MIN_LENGTHS_METRES = (numpy.linspace(0.1, 1., num=10) * 1e6).astype(int)

ANNOTATION_STRING_BY_THRESHOLD = [
    '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)'
]
UNIQUE_MIN_AREA_STRINGS = [
    '{0:.1f}'.format(a * METRES2_TO_MILLION_KM2)
    for a in UNIQUE_MIN_AREAS_METRES2
]
UNIQUE_MIN_LENGTH_STRINGS = [
    '{0:d}'.format(int(numpy.round(l * METRES_TO_KM)))
    for l in UNIQUE_MIN_LENGTHS_METRES
]

MIN_AREA_AXIS_LABEL = r'Minimum area of frontal region ($10^6$ km$^2$)'
MIN_LENGTH_AXIS_LABEL = 'Minimum end-to-end length (km)'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
FIGURE_SIZE_PIXELS = int(1e7)

MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.
SEQUENTIAL_COLOUR_MAP_OBJECT = pyplot.cm.plasma
DIVERGENT_COLOUR_MAP_OBJECT = pyplot.cm.seismic

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = (
    'Name of directory with predicted and observed objects.  Should contain one'
    ' file for each experiment trial.  These files will be read by '
    '`object_eval.read_evaluation_results`.')
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_scores_as_grid(
        score_matrix, colour_map_object, min_colour_value, max_colour_value,
        x_tick_labels, x_axis_label, y_tick_labels, y_axis_label,
        annotation_string, output_file_name):
    """Plots model scores as 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of model scores.
    :param colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param x_tick_labels: length-N list of string labels.
    :param x_axis_label: String label for the entire x-axis.
    :param y_tick_labels: length-M list of string labels.
    :param y_axis_label: String label for the entire y-axis.
    :param annotation_string: Text annotation (will be placed in top left of
        figure).
    :param output_file_name: Path to output file (the figure will be saved
        here).
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    score_matrix = numpy.ma.masked_where(
        numpy.isnan(score_matrix), score_matrix)
    pyplot.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value)

    num_columns = score_matrix.shape[1]
    x_tick_values = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float)
    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.xlabel(x_axis_label)

    num_rows = score_matrix.shape[0]
    y_tick_values = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=float)
    pyplot.yticks(y_tick_values, y_tick_labels)
    pyplot.ylabel(y_axis_label)

    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=score_matrix,
        colour_map=colour_map_object, colour_min=min_colour_value,
        colour_max=max_colour_value, orientation='vertical', extend_min=True,
        extend_max=True, fraction_of_axis_length=0.8)

    print '\n\n'
    print score_matrix
    print '\n\n'

    plotting_utils.annotate_axes(
        axes_object=axes_object, annotation_string=annotation_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run(experiment_dir_name, output_dir_name):
    """Plots results of object-conversion experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    num_binarization_thresholds = len(UNIQUE_BINARIZATION_THRESHOLDS)
    num_min_areas = len(UNIQUE_MIN_AREAS_METRES2)
    num_min_lengths = len(UNIQUE_MIN_LENGTHS_METRES)

    csi_matrix = numpy.full(
        (num_binarization_thresholds, num_min_areas, num_min_lengths),
        numpy.nan)
    pod_matrix = csi_matrix + 0.
    success_ratio_matrix = csi_matrix + 0.
    frequency_bias_matrix = csi_matrix + 0.

    for i in range(num_binarization_thresholds):
        for j in range(num_min_areas):
            for k in range(num_min_lengths):
                this_file_name = (
                    '{0:s}/obe_validation_binarization-threshold={1:.4f}_'
                    'min-area-metres2={2:012d}_min-length-metres={3:06d}.p'
                ).format(
                    experiment_dir_name, UNIQUE_BINARIZATION_THRESHOLDS[i],
                    UNIQUE_MIN_AREAS_METRES2[j], UNIQUE_MIN_LENGTHS_METRES[k]
                )

                if not os.path.isfile(this_file_name):
                    warning_string = (
                        'Cannot find file.  Expected at: "{0:s}"'
                    ).format(this_file_name)
                    warnings.warn(warning_string)
                    continue

                print 'Reading data from: "{0:s}"...'.format(this_file_name)
                this_evaluation_dict = object_eval.read_evaluation_results(
                    this_file_name)

                csi_matrix[i, j, k] = this_evaluation_dict[
                    object_eval.BINARY_CSI_KEY]
                pod_matrix[i, j, k] = this_evaluation_dict[
                    object_eval.BINARY_POD_KEY]
                success_ratio_matrix[i, j, k] = this_evaluation_dict[
                    object_eval.BINARY_SUCCESS_RATIO_KEY]
                frequency_bias_matrix[i, j, k] = this_evaluation_dict[
                    object_eval.BINARY_FREQUENCY_BIAS_KEY]

    this_offset = numpy.nanpercentile(
        numpy.absolute(frequency_bias_matrix - 1), MAX_COLOUR_PERCENTILE)
    min_colour_frequency_bias = 1 - this_offset
    max_colour_frequency_bias = 1 + this_offset

    csi_file_names = []
    pod_file_names = []
    success_ratio_file_names = []
    frequency_bias_file_names = []

    for i in range(num_binarization_thresholds):
        this_file_name = '{0:s}/csi_binarization-threshold={1:.4f}.jpg'.format(
            output_dir_name, UNIQUE_BINARIZATION_THRESHOLDS[i])
        csi_file_names.append(this_file_name)

        _plot_scores_as_grid(
            score_matrix=csi_matrix[i, ...],
            colour_map_object=SEQUENTIAL_COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                csi_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                csi_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_MIN_LENGTH_STRINGS,
            x_axis_label=MIN_LENGTH_AXIS_LABEL,
            y_tick_labels=UNIQUE_MIN_AREA_STRINGS,
            y_axis_label=MIN_AREA_AXIS_LABEL,
            annotation_string=ANNOTATION_STRING_BY_THRESHOLD[i],
            output_file_name=csi_file_names[-1])

        this_file_name = '{0:s}/pod_binarization-threshold={1:.4f}.jpg'.format(
            output_dir_name, UNIQUE_BINARIZATION_THRESHOLDS[i])
        pod_file_names.append(this_file_name)

        _plot_scores_as_grid(
            score_matrix=pod_matrix[i, ...],
            colour_map_object=SEQUENTIAL_COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                pod_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                pod_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_MIN_LENGTH_STRINGS,
            x_axis_label=MIN_LENGTH_AXIS_LABEL,
            y_tick_labels=UNIQUE_MIN_AREA_STRINGS,
            y_axis_label=MIN_AREA_AXIS_LABEL,
            annotation_string=ANNOTATION_STRING_BY_THRESHOLD[i],
            output_file_name=pod_file_names[-1])

        this_file_name = (
            '{0:s}/success_ratio_binarization-threshold={1:.4f}.jpg'
        ).format(output_dir_name, UNIQUE_BINARIZATION_THRESHOLDS[i])
        success_ratio_file_names.append(this_file_name)

        _plot_scores_as_grid(
            score_matrix=success_ratio_matrix[i, ...],
            colour_map_object=SEQUENTIAL_COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                success_ratio_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                success_ratio_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_MIN_LENGTH_STRINGS,
            x_axis_label=MIN_LENGTH_AXIS_LABEL,
            y_tick_labels=UNIQUE_MIN_AREA_STRINGS,
            y_axis_label=MIN_AREA_AXIS_LABEL,
            annotation_string=ANNOTATION_STRING_BY_THRESHOLD[i],
            output_file_name=success_ratio_file_names[-1])

        this_file_name = (
            '{0:s}/frequency_bias_binarization-threshold={1:.4f}.jpg'
        ).format(output_dir_name, UNIQUE_BINARIZATION_THRESHOLDS[i])
        frequency_bias_file_names.append(this_file_name)

        _plot_scores_as_grid(
            score_matrix=frequency_bias_matrix[i, ...],
            colour_map_object=DIVERGENT_COLOUR_MAP_OBJECT,
            min_colour_value=min_colour_frequency_bias,
            max_colour_value=max_colour_frequency_bias,
            x_tick_labels=UNIQUE_MIN_LENGTH_STRINGS,
            x_axis_label=MIN_LENGTH_AXIS_LABEL,
            y_tick_labels=UNIQUE_MIN_AREA_STRINGS,
            y_axis_label=MIN_AREA_AXIS_LABEL,
            annotation_string=ANNOTATION_STRING_BY_THRESHOLD[i],
            output_file_name=frequency_bias_file_names[-1])

        print '\n'

    this_file_name = '{0:s}/csi.jpg'.format(output_dir_name)
    print 'Concatenating panels to: "{0:s}"...'.format(this_file_name)
    imagemagick_utils.concatenate_images(
        input_file_names=csi_file_names, output_file_name=this_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS,
        output_size_pixels=FIGURE_SIZE_PIXELS)

    this_file_name = '{0:s}/pod.jpg'.format(output_dir_name)
    print 'Concatenating panels to: "{0:s}"...'.format(this_file_name)
    imagemagick_utils.concatenate_images(
        input_file_names=pod_file_names, output_file_name=this_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS,
        output_size_pixels=FIGURE_SIZE_PIXELS)

    this_file_name = '{0:s}/success_ratio.jpg'.format(output_dir_name)
    print 'Concatenating panels to: "{0:s}"...'.format(this_file_name)
    imagemagick_utils.concatenate_images(
        input_file_names=success_ratio_file_names,
        output_file_name=this_file_name, num_panel_rows=NUM_PANEL_ROWS,
        num_panel_columns=NUM_PANEL_COLUMNS,
        output_size_pixels=FIGURE_SIZE_PIXELS)

    this_file_name = '{0:s}/frequency_bias.jpg'.format(output_dir_name)
    print 'Concatenating panels to: "{0:s}"...'.format(this_file_name)
    imagemagick_utils.concatenate_images(
        input_file_names=frequency_bias_file_names,
        output_file_name=this_file_name, num_panel_rows=NUM_PANEL_ROWS,
        num_panel_columns=NUM_PANEL_COLUMNS,
        output_size_pixels=FIGURE_SIZE_PIXELS)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))