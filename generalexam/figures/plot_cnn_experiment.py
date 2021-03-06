"""Plots results of CNN experiment."""

import os.path
import warnings
import numpy
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.machine_learning import evaluation_utils as eval_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
FIGURE_SIZE_PIXELS = int(1e7)

MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.
COLOUR_MAP_OBJECT = pyplot.cm.plasma
WHITE_COLOUR = numpy.full(3, 253. / 255)
BLACK_COLOUR = numpy.full(3, 0.)

FONT_SIZE = 40
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

TOP_EXPERIMENT_DIR_NAME = (
    '/condo/swatwork/ralager/paper_experiment_1000mb/quick_training')

UNIQUE_PREDICTOR_COMBO_STRINGS = [
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 '
    'wet_bulb_potential_temperature_kelvins',
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 temperature_kelvins '
    'specific_humidity_kg_kg01',
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 temperature_kelvins '
    'specific_humidity_kg_kg01 wet_bulb_potential_temperature_kelvins',
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 '
    'wet_bulb_potential_temperature_kelvins height_m_asl',
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 temperature_kelvins '
    'specific_humidity_kg_kg01 height_m_asl',
    'u_wind_grid_relative_m_s01 v_wind_grid_relative_m_s01 temperature_kelvins '
    'specific_humidity_kg_kg01 wet_bulb_potential_temperature_kelvins '
    'height_m_asl'
]

UNIQUE_HALF_IMAGE_SIZES = numpy.array([4, 8, 12, 16], dtype=int)
UNIQUE_DROPOUT_FRACTIONS = numpy.array([0.25, 0.5])

UNIQUE_IMAGE_SIZE_STRINGS = [
    r'{0:d}'.format(2*n + 1) for n in UNIQUE_HALF_IMAGE_SIZES
]
UNIQUE_PREDICTOR_ABBREV_STRINGS = [
    r'$u, v, \theta_w$',
    r'$u, v, T, q$',
    r'$u, v, T, q, \theta_w$',
    r'$u, v, \theta_w$, Z',
    r'$u, v, T, q, Z$',
    'All'
]

IMAGE_SIZE_AXIS_LABEL = 'Grid size'
PREDICTORS_AXIS_LABEL = ''


def _plot_scores_as_grid(
        score_matrix, colour_map_object, min_colour_value, max_colour_value,
        x_tick_labels, x_axis_label, x_axis_text_colour, y_tick_labels,
        y_axis_label, y_axis_text_colour, title_string, output_file_name):
    """Plots model scores as 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of model scores.
    :param colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param x_tick_labels: length-N list of string labels.
    :param x_axis_label: String label for the entire x-axis.
    :param x_axis_text_colour: Colour for all text labels along x-axis.
    :param y_tick_labels: length-M list of string labels.
    :param y_axis_label: String label for the entire y-axis.
    :param y_axis_text_colour: Colour for all text labels along y-axis.
    :param title_string: Figure title.
    :param output_file_name: Path to output file (the figure will be saved
        here).
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    score_matrix_to_plot = score_matrix + 0.
    score_matrix_to_plot[numpy.isnan(score_matrix_to_plot)] = 0.
    pyplot.imshow(
        score_matrix_to_plot, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value)

    x_tick_values = numpy.linspace(
        0, score_matrix_to_plot.shape[1] - 1,
        num=score_matrix_to_plot.shape[1], dtype=float)
    pyplot.xticks(x_tick_values, x_tick_labels, color=x_axis_text_colour)
    pyplot.xlabel(x_axis_label, color=x_axis_text_colour)

    y_tick_values = numpy.linspace(
        0, score_matrix_to_plot.shape[0] - 1,
        num=score_matrix_to_plot.shape[0], dtype=float)
    pyplot.yticks(y_tick_values, y_tick_labels, color=y_axis_text_colour)
    pyplot.ylabel(y_axis_label, color=y_axis_text_colour)

    pyplot.title(title_string)
    plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object,
        values_to_colour=score_matrix_to_plot,
        colour_map=colour_map_object, colour_min=min_colour_value,
        colour_max=max_colour_value, orientation='vertical',
        extend_min=True, extend_max=True, font_size=FONT_SIZE)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name)


def _run():
    """Plots results of CNN experiment.

    This is effectively the main method.

    :raises: ValueError: if any Unix command fails.
    """

    num_predictor_combos = len(UNIQUE_PREDICTOR_COMBO_STRINGS)
    num_image_sizes = len(UNIQUE_HALF_IMAGE_SIZES)
    num_dropout_fractions = len(UNIQUE_DROPOUT_FRACTIONS)

    gerrity_score_matrix = numpy.full(
        (num_predictor_combos, num_image_sizes, num_dropout_fractions),
        numpy.nan)
    peirce_score_matrix = gerrity_score_matrix + 0.
    hss_matrix = gerrity_score_matrix + 0.
    accuracy_matrix = gerrity_score_matrix + 0.

    for i in range(num_predictor_combos):
        for j in range(num_image_sizes):
            for k in range(num_dropout_fractions):
                this_num_predictors = len(
                    UNIQUE_PREDICTOR_COMBO_STRINGS[i].split())
                this_eval_file_name = (
                    '{0:s}/{1:s}_init-num-filters={2:d}_half-image-size-px='
                    '{3:d}_num-conv-layer-sets={4:d}_dropout={5:.2f}/validation'
                    '/model_evaluation.p'
                ).format(
                    TOP_EXPERIMENT_DIR_NAME,
                    UNIQUE_PREDICTOR_COMBO_STRINGS[i].replace(
                        '_', '-').replace(' ', '_'),
                    this_num_predictors * 8,
                    UNIQUE_HALF_IMAGE_SIZES[j],
                    2 + int(UNIQUE_HALF_IMAGE_SIZES[j] > 8),
                    UNIQUE_DROPOUT_FRACTIONS[k]
                )

                if not os.path.isfile(this_eval_file_name):
                    warning_string = (
                        'POTENTIAL PROBLEM.  Cannot find file expected at: '
                        '"{0:s}"'
                    ).format(this_eval_file_name)
                    warnings.warn(warning_string)
                    continue

                print 'Reading data from: "{0:s}"...'.format(
                    this_eval_file_name)
                this_evaluation_dict = eval_utils.read_evaluation_results(
                    this_eval_file_name)

                gerrity_score_matrix[i, j, k] = this_evaluation_dict[
                    eval_utils.GERRITY_SCORE_KEY]
                peirce_score_matrix[i, j, k] = this_evaluation_dict[
                    eval_utils.PEIRCE_SCORE_KEY]
                hss_matrix[i, j, k] = this_evaluation_dict[
                    eval_utils.HEIDKE_SCORE_KEY]
                accuracy_matrix[i, j, k] = this_evaluation_dict[
                    eval_utils.ACCURACY_KEY]

    print SEPARATOR_STRING
    panel_file_names = numpy.full((4, num_dropout_fractions), '', dtype=object)

    for k in range(num_dropout_fractions):
        if k == 0:
            this_y_axis_text_colour = BLACK_COLOUR + 0.
        else:
            this_y_axis_text_colour = WHITE_COLOUR + 0.
        
        this_title_string = 'Gerrity score; dropout = {0:.2f}'.format(
            UNIQUE_DROPOUT_FRACTIONS[k])
        panel_file_names[0, k] = (
            '{0:s}/gerrity_score_dropout-fraction={1:.2f}.jpg'
        ).format(TOP_EXPERIMENT_DIR_NAME, UNIQUE_DROPOUT_FRACTIONS[k])

        print gerrity_score_matrix[..., k]

        _plot_scores_as_grid(
            score_matrix=gerrity_score_matrix[..., k],
            colour_map_object=COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                gerrity_score_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                gerrity_score_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_IMAGE_SIZE_STRINGS,
            x_axis_label=IMAGE_SIZE_AXIS_LABEL,
            x_axis_text_colour=WHITE_COLOUR,
            y_tick_labels=UNIQUE_PREDICTOR_ABBREV_STRINGS,
            y_axis_label='',
            y_axis_text_colour=this_y_axis_text_colour,
            title_string=this_title_string,
            output_file_name=panel_file_names[0, k])

        this_title_string = 'Peirce score; dropout = {0:.2f}'.format(
            UNIQUE_DROPOUT_FRACTIONS[k])
        panel_file_names[1, k] = (
            '{0:s}/peirce_score_dropout-fraction={1:.2f}.jpg'
        ).format(TOP_EXPERIMENT_DIR_NAME, UNIQUE_DROPOUT_FRACTIONS[k])

        _plot_scores_as_grid(
            score_matrix=peirce_score_matrix[..., k],
            colour_map_object=COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                peirce_score_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                peirce_score_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_IMAGE_SIZE_STRINGS,
            x_axis_label=IMAGE_SIZE_AXIS_LABEL,
            x_axis_text_colour=WHITE_COLOUR,
            y_tick_labels=UNIQUE_PREDICTOR_ABBREV_STRINGS,
            y_axis_label='',
            y_axis_text_colour=this_y_axis_text_colour,
            title_string=this_title_string,
            output_file_name=panel_file_names[1, k])

        this_title_string = 'Heidke skill score; dropout = {0:.2f}'.format(
            UNIQUE_DROPOUT_FRACTIONS[k])
        panel_file_names[2, k] = (
            '{0:s}/hss_dropout-fraction={1:.2f}.jpg'
        ).format(TOP_EXPERIMENT_DIR_NAME, UNIQUE_DROPOUT_FRACTIONS[k])

        _plot_scores_as_grid(
            score_matrix=hss_matrix[..., k],
            colour_map_object=COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                hss_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                hss_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_IMAGE_SIZE_STRINGS,
            x_axis_label=IMAGE_SIZE_AXIS_LABEL,
            x_axis_text_colour=WHITE_COLOUR,
            y_tick_labels=UNIQUE_PREDICTOR_ABBREV_STRINGS,
            y_axis_label='',
            y_axis_text_colour=this_y_axis_text_colour,
            title_string=this_title_string,
            output_file_name=panel_file_names[2, k])

        this_title_string = 'Accuracy; dropout = {0:.2f}'.format(
            UNIQUE_DROPOUT_FRACTIONS[k])
        panel_file_names[3, k] = (
            '{0:s}/accuracy_dropout-fraction={1:.2f}.jpg'
        ).format(TOP_EXPERIMENT_DIR_NAME, UNIQUE_DROPOUT_FRACTIONS[k])

        _plot_scores_as_grid(
            score_matrix=accuracy_matrix[..., k],
            colour_map_object=COLOUR_MAP_OBJECT,
            min_colour_value=numpy.nanpercentile(
                accuracy_matrix, MIN_COLOUR_PERCENTILE),
            max_colour_value=numpy.nanpercentile(
                accuracy_matrix, MAX_COLOUR_PERCENTILE),
            x_tick_labels=UNIQUE_IMAGE_SIZE_STRINGS,
            x_axis_label=IMAGE_SIZE_AXIS_LABEL,
            x_axis_text_colour=BLACK_COLOUR,
            y_tick_labels=UNIQUE_PREDICTOR_ABBREV_STRINGS,
            y_axis_label='',
            y_axis_text_colour=this_y_axis_text_colour,
            title_string=this_title_string,
            output_file_name=panel_file_names[3, k])

    for m in range(4):
        second_image_object = Image.open(panel_file_names[m, 1])
        command_string = '"{0:s}" "{1:s}" -resize {2:d}x{3:d}\! "{1:s}"'.format(
            imagemagick_utils.DEFAULT_CONVERT_EXE_NAME, panel_file_names[m, 0],
            second_image_object.size[0], second_image_object.size[1])

        print 'Resizing image: "{0:s}"...'.format(panel_file_names[m, 0])
        exit_code = os.system(command_string)
        if exit_code != 0:
            raise ValueError('\nUnix command failed (log messages shown '
                             'above should explain why).')

    concat_file_name = '{0:s}/validation.jpg'.format(TOP_EXPERIMENT_DIR_NAME)
    print 'Concatenating panels to: "{0:s}"...'.format(concat_file_name)

    imagemagick_utils.concatenate_images(
        input_file_names=numpy.ravel(panel_file_names).tolist(),
        output_file_name=concat_file_name, num_panel_rows=4,
        num_panel_columns=num_dropout_fractions)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=FIGURE_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
