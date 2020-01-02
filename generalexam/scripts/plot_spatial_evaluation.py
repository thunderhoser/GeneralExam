"""Plots results of neighbourhood evaluation at each grid point."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import neigh_evaluation

FAR_WEIGHT_FOR_CSI = 0.5

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
BORDER_COLOUR = numpy.full(3, 0.)

MAX_COLOUR_PERCENTILE = 99.
SCORE_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
COUNT_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
CONCAT_ARG_NAME = 'concat_figures'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`neigh_evaluation.read_spatial_results`.'
)
CONCAT_HELP_STRING = (
    'Boolean flag.  If 1, will concatenate all figures into one, using '
    'ImageMagick.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONCAT_ARG_NAME, type=int, required=False, default=1,
    help=CONCAT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_bias_colour_scheme(max_value):
    """Returns colour scheme for frequency bias.

    :param max_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour normalization (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    # TODO(thunderhoser): Put this method in a module.

    orig_colour_map_object = pyplot.get_cmap('seismic')

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_one_score(score_matrix, is_frequency_bias, is_count, output_file_name,
                    title_string=None, panel_letter=None):
    """Plots one score as grid with basemap.

    :param score_matrix: 2-D numpy array of scores.
    :param is_frequency_bias: Boolean flag.
    :param is_count: Boolean flag.
    :param output_file_name: Path to output file (figure will be saved here).
    :param title_string: Title (will be added above figure).  If you do not want
        a title, make this None.
    :param panel_letter: Panel letter.  For example, if the letter is "a", will
        add "(a)" at top-left of figure, assuming that it will eventually be a
        panel in a larger figure.  If you do not want a panel letter, make this
        None.
    """

    num_grid_rows = score_matrix.shape[0]
    num_grid_columns = score_matrix.shape[1]

    full_grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    full_grid_row_limits, full_grid_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name)
    )

    matrix_to_plot = score_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name,
        first_row_in_full_grid=full_grid_row_limits[0],
        last_row_in_full_grid=full_grid_row_limits[1],
        first_column_in_full_grid=full_grid_column_limits[0],
        last_column_in_full_grid=full_grid_column_limits[1]
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS
    )

    if is_frequency_bias:
        this_offset = numpy.nanpercentile(
            numpy.absolute(matrix_to_plot - 1.), MAX_COLOUR_PERCENTILE
        )
        min_colour_value = 0.
        max_colour_value = 1. + this_offset

        colour_map_object, colour_norm_object = _get_bias_colour_scheme(
            max_colour_value
        )
    else:
        if is_count:
            colour_map_object = COUNT_COLOUR_MAP_OBJECT
            min_colour_value = 0.
        else:
            colour_map_object = SCORE_COLOUR_MAP_OBJECT
            min_colour_value = numpy.nanpercentile(
                matrix_to_plot, 100. - MAX_COLOUR_PERCENTILE
            )

        colour_norm_object = None
        max_colour_value = numpy.nanpercentile(
            matrix_to_plot, MAX_COLOUR_PERCENTILE
        )

    nwp_plotting.plot_subgrid(
        field_matrix=matrix_to_plot,
        model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    if is_frequency_bias:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            padding=0.05, orientation_string='horizontal',
            extend_min=False, extend_max=True
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    else:
        colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            padding=0.05, orientation_string='horizontal',
            extend_min=min_colour_value > 1e-6,
            extend_max=max_colour_value < 1. - 1e-6
        )

        tick_values = colour_bar_object.get_ticks()

        if is_count:
            tick_strings = [
                '{0:d}'.format(int(numpy.round(v))) for v in tick_values
            ]
        else:
            tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    if title_string is not None:
        axes_object.set_title(title_string)

    if panel_letter is not None:
        plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(panel_letter)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


def _run(evaluation_file_name, concat_figures, output_dir_name):
    """Plots results of neighbourhood evaluation at each grid point.

    This is effectively the main method.

    :param evaluation_file_name: See documentation at top of file.
    :param concat_figures: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_dict = neigh_evaluation.read_spatial_results(
        evaluation_file_name
    )
    binary_ct_dict_matrix = evaluation_dict[
        neigh_evaluation.BINARY_CT_MATRIX_KEY
    ]

    num_grid_rows = binary_ct_dict_matrix.shape[0]
    num_grid_columns = binary_ct_dict_matrix.shape[1]
    these_dim = (num_grid_rows, num_grid_columns)

    num_actual_matrix = numpy.full(these_dim, numpy.nan)
    num_predicted_matrix = numpy.full(these_dim, numpy.nan)
    pod_matrix = numpy.full(these_dim, numpy.nan)
    far_matrix = numpy.full(these_dim, numpy.nan)
    csi_matrix = numpy.full(these_dim, numpy.nan)
    weighted_csi_matrix = numpy.full(these_dim, numpy.nan)
    frequency_bias_matrix = numpy.full(these_dim, numpy.nan)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_binary_ct_dict = binary_ct_dict_matrix[i, j]

            val = this_binary_ct_dict[neigh_evaluation.NUM_FALSE_POSITIVES_KEY]
            if numpy.isnan(val):
                continue

            num_actual_matrix[i, j] = (
                this_binary_ct_dict[neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY]
                + this_binary_ct_dict[neigh_evaluation.NUM_FALSE_NEGATIVES_KEY]
            )

            num_predicted_matrix[i, j] = (
                this_binary_ct_dict[
                    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY
                ]
                + this_binary_ct_dict[neigh_evaluation.NUM_FALSE_POSITIVES_KEY]
            )

            pod_matrix[i, j] = neigh_evaluation.get_pod(this_binary_ct_dict)
            far_matrix[i, j] = neigh_evaluation.get_far(this_binary_ct_dict)
            csi_matrix[i, j] = neigh_evaluation.get_csi(
                binary_ct_as_dict=this_binary_ct_dict, far_weight=1.
            )
            weighted_csi_matrix[i, j] = neigh_evaluation.get_csi(
                binary_ct_as_dict=this_binary_ct_dict,
                far_weight=FAR_WEIGHT_FOR_CSI
            )
            frequency_bias_matrix[i, j] = (
                neigh_evaluation.get_frequency_bias(this_binary_ct_dict)
            )

    num_actual_file_name = '{0:s}/num_actual.jpg'.format(output_dir_name)
    num_predicted_file_name = '{0:s}/num_predicted.jpg'.format(output_dir_name)
    pod_file_name = '{0:s}/pod.jpg'.format(output_dir_name)
    far_file_name = '{0:s}/far.jpg'.format(output_dir_name)
    frequency_bias_file_name = '{0:s}/frequency_bias.jpg'.format(
        output_dir_name
    )
    csi_file_name = '{0:s}/csi.jpg'.format(output_dir_name)
    weighted_csi_file_name = '{0:s}/weighted_csi.jpg'.format(output_dir_name)

    this_title_string = (
        'Number of actual fronts' if concat_figures else None
    )
    this_panel_letter = 'a' if concat_figures else None
    _plot_one_score(
        score_matrix=num_actual_matrix, is_frequency_bias=False, is_count=True,
        output_file_name=num_actual_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = (
        'Number of predicted fronts' if concat_figures else None
    )
    this_panel_letter = 'b' if concat_figures else None
    _plot_one_score(
        score_matrix=num_predicted_matrix, is_frequency_bias=False,
        is_count=True, output_file_name=num_predicted_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = (
        'Probability of detection (POD)' if concat_figures else None
    )
    this_panel_letter = 'c' if concat_figures else None
    _plot_one_score(
        score_matrix=pod_matrix, is_frequency_bias=False, is_count=False,
        output_file_name=pod_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = (
        'False-alarm ratio (FAR)' if concat_figures else None
    )
    this_panel_letter = 'd' if concat_figures else None
    _plot_one_score(
        score_matrix=far_matrix, is_frequency_bias=False, is_count=False,
        output_file_name=far_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = 'Frequency bias' if concat_figures else None
    this_panel_letter = 'e' if concat_figures else None
    _plot_one_score(
        score_matrix=frequency_bias_matrix, is_frequency_bias=True,
        is_count=False, output_file_name=frequency_bias_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = (
        'Critical success index (CSI)' if concat_figures else None
    )
    this_panel_letter = 'f' if concat_figures else None
    _plot_one_score(
        score_matrix=csi_matrix, is_frequency_bias=False, is_count=False,
        output_file_name=csi_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    this_title_string = 'Weighted CSI' if concat_figures else None
    this_panel_letter = 'g' if concat_figures else None
    _plot_one_score(
        score_matrix=weighted_csi_matrix, is_frequency_bias=False,
        is_count=False, output_file_name=weighted_csi_file_name,
        title_string=this_title_string, panel_letter=this_panel_letter
    )

    if not concat_figures:
        return

    panel_file_names = [
        num_actual_file_name, num_predicted_file_name, pod_file_name,
        far_file_name, frequency_bias_file_name, csi_file_name,
        weighted_csi_file_name
    ]
    concat_file_name = '{0:s}/spatial_evaluation.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=4, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        concat_figures=bool(getattr(INPUT_ARG_OBJECT, CONCAT_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
