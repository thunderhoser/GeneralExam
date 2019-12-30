"""Plots validation scores for object-conversion experiment."""

import os.path
import pickle
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as gg_evaluation
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 0.001
FAR_WEIGHT_FOR_CSI = 0.5

NEIGH_DISTANCES_METRES = numpy.array([50000, 100000, 150000, 200000], dtype=int)
WF_PROB_THRESHOLDS = numpy.linspace(0.2, 0.8, num=25)
CF_PROB_THRESHOLDS = numpy.linspace(0.2, 0.8, num=25)

TOP_EXPERIMENT_DIR_NAME = (
    '/condo/swatwork/ralager/era5_experiment_with_orography/'
    'h-u-v-T-q-thetaw-Z_sfc-900_num-blocks=3_num-layers-per-block=2/'
    'validation/gridded_predictions'
)

OUTPUT_DIR_NAME = '{0:s}/evaluation'.format(TOP_EXPERIMENT_DIR_NAME)

POD_MATRIX_KEY = 'pod_matrix'
FAR_MATRIX_KEY = 'far_matrix'
CSI_MATRIX_KEY = 'csi_matrix'
WEIGHTED_CSI_MATRIX_KEY = 'weighted_csi_matrix'
BIAS_MATRIX_KEY = 'frequency_bias_matrix'

DEFAULT_FONT_SIZE = 20
TITLE_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 20
DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')

MARKER_COLOUR = numpy.full(3, 0.)
MARKER_SIZE = 22
MARKER_EDGE_WIDTH = 0

SELECTED_MODEL_MARKER_TYPE = 'o'
SELECTED_WF_INDEX = 18
SELECTED_CF_INDEX = 18

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)


def _get_bias_colour_scheme(max_value):
    """Returns colour scheme for frequency bias.

    :param max_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour normalization (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

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


def _plot_one_score_one_neigh(
        score_matrix, label_x_axis, label_y_axis, is_frequency_bias,
        neigh_distance_metres, output_file_name):
    """Plots one score for one neighbourhood distance.

    M = number of WF-probability thresholds
    N = number of CF-probability thresholds

    :param score_matrix: M-by-N numpy array of scores.
    :param label_x_axis: Boolean flag.
    :param label_y_axis: Boolean flag.
    :param is_frequency_bias: Boolean flag.
    :param neigh_distance_metres: Neighbourhood distance.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    num_wf_thresholds = len(WF_PROB_THRESHOLDS)
    num_cf_thresholds = len(CF_PROB_THRESHOLDS)

    if label_y_axis:
        y_tick_labels = [
            '{0:.3f}'.format(p) for p in WF_PROB_THRESHOLDS
        ]
    else:
        y_tick_labels = [' '] * num_wf_thresholds

    if label_x_axis:
        x_tick_labels = [
            '{0:.3f}'.format(p) for p in CF_PROB_THRESHOLDS
        ]
    else:
        x_tick_labels = [' '] * num_cf_thresholds

    if is_frequency_bias:
        this_offset = numpy.nanpercentile(numpy.absolute(score_matrix - 1.), 99)
        max_colour_value = 1. + this_offset
        min_colour_value = 0.

        colour_map_object, colour_norm_object = _get_bias_colour_scheme(
            max_colour_value
        )
    else:
        colour_map_object = DEFAULT_COLOUR_MAP_OBJECT
        colour_norm_object = None

        min_colour_value = numpy.nanpercentile(score_matrix, 1)
        max_colour_value = numpy.nanpercentile(score_matrix, 99)

    axes_object = gg_evaluation.plot_hyperparam_grid(
        score_matrix=score_matrix,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    neigh_distance_km = int(numpy.round(neigh_distance_metres * METRES_TO_KM))
    title_string = 'Neighbourhood distance = {0:d} km'.format(neigh_distance_km)
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    axes_object.set_xticklabels(
        x_tick_labels, fontsize=TICK_LABEL_FONT_SIZE, rotation=90.
    )
    axes_object.set_yticklabels(
        y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE
    )

    if label_y_axis:
        axes_object.set_ylabel(
            'WF threshold', fontsize=TICK_LABEL_FONT_SIZE
        )

    if label_x_axis:
        axes_object.set_xlabel(
            'CF threshold', fontsize=TICK_LABEL_FONT_SIZE
        )

    if is_frequency_bias:
        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=True,
            fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE)

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1f}'.format(v) for v in tick_values]

        colour_bar_object.set_ticks(tick_values)
        colour_bar_object.set_ticklabels(tick_strings)
    else:
        plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string='vertical', extend_min=True, extend_max=True,
            fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE)

    axes_object.plot(
        SELECTED_CF_INDEX, SELECTED_WF_INDEX, linestyle='None',
        marker=SELECTED_MODEL_MARKER_TYPE,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


def _run():
    """Plots validation scores for determinization experiment with ERA5 data.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_neigh_distances = len(NEIGH_DISTANCES_METRES)
    num_wf_thresholds = len(WF_PROB_THRESHOLDS)
    num_cf_thresholds = len(CF_PROB_THRESHOLDS)
    summary_file_name = '{0:s}/evaluation.p'.format(OUTPUT_DIR_NAME)

    if os.path.isfile(summary_file_name):
        print('Reading data from: "{0:s}"...'.format(summary_file_name))

        pickle_file_handle = open(summary_file_name, 'rb')
        summary_dict = pickle.load(pickle_file_handle)
        pickle_file_handle.close()

        pod_matrix = summary_dict[POD_MATRIX_KEY]
        far_matrix = summary_dict[FAR_MATRIX_KEY]
        csi_matrix = summary_dict[CSI_MATRIX_KEY]
        weighted_csi_matrix = summary_dict[WEIGHTED_CSI_MATRIX_KEY]
        frequency_bias_matrix = summary_dict[BIAS_MATRIX_KEY]

    else:
        these_dim = (num_neigh_distances, num_wf_thresholds, num_cf_thresholds)
        pod_matrix = numpy.full(these_dim, numpy.nan)
        far_matrix = numpy.full(these_dim, numpy.nan)
        csi_matrix = numpy.full(these_dim, numpy.nan)
        weighted_csi_matrix = numpy.full(these_dim, numpy.nan)
        frequency_bias_matrix = numpy.full(these_dim, numpy.nan)

        for i in range(num_neigh_distances):
            for j in range(num_wf_thresholds):
                for k in range(num_cf_thresholds):
                    this_file_name = (
                        '{0:s}/deterministic_wf-threshold={1:.3f}_'
                        'cf-threshold={2:.3f}/evaluation/'
                        'evaluation_neigh-distance-metres={3:06d}.p'
                    ).format(
                        TOP_EXPERIMENT_DIR_NAME, WF_PROB_THRESHOLDS[j],
                        CF_PROB_THRESHOLDS[k], NEIGH_DISTANCES_METRES[i]
                    )

                    if not os.path.isfile(this_file_name):
                        continue

                    print('Reading data from: "{0:s}"...'.format(
                        this_file_name
                    ))
                    this_evaluation_dict = (
                        neigh_evaluation.read_nonspatial_results(this_file_name)
                    )

                    this_binary_ct = this_evaluation_dict[
                        neigh_evaluation.BINARY_TABLES_KEY
                    ][0]

                    pod_matrix[i, j, k] = neigh_evaluation.get_pod(
                        this_binary_ct)
                    far_matrix[i, j, k] = neigh_evaluation.get_far(
                        this_binary_ct)
                    csi_matrix[i, j, k] = neigh_evaluation.get_csi(
                        binary_ct_as_dict=this_binary_ct, far_weight=1.
                    )
                    weighted_csi_matrix[i, j, k] = neigh_evaluation.get_csi(
                        binary_ct_as_dict=this_binary_ct,
                        far_weight=FAR_WEIGHT_FOR_CSI
                    )
                    frequency_bias_matrix[i, j, k] = (
                        neigh_evaluation.get_frequency_bias(this_binary_ct)
                    )

                if not (i == num_neigh_distances - 1 and
                        j == num_wf_thresholds - 1):
                    print('\n')

        print(SEPARATOR_STRING)

        summary_dict = {
            POD_MATRIX_KEY: pod_matrix,
            FAR_MATRIX_KEY: far_matrix,
            CSI_MATRIX_KEY: csi_matrix,
            WEIGHTED_CSI_MATRIX_KEY: weighted_csi_matrix,
            BIAS_MATRIX_KEY: frequency_bias_matrix
        }

        print('Writing validation scores to: "{0:s}"...'.format(
            summary_file_name
        ))

        pickle_file_handle = open(summary_file_name, 'wb')
        pickle.dump(summary_dict, pickle_file_handle)
        pickle_file_handle.close()

    print(SEPARATOR_STRING)

    pod_file_names = []
    far_file_names = []
    csi_file_names = []
    weighted_csi_file_names = []
    frequency_bias_file_names = []

    for i in range(num_neigh_distances):
        these_sort_indices = numpy.argsort(
            -1 * numpy.ravel(weighted_csi_matrix[i, ...])
        )

        for q in range(len(these_sort_indices)):
            j, k = numpy.unravel_index(
                these_sort_indices[q], weighted_csi_matrix[i, ...].shape
            )

            print((
                '{0:d}th-best weighted CSI for {1:.1f}-km neigh distance = '
                '{2:.3f} ... WF-probability threshold = {3:.3f} ... '
                'CF-probability threshold = {4:.3f}'
            ).format(
                q + 1,
                NEIGH_DISTANCES_METRES[i] * METRES_TO_KM,
                weighted_csi_matrix[i, j, k],
                WF_PROB_THRESHOLDS[j], CF_PROB_THRESHOLDS[k]
            ))

            print((
                'Corresponding actual CSI = {0:.3f} ... POD = {1:.3f} ... '
                'FAR = {2:.3f} ... frequency bias = {3:.3f}\n'
            ).format(
                csi_matrix[i, j, k], pod_matrix[i, j, k],
                far_matrix[i, j, k], frequency_bias_matrix[i, j, k]
            ))

        this_file_name = '{0:s}/pod_neigh-distance-metres={1:06d}.jpg'.format(
            OUTPUT_DIR_NAME, NEIGH_DISTANCES_METRES[i]
        )
        pod_file_names.append(this_file_name)

        _plot_one_score_one_neigh(
            score_matrix=pod_matrix[i, ...],
            label_x_axis=i >= 2, label_y_axis=i in [0, 2],
            is_frequency_bias=False,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            output_file_name=this_file_name
        )

        this_file_name = '{0:s}/far_neigh-distance-metres={1:06d}.jpg'.format(
            OUTPUT_DIR_NAME, NEIGH_DISTANCES_METRES[i]
        )
        far_file_names.append(this_file_name)

        _plot_one_score_one_neigh(
            score_matrix=far_matrix[i, ...],
            label_x_axis=i >= 2, label_y_axis=i in [0, 2],
            is_frequency_bias=False,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            output_file_name=this_file_name
        )

        this_file_name = '{0:s}/csi_neigh-distance-metres={1:06d}.jpg'.format(
            OUTPUT_DIR_NAME, NEIGH_DISTANCES_METRES[i]
        )
        csi_file_names.append(this_file_name)

        _plot_one_score_one_neigh(
            score_matrix=csi_matrix[i, ...],
            label_x_axis=i >= 2, label_y_axis=i in [0, 2],
            is_frequency_bias=False,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            output_file_name=this_file_name
        )

        this_file_name = (
            '{0:s}/weighted_csi_neigh-distance-metres={1:06d}.jpg'
        ).format(
            OUTPUT_DIR_NAME, NEIGH_DISTANCES_METRES[i]
        )
        weighted_csi_file_names.append(this_file_name)

        _plot_one_score_one_neigh(
            score_matrix=weighted_csi_matrix[i, ...],
            label_x_axis=i >= 2, label_y_axis=i in [0, 2],
            is_frequency_bias=False,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            output_file_name=this_file_name
        )

        this_file_name = (
            '{0:s}/frequency_bias_neigh-distance-metres={1:06d}.jpg'
        ).format(
            OUTPUT_DIR_NAME, NEIGH_DISTANCES_METRES[i]
        )
        frequency_bias_file_names.append(this_file_name)

        _plot_one_score_one_neigh(
            score_matrix=frequency_bias_matrix[i, ...],
            label_x_axis=i >= 2, label_y_axis=i in [0, 2],
            is_frequency_bias=True,
            neigh_distance_metres=NEIGH_DISTANCES_METRES[i],
            output_file_name=this_file_name
        )

        print(SEPARATOR_STRING)

    main_pod_file_name = '{0:s}/pod.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(main_pod_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=pod_file_names, output_file_name=main_pod_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=main_pod_file_name,
        output_file_name=main_pod_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    main_far_file_name = '{0:s}/far.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(main_far_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=far_file_names, output_file_name=main_far_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=main_far_file_name,
        output_file_name=main_far_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    main_csi_file_name = '{0:s}/csi.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(main_csi_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=csi_file_names, output_file_name=main_csi_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=main_csi_file_name,
        output_file_name=main_csi_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    main_weighted_csi_file_name = '{0:s}/weighted_csi.jpg'.format(
        OUTPUT_DIR_NAME
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        main_weighted_csi_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=weighted_csi_file_names,
        output_file_name=main_weighted_csi_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=main_weighted_csi_file_name,
        output_file_name=main_weighted_csi_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    main_bias_file_name = '{0:s}/frequency_bias.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(main_bias_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=frequency_bias_file_names,
        output_file_name=main_bias_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=main_bias_file_name,
        output_file_name=main_bias_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    _run()
