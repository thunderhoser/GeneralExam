"""Plots validation scores for CNN experiment."""

import copy
import pickle
import os.path
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as gg_evaluation
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import pixelwise_evaluation as pixelwise_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIELD_COMBO_ABBREVS = [
    'h-u-v-thetaw', 'h-u-v-T-q', 'h-u-v-T-q-thetaw', 'h-u-v-thetaw-Z',
    'h-u-v-T-q-Z', 'h-u-v-T-q-thetaw-Z'
]
FIELD_COMBO_STRINGS = [
    r'$\theta_w$', r'$T$ and $q_v$', r'$T$, $q_v$, and $\theta_w$',
    r'$\theta_w$ and $Z$', r'$T$, $q_v$, and $Z$',
    r'$T$, $q_v$, $\theta_w$, and $Z$'
]

PRESSURE_COMBO_ABBREVS = [
    'sfc', '1000', 'sfc-1000', 'sfc-950', 'sfc-900', 'sfc-850'
]
PRESSURE_COMBO_STRINGS = [
    'Sfc', '1000 mb', 'Sfc + 1000 mb', 'Sfc + 950 mb', 'Sfc + 900 mb',
    'Sfc + 850 mb'
]

CONV_BLOCK_COUNTS = numpy.array([2, 3], dtype=int)
CONV_LAYER_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)

TOP_EXPERIMENT_DIR_NAME = (
    '/condo/swatwork/ralager/era5_experiment_with_orography'
)

AUC_MATRIX_KEY = 'auc_matrix'
AUPD_MATRIX_KEY = 'aupd_matrix'
CSI_MATRIX_KEY = 'csi_matrix'

FONT_SIZE = 35
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')

MARKER_COLOUR = numpy.full(3, 0.)
BEST_MODEL_MARKER_TYPE = '*'
BEST_MODEL_MARKER_SIZE = 48
BEST_MODEL_MARKER_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)


def _plot_one_score(score_matrix, score_abbrev, best_index):
    """Plots one score.

    :param score_matrix: 4-D numpy array of values.  Axes should be
        (num conv blocks, pressures, fields, num layers per block).
    :param score_abbrev: Abbreviated name for score.
    :param best_index: Linear index of best model.
    """

    num_conv_block_counts = len(CONV_BLOCK_COUNTS)
    num_conv_layer_counts = len(CONV_LAYER_COUNTS)

    best_i, best_j, best_k, best_m = numpy.unravel_index(
        best_index, score_matrix.shape
    )

    min_colour_value = numpy.percentile(score_matrix, 1.)
    max_colour_value = numpy.percentile(score_matrix, 99.)
    panel_file_names = []

    for m in range(num_conv_layer_counts):
        for i in range(num_conv_block_counts):
            if m == num_conv_layer_counts - 1:
                these_x_tick_labels = copy.deepcopy(FIELD_COMBO_STRINGS)
            else:
                these_x_tick_labels = [' '] * len(FIELD_COMBO_STRINGS)

            if i == 0:
                these_y_tick_labels = copy.deepcopy(PRESSURE_COMBO_STRINGS)
            else:
                these_y_tick_labels = [' '] * len(PRESSURE_COMBO_STRINGS)

            this_axes_object = gg_evaluation.plot_hyperparam_grid(
                score_matrix=score_matrix[i, ..., m],
                colour_map_object=COLOUR_MAP_OBJECT,
                min_colour_value=min_colour_value,
                max_colour_value=max_colour_value
            )

            this_axes_object.set_xticklabels(
                these_x_tick_labels, fontsize=FONT_SIZE, rotation=90.
            )
            this_axes_object.set_yticklabels(
                these_y_tick_labels, fontsize=FONT_SIZE
            )

            if i == best_i and m == best_m:
                this_axes_object.plot(
                    best_k, best_j, linestyle='None',
                    marker=BEST_MODEL_MARKER_TYPE,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR,
                    markersize=BEST_MODEL_MARKER_SIZE,
                    markeredgewidth=BEST_MODEL_MARKER_WIDTH
                )

            this_title_string = (
                '{0:d} blocks, {1:d} layer{2:s} per block'
            ).format(
                CONV_BLOCK_COUNTS[i], CONV_LAYER_COUNTS[m],
                's' if CONV_LAYER_COUNTS[m] > 1 else ''
            )

            pyplot.title(this_title_string)

            colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=this_axes_object,
                data_matrix=score_matrix[i, ..., m],
                colour_map_object=COLOUR_MAP_OBJECT,
                min_value=min_colour_value, max_value=max_colour_value,
                orientation_string='vertical', fraction_of_axis_length=0.85,
                extend_min=True, extend_max=True, font_size=FONT_SIZE
            )

            if i != num_conv_block_counts - 1:
                colour_bar_object.remove()

            this_file_name = (
                '{0:s}/{1:s}_num-blocks={2:d}_num-layers={3:d}.jpg'
            ).format(
                TOP_EXPERIMENT_DIR_NAME, score_abbrev,
                CONV_BLOCK_COUNTS[i], CONV_LAYER_COUNTS[m]
            )

            panel_file_names.append(this_file_name)

            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close()

    concat_file_name = '{0:s}/{1:s}.jpg'.format(
        TOP_EXPERIMENT_DIR_NAME, score_abbrev
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=num_conv_layer_counts,
        num_panel_columns=num_conv_block_counts
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


def _read_evaluation_file(evaluation_file_name):
    """Reads evaluation file for one CNN.

    :param evaluation_file_name: Path to input file.
    :return: auc: Area under ROC curve.
    :return: aupd: Area under performance diagram.
    :return: csi: Critical success index.
    """

    result_table_xarray = pixelwise_eval.read_file(evaluation_file_name)

    best_threshold = (
        result_table_xarray.attrs[pixelwise_eval.BEST_THRESHOLD_KEY]
    )
    all_thresholds = result_table_xarray.coords[
        pixelwise_eval.DETERMINIZN_THRESHOLD_DIM
    ].values
    best_threshold_index = numpy.argmin(numpy.absolute(
        best_threshold - all_thresholds
    ))

    auc = numpy.mean(
        result_table_xarray[pixelwise_eval.AREA_UNDER_ROCC_KEY].values
    )
    aupd = numpy.mean(
        result_table_xarray[pixelwise_eval.AREA_UNDER_PD_KEY].values
    )
    csi = numpy.mean(
        result_table_xarray[pixelwise_eval.CSI_KEY].values[
            :, best_threshold_index
        ]
    )

    return auc, aupd, csi


def _run():
    """Plots validation scores for CNN experiment.

    This is effectively the main method.
    """

    num_field_combos = len(FIELD_COMBO_ABBREVS)
    num_pressure_combos = len(PRESSURE_COMBO_ABBREVS)
    num_conv_block_counts = len(CONV_BLOCK_COUNTS)
    num_conv_layer_counts = len(CONV_LAYER_COUNTS)

    summary_file_name = '{0:s}/validation_grid.p'.format(
        TOP_EXPERIMENT_DIR_NAME
    )

    if os.path.isfile(summary_file_name):
        print('Reading data from: "{0:s}"...'.format(summary_file_name))

        pickle_file_handle = open(summary_file_name, 'rb')
        summary_dict = pickle.load(pickle_file_handle)
        pickle_file_handle.close()

        auc_matrix = summary_dict[AUC_MATRIX_KEY]
        aupd_matrix = summary_dict[AUPD_MATRIX_KEY]
        csi_matrix = summary_dict[CSI_MATRIX_KEY]
    else:
        dimensions = (
            num_conv_block_counts, num_pressure_combos, num_field_combos,
            num_conv_layer_counts
        )

        auc_matrix = numpy.full(dimensions, numpy.nan)
        aupd_matrix = numpy.full(dimensions, numpy.nan)
        csi_matrix = numpy.full(dimensions, numpy.nan)

        for i in range(num_conv_block_counts):
            for j in range(num_pressure_combos):
                for k in range(num_field_combos):
                    for m in range(num_conv_layer_counts):
                        this_eval_file_name = (
                            '{0:s}/{1:s}_{2:s}_num-blocks={3:d}_'
                            'num-layers-per-block={4:d}/validation/'
                            'validation_scores.nc'
                        ).format(
                            TOP_EXPERIMENT_DIR_NAME, FIELD_COMBO_ABBREVS[k],
                            PRESSURE_COMBO_ABBREVS[j], CONV_BLOCK_COUNTS[i],
                            CONV_LAYER_COUNTS[m]
                        )

                        print('Reading data from: "{0:s}"...'.format(
                            this_eval_file_name
                        ))

                        this_auc, this_aupd, this_csi = _read_evaluation_file(
                            this_eval_file_name
                        )

                        auc_matrix[i, j, k, m] = this_auc
                        aupd_matrix[i, j, k, m] = this_aupd
                        csi_matrix[i, j, k, m] = this_csi

                print_line = not (
                    i == num_conv_block_counts - 1
                    and j == num_pressure_combos - 1
                )

                if print_line:
                    print('\n')

        print(SEPARATOR_STRING)

        summary_dict = {
            AUC_MATRIX_KEY: auc_matrix,
            AUPD_MATRIX_KEY: aupd_matrix,
            CSI_MATRIX_KEY: csi_matrix
        }

        print('Writing results to: "{0:s}"...'.format(summary_file_name))
        pickle_file_handle = open(summary_file_name, 'wb')
        pickle.dump(summary_dict, pickle_file_handle)
        pickle_file_handle.close()

    sort_indices = numpy.argsort(-1 * numpy.ravel(auc_matrix))

    for q in range(len(sort_indices)):
        i, j, k, m = numpy.unravel_index(sort_indices[q], auc_matrix.shape)

        print((
            '{0:d}th-best AUC = {1:.3f} ... fields = {2:s} ... pressure '
            'levels = {3:s} ... num conv blocks = {4:d} ... num layers per '
            'block = {5:d}'
        ).format(
            q + 1, auc_matrix[i, j, k, m],
            FIELD_COMBO_ABBREVS[k], PRESSURE_COMBO_ABBREVS[j],
            CONV_BLOCK_COUNTS[i], CONV_LAYER_COUNTS[m]
        ))

        print('AUPD = {0:.3f} ... CSI = {1:.3f}\n'.format(
            aupd_matrix[i, j, k, m], csi_matrix[i, j, k, m]
        ))

    _plot_one_score(
        score_matrix=auc_matrix, score_abbrev='auc', best_index=sort_indices[0]
    )
    print('\n')

    _plot_one_score(
        score_matrix=aupd_matrix, score_abbrev='aupd',
        best_index=sort_indices[0]
    )
    print('\n')

    _plot_one_score(
        score_matrix=csi_matrix, score_abbrev='csi', best_index=sort_indices[0]
    )


if __name__ == '__main__':
    _run()
