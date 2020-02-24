"""Plots sensitivity to minimum frontal-zone length."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.climo_paper import \
    make_monte_carlo_figure_1season as make_mc_figure

KM_TO_METRES = 1000
MIN_LENGTHS_KM = numpy.array([200, 400, 600], dtype=int)

CONCAT_FIGURE_SIZE_PX = int(1e7)

PREDICTION_DIR_ARG_NAME = 'prediction_dir_name'
MAX_FDR_ARG_NAME = 'monte_carlo_max_fdr'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with deterministic CNN-predicted fronts.'
)
MAX_FDR_HELP_STRING = (
    'Max FDR (false-discovery rate) for field-based version of Monte Carlo '
    'significance test.  If you do not want to use field-based version, leave '
    'this argument alone.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FDR_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_FDR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_prediction_dir_name, monte_carlo_max_fdr, output_file_name):
    """Plots sensitivity to minimum frontal-zone length.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param monte_carlo_max_fdr: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    if monte_carlo_max_fdr <= 0:
        monte_carlo_max_fdr = None

    property_names = [
        climo_utils.CF_FREQ_PROPERTY_NAME, climo_utils.WF_FREQ_PROPERTY_NAME
    ]

    output_dir_name, pathless_output_file_name = os.path.split(output_file_name)
    extensionless_output_file_name = '{0:s}/{1:s}'.format(
        output_dir_name, os.path.splitext(pathless_output_file_name)[0]
    )

    num_properties = len(property_names)
    num_min_lengths = len(MIN_LENGTHS_KM)
    panel_file_names = []
    letter_label = None

    sig_marker_size = make_mc_figure.SIG_MARKER_SIZE * (
        1 + int(monte_carlo_max_fdr is not None)
    )

    for j in range(num_min_lengths):
        for i in range(num_properties):
            this_max_colour_value = (
                make_mc_figure.PROPERTY_TO_MAX_COLOUR_VALUE_DICT[
                    property_names[i]
                ]
            )

            if j == 0:
                this_input_dir_name = (
                    '{0:s}/determinized_wf-threshold=0.650_cf-threshold=0.650/'
                    'new_climatology'
                ).format(top_prediction_dir_name)
            else:
                this_input_dir_name = (
                    '{0:s}/determinized/'
                    'prob-threshold=0.650_min-length-metres={1:d}/climatology'
                ).format(
                    top_prediction_dir_name, KM_TO_METRES * MIN_LENGTHS_KM[j]
                )

            this_input_dir_name += (
                '/labels/counts/monte_carlo_tests/strong_el_nino/winter/'
                'stitched'
            )

            this_input_file_name = climo_utils.find_monte_carlo_file(
                directory_name=this_input_dir_name,
                property_name=property_names[i],
                first_grid_row=0, first_grid_column=0,
                raise_error_if_missing=True
            )

            print('Reading data from: "{0:s}"...'.format(this_input_file_name))
            this_monte_carlo_dict = climo_utils.read_monte_carlo_test(
                this_input_file_name
            )

            this_difference_matrix = (
                this_monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY] -
                this_monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY]
            )

            this_p_value_matrix = (
                this_monte_carlo_dict[climo_utils.P_VALUE_MATRIX_KEY]
            )

            this_p_value_matrix[
                numpy.isnan(this_difference_matrix)
            ] = numpy.nan

            if monte_carlo_max_fdr is None:
                this_significance_matrix = this_p_value_matrix <= 0.05
            else:
                this_p_value_matrix[
                    numpy.absolute(this_difference_matrix)
                    < this_max_colour_value / 2
                ] = numpy.nan

                this_significance_matrix = climo_utils.find_sig_grid_points(
                    p_value_matrix=this_p_value_matrix,
                    max_false_discovery_rate=monte_carlo_max_fdr
                )

            this_title_string = '{0:s} with {1:d}-km minimum length'.format(
                make_mc_figure.PROPERTY_ABBREV_TO_VERBOSE_DICT[
                    property_names[i]
                ],
                KM_TO_METRES * MIN_LENGTHS_KM[j]
            )

            this_output_file_name = (
                '{0:s}_{1:s}_min-length-km={2:d}.jpg'
            ).format(
                extensionless_output_file_name, property_names[i],
                MIN_LENGTHS_KM[j]
            )

            panel_file_names.append(this_output_file_name)

            if letter_label is None:
                letter_label = 'a'
            else:
                letter_label = chr(ord(letter_label) + 1)

            make_mc_figure._plot_one_difference(
                difference_matrix=this_difference_matrix,
                significance_matrix=this_significance_matrix,
                sig_marker_size=sig_marker_size,
                max_colour_value=this_max_colour_value,
                plot_latitudes=True, plot_longitudes=True,
                plot_colour_bar=j == num_min_lengths - 1,
                title_string=this_title_string, letter_label=letter_label,
                output_file_name=panel_file_names[-1]
            )

    print('Concatenating panels to: "{0:s}"...'.format(output_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=output_file_name,
        num_panel_rows=num_min_lengths, num_panel_columns=num_properties
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for this_file_name in panel_file_names:
        print('Removing temporary file "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        monte_carlo_max_fdr=getattr(INPUT_ARG_OBJECT, MAX_FDR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
