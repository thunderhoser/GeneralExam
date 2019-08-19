"""Finds colour limits for Monte Carlo plots."""

import argparse
import numpy
from generalexam.ge_utils import climatology_utils as climo_utils

COMPARISON_NAMES = ['el_nino', 'la_nina', 'strong_el_nino', 'strong_la_nina']

INPUT_DIR_ARG_NAME = 'top_input_dir_name'
PROPERTY_ARG_NAME = 'property_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  For each season and each comparison, a'
    ' Monte Carlo file for property `{0:s}` will be found.'
).format(PROPERTY_ARG_NAME)

PROPERTY_HELP_STRING = (
    'Name of property tested in Monte Carlo files.  Must be in the following '
    'list:\n{0:s}'
).format(str(climo_utils.VALID_PROPERTY_NAMES))

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in each colour scheme.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PROPERTY_ARG_NAME, type=str, required=True,
    help=PROPERTY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=True,
    help=MAX_PERCENTILE_HELP_STRING)


def _run(top_input_dir_name, property_name, max_colour_percentile):
    """Finds colour limits for Monte Carlo plots.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param property_name: Same.
    :param max_colour_percentile: Same.
    """

    num_comparisons = len(COMPARISON_NAMES)
    num_seasons = len(climo_utils.VALID_SEASON_STRINGS)

    difference_values = numpy.array([], dtype=float)
    mean_values = numpy.array([], dtype=float)

    for i in range(num_comparisons):
        for j in range(num_seasons):
            this_input_dir_name = '{0:s}/{1:s}/{2:s}/stitched'.format(
                top_input_dir_name, COMPARISON_NAMES[i],
                climo_utils.VALID_SEASON_STRINGS[j]
            )

            this_file_name = climo_utils.find_monte_carlo_file(
                directory_name=this_input_dir_name, property_name=property_name,
                first_grid_row=0, first_grid_column=0,
                raise_error_if_missing=True)

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            this_monte_carlo_dict = climo_utils.read_monte_carlo_test(
                this_file_name)

            this_baseline_matrix = this_monte_carlo_dict[
                climo_utils.BASELINE_MATRIX_KEY]
            this_trial_matrix = this_monte_carlo_dict[
                climo_utils.TRIAL_MATRIX_KEY]

            mean_values = numpy.concatenate((
                mean_values,
                numpy.ravel(this_baseline_matrix),
                numpy.ravel(this_trial_matrix)
            ))

            difference_values = numpy.concatenate((
                difference_values,
                numpy.ravel(this_trial_matrix - this_baseline_matrix)
            ))

    max_difference_value = numpy.nanpercentile(
        numpy.absolute(difference_values), max_colour_percentile
    )
    max_mean_value = numpy.nanpercentile(mean_values, max_colour_percentile)

    print((
        '{0:.1f}th percentile of difference values = {1:.4e} ... of mean values'
        ' = {2:.4e}'
    ).format(
        max_colour_percentile, max_difference_value, max_mean_value
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        property_name=getattr(INPUT_ARG_OBJECT, PROPERTY_ARG_NAME),
        max_colour_percentile=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME)
    )
