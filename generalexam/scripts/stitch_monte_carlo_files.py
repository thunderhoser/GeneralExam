"""Stitches together Monte Carlo files with different grid cells."""

import argparse
import numpy
from generalexam.ge_utils import climatology_utils as climo_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_dir_name'
PROPERTY_ARG_NAME = 'property_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory (with Monte Carlo files to be stitched).  Files '
    'therein will be found by `climatology_utils.find_many_monte_carlo_files` '
    'and read by `climatology_utils.read_monte_carlo_test`.')

PROPERTY_HELP_STRING = (
    'Name of property tested in Monte Carlo files.  Must be in the following '
    'list:\n{0:s}'
).format(str(climo_utils.VALID_PROPERTY_NAMES))

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  One stitched Monte Carlo file (with results for'
    ' all grid cells) will be written here by '
    '`climatology_utils.write_monte_carlo_test`, to an exact location '
    'determined by `climatology_utils.find_monte_carlo_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PROPERTY_ARG_NAME, type=str, required=True,
    help=PROPERTY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_dir_name, property_name, output_dir_name):
    """Stitches together Monte Carlo files with different grid cells.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param property_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if cannot find enough input files to stitch together a
        full grid.
    """

    input_file_names = climo_utils.find_many_monte_carlo_files(
        directory_name=input_dir_name, property_name=property_name,
        raise_error_if_none_found=True)

    num_files = len(input_file_names)
    first_row_by_file = numpy.full(num_files, -1, dtype=int)
    first_column_by_file = numpy.full(num_files, -1, dtype=int)
    last_row_by_file = numpy.full(num_files, -1, dtype=int)
    last_column_by_file = numpy.full(num_files, -1, dtype=int)

    for i in range(num_files):
        print('Reading metadata from: "{0:s}"...'.format(input_file_names[i]))
        this_monte_carlo_dict = climo_utils.read_monte_carlo_test(
            input_file_names[i]
        )

        first_row_by_file[i] = this_monte_carlo_dict[
            climo_utils.FIRST_GRID_ROW_KEY]
        first_column_by_file[i] = this_monte_carlo_dict[
            climo_utils.FIRST_GRID_COLUMN_KEY]

        this_sig_matrix = this_monte_carlo_dict[
            climo_utils.SIGNIFICANCE_MATRIX_KEY]

        last_row_by_file[i] = (
            first_row_by_file[i] + this_sig_matrix.shape[0] - 1
        )
        last_column_by_file[i] = (
            first_column_by_file[i] + this_sig_matrix.shape[1] - 1
        )

    print(SEPARATOR_STRING)

    first_grid_rows = numpy.unique(first_row_by_file)
    first_grid_columns = numpy.unique(first_column_by_file)

    num_first_rows = len(first_grid_rows)
    num_first_columns = len(first_grid_columns)
    input_file_name_matrix = numpy.full(
        (num_first_rows, num_first_columns), '', dtype=object
    )

    for j in range(num_first_rows):
        for k in range(num_first_columns):
            these_indices = numpy.where(numpy.logical_and(
                first_row_by_file == first_grid_rows[j],
                first_column_by_file == first_grid_columns[k],
            ))[0]

            if len(these_indices) == 0:
                error_string = (
                    'Cannot find file with first grid row = {0:d} and first '
                    'column = {1:d}.'
                ).format(
                    first_grid_rows[j], first_grid_columns[k]
                )

                raise ValueError(error_string)

            input_file_name_matrix[j, k] = input_file_names[these_indices[0]]

    num_rows_total = 1 + numpy.max(last_row_by_file)
    num_columns_total = 1 + numpy.max(last_column_by_file)
    dimensions = (num_rows_total, num_columns_total)

    baseline_mean_matrix = numpy.full(dimensions, numpy.nan)
    trial_mean_matrix = numpy.full(dimensions, numpy.nan)
    significance_matrix = numpy.full(dimensions, False, dtype=bool)
    num_labels_matrix = numpy.full(dimensions, -1, dtype=int)

    baseline_input_file_names = []
    trial_input_file_names = []
    num_iterations = None
    confidence_level = None

    for j in range(num_first_rows):
        for k in range(num_first_columns):
            print('Reading all data from: "{0:s}"...'.format(
                input_file_name_matrix[j, k]
            ))

            this_monte_carlo_dict = climo_utils.read_monte_carlo_test(
                input_file_name_matrix[j, k]
            )

            if num_iterations is None:
                num_iterations = this_monte_carlo_dict[
                    climo_utils.NUM_ITERATIONS_KEY]
                confidence_level = this_monte_carlo_dict[
                    climo_utils.CONFIDENCE_LEVEL_KEY]

            assert (
                num_iterations ==
                this_monte_carlo_dict[climo_utils.NUM_ITERATIONS_KEY]
            )

            assert numpy.isclose(
                confidence_level,
                this_monte_carlo_dict[climo_utils.CONFIDENCE_LEVEL_KEY],
                atol=1e-6
            )

            this_baseline_mean_matrix = this_monte_carlo_dict[
                climo_utils.BASELINE_MATRIX_KEY]

            this_last_row = (
                first_grid_rows[j] + this_baseline_mean_matrix.shape[0] - 1
            )
            this_last_column = (
                first_grid_columns[k] + this_baseline_mean_matrix.shape[1] - 1
            )

            baseline_mean_matrix[
                first_grid_rows[j]:(this_last_row + 1),
                first_grid_columns[k]:(this_last_column + 1)
            ] = this_baseline_mean_matrix

            trial_mean_matrix[
                first_grid_rows[j]:(this_last_row + 1),
                first_grid_columns[k]:(this_last_column + 1)
            ] = this_monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY]

            significance_matrix[
                first_grid_rows[j]:(this_last_row + 1),
                first_grid_columns[k]:(this_last_column + 1)
            ] = this_monte_carlo_dict[climo_utils.SIGNIFICANCE_MATRIX_KEY]

            num_labels_matrix[
                first_grid_rows[j]:(this_last_row + 1),
                first_grid_columns[k]:(this_last_column + 1)
            ] = this_monte_carlo_dict[climo_utils.NUM_LABELS_MATRIX_KEY]

            baseline_input_file_names += this_monte_carlo_dict[
                climo_utils.BASELINE_INPUT_FILES_KEY]
            trial_input_file_names += this_monte_carlo_dict[
                climo_utils.TRIAL_INPUT_FILES_KEY]

    output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name, property_name=property_name,
        first_grid_row=numpy.min(first_grid_rows),
        first_grid_column=numpy.min(first_grid_columns),
        raise_error_if_missing=False
    )

    print('Writing stitched grid to: "{0:s}"...'.format(output_file_name))

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=output_file_name,
        baseline_mean_matrix=baseline_mean_matrix,
        trial_mean_matrix=trial_mean_matrix,
        significance_matrix=significance_matrix,
        num_labels_matrix=num_labels_matrix, property_name=property_name,
        baseline_input_file_names=baseline_input_file_names,
        trial_input_file_names=trial_input_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=numpy.min(first_grid_rows),
        first_grid_column=numpy.min(first_grid_columns)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        property_name=getattr(INPUT_ARG_OBJECT, PROPERTY_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
