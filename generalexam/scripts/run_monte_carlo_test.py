"""Runs Monte Carlo significance test for gridded front properties."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import climatology_utils as climo_utils

# TODO(thunderhoser): Allow the same to be done for gridded frequency.  Do not
# run this test on raw counts, since the two composites are almost guaranteed to
# have unequal time spans.

TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_property_dir_name'
FIRST_START_TIMES_ARG_NAME = 'first_start_time_strings'
FIRST_END_TIMES_ARG_NAME = 'first_end_time_strings'
SECOND_START_TIMES_ARG_NAME = 'second_start_time_strings'
SECOND_END_TIMES_ARG_NAME = 'second_end_time_strings'
FIRST_ROW_ARG_NAME = 'first_grid_row'
LAST_ROW_ARG_NAME = 'last_grid_row'
FIRST_COLUMN_ARG_NAME = 'first_grid_column'
LAST_COLUMN_ARG_NAME = 'last_grid_column'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_many_property_files` and read by '
    '`climatology_utils.average_many_property_files`.  This script will look '
    'for one file per 3-hour time step.')

FIRST_START_TIMES_HELP_STRING = (
    'List of start times (format "yyyymmddHH") for first composite.')

FIRST_END_TIMES_HELP_STRING = (
    'List of end times (format "yyyymmddHH") for first composite.')

SECOND_START_TIMES_HELP_STRING = (
    'List of start times (format "yyyymmddHH") for second composite.')

SECOND_END_TIMES_HELP_STRING = (
    'List of end times (format "yyyymmddHH") for second composite.')

FIRST_ROW_HELP_STRING = (
    'First grid row to test.  This script will test only a subset of grid '
    'cells, since testing all grid cells would be too memory-intensive.')

LAST_ROW_HELP_STRING = (
    'Last grid row to test.  See help string for `{0:s}`.'
).format(FIRST_ROW_ARG_NAME)

FIRST_COLUMN_HELP_STRING = (
    'First grid column to test.  See help string for `{0:s}`.'
).format(FIRST_ROW_ARG_NAME)

LAST_COLUMN_HELP_STRING = (
    'Last grid column to test.  See help string for `{0:s}`.'
).format(FIRST_ROW_ARG_NAME)

NUM_ITERATIONS_HELP_STRING = (
    'Number of Monte Carlo iterations (number of times, at each grid cell, that'
    ' data will be shuffled between the two composites).')

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for two-tailed Monte Carlo test.  At each grid cell, '
    'difference between the composites will be deemed statistically significant'
    ' iff it reaches or exceeds this confidence level.')

# TODO(thunderhoser): I haven't decided on the output format.
OUTPUT_DIR_HELP_STRING = 'Name of output directory.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_START_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=FIRST_START_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_END_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=FIRST_END_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SECOND_START_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=SECOND_START_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SECOND_END_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=SECOND_END_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_ROW_ARG_NAME, type=int, required=True,
    help=FIRST_ROW_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_ROW_ARG_NAME, type=int, required=True,
    help=LAST_ROW_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_COLUMN_ARG_NAME, type=int, required=True,
    help=FIRST_COLUMN_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_COLUMN_ARG_NAME, type=int, required=True,
    help=LAST_COLUMN_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False, default=5000,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_input_times(start_time_strings, end_time_strings):
    """Converts and error-checks set of input times.

    :param start_time_strings: See documentation at top of file for either
        `first_start_time_strings` or `second_start_time_strings`.
    :param end_time_strings: See doc at top of file for either
        `first_end_time_strings` or `second_end_time_strings`.
    :return: valid_times_unix_sec: 1-D numpy array of time steps.
    """

    num_periods = len(start_time_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(end_time_strings),
        exact_dimensions=numpy.array([num_periods], dtype=int)
    )

    start_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in start_time_strings
    ], dtype=int)

    end_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in end_time_strings
    ], dtype=int)

    sort_indices = numpy.argsort(start_times_unix_sec)
    start_times_unix_sec = start_times_unix_sec[sort_indices]
    end_times_unix_sec = end_times_unix_sec[sort_indices]

    for i in range(1, num_periods):
        error_checking.assert_is_greater(
            start_times_unix_sec[i], end_times_unix_sec[i - 1]
        )

    valid_times_unix_sec = numpy.array([], dtype=int)

    for i in range(num_periods):
        these_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=start_times_unix_sec[i],
            end_time_unix_sec=end_times_unix_sec[i],
            time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

        valid_times_unix_sec = numpy.concatenate((
            valid_times_unix_sec, these_times_unix_sec))

    return valid_times_unix_sec


def _read_properties_one_composite(
        property_file_names, first_grid_row, last_grid_row, first_grid_column,
        last_grid_column):
    """Reads gridded front properties for one composite.

    :param property_file_names: 1-D list of paths to input files.
    :param first_grid_row: See documentation at top of file.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :return: front_property_dict: Dictionary with the following keys.
    front_property_dict["wf_length_matrix_metres"]: See doc for
        `climatology_utils.write_gridded_properties`.
    front_property_dict["wf_area_matrix_m2"]: Same.
    front_property_dict["cf_length_matrix_metres"]: Same.
    front_property_dict["cf_area_matrix_m2"]: Same.
    """

    wf_length_matrix_metres = None
    wf_area_matrix_m2 = None
    cf_length_matrix_metres = None
    cf_area_matrix_m2 = None

    for this_file_name in property_file_names:
        print('Reading front properties from: "{0:s}"...'.format(
            this_file_name))

        this_property_dict = climo_utils.read_gridded_properties(this_file_name)

        this_wf_length_matrix_metres = this_property_dict[
            climo_utils.WARM_FRONT_LENGTHS_KEY
        ][
            :, first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_wf_area_matrix_m2 = this_property_dict[
            climo_utils.WARM_FRONT_AREAS_KEY
        ][
            :, first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_cf_length_matrix_metres = this_property_dict[
            climo_utils.COLD_FRONT_LENGTHS_KEY
        ][
            :, first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_cf_area_matrix_m2 = this_property_dict[
            climo_utils.COLD_FRONT_AREAS_KEY
        ][
            :, first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        if wf_length_matrix_metres is None:
            wf_length_matrix_metres = this_wf_length_matrix_metres + 0.
            wf_area_matrix_m2 = this_wf_area_matrix_m2 + 0.
            cf_length_matrix_metres = this_cf_length_matrix_metres + 0.
            cf_area_matrix_m2 = this_cf_area_matrix_m2 + 0.
        else:
            wf_length_matrix_metres = numpy.concatenate(
                (wf_length_matrix_metres, this_wf_length_matrix_metres), axis=0
            )
            wf_area_matrix_m2 = numpy.concatenate(
                (wf_area_matrix_m2, this_wf_area_matrix_m2), axis=0
            )
            cf_length_matrix_metres = numpy.concatenate(
                (cf_length_matrix_metres, this_cf_length_matrix_metres), axis=0
            )
            cf_area_matrix_m2 = numpy.concatenate(
                (cf_area_matrix_m2, this_cf_area_matrix_m2), axis=0
            )

    return {
        climo_utils.WARM_FRONT_LENGTHS_KEY: wf_length_matrix_metres,
        climo_utils.WARM_FRONT_AREAS_KEY: wf_area_matrix_m2,
        climo_utils.COLD_FRONT_LENGTHS_KEY: cf_length_matrix_metres,
        climo_utils.COLD_FRONT_AREAS_KEY: cf_area_matrix_m2
    }


def _mc_test_one_property(
        first_property_matrix, second_property_matrix, num_iterations,
        confidence_level):
    """Runs Monte Carlo test for one property.

    The "one property" could be WF length, WF area, CF length, or CF area.

    F = number of times in first composite
    S = number of times in second composite
    M = number of rows in grid
    N = number of columns in grid

    :param first_property_matrix: F-by-M-by-N numpy array with values of given
        property.
    :param second_property_matrix: S-by-M-by-N numpy array with values of given
        property.
    :param num_iterations: See documentation at top of file.
    :param confidence_level: Same.
    :return: actual_difference_matrix: M-by-N numpy array with difference (mean
        of second composite minus mean of first composite) at each grid cell.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between means is significant.
    """

    first_num_times = first_property_matrix.shape[0]
    concat_property_matrix = numpy.concatenate(
        (first_property_matrix, second_property_matrix), axis=0
    )

    num_grid_rows = first_property_matrix.shape[1]
    num_grid_columns = first_property_matrix.shape[2]
    mc_difference_matrix = numpy.full(
        (num_iterations, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_iterations):
        if numpy.mod(k, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                k, num_iterations
            ))

        numpy.random.shuffle(concat_property_matrix)
        this_first_mean_matrix = numpy.nanmean(
            concat_property_matrix[:first_num_times, ...], axis=0
        )
        this_second_mean_matrix = numpy.nanmean(
            concat_property_matrix[first_num_times:, ...], axis=0
        )

        # TODO(thunderhoser): This works only for values (like length and area)
        # that must be positive.
        this_first_mean_matrix[this_first_mean_matrix == 0] = numpy.nan
        this_second_mean_matrix[this_second_mean_matrix == 0] = numpy.nan

        mc_difference_matrix[k, ...] = (
            this_second_mean_matrix - this_first_mean_matrix
        )

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))

    first_mean_matrix = numpy.nanmean(first_property_matrix, axis=0)
    second_mean_matrix = numpy.nanmean(second_property_matrix, axis=0)
    first_mean_matrix[first_mean_matrix == 0] = numpy.nan
    second_mean_matrix[second_mean_matrix == 0] = numpy.nan

    actual_difference_matrix = second_mean_matrix - first_mean_matrix

    # TODO(thunderhoser): nanpercentile method ignores NaN's completely.  Is
    # this what I want?
    min_difference_matrix = numpy.nanpercentile(
        a=mc_difference_matrix, q=50. * (1 - confidence_level), axis=0
    )
    max_difference_matrix = numpy.nanpercentile(
        a=mc_difference_matrix, q=50. * (1 + confidence_level), axis=0
    )

    significance_matrix = numpy.logical_or(
        actual_difference_matrix < min_difference_matrix,
        actual_difference_matrix > max_difference_matrix
    )

    print((
        'Difference between means is significant at {0:d} of {1:d} grid cells!'
    ).format(
        numpy.sum(significance_matrix.astype(int)), significance_matrix.size
    ))

    return actual_difference_matrix, significance_matrix


def _run(input_property_dir_name, first_start_time_strings,
         first_end_time_strings, second_start_time_strings,
         second_end_time_strings, first_grid_row, last_grid_row,
         first_grid_column, last_grid_column, num_iterations, confidence_level,
         output_dir_name):
    """Runs Monte Carlo significance test for gridded front properties.

    This is effectively the main method.

    :param input_property_dir_name: See documentation at top of file.
    :param first_start_time_strings: Same.
    :param first_end_time_strings: Same.
    :param second_start_time_strings: Same.
    :param second_end_time_strings: Same.
    :param first_grid_row: Same.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :param num_iterations: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(first_grid_row, 0)
    error_checking.assert_is_greater(last_grid_row, first_grid_row)
    error_checking.assert_is_geq(first_grid_column, 0)
    error_checking.assert_is_greater(last_grid_column, first_grid_column)
    error_checking.assert_is_geq(num_iterations, 1000)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    first_times_unix_sec = _convert_input_times(
        start_time_strings=first_start_time_strings,
        end_time_strings=first_end_time_strings)

    second_times_unix_sec = _convert_input_times(
        start_time_strings=second_start_time_strings,
        end_time_strings=second_end_time_strings)

    first_property_file_names = [
        climo_utils.find_basic_file(
            directory_name=input_property_dir_name,
            file_type_string=climo_utils.FRONT_PROPERTIES_STRING,
            valid_time_unix_sec=t)
        for t in first_times_unix_sec
    ]

    second_property_file_names = [
        climo_utils.find_basic_file(
            directory_name=input_property_dir_name,
            file_type_string=climo_utils.FRONT_PROPERTIES_STRING,
            valid_time_unix_sec=t)
        for t in second_times_unix_sec
    ]

    first_property_dict = _read_properties_one_composite(
        property_file_names=first_property_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    second_property_dict = _read_properties_one_composite(
        property_file_names=second_property_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    wf_length_diff_matrix_metres, wf_length_sig_matrix = _mc_test_one_property(
        first_property_matrix=first_property_dict[
            climo_utils.WARM_FRONT_LENGTHS_KEY],
        second_property_matrix=second_property_dict[
            climo_utils.WARM_FRONT_LENGTHS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WARM_FRONT_LENGTHS_KEY,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        difference_matrix=wf_length_diff_matrix_metres,
        significance_matrix=wf_length_sig_matrix,
        property_name=climo_utils.WARM_FRONT_LENGTHS_KEY,
        first_property_file_names=first_property_file_names,
        second_property_file_names=second_property_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)
    print(SEPARATOR_STRING)

    wf_area_diff_matrix_m2, wf_area_sig_matrix = _mc_test_one_property(
        first_property_matrix=first_property_dict[
            climo_utils.WARM_FRONT_AREAS_KEY],
        second_property_matrix=second_property_dict[
            climo_utils.WARM_FRONT_AREAS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WARM_FRONT_AREAS_KEY,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        difference_matrix=wf_area_diff_matrix_m2,
        significance_matrix=wf_area_sig_matrix,
        property_name=climo_utils.WARM_FRONT_AREAS_KEY,
        first_property_file_names=first_property_file_names,
        second_property_file_names=second_property_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)
    print(SEPARATOR_STRING)

    cf_length_diff_matrix_metres, cf_length_sig_matrix = _mc_test_one_property(
        first_property_matrix=first_property_dict[
            climo_utils.COLD_FRONT_LENGTHS_KEY],
        second_property_matrix=second_property_dict[
            climo_utils.COLD_FRONT_LENGTHS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.COLD_FRONT_LENGTHS_KEY,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        difference_matrix=cf_length_diff_matrix_metres,
        significance_matrix=cf_length_sig_matrix,
        property_name=climo_utils.COLD_FRONT_LENGTHS_KEY,
        first_property_file_names=first_property_file_names,
        second_property_file_names=second_property_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)
    print(SEPARATOR_STRING)

    cf_area_diff_matrix_m2, cf_area_sig_matrix = _mc_test_one_property(
        first_property_matrix=first_property_dict[
            climo_utils.COLD_FRONT_AREAS_KEY],
        second_property_matrix=second_property_dict[
            climo_utils.COLD_FRONT_AREAS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.COLD_FRONT_AREAS_KEY,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        difference_matrix=cf_area_diff_matrix_m2,
        significance_matrix=cf_area_sig_matrix,
        property_name=climo_utils.COLD_FRONT_AREAS_KEY,
        first_property_file_names=first_property_file_names,
        second_property_file_names=second_property_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_property_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_start_time_strings=getattr(
            INPUT_ARG_OBJECT, FIRST_START_TIMES_ARG_NAME),
        first_end_time_strings=getattr(
            INPUT_ARG_OBJECT, FIRST_END_TIMES_ARG_NAME),
        second_start_time_strings=getattr(
            INPUT_ARG_OBJECT, SECOND_START_TIMES_ARG_NAME),
        second_end_time_strings=getattr(
            INPUT_ARG_OBJECT, SECOND_END_TIMES_ARG_NAME),
        first_grid_row=getattr(INPUT_ARG_OBJECT, FIRST_ROW_ARG_NAME),
        last_grid_row=getattr(INPUT_ARG_OBJECT, LAST_ROW_ARG_NAME),
        first_grid_column=getattr(INPUT_ARG_OBJECT, FIRST_COLUMN_ARG_NAME),
        last_grid_column=getattr(INPUT_ARG_OBJECT, LAST_COLUMN_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
