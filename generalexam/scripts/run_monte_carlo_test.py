"""Runs Monte Carlo test for gridded front frequencies or properties."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import climatology_utils as climo_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

YEAR_MONTH_FORMAT = '%Y%m'
TIME_INTERVAL_SEC = 10800

NUM_TIMES_BY_MONTH_KEY = 'num_time_steps_by_month'

INPUT_DIR_ARG_NAME = 'input_dir_name'
FILE_TYPE_ARG_NAME = 'file_type_string'
BASELINE_MONTHS_ARG_NAME = 'baseline_month_strings'
TRIAL_MONTHS_ARG_NAME = 'trial_month_strings'
FIRST_ROW_ARG_NAME = 'first_grid_row'
LAST_ROW_ARG_NAME = 'last_grid_row'
FIRST_COLUMN_ARG_NAME = 'first_grid_column'
LAST_COLUMN_ARG_NAME = 'last_grid_column'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (one per month) will be found by '
    '`climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_counts` or '
    '`climatology_utils.read_gridded_stats`.')

FILE_TYPE_HELP_STRING = (
    'File type (determines whether this script will test front frequencies or '
    'statistics).  Must be in the following list:\n{0:s}'
).format(str(climo_utils.AGGREGATED_FILE_TYPE_STRINGS))

BASELINE_MONTHS_HELP_STRING = (
    'List of months (format "yyyymm") for baseline composite.')

TRIAL_MONTHS_HELP_STRING = (
    'List of months (format "yyyymm") for trial composite.')

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

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`climatology_utils.write_monte_carlo_test`, to exact locations determined '
    'by `climatology_utils.find_monte_carlo_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FILE_TYPE_ARG_NAME, type=str, required=True,
    help=FILE_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_MONTHS_ARG_NAME, nargs='+', type=str, required=True,
    help=BASELINE_MONTHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_MONTHS_ARG_NAME, nargs='+', type=str, required=True,
    help=TRIAL_MONTHS_HELP_STRING)

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


def _months_to_start_end_times(month_strings):
    """Converts list of months to list of start/end times.

    T = number of months

    :param month_strings: length-T list of months (format "yyyymm").
    :return: start_times_unix_sec: length-T numpy array of month-start times.
    :return: end_times_unix_sec: length-T numpy array of month-end times.
    """

    months_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(m, YEAR_MONTH_FORMAT)
        for m in month_strings
    ], dtype=int)

    start_times_unix_sec = numpy.array([
        time_conversion.first_and_last_times_in_month(m)[0]
        for m in months_unix_sec
    ], dtype=int)

    end_times_unix_sec = numpy.array([
        time_conversion.first_and_last_times_in_month(m)[1]
        for m in months_unix_sec
    ], dtype=int)

    end_times_unix_sec += 1 - TIME_INTERVAL_SEC

    return start_times_unix_sec, end_times_unix_sec


def _get_weighted_mean_for_statistic(num_labels_matrix, statistic_matrix):
    """Computes weighted mean for one statistic.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param num_labels_matrix: T-by-M-by-N numpy array with number of labels
        (fronts) used to compute statistic.
    :param statistic_matrix: T-by-M-by-N numpy array with values of statistic.
    :return: mean_statistic__matrix: M-by-N numpy array with mean statistic at
        each grid cell (over all T time steps)
    """

    count_matrix = numpy.sum(num_labels_matrix, axis=0).astype(float)
    sum_statistic_matrix = numpy.nansum(
        num_labels_matrix * statistic_matrix, axis=0
    )

    count_matrix[count_matrix == 0] = numpy.nan
    return sum_statistic_matrix / count_matrix


def _read_frequencies_one_composite(
        count_file_names, first_grid_row, last_grid_row, first_grid_column,
        last_grid_column):
    """Reads gridded front frequencies for one composite.

    T = number of months
    M = number of rows in subgrid
    N = number of columns in subgrid

    :param count_file_names: 1-D list of paths to input files.
    :param first_grid_row: See documentation at top of file.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :return: front_count_dict: Dictionary with the following keys.
    front_count_dict["num_wf_labels_matrix"]: T-by-M-by-N numpy array with
        number of warm fronts.
    front_count_dict["num_cf_labels_matrix"]: Same but for cold fronts.
    front_count_dict["num_time_steps_by_month"]: length-T numpy array with
        number of time steps (max possible number of fronts) in each month.
    """

    num_months = len(count_file_names)

    num_wf_labels_matrix = None
    num_cf_labels_matrix = None
    num_time_steps_by_month = numpy.full(num_months, -1, dtype=int)

    for i in range(num_months):
        print('Reading front counts from: "{0:s}"...'.format(
            count_file_names[i]
        ))

        this_count_dict = climo_utils.read_gridded_counts(count_file_names[i])

        num_time_steps_by_month[i] = 1 + int(numpy.round(
            float(
                this_count_dict[climo_utils.LAST_TIME_KEY] -
                this_count_dict[climo_utils.FIRST_TIME_KEY]
            )
            / TIME_INTERVAL_SEC
        ))

        print('Number of time steps in month = {0:d}'.format(
            num_time_steps_by_month[i]
        ))

        this_num_wf_labels_matrix = (
            this_count_dict[climo_utils.NUM_WF_LABELS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_num_cf_labels_matrix = (
            this_count_dict[climo_utils.NUM_CF_LABELS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        if num_wf_labels_matrix is None:
            num_grid_rows = this_num_wf_labels_matrix.shape[0]
            num_grid_columns = this_num_wf_labels_matrix.shape[1]
            dimensions = (num_months, num_grid_rows, num_grid_columns)

            num_wf_labels_matrix = numpy.full(dimensions, numpy.nan)
            num_cf_labels_matrix = numpy.full(dimensions, numpy.nan)

        num_wf_labels_matrix[i, ...] = this_num_wf_labels_matrix
        num_cf_labels_matrix[i, ...] = this_num_cf_labels_matrix

    return {
        climo_utils.NUM_WF_LABELS_KEY: num_wf_labels_matrix,
        climo_utils.NUM_CF_LABELS_KEY: num_cf_labels_matrix,
        NUM_TIMES_BY_MONTH_KEY: num_time_steps_by_month
    }


def _read_stats_one_composite(
        statistic_file_names, first_grid_row, last_grid_row, first_grid_column,
        last_grid_column):
    """Reads gridded front statistics for one composite.

    T = number of months
    M = number of rows in subgrid
    N = number of columns in subgrid

    :param statistic_file_names: 1-D list of paths to input files.
    :param first_grid_row: See documentation at top of file.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :return: front_statistic_dict: Dictionary with the following keys.
    front_statistic_dict["num_wf_labels_matrix"]: T-by-M-by-N numpy array with
        number of warm fronts used to compute stats.
    front_statistic_dict["mean_wf_length_matrix_metres"]: Same but for
        warm-front length.
    front_statistic_dict["mean_wf_area_matrix_m2"]: Same but for warm-front
        area.
    front_statistic_dict["num_cf_labels_matrix"]: T-by-M-by-N numpy array with
        number of cold fronts used to compute stats.
    front_statistic_dict["mean_cf_length_matrix_metres"]: Same but for
        cold-front length.
    front_statistic_dict["mean_cf_area_matrix_m2"]: Same but for cold-front
        area.
    """

    num_months = len(statistic_file_names)

    num_wf_labels_matrix = None
    mean_wf_length_matrix_metres = None
    mean_wf_area_matrix_m2 = None
    num_cf_labels_matrix = None
    mean_cf_length_matrix_metres = None
    mean_cf_area_matrix_m2 = None

    for i in range(num_months):
        print('Reading front statistics from: "{0:s}"...'.format(
            statistic_file_names[i]
        ))

        this_statistic_dict = climo_utils.read_gridded_stats(
            statistic_file_names[i]
        )

        this_num_wf_labels_matrix = (
            this_statistic_dict[climo_utils.NUM_WF_LABELS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_wf_length_matrix_metres = (
            this_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_wf_area_matrix_m2 = (
            this_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_num_cf_labels_matrix = (
            this_statistic_dict[climo_utils.NUM_CF_LABELS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_cf_length_matrix_metres = (
            this_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        this_cf_area_matrix_m2 = (
            this_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ]
        )

        if num_wf_labels_matrix is None:
            num_grid_rows = this_num_wf_labels_matrix.shape[0]
            num_grid_columns = this_num_wf_labels_matrix.shape[1]
            dimensions = (num_months, num_grid_rows, num_grid_columns)

            num_wf_labels_matrix = numpy.full(dimensions, numpy.nan)
            mean_wf_length_matrix_metres = numpy.full(dimensions, numpy.nan)
            mean_wf_area_matrix_m2 = numpy.full(dimensions, numpy.nan)
            num_cf_labels_matrix = numpy.full(dimensions, numpy.nan)
            mean_cf_length_matrix_metres = numpy.full(dimensions, numpy.nan)
            mean_cf_area_matrix_m2 = numpy.full(dimensions, numpy.nan)

        num_wf_labels_matrix[i, ...] = this_num_wf_labels_matrix
        mean_wf_length_matrix_metres[i, ...] = this_wf_length_matrix_metres
        mean_wf_area_matrix_m2[i, ...] = this_wf_area_matrix_m2
        num_cf_labels_matrix[i, ...] = this_num_cf_labels_matrix
        mean_cf_length_matrix_metres[i, ...] = this_cf_length_matrix_metres
        mean_cf_area_matrix_m2[i, ...] = this_cf_area_matrix_m2

    return {
        climo_utils.NUM_WF_LABELS_KEY: num_wf_labels_matrix,
        climo_utils.MEAN_WF_LENGTHS_KEY: mean_wf_length_matrix_metres,
        climo_utils.MEAN_WF_AREAS_KEY: mean_wf_area_matrix_m2,
        climo_utils.NUM_CF_LABELS_KEY: num_cf_labels_matrix,
        climo_utils.MEAN_CF_LENGTHS_KEY: mean_cf_length_matrix_metres,
        climo_utils.MEAN_CF_AREAS_KEY: mean_cf_area_matrix_m2
    }


def _mc_test_frequency(
        baseline_num_labels_matrix, baseline_time_step_counts,
        trial_num_labels_matrix, trial_time_step_counts, num_iterations,
        confidence_level):
    """Runs Monte Carlo test for frequency of one front type.

    B = number of months in baseline composite
    T = number of months in trial composite
    M = number of rows in grid
    N = number of columns in grid

    :param baseline_num_labels_matrix: B-by-M-by-N numpy array with number of
        fronts.
    :param baseline_time_step_counts: length-B numpy array with number of time
        steps in each month.
    :param trial_num_labels_matrix: T-by-M-by-N numpy array with number of
        fronts.
    :param trial_time_step_counts: length-T numpy array with number of time
        steps in each month.
    :param num_iterations: See documentation at top of file.
    :param confidence_level: Same.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between frequencies is significant.
    :return: baseline_freq_matrix: M-by-N numpy array with overall frequencies
        for baseline composite.
    :return: trial_freq_matrix: Same but for trial composite.
    """

    concat_num_labels_matrix = numpy.concatenate(
        (baseline_num_labels_matrix, trial_num_labels_matrix), axis=0
    )
    concat_time_step_counts = numpy.concatenate(
        (baseline_time_step_counts, trial_time_step_counts), axis=0
    )

    num_baseline_months = baseline_num_labels_matrix.shape[0]
    num_months = concat_num_labels_matrix.shape[0]
    month_indices = numpy.linspace(0, num_months - 1, num=num_months, dtype=int)

    num_grid_rows = baseline_num_labels_matrix.shape[1]
    num_grid_columns = baseline_num_labels_matrix.shape[2]
    mc_difference_matrix = numpy.full(
        (num_iterations, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_iterations):
        if numpy.mod(k, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                k, num_iterations
            ))

        numpy.random.shuffle(month_indices)

        this_baseline_freq_matrix = numpy.sum(
            concat_num_labels_matrix[month_indices[:num_baseline_months], ...],
            axis=0
        )

        this_baseline_freq_matrix = (
            this_baseline_freq_matrix /
            numpy.sum(concat_time_step_counts[:num_baseline_months])
        )

        this_trial_freq_matrix = numpy.sum(
            concat_num_labels_matrix[month_indices[num_baseline_months:], ...],
            axis=0
        )

        this_trial_freq_matrix = (
            this_trial_freq_matrix /
            numpy.sum(concat_time_step_counts[num_baseline_months:])
        )

        mc_difference_matrix[k, ...] = (
            this_trial_freq_matrix - this_baseline_freq_matrix
        )

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))

    baseline_freq_matrix = (
        numpy.sum(baseline_num_labels_matrix, axis=0) /
        numpy.sum(baseline_time_step_counts)
    )

    trial_freq_matrix = (
        numpy.sum(trial_num_labels_matrix, axis=0) /
        numpy.sum(trial_time_step_counts)
    )

    actual_difference_matrix = trial_freq_matrix - baseline_freq_matrix

    min_frequency_diff_matrix = numpy.percentile(
        a=mc_difference_matrix, q=50. * (1 - confidence_level), axis=0
    )
    max_frequency_diff_matrix = numpy.percentile(
        a=mc_difference_matrix, q=50. * (1 + confidence_level), axis=0
    )

    significance_matrix = numpy.logical_or(
        actual_difference_matrix < min_frequency_diff_matrix,
        actual_difference_matrix > max_frequency_diff_matrix
    )

    print((
        'Difference between frequencies is significant at {0:d} of {1:d} grid '
        'cells!'
    ).format(
        numpy.sum(significance_matrix.astype(int)), significance_matrix.size
    ))

    return significance_matrix, baseline_freq_matrix, trial_freq_matrix


def _mc_test_one_statistic(
        baseline_num_labels_matrix, baseline_stat_matrix,
        trial_num_labels_matrix, trial_stat_matrix, num_iterations,
        confidence_level):
    """Runs Monte Carlo test for one statistic.

    The "one statistic" could be WF length, WF area, CF length, or CF area.

    B = number of months in baseline composite
    T = number of months in trial composite
    M = number of rows in grid
    N = number of columns in grid

    :param baseline_num_labels_matrix: B-by-M-by-N numpy array with number of
        fronts used to compute statistic.
    :param baseline_stat_matrix: B-by-M-by-N numpy array with statistic itself.
    :param trial_num_labels_matrix: T-by-M-by-N numpy array with number of
        fronts used to compute statistic.
    :param trial_stat_matrix: T-by-M-by-N numpy array with statistic itself.
    :param num_iterations: See documentation at top of file.
    :param confidence_level: Same.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between means is significant.
    :return: baseline_mean_matrix: M-by-N numpy array with mean values for
        baseline composite.
    :return: trial_mean_matrix: Same but for trial composite.
    """

    concat_num_labels_matrix = numpy.concatenate(
        (baseline_num_labels_matrix, trial_num_labels_matrix), axis=0
    )
    concat_stat_matrix = numpy.concatenate(
        (baseline_stat_matrix, trial_stat_matrix), axis=0
    )

    num_baseline_months = baseline_stat_matrix.shape[0]
    num_months = concat_stat_matrix.shape[0]
    month_indices = numpy.linspace(0, num_months - 1, num=num_months, dtype=int)

    num_grid_rows = baseline_stat_matrix.shape[1]
    num_grid_columns = baseline_stat_matrix.shape[2]
    mc_difference_matrix = numpy.full(
        (num_iterations, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_iterations):
        if numpy.mod(k, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                k, num_iterations
            ))

        numpy.random.shuffle(month_indices)

        this_baseline_mean_matrix = _get_weighted_mean_for_statistic(
            num_labels_matrix=
            concat_num_labels_matrix[month_indices[:num_baseline_months], ...],
            statistic_matrix=
            concat_stat_matrix[month_indices[:num_baseline_months], ...]
        )

        this_trial_mean_matrix = _get_weighted_mean_for_statistic(
            num_labels_matrix=
            concat_num_labels_matrix[month_indices[num_baseline_months:], ...],
            statistic_matrix=
            concat_stat_matrix[month_indices[num_baseline_months:], ...]
        )

        mc_difference_matrix[k, ...] = (
            this_trial_mean_matrix - this_baseline_mean_matrix
        )

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))

    baseline_mean_matrix = _get_weighted_mean_for_statistic(
        num_labels_matrix=baseline_num_labels_matrix,
        statistic_matrix=baseline_stat_matrix)

    trial_mean_matrix = _get_weighted_mean_for_statistic(
        num_labels_matrix=trial_num_labels_matrix,
        statistic_matrix=trial_stat_matrix)

    actual_difference_matrix = trial_mean_matrix - baseline_mean_matrix

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

    return significance_matrix, baseline_mean_matrix, trial_mean_matrix


def _run(input_dir_name, file_type_string, baseline_month_strings,
         trial_month_strings, first_grid_row, last_grid_row, first_grid_column,
         last_grid_column, num_iterations, confidence_level, output_dir_name):
    """Runs Monte Carlo significance test for gridded front properties.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param file_type_string: Same.
    :param baseline_month_strings: Same.
    :param trial_month_strings: Same.
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
    error_checking.assert_is_geq(num_iterations, 10)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    baseline_start_times_unix_sec, baseline_end_times_unix_sec = (
        _months_to_start_end_times(baseline_month_strings)
    )

    trial_start_times_unix_sec, trial_end_times_unix_sec = (
        _months_to_start_end_times(trial_month_strings)
    )

    baseline_input_file_names = [
        climo_utils.find_aggregated_file(
            directory_name=input_dir_name, file_type_string=file_type_string,
            first_time_unix_sec=f, last_time_unix_sec=l
        ) for f, l in zip(
            baseline_start_times_unix_sec, baseline_end_times_unix_sec
        )
    ]

    trial_input_file_names = [
        climo_utils.find_aggregated_file(
            directory_name=input_dir_name, file_type_string=file_type_string,
            first_time_unix_sec=f, last_time_unix_sec=l
        ) for f, l in zip(trial_start_times_unix_sec, trial_end_times_unix_sec)
    ]

    if file_type_string == climo_utils.FRONT_STATS_STRING:
        baseline_statistic_dict = _read_stats_one_composite(
            statistic_file_names=baseline_input_file_names,
            first_grid_row=first_grid_row, last_grid_row=last_grid_row,
            first_grid_column=first_grid_column,
            last_grid_column=last_grid_column)
        print(SEPARATOR_STRING)

        trial_statistic_dict = _read_stats_one_composite(
            statistic_file_names=trial_input_file_names,
            first_grid_row=first_grid_row, last_grid_row=last_grid_row,
            first_grid_column=first_grid_column,
            last_grid_column=last_grid_column)
        print(SEPARATOR_STRING)

        this_sig_matrix, this_baseline_mean_matrix, this_trial_mean_matrix = (
            _mc_test_one_statistic(
                baseline_num_labels_matrix=baseline_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                baseline_stat_matrix=baseline_statistic_dict[
                    climo_utils.MEAN_WF_LENGTHS_KEY],
                trial_num_labels_matrix=trial_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                trial_stat_matrix=trial_statistic_dict[
                    climo_utils.MEAN_WF_LENGTHS_KEY],
                num_iterations=num_iterations,
                confidence_level=confidence_level)
        )
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_sig_matrix,
            property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_sig_matrix, this_baseline_mean_matrix, this_trial_mean_matrix = (
            _mc_test_one_statistic(
                baseline_num_labels_matrix=baseline_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                baseline_stat_matrix=baseline_statistic_dict[
                    climo_utils.MEAN_WF_AREAS_KEY],
                trial_num_labels_matrix=trial_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                trial_stat_matrix=trial_statistic_dict[
                    climo_utils.MEAN_WF_AREAS_KEY],
                num_iterations=num_iterations,
                confidence_level=confidence_level)
        )
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.WF_AREA_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_sig_matrix,
            property_name=climo_utils.WF_AREA_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_sig_matrix, this_baseline_mean_matrix, this_trial_mean_matrix = (
            _mc_test_one_statistic(
                baseline_num_labels_matrix=baseline_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                baseline_stat_matrix=baseline_statistic_dict[
                    climo_utils.MEAN_CF_LENGTHS_KEY],
                trial_num_labels_matrix=trial_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                trial_stat_matrix=trial_statistic_dict[
                    climo_utils.MEAN_CF_LENGTHS_KEY],
                num_iterations=num_iterations,
                confidence_level=confidence_level)
        )
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_sig_matrix,
            property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_sig_matrix, this_baseline_mean_matrix, this_trial_mean_matrix = (
            _mc_test_one_statistic(
                baseline_num_labels_matrix=baseline_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                baseline_stat_matrix=baseline_statistic_dict[
                    climo_utils.MEAN_CF_AREAS_KEY],
                trial_num_labels_matrix=trial_statistic_dict[
                    climo_utils.NUM_WF_LABELS_KEY],
                trial_stat_matrix=trial_statistic_dict[
                    climo_utils.MEAN_CF_AREAS_KEY],
                num_iterations=num_iterations,
                confidence_level=confidence_level)
        )
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.CF_AREA_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_sig_matrix,
            property_name=climo_utils.CF_AREA_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)

        return

    baseline_count_dict = _read_frequencies_one_composite(
        count_file_names=baseline_input_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    trial_count_dict = _read_frequencies_one_composite(
        count_file_names=trial_input_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    this_sig_matrix, this_baseline_freq_matrix, this_trial_freq_matrix = (
        _mc_test_frequency(
            baseline_num_labels_matrix=baseline_count_dict[
                climo_utils.NUM_WF_LABELS_KEY],
            baseline_time_step_counts=baseline_count_dict[
                NUM_TIMES_BY_MONTH_KEY],
            trial_num_labels_matrix=trial_count_dict[
                climo_utils.NUM_WF_LABELS_KEY],
            trial_time_step_counts=trial_count_dict[NUM_TIMES_BY_MONTH_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        baseline_mean_matrix=this_baseline_freq_matrix,
        trial_mean_matrix=this_trial_freq_matrix,
        significance_matrix=this_sig_matrix,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        baseline_input_file_names=baseline_input_file_names,
        trial_input_file_names=trial_input_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)

    this_sig_matrix, this_baseline_freq_matrix, this_trial_freq_matrix = (
        _mc_test_frequency(
            baseline_num_labels_matrix=baseline_count_dict[
                climo_utils.NUM_CF_LABELS_KEY],
            baseline_time_step_counts=baseline_count_dict[
                NUM_TIMES_BY_MONTH_KEY],
            trial_num_labels_matrix=trial_count_dict[
                climo_utils.NUM_CF_LABELS_KEY],
            trial_time_step_counts=trial_count_dict[NUM_TIMES_BY_MONTH_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
    )
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        baseline_mean_matrix=this_baseline_freq_matrix,
        trial_mean_matrix=this_trial_freq_matrix,
        significance_matrix=this_sig_matrix,
        property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
        baseline_input_file_names=baseline_input_file_names,
        trial_input_file_names=trial_input_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        file_type_string=getattr(INPUT_ARG_OBJECT, FILE_TYPE_ARG_NAME),
        baseline_month_strings=getattr(
            INPUT_ARG_OBJECT, BASELINE_MONTHS_ARG_NAME),
        trial_month_strings=getattr(INPUT_ARG_OBJECT, TRIAL_MONTHS_ARG_NAME),
        first_grid_row=getattr(INPUT_ARG_OBJECT, FIRST_ROW_ARG_NAME),
        last_grid_row=getattr(INPUT_ARG_OBJECT, LAST_ROW_ARG_NAME),
        first_grid_column=getattr(INPUT_ARG_OBJECT, FIRST_COLUMN_ARG_NAME),
        last_grid_column=getattr(INPUT_ARG_OBJECT, LAST_COLUMN_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
