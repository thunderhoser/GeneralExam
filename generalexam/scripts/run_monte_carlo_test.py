"""Runs Monte Carlo test for gridded front frequencies or properties."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800

WF_FLAGS_KEY = 'wf_flag_matrix'
CF_FLAGS_KEY = 'cf_flag_matrix'

INPUT_DIR_ARG_NAME = 'input_dir_name'
FILE_TYPE_ARG_NAME = 'file_type_string'
BASELINE_START_TIMES_ARG_NAME = 'baseline_start_time_strings'
BASELINE_END_TIMES_ARG_NAME = 'baseline_end_time_strings'
TRIAL_START_TIMES_ARG_NAME = 'trial_start_time_strings'
TRIAL_END_TIMES_ARG_NAME = 'trial_end_time_strings'
FIRST_ROW_ARG_NAME = 'first_grid_row'
LAST_ROW_ARG_NAME = 'last_grid_row'
FIRST_COLUMN_ARG_NAME = 'first_grid_column'
LAST_COLUMN_ARG_NAME = 'last_grid_column'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_basic_file` and read by '
    '`climatology_utils.read_gridded_labels` or '
    '`climatology_utils.read_gridded_properties`.')

FILE_TYPE_HELP_STRING = (
    'File type (determines whether this script will test front frequencies or '
    'properties).  Must be in the following list:\n{0:s}'
).format(str(climo_utils.BASIC_FILE_TYPE_STRINGS))

BASELINE_START_TIMES_HELP_STRING = (
    'List of start times (format "yyyymmddHH") for baseline composite.')

BASELINE_END_TIMES_HELP_STRING = (
    'List of end times (format "yyyymmddHH") for baseline composite.')

TRIAL_START_TIMES_HELP_STRING = (
    'List of start times (format "yyyymmddHH") for trial composite.')

TRIAL_END_TIMES_HELP_STRING = (
    'List of end times (format "yyyymmddHH") for trial composite.')

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
    '--' + BASELINE_START_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=BASELINE_START_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_END_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=BASELINE_END_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_START_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=TRIAL_START_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_END_TIMES_ARG_NAME, nargs='+', type=str, required=True,
    help=TRIAL_END_TIMES_HELP_STRING)

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
        `baseline_start_time_strings` or `trial_start_time_strings`.
    :param end_time_strings: See doc at top of file for either
        `baseline_end_time_strings` or `trial_end_time_strings`.
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


def _read_labels_one_composite(
        label_file_names, first_grid_row, last_grid_row, first_grid_column,
        last_grid_column):
    """Reads gridded front labels for one composite.

    T = number of time steps
    M = number of rows in subgrid
    N = number of columns in subgrid

    :param label_file_names: 1-D list of paths to input files.
    :param first_grid_row: See documentation at top of file.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :return: front_label_dict: Dictionary with the following keys.
    front_label_dict["wf_flag_matrix"]: T-by-M-by-N numpy array of flags.  0
        indicates no warm front; 1 indicates warm front; NaN indicates no data.
    front_label_dict["cf_flag_matrix"]: Same but for cold fronts.
    """

    wf_flag_matrix = None
    cf_flag_matrix = None

    for this_file_name in label_file_names:
        print('Reading front labels from: "{0:s}"...'.format(this_file_name))
        this_label_dict = climo_utils.read_gridded_labels(this_file_name)

        this_wf_flag_matrix = numpy.expand_dims(
            this_label_dict[climo_utils.FRONT_LABELS_KEY][
                first_grid_row:(last_grid_row + 1),
                first_grid_column:(last_grid_column + 1)
            ],
            axis=0
        )

        this_cf_flag_matrix = this_wf_flag_matrix + 0.

        this_wf_flag_matrix[
            this_wf_flag_matrix == front_utils.COLD_FRONT_ENUM
        ] = 0.
        this_wf_flag_matrix[
            this_wf_flag_matrix == front_utils.WARM_FRONT_ENUM
        ] = 1.

        this_cf_flag_matrix[
            this_cf_flag_matrix == front_utils.WARM_FRONT_ENUM
        ] = 0.
        this_cf_flag_matrix[
            this_cf_flag_matrix == front_utils.COLD_FRONT_ENUM
        ] = 1.

        if wf_flag_matrix is None:
            wf_flag_matrix = this_wf_flag_matrix + 0.
            cf_flag_matrix = this_cf_flag_matrix + 0.
        else:
            wf_flag_matrix = numpy.concatenate(
                (wf_flag_matrix, this_wf_flag_matrix), axis=0
            )
            cf_flag_matrix = numpy.concatenate(
                (cf_flag_matrix, this_cf_flag_matrix), axis=0
            )

    return {
        WF_FLAGS_KEY: wf_flag_matrix,
        CF_FLAGS_KEY: cf_flag_matrix
    }


def _read_properties_one_composite(
        property_file_names, first_grid_row, last_grid_row, first_grid_column,
        last_grid_column):
    """Reads gridded front properties for one composite.

    T = number of time steps
    M = number of rows in subgrid
    N = number of columns in subgrid

    :param property_file_names: 1-D list of paths to input files.
    :param first_grid_row: See documentation at top of file.
    :param last_grid_row: Same.
    :param first_grid_column: Same.
    :param last_grid_column: Same.
    :return: front_property_dict: Dictionary with the following keys.
    front_property_dict["wf_length_matrix_metres"]: T-by-M-by-N numpy array of
        warm-front lengths.
    front_property_dict["wf_area_matrix_m2"]: Same but for warm-front areas.
    front_property_dict["cf_length_matrix_metres"]: Same but for cold-front
        lengths.
    front_property_dict["cf_area_matrix_m2"]: Same but for cold-front areas.
    """

    num_times = len(property_file_names)

    wf_length_matrix_metres = None
    wf_area_matrix_m2 = None
    cf_length_matrix_metres = None
    cf_area_matrix_m2 = None

    for i in range(len(property_file_names)):
        print('Reading front properties from: "{0:s}"...'.format(
            property_file_names[i]
        ))

        this_property_dict = climo_utils.read_gridded_properties(
            property_file_names[i]
        )

        this_wf_length_matrix_metres = this_property_dict[
            climo_utils.WARM_FRONT_LENGTHS_KEY
        ][
            first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_wf_area_matrix_m2 = this_property_dict[
            climo_utils.WARM_FRONT_AREAS_KEY
        ][
            first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_cf_length_matrix_metres = this_property_dict[
            climo_utils.COLD_FRONT_LENGTHS_KEY
        ][
            first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        this_cf_area_matrix_m2 = this_property_dict[
            climo_utils.COLD_FRONT_AREAS_KEY
        ][
            first_grid_row:(last_grid_row + 1),
            first_grid_column:(last_grid_column + 1)
        ]

        if wf_length_matrix_metres is None:
            num_grid_rows = this_wf_length_matrix_metres.shape[0]
            num_grid_columns = this_wf_length_matrix_metres.shape[1]

            wf_length_matrix_metres = numpy.full(
                (num_times, num_grid_rows, num_grid_columns), numpy.nan
            )
            wf_area_matrix_m2 = wf_length_matrix_metres + 0.
            cf_length_matrix_metres = wf_length_matrix_metres + 0.
            cf_area_matrix_m2 = wf_length_matrix_metres + 0.

        wf_length_matrix_metres[i, ...] = this_wf_length_matrix_metres
        wf_area_matrix_m2[i, ...] = this_wf_area_matrix_m2
        cf_length_matrix_metres[i, ...] = this_cf_length_matrix_metres
        cf_area_matrix_m2[i, ...] = this_cf_area_matrix_m2

    return {
        climo_utils.WARM_FRONT_LENGTHS_KEY: wf_length_matrix_metres,
        climo_utils.WARM_FRONT_AREAS_KEY: wf_area_matrix_m2,
        climo_utils.COLD_FRONT_LENGTHS_KEY: cf_length_matrix_metres,
        climo_utils.COLD_FRONT_AREAS_KEY: cf_area_matrix_m2
    }


def _mc_test_frequency(
        baseline_flag_matrix, trial_flag_matrix, num_iterations,
        confidence_level):
    """Runs Monte Carlo for frequency of one front type (warm or cold).

    F = number of times in baseline composite
    S = number of times in trial composite
    M = number of rows in grid
    N = number of columns in grid

    :param baseline_flag_matrix: F-by-M-by-N numpy array of flags.  1 indicates a
        front; 0 indicates no front; NaN indicates no data.
    :param trial_flag_matrix: S-by-M-by-N numpy array with same format.
    :param num_iterations: See documentation at top of file.
    :param confidence_level: Same.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between frequencies is significant.
    """

    num_baseline_times = baseline_flag_matrix.shape[0]
    concat_label_matrix = numpy.concatenate(
        (baseline_flag_matrix, trial_flag_matrix), axis=0
    )

    num_grid_rows = baseline_flag_matrix.shape[1]
    num_grid_columns = baseline_flag_matrix.shape[2]
    mc_frequency_diff_matrix = numpy.full(
        (num_iterations, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_iterations):
        if numpy.mod(k, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                k, num_iterations
            ))

        numpy.random.shuffle(concat_label_matrix)
        this_baseline_freq_matrix = numpy.mean(
            concat_label_matrix[:num_baseline_times, ...], axis=0
        )
        this_trial_freq_matrix = numpy.mean(
            concat_label_matrix[num_baseline_times:, ...], axis=0
        )

        mc_frequency_diff_matrix[k, ...] = (
            this_trial_freq_matrix - this_baseline_freq_matrix
        )

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))

    actual_frequency_diff_matrix = (
        numpy.mean(trial_flag_matrix, axis=0) -
        numpy.mean(baseline_flag_matrix, axis=0)
    )

    min_frequency_diff_matrix = numpy.percentile(
        a=mc_frequency_diff_matrix, q=50. * (1 - confidence_level), axis=0
    )
    max_frequency_diff_matrix = numpy.percentile(
        a=mc_frequency_diff_matrix, q=50. * (1 + confidence_level), axis=0
    )

    significance_matrix = numpy.logical_or(
        actual_frequency_diff_matrix < min_frequency_diff_matrix,
        actual_frequency_diff_matrix > max_frequency_diff_matrix
    )

    print((
        'Difference between frequencies is significant at {0:d} of {1:d} grid '
        'cells!'
    ).format(
        numpy.sum(significance_matrix.astype(int)), significance_matrix.size
    ))

    return significance_matrix


def _mc_test_one_property(
        baseline_property_matrix, trial_property_matrix, num_iterations,
        confidence_level):
    """Runs Monte Carlo test for one property.

    The "one property" could be WF length, WF area, CF length, or CF area.

    F = number of times in baseline composite
    S = number of times in trial composite
    M = number of rows in grid
    N = number of columns in grid

    :param baseline_property_matrix: F-by-M-by-N numpy array with values of given
        property.
    :param trial_property_matrix: S-by-M-by-N numpy array with values of given
        property.
    :param num_iterations: See documentation at top of file.
    :param confidence_level: Same.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between means is significant.
    """

    num_baseline_times = baseline_property_matrix.shape[0]
    concat_property_matrix = numpy.concatenate(
        (baseline_property_matrix, trial_property_matrix), axis=0
    )

    num_grid_rows = baseline_property_matrix.shape[1]
    num_grid_columns = baseline_property_matrix.shape[2]
    mc_difference_matrix = numpy.full(
        (num_iterations, num_grid_rows, num_grid_columns), numpy.nan
    )

    for k in range(num_iterations):
        if numpy.mod(k, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                k, num_iterations
            ))

        numpy.random.shuffle(concat_property_matrix)

        # numpy.nanmean on all-NaN slice returns NaN.
        this_baseline_mean_matrix = numpy.nanmean(
            concat_property_matrix[:num_baseline_times, ...], axis=0
        )
        this_trial_mean_matrix = numpy.nanmean(
            concat_property_matrix[num_baseline_times:, ...], axis=0
        )

        mc_difference_matrix[k, ...] = (
            this_trial_mean_matrix - this_baseline_mean_matrix
        )

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))

    actual_difference_matrix = (
        numpy.nanmean(trial_property_matrix, axis=0) -
        numpy.nanmean(baseline_property_matrix, axis=0)
    )

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

    return significance_matrix


def _run(input_dir_name, file_type_string, baseline_start_time_strings,
         baseline_end_time_strings, trial_start_time_strings,
         trial_end_time_strings, first_grid_row, last_grid_row,
         first_grid_column, last_grid_column, num_iterations, confidence_level,
         output_dir_name):
    """Runs Monte Carlo significance test for gridded front properties.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param file_type_string: Same.
    :param baseline_start_time_strings: Same.
    :param baseline_end_time_strings: Same.
    :param trial_start_time_strings: Same.
    :param trial_end_time_strings: Same.
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

    baseline_times_unix_sec = _convert_input_times(
        start_time_strings=baseline_start_time_strings,
        end_time_strings=baseline_end_time_strings)

    trial_times_unix_sec = _convert_input_times(
        start_time_strings=trial_start_time_strings,
        end_time_strings=trial_end_time_strings)

    baseline_input_file_names = [
        climo_utils.find_basic_file(
            directory_name=input_dir_name, file_type_string=file_type_string,
            valid_time_unix_sec=t
        )
        for t in baseline_times_unix_sec
    ]

    trial_input_file_names = [
        climo_utils.find_basic_file(
            directory_name=input_dir_name, file_type_string=file_type_string,
            valid_time_unix_sec=t
        )
        for t in trial_times_unix_sec
    ]

    if file_type_string == climo_utils.FRONT_PROPERTIES_STRING:
        baseline_property_dict = _read_properties_one_composite(
            property_file_names=baseline_input_file_names,
            first_grid_row=first_grid_row, last_grid_row=last_grid_row,
            first_grid_column=first_grid_column,
            last_grid_column=last_grid_column)
        print(SEPARATOR_STRING)

        trial_property_dict = _read_properties_one_composite(
            property_file_names=trial_input_file_names,
            first_grid_row=first_grid_row, last_grid_row=last_grid_row,
            first_grid_column=first_grid_column,
            last_grid_column=last_grid_column)
        print(SEPARATOR_STRING)

        this_significance_matrix = _mc_test_one_property(
            baseline_property_matrix=baseline_property_dict[
                climo_utils.WARM_FRONT_LENGTHS_KEY],
            trial_property_matrix=trial_property_dict[
                climo_utils.WARM_FRONT_LENGTHS_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_baseline_mean_matrix = numpy.nanmean(
            baseline_property_dict[climo_utils.WARM_FRONT_LENGTHS_KEY], axis=0
        )
        this_trial_mean_matrix = numpy.nanmean(
            trial_property_dict[climo_utils.WARM_FRONT_LENGTHS_KEY], axis=0
        )

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_significance_matrix,
            property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_significance_matrix = _mc_test_one_property(
            baseline_property_matrix=baseline_property_dict[
                climo_utils.WARM_FRONT_AREAS_KEY],
            trial_property_matrix=trial_property_dict[
                climo_utils.WARM_FRONT_AREAS_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_baseline_mean_matrix = numpy.nanmean(
            baseline_property_dict[climo_utils.WARM_FRONT_AREAS_KEY], axis=0
        )
        this_trial_mean_matrix = numpy.nanmean(
            trial_property_dict[climo_utils.WARM_FRONT_AREAS_KEY], axis=0
        )

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.WF_AREA_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_significance_matrix,
            property_name=climo_utils.WF_AREA_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_significance_matrix = _mc_test_one_property(
            baseline_property_matrix=baseline_property_dict[
                climo_utils.COLD_FRONT_LENGTHS_KEY],
            trial_property_matrix=trial_property_dict[
                climo_utils.COLD_FRONT_LENGTHS_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_baseline_mean_matrix = numpy.nanmean(
            baseline_property_dict[climo_utils.COLD_FRONT_LENGTHS_KEY], axis=0
        )
        this_trial_mean_matrix = numpy.nanmean(
            trial_property_dict[climo_utils.COLD_FRONT_LENGTHS_KEY], axis=0
        )

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_significance_matrix,
            property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)
        print(SEPARATOR_STRING)

        this_significance_matrix = _mc_test_one_property(
            baseline_property_matrix=baseline_property_dict[
                climo_utils.COLD_FRONT_AREAS_KEY],
            trial_property_matrix=trial_property_dict[
                climo_utils.COLD_FRONT_AREAS_KEY],
            num_iterations=num_iterations, confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_baseline_mean_matrix = numpy.nanmean(
            baseline_property_dict[climo_utils.COLD_FRONT_AREAS_KEY], axis=0
        )
        this_trial_mean_matrix = numpy.nanmean(
            trial_property_dict[climo_utils.COLD_FRONT_AREAS_KEY], axis=0
        )

        this_output_file_name = climo_utils.find_monte_carlo_file(
            directory_name=output_dir_name,
            property_name=climo_utils.CF_AREA_PROPERTY_NAME,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column,
            raise_error_if_missing=False)

        climo_utils.write_monte_carlo_test(
            netcdf_file_name=this_output_file_name,
            baseline_mean_matrix=this_baseline_mean_matrix,
            trial_mean_matrix=this_trial_mean_matrix,
            significance_matrix=this_significance_matrix,
            property_name=climo_utils.CF_AREA_PROPERTY_NAME,
            baseline_input_file_names=baseline_input_file_names,
            trial_input_file_names=trial_input_file_names,
            num_iterations=num_iterations, confidence_level=confidence_level,
            first_grid_row=first_grid_row, first_grid_column=first_grid_column)

        return

    baseline_label_dict = _read_labels_one_composite(
        label_file_names=baseline_input_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    trial_label_dict = _read_labels_one_composite(
        label_file_names=trial_input_file_names,
        first_grid_row=first_grid_row, last_grid_row=last_grid_row,
        first_grid_column=first_grid_column, last_grid_column=last_grid_column)
    print(SEPARATOR_STRING)

    this_significance_matrix = _mc_test_frequency(
        baseline_flag_matrix=baseline_label_dict[WF_FLAGS_KEY],
        trial_flag_matrix=trial_label_dict[WF_FLAGS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_baseline_mean_matrix = numpy.mean(
        baseline_label_dict[WF_FLAGS_KEY], axis=0
    )
    this_trial_mean_matrix = numpy.mean(
        trial_label_dict[WF_FLAGS_KEY], axis=0
    )

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        baseline_mean_matrix=this_baseline_mean_matrix,
        trial_mean_matrix=this_trial_mean_matrix,
        significance_matrix=this_significance_matrix,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        baseline_input_file_names=baseline_input_file_names,
        trial_input_file_names=trial_input_file_names,
        num_iterations=num_iterations, confidence_level=confidence_level,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column)

    this_significance_matrix = _mc_test_frequency(
        baseline_flag_matrix=baseline_label_dict[CF_FLAGS_KEY],
        trial_flag_matrix=trial_label_dict[CF_FLAGS_KEY],
        num_iterations=num_iterations, confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_baseline_mean_matrix = numpy.mean(
        baseline_label_dict[CF_FLAGS_KEY], axis=0
    )
    this_trial_mean_matrix = numpy.mean(
        trial_label_dict[CF_FLAGS_KEY], axis=0
    )

    this_output_file_name = climo_utils.find_monte_carlo_file(
        directory_name=output_dir_name,
        property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
        first_grid_row=first_grid_row, first_grid_column=first_grid_column,
        raise_error_if_missing=False)

    climo_utils.write_monte_carlo_test(
        netcdf_file_name=this_output_file_name,
        baseline_mean_matrix=this_baseline_mean_matrix,
        trial_mean_matrix=this_trial_mean_matrix,
        significance_matrix=this_significance_matrix,
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
        baseline_start_time_strings=getattr(
            INPUT_ARG_OBJECT, BASELINE_START_TIMES_ARG_NAME),
        baseline_end_time_strings=getattr(
            INPUT_ARG_OBJECT, BASELINE_END_TIMES_ARG_NAME),
        trial_start_time_strings=getattr(
            INPUT_ARG_OBJECT, TRIAL_START_TIMES_ARG_NAME),
        trial_end_time_strings=getattr(
            INPUT_ARG_OBJECT, TRIAL_END_TIMES_ARG_NAME),
        first_grid_row=getattr(INPUT_ARG_OBJECT, FIRST_ROW_ARG_NAME),
        last_grid_row=getattr(INPUT_ARG_OBJECT, LAST_ROW_ARG_NAME),
        first_grid_column=getattr(INPUT_ARG_OBJECT, FIRST_COLUMN_ARG_NAME),
        last_grid_column=getattr(INPUT_ARG_OBJECT, LAST_COLUMN_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
