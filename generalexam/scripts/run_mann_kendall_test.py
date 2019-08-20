"""Runs Mann-Kendall test for gridded front frequencies or statistics."""

import argparse
import numpy
from scipy.interpolate import interp1d as scipy_interp1d
import pymannkendall
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import climatology_utils as climo_utils

TIME_INTERVAL_SEC = 10800
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WF_FREQUENCY_KEY = 'wf_frequency_matrix'
CF_FREQUENCY_KEY = 'cf_frequency_matrix'

INPUT_DIR_ARG_NAME = 'input_dir_name'
FILE_TYPE_ARG_NAME = 'file_type_string'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
SEASON_ARG_NAME = 'season_string'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (one per year) will be found by '
    '`climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_counts` or '
    '`climatology_utils.read_gridded_stats`.')

FILE_TYPE_HELP_STRING = (
    'File type (determines whether this script will test front frequencies or '
    'statistics).  Must be in the following list:\n{0:s}'
).format(str(climo_utils.AGGREGATED_FILE_TYPE_STRINGS))

YEAR_HELP_STRING = (
    'This script will use the Mann-Kendall test to assess linear trends for the'
    ' period `{0:s}`...`{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

SEASON_HELP_STRING = (
    'Season.  If you do *not* want to subset by season, leave this empty.  If '
    'you want to subset by season, must be in the following list:\n{0:s}'
).format(str(climo_utils.VALID_SEASON_STRINGS))

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for two-tailed Mann-Kendall test.  At each grid cell, '
    'positive or negative linear trend will be deemed statistically significant'
    ' iff it reaches or exceeds this confidence level.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`climatology_utils.write_mann_kendall_test`, to exact locations determined '
    'by `climatology_utils.find_mann_kendall_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FILE_TYPE_ARG_NAME, type=str, required=True,
    help=FILE_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEASON_ARG_NAME, type=str, required=False, default='',
    help=SEASON_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _find_input_files(input_dir_name, file_type_string, first_year, last_year,
                      season_string):
    """Finds input files for Mann-Kendall test.

    :param input_dir_name: See documentation at top of file.
    :param file_type_string: Same.
    :param first_year: Same.
    :param last_year: Same.
    :param season_string: Same.
    :return: input_file_name_matrix: 2-D numpy array, where
        input_file_name_matrix[i, j] is the path to the input file for the [i]th
        year and [j]th month.
    """

    if season_string in ['', 'None']:
        desired_months = numpy.linspace(1, 12, num=12, dtype=int)
    else:
        desired_months = climo_utils.season_to_months(season_string)

    num_months = len(desired_months)

    error_checking.assert_is_geq(last_year, first_year + 9)
    num_years = last_year - first_year + 1
    desired_years = numpy.linspace(
        first_year, last_year, num=num_years, dtype=int)

    input_file_name_matrix = numpy.full(
        (num_years, num_months), '', dtype=object
    )

    for i in range(num_years):
        for j in range(num_months):
            this_month_string = '{0:04d}{1:02d}'.format(
                desired_years[i], desired_months[j]
            )

            this_month_unix_sec = time_conversion.string_to_unix_sec(
                this_month_string, '%Y%m')

            this_first_time_unix_sec, this_last_time_unix_sec = (
                time_conversion.first_and_last_times_in_month(
                    this_month_unix_sec)
            )

            this_last_time_unix_sec += 1 - TIME_INTERVAL_SEC

            input_file_name_matrix[i, j] = climo_utils.find_aggregated_file(
                directory_name=input_dir_name,
                file_type_string=file_type_string,
                first_time_unix_sec=this_first_time_unix_sec,
                last_time_unix_sec=this_last_time_unix_sec,
                hours=None, months=None, raise_error_if_missing=True)

    return input_file_name_matrix


def _read_frequencies(input_file_name_matrix):
    """Reads front frequencies into two numpy arrays.

    T = number of years
    M = number of rows in grid
    N = number of columns in grid

    :param input_file_name_matrix: See output doc for `_find_input_files`.
    :return: wf_frequency_matrix: T-by-M-by-N numpy array.
        wf_frequency_matrix[t, i, j] is the frequency of warm fronts (fraction
        of time steps with warm fronts) at grid cell [i, j] in the [t]th year.
    :return: front_count_dict: Dictionary with the following keys.
    front_count_dict["num_wf_labels_matrix"]: M-by-N numpy array with number of
        WF labels used at each grid cell.
    front_count_dict["wf_frequency_matrix"]: T-by-M-by-N numpy array with
        warm-front frequency at each grid cell in each year.
    front_count_dict["num_cf_labels_matrix"]: M-by-N numpy array with number of
        CF labels used at each grid cell.
    front_count_dict["cf_frequency_matrix"]: T-by-M-by-N numpy array with
        cold-front frequency at each grid cell in each year.
    """

    num_years = input_file_name_matrix.shape[0]
    num_months = input_file_name_matrix.shape[1]

    num_wf_labels_matrix = None
    wf_frequency_matrix = None
    num_cf_labels_matrix = None
    cf_frequency_matrix = None

    for i in range(num_years):
        num_times_in_year = 0

        for j in range(num_months):
            print('Reading data from: "{0:s}"...'.format(
                input_file_name_matrix[i, j]
            ))

            this_count_dict = climo_utils.read_gridded_counts(
                input_file_name_matrix[i, j]
            )

            num_times_in_month = 1 + int(numpy.round(
                float(
                    this_count_dict[climo_utils.LAST_TIME_KEY] -
                    this_count_dict[climo_utils.FIRST_TIME_KEY]
                ) / TIME_INTERVAL_SEC
            ))

            print('Number of time steps in file = {0:d}'.format(
                num_times_in_month))

            num_times_in_year += num_times_in_month
            this_num_wf_matrix = this_count_dict[climo_utils.NUM_WF_LABELS_KEY]
            this_num_cf_matrix = this_count_dict[climo_utils.NUM_CF_LABELS_KEY]

            if wf_frequency_matrix is None:
                dimensions = (num_years,) + this_num_wf_matrix.shape
                num_wf_labels_matrix = numpy.full(dimensions, 0.)
                wf_frequency_matrix = numpy.full(dimensions, 0.)
                num_cf_labels_matrix = numpy.full(dimensions, 0.)
                cf_frequency_matrix = numpy.full(dimensions, 0.)

            num_wf_labels_matrix[i, ...] += this_num_wf_matrix
            wf_frequency_matrix[i, ...] += this_num_wf_matrix
            num_cf_labels_matrix[i, ...] += this_num_cf_matrix
            cf_frequency_matrix[i, ...] += this_num_cf_matrix

        wf_frequency_matrix[i, ...] = (
            wf_frequency_matrix[i, ...] / num_times_in_year
        )
        cf_frequency_matrix[i, ...] = (
            cf_frequency_matrix[i, ...] / num_times_in_year
        )

    num_wf_labels_matrix = numpy.round(
        numpy.sum(num_wf_labels_matrix, axis=0)
    ).astype(int)
    num_cf_labels_matrix = numpy.round(
        numpy.sum(num_cf_labels_matrix, axis=0)
    ).astype(int)

    return {
        climo_utils.NUM_WF_LABELS_KEY: num_wf_labels_matrix,
        WF_FREQUENCY_KEY: wf_frequency_matrix,
        climo_utils.NUM_CF_LABELS_KEY: num_cf_labels_matrix,
        CF_FREQUENCY_KEY: cf_frequency_matrix
    }


def _read_statistics(input_file_name_matrix):
    """Reads front statistics into several numpy arrays.

    T = number of years
    M = number of rows in grid
    N = number of columns in grid

    :param input_file_name_matrix: See output doc for `_find_input_files`.
    :return: front_statistic_dict: Dictionary with the following keys.
    front_statistic_dict["num_wf_labels_matrix"]: M-by-N numpy array with number
        of WF labels used at each grid cell.
    front_statistic_dict["mean_wf_length_matrix_metres"]: T-by-M-by-N numpy
        array.  wf_length_matrix_metres[t, i, j] is the average length of warm
        fronts at grid cell [i, j] in the [t]th year.
    front_statistic_dict["mean_cf_length_matrix_metres"]: Same but for cold-
        front length.
    front_statistic_dict["num_cf_labels_matrix"]: M-by-N numpy array with number
        of CF labels used at each grid cell.
    front_statistic_dict["mean_wf_area_matrix_m2"]: Same but for warm-front
        area.
    front_statistic_dict["mean_cf_area_matrix_m2"]: Same but for cold-front
        area.
    """

    num_years = input_file_name_matrix.shape[0]
    num_months = input_file_name_matrix.shape[1]

    num_wf_labels_matrix = None
    sum_wf_length_matrix_metres = None
    sum_wf_area_matrix_m2 = None
    num_cf_labels_matrix = None
    sum_cf_length_matrix_metres = None
    sum_cf_area_matrix_m2 = None

    for i in range(num_years):
        for j in range(num_months):
            print('Reading data from: "{0:s}"...'.format(
                input_file_name_matrix[i, j]
            ))

            this_statistic_dict = climo_utils.read_gridded_stats(
                input_file_name_matrix[i, j]
            )

            this_num_wf_matrix = this_statistic_dict[
                climo_utils.NUM_WF_LABELS_KEY]
            this_num_cf_matrix = this_statistic_dict[
                climo_utils.NUM_CF_LABELS_KEY]

            if sum_wf_length_matrix_metres is None:
                dimensions = (
                    (num_years,) +
                    this_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY].shape
                )

                num_wf_labels_matrix = numpy.full(dimensions, 0.)
                sum_wf_length_matrix_metres = numpy.full(dimensions, 0.)
                sum_wf_area_matrix_m2 = numpy.full(dimensions, 0.)
                num_cf_labels_matrix = numpy.full(dimensions, 0.)
                sum_cf_length_matrix_metres = numpy.full(dimensions, 0.)
                sum_cf_area_matrix_m2 = numpy.full(dimensions, 0.)

            num_wf_labels_matrix[i, ...] += this_num_wf_matrix
            num_cf_labels_matrix[i, ...] += this_num_cf_matrix

            this_sum_matrix = (
                this_num_wf_matrix *
                this_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY]
            )
            this_sum_matrix[numpy.isnan(this_sum_matrix)] = 0
            sum_wf_length_matrix_metres[i, ...] += this_sum_matrix

            this_sum_matrix = (
                this_num_wf_matrix *
                this_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY]
            )
            this_sum_matrix[numpy.isnan(this_sum_matrix)] = 0
            sum_wf_area_matrix_m2[i, ...] += this_sum_matrix

            this_sum_matrix = (
                this_num_cf_matrix *
                this_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY]
            )
            this_sum_matrix[numpy.isnan(this_sum_matrix)] = 0
            sum_cf_length_matrix_metres[i, ...] += this_sum_matrix

            this_sum_matrix = (
                this_num_cf_matrix *
                this_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY]
            )
            this_sum_matrix[numpy.isnan(this_sum_matrix)] = 0
            sum_cf_area_matrix_m2[i, ...] += this_sum_matrix

    num_wf_labels_matrix[num_wf_labels_matrix == 0] = numpy.nan
    num_cf_labels_matrix[num_cf_labels_matrix == 0] = numpy.nan

    front_statistic_dict = {
        climo_utils.MEAN_WF_LENGTHS_KEY:
            sum_wf_length_matrix_metres / num_wf_labels_matrix,
        climo_utils.MEAN_WF_AREAS_KEY:
            sum_wf_area_matrix_m2 / num_wf_labels_matrix,
        climo_utils.MEAN_CF_LENGTHS_KEY:
            sum_cf_length_matrix_metres / num_cf_labels_matrix,
        climo_utils.MEAN_CF_AREAS_KEY:
            sum_cf_area_matrix_m2 / num_cf_labels_matrix
    }

    num_wf_labels_matrix = numpy.round(
        numpy.nansum(num_wf_labels_matrix, axis=0)
    ).astype(int)
    num_cf_labels_matrix = numpy.round(
        numpy.nansum(num_cf_labels_matrix, axis=0)
    ).astype(int)

    front_statistic_dict.update({
        climo_utils.NUM_WF_LABELS_KEY: num_wf_labels_matrix,
        climo_utils.NUM_CF_LABELS_KEY: num_cf_labels_matrix
    })

    return front_statistic_dict


def _mk_test_frequency(frequency_matrix, confidence_level):
    """Runs Mann-Kendall test for frequency of one front type (warm or cold).

    M = number of rows in grid
    N = number of columns in grid

    :param frequency_matrix: See doc for `wf_frequency_matrix` or
        `cf_frequency_matrix` in method `_read_frequencies`.
    :param confidence_level: See documentation at top of file.
    :return: trend_matrix_year01: M-by-N numpy array of Theil-Sen slopes.  Units
        are years^-1.
    :return: significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where linear trend is significant.
    """

    num_grid_rows = frequency_matrix.shape[1]
    num_grid_columns = frequency_matrix.shape[2]

    trend_matrix_year01 = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )
    significance_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_num_done = i * num_grid_columns + j

            if numpy.mod(this_num_done, 1000) == 0:
                print((
                    'Have run Mann-Kendall test for {0:d} of {1:d} grid '
                    'cells...'
                ).format(
                    this_num_done, num_grid_rows * num_grid_columns
                ))

            these_frequencies = frequency_matrix[:, i, j]
            if numpy.all(numpy.isnan(these_frequencies)):
                continue

            this_result_tuple = pymannkendall.original_test(
                x=these_frequencies, alpha=1. - confidence_level)

            trend_matrix_year01[i, j] = this_result_tuple.slope
            significance_matrix[i, j] = this_result_tuple.h

    print('Have run Mann-Kendall test for all {0:d} grid cells!'.format(
        num_grid_rows * num_grid_columns
    ))

    return trend_matrix_year01, significance_matrix


def _fill_nans_in_series(data_series):
    """Fills NaN's in data series (via linear interpolation).

    This method assumes that the data series is equally spaced.

    :param data_series: 1-D numpy array of values.
    :return: data_series: Same but without NaN's.
    """

    nan_flags = numpy.isnan(data_series)
    if not numpy.any(nan_flags):
        return data_series

    nan_indices = numpy.where(nan_flags)[0]
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    if len(real_indices) == 1:
        data_series[nan_indices] = data_series[real_indices[0]]
        return data_series

    num_times = len(data_series)
    time_indices = numpy.linspace(0, num_times - 1, num=num_times, dtype=float)

    interp_object = scipy_interp1d(
        time_indices[real_indices], data_series[real_indices],
        kind='linear', assume_sorted=True, bounds_error=False,
        fill_value='extrapolate'
    )

    data_series[nan_indices] = interp_object(time_indices[nan_indices])
    return data_series


def _mk_test_one_statistic(statistic_matrix, confidence_level):
    """Runs Mann-Kendall test for one statistic.

    :param statistic_matrix: See doc for any output variable in method
        `_read_statistics`.
    :param confidence_level: See documentation at top of file.
    :return: trend_matrix_year01: See doc for `_mk_test_frequency`.
    :return: significance_matrix: Same.
    """

    num_grid_rows = statistic_matrix.shape[1]
    num_grid_columns = statistic_matrix.shape[2]

    trend_matrix_year01 = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )
    significance_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_num_done = i * num_grid_columns + j

            if numpy.mod(this_num_done, 1000) == 0:
                print((
                    'Have run Mann-Kendall test for {0:d} of {1:d} grid '
                    'cells...'
                ).format(
                    this_num_done, num_grid_rows * num_grid_columns
                ))

            these_values = statistic_matrix[:, i, j]
            if numpy.all(numpy.isnan(these_values)):
                continue

            these_values = _fill_nans_in_series(these_values)
            this_result_tuple = pymannkendall.original_test(
                x=these_values, alpha=1. - confidence_level)

            trend_matrix_year01[i, j] = this_result_tuple.slope
            significance_matrix[i, j] = this_result_tuple.h

    print('Have run Mann-Kendall test for all {0:d} grid cells!'.format(
        num_grid_rows * num_grid_columns
    ))

    return trend_matrix_year01, significance_matrix


def _run(input_dir_name, file_type_string, first_year, last_year, season_string,
         confidence_level, output_dir_name):
    """Runs Mann-Kendall test for gridded front frequencies or statistics.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param file_type_string: Same.
    :param first_year: Same.
    :param last_year: Same.
    :param season_string: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    input_file_name_matrix = _find_input_files(
        input_dir_name=input_dir_name, file_type_string=file_type_string,
        first_year=first_year, last_year=last_year, season_string=season_string)

    input_file_name_list = numpy.ravel(input_file_name_matrix).tolist()

    if file_type_string == climo_utils.FRONT_COUNTS_STRING:
        front_count_dict = _read_frequencies(input_file_name_matrix)
        print(SEPARATOR_STRING)

        wf_trend_matrix_year01, wf_significance_matrix = _mk_test_frequency(
            frequency_matrix=front_count_dict[WF_FREQUENCY_KEY],
            confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_mann_kendall_file(
            directory_name=output_dir_name,
            property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_mann_kendall_test(
            netcdf_file_name=this_output_file_name,
            trend_matrix_year01=wf_trend_matrix_year01,
            significance_matrix=wf_significance_matrix,
            num_labels_matrix=front_count_dict[climo_utils.NUM_WF_LABELS_KEY],
            property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
            input_file_names=input_file_name_list,
            confidence_level=confidence_level)

        cf_trend_matrix_year01, cf_significance_matrix = _mk_test_frequency(
            frequency_matrix=front_count_dict[CF_FREQUENCY_KEY],
            confidence_level=confidence_level)
        print(SEPARATOR_STRING)

        this_output_file_name = climo_utils.find_mann_kendall_file(
            directory_name=output_dir_name,
            property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
            raise_error_if_missing=False)

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        climo_utils.write_mann_kendall_test(
            netcdf_file_name=this_output_file_name,
            trend_matrix_year01=cf_trend_matrix_year01,
            significance_matrix=cf_significance_matrix,
            num_labels_matrix=front_count_dict[climo_utils.NUM_CF_LABELS_KEY],
            property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
            input_file_names=input_file_name_list,
            confidence_level=confidence_level)

        return

    front_statistic_dict = _read_statistics(input_file_name_matrix)
    print(SEPARATOR_STRING)

    wf_length_matrix_m_y01, wf_length_sig_matrix = _mk_test_one_statistic(
        statistic_matrix=front_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY],
        confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_mann_kendall_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_mann_kendall_test(
        netcdf_file_name=this_output_file_name,
        trend_matrix_year01=wf_length_matrix_m_y01,
        significance_matrix=wf_length_sig_matrix,
        num_labels_matrix=front_statistic_dict[climo_utils.NUM_WF_LABELS_KEY],
        property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
        input_file_names=input_file_name_list,
        confidence_level=confidence_level)

    wf_area_matrix_m2_y01, wf_area_sig_matrix = _mk_test_one_statistic(
        statistic_matrix=front_statistic_dict[climo_utils.MEAN_WF_AREAS_KEY],
        confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_mann_kendall_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WF_AREA_PROPERTY_NAME,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_mann_kendall_test(
        netcdf_file_name=this_output_file_name,
        trend_matrix_year01=wf_area_matrix_m2_y01,
        significance_matrix=wf_area_sig_matrix,
        num_labels_matrix=front_statistic_dict[climo_utils.NUM_WF_LABELS_KEY],
        property_name=climo_utils.WF_AREA_PROPERTY_NAME,
        input_file_names=input_file_name_list,
        confidence_level=confidence_level)

    cf_length_matrix_m_y01, cf_length_sig_matrix = _mk_test_one_statistic(
        statistic_matrix=front_statistic_dict[climo_utils.MEAN_CF_LENGTHS_KEY],
        confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_mann_kendall_file(
        directory_name=output_dir_name,
        property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_mann_kendall_test(
        netcdf_file_name=this_output_file_name,
        trend_matrix_year01=cf_length_matrix_m_y01,
        significance_matrix=cf_length_sig_matrix,
        num_labels_matrix=front_statistic_dict[climo_utils.NUM_CF_LABELS_KEY],
        property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
        input_file_names=input_file_name_list,
        confidence_level=confidence_level)

    cf_area_matrix_m2_y01, cf_area_sig_matrix = _mk_test_one_statistic(
        statistic_matrix=front_statistic_dict[climo_utils.MEAN_CF_AREAS_KEY],
        confidence_level=confidence_level)
    print(SEPARATOR_STRING)

    this_output_file_name = climo_utils.find_mann_kendall_file(
        directory_name=output_dir_name,
        property_name=climo_utils.CF_AREA_PROPERTY_NAME,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(this_output_file_name))

    climo_utils.write_mann_kendall_test(
        netcdf_file_name=this_output_file_name,
        trend_matrix_year01=cf_area_matrix_m2_y01,
        significance_matrix=cf_area_sig_matrix,
        num_labels_matrix=front_statistic_dict[climo_utils.NUM_CF_LABELS_KEY],
        property_name=climo_utils.CF_AREA_PROPERTY_NAME,
        input_file_names=input_file_name_list,
        confidence_level=confidence_level)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        file_type_string=getattr(INPUT_ARG_OBJECT, FILE_TYPE_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        season_string=getattr(INPUT_ARG_OBJECT, SEASON_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
