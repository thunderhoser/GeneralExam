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

INPUT_DIR_ARG_NAME = 'input_dir_name'
FILE_TYPE_ARG_NAME = 'file_type_string'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
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
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_frequencies(input_file_names):
    """Reads front frequencies into two numpy arrays.

    T = number of years
    M = number of rows in grid
    N = number of columns in grid

    :param input_file_names: length-T list of paths to input files (will be read
        by `climatology_utils.read_gridded_counts`).
    :return: wf_frequency_matrix: T-by-M-by-N numpy array.
        wf_frequency_matrix[t, i, j] is the frequency of warm fronts (fraction
        of time steps with warm fronts) at grid cell [i, j] in the [t]th year.
    :return: cf_frequency_matrix: Same but for cold fronts.
    """

    num_years = len(input_file_names)
    wf_frequency_matrix = None
    cf_frequency_matrix = None

    for i in range(num_years):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_count_dict = climo_utils.read_gridded_counts(input_file_names[i])

        this_num_time_steps = 1 + int(numpy.round(
            float(
                this_count_dict[climo_utils.LAST_TIME_KEY] -
                this_count_dict[climo_utils.FIRST_TIME_KEY]
            ) / TIME_INTERVAL_SEC
        ))

        this_wf_frequency_matrix = (
            this_count_dict[climo_utils.NUM_WF_LABELS_KEY] / this_num_time_steps
        )
        this_cf_frequency_matrix = (
            this_count_dict[climo_utils.NUM_CF_LABELS_KEY] / this_num_time_steps
        )

        if wf_frequency_matrix is None:
            wf_frequency_matrix = numpy.full(
                (num_years,) + this_wf_frequency_matrix.shape, numpy.nan
            )
            cf_frequency_matrix = numpy.full(
                (num_years,) + this_wf_frequency_matrix.shape, numpy.nan
            )

        wf_frequency_matrix[i, ...] = this_wf_frequency_matrix
        cf_frequency_matrix[i, ...] = this_cf_frequency_matrix

    return wf_frequency_matrix, cf_frequency_matrix


def _read_statistics(input_file_names):
    """Reads front statistics into several numpy arrays.

    T = number of years
    M = number of rows in grid
    N = number of columns in grid

    :param input_file_names: length-T list of paths to input files (will be read
        by `climatology_utils.read_gridded_stats`).
    :return: front_statistic_dict: Dictionary with the following keys.
    front_statistic_dict["mean_wf_length_matrix_metres"]: T-by-M-by-N numpy
        array.  wf_length_matrix_metres[t, i, j] is the average length of warm
        fronts at grid cell [i, j] in the [t]th year.
    front_statistic_dict["mean_cf_length_matrix_metres"]: Same but for cold-
        front length.
    front_statistic_dict["mean_wf_area_matrix_m2"]: Same but for warm-front
        area.
    front_statistic_dict["mean_cf_area_matrix_m2"]: Same but for cold-front
        area.
    """

    num_years = len(input_file_names)
    mean_wf_length_matrix_metres = None
    mean_cf_length_matrix_metres = None
    mean_wf_area_matrix_m2 = None
    mean_cf_area_matrix_m2 = None

    for i in range(num_years):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_statistic_dict = climo_utils.read_gridded_stats(
            input_file_names[i]
        )

        if mean_wf_length_matrix_metres is None:
            dimensions = (
                (num_years,) +
                this_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY].shape
            )

            mean_wf_length_matrix_metres = numpy.full(dimensions, numpy.nan)
            mean_cf_length_matrix_metres = numpy.full(dimensions, numpy.nan)
            mean_wf_area_matrix_m2 = numpy.full(dimensions, numpy.nan)
            mean_cf_area_matrix_m2 = numpy.full(dimensions, numpy.nan)

        mean_wf_length_matrix_metres[i, ...] = this_statistic_dict[
            climo_utils.MEAN_WF_LENGTHS_KEY]
        mean_cf_length_matrix_metres[i, ...] = this_statistic_dict[
            climo_utils.MEAN_CF_LENGTHS_KEY]
        mean_wf_area_matrix_m2[i, ...] = this_statistic_dict[
            climo_utils.MEAN_WF_AREAS_KEY]
        mean_cf_area_matrix_m2[i, ...] = this_statistic_dict[
            climo_utils.MEAN_CF_AREAS_KEY]

    return {
        climo_utils.MEAN_WF_LENGTHS_KEY: mean_wf_length_matrix_metres,
        climo_utils.MEAN_CF_LENGTHS_KEY: mean_cf_length_matrix_metres,
        climo_utils.MEAN_WF_AREAS_KEY: mean_wf_area_matrix_m2,
        climo_utils.MEAN_CF_AREAS_KEY: mean_cf_area_matrix_m2
    }


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

    num_times = len(data_series)
    time_indices = numpy.linspace(0, num_times - 1, num=num_times, dtype=float)

    nan_indices = numpy.where(nan_flags)[0]
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

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


def _run(input_dir_name, file_type_string, first_year, last_year,
         confidence_level, output_dir_name):
    """Runs Mann-Kendall test for gridded front frequencies or statistics.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param file_type_string: Same.
    :param first_year: Same.
    :param last_year: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_geq(last_year, first_year + 9)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    num_years = last_year - first_year + 1
    years = numpy.linspace(first_year, last_year, num=num_years, dtype=int)

    input_file_names = [None] * num_years

    for i in range(num_years):
        this_first_time_unix_sec, this_last_time_unix_sec = (
            time_conversion.first_and_last_times_in_year(years[i])
        )

        this_last_time_unix_sec += 1 - TIME_INTERVAL_SEC

        input_file_names[i] = climo_utils.find_aggregated_file(
            directory_name=input_dir_name, file_type_string=file_type_string,
            first_time_unix_sec=this_first_time_unix_sec,
            last_time_unix_sec=this_last_time_unix_sec, hours=None, months=None,
            raise_error_if_missing=True)

    if file_type_string == climo_utils.FRONT_COUNTS_STRING:
        wf_frequency_matrix, cf_frequency_matrix = _read_frequencies(
            input_file_names)
        print(SEPARATOR_STRING)

        wf_trend_matrix_year01, wf_significance_matrix = _mk_test_frequency(
            frequency_matrix=wf_frequency_matrix,
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
            property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
            input_file_names=input_file_names,
            confidence_level=confidence_level)

        cf_trend_matrix_year01, cf_significance_matrix = _mk_test_frequency(
            frequency_matrix=cf_frequency_matrix,
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
            property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
            input_file_names=input_file_names,
            confidence_level=confidence_level)

        return

    front_statistic_dict = _read_statistics(input_file_names)
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
        property_name=climo_utils.WF_LENGTH_PROPERTY_NAME,
        input_file_names=input_file_names,
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
        property_name=climo_utils.WF_AREA_PROPERTY_NAME,
        input_file_names=input_file_names,
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
        property_name=climo_utils.CF_LENGTH_PROPERTY_NAME,
        input_file_names=input_file_names,
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
        property_name=climo_utils.CF_AREA_PROPERTY_NAME,
        input_file_names=input_file_names,
        confidence_level=confidence_level)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        file_type_string=getattr(INPUT_ARG_OBJECT, FILE_TYPE_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
