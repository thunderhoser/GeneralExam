"""Finds time periods for ENSO-based composites."""

import argparse
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FIRST_YEAR = 1979
LAST_YEAR = 2018
TIME_INTERVAL_SEC = 10800

YEAR_COLUMN = 0
MONTH_COLUMN = 1
NINO_3POINT4_COLUMN = 9

ENSO_FILE_ARG_NAME = 'input_enso_file_name'
BASELINE_THRES_ARG_NAME = 'baseline_threshold'
TRIAL_THRES_ARG_NAME = 'trial_threshold'

ENSO_FILE_HELP_STRING = (
    'Path to input file, containing monthly time series of ENSO indices '
    '(including the Nino 3.4 index, which will be used here).  The file should '
    'come from here: '
    'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii')

BASELINE_THRES_HELP_STRING = (
    'Absolute threshold for baseline period.  Months with '
    'abs(nino_3point4_index) < `{0:s}` will be in the baseline period.'
).format(BASELINE_THRES_ARG_NAME)

TRIAL_THRES_HELP_STRING = (
    'Threshold for trial period.  If this is positive, months with '
    'nino_3point4_index >= `{0:s}` will be in the trial period.  If negative, '
    'months with nino_3point4_index <= `{0:s}` will be in the trial period.'
).format(TRIAL_THRES_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ENSO_FILE_ARG_NAME, type=str, required=True,
    help=ENSO_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_THRES_ARG_NAME, type=float, required=True,
    help=BASELINE_THRES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_THRES_ARG_NAME, type=float, required=True,
    help=TRIAL_THRES_HELP_STRING)


def _months_to_start_end_times(years, months):
    """Converts list of months to list of start and end times.

    N = number of months in set

    :param years: length-N numpy array of years.
    :param months: length-N numpy array of months.
    :return: start_time_strings: length-N list of start times (format
        "yyyymmddHH").
    :return: end_time_strings: length-N list of end times (format "yyyymmddHH").
    """

    num_months = len(months)
    start_times_unix_sec = numpy.full(num_months, -1, dtype=int)
    end_times_unix_sec = numpy.full(num_months, -1, dtype=int)

    for i in range(num_months):
        this_month_unix_sec = time_conversion.string_to_unix_sec(
            '{0:04d}{1:02d}'.format(years[i], months[i]), '%Y%m'
        )

        start_times_unix_sec[i], end_times_unix_sec[i] = (
            time_conversion.first_and_last_times_in_month(this_month_unix_sec)
        )

    end_times_unix_sec += 1 - TIME_INTERVAL_SEC

    start_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in start_times_unix_sec
    ]

    end_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in end_times_unix_sec
    ]

    return start_time_strings, end_time_strings


def _run(enso_file_name, baseline_threshold, trial_threshold):
    """Finds time periods for ENSO-based composites.

    This is effectively the main method.

    :param enso_file_name: See documentation at top of file.
    :param baseline_threshold: Same.
    :param trial_threshold: Same.
    """

    error_checking.assert_is_greater(baseline_threshold, 0.)
    error_checking.assert_is_geq(
        numpy.absolute(trial_threshold), baseline_threshold
    )

    print('Reading Nino 3.4 indices from: "{0:s}"...'.format(enso_file_name))
    enso_table = pandas.read_csv(
        enso_file_name, skiprows=[0], header=None, delim_whitespace=True
    )

    enso_table = enso_table.loc[
        (enso_table[YEAR_COLUMN] >= FIRST_YEAR) &
        (enso_table[YEAR_COLUMN] <= LAST_YEAR)
    ]

    years = enso_table[YEAR_COLUMN].values
    months = enso_table[MONTH_COLUMN].values
    nino_3point4_indices = enso_table[NINO_3POINT4_COLUMN].values

    baseline_indices = numpy.where(
        numpy.absolute(nino_3point4_indices) < baseline_threshold
    )[0]

    print((
        '{0:d} of {1:d} months are in baseline period (absolute '
        'nino_3point4_index < {2:f}).'
    ).format(
        len(baseline_indices), len(years), baseline_threshold
    ))

    if trial_threshold > 0:
        trial_indices = numpy.where(nino_3point4_indices >= trial_threshold)[0]
    else:
        trial_indices = numpy.where(nino_3point4_indices <= trial_threshold)[0]

    print((
        '{0:d} of {1:d} months are in trial period (nino_3point4_index '
        '{2:s} {3:f}).'
    ).format(
        len(trial_indices), len(years), '>=' if trial_threshold > 0 else '<=',
        trial_threshold
    ))

    baseline_start_time_strings, baseline_end_time_strings = (
        _months_to_start_end_times(
            years=years[baseline_indices], months=months[baseline_indices]
        )
    )

    trial_start_time_strings, trial_end_time_strings = (
        _months_to_start_end_times(
            years=years[trial_indices], months=months[trial_indices]
        )
    )

    print(SEPARATOR_STRING)
    print('Baseline start times:\n')
    print(' '.join(baseline_start_time_strings))

    print(SEPARATOR_STRING)
    print('Baseline end times:\n')
    print(' '.join(baseline_end_time_strings))

    print(SEPARATOR_STRING)
    print('Trial start times:\n')
    print(' '.join(trial_start_time_strings))

    print(SEPARATOR_STRING)
    print('Trial end times:\n')
    print(' '.join(trial_end_time_strings))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        enso_file_name=getattr(INPUT_ARG_OBJECT, ENSO_FILE_ARG_NAME),
        baseline_threshold=getattr(INPUT_ARG_OBJECT, BASELINE_THRES_ARG_NAME),
        trial_threshold=getattr(INPUT_ARG_OBJECT, TRIAL_THRES_ARG_NAME)
    )
