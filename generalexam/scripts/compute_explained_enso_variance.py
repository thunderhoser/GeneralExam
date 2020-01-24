"""Computes percent variance in CF and WF frequency explained by ENSO phase."""

import argparse
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_utils import climatology_utils as climo_utils

YEAR_MONTH_FORMAT = '%Y%m'
TIME_INTERVAL_SEC = 10800
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WF_FREQ_MATRIX_KEY = 'wf_frequency_matrix'
CF_FREQ_MATRIX_KEY = 'cf_frequency_matrix'

ENSO_YEAR_COLUMN = 0
ENSO_MONTH_COLUMN = 1
NINO_3POINT4_COLUMN = 9

COUNT_DIR_ARG_NAME = 'input_count_dir_name'
ENSO_FILE_ARG_NAME = 'input_enso_file_name'
FIRST_MONTH_ARG_NAME = 'first_month_string'
LAST_MONTH_ARG_NAME = 'last_month_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

COUNT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (one per month) will be found by '
    '`climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_counts`.'
)
ENSO_FILE_HELP_STRING = (
    'Path to input file, containing monthly time series of ENSO indices '
    '(including the Nino 3.4 index, which will be used here).  The file should '
    'come from here: '
    'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii'
)
MONTH_HELP_STRING = (
    'Month (format "yyyymm").  Will compute explained variances at each grid '
    'point over the period `{0:s}`...`{1:s}`.'
).format(FIRST_MONTH_ARG_NAME, LAST_MONTH_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`climatology_utils.write_explained_enso_variance`, to exact locations '
    'determined by `climatology_utils.find_explained_variance_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + COUNT_DIR_ARG_NAME, type=str, required=True,
    help=COUNT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ENSO_FILE_ARG_NAME, type=str, required=True,
    help=ENSO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_MONTH_ARG_NAME, type=str, required=True, help=MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_MONTH_ARG_NAME, type=str, required=True, help=MONTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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


def _read_frequencies_one_composite(count_file_names):
    """Reads gridded front frequencies for one composite.

    T = number of months
    M = number of rows in subgrid
    N = number of columns in subgrid

    :param count_file_names: 1-D list of paths to input files.
    :return: front_count_dict: Dictionary with the following keys.
    front_count_dict["wf_frequency_matrix"]: T-by-M-by-N numpy array with
        monthly warm-front frequencies.
    front_count_dict["cf_frequency_matrix"]: Same but for cold fronts.
    """

    wf_frequency_matrix = None
    cf_frequency_matrix = None
    num_months = len(count_file_names)

    for i in range(num_months):
        print('Reading front counts from: "{0:s}"...'.format(
            count_file_names[i]
        ))

        this_count_dict = climo_utils.read_gridded_counts(count_file_names[i])

        this_num_times = 1 + int(numpy.round(
            float(
                this_count_dict[climo_utils.LAST_TIME_KEY] -
                this_count_dict[climo_utils.FIRST_TIME_KEY]
            )
            / TIME_INTERVAL_SEC
        ))

        print('Number of time steps in month = {0:d}'.format(this_num_times))

        this_wf_frequency_matrix = (
            this_count_dict[climo_utils.NUM_WF_LABELS_KEY] / this_num_times
        )
        this_cf_frequency_matrix = (
            this_count_dict[climo_utils.NUM_CF_LABELS_KEY] / this_num_times
        )

        if wf_frequency_matrix is None:
            num_grid_rows = this_wf_frequency_matrix.shape[0]
            num_grid_columns = this_wf_frequency_matrix.shape[1]
            dimensions = (num_months, num_grid_rows, num_grid_columns)

            wf_frequency_matrix = numpy.full(dimensions, numpy.nan)
            cf_frequency_matrix = numpy.full(dimensions, numpy.nan)

        wf_frequency_matrix[i, ...] = this_wf_frequency_matrix
        cf_frequency_matrix[i, ...] = this_cf_frequency_matrix

    return {
        WF_FREQ_MATRIX_KEY: wf_frequency_matrix,
        CF_FREQ_MATRIX_KEY: cf_frequency_matrix
    }


def _run(count_dir_name, enso_file_name, first_month_string, last_month_string,
         output_dir_name):
    """Computes percent variance in CF and WF frequency explained by ENSO phase.

    This is effectively the main method.

    :param count_dir_name: See documentation at top of file.
    :param enso_file_name: Same.
    :param first_month_string: Same.
    :param last_month_string: Same.
    :param output_dir_name: Same.
    """

    first_year = int(first_month_string[:4])
    first_month = int(first_month_string[4:])
    last_year = int(last_month_string[:4])
    last_month = int(last_month_string[4:])

    print('Reading Nino 3.4 indices from: "{0:s}"...'.format(enso_file_name))
    enso_table = pandas.read_csv(
        enso_file_name, skiprows=[0], header=None, delim_whitespace=True
    )

    enso_month_strings = [
        '{0:04d}{1:02d}'.format(y, m) for y, m in zip(
            enso_table[ENSO_YEAR_COLUMN].values,
            enso_table[ENSO_MONTH_COLUMN].values
        )
    ]

    month_strings = []
    nino_3point4_indices = []

    for y in range(first_year, last_year + 1):
        if y == first_year:
            this_first_month = first_month
        else:
            this_first_month = 1

        if y == last_year:
            this_last_month = last_month
        else:
            this_last_month = 12

        for m in range(this_first_month, this_last_month + 1):
            month_strings.append('{0:04d}{1:02d}'.format(y, m))

            this_row = numpy.where(
                numpy.array(enso_month_strings) == month_strings[-1]
            )[0][0]

            nino_3point4_indices.append(
                enso_table[NINO_3POINT4_COLUMN].values[this_row]
            )

    nino_3point4_indices = numpy.array(nino_3point4_indices)
    start_times_unix_sec, end_times_unix_sec = (
        _months_to_start_end_times(month_strings)
    )

    count_file_names = [
        climo_utils.find_aggregated_file(
            directory_name=count_dir_name,
            file_type_string=climo_utils.FRONT_COUNTS_STRING,
            first_time_unix_sec=f, last_time_unix_sec=l
        )
        for f, l in zip(start_times_unix_sec, end_times_unix_sec)
    ]

    count_dict = _read_frequencies_one_composite(count_file_names)
    wf_frequency_matrix = count_dict[WF_FREQ_MATRIX_KEY]
    cf_frequency_matrix = count_dict[CF_FREQ_MATRIX_KEY]
    print(SEPARATOR_STRING)

    num_grid_rows = wf_frequency_matrix.shape[1]
    num_grid_columns = wf_frequency_matrix.shape[2]
    dimensions = (num_grid_rows, num_grid_columns)
    wf_explained_variance_matrix = numpy.full(dimensions, numpy.nan)
    cf_explained_variance_matrix = numpy.full(dimensions, numpy.nan)

    nino_3point4_variance = numpy.var(nino_3point4_indices, ddof=1)

    for i in range(num_grid_rows):
        print((
            'Computing explained variances for {0:d}th of {1:d} rows in grid...'
        ).format(
            i + 1, num_grid_rows
        ))

        for j in range(num_grid_columns):
            this_covar_matrix = numpy.cov(
                nino_3point4_indices, wf_frequency_matrix[:, i, j],
                bias=False, ddof=1
            )
            wf_explained_variance_matrix[i, j] = (
                this_covar_matrix[0, 1] ** 2 /
                (this_covar_matrix[0, 0] * this_covar_matrix[1, 1])
            )

            this_covar_matrix = numpy.cov(
                nino_3point4_indices, cf_frequency_matrix[:, i, j],
                bias=False, ddof=1
            )
            cf_explained_variance_matrix[i, j] = (
                this_covar_matrix[0, 1] ** 2 /
                (this_covar_matrix[0, 0] * this_covar_matrix[1, 1])
            )

    print(SEPARATOR_STRING)

    wf_output_file_name = climo_utils.find_explained_variance_file(
        directory_name=output_dir_name,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        enso_flag=True, raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(wf_output_file_name))
    climo_utils.write_explained_variances(
        netcdf_file_name=wf_output_file_name,
        explained_variance_matrix=wf_explained_variance_matrix,
        property_name=climo_utils.WF_FREQ_PROPERTY_NAME,
        first_month_string=first_month_string,
        last_month_string=last_month_string, enso_flag=True
    )

    cf_output_file_name = climo_utils.find_explained_variance_file(
        directory_name=output_dir_name,
        property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
        enso_flag=True, raise_error_if_missing=False
    )

    print('Writing results to: "{0:s}"...'.format(cf_output_file_name))
    climo_utils.write_explained_variances(
        netcdf_file_name=cf_output_file_name,
        explained_variance_matrix=cf_explained_variance_matrix,
        property_name=climo_utils.CF_FREQ_PROPERTY_NAME,
        first_month_string=first_month_string,
        last_month_string=last_month_string, enso_flag=True
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_dir_name=getattr(INPUT_ARG_OBJECT, COUNT_DIR_ARG_NAME),
        enso_file_name=getattr(INPUT_ARG_OBJECT, ENSO_FILE_ARG_NAME),
        first_month_string=getattr(INPUT_ARG_OBJECT, FIRST_MONTH_ARG_NAME),
        last_month_string=getattr(INPUT_ARG_OBJECT, LAST_MONTH_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
