"""Sums gridded front counts over many files."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import climatology_utils as climo_utils

INPUT_TIME_FORMAT = '%Y%m%d%H'

INPUT_DIR_ARG_NAME = 'input_count_dir_name'
FIRST_TIMES_ARG_NAME = 'first_time_strings'
LAST_TIMES_ARG_NAME = 'last_time_strings'
HOURS_ARG_NAME = 'hours'
MONTHS_ARG_NAME = 'months'
OUTPUT_DIR_ARG_NAME = 'output_count_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input files.  Files therein will be found'
    ' by `climatology_utils.find_aggregated_file` and read by '
    '`climatology_utils.read_gridded_counts`.'
)
FIRST_TIMES_HELP_STRING = (
    'First times (one per file).  This should be a space-separated list, with '
    'each time in format "yyyymmddHH".'
)
LAST_TIMES_HELP_STRING = (
    'Last times (one per file).  This should be a space-separated list, with '
    'each time in format "yyyymmddHH".'
)
HOURS_HELP_STRING = (
    'Hours in files (space-separated list of integers).  To use all hours, '
    'leave this alone.'
)
MONTHS_HELP_STRING = (
    'Months in files (space-separated list of integers).  To use all hours, '
    'leave this alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  The sum over all input files will be '
    'written here by `climatology_utils.write_gridded_counts`, to an exact '
    'location determined by `climatology_utils.find_aggregated_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIMES_ARG_NAME, type=str, nargs='+', required=True,
    help=FIRST_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIMES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAST_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HOURS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=HOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MONTHS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=MONTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, first_time_strings, last_time_strings, hours,
         months, top_output_dir_name):
    """Sums gridded front counts over many files.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_time_strings: Same.
    :param last_time_strings: Same.
    :param hours: Same.
    :param months: Same.
    :param top_output_dir_name: Same.
    """

    num_files = len(first_time_strings)
    expected_dim = numpy.array([num_files], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(last_time_strings), exact_dimensions=expected_dim
    )

    first_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, INPUT_TIME_FORMAT)
        for t in first_time_strings
    ], dtype=int)

    last_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, INPUT_TIME_FORMAT)
        for t in last_time_strings
    ], dtype=int)

    if len(hours) == 1 and hours[0] < 0:
        hours = None
    if len(months) == 1 and months[0] < 1:
        months = None

    num_wf_labels_matrix = []
    num_unique_wf_matrix = []
    num_cf_labels_matrix = []
    num_unique_cf_matrix = []
    prediction_file_names = []

    for i in range(num_files):
        this_file_name = climo_utils.find_aggregated_file(
            directory_name=top_input_dir_name,
            file_type_string=climo_utils.FRONT_COUNTS_STRING,
            first_time_unix_sec=first_times_unix_sec[i],
            last_time_unix_sec=last_times_unix_sec[i],
            hours=hours, months=months
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_count_dict = climo_utils.read_gridded_counts(this_file_name)

        if num_wf_labels_matrix is None:
            num_wf_labels_matrix = (
                this_count_dict[climo_utils.NUM_WF_LABELS_KEY] + 0.
            )
            num_unique_wf_matrix = (
                this_count_dict[climo_utils.NUM_UNIQUE_WF_KEY] + 0.
            )
            num_cf_labels_matrix = (
                this_count_dict[climo_utils.NUM_CF_LABELS_KEY] + 0.
            )
            num_unique_cf_matrix = (
                this_count_dict[climo_utils.NUM_UNIQUE_CF_KEY] + 0.
            )
        else:
            num_wf_labels_matrix += (
                this_count_dict[climo_utils.NUM_WF_LABELS_KEY]
            )
            num_unique_wf_matrix += (
                this_count_dict[climo_utils.NUM_UNIQUE_WF_KEY]
            )
            num_cf_labels_matrix += (
                this_count_dict[climo_utils.NUM_CF_LABELS_KEY]
            )
            num_unique_cf_matrix += (
                this_count_dict[climo_utils.NUM_UNIQUE_CF_KEY]
            )

        prediction_file_names += this_count_dict[
            climo_utils.PREDICTION_FILES_KEY
        ]

    output_file_name = climo_utils.find_aggregated_file(
        directory_name=top_output_dir_name,
        file_type_string=climo_utils.FRONT_COUNTS_STRING,
        first_time_unix_sec=numpy.min(first_times_unix_sec),
        last_time_unix_sec=numpy.max(last_times_unix_sec),
        hours=hours, months=months
    )

    print('Writing summed counts to: "{0:s}"...'.format(output_file_name))

    climo_utils.write_gridded_counts(
        netcdf_file_name=output_file_name,
        num_wf_labels_matrix=num_wf_labels_matrix,
        num_unique_wf_matrix=num_unique_wf_matrix,
        num_cf_labels_matrix=num_cf_labels_matrix,
        num_unique_cf_matrix=num_unique_cf_matrix,
        first_time_unix_sec=numpy.min(first_times_unix_sec),
        last_time_unix_sec=numpy.max(last_times_unix_sec),
        prediction_file_names=prediction_file_names, hours=hours, months=months
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_strings=getattr(INPUT_ARG_OBJECT, FIRST_TIMES_ARG_NAME),
        last_time_strings=getattr(INPUT_ARG_OBJECT, LAST_TIMES_ARG_NAME),
        hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, HOURS_ARG_NAME), dtype=int
        ),
        months=numpy.array(
            getattr(INPUT_ARG_OBJECT, MONTHS_ARG_NAME), dtype=int
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
