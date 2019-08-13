"""Counts WF and CF labels at each grid cell over a given time period."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_WF_LABELS_KEY = 'num_wf_labels_matrix'
NUM_CF_LABELS_KEY = 'num_cf_labels_matrix'
NUM_UNIQUE_WF_KEY = 'num_unique_wf_matrix'
NUM_UNIQUE_CF_KEY = 'num_unique_cf_matrix'
SECOND_UNIQUE_LABELS_KEY = 'second_unique_label_matrix'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
HOURS_ARG_NAME = 'hours_to_keep'
MONTHS_ARG_NAME = 'months_to_keep'
SEPARATION_TIME_ARG_NAME = 'separation_time_sec'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file` with '
    '`read_deterministic == True`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will count WF and CF labels at '
    'each grid cell for the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

HOURS_HELP_STRING = (
    'List of UTC hours (integers in 0...23).  This script will count WF and CF '
    'labels only at the given hours.  If you want do not want to filter by '
    'hour, leave this argument alone.')

MONTHS_HELP_STRING = (
    'List of months (integers in 1...12).  This script will count WF and CF '
    'labels only for the given months.  If you want do not want to filter by '
    'month, leave this argument alone.')

SEPARATION_TIME_HELP_STRING = (
    'Separation time (used to remove redundant front labels).  If grid cell '
    '[i, j] has multiple front labels of the same type within `{0:s}` seconds, '
    'only one such label will count.'
).format(SEPARATION_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  File will be written by '
    '`climatology_utils.write_gridded_counts`, to a location therein determined'
    ' by `climatology_utils.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HOURS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=HOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MONTHS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=MONTHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEPARATION_TIME_ARG_NAME, type=int, required=False, default=86400,
    help=SEPARATION_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _update_counts(
        first_label_matrix, first_unique_label_matrix, first_times_unix_sec,
        second_label_matrix, second_times_unix_sec, separation_time_sec,
        count_second_period):
    """Updates WF and CF count at each grid cell.

    F = number of time steps in first period
    S = number of time steps in second period
    M = number of rows in grid
    N = number of columns in grid

    :param first_label_matrix: F-by-M-by-N numpy array with integer front
        labels.
    :param first_unique_label_matrix: Same but after applying separation time.
    :param first_times_unix_sec: length-F numpy array of valid times.
    :param second_label_matrix: S-by-M-by-N numpy array with integer front
        labels.  This may be None.
    :param second_times_unix_sec: length-S numpy array of valid times.  This may
        be None.
    :param separation_time_sec: See documentation at top of file.
    :param count_second_period: Boolean flag.  If True, will count number of
        fronts in both periods.  If False, will count only in first period.
    :return: count_dict: Dictionary with the following keys.
    count_dict["num_wf_labels_matrix"]: M-by-N numpy array with number of
        warm-front labels (before applying separation time).
    count_dict["num_cf_labels_matrix"]: Same but for cold fronts.
    count_dict["num_unique_wf_matrix"]: M-by-N numpy array with number of unique
        warm fronts (after applying separation time).
    count_dict["num_unique_cf_matrix"]: Same but for cold fronts.
    count_dict["second_unique_label_matrix"]: Same as input
        `second_label_matrix` but after applying separation time.
    """

    have_second_period = second_label_matrix is not None

    if have_second_period and count_second_period:
        this_label_matrix = numpy.concatenate(
            (first_label_matrix, second_label_matrix), axis=0
        )
    else:
        this_label_matrix = first_label_matrix + 0

    num_wf_labels_matrix = numpy.sum(
        this_label_matrix == front_utils.WARM_FRONT_ENUM, axis=0)
    num_cf_labels_matrix = numpy.sum(
        this_label_matrix == front_utils.COLD_FRONT_ENUM, axis=0)

    if have_second_period:
        unique_label_matrix = numpy.concatenate(
            (first_unique_label_matrix, second_label_matrix), axis=0
        )
        valid_times_unix_sec = numpy.concatenate((
            first_times_unix_sec, second_times_unix_sec))
    else:
        unique_label_matrix = first_unique_label_matrix + 0
        valid_times_unix_sec = first_times_unix_sec + 0

    num_grid_rows = unique_label_matrix.shape[1]
    num_grid_columns = unique_label_matrix.shape[2]

    this_num_fronts_matrix = numpy.sum(
        unique_label_matrix > front_utils.NO_FRONT_ENUM, axis=0)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if this_num_fronts_matrix[i, j] == 0:
                continue

            unique_label_matrix[:, i, j] = climo_utils.apply_separation_time(
                front_type_enums=unique_label_matrix[:, i, j],
                valid_times_unix_sec=valid_times_unix_sec,
                separation_time_sec=separation_time_sec
            )[0]

    first_num_times = len(first_times_unix_sec)
    if count_second_period:
        this_label_matrix = unique_label_matrix + 0
    else:
        this_label_matrix = unique_label_matrix[:first_num_times, ...]

    num_unique_wf_matrix = numpy.sum(
        this_label_matrix == front_utils.WARM_FRONT_ENUM, axis=0)
    num_unique_cf_matrix = numpy.sum(
        this_label_matrix == front_utils.COLD_FRONT_ENUM, axis=0)

    if have_second_period:
        second_unique_label_matrix = unique_label_matrix[first_num_times:, ...]
    else:
        second_unique_label_matrix = None

    num_front_labels = numpy.sum(num_wf_labels_matrix + num_cf_labels_matrix)
    num_unique_fronts = numpy.sum(num_unique_wf_matrix + num_unique_cf_matrix)

    print((
        'Number of front labels (unique fronts) increased by {0:d} ({1:d})!'
    ).format(
        num_front_labels, num_unique_fronts
    ))

    return {
        NUM_WF_LABELS_KEY: num_wf_labels_matrix,
        NUM_CF_LABELS_KEY: num_cf_labels_matrix,
        NUM_UNIQUE_WF_KEY: num_unique_wf_matrix,
        NUM_UNIQUE_CF_KEY: num_unique_cf_matrix,
        SECOND_UNIQUE_LABELS_KEY: second_unique_label_matrix
    }


def _run(prediction_dir_name, first_time_string, last_time_string,
         hours_to_keep, months_to_keep, separation_time_sec, output_dir_name):
    """Counts WF and CF labels at each grid cell over a given time period.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param hours_to_keep: Same.
    :param months_to_keep: Same.
    :param separation_time_sec: Same.
    :param output_dir_name: Same.
    """

    if len(hours_to_keep) == 1 and hours_to_keep[0] == -1:
        hours_to_keep = None

    if len(months_to_keep) == 1 and months_to_keep[0] == -1:
        months_to_keep = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT)

    prediction_file_names, valid_times_unix_sec = (
        prediction_io.find_files_for_climo(
            directory_name=prediction_dir_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec,
            hours_to_keep=hours_to_keep, months_to_keep=months_to_keep)
    )

    if separation_time_sec <= 0:
        num_times_per_block = 50
    else:
        smallest_time_step_sec = numpy.min(numpy.diff(valid_times_unix_sec))
        num_times_per_block = 5 * int(
            numpy.ceil(float(separation_time_sec) / smallest_time_step_sec)
        )

    first_label_matrix = None
    mask_matrix = None

    for k in range(num_times_per_block):
        print('Reading deterministic labels from: "{0:s}"...'.format(
            prediction_file_names[k]
        ))

        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=prediction_file_names[k], read_deterministic=True)

        if mask_matrix is None:
            mask_matrix = numpy.invert(numpy.isnan(
                this_prediction_dict[prediction_io.CLASS_PROBABILITIES_KEY][
                    0, ...]
            ))

        this_label_matrix = this_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]

        if first_label_matrix is None:
            first_label_matrix = this_label_matrix + 0
        else:
            first_label_matrix = numpy.concatenate(
                (first_label_matrix, this_label_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    first_num_times = first_label_matrix.shape[0]
    first_times_unix_sec = valid_times_unix_sec[:first_num_times]
    first_unique_label_matrix = first_label_matrix + 0

    num_times = len(prediction_file_names)
    num_grid_rows = first_label_matrix.shape[1]
    num_grid_columns = first_label_matrix.shape[2]

    this_num_fronts_matrix = numpy.sum(
        first_label_matrix > front_utils.NO_FRONT_ENUM, axis=0)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if this_num_fronts_matrix[i, j] == 0:
                continue

            first_unique_label_matrix[:, i, j] = (
                climo_utils.apply_separation_time(
                    front_type_enums=first_label_matrix[:, i, j],
                    valid_times_unix_sec=first_times_unix_sec,
                    separation_time_sec=separation_time_sec)
            )[0]

    num_wf_labels_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=float
    )
    num_cf_labels_matrix = num_wf_labels_matrix + 0.
    num_unique_wf_matrix = num_wf_labels_matrix + 0.
    num_unique_cf_matrix = num_wf_labels_matrix + 0.

    second_label_matrix = None

    for k in range(first_num_times, num_times):
        if numpy.mod(k, num_times_per_block) == 0 and k != num_times_per_block:
            second_num_times = second_label_matrix.shape[0]
            second_times_unix_sec = valid_times_unix_sec[
                (k - second_num_times):k
            ]

            this_count_dict = _update_counts(
                first_label_matrix=first_label_matrix,
                first_unique_label_matrix=first_unique_label_matrix,
                first_times_unix_sec=first_times_unix_sec,
                second_label_matrix=second_label_matrix,
                second_times_unix_sec=second_times_unix_sec,
                separation_time_sec=separation_time_sec,
                count_second_period=False)
            print(SEPARATOR_STRING)

            num_wf_labels_matrix = (
                num_wf_labels_matrix + this_count_dict[NUM_WF_LABELS_KEY]
            )
            num_cf_labels_matrix = (
                num_cf_labels_matrix + this_count_dict[NUM_CF_LABELS_KEY]
            )
            num_unique_wf_matrix = (
                num_unique_wf_matrix + this_count_dict[NUM_UNIQUE_WF_KEY]
            )
            num_unique_cf_matrix = (
                num_unique_cf_matrix + this_count_dict[NUM_UNIQUE_CF_KEY]
            )

            first_label_matrix = second_label_matrix + 0
            first_times_unix_sec = second_times_unix_sec + 0
            first_unique_label_matrix = this_count_dict[
                SECOND_UNIQUE_LABELS_KEY]
            second_label_matrix = None

        print('Reading deterministic labels from: "{0:s}"...'.format(
            prediction_file_names[k]
        ))

        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=prediction_file_names[k], read_deterministic=True)

        this_label_matrix = this_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]

        if second_label_matrix is None:
            second_label_matrix = this_label_matrix + 0
        else:
            second_label_matrix = numpy.concatenate(
                (second_label_matrix, this_label_matrix), axis=0
            )

    if second_label_matrix is None:
        second_times_unix_sec = None
    else:
        second_num_times = second_label_matrix.shape[0]
        second_times_unix_sec = valid_times_unix_sec[
            (num_times - second_num_times):num_times
        ]

    this_count_dict = _update_counts(
        first_label_matrix=first_label_matrix,
        first_unique_label_matrix=first_unique_label_matrix,
        first_times_unix_sec=first_times_unix_sec,
        second_label_matrix=second_label_matrix,
        second_times_unix_sec=second_times_unix_sec,
        separation_time_sec=separation_time_sec,
        count_second_period=True)
    print(SEPARATOR_STRING)

    num_wf_labels_matrix = (
        num_wf_labels_matrix + this_count_dict[NUM_WF_LABELS_KEY]
    )
    num_cf_labels_matrix = (
        num_cf_labels_matrix + this_count_dict[NUM_CF_LABELS_KEY]
    )
    num_unique_wf_matrix = (
        num_unique_wf_matrix + this_count_dict[NUM_UNIQUE_WF_KEY]
    )
    num_unique_cf_matrix = (
        num_unique_cf_matrix + this_count_dict[NUM_UNIQUE_CF_KEY]
    )

    for i in range(num_times):
        num_wf_labels_matrix[i, ...][mask_matrix == False] = numpy.nan
        num_cf_labels_matrix[i, ...][mask_matrix == False] = numpy.nan
        num_unique_wf_matrix[i, ...][mask_matrix == False] = numpy.nan
        num_unique_cf_matrix[i, ...][mask_matrix == False] = numpy.nan

    output_file_name = climo_utils.find_file(
        directory_name=output_dir_name,
        file_type_string=climo_utils.FRONT_COUNTS_STRING,
        first_time_unix_sec=valid_times_unix_sec[0],
        last_time_unix_sec=valid_times_unix_sec[-1],
        hours=hours_to_keep, months=months_to_keep,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    climo_utils.write_gridded_counts(
        netcdf_file_name=output_file_name,
        num_wf_labels_matrix=num_wf_labels_matrix,
        num_cf_labels_matrix=num_cf_labels_matrix,
        num_unique_wf_matrix=num_unique_wf_matrix,
        num_unique_cf_matrix=num_unique_cf_matrix,
        first_time_unix_sec=valid_times_unix_sec[0],
        last_time_unix_sec=valid_times_unix_sec[-1],
        prediction_file_names=prediction_file_names,
        separation_time_sec=separation_time_sec,
        hours=hours_to_keep, months=months_to_keep)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        hours_to_keep=numpy.array(
            getattr(INPUT_ARG_OBJECT, HOURS_ARG_NAME), dtype=int
        ),
        months_to_keep=numpy.array(
            getattr(INPUT_ARG_OBJECT, MONTHS_ARG_NAME), dtype=int
        ),
        separation_time_sec=getattr(INPUT_ARG_OBJECT, SEPARATION_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
