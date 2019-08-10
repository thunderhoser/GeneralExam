"""Counts WF and CF labels at each grid cell over a given time period."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800
ACCEPTED_HOURS = numpy.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=int)

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
    ' by `climatology_utils.find_gridded_count_file`.')

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
    :raises: ValueError: if any value in `hours_to_keep` is not in the list
        `ACCEPTED_HOURS`.
    """

    if len(hours_to_keep) == 1 and hours_to_keep[0] == -1:
        hours_to_keep = None

    if len(months_to_keep) == 1 and months_to_keep[0] == -1:
        months_to_keep = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

    if hours_to_keep is not None:
        these_flags = numpy.array(
            [h in ACCEPTED_HOURS for h in hours_to_keep], dtype=bool
        )

        if not numpy.all(these_flags):
            error_string = (
                '\n{0:s}\nAt least one hour (listed above) is not in the list '
                'of accepted hours (listed below).\n{1:s}'
            ).format(str(hours_to_keep), str(ACCEPTED_HOURS))

            raise ValueError(error_string)

        indices_to_keep = climatology_utils.filter_by_hour(
            all_times_unix_sec=valid_times_unix_sec,
            hours_to_keep=hours_to_keep)

        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]

    if months_to_keep is not None:
        indices_to_keep = climatology_utils.filter_by_month(
            all_times_unix_sec=valid_times_unix_sec,
            months_to_keep=months_to_keep)

        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]

    prediction_file_names = [
        prediction_io.find_file(
            directory_name=prediction_dir_name,
            first_time_unix_sec=t, last_time_unix_sec=t,
            raise_error_if_missing=True)
        for t in valid_times_unix_sec
    ]

    predicted_label_matrix = None

    for this_file_name in prediction_file_names:
        print('Reading deterministic labels from: "{0:s}"...'.format(
            this_file_name
        ))

        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_file_name, read_deterministic=True)

        this_label_matrix = this_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]

        if predicted_label_matrix is None:
            predicted_label_matrix = this_label_matrix + 0
        else:
            predicted_label_matrix = numpy.concatenate(
                (predicted_label_matrix, this_label_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    num_grid_rows = predicted_label_matrix.shape[1]
    num_grid_columns = predicted_label_matrix.shape[2]

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_orig_num_wf = numpy.sum(
                predicted_label_matrix[:, i, j] == front_utils.WARM_FRONT_ENUM
            )
            this_orig_num_cf = numpy.sum(
                predicted_label_matrix[:, i, j] == front_utils.COLD_FRONT_ENUM
            )

            if this_orig_num_wf == this_orig_num_cf == 0:
                continue

            print((
                'Applying {0:d}-second separation time to grid cell '
                '[{1:d}, {2:d}]...'
            ).format(
                separation_time_sec, i, j
            ))

            predicted_label_matrix[:, i, j] = (
                climatology_utils.apply_separation_time(
                    front_type_enums=predicted_label_matrix[:, i, j],
                    valid_times_unix_sec=valid_times_unix_sec,
                    separation_time_sec=separation_time_sec)
            )[0]

            this_new_num_wf = numpy.sum(
                predicted_label_matrix[:, i, j] == front_utils.WARM_FRONT_ENUM
            )
            this_new_num_cf = numpy.sum(
                predicted_label_matrix[:, i, j] == front_utils.COLD_FRONT_ENUM
            )

            print((
                'Number of WF labels reduced from {0:d} to {1:d} ... CF labels '
                'reduced from {2:d} to {3:d}\n'
            ).format(
                this_orig_num_wf, this_new_num_wf, this_orig_num_cf,
                this_new_num_cf
            ))

    output_file_name = climatology_utils.find_gridded_count_file(
        directory_name=output_dir_name,
        first_time_unix_sec=valid_times_unix_sec[0],
        last_time_unix_sec=valid_times_unix_sec[-1],
        hours=hours_to_keep, months=months_to_keep,
        raise_error_if_missing=False)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    num_warm_fronts_matrix = numpy.sum(
        predicted_label_matrix == front_utils.WARM_FRONT_ENUM, axis=0)
    num_cold_fronts_matrix = numpy.sum(
        predicted_label_matrix == front_utils.COLD_FRONT_ENUM, axis=0)

    climatology_utils.write_gridded_counts(
        netcdf_file_name=output_file_name,
        num_warm_fronts_matrix=num_warm_fronts_matrix,
        num_cold_fronts_matrix=num_cold_fronts_matrix,
        prediction_file_names=prediction_file_names,
        separation_time_sec=separation_time_sec)


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
