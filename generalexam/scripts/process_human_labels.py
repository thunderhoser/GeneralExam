"""Processes gridded WF and CF labels from human meteorologists."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import fronts_io
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import predictor_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

NUM_NARR_ROWS, NUM_NARR_COLUMNS = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME,
    grid_name=nwp_model_utils.NAME_OF_221GRID
)

TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_label_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
SEPARATION_TIME_ARG_NAME = 'separation_time_sec'
PREDICTOR_FILE_ARG_NAME = 'predictor_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`fronts_io.find_gridded_file` and read by `fronts_io.read_grid_from_file`.'
)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will process labels for all times'
    ' from `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance (will be used to "fatten" human-drawn fronts).'
)

SEPARATION_TIME_HELP_STRING = (
    'Separation time (used to remove redundant front labels).  If grid cell '
    '[i, j] has multiple front labels of the same type within `{0:s}` seconds, '
    'only one such label will count.'
).format(SEPARATION_TIME_ARG_NAME)

PREDICTOR_FILE_HELP_STRING = (
    'Path to predictor file (readable by `predictor_io.read_file`), on the same'
    ' grid as human labels.  This will be used to mask out grid cells with no '
    'reanalysis data.  If you do not want to bother with masking, leave this '
    'alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each time step, file will be written by '
    '`climatology_utils.write_gridded_labels`, to a location therein determined'
    ' by `climatology_utils.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False,
    default=50000, help=DILATION_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEPARATION_TIME_ARG_NAME, type=int, required=False, default=86400,
    help=SEPARATION_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_FILE_ARG_NAME, type=str, required=False, default='',
    help=PREDICTOR_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _label_file_name_to_time(human_label_file_name):
    """Parses time from name of label file.

    :param human_label_file_name: Path to label file.
    :return: valid_time_unix_sec: Valid time.
    """

    valid_time_string = (
        os.path.split(human_label_file_name)[-1].split('_')[-1].split('.')[0]
    )

    return time_conversion.string_to_unix_sec(valid_time_string, TIME_FORMAT)


def _write_new_labels(
        first_label_matrix, first_unique_label_matrix,
        first_label_file_names, second_label_matrix,
        second_label_file_names, write_second_period, separation_time_sec,
        output_dir_name, mask_matrix=None):
    """Writes new labels to files (one per time step).

    F = number of times in first period
    S = number of times in second period
    M = number of rows in grid
    N = number of columns in grid

     :param first_label_matrix: F-by-M-by-N numpy array of front labels
        (integers) for first period.
    :param first_unique_label_matrix: Same but after applying separation time.
    :param first_label_file_names: length-F list of paths to label files.
    :param second_label_matrix: S-by-M-by-N numpy array of front labels
        (integers) for second period.
    :param second_label_file_names: length-S list of paths to label files.
    :param write_second_period: Boolean flag.  If True, will write labels for
        both periods.  If False, will write labels only for first period.
    :param separation_time_sec: See documentation at top of file.
    :param output_dir_name: Same.
    :param mask_matrix: M-by-N numpy array of integers in 0...1.  If
        mask_matrix[i, j] = 0, grid cell [i, j] will be masked out.  In other
        words, all labels at grid cell [i, j] will be turned into NaN.
    :return: second_unique_label_matrix: Same as input `second_label_matrix` but
        after applying separation time.
    """

    first_times_unix_sec = numpy.array([
        _label_file_name_to_time(f) for f in first_label_file_names
    ], dtype=int)

    have_second_period = second_label_matrix is not None

    if have_second_period:
        second_times_unix_sec = numpy.array([
            _label_file_name_to_time(f) for f in second_label_file_names
        ], dtype=int)

        unique_label_matrix = numpy.concatenate(
            (first_unique_label_matrix, second_label_matrix), axis=0
        )
        valid_times_unix_sec = numpy.concatenate((
            first_times_unix_sec, second_times_unix_sec
        ))

        num_grid_rows = unique_label_matrix.shape[1]
        num_grid_columns = unique_label_matrix.shape[2]

        this_num_fronts_matrix = numpy.sum(
            unique_label_matrix > front_utils.NO_FRONT_ENUM, axis=0
        )

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                if this_num_fronts_matrix[i, j] == 0:
                    continue

                unique_label_matrix[:, i, j] = (
                    climo_utils.apply_separation_time(
                        front_type_enums=unique_label_matrix[:, i, j],
                        valid_times_unix_sec=valid_times_unix_sec,
                        separation_time_sec=separation_time_sec
                    )[0]
                )

        second_unique_label_matrix = (
            unique_label_matrix[len(first_times_unix_sec):, ...]
        )
    else:
        second_unique_label_matrix = None
        second_times_unix_sec = None

    for i in range(len(first_times_unix_sec)):
        this_output_file_name = climo_utils.find_basic_file(
            directory_name=output_dir_name,
            file_type_string=climo_utils.FRONT_LABELS_STRING,
            valid_time_unix_sec=first_times_unix_sec[i],
            raise_error_if_missing=False
        )

        print('Writing labels to: "{0:s}"...'.format(this_output_file_name))

        this_label_matrix = first_label_matrix[i, ...].astype(float)
        this_unique_label_matrix = (
            first_unique_label_matrix[i, ...].astype(float)
        )

        if mask_matrix is not None:
            this_label_matrix[mask_matrix == 0] = numpy.nan
            this_unique_label_matrix[mask_matrix == 0] = numpy.nan

        climo_utils.write_gridded_labels(
            netcdf_file_name=this_output_file_name,
            label_matrix=this_label_matrix,
            unique_label_matrix=this_unique_label_matrix,
            prediction_file_name=first_label_file_names[i],
            separation_time_sec=separation_time_sec
        )

    if not(have_second_period and write_second_period):
        return second_unique_label_matrix

    for i in range(len(second_times_unix_sec)):
        this_output_file_name = climo_utils.find_basic_file(
            directory_name=output_dir_name,
            file_type_string=climo_utils.FRONT_LABELS_STRING,
            valid_time_unix_sec=second_times_unix_sec[i],
            raise_error_if_missing=False
        )

        print('Writing labels to: "{0:s}"...'.format(this_output_file_name))

        this_label_matrix = second_label_matrix[i, ...].astype(float)
        this_unique_label_matrix = (
            second_unique_label_matrix[i, ...].astype(float)
        )

        if mask_matrix is not None:
            this_label_matrix[mask_matrix == 0] = numpy.nan
            this_unique_label_matrix[mask_matrix == 0] = numpy.nan

        climo_utils.write_gridded_labels(
            netcdf_file_name=this_output_file_name,
            label_matrix=this_label_matrix,
            unique_label_matrix=this_unique_label_matrix,
            prediction_file_name=second_label_file_names[i],
            separation_time_sec=separation_time_sec
        )

    return second_unique_label_matrix


def _run(top_label_dir_name, first_time_string, last_time_string,
         dilation_distance_metres, separation_time_sec, predictor_file_name,
         output_dir_name):
    """Processes gridded WF and CF labels from human meteorologists.

    This is effectively the main method.

    :param top_label_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param dilation_distance_metres: Same.
    :param separation_time_sec: Same.
    :param predictor_file_name: Same.
    :param output_dir_name: Same.
    """

    if predictor_file_name in ['', 'None']:
        mask_matrix = None
    else:
        print('Reading predictors from: "{0:s}"...'.format(predictor_file_name))
        predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_name
        )

        mask_matrix = numpy.invert(numpy.isnan(
            predictor_dict[predictor_utils.DATA_MATRIX_KEY][0, ..., 0]
        ))
        mask_matrix = mask_matrix.astype(int)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )
    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    valid_times_unix_sec = []
    human_label_file_names = []

    for i in range(len(all_times_unix_sec)):
        this_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_label_dir_name,
            valid_time_unix_sec=all_times_unix_sec[i],
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_file_name):
            continue

        valid_times_unix_sec.append(all_times_unix_sec[i])
        human_label_file_names.append(this_file_name)

    valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)

    if separation_time_sec <= 0:
        num_times_per_block = 100
    else:
        smallest_time_step_sec = numpy.min(numpy.diff(valid_times_unix_sec))
        num_times_per_block = 10 * int(
            numpy.ceil(float(separation_time_sec) / smallest_time_step_sec)
        )

    num_times = len(human_label_file_names)
    num_times_per_block = min([num_times_per_block, num_times])

    first_label_matrix = None

    for k in range(num_times_per_block):
        print('Reading human labels from: "{0:s}"...'.format(
            human_label_file_names[k]
        ))

        this_front_table = fronts_io.read_grid_from_file(
            human_label_file_names[k]
        )

        this_label_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=this_front_table,
            num_rows_per_image=NUM_NARR_ROWS,
            num_columns_per_image=NUM_NARR_COLUMNS
        )

        this_label_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_label_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False
        )

        this_label_matrix = numpy.pad(
            this_label_matrix,
            pad_width=((0, 0), (100, 100), (100, 100)),
            mode='constant', constant_values=0
        )

        if first_label_matrix is None:
            first_label_matrix = this_label_matrix + 0
        else:
            first_label_matrix = numpy.concatenate(
                (first_label_matrix, this_label_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    first_num_times = first_label_matrix.shape[0]
    first_unique_label_matrix = first_label_matrix + 0

    num_grid_rows = first_label_matrix.shape[1]
    num_grid_columns = first_label_matrix.shape[2]

    this_num_fronts_matrix = numpy.sum(
        first_label_matrix > front_utils.NO_FRONT_ENUM, axis=0
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if this_num_fronts_matrix[i, j] == 0:
                continue

            first_unique_label_matrix[:, i, j] = (
                climo_utils.apply_separation_time(
                    front_type_enums=first_label_matrix[:, i, j],
                    valid_times_unix_sec=valid_times_unix_sec[:first_num_times],
                    separation_time_sec=separation_time_sec)
            )[0]

    second_label_matrix = None
    first_label_file_names = human_label_file_names[:first_num_times]

    for k in range(first_num_times, num_times):
        if numpy.mod(k, num_times_per_block) == 0 and k != num_times_per_block:
            second_num_times = second_label_matrix.shape[0]
            second_label_file_names = (
                human_label_file_names[(k - second_num_times):k]
            )

            print(SEPARATOR_STRING)
            first_unique_label_matrix = _write_new_labels(
                first_label_matrix=first_label_matrix,
                first_unique_label_matrix=first_unique_label_matrix,
                first_label_file_names=first_label_file_names,
                second_label_matrix=second_label_matrix,
                second_label_file_names=second_label_file_names,
                write_second_period=False, mask_matrix=mask_matrix,
                separation_time_sec=separation_time_sec,
                output_dir_name=output_dir_name
            )
            print(SEPARATOR_STRING)

            first_label_matrix = second_label_matrix + 0
            first_label_file_names = copy.deepcopy(second_label_file_names)
            second_label_matrix = None

        print('Reading human labels from: "{0:s}"...'.format(
            human_label_file_names[k]
        ))

        this_front_table = fronts_io.read_grid_from_file(
            human_label_file_names[k]
        )

        this_label_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=this_front_table,
            num_rows_per_image=NUM_NARR_ROWS,
            num_columns_per_image=NUM_NARR_COLUMNS
        )

        this_label_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_label_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False
        )

        this_label_matrix = numpy.pad(
            this_label_matrix,
            pad_width=((0, 0), (100, 100), (100, 100)),
            mode='constant', constant_values=0
        )

        if second_label_matrix is None:
            second_label_matrix = this_label_matrix + 0
        else:
            second_label_matrix = numpy.concatenate(
                (second_label_matrix, this_label_matrix), axis=0
            )

    if second_label_matrix is None:
        second_label_file_names = None
    else:
        second_num_times = second_label_matrix.shape[0]
        second_label_file_names = (
            human_label_file_names[(num_times - second_num_times):num_times]
        )

    print(SEPARATOR_STRING)
    _write_new_labels(
        first_label_matrix=first_label_matrix,
        first_unique_label_matrix=first_unique_label_matrix,
        first_label_file_names=first_label_file_names,
        second_label_matrix=second_label_matrix,
        second_label_file_names=second_label_file_names,
        write_second_period=True, mask_matrix=mask_matrix,
        separation_time_sec=separation_time_sec,
        output_dir_name=output_dir_name
    )

if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_label_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        separation_time_sec=getattr(INPUT_ARG_OBJECT, SEPARATION_TIME_ARG_NAME),
        predictor_file_name=getattr(INPUT_ARG_OBJECT, PREDICTOR_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
