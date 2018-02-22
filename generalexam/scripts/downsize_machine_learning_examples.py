"""Downsizes machine-learning examples.

Specifically, this script selects target points, creates a subgrid (smaller
version of the full grid) around each target point, then writes these subgrids
to a file.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_io as ml_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

# TODO(thunderhoser): This file contains a lot of methods (the helper methods)
# that should be elsewhere.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_POINTS_TO_SAMPLE_PER_TIME = 1000

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'
HOURS_TO_SECONDS = 3600

NARR_DIR_INPUT_ARG = 'input_narr_dir_name'
FRONTAL_GRID_DIR_INPUT_ARG = 'input_frontal_grid_dir_name'
PREDICTOR_NAMES_INPUT_ARG = 'narr_predictor_names'
PRESSURE_LEVEL_INPUT_ARG = 'pressure_level_mb'
TIME_INPUT_ARG = 'time_string'
DILATION_HALF_WIDTH_INPUT_ARG = 'dilation_half_width_in_grid_cells'
POSITIVE_FRACTION_INPUT_ARG = 'positive_fraction'
NUM_ROWS_INPUT_ARG = 'num_rows_in_half_window'
NUM_COLUMNS_INPUT_ARG = 'num_columns_in_half_window'
OUTPUT_DIR_INPUT_ARG = 'output_dir_name'

NARR_DIR_HELP_STRING = (
    'Name of directory with NARR files (one for each variable, pressure level, '
    'and month -- readable by `processed_narr_io.read_fields_from_file`).')
FRONTAL_GRID_DIR_HELP_STRING = (
    'Name of directory with frontal-grid files (one for each year, readable by '
    '`fronts_io.read_narr_grids_from_file`).')
PREDICTOR_NAMES_HELP_STRING = (
    'List of NARR variables to use as predictors.  Each must belong to the list'
    ' `processed_narr_io.FIELD_NAMES`.')
PRESSURE_LEVEL_HELP_STRING = (
    'NARR variables will be taken from this pressure level (millibars).')
TIME_HELP_STRING = (
    'Downsized ML examples will be created for this time step (format '
    '"yyyymmddHH").')
DILATION_HALF_WIDTH_HELP_STRING = (
    'Dilation half-width.  For each downsized grid, if there is any positive '
    'label (grid cell intersected by front) within `{0:s}` grid cells of the '
    'center, label will be positive (= 1 = yes).').format(
        DILATION_HALF_WIDTH_INPUT_ARG)
POSITIVE_FRACTION_HELP_STRING = (
    'Fraction of positive examples in downsized dataset.  A "positive example" '
    'is a point [time, row, column] with either a front.')
NUM_ROWS_HELP_STRING = 'Number of rows in half-window for downsizing.'
NUM_COLUMNS_HELP_STRING = 'Number of columns in half-window for downsizing.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (downsized examples will be saved in a Pickle '
    'file here).')

DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_FRONTAL_GRID_DIR_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/no_dilation')
DEFAULT_OUTPUT_DIR_NAME = '/condo/swatwork/ralager/ml_examples/downsized'

DEFAULT_NARR_PREDICTOR_NAMES = [
    processed_narr_io.WET_BULB_TEMP_NAME,
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME]

DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_DILATION_HALF_WIDTH_IN_GRID_CELLS = 8
DEFAULT_POSITIVE_FRACTION = 0.5
DEFAULT_NUM_ROWS_IN_HALF_WINDOW = 32
DEFAULT_NUM_COLUMNS_IN_HALF_WINDOW = 32

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONTAL_GRID_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_FRONTAL_GRID_DIR_NAME, help=FRONTAL_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_NAMES_INPUT_ARG, type=str, nargs='+', required=False,
    default=DEFAULT_NARR_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_INPUT_ARG, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TIME_INPUT_ARG, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_HALF_WIDTH_INPUT_ARG, type=int, required=False,
    default=DEFAULT_DILATION_HALF_WIDTH_IN_GRID_CELLS,
    help=DILATION_HALF_WIDTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + POSITIVE_FRACTION_INPUT_ARG, type=float, required=False,
    default=DEFAULT_POSITIVE_FRACTION, help=POSITIVE_FRACTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_INPUT_ARG, type=int, required=False,
    default=DEFAULT_NUM_ROWS_IN_HALF_WINDOW, help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_INPUT_ARG, type=int, required=False,
    default=DEFAULT_NUM_COLUMNS_IN_HALF_WINDOW, help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _month_to_start_end_times(month_string):
    """Returns start/end times for month.

    :param month_string: Month (format "yyyymm").
    :return: start_time_string: Start time (first NARR initialization in month)
        (format "yyyymmddHH")
    :return: end_time_string: End time (last NARR initialization in month)
        (format "yyyymmddHH")
    """

    month_unix_sec = time_conversion.string_to_unix_sec(
        month_string, TIME_FORMAT_MONTH)
    start_time_unix_sec, end_time_unix_sec = (
        time_conversion.first_and_last_times_in_month(month_unix_sec))

    narr_time_step_hours = nwp_model_utils.get_time_steps(
        nwp_model_utils.NARR_MODEL_NAME)[1]
    narr_time_step_seconds = narr_time_step_hours * HOURS_TO_SECONDS
    end_time_unix_sec += 1 - narr_time_step_seconds

    start_time_string = time_conversion.unix_sec_to_string(
        start_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    end_time_string = time_conversion.unix_sec_to_string(
        end_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)

    return start_time_string, end_time_string


def _find_narr_file(
        directory_name, field_name, pressure_level_mb, month_string):
    """Finds NARR file.

    This file should contain one variable at one pressure level for one month.

    :param directory_name: Path to directory.
    :param field_name: Field name (must belong to the list
        `processed_narr_io.FIELD_NAMES`).
    :param pressure_level_mb: Pressure level (millibars).
    :param month_string: Month (format "yyyymm").
    :return: narr_file_name: Path to file.
    """

    start_time_string, end_time_string = _month_to_start_end_times(
        month_string)

    field_name_unitless = field_name.replace('_kelvins', '').replace(
        '_m_s01', '')

    return '{0:s}/{1:s}_{2:04d}mb_{3:s}-{4:s}.p'.format(
        directory_name, field_name_unitless, pressure_level_mb,
        start_time_string, end_time_string)


def _find_frontal_grid_file(directory_name, year_string):
    """Finds file with front labels.

    Specifically, this file should contain NARR grid points intersected by a
    front for each time step in one year.

    :param directory_name: Path to directory.
    :param year_string: Year (format "yyyy").
    :return: frontal_grid_file_name: Path to file.
    """

    start_time_unix_sec, end_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(int(year_string)))

    narr_time_step_hours = nwp_model_utils.get_time_steps(
        nwp_model_utils.NARR_MODEL_NAME)[1]
    narr_time_step_seconds = narr_time_step_hours * HOURS_TO_SECONDS
    end_time_unix_sec += 1 - narr_time_step_seconds

    start_time_string = time_conversion.unix_sec_to_string(
        start_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    end_time_string = time_conversion.unix_sec_to_string(
        end_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)

    return '{0:s}/narr_frontal_grids_{1:s}-{2:s}.p'.format(
        directory_name, start_time_string, end_time_string)


def _downsize_ml_examples(
        input_narr_dir_name, input_frontal_grid_dir_name, narr_predictor_names,
        pressure_level_mb, time_string, dilation_half_width_in_grid_cells,
        positive_fraction, num_rows_in_half_window, num_columns_in_half_window,
        output_dir_name):
    """Downsizes machine-learning examples.

    :param input_narr_dir_name: Name of directory with NARR files (one for each
        variable, pressure level, and month -- readable by
        `processed_narr_io.read_fields_from_file`)
    :param input_frontal_grid_dir_name: Name of directory with frontal-grid
        files (one for each year, readable by
        `fronts_io.read_narr_grids_from_file`).
    :param narr_predictor_names: List of NARR variables to use as predictors.
        Each must belong to the list `processed_narr_io.FIELD_NAMES`.
    :param pressure_level_mb: NARR variables will be taken from this pressure
        level (millibars).
    :param time_string: Downsized ML examples will be created for this time step
        (format "yyyymmddHH").
    :param dilation_half_width_in_grid_cells: Dilation half-width.  For each
        downsized grid, if there is any positive label (grid cell intersected by
        front) within `dilation_half_width_in_grid_cells` grid cells of the
        center, label will be positive (= 1 = yes).
    :param positive_fraction: Fraction of positive examples in downsized
        dataset.  A "positive example" is a point [time, row, column] with
        either a front.
    :param num_rows_in_half_window: Number of rows in half-window for
        downsizing.
    :param num_columns_in_half_window: Number of columns in half-window for
        downsizing.
    :param output_dir_name: Name of output directory (downsized examples will be
        saved in a Pickle file here).
    :raises: ValueError: if `dilation_half_width_in_grid_cells` >=
        `num_rows_in_half_window` or `num_columns_in_half_window`.
    """

    if dilation_half_width_in_grid_cells >= num_rows_in_half_window:
        error_string = (
            '`dilation_half_width_in_grid_cells` ({0:d}) should be < '
            '`num_rows_in_half_window` ({1:d}).').format(
                dilation_half_width_in_grid_cells, num_rows_in_half_window)
        raise ValueError(error_string)

    if dilation_half_width_in_grid_cells >= num_columns_in_half_window:
        error_string = (
            '`dilation_half_width_in_grid_cells` ({0:d}) should be < '
            '`num_columns_in_half_window` ({1:d}).').format(
                dilation_half_width_in_grid_cells, num_columns_in_half_window)
        raise ValueError(error_string)

    num_predictors = len(narr_predictor_names)
    predictor_times_unix_sec = None
    keep_predictor_time_indices = None
    tuple_of_predictor_matrices = ()

    for m in range(num_predictors):
        this_predictor_file_name = _find_narr_file(
            directory_name=input_narr_dir_name,
            field_name=narr_predictor_names[m],
            pressure_level_mb=pressure_level_mb, month_string=time_string[:6])

        print 'Reading predictor fields at {0:s} from: "{1:s}"...'.format(
            time_string, this_predictor_file_name)

        if m == 0:
            this_predictor_matrix, _, _, predictor_times_unix_sec = (
                processed_narr_io.read_fields_from_file(
                    this_predictor_file_name))

            these_time_strings = [
                time_conversion.unix_sec_to_string(t, INPUT_TIME_FORMAT) for
                t in predictor_times_unix_sec]

            keep_predictor_time_flags = numpy.array(
                [s == time_string for s in these_time_strings])
            keep_predictor_time_indices = numpy.where(
                keep_predictor_time_flags)[0]
            predictor_times_unix_sec = predictor_times_unix_sec[
                keep_predictor_time_indices]

        else:
            this_predictor_matrix, _, _, _ = (
                processed_narr_io.read_fields_from_file(
                    this_predictor_file_name))

        this_predictor_matrix = this_predictor_matrix[
            keep_predictor_time_indices, ...]
        tuple_of_predictor_matrices += (this_predictor_matrix,)

    predictor_matrix = ml_utils.stack_predictor_variables(
        tuple_of_predictor_matrices)

    frontal_grid_file_name = _find_frontal_grid_file(
        directory_name=input_frontal_grid_dir_name, year_string=time_string[:4])

    print 'Reading target labels from: "{0:s}"...'.format(
        frontal_grid_file_name)
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        frontal_grid_file_name)

    print 'Matching time steps of predictor fields and target labels...'
    frontal_grid_table = frontal_grid_table.loc[
        frontal_grid_table[front_utils.TIME_COLUMN].isin(
            predictor_times_unix_sec)]
    frontal_grid_table.sort_values(
        [front_utils.TIME_COLUMN], axis=0, ascending=[True], inplace=True)

    print 'Converting target labels to grids...'
    num_grid_rows = predictor_matrix.shape[1]
    num_grid_columns = predictor_matrix.shape[2]
    frontal_grid_matrix = ml_utils.front_table_to_matrices(
        frontal_grid_table=frontal_grid_table, num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns)

    print 'Binarizing target labels...'
    frontal_grid_matrix = ml_utils.binarize_front_labels(frontal_grid_matrix)

    print 'Removing NaN''s from predictor and target grids...'
    predictor_matrix = ml_utils.remove_nans_from_narr_grid(predictor_matrix)
    frontal_grid_matrix = ml_utils.remove_nans_from_narr_grid(
        frontal_grid_matrix)
    print SEPARATOR_STRING

    frontal_grid_matrix = ml_utils.dilate_target_grids(
        binary_target_matrix=frontal_grid_matrix,
        num_grid_cells_in_half_window=dilation_half_width_in_grid_cells)

    print ('Downsampling target points (so that fraction of positive cases = '
           '{0:f})...').format(positive_fraction)
    sampled_target_point_dict = ml_utils.sample_target_points(
        binary_target_matrix=frontal_grid_matrix,
        positive_fraction=positive_fraction,
        num_points_per_time=NUM_POINTS_TO_SAMPLE_PER_TIME)
    print SEPARATOR_STRING

    (downsized_predictor_matrix,
     target_values,
     time_indices,
     center_grid_rows,
     center_grid_columns) = ml_utils.downsize_grids_around_selected_points(
         predictor_matrix=predictor_matrix, target_matrix=frontal_grid_matrix,
         num_rows_in_half_window=num_rows_in_half_window,
         num_columns_in_half_window=num_columns_in_half_window,
         target_point_dict=sampled_target_point_dict)
    print SEPARATOR_STRING

    downsized_times_unix_sec = predictor_times_unix_sec[time_indices]

    output_file_name = ml_io.find_downsized_example_file(
        top_directory_name=output_dir_name,
        valid_time_unix_sec=
        time_conversion.string_to_unix_sec(time_string, INPUT_TIME_FORMAT),
        pressure_level_mb=pressure_level_mb, raise_error_if_missing=False)

    print 'Writing downsized examples to: "{0:s}"...'.format(output_file_name)
    print downsized_predictor_matrix.shape
    ml_io.write_downsized_examples_to_file(
        predictor_matrix=downsized_predictor_matrix,
        target_values=target_values, unix_times_sec=downsized_times_unix_sec,
        center_grid_rows=center_grid_rows,
        center_grid_columns=center_grid_columns,
        predictor_names=narr_predictor_names, pickle_file_name=output_file_name)


def add_input_arguments(argument_parser_object):
    """Adds input args for this script to `argparse.ArgumentParser` object.

    :param argument_parser_object: `argparse.ArgumentParser` object, which may
        or may not already contain input args.
    :return: argument_parser_object: Same as input object, but with new input
        args added.
    """

    argument_parser_object.add_argument(
        '--' + NARR_DIR_INPUT_ARG, type=str, required=False,
        default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + FRONTAL_GRID_DIR_INPUT_ARG, type=str, required=False,
        default=DEFAULT_FRONTAL_GRID_DIR_NAME,
        help=FRONTAL_GRID_DIR_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PREDICTOR_NAMES_INPUT_ARG, type=str, nargs='+', required=False,
        default=DEFAULT_NARR_PREDICTOR_NAMES, help=PREDICTOR_NAMES_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + PRESSURE_LEVEL_INPUT_ARG, type=int, required=False,
        default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + DILATION_HALF_WIDTH_INPUT_ARG, type=int, required=False,
        default=DEFAULT_DILATION_HALF_WIDTH_IN_GRID_CELLS,
        help=DILATION_HALF_WIDTH_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + POSITIVE_FRACTION_INPUT_ARG, type=float, required=False,
        default=DEFAULT_POSITIVE_FRACTION, help=POSITIVE_FRACTION_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_ROWS_INPUT_ARG, type=int, required=False,
        default=DEFAULT_NUM_ROWS_IN_HALF_WINDOW, help=NUM_ROWS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + NUM_COLUMNS_INPUT_ARG, type=int, required=False,
        default=DEFAULT_NUM_COLUMNS_IN_HALF_WINDOW,
        help=NUM_COLUMNS_HELP_STRING)

    argument_parser_object.add_argument(
        '--' + OUTPUT_DIR_INPUT_ARG, type=str, required=False,
        default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)

    return argument_parser_object


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    INPUT_NARR_DIR_NAME = getattr(INPUT_ARG_OBJECT, NARR_DIR_INPUT_ARG)
    INPUT_FRONTAL_GRID_DIR_NAME = getattr(
        INPUT_ARG_OBJECT, FRONTAL_GRID_DIR_INPUT_ARG)

    NARR_PREDICTOR_NAMES = getattr(INPUT_ARG_OBJECT, PREDICTOR_NAMES_INPUT_ARG)
    PRESSURE_LEVEL_MB = getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_INPUT_ARG)
    TIME_STRING = getattr(INPUT_ARG_OBJECT, TIME_INPUT_ARG)
    DILATION_HALF_WIDTH_IN_GRID_CELLS = getattr(
        INPUT_ARG_OBJECT, DILATION_HALF_WIDTH_INPUT_ARG)

    POSITIVE_FRACTION = getattr(INPUT_ARG_OBJECT, POSITIVE_FRACTION_INPUT_ARG)
    NUM_ROWS_IN_HALF_WINDOW = getattr(INPUT_ARG_OBJECT, NUM_ROWS_INPUT_ARG)
    NUM_COLUMNS_IN_HALF_WINDOW = getattr(
        INPUT_ARG_OBJECT, NUM_COLUMNS_INPUT_ARG)

    OUTPUT_DIR_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_INPUT_ARG)

    _downsize_ml_examples(
        input_narr_dir_name=INPUT_NARR_DIR_NAME,
        input_frontal_grid_dir_name=INPUT_FRONTAL_GRID_DIR_NAME,
        narr_predictor_names=NARR_PREDICTOR_NAMES,
        pressure_level_mb=PRESSURE_LEVEL_MB, time_string=TIME_STRING,
        dilation_half_width_in_grid_cells=DILATION_HALF_WIDTH_IN_GRID_CELLS,
        positive_fraction=POSITIVE_FRACTION,
        num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
        num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW,
        output_dir_name=OUTPUT_DIR_NAME)
