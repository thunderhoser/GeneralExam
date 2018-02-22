"""IO methods for machine learning."""

import os.path
import pickle
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%d-%H'

def write_downsized_examples_to_file(
        predictor_matrix, target_values, unix_times_sec, center_grid_rows,
        center_grid_columns, predictor_names, pickle_file_name):
    """Writes downsized examples (defined over subgrid, not full grid) to file.

    Downsized ML examples can be created by
    `machine_learning_utils.downsize_grids_around_each_point` or
    `machine_learning_utils.downsize_grids_around_selected_points`.

    C = number of predictor variables (image channels)

    :param predictor_matrix: See documentation for
        `downsize_grids_around_each_point` or
        `downsize_grids_around_selected_points`.
    :param target_values: Same.
    :param unix_times_sec: Same.
    :param center_grid_rows: Same.
    :param center_grid_columns: Same.
    :param predictor_names: length-C list with names of predictor variables.
    :param pickle_file_name: Path to output file.
    """

    ml_utils.check_downsized_examples(
        predictor_matrix=predictor_matrix, target_values=target_values,
        unix_times_sec=unix_times_sec, center_grid_rows=center_grid_rows,
        center_grid_columns=center_grid_columns,
        predictor_names=predictor_names, assert_binary_target_matrix=False)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(predictor_matrix, pickle_file_handle)
    pickle.dump(target_values, pickle_file_handle)
    pickle.dump(unix_times_sec, pickle_file_handle)
    pickle.dump(center_grid_rows, pickle_file_handle)
    pickle.dump(center_grid_columns, pickle_file_handle)
    pickle.dump(predictor_names, pickle_file_handle)
    pickle_file_handle.close()


def read_downsized_examples_from_file(pickle_file_name):
    """Reads downsized examples (defined over subgrid, not full grid) from file.

    :param pickle_file_name: Path to input file.
    :return: predictor_matrix: See documentation for
        `write_downsized_examples_to_file`.
    :return: target_values: See doc for `write_downsized_examples_to_file`.
    :return: unix_times_sec: See doc for `write_downsized_examples_to_file`.
    :return: center_grid_rows: See doc for `write_downsized_examples_to_file`.
    :return: center_grid_columns: See doc for
        `write_downsized_examples_to_file`.
    :return: predictor_names: See doc for `write_downsized_examples_to_file`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    predictor_matrix = pickle.load(pickle_file_handle)
    target_values = pickle.load(pickle_file_handle)
    unix_times_sec = pickle.load(pickle_file_handle)
    center_grid_rows = pickle.load(pickle_file_handle)
    center_grid_columns = pickle.load(pickle_file_handle)
    predictor_names = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    ml_utils.check_downsized_examples(
        predictor_matrix=predictor_matrix, target_values=target_values,
        unix_times_sec=unix_times_sec, center_grid_rows=center_grid_rows,
        center_grid_columns=center_grid_columns,
        predictor_names=predictor_names, assert_binary_target_matrix=False)

    return (predictor_matrix, target_values, unix_times_sec, center_grid_rows,
            center_grid_columns, predictor_names)


def find_downsized_example_file(
        top_directory_name, valid_time_unix_sec, pressure_level_mb,
        raise_error_if_missing=True):
    """Finds file with downsized examples (defined over subgrid, not full grid).

    :param top_directory_name: Name of top-level directory containing files with
        downsized machine-learning examples.
    :param valid_time_unix_sec: Valid time.
    :param pressure_level_mb: Pressure level (millibars).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: downsized_example_file_name: Path to file.
    """

    month_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_MONTH)
    time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAME)

    downsized_example_file_name = (
        '{0:s}/{1:s}/downsized_ml_examples_{2:04d}mb_{3:s}.p'.format(
            top_directory_name, month_string, pressure_level_mb, time_string))

    if raise_error_if_missing and not os.path.isfile(
            downsized_example_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                downsized_example_file_name))
        raise ValueError(error_string)

    return downsized_example_file_name
