"""IO methods for machine learning."""

import glob
import random
import os.path
import pickle
import numpy
import keras
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%d-%H'

NUM_CLASSES_FOR_DOWNSIZED_EXAMPLES = 2


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


def downsized_example_generator(input_file_pattern, num_examples_per_batch):
    """Generates downsized input examples for a Keras model.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.downsized_example_generator(
            training_file_pattern, batch_size),
        ...)

    B = batch size (number of examples per batch)
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    C = number of channels (predictor variables)

    :param input_file_pattern: Glob pattern for input files (example:
        "ml_examples/downsized/201712/*.p").  All files matching this pattern
        will be used.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is just known as "batch_size" in Keras.
    :return: predictor_matrix: B-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-B numpy array of corresponding targets
        (labels).
    """

    input_file_names = glob.glob(input_file_pattern)
    random.shuffle(input_file_names)
    num_files = len(input_file_names)

    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    file_index = 0
    num_examples_in_memory = 0
    predictor_matrix = None
    target_values = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            if target_values is None:
                predictor_matrix, target_values, _, _, _, _ = (
                    read_downsized_examples_from_file(
                        input_file_names[file_index]))
            else:
                this_predictor_matrix, these_target_values, _, _, _, _ = (
                    read_downsized_examples_from_file(
                        input_file_names[file_index]))

                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0)
                target_values = numpy.concatenate(
                    (target_values, these_target_values))

            num_examples_in_memory = len(target_values)
            file_index += 1
            if file_index >= num_files:
                file_index = 0

        example_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)

        numpy.random.shuffle(example_indices)
        predictor_matrix = predictor_matrix[example_indices, ...]
        target_values = target_values[example_indices]

        predictor_matrix_to_return = predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values_to_return = keras.utils.to_categorical(
            target_values[batch_indices], NUM_CLASSES_FOR_DOWNSIZED_EXAMPLES)

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_values_to_return)
