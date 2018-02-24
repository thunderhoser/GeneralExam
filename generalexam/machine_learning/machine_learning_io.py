"""IO methods for machine learning."""

import glob
import random
import os.path
import pickle
import numpy
import keras
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%d-%H'

NUM_CLASSES_FOR_DOWNSIZED_EXAMPLES = 2
NARR_TIME_INTERVAL_SECONDS = nwp_model_utils.get_time_steps(
    nwp_model_utils.NARR_MODEL_NAME)[1]


def _check_input_args_for_otf_generator(
        num_examples_per_batch, num_examples_per_time, narr_predictor_names,
        dilation_half_width_for_target, num_rows_in_half_grid,
        num_columns_in_half_grid):
    """Error-checks input args for `downsized_example_generator_on_the_fly`.

    :param num_examples_per_batch: See documentation for
        `downsized_example_generator_on_the_fly`.
    :param num_examples_per_time: Same.
    :param narr_predictor_names: Same.
    :param dilation_half_width_for_target: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :raises: ValueError: if `dilation_half_width_for_target` >=
        `num_rows_in_half_grid` or `num_columns_in_half_grid`.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 2)
    error_checking.assert_is_integer(num_examples_per_time)
    error_checking.assert_is_geq(num_examples_per_time, 2)
    error_checking.assert_is_string_list(narr_predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(narr_predictor_names), num_dimensions=1)

    error_checking.assert_is_integer(dilation_half_width_for_target)
    error_checking.assert_is_greater(dilation_half_width_for_target, 0)
    error_checking.assert_is_integer(num_rows_in_half_grid)
    error_checking.assert_is_greater(num_rows_in_half_grid, 0)
    error_checking.assert_is_integer(num_columns_in_half_grid)
    error_checking.assert_is_greater(num_columns_in_half_grid, 0)

    if dilation_half_width_for_target >= num_rows_in_half_grid:
        error_string = (
            'dilation_half_width_for_target ({0:d}) should be < '
            'num_rows_in_half_grid ({1:d}).').format(
                dilation_half_width_for_target, num_rows_in_half_grid)
        raise ValueError(error_string)

    if dilation_half_width_for_target >= num_columns_in_half_grid:
        error_string = (
            'dilation_half_width_for_target ({0:d}) should be < '
            'num_columns_in_half_grid ({1:d}).').format(
                dilation_half_width_for_target, num_columns_in_half_grid)
        raise ValueError(error_string)


def _find_files_for_otf_generator(
        start_time_unix_sec, end_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb):
    """Finds input files for `downsized_example_generator_on_the_fly`.

    T = number of time steps
    P = number of predictor variables

    :param start_time_unix_sec: See documentation for
        `downsized_example_generator_on_the_fly`.
    :param end_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :return: narr_file_names_by_predictor: P-by-T list of paths to NARR files,
        each containing the grid for one predictor field at one time step.
    :return: frontal_grid_file_names: length-T list of paths to frontal-grid
        files.  Each contains the indices of all NARR grid points intersected by
        a front (warm or cold) at one time step.
    """

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=start_time_unix_sec,
        end_time_unix_sec=end_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(valid_times_unix_sec)

    num_times = len(valid_times_unix_sec)
    num_predictors = len(narr_predictor_names)
    narr_file_names_by_predictor = [[] * num_times] * num_predictors
    frontal_grid_file_names = [] * num_times

    for i in range(num_times):
        for j in range(num_predictors):
            narr_file_names_by_predictor[j][i] = (
                processed_narr_io.find_file_for_one_time(
                    top_directory_name=top_narr_directory_name,
                    field_name=narr_predictor_names[j],
                    pressure_level_mb=pressure_level_mb,
                    valid_time_unix_sec=valid_times_unix_sec[i],
                    raise_error_if_missing=True))

        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=True)

    return narr_file_names_by_predictor, frontal_grid_file_names


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


def downsized_example_generator_from_files(
        input_file_pattern, num_examples_per_batch):
    """Generates downsized input examples for a Keras model.

    This function creates examples by reading them from pre-existing files
    (rather than "on the fly" -- i.e., by reading and processing raw data at
    training time).  If you want to create examples on the fly, use
    `downsized_example_generator_on_the_fly`.

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
        predictor_matrix_to_return = (
            ml_utils.normalize_predictor_matrix(
                predictor_matrix=predictor_matrix_to_return,
                normalize_by_image=True))

        target_values_to_return = keras.utils.to_categorical(
            target_values[batch_indices], NUM_CLASSES_FOR_DOWNSIZED_EXAMPLES)

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_values_to_return)


def downsized_example_generator_on_the_fly(
        num_examples_per_batch, num_examples_per_time, start_time_unix_sec,
        end_time_unix_sec, top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        positive_fraction, num_rows_in_half_grid, num_columns_in_half_grid):
    """Generates downsized input examples for a Keras model.

    This function creates examples on the fly (by reading and processing raw
    data, instead of reading examples from pre-existing files).  If you want to
    read examples from pre-existing files, use
    `downsized_example_generator_from_files`.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.downsized_example_generator(
            training_file_pattern, batch_size),
        ...)

    B = batch size (number of examples per batch)
    M = number of rows (unique grid-point y-coordinates)
      = 2 * num_rows_in_half_grid + 1
    N = number of columns (unique grid-point x-coordinates)
      = 2 * num_columns_in_half_grid + 1
    C = number of channels (predictor variables)

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is just known as "batch_size" in Keras.
    :param num_examples_per_time: Number of examples to draw from each time
        step.
    :param start_time_unix_sec: Start of time period (from which examples will
        be randomly chosen).
    :param end_time_unix_sec: End of time period (from which examples will
        be randomly chosen).
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one file per time step).
    :param narr_predictor_names: length-C list of NARR fields to use as
        predictors.
    :param pressure_level_mb: Pressure level (millibars).
    :param dilation_half_width_for_target: Half-width of dilation window for
        target variable.  For each downsized example, if a front occurs within
        `dilation_half_width_for_target` grid cells of the middle, the label
        will be positive (= 1 = yes).
    :param positive_fraction: Fraction of examples with positive labels.
    :param num_rows_in_half_grid: Number of rows in half-grid for each downsized
        example.
    :param num_columns_in_half_grid: Number of columns in half-grid for each
        downsized example.
    :return: predictor_matrix: B-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-B numpy array of corresponding targets
        (labels).
    """

    _check_input_args_for_otf_generator(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_time=num_examples_per_time,
        narr_predictor_names=narr_predictor_names,
        dilation_half_width_for_target=dilation_half_width_for_target,
        num_rows_in_half_grid=num_rows_in_half_grid,
        num_columns_in_half_grid=num_columns_in_half_grid)

    narr_file_names_by_predictor, frontal_grid_file_names = (
        _find_files_for_otf_generator(
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    num_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    time_index = 0
    num_examples_in_memory = 0
    downsized_predictor_matrix = None
    target_values = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            tuple_of_full_predictor_matrices = ()

            for j in range(num_predictors):
                this_field_full_predictor_matrix, _, _, _ = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_names_by_predictor[j][time_index]))

                tuple_of_full_predictor_matrices += (
                    this_field_full_predictor_matrix,)

            time_index += 1
            if time_index >= num_times:
                time_index = 0

            this_full_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_full_predictor_matrices)
            this_full_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_full_predictor_matrix,
                normalize_by_image=True)

            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[time_index])
            if this_frontal_grid_table.empty:
                continue

            this_frontal_grid_matrix = ml_utils.front_table_to_matrices(
                frontal_grid_table=this_frontal_grid_table,
                num_grid_rows=this_full_predictor_matrix.shape[1],
                num_grid_columns=this_full_predictor_matrix.shape[2])

            this_frontal_grid_matrix = ml_utils.binarize_front_labels(
                this_frontal_grid_matrix)

            this_full_predictor_matrix = (
                ml_utils.remove_nans_from_narr_grid(
                    this_full_predictor_matrix))
            this_frontal_grid_matrix = ml_utils.remove_nans_from_narr_grid(
                this_frontal_grid_matrix)

            this_frontal_grid_matrix = ml_utils.dilate_target_grids(
                binary_target_matrix=this_frontal_grid_matrix,
                num_grid_cells_in_half_window=
                dilation_half_width_for_target)

            this_sampled_target_point_dict = ml_utils.sample_target_points(
                binary_target_matrix=this_frontal_grid_matrix,
                positive_fraction=positive_fraction,
                num_points_per_time=num_examples_per_time)
            if this_sampled_target_point_dict is None:
                continue

            (this_downsized_predictor_matrix, these_target_values,
             _, _, _) = ml_utils.downsize_grids_around_selected_points(
                 predictor_matrix=this_full_predictor_matrix,
                 target_matrix=this_frontal_grid_matrix,
                 num_rows_in_half_window=num_rows_in_half_grid,
                 num_columns_in_half_window=num_columns_in_half_grid,
                 target_point_dict=this_sampled_target_point_dict)

            downsized_predictor_matrix = numpy.concatenate(
                (downsized_predictor_matrix, this_downsized_predictor_matrix),
                axis=0)
            target_values = numpy.concatenate(
                (target_values, these_target_values))
            num_examples_in_memory = len(target_values)

        example_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)

        numpy.random.shuffle(example_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            example_indices, ...]
        target_values = target_values[example_indices]

        predictor_matrix_to_return = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values_to_return = keras.utils.to_categorical(
            target_values[batch_indices], NUM_CLASSES_FOR_DOWNSIZED_EXAMPLES)

        downsized_predictor_matrix = numpy.delete(
            downsized_predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_values_to_return)
