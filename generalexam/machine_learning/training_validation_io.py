"""IO methods for training or validation of a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

K = number of classes (possible values of target label)
E = number of examples.  Each example is one image or a time sequence of images.
M = number of pixel rows in each image
N = number of pixel columns in each image
T = number of predictor times per example (images per sequence)
C = number of channels (predictor variables) in each image

--- DEFINITIONS ---

A "downsized" example covers only a portion of the NARR grid (as opposed to
a full-size example, which covers the entire NARR grid).

For a 3-D example, the dimensions are M x N x C (M rows, N columns, C predictor
variables).

For a 4-D example, the dimensions are M x N x T x C (M rows, N columns, T time
steps, C predictor variables).
"""

import copy
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

NUM_CLASSES_IN_DOWNSIZED_FILES = 2

HOURS_TO_SECONDS = 3600
NARR_TIME_INTERVAL_SECONDS = HOURS_TO_SECONDS * nwp_model_utils.get_time_steps(
    nwp_model_utils.NARR_MODEL_NAME)[1]


def find_input_files_for_3d_examples(
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb):
    """Finds input files for 3-D machine-learning examples.

    Q = number of target times
    C = number of channels (predictor variables) in each image

    :param first_target_time_unix_sec: First target time.  Files will be
        returned for all target times from `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one file per time step).
    :param narr_predictor_names: 1-D list of NARR fields to use as predictors.
    :param pressure_level_mb: Pressure level (millibars).
    :return: narr_file_name_matrix: Q-by-C list of paths to NARR files, each
        containing the grid for one predictor field at one time step.
    :return: frontal_grid_file_names: length-Q list of paths to frontal-grid
        files, each containing a list of NARR grid points intersected by a front
        at one time step.
    """

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(target_times_unix_sec)

    num_target_times = len(target_times_unix_sec)
    num_predictors = len(narr_predictor_names)
    frontal_grid_file_names = [''] * num_target_times
    narr_file_name_matrix = numpy.full(
        (num_target_times, num_predictors), '', dtype=numpy.object)

    for i in range(num_target_times):
        for j in range(num_predictors):
            narr_file_name_matrix[i, j] = (
                processed_narr_io.find_file_for_one_time(
                    top_directory_name=top_narr_directory_name,
                    field_name=narr_predictor_names[j],
                    pressure_level_mb=pressure_level_mb,
                    valid_time_unix_sec=target_times_unix_sec[i],
                    raise_error_if_missing=True))

        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=True)

    return narr_file_name_matrix, frontal_grid_file_names


def find_input_files_for_4d_examples(
        first_target_time_unix_sec, last_target_time_unix_sec,
        num_predictor_time_steps, num_lead_time_steps, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb):
    """Finds input files for 4-D machine-learning examples.

    Q = number of target times
    T = num_predictor_time_steps
    C = number of channels (predictor variables) in each image

    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_3d_examples`.
    :param last_target_time_unix_sec: Same.
    :param num_predictor_time_steps: Number of predictor times per example
        (images per sequence).  This is T in the general discussion above.
    :param num_lead_time_steps: Number of time steps separating latest
        predictor time from target time.
    :param top_narr_directory_name: See documentation for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :return: narr_file_name_matrix: Q-by-T-by-C list of paths to NARR files,
        each containing the grid for one predictor field at one time step.
    :return: frontal_grid_file_names: length-Q list of paths to frontal-grid
        files, each containing a list of NARR grid points intersected by a front
        at one time step.
    """

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(target_times_unix_sec)

    num_target_times = len(target_times_unix_sec)
    num_predictors = len(narr_predictor_names)
    frontal_grid_file_names = [''] * num_target_times
    narr_file_name_matrix = numpy.full(
        (num_target_times, num_predictor_time_steps, num_predictors), '',
        dtype=numpy.object)

    for i in range(num_target_times):
        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=True)

        this_last_narr_time_unix_sec = target_times_unix_sec[i] - (
            num_lead_time_steps * NARR_TIME_INTERVAL_SECONDS)
        this_first_narr_time_unix_sec = this_last_narr_time_unix_sec - (
            (num_predictor_time_steps - 1) * NARR_TIME_INTERVAL_SECONDS)
        these_narr_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=this_first_narr_time_unix_sec,
            end_time_unix_sec=this_last_narr_time_unix_sec,
            time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

        for j in range(num_predictor_time_steps):
            for k in range(num_predictors):
                narr_file_name_matrix[i, j, k] = (
                    processed_narr_io.find_file_for_one_time(
                        top_directory_name=top_narr_directory_name,
                        field_name=narr_predictor_names[k],
                        pressure_level_mb=pressure_level_mb,
                        valid_time_unix_sec=these_narr_times_unix_sec[j],
                        raise_error_if_missing=True))

    return narr_file_name_matrix, frontal_grid_file_names


def write_downsized_examples_to_file(
        predictor_matrix, target_values, target_times_unix_sec,
        center_grid_rows, center_grid_columns, predictor_names,
        pickle_file_name, predictor_time_matrix_unix_sec=None):
    """Writes downsized machine-learning examples to file.

    Downsized ML examples are created by
    `machine_learning_utils.downsize_grids_around_selected_points`.

    :param predictor_matrix: See documentation for
        `machine_learning_utils.check_downsized_examples`.
    :param target_values: Same.
    :param target_times_unix_sec: Same.
    :param center_grid_rows: Same.
    :param center_grid_columns: Same.
    :param predictor_names: Same.
    :param pickle_file_name: Path to output file.
    :param predictor_time_matrix_unix_sec: See documentation for
        `machine_learning_utils.check_downsized_examples`.
    """

    ml_utils.check_downsized_examples(
        predictor_matrix=predictor_matrix, target_values=target_values,
        target_times_unix_sec=target_times_unix_sec,
        center_grid_rows=center_grid_rows,
        center_grid_columns=center_grid_columns,
        predictor_names=predictor_names,
        predictor_time_matrix_unix_sec=predictor_time_matrix_unix_sec,
        assert_binary_target_matrix=False)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(predictor_matrix, pickle_file_handle)
    pickle.dump(target_values, pickle_file_handle)
    pickle.dump(target_times_unix_sec, pickle_file_handle)
    pickle.dump(center_grid_rows, pickle_file_handle)
    pickle.dump(center_grid_columns, pickle_file_handle)
    pickle.dump(predictor_names, pickle_file_handle)
    pickle.dump(predictor_time_matrix_unix_sec, pickle_file_handle)
    pickle_file_handle.close()


def read_downsized_examples_from_file(pickle_file_name):
    """Reads downsized machine-learning examples from file.

    Downsized ML examples are created by
    `machine_learning_utils.downsize_grids_around_selected_points`.

    :param pickle_file_name: Path to input file.
    :return: predictor_matrix: See documentation for
        `machine_learning_utils.check_downsized_examples`.
    :return: target_values: Same.
    :return: target_times_unix_sec: Same.
    :return: center_grid_rows: Same.
    :return: center_grid_columns: Same.
    :return: predictor_names: Same.
    :return: predictor_time_matrix_unix_sec: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    predictor_matrix = pickle.load(pickle_file_handle)
    target_values = pickle.load(pickle_file_handle)
    target_times_unix_sec = pickle.load(pickle_file_handle)
    center_grid_rows = pickle.load(pickle_file_handle)
    center_grid_columns = pickle.load(pickle_file_handle)
    predictor_names = pickle.load(pickle_file_handle)
    predictor_time_matrix_unix_sec = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    ml_utils.check_downsized_examples(
        predictor_matrix=predictor_matrix, target_values=target_values,
        target_times_unix_sec=target_times_unix_sec,
        center_grid_rows=center_grid_rows,
        center_grid_columns=center_grid_columns,
        predictor_names=predictor_names,
        predictor_time_matrix_unix_sec=predictor_time_matrix_unix_sec,
        assert_binary_target_matrix=False)

    return (predictor_matrix, target_values, target_times_unix_sec,
            center_grid_rows, center_grid_columns, predictor_names,
            predictor_time_matrix_unix_sec)


def find_downsized_example_file(
        top_directory_name, target_time_unix_sec, pressure_level_mb,
        raise_error_if_missing=True):
    """Finds file with downsized machine-learning examples.

    Downsized ML examples are created by
    `machine_learning_utils.downsize_grids_around_selected_points`.

    :param top_directory_name: Name of top-level directory with downsized ML
        examples.
    :param target_time_unix_sec: Target time.
    :param pressure_level_mb: Pressure level (millibars).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: downsized_example_file_name: Path to file.
    """

    month_string = time_conversion.unix_sec_to_string(
        target_time_unix_sec, TIME_FORMAT_MONTH)
    time_string = time_conversion.unix_sec_to_string(
        target_time_unix_sec, TIME_FORMAT_IN_FILE_NAME)

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


def downsized_3d_example_generator_from_files(
        input_file_pattern, num_examples_per_batch):
    """Generates downsized 3-D examples for a Keras model.

    This function reads examples from pre-existing files, rather than creating
    them on the fly.  If you want to create examples on the fly, use
    `downsized_3d_example_generator`.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.downsized_3d_example_generator_from_files(
            training_file_pattern, batch_size),
        ...)

    E = num_examples_per_batch

    :param input_file_pattern: Glob pattern for input files (example:
        "ml_examples/downsized/201712/*.p").  All files matching this pattern
        will be used.
    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :return: target_matrix: E-by-2 numpy array of Boolean labels (all 0 or 1,
        although technically the type is "float64").
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
        print '\n'

        while num_examples_in_memory < num_examples_per_batch:
            print 'Reading data from: "{0:s}"...'.format(
                input_file_names[file_index])

            if target_values is None or len(target_values) == 0:
                predictor_matrix, target_values, _, _, _, _, _ = (
                    read_downsized_examples_from_file(
                        input_file_names[file_index]))
            else:
                this_predictor_matrix, these_target_values, _, _, _, _, _ = (
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

        print 'Fraction of positive examples = {0:.4f}'.format(
            numpy.mean(target_values[batch_indices].astype('float')))
        target_matrix_to_return = keras.utils.to_categorical(
            target_values[batch_indices], NUM_CLASSES_IN_DOWNSIZED_FILES)

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_matrix_to_return)


def downsized_3d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid):
    """Generates downsized 3-D examples for a Keras model.

    This function creates examples on the fly, rather than reading them from
    pre-existing files.  If you want to read examples from pre-existing files,
    use `downsized_3d_example_generator_from_files`.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.downsized_3d_example_generator(
            num_examples_per_batch, num_examples_per_target_time, ...),
        ...)

    E = num_examples_per_batch
    M = number of pixel rows = 2 * num_rows_in_half_grid + 1
    N = number of pixel columns = 2 * num_columns_in_half_grid + 1
    C = number of channels (predictor variables) in each image

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_examples_per_target_time: Number of downsized examples to create
        for each target time.
    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_3d_examples`.
    :param last_target_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: Dilation distance for target
        variable.  If a front occurs within
        `dilation_distance_for_target_metres` of grid cell [j, k] at time t, the
        label at [t, j, k] will be positive.
    :param class_fractions: 1-D numpy array with fraction of each class in
        batches generated by this function.  If you want 2 classes, the array
        should be (no_front_fraction, front_fraction).  If you want 3 classes,
        make it (no_front_fraction, warm_front_fraction, cold_front_fraction).
    :param num_rows_in_half_grid: See general discussion above.
    :param num_columns_in_half_grid: See general discussion above.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :return: target_matrix: E-by-K numpy array of Boolean labels (all 0 or 1,
        although technically the type is "float64").
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        find_input_files_for_3d_examples(
            first_target_time_unix_sec=first_target_time_unix_sec,
            last_target_time_unix_sec=last_target_time_unix_sec,
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
            print '\n'
            tuple_of_full_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[time_index, j])
                this_field_full_predictor_matrix, _, _, _ = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[time_index, j]))

                tuple_of_full_predictor_matrices += (
                    this_field_full_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[time_index])

            time_index += 1
            if time_index >= num_times:
                time_index = 0
            if this_frontal_grid_table.empty:
                continue

            print 'Creating downsized 3-D machine-learning examples...'

            this_full_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_full_predictor_matrices)
            this_full_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_full_predictor_matrix,
                normalize_by_example=True)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_full_predictor_matrix = (
                ml_utils.remove_nans_from_narr_grid(
                    this_full_predictor_matrix))
            this_frontal_grid_matrix = ml_utils.remove_nans_from_narr_grid(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            this_sampled_target_point_dict = ml_utils.sample_target_points(
                target_matrix=this_frontal_grid_matrix,
                class_fractions=class_fractions,
                num_points_per_time=num_examples_per_target_time)
            if this_sampled_target_point_dict is None:
                continue

            if target_values is None or len(target_values) == 0:
                downsized_predictor_matrix, target_values, _, _, _ = (
                    ml_utils.downsize_grids_around_selected_points(
                        predictor_matrix=this_full_predictor_matrix,
                        target_matrix=this_frontal_grid_matrix,
                        num_rows_in_half_window=num_rows_in_half_grid,
                        num_columns_in_half_window=num_columns_in_half_grid,
                        target_point_dict=this_sampled_target_point_dict,
                        verbose=False))
            else:
                (this_downsized_predictor_matrix, these_target_values,
                 _, _, _) = ml_utils.downsize_grids_around_selected_points(
                     predictor_matrix=this_full_predictor_matrix,
                     target_matrix=this_frontal_grid_matrix,
                     num_rows_in_half_window=num_rows_in_half_grid,
                     num_columns_in_half_window=num_columns_in_half_grid,
                     target_point_dict=this_sampled_target_point_dict,
                     verbose=False)

                downsized_predictor_matrix = numpy.concatenate(
                    (downsized_predictor_matrix,
                     this_downsized_predictor_matrix), axis=0)
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

        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_values[batch_indices] > 0))
        target_matrix_to_return = keras.utils.to_categorical(
            target_values[batch_indices], num_classes)

        downsized_predictor_matrix = numpy.delete(
            downsized_predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_matrix_to_return)


def downsized_4d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        num_predictor_time_steps, num_lead_time_steps,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid):
    """Generates downsized 4-D examples for a Keras model.

    This function creates examples on the fly, rather than reading them from
    pre-existing files.  As yet, this function has no counterpart that reads
    from files.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.downsized_4d_example_generator(
            num_examples_per_batch, num_examples_per_target_time, ...),
        ...)

    E = num_examples_per_batch
    M = number of pixel rows = 2 * num_rows_in_half_grid + 1
    N = number of pixel columns = 2 * num_columns_in_half_grid + 1

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param num_examples_per_target_time: Number of downsized examples to create
        for each target time.
    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_4d_examples`.
    :param last_target_time_unix_sec: Same.
    :param num_predictor_time_steps: Same.
    :param num_lead_time_steps: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: See documentation for
        `downsized_3d_example_generator`.
    :param class_fractions: Same.
    :param num_rows_in_half_grid: See general discussion above.
    :param num_columns_in_half_grid: See general discussion above.
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        images.
    :return: target_matrix: E-by-K numpy array of Boolean labels (all 0 or 1,
        although technically the type is "float64").
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        find_input_files_for_4d_examples(
            first_target_time_unix_sec=first_target_time_unix_sec,
            last_target_time_unix_sec=last_target_time_unix_sec,
            num_predictor_time_steps=num_predictor_time_steps,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    num_target_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_examples_in_memory = 0
    downsized_predictor_matrix = None
    target_values = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            for i in range(num_predictor_time_steps):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])
                    this_field_predictor_matrix, _, _, _ = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j]))

                    tuple_of_3d_predictor_matrices += (
                        this_field_predictor_matrix,)

                tuple_of_4d_predictor_matrices += (
                    ml_utils.stack_predictor_variables(
                        tuple_of_3d_predictor_matrices),)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            if this_frontal_grid_table.empty:
                continue

            print 'Creating downsized 4-D machine-learning examples...'

            this_full_predictor_matrix = ml_utils.stack_time_steps(
                tuple_of_4d_predictor_matrices)
            this_full_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_full_predictor_matrix,
                normalize_by_example=True)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_full_predictor_matrix = ml_utils.remove_nans_from_narr_grid(
                this_full_predictor_matrix)
            this_frontal_grid_matrix = ml_utils.remove_nans_from_narr_grid(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            this_sampled_target_point_dict = ml_utils.sample_target_points(
                target_matrix=this_frontal_grid_matrix,
                class_fractions=class_fractions,
                num_points_per_time=num_examples_per_target_time)
            if this_sampled_target_point_dict is None:
                continue

            if target_values is None or len(target_values) == 0:
                downsized_predictor_matrix, target_values, _, _, _ = (
                    ml_utils.downsize_grids_around_selected_points(
                        predictor_matrix=this_full_predictor_matrix,
                        target_matrix=this_frontal_grid_matrix,
                        num_rows_in_half_window=num_rows_in_half_grid,
                        num_columns_in_half_window=num_columns_in_half_grid,
                        target_point_dict=this_sampled_target_point_dict,
                        verbose=False))
            else:
                (this_downsized_predictor_matrix, these_target_values,
                 _, _, _) = ml_utils.downsize_grids_around_selected_points(
                     predictor_matrix=this_full_predictor_matrix,
                     target_matrix=this_frontal_grid_matrix,
                     num_rows_in_half_window=num_rows_in_half_grid,
                     num_columns_in_half_window=num_columns_in_half_grid,
                     target_point_dict=this_sampled_target_point_dict,
                     verbose=False)

                downsized_predictor_matrix = numpy.concatenate(
                    (downsized_predictor_matrix,
                     this_downsized_predictor_matrix), axis=0)
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

        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_values[batch_indices] > 0))
        target_matrix_to_return = keras.utils.to_categorical(
            target_values[batch_indices], num_classes)

        downsized_predictor_matrix = numpy.delete(
            downsized_predictor_matrix, batch_indices, axis=0)
        target_values = numpy.delete(target_values, batch_indices, axis=0)
        num_examples_in_memory = len(target_values)

        yield (predictor_matrix_to_return, target_matrix_to_return)


def full_size_3d_example_generator(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_classes):
    """Generates full-size 3-D examples for a Keras model.

    This function creates examples on the fly, rather than reading them from
    pre-existing files.  As yet, this function has no counterpart that reads
    from files.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.full_size_3d_example_generator(
            num_examples_per_batch, first_target_time_unix_sec, ...),
        ...)

    E = num_examples_per_batch

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_3d_examples`.
    :param last_target_time_unix_sec: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: See documentation for
        `downsized_3d_example_generator`.
    :param num_classes: Number of classes (possible target values).
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :return: target_matrix: E-by-M-by-N numpy array of target images, where
        target_matrix[i, j, k] is the integer label for the [i]th time step,
        [j]th pixel row, and [k]th pixel column.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        find_input_files_for_3d_examples(
            first_target_time_unix_sec=first_target_time_unix_sec,
            last_target_time_unix_sec=last_target_time_unix_sec,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    num_target_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_examples_in_memory = 0
    predictor_matrix = None
    target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[target_time_index, j])
                this_field_predictor_matrix, _, _, _ = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[target_time_index, j]))

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            print 'Processing full-size machine-learning example...'

            this_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_predictor_matrix,
                normalize_by_example=True)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_predictor_matrix.shape[1],
                num_columns_per_image=this_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_predictor_matrix)
            this_frontal_grid_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            if target_matrix is None or target_matrix.size == 0:
                predictor_matrix = copy.deepcopy(this_predictor_matrix)
                target_matrix = copy.deepcopy(this_frontal_grid_matrix)
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0)
                target_matrix = numpy.concatenate(
                    (target_matrix, this_frontal_grid_matrix), axis=0)

            num_examples_in_memory = target_matrix.shape[0]

        predictor_matrix_to_return = predictor_matrix[
            batch_indices, ...].astype('float32')
        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_matrix[batch_indices, ...] > 0))

        target_matrix_to_return = keras.utils.to_categorical(
            target_matrix[batch_indices, ...], num_classes)
        target_matrix_to_return = numpy.reshape(
            target_matrix_to_return, target_matrix.shape + (num_classes,))

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_matrix = numpy.delete(target_matrix, batch_indices, axis=0)
        num_examples_in_memory = target_matrix.shape[0]

        yield (predictor_matrix_to_return, target_matrix_to_return)


def full_size_4d_example_generator(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, num_predictor_time_steps,
        num_lead_time_steps, top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_classes):
    """Generates full-size 4-D examples for a Keras model.

    This function creates examples on the fly, rather than reading them from
    pre-existing files.  As yet, this function has no counterpart that reads
    from files.

    This function fits the template specified by `keras.models.*.fit_generator`.
    Thus, when training a Keras model with the `fit_generator` method, the input
    argument "generator" should be this function.  For example:

    model_object.fit_generator(
        generator=machine_learning_io.full_size_4d_example_generator(
            num_examples_per_batch, first_target_time_unix_sec, ...),
        ...)

    E = num_examples_per_batch
    T = num_predictor_time_steps

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_4d_examples`.
    :param last_target_time_unix_sec: Same.
    :param num_predictor_time_steps: Same.
    :param num_lead_time_steps: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: See documentation for
        `downsized_3d_example_generator`.
    :param num_classes: Number of classes (possible target values).
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        images.
    :return: target_matrix: E-by-M-by-N numpy array of target images, where
        target_matrix[i, j, k] is the integer label for the [i]th time step,
        [j]th pixel row, and [k]th pixel column.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        find_input_files_for_4d_examples(
            first_target_time_unix_sec=first_target_time_unix_sec,
            last_target_time_unix_sec=last_target_time_unix_sec,
            num_predictor_time_steps=num_predictor_time_steps,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    num_target_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_examples_in_memory = 0
    predictor_matrix = None
    target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            for i in range(num_predictor_time_steps):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])
                    this_field_predictor_matrix, _, _, _ = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j]))

                    tuple_of_3d_predictor_matrices += (
                        this_field_predictor_matrix,)

                tuple_of_4d_predictor_matrices += (
                    ml_utils.stack_predictor_variables(
                        tuple_of_3d_predictor_matrices),)

            this_predictor_matrix = ml_utils.stack_time_steps(
                tuple_of_4d_predictor_matrices)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            print 'Processing full-size 4-D machine-learning example...'

            this_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_predictor_matrix,
                normalize_by_example=True)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_predictor_matrix.shape[1],
                num_columns_per_image=this_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_predictor_matrix)
            this_frontal_grid_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            if target_matrix is None or target_matrix.size == 0:
                predictor_matrix = copy.deepcopy(this_predictor_matrix)
                target_matrix = copy.deepcopy(this_frontal_grid_matrix)
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0)
                target_matrix = numpy.concatenate(
                    (target_matrix, this_frontal_grid_matrix), axis=0)

            num_examples_in_memory = target_matrix.shape[0]

        predictor_matrix_to_return = predictor_matrix[
            batch_indices, ...].astype('float32')
        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_matrix[batch_indices, ...] > 0))

        target_matrix_to_return = keras.utils.to_categorical(
            target_matrix[batch_indices, ...], num_classes)
        target_matrix_to_return = numpy.reshape(
            target_matrix_to_return, target_matrix.shape + (num_classes,))

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_matrix = numpy.delete(target_matrix, batch_indices, axis=0)
        num_examples_in_memory = target_matrix.shape[0]

        yield (predictor_matrix_to_return, target_matrix_to_return)
