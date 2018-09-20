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
import numpy
import keras
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAME = '%Y-%m-%d-%H'

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

    print first_target_time_unix_sec
    print last_target_time_unix_sec
    print NARR_TIME_INTERVAL_SECONDS

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
        predictor_time_step_offsets, num_lead_time_steps,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb):
    """Finds input files for 4-D machine-learning examples.

    Q = number of target times
    T = number of predictor times per example
    C = number of channels (predictor variables) in each image

    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_3d_examples`.
    :param last_target_time_unix_sec: Same.
    :param predictor_time_step_offsets: length-T numpy array of offsets between
        predictor times and (target time - lead time).
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

    error_checking.assert_is_integer_numpy_array(predictor_time_step_offsets)
    error_checking.assert_is_numpy_array(
        predictor_time_step_offsets, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(predictor_time_step_offsets, 0)

    predictor_time_step_offsets = numpy.unique(
        predictor_time_step_offsets)[::-1]

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(target_times_unix_sec)

    num_target_times = len(target_times_unix_sec)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)
    frontal_grid_file_names = [''] * num_target_times
    narr_file_name_matrix = numpy.full(
        (num_target_times, num_predictor_times_per_example, num_predictors), '',
        dtype=numpy.object)

    for i in range(num_target_times):
        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=True)

        this_last_time_unix_sec = target_times_unix_sec[i] - (
            num_lead_time_steps * NARR_TIME_INTERVAL_SECONDS)
        these_narr_times_unix_sec = this_last_time_unix_sec - (
            predictor_time_step_offsets * NARR_TIME_INTERVAL_SECONDS)

        for j in range(num_predictor_times_per_example):
            for k in range(num_predictors):
                narr_file_name_matrix[i, j, k] = (
                    processed_narr_io.find_file_for_one_time(
                        top_directory_name=top_narr_directory_name,
                        field_name=narr_predictor_names[k],
                        pressure_level_mb=pressure_level_mb,
                        valid_time_unix_sec=these_narr_times_unix_sec[j],
                        raise_error_if_missing=True))

    return narr_file_name_matrix, frontal_grid_file_names


def downsized_3d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid, narr_mask_matrix=None):
    """Generates downsized 3-D examples for a Keras model.

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
    :param narr_mask_matrix: See doc for
        `machine_learning_utils.check_narr_mask`.  If
        narr_mask_matrix[i, j] = 0, cell [i, j] in the full grid will never be
        used for downsizing -- i.e., will never be used as the center of a
        downsized grid.  If `narr_mask_matrix is None`, any cell in the full
        grid can be used as the center of a downsized grid.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :return: target_matrix: E-by-K numpy array of Boolean labels (all 0 or 1,
        although technically the type is "float64").
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_target_time)
    error_checking.assert_is_geq(num_examples_per_target_time, 2)
    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    time_index = 0
    num_times_in_memory = 0
    num_times_needed_in_memory = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_target_time))
    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_times_in_memory < num_times_needed_in_memory:
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[time_index, j])

                this_field_predictor_matrix, _, _, _ = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[time_index, j]))
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix))

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[time_index])

            time_index += 1
            if time_index >= num_times:
                time_index = 0

            this_full_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_full_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_full_predictor_matrix,
                normalize_by_example=True)

            this_full_target_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_full_target_matrix = ml_utils.binarize_front_images(
                    this_full_target_matrix)

            if num_classes == 2:
                this_full_target_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_full_target_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_full_target_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_full_target_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = copy.deepcopy(
                    this_full_predictor_matrix)
                full_target_matrix = copy.deepcopy(this_full_target_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_full_predictor_matrix), axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_full_target_matrix), axis=0)

            num_times_in_memory = full_target_matrix.shape[0]

        print 'Creating downsized 3-D machine-learning examples...'
        sampled_target_point_dict = ml_utils.sample_target_points(
            target_matrix=full_target_matrix, class_fractions=class_fractions,
            num_points_to_sample=num_examples_per_batch,
            mask_matrix=narr_mask_matrix)

        downsized_predictor_matrix, target_values, _, _, _ = (
            ml_utils.downsize_grids_around_selected_points(
                predictor_matrix=full_predictor_matrix,
                target_matrix=full_target_matrix,
                num_rows_in_half_window=num_rows_in_half_grid,
                num_columns_in_half_window=num_columns_in_half_grid,
                target_point_dict=sampled_target_point_dict,
                verbose=False))

        numpy.random.shuffle(batch_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = target_values[batch_indices]

        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_values > 0))
        target_matrix = keras.utils.to_categorical(target_values, num_classes)

        full_predictor_matrix = None
        full_target_matrix = None
        num_times_in_memory = 0

        yield (downsized_predictor_matrix, target_matrix)


def downsized_4d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        predictor_time_step_offsets, num_lead_time_steps,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid, narr_mask_matrix=None):
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
    :param predictor_time_step_offsets: length-T numpy array of offsets between
        predictor times and (target time - lead time).
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
    :param mask_matrix: See doc for `downsized_3d_example_generator`.
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        images.
    :return: target_matrix: E-by-K numpy array of Boolean labels (all 0 or 1,
        although technically the type is "float64").
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_target_time)
    error_checking.assert_is_geq(num_examples_per_target_time, 2)
    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_4d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        predictor_time_step_offsets=predictor_time_step_offsets,
        num_lead_time_steps=num_lead_time_steps,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_target_times = len(frontal_grid_file_names)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_times_in_memory = 0
    num_times_needed_in_memory = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_target_time))
    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_times_in_memory < num_times_needed_in_memory:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            for i in range(num_predictor_times_per_example):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])

                    this_field_predictor_matrix, _, _, _ = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j]))
                    this_field_predictor_matrix = (
                        ml_utils.fill_nans_in_predictor_images(
                            this_field_predictor_matrix))

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

            this_full_predictor_matrix = ml_utils.stack_time_steps(
                tuple_of_4d_predictor_matrices)
            this_full_predictor_matrix = ml_utils.normalize_predictor_matrix(
                predictor_matrix=this_full_predictor_matrix,
                normalize_by_example=True)

            this_full_target_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_full_target_matrix = ml_utils.binarize_front_images(
                    this_full_target_matrix)

            if num_classes == 2:
                this_full_target_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_full_target_matrix,
                    dilation_distance_metres=
                    dilation_distance_for_target_metres, verbose=False)
            else:
                this_full_target_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_full_target_matrix,
                        dilation_distance_metres=
                        dilation_distance_for_target_metres, verbose=False))

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = copy.deepcopy(
                    this_full_predictor_matrix)
                full_target_matrix = copy.deepcopy(this_full_target_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_full_predictor_matrix), axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_full_target_matrix), axis=0)

            num_times_in_memory = full_target_matrix.shape[0]

        sampled_target_point_dict = ml_utils.sample_target_points(
            target_matrix=full_target_matrix, class_fractions=class_fractions,
            num_points_to_sample=num_examples_per_batch,
            mask_matrix=narr_mask_matrix)

        downsized_predictor_matrix, target_values, _, _, _ = (
            ml_utils.downsize_grids_around_selected_points(
                predictor_matrix=full_predictor_matrix,
                target_matrix=full_target_matrix,
                num_rows_in_half_window=num_rows_in_half_grid,
                num_columns_in_half_window=num_columns_in_half_grid,
                target_point_dict=sampled_target_point_dict,
                verbose=False))

        numpy.random.shuffle(batch_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = target_values[batch_indices]

        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_values > 0))
        target_matrix = keras.utils.to_categorical(target_values, num_classes)

        full_predictor_matrix = None
        full_target_matrix = None
        num_times_in_memory = 0

        yield (downsized_predictor_matrix, target_matrix)


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
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix))

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
        last_target_time_unix_sec, predictor_time_step_offsets,
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
    T = number of predictor times per example

    :param num_examples_per_batch: Number of examples per batch.  This argument
        is known as "batch_size" in Keras.
    :param first_target_time_unix_sec: See documentation for
        `find_input_files_for_4d_examples`.
    :param last_target_time_unix_sec: Same.
    :param predictor_time_step_offsets: Same.
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
            predictor_time_step_offsets=predictor_time_step_offsets,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    num_target_times = len(frontal_grid_file_names)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
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

            for i in range(num_predictor_times_per_example):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])

                    this_field_predictor_matrix, _, _, _ = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j]))
                    this_field_predictor_matrix = (
                        ml_utils.fill_nans_in_predictor_images(
                            this_field_predictor_matrix))

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
