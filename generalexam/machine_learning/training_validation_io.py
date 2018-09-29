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
import os.path
import numpy
import keras
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

HOURS_TO_SECONDS = 3600
NARR_TIME_INTERVAL_SECONDS = HOURS_TO_SECONDS * nwp_model_utils.get_time_steps(
    nwp_model_utils.NARR_MODEL_NAME)[1]

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
TARGET_TIMES_KEY = 'target_times_unix_sec'
ROW_INDICES_KEY = 'row_indices'
COLUMN_INDICES_KEY = 'column_indices'
PREDICTOR_NAMES_KEY = 'narr_predictor_names'
NARR_MASK_KEY = 'narr_mask_matrix'

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
DILATION_DISTANCE_KEY = 'dilation_distance_metres'
NARR_ROW_DIMENSION_KEY = 'narr_row'
NARR_COLUMN_DIMENSION_KEY = 'narr_column'
EXAMPLE_DIMENSION_KEY = 'example'
EXAMPLE_ROW_DIMENSION_KEY = 'example_row'
EXAMPLE_COLUMN_DIMENSION_KEY = 'example_column'
PREDICTOR_DIMENSION_KEY = 'predictor_variable'
CHARACTER_DIMENSION_KEY = 'predictor_variable_char'
CLASS_DIMENSION_KEY = 'class'


def _decrease_example_size(predictor_matrix, num_half_rows, num_half_columns):
    """Decreases the grid size for each example.

    M = original number of grid rows per example
    N = original number of grid columns per example
    m = new number of rows per example
    n = new number of columns per example

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :param num_half_rows: Determines number of rows returned for each example.
        Examples will be cropped so that the center of the original image is the
        center of the new image.  If `num_half_rows`, examples will not be
        cropped.
    :param num_half_columns: Same but for columns.
    :return: predictor_matrix: E-by-m-by-n-by-C numpy array of predictor images.
    """

    if num_half_rows is not None:
        error_checking.assert_is_integer(num_half_rows)
        error_checking.assert_is_greater(num_half_rows, 0)

        center_row_index = int(
            numpy.floor(float(predictor_matrix.shape[1]) / 2)
        )
        first_row_index = center_row_index - num_half_rows
        last_row_index = center_row_index + num_half_rows
        predictor_matrix = predictor_matrix[
            :, first_row_index:(last_row_index + 1), ...
        ]

    if num_half_columns is not None:
        error_checking.assert_is_integer(num_half_columns)
        error_checking.assert_is_greater(num_half_columns, 0)

        center_column_index = int(
            numpy.floor(float(predictor_matrix.shape[2]) / 2)
        )
        first_column_index = center_column_index - num_half_columns
        last_column_index = center_column_index + num_half_columns
        predictor_matrix = predictor_matrix[
            :, :, first_column_index:(last_column_index + 1), ...
        ]

    return predictor_matrix


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
        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(frontal_grid_file_names[i]):
            frontal_grid_file_names[i] = ''
            continue

        for j in range(num_predictors):
            narr_file_name_matrix[i, j] = (
                processed_narr_io.find_file_for_one_time(
                    top_directory_name=top_narr_directory_name,
                    field_name=narr_predictor_names[j],
                    pressure_level_mb=pressure_level_mb,
                    valid_time_unix_sec=target_times_unix_sec[i],
                    raise_error_if_missing=True))

    keep_time_flags = numpy.array(
        [f != '' for f in frontal_grid_file_names], dtype=bool)
    keep_time_indices = numpy.where(keep_time_flags)[0]

    return (narr_file_name_matrix[keep_time_indices, ...],
            frontal_grid_file_names[keep_time_indices])


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
            raise_error_if_missing=False)

        if not os.path.isfile(frontal_grid_file_names[i]):
            frontal_grid_file_names[i] = ''
            continue

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

    keep_time_flags = numpy.array(
        [f != '' for f in frontal_grid_file_names], dtype=bool)
    keep_time_indices = numpy.where(keep_time_flags)[0]

    return (narr_file_name_matrix[keep_time_indices, ...],
            frontal_grid_file_names[keep_time_indices])


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


def prep_downsized_3d_examples_to_write(
        target_time_unix_sec, max_num_examples, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names,
        pressure_level_mb, dilation_distance_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid, narr_mask_matrix=None):
    """Prepares downsized 3-D examples for writing to a file.

    :param target_time_unix_sec: Target time.
    :param max_num_examples: Maximum number of examples.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param class_fractions: length-3 numpy array of sampling fractions for
        target variable.  These are the fractions of (NF, WF, CF) examples,
        respectively, to be created.
    :param num_rows_in_half_grid: See doc for
        `downsized_3d_example_generator`.
    :param num_columns_in_half_grid: Same.
    :param narr_mask_matrix: Same.
    :return: example_dict: Dictionary with the following keys.
    example_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        images.
    example_dict['target_matrix']: E-by-K numpy array of Boolean labels (all 0
        or 1, but technically the type is "float64").
    example_dict['target_times_unix_sec']: length-E numpy array of target times
        (these will all be the same).
    example_dict['row_indices']: length-E numpy array with NARR row at the
        center of each example.
    example_dict['column_indices']: length-E numpy array with NARR column at the
        center of each example.
    """

    error_checking.assert_is_numpy_array(
        class_fractions, exact_dimensions=numpy.array([3]))
    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=target_time_unix_sec,
        last_target_time_unix_sec=target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    narr_file_names = narr_file_name_matrix[0, ...]
    frontal_grid_file_name = frontal_grid_file_names[0]

    tuple_of_predictor_matrices = ()
    num_predictors = len(narr_predictor_names)

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])

        this_field_matrix = processed_narr_io.read_fields_from_file(
            narr_file_names[j])[0]
        this_field_matrix = ml_utils.fill_nans_in_predictor_images(
            this_field_matrix)
        tuple_of_predictor_matrices += (this_field_matrix,)

    predictor_matrix = ml_utils.stack_predictor_variables(
        tuple_of_predictor_matrices)
    predictor_matrix = ml_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_example=True)

    print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        frontal_grid_file_name)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=frontal_grid_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    target_matrix = ml_utils.dilate_ternary_target_images(
        target_matrix=target_matrix,
        dilation_distance_metres=dilation_distance_metres, verbose=False)

    sampled_target_point_dict = ml_utils.sample_target_points(
        target_matrix=target_matrix, class_fractions=class_fractions,
        num_points_to_sample=max_num_examples, mask_matrix=narr_mask_matrix)

    (predictor_matrix, target_values, _, row_indices, column_indices
    ) = ml_utils.downsize_grids_around_selected_points(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        num_rows_in_half_window=num_rows_in_half_grid,
        num_columns_in_half_window=num_columns_in_half_grid,
        target_point_dict=sampled_target_point_dict, verbose=False)

    target_matrix = keras.utils.to_categorical(target_values, 3)
    actual_class_fractions = numpy.sum(target_matrix, axis=0)
    print 'Fraction of examples in each class: {0:s}'.format(
        str(actual_class_fractions))

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY:
            numpy.full(target_matrix.shape[0], target_time_unix_sec, dtype=int),
        ROW_INDICES_KEY: row_indices,
        COLUMN_INDICES_KEY: column_indices
    }


def find_downsized_3d_example_file(
        directory_name, first_target_time_unix_sec, last_target_time_unix_sec,
        raise_error_if_missing=True):
    """Finds file with downsized 3-D examples.

    :param directory_name: Name of directory.
    :param first_target_time_unix_sec: First target time in file.
    :param last_target_time_unix_sec: Last target time in file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: downsized_3d_file_name: Path to file with downsized 3-D examples.
        If file is missing and `raise_error_if_missing = False`, this is the
        *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(first_target_time_unix_sec)
    error_checking.assert_is_integer(last_target_time_unix_sec)
    error_checking.assert_is_geq(
        last_target_time_unix_sec, first_target_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    downsized_3d_file_name = (
        '{0:s}/downsized_3d_examples_{1:s}-{2:s}.nc'
    ).format(
        directory_name,
        time_conversion.unix_sec_to_string(
            first_target_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            last_target_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    )

    if raise_error_if_missing and not os.path.isfile(downsized_3d_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            downsized_3d_file_name)
        raise ValueError(error_string)

    return downsized_3d_file_name


def write_downsized_3d_examples(
        netcdf_file_name, example_dict, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, narr_mask_matrix=None):
    """Writes downsized 3-D examples to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param example_dict: Dictionary created by
        `prep_downsized_3d_examples_to_write`.
    :param narr_predictor_names: See doc for
        `prep_downsized_3d_examples_to_write`.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param narr_mask_matrix: Same.
    """

    error_checking.assert_is_string_list(narr_predictor_names)
    num_predictors = example_dict[PREDICTOR_MATRIX_KEY].shape[3]
    error_checking.assert_is_numpy_array(
        numpy.array(narr_predictor_names),
        exact_dimensions=numpy.array([num_predictors]))

    error_checking.assert_is_integer(pressure_level_mb)
    error_checking.assert_is_greater(pressure_level_mb, 0)
    error_checking.assert_is_geq(dilation_distance_metres, 0.)

    num_narr_rows, num_narr_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    if narr_mask_matrix is None:
        narr_mask_matrix = numpy.full(
            (num_narr_rows, num_narr_columns), 1, dtype=int)

    ml_utils.check_narr_mask(narr_mask_matrix)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(PRESSURE_LEVEL_KEY, int(pressure_level_mb))
    netcdf_dataset.setncattr(DILATION_DISTANCE_KEY, dilation_distance_metres)

    num_classes = example_dict[TARGET_MATRIX_KEY].shape[1]
    num_examples = example_dict[PREDICTOR_MATRIX_KEY].shape[0]
    num_rows_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[1]
    num_columns_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[2]

    num_predictor_chars = 1
    for m in range(num_predictors):
        num_predictor_chars = max(
            [num_predictor_chars, len(narr_predictor_names[m])]
        )

    netcdf_dataset.createDimension(NARR_ROW_DIMENSION_KEY, num_narr_rows)
    netcdf_dataset.createDimension(NARR_COLUMN_DIMENSION_KEY, num_narr_columns)
    netcdf_dataset.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    netcdf_dataset.createDimension(
        EXAMPLE_ROW_DIMENSION_KEY, num_rows_per_example)
    netcdf_dataset.createDimension(
        EXAMPLE_COLUMN_DIMENSION_KEY, num_columns_per_example)
    netcdf_dataset.createDimension(PREDICTOR_DIMENSION_KEY, num_predictors)
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_predictor_chars)
    netcdf_dataset.createDimension(CLASS_DIMENSION_KEY, num_classes)

    string_type = 'S{0:d}'.format(num_predictor_chars)
    predictor_names_as_char_array = netCDF4.stringtochar(numpy.array(
        narr_predictor_names, dtype=string_type))

    netcdf_dataset.createVariable(
        PREDICTOR_NAMES_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIMENSION_KEY, CHARACTER_DIMENSION_KEY))
    netcdf_dataset.variables[PREDICTOR_NAMES_KEY][:] = numpy.array(
        predictor_names_as_char_array)

    netcdf_dataset.createVariable(
        NARR_MASK_KEY, datatype=numpy.int32,
        dimensions=(NARR_ROW_DIMENSION_KEY, NARR_COLUMN_DIMENSION_KEY))
    netcdf_dataset.variables[NARR_MASK_KEY][:] = narr_mask_matrix

    netcdf_dataset.createVariable(
        PREDICTOR_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ROW_DIMENSION_KEY,
                    EXAMPLE_COLUMN_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    netcdf_dataset.variables[PREDICTOR_MATRIX_KEY][:] = example_dict[
        PREDICTOR_MATRIX_KEY]

    netcdf_dataset.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY))
    netcdf_dataset.variables[TARGET_MATRIX_KEY][:] = example_dict[
        TARGET_MATRIX_KEY]

    netcdf_dataset.createVariable(
        TARGET_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[TARGET_TIMES_KEY][:] = example_dict[
        TARGET_TIMES_KEY]

    netcdf_dataset.createVariable(
        ROW_INDICES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[ROW_INDICES_KEY][:] = example_dict[ROW_INDICES_KEY]

    netcdf_dataset.createVariable(
        COLUMN_INDICES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[COLUMN_INDICES_KEY][:] = example_dict[
        COLUMN_INDICES_KEY]

    netcdf_dataset.close()


def read_downsized_3d_examples(
        netcdf_file_name, predictor_names_to_keep=None,
        num_half_rows_to_keep=None, num_half_columns_to_keep=None,
        first_time_to_keep_unix_sec=None, last_time_to_keep_unix_sec=None):
    """Reads downsized 3-D examples from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param predictor_names_to_keep: 1-D list with names of predictor variables to
        keep (each name must be accepted by `check_field_name`).  If
        `predictor_names_to_keep is None`, all predictors in the file will be
        returned.
    :param num_half_rows_to_keep: Determines number of rows to keep for each
        example.  Examples will be cropped so that the center of the original
        image is the center of the new image.  If
        `num_half_rows_to_keep is None`, examples will not be cropped.
    :param num_half_columns_to_keep: Same but for columns.
    :param first_time_to_keep_unix_sec: Will throw out earlier target times.
    :param last_time_to_keep_unix_sec: Will throw out later target times.
    :return: example_dict: Dictionary with the following keys.
    example_dict['predictor_matrix']: See doc for
        `prep_downsized_3d_examples_to_write`.
    example_dict['target_matrix']: Same.
    example_dict['target_times_unix_sec']: Same.
    example_dict['row_indices']: Same.
    example_dict['column_indices']: Same.
    example_dict['predictor_names_to_keep']: See doc for
        `write_downsized_3d_examples`.
    example_dict['pressure_level_mb']: Same.
    example_dict['dilation_distance_metres']: Same.
    example_dict['narr_mask_matrix']: Same.
    """

    if predictor_names_to_keep is not None:
        error_checking.assert_is_numpy_array(
            numpy.array(predictor_names_to_keep), num_dimensions=1)
        for this_name in predictor_names_to_keep:
            processed_narr_io.check_field_name(this_name)
    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = int(1e11)

    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name)

    narr_predictor_names = netCDF4.chartostring(
        netcdf_dataset.variables[PREDICTOR_NAMES_KEY][:])
    narr_predictor_names = [str(s) for s in narr_predictor_names]
    predictor_matrix = numpy.array(
        netcdf_dataset.variables[PREDICTOR_MATRIX_KEY][:])

    if predictor_names_to_keep is None:
        predictor_names_to_keep = narr_predictor_names + []
    else:
        these_indices = numpy.array(
            [narr_predictor_names.index(p) for p in predictor_names_to_keep],
            dtype=int)
        predictor_matrix = predictor_matrix[..., these_indices]

    predictor_matrix = _decrease_example_size(
        predictor_matrix=predictor_matrix, num_half_rows=num_half_rows_to_keep,
        num_half_columns=num_half_columns_to_keep)

    target_times_unix_sec = numpy.array(
        netcdf_dataset.variables[TARGET_TIMES_KEY][:], dtype=int)
    indices_to_keep = numpy.where(numpy.logical_and(
        target_times_unix_sec >= first_time_to_keep_unix_sec,
        target_times_unix_sec <= last_time_to_keep_unix_sec
    ))[0]

    target_times_unix_sec = target_times_unix_sec[indices_to_keep]
    predictor_matrix = predictor_matrix[indices_to_keep, ...].astype('float32')
    target_matrix = numpy.array(
        netcdf_dataset.variables[TARGET_MATRIX_KEY][indices_to_keep, ...]
    ).astype('float64')
    row_indices = numpy.array(
        netcdf_dataset.variables[ROW_INDICES_KEY][indices_to_keep], dtype=int)
    column_indices = numpy.array(
        netcdf_dataset.variables[COLUMN_INDICES_KEY][indices_to_keep],
        dtype=int)

    example_dict = {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY: target_times_unix_sec,
        ROW_INDICES_KEY: row_indices,
        COLUMN_INDICES_KEY: column_indices,
        PREDICTOR_NAMES_KEY: predictor_names_to_keep,
        PRESSURE_LEVEL_KEY: int(getattr(netcdf_dataset, PRESSURE_LEVEL_KEY)),
        DILATION_DISTANCE_KEY: getattr(netcdf_dataset, DILATION_DISTANCE_KEY),
        NARR_MASK_KEY:
            numpy.array(netcdf_dataset.variables[NARR_MASK_KEY][:], dtype=int)
    }

    netcdf_dataset.close()
    return example_dict
