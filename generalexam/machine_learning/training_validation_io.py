"""IO methods for training and on-the-fly validation (during training)."""

import copy
from random import shuffle
import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

LARGE_INTEGER = int(1e10)
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
TARGET_TIMES_KEY = 'target_times_unix_sec'


def downsized_3d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_gridded_front_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        class_fractions, num_rows_in_half_grid, num_columns_in_half_grid,
        narr_mask_matrix=None):
    """Generates downsized 3-D examples from raw files.

    :param num_examples_per_batch: Number of examples per batch.
    :param num_examples_per_target_time: Number of examples (target pixels) per
        target time.
    :param first_target_time_unix_sec: First target time.  Examples will be
        randomly drawn from the period `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Dilation distance.  Will be used to dilate
        WF and CF labels, which effectively creates a distance buffer around
        each front, thus accounting for spatial uncertainty in front placement.
    :param class_fractions: List of downsampling fractions.  Must have length 3,
        where the elements are (NF, WF, CF).  The sum of all fractions must be
        1.0.
    :param num_rows_in_half_grid: Number of rows in half-grid for each example.
        Actual number of rows will be 2 * `num_rows_in_half_grid` + 1.
    :param num_columns_in_half_grid: Same but for columns.
    :param narr_mask_matrix: See doc for
        `machine_learning_utils.check_narr_mask`.  If narr_mask_matrix[i, j]
        = 0, cell [i, j] in the full NARR grid will never be used as the center
        of a downsized example.  If you do not want masking, leave this alone.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: E-by-K numpy array of target values.  All values are
        0 or 1, but the array type is "float64".  Columns are mutually exclusive
        and collectively exhaustive, so the sum across each row is 1.
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

    (narr_file_name_matrix, gridded_front_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_gridded_front_dir_name=top_gridded_front_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_times = len(gridded_front_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    time_index = 0
    num_times_in_memory = 0
    num_times_needed_in_memory = int(
        numpy.ceil(float(num_examples_per_batch) / num_examples_per_target_time)
    )

    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_times_in_memory < num_times_needed_in_memory:
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[time_index, j])

                this_field_predictor_matrix = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[time_index, j])
                )[0]
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix)
                )

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                gridded_front_file_names[time_index])
            this_gridded_front_table = fronts_io.read_grid_from_file(
                gridded_front_file_names[time_index])

            time_index += 1
            if time_index >= num_times:
                time_index = 0

            this_full_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_full_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_full_predictor_matrix)

            this_full_target_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_gridded_front_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_full_target_matrix = ml_utils.binarize_front_images(
                    this_full_target_matrix)

            if num_classes == 2:
                this_full_target_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_full_target_matrix,
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_full_target_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_full_target_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

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

        print 'Creating downsized 3-D examples...'
        sampled_target_point_dict = ml_utils.sample_target_points(
            target_matrix=full_target_matrix, class_fractions=class_fractions,
            num_points_to_sample=num_examples_per_batch,
            mask_matrix=narr_mask_matrix)

        (downsized_predictor_matrix, target_values
        ) = ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=sampled_target_point_dict,
            verbose=False)[:2]

        numpy.random.shuffle(batch_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = target_values[batch_indices]

        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        actual_class_fractions = numpy.sum(target_matrix, axis=0)
        print 'Fraction of examples in each class: {0:s}'.format(
            str(actual_class_fractions))

        full_predictor_matrix = None
        full_target_matrix = None
        num_times_in_memory = 0

        yield (downsized_predictor_matrix, target_matrix)


def quick_downsized_3d_example_gen(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, top_input_dir_name, narr_predictor_names,
        num_classes, num_rows_in_half_grid, num_columns_in_half_grid):
    """Generates downsized 3-D examples from processed files.

    These "processed files" are created by `write_downsized_3d_examples`.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param top_input_dir_name: Name of top-level directory for files with
        downsized 3-D examples.  Files therein will be found by
        `find_downsized_3d_example_file` (with `shuffled == True`) and read by
        `read_downsized_3d_examples`.
    :param narr_predictor_names: See doc for `downsized_3d_example_generator`.
    :param num_classes: Number of target classes (2 or 3).
    :param num_rows_in_half_grid: See doc for `downsized_3d_example_generator`.
    :param num_columns_in_half_grid: Same.
    :return: predictor_matrix: See doc for `downsized_3d_example_generator`.
    :return: target_matrix: Same.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    example_file_names = find_downsized_3d_example_files(
        top_directory_name=top_input_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=LARGE_INTEGER)
    shuffle(example_file_names)

    num_files = len(example_file_names)
    file_index = 0
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])
            this_example_dict = read_downsized_3d_examples(
                netcdf_file_name=example_file_names[file_index],
                predictor_names_to_keep=narr_predictor_names,
                num_half_rows_to_keep=num_rows_in_half_grid,
                num_half_columns_to_keep=num_columns_in_half_grid,
                first_time_to_keep_unix_sec=first_target_time_unix_sec,
                last_time_to_keep_unix_sec=last_target_time_unix_sec)

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            this_num_examples = len(this_example_dict[TARGET_TIMES_KEY])
            if this_num_examples == 0:
                continue

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = this_example_dict[
                    PREDICTOR_MATRIX_KEY] + 0.
                full_target_matrix = this_example_dict[TARGET_MATRIX_KEY] + 0
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_example_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_example_dict[TARGET_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        predictor_matrix = full_predictor_matrix[batch_indices, ...].astype(
            'float32')
        target_matrix = full_target_matrix[batch_indices, ...].astype('float64')

        if num_classes == 2:
            target_values = numpy.argmax(target_matrix, axis=1)
            target_matrix = keras.utils.to_categorical(
                target_values, num_classes)

        actual_class_fractions = numpy.sum(target_matrix, axis=0)
        print 'Number of examples in each class: {0:s}'.format(
            str(actual_class_fractions))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_matrix)


def full_size_3d_example_generator(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, top_narr_directory_name,
        top_gridded_front_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes):
    """Generates full-size 3-D examples from raw files.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param num_classes: Same.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: E-by-M-by-N numpy array of target values.  Each
        value is an integer from the list `front_utils.VALID_INTEGER_IDS`.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    (narr_file_name_matrix, gridded_front_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_gridded_front_dir_name=top_gridded_front_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_target_times = len(gridded_front_file_names)
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

                this_field_predictor_matrix = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[target_time_index, j])
                )[0]
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix)
                )

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                gridded_front_file_names[target_time_index])
            this_gridded_front_table = fronts_io.read_grid_from_file(
                gridded_front_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            this_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_predictor_matrix)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_gridded_front_table,
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
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

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
