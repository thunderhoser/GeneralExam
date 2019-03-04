"""IO methods for testing and deployment.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples.  Examples may be either downsized or full-size.  See
    below for a definition of each.
K = number of classes (possible target values).  See below for the definition of
    "target value".
M = number of spatial rows per example
N = number of spatial columns per example
T = number of predictor times per example (number of images per sequence)
C = number of channels (predictor variables) per example

--- DEFINITIONS ---

A "downsized" example covers only a subset of the NARR grid, while a full-size
example covers the entire NARR grid.

The dimensions of a 3-D example are M x N x C (only one predictor time).

The dimensions of a 4-D example are M x N x T x C.

NF = no front
WF = warm front
CF = cold front

Target variable = label at one pixel.  For a downsized example, there is only
one target variable (the label at the center pixel).  For a full-size example,
there are M*N target variables (the label at each pixel).
"""

import numpy
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import training_validation_io as trainval_io
from generalexam.machine_learning import machine_learning_utils as ml_utils


def create_downsized_3d_examples(
        center_row_indices, center_column_indices, num_rows_in_half_grid,
        num_columns_in_half_grid, full_predictor_matrix=None,
        full_target_matrix=None, target_time_unix_sec=None,
        top_narr_directory_name=None, top_gridded_front_dir_name=None,
        narr_predictor_names=None, pressure_level_mb=None,
        dilation_distance_metres=None, num_classes=None):
    """Creates downsized 3-D examples from raw files.

    This method creates one example for each center pixel, where center pixels
    are specified by `center_row_indices` and `center_column_indices`.

    If `full_predictor_matrix` and `full_target_matrix` are specified, all input
    args thereafter will be ignored.

    E = number of examples (center pixels)
    P = number of rows in full NARR grid
    Q = number of columns in full NARR grid

    :param center_row_indices: length-E numpy array with row for each center
        pixel.
    :param center_column_indices: length-E numpy array with column for each
        center pixel.
    :param num_rows_in_half_grid: Number of rows in half-grid for each example.
        Actual number of rows will be 2 * `num_rows_in_half_grid` + 1.
    :param num_columns_in_half_grid: Same but for columns.
    :param full_predictor_matrix: 1-by-P-by-Q-by-C numpy array of predictor
        values.
    :param full_target_matrix: 1-by-P-by-Q-by-C numpy array of target values.
        Each value must be accepted by `front_utils.check_front_type_enum`.
    :param target_time_unix_sec: Target time.
    :param top_narr_directory_name: Name of top-level directory with NARR data.
        Files therein will be found by
        `processed_narr_io.find_file_for_one_time` and read by
        `processed_narr_io.read_fields_from_file`.
    :param top_gridded_front_dir_name: Name of top-level directory with target
        values (gridded front labels).  Files therein will be found by
        `fronts_io.find_gridded_file` and read by
        `fronts_io.read_grid_from_file`.
    :param narr_predictor_names: length-C list with names of predictor
        variables.  Each must be accepted by
        `processed_narr_io.check_field_name`.
    :param pressure_level_mb: Pressure level (millibars) for predictors.
    :param dilation_distance_metres: Dilation distance.  Will be used to dilate
        WF and CF labels, which effectively creates a distance buffer around
        each front, thus accounting for spatial uncertainty in front placement.
    :param num_classes: Number of target classes (either 2 or 3).
    :return: downsized_predictor_matrix: E-by-M-by-N-by-C numpy array of
        predictor values.
    :return: target_values: length-E numpy array of target values (integers from
        the list `front_utils.VALID_INTEGER_IDS`).
    :return: full_predictor_matrix: See input doc.
    :return: full_target_matrix: See input doc.
    """

    if full_predictor_matrix is None or full_target_matrix is None:
        error_checking.assert_is_integer(num_classes)
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

        narr_file_name_matrix, gridded_front_file_names = (
            trainval_io.find_input_files_for_3d_examples(
                first_target_time_unix_sec=target_time_unix_sec,
                last_target_time_unix_sec=target_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb)
        )

        narr_file_names = narr_file_name_matrix[0, :]
        gridded_front_file_name = gridded_front_file_names[0]

        num_predictors = len(narr_predictor_names)
        tuple_of_full_predictor_matrices = ()

        for j in range(num_predictors):
            print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])

            this_field_predictor_matrix = (
                processed_narr_io.read_fields_from_file(narr_file_names[j])
            )[0]

            this_field_predictor_matrix = (
                ml_utils.fill_nans_in_predictor_images(
                    this_field_predictor_matrix)
            )

            tuple_of_full_predictor_matrices += (this_field_predictor_matrix,)

        print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
        gridded_front_table = fronts_io.read_grid_from_file(
            gridded_front_file_name)

        full_predictor_matrix = ml_utils.stack_predictor_variables(
            tuple_of_full_predictor_matrices)
        full_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=full_predictor_matrix)

        full_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=gridded_front_table,
            num_rows_per_image=full_predictor_matrix.shape[1],
            num_columns_per_image=full_predictor_matrix.shape[2])

        if num_classes == 2:
            full_target_matrix = ml_utils.binarize_front_images(
                full_target_matrix)

        if num_classes == 2:
            full_target_matrix = ml_utils.dilate_binary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)
        else:
            full_target_matrix = ml_utils.dilate_ternary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)

    print 'Creating {0:d} downsized 3-D examples...'.format(
        len(center_row_indices))

    this_target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }

    downsized_predictor_matrix, target_values = (
        ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=this_target_point_dict,
            verbose=False
        )[:2]
    )

    downsized_predictor_matrix = downsized_predictor_matrix.astype('float32')

    return (downsized_predictor_matrix, target_values, full_predictor_matrix,
            full_target_matrix)


def create_downsized_4d_examples(
        center_row_indices, center_column_indices, num_rows_in_half_grid,
        num_columns_in_half_grid, full_predictor_matrix=None,
        full_target_matrix=None, target_time_unix_sec=None,
        num_lead_time_steps=None, predictor_time_step_offsets=None,
        top_narr_directory_name=None, top_gridded_front_dir_name=None,
        narr_predictor_names=None, pressure_level_mb=None,
        dilation_distance_metres=None, num_classes=None):
    """Creates downsized 4-D examples from raw files.

    This method creates one example for each center pixel, where center pixels
    are specified by `center_row_indices` and `center_column_indices`.

    If `full_predictor_matrix` and `full_target_matrix` are specified, all input
    args thereafter will be ignored.

    E = number of examples (center pixels)
    P = number of rows in full NARR grid
    Q = number of columns in full NARR grid

    :param center_row_indices: See doc for `create_downsized_3d_examples`.
    :param center_column_indices: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param full_predictor_matrix: 1-by-P-by-Q-by-T-by-C numpy array of
        predictor values.
    :param full_target_matrix: 1-by-P-by-Q-by-C numpy array of target values.
        Each value must be accepted by `front_utils.check_front_type_enum`.
    :param target_time_unix_sec: See doc for `create_downsized_3d_examples`.
    :param num_lead_time_steps: Number of time steps between target time and
        latest possible predictor time.
    :param predictor_time_step_offsets: length-T numpy array of offsets between
        predictor time and latest possible predictor time (target time minus
        lead time).
    :param top_narr_directory_name: See doc for `create_downsized_3d_examples`.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Same.
    :return: downsized_predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of
        predictor values.
    :return: target_values: length-E numpy array of target values (integers from
        the list `front_utils.VALID_INTEGER_IDS`).
    :return: full_predictor_matrix: See input doc.
    :return: full_target_matrix: See input doc.
    """

    if full_predictor_matrix is None or full_target_matrix is None:
        error_checking.assert_is_integer(num_classes)
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

        narr_file_name_matrix, gridded_front_file_names = (
            trainval_io.find_input_files_for_4d_examples(
                first_target_time_unix_sec=target_time_unix_sec,
                last_target_time_unix_sec=target_time_unix_sec,
                predictor_time_step_offsets=predictor_time_step_offsets,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb)
        )

        narr_file_name_matrix = narr_file_name_matrix[0, ...]
        gridded_front_file_name = gridded_front_file_names[0]

        num_predictor_times_per_example = len(predictor_time_step_offsets)
        num_predictors = len(narr_predictor_names)
        tuple_of_4d_predictor_matrices = ()

        for i in range(num_predictor_times_per_example):
            tuple_of_3d_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[i, j])

                this_field_predictor_matrix = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[i, j])
                )[0]

                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix)
                )

                tuple_of_3d_predictor_matrices += (this_field_predictor_matrix,)

            tuple_of_4d_predictor_matrices += (
                ml_utils.stack_predictor_variables(
                    tuple_of_3d_predictor_matrices),
            )

        full_predictor_matrix = ml_utils.stack_time_steps(
            tuple_of_4d_predictor_matrices)

        print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
        gridded_front_table = fronts_io.read_grid_from_file(
            gridded_front_file_name)

        full_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=full_predictor_matrix)

        full_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=gridded_front_table,
            num_rows_per_image=full_predictor_matrix.shape[1],
            num_columns_per_image=full_predictor_matrix.shape[2])

        if num_classes == 2:
            full_target_matrix = ml_utils.binarize_front_images(
                full_target_matrix)

        if num_classes == 2:
            full_target_matrix = ml_utils.dilate_binary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)
        else:
            full_target_matrix = ml_utils.dilate_ternary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)

    print 'Creating {0:d} downsized 4-D examples...'.format(
        len(center_row_indices))

    this_target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }

    downsized_predictor_matrix, target_values = (
        ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=this_target_point_dict,
            verbose=False
        )[:2]
    )

    downsized_predictor_matrix = downsized_predictor_matrix.astype('float32')

    return (downsized_predictor_matrix, target_values, full_predictor_matrix,
            full_target_matrix)


def create_full_size_3d_example(
        target_time_unix_sec, top_narr_directory_name,
        top_gridded_front_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes):
    """Creates full-size 3-D examples from raw files.

    :param target_time_unix_sec: See doc for `create_downsized_3d_examples`.
    :param top_narr_directory_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Same.
    :return: predictor_matrix: 1-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: 1-by-M-by-N numpy array of target values.  Each
        value is an integer from the list `front_utils.VALID_INTEGER_IDS`.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, gridded_front_file_names = (
        trainval_io.find_input_files_for_3d_examples(
            first_target_time_unix_sec=target_time_unix_sec,
            last_target_time_unix_sec=target_time_unix_sec,
            top_narr_directory_name=top_narr_directory_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb)
    )

    narr_file_names = narr_file_name_matrix[0, :]
    gridded_front_file_name = gridded_front_file_names[0]

    tuple_of_predictor_matrices = ()
    num_predictors = len(narr_predictor_names)

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])

        this_field_predictor_matrix = (
            processed_narr_io.read_fields_from_file(narr_file_names[j])
        )[0]

        this_field_predictor_matrix = (
            ml_utils.fill_nans_in_predictor_images(
                this_field_predictor_matrix)
        )

        tuple_of_predictor_matrices += (this_field_predictor_matrix,)

    print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
    gridded_front_table = fronts_io.read_grid_from_file(gridded_front_file_name)

    print 'Processing full-size 3-D machine-learning example...'

    predictor_matrix = ml_utils.stack_predictor_variables(
        tuple_of_predictor_matrices)
    predictor_matrix, _ = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    if num_classes == 2:
        target_matrix = ml_utils.binarize_front_images(target_matrix)

    predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(predictor_matrix)
    target_matrix = ml_utils.subset_narr_grid_for_fcn_input(target_matrix)

    if num_classes == 2:
        target_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False)
    else:
        target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False)

    predictor_matrix = predictor_matrix.astype('float32')
    print 'Fraction of pixels with a front = {0:.4f}'.format(
        numpy.mean(target_matrix > 0)
    )

    target_matrix = numpy.expand_dims(target_matrix, axis=-1)
    return predictor_matrix, target_matrix


def create_full_size_4d_example(
        target_time_unix_sec, num_lead_time_steps, predictor_time_step_offsets,
        top_narr_directory_name, top_gridded_front_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        num_classes):
    """Creates full-size 4-D examples from raw files.

    :param target_time_unix_sec: See doc for `create_downsized_3d_examples`.
    :param num_lead_time_steps: See doc for `create_downsized_4d_examples`.
    :param predictor_time_step_offsets: Same.
    :param top_narr_directory_name: See doc for `create_downsized_3d_examples`.
    :param top_gridded_front_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Same.
    :return: predictor_matrix: 1-by-M-by-N-by-T-by-C numpy array of predictor
        values.
    :return: target_matrix: 1-by-M-by-N numpy array of target values.  Each
        value is an integer from the list `front_utils.VALID_INTEGER_IDS`.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, gridded_front_file_names = (
        trainval_io.find_input_files_for_4d_examples(
            first_target_time_unix_sec=target_time_unix_sec,
            last_target_time_unix_sec=target_time_unix_sec,
            predictor_time_step_offsets=predictor_time_step_offsets,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb)
    )

    narr_file_name_matrix = narr_file_name_matrix[0, ...]
    gridded_front_file_name = gridded_front_file_names[0]

    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)
    tuple_of_4d_predictor_matrices = ()

    for i in range(num_predictor_times_per_example):
        tuple_of_3d_predictor_matrices = ()

        for j in range(num_predictors):
            print 'Reading data from: "{0:s}"...'.format(
                narr_file_name_matrix[i, j])

            this_field_predictor_matrix = (
                processed_narr_io.read_fields_from_file(
                    narr_file_name_matrix[i, j])
            )[0]

            this_field_predictor_matrix = (
                ml_utils.fill_nans_in_predictor_images(
                    this_field_predictor_matrix)
            )

            tuple_of_3d_predictor_matrices += (this_field_predictor_matrix,)

        tuple_of_4d_predictor_matrices += (
            ml_utils.stack_predictor_variables(tuple_of_3d_predictor_matrices),
        )

    predictor_matrix = ml_utils.stack_time_steps(tuple_of_4d_predictor_matrices)

    print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
    gridded_front_table = fronts_io.read_grid_from_file(gridded_front_file_name)

    print 'Processing full-size 4-D machine-learning example...'
    predictor_matrix, _ = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    if num_classes == 2:
        target_matrix = ml_utils.binarize_front_images(target_matrix)

    predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(predictor_matrix)
    target_matrix = ml_utils.subset_narr_grid_for_fcn_input(target_matrix)

    if num_classes == 2:
        target_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False)
    else:
        target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False)

    predictor_matrix = predictor_matrix.astype('float32')
    print 'Fraction of pixels with a front = {0:.4f}'.format(
        numpy.mean(target_matrix > 0)
    )

    target_matrix = numpy.expand_dims(target_matrix, axis=-1)
    return predictor_matrix, target_matrix
