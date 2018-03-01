"""IO methods for testing a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

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

import numpy
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import training_validation_io
from generalexam.machine_learning import machine_learning_utils as ml_utils


def create_downsized_3d_examples(
        center_row_indices, center_column_indices, num_rows_in_half_grid,
        num_columns_in_half_grid, full_predictor_matrix=None,
        full_target_matrix=None, target_time_unix_sec=None,
        top_narr_directory_name=None, top_frontal_grid_dir_name=None,
        narr_predictor_names=None, pressure_level_mb=None,
        dilation_distance_for_target_metres=None, num_classes=None):
    """Creates downsized 3-D examples for testing a Keras model.

    Specifically, this method creates one example for each sample point at one
    time step.  Sample points are specified by `center_row_indices` and
    `center_column_indices`.

    Below is an example of how to use this method with a Keras model.

    predictor_matrix, actual_target_values = create_downsized_3d_examples(
        center_row_indices, center_column_indices, ...)
    predicted_target_values = model_object.predict(predictor_matrix, ...)

    E = number of sample points
    M = number of pixel rows in full NARR grid
    N = number of pixel columns in full NARR grid
    C = number of channels (predictor variables)

    m = number of pixel rows in each downsized grid
      = 2 * num_rows_in_half_grid + 1
    n = number of pixel columns in each downsized grid
      = 2 * num_columns_in_half_grid + 1

    If `full_predictor_matrix` and `full_target_matrix` are both given, this
    method will ignore all input args thereafter.

    :param center_row_indices: length-E numpy array of rows that will serve as
        center points of downsized examples.  These are indices into the full
        NARR grid.
    :param center_column_indices: Same as above, except for columns.
    :param num_rows_in_half_grid: See general discussion above.
    :param num_columns_in_half_grid: See general discussion above.
    :param full_predictor_matrix: 1-by-M-by-N-by-C numpy array with predictor
        image.
    :param full_target_matrix: 1-by-M-by-N numpy array with target image.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one file per time step).
    :param narr_predictor_names: 1-D list of NARR fields to use as predictors.
    :param pressure_level_mb: Pressure level (millibars).
    :param dilation_distance_for_target_metres: Dilation distance for target
        variable.  If a front occurs within
        `dilation_distance_for_target_metres` of grid cell [j, k] at time t, the
        label at [t, j, k] will be positive.
    :param num_classes: Number of classes (possible target values).
    :return: downsized_predictor_matrix: E-by-m-by-n-by-C numpy array of
        predictor images.
    :return: target_values: length-E numpy array of integer targets (labels).
    :return: full_predictor_matrix: 1-by-M-by-N-by-C numpy array with predictor
        image.
    :return: full_target_matrix: 1-by-M-by-N numpy array with target image.
    """

    if full_predictor_matrix is None or full_target_matrix is None:
        error_checking.assert_is_integer(num_classes)
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

        narr_file_name_matrix, frontal_grid_file_names = (
            training_validation_io.find_input_files_for_3d_examples(
                first_target_time_unix_sec=target_time_unix_sec,
                last_target_time_unix_sec=target_time_unix_sec,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb))

        narr_file_names = narr_file_name_matrix[0, :]
        frontal_grid_file_name = frontal_grid_file_names[0]

        num_predictors = len(narr_predictor_names)
        tuple_of_full_predictor_matrices = ()

        for j in range(num_predictors):
            print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])
            this_field_predictor_matrix, _, _, _ = (
                processed_narr_io.read_fields_from_file(narr_file_names[j]))
            tuple_of_full_predictor_matrices += (this_field_predictor_matrix,)

        print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
        frontal_grid_table = fronts_io.read_narr_grids_from_file(
            frontal_grid_file_name)

        full_predictor_matrix = ml_utils.stack_predictor_variables(
            tuple_of_full_predictor_matrices)
        full_predictor_matrix = ml_utils.normalize_predictor_matrix(
            predictor_matrix=full_predictor_matrix,
            normalize_by_example=True)

        full_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=frontal_grid_table,
            num_rows_per_image=full_predictor_matrix.shape[1],
            num_columns_per_image=full_predictor_matrix.shape[2])

        if num_classes == 2:
            full_target_matrix = ml_utils.binarize_front_images(
                full_target_matrix)

        full_predictor_matrix = ml_utils.remove_nans_from_narr_grid(
            full_predictor_matrix)
        full_target_matrix = ml_utils.remove_nans_from_narr_grid(
            full_target_matrix)

        if num_classes == 2:
            full_target_matrix = ml_utils.dilate_binary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_for_target_metres,
                verbose=False)
        else:
            full_target_matrix = ml_utils.dilate_ternary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_for_target_metres,
                verbose=False)

    num_sample_points = len(center_row_indices)
    print ('Creating downsized 3-D example for each of {0:d} sample '
           'points...').format(num_sample_points)

    this_target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }
    downsized_predictor_matrix, target_values, _, _, _ = (
        ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=this_target_point_dict,
            verbose=False))

    downsized_predictor_matrix = downsized_predictor_matrix.astype('float32')
    return (downsized_predictor_matrix, target_values, full_predictor_matrix,
            full_target_matrix)


def create_downsized_4d_examples(
        center_row_indices, center_column_indices, num_rows_in_half_grid,
        num_columns_in_half_grid, full_predictor_matrix=None,
        full_target_matrix=None, target_time_unix_sec=None,
        num_predictor_time_steps=None, num_lead_time_steps=None,
        top_narr_directory_name=None, top_frontal_grid_dir_name=None,
        narr_predictor_names=None, pressure_level_mb=None,
        dilation_distance_for_target_metres=None, num_classes=None):
    """Creates downsized 4-D examples for testing a Keras model.

    Specifically, this method creates one example for each sample point at one
    time step.  Sample points are specified by `center_row_indices` and
    `center_column_indices`.

    Below is an example of how to use this method with a Keras model.

    predictor_matrix, actual_target_values = create_downsized_4d_examples(
        center_row_indices, center_column_indices, ...)
    predicted_target_values = model_object.predict(predictor_matrix, ...)

    E = number of sample points (length of `center_row_indices` or
        `center_column_indices`)
    m = number of pixel rows in each downsized grid
      = 2 * num_rows_in_half_grid + 1
    n = number of pixel columns in each downsized grid
      = 2 * num_columns_in_half_grid + 1

    :param center_row_indices: See documentation for
        `create_downsized_3d_examples`.
    :param center_column_indices: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param full_predictor_matrix: Same.
    :param full_target_matrix: Same.
    :param target_time_unix_sec: Same.
    :param num_predictor_time_steps:
    :param num_lead_time_steps:
    :param top_narr_directory_name: See documentation for
        `create_downsized_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: Same.
    :param num_classes: Same.
    :return: downsized_predictor_matrix: E-by-m-by-n-by-T-by-C numpy array of
        predictor images.
    :return: target_values: length-E numpy array of integer targets (labels).
    :return: full_predictor_matrix: 1-by-M-by-N-by-T-by-C numpy array with
        predictor image.
    :return: full_target_matrix: 1-by-M-by-N numpy array with target image.
    """

    if full_predictor_matrix is None or full_target_matrix is None:
        error_checking.assert_is_integer(num_classes)
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

        narr_file_name_matrix, frontal_grid_file_names = (
            training_validation_io.find_input_files_for_4d_examples(
                first_target_time_unix_sec=target_time_unix_sec,
                last_target_time_unix_sec=target_time_unix_sec,
                num_predictor_time_steps=num_predictor_time_steps,
                num_lead_time_steps=num_lead_time_steps,
                top_narr_directory_name=top_narr_directory_name,
                top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                narr_predictor_names=narr_predictor_names,
                pressure_level_mb=pressure_level_mb))

        narr_file_name_matrix = narr_file_name_matrix[0, ...]
        frontal_grid_file_name = frontal_grid_file_names[0]

        num_predictors = len(narr_predictor_names)
        tuple_of_4d_predictor_matrices = ()

        for i in range(num_predictor_time_steps):
            tuple_of_3d_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[i, j])
                this_field_predictor_matrix, _, _, _ = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[i, j]))

                tuple_of_3d_predictor_matrices += (this_field_predictor_matrix,)

            tuple_of_4d_predictor_matrices += (
                ml_utils.stack_predictor_variables(
                    tuple_of_3d_predictor_matrices),)

        full_predictor_matrix = ml_utils.stack_time_steps(
            tuple_of_4d_predictor_matrices)

        print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
        frontal_grid_table = fronts_io.read_narr_grids_from_file(
            frontal_grid_file_name)

        full_predictor_matrix = ml_utils.normalize_predictor_matrix(
            predictor_matrix=full_predictor_matrix, normalize_by_example=True)

        full_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=frontal_grid_table,
            num_rows_per_image=full_predictor_matrix.shape[1],
            num_columns_per_image=full_predictor_matrix.shape[2])

        if num_classes == 2:
            full_target_matrix = ml_utils.binarize_front_images(
                full_target_matrix)

        full_predictor_matrix = ml_utils.remove_nans_from_narr_grid(
            full_predictor_matrix)
        full_target_matrix = ml_utils.remove_nans_from_narr_grid(
            full_target_matrix)

        if num_classes == 2:
            full_target_matrix = ml_utils.dilate_binary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_for_target_metres,
                verbose=False)
        else:
            full_target_matrix = ml_utils.dilate_ternary_target_images(
                target_matrix=full_target_matrix,
                dilation_distance_metres=dilation_distance_for_target_metres,
                verbose=False)

    num_sample_points = len(center_row_indices)
    print ('Creating downsized 3-D example for each of {0:d} sample '
           'points...').format(num_sample_points)

    this_target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }
    downsized_predictor_matrix, target_values, _, _, _ = (
        ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=this_target_point_dict,
            verbose=False))

    downsized_predictor_matrix = downsized_predictor_matrix.astype('float32')
    return (downsized_predictor_matrix, target_values, full_predictor_matrix,
            full_target_matrix)


def create_full_size_3d_example(
        target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_classes):
    """Creates one full-size 3-D testing example for a Keras model.

    Below is an example of how to use this method with a Keras model.

    predictor_matrix, actual_target_matrix = create_full_size_3d_example(
        target_time_unix_sec, top_narr_directory_name, ...)
    predicted_target_matrix = model_object.predict(predictor_matrix, ...)

    :param target_time_unix_sec: See documentation for
        `downsized_3d_example_generator`.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: Same.
    :param num_classes: Same.
    :return: predictor_matrix: 1-by-M-by-N-by-C numpy array with predictor
        image.
    :return: target_matrix: 1-by-M-by-N numpy array with target image.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        training_validation_io.find_input_files_for_3d_examples(
            first_target_time_unix_sec=target_time_unix_sec,
            last_target_time_unix_sec=target_time_unix_sec,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    narr_file_names = narr_file_name_matrix[0, :]
    frontal_grid_file_name = frontal_grid_file_names[0]
    num_predictors = len(narr_predictor_names)

    tuple_of_predictor_matrices = ()

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])
        this_field_predictor_matrix, _, _, _ = (
            processed_narr_io.read_fields_from_file(narr_file_names[j]))
        tuple_of_predictor_matrices += (this_field_predictor_matrix,)

    print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        frontal_grid_file_name)

    print 'Processing full-size 3-D machine-learning example...'

    predictor_matrix = ml_utils.stack_predictor_variables(
        tuple_of_predictor_matrices)
    predictor_matrix = ml_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_example=True)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=frontal_grid_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    if num_classes == 2:
        target_matrix = ml_utils.binarize_front_images(target_matrix)

    predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(predictor_matrix)
    target_matrix = ml_utils.subset_narr_grid_for_fcn_input(target_matrix)

    if num_classes == 2:
        target_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_for_target_metres,
            verbose=False)
    else:
        target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_for_target_metres,
            verbose=False)

    predictor_matrix = predictor_matrix.astype('float32')
    print 'Fraction of pixels with a front = {0:.4f}'.format(
        numpy.mean(target_matrix > 0))

    target_matrix = numpy.expand_dims(target_matrix, axis=-1)
    return predictor_matrix, target_matrix


def create_full_size_4d_example(
        target_time_unix_sec, num_predictor_time_steps, num_lead_time_steps,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb,
        dilation_distance_for_target_metres, num_classes):
    """Creates one full-size 4-D testing example for a Keras model.

    Unlike `downsized_3d_example_generator`, this method is not a generator.  In
    other words, it does not fit the template specified by
    `keras.models.*.predict_generator`.  Thus, when using this method to test a
    Keras model, you should use the  `keras.models.*.predict` method.  For
    example:

    predictor_matrix, actual_target_matrix = create_full_size_4d_example(
        target_time_unix_sec, num_predictor_time_steps, ...)
    predicted_target_matrix = model_object.predict(predictor_matrix, ...)

    :param target_time_unix_sec: See documentation for
        `downsized_3d_example_generator`.
    :param num_predictor_time_steps: Number of predictor times per example.
    :param num_lead_time_steps: Number of time steps separating latest predictor
        time from target time.
    :param top_narr_directory_name: See documentation for
        `downsized_3d_example_generator`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_for_target_metres: Same.
    :param num_classes: Same.
    :return: predictor_matrix: 1-by-M-by-N-by-T-by-C numpy array of predictor
        images.
    :return: target_matrix: 1-by-M-by-N numpy array with target image.
    """

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    narr_file_name_matrix, frontal_grid_file_names = (
        training_validation_io.find_input_files_for_4d_examples(
            first_target_time_unix_sec=target_time_unix_sec,
            last_target_time_unix_sec=target_time_unix_sec,
            num_predictor_time_steps=num_predictor_time_steps,
            num_lead_time_steps=num_lead_time_steps,
            top_narr_directory_name=top_narr_directory_name,
            top_frontal_grid_dir_name=top_frontal_grid_dir_name,
            narr_predictor_names=narr_predictor_names,
            pressure_level_mb=pressure_level_mb))

    narr_file_name_matrix = narr_file_name_matrix[0, ...]
    frontal_grid_file_name = frontal_grid_file_names[0]
    num_predictors = len(narr_predictor_names)

    tuple_of_4d_predictor_matrices = ()

    for i in range(num_predictor_time_steps):
        tuple_of_3d_predictor_matrices = ()

        for j in range(num_predictors):
            print 'Reading data from: "{0:s}"...'.format(
                narr_file_name_matrix[i, j])
            this_field_predictor_matrix, _, _, _ = (
                processed_narr_io.read_fields_from_file(
                    narr_file_name_matrix[i, j]))

            tuple_of_3d_predictor_matrices += (this_field_predictor_matrix,)

        tuple_of_4d_predictor_matrices += (
            ml_utils.stack_predictor_variables(tuple_of_3d_predictor_matrices),)

    predictor_matrix = ml_utils.stack_time_steps(tuple_of_4d_predictor_matrices)

    print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        frontal_grid_file_name)

    print 'Processing full-size 4-D machine-learning example...'

    predictor_matrix = ml_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_example=True)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=frontal_grid_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    if num_classes == 2:
        target_matrix = ml_utils.binarize_front_images(target_matrix)

    predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(predictor_matrix)
    target_matrix = ml_utils.subset_narr_grid_for_fcn_input(target_matrix)

    if num_classes == 2:
        target_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_for_target_metres,
            verbose=False)
    else:
        target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_for_target_metres,
            verbose=False)

    predictor_matrix = predictor_matrix.astype('float32')
    print 'Fraction of pixels with a front = {0:.4f}'.format(
        numpy.mean(target_matrix > 0))

    target_matrix = numpy.expand_dims(target_matrix, axis=-1)
    return predictor_matrix, target_matrix
