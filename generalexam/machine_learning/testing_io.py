"""IO methods for testing and deploying a trained model.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples
M = number of rows in full grid
N = number of columns in full grid
m = number of rows in downsized grid
n = number of columns in downsized grid
C = number of channels (predictors)
"""

import os.path
import warnings
import numpy
from keras.utils import to_categorical
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import fronts_io
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_VALUES_KEY = 'target_values'
FULL_PREDICTOR_MATRIX_KEY = 'full_size_predictor_matrix'
FULL_TARGET_MATRIX_KEY = 'full_size_target_matrix'


def create_downsized_examples_no_targets(
        center_row_indices, center_column_indices, num_half_rows,
        num_half_columns, full_size_predictor_matrix=None,
        top_predictor_dir_name=None, valid_time_unix_sec=None,
        pressure_levels_mb=None, predictor_names=None,
        normalization_type_string=None):
    """Created downsized examples without target values.

    If `full_size_predictor_matrix` is defined, this method will not use the
    following input args.

    :param center_row_indices: length-E numpy array of row indices.  If
        center_row_indices[i] = j, the center of the predictor grid for the
        [i]th example will be the [j]th row of the full grid.
    :param center_column_indices: Same but for columns.
    :param num_half_rows: Number of half-rows in predictor grid.  m (defined in
        the above discussion) will be `2 * num_half_rows + 1`.
    :param num_half_columns: Same but for columns.
    :param full_size_predictor_matrix: 1-by-M-by-N-by-C numpy array of predictor
        values on the full grid.
    :param top_predictor_dir_name: Name of top-level directory with predictors.
        Files therein will be found by `predictor_io.find_file` and read by
        `predictor_io.read_file`.
    :param valid_time_unix_sec: Valid time.
    :param pressure_levels_mb: length-C numpy array of pressure levels
        (millibars).
    :param predictor_names: length-C list of predictor names (each must be
        accepted by `predictor_utils.check_field_name`).
    :param normalization_type_string: Normalization method for predictors (see
        doc for `machine_learning_utils.normalize_predictors`).
    :return: result_dict: Dictionary with the following keys.
    result_dict['predictor_matrix']: E-by-m-by-n-by-C numpy array of predictor
        values.
    result_dict['full_size_predictor_matrix']: See input doc.
    """

    if full_size_predictor_matrix is None:
        predictor_file_name = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_time_unix_sec,
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(predictor_file_name)
        predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_name,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            field_names_to_keep=predictor_names)

        full_size_predictor_matrix = predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][[0], ...]

        for j in range(len(predictor_names)):
            full_size_predictor_matrix[..., j] = (
                ml_utils.fill_nans_in_predictor_images(
                    full_size_predictor_matrix[..., j]
                )
            )

        full_size_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=full_size_predictor_matrix,
            normalization_type_string=normalization_type_string)

    error_checking.assert_is_integer_numpy_array(center_row_indices)
    error_checking.assert_is_numpy_array(center_row_indices, num_dimensions=1)

    num_examples = len(center_row_indices)
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(center_column_indices)
    error_checking.assert_is_numpy_array(
        center_column_indices, exact_dimensions=these_expected_dim)

    print 'Creating {0:d} downsized learning examples...'.format(num_examples)

    target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }

    dummy_target_matrix = numpy.full(
        full_size_predictor_matrix.shape[:-1], 0, dtype=int
    )

    predictor_matrix = ml_utils.downsize_grids_around_selected_points(
        predictor_matrix=full_size_predictor_matrix,
        target_matrix=dummy_target_matrix,
        num_rows_in_half_window=num_half_rows,
        num_columns_in_half_window=num_half_columns,
        target_point_dict=target_point_dict, verbose=False
    )[0]

    predictor_matrix = predictor_matrix.astype('float32')

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        FULL_PREDICTOR_MATRIX_KEY: full_size_predictor_matrix
    }


def create_downsized_examples_with_targets(
        center_row_indices, center_column_indices, num_half_rows,
        num_half_columns, full_size_predictor_matrix=None,
        full_size_target_matrix=None, top_predictor_dir_name=None,
        top_gridded_front_dir_name=None, valid_time_unix_sec=None,
        pressure_levels_mb=None, predictor_names=None,
        normalization_type_string=None, dilation_distance_metres=None,
        num_classes=3):
    """Created downsized examples with target values.

    If `full_size_predictor_matrix` is defined, this method will not use input
    args after `full_size_target_matrix`.

    :param center_row_indices: See doc for
        `create_downsized_examples_no_targets`.
    :param center_column_indices: Same.
    :param num_half_rows: Same.
    :param num_half_columns: Same.
    :param full_size_predictor_matrix: Same.
    :param full_size_target_matrix: 1-by-M-by-N numpy array of target
        values on the full grid.  All values must be in 0...(num_classes - 1).
        If full_size_target_matrix[0, i, j] = k, grid cell [i, j] belongs to
        class k.
    :param top_predictor_dir_name: See doc for
        `create_downsized_examples_no_targets`.
    :param top_gridded_front_dir_name: Name of top-level directory with gridded
        front labels.  Files therein will be found by
        `fronts_io.find_gridded_file` and read by
        `fronts_io.read_grid_from_file`.
    :param valid_time_unix_sec: See doc for
        `create_downsized_examples_no_targets`.
    :param pressure_levels_mb: Same.
    :param predictor_names: Same.
    :param normalization_type_string: Same.
    :param dilation_distance_metres:
    :param dilation_distance_metres: Dilation distance for gridded warm-front
        and cold-front labels.
    :param num_classes: Number of classes.  If `num_classes == 3`, the problem
        will remain multiclass (no front, warm front, or cold front).  If
        `num_classes == 2`, the problem will be simplified to binary (front or
        no front).
    :return: result_dict: Dictionary with the following keys.
    result_dict['predictor_matrix']: E-by-m-by-n-by-C numpy array of predictor
        values.
    result_dict['target_values']: length-E numpy array of target values
        (integers in range 0...[num_classes - 1]).
    result_dict['full_size_predictor_matrix']: See input doc.
    result_dict['full_size_target_matrix']: See input doc.
    """

    if full_size_predictor_matrix is None:
        error_checking.assert_is_integer(num_classes)
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

        predictor_file_name = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_time_unix_sec,
            raise_error_if_missing=True)

        print 'Reading data from: "{0:s}"...'.format(predictor_file_name)
        predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_name,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            field_names_to_keep=predictor_names)

        full_size_predictor_matrix = predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][[0], ...]

        for j in range(len(predictor_names)):
            full_size_predictor_matrix[..., j] = (
                ml_utils.fill_nans_in_predictor_images(
                    full_size_predictor_matrix[..., j]
                )
            )

        full_size_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=full_size_predictor_matrix,
            normalization_type_string=normalization_type_string)

        gridded_front_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_gridded_front_dir_name,
            valid_time_unix_sec=valid_time_unix_sec,
            raise_error_if_missing=False)

        if not os.path.isfile(gridded_front_file_name):
            warning_string = (
                'POTENTIAL PROBLEM.  Cannot find file expected at: "{0:s}"'
            ).format(gridded_front_file_name)

            warnings.warn(warning_string)
            return None

        print 'Reading data from: "{0:s}"...'.format(
            gridded_front_file_name)
        gridded_front_table = fronts_io.read_grid_from_file(
            gridded_front_file_name)

        full_size_target_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=gridded_front_table,
            num_rows_per_image=full_size_predictor_matrix.shape[1],
            num_columns_per_image=full_size_predictor_matrix.shape[2]
        )

        if num_classes == 2:
            full_size_target_matrix = ml_utils.binarize_front_images(
                full_size_target_matrix)

        if num_classes == 2:
            full_size_target_matrix = ml_utils.dilate_binary_target_images(
                target_matrix=full_size_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)
        else:
            full_size_target_matrix = ml_utils.dilate_ternary_target_images(
                target_matrix=full_size_target_matrix,
                dilation_distance_metres=dilation_distance_metres,
                verbose=False)

    error_checking.assert_is_integer_numpy_array(center_row_indices)
    error_checking.assert_is_numpy_array(center_row_indices, num_dimensions=1)

    num_examples = len(center_row_indices)
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(center_column_indices)
    error_checking.assert_is_numpy_array(
        center_column_indices, exact_dimensions=these_expected_dim)

    print 'Creating {0:d} downsized learning examples...'.format(num_examples)

    target_point_dict = {
        ml_utils.ROW_INDICES_BY_TIME_KEY: [center_row_indices],
        ml_utils.COLUMN_INDICES_BY_TIME_KEY: [center_column_indices]
    }

    predictor_matrix, target_values = (
        ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_size_predictor_matrix,
            target_matrix=full_size_target_matrix,
            num_rows_in_half_window=num_half_rows,
            num_columns_in_half_window=num_half_columns,
            target_point_dict=target_point_dict, verbose=False
        )[:2]
    )

    predictor_matrix = predictor_matrix.astype('float32')

    num_examples_by_class = numpy.array(
        [numpy.sum(target_values == k) for k in range(num_classes)], dtype=int
    )

    print 'Number of examples in each class: {0:s}'.format(
        str(num_examples_by_class)
    )

    return {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_VALUES_KEY: target_values,
        FULL_PREDICTOR_MATRIX_KEY: full_size_predictor_matrix,
        FULL_TARGET_MATRIX_KEY: full_size_target_matrix
    }


def create_full_size_example(
        top_predictor_dir_name, top_gridded_front_dir_name, valid_time_unix_sec,
        pressure_levels_mb, predictor_names, normalization_type_string,
        dilation_distance_metres, num_classes):
    """Creates full-size example (for semantic segmentation).

    M = number of rows in full grid
    N = number of columns in full grid
    C = number of channels (predictors)
    K = number of classes (either 2 or 3)

    :param top_predictor_dir_name: See doc for `create_downsized_examples`.
    :param top_gridded_front_dir_name: Same.
    :param valid_time_unix_sec: Same.
    :param pressure_levels_mb: Same.
    :param predictor_names: Same.
    :param normalization_type_string: Same.
    :param dilation_distance_metres: Same.
    :param num_classes: Same.
    :return: predictor_matrix: 1-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: 1-by-M-by-N-by-K numpy array of zeros and ones (but
        type is "float64").  If target_matrix[0, i, j, k] = 1, grid cell [i, j]
        belongs to the [k]th class.
    """

    # TODO(thunderhoser): Probably need to use mask here as well.

    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    gridded_front_file_name = fronts_io.find_gridded_file(
        top_directory_name=top_gridded_front_dir_name,
        valid_time_unix_sec=valid_time_unix_sec,
        raise_error_if_missing=False)

    if not os.path.isfile(gridded_front_file_name):
        warning_string = (
            'POTENTIAL PROBLEM.  Cannot find file expected at: "{0:s}"'
        ).format(gridded_front_file_name)

        warnings.warn(warning_string)
        return None, None

    predictor_file_name = predictor_io.find_file(
        top_directory_name=top_predictor_dir_name,
        valid_time_unix_sec=valid_time_unix_sec,
        raise_error_if_missing=True)

    print 'Reading data from: "{0:s}"...'.format(predictor_file_name)
    predictor_dict = predictor_io.read_file(
        netcdf_file_name=predictor_file_name,
        pressure_levels_to_keep_mb=pressure_levels_mb,
        field_names_to_keep=predictor_names)

    predictor_matrix = predictor_dict[predictor_utils.DATA_MATRIX_KEY][[0], ...]

    for j in range(len(predictor_names)):
        predictor_matrix[..., j] = ml_utils.fill_nans_in_predictor_images(
            predictor_matrix[..., j]
        )

    predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(predictor_matrix)
    predictor_matrix, _ = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix,
        normalization_type_string=normalization_type_string)

    print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
    gridded_front_table = fronts_io.read_grid_from_file(
        gridded_front_file_name)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    target_matrix = ml_utils.subset_narr_grid_for_fcn_input(target_matrix)

    if num_classes == 2:
        target_matrix = ml_utils.binarize_front_images(target_matrix)
        target_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres, verbose=False)
    else:
        target_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=target_matrix,
            dilation_distance_metres=dilation_distance_metres, verbose=False)

    predictor_matrix = predictor_matrix.astype('float32')
    target_matrix = to_categorical(target_matrix, num_classes)

    num_instances_by_class = numpy.array(
        [numpy.sum(target_matrix[..., k]) for k in range(num_classes)],
        dtype=int
    )

    print 'Number of instances of each class: {0:s}'.format(
        str(num_instances_by_class)
    )

    return predictor_matrix, target_matrix
