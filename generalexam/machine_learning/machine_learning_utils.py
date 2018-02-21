"""Helper methods for machine learning."""

import copy
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

NARR_COLUMNS_WITHOUT_NAN = numpy.linspace(7, 264, num=264 - 7 + 1, dtype=int)


def _check_predictor_matrix(predictor_matrix, allow_nan=False):
    """Checks predictor matrix for errors.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    :param predictor_matrix: numpy array.  Dimensions must be either T-by-M-by-N
        or T-by-M-by-N-by-C.
    :param allow_nan: Boolean flag.  If True, will allow some elements to be
        NaN.
    """

    if allow_nan:
        error_checking.assert_is_real_numpy_array(predictor_matrix)
    else:
        error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)


def _check_target_matrix(target_matrix):
    """Checks target matrix (containing labels, or "ground truth") for errors.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param target_matrix: T-by-M-by-N numpy array of labels.
    """

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)

    error_checking.assert_is_geq_numpy_array(
        target_matrix, front_utils.NO_FRONT_INTEGER_ID)
    error_checking.assert_is_leq_numpy_array(
        target_matrix, front_utils.COLD_FRONT_INTEGER_ID)


def _check_predictor_and_target_matrices(
        predictor_matrix, target_matrix, allow_nan_predictors=False):
    """Checks both predictor and target matrices together.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    :param predictor_matrix: numpy array.  Dimensions must be either T-by-M-by-N
        or T-by-M-by-N-by-C.
    :param target_matrix: T-by-M-by-N numpy array of labels.
    :param allow_nan_predictors: Boolean flag.  If True, will allow some
        predictor values to be NaN.
    """

    _check_predictor_matrix(predictor_matrix, allow_nan=allow_nan_predictors)
    _check_target_matrix(target_matrix)

    expected_dimensions = numpy.array([predictor_matrix.shape])[:-1]
    error_checking.assert_is_numpy_array(
        target_matrix, exact_dimensions=expected_dimensions)


def _subset_grid(
        full_matrix, center_row, center_column, num_rows_in_half_window,
        num_columns_in_half_window):
    """Takes smaller grid ("window") from the original grid.

    M = number of rows in full grid (unique y-coordinates)
    N = number of columns in full grid (unique x-coordinates)

    m = number of rows in window = 2 * `num_rows_in_half_window` + 1
    n = number of columns in window = 2 * `num_columns_in_half_window` + 1

    If the center point is too near an edge of the full grid, edge padding will
    be done (i.e., the edge value will be repeated as many times as necessary).

    :param full_matrix: M-by-N numpy array.
    :param center_row: Row at center of window.
    :param center_column: Column at center of window.
    :param num_rows_in_half_window: Number of rows in half-window (on both top
        and bottom of center point).
    :param num_columns_in_half_window: Number of columns in half-window (to both
        left and right of center point).
    :return: small_matrix: m-by-n numpy array, created by sampling from
        `full_matrix`.
    """

    # TODO(thunderhoser): make this work for more than 2-D matrices.

    num_rows_in_full_grid = full_matrix.shape[0]
    num_columns_in_full_grid = full_matrix.shape[1]

    first_row = center_row - num_rows_in_half_window
    last_row = center_row + num_rows_in_half_window
    first_column = center_column - num_columns_in_half_window
    last_column = center_column + num_columns_in_half_window

    if first_row < 0:
        num_padding_rows_at_top = 0 - first_row
        first_row = 0
    else:
        num_padding_rows_at_top = 0

    if last_row > num_rows_in_full_grid - 1:
        num_padding_rows_at_bottom = last_row - (num_rows_in_full_grid - 1)
        last_row = num_rows_in_full_grid - 1
    else:
        num_padding_rows_at_bottom = 0

    if first_column < 0:
        num_padding_columns_at_left = 0 - first_column
        first_column = 0
    else:
        num_padding_columns_at_left = 0

    if last_column > num_columns_in_full_grid - 1:
        num_padding_columns_at_right = last_column - (
            num_columns_in_full_grid - 1)
        last_column = num_columns_in_full_grid - 1
    else:
        num_padding_columns_at_right = 0

    small_matrix = full_matrix[
        first_row:(last_row + 1), first_column:(last_column + 1)]
    return numpy.pad(
        small_matrix, pad_width=
        ((num_padding_rows_at_top, num_padding_rows_at_bottom),
         (num_padding_columns_at_left, num_padding_columns_at_right)),
        mode='edge')


def front_table_to_matrices(
        frontal_grid_table, num_grid_rows, num_grid_columns):
    """For each time step, converts lists of frontal points to a grid.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param frontal_grid_table: T-row pandas DataFrame with columns documented in
        `fronts_io.write_narr_grids_to_file`.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :return: frontal_grid_matrix: T-by-M-by-N numpy array.  If
        frontal_grid_matrix[i, j, k] = `front_utils.NO_FRONT_INTEGER_ID`, there
        is no front at grid point [j, k] at the [i]th time step.  If
        `front_utils.WARM_FRONT_INTEGER_ID`, there is a warm front; if
        `front_utils.COLD_FRONT_INTEGER_ID`, there is a cold front.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    num_times = len(frontal_grid_table.index)
    frontal_grid_matrix = None

    for i in range(num_times):
        this_frontal_grid_matrix = front_utils.frontal_points_to_grid(
            frontal_grid_dict=frontal_grid_table.iloc[[i]].to_dict(),
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

        this_frontal_grid_matrix = numpy.reshape(
            this_frontal_grid_matrix, (1, num_grid_rows, num_grid_columns))

        if frontal_grid_matrix is None:
            frontal_grid_matrix = copy.deepcopy(this_frontal_grid_matrix)
        else:
            frontal_grid_matrix = numpy.concatenate(
                (frontal_grid_matrix, this_frontal_grid_matrix), axis=0)

    return frontal_grid_matrix


def binarize_front_labels(frontal_grid_matrix):
    """Turns front labels from multi-class into binary.

    The original (multi-class) labels are "no front," "warm front," and "cold
    front".  The new (binary) labels will just be "front" and "no front".

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param frontal_grid_matrix: T-by-M-by-N numpy array.  If
        frontal_grid_matrix[i, j, k] = `front_utils.NO_FRONT_INTEGER_ID`, there
        is no front at grid point [j, k] at the [i]th time step.  If
        `front_utils.WARM_FRONT_INTEGER_ID`, there is a warm front; if
        `front_utils.COLD_FRONT_INTEGER_ID`, there is a cold front.
    :return: frontal_grid_matrix: T-by-M-by-N numpy array.  If
        frontal_grid_matrix[i, j, k] = `front_utils.ANY_FRONT_INTEGER_ID`, there
        is a front at grid point [j, k] at the [i]th time step.  Otherwise,
        there is none.
    """

    _check_target_matrix(frontal_grid_matrix)

    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.WARM_FRONT_INTEGER_ID
    ] = front_utils.ANY_FRONT_INTEGER_ID
    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.COLD_FRONT_INTEGER_ID
    ] = front_utils.ANY_FRONT_INTEGER_ID

    return frontal_grid_matrix


def remove_nans_from_narr_grid(data_matrix):
    """Removes all grid rows and columns with at least one NaN.

    These rows and columns are always the same in NARR data.

    T = number of time steps
    M = original number of rows (unique grid-point y-coordinates)
    N = original number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    m = new number of rows (after removing NaN's)
    n = new number of columns (after removing NaN's)

    :param data_matrix: Input matrix (numpy array).  May be either T-by-M-by-N
        or T-by-M-by-N-by-C.
    :return: data_matrix: Same as input, except without NaN's.  If original
        dimensions were T-by-M-by-N, new dimensions are T-by-m-by-n.  If
        original dimensions were T-by-M-by-N-by-C, new dimensions are
        T-by-m-by-n-by-C.
    """

    error_checking.assert_is_numpy_array(data_matrix)

    num_dimensions = len(data_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    if num_dimensions == 3:
        expected_dimensions = numpy.array(
            [data_matrix.shape[0], num_grid_rows, num_grid_columns])
    else:
        expected_dimensions = numpy.array(
            [data_matrix.shape[0], num_grid_rows, num_grid_columns,
             data_matrix.shape[3]])

    error_checking.assert_is_numpy_array(
        data_matrix, exact_dimensions=expected_dimensions)

    return numpy.take(data_matrix, NARR_COLUMNS_WITHOUT_NAN, axis=2)


def subdivide_grids(predictor_matrix, target_matrix, num_rows_per_subgrid,
                    num_columns_per_subgrid):
    """Subdivides grids (with both predictors and target) into many subgrids.

    T = number of time steps
    M = number of rows in full grid (unique grid-point y-coordinates)
    N = number of columns in full grid (unique grid-point x-coordinates)
    C = number of image channels (variables)

    m = number of rows in each subgrid
    n = number of columns in each subgrid
    G = T * M * N = number of resulting subgrids

    :param predictor_matrix: Matrix with predictor variables (numpy array).  May
        be either T-by-M-by-N or T-by-M-by-N-by-C.
    :param target_matrix: T-by-M-by-N numpy array with target variable.
    :param num_rows_per_subgrid: Number of rows in each subgrid.  Must be an odd
        number.
    :param num_columns_per_subgrid: Number of columns in each subgrid.  Must be
        an odd number.
    :return: predictor_matrix: Same as input, but with different dimensions.  If
        original dimensions were T-by-M-by-N, new dimensions are G-by-m-by-n.
        If original dimensions were T-by-M-by-N-by-C, new dimensions are
        G-by-m-by-n-by-C.
    :return: target_values: Same as input, but with different dimensions.  Now a
        length-G numpy array.
    """

    # TODO(thunderhoser): Fix this after unit tests on the private subdividing
    # method.

    error_checking.assert_is_numpy_array(predictor_matrix)

    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)

    error_checking.assert_is_integer_numpy_array(target_matrix)
    expected_dimensions = numpy.array([predictor_matrix.shape])[:-1]
    error_checking.assert_is_numpy_array(
        target_matrix, exact_dimensions=expected_dimensions)

    num_grid_rows = predictor_matrix.shape[1]
    error_checking.assert_is_integer(num_rows_per_subgrid)
    error_checking.assert_is_greater(num_rows_per_subgrid, 0)

    num_rows_per_half_subgrid = int(
        numpy.floor(float(num_rows_per_subgrid) / 2))
    num_rows_per_subgrid = 2 * num_rows_per_half_subgrid + 1
    error_checking.assert_is_less_than(num_rows_per_subgrid, num_grid_rows)

    num_grid_columns = predictor_matrix.shape[2]
    error_checking.assert_is_integer(num_columns_per_subgrid)
    error_checking.assert_is_greater(num_columns_per_subgrid, 0)

    num_columns_per_half_subgrid = int(
        numpy.floor(float(num_columns_per_subgrid) / 2))
    num_columns_per_subgrid = 2 * num_columns_per_half_subgrid + 1
    error_checking.assert_is_less_than(
        num_columns_per_subgrid, num_grid_columns)

    num_times = predictor_matrix.shape[0]
    new_predictor_matrix = None
    target_values = numpy.array([], dtype=int)

    for j in range(num_grid_rows):
        this_min_row = j - num_rows_per_half_subgrid
        this_max_row = j + num_rows_per_half_subgrid

        if this_min_row < 0:
            this_top_padding = this_min_row - 0
            this_min_row = 0
        else:
            this_top_padding = 0

        if this_max_row > num_grid_rows - 1:
            this_bottom_padding = this_max_row - (num_grid_rows - 1)
            this_max_row = num_grid_rows - 1
        else:
            this_bottom_padding = 0

        these_rows = numpy.linspace(
            this_min_row, this_max_row, num=this_max_row - this_min_row + 1,
            dtype=int)

        for k in range(num_grid_columns):
            this_min_column = k - num_columns_per_half_subgrid
            this_max_column = k + num_columns_per_half_subgrid

            if this_min_column < 0:
                this_left_padding = this_min_column - 0
                this_min_column = 0
            else:
                this_left_padding = 0

            if this_max_column > num_grid_columns - 1:
                this_right_padding = this_max_column - (num_grid_columns - 1)
                this_max_column = num_grid_columns - 1
            else:
                this_right_padding = 0

            these_columns = numpy.linspace(
                this_min_column, this_max_column,
                num=this_max_column - this_min_column + 1, dtype=int)

            this_predictor_matrix = numpy.take(
                predictor_matrix, these_rows, axis=1)
            this_predictor_matrix = numpy.take(
                this_predictor_matrix, these_columns, axis=2)
            this_predictor_matrix = numpy.pad(
                this_predictor_matrix, pad_width=
                ((this_top_padding, this_bottom_padding),
                 (this_left_padding, this_right_padding)),
                mode='edge')

            if new_predictor_matrix is None:
                new_predictor_matrix = copy.deepcopy(this_predictor_matrix)
            else:
                new_predictor_matrix = numpy.concatenate(
                    (new_predictor_matrix, this_predictor_matrix), axis=0)

            these_target_values = numpy.linspace(
                target_matrix[j, k], target_matrix[j, k], num=num_times,
                dtype=int)
            target_values = numpy.concatenate((
                target_values, these_target_values))

    return new_predictor_matrix, target_values
