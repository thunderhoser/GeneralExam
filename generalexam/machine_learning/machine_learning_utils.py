"""Helper methods for machine learning."""

import copy
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

FIRST_NARR_COLUMN_WITHOUT_NAN = 7
LAST_NARR_COLUMN_WITHOUT_NAN = 264
NARR_COLUMNS_WITHOUT_NAN = numpy.linspace(
    FIRST_NARR_COLUMN_WITHOUT_NAN, LAST_NARR_COLUMN_WITHOUT_NAN,
    num=LAST_NARR_COLUMN_WITHOUT_NAN - FIRST_NARR_COLUMN_WITHOUT_NAN + 1,
    dtype=int)


def _check_predictor_matrix(predictor_matrix, allow_nan=False):
    """Checks predictor matrix for errors.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    :param predictor_matrix: numpy array with predictor variables.  Dimensions
        may be either T-by-M-by-N or T-by-M-by-N-by-C.
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


def _check_target_matrix(target_matrix, assert_binary=False):
    """Checks target matrix (containing labels, or "ground truth") for errors.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param target_matrix: T-by-M-by-N numpy array of front labels.  If
        target_matrix[i, j, k] = `front_utils.NO_FRONT_INTEGER_ID`, there is no
        front at grid point [j, k] and the [i]th time step.  If
        `front_utils.ANY_FRONT_INTEGER_ID`, there is a front, the type of which
        is not specified.  If `front_utils.WARM_FRONT_INTEGER_ID`, there is a
        warm front; if `front_utils.COLD_FRONT_INTEGER_ID`, there is a cold
        front.
    :param assert_binary: Boolean flag.  If True, the matrix must be binary,
        which means that the only valid entries are
        `front_utils.NO_FRONT_INTEGER_ID` and `front_utils.ANY_FRONT_INTEGER_ID`.
    """

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)

    error_checking.assert_is_geq_numpy_array(
        target_matrix, front_utils.NO_FRONT_INTEGER_ID)

    if assert_binary:
        error_checking.assert_is_leq_numpy_array(
            target_matrix, front_utils.ANY_FRONT_INTEGER_ID)
    else:
        max_entry = max([front_utils.COLD_FRONT_INTEGER_ID,
                         front_utils.WARM_FRONT_INTEGER_ID])

        error_checking.assert_is_leq_numpy_array(target_matrix, max_entry)


def _check_predictor_and_target_matrices(
        predictor_matrix, target_matrix, allow_nan_predictors=False,
        assert_binary_target_matrix=False):
    """Checks both predictor and target matrices together.

    :param predictor_matrix: See documentation for `_check_predictor_matrix`.
    :param target_matrix: See documentation for `_check_target_matrix`.
    :param allow_nan_predictors: Boolean flag.  If True, will allow some
        predictor values to be NaN.
    :param assert_binary_target_matrix: See documentation for
        `_check_target_matrix`.
    """

    _check_predictor_matrix(predictor_matrix, allow_nan=allow_nan_predictors)
    _check_target_matrix(
        target_matrix, assert_binary=assert_binary_target_matrix)

    expected_dimensions = numpy.array(predictor_matrix.shape)[:-1]
    error_checking.assert_is_numpy_array(
        target_matrix, exact_dimensions=expected_dimensions)


def _subset_grid(
        full_grid_matrix, center_row, center_column, num_rows_in_half_window,
        num_columns_in_half_window):
    """Takes smaller grid ("window") from the original grid.

    M = number of rows in full grid (unique grid-point y-coordinates)
    N = number of columns in full grid (unique grid-point x-coordinates)

    m = number of rows in window = 2 * `num_rows_in_half_window` + 1
    n = number of columns in window = 2 * `num_columns_in_half_window` + 1

    If the window runs off the edge of the grid, edge padding will be used
    (i.e., values from the edge of the full grid will repeated to fill the
    window).

    In this case, "rows" correspond to the second dimension (axis = 1) and
    "columns" correspond to the third dimensions (axis = 2).

    :param full_grid_matrix: Input matrix (numpy array) defined over the full
        grid.  The second dimension should have length M; the third should have
        length N.
    :param center_row: Row at center of window.
    :param center_column: Column at center of window.
    :param num_rows_in_half_window: Number of rows in half-window (on both top
        and bottom of center point).
    :param num_columns_in_half_window: Number of columns in half-window (to both
        left and right of center point).
    :return: small_grid_matrix: Subset version of input array, where the second
        dimension has length m; third dimension has length n.
    """

    num_rows_in_full_grid = full_grid_matrix.shape[1]
    num_columns_in_full_grid = full_grid_matrix.shape[2]

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

    these_rows = numpy.linspace(
        first_row, last_row, num=last_row - first_row + 1, dtype=int)
    these_columns = numpy.linspace(
        first_column, last_column, num=last_column - first_column + 1,
        dtype=int)

    small_grid_matrix = numpy.take(full_grid_matrix, these_rows, axis=1)
    small_grid_matrix = numpy.take(small_grid_matrix, these_columns, axis=2)

    pad_width_input_arg = (
        (0, 0), (num_padding_rows_at_top, num_padding_rows_at_bottom),
        (num_padding_columns_at_left, num_padding_columns_at_right))

    num_dimensions = len(full_grid_matrix.shape)
    for _ in range(3, num_dimensions):
        pad_width_input_arg += ((0, 0), )

    return numpy.pad(
        small_grid_matrix, pad_width=pad_width_input_arg, mode='edge')


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
    :return: frontal_grid_matrix: T-by-M-by-N numpy array with 3 possible
        entries.  See documentation for `_check_target_matrix`.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    num_times = len(frontal_grid_table.index)
    frontal_grid_matrix = None

    for i in range(num_times):
        this_frontal_grid_dict = {
            front_utils.WARM_FRONT_ROW_INDICES_COLUMN: frontal_grid_table[
                front_utils.WARM_FRONT_ROW_INDICES_COLUMN].values[i],
            front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN: frontal_grid_table[
                front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN].values[i],
            front_utils.COLD_FRONT_ROW_INDICES_COLUMN: frontal_grid_table[
                front_utils.COLD_FRONT_ROW_INDICES_COLUMN].values[i],
            front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN: frontal_grid_table[
                front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN].values[i]
        }

        this_frontal_grid_matrix = front_utils.frontal_points_to_grid(
            frontal_grid_dict=this_frontal_grid_dict,
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

    :param frontal_grid_matrix: T-by-M-by-N numpy array with 3 possible
        entries.  See documentation for `_check_target_matrix`.
    :return: frontal_grid_matrix: T-by-M-by-N numpy array with 2 possible
        entries.  See documentation for `_check_target_matrix`.
    """

    _check_target_matrix(frontal_grid_matrix, assert_binary=False)

    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.WARM_FRONT_INTEGER_ID
    ] = front_utils.ANY_FRONT_INTEGER_ID
    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.COLD_FRONT_INTEGER_ID
    ] = front_utils.ANY_FRONT_INTEGER_ID

    return frontal_grid_matrix


def stack_predictor_variables(tuple_of_predictor_matrices):
    """Stacks matrices with different predictor variables.

    T = number of time steps
    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    :param tuple_of_predictor_matrices: length-C tuple, where each element is a
        T-by-M-by-N numpy array with values of one predictor variable.
    :return: predictor_matrix: T-by-M-by-N-by-C numpy array.
    """

    predictor_matrix = numpy.stack(tuple_of_predictor_matrices, axis=-1)
    _check_predictor_matrix(predictor_matrix, allow_nan=True)
    return predictor_matrix


def remove_nans_from_narr_grid(narr_matrix):
    """Removes all grid rows and columns with at least one NaN.

    These rows and columns are always the same in NARR data.

    T = number of time steps
    M = original number of rows (unique grid-point y-coordinates)
    N = original number of columns (unique grid-point x-coordinates)
    C = number of image channels (variables)

    m = new number of rows (after removing NaN's)
    n = new number of columns (after removing NaN's)

    :param narr_matrix: Input matrix (numpy array).  May be either T-by-M-by-N
        or T-by-M-by-N-by-C.
    :return: narr_matrix: Same as input, except without NaN's.  If original
        dimensions were T-by-M-by-N, new dimensions are T-by-m-by-n.  If
        original dimensions were T-by-M-by-N-by-C, new dimensions are
        T-by-m-by-n-by-C.
    """

    error_checking.assert_is_numpy_array(narr_matrix)

    num_dimensions = len(narr_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    if num_dimensions == 3:
        expected_dimensions = numpy.array(
            [narr_matrix.shape[0], num_grid_rows, num_grid_columns])
    else:
        expected_dimensions = numpy.array(
            [narr_matrix.shape[0], num_grid_rows, num_grid_columns,
             narr_matrix.shape[3]])

    error_checking.assert_is_numpy_array(
        narr_matrix, exact_dimensions=expected_dimensions)

    return numpy.take(narr_matrix, NARR_COLUMNS_WITHOUT_NAN, axis=2)


def subset_grids_around_each_point(
        predictor_matrix, target_matrix, num_rows_in_half_window,
        num_columns_in_half_window, test_mode=False):
    """At each point P in full grid, takes smaller grid ("window") around P.

    For more details, see `subset_grid`, which is called by this method for each
    point in the full grid.

    T = number of time steps
    M = number of rows in full grid (unique grid-point y-coordinates)
    N = number of columns in full grid (unique grid-point x-coordinates)
    C = number of image channels (variables)

    m = number of rows in window = 2 * `num_rows_in_half_window` + 1
    n = number of columns in window = 2 * `num_columns_in_half_window` + 1
    G = T * M * N = number of resulting subgrids.

    :param predictor_matrix: numpy array with predictor variables.  Dimensions
        may be either T-by-M-by-N or T-by-M-by-N-by-C.
    :param target_matrix: T-by-M-by-N numpy array with labels ("ground truth").
    :param num_rows_in_half_window: Determines number of rows in each subgrid
        (see general discussion above).
    :param num_columns_in_half_window: Determines number of columns in each
        subgrid (see general discussion above).
    :param test_mode: Boolean flag.  Always leave this False.
    :return: predictor_matrix: numpy array with predictor variables.  Dimensions
        may be either G-by-m-by-n or G-by-m-by-n-by-C.
    :return: target_values: length-G numpy array of corresponding labels.
    """

    _check_predictor_and_target_matrices(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        allow_nan_predictors=False, assert_binary_target_matrix=False)
    error_checking.assert_is_boolean(test_mode)

    num_rows_in_full_grid = predictor_matrix.shape[1]
    num_columns_in_full_grid = predictor_matrix.shape[2]

    error_checking.assert_is_integer(num_rows_in_half_window)
    error_checking.assert_is_greater(num_rows_in_half_window, 0)
    num_rows_in_subgrid = 2 * num_rows_in_half_window + 1

    if not test_mode:
        error_checking.assert_is_less_than(
            num_rows_in_subgrid, num_rows_in_full_grid)

    error_checking.assert_is_integer(num_columns_in_half_window)
    error_checking.assert_is_greater(num_columns_in_half_window, 0)
    num_columns_in_subgrid = 2 * num_columns_in_half_window + 1

    if not test_mode:
        error_checking.assert_is_less_than(
            num_columns_in_subgrid, num_columns_in_full_grid)

    new_predictor_matrix = None
    target_values = numpy.array([], dtype=int)

    for j in range(num_rows_in_full_grid):
        for k in range(num_columns_in_full_grid):
            this_new_predictor_matrix = _subset_grid(
                full_grid_matrix=predictor_matrix, center_row=j,
                center_column=k,
                num_rows_in_half_window=num_rows_in_half_window,
                num_columns_in_half_window=num_columns_in_half_window)

            if new_predictor_matrix is None:
                new_predictor_matrix = copy.deepcopy(this_new_predictor_matrix)
            else:
                new_predictor_matrix = numpy.concatenate(
                    (new_predictor_matrix, this_new_predictor_matrix), axis=0)

            target_values = numpy.concatenate((
                target_values, target_matrix[:, j, k]))

    return new_predictor_matrix, target_values
