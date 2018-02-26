"""Helper methods for machine learning.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples.  Each example is one image or a time sequence of images.
T = number of time steps per example (i.e., number of images in each sequence)
M = number of pixel rows in each image
N = number of pixel columns in each image
C = number of channels (predictor variables) in each image
"""

import copy
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import front_utils

TOLERANCE_FOR_FREQUENCY_SUM = 1e-3

DEFAULT_NUM_SAMPLE_PTS_PER_TIME = 1000

FIRST_NARR_COLUMN_WITHOUT_NAN = 7
LAST_NARR_COLUMN_WITHOUT_NAN = 264
NARR_COLUMNS_WITHOUT_NAN = numpy.linspace(
    FIRST_NARR_COLUMN_WITHOUT_NAN, LAST_NARR_COLUMN_WITHOUT_NAN,
    num=LAST_NARR_COLUMN_WITHOUT_NAN - FIRST_NARR_COLUMN_WITHOUT_NAN + 1,
    dtype=int)

ROW_INDICES_BY_TIME_KEY = 'row_indices_by_time'
COLUMN_INDICES_BY_TIME_KEY = 'column_indices_by_time'

DEFAULT_PREDICTOR_NORMALIZATION_DICT = {
    processed_narr_io.WET_BULB_TEMP_NAME: numpy.array([240., 305.]),
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME: numpy.array([-20., 20.]),
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME: numpy.array([-20., 20.])
}


def _check_predictor_matrix(
        predictor_matrix, allow_nan=False, min_num_dimensions=3,
        max_num_dimensions=5):
    """Checks predictor matrix for errors.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param allow_nan: Boolean flag.  If allow_nan = False and this method finds
        a NaN, it will error out.
    :param min_num_dimensions: Minimum number of dimensions expected in
        `predictor_matrix`.
    :param max_num_dimensions: Max number of dimensions expected in
        `predictor_matrix`.
    """

    if allow_nan:
        error_checking.assert_is_real_numpy_array(predictor_matrix)
    else:
        error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, min_num_dimensions)
    error_checking.assert_is_leq(num_dimensions, max_num_dimensions)


def _check_target_matrix(target_matrix, assert_binary=False, num_dimensions=3):
    """Checks target matrix for errors.

    :param target_matrix: numpy array of targets.  Dimensions may be E x M x N
        or just length E (one-dimensional).  If matrix is binary, there are 2
        possible entries (both integers): `front_utils.NO_FRONT_INTEGER_ID` and
        `front_utils.ANY_FRONT_INTEGER_ID`.  If matrix is multi-class, there are
        3 possible entries (all integers): `front_utils.NO_FRONT_INTEGER_ID`,
        `front_utils.COLD_FRONT_INTEGER_ID`, and
        `front_utils.WARM_FRONT_INTEGER_ID`.
    :param assert_binary: Boolean flag.  If assert_binary = True and this method
        finds a non-binary element, it will error out.
    :param num_dimensions: Number of dimensions expected in `target_matrix`
        (either 3 or 1).
    """

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(
        target_matrix, num_dimensions=num_dimensions)

    error_checking.assert_is_geq_numpy_array(
        target_matrix, front_utils.NO_FRONT_INTEGER_ID)

    if assert_binary:
        error_checking.assert_is_leq_numpy_array(
            target_matrix, front_utils.ANY_FRONT_INTEGER_ID)
    else:
        error_checking.assert_is_leq_numpy_array(
            target_matrix,
            max([front_utils.COLD_FRONT_INTEGER_ID,
                 front_utils.WARM_FRONT_INTEGER_ID]))


def _check_predictor_and_target_matrices(
        predictor_matrix, target_matrix, allow_nan_predictors=False,
        assert_binary_target_matrix=False, num_target_dimensions=3):
    """Checks both predictor and target matrices for errors.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param target_matrix: numpy array of targets.  Dimensions may be E x M x N
        or just length E (one-dimensional).
    :param allow_nan_predictors: Boolean flag.  If allow_nan_predictors = False
        and this method finds a NaN in `predictor_matrix`, it will error out.
    :param assert_binary_target_matrix: Boolean flag.  If assert_binary = True
        and this method finds a non-binary element in `target_matrix`, it will
        error out.
    :param num_target_dimensions: Number of dimensions expected in
        `target_matrix` (either 3 or 1).
    """

    _check_predictor_matrix(predictor_matrix, allow_nan=allow_nan_predictors)
    _check_target_matrix(
        target_matrix, assert_binary=assert_binary_target_matrix,
        num_dimensions=num_target_dimensions)

    if num_target_dimensions == 3:
        expected_target_dimensions = numpy.array(predictor_matrix.shape)[:3]
    else:
        expected_target_dimensions = numpy.array([predictor_matrix.shape[0]])

    error_checking.assert_is_numpy_array(
        target_matrix, exact_dimensions=expected_target_dimensions)


def _check_downsizing_args(
        predictor_matrix, target_matrix, num_rows_in_half_window,
        num_columns_in_half_window, test_mode=False):
    """Checks downsizing arguments for errors.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param target_matrix: numpy array of targets.  Dimensions must be E x M x N.
    :param num_rows_in_half_window: See documentation for
        `_downsize_predictor_images`.
    :param num_columns_in_half_window: See doc for `_downsize_predictor_images`.
    :param test_mode: Boolean flag.  Always keep this False.
    :return: num_downsized_rows: Number of rows in each downsized image.
    :return: num_downsized_columns: Number of columns in each downsized image.
    """

    _check_predictor_and_target_matrices(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        allow_nan_predictors=False, assert_binary_target_matrix=False,
        num_target_dimensions=3)
    error_checking.assert_is_boolean(test_mode)

    num_rows_orig = predictor_matrix.shape[1]
    num_columns_orig = predictor_matrix.shape[2]

    error_checking.assert_is_integer(num_rows_in_half_window)
    error_checking.assert_is_greater(num_rows_in_half_window, 0)
    num_rows_in_subgrid = 2 * num_rows_in_half_window + 1

    if not test_mode:
        error_checking.assert_is_less_than(num_rows_in_subgrid, num_rows_orig)

    error_checking.assert_is_integer(num_columns_in_half_window)
    error_checking.assert_is_greater(num_columns_in_half_window, 0)
    num_columns_in_subgrid = 2 * num_columns_in_half_window + 1

    if not test_mode:
        error_checking.assert_is_less_than(
            num_columns_in_subgrid, num_columns_orig)

    return num_rows_in_subgrid, num_columns_in_subgrid


def _downsize_predictor_images(
        predictor_matrix, center_row, center_column, num_rows_in_half_window,
        num_columns_in_half_window):
    """Downsizes the original images into smaller images ("windows").

    This procedure is done independently for each spatial grid (i.e., each
    example, time step, and channel).

    M = number of pixel rows in each original image
    N = number of pixel columns in each original image

    m = number of pixel rows in each new image = 2 * num_rows_in_half_window + 1
    n = number of pixel columns in each new image =
        2 * num_columns_in_half_window + 1

    If the window runs off the edge of the original image, edge padding will be
    used (i.e., values from the edge of the original image will be repeated to
    fill the window).

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param center_row: Row at center of window.  Each new image will be centered
        around the [i]th row in the original image (i = center_row).
    :param center_column: Column at center of window.  Each new image will be
        centered around the [j]th column in the original image
        (j = center_column).
    :param num_rows_in_half_window: Determines the size of the full window
        (m = 2 * num_rows_in_half_window + 1).
    :param num_columns_in_half_window: Determines the size of the full window
        (n = 2 * num_columns_in_half_window + 1).
    :return: downsized_predictor_matrix: Subset of the input array.  Dimensions
        may be E x m x n, E x m x n x C, or E x m x n x T x C.  Number of
        dimensions is the same as the original image.
    """

    num_rows_orig = predictor_matrix.shape[1]
    num_columns_orig = predictor_matrix.shape[2]

    first_row = center_row - num_rows_in_half_window
    last_row = center_row + num_rows_in_half_window
    first_column = center_column - num_columns_in_half_window
    last_column = center_column + num_columns_in_half_window

    if first_row < 0:
        num_padding_rows_at_top = 0 - first_row
        first_row = 0
    else:
        num_padding_rows_at_top = 0

    if last_row > num_rows_orig - 1:
        num_padding_rows_at_bottom = last_row - (num_rows_orig - 1)
        last_row = num_rows_orig - 1
    else:
        num_padding_rows_at_bottom = 0

    if first_column < 0:
        num_padding_columns_at_left = 0 - first_column
        first_column = 0
    else:
        num_padding_columns_at_left = 0

    if last_column > num_columns_orig - 1:
        num_padding_columns_at_right = last_column - (
            num_columns_orig - 1)
        last_column = num_columns_orig - 1
    else:
        num_padding_columns_at_right = 0

    selected_rows = numpy.linspace(
        first_row, last_row, num=last_row - first_row + 1, dtype=int)
    selected_columns = numpy.linspace(
        first_column, last_column, num=last_column - first_column + 1,
        dtype=int)

    downsized_predictor_matrix = numpy.take(
        predictor_matrix, selected_rows, axis=1)
    downsized_predictor_matrix = numpy.take(
        downsized_predictor_matrix, selected_columns, axis=2)

    pad_width_input_arg = (
        (0, 0), (num_padding_rows_at_top, num_padding_rows_at_bottom),
        (num_padding_columns_at_left, num_padding_columns_at_right))

    num_dimensions = len(predictor_matrix.shape)
    for _ in range(3, num_dimensions):
        pad_width_input_arg += ((0, 0), )

    return numpy.pad(
        downsized_predictor_matrix, pad_width=pad_width_input_arg, mode='edge')


def get_class_weight_dict(class_frequencies):
    """Returns dictionary of class weights.

    This dictionary will be used to weight the loss function for a Keras model.

    K = number of classes

    :param class_frequencies: length-K numpy array, where class_frequencies[i]
        is the frequency of the [i]th class in the training data.
    :return: class_weight_dict: Dictionary, where each key is an integer from
        0...(K - 1) and each value is the corresponding weight.
    :raises: ValueError: if the sum of class frequencies is not 1.
    """

    error_checking.assert_is_numpy_array(class_frequencies, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(class_frequencies, 0.)
    error_checking.assert_is_less_than_numpy_array(class_frequencies, 1.)

    sum_of_class_frequencies = numpy.sum(class_frequencies)
    absolute_diff = numpy.absolute(sum_of_class_frequencies - 1.)
    if absolute_diff > TOLERANCE_FOR_FREQUENCY_SUM:
        error_string = (
            '{0:s}Sum of class frequencies (shown above) should be 1.  Instead,'
            ' got {1:.4f}.').format(str(class_frequencies),
                                    sum_of_class_frequencies)
        raise ValueError(error_string)

    class_weights = 1. / class_frequencies
    class_weights = class_weights / numpy.sum(class_weights)
    class_weight_dict = {}

    for k in range(len(class_weights)):
        class_weight_dict.update({k: class_weights[k]})
    return class_weight_dict


def normalize_predictor_matrix(
        predictor_matrix, normalize_by_example=False, predictor_names=None,
        normalization_dict=DEFAULT_PREDICTOR_NORMALIZATION_DICT,
        percentile_offset=1.):
    """Normalizes predictor matrix.

    Specifically, each value will be normalized as follows.

    new_value = (old_value - min_value) / (max_value - min_value)

    If normalize_by_example = False, min_value and max_value will be the same
    for each example (taken from normalization_dict).

    If normalize_by_example = True, min_value and max_value will be different
    for each example and predictor variable (channel).  Specifically, for the
    [i]th example and [j]th predictor variable, they will be the [k]th and
    [100 - k]th percentiles of the [j]th predictor variable in the [i]th
    example, where k = percentile_offset.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N x C or E x M x N x T x C.
    :param normalize_by_example: Boolean flag (see general discussion above).
    :param predictor_names: [used only if normalize_by_example = False]
        length-C list of predictor names (strings).
    :param normalization_dict: [used only if normalize_by_image = False]
        Dictionary, where each key is the name of a predictor (from
        `predictor_names`) and each value is a length-2 numpy array with
        (min_value, max_value).
    [used only if normalize_by_image = True]
        See k in the general discussion above.
    :return: predictor_matrix: Same as input, except normalized.
    """

    _check_predictor_matrix(
        predictor_matrix, allow_nan=True, min_num_dimensions=4,
        max_num_dimensions=5)
    num_predictors = predictor_matrix.shape[-1]

    error_checking.assert_is_boolean(normalize_by_example)

    if normalize_by_example:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_less_than(percentile_offset, 50.)

        num_examples = predictor_matrix.shape[0]
        for i in range(num_examples):
            for m in range(num_predictors):
                this_min_value = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], percentile_offset)
                this_max_value = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], 100. - percentile_offset)

                predictor_matrix[i, ..., m] = (
                    (predictor_matrix[i, ..., m] - this_min_value) /
                    (this_max_value - this_min_value))
    else:
        error_checking.assert_is_string_list(predictor_names)
        error_checking.assert_is_numpy_array(
            numpy.array(predictor_names),
            exact_dimensions=numpy.array([num_predictors]))

        for m in range(num_predictors):
            this_min_value = normalization_dict[predictor_names[m]][0]
            this_max_value = normalization_dict[predictor_names[m]][1]
            predictor_matrix[..., m] = (
                (predictor_matrix[..., m] - this_min_value) /
                (this_max_value - this_min_value))

    return predictor_matrix


def check_downsized_examples(
        predictor_matrix, target_values, target_times_unix_sec,
        center_grid_rows, center_grid_columns, predictor_names,
        predictor_time_matrix_unix_sec=None, assert_binary_target_matrix=False):
    """Checks downsized machine-learning examples for errors.

    Downsized ML examples are created by
    `downsize_grids_around_selected_points`.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param target_values: length-E numpy array of target values.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param center_grid_rows: length-E numpy array with row index at the center
        of each downsized image.  This can be used to map back to the original
        grid.  For example, if center_grid_rows[i] = j, the center row in the
        [i]th example is the [j]th row in the original grid.
    :param center_grid_columns: Same as above, except for columns.
    :param predictor_names: length-C list with names of predictor variables.
    :param predictor_time_matrix_unix_sec: [used only if predictor_matrix is 5D]
        E-by-T numpy array of predictor times.
    :param assert_binary_target_matrix: See documentation for
        `_check_target_matrix`.
    """

    _check_predictor_and_target_matrices(
        predictor_matrix=predictor_matrix, target_matrix=target_values,
        allow_nan_predictors=False, num_target_dimensions=1,
        assert_binary_target_matrix=assert_binary_target_matrix)

    num_examples = predictor_matrix.shape[0]
    num_predictor_dimensions = len(predictor_matrix.shape)
    if num_predictor_dimensions > 3:
        num_predictor_variables = predictor_matrix.shape[-1]
    else:
        num_predictor_variables = 1

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, exact_dimensions=numpy.array([num_examples]))

    error_checking.assert_is_integer_numpy_array(center_grid_rows)
    error_checking.assert_is_geq_numpy_array(center_grid_rows, 0)
    error_checking.assert_is_numpy_array(
        center_grid_rows, exact_dimensions=numpy.array([num_examples]))

    error_checking.assert_is_integer_numpy_array(center_grid_columns)
    error_checking.assert_is_geq_numpy_array(center_grid_columns, 0)
    error_checking.assert_is_numpy_array(
        center_grid_columns, exact_dimensions=numpy.array([num_examples]))

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names),
        exact_dimensions=numpy.array([num_predictor_variables]))

    if num_predictor_dimensions == 5:
        num_time_steps = predictor_matrix.shape[3]

        error_checking.assert_is_integer_numpy_array(
            predictor_time_matrix_unix_sec)
        error_checking.assert_is_numpy_array(
            predictor_time_matrix_unix_sec,
            exact_dimensions=numpy.array([num_examples, num_time_steps]))


def sample_target_points(
        binary_target_matrix, positive_fraction,
        num_points_per_time=DEFAULT_NUM_SAMPLE_PTS_PER_TIME, test_mode=False):
    """Samples target points (for desired balance of positive/negative cases).

    If the input matrix has either no positive points (points intersected by a
    front) or no negative points, this method will return None.

    P_i = number of grid points selected at the [i]th time

    :param binary_target_matrix: numpy array of targets.  Dimensions must be
        E x M x N.
    :param positive_fraction: Fraction of positive cases in resulting sample.
    :param num_points_per_time: Number of points to take from each example (for
        target variable, one example = one time).
    :param test_mode: Boolean flag.  Always leave this False.
    :return: target_point_dict: Dictionary with the following keys.
    target_point_dict['row_indices_by_time']: length-T list, where the [i]th
        element is a numpy array (length P_i) with row indices of grid points
        selected at the [i]th time.
    target_point_dict['column_indices_by_time']: Same as above, except for
        columns.
    """

    # TODO(thunderhoser): Allow non-binary target labels.
    _check_target_matrix(binary_target_matrix, assert_binary=True)

    error_checking.assert_is_greater(positive_fraction, 0.)
    error_checking.assert_is_less_than(positive_fraction, 1.)
    error_checking.assert_is_integer(num_points_per_time)
    error_checking.assert_is_boolean(test_mode)

    num_times = binary_target_matrix.shape[0]
    num_points_to_sample = num_points_per_time * num_times
    error_checking.assert_is_geq(num_points_to_sample, 2)

    num_positive_cases = int(
        numpy.round(positive_fraction * num_points_to_sample))
    num_positive_cases = max([num_positive_cases, 0])
    num_positive_cases = min([num_positive_cases, num_points_to_sample - 1])
    num_negative_cases = num_points_to_sample - num_positive_cases

    positive_flags_linear = (
        numpy.reshape(binary_target_matrix, binary_target_matrix.size) ==
        front_utils.ANY_FRONT_INTEGER_ID)
    positive_indices_linear = numpy.where(positive_flags_linear)[0]
    negative_indices_linear = numpy.where(
        numpy.invert(positive_flags_linear))[0]

    if not len(positive_indices_linear):
        return None
    if not len(negative_indices_linear):
        return None

    if len(positive_indices_linear) < num_positive_cases:
        num_positive_cases = len(positive_indices_linear)
        num_negative_cases = int(numpy.round(
            num_positive_cases * (1. - positive_fraction) / positive_fraction))

    if len(negative_indices_linear) < num_negative_cases:
        num_negative_cases = len(negative_indices_linear)
        num_positive_cases = int(numpy.round(
            num_negative_cases * positive_fraction / (1. - positive_fraction)))

    if test_mode:
        positive_indices_linear = positive_indices_linear[:num_positive_cases]
    else:
        positive_indices_linear = numpy.random.choice(
            positive_indices_linear, size=num_positive_cases, replace=False)

    if test_mode:
        negative_indices_linear = negative_indices_linear[:num_negative_cases]
    else:
        negative_indices_linear = numpy.random.choice(
            negative_indices_linear, size=num_negative_cases, replace=False)

    positive_time_indices, positive_row_indices, positive_column_indices = (
        numpy.unravel_index(
            positive_indices_linear, binary_target_matrix.shape))
    negative_time_indices, negative_row_indices, negative_column_indices = (
        numpy.unravel_index(
            negative_indices_linear, binary_target_matrix.shape))

    row_indices_by_time = [numpy.array([], dtype=int)] * num_times
    column_indices_by_time = [numpy.array([], dtype=int)] * num_times

    for i in range(num_times):
        these_positive_indices = numpy.where(positive_time_indices == i)[0]
        row_indices_by_time[i] = positive_row_indices[these_positive_indices]
        column_indices_by_time[i] = positive_column_indices[
            these_positive_indices]

        these_negative_indices = numpy.where(negative_time_indices == i)[0]
        row_indices_by_time[i] = numpy.concatenate((
            row_indices_by_time[i],
            negative_row_indices[these_negative_indices]))
        column_indices_by_time[i] = numpy.concatenate((
            column_indices_by_time[i],
            negative_column_indices[these_negative_indices]))

    return {
        ROW_INDICES_BY_TIME_KEY: row_indices_by_time,
        COLUMN_INDICES_BY_TIME_KEY: column_indices_by_time
    }


def front_table_to_images(
        frontal_grid_table, num_rows_per_image, num_columns_per_image):
    """For each time step, converts list of frontal points to an image.

    A "frontal point" is a grid point (pixel) intersected by a front.

    E = number of examples = number of time steps.  For target variable, each
    example is one time step.

    :param frontal_grid_table: E-row pandas DataFrame with columns documented in
        `fronts_io.write_narr_grids_to_file`.
    :param num_rows_per_image: Number of pixel rows in each image (M).
    :param num_columns_per_image: Number of pixel columns in each image (N).
    :return: frontal_grid_matrix: E-by-M-by-N numpy array with 3 possible
        entries (see documentation for `_check_target_matrix`).
    """

    error_checking.assert_is_integer(num_rows_per_image)
    error_checking.assert_is_greater(num_rows_per_image, 0)
    error_checking.assert_is_integer(num_columns_per_image)
    error_checking.assert_is_greater(num_columns_per_image, 0)

    if frontal_grid_table.empty:
        return numpy.full(
            (1, num_rows_per_image, num_columns_per_image), 0, dtype=int)

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
            num_grid_rows=num_rows_per_image,
            num_grid_columns=num_columns_per_image)

        this_frontal_grid_matrix = numpy.reshape(
            this_frontal_grid_matrix,
            (1, num_rows_per_image, num_columns_per_image))

        if frontal_grid_matrix is None:
            frontal_grid_matrix = copy.deepcopy(this_frontal_grid_matrix)
        else:
            frontal_grid_matrix = numpy.concatenate(
                (frontal_grid_matrix, this_frontal_grid_matrix), axis=0)

    return frontal_grid_matrix


def binarize_front_images(frontal_grid_matrix):
    """Converts front images from multi-class to binary.

    The original (multi-class) labels are "no front," "warm front," and
    "cold front".

    The new (binary) labels are "front" and "no front".

    :param frontal_grid_matrix: E-by-M-by-N numpy array with 3 possible
        entries (see documentation for `_check_target_matrix`).
    :return: frontal_grid_matrix: E-by-M-by-N numpy array with 2 possible
        entries (see documentation for `_check_target_matrix`).
    """

    _check_target_matrix(frontal_grid_matrix, assert_binary=False)

    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.WARM_FRONT_INTEGER_ID
        ] = front_utils.ANY_FRONT_INTEGER_ID
    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.COLD_FRONT_INTEGER_ID
        ] = front_utils.ANY_FRONT_INTEGER_ID

    return frontal_grid_matrix


def dilate_target_images(
        binary_target_matrix, num_pixels_in_half_window, verbose=True):
    """Dilates target image at each time step.

    :param binary_target_matrix: E-by-M-by-N numpy array with 2 possible
        entries (see documentation for `_check_target_matrix`).
    :param num_pixels_in_half_window: Number of pixels in dilation half-window.
    :param verbose: Boolean flag.  If True, this method will print progress
        messages.
    :return: binary_target_matrix: Dilated version of input.
    """

    _check_target_matrix(binary_target_matrix, assert_binary=True)
    error_checking.assert_is_boolean(verbose)

    num_times = binary_target_matrix.shape[0]
    for i in range(num_times):
        if verbose:
            print ('Dilating target grid at {0:d}th of {1:d} time '
                   'steps...').format(i + 1, num_times)

        binary_target_matrix[i, :, :] = front_utils.dilate_binary_image(
            binary_matrix=binary_target_matrix[i, :, :],
            dilation_half_width_in_grid_cells=num_pixels_in_half_window,
            include_diagonal_neighbours=True)

    return binary_target_matrix


def stack_predictor_variables(tuple_of_3d_predictor_matrices):
    """Stacks images with different predictor variables.

    C = number of predictor variables (channels)

    :param tuple_of_3d_predictor_matrices: length-C tuple, where each element is
        an E-by-M-by-N numpy array of predictor images.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor
        images.
    """

    predictor_matrix = numpy.stack(tuple_of_3d_predictor_matrices, axis=-1)
    _check_predictor_matrix(
        predictor_matrix, allow_nan=True, min_num_dimensions=4,
        max_num_dimensions=4)

    return predictor_matrix


def stack_time_steps(tuple_of_4d_predictor_matrices):
    """Stacks predictor images from different time steps.

    T = number of time steps

    :param tuple_of_4d_predictor_matrices: length-T tuple, where each element is
        an E-by-M-by-N-by-C numpy array of predictor images.
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        images.
    """

    predictor_matrix = numpy.stack(tuple_of_4d_predictor_matrices, axis=-2)
    _check_predictor_matrix(
        predictor_matrix, allow_nan=True, min_num_dimensions=5,
        max_num_dimensions=5)

    return predictor_matrix


def remove_nans_from_narr_grid(narr_matrix):
    """Removes NaN's from image defined on NARR grid.

    M = number of pixel rows in each original image
    N = number of pixel columns in each original image

    m = number of pixel rows in each new image
    n = number of pixel columns in each new image

    :param narr_matrix: numpy array of either predictor images or target images.
        Dimensions may be E x M x N, E x M x N x C, or E x M x N x T x C.
    :return: narr_matrix: Subset of the input array.  Dimensions
        may be E x m x n, E x m x n x C, or E x m x n x T x C.  Number of
        dimensions is the same as the original image.
    """

    error_checking.assert_is_numpy_array(narr_matrix)

    num_dimensions = len(narr_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 5)

    num_rows_in_narr, num_columns_in_narr = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    expected_dimensions = (
        narr_matrix.shape[0], num_rows_in_narr, num_columns_in_narr)
    for i in range(3, num_dimensions):
        expected_dimensions += (narr_matrix.shape[i],)

    error_checking.assert_is_numpy_array(
        narr_matrix, exact_dimensions=numpy.array(expected_dimensions))

    return numpy.take(narr_matrix, NARR_COLUMNS_WITHOUT_NAN, axis=2)


def downsize_grids_around_selected_points(
        predictor_matrix, target_matrix, num_rows_in_half_window,
        num_columns_in_half_window, target_point_dict, verbose=True,
        test_mode=False):
    """Creates small images ("windows") around each selected point in full grid.

    E = number of examples.  Each example is one image or a time sequence of
        images.
    T = number of time steps per example (i.e., number of images in each
        sequence)
    M = number of rows in full grid
    N = number of columns in full grid
    C = number of channels (predictor variables) in each image

    m = number of rows in small grid = 2 * num_rows_in_half_window + 1
    n = number of columns in small grid = 2 * num_columns_in_half_window + 1
    S = number of downsized examples created

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x T x C.
    :param target_matrix: numpy array of targets.  Dimensions must be E x M x N.
    :param num_rows_in_half_window: See documentation for
        `downsize_predictor_images`.
    :param num_columns_in_half_window: See doc for `downsize_predictor_images`.
    :param target_point_dict: Dictionary created by `sample_target_points`.
    :param verbose: Boolean flag.  If True, this method will print progress
        messages.
    :param test_mode: Boolean flag.  Always leave this False.
    :return: predictor_matrix: numpy array of predictor images.  Dimensions may
        be S x m x n, S x m x n x C, or S x m x n x T x C.
    :return: target_values: length-S numpy array of corresponding labels.
    :return: example_indices: length-S numpy array with example index for each
        small grid.  This can be used to map back to the original examples.
    :return: center_grid_rows: length-S numpy array with row index at the center
        of each small example.  This can be used to map back to the original
        grid.  For example, if center_grid_rows[i] = j, the center row in the
        [i]th example is the [j]th row in the original grid.
    :return: center_grid_columns: Same as above, but for columns.
    """

    num_downsized_rows, num_downsized_columns = _check_downsizing_args(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        num_rows_in_half_window=num_rows_in_half_window,
        num_columns_in_half_window=num_columns_in_half_window,
        test_mode=test_mode)

    num_full_examples = predictor_matrix.shape[0]
    num_downsized_examples = 0
    for i in range(num_full_examples):
        num_downsized_examples += len(
            target_point_dict[ROW_INDICES_BY_TIME_KEY][i])

    new_dimensions = (
        num_downsized_examples, num_downsized_rows, num_downsized_columns)
    new_dimensions += predictor_matrix.shape[3:]
    new_predictor_matrix = numpy.full(new_dimensions, numpy.nan)

    target_values = numpy.full(num_downsized_examples, -1, dtype=int)
    example_indices = numpy.full(num_downsized_examples, -1, dtype=int)
    center_grid_rows = numpy.full(num_downsized_examples, -1, dtype=int)
    center_grid_columns = numpy.full(num_downsized_examples, -1, dtype=int)
    last_downsized_example_index = -1

    for i in range(num_full_examples):
        if verbose:
            print (
                'Downsizing images around selected points for {0:d}th of {1:d} '
                'full-size examples...').format(i + 1, num_full_examples)

        these_target_point_rows = target_point_dict[ROW_INDICES_BY_TIME_KEY][i]
        these_target_point_columns = target_point_dict[
            COLUMN_INDICES_BY_TIME_KEY][i]
        this_num_target_points = len(these_target_point_rows)

        for j in range(this_num_target_points):
            new_predictor_matrix[[last_downsized_example_index + 1], ...] = (
                _downsize_predictor_images(
                    predictor_matrix=predictor_matrix[[i], ...],
                    center_row=these_target_point_rows[j],
                    center_column=these_target_point_columns[j],
                    num_rows_in_half_window=num_rows_in_half_window,
                    num_columns_in_half_window=num_columns_in_half_window))

            target_values[last_downsized_example_index + 1] = target_matrix[
                i, these_target_point_rows[j], these_target_point_columns[j]]
            example_indices[last_downsized_example_index + 1] = i

            center_grid_rows[
                last_downsized_example_index + 1] = these_target_point_rows[j]
            center_grid_columns[
                last_downsized_example_index + 1
            ] = these_target_point_columns[j]

            last_downsized_example_index += 1

    return (new_predictor_matrix, target_values, example_indices,
            center_grid_rows, center_grid_columns)
