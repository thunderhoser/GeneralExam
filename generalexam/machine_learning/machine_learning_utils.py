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
import pickle
import os.path
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import utils
from generalexam.ge_utils import front_utils

TOLERANCE_FOR_FREQUENCY_SUM = 1e-3
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

NARR_GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME
)[0]

# Subset of NARR grid for FCN (fully convolutional net).  This gives dimensions
# of 272 x 336, which are integer-divisible by 2 many times, which obviates the
# need for padding and cropping layers.
FIRST_NARR_COLUMN_FOR_FCN_INPUT = 0
LAST_NARR_COLUMN_FOR_FCN_INPUT = 335
FIRST_NARR_ROW_FOR_FCN_INPUT = 5
LAST_NARR_ROW_FOR_FCN_INPUT = 276

ROW_INDICES_BY_TIME_KEY = 'row_indices_by_time'
COLUMN_INDICES_BY_TIME_KEY = 'column_indices_by_time'

PROBABILITY_MATRIX_KEY = 'class_probability_matrix'
TARGET_TIMES_KEY = 'target_times_unix_sec'
TARGET_MATRIX_KEY = 'target_matrix'
MODEL_FILE_NAME_KEY = 'model_file_name'
USED_ISOTONIC_KEY = 'used_isotonic_regression'

MINMAX_STRING = 'minmax'
Z_SCORE_STRING = 'z_score'
VALID_NORM_TYPE_STRINGS = [MINMAX_STRING, Z_SCORE_STRING]

MIN_VALUE_MATRIX_KEY = 'min_value_matrix'
MAX_VALUE_MATRIX_KEY = 'max_value_matrix'
MEAN_VALUE_MATRIX_KEY = 'mean_value_matrix'
STDEV_MATRIX_KEY = 'standard_deviation_matrix'


def _check_full_narr_matrix(full_narr_matrix):
    """Checks input matrix (must contain data on full NARR grid).

    :param full_narr_matrix: numpy array of either predictor images or target
        images.  Dimensions may be E x M x N, E x M x N x C,
        or E x M x N x T x C.
    """

    error_checking.assert_is_numpy_array(full_narr_matrix)

    num_dimensions = len(full_narr_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 5)

    num_rows_in_narr, num_columns_in_narr = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    expected_dimensions = (
        full_narr_matrix.shape[0], num_rows_in_narr, num_columns_in_narr)

    for i in range(3, num_dimensions):
        expected_dimensions += (full_narr_matrix.shape[i],)

    expected_dimensions = numpy.array(expected_dimensions, dtype=int)
    error_checking.assert_is_numpy_array(
        full_narr_matrix, exact_dimensions=expected_dimensions)


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
        possible entries (both integers): `front_utils.NO_FRONT_ENUM` and
        `front_utils.ANY_FRONT_ENUM`.  If matrix is multi-class, there are
        3 possible entries (all integers): `front_utils.NO_FRONT_ENUM`,
        `front_utils.COLD_FRONT_ENUM`, and
        `front_utils.WARM_FRONT_ENUM`.
    :param assert_binary: Boolean flag.  If assert_binary = True and this method
        finds a non-binary element, it will error out.
    :param num_dimensions: Number of dimensions expected in `target_matrix`
        (either 3 or 1).
    """

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(
        target_matrix, num_dimensions=num_dimensions)

    error_checking.assert_is_geq_numpy_array(
        target_matrix, front_utils.NO_FRONT_ENUM)

    if assert_binary:
        error_checking.assert_is_leq_numpy_array(
            target_matrix, front_utils.ANY_FRONT_ENUM)
    else:
        error_checking.assert_is_leq_numpy_array(
            target_matrix,
            max([front_utils.COLD_FRONT_ENUM, front_utils.WARM_FRONT_ENUM])
        )


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
        (num_padding_columns_at_left, num_padding_columns_at_right)
    )

    num_dimensions = len(predictor_matrix.shape)
    for _ in range(3, num_dimensions):
        pad_width_input_arg += ((0, 0), )

    return numpy.pad(
        downsized_predictor_matrix, pad_width=pad_width_input_arg, mode='edge')


def _class_fractions_to_num_points(class_fractions, num_points_total):
    """For each class, converts fraction of total points to number of points.

    K = number of classes

    :param class_fractions: length-K numpy array, where the [i]th element is the
        desired fraction for the [i]th class.  Must sum to 1.0.
    :param num_points_total: Total number of points to sample.
    :return: num_points_by_class: length-K numpy array, where the [i]th element
        is the number of points to sample for the [i]th class.
    """

    num_classes = len(class_fractions)
    num_points_by_class = numpy.full(num_classes, -1, dtype=int)

    for i in range(num_classes - 1):
        num_points_by_class[i] = int(numpy.round(
            class_fractions[i] * num_points_total
        ))
        num_points_by_class[i] = max([num_points_by_class[i], 1])

        num_points_left = num_points_total - numpy.sum(num_points_by_class[:i])
        num_classes_left = (num_classes - 1) - i
        num_points_by_class[i] = min(
            [num_points_by_class[i], num_points_left - num_classes_left]
        )

    num_points_by_class[-1] = num_points_total - numpy.sum(
        num_points_by_class[:-1])

    return num_points_by_class


def _check_normalization_type(normalization_type_string):
    """Ensures that normalization type is valid.

    :param normalization_type_string: Normalization type.
    :raises: ValueError: if
        `normalization_type_string not in VALID_NORM_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(normalization_type_string)

    if normalization_type_string not in VALID_NORM_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid normalization types (listed above) do not include'
            ' "{1:s}".'
        ).format(str(VALID_NORM_TYPE_STRINGS), normalization_type_string)

        raise ValueError(error_string)


def check_narr_mask(mask_matrix):
    """Error-checks NARR mask.

    :param mask_matrix: M-by-N numpy array of integers (0 or 1).  If
        mask_matrix[i, j] = 0, grid cell [i, j] will never be used as the center
        of a downsized grid.
    """

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    error_checking.assert_is_integer_numpy_array(mask_matrix)
    error_checking.assert_is_geq_numpy_array(mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(mask_matrix, 1)

    expected_dimensions = numpy.array(
        [num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        mask_matrix, exact_dimensions=expected_dimensions)


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
            '\n{0:s}\nSum of class frequencies (shown above) should be 1.  '
            'Instead, got {1:.4f}.'
        ).format(str(class_frequencies), sum_of_class_frequencies)

        raise ValueError(error_string)

    class_weights = 1. / class_frequencies
    class_weights = class_weights / numpy.sum(class_weights)
    class_weight_dict = {}

    for k in range(len(class_weights)):
        class_weight_dict.update({k: class_weights[k]})

    return class_weight_dict


def normalize_predictors(
        predictor_matrix, normalization_type_string=MINMAX_STRING,
        percentile_offset=1.):
    """Normalizes predictor variables.

    If normalization_type_string = "z_score", each variable is normalized with
    the following equation, where `mean_value` and `standard_deviation` are the
    mean and standard deviation over the given example.

    normalized_value = (unnormalized_value - mean_value) / standard_deviation

    If normalization_type_string = "minmax", each variable is normalized with
    the following equation, where `min_value` is the [q]th percentile over the
    given example; `max_value` is the [100 - q]th percentile over the given
    example; and q = `percentile_offset`.

    normalized_value = (unnormalized_value - min_value) /
                       (max_value - min_value)

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N x C or E x M x N x T x C.
    :param normalization_type_string: See general discussion above.
    :param percentile_offset: See general discussion above.
    :return: predictor_matrix: Normalized version of input (same dimensions).
    :return: normalization_dict: Dictionary with the following keys.
    normalization_dict['min_value_matrix']: E-by-C numpy array of minimum values
        (actually [q]th percentiles).  If normalization_type_string != "minmax",
        this is None.
    normalization_dict['max_value_matrix']: Same but for max values (actually
        [100 - q]th percentiles).
    normalization_dict['mean_value_matrix']: E-by-C numpy array of mean values.
        If normalization_type_string != "z_score", this is None.
    normalization_dict['standard_deviation_matrix']: Same but for standard
        deviations.
    """

    _check_predictor_matrix(
        predictor_matrix, allow_nan=True, min_num_dimensions=4,
        max_num_dimensions=5)

    _check_normalization_type(normalization_type_string)

    num_examples = predictor_matrix.shape[0]
    num_predictors = predictor_matrix.shape[-1]

    min_value_matrix = None
    max_value_matrix = None
    mean_value_matrix = None
    standard_deviation_matrix = None

    if normalization_type_string == MINMAX_STRING:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_less_than(percentile_offset, 50.)

        min_value_matrix = numpy.full((num_examples, num_predictors), numpy.nan)
        max_value_matrix = numpy.full((num_examples, num_predictors), numpy.nan)

        for i in range(num_examples):
            for m in range(num_predictors):
                min_value_matrix[i, m] = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], percentile_offset)
                max_value_matrix[i, m] = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], 100. - percentile_offset)

                predictor_matrix[i, ..., m] = (
                    (predictor_matrix[i, ..., m] - min_value_matrix[i, m]) /
                    (max_value_matrix[i, m] - min_value_matrix[i, m])
                )
    else:
        mean_value_matrix = numpy.full(
            (num_examples, num_predictors), numpy.nan)
        standard_deviation_matrix = numpy.full(
            (num_examples, num_predictors), numpy.nan)

        for i in range(num_examples):
            for m in range(num_predictors):
                mean_value_matrix[i, m] = numpy.nanmean(
                    predictor_matrix[i, ..., m])
                standard_deviation_matrix[i, m] = numpy.nanstd(
                    predictor_matrix[i, ..., m], ddof=1)

                predictor_matrix[i, ..., m] = (
                    (predictor_matrix[i, ..., m] - mean_value_matrix[i, m]) /
                    standard_deviation_matrix[i, m]
                )

    normalization_dict = {
        MIN_VALUE_MATRIX_KEY: min_value_matrix,
        MAX_VALUE_MATRIX_KEY: max_value_matrix,
        MEAN_VALUE_MATRIX_KEY: mean_value_matrix,
        STDEV_MATRIX_KEY: standard_deviation_matrix
    }

    return predictor_matrix, normalization_dict


def denormalize_predictors(predictor_matrix, normalization_dict):
    """Deormalizes predictor variables.

    :param predictor_matrix: See output doc for `normalize_predictors`.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input (same dimensions).
    """

    _check_predictor_matrix(
        predictor_matrix, allow_nan=True, min_num_dimensions=4,
        max_num_dimensions=5)

    num_examples = predictor_matrix.shape[0]
    num_predictors = predictor_matrix.shape[-1]
    expected_param_dimensions = numpy.array(
        [num_examples, num_predictors], dtype=int)

    if normalization_dict[MIN_VALUE_MATRIX_KEY] is None:
        normalization_type_string = Z_SCORE_STRING + ''
    else:
        normalization_type_string = MINMAX_STRING + ''

    if normalization_type_string == MINMAX_STRING:
        min_value_matrix = normalization_dict[MIN_VALUE_MATRIX_KEY]
        max_value_matrix = normalization_dict[MAX_VALUE_MATRIX_KEY]

        error_checking.assert_is_numpy_array_without_nan(min_value_matrix)
        error_checking.assert_is_numpy_array(
            min_value_matrix, exact_dimensions=expected_param_dimensions)

        error_checking.assert_is_numpy_array_without_nan(max_value_matrix)
        error_checking.assert_is_numpy_array(
            max_value_matrix, exact_dimensions=expected_param_dimensions)

        for i in range(num_examples):
            for m in range(num_predictors):
                predictor_matrix[i, ..., m] = (
                    min_value_matrix[i, m] +
                    predictor_matrix[i, ..., m] *
                    (max_value_matrix[i, m] - min_value_matrix[i, m])
                )
    else:
        mean_value_matrix = normalization_dict[MEAN_VALUE_MATRIX_KEY]
        standard_deviation_matrix = normalization_dict[STDEV_MATRIX_KEY]

        error_checking.assert_is_numpy_array_without_nan(mean_value_matrix)
        error_checking.assert_is_numpy_array(
            mean_value_matrix, exact_dimensions=expected_param_dimensions)

        error_checking.assert_is_numpy_array_without_nan(
            standard_deviation_matrix)
        error_checking.assert_is_numpy_array(
            standard_deviation_matrix,
            exact_dimensions=expected_param_dimensions)

        for i in range(num_examples):
            for m in range(num_predictors):
                predictor_matrix[i, ..., m] = (
                    mean_value_matrix[i, m] +
                    predictor_matrix[i, ..., m] *
                    standard_deviation_matrix[i, m]
                )

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

    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer_numpy_array(center_grid_rows)
    error_checking.assert_is_geq_numpy_array(center_grid_rows, 0)
    error_checking.assert_is_numpy_array(
        center_grid_rows, exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer_numpy_array(center_grid_columns)
    error_checking.assert_is_geq_numpy_array(center_grid_columns, 0)
    error_checking.assert_is_numpy_array(
        center_grid_columns, exact_dimensions=these_expected_dim)

    these_expected_dim = numpy.array([num_predictor_variables], dtype=int)

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names), exact_dimensions=these_expected_dim)

    if num_predictor_dimensions == 5:
        num_time_steps = predictor_matrix.shape[3]
        these_expected_dim = numpy.array(
            [num_examples, num_time_steps], dtype=int)

        error_checking.assert_is_integer_numpy_array(
            predictor_time_matrix_unix_sec)
        error_checking.assert_is_numpy_array(
            predictor_time_matrix_unix_sec, exact_dimensions=these_expected_dim)


def sample_target_points(
        target_matrix, class_fractions, num_points_to_sample, mask_matrix=None,
        test_mode=False):
    """Samples target points to achieve desired class balance.

    If any class is missing from `target_matrix`, this method will return None.

    P_i = number of grid points selected at the [i]th time

    :param target_matrix: E-by-M-by-N numpy array of target images.  May be
        either binary (2-class) or ternary (3-class).
    :param class_fractions: 1-D numpy array of desired class fractions.  If
        `target_matrix` is binary, this array must have length 2; if
        `target_matrix` is ternary, this array must have length 3.
    :param num_points_to_sample: Number of points to sample.
    :param mask_matrix: M-by-N numpy array of integers (0 or 1).  If
        mask_matrix[i, j] = 0, grid cell [i, j] will never be sampled -- i.e.,
        used as the center of a downsized grid.
    :param test_mode: Leave this alone.
    :return: target_point_dict: Dictionary with the following keys.
    target_point_dict['row_indices_by_time']: length-T list, where the [i]th
        element is a numpy array (length P_i) with row indices of grid points
        selected at the [i]th time.
    target_point_dict['column_indices_by_time']: Same as above, except for
        columns.
    """

    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(class_fractions, 0.)
    error_checking.assert_is_leq_numpy_array(class_fractions, 1.)

    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    sum_of_class_fractions = numpy.sum(class_fractions)
    absolute_diff = numpy.absolute(sum_of_class_fractions - 1.)

    if absolute_diff > TOLERANCE_FOR_FREQUENCY_SUM:
        error_string = (
            '\n{0:s}\nSum of class fractions (shown above) should be 1.  '
            'Instead, got {1:.4f}.'
        ).format(str(class_fractions), sum_of_class_fractions)

        raise ValueError(error_string)

    _check_target_matrix(
        target_matrix, assert_binary=num_classes == 2, num_dimensions=3)

    if mask_matrix is None:
        mask_matrix_2d = numpy.full(target_matrix.shape[1:], 1, dtype=int)
    else:
        mask_matrix_2d = mask_matrix + 0

    error_checking.assert_is_integer_numpy_array(mask_matrix_2d)
    error_checking.assert_is_geq_numpy_array(mask_matrix_2d, 0)
    error_checking.assert_is_leq_numpy_array(mask_matrix_2d, 1)
    error_checking.assert_is_numpy_array(
        mask_matrix_2d,
        exact_dimensions=numpy.array(target_matrix.shape[1:], dtype=int)
    )

    mask_matrix = numpy.expand_dims(mask_matrix_2d, 0)
    mask_matrix = numpy.tile(mask_matrix, (target_matrix.shape[0], 1, 1))

    error_checking.assert_is_integer(num_points_to_sample)
    error_checking.assert_is_geq(num_points_to_sample, 3)
    error_checking.assert_is_boolean(test_mode)

    num_points_to_sample_by_class = _class_fractions_to_num_points(
        class_fractions=class_fractions, num_points_total=num_points_to_sample)

    num_classes = len(num_points_to_sample_by_class)
    num_points_found_by_class = numpy.full(num_classes, -1, dtype=int)
    time_indices_by_class = [numpy.array([], dtype=int)] * num_classes
    row_indices_by_class = [numpy.array([], dtype=int)] * num_classes
    column_indices_by_class = [numpy.array([], dtype=int)] * num_classes

    for i in range(num_classes):
        this_num_examples_unmasked = numpy.sum(target_matrix == i)
        this_num_examples_masked = numpy.sum(
            numpy.logical_and(target_matrix == i, mask_matrix == 1))

        print (
            'Number of examples for class {0:d} before mask = {1:d} ... after '
            'mask = {2:d}'
        ).format(i, this_num_examples_unmasked, this_num_examples_masked)

        (time_indices_by_class[i], row_indices_by_class[i],
         column_indices_by_class[i]
        ) = numpy.where(numpy.logical_and(target_matrix == i, mask_matrix == 1))

        num_points_found_by_class[i] = len(time_indices_by_class[i])
        if num_points_found_by_class[i] == 0:
            return None
        if num_points_found_by_class[i] <= num_points_to_sample_by_class[i]:
            continue

        these_indices = numpy.linspace(
            0, num_points_found_by_class[i] - 1,
            num=num_points_found_by_class[i], dtype=int)

        if test_mode:
            these_indices = these_indices[:num_points_to_sample_by_class[i]]
        else:
            these_indices = numpy.random.choice(
                these_indices, size=num_points_to_sample_by_class[i],
                replace=False)

        time_indices_by_class[i] = time_indices_by_class[i][these_indices]
        row_indices_by_class[i] = row_indices_by_class[i][these_indices]
        column_indices_by_class[i] = column_indices_by_class[i][these_indices]
        num_points_found_by_class[i] = len(time_indices_by_class[i])

    if numpy.any(num_points_found_by_class < num_points_to_sample_by_class):
        fraction_of_desired_num_by_class = num_points_found_by_class.astype(
            float) / num_points_to_sample_by_class

        num_points_to_sample = int(numpy.floor(
            num_points_to_sample * numpy.min(fraction_of_desired_num_by_class)))

        num_points_to_sample_by_class = _class_fractions_to_num_points(
            class_fractions=class_fractions,
            num_points_total=num_points_to_sample)

        for i in range(num_classes):
            if num_points_found_by_class[i] <= num_points_to_sample_by_class[i]:
                continue

            these_indices = numpy.linspace(
                0, num_points_found_by_class[i] - 1,
                num=num_points_found_by_class[i], dtype=int)

            if test_mode:
                these_indices = these_indices[:num_points_to_sample_by_class[i]]
            else:
                these_indices = numpy.random.choice(
                    these_indices, size=num_points_to_sample_by_class[i],
                    replace=False)

            time_indices_by_class[i] = time_indices_by_class[i][these_indices]
            row_indices_by_class[i] = row_indices_by_class[i][these_indices]
            column_indices_by_class[i] = column_indices_by_class[i][
                these_indices]

    num_times = target_matrix.shape[0]
    row_indices_by_time = [numpy.array([], dtype=int)] * num_times
    column_indices_by_time = [numpy.array([], dtype=int)] * num_times

    for i in range(num_times):
        for j in range(num_classes):
            this_time_indices = numpy.where(time_indices_by_class[j] == i)[0]

            row_indices_by_time[i] = numpy.concatenate((
                row_indices_by_time[i],
                row_indices_by_class[j][this_time_indices]
            ))
            column_indices_by_time[i] = numpy.concatenate((
                column_indices_by_time[i],
                column_indices_by_class[j][this_time_indices]
            ))

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
            (1, num_rows_per_image, num_columns_per_image), 0, dtype=int
        )

    num_times = len(frontal_grid_table.index)
    frontal_grid_matrix = None

    for i in range(num_times):
        this_gridded_front_dict = {
            front_utils.WARM_FRONT_ROWS_COLUMN: frontal_grid_table[
                front_utils.WARM_FRONT_ROWS_COLUMN].values[i],

            front_utils.WARM_FRONT_COLUMNS_COLUMN: frontal_grid_table[
                front_utils.WARM_FRONT_COLUMNS_COLUMN].values[i],

            front_utils.COLD_FRONT_ROWS_COLUMN: frontal_grid_table[
                front_utils.COLD_FRONT_ROWS_COLUMN].values[i],

            front_utils.COLD_FRONT_COLUMNS_COLUMN: frontal_grid_table[
                front_utils.COLD_FRONT_COLUMNS_COLUMN].values[i]
        }

        this_gridded_front_matrix = front_utils.points_to_gridded_labels(
            gridded_label_dict=this_gridded_front_dict,
            num_grid_rows=num_rows_per_image,
            num_grid_columns=num_columns_per_image)

        this_gridded_front_matrix = numpy.reshape(
            this_gridded_front_matrix,
            (1, num_rows_per_image, num_columns_per_image)
        )

        if frontal_grid_matrix is None:
            frontal_grid_matrix = copy.deepcopy(this_gridded_front_matrix)
        else:
            frontal_grid_matrix = numpy.concatenate(
                (frontal_grid_matrix, this_gridded_front_matrix), axis=0)

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
        frontal_grid_matrix == front_utils.WARM_FRONT_ENUM
    ] = front_utils.ANY_FRONT_ENUM

    frontal_grid_matrix[
        frontal_grid_matrix == front_utils.COLD_FRONT_ENUM
    ] = front_utils.ANY_FRONT_ENUM

    return frontal_grid_matrix


def dilate_ternary_target_images(
        target_matrix, dilation_distance_metres, verbose=True):
    """Dilates ternary (3-class) target image at each time step.

    :param target_matrix: E-by-M-by-N numpy array with 3 possible
        entries (see documentation for `_check_target_matrix`).
    :param dilation_distance_metres: Dilation distance.
    :param verbose: Boolean flag.  If True, this method will print progress
        messages.
    :return: target_matrix: Dilated version of input.
    """

    _check_target_matrix(target_matrix, assert_binary=False, num_dimensions=3)
    error_checking.assert_is_boolean(verbose)

    dilation_mask_matrix = front_utils.buffer_distance_to_dilation_mask(
        buffer_distance_metres=dilation_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES
    ).astype(int)

    num_times = target_matrix.shape[0]

    for i in range(num_times):
        if verbose:
            print (
                'Dilating 3-class target image at {0:d}th of {1:d} time '
                'steps...'
            ).format(i + 1, num_times)

        target_matrix[i, :, :] = front_utils.dilate_ternary_label_matrix(
            ternary_label_matrix=target_matrix[i, :, :],
            dilation_mask_matrix=dilation_mask_matrix)

    return target_matrix


def dilate_binary_target_images(
        target_matrix, dilation_distance_metres, verbose=True):
    """Dilates binary (2-class) target image at each time step.

    :param target_matrix: E-by-M-by-N numpy array with 2 possible
        entries (see documentation for `_check_target_matrix`).
    :param dilation_distance_metres: Dilation distance.
    :param verbose: Boolean flag.  If True, this method will print progress
        messages.
    :return: target_matrix: Dilated version of input.
    """

    _check_target_matrix(target_matrix, assert_binary=True, num_dimensions=3)
    error_checking.assert_is_boolean(verbose)

    dilation_mask_matrix = front_utils.buffer_distance_to_dilation_mask(
        buffer_distance_metres=dilation_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES
    ).astype(int)

    num_times = target_matrix.shape[0]

    for i in range(num_times):
        if verbose:
            print (
                'Dilating 2-class target image at {0:d}th of {1:d} time '
                'steps...'
            ).format(i + 1, num_times)

        target_matrix[i, :, :] = front_utils.dilate_binary_label_matrix(
            binary_label_matrix=target_matrix[i, :, :],
            dilation_mask_matrix=dilation_mask_matrix)

    return target_matrix


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


def fill_nans_in_predictor_images(predictor_matrix):
    """Fills NaN's in predictor images.

    :param predictor_matrix: E-by-M-by-N numpy array of predictor images.
    :return: predictor_matrix: Same but without NaN's.
    """

    _check_predictor_matrix(
        predictor_matrix=predictor_matrix, allow_nan=True, min_num_dimensions=3,
        max_num_dimensions=3)

    num_times = predictor_matrix.shape[0]
    for i in range(num_times):
        predictor_matrix[i, ...] = utils.fill_nans(predictor_matrix[i, ...])

    _check_predictor_matrix(
        predictor_matrix=predictor_matrix, allow_nan=False,
        min_num_dimensions=3, max_num_dimensions=3)

    return predictor_matrix


def subset_narr_grid_for_fcn_input(narr_matrix):
    """Subsets NARR grid for input to an FCN (fully convolutional net).

    This gives dimensions of 272 x 336, which are integer-divisible by 2 many
    times, which obviates the need for padding and cropping layers.

    :param narr_matrix: numpy array of either predictor images or target images.
        Dimensions may be E x 277 x 349, E x 277 x 349 x C, or
        E x 277 x 349 x T x C.
    :return: narr_matrix: Subset of the input array.  Dimensions may be
        E x 272 x 336, E x 272 x 336 x C, or E x 272 x 336 x T x C.
    """

    _check_full_narr_matrix(narr_matrix)

    return narr_matrix[
        :,
        FIRST_NARR_ROW_FOR_FCN_INPUT:(LAST_NARR_ROW_FOR_FCN_INPUT + 1),
        FIRST_NARR_COLUMN_FOR_FCN_INPUT:(LAST_NARR_COLUMN_FOR_FCN_INPUT + 1),
        ...
    ]


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
            target_point_dict[ROW_INDICES_BY_TIME_KEY][i]
        )

    new_dimensions = (
        num_downsized_examples, num_downsized_rows, num_downsized_columns
    )

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
                'full-size examples...'
            ).format(i + 1, num_full_examples)

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
                    num_columns_in_half_window=num_columns_in_half_window)
            )

            target_values[last_downsized_example_index + 1] = target_matrix[
                i, these_target_point_rows[j], these_target_point_columns[j]
            ]

            example_indices[last_downsized_example_index + 1] = i

            center_grid_rows[
                last_downsized_example_index + 1
            ] = these_target_point_rows[j]

            center_grid_columns[
                last_downsized_example_index + 1
            ] = these_target_point_columns[j]

            last_downsized_example_index += 1

    return (new_predictor_matrix, target_values, example_indices,
            center_grid_rows, center_grid_columns)


def write_narr_mask(mask_matrix, num_warm_fronts_matrix, num_cold_fronts_matrix,
                    pickle_file_name):
    """Writes NARR mask to Pickle file.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of integers in range 0...1, where 0
        means that the grid cell is masked.
    :param num_warm_fronts_matrix: M-by-N numpy array with number of warm fronts
        per grid cell.
    :param num_cold_fronts_matrix: M-by-N numpy array with number of cold fronts
        per grid cell.
    :param pickle_file_name: Path to output file.
    """

    check_narr_mask(mask_matrix)
    expected_dimensions = numpy.array(mask_matrix.shape, dtype=int)

    error_checking.assert_is_integer_numpy_array(num_warm_fronts_matrix)
    error_checking.assert_is_numpy_array(
        num_warm_fronts_matrix, exact_dimensions=expected_dimensions)

    error_checking.assert_is_integer_numpy_array(num_cold_fronts_matrix)
    error_checking.assert_is_numpy_array(
        num_cold_fronts_matrix, exact_dimensions=expected_dimensions)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mask_matrix, pickle_file_handle)
    pickle.dump(num_warm_fronts_matrix, pickle_file_handle)
    pickle.dump(num_cold_fronts_matrix, pickle_file_handle)
    pickle_file_handle.close()


def read_narr_mask(pickle_file_name):
    """Reads NARR mask from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mask_matrix: See doc for `write_narr_mask`.
    :return: num_warm_fronts_matrix: See doc for `write_narr_mask`.  If the file
        does not contain this matrix, the return value will be None.
    :return: num_cold_fronts_matrix: Same.
    """

    num_warm_fronts_matrix = None
    num_cold_fronts_matrix = None

    pickle_file_handle = open(pickle_file_name, 'rb')
    mask_matrix = pickle.load(pickle_file_handle)
    check_narr_mask(mask_matrix)

    try:
        num_warm_fronts_matrix = pickle.load(pickle_file_handle)
        num_cold_fronts_matrix = pickle.load(pickle_file_handle)
    except EOFError:
        pass

    pickle_file_handle.close()

    return mask_matrix, num_warm_fronts_matrix, num_cold_fronts_matrix


def find_gridded_prediction_file(
        directory_name, first_target_time_unix_sec, last_target_time_unix_sec,
        raise_error_if_missing=True):
    """Finds Pickle file with gridded predictions.

    This type of file should be written by `write_gridded_predictions`.

    :param directory_name: Name of directory with prediction file.
    :param first_target_time_unix_sec: First target time in file.
    :param last_target_time_unix_sec: Last target time in file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(first_target_time_unix_sec)
    error_checking.assert_is_integer(last_target_time_unix_sec)
    error_checking.assert_is_geq(
        last_target_time_unix_sec, first_target_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/gridded_predictions_{1:s}-{2:s}.p'.format(
        directory_name,
        time_conversion.unix_sec_to_string(
            first_target_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            last_target_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    )

    if not os.path.isfile(prediction_file_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_gridded_predictions(
        pickle_file_name, class_probability_matrix, target_times_unix_sec,
        model_file_name, used_isotonic_regression=False, target_matrix=None):
    """Writes gridded predictions to Pickle file.

    :param pickle_file_name: Path to output file.
    :param class_probability_matrix: E-by-M-by-N-by-K numpy array of predicted
        class probabilities.
    :param target_times_unix_sec: length-E numpy array of target times.
    :param model_file_name: Path to file containing the model used to generate
        predictions.
    :param used_isotonic_regression: Boolean flag.  If True, isotonic regression
        was used to calibrate probabilities from the base model (contained in
        `model_file_name`).
    :param target_matrix: E-by-M-by-N numpy array of target classes.  If
        `target_matrix is None`, this method will write only the predictions.
    """

    error_checking.assert_is_geq_numpy_array(
        class_probability_matrix, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        class_probability_matrix, 1., allow_nan=True)
    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=4)

    if target_matrix is not None:
        _check_target_matrix(
            target_matrix=target_matrix, assert_binary=False, num_dimensions=3)

        num_classes = class_probability_matrix.shape[-1]
        these_expected_dim = numpy.array(
            target_matrix.shape + (num_classes,), dtype=int
        )

        error_checking.assert_is_numpy_array(
            class_probability_matrix, exact_dimensions=these_expected_dim)

    these_expected_dim = numpy.array(
        [class_probability_matrix.shape[0]], dtype=int)

    error_checking.assert_is_integer_numpy_array(target_times_unix_sec)
    error_checking.assert_is_numpy_array(
        target_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(used_isotonic_regression)

    prediction_dict = {
        PROBABILITY_MATRIX_KEY: class_probability_matrix,
        TARGET_TIMES_KEY: target_times_unix_sec,
        TARGET_MATRIX_KEY: target_matrix,
        MODEL_FILE_NAME_KEY: model_file_name,
        USED_ISOTONIC_KEY: used_isotonic_regression
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(prediction_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_gridded_predictions(pickle_file_name):
    """Reads gridded predictions from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['class_probability_matrix']: See doc for
        `write_gridded_predictions`.
    prediction_dict['target_times_unix_sec']: Same.
    prediction_dict['target_matrix']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['used_isotonic_regression']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    prediction_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return prediction_dict
