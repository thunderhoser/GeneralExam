"""Unit tests for machine_learning_utils.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6
TOLERANCE_FOR_CLASS_WEIGHT = 1e-3

# The following constants are used to test _check_full_narr_matrix.
NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME,
    grid_name=nwp_model_utils.NAME_OF_221GRID)

FULL_NARR_MATRIX_2D = numpy.random.uniform(
    low=0., high=1., size=(NUM_ROWS_IN_NARR, NUM_COLUMNS_IN_NARR)
)

FULL_NARR_MATRIX_3D = numpy.stack(
    (FULL_NARR_MATRIX_2D, FULL_NARR_MATRIX_2D), axis=0)
FULL_NARR_MATRIX_4D = numpy.stack(
    (FULL_NARR_MATRIX_3D, FULL_NARR_MATRIX_3D), axis=-1)
FULL_NARR_MATRIX_5D = numpy.stack(
    (FULL_NARR_MATRIX_4D, FULL_NARR_MATRIX_4D), axis=-1)

# The following constants are used to test _check_predictor_matrix.
PREDICTOR_MATRIX_1D = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
PREDICTOR_MATRIX_2D = numpy.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12]], dtype=numpy.float32)

TUPLE_OF_2D_PREDICTOR_MATRICES = (PREDICTOR_MATRIX_2D, PREDICTOR_MATRIX_2D)
PREDICTOR_MATRIX_3D = numpy.stack(TUPLE_OF_2D_PREDICTOR_MATRICES, axis=0)
PREDICTOR_MATRIX_3D[0, 0, 0] = numpy.nan

TUPLE_OF_3D_PREDICTOR_MATRICES = (
    PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D,
    PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D)
PREDICTOR_MATRIX_4D = numpy.stack(TUPLE_OF_3D_PREDICTOR_MATRICES, axis=-1)

TUPLE_OF_4D_PREDICTOR_MATRICES = (
    PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D,
    PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D)
PREDICTOR_MATRIX_5D = numpy.stack(TUPLE_OF_4D_PREDICTOR_MATRICES, axis=-2)

# The following constants are used to test _check_target_matrix.
TARGET_VALUES_BINARY_1D = numpy.array([0, 1, 1, 0], dtype=int)
TARGET_VALUES_TERNARY_1D = numpy.array([0, 2, 1, 0], dtype=int)
TARGET_VALUES_BINARY_2D = numpy.array([[0, 1, 1, 0],
                                       [1, 0, 0, 1]], dtype=int)
TARGET_VALUES_BINARY_3D = numpy.stack(
    (TARGET_VALUES_BINARY_2D, TARGET_VALUES_BINARY_2D), axis=0)

# The following constants are used to test _downsize_predictor_images.
PRE_DOWNSIZED_MATRIX = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [10, 11, 12, 13, 14, 15, 16, 17, 18],
                                    [19, 20, 21, 22, 23, 24, 25, 26, 27],
                                    [28, 29, 30, 31, 32, 33, 34, 35, 36],
                                    [37, 38, 39, 40, 41, 42, 43, 44, 45],
                                    [46, 47, 48, 49, 50, 51, 52, 53, 54],
                                    [55, 56, 57, 58, 59, 60, 61, 62, 63]],
                                   dtype=numpy.float32)

DOWNSIZING_CENTER_ROW_TOP_LEFT = 1
DOWNSIZING_CENTER_COLUMN_TOP_LEFT = 1
DOWNSIZING_CENTER_ROW_BOTTOM_LEFT = 5
DOWNSIZING_CENTER_COLUMN_BOTTOM_LEFT = 1
DOWNSIZING_CENTER_ROW_TOP_RIGHT = 1
DOWNSIZING_CENTER_COLUMN_TOP_RIGHT = 7
DOWNSIZING_CENTER_ROW_BOTTOM_RIGHT = 5
DOWNSIZING_CENTER_COLUMN_BOTTOM_RIGHT = 7
DOWNSIZING_CENTER_ROW_MIDDLE = 3
DOWNSIZING_CENTER_COLUMN_MIDDLE = 4

NUM_ROWS_IN_DOWNSIZED_HALF_GRID = 2
NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID = 3

DOWNSIZED_MATRIX_TOP_LEFT = numpy.array([[1, 1, 1, 2, 3, 4, 5],
                                         [1, 1, 1, 2, 3, 4, 5],
                                         [10, 10, 10, 11, 12, 13, 14],
                                         [19, 19, 19, 20, 21, 22, 23],
                                         [28, 28, 28, 29, 30, 31, 32]],
                                        dtype=numpy.float32)

DOWNSIZED_MATRIX_BOTTOM_LEFT = numpy.array([[28, 28, 28, 29, 30, 31, 32],
                                            [37, 37, 37, 38, 39, 40, 41],
                                            [46, 46, 46, 47, 48, 49, 50],
                                            [55, 55, 55, 56, 57, 58, 59],
                                            [55, 55, 55, 56, 57, 58, 59]],
                                           dtype=numpy.float32)

DOWNSIZED_MATRIX_TOP_RIGHT = numpy.array([[5, 6, 7, 8, 9, 9, 9],
                                          [5, 6, 7, 8, 9, 9, 9],
                                          [14, 15, 16, 17, 18, 18, 18],
                                          [23, 24, 25, 26, 27, 27, 27],
                                          [32, 33, 34, 35, 36, 36, 36]],
                                         dtype=numpy.float32)

DOWNSIZED_MATRIX_BOTTOM_RIGHT = numpy.array([[32, 33, 34, 35, 36, 36, 36],
                                             [41, 42, 43, 44, 45, 45, 45],
                                             [50, 51, 52, 53, 54, 54, 54],
                                             [59, 60, 61, 62, 63, 63, 63],
                                             [59, 60, 61, 62, 63, 63, 63]],
                                            dtype=numpy.float32)

DOWNSIZED_MATRIX_MIDDLE = numpy.array([[11, 12, 13, 14, 15, 16, 17],
                                       [20, 21, 22, 23, 24, 25, 26],
                                       [29, 30, 31, 32, 33, 34, 35],
                                       [38, 39, 40, 41, 42, 43, 44],
                                       [47, 48, 49, 50, 51, 52, 53]],
                                      dtype=numpy.float32)

PRE_DOWNSIZED_MATRIX_3D = numpy.stack(
    (PRE_DOWNSIZED_MATRIX, PRE_DOWNSIZED_MATRIX), axis=0)
PRE_DOWNSIZED_MATRIX_4D = numpy.stack(
    (PRE_DOWNSIZED_MATRIX_3D, PRE_DOWNSIZED_MATRIX_3D), axis=-1)
PRE_DOWNSIZED_MATRIX_5D = numpy.stack(
    (PRE_DOWNSIZED_MATRIX_4D, PRE_DOWNSIZED_MATRIX_4D), axis=-2)

DOWNSIZED_MATRIX_TOP_LEFT_3D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_LEFT, DOWNSIZED_MATRIX_TOP_LEFT), axis=0)
DOWNSIZED_MATRIX_TOP_LEFT_4D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_LEFT_3D, DOWNSIZED_MATRIX_TOP_LEFT_3D), axis=-1)
DOWNSIZED_MATRIX_TOP_LEFT_5D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_LEFT_4D, DOWNSIZED_MATRIX_TOP_LEFT_4D), axis=-2)

DOWNSIZED_MATRIX_BOTTOM_LEFT_3D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_LEFT, DOWNSIZED_MATRIX_BOTTOM_LEFT), axis=0)
DOWNSIZED_MATRIX_BOTTOM_LEFT_4D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_LEFT_3D, DOWNSIZED_MATRIX_BOTTOM_LEFT_3D),
    axis=-1)
DOWNSIZED_MATRIX_BOTTOM_LEFT_5D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_LEFT_4D, DOWNSIZED_MATRIX_BOTTOM_LEFT_4D),
    axis=-2)

DOWNSIZED_MATRIX_TOP_RIGHT_3D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_RIGHT, DOWNSIZED_MATRIX_TOP_RIGHT), axis=0)
DOWNSIZED_MATRIX_TOP_RIGHT_4D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_RIGHT_3D, DOWNSIZED_MATRIX_TOP_RIGHT_3D), axis=-1)
DOWNSIZED_MATRIX_TOP_RIGHT_5D = numpy.stack(
    (DOWNSIZED_MATRIX_TOP_RIGHT_4D, DOWNSIZED_MATRIX_TOP_RIGHT_4D), axis=-2)

DOWNSIZED_MATRIX_BOTTOM_RIGHT_3D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_RIGHT, DOWNSIZED_MATRIX_BOTTOM_RIGHT), axis=0)
DOWNSIZED_MATRIX_BOTTOM_RIGHT_4D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_RIGHT_3D, DOWNSIZED_MATRIX_BOTTOM_RIGHT_3D),
    axis=-1)
DOWNSIZED_MATRIX_BOTTOM_RIGHT_5D = numpy.stack(
    (DOWNSIZED_MATRIX_BOTTOM_RIGHT_4D, DOWNSIZED_MATRIX_BOTTOM_RIGHT_4D),
    axis=-2)

DOWNSIZED_MATRIX_MIDDLE_3D = numpy.stack(
    (DOWNSIZED_MATRIX_MIDDLE, DOWNSIZED_MATRIX_MIDDLE), axis=0)
DOWNSIZED_MATRIX_MIDDLE_4D = numpy.stack(
    (DOWNSIZED_MATRIX_MIDDLE_3D, DOWNSIZED_MATRIX_MIDDLE_3D), axis=-1)
DOWNSIZED_MATRIX_MIDDLE_5D = numpy.stack(
    (DOWNSIZED_MATRIX_MIDDLE_4D, DOWNSIZED_MATRIX_MIDDLE_4D), axis=-2)

# The following constants are used to test _class_fractions_to_num_points.
CLASS_FRACTIONS_BINARY = numpy.array([0.1, 0.9])
NUM_POINTS_AVAILABLE_LARGE = 17
NUM_POINTS_BY_CLASS_BINARY_LARGE = numpy.array([2, 15])
NUM_POINTS_BY_CLASS_TERNARY_LARGE = numpy.array([2, 3, 12])

CLASS_FRACTIONS_TERNARY = numpy.array([0.1, 0.2, 0.7])
NUM_POINTS_AVAILABLE_SMALL = 4
NUM_POINTS_BY_CLASS_BINARY_SMALL = numpy.array([1, 3])
NUM_POINTS_BY_CLASS_TERNARY_SMALL = numpy.array([1, 1, 2])

# The following constants are used to test get_class_weight_dict.
CLASS_WEIGHT_DICT_BINARY = {0: 0.9, 1: 0.1}
CLASS_WEIGHT_DICT_TERNARY = {0: 0.6087, 1: 0.3043, 2: 0.0870}

# The following constants are used to test normalize_predictors with
# normalization type = "minmax".
PRCTILE_OFFSET_FOR_NORMALIZATION = 0.

FIRST_PREDICTOR_MATRIX_2D = numpy.array(
    [[0, 1, 2, 3],
     [4, 5, 6, 7]], dtype=float
)

SECOND_PREDICTOR_MATRIX_2D = numpy.array(
    [[2, 4, 6, numpy.nan],
     [-1, -3, -5, -7]]
)

THIS_FIRST_MATRIX_3D = numpy.stack(
    (FIRST_PREDICTOR_MATRIX_2D, FIRST_PREDICTOR_MATRIX_2D), axis=-1)
PREDICTOR_MATRIX_4D_DENORM = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_FIRST_MATRIX_3D), axis=0)

THIS_SECOND_MATRIX_3D = numpy.stack(
    (SECOND_PREDICTOR_MATRIX_2D, SECOND_PREDICTOR_MATRIX_2D), axis=-1)
THIS_SECOND_MATRIX_4D = numpy.stack(
    (THIS_SECOND_MATRIX_3D, THIS_SECOND_MATRIX_3D), axis=0)
PREDICTOR_MATRIX_5D_DENORM = numpy.stack(
    (PREDICTOR_MATRIX_4D_DENORM, THIS_SECOND_MATRIX_4D), axis=-2)

THIS_MIN = 0.
THIS_MAX_LESS_MIN = 7.

THIS_FIRST_MATRIX_3D = numpy.stack((
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN,
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN
), axis=-1)
PREDICTOR_MATRIX_4D_MINMAX_NORM = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_FIRST_MATRIX_3D), axis=0)

THIS_MIN = -7.
THIS_MAX_LESS_MIN = 14.

THIS_FIRST_MATRIX_3D = numpy.stack((
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN,
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN
), axis=-1)
THIS_FIRST_MATRIX_4D = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_FIRST_MATRIX_3D), axis=0)

THIS_SECOND_MATRIX_3D = numpy.stack((
    (SECOND_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN,
    (SECOND_PREDICTOR_MATRIX_2D - THIS_MIN) / THIS_MAX_LESS_MIN
), axis=-1)
THIS_SECOND_MATRIX_4D = numpy.stack(
    (THIS_SECOND_MATRIX_3D, THIS_SECOND_MATRIX_3D), axis=0)

PREDICTOR_MATRIX_5D_MINMAX_NORM = numpy.stack(
    (THIS_FIRST_MATRIX_4D, THIS_SECOND_MATRIX_4D), axis=-2)

# The following constants are used to test normalize_predictors with
# normalization type = "z_score".
THIS_MEAN = numpy.mean(FIRST_PREDICTOR_MATRIX_2D)
THIS_STDEV = numpy.std(FIRST_PREDICTOR_MATRIX_2D, ddof=1)

THIS_FIRST_MATRIX_3D = numpy.stack((
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV,
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV
), axis=-1)
PREDICTOR_MATRIX_4D_Z_NORM = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_FIRST_MATRIX_3D), axis=0)

ALL_PREDICTORS = numpy.stack(
    (FIRST_PREDICTOR_MATRIX_2D, SECOND_PREDICTOR_MATRIX_2D), axis=-1)
THIS_MEAN = numpy.nanmean(ALL_PREDICTORS)
THIS_STDEV = numpy.nanstd(ALL_PREDICTORS, ddof=1)

THIS_FIRST_MATRIX_3D = numpy.stack((
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV,
    (FIRST_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV
), axis=-1)
THIS_FIRST_MATRIX_4D = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_FIRST_MATRIX_3D), axis=0)

THIS_SECOND_MATRIX_3D = numpy.stack((
    (SECOND_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV,
    (SECOND_PREDICTOR_MATRIX_2D - THIS_MEAN) / THIS_STDEV
), axis=-1)
THIS_SECOND_MATRIX_4D = numpy.stack(
    (THIS_SECOND_MATRIX_3D, THIS_SECOND_MATRIX_3D), axis=0)

PREDICTOR_MATRIX_5D_Z_NORM = numpy.stack(
    (THIS_FIRST_MATRIX_4D, THIS_SECOND_MATRIX_4D), axis=-2)

# The following constants are used to test front_table_to_images.
NUM_GRID_ROWS = 6
NUM_GRID_COLUMNS = 8
THESE_WF_ROW_INDICES = [numpy.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)]
THESE_WF_COLUMN_INDICES = [numpy.array([1, 2, 7, 2, 3, 4, 5, 6, 7], dtype=int)]
THESE_CF_ROW_INDICES = [numpy.array([1, 2, 3, 4, 4, 5], dtype=int)]
THESE_CF_COLUMN_INDICES = [numpy.array([1, 1, 1, 0, 1, 0], dtype=int)]

THIS_DICT = {
    front_utils.WARM_FRONT_ROWS_COLUMN: THESE_WF_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMNS_COLUMN: THESE_WF_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROWS_COLUMN: THESE_CF_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMNS_COLUMN: THESE_CF_COLUMN_INDICES
}
FRONTAL_GRID_TABLE1 = pandas.DataFrame.from_dict(THIS_DICT)

THESE_WF_ROW_INDICES = [numpy.array([1, 1, 2, 2, 3, 3], dtype=int)]
THESE_WF_COLUMN_INDICES = [numpy.array([4, 5, 5, 6, 6, 7], dtype=int)]
THESE_CF_ROW_INDICES = [
    numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=int)]
THESE_CF_COLUMN_INDICES = [
    numpy.array([2, 3, 1, 2, 0, 1, 0, 1, 0, 1, 1], dtype=int)]

THIS_DICT = {
    front_utils.WARM_FRONT_ROWS_COLUMN: THESE_WF_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMNS_COLUMN: THESE_WF_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROWS_COLUMN: THESE_CF_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMNS_COLUMN: THESE_CF_COLUMN_INDICES
}
FRONTAL_GRID_TABLE2 = pandas.DataFrame.from_dict(THIS_DICT)
FRONTAL_GRID_TABLE = pandas.concat(
    [FRONTAL_GRID_TABLE1, FRONTAL_GRID_TABLE2], axis=0, ignore_index=True)

THIS_FIRST_MATRIX = numpy.array([[0, 1, 1, 0, 0, 0, 0, 1],
                                 [0, 2, 1, 1, 1, 1, 1, 1],
                                 [0, 2, 0, 0, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0, 0, 0, 0],
                                 [2, 2, 0, 0, 0, 0, 0, 0],
                                 [2, 0, 0, 0, 0, 0, 0, 0]])

THIS_SECOND_MATRIX = numpy.array([[0, 0, 2, 2, 0, 0, 0, 0],
                                  [0, 2, 2, 0, 1, 1, 0, 0],
                                  [2, 2, 0, 0, 0, 1, 1, 0],
                                  [2, 2, 0, 0, 0, 0, 1, 1],
                                  [2, 2, 0, 0, 0, 0, 0, 0],
                                  [0, 2, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX_TERNARY = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0).astype(int)

# The following constants are used to test binarize_front_images.
THIS_FIRST_MATRIX = numpy.array([[0, 1, 1, 0, 0, 0, 0, 1],
                                 [0, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0]])

THIS_SECOND_MATRIX = numpy.array([[0, 0, 1, 1, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 1, 1, 0, 0],
                                  [1, 1, 0, 0, 0, 1, 1, 0],
                                  [1, 1, 0, 0, 0, 0, 1, 1],
                                  [1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX_BINARY = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0).astype(int)

# The following constants are used to test sample_target_points with 2 classes.
NUM_POINTS_TO_SAMPLE = 50
CLASS_FRACTIONS_FOR_BINARY_SAMPLING = numpy.array([0.5, 0.5])

MASK_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)

NEGATIVE_ROWS_TIME1_NO_MASK = numpy.array(
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    dtype=int)
NEGATIVE_COLUMNS_TIME1_NO_MASK = numpy.array(
    [0, 3, 4, 5, 6, 0, 0, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6],
    dtype=int)

POSITIVE_ROWS_TIME1_NO_MASK = numpy.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5], dtype=int)
POSITIVE_COLUMNS_TIME1_NO_MASK = numpy.array(
    [1, 2, 7, 1, 2, 3, 4, 5, 6, 7, 1, 1, 0, 1, 0], dtype=int)

ROW_INDICES_TIME1_NO_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME1_NO_MASK, POSITIVE_ROWS_TIME1_NO_MASK)).astype(int)
COLUMN_INDICES_TIME1_NO_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME1_NO_MASK, POSITIVE_COLUMNS_TIME1_NO_MASK)).astype(int)

NEGATIVE_ROWS_TIME1_WITH_MASK = numpy.array(
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int)
NEGATIVE_COLUMNS_TIME1_WITH_MASK = numpy.array(
    [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6], dtype=int)

POSITIVE_ROWS_TIME1_WITH_MASK = numpy.array(
    [1, 1, 1, 1, 1, 1, 2, 3, 4], dtype=int)
POSITIVE_COLUMNS_TIME1_WITH_MASK = numpy.array(
    [1, 2, 3, 4, 5, 6, 1, 1, 1], dtype=int)

ROW_INDICES_TIME1_WITH_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME1_WITH_MASK, POSITIVE_ROWS_TIME1_WITH_MASK)).astype(int)
COLUMN_INDICES_TIME1_WITH_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME1_WITH_MASK, POSITIVE_COLUMNS_TIME1_WITH_MASK
)).astype(int)

NEGATIVE_ROWS_TIME2_NO_MASK = numpy.array([], dtype=int)
NEGATIVE_COLUMNS_TIME2_NO_MASK = numpy.array([], dtype=int)

POSITIVE_ROWS_TIME2_NO_MASK = numpy.array(
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
POSITIVE_COLUMNS_TIME2_NO_MASK = numpy.array(
    [2, 3, 1, 2, 4, 5, 0, 1, 5, 6], dtype=int)

ROW_INDICES_TIME2_NO_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME2_NO_MASK, POSITIVE_ROWS_TIME2_NO_MASK)).astype(int)
COLUMN_INDICES_TIME2_NO_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME2_NO_MASK, POSITIVE_COLUMNS_TIME2_NO_MASK)).astype(int)

NEGATIVE_ROWS_TIME2_WITH_MASK = numpy.array([1, 1, 2, 2], dtype=int)
NEGATIVE_COLUMNS_TIME2_WITH_MASK = numpy.array([3, 6, 2, 3], dtype=int)

POSITIVE_ROWS_TIME2_WITH_MASK = numpy.array(
    [1, 1, 1, 1, 2, 2, 2, 3, 3, 4], dtype=int)
POSITIVE_COLUMNS_TIME2_WITH_MASK = numpy.array(
    [1, 2, 4, 5, 1, 5, 6, 1, 6, 1], dtype=int)

ROW_INDICES_TIME2_WITH_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME2_WITH_MASK, POSITIVE_ROWS_TIME2_WITH_MASK)).astype(int)
COLUMN_INDICES_TIME2_WITH_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME2_WITH_MASK, POSITIVE_COLUMNS_TIME2_WITH_MASK
)).astype(int)

TARGET_POINT_DICT_BINARY_NO_MASK = {
    ml_utils.ROW_INDICES_BY_TIME_KEY:
        [ROW_INDICES_TIME1_NO_MASK, ROW_INDICES_TIME2_NO_MASK],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY:
        [COLUMN_INDICES_TIME1_NO_MASK, COLUMN_INDICES_TIME2_NO_MASK]
}

TARGET_POINT_DICT_BINARY_WITH_MASK = {
    ml_utils.ROW_INDICES_BY_TIME_KEY:
        [ROW_INDICES_TIME1_WITH_MASK, ROW_INDICES_TIME2_WITH_MASK],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY:
        [COLUMN_INDICES_TIME1_WITH_MASK, COLUMN_INDICES_TIME2_WITH_MASK]
}

# The following constants are used to test sample_target_points with 3 classes.
CLASS_FRACTIONS_FOR_TERNARY_SAMPLING = numpy.array([0.5, 0.2, 0.3])

NEGATIVE_ROWS_TIME1_NO_MASK = numpy.array(
    [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    dtype=int)
NEGATIVE_COLUMNS_TIME1_NO_MASK = numpy.array(
    [0, 3, 4, 5, 6, 0, 0, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6],
    dtype=int)

NEGATIVE_ROWS_TIME2_NO_MASK = numpy.array([], dtype=int)
NEGATIVE_COLUMNS_TIME2_NO_MASK = numpy.array([], dtype=int)

WARM_FRONT_ROWS_TIME1_NO_MASK = numpy.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
WARM_FRONT_COLUMNS_TIME1_NO_MASK = numpy.array(
    [1, 2, 7, 2, 3, 4, 5, 6, 7], dtype=int)
WARM_FRONT_ROWS_TIME2_NO_MASK = numpy.array([1], dtype=int)
WARM_FRONT_COLUMNS_TIME2_NO_MASK = numpy.array([4], dtype=int)

COLD_FRONT_ROWS_TIME1_NO_MASK = numpy.array([1, 2, 3, 4, 4, 5], dtype=int)
COLD_FRONT_COLUMNS_TIME1_NO_MASK = numpy.array([1, 1, 1, 0, 1, 0], dtype=int)
COLD_FRONT_ROWS_TIME2_NO_MASK = numpy.array(
    [0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=int)
COLD_FRONT_COLUMNS_TIME2_NO_MASK = numpy.array(
    [2, 3, 1, 2, 0, 1, 0, 1, 0], dtype=int)

ROW_INDICES_TIME1_NO_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME1_NO_MASK, WARM_FRONT_ROWS_TIME1_NO_MASK,
    COLD_FRONT_ROWS_TIME1_NO_MASK
)).astype(int)
COLUMN_INDICES_TIME1_NO_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME1_NO_MASK, WARM_FRONT_COLUMNS_TIME1_NO_MASK,
    COLD_FRONT_COLUMNS_TIME1_NO_MASK
)).astype(int)

ROW_INDICES_TIME2_NO_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME2_NO_MASK, WARM_FRONT_ROWS_TIME2_NO_MASK,
    COLD_FRONT_ROWS_TIME2_NO_MASK
)).astype(int)
COLUMN_INDICES_TIME2_NO_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME2_NO_MASK, WARM_FRONT_COLUMNS_TIME2_NO_MASK,
    COLD_FRONT_COLUMNS_TIME2_NO_MASK
)).astype(int)

TARGET_POINT_DICT_TERNARY_NO_MASK = {
    ml_utils.ROW_INDICES_BY_TIME_KEY:
        [ROW_INDICES_TIME1_NO_MASK, ROW_INDICES_TIME2_NO_MASK],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY:
        [COLUMN_INDICES_TIME1_NO_MASK, COLUMN_INDICES_TIME2_NO_MASK]
}

NEGATIVE_ROWS_TIME1_WITH_MASK = numpy.array(
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int)
NEGATIVE_COLUMNS_TIME1_WITH_MASK = numpy.array(
    [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6], dtype=int)

NEGATIVE_ROWS_TIME2_WITH_MASK = numpy.array([], dtype=int)
NEGATIVE_COLUMNS_TIME2_WITH_MASK = numpy.array([], dtype=int)

WARM_FRONT_ROWS_TIME1_WITH_MASK = numpy.array([1, 1, 1, 1, 1], dtype=int)
WARM_FRONT_COLUMNS_TIME1_WITH_MASK = numpy.array([2, 3, 4, 5, 6], dtype=int)
WARM_FRONT_ROWS_TIME2_WITH_MASK = numpy.array([1], dtype=int)
WARM_FRONT_COLUMNS_TIME2_WITH_MASK = numpy.array([4], dtype=int)

COLD_FRONT_ROWS_TIME1_WITH_MASK = numpy.array([1, 2, 3, 4], dtype=int)
COLD_FRONT_COLUMNS_TIME1_WITH_MASK = numpy.array([1, 1, 1, 1], dtype=int)
COLD_FRONT_ROWS_TIME2_WITH_MASK = numpy.array([1, 1, 2, 3, 4], dtype=int)
COLD_FRONT_COLUMNS_TIME2_WITH_MASK = numpy.array([1, 2, 1, 1, 1], dtype=int)

ROW_INDICES_TIME1_WITH_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME1_WITH_MASK, WARM_FRONT_ROWS_TIME1_WITH_MASK,
    COLD_FRONT_ROWS_TIME1_WITH_MASK
)).astype(int)
COLUMN_INDICES_TIME1_WITH_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME1_WITH_MASK, WARM_FRONT_COLUMNS_TIME1_WITH_MASK,
    COLD_FRONT_COLUMNS_TIME1_WITH_MASK
)).astype(int)

ROW_INDICES_TIME2_WITH_MASK = numpy.concatenate((
    NEGATIVE_ROWS_TIME2_WITH_MASK, WARM_FRONT_ROWS_TIME2_WITH_MASK,
    COLD_FRONT_ROWS_TIME2_WITH_MASK
)).astype(int)
COLUMN_INDICES_TIME2_WITH_MASK = numpy.concatenate((
    NEGATIVE_COLUMNS_TIME2_WITH_MASK, WARM_FRONT_COLUMNS_TIME2_WITH_MASK,
    COLD_FRONT_COLUMNS_TIME2_WITH_MASK
)).astype(int)

TARGET_POINT_DICT_TERNARY_WITH_MASK = {
    ml_utils.ROW_INDICES_BY_TIME_KEY:
        [ROW_INDICES_TIME1_WITH_MASK, ROW_INDICES_TIME2_WITH_MASK],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY:
        [COLUMN_INDICES_TIME1_WITH_MASK, COLUMN_INDICES_TIME2_WITH_MASK]
}

# The following constants are used to test dilate_target_images.
DILATION_DISTANCE_METRES = 50000.

THIS_FIRST_MATRIX = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                 [2, 2, 1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 0, 0, 0, 0, 0],
                                 [2, 2, 2, 0, 0, 0, 0, 0],
                                 [2, 2, 2, 0, 0, 0, 0, 0]])

THIS_SECOND_MATRIX = numpy.array([[2, 2, 2, 2, 2, 1, 1, 0],
                                  [2, 2, 2, 2, 1, 1, 1, 1],
                                  [2, 2, 2, 2, 1, 1, 1, 1],
                                  [2, 2, 2, 0, 1, 1, 1, 1],
                                  [2, 2, 2, 0, 0, 1, 1, 1],
                                  [2, 2, 2, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX_TERNARY_DILATED = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0).astype(int)

THIS_FIRST_MATRIX = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 1],
                                 [1, 1, 1, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0, 0, 0]])

THIS_SECOND_MATRIX = numpy.array([[1, 1, 1, 1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 0, 1, 1, 1, 1],
                                  [1, 1, 1, 0, 0, 1, 1, 1],
                                  [1, 1, 1, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX_BINARY_DILATED = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0).astype(int)

# The following constants are used to test subset_narr_grid_for_fcn_input.
FCN_INPUT_MATRIX_2D = FULL_NARR_MATRIX_2D[
    ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT:
    (ml_utils.LAST_NARR_ROW_FOR_FCN_INPUT + 1),
    ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT:
    (ml_utils.LAST_NARR_COLUMN_FOR_FCN_INPUT + 1)
]

FCN_INPUT_MATRIX_3D = numpy.stack(
    (FCN_INPUT_MATRIX_2D, FCN_INPUT_MATRIX_2D), axis=0)
FCN_INPUT_MATRIX_4D = numpy.stack(
    (FCN_INPUT_MATRIX_3D, FCN_INPUT_MATRIX_3D), axis=-1)
FCN_INPUT_MATRIX_5D = numpy.stack(
    (FCN_INPUT_MATRIX_4D, FCN_INPUT_MATRIX_4D), axis=-2)

# The following constants are used to test
# downsize_grids_around_selected_points.
PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS = numpy.array([[1, 3, 5, 7],
                                                            [2, 4, 6, 8]],
                                                           dtype=numpy.float32)

PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS = numpy.stack(
    (PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS,), axis=0)
PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS = numpy.stack(
    (PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS,
     PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS,
     PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS), axis=-1)

DOWNSIZED_MATRIX_R1_C2 = numpy.array([[1, 3, 5],
                                      [1, 3, 5],
                                      [2, 4, 6]], dtype=numpy.float32)
DOWNSIZED_MATRIX_R1_C3 = numpy.array([[3, 5, 7],
                                      [3, 5, 7],
                                      [4, 6, 8]], dtype=numpy.float32)
DOWNSIZED_MATRIX_R2_C1 = numpy.array([[1, 1, 3],
                                      [2, 2, 4],
                                      [2, 2, 4]], dtype=numpy.float32)
DOWNSIZED_MATRIX_R2_C4 = numpy.array([[5, 7, 7],
                                      [6, 8, 8],
                                      [6, 8, 8]], dtype=numpy.float32)

TARGET_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS = numpy.array([[0, 0, 1, 1],
                                                         [2, 2, 0, 0]],
                                                        dtype=int)
TARGET_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS = numpy.stack(
    (TARGET_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS,), axis=0)

NUM_ROWS_IN_HALF_GRID_AROUND_SELECTED_PTS = 1
NUM_COLUMNS_IN_HALF_GRID_AROUND_SELECTED_PTS = 1

TARGET_POINT_DICT_FOR_DOWNSIZING = {
    ml_utils.ROW_INDICES_BY_TIME_KEY: [numpy.array([0, 0, 1, 1], dtype=int)],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY: [numpy.array([2, 1, 3, 0], dtype=int)]
}

DOWNSIZED_MATRIX_AT_SELECTED_POINTS = numpy.stack((
    DOWNSIZED_MATRIX_R1_C3, DOWNSIZED_MATRIX_R1_C2,
    DOWNSIZED_MATRIX_R2_C4, DOWNSIZED_MATRIX_R2_C1), axis=0)
DOWNSIZED_MATRIX_AT_SELECTED_POINTS = numpy.stack(
    (DOWNSIZED_MATRIX_AT_SELECTED_POINTS,
     DOWNSIZED_MATRIX_AT_SELECTED_POINTS,
     DOWNSIZED_MATRIX_AT_SELECTED_POINTS), axis=-1)

TARGET_VECTOR_AT_SELECTED_POINTS = numpy.array([1, 0, 0, 2], dtype=int)
EXAMPLE_INDICES_AT_SELECTED_POINTS = numpy.array([0, 0, 0, 0], dtype=int)
CENTER_ROWS_AT_SELECTED_POINTS = numpy.array([0, 0, 1, 1], dtype=int)
CENTER_COLUMNS_AT_SELECTED_POINTS = numpy.array([2, 1, 3, 0], dtype=int)


def _compare_target_point_dicts(
        first_target_point_dict, second_target_point_dict):
    """Compares two dictionaries with sampled target points.

    :param first_target_point_dict: First dictionary (in the format produced by
        `machine_learning_utils.sample_target_points`).
    :param second_target_point_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_target_point_dict.keys()
    second_keys = second_target_point_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    first_num_times = len(
        first_target_point_dict[ml_utils.ROW_INDICES_BY_TIME_KEY])
    second_num_times = len(
        second_target_point_dict[ml_utils.ROW_INDICES_BY_TIME_KEY])
    if first_num_times != second_num_times:
        return False

    for i in range(first_num_times):
        for this_key in first_keys:
            if not numpy.array_equal(first_target_point_dict[this_key][i],
                                     second_target_point_dict[this_key][i]):
                return False

    return True


class MachineLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for machine_learning_utils.py."""

    def test_check_full_narr_matrix_2d(self):
        """Ensures correct output from _check_full_narr_matrix.

        In this case, input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            ml_utils._check_full_narr_matrix(FULL_NARR_MATRIX_2D)

    def test_check_full_narr_matrix_3d(self):
        """Ensures correct output from _check_full_narr_matrix.

        In this case, input matrix is 3-D with the NARR's spatial dimensions
        (good).
        """

        ml_utils._check_full_narr_matrix(FULL_NARR_MATRIX_3D)

    def test_check_full_narr_matrix_4d(self):
        """Ensures correct output from _check_full_narr_matrix.

        In this case, input matrix is 4-D with the NARR's spatial dimensions
        (good).
        """

        ml_utils._check_full_narr_matrix(FULL_NARR_MATRIX_4D)

    def test_check_full_narr_matrix_5d(self):
        """Ensures correct output from _check_full_narr_matrix.

        In this case, input matrix is 5-D with the NARR's spatial dimensions
        (good).
        """

        ml_utils._check_full_narr_matrix(FULL_NARR_MATRIX_5D)

    def test_check_full_narr_matrix_bad_dimensions(self):
        """Ensures correct output from _check_full_narr_matrix.

        In this case, dimensions have been permuted, so that spatial dimensions
        are not along the expected axes.
        """

        with self.assertRaises(TypeError):
            ml_utils._check_full_narr_matrix(
                numpy.transpose(FULL_NARR_MATRIX_3D))

    def test_check_predictor_matrix_1d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 1-D (bad).
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_1D)

    def test_check_predictor_matrix_2d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_2D)

    def test_check_predictor_matrix_nan_allowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix contains NaN's (which are allowed).
        """

        ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_3D, allow_nan=True)

    def test_check_predictor_matrix_nan_disallowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix contains NaN's (which are *not* allowed).
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(
                PREDICTOR_MATRIX_3D, allow_nan=False)

    def test_check_predictor_matrix_4d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 4-D (good).
        """

        ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_4D, allow_nan=True)

    def test_check_predictor_matrix_5d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 5-D (good).
        """

        ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_5D, allow_nan=True)

    def test_check_target_matrix_1d_good(self):
        """Ensures correct output from _check_target_matrix.

        In this case, method expects 1-D array and receives 1-D array.
        """

        ml_utils._check_target_matrix(
            TARGET_VALUES_BINARY_1D, assert_binary=True, num_dimensions=1)

    def test_check_target_matrix_1d_bad(self):
        """Ensures correct output from _check_target_matrix.

        In this case, method expects 3-D matrix and receives 1-D array.
        """

        with self.assertRaises(TypeError):
            ml_utils._check_target_matrix(
                TARGET_VALUES_BINARY_1D, assert_binary=False, num_dimensions=3)

    def test_check_target_matrix_3d_good(self):
        """Ensures correct output from _check_target_matrix.

        In this case, method expects 3-D matrix and receives 3-D matrix.
        """

        ml_utils._check_target_matrix(
            TARGET_VALUES_BINARY_3D, assert_binary=True, num_dimensions=3)

    def test_check_target_matrix_3d_bad(self):
        """Ensures correct output from _check_target_matrix.

        In this case, method expects 1-D array and receives 3-D matrix.
        """

        with self.assertRaises(TypeError):
            ml_utils._check_target_matrix(
                TARGET_VALUES_BINARY_3D, assert_binary=False, num_dimensions=1)

    def test_check_target_matrix_2d(self):
        """Ensures correct output from _check_target_matrix.

        In this case, input matrix is 2-D (bad).
        """

        with self.assertRaises(TypeError):
            ml_utils._check_target_matrix(
                TARGET_VALUES_BINARY_2D, assert_binary=False)

    def test_check_target_matrix_non_binary_allowed(self):
        """Ensures correct output from _check_target_matrix.

        In this case, input matrix is non-binary, which is allowed.
        """

        ml_utils._check_target_matrix(
            TARGET_VALUES_BINARY_1D, assert_binary=False, num_dimensions=1)

    def test_check_target_matrix_non_binary_disallowed(self):
        """Ensures correct output from _check_target_matrix.

        In this case, input matrix is non-binary, which is *not* allowed.
        """

        with self.assertRaises(ValueError):
            ml_utils._check_target_matrix(
                TARGET_VALUES_TERNARY_1D, assert_binary=True, num_dimensions=1)

    def test_downsize_predictor_images_top_left_3d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 3-D; center point for extraction is at
        top-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_3D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_LEFT_3D, atol=TOLERANCE))

    def test_downsize_predictor_images_top_left_4d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 4-D; center point for extraction is at
        top-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_4D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_LEFT_4D, atol=TOLERANCE))

    def test_downsize_predictor_images_top_left_5d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 5-D; center point for extraction is at
        top-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_5D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_LEFT_5D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_left_3d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 3-D; center point for extraction is at
        bottom-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_3D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_LEFT_3D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_left_4d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 4-D; center point for extraction is at
        bottom-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_4D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_LEFT_4D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_left_5d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 5-D; center point for extraction is at
        bottom-left of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_5D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_LEFT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_LEFT_5D, atol=TOLERANCE))

    def test_downsize_predictor_images_top_right_3d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 3-D; center point for extraction is at
        top-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_3D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_RIGHT_3D, atol=TOLERANCE))

    def test_downsize_predictor_images_top_right_4d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 4-D; center point for extraction is at
        top-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_4D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_RIGHT_4D, atol=TOLERANCE))

    def test_downsize_predictor_images_top_right_5d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 5-D; center point for extraction is at
        top-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_5D,
            center_row=DOWNSIZING_CENTER_ROW_TOP_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_TOP_RIGHT_5D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_right_3d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 3-D; center point for extraction is at
        bottom-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_3D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_RIGHT_3D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_right_4d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 4-D; center point for extraction is at
        bottom-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_4D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_RIGHT_4D, atol=TOLERANCE))

    def test_downsize_predictor_images_bottom_right_5d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 5-D; center point for extraction is at
        bottom-right of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_5D,
            center_row=DOWNSIZING_CENTER_ROW_BOTTOM_RIGHT,
            center_column=DOWNSIZING_CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_BOTTOM_RIGHT_5D, atol=TOLERANCE))

    def test_downsize_predictor_images_middle_3d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 3-D; center point for extraction is in
        middle of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_3D,
            center_row=DOWNSIZING_CENTER_ROW_MIDDLE,
            center_column=DOWNSIZING_CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_MIDDLE_3D, atol=TOLERANCE))

    def test_downsize_predictor_images_middle_4d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 4-D; center point for extraction is in
        middle of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_4D,
            center_row=DOWNSIZING_CENTER_ROW_MIDDLE,
            center_column=DOWNSIZING_CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_MIDDLE_4D, atol=TOLERANCE))

    def test_downsize_predictor_images_middle_5d(self):
        """Ensures correct output from _downsize_predictor_images.

        In this case, input matrix is 5-D; center point for extraction is in
        middle of input matrix.
        """

        this_matrix = ml_utils._downsize_predictor_images(
            predictor_matrix=PRE_DOWNSIZED_MATRIX_5D,
            center_row=DOWNSIZING_CENTER_ROW_MIDDLE,
            center_column=DOWNSIZING_CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_DOWNSIZED_HALF_GRID,
            num_columns_in_half_window=NUM_COLUMNS_IN_DOWNSIZED_HALF_GRID)

        self.assertTrue(numpy.allclose(
            this_matrix, DOWNSIZED_MATRIX_MIDDLE_5D, atol=TOLERANCE))

    def test_class_fractions_to_num_points_large_binary(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, number of available points is large and there are 2
        classes.
        """

        this_num_points_by_class = ml_utils._class_fractions_to_num_points(
            class_fractions=CLASS_FRACTIONS_BINARY,
            num_points_total=NUM_POINTS_AVAILABLE_LARGE)
        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_LARGE))

    def test_class_fractions_to_num_points_large_ternary(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, number of available points is large and there are 3
        classes.
        """

        this_num_points_by_class = ml_utils._class_fractions_to_num_points(
            class_fractions=CLASS_FRACTIONS_TERNARY,
            num_points_total=NUM_POINTS_AVAILABLE_LARGE)
        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_LARGE))

    def test_class_fractions_to_num_points_small_binary(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, number of available points is small and there are 2
        classes.
        """

        this_num_points_by_class = ml_utils._class_fractions_to_num_points(
            class_fractions=CLASS_FRACTIONS_BINARY,
            num_points_total=NUM_POINTS_AVAILABLE_SMALL)
        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_SMALL))

    def test_class_fractions_to_num_points_small_ternary(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, number of available points is small and there are 3
        classes.
        """

        this_num_points_by_class = ml_utils._class_fractions_to_num_points(
            class_fractions=CLASS_FRACTIONS_TERNARY,
            num_points_total=NUM_POINTS_AVAILABLE_SMALL)
        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_SMALL))

    def test_get_class_weight_dict_binary(self):
        """Ensures correct output from get_class_weight_dict.

        In this case, input contains 2 classes.
        """

        this_class_weight_dict = ml_utils.get_class_weight_dict(
            CLASS_FRACTIONS_BINARY)

        self.assertTrue(set(this_class_weight_dict.keys()) ==
                        set(CLASS_WEIGHT_DICT_BINARY.keys()))

        for this_key in this_class_weight_dict.keys():
            self.assertTrue(numpy.isclose(
                this_class_weight_dict[this_key],
                CLASS_WEIGHT_DICT_BINARY[this_key],
                atol=TOLERANCE_FOR_CLASS_WEIGHT))

    def test_get_class_weight_dict_ternary(self):
        """Ensures correct output from get_class_weight_dict.

        In this case, input contains 3 classes.
        """

        this_class_weight_dict = ml_utils.get_class_weight_dict(
            CLASS_FRACTIONS_TERNARY)

        self.assertTrue(set(this_class_weight_dict.keys()) ==
                        set(CLASS_WEIGHT_DICT_TERNARY.keys()))

        for this_key in this_class_weight_dict.keys():
            self.assertTrue(numpy.isclose(
                this_class_weight_dict[this_key],
                CLASS_WEIGHT_DICT_TERNARY[this_key],
                atol=TOLERANCE_FOR_CLASS_WEIGHT))

    def test_normalize_predictors_4d_minmax(self):
        """Ensures correct output from normalize_predictors.

        In this case, predictor matrix is 4-D (no time dimension) and
        normalization method is min-max.
        """

        this_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=PREDICTOR_MATRIX_4D_DENORM + 0.,
            normalization_type_string=ml_utils.MINMAX_STRING,
            percentile_offset=PRCTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_MINMAX_NORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_normalize_predictors_5d_minmax(self):
        """Ensures correct output from normalize_predictors.

        In this case, predictor matrix is 5-D (has time dimension) and
        normalization method is min-max.
        """

        this_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=PREDICTOR_MATRIX_5D_DENORM + 0.,
            normalization_type_string=ml_utils.MINMAX_STRING,
            percentile_offset=PRCTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_MINMAX_NORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_normalize_predictors_4d_z(self):
        """Ensures correct output from normalize_predictors.

        In this case, predictor matrix is 4-D (no time dimension) and
        normalization method is z-score.
        """

        this_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=PREDICTOR_MATRIX_4D_DENORM + 0.,
            normalization_type_string=ml_utils.Z_SCORE_STRING)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_Z_NORM, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_normalize_predictors_5d_z(self):
        """Ensures correct output from normalize_predictors.

        In this case, predictor matrix is 5-D (has time dimension) and
        normalization method is z-score.
        """

        this_predictor_matrix, _ = ml_utils.normalize_predictors(
            predictor_matrix=PREDICTOR_MATRIX_5D_DENORM + 0.,
            normalization_type_string=ml_utils.Z_SCORE_STRING)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_Z_NORM, atol=TOLERANCE,
            equal_nan=True
        ))

    def test_denormalize_predictors_4d_minmax(self):
        """Ensures correct output from denormalize_predictors.

        In this case, predictor matrix is 4-D (no time dimension) and
        normalization method is min-max.
        """

        this_predictor_matrix, this_normalization_dict = (
            ml_utils.normalize_predictors(
                predictor_matrix=PREDICTOR_MATRIX_4D_DENORM + 0.,
                normalization_type_string=ml_utils.MINMAX_STRING,
                percentile_offset=PRCTILE_OFFSET_FOR_NORMALIZATION)
        )

        this_predictor_matrix = ml_utils.denormalize_predictors(
            predictor_matrix=this_predictor_matrix,
            normalization_dict=this_normalization_dict)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_DENORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_denormalize_predictors_5d_minmax(self):
        """Ensures correct output from denormalize_predictors.

        In this case, predictor matrix is 5-D (no time dimension) and
        normalization method is min-max.
        """

        this_predictor_matrix, this_normalization_dict = (
            ml_utils.normalize_predictors(
                predictor_matrix=PREDICTOR_MATRIX_5D_DENORM + 0.,
                normalization_type_string=ml_utils.MINMAX_STRING,
                percentile_offset=PRCTILE_OFFSET_FOR_NORMALIZATION)
        )

        this_predictor_matrix = ml_utils.denormalize_predictors(
            predictor_matrix=this_predictor_matrix,
            normalization_dict=this_normalization_dict)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_DENORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_denormalize_predictors_4d_z(self):
        """Ensures correct output from denormalize_predictors.

        In this case, predictor matrix is 4-D (no time dimension) and
        normalization method is z-score.
        """

        this_predictor_matrix, this_normalization_dict = (
            ml_utils.normalize_predictors(
                predictor_matrix=PREDICTOR_MATRIX_4D_DENORM + 0.,
                normalization_type_string=ml_utils.Z_SCORE_STRING)
        )

        this_predictor_matrix = ml_utils.denormalize_predictors(
            predictor_matrix=this_predictor_matrix,
            normalization_dict=this_normalization_dict)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_DENORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_denormalize_predictors_5d_z(self):
        """Ensures correct output from denormalize_predictors.

        In this case, predictor matrix is 5-D (no time dimension) and
        normalization method is z-score.
        """

        this_predictor_matrix, this_normalization_dict = (
            ml_utils.normalize_predictors(
                predictor_matrix=PREDICTOR_MATRIX_5D_DENORM + 0.,
                normalization_type_string=ml_utils.Z_SCORE_STRING)
        )

        this_predictor_matrix = ml_utils.denormalize_predictors(
            predictor_matrix=this_predictor_matrix,
            normalization_dict=this_normalization_dict)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_DENORM,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_sample_target_points_binary_no_mask(self):
        """Ensures correct output from sample_target_points.

        In this case, there are 2 classes and no mask.
        """

        this_target_point_dict = ml_utils.sample_target_points(
            target_matrix=FRONTAL_GRID_MATRIX_BINARY,
            class_fractions=CLASS_FRACTIONS_FOR_BINARY_SAMPLING,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, mask_matrix=None,
            test_mode=True)

        self.assertTrue(_compare_target_point_dicts(
            this_target_point_dict, TARGET_POINT_DICT_BINARY_NO_MASK))

    def test_sample_target_points_binary_with_mask(self):
        """Ensures correct output from sample_target_points.

        In this case, there are 2 classes with a mask.
        """

        this_target_point_dict = ml_utils.sample_target_points(
            target_matrix=FRONTAL_GRID_MATRIX_BINARY,
            class_fractions=CLASS_FRACTIONS_FOR_BINARY_SAMPLING,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, mask_matrix=MASK_MATRIX,
            test_mode=True)

        self.assertTrue(_compare_target_point_dicts(
            this_target_point_dict, TARGET_POINT_DICT_BINARY_WITH_MASK))

    def test_sample_target_points_ternary_no_mask(self):
        """Ensures correct output from sample_target_points.

        In this case, there are 3 classes and no mask.
        """

        this_target_point_dict = ml_utils.sample_target_points(
            target_matrix=FRONTAL_GRID_MATRIX_TERNARY,
            class_fractions=CLASS_FRACTIONS_FOR_TERNARY_SAMPLING,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, test_mode=True)

        self.assertTrue(_compare_target_point_dicts(
            this_target_point_dict, TARGET_POINT_DICT_TERNARY_NO_MASK))

    def test_sample_target_points_ternary_with_mask(self):
        """Ensures correct output from sample_target_points.

        In this case, there are 3 classes with a mask.
        """

        this_target_point_dict = ml_utils.sample_target_points(
            target_matrix=FRONTAL_GRID_MATRIX_TERNARY,
            class_fractions=CLASS_FRACTIONS_FOR_TERNARY_SAMPLING,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, mask_matrix=MASK_MATRIX,
            test_mode=True)

        self.assertTrue(_compare_target_point_dicts(
            this_target_point_dict, TARGET_POINT_DICT_TERNARY_WITH_MASK))

    def test_front_table_to_images(self):
        """Ensures correct output from front_table_to_images."""

        this_frontal_grid_matrix = ml_utils.front_table_to_images(
            frontal_grid_table=FRONTAL_GRID_TABLE,
            num_rows_per_image=NUM_GRID_ROWS,
            num_columns_per_image=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_frontal_grid_matrix, FRONTAL_GRID_MATRIX_TERNARY))

    def test_binarize_front_images(self):
        """Ensures correct output from binarize_front_images."""

        this_input_matrix = copy.deepcopy(FRONTAL_GRID_MATRIX_TERNARY)
        this_binary_matrix = ml_utils.binarize_front_images(this_input_matrix)

        self.assertTrue(numpy.array_equal(
            this_binary_matrix, FRONTAL_GRID_MATRIX_BINARY))

    def test_dilate_binary_target_images(self):
        """Ensures correct output from dilate_binary_target_images."""

        this_input_matrix = copy.deepcopy(FRONTAL_GRID_MATRIX_BINARY)
        this_dilated_matrix = ml_utils.dilate_binary_target_images(
            target_matrix=this_input_matrix,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        self.assertTrue(numpy.array_equal(
            this_dilated_matrix, FRONTAL_GRID_MATRIX_BINARY_DILATED))

    def test_dilate_ternary_target_images(self):
        """Ensures correct output from dilate_ternary_target_images."""

        this_input_matrix = copy.deepcopy(FRONTAL_GRID_MATRIX_TERNARY)
        this_dilated_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=this_input_matrix,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        self.assertTrue(numpy.array_equal(
            this_dilated_matrix, FRONTAL_GRID_MATRIX_TERNARY_DILATED))

    def test_subset_narr_grid_for_fcn_input_3d(self):
        """Ensures correct output from subset_narr_grid_for_fcn_input.

        In this case, input matrix is 3-D.
        """

        this_matrix = ml_utils.subset_narr_grid_for_fcn_input(
            FULL_NARR_MATRIX_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, FCN_INPUT_MATRIX_3D, atol=TOLERANCE))

    def test_subset_narr_grid_for_fcn_input_4d(self):
        """Ensures correct output from subset_narr_grid_for_fcn_input.

        In this case, input matrix is 4-D.
        """

        this_matrix = ml_utils.subset_narr_grid_for_fcn_input(
            FULL_NARR_MATRIX_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, FCN_INPUT_MATRIX_4D, atol=TOLERANCE))

    def test_subset_narr_grid_for_fcn_input_5d(self):
        """Ensures correct output from subset_narr_grid_for_fcn_input.

        In this case, input matrix is 5-D.
        """

        this_matrix = ml_utils.subset_narr_grid_for_fcn_input(
            FULL_NARR_MATRIX_5D)
        self.assertTrue(numpy.allclose(
            this_matrix, FCN_INPUT_MATRIX_5D, atol=TOLERANCE))

    def test_downsize_grids_around_selected_points(self):
        """Ensures correct output from downsize_grids_around_selected_points."""

        this_full_predictor_matrix = copy.deepcopy(
            PREDICTOR_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS)

        (this_small_predictor_matrix,
         this_target_vector,
         these_example_indices,
         these_center_rows,
         these_center_columns) = ml_utils.downsize_grids_around_selected_points(
             predictor_matrix=this_full_predictor_matrix,
             target_matrix=TARGET_MATRIX_TO_DOWNSIZE_AT_SELECTED_PTS,
             num_rows_in_half_window=NUM_ROWS_IN_HALF_GRID_AROUND_SELECTED_PTS,
             num_columns_in_half_window=
             NUM_COLUMNS_IN_HALF_GRID_AROUND_SELECTED_PTS,
             target_point_dict=TARGET_POINT_DICT_FOR_DOWNSIZING, test_mode=True)

        self.assertTrue(numpy.allclose(
            this_small_predictor_matrix, DOWNSIZED_MATRIX_AT_SELECTED_POINTS,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_target_vector, TARGET_VECTOR_AT_SELECTED_POINTS
        ))
        self.assertTrue(numpy.array_equal(
            these_example_indices, EXAMPLE_INDICES_AT_SELECTED_POINTS
        ))
        self.assertTrue(numpy.array_equal(
            these_center_rows, CENTER_ROWS_AT_SELECTED_POINTS
        ))
        self.assertTrue(numpy.array_equal(
            these_center_columns, CENTER_COLUMNS_AT_SELECTED_POINTS
        ))


if __name__ == '__main__':
    unittest.main()
