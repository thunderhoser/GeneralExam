"""Unit tests for machine_learning_utils.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6

# The following constants are used to test _check_predictor_matrix.
PREDICTOR_MATRIX_1D = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
PREDICTOR_MATRIX_2D = numpy.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8]], dtype=numpy.float32)

PREDICTOR_MATRIX_3D = numpy.stack(
    (PREDICTOR_MATRIX_2D, PREDICTOR_MATRIX_2D), axis=-1)
PREDICTOR_MATRIX_3D[0, 0, 0] = numpy.nan

PREDICTOR_MATRIX_4D = numpy.stack(
    (PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D), axis=-1)
PREDICTOR_MATRIX_5D = numpy.stack(
    (PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D), axis=-1)

# The following constants are used to test _downsize_grid.
FULL_GRID_MATRIX = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                                [19, 20, 21, 22, 23, 24, 25, 26, 27],
                                [28, 29, 30, 31, 32, 33, 34, 35, 36],
                                [37, 38, 39, 40, 41, 42, 43, 44, 45],
                                [46, 47, 48, 49, 50, 51, 52, 53, 54],
                                [55, 56, 57, 58, 59, 60, 61, 62, 63]],
                               dtype=numpy.float32)

CENTER_ROW_TOP_LEFT = 1
CENTER_COLUMN_TOP_LEFT = 1
CENTER_ROW_BOTTOM_LEFT = 5
CENTER_COLUMN_BOTTOM_LEFT = 1
CENTER_ROW_TOP_RIGHT = 1
CENTER_COLUMN_TOP_RIGHT = 7
CENTER_ROW_BOTTOM_RIGHT = 5
CENTER_COLUMN_BOTTOM_RIGHT = 7
CENTER_ROW_MIDDLE = 3
CENTER_COLUMN_MIDDLE = 4

NUM_ROWS_IN_HALF_WINDOW = 2
NUM_COLUMNS_IN_HALF_WINDOW = 3

SMALL_GRID_MATRIX_TOP_LEFT = numpy.array([[1, 1, 1, 2, 3, 4, 5],
                                          [1, 1, 1, 2, 3, 4, 5],
                                          [10, 10, 10, 11, 12, 13, 14],
                                          [19, 19, 19, 20, 21, 22, 23],
                                          [28, 28, 28, 29, 30, 31, 32]],
                                         dtype=numpy.float32)

SMALL_GRID_MATRIX_BOTTOM_LEFT = numpy.array([[28, 28, 28, 29, 30, 31, 32],
                                             [37, 37, 37, 38, 39, 40, 41],
                                             [46, 46, 46, 47, 48, 49, 50],
                                             [55, 55, 55, 56, 57, 58, 59],
                                             [55, 55, 55, 56, 57, 58, 59]],
                                            dtype=numpy.float32)

SMALL_GRID_MATRIX_TOP_RIGHT = numpy.array([[5, 6, 7, 8, 9, 9, 9],
                                           [5, 6, 7, 8, 9, 9, 9],
                                           [14, 15, 16, 17, 18, 18, 18],
                                           [23, 24, 25, 26, 27, 27, 27],
                                           [32, 33, 34, 35, 36, 36, 36]],
                                          dtype=numpy.float32)

SMALL_GRID_MATRIX_BOTTOM_RIGHT = numpy.array([[32, 33, 34, 35, 36, 36, 36],
                                              [41, 42, 43, 44, 45, 45, 45],
                                              [50, 51, 52, 53, 54, 54, 54],
                                              [59, 60, 61, 62, 63, 63, 63],
                                              [59, 60, 61, 62, 63, 63, 63]],
                                             dtype=numpy.float32)

SMALL_GRID_MATRIX_MIDDLE = numpy.array([[11, 12, 13, 14, 15, 16, 17],
                                        [20, 21, 22, 23, 24, 25, 26],
                                        [29, 30, 31, 32, 33, 34, 35],
                                        [38, 39, 40, 41, 42, 43, 44],
                                        [47, 48, 49, 50, 51, 52, 53]],
                                       dtype=numpy.float32)

FULL_GRID_MATRIX_3D = numpy.stack(
    (FULL_GRID_MATRIX, FULL_GRID_MATRIX), axis=0)
FULL_GRID_MATRIX_4D = numpy.stack(
    (FULL_GRID_MATRIX_3D, FULL_GRID_MATRIX_3D), axis=-1)

SMALL_GRID_MATRIX_TOP_LEFT_3D = numpy.stack(
    (SMALL_GRID_MATRIX_TOP_LEFT, SMALL_GRID_MATRIX_TOP_LEFT), axis=0)
SMALL_GRID_MATRIX_TOP_LEFT_4D = numpy.stack(
    (SMALL_GRID_MATRIX_TOP_LEFT_3D, SMALL_GRID_MATRIX_TOP_LEFT_3D), axis=-1)

SMALL_GRID_MATRIX_BOTTOM_LEFT_3D = numpy.stack(
    (SMALL_GRID_MATRIX_BOTTOM_LEFT, SMALL_GRID_MATRIX_BOTTOM_LEFT), axis=0)
SMALL_GRID_MATRIX_BOTTOM_LEFT_4D = numpy.stack(
    (SMALL_GRID_MATRIX_BOTTOM_LEFT_3D, SMALL_GRID_MATRIX_BOTTOM_LEFT_3D),
    axis=-1)

SMALL_GRID_MATRIX_TOP_RIGHT_3D = numpy.stack(
    (SMALL_GRID_MATRIX_TOP_RIGHT, SMALL_GRID_MATRIX_TOP_RIGHT), axis=0)
SMALL_GRID_MATRIX_TOP_RIGHT_4D = numpy.stack(
    (SMALL_GRID_MATRIX_TOP_RIGHT_3D, SMALL_GRID_MATRIX_TOP_RIGHT_3D), axis=-1)

SMALL_GRID_MATRIX_BOTTOM_RIGHT_3D = numpy.stack(
    (SMALL_GRID_MATRIX_BOTTOM_RIGHT, SMALL_GRID_MATRIX_BOTTOM_RIGHT), axis=0)
SMALL_GRID_MATRIX_BOTTOM_RIGHT_4D = numpy.stack(
    (SMALL_GRID_MATRIX_BOTTOM_RIGHT_3D, SMALL_GRID_MATRIX_BOTTOM_RIGHT_3D),
    axis=-1)

SMALL_GRID_MATRIX_MIDDLE_3D = numpy.stack(
    (SMALL_GRID_MATRIX_MIDDLE, SMALL_GRID_MATRIX_MIDDLE), axis=0)
SMALL_GRID_MATRIX_MIDDLE_4D = numpy.stack(
    (SMALL_GRID_MATRIX_MIDDLE_3D, SMALL_GRID_MATRIX_MIDDLE_3D), axis=-1)

# The following constants are used to test front_table_to_matrices.
NUM_GRID_ROWS = 6
NUM_GRID_COLUMNS = 8
THESE_WF_ROW_INDICES = [numpy.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)]
THESE_WF_COLUMN_INDICES = [numpy.array([1, 2, 7, 2, 3, 4, 5, 6, 7], dtype=int)]
THESE_CF_ROW_INDICES = [numpy.array([1, 2, 3, 4, 4, 5], dtype=int)]
THESE_CF_COLUMN_INDICES = [numpy.array([1, 1, 1, 0, 1, 0], dtype=int)]

THIS_DICT = {
    front_utils.WARM_FRONT_ROW_INDICES_COLUMN: THESE_WF_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN: THESE_WF_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROW_INDICES_COLUMN: THESE_CF_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN: THESE_CF_COLUMN_INDICES
}
FRONTAL_GRID_TABLE1 = pandas.DataFrame.from_dict(THIS_DICT)

THESE_WF_ROW_INDICES = [numpy.array([1, 1, 2, 2, 3, 3], dtype=int)]
THESE_WF_COLUMN_INDICES = [numpy.array([4, 5, 5, 6, 6, 7], dtype=int)]
THESE_CF_ROW_INDICES = [
    numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=int)]
THESE_CF_COLUMN_INDICES = [
    numpy.array([2, 3, 1, 2, 0, 1, 0, 1, 0, 1, 1], dtype=int)]

THIS_DICT = {
    front_utils.WARM_FRONT_ROW_INDICES_COLUMN: THESE_WF_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN: THESE_WF_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROW_INDICES_COLUMN: THESE_CF_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN: THESE_CF_COLUMN_INDICES
}
FRONTAL_GRID_TABLE2 = pandas.DataFrame.from_dict(THIS_DICT)
FRONTAL_GRID_TABLE = pandas.concat(
    [FRONTAL_GRID_TABLE1, FRONTAL_GRID_TABLE2], axis=0, ignore_index=True)

FRONTAL_GRID_MATRIX1 = numpy.array([[0, 1, 1, 0, 0, 0, 0, 1],
                                    [0, 2, 1, 1, 1, 1, 1, 1],
                                    [0, 2, 0, 0, 0, 0, 0, 0],
                                    [0, 2, 0, 0, 0, 0, 0, 0],
                                    [2, 2, 0, 0, 0, 0, 0, 0],
                                    [2, 0, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX2 = numpy.array([[0, 0, 2, 2, 0, 0, 0, 0],
                                    [0, 2, 2, 0, 1, 1, 0, 0],
                                    [2, 2, 0, 0, 0, 1, 1, 0],
                                    [2, 2, 0, 0, 0, 0, 1, 1],
                                    [2, 2, 0, 0, 0, 0, 0, 0],
                                    [0, 2, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX = numpy.stack(
    (FRONTAL_GRID_MATRIX1, FRONTAL_GRID_MATRIX2), axis=0)

# The following constants are used to test binarize_front_labels.
FRONTAL_GRID_MATRIX1_BINARY = numpy.array([[0, 1, 1, 0, 0, 0, 0, 1],
                                           [0, 1, 1, 1, 1, 1, 1, 1],
                                           [0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0, 0],
                                           [1, 1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX2_BINARY = numpy.array([[0, 0, 1, 1, 0, 0, 0, 0],
                                           [0, 1, 1, 0, 1, 1, 0, 0],
                                           [1, 1, 0, 0, 0, 1, 1, 0],
                                           [1, 1, 0, 0, 0, 0, 1, 1],
                                           [1, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0, 0]])

FRONTAL_GRID_MATRIX_BINARY = numpy.stack(
    (FRONTAL_GRID_MATRIX1_BINARY, FRONTAL_GRID_MATRIX2_BINARY), axis=0)

# The following constants are used to test _sample_target_points.
POSITIVE_FRACTION_FOR_SAMPLING = 0.5
POSITIVE_ROW_INDICES_TIME1 = numpy.array(
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5])
POSITIVE_COLUMN_INDICES_TIME1 = numpy.array(
    [1, 2, 7, 1, 2, 3, 4, 5, 6, 7, 1, 1, 0, 1, 0])

THESE_ROW_INDICES_TIME2 = numpy.array(
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5], dtype=int)
THESE_COLUMN_INDICES_TIME2 = numpy.array(
    [2, 3, 1, 2, 4, 5, 0, 1, 5, 6, 0, 1, 6, 7, 0, 1, 1], dtype=int)

NEGATIVE_ROW_INDICES_TIME1 = numpy.array([0, 0, 0, 0, 0,
                                          1,
                                          2, 2, 2, 2, 2, 2, 2,
                                          3, 3, 3, 3, 3, 3, 3,
                                          4, 4, 4, 4, 4, 4,
                                          5, 5, 5, 5, 5, 5])
NEGATIVE_COLUMN_INDICES_TIME1 = numpy.array([0, 3, 4, 5, 6,
                                             0,
                                             0, 2, 3, 4, 5, 6, 7,
                                             0, 2, 3, 4, 5, 6, 7,
                                             2, 3, 4, 5, 6, 7,
                                             1, 2, 3, 4, 5, 6])

THESE_ROW_INDICES_TIME1 = numpy.concatenate((
    POSITIVE_ROW_INDICES_TIME1, NEGATIVE_ROW_INDICES_TIME1)).astype(int)
THESE_COLUMN_INDICES_TIME1 = numpy.concatenate((
    POSITIVE_COLUMN_INDICES_TIME1, NEGATIVE_COLUMN_INDICES_TIME1)).astype(int)

THESE_ROW_INDICES_BY_TIME = [
    THESE_ROW_INDICES_TIME1, THESE_ROW_INDICES_TIME2]
THESE_COLUMN_INDICES_BY_TIME = [
    THESE_COLUMN_INDICES_TIME1, THESE_COLUMN_INDICES_TIME2]

SAMPLED_TARGET_POINT_DICT = {
    ml_utils.ROW_INDICES_BY_TIME_KEY: THESE_ROW_INDICES_BY_TIME,
    ml_utils.COLUMN_INDICES_BY_TIME_KEY: THESE_COLUMN_INDICES_BY_TIME
}

# The following constants are used to test stack_predictor_variables.
TUPLE_OF_PREDICTOR_MATRICES = (PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D)

# The following constants are used to test remove_nans_from_narr_grid.
NUM_NARR_ROWS, NUM_NARR_COLUMNS = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME)

NARR_MATRIX_2D = numpy.random.uniform(
    low=0., high=1., size=(NUM_NARR_ROWS, NUM_NARR_COLUMNS))
NARR_MATRIX_2D_WITHOUT_NAN = NARR_MATRIX_2D[
    :, ml_utils.FIRST_NARR_COLUMN_WITHOUT_NAN:
    (ml_utils.LAST_NARR_COLUMN_WITHOUT_NAN + 1)]

NARR_MATRIX_3D = numpy.stack((NARR_MATRIX_2D, NARR_MATRIX_2D), axis=0)
NARR_MATRIX_3D_WITHOUT_NAN = numpy.stack(
    (NARR_MATRIX_2D_WITHOUT_NAN, NARR_MATRIX_2D_WITHOUT_NAN), axis=0)

NARR_MATRIX_4D = numpy.stack((NARR_MATRIX_3D, NARR_MATRIX_3D), axis=-1)
NARR_MATRIX_4D_WITHOUT_NAN = numpy.stack(
    (NARR_MATRIX_3D_WITHOUT_NAN, NARR_MATRIX_3D_WITHOUT_NAN), axis=-1)

# The following constants are used to test downsize_grids_around_each_point.
FULL_PREDICTOR_MATRIX_TOY_EXAMPLE = numpy.array([[1, 3, 5, 7],
                                                 [2, 4, 6, 8]],
                                                dtype=numpy.float32)

FULL_PREDICTOR_MATRIX_TOY_EXAMPLE = numpy.stack(
    (FULL_PREDICTOR_MATRIX_TOY_EXAMPLE,), axis=0)
FULL_PREDICTOR_MATRIX_TOY_EXAMPLE = numpy.stack(
    (FULL_PREDICTOR_MATRIX_TOY_EXAMPLE, FULL_PREDICTOR_MATRIX_TOY_EXAMPLE,
     FULL_PREDICTOR_MATRIX_TOY_EXAMPLE), axis=-1)

SMALL_PREDICTOR_MATRIX_R1_C1 = numpy.array([[1, 1, 3],
                                            [1, 1, 3],
                                            [2, 2, 4]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R1_C2 = numpy.array([[1, 3, 5],
                                            [1, 3, 5],
                                            [2, 4, 6]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R1_C3 = numpy.array([[3, 5, 7],
                                            [3, 5, 7],
                                            [4, 6, 8]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R1_C4 = numpy.array([[5, 7, 7],
                                            [5, 7, 7],
                                            [6, 8, 8]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R2_C1 = numpy.array([[1, 1, 3],
                                            [2, 2, 4],
                                            [2, 2, 4]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R2_C2 = numpy.array([[1, 3, 5],
                                            [2, 4, 6],
                                            [2, 4, 6]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R2_C3 = numpy.array([[3, 5, 7],
                                            [4, 6, 8],
                                            [4, 6, 8]], dtype=numpy.float32)
SMALL_PREDICTOR_MATRIX_R2_C4 = numpy.array([[5, 7, 7],
                                            [6, 8, 8],
                                            [6, 8, 8]], dtype=numpy.float32)

SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE = numpy.stack((
    SMALL_PREDICTOR_MATRIX_R1_C1, SMALL_PREDICTOR_MATRIX_R1_C2,
    SMALL_PREDICTOR_MATRIX_R1_C3, SMALL_PREDICTOR_MATRIX_R1_C4,
    SMALL_PREDICTOR_MATRIX_R2_C1, SMALL_PREDICTOR_MATRIX_R2_C2,
    SMALL_PREDICTOR_MATRIX_R2_C3, SMALL_PREDICTOR_MATRIX_R2_C4), axis=0)
SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE = numpy.stack(
    (SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE, SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE,
     SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE), axis=-1)

TARGET_MATRIX_TOY_EXAMPLE = numpy.array([[0, 0, 1, 1],
                                         [2, 2, 0, 0]], dtype=int)
TARGET_MATRIX_TOY_EXAMPLE = numpy.stack((TARGET_MATRIX_TOY_EXAMPLE,), axis=0)

TARGET_VECTOR_TOY_EXAMPLE = numpy.array([0, 0, 1, 1, 2, 2, 0, 0], dtype=int)
NUM_ROWS_IN_HALF_WINDOW_TOY_EXAMPLE = 1
NUM_COLUMNS_IN_HALF_WINDOW_TOY_EXAMPLE = 1

TARGET_POINT_DICT_FOR_DOWNSIZING = {
    ml_utils.ROW_INDICES_BY_TIME_KEY: [numpy.array([0, 0, 1, 1], dtype=int)],
    ml_utils.COLUMN_INDICES_BY_TIME_KEY: [numpy.array([2, 1, 3, 0], dtype=int)]
}

SMALL_PREDICTOR_MATRIX_SELECTED_POINTS = numpy.stack((
    SMALL_PREDICTOR_MATRIX_R1_C3, SMALL_PREDICTOR_MATRIX_R1_C2,
    SMALL_PREDICTOR_MATRIX_R2_C4, SMALL_PREDICTOR_MATRIX_R2_C1), axis=0)
SMALL_PREDICTOR_MATRIX_SELECTED_POINTS = numpy.stack(
    (SMALL_PREDICTOR_MATRIX_SELECTED_POINTS,
     SMALL_PREDICTOR_MATRIX_SELECTED_POINTS,
     SMALL_PREDICTOR_MATRIX_SELECTED_POINTS), axis=-1)

TARGET_VECTOR_SELECTED_POINTS = numpy.array([1, 0, 0, 2], dtype=int)


class MachineLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for machine_learning_utils.py."""

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

    def test_check_predictor_matrix_3d_nan_allowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 3-D and NaN's are allowed.
        """

        ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_3D, allow_nan=True)

    def test_check_predictor_matrix_3d_nan_disallowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 3-D and NaN's are not allowed.
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(
                PREDICTOR_MATRIX_3D, allow_nan=False)

    def test_check_predictor_matrix_4d_nan_allowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 4-D and NaN's are allowed.
        """

        ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_4D, allow_nan=True)

    def test_check_predictor_matrix_4d_nan_disallowed(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 4-D and NaN's are not allowed.
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(
                PREDICTOR_MATRIX_4D, allow_nan=False)

    def test_check_predictor_matrix_5d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 5-D (bad).
        """

        with self.assertRaises(ValueError):
            ml_utils._check_predictor_matrix(PREDICTOR_MATRIX_5D)

    def test_downsize_grid_top_left_3d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 3-D; center point for extraction is at
        top-left of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_3D,
            center_row=CENTER_ROW_TOP_LEFT,
            center_column=CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_LEFT_3D, atol=TOLERANCE))

    def test_downsize_grid_top_left_4d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 4-D; center point for extraction is at
        top-left of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_4D,
            center_row=CENTER_ROW_TOP_LEFT,
            center_column=CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_LEFT_4D, atol=TOLERANCE))

    def test_downsize_grid_bottom_left_3d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 3-D; center point for extraction is at
        bottom-left of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_3D,
            center_row=CENTER_ROW_BOTTOM_LEFT,
            center_column=CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_LEFT_3D, atol=TOLERANCE))

    def test_downsize_grid_bottom_left_4d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 4-D; center point for extraction is at
        bottom-left of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_4D,
            center_row=CENTER_ROW_BOTTOM_LEFT,
            center_column=CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_LEFT_4D, atol=TOLERANCE))

    def test_downsize_grid_top_right_3d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 3-D; center point for extraction is at
        top-right of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_3D,
            center_row=CENTER_ROW_TOP_RIGHT,
            center_column=CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_RIGHT_3D, atol=TOLERANCE))

    def test_downsize_grid_top_right_4d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 4-D; center point for extraction is at
        top-right of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_4D,
            center_row=CENTER_ROW_TOP_RIGHT,
            center_column=CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_RIGHT_4D, atol=TOLERANCE))

    def test_downsize_grid_bottom_right_3d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 3-D; center point for extraction is at
        bottom-right of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_3D,
            center_row=CENTER_ROW_BOTTOM_RIGHT,
            center_column=CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_RIGHT_3D, atol=TOLERANCE))

    def test_downsize_grid_bottom_right_4d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 4-D; center point for extraction is at
        bottom-right of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_4D,
            center_row=CENTER_ROW_BOTTOM_RIGHT,
            center_column=CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_RIGHT_4D, atol=TOLERANCE))

    def test_downsize_grid_middle_3d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 3-D; center point for extraction is in
        middle of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_3D,
            center_row=CENTER_ROW_MIDDLE, center_column=CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_MIDDLE_3D, atol=TOLERANCE))

    def test_downsize_grid_middle_4d(self):
        """Ensures correct output from _downsize_grid.

        In this case, input matrix is 4-D; center point for extraction is in
        middle of input matrix.
        """

        this_matrix = ml_utils._downsize_grid(
            full_grid_matrix=FULL_GRID_MATRIX_4D,
            center_row=CENTER_ROW_MIDDLE, center_column=CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_MIDDLE_4D, atol=TOLERANCE))

    def test_sample_target_points(self):
        """Ensures correct output from _sample_target_points."""

        this_target_point_dict = ml_utils._sample_target_points(
            binary_target_matrix=FRONTAL_GRID_MATRIX_BINARY,
            positive_fraction=POSITIVE_FRACTION_FOR_SAMPLING, test_mode=True)

        self.assertTrue(set(this_target_point_dict.keys()) ==
                        set(SAMPLED_TARGET_POINT_DICT.keys()))

        this_num_times = len(
            this_target_point_dict[ml_utils.ROW_INDICES_BY_TIME_KEY])
        expected_num_times = len(
            SAMPLED_TARGET_POINT_DICT[ml_utils.ROW_INDICES_BY_TIME_KEY])
        self.assertTrue(this_num_times == expected_num_times)

        for i in range(expected_num_times):
            self.assertTrue(numpy.array_equal(
                this_target_point_dict[ml_utils.ROW_INDICES_BY_TIME_KEY][i],
                SAMPLED_TARGET_POINT_DICT[ml_utils.ROW_INDICES_BY_TIME_KEY][i]))
            self.assertTrue(numpy.array_equal(
                this_target_point_dict[ml_utils.COLUMN_INDICES_BY_TIME_KEY][i],
                SAMPLED_TARGET_POINT_DICT[
                    ml_utils.COLUMN_INDICES_BY_TIME_KEY][i]))

    def test_front_table_to_matrices(self):
        """Ensures correct output from front_table_to_matrices."""

        this_frontal_grid_matrix = ml_utils.front_table_to_matrices(
            frontal_grid_table=FRONTAL_GRID_TABLE, num_grid_rows=NUM_GRID_ROWS,
            num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_frontal_grid_matrix, FRONTAL_GRID_MATRIX))

    def test_binarize_front_labels(self):
        """Ensures correct output from binarize_front_labels."""

        this_input_matrix = copy.deepcopy(FRONTAL_GRID_MATRIX)
        this_binary_matrix = ml_utils.binarize_front_labels(this_input_matrix)

        self.assertTrue(numpy.array_equal(
            this_binary_matrix, FRONTAL_GRID_MATRIX_BINARY))

    def test_stack_predictor_variables(self):
        """Ensures correct output from stack_predictor_variables."""

        this_matrix = ml_utils.stack_predictor_variables(
            TUPLE_OF_PREDICTOR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, PREDICTOR_MATRIX_4D, atol=TOLERANCE, equal_nan=True))

    def test_remove_nans_from_narr_grid_3d(self):
        """Ensures correct output from remove_nans_from_narr_grid.

        In this case, input matrix is 3-D.
        """

        this_matrix = ml_utils.remove_nans_from_narr_grid(NARR_MATRIX_3D)
        self.assertTrue(numpy.allclose(
            this_matrix, NARR_MATRIX_3D_WITHOUT_NAN, atol=TOLERANCE))

    def test_remove_nans_from_narr_grid_4d(self):
        """Ensures correct output from remove_nans_from_narr_grid.

        In this case, input matrix is 4-D.
        """

        this_matrix = ml_utils.remove_nans_from_narr_grid(NARR_MATRIX_4D)
        self.assertTrue(numpy.allclose(
            this_matrix, NARR_MATRIX_4D_WITHOUT_NAN, atol=TOLERANCE))

    def test_downsize_grids_around_each_point(self):
        """Ensures correct output from downsize_grids_around_each_point."""

        this_full_predictor_matrix = copy.deepcopy(
            FULL_PREDICTOR_MATRIX_TOY_EXAMPLE)

        this_small_predictor_matrix, this_target_vector = (
            ml_utils.downsize_grids_around_each_point(
                predictor_matrix=this_full_predictor_matrix,
                target_matrix=TARGET_MATRIX_TOY_EXAMPLE,
                num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW_TOY_EXAMPLE,
                num_columns_in_half_window=
                NUM_COLUMNS_IN_HALF_WINDOW_TOY_EXAMPLE, test_mode=True))

        self.assertTrue(numpy.allclose(
            this_small_predictor_matrix, SMALL_PREDICTOR_MATRIX_TOY_EXAMPLE,
            atol=TOLERANCE))
        self.assertTrue(numpy.array_equal(
            this_target_vector, TARGET_VECTOR_TOY_EXAMPLE))

    def test_downsize_grids_around_selected_points(self):
        """Ensures correct output from downsize_grids_around_selected_points."""

        this_full_predictor_matrix = copy.deepcopy(
            FULL_PREDICTOR_MATRIX_TOY_EXAMPLE)

        this_small_predictor_matrix, this_target_vector = (
            ml_utils.downsize_grids_around_selected_points(
                predictor_matrix=this_full_predictor_matrix,
                target_matrix=TARGET_MATRIX_TOY_EXAMPLE,
                num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW_TOY_EXAMPLE,
                num_columns_in_half_window=
                NUM_COLUMNS_IN_HALF_WINDOW_TOY_EXAMPLE,
                target_point_dict=TARGET_POINT_DICT_FOR_DOWNSIZING,
                test_mode=True))

        self.assertTrue(numpy.allclose(
            this_small_predictor_matrix, SMALL_PREDICTOR_MATRIX_SELECTED_POINTS,
            atol=TOLERANCE))
        self.assertTrue(numpy.array_equal(
            this_target_vector, TARGET_VECTOR_SELECTED_POINTS))


if __name__ == '__main__':
    unittest.main()
