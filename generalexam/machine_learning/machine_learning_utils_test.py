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

# The following constants are used to test _subset_grid.
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

    def test_subset_grid_top_left(self):
        """Ensures correct output from _subset_grid.

        In this case, center point for extraction is at top-left of original
        matrix.
        """

        this_matrix = ml_utils._subset_grid(
            full_matrix=FULL_GRID_MATRIX, center_row=CENTER_ROW_TOP_LEFT,
            center_column=CENTER_COLUMN_TOP_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_LEFT, atol=TOLERANCE))

    def test_subset_grid_bottom_left(self):
        """Ensures correct output from _subset_grid.

        In this case, center point for extraction is at bottom-left of original
        matrix.
        """

        this_matrix = ml_utils._subset_grid(
            full_matrix=FULL_GRID_MATRIX, center_row=CENTER_ROW_BOTTOM_LEFT,
            center_column=CENTER_COLUMN_BOTTOM_LEFT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_LEFT, atol=TOLERANCE))

    def test_subset_grid_top_right(self):
        """Ensures correct output from _subset_grid.

        In this case, center point for extraction is at top-right of original
        matrix.
        """

        this_matrix = ml_utils._subset_grid(
            full_matrix=FULL_GRID_MATRIX, center_row=CENTER_ROW_TOP_RIGHT,
            center_column=CENTER_COLUMN_TOP_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_TOP_RIGHT, atol=TOLERANCE))

    def test_subset_grid_bottom_right(self):
        """Ensures correct output from _subset_grid.

        In this case, center point for extraction is at bottom-right of original
        matrix.
        """

        this_matrix = ml_utils._subset_grid(
            full_matrix=FULL_GRID_MATRIX, center_row=CENTER_ROW_BOTTOM_RIGHT,
            center_column=CENTER_COLUMN_BOTTOM_RIGHT,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_BOTTOM_RIGHT, atol=TOLERANCE))

    def test_subset_grid_middle(self):
        """Ensures correct output from _subset_grid.

        In this case, center point for extraction is in middle of original
        matrix.
        """

        this_matrix = ml_utils._subset_grid(
            full_matrix=FULL_GRID_MATRIX, center_row=CENTER_ROW_MIDDLE,
            center_column=CENTER_COLUMN_MIDDLE,
            num_rows_in_half_window=NUM_ROWS_IN_HALF_WINDOW,
            num_columns_in_half_window=NUM_COLUMNS_IN_HALF_WINDOW)

        self.assertTrue(numpy.allclose(
            this_matrix, SMALL_GRID_MATRIX_MIDDLE, atol=TOLERANCE))

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


if __name__ == '__main__':
    unittest.main()
