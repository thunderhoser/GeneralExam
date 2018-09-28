"""Unit tests for training_validation_io.py."""

import copy
import unittest
import numpy
from generalexam.machine_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

THIS_MATRIX_EXAMPLE1_PREDICTOR1 = numpy.array(
    [[1, 2, 3, 4, 5, 6, 7],
     [8, 9, 10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19, 20, 21],
     [22, 23, 24, 25, 26, 27, 28],
     [29, 30, 31, 32, 33, 34, 35]], dtype=float)

THIS_MATRIX_EXAMPLE1 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_PREDICTOR1, THIS_MATRIX_EXAMPLE1_PREDICTOR1 - 10,
     THIS_MATRIX_EXAMPLE1_PREDICTOR1 + 10),
    axis=-1)
LARGE_PREDICTOR_MATRIX = numpy.stack(
    (THIS_MATRIX_EXAMPLE1, THIS_MATRIX_EXAMPLE1 + 100), axis=0)

NUM_HALF_ROWS_TO_KEEP = 1
NUM_HALF_COLUMNS_TO_KEEP = 2

THIS_MATRIX_EXAMPLE1_PREDICTOR1 = numpy.array(
    [[9, 10, 11, 12, 13],
     [16, 17, 18, 19, 20],
     [23, 24, 25, 26, 27]], dtype=float)

THIS_MATRIX_EXAMPLE1 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_PREDICTOR1, THIS_MATRIX_EXAMPLE1_PREDICTOR1 - 10,
     THIS_MATRIX_EXAMPLE1_PREDICTOR1 + 10),
    axis=-1)
SMALL_PREDICTOR_MATRIX = numpy.stack(
    (THIS_MATRIX_EXAMPLE1, THIS_MATRIX_EXAMPLE1 + 100), axis=0)


class TrainingValidationIoTests(unittest.TestCase):
    """Each method is a unit test for training_validation_io.py."""

    def test_decrease_example_size(self):
        """Ensures correct output from _decrease_example_size."""

        this_predictor_matrix = trainval_io._decrease_example_size(
            predictor_matrix=copy.deepcopy(LARGE_PREDICTOR_MATRIX),
            num_half_rows=NUM_HALF_ROWS_TO_KEEP,
            num_half_columns=NUM_HALF_COLUMNS_TO_KEEP)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SMALL_PREDICTOR_MATRIX, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
