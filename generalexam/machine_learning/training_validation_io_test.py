"""Unit tests for training_validation_io.py."""

import copy
import unittest
import numpy
from generalexam.machine_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _decrease_example_size.
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

# The following constants are used to test find_downsized_3d_example_file.
DIRECTORY_NAME = 'poop'
FIRST_TARGET_TIME_UNIX_SEC = -84157200  # 2300 UTC 2 May 1967
LAST_TARGET_TIME_UNIX_SEC = -84146400  # 0200 UTC 3 May 1967
DOWNSIZED_3D_FILE_NAME = 'poop/downsized_3d_examples_1967050223-1967050302.nc'


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

    def test_find_downsized_3d_example_file(self):
        """Ensures correct output from find_downsized_3d_example_file."""

        this_file_name = trainval_io.find_downsized_3d_example_file(
            directory_name=DIRECTORY_NAME,
            first_target_time_unix_sec=FIRST_TARGET_TIME_UNIX_SEC,
            last_target_time_unix_sec=LAST_TARGET_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == DOWNSIZED_3D_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
