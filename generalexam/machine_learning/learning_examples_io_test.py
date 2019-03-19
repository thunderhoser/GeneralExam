"""Unit tests for learning_examples_io.py."""

import unittest
import numpy
from generalexam.machine_learning import learning_examples_io as examples_io

TOLERANCE = 1e-6

# The following constants are used to test _shrink_predictor_grid.
LARGE_2D_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28],
    [29, 30, 31, 32, 33, 34, 35]
], dtype=float)

LARGE_3D_MATRIX = numpy.stack(
    (LARGE_2D_MATRIX, LARGE_2D_MATRIX - 10, LARGE_2D_MATRIX + 10), axis=-1
)

LARGE_PREDICTOR_MATRIX = numpy.stack(
    (LARGE_3D_MATRIX, LARGE_3D_MATRIX + 100), axis=0
)

NUM_HALF_ROWS_SMALL = 1
NUM_HALF_COLUMNS_SMALL = 2

SMALL_2D_MATRIX = numpy.array(
    [[9, 10, 11, 12, 13],
     [16, 17, 18, 19, 20],
     [23, 24, 25, 26, 27]], dtype=float)

SMALL_3D_MATRIX = numpy.stack(
    (SMALL_2D_MATRIX, SMALL_2D_MATRIX - 10, SMALL_2D_MATRIX + 10), axis=-1
)

SMALL_PREDICTOR_MATRIX = numpy.stack(
    (SMALL_3D_MATRIX, SMALL_3D_MATRIX + 100), axis=0)

# The following constants are used to test find_file, _file_name_to_times, and
# _file_name_to_batch_number.
TOP_DIRECTORY_NAME = 'poop'
FIRST_VALID_TIME_UNIX_SEC = -84157200  # 2300 UTC 2 May 1967
LAST_VALID_TIME_UNIX_SEC = -84146400  # 0200 UTC 3 May 1967
NON_SHUFFLED_FILE_NAME = 'poop/downsized_3d_examples_1967050223-1967050302.nc'

BATCH_NUMBER = 1234
SHUFFLED_FILE_NAME = (
    'poop/batches0001000-0001999/downsized_3d_examples_batch0001234.nc')


class LearningExamplesIoTests(unittest.TestCase):
    """Each method is a unit test for learning_examples_io.py."""

    def test_shrink_predictor_grid(self):
        """Ensures correct output from _shrink_predictor_grid."""

        this_predictor_matrix = examples_io._shrink_predictor_grid(
            predictor_matrix=LARGE_PREDICTOR_MATRIX + 0.,
            num_half_rows=NUM_HALF_ROWS_SMALL,
            num_half_columns=NUM_HALF_COLUMNS_SMALL)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SMALL_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_find_file_shuffled(self):
        """Ensures correct output from find_file.

        In this case the file contains temporally shuffled data.
        """

        this_file_name = examples_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=True,
            batch_number=BATCH_NUMBER, raise_error_if_missing=False)

        self.assertTrue(this_file_name == SHUFFLED_FILE_NAME)

    def test_find_file_non_shuffled(self):
        """Ensures correct output from find_file.

        In this case the file does *not* contain temporally shuffled data.
        """

        this_file_name = examples_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=False,
            first_valid_time_unix_sec=FIRST_VALID_TIME_UNIX_SEC,
            last_valid_time_unix_sec=LAST_VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == NON_SHUFFLED_FILE_NAME)

    def test_file_name_to_times_shuffled(self):
        """Ensures correct output from _file_name_to_times.

        In this case the file contains temporally shuffled data.
        """

        with self.assertRaises(ValueError):
            examples_io._file_name_to_times(SHUFFLED_FILE_NAME)

    def test_file_name_to_times_non_shuffled(self):
        """Ensures correct output from _file_name_to_times.

        In this case the file does *not* contain temporally shuffled data.
        """

        this_first_time_unix_sec, this_last_time_unix_sec = (
            examples_io._file_name_to_times(NON_SHUFFLED_FILE_NAME)
        )

        self.assertTrue(this_first_time_unix_sec == FIRST_VALID_TIME_UNIX_SEC)
        self.assertTrue(this_last_time_unix_sec == LAST_VALID_TIME_UNIX_SEC)

    def test_file_name_to_batch_number_shuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the file contains temporally shuffled data.
        """

        this_batch_number = examples_io._file_name_to_batch_number(
            SHUFFLED_FILE_NAME)
        self.assertTrue(this_batch_number == BATCH_NUMBER)

    def test_file_name_to_batch_number_non_shuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the file does *not* contain temporally shuffled data.
        """

        with self.assertRaises(ValueError):
            examples_io._file_name_to_batch_number(NON_SHUFFLED_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
