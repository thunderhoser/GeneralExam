"""Unit tests for prediction_io.py."""

import unittest
import numpy
from generalexam.ge_io import prediction_io

TOLERANCE = 1e-6

# The following constants are used to test _fill_probabilities.
THIS_CLASS0_MATRIX = numpy.array([
    [-1, -1, -1, 0.1, 0.7, -1, -1, -1],
    [-1, 0.5, 0.3, 0.0, 0.5, 0.4, 0.7, -1],
    [1.0, 0.1, 0.4, 0.9, 0.1, 0.3, 0.4, 0.8],
    [0.4, 0.2, 0.1, 0.2, 0.4, 0.0, 0.8, 0.9],
    [-1, -1, 0.4, 0.2, 0.6, 0.2, 0.3, 0.9],
    [-1, -1, -1, 0.0, 0.5, 0.2, 0.5, -1]
])

THIS_CLASS1_MATRIX = numpy.array([
    [-1, -1, -1, 0.9, 0.1, -1, -1, -1],
    [-1, 0.3, 0.4, 0.3, 0.5, 0.1, 0.1, -1],
    [0.0, 0.3, 0.1, 0.1, 0.6, 0.2, 0.2, 0.0],
    [0.4, 0.8, 0.7, 0.7, 0.2, 0.7, 0.2, 0.1],
    [-1, -1, 0.2, 0.8, 0.1, 0.3, 0.5, 0.0],
    [-1, -1, -1, 0.2, 0.2, 0.4, 0.2, -1]
])

THIS_CLASS2_MATRIX = numpy.array([
    [-1, -1, -1, 0.0, 0.2, -1, -1, -1],
    [-1, 0.2, 0.3, 0.7, 0.0, 0.5, 0.2, -1],
    [0.0, 0.6, 0.5, 0.0, 0.3, 0.5, 0.4, 0.2],
    [0.2, 0.0, 0.2, 0.1, 0.4, 0.3, 0.0, 0.0],
    [-1, -1, 0.4, 0.0, 0.3, 0.5, 0.2, 0.1],
    [-1, -1, -1, 0.8, 0.3, 0.4, 0.3, -1]
])

CLASS_PROB_MATRIX_WITH_NAN = numpy.stack(
    (THIS_CLASS0_MATRIX, THIS_CLASS1_MATRIX, THIS_CLASS2_MATRIX), axis=-1
)
CLASS_PROB_MATRIX_WITH_NAN[CLASS_PROB_MATRIX_WITH_NAN < -0.1] = numpy.nan

THIS_CLASS0_MATRIX = numpy.array([
    [1.0, 1.0, 1.0, 0.1, 0.7, 1.0, 1.0, 1.0],
    [1.0, 0.5, 0.3, 0.0, 0.5, 0.4, 0.7, 1.0],
    [1.0, 0.1, 0.4, 0.9, 0.1, 0.3, 0.4, 0.8],
    [0.4, 0.2, 0.1, 0.2, 0.4, 0.0, 0.8, 0.9],
    [1.0, 1.0, 0.4, 0.2, 0.6, 0.2, 0.3, 0.9],
    [1.0, 1.0, 1.0, 0.0, 0.5, 0.2, 0.5, 1.0]
])

THIS_CLASS1_MATRIX = numpy.array([
    [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.4, 0.3, 0.5, 0.1, 0.1, 0.0],
    [0.0, 0.3, 0.1, 0.1, 0.6, 0.2, 0.2, 0.0],
    [0.4, 0.8, 0.7, 0.7, 0.2, 0.7, 0.2, 0.1],
    [0.0, 0.0, 0.2, 0.8, 0.1, 0.3, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.2, 0.0]
])

THIS_CLASS2_MATRIX = numpy.array([
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.3, 0.7, 0.0, 0.5, 0.2, 0.0],
    [0.0, 0.6, 0.5, 0.0, 0.3, 0.5, 0.4, 0.2],
    [0.2, 0.0, 0.2, 0.1, 0.4, 0.3, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.0, 0.3, 0.5, 0.2, 0.1],
    [0.0, 0.0, 0.0, 0.8, 0.3, 0.4, 0.3, 0.0]
])

CLASS_PROB_MATRIX_WITHOUT_NAN = numpy.stack(
    (THIS_CLASS0_MATRIX, THIS_CLASS1_MATRIX, THIS_CLASS2_MATRIX), axis=-1
)

# The following constants are used to test find_file.
DIRECTORY_NAME = 'foobar'
FIRST_TIME_UNIX_SEC = 1553385600  # 0000 UTC 24 Mar 2019
LAST_TIME_UNIX_SEC = 1553461200  # 2100 UTC 24 Mar 2019

ONE_TIME_FILE_NAME = 'foobar/2019/predictions_2019032400-2019032400.nc'
MANY_TIMES_FILE_NAME = 'foobar/2019/predictions_2019032400-2019032421.nc'


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_fill_probabilities(self):
        """Ensures correct output from _fill_probabilities."""

        this_probability_matrix = prediction_io._fill_probabilities(
            CLASS_PROB_MATRIX_WITH_NAN + 0.)

        self.assertTrue(numpy.allclose(
            this_probability_matrix, CLASS_PROB_MATRIX_WITHOUT_NAN,
            atol=TOLERANCE
        ))

    def test_find_file_one_time(self):
        """Ensures correct output from find_file.

        In this case the file contains one time step.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=FIRST_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == ONE_TIME_FILE_NAME)

    def test_find_file_many_times(self):
        """Ensures correct output from find_file.

        In this case the file contains many time steps.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == MANY_TIMES_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
