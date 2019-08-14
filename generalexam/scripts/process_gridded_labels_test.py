"""Unit tests for process_gridded_labels.py."""

import unittest
import numpy
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.scripts import process_gridded_labels as process_labels

DUMMY_INPUT_DIR_NAME = 'foo'
DUMMY_OUTPUT_DIR_NAME = 'bar'

SEPARATION_TIME_SEC = 25000
FIRST_TIMES_UNIX_SEC = numpy.array([0, 10800, 21600], dtype=int)
FIRST_PREDICTION_FILE_NAMES = [
    prediction_io.find_file(
        directory_name=DUMMY_INPUT_DIR_NAME, first_time_unix_sec=t,
        last_time_unix_sec=t, raise_error_if_missing=False
    )
    for t in FIRST_TIMES_UNIX_SEC
]

FIRST_LABEL_MATRIX_TIME1 = numpy.array([
    [0, 0, 1, 1, 1, 1],
    [2, 2, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0]
], dtype=int)

FIRST_LABEL_MATRIX_TIME2 = numpy.array([
    [0, 1, 0, 0, 1, 1],
    [0, 2, 2, 1, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 2, 2, 0, 0, 0]
], dtype=int)

FIRST_LABEL_MATRIX_TIME3 = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 1, 1],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0]
], dtype=int)

FIRST_LABEL_MATRIX = numpy.stack(
    (FIRST_LABEL_MATRIX_TIME1, FIRST_LABEL_MATRIX_TIME2,
     FIRST_LABEL_MATRIX_TIME3),
    axis=0
)

FIRST_UNIQUE_LABEL_MATRIX = FIRST_LABEL_MATRIX + 0

for i in range(FIRST_LABEL_MATRIX.shape[1]):
    for j in range(FIRST_LABEL_MATRIX.shape[2]):
        FIRST_UNIQUE_LABEL_MATRIX[:, i, j] = (
            climo_utils.apply_separation_time(
                front_type_enums=FIRST_LABEL_MATRIX[:, i, j],
                valid_times_unix_sec=FIRST_TIMES_UNIX_SEC,
                separation_time_sec=SEPARATION_TIME_SEC)
        )[0]

SECOND_TIMES_UNIX_SEC = numpy.array([32400, 64800, 75600], dtype=int)
SECOND_PREDICTION_FILE_NAMES = [
    prediction_io.find_file(
        directory_name=DUMMY_INPUT_DIR_NAME, first_time_unix_sec=t,
        last_time_unix_sec=t, raise_error_if_missing=False
    )
    for t in SECOND_TIMES_UNIX_SEC
]

THIS_MATRIX_TIME1 = numpy.array([
    [0, 0, 0, 0, 0, 1],
    [0, 0, 2, 1, 1, 1],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 2, 0, 0, 0]
], dtype=int)

THIS_MATRIX_TIME2 = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1],
    [0, 0, 2, 2, 0, 1]
], dtype=int)

THIS_MATRIX_TIME3 = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 1, 1]
], dtype=int)

SECOND_LABEL_MATRIX = numpy.stack(
    (THIS_MATRIX_TIME1, THIS_MATRIX_TIME2, THIS_MATRIX_TIME3), axis=0
)

THIS_MATRIX_TIME1 = numpy.array([
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=int)

THIS_MATRIX_TIME2 = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1],
    [0, 0, 2, 2, 0, 1]
], dtype=int)

THIS_MATRIX_TIME3 = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
], dtype=int)

SECOND_UNIQUE_LABEL_MATRIX = numpy.stack(
    (THIS_MATRIX_TIME1, THIS_MATRIX_TIME2, THIS_MATRIX_TIME3), axis=0
)


class ProcessGriddedLabelsTests(unittest.TestCase):
    """Each method is a unit test for process_gridded_labels.py."""

    def test_write_new_labels_1period(self):
        """Ensures correct output from _write_new_labels.

        In this case, `write_second_period == False`.
        """

        this_unique_label_matrix = process_labels._write_new_labels(
            first_label_matrix=FIRST_LABEL_MATRIX,
            first_unique_label_matrix=FIRST_UNIQUE_LABEL_MATRIX,
            first_prediction_file_names=FIRST_PREDICTION_FILE_NAMES,
            second_label_matrix=None,
            second_prediction_file_names=None,
            write_second_period=False,
            separation_time_sec=SEPARATION_TIME_SEC,
            output_dir_name=DUMMY_OUTPUT_DIR_NAME, test_mode=True)

        self.assertTrue(this_unique_label_matrix is None)

    def test_write_new_labels_2periods(self):
        """Ensures correct output from _write_new_labels.

        In this case, `write_second_period == True`.
        """

        this_unique_label_matrix = process_labels._write_new_labels(
            first_label_matrix=FIRST_LABEL_MATRIX,
            first_unique_label_matrix=FIRST_UNIQUE_LABEL_MATRIX,
            first_prediction_file_names=FIRST_PREDICTION_FILE_NAMES,
            second_label_matrix=SECOND_LABEL_MATRIX,
            second_prediction_file_names=SECOND_PREDICTION_FILE_NAMES,
            write_second_period=False,
            separation_time_sec=SEPARATION_TIME_SEC,
            output_dir_name=DUMMY_OUTPUT_DIR_NAME, test_mode=True)

        self.assertTrue(numpy.array_equal(
            this_unique_label_matrix, SECOND_UNIQUE_LABEL_MATRIX
        ))


if __name__ == '__main__':
    unittest.main()
