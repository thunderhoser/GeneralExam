"""Unit tests for count_gridded_labels.py."""

import unittest
import numpy
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.scripts import count_gridded_labels

NO_FRONT_ENUM = front_utils.NO_FRONT_ENUM
WARM_FRONT_ENUM = front_utils.WARM_FRONT_ENUM
COLD_FRONT_ENUM = front_utils.COLD_FRONT_ENUM

SEPARATION_TIME_SEC = 25000
FIRST_TIMES_UNIX_SEC = numpy.array([0, 10800, 21600], dtype=int)

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

NUM_WF_LABELS_MATRIX_FIRST = numpy.array([
    [0, 1, 1, 1, 2, 2],
    [0, 0, 1, 2, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=int)

NUM_CF_LABELS_MATRIX_FIRST = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0],
    [0, 1, 2, 0, 0, 0],
    [0, 2, 2, 0, 0, 0]
], dtype=int)

NUM_UNIQUE_WF_MATRIX_FIRST = numpy.array([
    [0, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=int)

NUM_UNIQUE_CF_MATRIX_FIRST = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0]
], dtype=int)

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

NUM_WF_LABELS_MATRIX_BOTH = numpy.array([
    [0, 1, 1, 1, 2, 3],
    [0, 0, 1, 3, 2, 2],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 2]
], dtype=int)

NUM_CF_LABELS_MATRIX_BOTH = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 2, 3, 0, 0, 0],
    [0, 1, 3, 1, 0, 0],
    [0, 2, 5, 2, 0, 0]
], dtype=int)

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

NUM_UNIQUE_WF_MATRIX_BOTH = numpy.array([
    [0, 1, 1, 1, 1, 2],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1]
], dtype=int)

NUM_UNIQUE_CF_MATRIX_BOTH = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 2, 1, 0, 0]
], dtype=int)


class CountGriddedLabelsTests(unittest.TestCase):
    """Each method is a unit test for count_gridded_labels.py."""

    def test_update_counts_without_second(self):
        """Ensures correct output from _update_counts.

        In this case, `count_second_period == False`.
        """

        this_count_dict = count_gridded_labels._update_counts(
            first_label_matrix=FIRST_LABEL_MATRIX,
            first_unique_label_matrix=FIRST_UNIQUE_LABEL_MATRIX,
            first_times_unix_sec=FIRST_TIMES_UNIX_SEC,
            second_label_matrix=SECOND_LABEL_MATRIX,
            second_times_unix_sec=SECOND_TIMES_UNIX_SEC,
            separation_time_sec=SEPARATION_TIME_SEC,
            count_second_period=False)

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_WF_LABELS_KEY],
            NUM_WF_LABELS_MATRIX_FIRST
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_CF_LABELS_KEY],
            NUM_CF_LABELS_MATRIX_FIRST
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_UNIQUE_WF_KEY],
            NUM_UNIQUE_WF_MATRIX_FIRST
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_UNIQUE_CF_KEY],
            NUM_UNIQUE_CF_MATRIX_FIRST
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.SECOND_UNIQUE_LABELS_KEY],
            SECOND_UNIQUE_LABEL_MATRIX
        ))

    def test_update_counts_with_second(self):
        """Ensures correct output from _update_counts.

        In this case, `count_second_period == True`.
        """

        this_count_dict = count_gridded_labels._update_counts(
            first_label_matrix=FIRST_LABEL_MATRIX,
            first_unique_label_matrix=FIRST_UNIQUE_LABEL_MATRIX,
            first_times_unix_sec=FIRST_TIMES_UNIX_SEC,
            second_label_matrix=SECOND_LABEL_MATRIX,
            second_times_unix_sec=SECOND_TIMES_UNIX_SEC,
            separation_time_sec=SEPARATION_TIME_SEC,
            count_second_period=True)

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_WF_LABELS_KEY],
            NUM_WF_LABELS_MATRIX_BOTH
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_CF_LABELS_KEY],
            NUM_CF_LABELS_MATRIX_BOTH
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_UNIQUE_WF_KEY],
            NUM_UNIQUE_WF_MATRIX_BOTH
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.NUM_UNIQUE_CF_KEY],
            NUM_UNIQUE_CF_MATRIX_BOTH
        ))

        self.assertTrue(numpy.array_equal(
            this_count_dict[count_gridded_labels.SECOND_UNIQUE_LABELS_KEY],
            SECOND_UNIQUE_LABEL_MATRIX
        ))


if __name__ == '__main__':
    unittest.main()
