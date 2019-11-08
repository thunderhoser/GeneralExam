"""Unit tests for find_extreme_examples.py."""

import unittest
import numpy
from generalexam.scripts import find_extreme_examples as find_extremes

OBSERVED_LABELS = numpy.array([
    0, 1, 2, 1, 2, 1, 1, 0, 2, 2, 2, 2, 1, 2, 1, 0, 1, 0, 2, 2
], dtype=int)

CLASS_PROBABILITY_MATRIX = numpy.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0.1, 0.9],
    [0.3, 0.1, 0.6],
    [0.6, 0.2, 0.2],
    [0.1, 0, 0.9],
    [0, 1, 0],
    [0.2, 0.8, 0],
    [0.1, 0.9, 0],
    [0, 0.8, 0.2],
    [0, 1, 0],
    [0, 0.7, 0.3],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [0.2, 0.8, 0],
    [1, 0, 0],
    [0.5, 0.2, 0.3],
    [1, 0, 0],
    [0, 0.6, 0.4]
])

NUM_EXAMPLES_PER_SET = 5

THIS_NF_INDEX_MATRIX = numpy.array([
    [17, 7, 15, 0, -1],
    [7, 15, 17, 0, -1],
    [0, 17, 7, 15, -1]
], dtype=int)

THIS_WF_INDEX_MATRIX = numpy.array([
    [16, 3, 5, 1, 6],
    [6, 3, 1, 5, 12],
    [1, 12, 14, 5, 3]
], dtype=int)

THIS_CF_INDEX_MATRIX = numpy.array([
    [18, 4, 8, 2, 9],
    [10, 13, 8, 9, 11],
    [2, 19, 11, 4, 9]
], dtype=int)

INDEX_MATRIX_CONDITIONAL = numpy.stack(
    (THIS_NF_INDEX_MATRIX, THIS_WF_INDEX_MATRIX, THIS_CF_INDEX_MATRIX), axis=1
)

INDEX_MATRIX_UNCONDITIONAL = numpy.array([
    [16, 18, 4, 17, 3],
    [13, 10, 6, 8, 9],
    [0, 1, 14, 12, 2]
], dtype=int)


class FindExtremeExamplesTests(unittest.TestCase):
    """Each method is a unit test for find_extreme_examples.py."""

    def test_get_conditional_extremes(self):
        """Ensures correct output from _get_conditional_extremes."""

        this_index_matrix = find_extremes._get_conditional_extremes(
            class_probability_matrix=CLASS_PROBABILITY_MATRIX,
            observed_labels=OBSERVED_LABELS,
            num_examples_per_set=NUM_EXAMPLES_PER_SET)

        self.assertTrue(numpy.array_equal(
            this_index_matrix, INDEX_MATRIX_CONDITIONAL
        ))

    def test_get_unconditional_extremes(self):
        """Ensures correct output from _get_unconditional_extremes."""

        this_index_matrix = find_extremes._get_unconditional_extremes(
            class_probability_matrix=CLASS_PROBABILITY_MATRIX,
            num_examples_per_set=NUM_EXAMPLES_PER_SET)

        self.assertTrue(numpy.array_equal(
            this_index_matrix, INDEX_MATRIX_UNCONDITIONAL
        ))


if __name__ == '__main__':
    unittest.main()
