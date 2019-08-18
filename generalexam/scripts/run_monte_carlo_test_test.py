"""Unit tests for run_monte_carlo_test.py."""

import unittest
import numpy
from generalexam.scripts import run_monte_carlo_test as run_mc_test

TOLERANCE = 1e-6

THIS_NUM_LABELS_MATRIX = numpy.array([
    [0, 0, 1, 1, 0, 0],
    [0, 2, 2, 2, 2, 0],
    [0, 2, 2, 2, 2, 0],
    [0, 0, 1, 1, 0, 0]
], dtype=int)

NUM_LABELS_MATRIX = numpy.stack((
    THIS_NUM_LABELS_MATRIX, THIS_NUM_LABELS_MATRIX * 2,
    THIS_NUM_LABELS_MATRIX * 4
), axis=0)

THIS_STATISTIC_MATRIX = numpy.array([
    [-1, -1, 10, 100, -1, -1],
    [-1, 500, 50, 1000, 200, -1],
    [-1, 750, 700, 60, 400, -1],
    [-1, -1, 34, 29, -1, -1]
], dtype=float)

THIS_STATISTIC_MATRIX[THIS_STATISTIC_MATRIX < 0] = numpy.nan

STATISTIC_MATRIX = numpy.stack((
    THIS_STATISTIC_MATRIX, THIS_STATISTIC_MATRIX * 2,
    THIS_STATISTIC_MATRIX * 4
), axis=0)

MEAN_STATISTIC_MATRIX = THIS_STATISTIC_MATRIX * 3


class RunMonteCarloTestTests(unittest.TestCase):
    """Each method is a unit test for run_monte_carlo_test.py."""

    def test_get_weighted_mean_for_statistic(self):
        """Ensures correct output from _get_weighted_mean_for_statistic."""

        this_mean_stat_matrix = run_mc_test._get_weighted_mean_for_statistic(
            num_labels_matrix=NUM_LABELS_MATRIX,
            statistic_matrix=STATISTIC_MATRIX)

        self.assertTrue(numpy.allclose(
            this_mean_stat_matrix, MEAN_STATISTIC_MATRIX, atol=TOLERANCE,
            equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
