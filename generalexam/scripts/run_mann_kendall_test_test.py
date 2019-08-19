"""Unit tests for run_mann_kendall_test.py."""

import unittest
import numpy
from generalexam.scripts import run_mann_kendall_test as run_mk_test

TOLERANCE = 1e-6

SERIES_WITHOUT_NANS = numpy.array([
    2.6, 9.9, 5, 0.6, 2.5, 7.5, -4.3, -7, 4.9, -1.3, 3.2, -7.9, -5.2, 6.9, 4.1
])

SERIES_WITH_NANS = numpy.array([
    numpy.nan, 9.9, 5, 0.6, numpy.nan, numpy.nan, -4.3, -7, numpy.nan, -1.3,
    3.2, -7.9, -5.2, numpy.nan, numpy.nan
])

SERIES_AFTER_INTERP = numpy.array([
    14.8, 9.9, 5, 0.6, -1.033333333, -2.666666667, -4.3, -7, -4.15, -1.3,
    3.2, -7.9, -5.2, -2.5, 0.2
])


class RunMannKendallTestTests(unittest.TestCase):
    """Each method is a unit test for run_mann_kendall_test.py."""

    def test_fill_nans_in_series_needed(self):
        """Ensures correct output from _fill_nans_in_series.

        In this case the input series contains NaN's.
        """

        this_series = run_mk_test._fill_nans_in_series(SERIES_WITH_NANS + 0)

        self.assertTrue(numpy.allclose(
            this_series, SERIES_AFTER_INTERP, atol=TOLERANCE
        ))

    def test_fill_nans_in_series_not_needed(self):
        """Ensures correct output from _fill_nans_in_series.

        In this case the input series does *not* contain NaN's.
        """

        this_series = run_mk_test._fill_nans_in_series(SERIES_WITHOUT_NANS + 0)

        self.assertTrue(numpy.allclose(
            this_series, SERIES_WITHOUT_NANS, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
