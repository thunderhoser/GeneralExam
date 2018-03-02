"""Unit tests for utils.py."""

import unittest
import numpy
from generalexam.ge_utils import utils

TOLERANCE = 1e-6

MATRIX_WITH_NANS_1D = numpy.array([1, 2, 3, numpy.nan])
MATRIX_WITHOUT_NANS_1D = numpy.array([1, 2, 3, 3], dtype=float)

MATRIX_WITH_NANS_2D = numpy.array([[1, 2, 3, 4, 5],
                                   [6, 7, numpy.nan, numpy.nan, 10],
                                   [numpy.nan, 12, numpy.nan, numpy.nan, 15]])
MATRIX_WITHOUT_NANS_2D = numpy.array([[1, 2, 3, 4, 5],
                                      [6, 7, 7, 4, 10],
                                      [6, 12, 12, 15, 15]])

MATRIX_WITH_NANS_3D = numpy.stack(
    (MATRIX_WITH_NANS_2D, MATRIX_WITH_NANS_2D), axis=0)
MATRIX_WITHOUT_NANS_3D = numpy.stack(
    (MATRIX_WITHOUT_NANS_2D, MATRIX_WITHOUT_NANS_2D), axis=0)


class UtilsTests(unittest.TestCase):
    """Each method is a unit test for utils.py."""

    def test_fill_nans_1d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = utils.fill_nans(MATRIX_WITH_NANS_1D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_1D, atol=TOLERANCE))

    def test_fill_nans_2d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = utils.fill_nans(MATRIX_WITH_NANS_2D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_2D, atol=TOLERANCE))

    def test_fill_nans_3d(self):
        """Ensures correct output from fill_nans."""

        this_matrix_without_nans = utils.fill_nans(MATRIX_WITH_NANS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_3D, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
