"""Unit tests for permutation.py."""

import unittest
import numpy
from sklearn.metrics import roc_auc_score as sklearn_auc
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import permutation

TOLERANCE = 1e-6

# The following constants are used to test negative_auc_function.
OBSERVED_LABELS_BINARY = numpy.array(
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=int
)
CLASS_PROB_MATRIX_BINARY = numpy.array([
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5]
])

NEGATIVE_AUC_BINARY = -0.75

OBSERVED_LABELS_TERNARY = numpy.array(
    [0, 0, 1, 0, 2, 2, 1, 1, 2, 0, 1, 0, 2, 2, 0], dtype=int
)
CLASS_PROB_MATRIX_TERNARY = numpy.array([
    [0.1, 0.5, 0.4],
    [0.1, 0.0, 0.9],
    [0.0, 0.8, 0.2],
    [0.3, 0.0, 0.7],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.1, 0.9, 0.0],
    [0.4, 0.5, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1],
    [0.9, 0.1, 0.0],
    [0.0, 0.4, 0.6],
    [0.4, 0.3, 0.2],
    [0.1, 0.8, 0.0]
])

THIS_FIRST_AUC = sklearn_auc(
    y_true=(OBSERVED_LABELS_TERNARY == 1).astype(int),
    y_score=CLASS_PROB_MATRIX_TERNARY[:, 1]
)

THIS_SECOND_AUC = sklearn_auc(
    y_true=(OBSERVED_LABELS_TERNARY == 2).astype(int),
    y_score=CLASS_PROB_MATRIX_TERNARY[:, 2]
)

NEGATIVE_AUC_TERNARY = -0.5 * (THIS_FIRST_AUC + THIS_SECOND_AUC)

# The following constants are used to test get_nice_predictor_names.
SURFACE_PRESSURE_MB = predictor_utils.DUMMY_SURFACE_PRESSURE_MB

PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME, predictor_utils.PRESSURE_NAME,
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.HEIGHT_NAME
]

PRESSURE_LEVELS_MB = numpy.array([
    SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB,
    SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB,
    950, 950, 900
], dtype=int)

NICE_PREDICTOR_NAMES = [
    'Surface temperature', 'Surface specific humidity',
    r'Surface $u$-wind', r'Surface $v$-wind', 'Surface pressure',
    '950-mb temperature', '950-mb specific humidity',
    '900-mb geopotential height'
]


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_negative_auc_function_binary(self):
        """Ensures correct output from negative_auc_function.

        In this case the problem is binary (2-class).
        """

        this_negative_auc = permutation.negative_auc_function(
            observed_labels=OBSERVED_LABELS_BINARY,
            class_probability_matrix=CLASS_PROB_MATRIX_BINARY)

        self.assertTrue(numpy.isclose(
            this_negative_auc, NEGATIVE_AUC_BINARY, atol=TOLERANCE
        ))

    def test_negative_auc_function_ternary(self):
        """Ensures correct output from negative_auc_function.

        In this case the problem is ternary (3-class).
        """

        this_negative_auc = permutation.negative_auc_function(
            observed_labels=OBSERVED_LABELS_TERNARY,
            class_probability_matrix=CLASS_PROB_MATRIX_TERNARY)

        self.assertTrue(numpy.isclose(
            this_negative_auc, NEGATIVE_AUC_TERNARY, atol=TOLERANCE
        ))

    def test_get_nice_predictor_names(self):
        """Ensures correct output from get_nice_predictor_names."""

        these_nice_predictor_names = permutation.get_nice_predictor_names(
            predictor_names=PREDICTOR_NAMES,
            pressure_levels_mb=PRESSURE_LEVELS_MB)

        self.assertTrue(these_nice_predictor_names == NICE_PREDICTOR_NAMES)


if __name__ == '__main__':
    unittest.main()
