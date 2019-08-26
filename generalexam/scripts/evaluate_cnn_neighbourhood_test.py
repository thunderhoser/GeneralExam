"""Unit tests for evaluate_cnn_neighbourhood.py."""

import unittest
import numpy
from generalexam.machine_learning import neigh_evaluation
from generalexam.scripts import evaluate_cnn_neighbourhood as eval_cnn_neigh

TOLERANCE = 1e-6

# The following constants are used to test `_decompose_contingency_tables` and
# `_bootstrap_contingency_tables`.
PREDICTION_ORIENTED_CT_MATRIX_DENORM = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [1, 4, 2],
    [5, 5, 10]
])

ACTUAL_ORIENTED_CT_MATRIX_DENORM = numpy.array([
    [numpy.nan, 2, 5],
    [numpy.nan, 5, 0],
    [numpy.nan, 10, 15]
])

PREDICTION_ORIENTED_CT_MATRIX_NORM = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [1. / 7, 4. / 7, 2. / 7],
    [0.25, 0.25, 0.5]
])

ACTUAL_ORIENTED_CT_MATRIX_NORM = numpy.array([
    [numpy.nan, 2. / 17, 0.25],
    [numpy.nan, 5. / 17, 0],
    [numpy.nan, 10. / 17, 0.75]
])

PREDICTED_FRONT_ENUMS = numpy.concatenate((
    numpy.full(7, 1, dtype=int), numpy.full(20, 2, dtype=int)
))
PREDICTED_TO_ACTUAL_FRONT_ENUMS = numpy.concatenate((
    numpy.full(1, 0, dtype=int), numpy.full(4, 1, dtype=int),
    numpy.full(2, 2, dtype=int),
    numpy.full(5, 0, dtype=int), numpy.full(5, 1, dtype=int),
    numpy.full(10, 2, dtype=int)
))

ACTUAL_FRONT_ENUMS = numpy.concatenate((
    numpy.full(2, 1, dtype=int), numpy.full(5, 2, dtype=int),
    numpy.full(5, 1, dtype=int),
    numpy.full(10, 1, dtype=int), numpy.full(15, 2, dtype=int)
))
ACTUAL_TO_PREDICTED_FRONT_ENUMS = numpy.concatenate((
    numpy.full(7, 0, dtype=int), numpy.full(5, 1, dtype=int),
    numpy.full(25, 2, dtype=int)
))

MATCH_DICT = {
    eval_cnn_neigh.PREDICTED_LABELS_KEY: PREDICTED_FRONT_ENUMS,
    eval_cnn_neigh.PREDICTED_TO_ACTUAL_FRONTS_KEY:
        PREDICTED_TO_ACTUAL_FRONT_ENUMS,
    eval_cnn_neigh.ACTUAL_LABELS_KEY: ACTUAL_FRONT_ENUMS,
    eval_cnn_neigh.ACTUAL_TO_PREDICTED_FRONTS_KEY:
        ACTUAL_TO_PREDICTED_FRONT_ENUMS
}

# The following constants are used to test `_bootstrap_contingency_tables`.
BINARY_CT_AS_DICT = {
    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 14,
    neigh_evaluation.NUM_FALSE_POSITIVES_KEY: 13,
    neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 20,
    neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: 17
}


class EvaluateCnnNeighbourhoodTests(unittest.TestCase):
    """Each method is a unit test for evaluate_cnn_neighbourhood.py."""

    def test_decompose_contingency_tables(self):
        """Ensures correct output from _decompose_contingency_tables."""

        this_match_dict = eval_cnn_neigh._decompose_contingency_tables(
            prediction_oriented_ct_matrix=PREDICTION_ORIENTED_CT_MATRIX_DENORM,
            actual_oriented_ct_matrix=ACTUAL_ORIENTED_CT_MATRIX_DENORM)

        actual_keys = list(this_match_dict.keys())
        expected_keys = list(MATCH_DICT.keys())
        self.assertTrue(set(actual_keys) == set(expected_keys))

        for this_key in actual_keys:
            self.assertTrue(numpy.array_equal(
                this_match_dict[this_key], MATCH_DICT[this_key]
            ))

    def test_bootstrap_contingency_tables(self):
        """Ensures correct output from _bootstrap_contingency_tables."""

        (this_binary_ct_as_dict, this_prediction_oriented_matrix,
         this_actual_oriented_matrix
        ) = eval_cnn_neigh._bootstrap_contingency_tables(
            match_dict=MATCH_DICT, test_mode=True)

        self.assertTrue(this_binary_ct_as_dict == BINARY_CT_AS_DICT)
        self.assertTrue(numpy.allclose(
            this_prediction_oriented_matrix, PREDICTION_ORIENTED_CT_MATRIX_NORM,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_actual_oriented_matrix, ACTUAL_ORIENTED_CT_MATRIX_NORM,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
