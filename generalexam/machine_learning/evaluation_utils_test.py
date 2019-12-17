"""Unit tests for evaluation_utils.py."""

import unittest
import numpy
from generalexam.machine_learning import evaluation_utils

TOLERANCE = 1e-6
GERRITY_TOLERANCE = 1e-4

# The following constants are used to test determinize_predictions.
CLASS_PROBABILITY_MATRIX = numpy.array([
    [0.5, 0.3, 0.2],
    [0.3, 0.6, 0.1],
    [0.7, 0.2, 0.1],
    [0.0, 0.4, 0.6],
    [0.3, 0.4, 0.3],
    [0.8, 0.2, 0.0],
    [0.2, 0.3, 0.5],
    [0.1, 0.2, 0.7],
    [0.0, 0.5, 0.5],
    [0.5, 0.4, 0.1]
])

DETERMINIZATION_THRESHOLD = 0.5
PREDICTED_LABELS = numpy.array([0, 1, 0, 2, 1, 0, 2, 2, 1, 0], dtype=int)

# The following constants are used to test get_contingency_table.
OBSERVED_LABELS = numpy.array([0, 0, 2, 2, 1, 1, 2, 2, 0, 0], dtype=int)
CONTINGENCY_MATRIX = numpy.array([
    [2, 1, 1],
    [2, 1, 0],
    [0, 0, 3]
], dtype=int)

# The following constants are used to test scores.
ACCURACY = 0.6
PEIRCE_SCORE = (0.6 - 0.34) / (1. - 0.36)
HEIDKE_SCORE = (0.6 - 0.34) / (1. - 0.34)

A_VECTOR_FOR_GERRITY_SCORE = numpy.array([1.5, 0.666667, 0.])
S_MATRIX_FOR_GERRITY_SCORE = numpy.array([
    [1.083333, -0.166667, -1.],
    [-0.166667, 0.666667, -0.166667],
    [-1., -0.166667, 1.083333]
])

GERRITY_SCORE = numpy.sum(
    S_MATRIX_FOR_GERRITY_SCORE * CONTINGENCY_MATRIX
) / len(OBSERVED_LABELS)

BINARY_POD = 4. / 6
BINARY_POFD = 2. / 4
BINARY_FAR = 2. / 6
BINARY_FREQ_BIAS = 1.
CRITICAL_SUCCESS_INDEX = 0.5


class EvaluationUtilsTests(unittest.TestCase):
    """Each method is a unit test for evaluation_utils.py."""

    def test_determinize_predictions(self):
        """Ensures correct output from determinize_predictions."""

        these_predicted_labels = evaluation_utils.determinize_predictions(
            class_probability_matrix=CLASS_PROBABILITY_MATRIX,
            threshold=DETERMINIZATION_THRESHOLD)

        self.assertTrue(numpy.array_equal(
            these_predicted_labels, PREDICTED_LABELS
        ))

    def test_get_contingency_table(self):
        """Ensures correct output from get_contingency_table."""

        this_contingency_matrix = evaluation_utils.get_contingency_table(
            predicted_labels=PREDICTED_LABELS, observed_labels=OBSERVED_LABELS)

        self.assertTrue(numpy.array_equal(
            this_contingency_matrix, CONTINGENCY_MATRIX
        ))

    def test_get_accuracy(self):
        """Ensures correct output from get_accuracy."""

        this_accuracy = evaluation_utils.get_accuracy(CONTINGENCY_MATRIX)
        self.assertTrue(numpy.isclose(this_accuracy, ACCURACY, atol=TOLERANCE))

    def test_get_peirce_score(self):
        """Ensures correct output from get_peirce_score."""

        this_peirce_score = evaluation_utils.get_peirce_score(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_peirce_score, PEIRCE_SCORE, atol=TOLERANCE
        ))

    def test_get_heidke_score(self):
        """Ensures correct output from get_heidke_score."""

        this_heidke_score = evaluation_utils.get_heidke_score(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_heidke_score, HEIDKE_SCORE, atol=TOLERANCE
        ))

    def test_get_a_for_gerrity_score(self):
        """Ensures correct output from _get_a_for_gerrity_score."""

        this_a_vector = evaluation_utils._get_a_for_gerrity_score(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.allclose(
            this_a_vector, A_VECTOR_FOR_GERRITY_SCORE, atol=GERRITY_TOLERANCE
        ))

    def test_get_s_for_gerrity_score(self):
        """Ensures correct output from _get_s_for_gerrity_score."""

        this_s_matrix = evaluation_utils._get_s_for_gerrity_score(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.allclose(
            this_s_matrix, S_MATRIX_FOR_GERRITY_SCORE, atol=GERRITY_TOLERANCE
        ))

    def test_get_gerrity_score(self):
        """Ensures correct output from get_gerrity_score."""

        this_gerrity_score = evaluation_utils.get_gerrity_score(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_gerrity_score, GERRITY_SCORE, atol=GERRITY_TOLERANCE
        ))

    def test_get_binary_pod(self):
        """Ensures correct output from get_binary_pod."""

        this_pod = evaluation_utils.get_binary_pod(CONTINGENCY_MATRIX)
        self.assertTrue(numpy.isclose(this_pod, BINARY_POD, atol=TOLERANCE))

    def test_get_binary_pofd(self):
        """Ensures correct output from get_binary_pofd."""

        this_pofd = evaluation_utils.get_binary_pofd(CONTINGENCY_MATRIX)
        self.assertTrue(numpy.isclose(this_pofd, BINARY_POFD, atol=TOLERANCE))

    def test_get_binary_far(self):
        """Ensures correct output from get_binary_far."""

        this_far = evaluation_utils.get_binary_far(CONTINGENCY_MATRIX)
        self.assertTrue(numpy.isclose(this_far, BINARY_FAR, atol=TOLERANCE))

    def test_get_binary_frequency_bias(self):
        """Ensures correct output from get_binary_frequency_bias."""

        this_bias = evaluation_utils.get_binary_frequency_bias(
            CONTINGENCY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_bias, BINARY_FREQ_BIAS, atol=TOLERANCE
        ))

    def test_get_binary_csi(self):
        """Ensures correct output from get_binary_csi."""

        this_csi = evaluation_utils.get_csi(CONTINGENCY_MATRIX)
        self.assertTrue(numpy.isclose(
            this_csi, CRITICAL_SUCCESS_INDEX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
