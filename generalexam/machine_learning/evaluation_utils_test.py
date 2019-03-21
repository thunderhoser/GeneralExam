"""Unit tests for evaluation_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.machine_learning import evaluation_utils

TOLERANCE = 1e-6
TOLERANCE_FOR_GERRITY_SCORE = 1e-4

# The following constants are used to test _get_random_sample_points.
NUM_POINTS_TO_SAMPLE = 10000
NUM_ROWS_FOR_FCN_INPUT = (
    ml_utils.LAST_NARR_ROW_FOR_FCN_INPUT -
    ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT + 1
)
NUM_COLUMNS_FOR_FCN_INPUT = (
    ml_utils.LAST_NARR_COLUMN_FOR_FCN_INPUT -
    ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT + 1
)

NARR_MASK_MATRIX = numpy.full(
    nwp_model_utils.get_grid_dimensions(
        nwp_model_utils.NARR_MODEL_NAME), 0, dtype=int)

FIRST_UNMASKED_ROW = 20
LAST_UNMASKED_ROW = 200
FIRST_UNMASKED_COLUMN = 10
LAST_UNMASKED_COLUMN = 100

NARR_MASK_MATRIX[
    FIRST_UNMASKED_ROW:(LAST_UNMASKED_ROW + 1),
    FIRST_UNMASKED_COLUMN:(LAST_UNMASKED_COLUMN + 1)
] = 1

# The following constants are used to test determinize_probabilities.
CLASS_PROBABILITY_MATRIX = numpy.array([[0.5, 0.3, 0.2],
                                        [0.3, 0.6, 0.1],
                                        [0.7, 0.2, 0.1],
                                        [0.0, 0.4, 0.6],
                                        [0.3, 0.4, 0.3],
                                        [0.8, 0.2, 0.0],
                                        [0.2, 0.3, 0.5],
                                        [0.1, 0.2, 0.7],
                                        [0.0, 0.5, 0.5],
                                        [0.5, 0.4, 0.1]])

BINARIZATION_THRESHOLD = 0.5
PREDICTED_LABELS = numpy.array([0, 1, 0, 2, 1, 0, 2, 2, 1, 0], dtype=int)

# The following constants are used to test get_contingency_table.
OBSERVED_LABELS = numpy.array([0, 0, 2, 2, 1, 1, 2, 2, 0, 0], dtype=int)
NUM_CLASSES = 3
CONTINGENCY_TABLE_AS_MATRIX = numpy.array([[2, 1, 1],
                                           [2, 1, 0],
                                           [0, 0, 3]], dtype=int)

# The following constants are used to test performance metrics.
ACCURACY = 0.6
PEIRCE_SCORE = (0.6 - 0.34) / (1. - 0.36)
HEIDKE_SCORE = (0.6 - 0.34) / (1. - 0.34)

A_VECTOR_FOR_GERRITY_SCORE = numpy.array([1.5, 0.666667, 0.])
S_MATRIX_FOR_GERRITY_SCORE = numpy.array([[1.083333, -0.166667, -1.],
                                          [-0.166667, 0.666667, -0.166667],
                                          [-1., -0.166667, 1.083333]])

NUM_EVALUATION_PAIRS = len(OBSERVED_LABELS)
GERRITY_SCORE = numpy.sum(
    S_MATRIX_FOR_GERRITY_SCORE * CONTINGENCY_TABLE_AS_MATRIX
) / NUM_EVALUATION_PAIRS

MULTICLASS_CSI = 0.5


class EvaluationUtilsTests(unittest.TestCase):
    """Each method is a unit test for evaluation_utils.py."""

    def test_get_random_sample_points_downsized_no_mask(self):
        """Ensures correct output from _get_random_sample_points.

        In this case,
        `for_downsized_examples = True and narr_mask_matrix is None`.
        """

        these_row_indices, these_column_indices = (
            evaluation_utils._get_random_sample_points(
                num_points=NUM_POINTS_TO_SAMPLE, for_downsized_examples=True,
                narr_mask_matrix=None)
        )

        error_checking.assert_is_integer_numpy_array(these_row_indices)
        error_checking.assert_is_geq_numpy_array(these_row_indices, 0)
        error_checking.assert_is_less_than_numpy_array(
            these_row_indices, NARR_MASK_MATRIX.shape[0])

        error_checking.assert_is_integer_numpy_array(these_column_indices)
        error_checking.assert_is_geq_numpy_array(these_column_indices, 0)
        error_checking.assert_is_less_than_numpy_array(
            these_column_indices, NARR_MASK_MATRIX.shape[1])

    def test_get_random_sample_points_downsized_with_mask(self):
        """Ensures correct output from _get_random_sample_points.

        In this case,
        `for_downsized_examples = True and narr_mask_matrix is not None`.
        """

        these_row_indices, these_column_indices = (
            evaluation_utils._get_random_sample_points(
                num_points=NUM_POINTS_TO_SAMPLE, for_downsized_examples=True,
                narr_mask_matrix=NARR_MASK_MATRIX)
        )

        error_checking.assert_is_integer_numpy_array(these_row_indices)
        error_checking.assert_is_geq_numpy_array(
            these_row_indices, FIRST_UNMASKED_ROW)
        error_checking.assert_is_leq_numpy_array(
            these_row_indices, LAST_UNMASKED_ROW)

        error_checking.assert_is_integer_numpy_array(these_column_indices)
        error_checking.assert_is_geq_numpy_array(
            these_column_indices, FIRST_UNMASKED_COLUMN)
        error_checking.assert_is_leq_numpy_array(
            these_column_indices, LAST_UNMASKED_COLUMN)

    def test_get_random_sample_points_full_size(self):
        """Ensures correct output from _get_random_sample_points.

        In this case, for_downsized_examples = False.
        """

        these_row_indices, these_column_indices = (
            evaluation_utils._get_random_sample_points(
                num_points=NUM_POINTS_TO_SAMPLE, for_downsized_examples=False)
        )

        error_checking.assert_is_integer_numpy_array(these_row_indices)
        error_checking.assert_is_geq_numpy_array(these_row_indices, 0)
        error_checking.assert_is_less_than_numpy_array(
            these_row_indices, NUM_ROWS_FOR_FCN_INPUT)

        error_checking.assert_is_integer_numpy_array(these_column_indices)
        error_checking.assert_is_geq_numpy_array(these_column_indices, 0)
        error_checking.assert_is_less_than_numpy_array(
            these_column_indices, NUM_COLUMNS_FOR_FCN_INPUT)

    def test_determinize_probabilities(self):
        """Ensures correct output from determinize_probabilities."""

        these_predicted_labels = evaluation_utils.determinize_probabilities(
            class_probability_matrix=CLASS_PROBABILITY_MATRIX,
            binarization_threshold=BINARIZATION_THRESHOLD)

        self.assertTrue(numpy.array_equal(
            these_predicted_labels, PREDICTED_LABELS
        ))

    def test_get_contingency_table(self):
        """Ensures correct output from get_contingency_table."""

        this_contingency_table = evaluation_utils.get_contingency_table(
            predicted_labels=PREDICTED_LABELS, observed_labels=OBSERVED_LABELS,
            num_classes=NUM_CLASSES)

        self.assertTrue(numpy.array_equal(
            this_contingency_table, CONTINGENCY_TABLE_AS_MATRIX
        ))

    def test_get_accuracy(self):
        """Ensures correct output from get_accuracy."""

        this_accuracy = evaluation_utils.get_accuracy(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.isclose(this_accuracy, ACCURACY, atol=TOLERANCE))

    def test_get_peirce_score(self):
        """Ensures correct output from get_peirce_score."""

        this_peirce_score = evaluation_utils.get_peirce_score(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.isclose(
            this_peirce_score, PEIRCE_SCORE, atol=TOLERANCE
        ))

    def test_get_heidke_score(self):
        """Ensures correct output from get_heidke_score."""

        this_heidke_score = evaluation_utils.get_heidke_score(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.isclose(
            this_heidke_score, HEIDKE_SCORE, atol=TOLERANCE
        ))

    def test_get_a_for_gerrity_score(self):
        """Ensures correct output from _get_a_for_gerrity_score."""

        this_a_vector = evaluation_utils._get_a_for_gerrity_score(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.allclose(
            this_a_vector, A_VECTOR_FOR_GERRITY_SCORE,
            atol=TOLERANCE_FOR_GERRITY_SCORE
        ))

    def test_get_s_for_gerrity_score(self):
        """Ensures correct output from _get_s_for_gerrity_score."""

        this_s_matrix = evaluation_utils._get_s_for_gerrity_score(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.allclose(
            this_s_matrix, S_MATRIX_FOR_GERRITY_SCORE,
            atol=TOLERANCE_FOR_GERRITY_SCORE
        ))

    def test_get_gerrity_score(self):
        """Ensures correct output from get_gerrity_score."""

        this_gerrity_score = evaluation_utils.get_gerrity_score(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.isclose(
            this_gerrity_score, GERRITY_SCORE,
            atol=TOLERANCE_FOR_GERRITY_SCORE
        ))

    def test_get_multiclass_csi(self):
        """Ensures correct output from get_multiclass_csi."""

        this_multiclass_csi = evaluation_utils.get_multiclass_csi(
            CONTINGENCY_TABLE_AS_MATRIX)

        self.assertTrue(numpy.isclose(
            this_multiclass_csi, MULTICLASS_CSI, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
