"""Unit tests for neigh_evaluation.py."""

import unittest
import numpy
from generalexam.machine_learning import neigh_evaluation

TOLERANCE = 1e-6

# The following constants are used to test determinize_predictions.
THIS_CLASS0_PROB_MATRIX = numpy.array([
    [0.7, 0.1, 0.4, 0.9, 0.6, 0.2, 0.5, 0.6],
    [0.7, 0.6, 0.6, 1.0, 0.7, 0.6, 0.3, 0.8],
    [0.5, 0.9, 0.6, 0.9, 0.6, 0.2, 0.4, 0.9],
    [0.8, 0.8, 0.2, 0.4, 1.0, 0.5, 0.9, 0.9],
    [0.2, 0.9, 0.9, 0.2, 0.2, 0.5, 0.1, 0.1]
])

THIS_CLASS1_PROB_MATRIX = numpy.array([
    [0.2, 0.7, 0.3, 0.1, 0.1, 0.2, 0.2, 0.1],
    [0.3, 0.2, 0.3, 0.0, 0.1, 0.1, 0.3, 0.0],
    [0.4, 0.0, 0.3, 0.0, 0.1, 0.3, 0.2, 0.0],
    [0.1, 0.0, 0.6, 0.2, 0.0, 0.2, 0.1, 0.1],
    [0.5, 0.05, 0.1, 0.5, 0.3, 0.0, 0.4, 0.3]
])

THIS_CLASS2_PROB_MATRIX = numpy.array([
    [0.1, 0.2, 0.3, 0.0, 0.3, 0.6, 0.3, 0.3],
    [0.0, 0.2, 0.1, 0.0, 0.2, 0.3, 0.4, 0.2],
    [0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.4, 0.1],
    [0.1, 0.2, 0.2, 0.4, 0.0, 0.3, 0.0, 0.0],
    [0.3, 0.05, 0.0, 0.3, 0.5, 0.5, 0.5, 0.6]
])

TOY_PROBABILITY_MATRIX = numpy.stack(
    (THIS_CLASS0_PROB_MATRIX, THIS_CLASS1_PROB_MATRIX, THIS_CLASS2_PROB_MATRIX),
    axis=-1
)
TOY_PROBABILITY_MATRIX = numpy.expand_dims(TOY_PROBABILITY_MATRIX, axis=0)

BINARIZATION_THRESHOLD = 0.75
TOY_LABEL_MATRIX = numpy.array([
    [1, 1, 1, 0, 2, 2, 2, 2],
    [1, 1, 1, 0, 2, 2, 2, 0],
    [1, 0, 1, 0, 2, 2, 2, 0],
    [0, 0, 1, 2, 0, 2, 0, 0],
    [1, 0, 0, 1, 2, 2, 2, 2]
], dtype=int)

TOY_LABEL_MATRIX = numpy.expand_dims(TOY_LABEL_MATRIX, axis=0)

# The following constants are used to test _match_actual_wf_grid_cells,
# _match_actual_cf_grid_cells, _match_predicted_wf_grid_cells,
# _match_predicted_cf_grid_cells, and get_binary_contingency_table.
ACTUAL_LABEL_MATRIX = numpy.array([
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

PREDICTED_LABEL_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2]
], dtype=int)

GRID_SPACING_METRES = 32.
SMALL_NEIGH_DISTANCE_METRES = 1.
LARGE_NEIGH_DISTANCE_METRES = 50.

ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH = numpy.array([11, 1, 3], dtype=int)
ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH = numpy.array([13, 2, 0], dtype=int)
PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH = numpy.array([11, 1, 2], dtype=int)
PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH = numpy.array([17, 3, 0], dtype=int)

ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH = numpy.array([3, 8, 4], dtype=int)
ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH = numpy.array([5, 4, 6], dtype=int)
PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH = numpy.array([0, 11, 3], dtype=int)
PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH = numpy.array([11, 5, 4], dtype=int)


class NeighEvaluationTests(unittest.TestCase):
    """Each method is a unit test for neigh_evaluation.py."""

    def test_determinize_predictions(self):
        """Ensures correct output from determinize_predictions."""

        this_predicted_label_matrix = neigh_evaluation.determinize_predictions(
            class_probability_matrix=TOY_PROBABILITY_MATRIX + 0.,
            binarization_threshold=BINARIZATION_THRESHOLD)

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, TOY_LABEL_MATRIX
        ))

    def test_match_actual_wf_grid_cells_small_neigh(self):
        """Ensures correct output from _match_actual_wf_grid_cells.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_actual_wf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH
        ))

    def test_match_actual_wf_grid_cells_large_neigh(self):
        """Ensures correct output from _match_actual_wf_grid_cells.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_actual_wf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH
        ))

    def test_match_actual_cf_grid_cells_small_neigh(self):
        """Ensures correct output from _match_actual_cf_grid_cells.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_actual_cf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH
        ))

    def test_match_actual_cf_grid_cells_large_neigh(self):
        """Ensures correct output from _match_actual_cf_grid_cells.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_actual_cf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH
        ))

    def test_match_predicted_wf_grid_cells_small_neigh(self):
        """Ensures correct output from _match_predicted_wf_grid_cells.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_predicted_wf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH
        ))

    def test_match_predicted_wf_grid_cells_large_neigh(self):
        """Ensures correct output from _match_predicted_wf_grid_cells.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_predicted_wf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH
        ))

    def test_match_predicted_cf_grid_cells_small_neigh(self):
        """Ensures correct output from _match_predicted_cf_grid_cells.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_predicted_cf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH
        ))

    def test_match_predicted_cf_grid_cells_large_neigh(self):
        """Ensures correct output from _match_predicted_cf_grid_cells.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_num_predicted_by_class = (
            neigh_evaluation._match_predicted_cf_grid_cells(
                predicted_label_matrix_one_time=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix_one_time=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_num_predicted_by_class, PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH
        ))


if __name__ == '__main__':
    unittest.main()
