"""Unit tests for neigh_evaluation.py."""

import unittest
import numpy
from generalexam.machine_learning import neigh_evaluation

TOLERANCE = 1e-6

# The following constants are used to test determinize_predictions_1threshold
# and determinize_predictions_2thresholds.
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
TOY_LABEL_MATRIX_1THRESHOLD = numpy.array([
    [1, 1, 1, 0, 2, 2, 2, 2],
    [1, 1, 1, 0, 2, 2, 2, 0],
    [1, 0, 1, 0, 2, 2, 2, 0],
    [0, 0, 1, 2, 0, 2, 0, 0],
    [1, 0, 0, 1, 2, 2, 2, 2]
], dtype=int)

TOY_LABEL_MATRIX_1THRESHOLD = numpy.expand_dims(
    TOY_LABEL_MATRIX_1THRESHOLD, axis=0)

WF_THRESHOLD = 0.2
CF_THRESHOLD = 0.4

TOY_LABEL_MATRIX_2THRESHOLDS = numpy.array([
    [1, 1, 1, 0, 0, 2, 0, 0],
    [1, 1, 1, 0, 0, 0, 2, 0],
    [1, 0, 1, 0, 0, 2, 2, 0],
    [0, 0, 1, 2, 0, 0, 0, 0],
    [1, 0, 0, 1, 2, 2, 2, 2]
], dtype=int)

TOY_LABEL_MATRIX_2THRESHOLDS = numpy.expand_dims(
    TOY_LABEL_MATRIX_2THRESHOLDS, axis=0)

# The following constants are used to test remove_small_regions_one_time.
LABEL_MATRIX_ALL_REGIONS = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0],
    [0, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 2]
], dtype=int)

GRID_SPACING_METRES = 32.
MIN_REGION_LENGTH_METRES = 120.

BUFFER1_DISTANCE_METRES = GRID_SPACING_METRES
BUFFER2_DISTANCE_METRES = numpy.sqrt(2) * GRID_SPACING_METRES
BUFFER3_DISTANCE_METRES = numpy.sqrt(8) * GRID_SPACING_METRES
BUFFER4_DISTANCE_METRES = numpy.sqrt(18) * GRID_SPACING_METRES
BUFFER5_DISTANCE_METRES = numpy.sqrt(32) * GRID_SPACING_METRES

LABEL_MATRIX_LARGE_REGIONS_BUFFER1 = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

LABEL_MATRIX_LARGE_REGIONS_BUFFER2 = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

LABEL_MATRIX_LARGE_REGIONS_BUFFER3 = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

LABEL_MATRIX_LARGE_REGIONS_BUFFER4 = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2]
], dtype=int)

LABEL_MATRIX_LARGE_REGIONS_BUFFER5 = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2]
], dtype=int)

# The following constants are used to test _match_actual_wf_grid_cells,
# _match_actual_cf_grid_cells, _match_predicted_wf_grid_cells,
# _match_predicted_cf_grid_cells, and make_contingency_tables.
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

BINARY_CT_AS_DICT_SMALL_NEIGH = {
    neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 1,
    neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: 29,
    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 1,
    neigh_evaluation.NUM_FALSE_POSITIVES_KEY: 33
}

PREDICTION_ORIENTED_CT_MATRIX_SMALL_NEIGH = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [11. / 14, 1. / 14, 2. / 14],
    [17. / 20, 3. / 20, 0]
])

ACTUAL_ORIENTED_CT_MATRIX_SMALL_NEIGH = numpy.array([
    [numpy.nan, 11. / 15, 13. / 15],
    [numpy.nan, 1. / 15, 2. / 15],
    [numpy.nan, 3. / 15, 0]
])

BINARY_CT_AS_DICT_LARGE_NEIGH = {
    neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 14,
    neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: 16,
    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 15,
    neigh_evaluation.NUM_FALSE_POSITIVES_KEY: 19
}

PREDICTION_ORIENTED_CT_MATRIX_LARGE_NEIGH = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [0, 11. / 14, 3. / 14],
    [11. / 20, 5. / 20, 4. / 20]
])

ACTUAL_ORIENTED_CT_MATRIX_LARGE_NEIGH = numpy.array([
    [numpy.nan, 3. / 15, 5. / 15],
    [numpy.nan, 8. / 15, 4. / 15],
    [numpy.nan, 4. / 15, 6. / 15]
])

# The following constants are used to test get_binary*.
BINARY_POD_LARGE_NEIGH = 14. / 30
BINARY_FOM_LARGE_NEIGH = 16. / 30
BINARY_SUCCESS_RATIO_LARGE_NEIGH = 15. / 34
BINARY_FAR_LARGE_NEIGH = 19. / 34
BINARY_CSI_LARGE_NEIGH = (30. / 14 + 34. / 15 - 1) ** -1
BINARY_FREQUENCY_BIAS_LARGE_NEIGH = float(14 * 34) / (30 * 15)


class NeighEvaluationTests(unittest.TestCase):
    """Each method is a unit test for neigh_evaluation.py."""

    def test_determinize_predictions_1threshold(self):
        """Ensures correct output from determinize_predictions_1threshold."""

        this_predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_1threshold(
                class_probability_matrix=TOY_PROBABILITY_MATRIX + 0.,
                binarization_threshold=BINARIZATION_THRESHOLD)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, TOY_LABEL_MATRIX_1THRESHOLD
        ))

    def test_determinize_predictions_2thresholds(self):
        """Ensures correct output from determinize_predictions_2thresholds."""

        this_predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_2thresholds(
                class_probability_matrix=TOY_PROBABILITY_MATRIX + 0.,
                wf_threshold=WF_THRESHOLD, cf_threshold=CF_THRESHOLD)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, TOY_LABEL_MATRIX_2THRESHOLDS
        ))

    def test_remove_small_regions_buffer1(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using first buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_region_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=BUFFER1_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, LABEL_MATRIX_LARGE_REGIONS_BUFFER1
        ))

    def test_remove_small_regions_buffer2(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using second buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_region_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=BUFFER2_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, LABEL_MATRIX_LARGE_REGIONS_BUFFER2
        ))

    def test_remove_small_regions_buffer3(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using third buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_region_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=BUFFER3_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, LABEL_MATRIX_LARGE_REGIONS_BUFFER3
        ))

    def test_remove_small_regions_one_time_buffer4(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using fourth buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_region_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=BUFFER4_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, LABEL_MATRIX_LARGE_REGIONS_BUFFER4
        ))

    def test_remove_small_regions_buffer5(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using fifth buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_region_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=BUFFER5_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, LABEL_MATRIX_LARGE_REGIONS_BUFFER5
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

    def test_make_contingency_tables_small_neigh(self):
        """Ensures correct output from make_contingency_tables.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        (this_binary_ct_as_dict, this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_contingency_tables(
            predicted_label_matrix=numpy.expand_dims(
                PREDICTED_LABEL_MATRIX, axis=0) + 0,
            actual_label_matrix=numpy.expand_dims(
                ACTUAL_LABEL_MATRIX, axis=0) + 0,
            neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(this_binary_ct_as_dict == BINARY_CT_AS_DICT_SMALL_NEIGH)

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix,
            PREDICTION_ORIENTED_CT_MATRIX_SMALL_NEIGH,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix,
            ACTUAL_ORIENTED_CT_MATRIX_SMALL_NEIGH,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_make_contingency_tables_large_neigh(self):
        """Ensures correct output from make_contingency_tables.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        (this_binary_ct_as_dict, this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_contingency_tables(
            predicted_label_matrix=numpy.expand_dims(
                PREDICTED_LABEL_MATRIX, axis=0) + 0,
            actual_label_matrix=numpy.expand_dims(
                ACTUAL_LABEL_MATRIX, axis=0) + 0,
            neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(this_binary_ct_as_dict == BINARY_CT_AS_DICT_LARGE_NEIGH)

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix,
            PREDICTION_ORIENTED_CT_MATRIX_LARGE_NEIGH,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix,
            ACTUAL_ORIENTED_CT_MATRIX_LARGE_NEIGH,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_binary_pod_large_neigh(self):
        """Ensures correct output from get_binary_pod."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_pod(BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_POD_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_binary_fom_large_neigh(self):
        """Ensures correct output from get_binary_fom."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_fom(BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_FOM_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_binary_success_ratio_large_neigh(self):
        """Ensures correct output from get_binary_success_ratio."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_success_ratio(
                BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_SUCCESS_RATIO_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_binary_far_large_neigh(self):
        """Ensures correct output from get_binary_far."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_far(BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_FAR_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_binary_csi_large_neigh(self):
        """Ensures correct output from get_binary_csi."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_csi(BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_CSI_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_binary_frequency_bias_large_neigh(self):
        """Ensures correct output from get_binary_frequency_bias."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_binary_frequency_bias(
                BINARY_CT_AS_DICT_LARGE_NEIGH),
            BINARY_FREQUENCY_BIAS_LARGE_NEIGH, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
