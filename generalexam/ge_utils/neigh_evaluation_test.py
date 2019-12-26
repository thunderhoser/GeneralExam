"""Unit tests for neigh_evaluation.py."""

import unittest
import numpy
from generalexam.ge_utils import neigh_evaluation

TOLERANCE = 1e-6

# The following constants are used to test dilate_narr_mask and erode_narr_mask.
FIRST_DILATION_DIST_METRES = 35000.
SECOND_DILATION_DIST_METRES = 50000.
THIRD_DILATION_DIST_METRES = 100000.
FOURTH_DILATION_DIST_METRES = 150000.
FIRST_EROSION_DIST_METRES = 35000.
SECOND_EROSION_DIST_METRES = 50000.
THIRD_EROSION_DIST_METRES = 100000.

ORIG_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
], dtype=int)

FIRST_DILATED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
], dtype=int)

SECOND_DILATED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=int)

THIRD_DILATED_MASK_MATRIX = numpy.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

FOURTH_DILATED_MASK_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

FIRST_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
], dtype=int)

SECOND_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
], dtype=int)

THIRD_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

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

NF_PROB_THRESHOLD = 0.75
TOY_LABEL_MATRIX_1THRESHOLD = numpy.array([
    [1, 1, 1, 0, 2, 2, 2, 2],
    [1, 1, 1, 0, 2, 2, 2, 0],
    [1, 0, 1, 0, 2, 2, 2, 0],
    [0, 0, 1, 2, 0, 2, 0, 0],
    [1, 0, 0, 1, 2, 2, 2, 2]
], dtype=int)

TOY_LABEL_MATRIX_1THRESHOLD = numpy.expand_dims(
    TOY_LABEL_MATRIX_1THRESHOLD, axis=0)

WF_PROB_THRESHOLD = 0.2
CF_PROB_THRESHOLD = 0.4

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

FIRST_BUFFER_DIST_METRES = GRID_SPACING_METRES
SECOND_BUFFER_DIST_METRES = numpy.sqrt(2) * GRID_SPACING_METRES
THIRD_BUFFER_DIST_METRES = numpy.sqrt(8) * GRID_SPACING_METRES
FOURTH_BUFFER_DIST_METRES = numpy.sqrt(18) * GRID_SPACING_METRES
FIFTH_BUFFER_DIST_METRES = numpy.sqrt(32) * GRID_SPACING_METRES

FIRST_LARGE_RGN_LABEL_MATRIX = numpy.array([
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

SECOND_LARGE_RGN_LABEL_MATRIX = numpy.array([
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

THIRD_LARGE_RGN_LABEL_MATRIX = numpy.array([
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

FOURTH_LARGE_RGN_LABEL_MATRIX = numpy.array([
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

FIFTH_LARGE_RGN_LABEL_MATRIX = numpy.array([
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

# The following constants are used to test _match_actual_wf_one_time,
# _match_actual_cf_one_time, _match_predicted_wf_one_time,
# _match_predicted_cf_one_time.
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

ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH = numpy.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 2, 1, 5, 5, 5, 5, 5, 0, 0],
    [5, 5, 5, 5, 2, 0, 0, 5, 0, 0, 0, 5],
    [5, 5, 5, 5, 5, 2, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH = numpy.array([
    [1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH = numpy.array([
    [2, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0],
    [2, 5, 5, 5, 1, 0, 0, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
], dtype=int)

PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH = numpy.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 0, 1, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 1, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0],
    [5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0],
    [5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0, 0]
], dtype=int)

ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH = numpy.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 2, 1, 5, 5, 5, 5, 5, 1, 1],
    [5, 5, 5, 5, 2, 1, 1, 5, 1, 1, 1, 5],
    [5, 5, 5, 5, 5, 2, 2, 0, 0, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH = numpy.array([
    [1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 2, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 2, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5],
], dtype=int)

PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH = numpy.array([
    [2, 2, 1, 1, 1, 5, 5, 5, 5, 5, 1, 1],
    [2, 5, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
], dtype=int)

PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH = numpy.array([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 2, 1, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 2, 1, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0],
    [5, 5, 5, 5, 2, 0, 5, 5, 5, 5, 5, 0],
    [5, 5, 5, 2, 0, 5, 5, 5, 5, 5, 0, 0]
], dtype=int)

ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH[ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH == 5] = -1
ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH[ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH == 5] = -1
PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH[PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH == 5] = -1
PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH[PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH == 5] = -1

ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH[ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH == 5] = -1
ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH[ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH == 5] = -1
PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH[PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH == 5] = -1
PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH[PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH == 5] = -1

# The following constants are used to test make_nonspatial_contingency_tables.
TRAINING_MASK_MATRIX = numpy.full(PREDICTED_LABEL_MATRIX.shape, 1, dtype=int)
TRAINING_MASK_MATRIX[0, 0] = 0

BINARY_CT_NONSPATIAL_SMALL = {
    neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 1,
    neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: 28,
    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 1,
    neigh_evaluation.NUM_FALSE_POSITIVES_KEY: 32
}

PREDICTION_ORIENTED_CT_NONSPATIAL_SMALL = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [11, 1, 1],
    [17, 3, 0]
])

ACTUAL_ORIENTED_CT_NONSPATIAL_SMALL = numpy.array([
    [numpy.nan, 11, 13],
    [numpy.nan, 1, 1],
    [numpy.nan, 3, 0]
])

BINARY_CT_NONSPATIAL_LARGE = {
    neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 14,
    neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: 15,
    neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 15,
    neigh_evaluation.NUM_FALSE_POSITIVES_KEY: 18
}

PREDICTION_ORIENTED_CT_NONSPATIAL_LARGE = numpy.array([
    [numpy.nan, numpy.nan, numpy.nan],
    [0, 11, 2],
    [11, 5, 4]
])

ACTUAL_ORIENTED_CT_NONSPATIAL_LARGE = numpy.array([
    [numpy.nan, 3, 5],
    [numpy.nan, 8, 3],
    [numpy.nan, 4, 6]
])

# The following constants are used to test evaluation scores.
FAR_WEIGHT = 0.5

POD_LARGE_NEIGH = 14. / 29
FAR_LARGE_NEIGH = 18. / 33
UNWEIGHTED_CSI_LARGE_NEIGH = (29. / 14 + 33. / 15 - 1) ** -1
WEIGHTED_CSI_LARGE_NEIGH = (29. / 14 + 33. / 24 - 1) ** -1
FREQUENCY_BIAS_LARGE_NEIGH = float(14 * 33) / (29 * 15)

# The following constants are used to test make_spatial_contingency_tables.
NUM_GRID_ROWS = ACTUAL_LABEL_MATRIX.shape[0]
NUM_GRID_COLUMNS = ACTUAL_LABEL_MATRIX.shape[1]
THESE_DIM = (NUM_GRID_ROWS, NUM_GRID_COLUMNS, 3, 3)

ACTUAL_ORIENTED_CT_SPATIAL_SMALL = numpy.full(THESE_DIM, 0.)
PRED_ORIENTED_CT_SPATIAL_SMALL = numpy.full(THESE_DIM, 0.)
ACTUAL_ORIENTED_CT_SPATIAL_LARGE = numpy.full(THESE_DIM, 0.)
PRED_ORIENTED_CT_SPATIAL_LARGE = numpy.full(THESE_DIM, 0.)

ACTUAL_ORIENTED_CT_SPATIAL_SMALL[..., 0] = numpy.nan
PRED_ORIENTED_CT_SPATIAL_SMALL[..., 0, :] = numpy.nan
ACTUAL_ORIENTED_CT_SPATIAL_LARGE[..., 0] = numpy.nan
PRED_ORIENTED_CT_SPATIAL_LARGE[..., 0, :] = numpy.nan

for k in range(3):
    ACTUAL_ORIENTED_CT_SPATIAL_SMALL[..., k, 1] += (
        ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH == k
    )
    ACTUAL_ORIENTED_CT_SPATIAL_SMALL[..., k, 2] += (
        ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH == k
    )

    PRED_ORIENTED_CT_SPATIAL_SMALL[..., 1, k] += (
        PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH == k
    )
    PRED_ORIENTED_CT_SPATIAL_SMALL[..., 2, k] += (
        PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH == k
    )

    ACTUAL_ORIENTED_CT_SPATIAL_LARGE[..., k, 1] += (
        ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH == k
    )
    ACTUAL_ORIENTED_CT_SPATIAL_LARGE[..., k, 2] += (
        ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH == k
    )

    PRED_ORIENTED_CT_SPATIAL_LARGE[..., 1, k] += (
        PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH == k
    )
    PRED_ORIENTED_CT_SPATIAL_LARGE[..., 2, k] += (
        PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH == k
    )

ACTUAL_ORIENTED_CT_SPATIAL_SMALL[0, 0, ...] = numpy.nan
PRED_ORIENTED_CT_SPATIAL_SMALL[0, 0, ...] = numpy.nan
ACTUAL_ORIENTED_CT_SPATIAL_LARGE[0, 0, ...] = numpy.nan
PRED_ORIENTED_CT_SPATIAL_LARGE[0, 0, ...] = numpy.nan

BINARY_CT_MATRIX_SMALL = numpy.full(
    (NUM_GRID_ROWS, NUM_GRID_COLUMNS), '', dtype=object
)
BINARY_CT_MATRIX_LARGE = numpy.full(
    (NUM_GRID_ROWS, NUM_GRID_COLUMNS), '', dtype=object
)

for r in range(NUM_GRID_ROWS):
    for c in range(NUM_GRID_COLUMNS):
        if r == c == 0:
            this_dict = {
                neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: numpy.nan,
                neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: numpy.nan,
                neigh_evaluation.NUM_FALSE_POSITIVES_KEY: numpy.nan,
                neigh_evaluation.NUM_FALSE_NEGATIVES_KEY: numpy.nan
            }

            BINARY_CT_MATRIX_SMALL[r, c] = this_dict
            BINARY_CT_MATRIX_LARGE[r, c] = this_dict
            continue

        BINARY_CT_MATRIX_SMALL[r, c] = dict()
        BINARY_CT_MATRIX_LARGE[r, c] = dict()

        val = (
            int(PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH[r, c] == 1) +
            int(PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH[r, c] == 2)
        )
        BINARY_CT_MATRIX_SMALL[r, c][
            neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY
        ] = val

        val = (
            int(PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH[r, c] == 1) +
            int(PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH[r, c] == 2)
        )
        BINARY_CT_MATRIX_LARGE[r, c][
            neigh_evaluation.NUM_PREDICTION_ORIENTED_TP_KEY
        ] = val

        val = (
            int(PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH[r, c] in [0, 2]) +
            int(PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH[r, c] in [0, 1])
        )
        BINARY_CT_MATRIX_SMALL[r, c][
            neigh_evaluation.NUM_FALSE_POSITIVES_KEY
        ] = val

        val = (
            int(PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH[r, c] in [0, 2]) +
            int(PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH[r, c] in [0, 1])
        )
        BINARY_CT_MATRIX_LARGE[r, c][
            neigh_evaluation.NUM_FALSE_POSITIVES_KEY
        ] = val

        val = (
            int(ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH[r, c] == 1) +
            int(ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH[r, c] == 2)
        )
        BINARY_CT_MATRIX_SMALL[r, c][
            neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY
        ] = val

        val = (
            int(ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH[r, c] == 1) +
            int(ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH[r, c] == 2)
        )
        BINARY_CT_MATRIX_LARGE[r, c][
            neigh_evaluation.NUM_ACTUAL_ORIENTED_TP_KEY
        ] = val

        val = (
            int(ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH[r, c] in [0, 2]) +
            int(ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH[r, c] in [0, 1])
        )
        BINARY_CT_MATRIX_SMALL[r, c][
            neigh_evaluation.NUM_FALSE_NEGATIVES_KEY
        ] = val

        val = (
            int(ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH[r, c] in [0, 2]) +
            int(ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH[r, c] in [0, 1])
        )
        BINARY_CT_MATRIX_LARGE[r, c][
            neigh_evaluation.NUM_FALSE_NEGATIVES_KEY
        ] = val


def _compare_spatial_binary_tables(first_binary_ct_matrix,
                                   second_binary_ct_matrix):
    """Compares two sets of spatial binary contingency tables.

    :param first_binary_ct_matrix: First set of tables (produced by
        `neigh_evaluation.make_spatial_contingency_tables`).
    :param second_binary_ct_matrix: Second set of tables.
    :return: are_sets_equal: Boolean flag.
    """

    num_rows = first_binary_ct_matrix.shape[0]
    num_columns = first_binary_ct_matrix.shape[1]
    keys = list(first_binary_ct_matrix[0, 0].keys())

    for i in range(num_rows):
        for j in range(num_columns):
            first_array = numpy.array([
                first_binary_ct_matrix[i, j][k] for k in keys
            ])

            second_array = numpy.array([
                second_binary_ct_matrix[i, j][k] for k in keys
            ])

            if not numpy.allclose(
                    first_array, second_array, atol=TOLERANCE, equal_nan=True
            ):
                return False

    return True


class NeighEvaluationTests(unittest.TestCase):
    """Each method is a unit test for neigh_evaluation.py."""

    def test_dilate_narr_mask_first(self):
        """Ensures correct output from dilate_narr_mask.

        In this case, using first dilation distance.
        """

        this_mask_matrix = neigh_evaluation.dilate_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=FIRST_DILATION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, FIRST_DILATED_MASK_MATRIX
        ))

    def test_dilate_narr_mask_second(self):
        """Ensures correct output from dilate_narr_mask.

        In this case, using second dilation distance.
        """

        this_mask_matrix = neigh_evaluation.dilate_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=SECOND_DILATION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, SECOND_DILATED_MASK_MATRIX
        ))

    def test_dilate_narr_mask_third(self):
        """Ensures correct output from dilate_narr_mask.

        In this case, using third dilation distance.
        """

        this_mask_matrix = neigh_evaluation.dilate_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=THIRD_DILATION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, THIRD_DILATED_MASK_MATRIX
        ))

    def test_dilate_narr_mask_fourth(self):
        """Ensures correct output from dilate_narr_mask.

        In this case, using fourth dilation distance.
        """

        this_mask_matrix = neigh_evaluation.dilate_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=FOURTH_DILATION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, FOURTH_DILATED_MASK_MATRIX
        ))

    def test_erode_narr_mask_first(self):
        """Ensures correct output from erode_narr_mask.

        In this case, using first erosion distance.
        """

        this_mask_matrix = neigh_evaluation.erode_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=FIRST_EROSION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, FIRST_ERODED_MASK_MATRIX
        ))

    def test_erode_narr_mask_second(self):
        """Ensures correct output from erode_narr_mask.

        In this case, using second erosion distance.
        """

        this_mask_matrix = neigh_evaluation.erode_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=SECOND_EROSION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, SECOND_ERODED_MASK_MATRIX
        ))

    def test_erode_narr_mask_third(self):
        """Ensures correct output from erode_narr_mask.

        In this case, using third erosion distance.
        """

        this_mask_matrix = neigh_evaluation.erode_narr_mask(
            narr_mask_matrix=ORIG_MASK_MATRIX + 0,
            neigh_distance_metres=THIRD_EROSION_DIST_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, THIRD_ERODED_MASK_MATRIX
        ))

    def test_determinize_predictions_1threshold(self):
        """Ensures correct output from determinize_predictions_1threshold."""

        this_predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_1threshold(
                class_probability_matrix=TOY_PROBABILITY_MATRIX + 0.,
                nf_threshold=NF_PROB_THRESHOLD)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, TOY_LABEL_MATRIX_1THRESHOLD
        ))

    def test_determinize_predictions_2thresholds(self):
        """Ensures correct output from determinize_predictions_2thresholds."""

        this_predicted_label_matrix = (
            neigh_evaluation.determinize_predictions_2thresholds(
                class_probability_matrix=TOY_PROBABILITY_MATRIX + 0.,
                wf_threshold=WF_PROB_THRESHOLD, cf_threshold=CF_PROB_THRESHOLD)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, TOY_LABEL_MATRIX_2THRESHOLDS
        ))

    def test_remove_small_regions_first(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using first buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=FIRST_BUFFER_DIST_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, FIRST_LARGE_RGN_LABEL_MATRIX
        ))

    def test_remove_small_regions_second(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using second buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=SECOND_BUFFER_DIST_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, SECOND_LARGE_RGN_LABEL_MATRIX
        ))

    def test_remove_small_regions_third(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using third buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=THIRD_BUFFER_DIST_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, THIRD_LARGE_RGN_LABEL_MATRIX
        ))

    def test_remove_small_regions_fourth(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using fourth buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=FOURTH_BUFFER_DIST_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, FOURTH_LARGE_RGN_LABEL_MATRIX
        ))

    def test_remove_small_regions_fifth(self):
        """Ensures correct output from remove_small_regions_one_time.

        In this case, using fifth buffer distance.
        """

        this_label_matrix = neigh_evaluation.remove_small_regions_one_time(
            predicted_label_matrix=LABEL_MATRIX_ALL_REGIONS + 0,
            min_length_metres=MIN_REGION_LENGTH_METRES,
            buffer_distance_metres=FIFTH_BUFFER_DIST_METRES,
            grid_spacing_metres=GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, FIFTH_LARGE_RGN_LABEL_MATRIX
        ))

    def test_match_actual_wf_small_neigh(self):
        """Ensures correct output from _match_actual_wf_one_time.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_predicted_label_matrix = (
            neigh_evaluation._match_actual_wf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, ACTUAL_WF_TO_PREDICTED_SMALL_NEIGH
        ))

    def test_match_actual_wf_large_neigh(self):
        """Ensures correct output from _match_actual_wf_one_time.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_predicted_label_matrix = (
            neigh_evaluation._match_actual_wf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, ACTUAL_WF_TO_PREDICTED_LARGE_NEIGH
        ))

    def test_match_actual_cf_small_neigh(self):
        """Ensures correct output from _match_actual_cf_one_time.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_predicted_label_matrix = (
            neigh_evaluation._match_actual_cf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, ACTUAL_CF_TO_PREDICTED_SMALL_NEIGH
        ))

    def test_match_actual_cf_large_neigh(self):
        """Ensures correct output from _match_actual_cf_one_time.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_predicted_label_matrix = (
            neigh_evaluation._match_actual_cf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, ACTUAL_CF_TO_PREDICTED_LARGE_NEIGH
        ))

    def test_match_predicted_wf_small_neigh(self):
        """Ensures correct output from _match_predicted_wf_one_time.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_actual_label_matrix = (
            neigh_evaluation._match_predicted_wf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_actual_label_matrix, PREDICTED_WF_TO_ACTUAL_SMALL_NEIGH
        ))

    def test_match_predicted_wf_large_neigh(self):
        """Ensures correct output from _match_predicted_wf_one_time.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_actual_label_matrix = (
            neigh_evaluation._match_predicted_wf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_actual_label_matrix, PREDICTED_WF_TO_ACTUAL_LARGE_NEIGH
        ))

    def test_match_predicted_cf_small_neigh(self):
        """Ensures correct output from _match_predicted_cf_one_time.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_actual_label_matrix = (
            neigh_evaluation._match_predicted_cf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_actual_label_matrix, PREDICTED_CF_TO_ACTUAL_SMALL_NEIGH
        ))

    def test_match_predicted_cf_large_neigh(self):
        """Ensures correct output from _match_predicted_cf_one_time.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_actual_label_matrix = (
            neigh_evaluation._match_predicted_cf_one_time(
                predicted_label_matrix=PREDICTED_LABEL_MATRIX + 0,
                actual_label_matrix=ACTUAL_LABEL_MATRIX + 0,
                neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
                grid_spacing_metres=GRID_SPACING_METRES)
        )

        self.assertTrue(numpy.array_equal(
            this_actual_label_matrix, PREDICTED_CF_TO_ACTUAL_LARGE_NEIGH
        ))

    def test_make_nonspatial_contingency_tables_small_neigh(self):
        """Ensures correct output from make_nonspatial_contingency_tables.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_predicted_label_matrix = numpy.expand_dims(
            PREDICTED_LABEL_MATRIX, axis=0
        )
        this_actual_label_matrix = numpy.expand_dims(
            ACTUAL_LABEL_MATRIX, axis=0
        )

        (this_binary_ct_dict, this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_nonspatial_contingency_tables(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES, normalize=False,
            training_mask_matrix=TRAINING_MASK_MATRIX + 0
        )

        self.assertTrue(this_binary_ct_dict == BINARY_CT_NONSPATIAL_SMALL)

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix,
            PREDICTION_ORIENTED_CT_NONSPATIAL_SMALL,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix,
            ACTUAL_ORIENTED_CT_NONSPATIAL_SMALL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_make_nonspatial_contingency_tables_large_neigh(self):
        """Ensures correct output from make_nonspatial_contingency_tables.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_predicted_label_matrix = numpy.expand_dims(
            PREDICTED_LABEL_MATRIX, axis=0
        )
        this_actual_label_matrix = numpy.expand_dims(
            ACTUAL_LABEL_MATRIX, axis=0
        )

        (this_binary_ct_dict, this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_nonspatial_contingency_tables(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES, normalize=False,
            training_mask_matrix=TRAINING_MASK_MATRIX + 0
        )

        self.assertTrue(this_binary_ct_dict == BINARY_CT_NONSPATIAL_LARGE)

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix,
            PREDICTION_ORIENTED_CT_NONSPATIAL_LARGE,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix,
            ACTUAL_ORIENTED_CT_NONSPATIAL_LARGE,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_make_spatial_contingency_tables_small_neigh(self):
        """Ensures correct output from make_spatial_contingency_tables.

        In this case the neighbourhood distance is small (forcing the
        neighbourhood to be only 1 grid cell).
        """

        this_predicted_label_matrix = numpy.expand_dims(
            PREDICTED_LABEL_MATRIX, axis=0
        )
        this_actual_label_matrix = numpy.expand_dims(
            ACTUAL_LABEL_MATRIX, axis=0
        )

        (this_binary_ct_matrix,
         this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_spatial_contingency_tables(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=SMALL_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES,
            training_mask_matrix=TRAINING_MASK_MATRIX + 0
        )

        _compare_spatial_binary_tables(
            this_binary_ct_matrix, BINARY_CT_MATRIX_SMALL
        )

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix, PRED_ORIENTED_CT_SPATIAL_SMALL,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix, ACTUAL_ORIENTED_CT_SPATIAL_SMALL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_make_spatial_contingency_tables_large_neigh(self):
        """Ensures correct output from make_spatial_contingency_tables.

        In this case the neighbourhood distance is large (allowing the
        neighbourhood to be 9 grid cells).
        """

        this_predicted_label_matrix = numpy.expand_dims(
            PREDICTED_LABEL_MATRIX, axis=0
        )
        this_actual_label_matrix = numpy.expand_dims(
            ACTUAL_LABEL_MATRIX, axis=0
        )

        (this_binary_ct_matrix,
         this_prediction_oriented_ct_matrix,
         this_actual_oriented_ct_matrix
        ) = neigh_evaluation.make_spatial_contingency_tables(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=LARGE_NEIGH_DISTANCE_METRES,
            grid_spacing_metres=GRID_SPACING_METRES,
            training_mask_matrix=TRAINING_MASK_MATRIX + 0
        )

        _compare_spatial_binary_tables(
            this_binary_ct_matrix, BINARY_CT_MATRIX_LARGE
        )

        self.assertTrue(numpy.allclose(
            this_prediction_oriented_ct_matrix, PRED_ORIENTED_CT_SPATIAL_LARGE,
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_actual_oriented_ct_matrix, ACTUAL_ORIENTED_CT_SPATIAL_LARGE,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_pod(self):
        """Ensures correct output from get_pod."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_pod(BINARY_CT_NONSPATIAL_LARGE),
            POD_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_far(self):
        """Ensures correct output from get_far."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_far(BINARY_CT_NONSPATIAL_LARGE),
            FAR_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_unweighted_csi(self):
        """Ensures correct output from get_csi (unweighted in this case)."""

        this_csi = neigh_evaluation.get_csi(
            binary_ct_as_dict=BINARY_CT_NONSPATIAL_LARGE, far_weight=1.)

        self.assertTrue(numpy.isclose(
            this_csi, UNWEIGHTED_CSI_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_weighted_csi(self):
        """Ensures correct output from get_csi (weighted in this case)."""

        this_csi = neigh_evaluation.get_csi(
            binary_ct_as_dict=BINARY_CT_NONSPATIAL_LARGE, far_weight=FAR_WEIGHT
        )

        self.assertTrue(numpy.isclose(
            this_csi, WEIGHTED_CSI_LARGE_NEIGH, atol=TOLERANCE
        ))

    def test_get_frequency_bias(self):
        """Ensures correct output from get_frequency_bias."""

        self.assertTrue(numpy.isclose(
            neigh_evaluation.get_frequency_bias(BINARY_CT_NONSPATIAL_LARGE),
            FREQUENCY_BIAS_LARGE_NEIGH, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
