"""Unit tests for object_based_evaluation.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.evaluation import object_based_evaluation as object_based_eval
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import a_star_search
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6
ARRAY_COLUMN_NAMES = [
    object_based_eval.ROW_INDICES_COLUMN,
    object_based_eval.COLUMN_INDICES_COLUMN,
    object_based_eval.X_COORDS_COLUMN, object_based_eval.Y_COORDS_COLUMN
]

# The following constants are used to test _one_region_to_binary_image and
# _one_binary_image_to_region.
BINARY_IMAGE_MATRIX_ONE_REGION = numpy.array(
    [[1, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 1, 1, 0],
     [0, 1, 1, 0, 0, 1, 1, 0],
     [0, 1, 1, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 0]], dtype=int)

ROW_INDICES_ONE_REGION = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int)
COLUMN_INDICES_ONE_REGION = numpy.array(
    [0, 1, 5, 1, 5, 6, 1, 2, 5, 6, 1, 2, 5, 2, 3, 4], dtype=int)

# The following constants are used to test _find_endpoints_of_skeleton.
BINARY_SKELETON_MATRIX = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 1, 0],
                                      [0, 0, 1, 1, 0, 0, 1, 0],
                                      [0, 0, 0, 1, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 1, 1, 0, 0]], dtype=int)
BINARY_ENDPOINT_MATRIX = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)

# The following constants are used to test _get_skeleton_line_endpoint_length,
# _get_skeleton_line_arc_length, and _get_skeleton_line_quality.
THIS_GRID_SEARCH_OBJECT = a_star_search.GridSearch(
    binary_region_matrix=BINARY_IMAGE_MATRIX_ONE_REGION)

(ROW_INDICES_ONE_SKELETON, COLUMN_INDICES_ONE_SKELETON
) = a_star_search.run_a_star(
    grid_search_object=THIS_GRID_SEARCH_OBJECT, start_row=0, start_column=0,
    end_row=1, end_column=6)

(X_GRID_SPACING_METRES, Y_GRID_SPACING_METRES
) = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME)

ENDPOINT_LENGTH_METRES = numpy.sqrt(37.) * X_GRID_SPACING_METRES
MIN_ENDPOINT_LENGTH_SMALL_METRES = ENDPOINT_LENGTH_METRES - 1.
MIN_ENDPOINT_LENGTH_LARGE_METRES = ENDPOINT_LENGTH_METRES + 1.

ARC_LENGTH_METRES = (3 + 5 * numpy.sqrt(2.)) * X_GRID_SPACING_METRES
SKELETON_LINE_QUALITY_POSITIVE = ENDPOINT_LENGTH_METRES ** 2 / ARC_LENGTH_METRES
SKELETON_LINE_QUALITY_NEGATIVE = (
    object_based_eval.NEGATIVE_SKELETON_LINE_QUALITY + 0.
)

# The following constants are used to test _get_distance_between_fronts.
THESE_FIRST_X_METRES = numpy.array([1, 1, 2, 2, 2, 2, 2], dtype=float)
THESE_FIRST_Y_METRES = numpy.array([6, 5, 5, 4, 3, 2, 1], dtype=float)
THESE_SECOND_X_METRES = numpy.array([1, 2, 2, 3, 4, 5, 6, 7, 7, 8], dtype=float)
THESE_SECOND_Y_METRES = numpy.array([5, 5, 4, 4, 4, 4, 4, 4, 5, 5], dtype=float)

THESE_SHORTEST_DISTANCES_METRES = numpy.array(
    [1, 0, 1, 0, 3, 3, 3], dtype=float)
INTERFRONT_DISTANCE_METRES = numpy.median(THESE_SHORTEST_DISTANCES_METRES)

# The following constants are used to test determinize_probabilities.
THIS_MATRIX_CLASS0 = numpy.array(
    [[0.7, 0.1, 0.4, 0.9, 0.6, 0.2, 0.5, 0.6],
     [0.7, 0.6, 0.6, 1.0, 0.7, 0.6, 0.3, 0.8],
     [0.5, 0.9, 0.6, 0.9, 0.6, 0.2, 0.4, 0.9],
     [0.8, 0.8, 0.2, 0.4, 1.0, 0.5, 0.9, 0.9],
     [0.2, 0.9, 0.9, 0.2, 0.2, 0.5, 0.1, 0.1]])

THIS_MATRIX_CLASS1 = numpy.array(
    [[0.2, 0.7, 0.3, 0.1, 0.1, 0.2, 0.2, 0.1],
     [0.3, 0.2, 0.3, 0.0, 0.1, 0.1, 0.3, 0.0],
     [0.4, 0.0, 0.3, 0.0, 0.1, 0.3, 0.2, 0.0],
     [0.1, 0.0, 0.6, 0.2, 0.0, 0.2, 0.1, 0.1],
     [0.5, 0.9, 0.1, 0.5, 0.3, 0.0, 0.4, 0.3]])

THIS_MATRIX_CLASS2 = numpy.array(
    [[0.1, 0.2, 0.3, 0.0, 0.3, 0.6, 0.3, 0.3],
     [0.0, 0.2, 0.1, 0.0, 0.2, 0.3, 0.4, 0.2],
     [0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.4, 0.1],
     [0.1, 0.2, 0.2, 0.4, 0.0, 0.3, 0.0, 0.0],
     [0.3, 0.1, 0.0, 0.3, 0.5, 0.5, 0.5, 0.6]])

PROBABILITY_MATRIX = numpy.stack(
    (THIS_MATRIX_CLASS0, THIS_MATRIX_CLASS1, THIS_MATRIX_CLASS2), axis=-1)
PROBABILITY_MATRIX = numpy.expand_dims(PROBABILITY_MATRIX, axis=0)

BINARIZATION_THRESHOLD = 0.75
PREDICTED_LABEL_MATRIX = numpy.array(
    [[1, 1, 1, 0, 2, 2, 2, 2],
     [1, 1, 1, 0, 2, 2, 2, 0],
     [1, 0, 1, 0, 2, 2, 2, 0],
     [0, 0, 1, 2, 0, 2, 0, 0],
     [1, 0, 0, 1, 2, 2, 2, 2]], dtype=int)
PREDICTED_LABEL_MATRIX = numpy.expand_dims(PREDICTED_LABEL_MATRIX, axis=0)

# The following constants are used to test images_to_regions and
# regions_to_images.
IMAGE_TIMES_UNIX_SEC = numpy.array([0], dtype=int)
REGION_TIMES_UNIX_SEC = numpy.array([0, 0, 0], dtype=int)
FRONT_TYPE_STRINGS = [
    front_utils.WARM_FRONT_STRING_ID, front_utils.COLD_FRONT_STRING_ID,
    front_utils.WARM_FRONT_STRING_ID
]

ROW_INDICES_LARGE_WF_REGION = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 3, 4], dtype=int)
COLUMN_INDICES_LARGE_WF_REGION = numpy.array(
    [0, 1, 2, 0, 1, 2, 0, 2, 2, 3], dtype=int)
ROW_INDICES_SMALL_WF_REGION = numpy.array([4], dtype=int)
COLUMN_INDICES_SMALL_WF_REGION = numpy.array([0], dtype=int)
ROW_INDICES_CF_REGION = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4], dtype=int)
COLUMN_INDICES_CF_REGION = numpy.array(
    [4, 5, 6, 7, 4, 5, 6, 4, 5, 6, 3, 5, 4, 5, 6, 7], dtype=int)

ROW_INDICES_BY_REGION = [
    ROW_INDICES_LARGE_WF_REGION, ROW_INDICES_CF_REGION,
    ROW_INDICES_SMALL_WF_REGION
]
COLUMN_INDICES_BY_REGION = [
    COLUMN_INDICES_LARGE_WF_REGION, COLUMN_INDICES_CF_REGION,
    COLUMN_INDICES_SMALL_WF_REGION
]

THIS_DICT = {
    front_utils.TIME_COLUMN: REGION_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: FRONT_TYPE_STRINGS,
    object_based_eval.ROW_INDICES_COLUMN: ROW_INDICES_BY_REGION,
    object_based_eval.COLUMN_INDICES_COLUMN: COLUMN_INDICES_BY_REGION
}
PREDICTED_REGION_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test discard_regions_with_small_area.
MIN_REGION_AREA_METRES2 = 1.2e10  # 12 000 km^2 (~11 grid cells)
PREDICTED_REGION_TABLE_SANS_SMALL_AREA = PREDICTED_REGION_TABLE.drop(
    PREDICTED_REGION_TABLE.index[[0, 2]], axis=0, inplace=False)

# The following constants are used to test skeletonize_frontal_regions.
# SKELETON_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 2, 2],
#                                [0, 1, 0, 0, 0, 2, 0, 0],
#                                [1, 0, 1, 0, 2, 0, 0, 0],
#                                [0, 0, 1, 2, 0, 2, 0, 0],
#                                [1, 0, 0, 1, 2, 2, 2, 2]], dtype=int)

THESE_ROW_INDICES_LARGE_WF = numpy.array([1, 2, 2, 3, 4], dtype=int)
THESE_COLUMN_INDICES_LARGE_WF = numpy.array([1, 0, 2, 2, 3], dtype=int)
THESE_ROW_INDICES_SMALL_WF = numpy.array([4], dtype=int)
THESE_COLUMN_INDICES_SMALL_WF = numpy.array([0], dtype=int)
THESE_ROW_INDICES_COLD_FRONT = numpy.array(
    [0, 0, 1, 2, 3, 3, 4, 4, 4, 4], dtype=int)
THESE_COLUMN_INDICES_COLD_FRONT = numpy.array(
    [6, 7, 5, 4, 3, 5, 4, 5, 6, 7], dtype=int)

ROW_INDICES_BY_SKELETON = [
    THESE_ROW_INDICES_LARGE_WF, THESE_ROW_INDICES_COLD_FRONT,
    THESE_ROW_INDICES_SMALL_WF
]
COLUMN_INDICES_BY_SKELETON = [
    THESE_COLUMN_INDICES_LARGE_WF, THESE_COLUMN_INDICES_COLD_FRONT,
    THESE_COLUMN_INDICES_SMALL_WF
]

PREDICTED_SKELETON_DICT = {
    front_utils.TIME_COLUMN: REGION_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: FRONT_TYPE_STRINGS,
    object_based_eval.ROW_INDICES_COLUMN: ROW_INDICES_BY_SKELETON,
    object_based_eval.COLUMN_INDICES_COLUMN: COLUMN_INDICES_BY_SKELETON
}
PREDICTED_SKELETON_TABLE = pandas.DataFrame.from_dict(PREDICTED_SKELETON_DICT)

# The following constants are used to test find_main_skeletons.
# MAIN_SKELETON_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0, 2, 2],
#                                     [0, 1, 0, 0, 0, 2, 0, 0],
#                                     [1, 0, 1, 0, 2, 0, 0, 0],
#                                     [0, 0, 1, 0, 0, 2, 0, 0],
#                                     [1, 0, 0, 1, 0, 0, 2, 2]], dtype=int)

THESE_ROW_INDICES_LARGE_WF = numpy.array([2, 1, 2, 3, 4], dtype=int)
THESE_COLUMN_INDICES_LARGE_WF = numpy.array([0, 1, 2, 2, 3], dtype=int)
THESE_ROW_INDICES_COLD_FRONT = numpy.array([0, 0, 1, 2, 3, 4, 4], dtype=int)
THESE_COLUMN_INDICES_COLD_FRONT = numpy.array([7, 6, 5, 4, 5, 6, 7], dtype=int)
THESE_INDICES = numpy.array([0, 1], dtype=int)

PREDICTED_MAIN_SKELETON_DICT = {
    front_utils.TIME_COLUMN: REGION_TIMES_UNIX_SEC[THESE_INDICES],
    front_utils.FRONT_TYPE_COLUMN:
        [FRONT_TYPE_STRINGS[k] for k in THESE_INDICES],
    object_based_eval.ROW_INDICES_COLUMN:
        [THESE_ROW_INDICES_LARGE_WF, THESE_ROW_INDICES_COLD_FRONT],
    object_based_eval.COLUMN_INDICES_COLUMN:
        [THESE_COLUMN_INDICES_LARGE_WF, THESE_COLUMN_INDICES_COLD_FRONT]
}
PREDICTED_MAIN_SKELETON_TABLE = pandas.DataFrame.from_dict(
    PREDICTED_MAIN_SKELETON_DICT)

# The following constants are used to test convert_regions_rowcol_to_narr_xy.
X_COORDS_LARGE_WF_REGION_METRES = (
    COLUMN_INDICES_LARGE_WF_REGION * X_GRID_SPACING_METRES
)
Y_COORDS_LARGE_WF_REGION_METRES = (
    ROW_INDICES_LARGE_WF_REGION * Y_GRID_SPACING_METRES
)
X_COORDS_SMALL_WF_REGION_METRES = (
    COLUMN_INDICES_SMALL_WF_REGION * X_GRID_SPACING_METRES
)
Y_COORDS_SMALL_WF_REGION_METRES = (
    ROW_INDICES_SMALL_WF_REGION * Y_GRID_SPACING_METRES
)
X_COORDS_CF_REGION_METRES = COLUMN_INDICES_CF_REGION * X_GRID_SPACING_METRES
Y_COORDS_CF_REGION_METRES = ROW_INDICES_CF_REGION * Y_GRID_SPACING_METRES

X_COORDS_BY_REGION_METRES = [
    X_COORDS_LARGE_WF_REGION_METRES, X_COORDS_CF_REGION_METRES,
    X_COORDS_SMALL_WF_REGION_METRES
]
Y_COORDS_BY_REGION_METRES = [
    Y_COORDS_LARGE_WF_REGION_METRES, Y_COORDS_CF_REGION_METRES,
    Y_COORDS_SMALL_WF_REGION_METRES
]

THIS_DICT = {
    object_based_eval.X_COORDS_COLUMN: X_COORDS_BY_REGION_METRES,
    object_based_eval.Y_COORDS_COLUMN: Y_COORDS_BY_REGION_METRES
}
PREDICTED_REGION_TABLE_WITH_XY_COORDS = PREDICTED_REGION_TABLE.assign(
    **THIS_DICT)

# The following constants are used to test get_binary_contingency_table.
THESE_X_COORDS_BY_FRONT_METRES = [
    numpy.array([1.]), numpy.array([2.]), numpy.array([3.]), numpy.array([4.]),
    numpy.array([1.]), numpy.array([2.]), numpy.array([3.])
]
THESE_Y_COORDS_BY_FRONT_METRES = [
    numpy.array([1.]), numpy.array([2.]), numpy.array([3.]), numpy.array([4.]),
    numpy.array([1.]), numpy.array([2.]), numpy.array([3.])
]

THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 0, 1, 1, 1], dtype=int)
THESE_FRONT_TYPE_STRINGS = [
    front_utils.COLD_FRONT_STRING_ID, front_utils.WARM_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID, front_utils.WARM_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID, front_utils.WARM_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID
]

THIS_DICT = {
    front_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: THESE_FRONT_TYPE_STRINGS,
    object_based_eval.X_COORDS_COLUMN: THESE_X_COORDS_BY_FRONT_METRES,
    object_based_eval.Y_COORDS_COLUMN: THESE_Y_COORDS_BY_FRONT_METRES
}
ACTUAL_POLYLINE_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_X_COORDS_BY_FRONT_METRES = [
    numpy.array([1.]), numpy.array([2.]), numpy.array([50.]),
    numpy.array([4.]), numpy.array([3.])
]
THESE_Y_COORDS_BY_FRONT_METRES = [
    numpy.array([0.]), numpy.array([2.]), numpy.array([-100.]),
    numpy.array([5.5]), numpy.array([3.])
]

THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 1, 1], dtype=int)
THESE_FRONT_TYPE_STRINGS = [
    front_utils.COLD_FRONT_STRING_ID, front_utils.COLD_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID, front_utils.WARM_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID
]

THIS_DICT = {
    front_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: THESE_FRONT_TYPE_STRINGS,
    object_based_eval.X_COORDS_COLUMN: THESE_X_COORDS_BY_FRONT_METRES,
    object_based_eval.Y_COORDS_COLUMN: THESE_Y_COORDS_BY_FRONT_METRES
}
PREDICTED_REGION_TABLE_FOR_CT = pandas.DataFrame.from_dict(THIS_DICT)

NEIGH_DISTANCE_METRES = 2.
BINARY_CONTINGENCY_TABLE_AS_DICT = {
    object_based_eval.NUM_ACTUAL_FRONTS_PREDICTED_KEY: 5,
    object_based_eval.NUM_FALSE_NEGATIVES_KEY: 2,
    object_based_eval.NUM_PREDICTED_FRONTS_VERIFIED_KEY: 3,
    object_based_eval.NUM_FALSE_POSITIVES_KEY: 2
}

ROW_NORMALIZED_CT_AS_MATRIX = numpy.array([[1., 0., 0.],
                                           [0.25, 0.25, 0.5]])

COLUMN_NORMALIZED_CT_AS_MATRIX = numpy.array([[1. / 3, 0.25],
                                              [0., 0.],
                                              [2. / 3, 0.75]])

# The following constants are used to test performance metrics.
FAKE_BINARY_CT_AS_DICT = {
    object_based_eval.NUM_ACTUAL_FRONTS_PREDICTED_KEY: 100,
    object_based_eval.NUM_PREDICTED_FRONTS_VERIFIED_KEY: 50,
    object_based_eval.NUM_FALSE_POSITIVES_KEY: 200,
    object_based_eval.NUM_FALSE_NEGATIVES_KEY: 10
}

BINARY_POD = 100. / 110
BINARY_FOM = 10. / 110
BINARY_SUCCESS_RATIO = 50. / 250
BINARY_FAR = 200. / 250
BINARY_CSI = (110. / 100 + 250. / 50 - 1.) ** -1
BINARY_FREQUENCY_BIAS = (100. / 110) * (250. / 50)


def _compare_tables(first_table, second_table):
    """Compares two pandas DataFrames.

    :param first_table: pandas DataFrame.
    :param second_table: pandas DataFrame.
    :return: are_tables_equal: Boolean flag.
    """

    these_column_names = list(first_table)
    expected_column_names = list(second_table)
    if set(these_column_names) != set(expected_column_names):
        return False

    this_num_regions = len(first_table.index)
    expected_num_regions = len(second_table.index)
    if this_num_regions != expected_num_regions:
        return False

    for i in range(this_num_regions):
        for this_column_name in these_column_names:
            if this_column_name in ARRAY_COLUMN_NAMES:
                if not numpy.allclose(first_table[this_column_name].values[i],
                                      second_table[this_column_name].values[i],
                                      atol=TOLERANCE):
                    return False

            else:
                if not (first_table[this_column_name].values[i] ==
                        second_table[this_column_name].values[i]):
                    return False

    return True


class ObjectBasedEvaluationTests(unittest.TestCase):
    """Each method is a unit test for object_based_evaluation.py."""

    def test_one_region_to_binary_image(self):
        """Ensures correct output from _one_region_to_binary_image."""

        this_binary_image_matrix = (
            object_based_eval._one_region_to_binary_image(
                row_indices_in_region=ROW_INDICES_ONE_REGION,
                column_indices_in_region=COLUMN_INDICES_ONE_REGION,
                num_grid_rows=BINARY_IMAGE_MATRIX_ONE_REGION.shape[0],
                num_grid_columns=BINARY_IMAGE_MATRIX_ONE_REGION.shape[1])
        )

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_IMAGE_MATRIX_ONE_REGION))

    def test_one_binary_image_to_region(self):
        """Ensures correct output from _one_binary_image_to_region."""

        (these_row_indices, these_column_indices
        ) = object_based_eval._one_binary_image_to_region(
            BINARY_IMAGE_MATRIX_ONE_REGION)

        self.assertTrue(numpy.array_equal(
            these_row_indices, ROW_INDICES_ONE_REGION))
        self.assertTrue(numpy.array_equal(
            these_column_indices, COLUMN_INDICES_ONE_REGION))

    def test_find_endpoints_of_skeleton(self):
        """Ensures correct output from _find_endpoints_of_skeleton."""

        this_binary_endpoint_matrix = (
            object_based_eval._find_endpoints_of_skeleton(
                BINARY_SKELETON_MATRIX)
        )
        self.assertTrue(numpy.array_equal(
            this_binary_endpoint_matrix, BINARY_ENDPOINT_MATRIX))

    def test_get_skeleton_line_endpoint_length(self):
        """Ensures correct output from _get_skeleton_line_endpoint_length."""

        this_length_metres = (
            object_based_eval._get_skeleton_line_endpoint_length(
                row_indices=ROW_INDICES_ONE_SKELETON,
                column_indices=COLUMN_INDICES_ONE_SKELETON,
                x_grid_spacing_metres=X_GRID_SPACING_METRES,
                y_grid_spacing_metres=Y_GRID_SPACING_METRES)
        )
        self.assertTrue(numpy.isclose(
            this_length_metres, ENDPOINT_LENGTH_METRES, atol=TOLERANCE))

    def test_get_skeleton_line_arc_length(self):
        """Ensures correct output from _get_skeleton_line_arc_length."""

        this_length_metres = (
            object_based_eval._get_skeleton_line_arc_length(
                row_indices=ROW_INDICES_ONE_SKELETON,
                column_indices=COLUMN_INDICES_ONE_SKELETON,
                x_grid_spacing_metres=X_GRID_SPACING_METRES,
                y_grid_spacing_metres=Y_GRID_SPACING_METRES)
        )
        self.assertTrue(numpy.isclose(
            this_length_metres, ARC_LENGTH_METRES, atol=TOLERANCE))

    def test_get_skeleton_line_quality_positive(self):
        """Ensures correct output from _get_skeleton_line_quality.

        In this case the quality should be positive, because the endpoint length
        meets the requirement.
        """

        this_quality = object_based_eval._get_skeleton_line_quality(
            row_indices=ROW_INDICES_ONE_SKELETON,
            column_indices=COLUMN_INDICES_ONE_SKELETON,
            x_grid_spacing_metres=X_GRID_SPACING_METRES,
            y_grid_spacing_metres=Y_GRID_SPACING_METRES,
            min_endpoint_length_metres=MIN_ENDPOINT_LENGTH_SMALL_METRES)

        self.assertTrue(numpy.isclose(
            this_quality, SKELETON_LINE_QUALITY_POSITIVE, atol=TOLERANCE))

    def test_get_skeleton_line_quality_negative(self):
        """Ensures correct output from _get_skeleton_line_quality.

        In this case the quality should be negative, because the endpoint length
        does not meet the requirement.
        """

        this_quality = object_based_eval._get_skeleton_line_quality(
            row_indices=ROW_INDICES_ONE_SKELETON,
            column_indices=COLUMN_INDICES_ONE_SKELETON,
            x_grid_spacing_metres=X_GRID_SPACING_METRES,
            y_grid_spacing_metres=Y_GRID_SPACING_METRES,
            min_endpoint_length_metres=MIN_ENDPOINT_LENGTH_LARGE_METRES)

        self.assertTrue(numpy.isclose(
            this_quality, SKELETON_LINE_QUALITY_NEGATIVE, atol=TOLERANCE))

    def test_get_distance_between_fronts(self):
        """Ensures correct output from _get_distance_between_fronts."""

        this_distance_metres = object_based_eval._get_distance_between_fronts(
            first_x_coords_metres=THESE_FIRST_X_METRES,
            first_y_coords_metres=THESE_FIRST_Y_METRES,
            second_x_coords_metres=THESE_SECOND_X_METRES,
            second_y_coords_metres=THESE_SECOND_Y_METRES)

        self.assertTrue(numpy.isclose(
            this_distance_metres, INTERFRONT_DISTANCE_METRES,
            atol=TOLERANCE))

    def test_determinize_probabilities(self):
        """Ensures correct output from determinize_probabilities."""

        this_label_matrix = object_based_eval.determinize_probabilities(
            class_probability_matrix=PROBABILITY_MATRIX,
            binarization_threshold=BINARIZATION_THRESHOLD)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, PREDICTED_LABEL_MATRIX))

    def test_images_to_regions(self):
        """Ensures correct output from images_to_regions."""

        this_region_table = object_based_eval.images_to_regions(
            predicted_label_matrix=PREDICTED_LABEL_MATRIX,
            image_times_unix_sec=IMAGE_TIMES_UNIX_SEC)

        self.assertTrue(_compare_tables(
            this_region_table, PREDICTED_REGION_TABLE))

    def test_regions_to_images(self):
        """Ensures correct output from regions_to_images."""

        this_label_matrix = object_based_eval.regions_to_images(
            predicted_region_table=PREDICTED_REGION_TABLE,
            num_grid_rows=PREDICTED_LABEL_MATRIX.shape[1],
            num_grid_columns=PREDICTED_LABEL_MATRIX.shape[2])

        self.assertTrue(numpy.array_equal(
            this_label_matrix, PREDICTED_LABEL_MATRIX))

    def test_discard_regions_with_small_area(self):
        """Ensures correct output from discard_regions_with_small_area."""

        this_region_table = object_based_eval.discard_regions_with_small_area(
            predicted_region_table=copy.deepcopy(PREDICTED_REGION_TABLE),
            x_grid_spacing_metres=X_GRID_SPACING_METRES,
            y_grid_spacing_metres=Y_GRID_SPACING_METRES,
            min_area_metres2=MIN_REGION_AREA_METRES2)

        self.assertTrue(_compare_tables(
            this_region_table, PREDICTED_REGION_TABLE_SANS_SMALL_AREA))

    def test_skeletonize_frontal_regions(self):
        """Ensures correct output from skeletonize_frontal_regions."""

        this_skeleton_table = object_based_eval.skeletonize_frontal_regions(
            predicted_region_table=copy.deepcopy(PREDICTED_REGION_TABLE),
            num_grid_rows=PREDICTED_LABEL_MATRIX.shape[1],
            num_grid_columns=PREDICTED_LABEL_MATRIX.shape[2])

        self.assertTrue(_compare_tables(
            this_skeleton_table, PREDICTED_SKELETON_TABLE))

    def test_find_main_skeletons(self):
        """Ensures correct output from find_main_skeletons."""

        this_main_skeleton_table = object_based_eval.find_main_skeletons(
            predicted_region_table=copy.deepcopy(PREDICTED_SKELETON_TABLE),
            class_probability_matrix=PROBABILITY_MATRIX,
            image_times_unix_sec=IMAGE_TIMES_UNIX_SEC,
            x_grid_spacing_metres=1., y_grid_spacing_metres=1.,
            min_endpoint_length_metres=0.001)

        self.assertTrue(_compare_tables(
            this_main_skeleton_table, PREDICTED_MAIN_SKELETON_TABLE))

    def test_convert_regions_rowcol_to_narr_xy_fcn_false(self):
        """Ensures correct output from convert_regions_rowcol_to_narr_xy.

        In this case, `are_predictions_from_fcn` is False.
        """

        this_region_table = object_based_eval.convert_regions_rowcol_to_narr_xy(
            predicted_region_table=copy.deepcopy(PREDICTED_REGION_TABLE),
            are_predictions_from_fcn=False)

        self.assertTrue(_compare_tables(
            this_region_table, PREDICTED_REGION_TABLE_WITH_XY_COORDS))

    def test_convert_regions_rowcol_to_narr_xy_fcn_true(self):
        """Ensures correct output from convert_regions_rowcol_to_narr_xy.

        In this case, `are_predictions_from_fcn` is True.
        """

        this_input_table = copy.deepcopy(PREDICTED_REGION_TABLE)
        this_input_table[
            object_based_eval.ROW_INDICES_COLUMN
        ] -= ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT
        this_input_table[
            object_based_eval.COLUMN_INDICES_COLUMN
        ] -= ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT

        this_region_table = object_based_eval.convert_regions_rowcol_to_narr_xy(
            predicted_region_table=this_input_table,
            are_predictions_from_fcn=True)

        this_region_table[
            object_based_eval.ROW_INDICES_COLUMN
        ] += ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT
        this_region_table[
            object_based_eval.COLUMN_INDICES_COLUMN
        ] += ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT

        self.assertTrue(_compare_tables(
            this_region_table, PREDICTED_REGION_TABLE_WITH_XY_COORDS))

    def test_get_binary_contingency_table(self):
        """Ensures correct output from get_binary_contingency_table."""

        this_contingency_table_as_dict = (
            object_based_eval.get_binary_contingency_table(
                predicted_region_table=PREDICTED_REGION_TABLE_FOR_CT,
                actual_polyline_table=ACTUAL_POLYLINE_TABLE,
                neigh_distance_metres=NEIGH_DISTANCE_METRES)
        )

        actual_keys = this_contingency_table_as_dict.keys()
        expected_keys = BINARY_CONTINGENCY_TABLE_AS_DICT.keys()
        self.assertTrue(set(actual_keys) == set(expected_keys))

        for this_key in actual_keys:
            self.assertTrue(this_contingency_table_as_dict[this_key] ==
                            BINARY_CONTINGENCY_TABLE_AS_DICT[this_key])

    def test_get_row_normalized_contingency_table(self):
        """Ensures correct output from get_row_normalized_contingency_table."""

        this_row_normalized_ct_matrix = (
            object_based_eval.get_row_normalized_contingency_table(
                predicted_region_table=PREDICTED_REGION_TABLE_FOR_CT,
                actual_polyline_table=ACTUAL_POLYLINE_TABLE,
                neigh_distance_metres=NEIGH_DISTANCE_METRES)
        )

        self.assertTrue(numpy.allclose(
            this_row_normalized_ct_matrix, ROW_NORMALIZED_CT_AS_MATRIX,
            atol=TOLERANCE, equal_nan=True))

    def test_get_column_normalized_contingency_table(self):
        """Ensures correctness of get_column_normalized_contingency_table."""

        this_column_normalized_ct_matrix = (
            object_based_eval.get_column_normalized_contingency_table(
                predicted_region_table=PREDICTED_REGION_TABLE_FOR_CT,
                actual_polyline_table=ACTUAL_POLYLINE_TABLE,
                neigh_distance_metres=NEIGH_DISTANCE_METRES)
        )

        self.assertTrue(numpy.allclose(
            this_column_normalized_ct_matrix, COLUMN_NORMALIZED_CT_AS_MATRIX,
            atol=TOLERANCE, equal_nan=True))

    def test_get_binary_pod(self):
        """Ensures correct output from get_binary_pod."""

        this_binary_pod = object_based_eval.get_binary_pod(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_pod, BINARY_POD, atol=TOLERANCE))

    def test_get_binary_fom(self):
        """Ensures correct output from get_binary_fom."""

        this_binary_fom = object_based_eval.get_binary_fom(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_fom, BINARY_FOM, atol=TOLERANCE))

    def test_get_binary_success_ratio(self):
        """Ensures correct output from get_binary_success_ratio."""

        this_binary_success_ratio = object_based_eval.get_binary_success_ratio(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_success_ratio, BINARY_SUCCESS_RATIO, atol=TOLERANCE))

    def test_get_binary_far(self):
        """Ensures correct output from get_binary_far."""

        this_binary_far = object_based_eval.get_binary_far(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_far, BINARY_FAR, atol=TOLERANCE))

    def test_get_binary_csi(self):
        """Ensures correct output from get_binary_csi."""

        this_binary_csi = object_based_eval.get_binary_csi(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_csi, BINARY_CSI, atol=TOLERANCE))

    def test_get_binary_frequency_bias(self):
        """Ensures correct output from get_binary_frequency_bias."""

        this_binary_freq_bias = object_based_eval.get_binary_frequency_bias(
            FAKE_BINARY_CT_AS_DICT)
        self.assertTrue(numpy.isclose(
            this_binary_freq_bias, BINARY_FREQUENCY_BIAS, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
