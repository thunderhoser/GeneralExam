"""Unit tests for object_based_evaluation.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.evaluation import object_based_evaluation as object_based_eval
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6
ARRAY_COLUMN_NAMES = [
    object_based_eval.ROW_INDICES_COLUMN,
    object_based_eval.COLUMN_INDICES_COLUMN]

# The following constants are used to test _one_region_to_binary_image and
# _one_binary_image_to_region.
BINARY_IMAGE_MATRIX_ONE_REGION = numpy.array([[1, 1, 0, 0, 0, 1, 0, 0],
                                              [0, 1, 0, 0, 0, 1, 1, 0],
                                              [0, 1, 1, 0, 0, 1, 1, 0],
                                              [0, 1, 1, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 1, 1, 0, 0, 0]])

ROW_INDICES_ONE_REGION = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int)
COLUMN_INDICES_ONE_REGION = numpy.array(
    [0, 1, 5, 1, 5, 6, 1, 2, 5, 6, 1, 2, 5, 2, 3, 4], dtype=int)

# The following constants are used to test _get_length_of_bounding_box_diagonal.
X_GRID_SPACING_METRES, Y_GRID_SPACING_METRES = (
    nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME))
LENGTH_OF_BB_DIAG_ONE_REGION_METRES = numpy.sqrt(52) * X_GRID_SPACING_METRES

# The following constants are used to test determinize_probabilities.
PROBABILITY_MATRIX_CLASS0 = numpy.array(
    [[0.7, 0.1, 0.4, 0.9, 0.6, 0.2, 0.5, 0.6],
     [0.7, 0.6, 0.6, 1.0, 0.7, 0.6, 0.3, 0.8],
     [0.5, 0.9, 0.6, 0.9, 0.6, 0.2, 0.4, 0.9],
     [0.8, 0.8, 0.2, 0.4, 1.0, 0.5, 0.9, 0.9],
     [0.2, 0.9, 0.9, 0.2, 0.2, 0.5, 0.1, 0.1]])

PROBABILITY_MATRIX_CLASS1 = numpy.array(
    [[0.2, 0.7, 0.3, 0.1, 0.1, 0.2, 0.2, 0.1],
     [0.3, 0.2, 0.3, 0.0, 0.1, 0.1, 0.3, 0.0],
     [0.4, 0.0, 0.3, 0.0, 0.1, 0.3, 0.2, 0.0],
     [0.1, 0.0, 0.6, 0.2, 0.0, 0.2, 0.1, 0.1],
     [0.5, 0.9, 0.1, 0.5, 0.3, 0.0, 0.4, 0.3]])

PROBABILITY_MATRIX_CLASS2 = numpy.array(
    [[0.1, 0.2, 0.3, 0.0, 0.3, 0.6, 0.3, 0.3],
     [0., 0.2, 0.1, 0.0, 0.2, 0.3, 0.4, 0.2],
     [0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.4, 0.1],
     [0.1, 0.2, 0.2, 0.4, 0.0, 0.3, 0.0, 0.0],
     [0.3, 0.1, 0.0, 0.3, 0.5, 0.5, 0.5, 0.6]])

PROBABILITY_MATRIX = numpy.stack(
    (PROBABILITY_MATRIX_CLASS0, PROBABILITY_MATRIX_CLASS1,
     PROBABILITY_MATRIX_CLASS2), axis=-1)
PROBABILITY_MATRIX = numpy.stack((PROBABILITY_MATRIX,), axis=0)

PREDICTED_LABEL_MATRIX = numpy.array(
    [[1, 1, 1, 0, 2, 2, 2, 2],
     [1, 1, 1, 0, 2, 2, 2, 0],
     [1, 0, 1, 0, 2, 2, 2, 0],
     [0, 0, 1, 2, 0, 2, 0, 0],
     [1, 0, 0, 1, 2, 2, 2, 2]], dtype=int)
PREDICTED_LABEL_MATRIX = numpy.stack((PREDICTED_LABEL_MATRIX,), axis=0)

BINARIZATION_THRESHOLD = 0.75

# The following constants are used to test images_to_regions.
IMAGE_TIMES_UNIX_SEC = numpy.array([0], dtype=int)
REGION_TIMES_UNIX_SEC = numpy.array([0, 0, 0], dtype=int)
FRONT_TYPES = [
    front_utils.WARM_FRONT_STRING_ID, front_utils.COLD_FRONT_STRING_ID,
    front_utils.WARM_FRONT_STRING_ID]

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
    ROW_INDICES_SMALL_WF_REGION]
COLUMN_INDICES_BY_REGION = [
    COLUMN_INDICES_LARGE_WF_REGION, COLUMN_INDICES_CF_REGION,
    COLUMN_INDICES_SMALL_WF_REGION]

PREDICTED_REGION_DICT = {
    front_utils.TIME_COLUMN: REGION_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: FRONT_TYPES,
    object_based_eval.ROW_INDICES_COLUMN: ROW_INDICES_BY_REGION,
    object_based_eval.COLUMN_INDICES_COLUMN: COLUMN_INDICES_BY_REGION
}
PREDICTED_REGION_TABLE = pandas.DataFrame.from_dict(PREDICTED_REGION_DICT)

# The following constants are used to test discard_small_regions.
MIN_REGION_LENGTH_METRES = 1e5
PREDICTED_LARGE_REGION_TABLE = PREDICTED_REGION_TABLE.drop(
    PREDICTED_REGION_TABLE.index[[2]], axis=0, inplace=False)


def _compare_tables(table1, table2):
    """Ensures that two pandas DataFrames are equal.

    :param table1: pandas DataFrame.
    :param table2: pandas DataFrame.
    :return: are_tables_equal: Boolean flag.
    """

    these_column_names = list(table1)
    expected_column_names = list(table2)
    if set(these_column_names) != set(expected_column_names):
        return False

    this_num_regions = len(table1.index)
    expected_num_regions = len(table2.index)
    if this_num_regions != expected_num_regions:
        return False

    for i in range(this_num_regions):
        for this_column_name in these_column_names:
            if this_column_name in ARRAY_COLUMN_NAMES:
                are_these_entries_equal = numpy.array_equal(
                    table1[this_column_name].values[i],
                    table2[this_column_name].values[i])

            else:
                are_these_entries_equal = (
                    table1[this_column_name].values[i] ==
                    table2[this_column_name].values[i])

            if not are_these_entries_equal:
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
                num_grid_columns=BINARY_IMAGE_MATRIX_ONE_REGION.shape[1]))

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_IMAGE_MATRIX_ONE_REGION))

    def test_one_binary_image_to_region(self):
        """Ensures correct output from _one_binary_image_to_region."""

        these_row_indices, these_column_indices = (
            object_based_eval._one_binary_image_to_region(
                BINARY_IMAGE_MATRIX_ONE_REGION))

        self.assertTrue(numpy.array_equal(
            these_row_indices, ROW_INDICES_ONE_REGION))
        self.assertTrue(numpy.array_equal(
            these_column_indices, COLUMN_INDICES_ONE_REGION))

    def test_get_length_of_bounding_box_diagonal(self):
        """Ensures correct output from _get_length_of_bounding_box_diagonal."""

        this_length_metres = (
            object_based_eval._get_length_of_bounding_box_diagonal(
                row_indices_in_region=ROW_INDICES_ONE_REGION,
                column_indices_in_region=COLUMN_INDICES_ONE_REGION,
                x_grid_spacing_metres=X_GRID_SPACING_METRES,
                y_grid_spacing_metres=Y_GRID_SPACING_METRES))

        self.assertTrue(numpy.isclose(
            this_length_metres, LENGTH_OF_BB_DIAG_ONE_REGION_METRES,
            atol=TOLERANCE))

    def test_determinize_probabilities(self):
        """Ensures correct output from determinize_probabilities."""

        this_predicted_label_matrix = (
            object_based_eval.determinize_probabilities(
                class_probability_matrix=PROBABILITY_MATRIX,
                binarization_threshold=BINARIZATION_THRESHOLD))

        self.assertTrue(numpy.array_equal(
            this_predicted_label_matrix, PREDICTED_LABEL_MATRIX))

    def test_images_to_regions(self):
        """Ensures correct output from images_to_regions."""

        this_predicted_region_table = object_based_eval.images_to_regions(
            predicted_label_matrix=PREDICTED_LABEL_MATRIX,
            image_times_unix_sec=IMAGE_TIMES_UNIX_SEC)

        self.assertTrue(_compare_tables(
            this_predicted_region_table, PREDICTED_REGION_TABLE))

    def test_discard_small_regions(self):
        """Ensures correct output from discard_small_regions."""

        this_input_table = copy.deepcopy(PREDICTED_REGION_TABLE)
        this_predicted_region_table = object_based_eval.discard_small_regions(
            predicted_region_table=this_input_table,
            min_bounding_box_diag_length_metres=MIN_REGION_LENGTH_METRES)

        self.assertTrue(_compare_tables(
            this_predicted_region_table, PREDICTED_LARGE_REGION_TABLE))


if __name__ == '__main__':
    unittest.main()
