"""Unit tests for a_star_search.py."""

import unittest
import numpy
from generalexam.ge_utils import a_star_search

BINARY_REGION_MATRIX_WITH_PATH = numpy.array([[1, 0, 0, 0, 1, 0, 1, 0],
                                              [1, 1, 0, 1, 0, 1, 0, 0],
                                              [0, 1, 1, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 1, 1, 1, 0, 0]])

BINARY_REGION_MATRIX_NO_PATH = numpy.array([[1, 0, 0, 0, 1, 0, 1, 0],
                                            [1, 1, 0, 1, 0, 0, 0, 0],
                                            [0, 1, 1, 0, 0, 1, 0, 0],
                                            [0, 0, 1, 0, 0, 1, 0, 0],
                                            [0, 0, 1, 1, 1, 1, 0, 0]])

START_ROW = 0
START_COLUMN = 0
END_ROW = 0
END_COLUMN = 6

EXPECTED_VISITED_ROWS = numpy.array([0, 1, 2, 1, 0, 1, 0], dtype=int)
EXPECTED_VISITED_COLUMNS = numpy.array([0, 1, 2, 3, 4, 5, 6], dtype=int)


class AStarSearchTests(unittest.TestCase):
    """Each method is a unit test for a_star_search.py."""

    def test_run_a_star_path_exists(self):
        """Ensures correct output from run_a_star.

        In this case, path to goal exists.
        """

        this_grid_search_object = a_star_search.GridSearch(
            binary_region_matrix=BINARY_REGION_MATRIX_WITH_PATH)
        these_visited_rows, these_visited_columns = a_star_search.run_a_star(
            grid_search_object=this_grid_search_object, start_row=START_ROW,
            start_column=START_COLUMN, end_row=END_ROW, end_column=END_COLUMN)

        self.assertTrue(numpy.array_equal(
            these_visited_rows, EXPECTED_VISITED_ROWS))
        self.assertTrue(numpy.array_equal(
            these_visited_columns, EXPECTED_VISITED_COLUMNS))

    def test_run_a_star_path_not_exists(self):
        """Ensures correct output from run_a_star.

        In this case, *no* path to goal exists.
        """

        this_grid_search_object = a_star_search.GridSearch(
            binary_region_matrix=BINARY_REGION_MATRIX_NO_PATH)
        these_visited_rows, these_visited_columns = a_star_search.run_a_star(
            grid_search_object=this_grid_search_object, start_row=START_ROW,
            start_column=START_COLUMN, end_row=END_ROW, end_column=END_COLUMN)

        self.assertTrue(these_visited_rows is None)
        self.assertTrue(these_visited_columns is None)


if __name__ == '__main__':
    unittest.main()
