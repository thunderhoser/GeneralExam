"""Unit tests for skeleton_lines.py."""

import copy
import collections
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from generalexam.ge_utils import skeleton_lines

TOLERANCE = 1e-6

INTEGER_ARRAY_COLUMNS = [
    skeleton_lines.VERTEX_INDICES_KEY, skeleton_lines.TRIANGLE_INDICES_KEY,
    skeleton_lines.TRIANGLE_POSITIONS_KEY, skeleton_lines.NEW_EDGE_INDICES_KEY,
    skeleton_lines.NODE_INDICES_KEY, skeleton_lines.LEFT_CHILD_INDICES_KEY,
    skeleton_lines.RIGHT_CHILD_INDICES_KEY]
STRING_COLUMNS = [skeleton_lines.NODE_TYPE_KEY]
FLOAT_COLUMNS = [
    skeleton_lines.NODE_X_COORDS_KEY, skeleton_lines.NODE_Y_COORDS_KEY]

VERTEX_INDICES_ADJACENT = numpy.array([0, 1], dtype=int)
VERTEX_INDICES_NON_ADJACENT = numpy.array([1, 3], dtype=int)
VERTEX_INDICES_ADJACENT_WRAPAROUND = numpy.array([10, 0], dtype=int)
NUM_VERTICES_FOR_ADJACENCY_TEST = 11

VERTEX_X_COORDS_EXPLICITLY_CLOSED = numpy.array(
    [0., 0., -1., -1., 0., 0., -2., -2., 0.])
VERTEX_Y_COORDS_EXPLICITLY_CLOSED = numpy.array(
    [0., 1., 1., 2., 2., 3., 2., 1., 0.])
POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=VERTEX_X_COORDS_EXPLICITLY_CLOSED,
    exterior_y_coords=VERTEX_Y_COORDS_EXPLICITLY_CLOSED)

VERTEX_X_COORDS = numpy.array([0., 0., -1., -1., 0., 0., -2., -2.])
VERTEX_Y_COORDS = numpy.array([0., 1., 1., 2., 2., 3., 2., 1.])
TRIANGLE_TO_VERTEX_MATRIX = numpy.array([[2, 7, 0],
                                         [3, 5, 6],
                                         [7, 3, 6],
                                         [3, 7, 2],
                                         [1, 2, 0],
                                         [3, 4, 5]])

NEW_EDGE_TO_VERTICES = [[2, 7], [0, 2], [3, 5], [3, 6], [3, 7]]
NEW_EDGE_TO_TRIANGLE_INDICES = [[0, 3], [0, 4], [1, 5], [1, 2], [2, 3]]
NEW_EDGE_TO_TRIANGLE_POSITIONS = [[0, 1], [2, 1], [0, 2], [2, 1], [0, 0]]

NEW_EDGE_NODE_TYPES = [
    skeleton_lines.JUMPER_NODE_TYPE, skeleton_lines.JUMPER_NODE_TYPE,
    skeleton_lines.JUMPER_NODE_TYPE, skeleton_lines.JUMPER_NODE_TYPE,
    skeleton_lines.JUMPER_NODE_TYPE]
NEW_EDGE_X_COORDS = numpy.array([-1.5, -0.5, -0.5, -1.5, -1.5])
NEW_EDGE_Y_COORDS = numpy.array([1., 0.5, 2.5, 2., 1.5])

NEW_EDGE_DICT = {
    skeleton_lines.VERTEX_INDICES_KEY: NEW_EDGE_TO_VERTICES,
    skeleton_lines.TRIANGLE_INDICES_KEY: NEW_EDGE_TO_TRIANGLE_INDICES,
    skeleton_lines.TRIANGLE_POSITIONS_KEY: NEW_EDGE_TO_TRIANGLE_POSITIONS
}
NEW_EDGE_TABLE = pandas.DataFrame.from_dict(NEW_EDGE_DICT)

TRIANGLE_TO_NEW_EDGE_INDICES = [[0, 1], [2, 3], [3, 4], [0, 4], [1], [2]]
TRIANGLE_TO_NEW_EDGE_DICT = {
    skeleton_lines.NEW_EDGE_INDICES_KEY: TRIANGLE_TO_NEW_EDGE_INDICES
}
TRIANGLE_TO_NEW_EDGE_TABLE = pandas.DataFrame.from_dict(
    TRIANGLE_TO_NEW_EDGE_DICT)

END_NODE_VERTEX_INDICES = numpy.array([1, 4], dtype=int)
END_NODE_TYPES = [skeleton_lines.END_NODE_TYPE, skeleton_lines.END_NODE_TYPE]
END_NODE_TO_TRIANGLE_INDICES = [[4], [5]]
END_NODE_TO_TRIANGLE_POSITIONS = [[], []]

NODE_DICTIONARY = {
    skeleton_lines.NODE_TYPE_KEY: NEW_EDGE_NODE_TYPES + END_NODE_TYPES,
    skeleton_lines.NODE_X_COORDS_KEY: numpy.concatenate((
        NEW_EDGE_X_COORDS, VERTEX_X_COORDS[END_NODE_VERTEX_INDICES])),
    skeleton_lines.NODE_Y_COORDS_KEY: numpy.concatenate((
        NEW_EDGE_Y_COORDS, VERTEX_Y_COORDS[END_NODE_VERTEX_INDICES])),
    skeleton_lines.TRIANGLE_INDICES_KEY:
        NEW_EDGE_TO_TRIANGLE_INDICES + END_NODE_TO_TRIANGLE_INDICES,
    skeleton_lines.TRIANGLE_POSITIONS_KEY:
        NEW_EDGE_TO_TRIANGLE_POSITIONS + END_NODE_TO_TRIANGLE_POSITIONS
}
NODE_TABLE_SANS_CHILDREN = pandas.DataFrame.from_dict(NODE_DICTIONARY)

TRIANGLE_TO_NODE_INDICES = [[0, 1], [2, 3], [3, 4], [0, 4], [1, 5], [2, 6]]
TRIANGLE_TO_NODE_DICT = {
    skeleton_lines.NODE_INDICES_KEY: TRIANGLE_TO_NODE_INDICES
}
TRIANGLE_TO_NODE_TABLE = pandas.DataFrame.from_dict(TRIANGLE_TO_NODE_DICT)

NODE_TO_LEFT_CHILD_INDICES = [[1, 4], [0, 5], [3, 6], [2, 4], [3, 0], [], []]
NODE_TO_RIGHT_CHILD_INDICES = [[], [], [], [], [], [], []]
THIS_DICT = {
    skeleton_lines.LEFT_CHILD_INDICES_KEY: NODE_TO_LEFT_CHILD_INDICES,
    skeleton_lines.RIGHT_CHILD_INDICES_KEY: NODE_TO_RIGHT_CHILD_INDICES
}
NODE_TABLE_WITH_CHILDREN = NODE_TABLE_SANS_CHILDREN.assign(**THIS_DICT)

CONVEX_HULL_VERTEX_INDICES = copy.deepcopy(END_NODE_VERTEX_INDICES)


def _compare_tables(expected_table, actual_table):
    """Determines whether or not two pandas DataFrames are equal.

    :param expected_table: expected pandas DataFrame.
    :param actual_table: actual pandas DataFrame.
    :return: tables_equal_flag: Boolean flag.
    """

    expected_num_rows = len(expected_table.index)
    actual_num_rows = len(actual_table.index)
    if expected_num_rows != actual_num_rows:
        return False

    expected_column_names = list(expected_table)
    actual_column_names = list(actual_table)
    if set(expected_column_names) != set(actual_column_names):
        return False

    for i in range(expected_num_rows):
        for this_column_name in expected_column_names:
            if this_column_name in STRING_COLUMNS:
                are_entries_equal = (
                    expected_table[this_column_name].values[i] ==
                    actual_table[this_column_name].values[i])

            elif this_column_name in INTEGER_ARRAY_COLUMNS:
                these_expected_values = expected_table[
                    this_column_name].values[i]
                if isinstance(these_expected_values, numpy.ndarray):
                    these_expected_values = these_expected_values.tolist()

                these_actual_values = actual_table[this_column_name].values[i]
                if isinstance(these_actual_values, numpy.ndarray):
                    these_actual_values = these_actual_values.tolist()

                are_entries_equal = (
                    set(these_expected_values) == set(these_actual_values))

            else:
                are_entries_equal = numpy.isclose(
                    expected_table[this_column_name].values[i],
                    actual_table[this_column_name].values[i], atol=TOLERANCE)

            if not are_entries_equal:
                return False

    return True


class SkeletonLinesTests(unittest.TestCase):
    """Each method is a unit test for skeleton_lines.py."""

    def test_polygon_to_vertex_arrays(self):
        """Ensures correct output from _polygon_to_vertex_arrays."""

        these_vertex_x_coords, these_vertex_y_coords = (
            skeleton_lines._polygon_to_vertex_arrays(POLYGON_OBJECT_XY))

        self.assertTrue(numpy.allclose(
            these_vertex_x_coords, VERTEX_X_COORDS, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_coords, VERTEX_Y_COORDS, atol=TOLERANCE))

    def test_are_vertices_adjacent_yes_ascending(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, vertices are adjacent and sorted in ascending order.
        """

        self.assertTrue(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_ADJACENT,
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_are_vertices_adjacent_yes_descending(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, vertices are adjacent and sorted in descending order.
        """

        self.assertTrue(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_ADJACENT[::-1],
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_are_vertices_adjacent_no_ascending(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, vertices are non-adjacent and sorted in ascending order.
        """

        self.assertFalse(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_NON_ADJACENT,
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_are_vertices_adjacent_no_descending(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, vertices are non-adjacent and sorted in descending order.
        """

        self.assertFalse(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_NON_ADJACENT[::-1],
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_are_vertices_adjacent_first_before_last(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, input vertices are first before last.
        """

        self.assertTrue(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_ADJACENT_WRAPAROUND,
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_are_vertices_adjacent_last_before_first(self):
        """Ensures correct output from _are_vertices_adjacent.

        In this case, input vertices are last before first.
        """

        self.assertTrue(skeleton_lines._are_vertices_adjacent(
            vertex_indices=VERTEX_INDICES_ADJACENT_WRAPAROUND[::-1],
            num_vertices_in_polygon=NUM_VERTICES_FOR_ADJACENCY_TEST))

    def test_get_delaunay_triangulation(self):
        """Ensures correct output from _get_delaunay_triangulation."""

        this_triangle_to_vertex_matrix = (
            skeleton_lines._get_delaunay_triangulation(POLYGON_OBJECT_XY))

        self.assertTrue(numpy.array_equal(
            this_triangle_to_vertex_matrix, TRIANGLE_TO_VERTEX_MATRIX))

    def test_find_new_edges_from_triangulation(self):
        """Ensures correct output from _find_new_edges_from_triangulation."""

        this_new_edge_table, this_triangle_to_new_edge_table = (
            skeleton_lines._find_new_edges_from_triangulation(
                polygon_object_xy=POLYGON_OBJECT_XY,
                triangle_to_vertex_matrix=TRIANGLE_TO_VERTEX_MATRIX))

        self.assertTrue(_compare_tables(NEW_EDGE_TABLE, this_new_edge_table))
        self.assertTrue(_compare_tables(
            TRIANGLE_TO_NEW_EDGE_TABLE, this_triangle_to_new_edge_table))

    def test_find_end_nodes_of_triangulation(self):
        """Ensures correct output from _find_end_nodes_of_triangulation."""

        these_end_node_vertex_indices = (
            skeleton_lines._find_end_nodes_of_triangulation(
                triangle_to_vertex_matrix=TRIANGLE_TO_VERTEX_MATRIX,
                new_edge_table=NEW_EDGE_TABLE))

        self.assertTrue(numpy.array_equal(
            these_end_node_vertex_indices, END_NODE_VERTEX_INDICES))

    def test_find_and_classify_nodes(self):
        """Ensures correct output from _find_and_classify_nodes."""

        this_node_table, this_triangle_to_node_table = (
            skeleton_lines._find_and_classify_nodes(
                polygon_object_xy=POLYGON_OBJECT_XY,
                new_edge_table=NEW_EDGE_TABLE,
                triangle_to_new_edge_table=TRIANGLE_TO_NEW_EDGE_TABLE,
                triangle_to_vertex_matrix=TRIANGLE_TO_VERTEX_MATRIX,
                end_node_vertex_indices=END_NODE_VERTEX_INDICES))

        self.assertTrue(_compare_tables(
            NODE_TABLE_SANS_CHILDREN, this_node_table))
        self.assertTrue(_compare_tables(
            TRIANGLE_TO_NODE_TABLE, this_triangle_to_node_table))

    def test_find_and_classify_node_children(self):
        """Ensures correct output from _find_and_classify_node_children."""

        # TODO(thunderhoser): This is a somewhat trivial test, because there are
        # only end/jumper nodes, no branch nodes.

        this_node_table = skeleton_lines._find_and_classify_node_children(
            node_table=NODE_TABLE_SANS_CHILDREN,
            triangle_to_new_edge_table=TRIANGLE_TO_NEW_EDGE_TABLE,
            triangle_to_node_table=TRIANGLE_TO_NODE_TABLE)

        self.assertTrue(_compare_tables(
            NODE_TABLE_WITH_CHILDREN, this_node_table))

    def test_get_convex_hull_of_end_nodes(self):
        """Ensures correct output from _get_convex_hull_of_end_nodes."""

        # TODO(thunderhoser): This is a trivial test, because there are only 2
        # end nodes.  When there are < 3 end nodes, the method just returns the
        # original end nodes and does not actually compute the convex hull.

        these_vertex_indices = skeleton_lines._get_convex_hull_of_end_nodes(
            polygon_object_xy=POLYGON_OBJECT_XY,
            end_node_vertex_indices=END_NODE_VERTEX_INDICES)

        self.assertTrue(numpy.array_equal(
            these_vertex_indices, CONVEX_HULL_VERTEX_INDICES))


if __name__ == '__main__':
    unittest.main()
