"""Helper methods for A-star search.

--- NOTATION ---

I will use the following letters to denote matrix dimensions.

M = number of grid rows (unique y-coordinates at grid points)
N = number of grid columns (unique x-coordinates at grid points)
r = row coordinate (integer)
c = column coordinate (integer)
"""

import numpy
from gewittergefahr.gg_utils import error_checking
from astar import AStar

NUM_GRID_ROWS_KEY = 'num_grid_rows'
NUM_GRID_COLUMNS_KEY = 'num_grid_columns'


class GridSearch(AStar):
    """Allows A-star to search through a physical grid.

    Each node is a grid cell at location (column, row)*** or (c, r).  The goal
    is to navigate from the start node (first grid cell) to the end node (last
    grid cell), using only grid cells that are part of a connected region.
    There may be no path through the connected region, in which case A-star
    fails.

    *** I also hate this notation (row should come first), but it's required by
    the AStar library.
    """

    def __init__(self, binary_region_matrix):
        """Creates new instance.

        :param binary_region_matrix: M-by-N numpy array of integers in 0...1.
            If binary_region_matrix[i, j] = 1, grid cell [i, j] is part of the
            connected region.
        """

        error_checking.assert_is_numpy_array(
            binary_region_matrix, num_dimensions=2)
        error_checking.assert_is_integer_numpy_array(binary_region_matrix)
        error_checking.assert_is_geq_numpy_array(binary_region_matrix, 0)
        error_checking.assert_is_leq_numpy_array(binary_region_matrix, 1)

        setattr(self, NUM_GRID_ROWS_KEY, binary_region_matrix.shape[0])
        setattr(self, NUM_GRID_COLUMNS_KEY, binary_region_matrix.shape[1])

        # self.num_grid_rows = binary_region_matrix.shape[0]
        # self.num_grid_columns = binary_region_matrix.shape[1]
        self.row_indices_in_region, self.column_indices_in_region = numpy.where(
            binary_region_matrix == 1)

    def heuristic_cost_estimate(self, first_node_object, second_node_object):
        """Returns heuristic cost estimate for path between two nodes.

        :param first_node_object: (column, row) tuple.
        :param second_node_object: (column, row) tuple.
        :return: heuristic_cost: Euclidean distance (number of grid lengths
            along straight line between nodes 1 and 2).
        """

        (first_column, first_row) = first_node_object
        (second_column, second_row) = second_node_object

        return numpy.sqrt((first_row - second_row) ** 2 +
                          (first_column - second_column) ** 2)

    def distance_between(self, first_node_object, second_node_object):
        """Returns distance between two neighbours.

        The inputs to this method are always neighbours, so Euclidean distance
        is sufficient.

        :param first_node_object: (column, row) tuple.
        :param second_node_object: (column, row) tuple.
        :return: distance: Euclidean distance -- i.e., number of grid lengths
            along straight line between grid points 1 and 2, where a "grid
            point" is the center of a grid cell.  This is greater for grid cells
            that are diagonal neighbours (1.4142 units, rather than 1.0).
        """

        (first_column, first_row) = first_node_object
        (second_column, second_row) = second_node_object

        return numpy.sqrt((first_row - second_row) ** 2 +
                          (first_column - second_column) ** 2)

    def neighbors(self, node_object):
        """Returns neighbours of node.

        :param node_object: (column, row) tuple.
        :return: list_of_node_objects: 1-D list, where each element is the
            (column, row) tuple for a neighbour.
        """

        (node_column, node_row) = node_object
        row_flags = numpy.logical_and(
            self.row_indices_in_region >= node_row - 1,
            self.row_indices_in_region <= node_row + 1)
        column_flags = numpy.logical_and(
            self.column_indices_in_region >= node_column - 1,
            self.column_indices_in_region <= node_column + 1)

        neighbour_indices = numpy.where(
            numpy.logical_and(row_flags, column_flags))[0]
        neighbour_indices = neighbour_indices.tolist()

        node_index = numpy.where(numpy.logical_and(
            self.row_indices_in_region == node_row,
            self.column_indices_in_region == node_column))[0][0]
        neighbour_indices.remove(node_index)

        return [(self.column_indices_in_region[i],
                 self.row_indices_in_region[i]) for i in neighbour_indices]


def run_a_star(
        grid_search_object, start_row, start_column, end_row, end_column):
    """Runs A-star search.

    If A-star cannot reach the end node, this method returns None for all
    outputs.

    N = number of nodes in final path

    :param grid_search_object: Instance of `GridSearch`.
    :param start_row: Row index of start node.
    :param start_column: Column index of start node.
    :param end_row: Row index of end node.
    :param end_column: Column index of end node.
    :return: visited_rows: length-N numpy array with row indices of nodes in
        final path.
    :return: visited_columns: length-N numpy array with column indices of nodes
        in final path.
    """

    error_checking.assert_is_integer(start_row)
    error_checking.assert_is_geq(start_row, 0)
    error_checking.assert_is_less_than(
        start_row, getattr(grid_search_object, NUM_GRID_ROWS_KEY))

    error_checking.assert_is_integer(end_row)
    error_checking.assert_is_geq(end_row, 0)
    error_checking.assert_is_less_than(
        end_row, getattr(grid_search_object, NUM_GRID_ROWS_KEY))

    error_checking.assert_is_integer(start_column)
    error_checking.assert_is_geq(start_column, 0)
    error_checking.assert_is_less_than(
        start_column, getattr(grid_search_object, NUM_GRID_COLUMNS_KEY))

    error_checking.assert_is_integer(end_column)
    error_checking.assert_is_geq(end_column, 0)
    error_checking.assert_is_less_than(
        end_column, getattr(grid_search_object, NUM_GRID_COLUMNS_KEY))

    visited_rowcol_tuples = grid_search_object.astar(
        (start_column, start_row), (end_column, end_row))
    if visited_rowcol_tuples is None:
        return None, None

    visited_rowcol_tuples = list(visited_rowcol_tuples)
    visited_rows = numpy.array([x[1] for x in visited_rowcol_tuples], dtype=int)
    visited_columns = numpy.array(
        [x[0] for x in visited_rowcol_tuples], dtype=int)

    return visited_rows, visited_columns
