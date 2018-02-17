"""Unit tests for front_utils.py."""

import copy
import unittest
import numpy
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6

GRID_SPACING_X_METRES = 1.
GRID_SPACING_Y_METRES = 1.
ONE_GRID_POINT_X_METRES = 1.
ONE_GRID_POINT_Y_METRES = 2.
ONE_GRID_VERTEX_COORDS_X_METRES = numpy.array([0.5, 1.5, 1.5, 0.5, 0.5])
ONE_GRID_VERTEX_COORDS_Y_METRES = numpy.array([1.5, 1.5, 2.5, 2.5, 1.5])

POLYLINE_X_COORDS_METRES = numpy.array(
    [3.3, 2.3, 1.8, 1.5, 1.3, 1.4, 1.7, 2.1, 2.6, 2.8])
POLYLINE_Y_COORDS_METRES = numpy.array(
    [5.5, 4.7, 4.1, 3.5, 2.5, 1.5, 0.9, 0.1, -0.2, -0.5])
POLYLINE_VERTEX_LIST_XY_METRES = [
    (3.3, 5.5), (2.3, 4.7), (1.8, 4.1), (1.5, 3.5), (1.3, 2.5),
    (1.4, 1.5), (1.7, 0.9), (2.1, 0.1), (2.6, -0.2), (2.8, -0.5)]

NUM_GRID_ROWS = 6
NUM_GRID_COLUMNS = 8
GRID_POINT_X_COORDS_METRES = numpy.array([0., 1., 2., 3., 4., 5., 6., 7.])
GRID_POINT_Y_COORDS_METRES = numpy.array([0., 1., 2., 3., 4., 5.])

GRID_ROWS_IN_POLYLINE = numpy.array(
    [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5], dtype=int)
GRID_COLUMNS_IN_POLYLINE = numpy.array(
    [2, 3, 1, 2, 1, 1, 2, 1, 2, 2, 3], dtype=int)

POLYLINE_AS_BINARY_MATRIX = numpy.array([[0, 0, 1, 1, 0, 0, 0, 0],
                                         [0, 1, 1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 1, 0, 0, 0, 0, 0],
                                         [0, 1, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 1, 0, 0, 0, 0]])

BINARY_MATRIX_DILATED_HALFWIDTH1 = numpy.array([[0, 1, 1, 1, 1, 0, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [0, 1, 1, 1, 1, 0, 0, 0]])

ROWS_IN_POLYLINE_DILATED_HALFWIDTH1 = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
    dtype=int)
COLUMNS_IN_POLYLINE_DILATED_HALFWIDTH1 = numpy.array(
    [1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4],
    dtype=int)

BINARY_MATRIX_DILATED_HALFWIDTH2 = numpy.array([[1, 1, 1, 1, 1, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 0, 0],
                                                [1, 1, 1, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 1, 0, 0]])


class FrontUtilsTests(unittest.TestCase):
    """Each method is a unit test for front_utils.py."""

    def test_vertex_arrays_to_list(self):
        """Ensures correct output from _vertex_arrays_to_list."""

        this_vertex_list_xy_metres = front_utils._vertex_arrays_to_list(
            vertex_x_coords_metres=POLYLINE_X_COORDS_METRES,
            vertex_y_coords_metres=POLYLINE_Y_COORDS_METRES)

        self.assertTrue(len(this_vertex_list_xy_metres) ==
                        len(POLYLINE_VERTEX_LIST_XY_METRES))

        for i in range(len(this_vertex_list_xy_metres)):
            self.assertTrue(numpy.allclose(
                this_vertex_list_xy_metres[i],
                POLYLINE_VERTEX_LIST_XY_METRES[i], atol=TOLERANCE))

    def test_grid_cell_to_polygon(self):
        """Ensures correct output from _grid_cell_to_polygon."""

        this_polygon_object_xy_metres = front_utils._grid_cell_to_polygon(
            grid_point_x_metres=ONE_GRID_POINT_X_METRES,
            grid_point_y_metres=ONE_GRID_POINT_Y_METRES,
            x_spacing_metres=GRID_SPACING_X_METRES,
            y_spacing_metres=GRID_SPACING_Y_METRES)

        these_vertex_x_metres = numpy.array(
            this_polygon_object_xy_metres.exterior.xy[0])
        these_vertex_y_metres = numpy.array(
            this_polygon_object_xy_metres.exterior.xy[1])

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, ONE_GRID_VERTEX_COORDS_X_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, ONE_GRID_VERTEX_COORDS_Y_METRES,
            atol=TOLERANCE))

    def test_polyline_to_grid_points(self):
        """Ensures correct output from _polyline_to_grid_points."""

        these_rows, these_columns = front_utils._polyline_to_grid_points(
            polyline_x_coords_metres=POLYLINE_X_COORDS_METRES,
            polyline_y_coords_metres=POLYLINE_Y_COORDS_METRES,
            grid_point_x_coords_metres=GRID_POINT_X_COORDS_METRES,
            grid_point_y_coords_metres=GRID_POINT_Y_COORDS_METRES)

        self.assertTrue(numpy.array_equal(these_rows, GRID_ROWS_IN_POLYLINE))
        self.assertTrue(numpy.array_equal(
            these_columns, GRID_COLUMNS_IN_POLYLINE))

    def test_grid_points_to_binary_image(self):
        """Ensures correct output from _grid_points_to_binary_image."""

        this_binary_matrix = front_utils._grid_points_to_binary_image(
            rows_in_object=GRID_ROWS_IN_POLYLINE,
            columns_in_object=GRID_COLUMNS_IN_POLYLINE,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_binary_matrix, POLYLINE_AS_BINARY_MATRIX))

    def test_dilate_binary_image_half_width_1(self):
        """Ensures correct output from _dilate_binary_image.

        In this case, half-width of dilation window is one grid cell.
        """

        this_input_matrix = copy.deepcopy(POLYLINE_AS_BINARY_MATRIX)
        this_dilated_matrix = front_utils._dilate_binary_image(
            binary_matrix=this_input_matrix,
            dilation_half_width_in_grid_cells=1)

        self.assertTrue(numpy.array_equal(
            this_dilated_matrix, BINARY_MATRIX_DILATED_HALFWIDTH1))

    def test_dilate_binary_image_half_width_2(self):
        """Ensures correct output from _dilate_binary_image.

        In this case, half-width of dilation window is 2 grid cells.
        """

        this_input_matrix = copy.deepcopy(POLYLINE_AS_BINARY_MATRIX)
        this_dilated_matrix = front_utils._dilate_binary_image(
            binary_matrix=this_input_matrix,
            dilation_half_width_in_grid_cells=2)

        self.assertTrue(numpy.array_equal(
            this_dilated_matrix, BINARY_MATRIX_DILATED_HALFWIDTH2))

    def test_binary_image_to_grid_points(self):
        """Ensures correct output from _binary_image_to_grid_points."""

        these_rows, these_columns = front_utils._binary_image_to_grid_points(
            BINARY_MATRIX_DILATED_HALFWIDTH1)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_POLYLINE_DILATED_HALFWIDTH1))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_POLYLINE_DILATED_HALFWIDTH1))


if __name__ == '__main__':
    unittest.main()
