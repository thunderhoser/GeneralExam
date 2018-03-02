"""Unit tests for front_utils.py."""

import copy
import unittest
import numpy
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6

# The following constants are used to test _vertex_arrays_to_list.
POLYLINE_X_COORDS_METRES = numpy.array(
    [3.3, 2.3, 1.8, 1.5, 1.3, 1.4, 1.7, 2.1, 2.6, 2.8])
POLYLINE_Y_COORDS_METRES = numpy.array(
    [5.5, 4.7, 4.1, 3.5, 2.5, 1.5, 0.9, 0.1, -0.2, -0.5])
POLYLINE_VERTEX_LIST_XY_METRES = [
    (3.3, 5.5), (2.3, 4.7), (1.8, 4.1), (1.5, 3.5), (1.3, 2.5),
    (1.4, 1.5), (1.7, 0.9), (2.1, 0.1), (2.6, -0.2), (2.8, -0.5)]

# The following constants are used to test _grid_cell_to_polygon.
GRID_SPACING_X_METRES = 1.
GRID_SPACING_Y_METRES = 1.
ONE_GRID_POINT_X_METRES = 1.
ONE_GRID_POINT_Y_METRES = 2.
ONE_GRID_POINT_VERTEX_COORDS_X_METRES = numpy.array([0.5, 1.5, 1.5, 0.5, 0.5])
ONE_GRID_POINT_VERTEX_COORDS_Y_METRES = numpy.array([1.5, 1.5, 2.5, 2.5, 1.5])

# The following constants are used to test _polyline_to_grid_points.
NUM_GRID_ROWS = 6
NUM_GRID_COLUMNS = 8
GRID_POINT_X_COORDS_METRES = numpy.array([0., 1., 2., 3., 4., 5., 6., 7.])
GRID_POINT_Y_COORDS_METRES = numpy.array([0., 1., 2., 3., 4., 5.])

GRID_ROWS_IN_POLYLINE = numpy.array(
    [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5], dtype=int)
GRID_COLUMNS_IN_POLYLINE = numpy.array(
    [2, 3, 1, 2, 1, 1, 2, 1, 2, 2, 3], dtype=int)

# The following constants are used to test _grid_points_to_binary_image and
# _binary_image_to_grid_points.
BINARY_IMAGE_MATRIX_UNDILATED = numpy.array([[0, 0, 1, 1, 0, 0, 0, 0],
                                             [0, 1, 1, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 1, 0, 0, 0, 0]])

BINARY_IMAGE_MATRIX_DILATED = numpy.array([[0, 1, 1, 1, 1, 0, 0, 0],
                                           [1, 1, 1, 1, 0, 0, 0, 0],
                                           [1, 1, 1, 0, 0, 0, 0, 0],
                                           [1, 1, 1, 1, 0, 0, 0, 0],
                                           [1, 1, 1, 1, 0, 0, 0, 0],
                                           [0, 1, 1, 1, 1, 0, 0, 0]])

ROWS_IN_DILATED_POLYLINE = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
    dtype=int)
COLUMNS_IN_DILATED_POLYLINE = numpy.array(
    [1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4],
    dtype=int)

# The following constants are used to test _is_polyline_closed.
CLOSED_POLYLINE_LATITUDES_DEG = numpy.array([51.1, 53.5, 53.5, 51.1])
CLOSED_POLYLINE_LONGITUDES_DEG = numpy.array([246., 246.5, 246., 246.])
OPEN_POLYLINE_LATITUDES_DEG = copy.deepcopy(POLYLINE_Y_COORDS_METRES)
OPEN_POLYLINE_LONGITUDES_DEG = copy.deepcopy(POLYLINE_X_COORDS_METRES)

# The following constants are used to test _close_frontal_image.
TERNARY_IMAGE_MATRIX_UNCLOSED = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 1, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 1, 1],
                                             [0, 0, 2, 2, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0],
                                             [2, 2, 2, 0, 0, 0, 0, 0]])

TERNARY_IMAGE_MATRIX_CLOSED = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 1, 1, 1],
                                           [0, 0, 0, 0, 0, 0, 1, 1],
                                           [0, 0, 2, 2, 0, 0, 0, 0],
                                           [0, 0, 2, 0, 0, 0, 0, 0],
                                           [2, 2, 2, 0, 0, 0, 0, 0]])

# The following constants are used to test buffer_distance_to_narr_mask.
SMALL_BUFFER_DISTANCE_METRES = 1.
MASK_MATRIX_FOR_SMALL_BUFFER = numpy.array([[1]], dtype=bool)

LARGE_BUFFER_DISTANCE_METRES = float(1e5)
MASK_MATRIX_FOR_LARGE_BUFFER = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                                            [0, 1, 1, 1, 1, 1, 0],
                                            [0, 1, 1, 1, 1, 1, 0],
                                            [1, 1, 1, 1, 1, 1, 1],
                                            [0, 1, 1, 1, 1, 1, 0],
                                            [0, 1, 1, 1, 1, 1, 0],
                                            [0, 0, 0, 1, 0, 0, 0]],
                                           dtype=bool)

# The following constants are used to test frontal_image_to_grid_points,
# grid_points_to_frontal_image, and frontal_image_to_objects.
TERNARY_IMAGE_MATRIX = numpy.array([[0, 1, 1, 1, 1, 0, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0],
                                    [2, 1, 1, 0, 0, 0, 0, 0],
                                    [2, 2, 2, 2, 0, 0, 0, 0],
                                    [2, 2, 2, 2, 0, 0, 0, 0],
                                    [0, 2, 2, 2, 2, 0, 0, 0]])

NUM_FRONTS = 2
WARM_FRONT_ROW_INDICES = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=int)
WARM_FRONT_COLUMN_INDICES = numpy.array(
    [1, 2, 3, 4, 0, 1, 2, 3, 1, 2], dtype=int)
COLD_FRONT_ROW_INDICES = numpy.array(
    [2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], dtype=int)
COLD_FRONT_COLUMN_INDICES = numpy.array(
    [0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4], dtype=int)

FRONTAL_GRID_POINT_DICT = {
    front_utils.WARM_FRONT_ROW_INDICES_COLUMN: WARM_FRONT_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN: WARM_FRONT_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROW_INDICES_COLUMN: COLD_FRONT_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN: COLD_FRONT_COLUMN_INDICES
}

# The following constants are used to test dilate_binary_narr_image.
DILATION_DISTANCE_METRES = float(1e5)
BINARY_NARR_MATRIX_UNDILATED = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                                           dtype=int)

BINARY_NARR_MATRIX_DILATED = numpy.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
                                         dtype=int)

# The following constants are used to test dilate_ternary_narr_image.
TERNARY_NARR_MATRIX_UNDILATED = numpy.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]], dtype=int)
TERNARY_NARR_MATRIX_DILATED = numpy.array(
    [[1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
     [1, 1, 1, 2, 2, 2, 2, 1, 0, 0],
     [1, 1, 2, 2, 2, 2, 1, 1, 0, 0],
     [0, 1, 2, 2, 2, 1, 1, 1, 2, 0],
     [0, 0, 2, 2, 1, 1, 1, 2, 2, 2],
     [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
     [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]], dtype=int)


class FrontUtilsTests(unittest.TestCase):
    """Each method is a unit test for front_utils.py."""

    def test_vertex_arrays_to_list(self):
        """Ensures correct output from _vertex_arrays_to_list."""

        this_vertex_list_xy_metres = front_utils._vertex_arrays_to_list(
            x_coords_metres=POLYLINE_X_COORDS_METRES,
            y_coords_metres=POLYLINE_Y_COORDS_METRES)

        this_num_vertices = len(this_vertex_list_xy_metres)
        expected_num_vertices = len(POLYLINE_VERTEX_LIST_XY_METRES)
        self.assertTrue(this_num_vertices == expected_num_vertices)

        for i in range(this_num_vertices):
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
            these_vertex_x_metres, ONE_GRID_POINT_VERTEX_COORDS_X_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, ONE_GRID_POINT_VERTEX_COORDS_Y_METRES,
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

        this_binary_image_matrix = front_utils._grid_points_to_binary_image(
            rows_in_object=GRID_ROWS_IN_POLYLINE,
            columns_in_object=GRID_COLUMNS_IN_POLYLINE,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_IMAGE_MATRIX_UNDILATED))

    def test_binary_image_to_grid_points(self):
        """Ensures correct output from _binary_image_to_grid_points."""

        these_rows, these_columns = front_utils._binary_image_to_grid_points(
            BINARY_IMAGE_MATRIX_DILATED)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_DILATED_POLYLINE))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_DILATED_POLYLINE))

    def test_is_polyline_closed_yes(self):
        """Ensures correct output from _is_polyline_closed."""

        self.assertTrue(front_utils._is_polyline_closed(
            latitudes_deg=CLOSED_POLYLINE_LATITUDES_DEG,
            longitudes_deg=CLOSED_POLYLINE_LONGITUDES_DEG))

    def test_is_polyline_closed_no(self):
        """Ensures correct output from _is_polyline_closed."""

        self.assertFalse(front_utils._is_polyline_closed(
            latitudes_deg=OPEN_POLYLINE_LATITUDES_DEG,
            longitudes_deg=OPEN_POLYLINE_LONGITUDES_DEG))

    def test_close_frontal_image(self):
        """Ensures correct output from _close_frontal_image."""

        this_input_matrix = copy.deepcopy(TERNARY_IMAGE_MATRIX_UNCLOSED)
        this_closed_matrix = front_utils._close_frontal_image(this_input_matrix)

        self.assertTrue(numpy.array_equal(
            this_closed_matrix, TERNARY_IMAGE_MATRIX_CLOSED))

    def test_buffer_distance_to_narr_mask_small(self):
        """Ensures correct output from buffer_distance_to_narr_mask."""

        this_mask_matrix = front_utils.buffer_distance_to_narr_mask(
            SMALL_BUFFER_DISTANCE_METRES)
        self.assertTrue(numpy.array_equal(
            this_mask_matrix, MASK_MATRIX_FOR_SMALL_BUFFER))

    def test_buffer_distance_to_narr_mask_large(self):
        """Ensures correct output from buffer_distance_to_narr_mask."""

        this_mask_matrix = front_utils.buffer_distance_to_narr_mask(
            LARGE_BUFFER_DISTANCE_METRES)
        self.assertTrue(numpy.array_equal(
            this_mask_matrix, MASK_MATRIX_FOR_LARGE_BUFFER))

    def test_dilate_binary_narr_image(self):
        """Ensures correct output from dilate_binary_narr_image."""

        this_input_matrix = copy.deepcopy(BINARY_NARR_MATRIX_UNDILATED)
        this_binary_image_matrix = front_utils.dilate_binary_narr_image(
            binary_image_matrix=this_input_matrix,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_NARR_MATRIX_DILATED))

    def test_dilate_ternary_narr_image(self):
        """Ensures correct output from dilate_ternary_narr_image."""

        this_input_matrix = copy.deepcopy(TERNARY_NARR_MATRIX_UNDILATED)
        this_ternary_image_matrix = front_utils.dilate_ternary_narr_image(
            ternary_image_matrix=this_input_matrix,
            dilation_distance_metres=DILATION_DISTANCE_METRES)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_NARR_MATRIX_DILATED))

    def test_frontal_image_to_grid_points(self):
        """Ensures correct output from frontal_image_to_grid_points."""

        this_grid_point_dict = front_utils.frontal_image_to_grid_points(
            TERNARY_IMAGE_MATRIX)

        self.assertTrue(set(this_grid_point_dict.keys()) ==
                        set(FRONTAL_GRID_POINT_DICT.keys()))

        for this_key in FRONTAL_GRID_POINT_DICT.keys():
            self.assertTrue(numpy.array_equal(
                this_grid_point_dict[this_key], FRONTAL_GRID_POINT_DICT[this_key]))

    def test_frontal_grid_points_to_image(self):
        """Ensures correct output from frontal_grid_points_to_image."""

        this_ternary_image_matrix = front_utils.frontal_grid_points_to_image(
            frontal_grid_point_dict=FRONTAL_GRID_POINT_DICT,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_IMAGE_MATRIX))

    def test_frontal_image_to_objects(self):
        """Ensures correct output from frontal_image_to_objects."""

        this_input_matrix = copy.deepcopy(TERNARY_IMAGE_MATRIX)
        this_frontal_region_dict = front_utils.frontal_image_to_objects(
            this_input_matrix)

        this_num_fronts = len(
            this_frontal_region_dict[front_utils.FRONT_TYPE_BY_REGION_KEY])
        self.assertTrue(this_num_fronts == NUM_FRONTS)

        for i in range(NUM_FRONTS):
            this_front_type_string = this_frontal_region_dict[
                front_utils.FRONT_TYPE_BY_REGION_KEY][i]

            if this_front_type_string == front_utils.WARM_FRONT_STRING_ID:
                self.assertTrue(numpy.array_equal(
                    this_frontal_region_dict[
                        front_utils.ROW_INDICES_BY_REGION_KEY][i],
                    WARM_FRONT_ROW_INDICES))
                self.assertTrue(numpy.array_equal(
                    this_frontal_region_dict[
                        front_utils.COLUMN_INDICES_BY_REGION_KEY][i],
                    WARM_FRONT_COLUMN_INDICES))

            else:
                self.assertTrue(numpy.array_equal(
                    this_frontal_region_dict[
                        front_utils.ROW_INDICES_BY_REGION_KEY][i],
                    COLD_FRONT_ROW_INDICES))
                self.assertTrue(numpy.array_equal(
                    this_frontal_region_dict[
                        front_utils.COLUMN_INDICES_BY_REGION_KEY][i],
                    COLD_FRONT_COLUMN_INDICES))


if __name__ == '__main__':
    unittest.main()
