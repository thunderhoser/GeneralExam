"""Unit tests for front_utils.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6

NARR_GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME
)[0]

# The following constants are used to test _vertex_arrays_to_list.
VERTEX_X_COORDS_METRES = numpy.array(
    [3.3, 2.3, 1.8, 1.5, 1.3, 1.4, 1.7, 2.1, 2.6, 2.8]
)
VERTEX_Y_COORDS_METRES = numpy.array(
    [5.5, 4.7, 4.1, 3.5, 2.5, 1.5, 0.9, 0.1, -0.2, -0.5]
)
LIST_OF_XY_TUPLES_METRES = [
    (3.3, 5.5), (2.3, 4.7), (1.8, 4.1), (1.5, 3.5), (1.3, 2.5),
    (1.4, 1.5), (1.7, 0.9), (2.1, 0.1), (2.6, -0.2), (2.8, -0.5)
]

# The following constants are used to test _grid_cell_to_polygon.
X_SPACING_METRES = 1.
Y_SPACING_METRES = 1.
ONE_GRID_POINT_X_METRES = 1.
ONE_GRID_POINT_Y_METRES = 2.
GRID_CELL_EDGE_COORDS_X_METRES = numpy.array([0.5, 1.5, 1.5, 0.5, 0.5])
GRID_CELL_EDGE_COORDS_Y_METRES = numpy.array([1.5, 1.5, 2.5, 2.5, 1.5])

# The following constants are used to test _polyline_to_grid_points.
NUM_GRID_ROWS = 6
NUM_GRID_COLUMNS = 8
GRID_POINT_X_COORDS_METRES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
GRID_POINT_Y_COORDS_METRES = numpy.array([0, 1, 2, 3, 4, 5], dtype=float)

ROWS_IN_POLYLINE = numpy.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5], dtype=int)
COLUMNS_IN_POLYLINE = numpy.array([2, 3, 1, 2, 1, 1, 2, 1, 2, 2, 3], dtype=int)

# The following constants are used to test _grid_points_to_boolean_matrix and
# _boolean_matrix_to_grid_points.
BINARY_MATRIX_UNDILATED = numpy.array(
    [[0, 0, 1, 1, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0, 0, 0]], dtype=int)

BINARY_MATRIX_DILATED = numpy.array(
    [[0, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 0, 0, 0]], dtype=int)

ROWS_IN_DILATED_POLYLINE = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
    dtype=int)
COLUMNS_IN_DILATED_POLYLINE = numpy.array(
    [1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4],
    dtype=int)

# The following constants are used to test _is_polyline_closed.
CLOSED_VERTEX_LATITUDES_DEG = numpy.array([51.1, 53.5, 53.5, 51.1])
CLOSED_VERTEX_LONGITUDES_DEG = numpy.array([246, 246.5, 246, 246])
OPEN_VERTEX_LATITUDES_DEG = VERTEX_Y_COORDS_METRES + 0.
OPEN_VERTEX_LONGITUDES_DEG = VERTEX_X_COORDS_METRES + 0.

# The following constants are used to test buffer_distance_to_dilation_mask.
SMALL_BUFFER_DISTANCE_METRES = 1.
SMALL_BUFFER_MASK_MATRIX = numpy.array([[1]], dtype=bool)

LARGE_BUFFER_DISTANCE_METRES = float(1e5)
LARGE_BUFFER_MASK_MATRIX = numpy.array(
    [[0, 0, 0, 1, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 1, 0, 0, 0]], dtype=bool)

# The following constants are used to test gridded_labels_to_points,
# grid_points_to_frontal_image, and gridded_labels_to_regions.
TERNARY_LABEL_MATRIX = numpy.array(
    [[0, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0],
     [2, 1, 1, 0, 0, 0, 0, 0],
     [2, 2, 2, 2, 0, 0, 0, 0],
     [2, 2, 2, 2, 0, 0, 0, 0],
     [0, 2, 2, 2, 2, 0, 0, 0]], dtype=int)

NUM_FRONTS = 2
WARM_FRONT_ROW_INDICES = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=int)
WARM_FRONT_COLUMN_INDICES = numpy.array(
    [1, 2, 3, 4, 0, 1, 2, 3, 1, 2], dtype=int)
COLD_FRONT_ROW_INDICES = numpy.array(
    [2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], dtype=int)
COLD_FRONT_COLUMN_INDICES = numpy.array(
    [0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4], dtype=int)

REGION_MAJOR_AXIS_LENGTHS_PX = numpy.array([4.750431, 5.535734])

GRIDDED_LABEL_DICT = {
    front_utils.WARM_FRONT_ROWS_COLUMN: WARM_FRONT_ROW_INDICES,
    front_utils.WARM_FRONT_COLUMNS_COLUMN: WARM_FRONT_COLUMN_INDICES,
    front_utils.COLD_FRONT_ROWS_COLUMN: COLD_FRONT_ROW_INDICES,
    front_utils.COLD_FRONT_COLUMNS_COLUMN: COLD_FRONT_COLUMN_INDICES
}

# The following constants are used to test dilate_binary_label_matrix.
DILATION_DISTANCE_METRES = float(1e5)
BINARY_NARR_MATRIX_UNDILATED = numpy.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=int)

BINARY_NARR_MATRIX_DILATED = numpy.array(
    [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=int)

# The following constants are used to test dilate_ternary_label_matrix.
TERNARY_NARR_MATRIX_UNDILATED = numpy.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]], dtype=int)

TERNARY_NARR_MATRIX_CF_TIEBREAKER = numpy.array(
    [[1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
     [1, 1, 1, 2, 2, 2, 2, 1, 0, 0],
     [1, 1, 2, 2, 2, 2, 1, 1, 0, 0],
     [0, 1, 2, 2, 2, 1, 1, 1, 2, 0],
     [0, 0, 2, 2, 1, 1, 1, 2, 2, 2],
     [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
     [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]], dtype=int)

TERNARY_NARR_MATRIX_WF_TIEBREAKER = numpy.array(
    [[1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 2, 2, 2, 0, 0, 0],
     [1, 1, 1, 2, 2, 2, 1, 1, 0, 0],
     [1, 1, 2, 2, 2, 1, 1, 1, 0, 0],
     [0, 1, 2, 2, 1, 1, 1, 1, 1, 0],
     [0, 0, 2, 1, 1, 1, 1, 1, 2, 2],
     [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
     [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]], dtype=int)

# The following constants are used to test close_binary_label_matrix.
SMALL_CLOSING_DISTANCE_METRES = 50000.
LARGE_CLOSING_DISTANCE_METRES = 110000.

BINARY_MATRIX_UNCLOSED = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

BINARY_MATRIX_CLOSED_SMALL_BUFFER = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

BINARY_MATRIX_CLOSED_LARGE_BUFFER = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

# The following constants are used to test close_ternary_label_matrix.
TERNARY_MATRIX_UNCLOSED = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

TERNARY_MATRIX_CLOSED_SMALL_BUFFER = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

TERNARY_MATRIX_CLOSED_LARGE_BUFFER = numpy.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

# The following constants are used to test remove_fronts_in_masked_area.
THESE_STRINGS = [
    front_utils.WARM_FRONT_STRING, front_utils.COLD_FRONT_STRING,
    front_utils.COLD_FRONT_STRING, front_utils.WARM_FRONT_STRING
]

THESE_TIMES_UNIX_SEC = numpy.array([0, 1, 2, 3], dtype=int)

FIRST_LATITUDES_DEG = numpy.array([50, 80], dtype=float)
SECOND_LATITUDES_DEG = numpy.array([51.1, 53.5])
THIRD_LATITUDES_DEG = numpy.array([51.1, 53.5])
FOURTH_LATITUDES_DEG = numpy.array([77, 82.5])

FIRST_LONGITUDES_DEG = numpy.array([246, 246], dtype=float)
SECOND_LONGITUDES_DEG = numpy.array([246, 246.5])
THIRD_LONGITUDES_DEG = numpy.array([246, 340], dtype=float)
FOURTH_LONGITUDES_DEG = numpy.array([246, 246.5])

THIS_DICT = {
    front_utils.LATITUDES_COLUMN: [
        FIRST_LATITUDES_DEG, SECOND_LATITUDES_DEG, THIRD_LATITUDES_DEG,
        FOURTH_LATITUDES_DEG
    ],
    front_utils.LONGITUDES_COLUMN: [
        FIRST_LONGITUDES_DEG, SECOND_LONGITUDES_DEG, THIRD_LONGITUDES_DEG,
        FOURTH_LONGITUDES_DEG
    ],
    front_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    front_utils.FRONT_TYPE_COLUMN: THESE_STRINGS
}

POLYLINE_TABLE_UNMASKED = pandas.DataFrame.from_dict(THIS_DICT)

THIS_LATITUDE_MATRIX_DEG, THIS_LONGITUDE_MATRIX_DEG = (
    nwp_model_utils.get_latlng_grid_point_matrices(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
)

THIS_LATITUDE_FLAG_MATRIX = numpy.logical_and(
    THIS_LATITUDE_MATRIX_DEG >= 30, THIS_LATITUDE_MATRIX_DEG <= 75)
THIS_LONGITUDE_FLAG_MATRIX = numpy.logical_and(
    THIS_LONGITUDE_MATRIX_DEG >= 220, THIS_LONGITUDE_MATRIX_DEG <= 320)
NARR_MASK_MATRIX = numpy.logical_and(
    THIS_LATITUDE_FLAG_MATRIX, THIS_LONGITUDE_FLAG_MATRIX
).astype(int)

THIS_DICT = {
    front_utils.LATITUDES_COLUMN:
        [FIRST_LATITUDES_DEG, SECOND_LATITUDES_DEG, THIRD_LATITUDES_DEG],
    front_utils.LONGITUDES_COLUMN:
        [FIRST_LONGITUDES_DEG, SECOND_LONGITUDES_DEG, THIRD_LONGITUDES_DEG],
    front_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC[:-1],
    front_utils.FRONT_TYPE_COLUMN: THESE_STRINGS[:-1]
}

POLYLINE_TABLE_MASKED = pandas.DataFrame.from_dict(THIS_DICT)


def _compare_polyline_tables(first_polyline_table, second_polyline_table):
    """Compares two tables (pandas DataFrames) with fronts as polylines.

    :param first_polyline_table: First table.
    :param second_polyline_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_columns = list(first_polyline_table)
    second_columns = list(second_polyline_table)
    if set(first_columns) != set(second_columns):
        return False

    first_num_fronts = len(first_polyline_table.index)
    second_num_fronts = len(second_polyline_table.index)
    if first_num_fronts != second_num_fronts:
        return False

    for this_column in first_columns:
        if this_column in [front_utils.LATITUDES_COLUMN,
                           front_utils.LONGITUDES_COLUMN]:
            for i in range(first_num_fronts):
                if not numpy.allclose(
                        first_polyline_table[this_column].values[i],
                        second_polyline_table[this_column].values[i],
                        atol=TOLERANCE):
                    return False
        else:
            if not numpy.array_equal(first_polyline_table[this_column].values,
                                     second_polyline_table[this_column].values):
                return False

    return True


class FrontUtilsTests(unittest.TestCase):
    """Each method is a unit test for front_utils.py."""

    def test_vertex_arrays_to_list(self):
        """Ensures correct output from _vertex_arrays_to_list."""

        this_vertex_list_xy_metres = front_utils._vertex_arrays_to_list(
            x_coords_metres=VERTEX_X_COORDS_METRES,
            y_coords_metres=VERTEX_Y_COORDS_METRES)

        this_num_vertices = len(this_vertex_list_xy_metres)
        expected_num_vertices = len(LIST_OF_XY_TUPLES_METRES)
        self.assertTrue(this_num_vertices == expected_num_vertices)

        for i in range(this_num_vertices):
            self.assertTrue(numpy.allclose(
                this_vertex_list_xy_metres[i],
                LIST_OF_XY_TUPLES_METRES[i], atol=TOLERANCE
            ))

    def test_grid_cell_to_polygon(self):
        """Ensures correct output from _grid_cell_to_polygon."""

        this_polygon_object_xy_metres = front_utils._grid_cell_to_polygon(
            grid_point_x_metres=ONE_GRID_POINT_X_METRES,
            grid_point_y_metres=ONE_GRID_POINT_Y_METRES,
            x_spacing_metres=X_SPACING_METRES,
            y_spacing_metres=Y_SPACING_METRES)

        these_vertex_x_metres = numpy.array(
            this_polygon_object_xy_metres.exterior.xy[0]
        )
        these_vertex_y_metres = numpy.array(
            this_polygon_object_xy_metres.exterior.xy[1]
        )

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, GRID_CELL_EDGE_COORDS_X_METRES,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, GRID_CELL_EDGE_COORDS_Y_METRES,
            atol=TOLERANCE
        ))

    def test_polyline_to_grid_points(self):
        """Ensures correct output from _polyline_to_grid_points."""

        these_rows, these_columns = front_utils._polyline_to_grid_points(
            polyline_x_coords_metres=VERTEX_X_COORDS_METRES,
            polyline_y_coords_metres=VERTEX_Y_COORDS_METRES,
            grid_point_x_coords_metres=GRID_POINT_X_COORDS_METRES,
            grid_point_y_coords_metres=GRID_POINT_Y_COORDS_METRES)

        self.assertTrue(numpy.array_equal(these_rows, ROWS_IN_POLYLINE))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_POLYLINE
        ))

    def test_grid_points_to_boolean_matrix(self):
        """Ensures correct output from _grid_points_to_boolean_matrix."""

        this_binary_image_matrix = front_utils._grid_points_to_boolean_matrix(
            rows_in_front=ROWS_IN_POLYLINE,
            columns_in_front=COLUMNS_IN_POLYLINE,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_MATRIX_UNDILATED
        ))

    def test_boolean_matrix_to_grid_points(self):
        """Ensures correct output from _boolean_matrix_to_grid_points."""

        these_rows, these_columns = front_utils._boolean_matrix_to_grid_points(
            BINARY_MATRIX_DILATED)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_DILATED_POLYLINE
        ))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_DILATED_POLYLINE
        ))

    def test_is_polyline_closed_yes(self):
        """Ensures correct output from _is_polyline_closed."""

        self.assertTrue(front_utils._is_polyline_closed(
            x_coords_metres=CLOSED_VERTEX_LONGITUDES_DEG,
            y_coords_metres=CLOSED_VERTEX_LATITUDES_DEG
        ))

    def test_is_polyline_closed_no(self):
        """Ensures correct output from _is_polyline_closed."""

        self.assertFalse(front_utils._is_polyline_closed(
            x_coords_metres=OPEN_VERTEX_LONGITUDES_DEG,
            y_coords_metres=OPEN_VERTEX_LATITUDES_DEG
        ))

    def test_buffer_distance_to_dilation_mask_small(self):
        """Ensures correct output from buffer_distance_to_dilation_mask."""

        this_mask_matrix = front_utils.buffer_distance_to_dilation_mask(
            buffer_distance_metres=SMALL_BUFFER_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, SMALL_BUFFER_MASK_MATRIX
        ))

    def test_buffer_distance_to_dilation_mask_large(self):
        """Ensures correct output from buffer_distance_to_dilation_mask."""

        this_mask_matrix = front_utils.buffer_distance_to_dilation_mask(
            buffer_distance_metres=LARGE_BUFFER_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, LARGE_BUFFER_MASK_MATRIX
        ))

    def test_dilate_binary_label_matrix(self):
        """Ensures correct output from dilate_binary_label_matrix."""

        this_binary_image_matrix = front_utils.dilate_binary_label_matrix(
            binary_label_matrix=BINARY_NARR_MATRIX_UNDILATED + 0,
            dilation_distance_metres=DILATION_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_binary_image_matrix, BINARY_NARR_MATRIX_DILATED
        ))

    def test_dilate_ternary_label_matrix_cf_tiebreaker(self):
        """Ensures correct output from dilate_ternary_label_matrix.

        In this case the tiebreaker is the CF label.
        """

        this_ternary_image_matrix = front_utils.dilate_ternary_label_matrix(
            ternary_label_matrix=TERNARY_NARR_MATRIX_UNDILATED + 0,
            dilation_distance_metres=DILATION_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES,
            tiebreaker_enum=front_utils.COLD_FRONT_ENUM)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_NARR_MATRIX_CF_TIEBREAKER
        ))

    def test_dilate_ternary_label_matrix_wf_tiebreaker(self):
        """Ensures correct output from dilate_ternary_label_matrix.

        In this case the tiebreaker is the WF label.
        """

        this_ternary_image_matrix = front_utils.dilate_ternary_label_matrix(
            ternary_label_matrix=TERNARY_NARR_MATRIX_UNDILATED + 0,
            dilation_distance_metres=DILATION_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES,
            tiebreaker_enum=front_utils.WARM_FRONT_ENUM)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_NARR_MATRIX_WF_TIEBREAKER
        ))

    def test_close_binary_label_matrix_small_dist(self):
        """Ensures correct output from close_binary_label_matrix.

        In this case the buffer distance is small.
        """

        this_label_matrix = front_utils.close_binary_label_matrix(
            binary_label_matrix=BINARY_MATRIX_UNCLOSED + 0,
            buffer_distance_metres=SMALL_CLOSING_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, BINARY_MATRIX_CLOSED_SMALL_BUFFER
        ))

    def test_close_binary_label_matrix_large_dist(self):
        """Ensures correct output from close_binary_label_matrix.

        In this case the buffer distance is large.
        """

        this_label_matrix = front_utils.close_binary_label_matrix(
            binary_label_matrix=BINARY_MATRIX_UNCLOSED + 0,
            buffer_distance_metres=LARGE_CLOSING_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, BINARY_MATRIX_CLOSED_LARGE_BUFFER
        ))

    def test_close_ternary_label_matrix_small_dist(self):
        """Ensures correct output from close_ternary_label_matrix.

        In this case the buffer distance is small.
        """

        this_label_matrix = front_utils.close_ternary_label_matrix(
            ternary_label_matrix=TERNARY_MATRIX_UNCLOSED + 0,
            buffer_distance_metres=SMALL_CLOSING_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, TERNARY_MATRIX_CLOSED_SMALL_BUFFER
        ))

    def test_close_ternary_label_matrix_large_dist(self):
        """Ensures correct output from close_ternary_label_matrix.

        In this case the buffer distance is large.
        """

        this_label_matrix = front_utils.close_ternary_label_matrix(
            ternary_label_matrix=TERNARY_MATRIX_UNCLOSED + 0,
            buffer_distance_metres=LARGE_CLOSING_DISTANCE_METRES,
            grid_spacing_metres=NARR_GRID_SPACING_METRES)

        self.assertTrue(numpy.array_equal(
            this_label_matrix, TERNARY_MATRIX_CLOSED_LARGE_BUFFER
        ))

    def test_gridded_labels_to_points(self):
        """Ensures correct output from gridded_labels_to_points."""

        this_gridded_label_dict = front_utils.gridded_labels_to_points(
            TERNARY_LABEL_MATRIX)

        these_keys = this_gridded_label_dict.keys()
        expected_keys = GRIDDED_LABEL_DICT.keys()
        self.assertTrue(set(these_keys) == set(expected_keys))

        for this_key in GRIDDED_LABEL_DICT.keys():
            self.assertTrue(numpy.array_equal(
                this_gridded_label_dict[this_key],
                GRIDDED_LABEL_DICT[this_key]
            ))

    def test_points_to_gridded_labels(self):
        """Ensures correct output from points_to_gridded_labels."""

        this_ternary_image_matrix = front_utils.points_to_gridded_labels(
            gridded_label_dict=GRIDDED_LABEL_DICT,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_LABEL_MATRIX
        ))

    def test_gridded_labels_to_regions(self):
        """Ensures correct output from gridded_labels_to_regions."""

        this_region_dict = front_utils.gridded_labels_to_regions(
            ternary_label_matrix=TERNARY_LABEL_MATRIX + 0, compute_lengths=True)

        self.assertTrue(numpy.allclose(
            this_region_dict[front_utils.MAJOR_AXIS_LENGTHS_KEY],
            REGION_MAJOR_AXIS_LENGTHS_PX, atol=TOLERANCE
        ))

        this_num_fronts = len(this_region_dict[front_utils.FRONT_TYPES_KEY])
        self.assertTrue(this_num_fronts == NUM_FRONTS)

        for i in range(NUM_FRONTS):
            this_front_type_string = this_region_dict[
                front_utils.FRONT_TYPES_KEY][i]

            if this_front_type_string == front_utils.WARM_FRONT_STRING:
                self.assertTrue(numpy.array_equal(
                    this_region_dict[front_utils.ROWS_BY_REGION_KEY][i],
                    WARM_FRONT_ROW_INDICES
                ))
                self.assertTrue(numpy.array_equal(
                    this_region_dict[
                        front_utils.COLUMNS_BY_REGION_KEY][i],
                    WARM_FRONT_COLUMN_INDICES
                ))

            else:
                self.assertTrue(numpy.array_equal(
                    this_region_dict[front_utils.ROWS_BY_REGION_KEY][i],
                    COLD_FRONT_ROW_INDICES
                ))
                self.assertTrue(numpy.array_equal(
                    this_region_dict[
                        front_utils.COLUMNS_BY_REGION_KEY][i],
                    COLD_FRONT_COLUMN_INDICES
                ))

    def test_remove_fronts_in_masked_area(self):
        """Ensures correct output from remove_fronts_in_masked_area."""

        this_polyline_table = front_utils.remove_fronts_in_masked_area(
            polyline_table=copy.deepcopy(POLYLINE_TABLE_UNMASKED),
            narr_mask_matrix=NARR_MASK_MATRIX)

        self.assertTrue(_compare_polyline_tables(
            POLYLINE_TABLE_MASKED, this_polyline_table
        ))


if __name__ == '__main__':
    unittest.main()
