"""Methods for handling atmospheric fronts.

A front may be represented as either a polyline or a set of grid points.
"""

import numpy
import pandas
import cv2
import shapely.geometry
from scipy.ndimage.morphology import binary_closing
from skimage.measure import label as label_image
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE_DEG = 1e-3
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'

STRUCTURE_MATRIX_FOR_BINARY_CLOSING = numpy.ones((3, 3))

FRONT_TYPE_COLUMN = 'front_type'
TIME_COLUMN = 'unix_time_sec'
LATITUDES_COLUMN = 'latitudes_deg'
LONGITUDES_COLUMN = 'longitudes_deg'

WARM_FRONT_ROW_INDICES_COLUMN = 'warm_front_row_indices'
WARM_FRONT_COLUMN_INDICES_COLUMN = 'warm_front_column_indices'
COLD_FRONT_ROW_INDICES_COLUMN = 'cold_front_row_indices'
COLD_FRONT_COLUMN_INDICES_COLUMN = 'cold_front_column_indices'

ROW_INDICES_BY_REGION_KEY = 'row_indices_by_region'
COLUMN_INDICES_BY_REGION_KEY = 'column_indices_by_region'
FRONT_TYPE_BY_REGION_KEY = 'front_type_by_region'

NO_FRONT_INTEGER_ID = 0
ANY_FRONT_INTEGER_ID = 1
WARM_FRONT_INTEGER_ID = 1
COLD_FRONT_INTEGER_ID = 2
VALID_INTEGER_IDS = [
    NO_FRONT_INTEGER_ID, ANY_FRONT_INTEGER_ID, WARM_FRONT_INTEGER_ID,
    COLD_FRONT_INTEGER_ID]

WARM_FRONT_STRING_ID = 'warm'
COLD_FRONT_STRING_ID = 'cold'
VALID_STRING_IDS = [WARM_FRONT_STRING_ID, COLD_FRONT_STRING_ID]


def _check_polyline(x_coords_metres, y_coords_metres):
    """Checks polyline for errors.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    """

    error_checking.assert_is_numpy_array_without_nan(x_coords_metres)
    error_checking.assert_is_numpy_array(x_coords_metres, num_dimensions=1)
    num_vertices = len(x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(y_coords_metres)
    error_checking.assert_is_numpy_array(
        y_coords_metres, exact_dimensions=numpy.array([num_vertices]))


def _check_frontal_image(image_matrix, assert_binary=False):
    """Checks frontal image for errors.

    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)

    :param image_matrix: M-by-N numpy array of integers.  May be either binary
        (2-class) or ternary (3-class).  If binary, all elements must be in
        {0, 1} and element [i, j] indicates whether or not a front intersects
        grid cell [i, j].  If ternary, elements must be in `VALID_INTEGER_IDS`
        and element [i, j] indicates the type of front (warm, cold, or none)
        intersecting grid cell [i, j].
    :param assert_binary: Boolean flag.  If True and image is non-binary, this
        method will error out.
    """

    error_checking.assert_is_numpy_array(image_matrix, num_dimensions=2)
    error_checking.assert_is_integer_numpy_array(image_matrix)
    error_checking.assert_is_geq_numpy_array(
        image_matrix, numpy.min(VALID_INTEGER_IDS))

    if assert_binary:
        error_checking.assert_is_leq_numpy_array(
            image_matrix, ANY_FRONT_INTEGER_ID)
    else:
        error_checking.assert_is_leq_numpy_array(
            image_matrix, numpy.max(VALID_INTEGER_IDS))


def _vertex_arrays_to_list(x_coords_metres, y_coords_metres):
    """Converts set of vertices from two arrays to one list.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    :return: vertex_list_xy_metres: length-V list, where each element is an
        (x, y) tuple.
    """

    _check_polyline(
        x_coords_metres=x_coords_metres, y_coords_metres=y_coords_metres)

    num_vertices = len(x_coords_metres)
    vertex_list_xy_metres = []
    for i in range(num_vertices):
        vertex_list_xy_metres.append((x_coords_metres[i], y_coords_metres[i]))

    return vertex_list_xy_metres


def _create_linestring(x_coords_metres, y_coords_metres):
    """Converts polyline from vertex arrays to `shapely.geometry.LineString`.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    :return: linestring_object_xy_metres: `shapely.geometry.LineString` object
        with coordinates in metres.
    """

    vertex_list_xy_metres = _vertex_arrays_to_list(
        x_coords_metres=x_coords_metres, y_coords_metres=y_coords_metres)

    linestring_object_xy_metres = shapely.geometry.LineString(
        vertex_list_xy_metres)
    if not linestring_object_xy_metres.is_valid:
        raise ValueError('Resulting LineString object is invalid.')

    return linestring_object_xy_metres


def _grid_cell_to_polygon(
        grid_point_x_metres, grid_point_y_metres, x_spacing_metres,
        y_spacing_metres):
    """Converts grid cell from center point to polygon.

    This method assumes that the grid has uniform spacing in both x- and y-
    directions.  In other words, the grid is regular in x-y (and not, for
    example, lat-long) coords.

    :param grid_point_x_metres: x-coordinate of center point.
    :param grid_point_y_metres: y-coordinate of center point.
    :param x_spacing_metres: Spacing between adjacent points along x-axis.
    :param y_spacing_metres: Spacing between adjacent points along y-axis.
    :return: polygon_object_xy_metres: `shapely.geometry.Polygon` object, where
        each vertex is a corner of the grid cell.  Coordinates are still in
        metres.
    """

    x_min_metres = grid_point_x_metres - x_spacing_metres / 2
    x_max_metres = grid_point_x_metres + x_spacing_metres / 2
    y_min_metres = grid_point_y_metres - y_spacing_metres / 2
    y_max_metres = grid_point_y_metres + y_spacing_metres / 2

    vertex_x_coords_metres = numpy.array(
        [x_min_metres, x_max_metres, x_max_metres, x_min_metres, x_min_metres])
    vertex_y_coords_metres = numpy.array(
        [y_min_metres, y_min_metres, y_max_metres, y_max_metres, y_min_metres])

    return polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=vertex_x_coords_metres,
        exterior_y_coords=vertex_y_coords_metres)


def _polyline_to_grid_points(
        polyline_x_coords_metres, polyline_y_coords_metres,
        grid_point_x_coords_metres, grid_point_y_coords_metres):
    """Finds grid cells intersected by polyline.

    V = number of vertices in polyline
    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)
    P = number of grid cells intersected by polyline

    This method assumes that `grid_point_x_coords_metres` and
    `grid_point_y_coords_metres` are both equally spaced and sorted in ascending
    order.

    :param polyline_x_coords_metres: length-V numpy array of x-coordinates.
    :param polyline_y_coords_metres: length-V numpy array of y-coordinates.
    :param grid_point_x_coords_metres: length-N numpy array of x-coordinates.
    :param grid_point_y_coords_metres: length-M numpy array of y-coordinates.
    :return: rows_in_polyline: length-P numpy array with row indices (integers)
        of grid cells intersected by polyline.
    :return: columns_in_polyline: Same as above, except for columns.
    """

    polyline_object_xy_metres = _create_linestring(
        x_coords_metres=polyline_x_coords_metres,
        y_coords_metres=polyline_y_coords_metres)

    x_spacing_metres = (
        grid_point_x_coords_metres[1] - grid_point_x_coords_metres[0])
    y_spacing_metres = (
        grid_point_y_coords_metres[1] - grid_point_y_coords_metres[0])

    x_min_to_consider_metres = numpy.min(
        polyline_x_coords_metres) - x_spacing_metres
    x_max_to_consider_metres = numpy.max(
        polyline_x_coords_metres) + x_spacing_metres
    y_min_to_consider_metres = numpy.min(
        polyline_y_coords_metres) - y_spacing_metres
    y_max_to_consider_metres = numpy.max(
        polyline_y_coords_metres) + y_spacing_metres

    x_in_range_indices = numpy.where(numpy.logical_and(
        grid_point_x_coords_metres >= x_min_to_consider_metres,
        grid_point_x_coords_metres <= x_max_to_consider_metres))[0]
    y_in_range_indices = numpy.where(numpy.logical_and(
        grid_point_y_coords_metres >= y_min_to_consider_metres,
        grid_point_y_coords_metres <= y_max_to_consider_metres))[0]

    row_offset = numpy.min(y_in_range_indices)
    column_offset = numpy.min(x_in_range_indices)

    grid_points_x_to_consider_metres = grid_point_x_coords_metres[
        x_in_range_indices]
    grid_points_y_to_consider_metres = grid_point_y_coords_metres[
        y_in_range_indices]

    rows_in_polyline = []
    columns_in_polyline = []
    num_rows_to_consider = len(grid_points_y_to_consider_metres)
    num_columns_to_consider = len(grid_points_x_to_consider_metres)

    for i in range(num_rows_to_consider):
        for j in range(num_columns_to_consider):
            this_grid_cell_edge_polygon_xy_metres = _grid_cell_to_polygon(
                grid_point_x_metres=grid_points_x_to_consider_metres[j],
                grid_point_y_metres=grid_points_y_to_consider_metres[i],
                x_spacing_metres=x_spacing_metres,
                y_spacing_metres=y_spacing_metres)

            this_intersection_flag = (
                this_grid_cell_edge_polygon_xy_metres.intersects(
                    polyline_object_xy_metres) or
                this_grid_cell_edge_polygon_xy_metres.touches(
                    polyline_object_xy_metres))
            if not this_intersection_flag:
                continue

            rows_in_polyline.append(i + row_offset)
            columns_in_polyline.append(j + column_offset)

    return numpy.array(rows_in_polyline), numpy.array(columns_in_polyline)


def _grid_points_to_binary_image(
        rows_in_object, columns_in_object, num_grid_rows, num_grid_columns):
    """Converts list of grid points to binary image.

    M = number of grid rows (unique y-coordinates at grid points)
    N = number of grid columns (unique x-coordinates at grid points)
    P = number of grid cells intersected by object

    :param rows_in_object: length-P numpy array with row indices (integers)
        of grid cells intersected by object.
    :param columns_in_object: Same as above, except for columns.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :return: binary_image_matrix: M-by-N numpy array of Boolean flags.  If
        binary_image_matrix[i, j] = True, grid cell [i, j] overlaps with the
        object.
    """

    binary_image_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool)
    binary_image_matrix[rows_in_object, columns_in_object] = True
    return binary_image_matrix


def _binary_image_to_grid_points(binary_image_matrix):
    """Converts binary image to list of grid points.

    This method is the inverse of `_grid_points_to_binary_image`.

    :param binary_image_matrix: See documentation for
        `_grid_points_to_binary_image`.
    :return: rows_in_object: Same.
    :return: columns_in_object: Same.
    """

    return numpy.where(binary_image_matrix)


def _is_polyline_closed(latitudes_deg, longitudes_deg):
    """Determines whether or not polyline is closed.

    V = number of vertices

    :param latitudes_deg: length-V numpy array of latitudes (deg N).
    :param longitudes_deg: length-V numpy array of longitudes (deg E).
    :return: is_closed: Boolean flag.
    """

    absolute_lat_diff_deg = numpy.absolute(latitudes_deg[0] - latitudes_deg[-1])
    absolute_lng_diff_deg = numpy.absolute(
        longitudes_deg[0] - longitudes_deg[-1])

    return (absolute_lat_diff_deg < TOLERANCE_DEG and
            absolute_lng_diff_deg < TOLERANCE_DEG)


def check_front_type(front_type_string):
    """Ensures that front type is valid.

    :param front_type_string: String ID for front type.
    :raises: ValueError: if front type is unrecognized.
    """

    error_checking.assert_is_string(front_type_string)
    if front_type_string not in VALID_STRING_IDS:
        error_string = (
            '\n\n{0:s}\nValid front types (listed above) do not include '
            '"{1:s}".').format(VALID_STRING_IDS, front_type_string)
        raise ValueError(error_string)


def string_id_to_integer(front_type_string):
    """Converts front type from string to integer.

    :param front_type_string: String ID for front type.
    :return: front_type_integer: Integer ID for front type.
    """

    check_front_type(front_type_string)
    if front_type_string == WARM_FRONT_STRING_ID:
        return WARM_FRONT_INTEGER_ID

    return COLD_FRONT_INTEGER_ID


def close_frontal_image(ternary_image_matrix, num_iterations=1):
    """Applies binary closing to both warm and cold fronts in image.

    :param ternary_image_matrix: See doc for `_check_frontal_image`.
    :param num_iterations: Number of iterations of binary closing.  The more
        iterations, the more frontal pixels will be created.
    :return: ternary_image_matrix: Same as input, but after closing.
    """

    _check_frontal_image(image_matrix=ternary_image_matrix, assert_binary=False)
    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)

    binary_warm_front_matrix = binary_closing(
        (ternary_image_matrix == WARM_FRONT_INTEGER_ID).astype(int),
        structure=STRUCTURE_MATRIX_FOR_BINARY_CLOSING, origin=0,
        iterations=num_iterations)
    binary_cold_front_matrix = binary_closing(
        (ternary_image_matrix == COLD_FRONT_INTEGER_ID).astype(int),
        structure=STRUCTURE_MATRIX_FOR_BINARY_CLOSING, origin=0,
        iterations=num_iterations)

    ternary_image_matrix[
        numpy.where(binary_warm_front_matrix)
    ] = WARM_FRONT_INTEGER_ID
    ternary_image_matrix[
        numpy.where(binary_cold_front_matrix)
    ] = COLD_FRONT_INTEGER_ID

    return ternary_image_matrix


def buffer_distance_to_narr_mask(buffer_distance_metres):
    """Converts buffer distance to mask over NARR grid.

    m = number of grid rows (unique y-coordinates at grid points) within buffer
        distance
    n = number of grid columns (unique x-coordinates at grid points) within
        buffer distance

    :param buffer_distance_metres: Buffer distance.
    :return: mask_matrix: m-by-n numpy array of Boolean flags.
        mask_matrix[floor(m/2), floor(n/2)] represents the center point, around
        which the buffer is defined.  Element [i, j] indicates whether or not
        grid point [i, j] is in the distance buffer.
    """

    error_checking.assert_is_greater(buffer_distance_metres, 0.)
    buffer_distance_metres = max([buffer_distance_metres, 1.])

    grid_spacing_metres, _ = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    max_row_or_column_offset = int(
        numpy.floor(float(buffer_distance_metres) / grid_spacing_metres))

    row_or_column_offsets = numpy.linspace(
        -max_row_or_column_offset, max_row_or_column_offset,
        num=2*max_row_or_column_offset + 1, dtype=int)

    column_offset_matrix, row_offset_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=row_or_column_offsets,
        y_unique_metres=row_or_column_offsets)
    row_offset_matrix = row_offset_matrix.astype(float)
    column_offset_matrix = column_offset_matrix.astype(float)

    distance_matrix_metres = grid_spacing_metres * numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)
    return distance_matrix_metres <= buffer_distance_metres


def dilate_binary_narr_image(binary_image_matrix, dilation_distance_metres=None,
                             dilation_kernel_matrix=None):
    """Dilates a binary (2-class) image over the NARR grid.

    m = number of rows in dilation kernel
    n = number of columns in dilation kernel

    If `dilation_kernel_matrix` is None, `dilation_distance_metres` will be
    used.

    :param binary_image_matrix: See doucmentation for `_check_frontal_image`.
    :param dilation_distance_metres: Dilation distance.
    :param dilation_kernel_matrix: m-by-n numpy array of integers (all 0 or 1).
        This may be created by `buffer_distance_to_narr_mask`.
    :return: binary_image_matrix: Same as input, except dilated.
    """

    _check_frontal_image(image_matrix=binary_image_matrix, assert_binary=True)

    if dilation_kernel_matrix is None:
        dilation_kernel_matrix = buffer_distance_to_narr_mask(
            dilation_distance_metres).astype(int)

    error_checking.assert_is_numpy_array(
        dilation_kernel_matrix, num_dimensions=2)
    error_checking.assert_is_integer_numpy_array(dilation_kernel_matrix)
    error_checking.assert_is_geq_numpy_array(dilation_kernel_matrix, 0)
    error_checking.assert_is_leq_numpy_array(dilation_kernel_matrix, 1)
    error_checking.assert_is_geq(numpy.sum(dilation_kernel_matrix), 1)

    binary_image_matrix = cv2.dilate(
        binary_image_matrix.astype(numpy.uint8),
        dilation_kernel_matrix.astype(numpy.uint8), iterations=1)
    return binary_image_matrix.astype(int)


def dilate_ternary_narr_image(
        ternary_image_matrix, dilation_distance_metres=None,
        dilation_kernel_matrix=None):
    """Dilates a ternary (3-class) image over the NARR grid.

    :param ternary_image_matrix: See doucmentation for `_check_frontal_image`.
    :param dilation_distance_metres: See documentation for
        `dilate_binary_narr_image`.
    :param dilation_kernel_matrix: See documentation for
        `dilate_binary_narr_image`.
    :return: ternary_image_matrix: Same as input, except dilated.
    """

    _check_frontal_image(image_matrix=ternary_image_matrix, assert_binary=False)

    binary_cold_front_matrix = numpy.full(
        ternary_image_matrix.shape, NO_FRONT_INTEGER_ID, dtype=int)
    binary_cold_front_matrix[
        ternary_image_matrix == COLD_FRONT_INTEGER_ID] = ANY_FRONT_INTEGER_ID
    binary_cold_front_matrix = dilate_binary_narr_image(
        binary_cold_front_matrix,
        dilation_distance_metres=dilation_distance_metres,
        dilation_kernel_matrix=dilation_kernel_matrix)

    binary_warm_front_matrix = numpy.full(
        ternary_image_matrix.shape, NO_FRONT_INTEGER_ID, dtype=int)
    binary_warm_front_matrix[
        ternary_image_matrix == WARM_FRONT_INTEGER_ID] = ANY_FRONT_INTEGER_ID
    binary_warm_front_matrix = dilate_binary_narr_image(
        binary_warm_front_matrix,
        dilation_distance_metres=dilation_distance_metres,
        dilation_kernel_matrix=dilation_kernel_matrix)

    cold_front_row_indices, cold_front_column_indices = numpy.where(
        ternary_image_matrix == COLD_FRONT_INTEGER_ID)
    warm_front_row_indices, warm_front_column_indices = numpy.where(
        ternary_image_matrix == WARM_FRONT_INTEGER_ID)
    both_fronts_row_indices, both_fronts_column_indices = numpy.where(
        numpy.logical_and(binary_cold_front_matrix == ANY_FRONT_INTEGER_ID,
                          binary_warm_front_matrix == ANY_FRONT_INTEGER_ID))

    num_points_to_resolve = len(both_fronts_row_indices)
    for i in range(num_points_to_resolve):
        these_row_diffs = both_fronts_row_indices[i] - cold_front_row_indices
        these_column_diffs = (
            both_fronts_column_indices[i] - cold_front_column_indices)
        this_min_cold_front_distance = numpy.min(
            these_row_diffs**2 + these_column_diffs**2)

        these_row_diffs = both_fronts_row_indices[i] - warm_front_row_indices
        these_column_diffs = (
            both_fronts_column_indices[i] - warm_front_column_indices)
        this_min_warm_front_distance = numpy.min(
            these_row_diffs**2 + these_column_diffs**2)

        if this_min_cold_front_distance <= this_min_warm_front_distance:
            binary_warm_front_matrix[
                both_fronts_row_indices[i],
                both_fronts_column_indices[i]] = NO_FRONT_INTEGER_ID
        else:
            binary_cold_front_matrix[
                both_fronts_row_indices[i],
                both_fronts_column_indices[i]] = NO_FRONT_INTEGER_ID

    ternary_image_matrix[
        binary_cold_front_matrix == ANY_FRONT_INTEGER_ID
    ] = COLD_FRONT_INTEGER_ID
    ternary_image_matrix[
        binary_warm_front_matrix == ANY_FRONT_INTEGER_ID
        ] = WARM_FRONT_INTEGER_ID
    return ternary_image_matrix


def frontal_image_to_grid_points(ternary_image_matrix):
    """Converts frontal image to a list of grid points.

    W = number of grid cells intersected by any warm front
    C = number of grid cells intersected by any cold front

    :param ternary_image_matrix: See documentation for `_check_frontal_image`.
    :return: frontal_grid_point_dict: Dictionary with the following keys.
    frontal_grid_point_dict['warm_front_row_indices']: length-W numpy array
        with row indices (integers) of grid cells intersected by a warm front.
    frontal_grid_point_dict['warm_front_column_indices']: Same as above, except
        for columns.
    frontal_grid_point_dict['cold_front_row_indices']: length-C numpy array
        with row indices (integers) of grid cells intersected by a cold front.
    frontal_grid_point_dict['cold_front_column_indices']: Same as above, except
        for columns.
    """

    _check_frontal_image(image_matrix=ternary_image_matrix, assert_binary=False)

    warm_front_row_indices, warm_front_column_indices = numpy.where(
        ternary_image_matrix == WARM_FRONT_INTEGER_ID)
    cold_front_row_indices, cold_front_column_indices = numpy.where(
        ternary_image_matrix == COLD_FRONT_INTEGER_ID)

    return {
        WARM_FRONT_ROW_INDICES_COLUMN: warm_front_row_indices,
        WARM_FRONT_COLUMN_INDICES_COLUMN: warm_front_column_indices,
        COLD_FRONT_ROW_INDICES_COLUMN: cold_front_row_indices,
        COLD_FRONT_COLUMN_INDICES_COLUMN: cold_front_column_indices
    }


def frontal_grid_points_to_image(
        frontal_grid_point_dict, num_grid_rows, num_grid_columns):
    """Converts list of grid points to a frontal image.

    :param frontal_grid_point_dict: See documentation for
        `frontal_image_to_grid_points`.
    :param num_grid_rows: Number of grid rows (unique y-coordinates at grid
        points).
    :param num_grid_columns: Number of grid columns (unique x-coordinates at
        grid points).
    :return: ternary_image_matrix: See documentation for `_check_frontal_image`.
    """

    ternary_image_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), NO_FRONT_INTEGER_ID, dtype=int)

    ternary_image_matrix[
        frontal_grid_point_dict[WARM_FRONT_ROW_INDICES_COLUMN],
        frontal_grid_point_dict[WARM_FRONT_COLUMN_INDICES_COLUMN]
    ] = WARM_FRONT_INTEGER_ID
    ternary_image_matrix[
        frontal_grid_point_dict[COLD_FRONT_ROW_INDICES_COLUMN],
        frontal_grid_point_dict[COLD_FRONT_COLUMN_INDICES_COLUMN]
    ] = COLD_FRONT_INTEGER_ID

    return ternary_image_matrix


def frontal_image_to_objects(ternary_image_matrix):
    """Converts frontal image to a list of objects (connected regions).

    R = number of regions
    P_i = number of grid points in the [i]th region

    :param ternary_image_matrix: See documentation for `_check_frontal_image`.
    :return: frontal_region_dict: Dictionary with the following keys.
    frontal_region_dict['row_indices_by_region']: length-R list, where the [i]th
        element is a numpy array (length P_i, integers) with indices of grid
        rows in the [i]th region.
    frontal_region_dict['column_indices_by_region']: Same as above, except for
        columns.
    frontal_region_dict['front_type_by_region']: length-R list of front types
        (either "warm" or "cold").
    """

    _check_frontal_image(image_matrix=ternary_image_matrix, assert_binary=False)
    ternary_image_matrix = close_frontal_image(
        ternary_image_matrix=ternary_image_matrix, num_iterations=1)
    region_matrix = label_image(ternary_image_matrix, connectivity=2)

    num_regions = numpy.max(region_matrix)
    row_indices_by_region = [[]] * num_regions
    column_indices_by_region = [[]] * num_regions
    front_type_by_region = [''] * num_regions

    for i in range(num_regions):
        row_indices_by_region[i], column_indices_by_region[i] = numpy.where(
            region_matrix == i + 1)
        this_integer_id = ternary_image_matrix[
            row_indices_by_region[i][0], column_indices_by_region[i][0]]

        if this_integer_id == WARM_FRONT_INTEGER_ID:
            front_type_by_region[i] = WARM_FRONT_STRING_ID
        elif this_integer_id == COLD_FRONT_INTEGER_ID:
            front_type_by_region[i] = COLD_FRONT_STRING_ID

    return {
        ROW_INDICES_BY_REGION_KEY: row_indices_by_region,
        COLUMN_INDICES_BY_REGION_KEY: column_indices_by_region,
        FRONT_TYPE_BY_REGION_KEY: front_type_by_region
    }


def polyline_to_narr_grid(
        polyline_latitudes_deg, polyline_longitudes_deg,
        dilation_distance_metres):
    """Converts polyline to binary image over NARR grid.

    V = number of vertices in polyline

    :param polyline_latitudes_deg: length-V numpy array of latitudes (deg N).
    :param polyline_longitudes_deg: length-V numpy array of longitudes (deg E).
    :param dilation_distance_metres: Dilation distance.  This gives fronts a
        non-infinitesimal width, which allows them to be more than one grid cell
        (pixel) wide.  This accounts for spatial uncertainty in the placement of
        fronts.
    :return: binary_image_matrix: See documentation for `_check_frontal_image`.
        This will be a 277-by-349 matrix, to match the dimensions of the NARR
        grid.
    """

    polyline_x_coords_metres, polyline_y_coords_metres = (
        nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=polyline_latitudes_deg,
            longitudes_deg=polyline_longitudes_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME))

    grid_point_x_coords_metres, grid_point_y_coords_metres = (
        nwp_model_utils.get_xy_grid_points(
            model_name=nwp_model_utils.NARR_MODEL_NAME))

    rows_in_polyline, columns_in_polyline = _polyline_to_grid_points(
        polyline_x_coords_metres=polyline_x_coords_metres,
        polyline_y_coords_metres=polyline_y_coords_metres,
        grid_point_x_coords_metres=grid_point_x_coords_metres,
        grid_point_y_coords_metres=grid_point_y_coords_metres)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    binary_image_matrix = _grid_points_to_binary_image(
        rows_in_object=rows_in_polyline, columns_in_object=columns_in_polyline,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    return dilate_binary_narr_image(
        binary_image_matrix=binary_image_matrix.astype(int),
        dilation_distance_metres=dilation_distance_metres)


def many_polylines_to_narr_grid(polyline_table, dilation_distance_metres):
    """For each time step, converts polylines to list of NARR grid points.

    W = number of grid cells intersected by any warm front at a given time
    C = number of grid cells intersected by any cold front at a given time

    :param polyline_table: See documentation for
        `fronts_io.write_polylines_to_file`.
    :param dilation_distance_metres: Dilation distance.
    :return: frontal_grid_point_table: pandas DataFrame with the following
        columns (and one row for each valid time).
    frontal_grid_point_table.unix_time_sec: Valid time.
    frontal_grid_point_table.warm_front_row_indices: length-W numpy array
        with row indices (integers) of grid cells intersected by a warm front.
    frontal_grid_point_table.warm_front_column_indices: Same but for columns.
    frontal_grid_point_table.cold_front_row_indices: length-C numpy array
        with row indices (integers) of grid cells intersected by a cold front.
    frontal_grid_point_table.cold_front_column_indices: Same but for columns.
    """

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    valid_times_unix_sec = numpy.unique(polyline_table[TIME_COLUMN].values)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in valid_times_unix_sec]
    num_valid_times = len(valid_times_unix_sec)

    warm_front_row_indices_by_time = [[]] * num_valid_times
    warm_front_column_indices_by_time = [[]] * num_valid_times
    cold_front_row_indices_by_time = [[]] * num_valid_times
    cold_front_column_indices_by_time = [[]] * num_valid_times

    for i in range(num_valid_times):
        print 'Converting polylines to NARR grid for {0:s}...'.format(
            valid_time_strings[i])

        these_polyline_indices = numpy.where(
            polyline_table[TIME_COLUMN].values == valid_times_unix_sec[i])[0]
        this_ternary_image_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), NO_FRONT_INTEGER_ID, dtype=int)

        for j in these_polyline_indices:
            skip_this_front = _is_polyline_closed(
                latitudes_deg=polyline_table[LATITUDES_COLUMN].values[j],
                longitudes_deg=polyline_table[LONGITUDES_COLUMN].values[j])

            if skip_this_front:
                this_num_points = len(
                    polyline_table[LATITUDES_COLUMN].values[j])
                print (
                    'SKIPPING front with {0:d} points (closed polyline).'
                ).format(this_num_points)
                continue

            this_binary_image_matrix = polyline_to_narr_grid(
                polyline_latitudes_deg=
                polyline_table[LATITUDES_COLUMN].values[j],
                polyline_longitudes_deg=
                polyline_table[LONGITUDES_COLUMN].values[j],
                dilation_distance_metres=dilation_distance_metres)
            this_binary_image_matrix = this_binary_image_matrix.astype(bool)

            if (polyline_table[FRONT_TYPE_COLUMN].values[j] ==
                    WARM_FRONT_STRING_ID):
                this_ternary_image_matrix[
                    numpy.where(this_binary_image_matrix)
                ] = WARM_FRONT_INTEGER_ID
            else:
                this_ternary_image_matrix[
                    numpy.where(this_binary_image_matrix)
                ] = COLD_FRONT_INTEGER_ID

        this_grid_point_dict = frontal_image_to_grid_points(
            this_ternary_image_matrix)
        warm_front_row_indices_by_time[i] = this_grid_point_dict[
            WARM_FRONT_ROW_INDICES_COLUMN]
        warm_front_column_indices_by_time[i] = this_grid_point_dict[
            WARM_FRONT_COLUMN_INDICES_COLUMN]
        cold_front_row_indices_by_time[i] = this_grid_point_dict[
            COLD_FRONT_ROW_INDICES_COLUMN]
        cold_front_column_indices_by_time[i] = this_grid_point_dict[
            COLD_FRONT_COLUMN_INDICES_COLUMN]

    frontal_grid_dict = {
        TIME_COLUMN: valid_times_unix_sec,
        WARM_FRONT_ROW_INDICES_COLUMN: warm_front_row_indices_by_time,
        WARM_FRONT_COLUMN_INDICES_COLUMN: warm_front_column_indices_by_time,
        COLD_FRONT_ROW_INDICES_COLUMN: cold_front_row_indices_by_time,
        COLD_FRONT_COLUMN_INDICES_COLUMN: cold_front_column_indices_by_time
    }
    return pandas.DataFrame.from_dict(frontal_grid_dict)


def remove_polylines_in_masked_area(
        polyline_table, narr_mask_matrix, verbose=True):
    """Removes any polyline that touches only masked grid cells.

    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param polyline_table: See documentation for
        `fronts_io.write_polylines_to_file`.  Each row is one front.
    :param narr_mask_matrix: M-by-N numpy array of integers (0 or 1).  If
        narr_mask_matrix[i, j] = 0, grid cell [i, j] is masked.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: polyline_table: Same as input, except that some rows may have been
        removed.
    """

    error_checking.assert_is_integer_numpy_array(narr_mask_matrix)
    error_checking.assert_is_geq_numpy_array(narr_mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(narr_mask_matrix, 1)
    error_checking.assert_is_boolean(verbose)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    expected_dimensions = numpy.array(
        [num_grid_rows, num_grid_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        narr_mask_matrix, exact_dimensions=expected_dimensions)

    num_fronts = len(polyline_table.index)
    indices_to_drop = []

    for i in range(num_fronts):
        if numpy.mod(i, 25) == 0 and verbose:
            print (
                'Have checked {0:d} of {1:d} polylines; have removed {2:d} of '
                '{0:d} because they exist only in masked area...'
            ).format(i, num_fronts, len(indices_to_drop))

        skip_this_front = _is_polyline_closed(
            latitudes_deg=polyline_table[LATITUDES_COLUMN].values[i],
            longitudes_deg=polyline_table[LONGITUDES_COLUMN].values[i])

        if skip_this_front:
            indices_to_drop.append(i)
            continue

        this_binary_matrix = polyline_to_narr_grid(
            polyline_latitudes_deg=polyline_table[LATITUDES_COLUMN].values[i],
            polyline_longitudes_deg=polyline_table[LONGITUDES_COLUMN].values[i],
            dilation_distance_metres=1.)

        if not numpy.any(
                numpy.logical_and(
                    this_binary_matrix == 1, narr_mask_matrix == 1)):
            indices_to_drop.append(i)

    if len(indices_to_drop) == 0:
        return polyline_table

    indices_to_drop = numpy.array(indices_to_drop, dtype=int)
    return polyline_table.drop(
        polyline_table.index[indices_to_drop], axis=0, inplace=False)
