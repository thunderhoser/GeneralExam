"""Methods for handling atmospheric fronts.

A front may be represented as either a polyline or a set of grid points.
"""

import numpy
import pandas
import shapely.geometry
from scipy.ndimage.morphology import binary_dilation
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE_DEG = 1e-3
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H'

FRONT_TYPE_COLUMN = 'front_type'
TIME_COLUMN = 'unix_time_sec'
LATITUDES_COLUMN = 'latitudes_deg'
LONGITUDES_COLUMN = 'longitudes_deg'

WARM_FRONT_ROW_INDICES_COLUMN = 'warm_front_row_indices'
WARM_FRONT_COLUMN_INDICES_COLUMN = 'warm_front_column_indices'
COLD_FRONT_ROW_INDICES_COLUMN = 'cold_front_row_indices'
COLD_FRONT_COLUMN_INDICES_COLUMN = 'cold_front_column_indices'

NO_FRONT_INTEGER_ID = 0
WARM_FRONT_INTEGER_ID = 1
COLD_FRONT_INTEGER_ID = 2

WARM_FRONT_STRING_ID = 'warm'
COLD_FRONT_STRING_ID = 'cold'
VALID_STRING_IDS = [WARM_FRONT_STRING_ID, COLD_FRONT_STRING_ID]


def _check_polyline(vertex_x_coords_metres, vertex_y_coords_metres):
    """Checks polyline for errors.

    V = number of vertices

    :param vertex_x_coords_metres: length-V numpy array with x-coordinates of
        vertices.
    :param vertex_y_coords_metres: length-V numpy array with y-coordinates of
        vertices.
    """

    error_checking.assert_is_numpy_array_without_nan(vertex_x_coords_metres)
    error_checking.assert_is_numpy_array(
        vertex_x_coords_metres, num_dimensions=1)
    num_vertices = len(vertex_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(vertex_y_coords_metres)
    error_checking.assert_is_numpy_array(
        vertex_y_coords_metres, exact_dimensions=numpy.array([num_vertices]))


def _vertex_arrays_to_list(vertex_x_coords_metres, vertex_y_coords_metres):
    """Converts set of vertices from two arrays to one list.

    V = number of vertices

    :param vertex_x_coords_metres: length-V numpy array with x-coordinates of
        vertices.
    :param vertex_y_coords_metres: length-V numpy array with y-coordinates of
        vertices.
    :return: vertex_list_xy_metres: length-V list, where each element is an
        (x, y) tuple.
    """

    _check_polyline(vertex_x_coords_metres=vertex_x_coords_metres,
                    vertex_y_coords_metres=vertex_y_coords_metres)

    num_vertices = len(vertex_x_coords_metres)
    vertex_list_xy_metres = []
    for i in range(num_vertices):
        vertex_list_xy_metres.append(
            (vertex_x_coords_metres[i], vertex_y_coords_metres[i]))

    return vertex_list_xy_metres


def _polyline_from_vertex_arrays_to_linestring(
        vertex_x_coords_metres, vertex_y_coords_metres):
    """Converts polyline from vertex arrays to `shapely.geometry.LineString`.

    V = number of vertices

    :param vertex_x_coords_metres: length-V numpy array with x-coordinates of
        vertices.
    :param vertex_y_coords_metres: length-V numpy array with y-coordinates of
        vertices.
    :return: linestring_object_xy_metres: Instance of
        `shapely.geometry.LineString`, with vertex coordinates in metres.
    :raises: ValueError: if resulting LineString object is invalid.
    """

    vertex_list_xy_metres = _vertex_arrays_to_list(
        vertex_x_coords_metres=vertex_x_coords_metres,
        vertex_y_coords_metres=vertex_y_coords_metres)

    linestring_object_xy_metres = shapely.geometry.LineString(
        vertex_list_xy_metres)
    if not linestring_object_xy_metres.is_valid:
        raise ValueError('Resulting LineString object is invalid.')

    return linestring_object_xy_metres


def _grid_cell_to_polygon(
        grid_point_x_metres, grid_point_y_metres, x_spacing_metres,
        y_spacing_metres):
    """Converts a grid cell to a polygon.

    This method assumes that the grid is regular in x-y coordinates, not in lat-
    long coordinates.

    :param grid_point_x_metres: x-coordinate of center point ("grid point").
    :param grid_point_y_metres: y-coordinate of center point ("grid point").
    :param x_spacing_metres: Spacing between adjacent grid points in
        x-direction.
    :param y_spacing_metres: Spacing between adjacent grid points in
        y-direction.
    :return: grid_cell_edge_polygon_xy_metres: Instance of
        `shapely.geometry.Polygon`, where each vertex is a corner of the grid
        cell.  Coordinates are in metres.
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
    """Converts a polyline to a set of grid points.

    P = number of vertices in polyline
    M = number of grid rows (unique grid-point y-coordinates)
    N = number of grid columns (unique grid-point x-coordinates)
    Q = number of grid points in polyline

    This method assumes that `grid_point_x_coords_metres` and
    `grid_point_y_coords_metres` are sorted in ascending order and equally
    spaced.  In other words, the grid must be *regular* in x-y.

    :param polyline_x_coords_metres: length-P numpy array with x-coordinates in
        polyline.
    :param polyline_y_coords_metres: length-P numpy array with y-coordinates in
        polyline.
    :param grid_point_x_coords_metres: length-N numpy array with unique
        x-coordinates of grid points.
    :param grid_point_y_coords_metres: length-M numpy array with unique
        y-coordinates of grid points.
    :return: rows_in_polyline: length-Q numpy array with row indices (integers)
        of grid points in polyline.
    :return: columns_in_polyline: length-Q numpy array with column indices
        (integers) of grid points in polyline.
    """

    polyline_object_xy_metres = _polyline_from_vertex_arrays_to_linestring(
        vertex_x_coords_metres=polyline_x_coords_metres,
        vertex_y_coords_metres=polyline_y_coords_metres)

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
    """Converts set of grid points in object to a binary image matrix.

    P = number of grid points in object
    M = number of grid rows (unique grid-point y-coordinates)
    N = number of grid columns (unique grid-point x-coordinates)

    :param rows_in_object: length-P numpy array with indices (integers) of rows
        in object.
    :param columns_in_object: length-P numpy array with indices (integers) of
        columns in object.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :return: binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, grid cell [i, j] is part of the object.
        Otherwise, grid cell [i, j] is *not* part of the object.
    """

    binary_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool)
    binary_matrix[rows_in_object, columns_in_object] = True
    return binary_matrix


def _binary_image_to_grid_points(binary_matrix):
    """Converts binary image matrix to set of grid points in object.

    P = number of grid points in object
    M = number of grid rows (unique grid-point y-coordinates)
    N = number of grid columns (unique grid-point x-coordinates)

    :param binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, grid cell [i, j] is part of the object.
        Otherwise, grid cell [i, j] is *not* part of the object.
    :return: rows_in_object: length-P numpy array with indices (integers) of
        rows in object.
    :return: columns_in_object: length-P numpy array with indices (integers) of
        columns in object.
    """

    return numpy.where(binary_matrix)


def _dilate_binary_image(binary_matrix, dilation_half_width_in_grid_cells):
    """Dilates a binary image matrix.

    M = number of grid rows (unique grid-point y-coordinates)
    N = number of grid columns (unique grid-point x-coordinates)

    :param binary_matrix: M-by-N numpy array of Boolean flags.
    :param dilation_half_width_in_grid_cells: Half-width of dilation window.
    :return: binary_matrix: Same as input, except dilated.
    """

    return binary_dilation(
        binary_matrix, iterations=dilation_half_width_in_grid_cells, origin=0,
        border_value=0)


def check_front_type(front_string_id):
    """Ensures that front type is valid.

    :param front_string_id: String ID for front type.
    :raises: ValueError: if front type is unrecognized.
    """

    error_checking.assert_is_string(front_string_id)
    if front_string_id not in VALID_STRING_IDS:
        error_string = (
            '\n\n' + str(VALID_STRING_IDS) +
            '\n\nValid front types (listed above) do not include "' +
            front_string_id + '".')
        raise ValueError(error_string)


def polyline_to_binary_narr_grid(
        polyline_latitudes_deg, polyline_longitudes_deg,
        dilation_half_width_in_grid_cells=1):
    """Converts polyline to binary image with dimensions of NARR* grid.

    * NARR = North American Regional Reanalysis

    P = number of vertices in polyline
    M = number of rows in NARR grid = 277
    N = number of columns in NARR grid = 349

    :param polyline_latitudes_deg: length-P numpy array with latitudes (deg N)
        in polyline.
    :param polyline_longitudes_deg: length-P numpy array with longitudes (deg E)
        in polyline.
    :param dilation_half_width_in_grid_cells: Half-width of dilation window for
        `_dilate_binary_image`.
    :return: binary_matrix: M-by-N numpy array of Boolean flags.  If
        binary_matrix[i, j] = True, the polyline passes through grid cell
        [i, j].  Otherwise, the polyline does not pass through grid cell [i, j].
    """

    error_checking.assert_is_integer(dilation_half_width_in_grid_cells)
    error_checking.assert_is_greater(dilation_half_width_in_grid_cells, 0)

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

    binary_matrix = _grid_points_to_binary_image(
        rows_in_object=rows_in_polyline, columns_in_object=columns_in_polyline,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    return _dilate_binary_image(
        binary_matrix=binary_matrix,
        dilation_half_width_in_grid_cells=dilation_half_width_in_grid_cells)


def many_polylines_to_narr_grid(
        front_table, dilation_half_width_in_grid_cells=1):
    """For each time step, converts frontal polylines to image over NARR* grid.

    * NARR = North American Regional Reanalysis

    M = number of rows in NARR grid = 277
    N = number of columns in NARR grid = 349
    W = number of grid cells intersected by warm front at a given valid time
    C = number of grid cells intersected by cold front at a given valid time

    :param front_table: See documentation for
        `fronts_io.write_polylines_to_file`.
    :param dilation_half_width_in_grid_cells: Half-width of dilation window for
        `_dilate_binary_image`.
    :return: frontal_grid_table: pandas DataFrame with the following columns.
        Each row is one valid time.
    frontal_grid_table.unix_time_sec: Valid time.
    frontal_grid_table.warm_front_row_indices: length-W numpy array with row
        indices (integers) of grid cells intersected by a warm front.
    frontal_grid_table.warm_front_column_indices: Same as above, except for
        columns.
    frontal_grid_table.cold_front_row_indices: length-C numpy array with row
        indices (integers) of grid cells intersected by a cold front.
    frontal_grid_table.cold_front_column_indices: Same as above, except for
        columns.
    """

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    valid_times_unix_sec = numpy.unique(front_table[TIME_COLUMN].values)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_LOG_MESSAGES)
        for t in valid_times_unix_sec]
    num_valid_times = len(valid_times_unix_sec)

    warm_front_row_indices_by_time = [[]] * num_valid_times
    warm_front_column_indices_by_time = [[]] * num_valid_times
    cold_front_row_indices_by_time = [[]] * num_valid_times
    cold_front_column_indices_by_time = [[]] * num_valid_times

    for i in range(num_valid_times):
        print ('Converting frontal polylines to image over NARR grid for '
               '{0:s}...').format(valid_time_strings[i])

        these_front_indices = numpy.where(
            front_table[TIME_COLUMN].values == valid_times_unix_sec[i])[0]
        this_front_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), NO_FRONT_INTEGER_ID, dtype=int)

        for j in these_front_indices:

            # TODO(thunderhoser): This is a hack to account for very short
            # polylines (where all points are essentially the same).  Should put
            # this check in pre-processing.
            these_latitudes_deg = front_table[LATITUDES_COLUMN].values[j]
            these_longitudes_deg = front_table[LONGITUDES_COLUMN].values[j]

            this_absolute_lat_diff_deg = numpy.absolute(
                these_latitudes_deg[0] - these_latitudes_deg[-1])
            this_absolute_lng_diff_deg = numpy.absolute(
                these_longitudes_deg[0] - these_longitudes_deg[-1])
            if (this_absolute_lat_diff_deg < TOLERANCE_DEG and
                    this_absolute_lng_diff_deg < TOLERANCE_DEG):
                print ('SKIPPING front with {0:d} points.  First and last '
                       'points touch.'.format(len(these_latitudes_deg)))
                continue

            this_binary_matrix = polyline_to_binary_narr_grid(
                polyline_latitudes_deg=
                front_table[LATITUDES_COLUMN].values[j],
                polyline_longitudes_deg=
                front_table[LONGITUDES_COLUMN].values[j],
                dilation_half_width_in_grid_cells=
                dilation_half_width_in_grid_cells)

            if front_table[FRONT_TYPE_COLUMN].values[j] == WARM_FRONT_STRING_ID:
                this_front_matrix[
                    numpy.where(this_binary_matrix)] = WARM_FRONT_INTEGER_ID

            elif (front_table[FRONT_TYPE_COLUMN].values[i] ==
                  COLD_FRONT_STRING_ID):
                this_front_matrix[
                    numpy.where(this_binary_matrix)] = COLD_FRONT_INTEGER_ID

        (warm_front_row_indices_by_time[i],
         warm_front_column_indices_by_time[i]) = numpy.where(
             this_front_matrix == WARM_FRONT_INTEGER_ID)
        (cold_front_row_indices_by_time[i],
         cold_front_column_indices_by_time[i]) = numpy.where(
             this_front_matrix == COLD_FRONT_INTEGER_ID)

    frontal_grid_dict = {
        TIME_COLUMN: valid_times_unix_sec,
        WARM_FRONT_ROW_INDICES_COLUMN: warm_front_row_indices_by_time,
        WARM_FRONT_COLUMN_INDICES_COLUMN: warm_front_column_indices_by_time,
        COLD_FRONT_ROW_INDICES_COLUMN: cold_front_row_indices_by_time,
        COLD_FRONT_COLUMN_INDICES_COLUMN: cold_front_column_indices_by_time
    }
    return pandas.DataFrame.from_dict(frontal_grid_dict)
