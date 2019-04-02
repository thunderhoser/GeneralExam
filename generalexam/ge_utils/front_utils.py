"""Methods for handling atmospheric fronts.

A front may be represented as either a polyline or a set of grid points.
"""

import numpy
import pandas
import cv2
import shapely.geometry
from scipy.ndimage.morphology import binary_closing
from skimage.measure import label as label_image
from skimage.measure import regionprops
from sklearn.metrics.pairwise import euclidean_distances
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 0.001
TIME_FORMAT = '%Y-%m-%d-%H'

ERA5_MODEL_NAME = 'era5'
VALID_NWP_MODEL_NAMES = [ERA5_MODEL_NAME, nwp_model_utils.NARR_MODEL_NAME]

FRONT_TYPE_COLUMN = 'front_type_string'
TIME_COLUMN = 'valid_time_unix_sec'
LATITUDES_COLUMN = 'latitudes_deg'
LONGITUDES_COLUMN = 'longitudes_deg'

COLD_FRONT_ROWS_COLUMN = 'cold_front_rows'
COLD_FRONT_COLUMNS_COLUMN = 'cold_front_columns'
WARM_FRONT_ROWS_COLUMN = 'warm_front_rows'
WARM_FRONT_COLUMNS_COLUMN = 'warm_front_columns'
DILATION_DISTANCE_COLUMN = 'dilation_distance_metres'
MODEL_NAME_COLUMN = 'model_name'

ROWS_BY_REGION_KEY = 'row_indices_by_region'
COLUMNS_BY_REGION_KEY = 'column_indices_by_region'
FRONT_TYPES_KEY = 'front_type_strings'
MAJOR_AXIS_LENGTHS_KEY = 'major_axis_lengths_px'

NO_FRONT_ENUM = 0
ANY_FRONT_ENUM = 1
WARM_FRONT_ENUM = 1
COLD_FRONT_ENUM = 2

VALID_FRONT_TYPE_ENUMS = [
    NO_FRONT_ENUM, ANY_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM
]

WARM_FRONT_STRING = 'warm'
COLD_FRONT_STRING = 'cold'
VALID_FRONT_TYPE_STRINGS = [WARM_FRONT_STRING, COLD_FRONT_STRING]


def _vertex_arrays_to_list(x_coords_metres, y_coords_metres):
    """Converts set of vertices from two arrays to one list.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    :return: list_of_xy_tuples_metres: length-V list of (x, y) tuples.
    """

    check_polyline(x_coords_metres=x_coords_metres,
                   y_coords_metres=y_coords_metres)

    num_vertices = len(x_coords_metres)
    list_of_xy_tuples_metres = []

    for i in range(num_vertices):
        list_of_xy_tuples_metres.append(
            (x_coords_metres[i], y_coords_metres[i])
        )

    return list_of_xy_tuples_metres


def _create_linestring(x_coords_metres, y_coords_metres):
    """Converts polyline from two arrays to `shapely.geometry.LineString`.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    :return: linestring_object_xy_metres: `shapely.geometry.LineString` object.
    """

    list_of_xy_tuples_metres = _vertex_arrays_to_list(
        x_coords_metres=x_coords_metres, y_coords_metres=y_coords_metres)

    linestring_object_xy_metres = shapely.geometry.LineString(
        list_of_xy_tuples_metres)

    if not linestring_object_xy_metres.is_valid:
        raise ValueError('Resulting LineString object is invalid.')

    return linestring_object_xy_metres


def _grid_cell_to_polygon(grid_point_x_metres, grid_point_y_metres,
                          x_spacing_metres, y_spacing_metres):
    """Converts grid cell from single point to polygon.

    This method assumes that the grid is regular in x-y space.

    :param grid_point_x_metres: x-coordinate at center of grid cell.
    :param grid_point_y_metres: y-coordinate at center of grid cell.
    :param x_spacing_metres: Spacing between grid points in adjacent columns.
    :param y_spacing_metres: Spacing between grid points in adjacent rows.
    :return: polygon_object_xy_metres: `shapely.geometry.Polygon` object.
    """

    x_min_metres = grid_point_x_metres - x_spacing_metres / 2
    x_max_metres = grid_point_x_metres + x_spacing_metres / 2
    y_min_metres = grid_point_y_metres - y_spacing_metres / 2
    y_max_metres = grid_point_y_metres + y_spacing_metres / 2

    vertex_x_coords_metres = numpy.array(
        [x_min_metres, x_max_metres, x_max_metres, x_min_metres, x_min_metres]
    )
    vertex_y_coords_metres = numpy.array(
        [y_min_metres, y_min_metres, y_max_metres, y_max_metres, y_min_metres]
    )

    return polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=vertex_x_coords_metres,
        exterior_y_coords=vertex_y_coords_metres)


def _polyline_to_grid_points(
        polyline_x_coords_metres, polyline_y_coords_metres,
        grid_point_x_coords_metres, grid_point_y_coords_metres):
    """Converts polyline from list of vertices to list of grid points.

    V = number of vertices
    M = number of rows in grid
    N = number of columns in grid
    P = number of grid cells intersected by polyline

    This method assumes that the grid is regular in x-y space.

    :param polyline_x_coords_metres: length-V numpy array of x-coordinates.
    :param polyline_y_coords_metres: length-V numpy array of y-coordinates.
    :param grid_point_x_coords_metres: length-N numpy array of x-coordinates.
    :param grid_point_y_coords_metres: length-M numpy array of y-coordinates.
    :return: rows_in_polyline: length-P numpy array with row indices of grid
        cells.
    :return: columns_in_polyline: length-P numpy array with column indices of
        grid cells.
    """

    polyline_object_xy_metres = _create_linestring(
        x_coords_metres=polyline_x_coords_metres,
        y_coords_metres=polyline_y_coords_metres)

    x_spacing_metres = numpy.diff(grid_point_x_coords_metres[:2])[0]
    y_spacing_metres = numpy.diff(grid_point_y_coords_metres[:2])[0]

    x_min_to_consider_metres = (
        numpy.min(polyline_x_coords_metres) - x_spacing_metres
    )
    x_max_to_consider_metres = (
        numpy.max(polyline_x_coords_metres) + x_spacing_metres
    )
    y_min_to_consider_metres = (
        numpy.min(polyline_y_coords_metres) - y_spacing_metres
    )
    y_max_to_consider_metres = (
        numpy.max(polyline_y_coords_metres) + y_spacing_metres
    )

    x_in_range_indices = numpy.where(numpy.logical_and(
        grid_point_x_coords_metres >= x_min_to_consider_metres,
        grid_point_x_coords_metres <= x_max_to_consider_metres
    ))[0]

    y_in_range_indices = numpy.where(numpy.logical_and(
        grid_point_y_coords_metres >= y_min_to_consider_metres,
        grid_point_y_coords_metres <= y_max_to_consider_metres
    ))[0]

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
                    polyline_object_xy_metres)
            )

            if not this_intersection_flag:
                continue

            rows_in_polyline.append(i + row_offset)
            columns_in_polyline.append(j + column_offset)

    rows_in_polyline = numpy.array(rows_in_polyline, dtype=int)
    columns_in_polyline = numpy.array(columns_in_polyline, dtype=int)
    return rows_in_polyline, columns_in_polyline


def _grid_points_to_boolean_matrix(
        rows_in_front, columns_in_front, num_grid_rows, num_grid_columns):
    """Converts list of frontal grid points to Boolean matrix.

    M = number of rows in grid
    N = number of columns in grid
    P = number of grid cells in front

    :param rows_in_front: length-P numpy array with row indices of grid cells in
        front.
    :param columns_in_front: length-P numpy array with column indices of grid
        cells in front.
    :param num_grid_rows: M in the above discussion.
    :param num_grid_columns: N in the above discussion.
    :return: boolean_front_matrix: M-by-N numpy array of Boolean flags,
        indicating which grid cells are in the front.
    """

    boolean_front_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool)

    boolean_front_matrix[rows_in_front, columns_in_front] = True
    return boolean_front_matrix


def _boolean_matrix_to_grid_points(boolean_front_matrix):
    """Converts Boolean matrix to list of frontal grid points.

    This method is the inverse of `_grid_points_to_boolean_matrix`.

    :param boolean_front_matrix: See doc for `_grid_points_to_boolean_matrix`.
    :return: rows_in_front: Same.
    :return: columns_in_front: Same.
    """

    return numpy.where(boolean_front_matrix)


def _is_polyline_closed(x_coords_metres, y_coords_metres):
    """Determines whether or not polyline is closed.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    :return: is_closed: Boolean flag.
    """

    x_flag = (
        numpy.absolute(x_coords_metres[0] - x_coords_metres[-1]) < TOLERANCE
    )

    y_flag = (
        numpy.absolute(y_coords_metres[0] - y_coords_metres[-1]) < TOLERANCE
    )

    return x_flag and y_flag


def _break_wf_cf_ties(ternary_label_matrix, new_wf_flag_matrix,
                      new_cf_flag_matrix, tiebreaker_enum=COLD_FRONT_ENUM):
    """Breaks ties between WF and CF labels.

    Specifically, at any grid cell [i, j] where both new_wf_flag_matrix[i, j] =
    new_cf_flag_matrix[i, j] = 1, this method determines what the best label is.

    M = number of rows in grid
    N = number of columns in grid

    :param ternary_label_matrix: M-by-N numpy array with original ternary
        labels (no front, warm front, or cold front) before some image-
        morphology operation (e.g., dilation or closing).
    :param new_wf_flag_matrix: M-by-N numpy array (all 0 or 1) of warm-front
        flags after the image-morphology operation.
    :param new_cf_flag_matrix: Same but for cold fronts.
    :param tiebreaker_enum: Front type in case of a tie (must be accepted by
        `check_front_type_enum`).
    :return: ternary_label_matrix: Same as input but maybe with different
        elements.
    """

    cold_front_row_indices, cold_front_column_indices = numpy.where(
        ternary_label_matrix == COLD_FRONT_ENUM)
    warm_front_row_indices, warm_front_column_indices = numpy.where(
        ternary_label_matrix == WARM_FRONT_ENUM)
    both_fronts_row_indices, both_fronts_column_indices = numpy.where(
        numpy.logical_and(new_cf_flag_matrix == 1, new_wf_flag_matrix == 1)
    )

    num_points_to_resolve = len(both_fronts_row_indices)

    for i in range(num_points_to_resolve):
        these_row_diffs = both_fronts_row_indices[i] - cold_front_row_indices
        these_column_diffs = (
            both_fronts_column_indices[i] - cold_front_column_indices
        )
        this_min_cold_front_distance = numpy.min(
            these_row_diffs**2 + these_column_diffs**2)

        these_row_diffs = both_fronts_row_indices[i] - warm_front_row_indices
        these_column_diffs = (
            both_fronts_column_indices[i] - warm_front_column_indices
        )
        this_min_warm_front_distance = numpy.min(
            these_row_diffs ** 2 + these_column_diffs ** 2)

        if numpy.isclose(this_min_cold_front_distance,
                         this_min_warm_front_distance, atol=TOLERANCE):
            if tiebreaker_enum == COLD_FRONT_ENUM:
                new_wf_flag_matrix[
                    both_fronts_row_indices[i], both_fronts_column_indices[i]
                ] = NO_FRONT_ENUM
            else:
                new_cf_flag_matrix[
                    both_fronts_row_indices[i], both_fronts_column_indices[i]
                ] = NO_FRONT_ENUM

            continue

        if this_min_cold_front_distance < this_min_warm_front_distance:
            new_wf_flag_matrix[
                both_fronts_row_indices[i], both_fronts_column_indices[i]
            ] = NO_FRONT_ENUM
        else:
            new_cf_flag_matrix[
                both_fronts_row_indices[i], both_fronts_column_indices[i]
            ] = NO_FRONT_ENUM

    ternary_label_matrix[new_cf_flag_matrix == 1] = COLD_FRONT_ENUM
    ternary_label_matrix[new_wf_flag_matrix == 1] = WARM_FRONT_ENUM

    return ternary_label_matrix


def check_polyline(x_coords_metres, y_coords_metres):
    """Error-checks polyline.

    V = number of vertices

    :param x_coords_metres: length-V numpy array of x-coordinates.
    :param y_coords_metres: length-V numpy array of y-coordinates.
    """

    error_checking.assert_is_numpy_array_without_nan(x_coords_metres)
    error_checking.assert_is_numpy_array(x_coords_metres, num_dimensions=1)

    num_vertices = len(x_coords_metres)
    expected_dimensions = numpy.array([num_vertices], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(y_coords_metres)
    error_checking.assert_is_numpy_array(
        y_coords_metres, exact_dimensions=expected_dimensions)


def check_gridded_labels(label_matrix, assert_binary=False):
    """Error-checks gridded front labels.

    M = number of rows in grid
    N = number of columns in grid

    :param label_matrix: M-by-N numpy array of integers.  May be either binary
        (ranging from 0...1) or ternary (ranging from 0...2).  If binary,
        label_matrix[i, j] indicates whether or not a front (of any type) passes
        through grid cell [i, j].  If ternary, label_matrix[i, j] indicates the
        *type* of front (warm, cold, or none) passing through grid cell [i, j].
    :param assert_binary: Boolean flag.  If True and image is non-binary, this
        method will error out.
    """

    error_checking.assert_is_numpy_array(label_matrix, num_dimensions=2)
    error_checking.assert_is_integer_numpy_array(label_matrix)
    error_checking.assert_is_geq_numpy_array(
        label_matrix, numpy.min(VALID_FRONT_TYPE_ENUMS)
    )

    error_checking.assert_is_boolean(assert_binary)

    if assert_binary:
        error_checking.assert_is_leq_numpy_array(
            label_matrix, ANY_FRONT_ENUM)
    else:
        error_checking.assert_is_leq_numpy_array(
            label_matrix, numpy.max(VALID_FRONT_TYPE_ENUMS)
        )


def check_front_type_string(front_type_string):
    """Error-checks front type.

    :param front_type_string: Front type (string).
    :raises: ValueError: if `front_type_string not in VALID_FRONT_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(front_type_string)

    if front_type_string not in VALID_FRONT_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid front types (listed above) do not include "{1:s}".'
        ).format(str(VALID_FRONT_TYPE_STRINGS), front_type_string)

        raise ValueError(error_string)


def check_front_type_enum(front_type_enum):
    """Error-checks front type.

    :param front_type_enum: Front type (integer).
    :raises: ValueError: if `front_type_enum not in VALID_FRONT_TYPE_ENUMS`.
    """

    error_checking.assert_is_integer(front_type_enum)

    if front_type_enum not in VALID_FRONT_TYPE_ENUMS:
        error_string = (
            '\n{0:s}\nValid front types (listed above) do not include {1:d}.'
        ).format(str(VALID_FRONT_TYPE_ENUMS), front_type_enum)

        raise ValueError(error_string)


def check_nwp_model_name(model_name):
    """Error-checks name of NWP (numerical weather prediction) model.

    :param model_name: Model name.
    :raises: ValueError: if `model_name not in VALID_NWP_MODEL_NAMES`.
    """

    error_checking.assert_is_string(model_name)

    if model_name not in VALID_NWP_MODEL_NAMES:
        error_string = (
            '\n{0:s}\nValid model names (listed above) do not include "{1:s}".'
        ).format(str(VALID_NWP_MODEL_NAMES), model_name)

        raise ValueError(error_string)


def front_type_string_to_int(front_type_string):
    """Converts front type from string to integer.

    :param front_type_string: Front type (string).
    :return: front_type_enum: Front type (integer).
    """

    check_front_type_string(front_type_string)

    if front_type_string == WARM_FRONT_STRING:
        return WARM_FRONT_ENUM

    return COLD_FRONT_ENUM


def buffer_distance_to_dilation_mask(
        buffer_distance_metres, grid_spacing_metres):
    """Converts buffer distance to mask over model grid.

    This method assumes that the grid is equidistant.

    P = max number of grid cells within buffer distance

    :param buffer_distance_metres: Buffer distance.
    :param grid_spacing_metres: Spacing between adjacent grid points in model
        grid.
    :return: mask_matrix: P-by-P numpy array of Boolean flags.
        mask_matrix[floor(P/2), floor(P/2)] represents the center point, around
        which the buffer is defined.
        mask_matrix[floor(P/2) + i, floor(P/2) + j] indicates whether or not the
        grid point i rows down and j columns to the right is within the buffer
        distance.
    """

    error_checking.assert_is_greater(buffer_distance_metres, 0)
    error_checking.assert_is_greater(grid_spacing_metres, 0)
    buffer_distance_metres = max([buffer_distance_metres, 1.])

    max_pixel_offset = int(numpy.floor(
        float(buffer_distance_metres) / grid_spacing_metres
    ))

    pixel_offsets = numpy.linspace(
        -max_pixel_offset, max_pixel_offset, num=2*max_pixel_offset + 1,
        dtype=float)

    column_offset_matrix, row_offset_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=pixel_offsets, y_unique_metres=pixel_offsets)

    distance_matrix_metres = grid_spacing_metres * numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    return distance_matrix_metres <= buffer_distance_metres


def dilate_binary_label_matrix(
        binary_label_matrix, dilation_mask_matrix=None,
        dilation_distance_metres=None, grid_spacing_metres=None):
    """Dilates gridded binary ("front or no front") labels.

    :param binary_label_matrix: See doc for `check_gridded_labels`.
    :param dilation_mask_matrix: numpy array created by
        `buffer_distance_to_dilation_mask`.
    :param dilation_distance_metres:
        [used only if `dilation_mask_matrix is None`]
        Dilation distance.
    :param grid_spacing_metres: [used only if `dilation_mask_matrix is None`]
        Spacing between adjacent grid points in model grid.
    :return: binary_label_matrix: Same as input but dilated.
    """

    check_gridded_labels(label_matrix=binary_label_matrix, assert_binary=True)
    if dilation_distance_metres is not None and dilation_distance_metres <= 0:
        return binary_label_matrix

    if dilation_mask_matrix is None:
        dilation_mask_matrix = buffer_distance_to_dilation_mask(
            buffer_distance_metres=dilation_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        ).astype(int)

    error_checking.assert_is_integer_numpy_array(dilation_mask_matrix)
    error_checking.assert_is_numpy_array(
        dilation_mask_matrix, num_dimensions=2)

    error_checking.assert_is_geq_numpy_array(dilation_mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(dilation_mask_matrix, 1)
    error_checking.assert_is_geq(numpy.sum(dilation_mask_matrix), 1)

    binary_label_matrix = cv2.dilate(
        binary_label_matrix.astype(numpy.uint8),
        dilation_mask_matrix.astype(numpy.uint8),
        iterations=1
    )

    return binary_label_matrix.astype(int)


def dilate_ternary_label_matrix(
        ternary_label_matrix, dilation_mask_matrix=None,
        dilation_distance_metres=None, grid_spacing_metres=None,
        tiebreaker_enum=COLD_FRONT_ENUM):
    """Dilates gridded ternary ("no front, warm front, or cold front") labels.

    :param ternary_label_matrix: See doc for `check_gridded_labels`.
    :param dilation_mask_matrix: numpy array created by
        `buffer_distance_to_dilation_mask`.
    :param dilation_distance_metres:
        [used only if `dilation_mask_matrix is None`]
        Dilation distance.
    :param grid_spacing_metres: [used only if `dilation_mask_matrix is None`]
        Spacing between adjacent grid points in model grid.
    :param tiebreaker_enum: Front type in case of a tie (must be accepted by
        `check_front_type_enum`).
    :return: ternary_label_matrix: Same as input but dilated.
    """

    check_gridded_labels(label_matrix=ternary_label_matrix, assert_binary=False)
    if dilation_distance_metres is not None and dilation_distance_metres <= 0:
        return ternary_label_matrix

    check_front_type_enum(tiebreaker_enum)
    error_checking.assert_is_greater(tiebreaker_enum, NO_FRONT_ENUM)

    if dilation_mask_matrix is None:
        dilation_mask_matrix = buffer_distance_to_dilation_mask(
            buffer_distance_metres=dilation_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        ).astype(int)

    new_cf_flag_matrix = (ternary_label_matrix == COLD_FRONT_ENUM).astype(int)
    new_cf_flag_matrix = dilate_binary_label_matrix(
        binary_label_matrix=new_cf_flag_matrix,
        dilation_mask_matrix=dilation_mask_matrix)

    new_wf_flag_matrix = (ternary_label_matrix == WARM_FRONT_ENUM).astype(int)
    new_wf_flag_matrix = dilate_binary_label_matrix(
        binary_label_matrix=new_wf_flag_matrix,
        dilation_mask_matrix=dilation_mask_matrix)

    return _break_wf_cf_ties(
        ternary_label_matrix=ternary_label_matrix,
        new_wf_flag_matrix=new_wf_flag_matrix,
        new_cf_flag_matrix=new_cf_flag_matrix, tiebreaker_enum=tiebreaker_enum)


def close_binary_label_matrix(
        binary_label_matrix, mask_matrix=None, buffer_distance_metres=None,
        grid_spacing_metres=None):
    """Closes gridded binary ("front or no front") labels.

    :param binary_label_matrix: See doc for `dilate_binary_label_matrix`.
    :param mask_matrix: Same.
    :param buffer_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: binary_label_matrix: Same as input but closed.
    """

    # TODO(thunderhoser): I still don't really understand the behaviour of
    # binary closing.  It generally does not do what I want.

    check_gridded_labels(label_matrix=binary_label_matrix, assert_binary=True)

    if mask_matrix is None:
        mask_matrix = buffer_distance_to_dilation_mask(
            buffer_distance_metres=buffer_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        ).astype(int)

    error_checking.assert_is_integer_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    error_checking.assert_is_geq_numpy_array(mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(mask_matrix, 1)
    error_checking.assert_is_geq(numpy.sum(mask_matrix), 1)

    this_num_cells = int(numpy.ceil(float(mask_matrix.shape[0]) / 2))
    pad_width_arg = (
        (this_num_cells, this_num_cells),
        (this_num_cells, this_num_cells)
    )

    binary_label_matrix = numpy.pad(
        binary_label_matrix, pad_width=pad_width_arg, mode='constant',
        constant_values=0)

    binary_label_matrix = binary_closing(
        binary_label_matrix, structure=mask_matrix, origin=0, iterations=1
    ).astype(int)

    return binary_label_matrix[
        this_num_cells:-this_num_cells,
        this_num_cells:-this_num_cells
    ]


def close_ternary_label_matrix(
        ternary_label_matrix, mask_matrix=None, buffer_distance_metres=None,
        grid_spacing_metres=None, tiebreaker_enum=COLD_FRONT_ENUM):
    """Closes gridded ternary ("no front, warm front, or cold front") labels.

    :param ternary_label_matrix: See doc for `dilate_ternary_label_matrix`.
    :param mask_matrix: Same.
    :param buffer_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :param tiebreaker_enum: Front type in case of a tie (must be accepted by
        `check_front_type_enum`).
    :return: ternary_label_matrix: Same as input but closed.
    """

    # TODO(thunderhoser): I still don't really understand the behaviour of
    # binary closing.  It generally does not do what I want.

    check_gridded_labels(label_matrix=ternary_label_matrix, assert_binary=False)

    new_wf_flag_matrix = (ternary_label_matrix == WARM_FRONT_ENUM).astype(int)
    new_wf_flag_matrix = close_binary_label_matrix(
        binary_label_matrix=new_wf_flag_matrix,
        mask_matrix=mask_matrix, buffer_distance_metres=buffer_distance_metres,
        grid_spacing_metres=grid_spacing_metres
    )

    new_cf_flag_matrix = (ternary_label_matrix == COLD_FRONT_ENUM).astype(int)
    new_cf_flag_matrix = close_binary_label_matrix(
        binary_label_matrix=new_cf_flag_matrix,
        mask_matrix=mask_matrix, buffer_distance_metres=buffer_distance_metres,
        grid_spacing_metres=grid_spacing_metres
    )

    return _break_wf_cf_ties(
        ternary_label_matrix=ternary_label_matrix,
        new_wf_flag_matrix=new_wf_flag_matrix,
        new_cf_flag_matrix=new_cf_flag_matrix, tiebreaker_enum=tiebreaker_enum)


def gridded_labels_to_points(ternary_label_matrix):
    """Converts gridded labels to two lists: warm- and cold-frontal points.

    C = number of pixels intersected by any cold front
    W = number of pixels intersected by any warm front

    :param ternary_label_matrix: See doc for `check_gridded_labels`.
    :return: gridded_label_dict: Dictionary with the following keys.
    gridded_label_dict['cold_front_rows']: length-C numpy array with row indices
        of cold-frontal pixels.
    gridded_label_dict['cold_front_columns']: length-C numpy array with column
        indices of cold-frontal pixels.
    gridded_label_dict['warm_front_rows']: Same but for warm fronts.
    gridded_label_dict['warm_front_columns']: Same but for warm fronts.
    """

    check_gridded_labels(label_matrix=ternary_label_matrix, assert_binary=False)

    warm_front_row_indices, warm_front_column_indices = numpy.where(
        ternary_label_matrix == WARM_FRONT_ENUM)
    cold_front_row_indices, cold_front_column_indices = numpy.where(
        ternary_label_matrix == COLD_FRONT_ENUM)

    return {
        WARM_FRONT_ROWS_COLUMN: warm_front_row_indices,
        WARM_FRONT_COLUMNS_COLUMN: warm_front_column_indices,
        COLD_FRONT_ROWS_COLUMN: cold_front_row_indices,
        COLD_FRONT_COLUMNS_COLUMN: cold_front_column_indices
    }


def points_to_gridded_labels(gridded_label_dict, num_grid_rows,
                             num_grid_columns):
    """Converts list of warm- and cold-frontal grid points to actual grid.

    M = number of rows in grid
    N = number of columns in grid

    :param gridded_label_dict: See doc for `gridded_labels_to_points`.
    :param num_grid_rows: M in the above discussion.
    :param num_grid_columns: N in the above discussion.
    :return: ternary_label_matrix: M-by-N numpy array of integers in range
        0...2.  ternary_label_matrix[i, j] indicates the *type* of front (warm,
        cold, or none) passing through grid cell [i, j].
    """

    ternary_image_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), NO_FRONT_ENUM, dtype=int)

    ternary_image_matrix[
        gridded_label_dict[WARM_FRONT_ROWS_COLUMN],
        gridded_label_dict[WARM_FRONT_COLUMNS_COLUMN]
    ] = WARM_FRONT_ENUM

    ternary_image_matrix[
        gridded_label_dict[COLD_FRONT_ROWS_COLUMN],
        gridded_label_dict[COLD_FRONT_COLUMNS_COLUMN]
    ] = COLD_FRONT_ENUM

    return ternary_image_matrix


def gridded_labels_to_regions(ternary_label_matrix, compute_lengths=False):
    """Converts gridded labels to list of connected regions.

    R = number of connected regions
    P_i = number of grid points in the [i]th region

    :param ternary_label_matrix: See doc for `check_gridded_labels`.
    :param compute_lengths: Boolean flag.  If True, will compute major-axis
        length for each region.

    :return: region_dict: Dictionary with the following keys.
    region_dict['row_array_by_region']: length-R list, where the [i]th element
        is an integer numpy array (length P_i) of rows in the [i]th region.
    region_dict['column_array_by_region']: Same but for columns.
    region_dict['front_type_strings']: length-R list of front types.

    If `compute_lengths == True`, also contains the following keys.

    region_dict['major_axis_lengths_px']: length-R numpy array of major-axis
        lengths (pixels).
    """

    error_checking.assert_is_boolean(compute_lengths)
    check_gridded_labels(label_matrix=ternary_label_matrix, assert_binary=False)

    region_id_matrix = label_image(ternary_label_matrix, connectivity=2)

    if compute_lengths:
        list_of_regionprop_objects = regionprops(region_id_matrix)
        # major_axis_lengths_px = numpy.array(
        #     [r.major_axis_length for r in list_of_regionprop_objects]
        # )

        num_regions = len(list_of_regionprop_objects)
        major_axis_lengths_px = numpy.full(num_regions, numpy.nan)

        for i in range(num_regions):
            these_rows, these_columns = numpy.where(
                list_of_regionprop_objects[i].convex_image)

            this_coord_matrix = numpy.hstack((
                numpy.reshape(these_columns, (these_columns.size, 1)),
                numpy.reshape(these_rows, (these_rows.size, 1))
            ))

            this_distance_matrix_px = euclidean_distances(
                X=this_coord_matrix.astype(float)
            )

            major_axis_lengths_px[i] = numpy.max(this_distance_matrix_px)

    num_regions = numpy.max(region_id_matrix)
    row_array_by_region = [[]] * num_regions
    column_array_by_region = [[]] * num_regions
    front_type_strings = [''] * num_regions

    for i in range(num_regions):
        row_array_by_region[i], column_array_by_region[i] = numpy.where(
            region_id_matrix == i + 1
        )

        this_integer_id = ternary_label_matrix[
            row_array_by_region[i][0], column_array_by_region[i][0]
        ]

        if this_integer_id == WARM_FRONT_ENUM:
            front_type_strings[i] = WARM_FRONT_STRING
        elif this_integer_id == COLD_FRONT_ENUM:
            front_type_strings[i] = COLD_FRONT_STRING

    region_dict = {
        ROWS_BY_REGION_KEY: row_array_by_region,
        COLUMNS_BY_REGION_KEY: column_array_by_region,
        FRONT_TYPES_KEY: front_type_strings
    }

    if compute_lengths:
        region_dict.update({MAJOR_AXIS_LENGTHS_KEY: major_axis_lengths_px})

    return region_dict


def polyline_to_narr_grid(vertex_latitudes_deg, vertex_longitudes_deg,
                          dilation_distance_metres):
    """Converts polyline to gridded labels on NARR grid.

    V = number of vertices
    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param vertex_latitudes_deg: length-V numpy array of latitudes (deg N).
    :param vertex_longitudes_deg: length-V numpy array of longitudes (deg E).
    :param dilation_distance_metres: Dilation distance for gridded labels.
    :return: binary_label_matrix: M-by-N numpy array of integers in range 0...1.
        label_matrix[i, j] indicates whether or not a front (of any type) passes
        through grid cell [i, j].
    """

    vertex_x_coords_metres, vertex_y_coords_metres = (
        nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=vertex_latitudes_deg,
            longitudes_deg=vertex_longitudes_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    grid_point_x_coords_metres, grid_point_y_coords_metres = (
        nwp_model_utils.get_xy_grid_points(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    )

    rows_in_front, columns_in_front = _polyline_to_grid_points(
        polyline_x_coords_metres=vertex_x_coords_metres,
        polyline_y_coords_metres=vertex_y_coords_metres,
        grid_point_x_coords_metres=grid_point_x_coords_metres,
        grid_point_y_coords_metres=grid_point_y_coords_metres)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    binary_label_matrix = _grid_points_to_boolean_matrix(
        rows_in_front=rows_in_front, columns_in_front=columns_in_front,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

    grid_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME
    )[0]

    return dilate_binary_label_matrix(
        binary_label_matrix=binary_label_matrix.astype(int),
        dilation_distance_metres=dilation_distance_metres,
        grid_spacing_metres=grid_spacing_metres)


def many_polylines_to_narr_grid(polyline_table, dilation_distance_metres):
    """For each time step, converts polylines to gridded labels on NARR grid.

    P = number of points in a given front
    C = number of pixels intersected by cold front at a given time
    W = number of pixels intersected by warm front at a given time

    :param polyline_table: pandas DataFrame with the following columns.  Each
        row is one front.
    polyline_table.front_type_string: Front type ("warm" or "cold").
    polyline_table.valid_time_unix_sec: Valid time.
    polyline_table.latitudes_deg: length-P numpy array of latitudes (deg N)
        along front.
    polyline_table.longitudes_deg: length-P numpy array of longitudes (deg E)
        along front.

    :param dilation_distance_metres: Dilation distance for gridded labels.
    :return: gridded_label_table: pandas DataFrame with the following columns.
        Each row is one valid time.
    gridded_label_table.cold_front_rows: length-C numpy array with row indices
        of cold-frontal pixels.
    gridded_label_table.cold_front_columns: Same but for columns.
    gridded_label_table.warm_front_rows: length-W numpy array with row indices
        of warm-frontal pixels.
    gridded_label_table.warm_front_columns: Same but for columns.
    gridded_label_table.valid_time_unix_sec: Valid time.
    """

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    valid_times_unix_sec = numpy.unique(polyline_table[TIME_COLUMN].values)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    num_times = len(valid_times_unix_sec)
    warm_front_rows_by_time = [[]] * num_times
    warm_front_columns_by_time = [[]] * num_times
    cold_front_rows_by_time = [[]] * num_times
    cold_front_columns_by_time = [[]] * num_times

    for i in range(num_times):
        print 'Converting polylines to NARR grid for {0:s}...'.format(
            valid_time_strings[i])

        these_polyline_indices = numpy.where(
            polyline_table[TIME_COLUMN].values == valid_times_unix_sec[i]
        )[0]

        this_ternary_label_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), NO_FRONT_ENUM, dtype=int)

        for j in these_polyline_indices:
            skip_this_front = _is_polyline_closed(
                x_coords_metres=polyline_table[LONGITUDES_COLUMN].values[j],
                y_coords_metres=polyline_table[LATITUDES_COLUMN].values[j]
            )

            if skip_this_front:
                this_num_points = len(
                    polyline_table[LATITUDES_COLUMN].values[j]
                )

                print (
                    'SKIPPING front with {0:d} points (closed polyline).'
                ).format(this_num_points)

                continue

            this_binary_label_matrix = polyline_to_narr_grid(
                vertex_latitudes_deg=polyline_table[LATITUDES_COLUMN].values[j],
                vertex_longitudes_deg=polyline_table[
                    LONGITUDES_COLUMN].values[j],
                dilation_distance_metres=dilation_distance_metres
            )

            this_binary_label_matrix = this_binary_label_matrix.astype(bool)
            this_front_type_string = polyline_table[FRONT_TYPE_COLUMN].values[j]

            if this_front_type_string == WARM_FRONT_STRING:
                this_ternary_label_matrix[
                    numpy.where(this_binary_label_matrix)
                ] = WARM_FRONT_ENUM
            else:
                this_ternary_label_matrix[
                    numpy.where(this_binary_label_matrix)
                ] = COLD_FRONT_ENUM

        this_gridded_label_dict = gridded_labels_to_points(
            this_ternary_label_matrix)

        warm_front_rows_by_time[i] = this_gridded_label_dict[
            WARM_FRONT_ROWS_COLUMN]
        warm_front_columns_by_time[i] = this_gridded_label_dict[
            WARM_FRONT_COLUMNS_COLUMN]
        cold_front_rows_by_time[i] = this_gridded_label_dict[
            COLD_FRONT_ROWS_COLUMN]
        cold_front_columns_by_time[i] = this_gridded_label_dict[
            COLD_FRONT_COLUMNS_COLUMN]

    gridded_label_dict = {
        TIME_COLUMN: valid_times_unix_sec,
        WARM_FRONT_ROWS_COLUMN: warm_front_rows_by_time,
        WARM_FRONT_COLUMNS_COLUMN: warm_front_columns_by_time,
        COLD_FRONT_ROWS_COLUMN: cold_front_rows_by_time,
        COLD_FRONT_COLUMNS_COLUMN: cold_front_columns_by_time
    }

    return pandas.DataFrame.from_dict(gridded_label_dict)


def remove_fronts_in_masked_area(
        polyline_table, narr_mask_matrix, verbose=True):
    """Removes fronts in masked area (polylines that touch only masked pixels).

    M = number of rows in NARR grid
    N = number of columns in NARR grid

    :param polyline_table: See doc for `many_polylines_to_narr_grid`.
    :param narr_mask_matrix: M-by-N numpy array of integers in range 0...1.  If
        narr_mask_matrix[i, j] = 1, grid cell [i, j] is masked.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: polyline_table: Same as input, but some rows may have been removed.
    """

    error_checking.assert_is_integer_numpy_array(narr_mask_matrix)
    error_checking.assert_is_geq_numpy_array(narr_mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(narr_mask_matrix, 1)
    error_checking.assert_is_boolean(verbose)

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    these_expected_dim = numpy.array(
        [num_grid_rows, num_grid_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        narr_mask_matrix, exact_dimensions=these_expected_dim)

    num_fronts = len(polyline_table.index)
    bad_indices = []

    for i in range(num_fronts):
        skip_this_front = _is_polyline_closed(
            x_coords_metres=polyline_table[LONGITUDES_COLUMN].values[i],
            y_coords_metres=polyline_table[LATITUDES_COLUMN].values[i]
        )

        if skip_this_front:
            bad_indices.append(i)
            continue

        this_binary_label_matrix = polyline_to_narr_grid(
            vertex_latitudes_deg=polyline_table[LATITUDES_COLUMN].values[i],
            vertex_longitudes_deg=polyline_table[LONGITUDES_COLUMN].values[i],
            dilation_distance_metres=0.
        )

        found_unmasked_frontal_pixel = numpy.any(numpy.logical_and(
            this_binary_label_matrix == 1, narr_mask_matrix == 1
        ))

        if not found_unmasked_frontal_pixel:
            bad_indices.append(i)

    if verbose:
        print (
            'Removed {0:d} of {1:d} polylines (because they touch only masked '
            'grid cells).'
        ).format(len(bad_indices), num_fronts)

    if len(bad_indices) == 0:
        return polyline_table

    bad_indices = numpy.array(bad_indices, dtype=int)
    return polyline_table.drop(
        polyline_table.index[bad_indices], axis=0, inplace=False)
