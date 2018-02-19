"""Finds main skeleton line of polygon.

This module uses the backtracking algorithm from Wang et al. (2013).

--- REFERENCES ---

Wang, Z., H. Yan, and Y. Yang, 2013: "A research on automatic extraction of main
    skeleton lines of polygons". Communications in Information Science and
    Management Engineering, 3 (4), 175-180.
"""

import copy
import numpy
import pandas
import scipy.spatial
import shapely.geometry
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6

VERTEX_INDICES_KEY = 'vertex_indices'
TRIANGLE_INDICES_KEY = 'triangle_indices'
TRIANGLE_POSITIONS_KEY = 'triangle_positions'
NEW_EDGE_INDICES_KEY = 'new_edge_indices'

END_NODE_TYPE = 'end_node'
BRANCH_NODE_TYPE = 'branch_node'
JUMPER_NODE_TYPE = 'jumper_node'
LEFT_CHILD_TYPE = 'left'
RIGHT_CHILD_TYPE = 'right'

NORMAL_STEP_TYPE = 'left_child_step'
ABNORMAL_STEP_TYPE = 'right_child_or_backtrack_step'

NODE_TYPE_KEY = 'node_type'
NODE_X_COORDS_KEY = 'node_x_coordinates'
NODE_Y_COORDS_KEY = 'node_y_coordinates'
NODE_INDICES_KEY = 'node_indices'

LEFT_CHILD_INDICES_KEY = 'left_child_indices'
RIGHT_CHILD_INDICES_KEY = 'right_child_indices'


def _polygon_to_vertex_arrays(polygon_object_xy):
    """Converts polygon to vertex arrays.

    If the polygon is explicitly closed (i.e., first and last vertices are the
    same), this method will remove the last vertex, since it confuses Delaunay
    triangulation.

    V = number of vertices (after removing closure)

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :return: vertex_x_coords: length-V numpy array with x-coordinates of
        vertices.
    :return: vertex_y_coords: length-V numpy array with y-coordinates of
        vertices.
    """

    vertex_x_coords = numpy.array(polygon_object_xy.exterior.xy[0])
    vertex_y_coords = numpy.array(polygon_object_xy.exterior.xy[1])

    wraparound_x_diff = numpy.absolute(vertex_x_coords[0] - vertex_x_coords[-1])
    wraparound_y_diff = numpy.absolute(vertex_y_coords[0] - vertex_y_coords[-1])
    if wraparound_x_diff < TOLERANCE and wraparound_y_diff < TOLERANCE:
        vertex_x_coords = vertex_x_coords[:-1]
        vertex_y_coords = vertex_y_coords[:-1]

    return vertex_x_coords, vertex_y_coords


def _are_vertices_adjacent(vertex_indices, num_vertices_in_polygon):
    """Determines whether or not two vertices are adjacent.

    :param vertex_indices: length-2 numpy array of vertex indices.  For example,
        if vertex_indices = [3, 6], these are the 3rd and 6th vertices in the
        original polygon.
    :param num_vertices_in_polygon: Number of vertices in polygon.
    :return: adjacent_flag: Boolean flag.
    """

    index_diff = numpy.absolute(vertex_indices[1] - vertex_indices[0])
    if index_diff == 1:
        return True

    return (0 in vertex_indices and
            num_vertices_in_polygon - 1 in vertex_indices)


def _remove_delaunay_triangles_outside_polygon(
        polygon_object_xy, triangulation_object):
    """Removes Delaunay triangles outside of original polygon.

    N = number of triangles remaining

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :param triangulation_object: Instance of `scipy.spatial.qhull.Delaunay`.
    :return: triangle_to_vertex_matrix: N-by-3 numpy array, where
        triangle_to_vertex_matrix[i, j] is the index of the [j]th vertex in the
        [i]th triangle.  Thus, if triangle_to_vertex_matrix[i, j] = k, the [j]th
        vertex in the [i]th triangle is the [k]th vertex in the original
        polygon.
    """

    vertex_x_coords, vertex_y_coords = _polygon_to_vertex_arrays(
        polygon_object_xy)

    triangle_to_vertex_matrix = triangulation_object.simplices
    num_triangles = triangle_to_vertex_matrix.shape[0]
    keep_triangle_flags = numpy.full(num_triangles, True, dtype=bool)

    for i in range(num_triangles):
        for j in range(3):
            if j == 2:
                these_vertex_indices = numpy.array(
                    [triangle_to_vertex_matrix[i, j],
                     triangle_to_vertex_matrix[i, 0]], dtype=int)
            else:
                these_vertex_indices = numpy.array(
                    [triangle_to_vertex_matrix[i, j],
                     triangle_to_vertex_matrix[i, j + 1]], dtype=int)

            if _are_vertices_adjacent(
                    vertex_indices=these_vertex_indices,
                    num_vertices_in_polygon=len(vertex_x_coords)):
                continue

            # TODO(thunderhoser): stop using protected method.
            this_vertex_list_xy = front_utils._vertex_arrays_to_list(
                vertex_x_coords_metres=vertex_x_coords[these_vertex_indices],
                vertex_y_coords_metres=vertex_y_coords[these_vertex_indices])

            this_linestring_object_xy = shapely.geometry.LineString(
                this_vertex_list_xy)
            if polygon_object_xy.contains(this_linestring_object_xy):
                continue

            keep_triangle_flags[i] = False
            break

    keep_triangle_indices = numpy.where(keep_triangle_flags)[0]
    return triangle_to_vertex_matrix[keep_triangle_indices, :]


def _get_delaunay_triangulation(polygon_object_xy):
    """Returns Delaunay triangulation of polygon.

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :return: triangulation_object: Instance of `scipy.spatial.qhull.Delaunay`.
    """

    vertex_x_coords, vertex_y_coords = _polygon_to_vertex_arrays(
        polygon_object_xy)

    num_vertices = len(vertex_x_coords)
    vertex_matrix_xy = numpy.hstack((
        numpy.reshape(vertex_x_coords, (num_vertices, 1)),
        numpy.reshape(vertex_y_coords, (num_vertices, 1))))

    triangulation_object = scipy.spatial.Delaunay(
        vertex_matrix_xy, qhull_options='QJ')
    return _remove_delaunay_triangles_outside_polygon(
        polygon_object_xy=polygon_object_xy,
        triangulation_object=triangulation_object)


def _find_new_edges_from_triangulation(
        polygon_object_xy, triangle_to_vertex_matrix):
    """Finds new edges created by Delaunay triangulation.

    A "new edge" does not coincide with an edge in the original polygon.  Wang
    et al. (2013) call these "jumper edges," and the midpoint of a jumper edge
    may be either a "branch node" or a "jumper node".

    T = number of triangles to which a given edge belongs

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :param triangle_to_vertex_matrix: See documentation for
        `_remove_delaunay_triangles_outside_polygon`.
    :return: new_edge_table: pandas DataFrame with the following columns.  Each
        row is one new edge.
    new_edge_table.vertex_indices: Indices of polygon vertices that belong to
        this edge.
    new_edge_table.triangle_indices: length-T numpy array with indices of
        triangles to which this edge belongs.
    new_edge_table.triangle_positions: length-T numpy array with positions of
        edge in triangles.  0 means that the edge connects vertices 0 and 1 (the
        first and second); 1 means that the edge connects vertices 1 and 2; 2
        means that the edge connects vertices 2 and 0.

    :return: triangle_to_new_edge_table: pandas DataFrame with the following
        columns.  Each row is one triangle.
    triangle_to_new_edge_table.new_edge_indices: 1-D numpy array with indices of
        new edges belonging to triangle.  These are row indices in
        `new_edge_table`.
    """

    vertex_x_coords, _ = _polygon_to_vertex_arrays(polygon_object_xy)
    num_vertices = len(vertex_x_coords)

    num_triangles = triangle_to_vertex_matrix.shape[0]
    triangle_to_new_edge_index_list = [
        numpy.array([], dtype=int)] * num_triangles

    new_edge_to_vertex_matrix = None
    new_edge_to_triangle_index_list = []
    new_edge_to_triangle_position_list = []

    for i in range(num_triangles):
        for j in range(3):
            if j == 2:
                these_vertex_indices = numpy.array(
                    [triangle_to_vertex_matrix[i, j],
                     triangle_to_vertex_matrix[i, 0]], dtype=int)
            else:
                these_vertex_indices = numpy.array(
                    [triangle_to_vertex_matrix[i, j],
                     triangle_to_vertex_matrix[i, j + 1]], dtype=int)

            if _are_vertices_adjacent(
                    vertex_indices=these_vertex_indices,
                    num_vertices_in_polygon=num_vertices):
                continue

            if new_edge_to_vertex_matrix is None:
                this_new_edge_indices = numpy.array([])
            else:
                num_new_edges_found = new_edge_to_vertex_matrix.shape[0]
                this_new_edge_flags = numpy.array(
                    [set(these_vertex_indices.tolist()) ==
                     set(new_edge_to_vertex_matrix[k, :].tolist())
                     for k in range(num_new_edges_found)])

                this_new_edge_indices = numpy.where(this_new_edge_flags)[0]

            if len(this_new_edge_indices):
                this_new_edge_index = this_new_edge_indices[0]
                new_edge_to_triangle_index_list[this_new_edge_index].append(i)
                new_edge_to_triangle_position_list[
                    this_new_edge_index].append(j)

            else:
                if new_edge_to_vertex_matrix is None:
                    new_edge_to_vertex_matrix = numpy.reshape(
                        these_vertex_indices, (1, 2))
                else:
                    new_edge_to_vertex_matrix = numpy.vstack((
                        new_edge_to_vertex_matrix, these_vertex_indices))

                new_edge_to_triangle_index_list.append([i])
                new_edge_to_triangle_position_list.append([j])
                this_new_edge_index = new_edge_to_vertex_matrix.shape[0] - 1

            triangle_to_new_edge_index_list[i] = numpy.concatenate((
                triangle_to_new_edge_index_list[i],
                numpy.array([this_new_edge_index])))

    new_edge_dict = {
        VERTEX_INDICES_KEY: new_edge_to_vertex_matrix.tolist(),
        TRIANGLE_INDICES_KEY: new_edge_to_triangle_index_list,
        TRIANGLE_POSITIONS_KEY: new_edge_to_triangle_position_list
    }
    triangle_to_new_edge_dict = {
        NEW_EDGE_INDICES_KEY: triangle_to_new_edge_index_list
    }

    return (pandas.DataFrame.from_dict(new_edge_dict),
            pandas.DataFrame.from_dict(triangle_to_new_edge_dict))


def _find_end_nodes_of_triangulation(triangle_to_vertex_matrix, new_edge_table):
    """Finds end nodes of Delaunay triangulation.

    :param triangle_to_vertex_matrix: See documentation for
        `_remove_delaunay_triangles_outside_polygon`.
    :param new_edge_table: See documentation for
        `_find_new_edges_from_triangulation`.
    :return: end_node_vertex_indices: 1-D numpy array with indices of polygon
        vertices that are also end nodes.
    """

    end_node_vertex_indices = numpy.reshape(
        triangle_to_vertex_matrix,
        triangle_to_vertex_matrix.size).astype(int).tolist()
    end_node_vertex_indices = list(set(end_node_vertex_indices))

    new_edge_vertex_indices = numpy.array([], dtype=int)
    for i in range(len(new_edge_table.index)):
        new_edge_vertex_indices = numpy.concatenate((
            new_edge_vertex_indices,
            new_edge_table[VERTEX_INDICES_KEY].values[i]))

    new_edge_vertex_indices = numpy.unique(new_edge_vertex_indices)

    for i in new_edge_vertex_indices:
        if i in end_node_vertex_indices:
            end_node_vertex_indices.remove(i)

    return numpy.array(end_node_vertex_indices)


def _classify_branch_or_jumper_node(
        triangle_to_new_edge_table, target_edge_index,
        target_edge_to_triangle_indices):
    """Classifies node as either a branch or a jumper.

    Definitions of "branch node" and "jumper node" are as in Wang et al. (2013).

    :param triangle_to_new_edge_table: See doc for
        `_find_new_edges_from_triangulation`.
    :param target_edge_index: Index of new edge to be classified.
    :param target_edge_to_triangle_indices: 1-D numpy array with indices of
        triangles to which target edge belongs.  These indices are rows in
        `triangle_to_new_edge_table`.
    :return: node_type: Either "branch" or "jumper".
    """

    new_edges_connected_to_target = numpy.array([], dtype=int)
    for i in target_edge_to_triangle_indices:
        new_edges_connected_to_target = numpy.concatenate((
            new_edges_connected_to_target,
            triangle_to_new_edge_table[NEW_EDGE_INDICES_KEY].values[i]))

    new_edges_connected_to_target = numpy.unique(
        new_edges_connected_to_target).tolist()
    if target_edge_index in new_edges_connected_to_target:
        new_edges_connected_to_target.remove(target_edge_index)

    if len(new_edges_connected_to_target) >= 3:
        return BRANCH_NODE_TYPE
    return JUMPER_NODE_TYPE


def _find_and_classify_nodes(
        polygon_object_xy, new_edge_table, triangle_to_new_edge_table,
        triangle_to_vertex_matrix, end_node_vertex_indices):
    """Finds and classifies all nodes in Delaunay triangulation.

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :param new_edge_table: See documentation for
        `_find_new_edges_from_triangulation`.
    :param triangle_to_new_edge_table: See doc for
        `_find_new_edges_from_triangulation`.
    :param triangle_to_vertex_matrix: See doc for
        `_remove_delaunay_triangles_outside_polygon`.
    :param end_node_vertex_indices: See doc for
        `_find_end_nodes_of_triangulation`.
    :return: node_table: pandas DataFrame with the following columns.
    node_table.node_type: Type of node ("branch", "jumper", or "end").
    node_table.x_coordinate: x-coordinate of node.
    node_table.y_coordinate: y-coordinate of node.
    node_table.triangle_indices: length-T numpy array with indices of all
        triangles touched by node.  There are row indices into the matrix
        `triangle_to_vertex_matrix`.
    node_table.triangle_positions: length-T numpy array, indicating the node's
        position in each triangle that it touches.  0 means that the node is on
        the edge connecting vertices 0 and 1 (the first and second); 1 means
        on the edge connecting vertices 1 and 2; 2 means on the edge connecting
        vertices 2 and 0.

    :return triangle_to_node_table: pandas DataFrame with the following
        columns.  Each row is one triangle.
    triangle_to_node_table.node_indices: 1-D numpy array with indices of nodes
        belonging to triangle.  These are row indices in `node_table`.
    """

    num_end_nodes = len(end_node_vertex_indices)
    num_new_edges = len(new_edge_table.index)
    num_nodes = num_end_nodes + num_new_edges

    node_types = [''] * num_nodes
    node_x_coords = numpy.full(num_nodes, numpy.nan)
    node_y_coords = numpy.full(num_nodes, numpy.nan)
    triangle_indices_2d_list = [[]] * num_nodes
    triangle_positions_2d_list = [[]] * num_nodes

    vertex_x_coords, vertex_y_coords = _polygon_to_vertex_arrays(
        polygon_object_xy)

    for i in range(num_new_edges):
        node_types[i] = _classify_branch_or_jumper_node(
            triangle_to_new_edge_table=triangle_to_new_edge_table,
            target_edge_index=i,
            target_edge_to_triangle_indices=
            new_edge_table[TRIANGLE_INDICES_KEY].values[i])

        these_vertex_indices = new_edge_table[VERTEX_INDICES_KEY].values[i]
        node_x_coords[i] = numpy.mean(vertex_x_coords[these_vertex_indices])
        node_y_coords[i] = numpy.mean(vertex_y_coords[these_vertex_indices])
        triangle_indices_2d_list[i] = new_edge_table[
            TRIANGLE_INDICES_KEY].values[i]
        triangle_positions_2d_list[i] = new_edge_table[
            TRIANGLE_POSITIONS_KEY].values[i]

    triangle_to_node_table = copy.deepcopy(triangle_to_new_edge_table)
    triangle_to_node_table.rename(
        columns={NEW_EDGE_INDICES_KEY: NODE_INDICES_KEY}, inplace=True)

    for i in range(num_end_nodes):
        this_node_index = i + num_new_edges

        node_types[this_node_index] = END_NODE_TYPE
        node_x_coords[this_node_index] = vertex_x_coords[
            end_node_vertex_indices[i]]
        node_y_coords[this_node_index] = vertex_y_coords[
            end_node_vertex_indices[i]]

        these_triangle_indices = numpy.unique(numpy.where(
            triangle_to_vertex_matrix == end_node_vertex_indices[i])[0])
        triangle_indices_2d_list[this_node_index] = these_triangle_indices

        for j in these_triangle_indices:
            this_triangle_node_indices = triangle_to_node_table[
                NODE_INDICES_KEY].values[j]
            if this_node_index in this_triangle_node_indices:
                continue

            triangle_to_node_table[NODE_INDICES_KEY].values[j] = (
                numpy.concatenate((this_triangle_node_indices,
                                   numpy.array([this_node_index]))))

    node_dict = {
        NODE_TYPE_KEY: node_types,
        NODE_X_COORDS_KEY: node_x_coords,
        NODE_Y_COORDS_KEY: node_y_coords,
        TRIANGLE_INDICES_KEY: triangle_indices_2d_list,
        TRIANGLE_POSITIONS_KEY: triangle_positions_2d_list
    }
    return pandas.DataFrame.from_dict(node_dict), triangle_to_node_table


def _find_and_classify_node_children(
        node_table, triangle_to_new_edge_table, triangle_to_node_table):
    """Finds and classifies children for each node.

    Each child may be classified as either "left" or "right".

    :param node_table: See documentation for `_find_and_classify_nodes`.
    :param triangle_to_new_edge_table: See doc for
        `_find_new_edges_from_triangulation`.
    :param triangle_to_node_table: See doc for `_find_and_classify_nodes`.
    :return: node_table: Same as input, but with two extra columns.
    node_table.left_child_indices: 1-D numpy array with indices of left
        children.  If left_child_indices[i] -- the entry for
        `left_child_indices` in the [i]th row -- contains j, this means that the
        [j]th node is a left child of the [i]th node.
    node_table.right_child_indices: Same as above, except for right children.
    """

    num_nodes = len(node_table.index)
    node_to_left_child_indices = [numpy.array([], dtype=int)] * num_nodes
    node_to_right_child_indices = [numpy.array([], dtype=int)] * num_nodes

    for parent_index in range(num_nodes):
        parent_triangle_indices = node_table[TRIANGLE_INDICES_KEY].values[
            parent_index]

        for j in range(len(parent_triangle_indices)):
            these_node_indices = triangle_to_node_table[
                NODE_INDICES_KEY].values[parent_triangle_indices[j]]

            for child_index in these_node_indices:
                if child_index == parent_index:
                    continue  # A child cannot be its own parent.

                # Jumper and end nodes can have only left children.
                if (node_table[NODE_TYPE_KEY].values[parent_index] in
                        [JUMPER_NODE_TYPE, END_NODE_TYPE]):
                    node_to_left_child_indices[parent_index] = (
                        numpy.concatenate((
                            node_to_left_child_indices[parent_index],
                            numpy.array([child_index]))))
                    continue

                # Triangle T shares an edge with the polygon, so the current
                # node (which must be a branch node) can have only left children
                # through T.
                these_new_edge_indices = triangle_to_new_edge_table[
                    NEW_EDGE_INDICES_KEY].values[parent_triangle_indices[j]]
                if len(these_new_edge_indices) < 3:
                    node_to_left_child_indices[parent_index] = (
                        numpy.concatenate((
                            node_to_left_child_indices[parent_index],
                            numpy.array([child_index]))))
                    continue

                # Triangle shares no edge with the polygon, so the current node
                # (which must be a branch node) can have either left or right
                # children through T.  Direction depends on positions of child
                # and parent nodes in T.
                parent_triangle_position = node_table[
                    TRIANGLE_POSITIONS_KEY].values[parent_index][j]
                triangle_index_in_child_array = numpy.where(numpy.array(
                    node_table[TRIANGLE_INDICES_KEY].values[child_index]
                ) == parent_triangle_indices[j])[0][0]
                child_triangle_position = node_table[
                    TRIANGLE_POSITIONS_KEY].values[child_index][
                        triangle_index_in_child_array]

                child_minus_parent_position = (
                    child_triangle_position - parent_triangle_position)

                if child_minus_parent_position in [1, -2]:
                    node_to_right_child_indices[parent_index] = (
                        numpy.concatenate((
                            node_to_right_child_indices[parent_index],
                            numpy.array([child_index]))))
                else:
                    node_to_left_child_indices[parent_index] = (
                        numpy.concatenate((
                            node_to_left_child_indices[parent_index],
                            numpy.array([child_index]))))

    argument_dict = {
        LEFT_CHILD_INDICES_KEY: node_to_left_child_indices,
        RIGHT_CHILD_INDICES_KEY: node_to_right_child_indices
    }
    return node_table.assign(**argument_dict)


def _delete_last_node_added(
        node_table, used_node_indices, used_triangle_indices):
    """Deletes last node added to skeleton line.

    This method implements the "backtracking step".

    :param node_table: pandas DataFrame created by
        `_find_and_classify_node_children`.
    :param used_node_indices: 1-D list with indices of nodes added to skeleton
        line.
    :param used_triangle_indices: 1-D list with indices of triangles crossed by
        skeleton line.
    :return: node_table: Same as input, except that the node removed is no
        longer considered a child of its predecessor in the skeleton line.
    :return: used_node_indices: Same as input, but without last element.
    :return: used_triangle_indices: Same as input, but without last element.
    """

    predecessor_left_child_indices = list(
        node_table[LEFT_CHILD_INDICES_KEY].values[used_node_indices[-2]])
    predecessor_right_child_indices = list(
        node_table[RIGHT_CHILD_INDICES_KEY].values[used_node_indices[-2]])

    if used_node_indices[-1] in predecessor_left_child_indices:
        predecessor_left_child_indices.remove(used_node_indices[-1])
    if used_node_indices[-1] in predecessor_right_child_indices:
        predecessor_right_child_indices.remove(used_node_indices[-1])

    node_table[LEFT_CHILD_INDICES_KEY].values[
        used_node_indices[-2]] = predecessor_left_child_indices
    node_table[RIGHT_CHILD_INDICES_KEY].values[
        used_node_indices[-2]] = predecessor_right_child_indices

    used_node_indices.remove(used_node_indices[-1])
    used_triangle_indices.remove(used_triangle_indices[-1])

    return node_table, used_node_indices, used_triangle_indices


def _get_skeleton_line(
        node_table, triangle_to_node_table, start_node_index, end_node_index):
    """Finds one skeleton line through a polygon.

    This method uses the "backtracking algorithm" of Wang et al. (2013).

    Keep in mind that, in Wang et al. (2013), both start and end nodes are
    called "end nodes".  They are also classified as such in `node_table`.

    P = number of points in resulting skeleton line

    :param node_table: pandas DataFrame created by
        `_find_and_classify_node_children`.
    :param triangle_to_node_table: pandas DataFrame created by
        `_find_and_classify_nodes`.
    :param start_node_index: Index of start node.  This is a row index into
        `node_table`.
    :param end_node_index: Index of end node.  This is a row index into
        `node_table`.
    :return: skeleton_line_x_coords: length-P numpy array of x-coordinates along
        skeleton line.
    :return: skeleton_line_y_coords: length-P numpy array of y-coordinates along
        skeleton line.
    """

    # Go from start node to its left child.
    previous_node_index = node_table[LEFT_CHILD_INDICES_KEY].values[
        start_node_index][0]
    last_triangle_used_index = node_table[TRIANGLE_INDICES_KEY].values[
        start_node_index][0]

    used_node_indices = [start_node_index, previous_node_index]
    used_triangle_indices = [last_triangle_used_index]

    current_step_type = copy.deepcopy(NORMAL_STEP_TYPE)
    force_delete_prev_node = False

    while True:
        if current_step_type == NORMAL_STEP_TYPE:

            # Determine which triangle to cut across.
            candidate_triangle_indices = list(
                node_table[TRIANGLE_INDICES_KEY].values[used_node_indices[-1]])

            if (len(used_triangle_indices)
                    and used_triangle_indices[-1] in
                    candidate_triangle_indices):
                candidate_triangle_indices.remove(used_triangle_indices[-1])

            if not len(candidate_triangle_indices):
                current_step_type = copy.deepcopy(ABNORMAL_STEP_TYPE)
                force_delete_prev_node = True
                continue

            current_triangle_index = candidate_triangle_indices[0]

            # Find candidates for next node.
            candidate_next_node_indices = list(
                triangle_to_node_table[NODE_INDICES_KEY].values[
                    current_triangle_index])
            candidate_next_node_indices = list(
                set(candidate_next_node_indices).difference(
                    set(used_node_indices)))

            if not len(candidate_next_node_indices):
                current_step_type = copy.deepcopy(ABNORMAL_STEP_TYPE)
                force_delete_prev_node = True
                continue

            # If desired end node is a candidate, go there and stop algorithm.
            if end_node_index in candidate_next_node_indices:
                used_node_indices.append(end_node_index)
                used_triangle_indices.append(current_triangle_index)
                break

            # If undesired end node is a candidate, backtrack.
            candidate_next_node_types = node_table[NODE_TYPE_KEY].values[
                numpy.array(candidate_next_node_indices)]
            if END_NODE_TYPE in candidate_next_node_types:
                current_step_type = copy.deepcopy(ABNORMAL_STEP_TYPE)
                force_delete_prev_node = True
                continue

            # Narrow down candidates to left children of previous node.
            prev_left_child_indices = list(
                node_table[LEFT_CHILD_INDICES_KEY].values[
                    used_node_indices[-1]])
            candidate_next_node_indices = list(
                set(candidate_next_node_indices).intersection(
                    set(prev_left_child_indices)))

            if not len(candidate_next_node_indices):
                # node_table, used_node_indices, used_triangle_indices = (
                #     _delete_last_node_added(
                #         node_table=node_table,
                #         used_node_indices=used_node_indices,
                #         used_triangle_indices=used_triangle_indices))

                current_step_type = copy.deepcopy(ABNORMAL_STEP_TYPE)
                force_delete_prev_node = True
                continue

            used_node_indices.append(candidate_next_node_indices[0])
            used_triangle_indices.append(current_triangle_index)

        else:
            previous_node_type = node_table[NODE_TYPE_KEY].values[
                used_node_indices[-1]]

            if previous_node_type == JUMPER_NODE_TYPE or force_delete_prev_node:
                node_table, used_node_indices, used_triangle_indices = (
                    _delete_last_node_added(
                        node_table=node_table,
                        used_node_indices=used_node_indices,
                        used_triangle_indices=used_triangle_indices))

                force_delete_prev_node = False
                continue

            if len(used_node_indices) == 1:
                current_step_type = copy.deepcopy(NORMAL_STEP_TYPE)
                continue

            # Determine which triangle to cut across.
            candidate_triangle_indices = list(
                node_table[TRIANGLE_INDICES_KEY].values[used_node_indices[-1]])
            if used_triangle_indices[-1] in candidate_triangle_indices:
                candidate_triangle_indices.remove(used_triangle_indices[-1])

            if not len(candidate_triangle_indices):
                force_delete_prev_node = True
                continue

            current_triangle_index = candidate_triangle_indices[0]

            # Find candidates for next node.
            candidate_next_node_indices = list(
                triangle_to_node_table[NODE_INDICES_KEY].values[
                    current_triangle_index])
            candidate_next_node_indices = list(
                set(candidate_next_node_indices).difference(
                    set(used_node_indices)))

            # Narrow down candidates to right children of previous node.
            prev_right_child_indices = list(
                node_table[RIGHT_CHILD_INDICES_KEY].values[
                    used_node_indices[-1]])
            candidate_next_node_indices = list(
                set(candidate_next_node_indices).intersection(
                    set(prev_right_child_indices)))

            if not len(candidate_next_node_indices):
                force_delete_prev_node = True
                continue

            used_node_indices.append(candidate_next_node_indices[0])
            used_triangle_indices.append(current_triangle_index)
            current_step_type = copy.deepcopy(NORMAL_STEP_TYPE)

    used_node_indices = numpy.array(used_node_indices)
    return (node_table[NODE_X_COORDS_KEY].values[used_node_indices],
            node_table[NODE_Y_COORDS_KEY].values[used_node_indices])


def _get_convex_hull(vertex_x_coords, vertex_y_coords):
    """Finds convex hull of polygon.

    V = number of vertices

    :param vertex_x_coords: length-V numpy array with x-coordinates of vertices.
    :param vertex_y_coords: length-V numpy array with y-coordinates of vertices.
    :return: convex_hull_indices: 1-D numpy array with indices of vertices on
        convex hull.
    """

    num_vertices = len(vertex_x_coords)
    vertex_indices = numpy.linspace(
        0, num_vertices - 1, num=num_vertices, dtype=int)

    if num_vertices == 2:
        return vertex_indices

    vertex_matrix_xy = numpy.hstack((
        numpy.reshape(vertex_x_coords, (num_vertices, 1)),
        numpy.reshape(vertex_y_coords, (num_vertices, 1))))
    convex_hull_object = scipy.spatial.ConvexHull(vertex_matrix_xy)

    return vertex_indices[convex_hull_object.vertices]


def _remove_node_from_children(node_table, target_node_index):
    """Removes a single node from all child lists.

    :param node_table: pandas DataFrame created by
        `_find_and_classify_node_children`.
    :param target_node_index: Index of node to remove from child lists.
    :return: node_table: Same as input, except with different children.
    """

    num_nodes = len(node_table.index)

    for i in range(num_nodes):
        these_left_child_indices = list(
            node_table[LEFT_CHILD_INDICES_KEY].values[i])
        if target_node_index in these_left_child_indices:
            these_left_child_indices.remove(target_node_index)
            node_table[LEFT_CHILD_INDICES_KEY].values[
                i] = these_left_child_indices

        these_right_child_indices = list(
            node_table[RIGHT_CHILD_INDICES_KEY].values[i])
        if target_node_index in these_right_child_indices:
            these_right_child_indices.remove(target_node_index)
            node_table[RIGHT_CHILD_INDICES_KEY].values[
                i] = these_right_child_indices

    return node_table


def get_main_skeleton_line(polygon_object_xy):
    """Finds the main (longest) skeleton line passing through a polygon.

    P = number of points in resulting skeleton line

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :return: skeleton_line_x_coords: length-P numpy array of x-coordinates along
        skeleton line.
    :return: skeleton_line_y_coords: length-P numpy array of y-coordinates along
        skeleton line.
    """

    triangle_to_vertex_matrix = _get_delaunay_triangulation(polygon_object_xy)
    new_edge_table, triangle_to_new_edge_table = (
        _find_new_edges_from_triangulation(
            polygon_object_xy=polygon_object_xy,
            triangle_to_vertex_matrix=triangle_to_vertex_matrix))

    end_node_vertex_indices = _find_end_nodes_of_triangulation(
        triangle_to_vertex_matrix=triangle_to_vertex_matrix,
        new_edge_table=new_edge_table)

    node_table, triangle_to_node_table = _find_and_classify_nodes(
        polygon_object_xy=polygon_object_xy, new_edge_table=new_edge_table,
        triangle_to_new_edge_table=triangle_to_new_edge_table,
        triangle_to_vertex_matrix=triangle_to_vertex_matrix,
        end_node_vertex_indices=end_node_vertex_indices)

    node_table = _find_and_classify_node_children(
        node_table=node_table,
        triangle_to_new_edge_table=triangle_to_new_edge_table,
        triangle_to_node_table=triangle_to_node_table)

    end_node_flags = numpy.array(
        [s == END_NODE_TYPE for s in node_table[NODE_TYPE_KEY].values])
    end_node_indices = numpy.where(end_node_flags)[0]

    vertex_x_coords, vertex_y_coords = _polygon_to_vertex_arrays(
        polygon_object_xy)

    convex_hull_indices = _get_convex_hull(
        vertex_x_coords[end_node_vertex_indices],
        vertex_y_coords[end_node_vertex_indices])
    end_node_in_convex_hull_indices = end_node_indices[convex_hull_indices]

    num_end_nodes_in_convex_hull = len(end_node_in_convex_hull_indices)
    max_skeleton_line_length = -1.
    skeleton_line_x_coords = None
    skeleton_line_y_coords = None

    num_skeleton_lines = (
        num_end_nodes_in_convex_hull * (num_end_nodes_in_convex_hull - 1) / 2)
    num_skeleton_lines_done = 0

    for i in range(num_end_nodes_in_convex_hull - 1):
        for j in range(i + 1, num_end_nodes_in_convex_hull):
            print 'Drawing skeleton line {0:d} of {1:d}...'.format(
                num_skeleton_lines_done + 1, num_skeleton_lines)

            this_node_table = copy.deepcopy(node_table)
            this_node_table = _remove_node_from_children(
                node_table=this_node_table,
                target_node_index=end_node_in_convex_hull_indices[i])

            these_x_coords, these_y_coords = _get_skeleton_line(
                node_table=this_node_table,
                triangle_to_node_table=triangle_to_node_table,
                start_node_index=end_node_in_convex_hull_indices[i],
                end_node_index=end_node_in_convex_hull_indices[j])
            num_skeleton_lines_done += 1

            this_vertex_list_xy = front_utils._vertex_arrays_to_list(
                vertex_x_coords_metres=these_x_coords,
                vertex_y_coords_metres=these_y_coords)
            this_linestring_object_xy = shapely.geometry.LineString(
                this_vertex_list_xy)
            this_length = this_linestring_object_xy.length

            if this_length <= max_skeleton_line_length:
                continue

            max_skeleton_line_length = copy.deepcopy(this_length)
            skeleton_line_x_coords = copy.deepcopy(these_x_coords)
            skeleton_line_y_coords = copy.deepcopy(these_y_coords)

    return skeleton_line_x_coords, skeleton_line_y_coords
