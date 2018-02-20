"""Plotting methods for skeleton lines.

A "skeleton line" is a polyline description of a polygon.  For more details, see
skeleton_lines.py.
"""

import numpy
import matplotlib
matplotlib.use('agg')
from generalexam.ge_utils import skeleton_lines
from gewittergefahr.gg_utils import error_checking

DEFAULT_POLYGON_COLOUR = numpy.array([0., 0., 0.]) / 255
DEFAULT_SKELETON_LINE_COLOUR = numpy.array([252., 141., 98.]) / 255
DEFAULT_END_NODE_COLOUR = numpy.array([252., 141., 98.]) / 255
DEFAULT_NEW_EDGE_COLOUR = numpy.array([102., 194., 165.]) / 255
DEFAULT_BRANCH_NODE_COLOUR = numpy.array([102., 194., 165.]) / 255
DEFAULT_JUMPER_NODE_COLOUR = numpy.array([141., 160., 203.]) / 255

DEFAULT_LINE_WIDTH = 2.
DEFAULT_MARKER_SIZE = 8

MARKER_TYPE = 'o'
FONT_SIZE = 16
HORIZONTAL_ALIGNMENT_FOR_NODES = 'left'
VERTICAL_ALIGNMENT_FOR_NODES = 'bottom'
HORIZONTAL_ALIGNMENT_FOR_POLYGON_VERTICES = 'right'
VERTICAL_ALIGNMENT_FOR_POLYGON_VERTICES = 'top'


def plot_polygon(
        polygon_object_xy, axes_object, line_colour=DEFAULT_POLYGON_COLOUR,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots original polygon (without skeleton line or Delaunay triangulation).

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_colour: Colour of polygon edges (in any format accepted by
        `matplotlib.colors`).
    :param line_width: Width of polygon edges (real positive number).
    """

    vertex_x_coords = numpy.array(polygon_object_xy.exterior.xy[0])
    vertex_y_coords = numpy.array(polygon_object_xy.exterior.xy[1])

    axes_object.plot(
        vertex_x_coords, vertex_y_coords, color=line_colour, linestyle='solid',
        linewidth=line_width)

    num_vertices = len(vertex_x_coords)
    for i in range(num_vertices - 1):
        axes_object.text(
            vertex_x_coords[i], vertex_y_coords[i], str(i), fontsize=FONT_SIZE,
            color=line_colour,
            horizontalalignment=HORIZONTAL_ALIGNMENT_FOR_POLYGON_VERTICES,
            verticalalignment=VERTICAL_ALIGNMENT_FOR_POLYGON_VERTICES)


def plot_delaunay_triangulation(
        polygon_object_xy, node_table, new_edge_table, axes_object,
        new_edge_colour=DEFAULT_NEW_EDGE_COLOUR,
        new_edge_width=DEFAULT_LINE_WIDTH,
        end_node_colour=DEFAULT_END_NODE_COLOUR,
        end_node_marker_size=DEFAULT_MARKER_SIZE,
        branch_node_colour=DEFAULT_BRANCH_NODE_COLOUR,
        branch_node_marker_size=DEFAULT_MARKER_SIZE,
        jumper_node_colour=DEFAULT_JUMPER_NODE_COLOUR,
        jumper_node_marker_size=DEFAULT_MARKER_SIZE):
    """Plots Delaunay triangulation of polygon.

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y (Cartesian) coordinates.
    :param node_table: pandas DataFrame created by
        `skeleton_lines._find_and_classify_nodes` or
        `skeleton_lines._find_and_classify_node_children`.
    :param new_edge_table: pandas DataFrame created by
        `skeleton_lines._find_new_edges_from_triangulation`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param new_edge_colour: Colour of new edges (those in triangulation and not
        in original polygon) (in any format accepted by `matplotlib.colors`).
    :param new_edge_width: Width of new edges.
    :param end_node_colour: Colour of end nodes.
    :param end_node_marker_size: Marker size for end nodes.
    :param branch_node_colour: Colour of branch nodes.
    :param branch_node_marker_size: Marker size for branch nodes.
    :param jumper_node_colour: Colour of jumper nodes.
    :param jumper_node_marker_size: Marker size for jumper nodes.
    """

    polygon_vertex_x_coords = numpy.array(polygon_object_xy.exterior.xy[0])
    polygon_vertex_y_coords = numpy.array(polygon_object_xy.exterior.xy[1])

    num_new_edges = len(new_edge_table.index)
    for i in range(num_new_edges):
        these_vertex_indices = new_edge_table[
            skeleton_lines.VERTEX_INDICES_KEY].values[i]

        axes_object.plot(
            polygon_vertex_x_coords[these_vertex_indices],
            polygon_vertex_y_coords[these_vertex_indices],
            color=new_edge_colour, linestyle='solid', linewidth=new_edge_width)

    num_nodes = len(node_table.index)
    for i in range(num_nodes):
        this_node_type = node_table[skeleton_lines.NODE_TYPE_KEY].values[i]

        if this_node_type == skeleton_lines.END_NODE_TYPE:
            this_colour = end_node_colour
            this_marker_size = end_node_marker_size
        elif this_node_type == skeleton_lines.BRANCH_NODE_TYPE:
            this_colour = branch_node_colour
            this_marker_size = branch_node_marker_size
        elif this_node_type == skeleton_lines.JUMPER_NODE_TYPE:
            this_colour = jumper_node_colour
            this_marker_size = jumper_node_marker_size

        axes_object.plot(
            node_table[skeleton_lines.NODE_X_COORDS_KEY].values[i],
            node_table[skeleton_lines.NODE_Y_COORDS_KEY].values[i],
            linestyle='None', marker=MARKER_TYPE, markerfacecolor=this_colour,
            markeredgecolor=this_colour, markersize=this_marker_size,
            markeredgewidth=1)
        axes_object.text(
            node_table[skeleton_lines.NODE_X_COORDS_KEY].values[i],
            node_table[skeleton_lines.NODE_Y_COORDS_KEY].values[i], str(i),
            fontsize=FONT_SIZE, color=this_colour,
            horizontalalignment=HORIZONTAL_ALIGNMENT_FOR_NODES,
            verticalalignment=VERTICAL_ALIGNMENT_FOR_NODES)


def plot_skeleton_line(
        skeleton_line_x_coords, skeleton_line_y_coords, axes_object,
        line_colour=DEFAULT_SKELETON_LINE_COLOUR,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots skeleton line through polygon.

    P = number of points in skeleton line

    :param skeleton_line_x_coords: length-P numpy array with x-coordinates on
        skeleton line.
    :param skeleton_line_y_coords: length-P numpy array with y-coordinates on
        skeleton line.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_colour: Colour of skeleton line (in any format accepted by
        `matplotlib.colors`).
    :param line_width: Width of skeleton line (real positive number).
    """

    error_checking.assert_is_numpy_array_without_nan(skeleton_line_x_coords)
    error_checking.assert_is_numpy_array(
        skeleton_line_x_coords, num_dimensions=1)
    num_points = len(skeleton_line_x_coords)

    error_checking.assert_is_numpy_array_without_nan(skeleton_line_y_coords)
    error_checking.assert_is_numpy_array(
        skeleton_line_y_coords, exact_dimensions=numpy.array([num_points]))

    axes_object.plot(
        skeleton_line_x_coords, skeleton_line_y_coords, color=line_colour,
        linestyle='solid', linewidth=line_width)
