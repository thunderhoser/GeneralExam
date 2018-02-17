"""Plotting methods for warm and cold fronts."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from generalexam.ge_utils import front_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_WARM_FRONT_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_COLD_FRONT_COLOUR = numpy.array([31., 120., 180.]) / 255
DEFAULT_LINE_WIDTH = 2.
DEFAULT_LINE_STYLE = 'solid'

DEFAULT_GRID_OPACITY = 0.5


def _get_colour_map_for_grid():
    """Returns colour map for frontal grid (to be used by `plot_frontal_grid`).

    N = number of colours

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: length-(N + 1) numpy array of colour boundaries.
        colour_bounds[0] and colour_bounds[1] are the boundaries for the 1st
        colour; colour_bounds[1] and colour_bounds[2] are the boundaries for the
        2nd colour; ...; etc.
    """

    main_colour_list = [DEFAULT_WARM_FRONT_COLOUR, DEFAULT_COLD_FRONT_COLOUR]
    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.array(
        [front_utils.WARM_FRONT_INTEGER_ID - 0.5,
         front_utils.WARM_FRONT_INTEGER_ID + 0.5,
         front_utils.COLD_FRONT_INTEGER_ID])
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds, numpy.array([100.])))
    return colour_map_object, colour_norm_object, colour_bounds


def plot_polyline(
        latitudes_deg, longitudes_deg, basemap_object, axes_object,
        front_type=None, line_colour=None, line_width=DEFAULT_LINE_WIDTH,
        line_style=DEFAULT_LINE_STYLE):
    """Plots either warm front or cold front as polyline.

    P = number of points in polyline

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg N).
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param front_type: Type of front (string).  Used only to determine line
        colour (if `line_colour` is left as None).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
        Defaults to `DEFAULT_WARM_FRONT_COLOUR` or `DEFAULT_COLD_FRONT_COLOUR`.
    :param line_width: Line width (real positive number).
    :param line_style: Line style (in any format accepted by
        `matplotlib.lines`).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    if line_colour is None:
        front_utils.check_front_type(front_type)
        if front_type == front_utils.WARM_FRONT_STRING_ID:
            line_colour = DEFAULT_WARM_FRONT_COLOUR
        else:
            line_colour = DEFAULT_COLD_FRONT_COLOUR

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg)
    axes_object.plot(
        x_coords_metres, y_coords_metres, color=line_colour,
        linestyle=line_style, linewidth=line_width)


def plot_frontal_grid(
        frontal_grid_matrix, basemap_object, axes_object,
        model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=None,
        opacity=DEFAULT_GRID_OPACITY):
    """Plots grid points intersected by a warm front or cold front.

    :param frontal_grid_matrix: See documentation for
        `front_utils.frontal_grid_to_points`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param model_name: Name of NWP (numerical weather prediction) model whose
        grid is being used.
    :param grid_id: String ID for model grid.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    nw_latitude_deg_as_array, nw_longitude_deg_as_array = (
        nwp_model_utils.project_xy_to_latlng(
            numpy.array([nwp_model_utils.MIN_GRID_POINT_X_METRES]),
            numpy.array([nwp_model_utils.MIN_GRID_POINT_Y_METRES]),
            projection_object=None, model_name=model_name, grid_id=grid_id))

    x_min_in_basemap_proj_metres, y_min_in_basemap_proj_metres = basemap_object(
        nw_longitude_deg_as_array[0], nw_latitude_deg_as_array[0])

    x_spacing_metres, y_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=model_name, grid_id=grid_id)

    (frontal_grid_matrix_at_edges,
     grid_cell_edge_x_metres,
     grid_cell_edge_y_metres) = grids.xy_field_grid_points_to_edges(
         field_matrix=frontal_grid_matrix,
         x_min_metres=x_min_in_basemap_proj_metres,
         y_min_metres=y_min_in_basemap_proj_metres,
         x_spacing_metres=x_spacing_metres,
         y_spacing_metres=y_spacing_metres)

    frontal_grid_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(frontal_grid_matrix_at_edges), frontal_grid_matrix_at_edges)

    colour_map_object, _, colour_bounds = _get_colour_map_for_grid()

    basemap_object.pcolormesh(
        grid_cell_edge_x_metres, grid_cell_edge_y_metres,
        frontal_grid_matrix_at_edges, cmap=colour_map_object,
        vmin=colour_bounds[1], vmax=colour_bounds[-2], shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e9, alpha=opacity)
