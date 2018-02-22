"""Plotting methods for NARR data."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
DEFAULT_BOUNDARY_RESOLUTION_STRING = 'l'

ELLIPSOID = 'sphere'
EARTH_RADIUS_METRES = 6370997.

NUM_ROWS_IN_NARR_GRID, NUM_COLUMNS_IN_NARR_GRID = (
    nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME))


def get_xy_grid_point_matrices(
        first_row_in_narr_grid, last_row_in_narr_grid,
        first_column_in_narr_grid, last_column_in_narr_grid,
        basemap_object=None):
    """Returns coordinate matrices for a contiguous subset of the NARR grid.

    However, this subset need not be *strictly* a subset.  In other words, the
    "subset" could be the full NARR grid.

    This method generates different x- and y-coordinates than
    `nwp_model_utils.get_xy_grid_point_matrices`, because (like
    `mpl_toolkits.basemap.Basemap`) this method assumes that false easting and
    northing are zero.

    :param first_row_in_narr_grid: Row 0 in the subgrid is row
        `first_row_in_narr_grid` in the full NARR grid.
    :param last_row_in_narr_grid: Last row (index -1) in the subgrid is row
        `last_row_in_narr_grid` in the full NARR grid.
    :param first_column_in_narr_grid: Column 0 in the subgrid is row
        `first_column_in_narr_grid` in the full NARR grid.
    :param last_column_in_narr_grid: Last column (index -1) in the subgrid is
        row `last_column_in_narr_grid` in the full NARR grid.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap` created
        for the NARR grid.  If you don't have one, no big deal -- leave this
        argument empty.
    :return: grid_point_x_matrix_metres: M-by-N numpy array of x-coordinates.
    :return: grid_point_y_matrix_metres: M-by-N numpy array of y-coordinates.
    """

    error_checking.assert_is_integer(first_row_in_narr_grid)
    error_checking.assert_is_geq(first_row_in_narr_grid, 0)
    error_checking.assert_is_integer(last_row_in_narr_grid)
    error_checking.assert_is_greater(
        last_row_in_narr_grid, first_row_in_narr_grid)
    error_checking.assert_is_less_than(
        last_row_in_narr_grid, NUM_ROWS_IN_NARR_GRID)

    error_checking.assert_is_integer(first_column_in_narr_grid)
    error_checking.assert_is_geq(first_column_in_narr_grid, 0)
    error_checking.assert_is_integer(last_column_in_narr_grid)
    error_checking.assert_is_greater(
        last_column_in_narr_grid, first_column_in_narr_grid)
    error_checking.assert_is_less_than(
        last_column_in_narr_grid, NUM_COLUMNS_IN_NARR_GRID)

    latitude_matrix_deg, longitude_matrix_deg = (
        nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME))

    latitude_matrix_deg = latitude_matrix_deg[
        first_row_in_narr_grid:(last_row_in_narr_grid + 1),
        first_column_in_narr_grid:(last_column_in_narr_grid + 1)]
    longitude_matrix_deg = longitude_matrix_deg[
        first_row_in_narr_grid:(last_row_in_narr_grid + 1),
        first_column_in_narr_grid:(last_column_in_narr_grid + 1)]

    if basemap_object is None:
        standard_latitudes_deg, central_longitude_deg = (
            nwp_model_utils.get_projection_params(
                nwp_model_utils.NARR_MODEL_NAME))
        projection_object = projections.init_lambert_conformal_projection(
            standard_latitudes_deg=standard_latitudes_deg,
            central_longitude_deg=central_longitude_deg)

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            projections.project_latlng_to_xy(
                latitude_matrix_deg, longitude_matrix_deg,
                projection_object=projection_object, false_northing_metres=0.,
                false_easting_metres=0.))

    else:
        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            basemap_object(longitude_matrix_deg, latitude_matrix_deg))

    return grid_point_x_matrix_metres, grid_point_y_matrix_metres


def init_basemap(
        fig_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        fig_height_inches=DEFAULT_FIG_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING,
        first_row_in_narr_grid=0,
        last_row_in_narr_grid=NUM_ROWS_IN_NARR_GRID - 1,
        first_column_in_narr_grid=0,
        last_column_in_narr_grid=NUM_COLUMNS_IN_NARR_GRID - 1):
    """Creates basemap with the NARR's Lambert conformal conic projection.

    :param fig_width_inches: Figure width.
    :param fig_height_inches: Figure height.
    :param resolution_string: Resolution for boundaries (e.g., coastlines and
        political borders).  Options are "c" for crude, "l" for low, "i" for
        intermediate, "h" for high, and "f" for full.  Keep in mind that higher-
        resolution boundaries take much longer to draw.
    :param first_row_in_narr_grid: See documentation for
        `get_xy_grid_point_matrices`.
    :param last_row_in_narr_grid: See doc for `get_xy_grid_point_matrices`.
    :param first_column_in_narr_grid: See doc for `get_xy_grid_point_matrices`.
    :param last_column_in_narr_grid: See doc for `get_xy_grid_point_matrices`.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    error_checking.assert_is_greater(fig_width_inches, 0)
    error_checking.assert_is_greater(fig_height_inches, 0)
    error_checking.assert_is_string(resolution_string)

    grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
        get_xy_grid_point_matrices(
            first_row_in_narr_grid=first_row_in_narr_grid,
            last_row_in_narr_grid=last_row_in_narr_grid,
            first_column_in_narr_grid=first_column_in_narr_grid,
            last_column_in_narr_grid=last_column_in_narr_grid))

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(fig_width_inches, fig_height_inches))

    standard_latitudes_deg, central_longitude_deg = (
        nwp_model_utils.get_projection_params(
            nwp_model_utils.NARR_MODEL_NAME))

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=EARTH_RADIUS_METRES, ellps=ELLIPSOID,
        resolution=resolution_string,
        llcrnrx=grid_point_x_matrix_metres[0, 0],
        llcrnry=grid_point_y_matrix_metres[0, 0],
        urcrnrx=grid_point_x_matrix_metres[-1, -1],
        urcrnry=grid_point_y_matrix_metres[-1, -1])

    return figure_object, axes_object, basemap_object


def plot_xy_grid(
        data_matrix, axes_object, basemap_object, colour_map, colour_minimum,
        colour_maximum, first_row_in_narr_grid=0, first_column_in_narr_grid=0,
        opacity=1.):
    """Plots data over a contiguous subset of the NARR grid.

    However, this subset need not be *strictly* a subset.  In other words, the
    "subset" could be the full NARR grid.

    M = number of rows (unique grid-point y-coordinates) in subgrid
    N = number of columns (unique grid-point x-coordinates) in subgrid

    :param data_matrix: M-by-N numpy array of values to plot.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_minimum: Minimum value for colour map.
    :param colour_maximum: Maximum value for colour map.
    :param first_row_in_narr_grid: See documentation for
        `get_xy_grid_point_matrices`.
    :param first_column_in_narr_grid: See doc for `get_xy_grid_point_matrices`.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_real_numpy_array(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)
    error_checking.assert_is_greater(colour_maximum, colour_minimum)

    num_rows = data_matrix.shape[0]
    num_columns = data_matrix.shape[1]

    grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
        get_xy_grid_point_matrices(
            first_row_in_narr_grid=first_row_in_narr_grid,
            last_row_in_narr_grid=first_row_in_narr_grid + num_rows - 1,
            first_column_in_narr_grid=first_column_in_narr_grid,
            last_column_in_narr_grid=
            first_column_in_narr_grid + num_columns - 1,
            basemap_object=basemap_object))

    x_spacing_metres = (
        (grid_point_x_matrix_metres[0, -1] - grid_point_x_matrix_metres[0, 0]) /
        (num_columns - 1))
    y_spacing_metres = (
        (grid_point_y_matrix_metres[-1, 0] - grid_point_y_matrix_metres[0, 0]) /
        (num_rows - 1))

    data_matrix_at_edges, grid_cell_edge_x_metres, grid_cell_edge_y_metres = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=data_matrix,
            x_min_metres=grid_point_x_matrix_metres[0, 0],
            y_min_metres=grid_point_y_matrix_metres[0, 0],
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres))

    data_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(data_matrix_at_edges), data_matrix_at_edges)

    basemap_object.pcolormesh(
        grid_cell_edge_x_metres, grid_cell_edge_y_metres,
        data_matrix_at_edges, cmap=colour_map, vmin=colour_minimum,
        vmax=colour_maximum, shading='flat', edgecolors='None',
        axes=axes_object, zorder=-1e9, alpha=opacity)
