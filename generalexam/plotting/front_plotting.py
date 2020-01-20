"""Plotting methods for warm and cold fronts."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import front_utils

DEFAULT_WARM_FRONT_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_COLD_FRONT_COLOUR = numpy.array([31., 120., 180.]) / 255
DEFAULT_LINE_WIDTH = 2.
DEFAULT_LINE_STYLE = 'solid'

DEFAULT_GRID_OPACITY = 0.5

DEFAULT_WF_MARKER_TYPE = 'o'
DEFAULT_CF_MARKER_TYPE = '>'
DEFAULT_MARKER_SPACING_METRES = 150000.
DEFAULT_MARKER_SIZE = 12
DEFAULT_MARKER_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255


def get_colour_map_for_grid():
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
        [front_utils.WARM_FRONT_ENUM - 0.5,
         front_utils.WARM_FRONT_ENUM + 0.5,
         front_utils.COLD_FRONT_ENUM]
    )

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([-100.]), main_colour_bounds, numpy.array([100.])
    ))

    return colour_map_object, colour_norm_object, colour_bounds


def plot_front_with_markers(
        line_latitudes_deg, line_longitudes_deg, axes_object, basemap_object,
        marker_spacing_metres=DEFAULT_MARKER_SPACING_METRES,
        marker_type=None, front_type_string=None,
        marker_colour=DEFAULT_MARKER_COLOUR, marker_size=DEFAULT_MARKER_SIZE):
    """Plots front with markers (instead of a line).

    P = number of points in line

    :param line_latitudes_deg: length-P numpy array of latitudes (deg N).
    :param line_longitudes_deg: length-P numpy array of longitudes (deg E).
    :param axes_object: Front will be plotted on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Basemap used to convert lat-long coordinates to x-y
        (instance of `mpl_toolkits.basemap.Basemap`).
    :param marker_spacing_metres: Spacing between successive markers.
    :param marker_type: Marker type (any format accepted by matplotlib).
    :param front_type_string: [used only if `marker_type is None`]
        Front type (determines marker type).
    :param marker_colour: Marker colour (any format accepted by matplotlib).
    :param marker_size: Marker size (any format accepted by matplotlib).
    """

    error_checking.assert_is_valid_lat_numpy_array(line_latitudes_deg)
    error_checking.assert_is_numpy_array(line_latitudes_deg, num_dimensions=1)

    num_points = len(line_latitudes_deg)
    these_expected_dim = numpy.array([num_points], dtype=int)
    error_checking.assert_is_numpy_array(
        line_longitudes_deg, exact_dimensions=these_expected_dim)

    line_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        line_longitudes_deg)

    error_checking.assert_is_greater(marker_spacing_metres, 0.)

    if marker_type is None:
        front_utils.check_front_type_string(front_type_string)

        if front_type_string == front_utils.WARM_FRONT_STRING:
            marker_type = DEFAULT_WF_MARKER_TYPE
        else:
            marker_type = DEFAULT_CF_MARKER_TYPE

    x_coords_metres, y_coords_metres = basemap_object(
        line_longitudes_deg, line_latitudes_deg)

    for i in range(num_points - 1):
        this_x_diff_metres = x_coords_metres[i + 1] - x_coords_metres[i]
        this_y_diff_metres = y_coords_metres[i + 1] - y_coords_metres[i]
        this_distance_metres = numpy.sqrt(
            this_x_diff_metres ** 2 + this_y_diff_metres ** 2)

        this_num_points = 1 + int(numpy.ceil(
            this_distance_metres / marker_spacing_metres
        ))

        these_x_coords_metres = numpy.linspace(
            x_coords_metres[i], x_coords_metres[i + 1], num=this_num_points)
        these_y_coords_metres = numpy.linspace(
            y_coords_metres[i], y_coords_metres[i + 1], num=this_num_points)

        axes_object.plot(
            these_x_coords_metres, these_y_coords_metres,
            linestyle='None', marker=marker_type, markerfacecolor=marker_colour,
            markeredgecolor=marker_colour, markersize=marker_size,
            markeredgewidth=0.1)


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
        front_utils.check_front_type_string(front_type)

        if front_type == front_utils.WARM_FRONT_STRING:
            line_colour = DEFAULT_WARM_FRONT_COLOUR
        else:
            line_colour = DEFAULT_COLD_FRONT_COLOUR

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg)
    axes_object.plot(
        x_coords_metres, y_coords_metres, color=line_colour,
        linestyle=line_style, linewidth=line_width)


def plot_labels_on_general_grid(
        label_matrix, latitude_matrix_deg, longitude_matrix_deg,
        axes_object, basemap_object, opacity=DEFAULT_GRID_OPACITY):
    """Plots front labels on general grid.

    M = number of rows in grid
    N = number of columns in grid

    :param label_matrix: M-by-N numpy array of labels (integers accepted by
        `front_utils.check_front_type_enum`).
    :param latitude_matrix_deg: M-by-N numpy array of latitudes (deg N).
    :param longitude_matrix_deg: M-by-N numpy array of longitudes (deg E).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will be used to convert between lat-long and x-y
        (projection) coordinates (instance of `mpl_toolkits.basemap.Basemap`).
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_integer_numpy_array(label_matrix)
    error_checking.assert_is_numpy_array(label_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(
        label_matrix, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS)
    )
    error_checking.assert_is_leq_numpy_array(
        label_matrix, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS)
    )

    expected_dim = numpy.array(label_matrix.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        latitude_matrix_deg, exact_dimensions=expected_dim
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=latitude_matrix_deg, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        longitude_matrix_deg, exact_dimensions=expected_dim
    )
    longitude_matrix_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=longitude_matrix_deg, allow_nan=False
    )

    colour_map_object, _, colour_bounds = get_colour_map_for_grid()

    matrix_to_plot = numpy.ma.masked_where(
        label_matrix == front_utils.NO_FRONT_ENUM, label_matrix
    )

    x_matrix_metres, y_matrix_metres = basemap_object(
        longitude_matrix_deg, latitude_matrix_deg
    )
    basemap_object.pcolormesh(
        x_matrix_metres, y_matrix_metres, matrix_to_plot,
        cmap=colour_map_object, vmin=colour_bounds[1], vmax=colour_bounds[-2],
        shading='flat', edgecolors='None',
        ax=axes_object, zorder=-1e12, alpha=opacity
    )


def plot_labels_on_model_grid(
        gridded_front_matrix, axes_object, basemap_object, full_grid_name,
        first_row_in_full_grid=0, first_column_in_full_grid=0,
        opacity=DEFAULT_GRID_OPACITY):
    """Plots front labels on model grid.

    :param gridded_front_matrix: See doc for
        `front_utils.gridded_labels_to_points`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will be used to convert between lat-long and x-y
        (projection) coordinates (instance of `mpl_toolkits.basemap.Basemap`).
    :param full_grid_name: Name of full grid (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param first_row_in_full_grid: First row in full grid.  In other words,
        row 0 in `gridded_front_matrix` = row `first_row_in_full_grid` in the
        full grid.
    :param first_column_in_full_grid: Same but for column.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_integer_numpy_array(gridded_front_matrix)
    error_checking.assert_is_numpy_array(gridded_front_matrix, num_dimensions=2)

    error_checking.assert_is_geq_numpy_array(
        gridded_front_matrix, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS)
    )
    error_checking.assert_is_leq_numpy_array(
        gridded_front_matrix, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS)
    )

    colour_map_object, _, colour_bounds = get_colour_map_for_grid()

    gridded_front_matrix = numpy.ma.masked_where(
        gridded_front_matrix == front_utils.NO_FRONT_ENUM,
        gridded_front_matrix
    )

    nwp_plotting.plot_subgrid(
        field_matrix=gridded_front_matrix,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=colour_map_object,
        min_colour_value=colour_bounds[1], max_colour_value=colour_bounds[-2],
        grid_id=full_grid_name, first_row_in_full_grid=first_row_in_full_grid,
        first_column_in_full_grid=first_column_in_full_grid, opacity=opacity
    )
